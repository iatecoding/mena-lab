"""island_context.py: the whole-structure channel of MenaACE (IC). Per-island pooled context broadcast back to atoms.

Borrowed ideas, labelled: attention pooling with learned scores (Li et al. 2016, gated graph neural networks' global
attention; Lee et al. 2019, Set Transformer PMA); a non-local context channel in an ML potential for charge and spin
(Unke et al. 2021, SpookyNet's non-local attention); zero-initialised residual output (Bachlechner et al. 2020, ReZero).
This repo's part: the pooling is restricted to islands (connected components of the r < r_cut graph), so
E(A + B far apart) = E(A) + E(B) holds exactly, as in charge_mode="island"; the island charge Q_I and the bounded size
term n_I^(-1/3) (-> 0 for large islands, no out-of-range input) enter the context.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NEG = -1.0e9  # finite fill for masked logits (an all-masked compile-padding graph must give 0, not NaN)


class _SparseWindowSumFn(torch.autograd.Function):
    """Memory-light weighted sparse window sum y_i = x_i + sum_{e: recv(e)=i} w_e * x_{send(e)} for x [N, C].
    Allocates at most ONE temporary [E_lr, C] tensor (C=16 floats/edge) during forward using in-place .mul_,
    and saves ONLY (x [N, C], recv, send, w [E_lr]) for backward (0 floats/edge of feature storage retained)."""

    @staticmethod
    def forward(ctx, x: Tensor, recv: Tensor, send: Tensor, w: Tensor) -> Tensor:
        ctx.save_for_backward(x, recv, send, w)
        out = x.clone()
        if recv.numel() == 0:
            return out
        msg = x.index_select(0, send).mul_(w.unsqueeze(-1))
        out.index_add_(0, recv, msg)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, recv, send, w = ctx.saved_tensors
        if recv.numel() == 0:
            return grad_out, None, None, torch.zeros_like(w)
        if grad_out.requires_grad or torch.is_grad_enabled():
            grad_x = grad_out + grad_out.new_zeros(x.shape).index_add_(0, send, w.unsqueeze(-1) * grad_out[recv])
            grad_w = (grad_out[recv] * x[send]).sum(dim=-1)
            return grad_x, None, None, grad_w
        grad_x = grad_out.new_zeros(x.shape)
        msg_g = grad_out.index_select(0, recv).mul_(w.unsqueeze(-1))
        grad_x.index_add_(0, send, msg_g)
        del msg_g
        grad_x.add_(grad_out)
        grad_w = grad_out.index_select(0, recv).mul_(x.index_select(0, send)).sum(dim=-1)
        return grad_x, None, None, grad_w


class IslandPool:
    """Per-island or per-window reductions, broadcast back to every atom. Backends:
    - dense_single (training single-island fast path: every graph is one island)
    - dense_matrix (dense inference or windowed context: R [B, N, N] 0/1 same-island matrix or smooth window W_ij in [0, 1]
      with diagonal 1 for real atoms)
    - sparse (legacy hard islands: group [N] island id 0..G-1)
    - sparse_window (continuous windowed context "lever G": directed edges (recv, send) with smooth C3 weights w_edge [E]
      plus implicit self-weight w_ii = 1.0 on every atom)."""

    def __init__(
        self,
        kind: str,
        mask: Tensor | None = None,
        R: Tensor | None = None,
        group: Tensor | None = None,
        n_groups: int = 0,
        recv: Tensor | None = None,
        send: Tensor | None = None,
        w_edge: Tensor | None = None,
        n_atoms: int = 0,
    ):
        self.kind = kind
        self.mask = mask
        self.R = R
        self.group = group
        self.G = int(n_groups)
        self.recv = recv
        self.send = send
        self.w_edge = w_edge
        self.N = int(n_atoms)

    @classmethod
    def dense_single(cls, mask: Tensor) -> "IslandPool":
        return cls("single", mask=mask)

    @classmethod
    def dense_matrix(cls, R: Tensor, mask: Tensor) -> "IslandPool":
        return cls("matrix", mask=mask, R=R)

    @classmethod
    def sparse(cls, group: Tensor, n_groups: int) -> "IslandPool":
        return cls("sparse", group=group, n_groups=n_groups)

    @classmethod
    def sparse_window(cls, recv: Tensor, send: Tensor, w_edge: Tensor, n_atoms: int) -> "IslandPool":
        return cls("sparse_window", recv=recv, send=send, w_edge=w_edge, n_atoms=n_atoms)

    def count(self, dtype: torch.dtype) -> Tensor:
        """Effective atom count in each atom's island or window, >= 1: [B, N] (dense) or [N] (sparse)."""
        if self.kind == "single":
            m = self.mask.to(dtype)
            return m.sum(-1, keepdim=True).clamp_min(1.0).expand_as(m)
        if self.kind == "matrix":
            return self.R.to(dtype).sum(-1).clamp_min(1.0)
        if self.kind == "sparse_window":
            w = self.w_edge.to(dtype)
            ones = torch.ones(self.N, dtype=dtype, device=w.device)
            return ones.index_add_(0, self.recv, w).clamp_min(1.0)
        if self.G == 1:
            return torch.full((self.group.shape[0],), max(1.0, float(self.group.shape[0])), dtype=dtype, device=self.group.device)
        ones = torch.ones(self.group.shape[0], dtype=dtype, device=self.group.device)
        return ones.new_zeros(self.G).index_add_(0, self.group, ones).clamp_min(1.0)[self.group]

    def sum(self, x: Tensor) -> Tensor:
        """Weighted sum over each atom's island or window: [B, N] / [N] (or [..., C])."""
        if self.kind == "single":
            m = self.mask.to(x.dtype)
            if x.ndim == 3:
                m = m.unsqueeze(-1)
            return (x * m).sum(1, keepdim=True).expand_as(x)
        if self.kind == "matrix":
            R = self.R.to(x.dtype)
            if x.ndim == 2:
                return torch.bmm(R, x.unsqueeze(-1)).squeeze(-1)
            return torch.bmm(R, x)
        if self.kind == "sparse_window":
            w = self.w_edge.to(x.dtype)
            if x.ndim == 1:
                return x.clone().index_add_(0, self.recv, w * x[self.send])
            return _SparseWindowSumFn.apply(x, self.recv, self.send, w)
        if self.G == 1:
            return x.sum(dim=0, keepdim=True).expand_as(x)
        return x.new_zeros(self.G, *x.shape[1:]).index_add_(0, self.group, x)[self.group]

    def mean(self, x: Tensor) -> Tensor:
        """x [B, N, C] or [N, C] -> island/window mean per atom, same shape."""
        if self.kind == "single":
            m = self.mask.to(x.dtype).unsqueeze(-1)
            return ((x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)).expand_as(x)
        if self.kind == "matrix":
            R = self.R.to(x.dtype)
            return torch.bmm(R, x) / R.sum(-1, keepdim=True).clamp_min(1.0)
        if self.kind == "sparse_window":
            w = self.w_edge.to(x.dtype)
            s = x.clone().index_add_(0, self.recv, w.unsqueeze(-1) * x[self.send])
            c = x.new_ones(self.N).index_add_(0, self.recv, w).clamp_min(1.0)
            return s / c.unsqueeze(-1)
        if self.G == 1:
            return x.mean(dim=0, keepdim=True).expand_as(x)
        s = x.new_zeros(self.G, x.shape[-1]).index_add_(0, self.group, x)
        c = x.new_zeros(self.G).index_add_(0, self.group, torch.ones_like(x[:, 0])).clamp_min(1.0)
        return (s / c.unsqueeze(-1))[self.group]

    def softmax_pool(self, logits: Tensor, values: Tensor) -> Tensor:
        """logits [..., H], values [..., H, Dh] -> sum_j softmax_j(logits) values_j over each atom's island/window, [..., H*Dh].
        The max subtracted for stability is detached (softmax is invariant to it)."""
        if self.kind == "single":
            m = self.mask.unsqueeze(-1)  # [B, N, 1]
            lg = logits.masked_fill(~m, NEG)
            e = torch.exp(lg - lg.amax(1, keepdim=True).detach()) * m.to(logits.dtype)  # [B, N, H]
            z = e.sum(1, keepdim=True).clamp_min(1e-12)
            out = ((e / z).unsqueeze(-1) * values).sum(1, keepdim=True)  # [B, 1, H, Dh]
            return out.flatten(-2).expand(*values.shape[:2], -1)
        if self.kind == "matrix":
            R = self.R.to(logits.dtype)  # [B, N, N]
            support = R.unsqueeze(-1) > 0
            lg = logits.unsqueeze(1).expand(-1, R.shape[1], -1, -1).masked_fill(~support, NEG)  # [B, i, j, H]
            mx = lg.amax(2, keepdim=True).detach()
            e = torch.exp(lg - mx) * R.unsqueeze(-1)
            w = e / e.sum(2, keepdim=True).clamp_min(1e-12)
            return torch.einsum("bijh,bjhd->bihd", w, values).flatten(-2)
        if self.kind == "sparse_window":
            H = logits.shape[-1]
            w_e = self.w_edge.to(logits.dtype)
            keep = w_e > 0
            r, s, w_e = self.recv[keep], self.send[keep], w_e[keep]
            mx = logits.detach().clone()
            if r.numel() > 0:
                mx = mx.scatter_reduce(
                    0, r.unsqueeze(-1).expand(-1, H), logits[s].detach(), reduce="amax", include_self=True
                )
            e_self = torch.exp(logits - mx)  # [N, H], w_ii = 1
            if r.numel() > 0:
                e_edge = torch.exp(logits[s] - mx[r]) * w_e.unsqueeze(-1)  # [E_pos, H]
                z = e_self.clone().index_add_(0, r, e_edge).clamp_min(1e-12)
                num = (e_self.unsqueeze(-1) * values).index_add_(0, r, e_edge.unsqueeze(-1) * values[s])
            else:
                z = e_self.clamp_min(1e-12)
                num = e_self.unsqueeze(-1) * values
            return (num / z.unsqueeze(-1)).flatten(-2)
        if self.G == 1:
            w = torch.softmax(logits, dim=0)
            return (w.unsqueeze(-1) * values).sum(dim=0, keepdim=True).flatten(-2).expand(logits.shape[0], -1)
        g, H = self.group, logits.shape[-1]
        mx = logits.new_full((self.G, H), NEG).scatter_reduce(
            0, g.unsqueeze(-1).expand(-1, H), logits, reduce="amax", include_self=True).detach()
        e = torch.exp(logits - mx[g])  # [N, H]
        z = logits.new_zeros(self.G, H).index_add_(0, g, e).clamp_min(1e-12)
        num = values.new_zeros(self.G, *values.shape[1:]).index_add_(0, g, e.unsqueeze(-1) * values)
        return (num / z.unsqueeze(-1))[g].flatten(-2)


class MultiWindowPool:
    """Several window pools, one per channel group (multi-scale context).
    Channel c of a [..., C] input is pooled by pools[group_of(c)]; a channel-less input ([B, N] dense, [N] sparse) and
    count() use the widest window. Members: IslandPool (dense_matrix / sparse_window) or latent_mesh.MeshWindowPool."""

    kind = "multi"

    def __init__(self, pools: list, widest: int, sparse: bool) -> None:
        self.pools, self.widest, self.sparse = pools, int(widest), bool(sparse)

    def _has_channels(self, x: Tensor) -> bool:
        return x.ndim == (2 if self.sparse else 3)

    def _split(self, C: int) -> list[tuple[int, int]]:
        k = len(self.pools)
        if C % k:
            raise ValueError(f"{C} channels do not split evenly over {k} windows")
        w = C // k
        return [(i * w, (i + 1) * w) for i in range(k)]

    def sum(self, x: Tensor) -> Tensor:
        if not self._has_channels(x):
            return self.pools[self.widest].sum(x)
        return torch.cat([pl.sum(x[..., a:b]) for pl, (a, b) in zip(self.pools, self._split(x.shape[-1]))], -1)

    def count(self, dtype: torch.dtype) -> Tensor:
        return self.pools[self.widest].count(dtype)

    def count_channels(self, dtype: torch.dtype, C: int) -> Tensor:
        """[..., C] effective atom count of each channel's window."""
        return torch.cat([pl.count(dtype).unsqueeze(-1).expand(*pl.count(dtype).shape, b - a)
                          for pl, (a, b) in zip(self.pools, self._split(C))], -1)


class IslandContext(nn.Module):
    """One IC point: s_i <- s_i + W_o(SiLU(a) * b), [a, b] = W_c [c_I, Q_I / 4, n_I^(-1/3)] + W_s RMSNorm(s_i).

    In 'window' mode (or when lr_dim > 0, default lr_dim=16), projects per-atom features h_i in R^d to width
    d_lr = 16 BEFORE pooling over long-range edges, never materialising an [E_lr, d] or [B, N, N, d] tensor."""

    def __init__(
        self,
        d_model: int,
        dim: int = 32,
        heads: int = 4,
        hidden: int = 64,
        lr_dim: int = 0,
        mode: str = "island",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.mode = str(mode)
        self.heads = int(heads)
        self.lr_dim = int(lr_dim) if lr_dim else (16 if self.mode == "window" else 0)
        if self.lr_dim > 0:
            if self.lr_dim % self.heads:
                raise ValueError(f"ic_lr_dim ({self.lr_dim}) must be a multiple of ic_heads ({self.heads})")
            self.dh = self.lr_dim // self.heads
            self.in_norm = nn.RMSNorm(d_model, eps=1e-6)
            self.lr_proj = nn.Linear(d_model, self.lr_dim)
            self.att_logit = nn.Linear(d_model, self.heads)
            self.ctx_norm = nn.RMSNorm(self.lr_dim, eps=1e-6)
            self.ctx_in = nn.Linear(self.lr_dim + 2, 2 * hidden)
            self.atom_in = nn.Linear(d_model, 2 * hidden, bias=False)
            self.out = nn.Linear(hidden, d_model)
            for lin in (self.att_logit, self.out):  # attention gate starts at 1.0; module starts as identity
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        else:
            if dim % heads:
                raise ValueError(f"ic_dim ({dim}) must be a multiple of ic_heads ({heads})")
            self.dh = dim // self.heads
            self.in_norm = nn.RMSNorm(d_model, eps=1e-6)
            self.mean_val = nn.Linear(d_model, dim)
            self.att_logit = nn.Linear(d_model, self.heads)
            self.att_val = nn.Linear(d_model, dim)
            self.ctx_norm = nn.RMSNorm(2 * dim, eps=1e-6)
            self.ctx_in = nn.Linear(2 * dim + 2, 2 * hidden)
            self.atom_in = nn.Linear(d_model, 2 * hidden, bias=False)
            self.out = nn.Linear(hidden, d_model)
            for lin in (self.att_logit, self.out):  # attention starts as mean pooling; the module starts as the identity
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def project_lr(self, n: Tensor) -> Tensor:
        """Project normalized atom features n [..., D] to light width lr_dim (16) BEFORE pooling over edges."""
        u = self.lr_proj(n).unflatten(-1, (self.heads, self.dh))
        gate = 2.0 * torch.sigmoid(self.att_logit(n)).unsqueeze(-1)
        return (u * gate).flatten(-2)

    def forward(self, s: Tensor, pool: IslandPool, q_isl: Tensor) -> Tensor:
        """s [B, N, D] or [N, D]; q_isl: the island/window charge of each atom, [B, N] or [N]. Returns the updated state
        (the caller re-applies the atom mask)."""
        n = self.in_norm(s)
        if self.lr_dim > 0:
            u = self.project_lr(n)
            cnt = pool.count(n.dtype)
            cnt_c = pool.count_channels(n.dtype, u.shape[-1]) if hasattr(pool, "count_channels") else cnt.unsqueeze(-1)
            pooled = pool.sum(u) / cnt_c
            ctx = self.ctx_norm(pooled)
            scal = torch.stack([q_isl.to(n.dtype) / 4.0, cnt.pow(-1.0 / 3.0)], -1)
        else:
            mean = pool.mean(self.mean_val(n))
            att = pool.softmax_pool(self.att_logit(n), self.att_val(n).unflatten(-1, (self.heads, self.dh)))
            ctx = self.ctx_norm(torch.cat([mean, att], -1))
            scal = torch.stack([q_isl.to(n.dtype) / 4.0, pool.count(n.dtype).pow(-1.0 / 3.0)], -1)
        a, b = (self.ctx_in(torch.cat([ctx, scal], -1)) + self.atom_in(n)).chunk(2, -1)
        return s + self.out(F.silu(a) * b)

    def forward_sparse(self, s: Tensor, recv: Tensor, send: Tensor, w_edge: Tensor, q_isl: Tensor | None = None) -> Tensor:
        """Convenience sparse window forward: s [N, D], recv/send [E_lr], w_edge [E_lr]."""
        N = s.shape[0]
        pool = IslandPool.sparse_window(recv, send, w_edge, N)
        if q_isl is None:
            q_isl = s.new_zeros(N)
        return self.forward(s, pool, q_isl)

    def forward_dense(self, s: Tensor, W_ij: Tensor, mask: Tensor | None = None, q_isl: Tensor | None = None) -> Tensor:
        """Convenience dense window forward: s [B, N, D], W_ij [B, N, N] (including self-diagonal 1 on real atoms)."""
        B, N, _ = s.shape
        if mask is None:
            mask = torch.ones((B, N), dtype=torch.bool, device=s.device)
        pool = IslandPool.dense_matrix(W_ij, mask)
        if q_isl is None:
            q_isl = s.new_zeros(B, N)
        return self.forward(s, pool, q_isl) * mask.unsqueeze(-1).to(s.dtype)


class IslandContextBlock(IslandContext):
    """Windowed Island Context block with light long-range edge projection (default lr_dim=16)."""

    def __init__(
        self,
        d_model: int,
        dim: int = 32,
        heads: int = 4,
        hidden: int = 64,
        lr_dim: int = 16,
        mode: str = "window",
    ):
        super().__init__(d_model=d_model, dim=dim, heads=heads, hidden=hidden, lr_dim=lr_dim, mode=mode)

