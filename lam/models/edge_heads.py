"""edge_heads.py: Multi-head edge factor on top of ChemicalEdgeWeights (Mena-LAM-3 R2).

Per stage, modulates each edge message with H learned positive factors
    ell^h_ij = q_h(s_hat_i) . k_h(s_hat_j) / sqrt(d_head) + rb_h(rbf_ij),
    phi^h_ij = exp(ell^h_ij) / <exp(ell^h)>_{env, i},
where s_hat_i = norm_in(s_i) is the stage's RMS-normalised scalar state and <exp(ell^h)>_{env, i} is the
cutoff-envelope-weighted mean over i's neighbours (including a unit self-reference weight w_ii = 1 with ell_ii = 0,
matching the smooth degree (1 + sum_j chi_ij) of ChemicalEdgeWeights):
    <exp(ell^h)>_{env, i} = (1 + sum_j chi_ij exp(ell^h_ij)) / (1 + sum_j chi_ij)
                          = 1 + (sum_j chi_ij (exp(ell^h_ij) - 1)) / (1 + sum_j chi_ij).

Properties:
- Identity at initialisation: `q` and `rb` are zero-initialised while `k` uses standard random initialisation (the
  bilinear init pattern of MenaACE: one random factor, one zero factor). At init, `ell^h_ij == 0` and `phi^h_ij == 1`
  identically to machine precision, while both `q` and `rb` receive non-zero gradients on the first training step and
  `k` on the second.
- C^3 smoothness at r_cut: as a neighbour j crosses r_cut, chi_ij -> 0 smoothly with zero derivatives up to third order,
  and the unit self-reference keeps the denominator >= 1/(1 + deg) > 0 even for an isolated or 1-neighbour atom.
- Channel split rule: each head h in 0..H-1 scales the h-th contiguous 1/H block of channels of every edge message in
  the stage: scalar convolution (`msg_dim`), l=1 density (`K`), l=1 full and longitudinal transport (`K`), l=2 density
  (`K2`), and l=3 density (`L3`). All active channel counts must be divisible by `H`.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import lam.models.edge_ops as eo

LOGIT_CLAMP = 20.0


class _SparseQKPairs(torch.autograd.Function):
    """Per-edge dot products `(q[recv] * k[send]) @ P` saving atom-level `q, k` (`[N, H*Dh]`) instead of edge-level
    `qi, kj` (`[E, H*Dh]`) in the autograd graph."""

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, recv: Tensor, send: Tensor, P: Tensor) -> Tensor:
        ctx.save_for_backward(q, k, recv, send, P)
        qi = q.index_select(0, recv)
        kj = k.index_select(0, send)
        return (qi * kj) @ P

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        q, k, recv, send, P = ctx.saved_tensors
        g_ek = grad_out @ P.t()
        dq = dk = None
        if ctx.needs_input_grad[0]:
            kj = k.index_select(0, send)
            dq = q.new_zeros(q.shape).index_add(0, recv, g_ek * kj)
        if ctx.needs_input_grad[1]:
            qi = q.index_select(0, recv)
            dk = k.new_zeros(k.shape).index_add(0, send, g_ek * qi)
        return dq, dk, None, None, None


class _ScaleChannels(torch.autograd.Function):
    """Channel scaling `y = x * (phi @ P)` saving `(x, phi, P)` so the expanded `phi @ P` tensor is not retained in the
    autograd graph while preserving exact second derivatives (`create_graph=True`) for conservative forces."""

    @staticmethod
    def forward(ctx, x: Tensor, phi: Tensor, P: Tensor) -> Tensor:
        ctx.save_for_backward(x, phi, P)
        return x * (phi @ P)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, phi, P = ctx.saved_tensors
        dx = _ScaleChannels.apply(grad_out, phi, P) if ctx.needs_input_grad[0] else None
        dphi = (grad_out * x) @ P.t() if ctx.needs_input_grad[1] else None
        return dx, dphi, None


class _EdgeTPhi(torch.autograd.Function):
    """Per-head factorised dense edge product `edge_t.unsqueeze(0) * phi_h.unsqueeze(3)` without 5D `.permute` copies."""

    @staticmethod
    def forward(ctx, edge_t: Tensor, phi_h: Tensor) -> Tensor:
        ctx.save_for_backward(edge_t, phi_h)
        return edge_t.unsqueeze(0) * phi_h.unsqueeze(3)

    @staticmethod
    def backward(ctx, g: Tensor):
        edge_t, phi_h = ctx.saved_tensors
        ge = gp = None
        if ctx.needs_input_grad[0]:
            ge = (g * phi_h.unsqueeze(3)).sum(dim=0)
        if ctx.needs_input_grad[1]:
            gp = torch.einsum("hbirj,birj->hbij", g, edge_t)
        return ge, gp


class EdgeHeadFactor(nn.Module):
    """Per-stage multi-head edge factor (`core.stages_list.{i}.ehead.{q,k,rb}`)."""

    def __init__(self, d_model: int, num_rbf: int, heads: int = 4, d_head: int = 16) -> None:
        super().__init__()
        if int(heads) <= 0 or int(d_head) <= 0:
            raise ValueError(f"heads ({heads}) and d_head ({d_head}) must be positive")
        self.d_model = int(d_model)
        self.num_rbf = int(num_rbf)
        self.heads = int(heads)
        self.d_head = int(d_head)
        self.scale = 1.0 / math.sqrt(self.d_head)
        dim = self.heads * self.d_head
        self.q = nn.Linear(self.d_model, dim, bias=False)
        self.k = nn.Linear(self.d_model, dim, bias=False)
        self.rb = nn.Linear(self.num_rbf, self.heads, bias=False)
        nn.init.zeros_(self.q.weight)
        nn.init.zeros_(self.rb.weight)
        self._qk_proj: dict[tuple[torch.device, torch.dtype], Tensor] = {}
        self._ch_proj: dict[tuple[tuple[int, ...], torch.device, torch.dtype], Tensor] = {}

    def _get_qk_proj(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (device, dtype)
        proj = self._qk_proj.get(key)
        if proj is None:
            H, Dh = self.heads, self.d_head
            proj = torch.zeros(H * Dh, H, device=device, dtype=dtype)
            for h in range(H):
                proj[h * Dh : (h + 1) * Dh, h] = self.scale
            if not torch.compiler.is_compiling() and not isinstance(proj, torch._subclasses.FakeTensor):
                self._qk_proj[key] = proj
        return proj

    def _get_ch_proj(self, slices: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (slices, device, dtype)
        proj = self._ch_proj.get(key)
        if proj is None:
            H = self.heads
            proj = torch.zeros(H, sum(slices), device=device, dtype=dtype)
            off = 0
            for s in slices:
                if s % H != 0:
                    raise ValueError(f"channel count {s} must be divisible by edge_heads ({H})")
                sh = s // H
                for h in range(H):
                    proj[h, off + h * sh : off + (h + 1) * sh] = 1.0
                off += s
            if not torch.compiler.is_compiling() and not isinstance(proj, torch._subclasses.FakeTensor):
                self._ch_proj[key] = proj
        return proj

    def logits_dense(self, hs: Tensor, rbf: Tensor) -> Tensor:
        """Dense per-pair logits ell [B, N, N, H] from normed state hs [B, N, D] and rbf [B, N, N, R]."""
        B, N, _ = hs.shape
        H, Dh = self.heads, self.d_head
        qk = F.linear(hs, torch.cat([self.q.weight, self.k.weight], dim=0)).view(B, N, 2, H, Dh)
        q = qk[:, :, 0].permute(0, 2, 1, 3).reshape(B * H, N, Dh)
        k = qk[:, :, 1].permute(0, 2, 3, 1).reshape(B * H, Dh, N)
        pair = torch.bmm(q, k).view(B, H, N, N).permute(0, 2, 3, 1) * self.scale
        return (pair + self.rb(rbf)).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)

    def forward(self, hs: Tensor, rbf: Tensor, chi: Tensor, deg1: Tensor | None = None) -> Tensor:
        """Dense multi-head edge factor phi [B, N, N, H] from hs [B, N, D], rbf [B, N, N, R], and chi [B, N, N]."""
        ell = self.logits_dense(hs, rbf)
        exp_l = torch.exp(ell)
        if deg1 is None:
            deg1 = chi.sum(dim=-1, keepdim=True) + 1.0  # [B, N, 1]
        denom = ((chi.unsqueeze(-1) * exp_l).sum(dim=2) + 1.0) / deg1  # [B, N, H]
        return exp_l / denom.unsqueeze(2)

    def logits_sparse(self, hs: Tensor, rbf: Tensor, recv: Tensor, send: Tensor) -> Tensor:
        """Sparse per-edge logits ell [E, H] from normed state hs [N, D] and rbf [E, R]."""
        dim = self.heads * self.d_head
        qk = F.linear(hs, torch.cat([self.q.weight, self.k.weight], dim=0))
        q, k = qk.split(dim, dim=-1)
        pair = _SparseQKPairs.apply(q, k, recv, send, self._get_qk_proj(hs.device, hs.dtype))  # [E, H]
        return (pair + self.rb(rbf)).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)

    def forward_sparse(
        self,
        hs: Tensor,
        rbf: Tensor,
        chi: Tensor,
        recv: Tensor,
        send: Tensor,
        deg1: Tensor | None = None,
    ) -> Tensor:
        """Sparse multi-head edge factor phi [E, H] from hs [N, D], rbf [E, R], and chi [E]."""
        N = hs.shape[0]
        H = self.heads
        ell = self.logits_sparse(hs, rbf, recv, send)
        exp_l = torch.exp(ell)
        if deg1 is None:
            deg1 = hs.new_zeros(N, 1).index_add(0, recv, chi.unsqueeze(-1)) + 1.0  # [N, 1]
        num = hs.new_zeros(N, H).index_add(0, recv, chi.unsqueeze(-1) * exp_l) + 1.0  # [N, H]
        denom = num / deg1  # [N, H]
        return exp_l / denom.index_select(0, recv)

    def expand_phi(self, phi: Tensor, slices: Sequence[int]) -> Tensor:
        """Expand phi [..., H] to [..., sum(slices)] via cached [H, sum(slices)] 0/1 block matrix."""
        P = self._get_ch_proj(tuple(int(s) for s in slices), phi.device, phi.dtype)
        return phi @ P

    def scale_channels(self, x: Tensor, phi: Tensor, slices: Sequence[int] | None = None) -> Tensor:
        """Multiply each of the H contiguous 1/H channel blocks of x [..., C] (or each slice in `slices`) by phi [..., H]."""
        s_tuple = (x.shape[-1],) if slices is None else tuple(int(s) for s in slices)
        P = self._get_ch_proj(s_tuple, phi.device, phi.dtype)
        return _ScaleChannels.apply(x, phi, P)

    def scalar_conv_dense(
        self,
        edge: Tensor,
        edge_t: Tensor | None,
        sp: Tensor,
        w_sr: Tensor,
        phi: Tensor,
        small: bool = False,
        factorised: bool = True,
    ) -> Tensor:
        """Dense scalar continuous-filter convolution with per-head edge modulation phi [B, N, N, H].

        Algebraic identity for the factorised form (N > R):
            conv_{i, (h, c')} = sum_j phi^h_ij (sum_r edge_{i,j,r} W_sr[(h, c'), r]) sp_{j, (h, c')}
                              = sum_r W_sr[(h, c'), r] (sum_j (edge_{i,j,r} phi^h_ij) sp_{j, (h, c')}),
        applying phi^h_ij to edge before contracting over j in a single batched GEMM of the same total MAC count
        (B * N^2 * R * m) as the unheaded factorised path.
        """
        B, N, _, R = edge.shape
        H = self.heads
        m = sp.shape[-1]
        if m % H != 0:
            raise ValueError(f"scalar msg_dim {m} must be divisible by edge_heads ({H})")
        mh = m // H
        if not factorised or N <= R or edge_t is None:
            sr = self.scale_channels(F.linear(edge, w_sr), phi)
            if small:
                return (sr * sp.unsqueeze(1)).sum(dim=2)
            return torch.einsum("bijd,bjd->bid", sr, sp)
        # Factorised per-head contraction: [H*B, N*R, N] @ [H*B, N, m//H] -> [H, B*N, R, m//H]
        phi_h = phi.permute(3, 0, 1, 2).contiguous()
        ep_t = _EdgeTPhi.apply(edge_t.reshape(B, N, R, N), phi_h).view(H * B, N * R, N)
        sp_h = sp.view(B, N, H, mh).permute(2, 0, 1, 3).reshape(H * B, N, mh)
        y = torch.bmm(ep_t, sp_h).view(H, B * N, R, mh)
        w_h = w_sr.view(H, mh, R).transpose(1, 2).unsqueeze(1)  # [H, 1, R, mh]
        return (y * w_h).sum(dim=2).view(H, B, N, mh).permute(1, 2, 0, 3).reshape(B, N, m)
