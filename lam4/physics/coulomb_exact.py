"""coulomb_exact.py: dense smeared Coulomb for open systems (training N <= cap, tests, small inference).

Kernel K(r) = erf(r / sigma) / r (Gaussian-smeared charges, exact 1/r tail, finite at 0), sigma = 2.0 A:
    phi_i = sum_{j != i} q_j K(r_ij)                 [e / A]
    E_i   = -grad_i phi_i (q fixed) = sum_j q_j Kvec_ij,   Kvec_ij = -K'(r) (r_i - r_j) / r      [e / A^2]
    U_g   = 1/2 k_C sum_{i in g} q_i phi_i,           k_C = 332.06371 kcal A / (mol e^2)
Same maths as earlier Mena electrostatics.ExactCoulomb, from a GraphBatch: static shapes depend only on (N, G, cap).
Pad atoms (atom_mask False) own no slot: they are neither sources nor targets, and phi / field / energy rows of pad
atoms and pad graphs are exactly 0. Graphs are one group each; only atoms of the same graph interact.

Signatures
    make_slots_static(atom_mask [N], graph [N], n_graphs, cap) -> SlotLayout      (atoms sorted by graph id)
    assert_capacity(atom_mask, graph, n_graphs, cap)     eager check for the data pipeline (host sync, not traced)
    ExactCoulomb(pos [N, 3], layout, sigma=2.0)          or ExactCoulomb.from_batch(batch, cap=350, sigma=2.0)
        .phi(q [N]) -> [N];  .phi_field(q [N]) -> (phi [N], field [N, 3])
    coulomb_energy(q [N], phi [N], graph [N], n_graphs, atom_mask=None) -> [G]
A real graph with more than `cap` atoms has its overflow atoms dropped from the electrostatics (never written into
another graph's slots); size the cap from the data (spec: 350) and call assert_capacity once per batch in the loader.
Not ported: PMECoulomb / mesh (batched SPME and single-box PME are separate tasks).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from lam4.core.consts import capturing, is_fake, proxy_tracing
from lam4.core.radial import fp32_guard, upcast_fp32

COULOMB_K = 332.06371  # kcal A / (mol e^2)
SIGMA = 2.0
DEFAULT_CAP = 350
_META_CACHE: dict[tuple, tuple[int, int]] = {}


@dataclass
class SlotLayout:
    """slots [B, cap]: atom index per slot (N = sentinel for an empty slot); atom_slot [N]: flat slot (b * cap + k)
    of every atom (pad atoms: 0, never read); atom_mask [N] bool: atoms that own a slot."""

    slots: Tensor
    atom_slot: Tensor
    atom_mask: Tensor

    @property
    def cap(self) -> int:
        return self.slots.shape[1]


def make_slots_static(atom_mask: Tensor, graph: Tensor, n_graphs: int, cap: int) -> SlotLayout:
    N, G = graph.shape[0], n_graphs
    real = atom_mask.bool()
    ri = real.long()
    counts = torch.zeros(G, dtype=torch.long, device=graph.device).index_add(0, graph, ri)
    start = torch.cumsum(counts, 0) - counts
    rank = torch.cumsum(ri, 0) - 1 - start[graph]              # rank among the real atoms of its graph
    keep = real & (rank < cap)
    trash = G * cap
    flat = torch.where(keep, graph * cap + rank, torch.full_like(graph, trash))
    slots = torch.full((trash + 1,), N, dtype=torch.long, device=graph.device)
    slots = slots.scatter(0, flat, torch.arange(N, device=graph.device))
    return SlotLayout(slots[:trash].view(G, cap), torch.where(keep, flat, torch.zeros_like(flat)), keep)


def assert_capacity(atom_mask: Tensor, graph: Tensor, n_graphs: int, cap: int) -> None:
    counts = torch.zeros(int(n_graphs), dtype=torch.long, device=graph.device).index_add(0, graph, atom_mask.long())
    if int(counts.max()) > cap:
        raise ValueError(f"a graph has {int(counts.max())} real atoms, above the exact-Coulomb cap {cap}")


def resolve_coulomb_meta(batch, default_cap: int = DEFAULT_CAP) -> tuple[int, int]:
    """Return `(eff_n_graphs, eff_cap)` without silent `cap=350` atom truncation or `n_graphs=2` pad-graph misrouting.

    - When `batch.coulomb_cap` is explicitly set, honors `(batch.n_graphs, batch.coulomb_cap)` unless at most 1 graph has real atoms
      and `batch.coulomb_cap >= max_real_n`.
    - When `batch.coulomb_cap is None`, inspects real atom counts per graph so single-molecule padded batches (`MDEngine` `n_graphs=2`
      with 1 pad graph) collapse to `eff_n_graphs = 1`, and multi-graph batches with `>350` atoms never drop atoms (`eff_cap >= max_real_n`).
    - Caches by `(device, data_ptr, _version, n_atoms, n_graphs)` so `MDEngine` eager steps and `CUDAGraph` capture incur zero host syncs.
    """
    exp_cap = getattr(batch, "coulomb_cap", None)
    am, gr = batch.atom_mask, batch.graph
    if torch.compiler.is_compiling() or proxy_tracing() or is_fake(am):
        eff_g = getattr(batch, "_coulomb_eff_g", None)
        return (int(eff_g) if eff_g is not None else batch.n_graphs), (exp_cap if exp_cap is not None else batch.n_atoms)
    G = int(batch.n_graphs)
    N = int(batch.n_atoms)
    if G <= 1:
        return 1, (int(exp_cap) if exp_cap is not None else max(int(default_cap), N))
    if exp_cap is not None:
        return G, int(exp_cap)
    dev = am.device
    key = (dev.type, dev.index, am.data_ptr(), gr.data_ptr(), am._version, gr._version, N, G)
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    if capturing():
        return G, max(int(default_cap), N)
    counts = torch.zeros(G, dtype=torch.long, device=dev).index_add(0, gr, am.to(torch.long))
    n_real_g, max_n = torch.stack(((counts > 0).sum(), counts.max())).tolist()
    res = (1 if int(n_real_g) <= 1 else G, max(int(default_cap), int(max_n)))
    if len(_META_CACHE) >= 128:
        _META_CACHE.clear()
    _META_CACHE[key] = res
    return res


def smeared_kernel(r: Tensor, sigma: float) -> tuple[Tensor, Tensor]:
    """(K(r), K'(r)) for r > 0."""
    with fp32_guard(r.device):
        rf = upcast_fp32(r)
        x = rf / sigma
        erf = torch.erf(x)
        k = erf / rf
        dk = (2.0 / (sigma * math.sqrt(math.pi))) * torch.exp(-x * x) / rf - erf / (rf * rf)
        return k, dk


class ExactCoulomb:
    def __init__(self, pos: Tensor, layout: SlotLayout, sigma: float = SIGMA) -> None:
        with fp32_guard(pos.device):
            posf = upcast_fp32(pos)
            N = posf.shape[0]
            self.layout, self.n_atoms = layout, N
            slots = layout.slots
            pos_ext = torch.cat([posf, posf.new_zeros(1, 3)], 0)
            P = pos_ext.index_select(0, slots.reshape(-1)).view(*slots.shape, 3)  # [B, c, 3]
            valid = slots < N
            c = slots.shape[1]
            pm = valid.unsqueeze(2) & valid.unsqueeze(1) & ~torch.eye(c, dtype=torch.bool, device=posf.device).unsqueeze(0)
            diff = P.unsqueeze(2) - P.unsqueeze(1)  # [B, c, c, 3]
            r2 = diff.pow(2).sum(-1)
            r = torch.where(pm, r2, torch.ones_like(r2)).sqrt()
            k, dk = smeared_kernel(r, sigma)
            pmf = pm.to(posf.dtype)
            self.K = k * pmf  # [B, c, c]
            self.Kvec = (-(dk / r) * pmf).unsqueeze(-1) * diff  # [B, c, c, 3]
            self._m = layout.atom_mask.to(posf.dtype)

    @classmethod
    def from_batch(cls, batch, cap: int | None = None, sigma: float = SIGMA) -> "ExactCoulomb":
        if cap is None:
            eff_g, eff_cap = resolve_coulomb_meta(batch, DEFAULT_CAP)
            eff_graph = torch.zeros_like(batch.graph) if (eff_g == 1 and batch.n_graphs > 1) else batch.graph
            return cls(batch.pos, make_slots_static(batch.atom_mask, eff_graph, eff_g, eff_cap), sigma)
        c = cap if (torch.compiler.is_compiling() or proxy_tracing() or is_fake(batch.pos)) else int(cap)
        return cls(batch.pos, make_slots_static(batch.atom_mask, batch.graph, batch.n_graphs, c), sigma)

    def _to_slots(self, x: Tensor) -> Tensor:
        pad = x.new_zeros((1,) + tuple(x.shape[1:]))
        return torch.cat([x, pad], 0).index_select(0, self.layout.slots.reshape(-1)).view(*self.layout.slots.shape, *x.shape[1:])

    def phi(self, q: Tensor) -> Tensor:
        with fp32_guard(q.device):
            qf = upcast_fp32(q).to(self.K.dtype)
            qs = self._to_slots(qf)
            phi = torch.einsum("bij,bj->bi", self.K, qs).reshape(-1)
            return phi.index_select(0, self.layout.atom_slot) * self._m

    def phi_field(self, q: Tensor) -> tuple[Tensor, Tensor]:
        with fp32_guard(q.device):
            qf = upcast_fp32(q).to(self.K.dtype)
            qs = self._to_slots(qf)  # [B, c]
            phi = torch.einsum("bij,bj->bi", self.K, qs).reshape(-1)
            fld = torch.einsum("bijc,bj->bic", self.Kvec, qs).reshape(-1, 3)
            idx = self.layout.atom_slot
            return phi.index_select(0, idx) * self._m, fld.index_select(0, idx) * self._m.unsqueeze(-1)


def coulomb_energy(q: Tensor, phi: Tensor, graph: Tensor, n_graphs: int, atom_mask: Tensor | None = None) -> Tensor:
    """U [n_graphs] = 1/2 k_C sum_i q_i phi_i (pad atoms and pad graphs contribute exactly 0)."""
    with fp32_guard(q.device):
        qf, phif = upcast_fp32(q), upcast_fp32(phi)
        out_dt = qf.dtype
        acc_dt = torch.float64 if qf.device.type != "mps" else out_dt
        t = qf.to(acc_dt) * phif.to(acc_dt)
        if atom_mask is not None:
            t = t * atom_mask.to(acc_dt)
        out = t.new_zeros(n_graphs).index_add(0, graph, t) * (0.5 * COULOMB_K)
        return out.to(out_dt) if out.dtype != out_dt else out


class _ChunkedCoulombFn(torch.autograd.Function):
    """Row-chunked O(N * chunk) smeared Coulomb forward + analytical backward for CPU/CUDA molecules (G >= 1)."""

    @staticmethod
    def forward(ctx, pos: Tensor, q: Tensor, mask_b: Tensor, graph: Tensor | None, sigma: float,
                need_field: bool, chunk: int = 512) -> tuple[Tensor, Tensor]:
        with fp32_guard(pos.device):
            pos, q = upcast_fp32(pos), upcast_fp32(q).to(upcast_fp32(pos).dtype)
            N = int(pos.shape[0])
            mf = mask_b.to(pos.dtype)
            qm = q * mf
            phi = pos.new_zeros(N)
            fld = pos.new_zeros((N, 3)) if need_field else pos.new_empty((0, 3))
            inv_sig = 1.0 / float(sigma)
            c0 = 2.0 * inv_sig / math.sqrt(math.pi)
            for i0 in range(0, N, chunk):
                i1 = min(i0 + chunk, N)
                d = pos[i0:i1].unsqueeze(1) - pos.unsqueeze(0)  # [C, N, 3]
                pm = mask_b[i0:i1].unsqueeze(1) & mask_b.unsqueeze(0)
                if graph is not None:
                    pm = pm & (graph[i0:i1].unsqueeze(1) == graph.unsqueeze(0))
                pm[:, i0:i1] &= ~torch.eye(i1 - i0, dtype=torch.bool, device=pos.device)
                r2 = torch.where(pm, d.pow(2).sum(-1), torch.ones((i1 - i0, N), dtype=pos.dtype, device=pos.device))
                r = r2.sqrt()
                k = torch.erf(r * inv_sig) / r * pm.to(pos.dtype)
                phi[i0:i1] = k @ qm
                if need_field:
                    f1 = ((k - c0 * torch.exp(-(r * inv_sig).pow(2)) * pm.to(pos.dtype)) / r2)
                    fld[i0:i1] = torch.einsum("ij,ijc,j->ic", f1, d, qm)
        ctx.has_graph = graph is not None
        ctx.save_for_backward(pos, q, mask_b, *( () if graph is None else (graph,) ))
        ctx.sigma = float(sigma)
        ctx.need_field = bool(need_field)
        ctx.chunk = int(chunk)
        return phi, fld

    @staticmethod
    @once_differentiable
    def backward(ctx, g_phi: Tensor | None, g_fld: Tensor | None):
        saved = ctx.saved_tensors
        pos, q, mask_b = saved[0], saved[1], saved[2]
        graph = saved[3] if ctx.has_graph else None
        need_dp = bool(ctx.needs_input_grad[0])
        need_dq = bool(ctx.needs_input_grad[1])
        N = int(pos.shape[0])
        if not (need_dp or need_dq) or N == 0:
            return (torch.zeros_like(pos) if need_dp else None), (torch.zeros_like(q) if need_dq else None), None, None, None, None, None
        with fp32_guard(pos.device):
            mf = mask_b.to(pos.dtype)
            qm = q * mf
            gp = (upcast_fp32(g_phi).to(pos.dtype) * mf) if g_phi is not None else pos.new_zeros(N)
            has_gf = bool(ctx.need_field and g_fld is not None)
            ge = (upcast_fp32(g_fld).to(pos.dtype) * mf.unsqueeze(-1)) if has_gf else pos.new_zeros((N, 3))
            d_pos = pos.new_zeros((N, 3)) if need_dp else None
            d_q = pos.new_zeros(N) if need_dq else None
            inv_sig = 1.0 / ctx.sigma
            inv_sig2 = inv_sig * inv_sig
            c0 = 2.0 * inv_sig / math.sqrt(math.pi)
            chunk = ctx.chunk
            for i0 in range(0, N, chunk):
                i1 = min(i0 + chunk, N)
                d = pos[i0:i1].unsqueeze(1) - pos.unsqueeze(0)
                pm = mask_b[i0:i1].unsqueeze(1) & mask_b.unsqueeze(0)
                if graph is not None:
                    pm = pm & (graph[i0:i1].unsqueeze(1) == graph.unsqueeze(0))
                pm[:, i0:i1] &= ~torch.eye(i1 - i0, dtype=torch.bool, device=pos.device)
                pmf = pm.to(pos.dtype)
                r2 = torch.where(pm, d.pow(2).sum(-1), torch.ones((i1 - i0, N), dtype=pos.dtype, device=pos.device))
                r = r2.sqrt()
                ex = torch.exp(-r2 * inv_sig2)
                k = (torch.erf(r * inv_sig) / r) * pmf
                f1 = ((k - c0 * ex * pmf) / r2) * pmf
                if need_dq:
                    if has_gf:
                        d_q[i0:i1] = k @ gp - (f1 * (d * ge.unsqueeze(0)).sum(-1)).sum(dim=1)
                    else:
                        d_q[i0:i1] = k @ gp
                if need_dp:
                    s_phi = gp[i0:i1].unsqueeze(1) * qm.unsqueeze(0) + qm[i0:i1].unsqueeze(1) * gp.unsqueeze(0)
                    if has_gf:
                        f2 = ((2.0 * c0 * inv_sig2 * ex * pmf - 3.0 * f1) / r2) * pmf
                        v = qm.view(1, N, 1) * ge[i0:i1].unsqueeze(1) - qm[i0:i1].view(-1, 1, 1) * ge.unsqueeze(0)
                        vd = (v * d).sum(-1)
                        d_pos[i0:i1] = ((-s_phi * f1 + f2 * vd).unsqueeze(-1) * d + f1.unsqueeze(-1) * v).sum(dim=1)
                    else:
                        d_pos[i0:i1] = ((-s_phi * f1).unsqueeze(-1) * d).sum(dim=1)
        return d_pos, d_q, None, None, None, None, None


class ChunkedExactCoulomb:
    """O(N * chunk) memory fallback for large open molecules or multi-graph batches on CPU/CUDA."""

    def __init__(self, pos: Tensor, atom_mask: Tensor, sigma: float = SIGMA, chunk: int = 512, *,
                 graph: Tensor | None = None, n_graphs: int = 1, cap: int | None = None) -> None:
        self.pos = pos
        mask_b = atom_mask.bool()
        sym = torch.compiler.is_compiling() or proxy_tracing() or is_fake(atom_mask)
        if cap is not None and (sym or int(cap) < int(mask_b.numel())):
            ri = mask_b.long()
            gr = torch.zeros_like(ri) if graph is None else graph
            g_cnt = n_graphs if sym else max(1, int(n_graphs))
            counts = torch.zeros(g_cnt, dtype=torch.long, device=pos.device).index_add(0, gr, ri)
            start = torch.cumsum(counts, 0) - counts
            rank = torch.cumsum(ri, 0) - 1 - start[gr]
            mask_b = mask_b & (rank < cap)
        self.mask_b = mask_b
        self.graph = None if n_graphs <= 1 else graph
        self.sigma = float(sigma)
        self.chunk = int(chunk)

    def phi(self, q: Tensor) -> Tensor:
        phi, _ = _ChunkedCoulombFn.apply(self.pos, q, self.mask_b, self.graph, self.sigma, False, self.chunk)
        return phi

    def phi_field(self, q: Tensor) -> tuple[Tensor, Tensor]:
        phi, fld = _ChunkedCoulombFn.apply(self.pos, q, self.mask_b, self.graph, self.sigma, True, self.chunk)
        return phi, fld

    def phi_field_efg(self, q: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return _ChunkedCoulombEfgFn.apply(self.pos, q, self.mask_b, self.graph, self.sigma, self.chunk)

    def phi_field_multipole(self, q: Tensor, p: Tensor | None = None, *, need_efg: bool = False) -> tuple[Tensor, Tensor, Tensor | None]:
        if p is not None:
            raise ValueError("ChunkedExactCoulomb.phi_field_multipole supports monopole-only (p=None); use OpenMultipoleBackend when sqe_pqeq=True.")
        if need_efg:
            return self.phi_field_efg(q)
        phi, fld = self.phi_field(q)
        return phi, fld, None


class _ChunkedCoulombEfgFn(torch.autograd.Function):
    """Row-chunked O(N * chunk) smeared Coulomb (phi, field, L=2 EFG) + 1st-order backward."""

    @staticmethod
    def forward(ctx, pos: Tensor, q: Tensor, mask_b: Tensor, graph: Tensor | None, sigma: float,
                chunk: int = 512) -> tuple[Tensor, Tensor, Tensor]:
        from lam4.core.angular import l2_pair
        from lam4.physics.sqe_pqeq import _smeared_k_f1_f2

        with fp32_guard(pos.device):
            pos_f = upcast_fp32(pos)
            q_f = upcast_fp32(q).to(pos_f.dtype)
            N = int(pos_f.shape[0])
            qm = q_f * mask_b.to(pos_f.dtype)
            phi = pos_f.new_zeros(N)
            fld = pos_f.new_zeros((N, 3))
            efg = pos_f.new_zeros((N, 5))
            for i0 in range(0, N, chunk):
                i1 = min(i0 + chunk, N)
                d = pos_f[i0:i1].unsqueeze(1) - pos_f.unsqueeze(0)
                pm = mask_b[i0:i1].unsqueeze(1) & mask_b.unsqueeze(0)
                if graph is not None:
                    pm = pm & (graph[i0:i1].unsqueeze(1) == graph.unsqueeze(0))
                pm[:, i0:i1] &= ~torch.eye(i1 - i0, dtype=torch.bool, device=pos_f.device)
                k, f1, f2, _ = _smeared_k_f1_f2(d.pow(2).sum(-1), pm, sigma, need_f3=False)
                phi[i0:i1] = k @ qm
                fld[i0:i1] = torch.einsum("ij,ijc,j->ic", f1, d, qm)
                efg[i0:i1] = torch.einsum("ij,ijm,j->im", f2, l2_pair(d, d), qm)
        ctx.has_graph = graph is not None
        ctx.save_for_backward(pos, q, mask_b, *( () if graph is None else (graph,) ))
        ctx.sigma, ctx.chunk = float(sigma), int(chunk)
        return phi, fld, efg

    @staticmethod
    @once_differentiable
    def backward(ctx, g_phi: Tensor | None, g_fld: Tensor | None, g_efg: Tensor | None):
        from lam4.core.angular import l2_mat6
        from lam4.physics.sqe_pqeq import _smeared_k_f1_f2

        saved = ctx.saved_tensors
        pos, q, mask_b = saved[0], saved[1], saved[2]
        graph = saved[3] if ctx.has_graph else None
        need_dp, need_dq = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        N = int(pos.shape[0])
        if not (need_dp or need_dq) or N == 0:
            return (torch.zeros_like(pos) if need_dp else None), (torch.zeros_like(q) if need_dq else None), None, None, None, None
        with fp32_guard(pos.device):
            pos_f = upcast_fp32(pos)
            q_f = upcast_fp32(q).to(pos_f.dtype)
            mf = mask_b.to(pos_f.dtype)
            qm = q_f * mf
            gp = (upcast_fp32(g_phi).to(pos_f.dtype) * mf) if g_phi is not None else pos_f.new_zeros(N)
            has_gf, has_ge = (g_fld is not None), (g_efg is not None)
            gf = (upcast_fp32(g_fld).to(pos_f.dtype) * mf.unsqueeze(-1)) if has_gf else pos_f.new_zeros((N, 3))
            if has_ge:
                gm_raw = torch.stack(l2_mat6(upcast_fp32(g_efg).to(pos_f.dtype) * mf.unsqueeze(-1)), dim=-1)
                gm = torch.stack([
                    torch.stack([gm_raw[:, 0], gm_raw[:, 3], gm_raw[:, 4]], -1),
                    torch.stack([gm_raw[:, 3], gm_raw[:, 1], gm_raw[:, 5]], -1),
                    torch.stack([gm_raw[:, 4], gm_raw[:, 5], gm_raw[:, 2]], -1),
                ], dim=-2)
            d_pos = pos_f.new_zeros((N, 3)) if need_dp else None
            d_q = pos_f.new_zeros(N) if need_dq else None
            for i0 in range(0, N, ctx.chunk):
                i1 = min(i0 + ctx.chunk, N)
                d = pos_f[i0:i1].unsqueeze(1) - pos_f.unsqueeze(0)
                pm = mask_b[i0:i1].unsqueeze(1) & mask_b.unsqueeze(0)
                if graph is not None:
                    pm = pm & (graph[i0:i1].unsqueeze(1) == graph.unsqueeze(0))
                pm[:, i0:i1] &= ~torch.eye(i1 - i0, dtype=torch.bool, device=pos_f.device)
                k, f1, f2, f3 = _smeared_k_f1_f2(d.pow(2).sum(-1), pm, ctx.sigma, need_f3=has_ge and need_dp)
                if has_ge:
                    mdj = torch.einsum("jab,ijb->ija", gm, d)
                    sj = (mdj * d).sum(-1)
                if need_dq:
                    dq_c = k @ gp
                    if has_gf:
                        dq_c = dq_c - (f1 * (d * gf.unsqueeze(0)).sum(-1)).sum(1)
                    if has_ge:
                        dq_c = dq_c + (f2 * sj).sum(1)
                    d_q[i0:i1] = dq_c
                if need_dp:
                    cd = -(gp[i0:i1].unsqueeze(1) * qm.unsqueeze(0) + qm[i0:i1].unsqueeze(1) * gp.unsqueeze(0)) * f1
                    v = pos_f.new_zeros((i1 - i0, N, 3))
                    if has_gf:
                        v = qm.view(1, N, 1) * gf[i0:i1].unsqueeze(1) - qm[i0:i1].view(-1, 1, 1) * gf.unsqueeze(0)
                        cd = cd + f2 * (v * d).sum(-1)
                    g_vec = pos_f.new_zeros((i1 - i0, N, 3))
                    if has_ge:
                        mdi = torch.einsum("iab,ijb->ija", gm[i0:i1], d)
                        si = (mdi * d).sum(-1)
                        cd = cd + f3 * (qm.view(1, N) * si + qm[i0:i1].view(-1, 1) * sj)
                        g_vec = (2.0 * f2).unsqueeze(-1) * (qm.view(1, N, 1) * mdi + qm[i0:i1].view(-1, 1, 1) * mdj)
                    d_pos[i0:i1] = (cd.unsqueeze(-1) * d + f1.unsqueeze(-1) * v + g_vec).sum(dim=1)
        return (d_pos.to(pos.dtype) if need_dp else None), (d_q.to(q.dtype) if need_dq else None), None, None, None, None
