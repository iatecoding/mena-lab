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

COULOMB_K = 332.06371  # kcal A / (mol e^2)
SIGMA = 2.0
DEFAULT_CAP = 350


@dataclass
class SlotLayout:
    """slots [B, cap]: atom index per slot (N = sentinel for an empty slot); atom_slot [N]: flat slot (b * cap + k)
    of every atom (pad atoms: 0, never read); atom_mask [N] bool: atoms that own a slot."""

    slots: Tensor
    atom_slot: Tensor
    atom_mask: Tensor

    @property
    def cap(self) -> int:
        return int(self.slots.shape[1])


def make_slots_static(atom_mask: Tensor, graph: Tensor, n_graphs: int, cap: int) -> SlotLayout:
    N, G = graph.shape[0], int(n_graphs)
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


def smeared_kernel(r: Tensor, sigma: float) -> tuple[Tensor, Tensor]:
    """(K(r), K'(r)) for r > 0."""
    x = r / sigma
    erf = torch.erf(x)
    k = erf / r
    dk = (2.0 / (sigma * math.sqrt(math.pi))) * torch.exp(-x * x) / r - erf / (r * r)
    return k, dk


class ExactCoulomb:
    def __init__(self, pos: Tensor, layout: SlotLayout, sigma: float = SIGMA) -> None:
        N = pos.shape[0]
        self.layout, self.n_atoms = layout, N
        slots = layout.slots
        pos_ext = torch.cat([pos, pos.new_zeros(1, 3)], 0)
        P = pos_ext.index_select(0, slots.reshape(-1)).view(*slots.shape, 3)  # [B, c, 3]
        valid = slots < N
        c = slots.shape[1]
        pm = valid.unsqueeze(2) & valid.unsqueeze(1) & ~torch.eye(c, dtype=torch.bool, device=pos.device).unsqueeze(0)
        diff = P.unsqueeze(2) - P.unsqueeze(1)  # [B, c, c, 3]
        r2 = diff.pow(2).sum(-1)
        r = torch.where(pm, r2, torch.ones_like(r2)).sqrt()
        k, dk = smeared_kernel(r, sigma)
        pmf = pm.to(pos.dtype)
        self.K = k * pmf  # [B, c, c]
        self.Kvec = (-(dk / r) * pmf).unsqueeze(-1) * diff  # [B, c, c, 3]
        self._m = layout.atom_mask.to(pos.dtype)

    @classmethod
    def from_batch(cls, batch, cap: int = DEFAULT_CAP, sigma: float = SIGMA) -> "ExactCoulomb":
        return cls(batch.pos, make_slots_static(batch.atom_mask, batch.graph, batch.n_graphs, cap), sigma)

    def _to_slots(self, x: Tensor) -> Tensor:
        pad = x.new_zeros((1,) + tuple(x.shape[1:]))
        return torch.cat([x, pad], 0).index_select(0, self.layout.slots.reshape(-1)).view(*self.layout.slots.shape, *x.shape[1:])

    def phi(self, q: Tensor) -> Tensor:
        qs = self._to_slots(q)
        phi = torch.einsum("bij,bj->bi", self.K, qs).reshape(-1)
        return phi.index_select(0, self.layout.atom_slot) * self._m

    def phi_field(self, q: Tensor) -> tuple[Tensor, Tensor]:
        qs = self._to_slots(q)  # [B, c]
        phi = torch.einsum("bij,bj->bi", self.K, qs).reshape(-1)
        fld = torch.einsum("bijc,bj->bic", self.Kvec, qs).reshape(-1, 3)
        idx = self.layout.atom_slot
        return phi.index_select(0, idx) * self._m, fld.index_select(0, idx) * self._m.unsqueeze(-1)


def coulomb_energy(q: Tensor, phi: Tensor, graph: Tensor, n_graphs: int, atom_mask: Tensor | None = None) -> Tensor:
    """U [n_graphs] = 1/2 k_C sum_i q_i phi_i (pad atoms and pad graphs contribute exactly 0)."""
    t = q * phi
    if atom_mask is not None:
        t = t * atom_mask.to(t.dtype)
    return q.new_zeros(int(n_graphs)).index_add(0, graph, t) * (0.5 * COULOMB_K)


class _ChunkedCoulombFn(torch.autograd.Function):
    """Row-chunked O(N * chunk) smeared Coulomb forward + analytical backward for CPU/CUDA large molecules (G=1)."""

    @staticmethod
    def forward(ctx, pos: Tensor, q: Tensor, mask_b: Tensor, sigma: float, need_field: bool, chunk: int = 512) -> tuple[Tensor, Tensor]:
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
            pm[:, i0:i1] &= ~torch.eye(i1 - i0, dtype=torch.bool, device=pos.device)
            r2 = torch.where(pm, d.pow(2).sum(-1), torch.ones((i1 - i0, N), dtype=pos.dtype, device=pos.device))
            r = r2.sqrt()
            k = torch.erf(r * inv_sig) / r * pm.to(pos.dtype)
            phi[i0:i1] = k @ qm
            if need_field:
                f1 = ((k - c0 * torch.exp(-(r * inv_sig).pow(2)) * pm.to(pos.dtype)) / r2)
                fld[i0:i1] = torch.einsum("ij,ijc,j->ic", f1, d, qm)
        ctx.save_for_backward(pos, q, mask_b)
        ctx.sigma = float(sigma)
        ctx.need_field = bool(need_field)
        ctx.chunk = int(chunk)
        return phi, fld

    @staticmethod
    @once_differentiable
    def backward(ctx, g_phi: Tensor | None, g_fld: Tensor | None):
        pos, q, mask_b = ctx.saved_tensors
        need_dp = bool(ctx.needs_input_grad[0])
        need_dq = bool(ctx.needs_input_grad[1])
        N = int(pos.shape[0])
        if not (need_dp or need_dq) or N == 0:
            return (torch.zeros_like(pos) if need_dp else None), (torch.zeros_like(q) if need_dq else None), None, None, None, None
        mf = mask_b.to(pos.dtype)
        qm = q * mf
        gp = (g_phi * mf) if g_phi is not None else pos.new_zeros(N)
        has_gf = bool(ctx.need_field and g_fld is not None)
        ge = (g_fld * mf.unsqueeze(-1)) if has_gf else pos.new_zeros((N, 3))
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
        return d_pos, d_q, None, None, None, None


class ChunkedExactCoulomb:
    """O(N * chunk) memory fallback for large open molecules on CPU/CUDA."""

    def __init__(self, pos: Tensor, atom_mask: Tensor, sigma: float = SIGMA, chunk: int = 512) -> None:
        self.pos = pos
        self.mask_b = atom_mask.bool()
        self.sigma = float(sigma)
        self.chunk = int(chunk)

    def phi(self, q: Tensor) -> Tensor:
        phi, _ = _ChunkedCoulombFn.apply(self.pos, q, self.mask_b, self.sigma, False, self.chunk)
        return phi

    def phi_field(self, q: Tensor) -> tuple[Tensor, Tensor]:
        return _ChunkedCoulombFn.apply(self.pos, q, self.mask_b, self.sigma, True, self.chunk)
