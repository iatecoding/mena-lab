"""Pad a GraphBatch to fixed (atoms, edges, graphs) so one traced graph / CUDA graph serves every step.

Design rule: padding is a physical no-op, not a masked-out special case.
 * Pad atoms live in pad graphs, Z = 0, masks False, labels exactly 0 (never NaN).
 * Pad atoms sit on a line spaced exactly r_cut apart, starting at the origin, so the distance between
   consecutive pad atoms is exactly r_cut (computed exactly in fp32: 0 and multiples of r_cut).
 * Pad edges join consecutive pad atoms and stay receiver-sorted after the real edges (the streaming edge op needs that). The C3 envelope and its first two derivatives are zero at r_cut,
   so pad edges add exactly nothing to values or gradients, and no pair sits at distance 0 (no NaN in r_hat).
"""
from __future__ import annotations

import torch

from lam4.data.batch import GraphBatch


def pad_batch(b: GraphBatch, n_atoms: int, n_edges: int, n_graphs: int, r_cut: float) -> GraphBatch:
    n0, e0, g0 = b.n_atoms, b.n_edges, b.n_graphs
    pa, pe, pg = n_atoms - n0, n_edges - e0, n_graphs - g0
    if min(pa, pe, pg) < 0:
        raise ValueError(f"batch ({n0} atoms, {e0} edges, {g0} graphs) exceeds caps ({n_atoms}, {n_edges}, {n_graphs})")
    if pe > 0 and pa < 2:
        raise ValueError("padding edges needs at least 2 pad atoms")
    if pa > 0 and pg < 1:
        raise ValueError("pad atoms need at least one pad graph")
    dev, dt = b.pos.device, b.pos.dtype
    fl, lg = dict(device=dev, dtype=dt), dict(device=dev, dtype=torch.long)

    def cat(x, pad):
        return torch.cat([x, pad.to(x.dtype)], dim=0)

    pad_pos = torch.zeros(pa, 3, **fl)
    pad_pos[:, 0] = torch.arange(pa, **fl) * r_cut
    # consecutive pad atoms, receiver-sorted in O(pe) without argsort; distance is identically r_cut
    if pe > 0:
        r_local = (torch.arange(pe, **lg) * pa) // pe
        s_local = torch.where(r_local + 1 < pa, r_local + 1, r_local - 1)
        pad_ei = torch.stack([s_local + n0, r_local + n0])
    else:
        pad_ei = torch.zeros(2, 0, **lg)
    f0 = int(b.frag_q.shape[0])
    return GraphBatch(
        z=cat(b.z, torch.zeros(pa, **lg)),
        pos=cat(b.pos, pad_pos),
        graph=cat(b.graph, torch.full((pa,), g0, **lg)),
        cell=cat(b.cell, torch.zeros(pg, 3, 3, **fl)),
        edge_index=torch.cat([b.edge_index, pad_ei], dim=1),
        edge_shift=cat(b.edge_shift, torch.zeros(pe, 3, **fl)),
        charge=cat(b.charge, torch.zeros(pg, **fl)),
        spin=cat(b.spin, torch.ones(pg, **fl)),
        u_flag=cat(b.u_flag, torch.zeros(pg, **lg)),
        token=cat(b.token, torch.zeros(pg, **lg)),
        ref_id=cat(b.ref_id, torch.zeros(pg, **lg)),
        domain=b.domain,
        atom_mask=cat(b.atom_mask, torch.zeros(pa, dtype=torch.bool, device=dev)),
        graph_mask=cat(b.graph_mask, torch.zeros(pg, dtype=torch.bool, device=dev)),
        energy=cat(b.energy, torch.zeros(pg, **fl)),
        forces=cat(b.forces, torch.zeros(pa, 3, **fl)),
        energy_mask=cat(b.energy_mask, torch.zeros(pg, dtype=torch.bool, device=dev)),
        force_mask=cat(b.force_mask, torch.zeros(pa, dtype=torch.bool, device=dev)),
        coulomb_cap=b.coulomb_cap,
        frag_id=None if b.frag_id is None else cat(b.frag_id, torch.full((pa,), f0, **lg)),
        frag_charge=None if b.frag_charge is None else cat(b.frag_charge, torch.zeros(pg, **fl)),
        frag_spin=None if b.frag_spin is None else cat(b.frag_spin, torch.ones(pg, **fl)),
        pbc=b.pbc,
        spme_grid=b.spme_grid,
        periodic_electro=b.periodic_electro,
        xc_id=None if b.xc_id is None else cat(b.xc_id, torch.zeros(pg, **lg)),
        paw_id=None if b.paw_id is None else cat(b.paw_id, torch.zeros(pg, **lg)),
        u_atom=None if b.u_atom is None else cat(b.u_atom, torch.zeros(pa, **lg)),
    )


def periodic_two_bucket_epads(n_pad: int, max_e_pad: int | None = None, sparse_ratio: float = 0.45, e_sp: int = 0) -> tuple[int, int]:
    """Return aligned `(e_pad_sparse, e_pad_dense)` for 2-bucket periodic batching (`sparse_periodic` vs `dense_bulk`)."""
    align = lambda v: max(1024, ((int(v) + 1023) // 1024) * 1024)
    dense = align(max_e_pad if (max_e_pad and max_e_pad > 0) else max(4096, int(n_pad) * 64))
    raw_sp = int(e_sp) if int(e_sp) > 0 else max(4096, int(round(dense * float(sparse_ratio))))
    sparse = min(dense, align(raw_sp))
    return sparse, dense


def classify_periodic_bucket(n_edges: int, e_pad_sparse: int, e_pad_dense: int) -> tuple[str, int]:
    """Map real periodic edge count `n_edges` into `('sparse_periodic', e_pad_sparse)` or `('dense_bulk', e_pad_dense)`."""
    if n_edges <= e_pad_sparse:
        return "sparse_periodic", int(e_pad_sparse)
    if n_edges <= e_pad_dense:
        return "dense_bulk", int(e_pad_dense)
    raise ValueError(f"periodic batch edges ({n_edges}) exceed dense_bulk e_pad ({e_pad_dense})")


def active_edge_count(b: GraphBatch) -> int:
    """Count real (non-pad) edges in a `pad_batch`-padded GraphBatch (real edges precede `z=0` pad edges)."""
    if b.n_edges == 0:
        return 0
    real_dst = (b.z.index_select(0, b.edge_index[1]) > 0) & b.atom_mask.index_select(0, b.edge_index[1])
    return int(real_dst.sum().item())


def pad_periodic_batch(b: GraphBatch, n_atoms: int, e_pad_sparse: int, e_pad_dense: int, n_graphs: int, r_cut: float) -> tuple[str, GraphBatch]:
    """Pad an unpadded periodic `GraphBatch` into either `sparse_periodic` or `dense_bulk` bucket."""
    name, target_e = classify_periodic_bucket(b.n_edges, e_pad_sparse, e_pad_dense)
    return name, pad_batch(b, n_atoms, target_e, n_graphs, r_cut)


def repad_periodic_two_bucket(b: GraphBatch, e_pad_sparse: int, e_pad_dense: int, r_cut: float = 6.0, *, e_real: int | None = None) -> tuple[str, GraphBatch]:
    """Repack an already-padded periodic batch cleanly to `sparse_periodic` when its real edges fit `e_pad_sparse`."""
    from dataclasses import replace
    n_real = active_edge_count(b) if e_real is None else int(e_real)
    name, target_e = classify_periodic_bucket(n_real, e_pad_sparse, e_pad_dense)
    if target_e == b.n_edges:
        return name, b
    if target_e > b.n_edges:
        raise ValueError(f"cannot expand padded edges {b.n_edges} -> {target_e}")
    return name, replace(b, edge_index=b.edge_index[:, :target_e].contiguous(), edge_shift=b.edge_shift[:target_e].contiguous())


class TwoBucketStream:
    """Wrap a periodic crystal stream so sparse structures compact to `e_pad_sparse` and dense ones stay in `e_pad_dense`."""

    def __init__(self, inner, e_pad_sparse: int, e_pad_dense: int, r_cut: float = 6.0, *, log=print, log_every: int = 500) -> None:
        self.inner, self.e_pad_sparse, self.e_pad_dense, self.r_cut = inner, int(e_pad_sparse), int(e_pad_dense), float(r_cut)
        self.device = getattr(inner, "device", "cpu")
        self._log, self._log_every = log, max(0, int(log_every))
        self.n_draws = 0
        self.n_sparse = 0

    @property
    def sparse_frac(self) -> float:
        return self.n_sparse / self.n_draws if self.n_draws else 0.0

    def next(self):
        from dataclasses import replace
        pk = self.inner.next()
        name, b2 = repad_periodic_two_bucket(pk.batch, self.e_pad_sparse, self.e_pad_dense, self.r_cut, e_real=getattr(pk, "n_real_edges", None))
        self.n_draws += 1
        if name == "sparse_periodic":
            self.n_sparse += 1
        if self._log is not None and self._log_every and (self.n_draws in (50, 200) or self.n_draws % self._log_every == 0):
            self._log(f"two_crystal_buckets: draws={self.n_draws} sparse_periodic={self.n_sparse} ({100.0 * self.sparse_frac:.1f}%) e_sp={self.e_pad_sparse} e_dn={self.e_pad_dense}")
        return replace(pk, batch=b2) if b2 is not pk.batch else pk

    def state(self) -> dict:
        return self.inner.state()

    def load(self, st: dict) -> None:
        self.inner.load(st)
