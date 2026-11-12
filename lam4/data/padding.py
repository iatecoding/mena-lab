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
    # consecutive pad atoms, both directions, cycled to fill the edge cap
    if pe > 0:
        k = torch.arange(pe, **lg) % (2 * (pa - 1))
        fwd = k < (pa - 1)
        a = torch.where(fwd, k, k - (pa - 1))
        send = torch.where(fwd, a, a + 1) + n0
        recv = torch.where(fwd, a + 1, a) + n0
        order = torch.argsort(recv, stable=True)   # real edges are receiver-sorted; keep the whole list sorted
        pad_ei = torch.stack([send[order], recv[order]])
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
    )
