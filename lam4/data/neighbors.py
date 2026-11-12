"""neighbors.py: radius graphs for the lam4 loader. Edges are (sender j -> receiver i), RECEIVER-SORTED, as `GraphBatch` wants.

* `dense_open_mask` / `edges_from_mask`: open molecules packed in a size bucket, [K, cap, cap] dense in-bucket distances on any device (the O4 `pack_batch` method,
  the earlier Mena training pipeline). Strict r < r_cut, no self pair; `torch.nonzero` order (b, i, j) makes the list receiver-sorted for free.
* `vesin_edges`: one periodic or open structure on the host with vesin (as lam.units.build_graph), integer image shifts; vesin's (i, j, S) means the vector
  r_j - r_i + S @ cell, i.e. i = receiver, j = sender in the `GraphBatch` convention (`edge_vectors = pos[send] - pos[recv] + shift @ cell`).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def dense_open_mask(pos: Tensor, mask: Tensor, r_cut: float) -> Tensor:
    """pos [K, cap, 3], mask [K, cap] bool -> m [K, cap, cap] bool, m[b, i, j] = atoms i != j of structure b, both real, |r_i - r_j| < r_cut."""
    cap = pos.shape[1]
    d2 = (pos.unsqueeze(2) - pos.unsqueeze(1)).pow(2).sum(-1)
    eye = torch.eye(cap, dtype=torch.bool, device=pos.device)
    return (d2 < r_cut * r_cut) & mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye


@torch.no_grad()
def edges_from_mask(m: Tensor, n_atoms: Tensor) -> Tensor:
    """m [K, cap, cap], n_atoms [K] (real atoms packed contiguously, structure after structure) -> edge_index [2, E] = (sender j, receiver i), receiver-sorted."""
    start = torch.cumsum(n_atoms, 0) - n_atoms
    b, i, j = torch.nonzero(m, as_tuple=True)
    return torch.stack([start[b] + j, start[b] + i])


def vesin_edges(pos: np.ndarray, cell: np.ndarray, pbc, r_cut: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sender j, receiver i, integer shift [E, 3]) of one structure, receiver-sorted (stable). pbc: bool or 3 bools; open structures use a zero box."""
    import vesin

    p = tuple(bool(x) for x in np.atleast_1d(pbc)) if np.size(pbc) == 3 else (bool(np.atleast_1d(pbc)[0]),) * 3
    box = np.asarray(cell, np.float64) if any(p) else np.zeros((3, 3))
    i, j, s = vesin.NeighborList(cutoff=float(r_cut), full_list=True).compute(
        points=np.asarray(pos, np.float64), box=box, periodic=p, quantities="ijS")
    i, j, s = np.asarray(i, np.int64), np.asarray(j, np.int64), np.asarray(s, np.int64)
    o = np.argsort(i, kind="stable")
    return j[o], i[o], s[o]
