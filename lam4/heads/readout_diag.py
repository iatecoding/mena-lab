"""readout_diag.py: per-eval energy-cancellation diagnostic of the readout (logged as `diag/readout_energy`).

An energy gap seen in an earlier model (large-molecule offsets) came from stage readouts that grew large and cancelled each other at constant LR. This reads the
terms of `Readout.final` (a dict of per-atom energies [N], insertion order = Readout.term_names) and returns, per graph size bucket and per |Q| class, how large
each term is and how much of it survives in the sum. Pure tensor ops, static shapes (one-hot matmuls, no boolean indexing), no host sync, no printing, no grad.

For graph g with n_g real atoms and terms t: E_gt = sum_i e_it.  Per bucket b (graphs with n_g in (edge_{b-1}, edge_b]; SIZE_EDGES = 8 .. 350 plus an open
last bucket) and per |Q| class (0, 1, 2, >= 3):
    count            [B]      real graphs in the bucket (empty buckets give 0 in every statistic, never NaN)
    abs_per_atom     [B, T]   mean_g |E_gt| / n_g        magnitude of each term
    signed_per_atom  [B, T]   mean_g  E_gt  / n_g        drift of each term (the offset that swings at constant LR)
    gross_per_atom   [B]      mean_g sum_t |E_gt| / n_g
    net_abs_per_atom [B]      mean_g |sum_t E_gt| / n_g
    cancel           [B]      mean_g |sum_t E_gt| / sum_t |E_gt|   in [0, 1]; 1 = no cancellation, small = terms cancel each other
Keys: size_* and q_* (q_count, q_abs_per_atom, ...), plus size_edges [B - 1]. `charge` is the graph charge [G]; pass graph_mask to drop pad graphs
(graphs without a real atom are dropped anyway).
"""
from __future__ import annotations

import torch
from torch import Tensor

SIZE_EDGES = (8, 16, 32, 64, 96, 128, 192, 350)
QCLASSES = 4   # |Q| = 0, 1, 2, >= 3


def _stats(prefix: str, onehot: Tensor, e_gt: Tensor, n_g: Tensor) -> dict[str, Tensor]:
    cnt = onehot.sum(0)
    inv = 1.0 / cnt.clamp_min(1.0)
    per_atom = e_gt / n_g.clamp_min(1.0).unsqueeze(-1)
    gross, net = e_gt.abs().sum(-1), e_gt.sum(-1).abs()
    return {
        f"{prefix}_count": cnt,
        f"{prefix}_abs_per_atom": (onehot.t() @ per_atom.abs()) * inv.unsqueeze(-1),
        f"{prefix}_signed_per_atom": (onehot.t() @ per_atom) * inv.unsqueeze(-1),
        f"{prefix}_gross_per_atom": (onehot.t() @ (gross / n_g.clamp_min(1.0))) * inv,
        f"{prefix}_net_abs_per_atom": (onehot.t() @ (net / n_g.clamp_min(1.0))) * inv,
        f"{prefix}_cancel": (onehot.t() @ (net / gross.clamp_min(1e-30))) * inv,
    }


@torch.no_grad()
def cancellation_diagnostic(terms: dict[str, Tensor], graph: Tensor, atom_mask: Tensor, charge: Tensor, graph_mask: Tensor | None = None,
                            size_edges: tuple[int, ...] = SIZE_EDGES) -> dict[str, Tensor]:
    """terms {name: [N]}, graph [N] atom -> graph, atom_mask [N] bool, charge [G] -> dict of tensors (see the module docstring)."""
    stack = torch.stack([t.detach() for t in terms.values()], dim=-1)                 # [N, T]
    mf = atom_mask.to(stack.dtype)
    g_n = charge.shape[0]
    e_gt = stack.new_zeros(g_n, stack.shape[-1]).index_add(0, graph, stack * mf.unsqueeze(-1))
    n_g = stack.new_zeros(g_n).index_add(0, graph, mf)
    valid = (n_g > 0) if graph_mask is None else ((n_g > 0) & graph_mask)
    vf = valid.to(stack.dtype)
    edges = torch.tensor(size_edges, dtype=stack.dtype, device=stack.device)
    size_b = torch.bucketize(n_g, edges)                                               # 0 .. len(edges); n_g <= edge_b -> bucket b
    q_b = charge.abs().round().clamp(max=QCLASSES - 1).long()
    oh_s = torch.nn.functional.one_hot(size_b, len(size_edges) + 1).to(stack.dtype) * vf.unsqueeze(-1)
    oh_q = torch.nn.functional.one_hot(q_b, QCLASSES).to(stack.dtype) * vf.unsqueeze(-1)
    out = {"size_edges": edges}
    out.update(_stats("size", oh_s, e_gt, n_g))
    out.update(_stats("q", oh_q, e_gt, n_g))
    return out
