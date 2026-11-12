"""Product-only edge aggregation for both stage layouts: oracle `edge_aggregate_ref` and chunked streaming `edge_aggregate_stream`.

Per receiver i, edge (i, j): d = x_j - x_i + shift, r = |d|, u = d / r, x = warp(r) (optional pair warp, radial.py), tab = R(x) = Phi(x) W
(C2 B-spline table of Gaussians at the warped x, spline_tables.py; no envelope inside), chi = env(r) on the UNWARPED r (as earlier Mena trunk). The message, head weights and normalisation are documented in
edge_terms.py; layout.py gives the column / output order of FIRST and LATER stages. Summary:

    N_i = sum_j msg_ij,  Z_i^h = sum_j chi_ij exp(logit_ij^h),  D_i = sum_j chi_ij        (ONE pass, no max, no second sweep)
    out_i = N_i * sqrt(D_i + 1) / (Z_i^h + 1)  per head on its channels   [= phi-normalised sum times the earlier Mena trunk's 1 / sqrt(D_i + 1)]

COMPAT mode (`edge_scale` [E]): head weights off (phi = 1, no q / k, no radial bias), every message multiplied by edge_scale_ij, Z := D,
so out_i = sum_j edge_scale_ij msg_ij / sqrt(D_i + 1): exactly the earlier Mena ChemicalEdgeWeights path, edge_scale = 1 + 0.49 (mod - mean_mod).

Padding contract: an edge with |d| = r_cut (lam4.data.padding.pad_batch) has chi = chi' = 0 exactly (and x = r_cut), so it adds exactly 0
to N, Z, D and (product rule) to every gradient, including w.r.t. positions, gates, vec, q, k and table weights.

Both functions take edges sorted by receiver and give the same values and gradients; `stream` never holds more than
[chunk_edges, ~700] floats of edge data, recomputes them in backward and double backward and allocates gradients only for the rows a
chunk touches (edge_stream.py).
"""
from __future__ import annotations

import torch
from torch import Tensor

from lam4.core.edge_chunks import DEFAULT_CHUNK_EDGES, Chunk, make_chunks
from lam4.core.edge_stream import StreamAgg, empty_z
from lam4.core.edge_terms import EdgeCfg, edge_terms, finalise, normalise_qk
from lam4.core.layout import HEADS, StageLayout, split_output  # noqa: F401
from lam4.core.spline_tables import windows


def _check(layout: StageLayout, pos: Tensor, gates: Tensor, coef: Tensor, vec: Tensor | None, q: Tensor | None,
           edge_scale: Tensor | None, ten: Tensor | None = None) -> None:
    if layout.later and vec is None:
        raise ValueError("LATER layout needs vec [N, 32, 3] (the sender's transported vector)")
    if layout.later and not (layout.k_tr == layout.k_v == layout.k_lon and layout.k_tr2 == layout.k_t == layout.k_lon2):
        raise ValueError("LATER layout needs k_tr == k_v == k_lon and k_tr2 == k_t == k_lon2")
    if ten is not None and ten.numel() > 0 and tuple(ten.shape) != (pos.shape[0], layout.k_t, 5):
        raise ValueError(f"ten must be [N, {layout.k_t}, 5]")
    if edge_scale is None and q is None:
        raise ValueError("q and k are required unless edge_scale is given (compat mode)")
    if gates.shape[-1] != layout.gate_dim or coef.shape[-1] != layout.table_dim:
        raise ValueError(f"gates must be [N, {layout.gate_dim}] and coef [G+3, {layout.table_dim}] for the {layout.name} layout")


def _gain(qk: tuple[Tensor, Tensor], gain: Tensor | None) -> tuple[Tensor, Tensor]:
    """Per-head gain [H] on the unit q: logit = gain_h <qn, kn>. gain = 0 gives logit exactly 0 with q, k at their normal init (no dead saddle)."""
    qn, kn = qk
    return (qn if gain is None else qn * gain.view(1, -1, 1).to(qn.dtype)), kn


def edge_aggregate_ref(pos: Tensor, recv: Tensor, send: Tensor, shift: Tensor, coef: Tensor, gates: Tensor, q: Tensor | None,
                       k: Tensor | None, *, layout: StageLayout, vec: Tensor | None = None, ten: Tensor | None = None,
                       rho: Tensor | None = None, edge_scale: Tensor | None = None, qk_gain: Tensor | None = None,
                       cfg: EdgeCfg = EdgeCfg(), z: Tensor | None = None,
                       geom: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None,
                       phi: Tensor | None = None) -> Tensor:
    """Oracle. pos [N, 3]; recv, send [E]; shift [E, 3]; coef [G+3, layout.table_dim] (BSplineTables()); gates [N, 184]; q, k [N, 4 Dh];
    vec [N, 32, 3] sender vectors (LATER); ten [N, 16, 5] sender quadrupoles (LATER, optional); rho [N] per-atom warp radii (PairWarp.node_rho) or None;
    edge_scale [E] or None (compat); z [N] long atomic numbers for the descriptor floor (cfg.floor; None = no floor).
    Returns [N, layout.out_dim]. Materialises [E, ~700] tensors."""
    _check(layout, pos, gates, coef, vec, q, edge_scale, ten)
    n, od = pos.shape[0], layout.out_dim
    if recv.numel() == 0:
        z0 = (pos.sum() * 0.0) if pos.requires_grad else 0.0
        return finalise(pos.new_zeros((n, od)) + z0, pos.new_zeros((n, HEADS)), pos.new_zeros(n), layout)
    qn, kn = (None, None) if edge_scale is not None else _gain(normalise_qk(q, k), qk_gain)
    msg, zc, chi = edge_terms(pos, pos, recv, send, shift, windows(coef), gates, vec, qn, kn, rho, rho, edge_scale, layout, cfg, z, z, ten, geom=geom, phi=phi)
    n, od = pos.shape[0], layout.out_dim
    packed = torch.cat((msg, zc, chi.unsqueeze(-1)), dim=-1)
    acc = packed.new_zeros(n, od + HEADS + 1).index_add(0, recv, packed)
    return finalise(acc[:, :od], acc[:, od:od + HEADS], acc[:, od + HEADS], layout)


def edge_aggregate_stream(pos: Tensor, recv: Tensor, send: Tensor, shift: Tensor, coef: Tensor, gates: Tensor, q: Tensor | None,
                          k: Tensor | None, *, layout: StageLayout, vec: Tensor | None = None, ten: Tensor | None = None,
                          rho: Tensor | None = None, edge_scale: Tensor | None = None, qk_gain: Tensor | None = None,
                          cfg: EdgeCfg = EdgeCfg(), chunks: list[Chunk] | None = None, chunk_edges: int = DEFAULT_CHUNK_EDGES,
                          z: Tensor | None = None) -> Tensor:
    """Same arguments and result as `edge_aggregate_ref`, edges sorted by receiver, processed in receiver chunks with recomputation in
    backward / double backward. `shift` is a constant (no gradient through it in the stream path).

    chunks: the plan from `edge_chunks.make_chunks(recv, send, N, chunk_edges)`, built once in the data pipeline (and `chunks_to` the
    device). If None the plan is computed here from recv / send, which copies them to the host and SYNCS the device: use it for tests only.
    """
    _check(layout, pos, gates, coef, vec, q, edge_scale, ten)
    if chunks is None:
        chunks = make_chunks(recv, send, pos.shape[0], chunk_edges)
    empty = pos.new_zeros(0)
    qn, kn = (empty, empty) if edge_scale is not None else _gain(normalise_qk(q, k), qk_gain)
    sh_c, sh_d = (empty, shift) if shift.requires_grad else (shift, empty)
    ins = (pos, windows(coef), gates, empty if vec is None else vec, qn, kn, empty if rho is None else rho,
           empty if edge_scale is None else edge_scale, empty if (ten is None or not layout.later) else ten, sh_d)
    num, zs, dsum = StreamAgg.apply(chunks, layout, cfg, sh_c, empty_z(pos, z), *ins)
    return finalise(num, zs, dsum, layout)
