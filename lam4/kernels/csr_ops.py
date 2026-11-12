"""Fused Dual-CSR autograd op (`CSRAgg`) and safe fallback wrapper (`edge_aggregate_csr`)."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from lam4.core import edge_terms
from lam4.core.edge_ops import _check, _gain, edge_aggregate_ref, edge_aggregate_stream
from lam4.core.edge_terms import EdgeCfg, finalise, normalise_qk
from lam4.core.layout import HEADS, StageLayout
from lam4.kernels.csr_plan import CSRPlan, build_csr_plan, coef_to_horner_poly
from lam4.kernels.metal_csr import has_metal_csr, metal_csr_backward, metal_csr_forward
from lam4.kernels.triton_csr import has_triton_csr, triton_csr_backward, triton_csr_forward


def device_supports_csr(device: torch.device | str) -> bool:
    dev = torch.device(device)
    if dev.type == "mps":
        return has_metal_csr()
    if dev.type == "cuda":
        return has_triton_csr()
    return False


def can_use_csr(
    pos: Tensor,
    coef: Tensor,
    q: Tensor | None,
    k: Tensor | None,
    *,
    layout: StageLayout,
    cfg: EdgeCfg,
    edge_scale: Tensor | None = None,
    create_graph: bool = False,
) -> bool:
    """Strict developer-safe check: returns True iff the Dual-CSR GPU kernel supports every input & flag."""
    if pos.dtype != torch.float32 or create_graph or torch.compiler.is_compiling():
        return False
    if edge_terms.bspline_eval is not edge_terms._ORIG_BSPLINE_EVAL:
        return False
    if not device_supports_csr(pos.device):
        return False
    if edge_scale is not None or cfg.r_cut_near > 0.0 or coef.requires_grad:
        return False
    if q is None or k is None or q.shape[-1] != 64 or k.shape[-1] != 64:
        return False
    if (layout.k_s, layout.k_v, layout.k_t, layout.k_o) != (128, 32, 16, 8):
        return False
    if layout.later and (layout.k_tr, layout.k_lon, layout.k_tr2, layout.k_lon2) != (32, 32, 16, 16):
        return False
    return True


class CSRAgg(torch.autograd.Function):
    """Atomic-free Dual-CSR edge aggregation returning `(num [N, out_dim], zc [N, 4], dsum [N])`."""

    @staticmethod
    def forward(
        ctx,
        plan: CSRPlan,
        layout: StageLayout,
        cfg: EdgeCfg,
        shift_const: Tensor,
        graph: Tensor,
        z: Tensor,
        pos: Tensor,
        poly: Tensor,
        gates: Tensor,
        vec: Tensor,
        ten: Tensor,
        qn: Tensor,
        kn: Tensor,
        rho: Tensor,
        cell: Tensor,
        shift_diff: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        shift_in = shift_diff if shift_diff.numel() > 0 else shift_const
        ctx.plan = plan
        ctx.layout = layout
        ctx.cfg = cfg
        ctx.used_shift_diff = bool(shift_diff.numel() > 0)
        ctx.save_for_backward(shift_in, graph, z, pos, poly, gates, vec, ten, qn, kn, rho, cell)
        fn = metal_csr_forward if pos.device.type == "mps" else triton_csr_forward
        return fn(plan, layout, cfg, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z)

    @staticmethod
    @once_differentiable
    def backward(ctx, g_num: Tensor, g_zc: Tensor, g_dsum: Tensor):
        shift_in, graph, z, pos, poly, gates, vec, ten, qn, kn, rho, cell = ctx.saved_tensors
        needs = ctx.needs_input_grad
        need_vir = bool(needs[14] and cell.numel() > 0)
        need_drho = bool(needs[13] and rho.numel() > 0)
        need_dshift = bool(needs[15] and ctx.used_shift_diff)
        fn = metal_csr_backward if pos.device.type == "mps" else triton_csr_backward
        d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell, d_shift = fn(
            ctx.plan,
            ctx.layout,
            ctx.cfg,
            pos,
            poly,
            gates,
            vec,
            ten,
            qn,
            kn,
            rho,
            shift_in,
            cell,
            graph,
            z,
            g_num.contiguous(),
            g_zc.contiguous(),
            g_dsum.contiguous(),
            need_vir=need_vir,
            need_drho=need_drho,
            need_dshift=need_dshift,
        )
        outs = (d_pos, None, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell, d_shift)
        masked = tuple(g if (nd and g is not None) else None for g, nd in zip(outs, needs[6:]))
        return (None, None, None, None, None, None, *masked)


def edge_aggregate_csr(
    pos: Tensor,
    recv: Tensor,
    send: Tensor,
    shift: Tensor | None,
    coef: Tensor,
    gates: Tensor,
    q: Tensor | None,
    k: Tensor | None,
    *,
    layout: StageLayout,
    vec: Tensor | None = None,
    ten: Tensor | None = None,
    rho: Tensor | None = None,
    edge_scale: Tensor | None = None,
    qk_gain: Tensor | None = None,
    cfg: EdgeCfg = EdgeCfg(),
    z: Tensor | None = None,
    csr_plan: CSRPlan | None = None,
    poly: Tensor | None = None,
    edge_shift: Tensor | None = None,
    cell: Tensor | None = None,
    graph: Tensor | None = None,
    create_graph: bool = False,
    low_memory_fallback: bool = False,
) -> Tensor:
    """Fused Dual-CSR edge aggregation (`[N, layout.out_dim]`), with automatic exact PyTorch fallback when unsupported."""
    _check(layout, pos, gates, coef, vec, q, edge_scale, ten)
    if recv.numel() == 0:
        n, od = pos.shape[0], layout.out_dim
        z0 = (pos.sum() * 0.0) if pos.requires_grad else 0.0
        return finalise(pos.new_zeros((n, od)) + z0, pos.new_zeros((n, 4)), pos.new_zeros(n), layout)
    if not can_use_csr(pos, coef, q, k, layout=layout, cfg=cfg, edge_scale=edge_scale, create_graph=create_graph):
        if shift is None:
            assert edge_shift is not None and cell is not None
            shift = (edge_shift @ cell[0]) if cell.shape[0] == 1 else torch.einsum("ek,ekl->el", edge_shift, cell[graph[recv]])
        kw = dict(layout=layout, vec=vec, ten=ten, rho=rho, edge_scale=edge_scale, qk_gain=qk_gain, cfg=cfg, z=z)
        if low_memory_fallback:
            return edge_aggregate_stream(pos, recv, send, shift, coef, gates, q, k, **kw)
        return edge_aggregate_ref(pos, recv, send, shift, coef, gates, q, k, **kw)

    assert q is not None and k is not None
    plan = build_csr_plan(recv, send, pos.shape[0]) if csr_plan is None else csr_plan
    poly_c = coef_to_horner_poly(coef, cfg.r_cut) if poly is None else poly.contiguous()
    qn, kn = _gain(normalise_qk(q, k), qk_gain)

    empty_f = pos.new_zeros(0)
    empty_i = pos.new_zeros(0, dtype=torch.int64)
    pos_c = pos.contiguous()
    gates_c = gates.contiguous()
    vec_c = vec.contiguous() if (layout.later and vec is not None) else empty_f
    ten_c = ten.contiguous() if (layout.later and ten is not None and ten.numel() > 0) else empty_f
    qn_c = qn.contiguous()
    kn_c = kn.contiguous()
    rho_c = rho.contiguous() if (rho is not None and rho.numel() > 0) else empty_f
    z_c = z.to(torch.int64).contiguous() if (cfg.floor and z is not None and z.numel() > 0) else empty_i

    if cell is not None and edge_shift is not None:
        cell_c = cell.contiguous()
        graph_c = graph.to(torch.int64).contiguous() if (graph is not None and cell_c.shape[0] > 1) else empty_i
        sh_const = edge_shift.to(torch.float32).contiguous()
        sh_diff = empty_f
    else:
        assert shift is not None
        cell_c = empty_f
        graph_c = empty_i
        sh_32 = shift.to(torch.float32).contiguous()
        sh_const, sh_diff = (empty_f, sh_32) if shift.requires_grad else (sh_32, empty_f)

    num, zc, dsum = CSRAgg.apply(
        plan, layout, cfg, sh_const, graph_c, z_c, pos_c, poly_c, gates_c, vec_c, ten_c, qn_c, kn_c, rho_c, cell_c, sh_diff
    )
    return finalise(num, zc, dsum, layout)
