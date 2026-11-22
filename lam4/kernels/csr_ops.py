"""Fused Dual-CSR custom_op (`lam4::csr_agg`, `bwd`, `bwd_p`, `bwd2`) + autograd (`CSRAgg`) and safe fallback."""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

from lam4.core import edge_terms as et_mod
from lam4.core.consts import storable
from lam4.core.edge_ops import _check, _gain, edge_aggregate_ref, edge_aggregate_stream
from lam4.core.edge_terms import EdgeCfg, edge_terms, finalise, normalise_qk
from lam4.core.layout import FIRST, HEADS, LATER, StageLayout
from lam4.kernels.cpu_csr import cpu_csr_backward, cpu_csr_forward, has_cpu_csr, rmin_table as cpu_rmin_table
from lam4.kernels.csr_plan import (
    CSRPlan, build_csr_plan, coef_to_horner_poly, horner_to_win, raw_proxy_tracing,
    rmin_table_f32, stream_vjp_csr_backward, warm_csr_nlh_caches,
)
from lam4.kernels.cuda_csr import cuda_csr_aggregate_final, has_cuda_csr
from lam4.kernels.metal_csr import has_metal_csr, metal_csr_backward, metal_csr_forward
from lam4.kernels.triton_csr import has_triton_csr, has_triton_csr_backward, triton_csr_backward, triton_csr_forward
from lam4.kernels.triton_csr_bwd2 import has_triton_csr_bwd2, triton_csr_backward_poly, triton_csr_bwd2
from lam4.physics.repulsion import with_rmin_table


def has_csr_custom_op() -> bool:
    return os.environ.get("LAM4_CSR_CUSTOM_OP", "1") != "0"


def device_supports_csr(device: torch.device | str) -> bool:
    dev = torch.device(device)
    if dev.type == "cpu":
        return has_cpu_csr()
    return has_metal_csr() if dev.type == "mps" else (has_triton_csr() if dev.type == "cuda" else False)


def dtype_supports_csr(pos: Tensor) -> bool:
    """float32 everywhere; float64 only on the CPU kernel (templated on float / double)."""
    return pos.dtype == torch.float32 or (pos.dtype == torch.float64 and pos.device.type == "cpu")


def _present(t: Tensor | None) -> bool:
    return False if t is None else (True if isinstance(t.shape[0], torch.SymInt) else (t.numel() > 0))


def can_use_csr(
    pos: Tensor, coef: Tensor, q: Tensor | None, k: Tensor | None, *, layout: StageLayout,
    cfg: EdgeCfg, edge_scale: Tensor | None = None, create_graph: bool = False, allow_bwd2: bool = False,
) -> bool:
    """Strict developer-safe check: returns True iff Dual-CSR GPU kernel supports every input & flag."""
    if not dtype_supports_csr(pos):
        return False
    bwd2_ok = bool(allow_bwd2 and pos.device.type == "cuda" and has_triton_csr_bwd2(pos.device) and has_csr_custom_op())
    if create_graph and not bwd2_ok:
        return False
    if raw_proxy_tracing() and not bwd2_ok:
        return False
    if torch.compiler.is_compiling() and not has_csr_custom_op():
        return False
    if et_mod.bspline_eval is not et_mod._ORIG_BSPLINE_EVAL or not device_supports_csr(pos.device):
        return False
    if edge_scale is not None or cfg.r_cut_near > 0.0 or (coef.requires_grad and not bwd2_ok):
        return False
    if q is None or k is None or q.shape[-1] != 64 or k.shape[-1] != 64:
        return False
    if (layout.k_s, layout.k_v, layout.k_t, layout.k_o) != (128, 32, 16, 8):
        return False
    if layout.later and (layout.k_tr, layout.k_lon, layout.k_tr2, layout.k_lon2) != (32, 32, 16, 16):
        return False
    return True


def _cpu_csr_forward(plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor, cell: Tensor, graph: Tensor, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    n, od = plan.n_atoms, layout.out_dim
    is_sym = isinstance(n, torch.SymInt) or isinstance(plan.n_edges, torch.SymInt)
    e_act = max(0, min(int(plan.recv_ptr[-1].item()), plan.n_edges)) if (not is_sym and storable(pos) and pos.device.type == "cpu" and plan.recv_ptr.numel() > 0) else plan.n_edges
    if not is_sym and (n == 0 or e_act == 0):
        return pos.new_zeros((n, od)), pos.new_zeros((n, 4)), pos.new_zeros((n,))
    recv_l = (plan.recv_idx if e_act == plan.n_edges else plan.recv_idx[:e_act]).to(torch.long)
    send_l = (plan.send_idx if e_act == plan.n_edges else plan.send_idx[:e_act]).to(torch.long)
    sh_src = shift_in if e_act == plan.n_edges else shift_in[:e_act]
    sh = (((sh_src @ cell[0]) if cell.shape[0] == 1 else torch.einsum("ek,ekl->el", sh_src, cell[graph[recv_l]])) if cell.ndim == 3 else sh_src)
    has_r, has_z = _present(rho), bool(cfg.floor and _present(z))
    w_poly = horner_to_win(poly.double()) if (poly.device.type == "cpu" and poly.dtype == torch.float32 and torch.is_grad_enabled()) else horner_to_win(poly)
    msg, zw, chi = edge_terms(
        pos, pos, recv_l, send_l, sh, w_poly, gates, vec if (layout.later and vec.ndim == 3) else None,
        qn, kn, rho if has_r else None, rho if has_r else None, None, layout, cfg, z if has_z else None, z if has_z else None,
        ten if (layout.later and ten.ndim == 3) else None,
    )
    packed = torch.cat((msg, zw, chi.unsqueeze(-1)), dim=-1)
    acc = packed.new_zeros(n, od + HEADS + 1).index_add(0, recv_l, packed)
    return acc[:, :od].contiguous(), acc[:, od : od + HEADS].contiguous(), acc[:, od + HEADS].contiguous()


def _ref_vjp_csr_backward(
    recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    layout: StageLayout, cfg: EdgeCfg, rmin: Tensor, pos: Tensor, poly: Tensor, gates: Tensor,
    vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor,
    cell: Tensor, graph: Tensor, z: Tensor, g_num: Tensor, g_zc: Tensor, g_dsum: Tensor,
    *, need_vir: bool, need_drho: bool, need_dshift: bool, need_dpoly: bool = False, create_graph: bool = False, live_attach: bool = False,
) -> tuple[Any, ...]:
    has_v, has_t, has_c = bool(layout.later and vec.ndim == 3), bool(layout.later and ten.ndim == 3 and _present(ten)), bool(cell.ndim == 3 and _present(cell))
    prep = (lambda x, f: x if (live_attach or not f) else x.detach().requires_grad_(True))
    if not need_dpoly:
        flags = (True, True, has_v, has_t, True, True, bool(need_drho and _present(rho)), bool(need_vir and has_c), bool(need_dshift))
        raw = (pos, gates, vec, ten, qn, kn, rho, cell, shift_in)
        with torch.enable_grad(), with_rmin_table(rmin):
            lvs = [prep(x, f) if live_attach else (x.detach().requires_grad_(f)) for x, f in zip(raw, flags)]
            plan = CSRPlan(pos.shape[0], send_idx.shape[0], recv_ptr, send_idx, recv_idx, send_ptr, send_perm)  # type: ignore[arg-type]
            outs = _cpu_csr_forward(plan, layout, cfg, lvs[0], poly, lvs[1], lvs[2], lvs[3], lvs[4], lvs[5], lvs[6], lvs[8], lvs[7], graph, z)
            act_idx = [k for k, f in enumerate(flags) if f and (not live_attach or lvs[k].requires_grad)]
            grads = torch.autograd.grad(outs, [lvs[k] for k in act_idx], (g_num, g_zc, g_dsum), create_graph=create_graph, allow_unused=True) if act_idx else ()
        res: list[Any] = [None] * 9
        for k, g in zip(act_idx, grads):
            res[k] = torch.zeros_like(raw[k]) if g is None else g
        for k, f in enumerate(flags):
            if f and res[k] is None:
                res[k] = torch.zeros_like(raw[k])
        return tuple(res)
    flags_p = (True, True, True, has_v, has_t, True, True, bool(need_drho and _present(rho)), bool(need_vir and has_c), bool(need_dshift))
    raw_p = (pos, poly, gates, vec, ten, qn, kn, rho, cell, shift_in)
    with torch.enable_grad(), with_rmin_table(rmin):
        lvs_p = [prep(x, f) if live_attach else (x.detach().requires_grad_(f)) for x, f in zip(raw_p, flags_p)]
        plan = CSRPlan(pos.shape[0], send_idx.shape[0], recv_ptr, send_idx, recv_idx, send_ptr, send_perm)  # type: ignore[arg-type]
        outs = _cpu_csr_forward(plan, layout, cfg, lvs_p[0], lvs_p[1], lvs_p[2], lvs_p[3], lvs_p[4], lvs_p[5], lvs_p[6], lvs_p[7], lvs_p[9], lvs_p[8], graph, z)
        act_p = [k for k, f in enumerate(flags_p) if f and (not live_attach or lvs_p[k].requires_grad)]
        grads_p = torch.autograd.grad(outs, [lvs_p[k] for k in act_p], (g_num, g_zc, g_dsum), create_graph=create_graph, allow_unused=True) if act_p else ()
    res_p: list[Any] = [None] * 10
    for k, g in zip(act_p, grads_p):
        res_p[k] = torch.zeros_like(raw_p[k]) if g is None else g
    for k, f in enumerate(flags_p):
        if f and res_p[k] is None:
            res_p[k] = torch.zeros_like(raw_p[k])
    return tuple(res_p)


def _ref_csr_bwd2_cpu(
    plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor,
    ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor, cell: Tensor, graph: Tensor, z: Tensor,
    g_num: Tensor, g_zc: Tensor, g_dsum: Tensor, v_pos: Tensor, v_poly: Tensor, v_gates: Tensor, v_vec: Tensor,
    v_ten: Tensor, v_qn: Tensor, v_kn: Tensor, v_rho: Tensor, v_cell: Tensor, *, need_dpoly: bool, need_vir: bool, need_drho: bool,
) -> tuple[Any, ...]:
    ht, hr, hc = bool(layout.later and ten.ndim == 3 and ten.numel() > 0), bool(rho.numel() > 0), bool(cell.ndim == 3 and cell.numel() > 0)
    with torch.enable_grad(), with_rmin_table(rmin_table_f32(pos.device)):
        ins = [
            pos.detach().requires_grad_(True), poly.detach().requires_grad_(True), gates.detach().requires_grad_(True),
            vec.detach().requires_grad_(True) if (layout.later and vec.ndim == 3) else vec.detach(),
            ten.detach().requires_grad_(True) if ht else ten.detach(), qn.detach().requires_grad_(True),
            kn.detach().requires_grad_(True), rho.detach().requires_grad_(True) if hr else rho.detach(),
            cell.detach().requires_grad_(True) if hc else cell.detach(),
        ]
        gn_l, gz_l, gd_l = g_num.detach().requires_grad_(True), g_zc.detach().requires_grad_(True), g_dsum.detach().requires_grad_(True)
        b1 = torch.autograd.grad(_cpu_csr_forward(plan, layout, cfg, ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6], ins[7], shift_in, ins[8], graph, z), [x for x in ins if x.requires_grad], (gn_l, gz_l, gd_l), create_graph=True, allow_unused=True)
        it1 = iter(b1)
        b1_full = [(next(it1) if x.requires_grad else None) for x in ins]
        pairs = [(b, v) for b, v in zip(b1_full, (v_pos, v_poly, v_gates, v_vec, v_ten, v_qn, v_kn, v_rho, v_cell)) if (b is not None and v.numel() > 0)]
        wrt = [x for x in ins if x.requires_grad] + [gn_l, gz_l, gd_l]
        g2 = torch.autograd.grad([b for b, _ in pairs], wrt, [v for _, v in pairs], allow_unused=True) if pairs else [None] * len(wrt)
        it2 = iter(g2)
        res = [(next(it2) if x.requires_grad else None) for x in ins]
        zor = lambda g, ref, ok: None if not ok else (torch.zeros_like(ref) if g is None else g)
        return (zor(res[0], pos, True), zor(res[1], poly, need_dpoly), zor(res[2], gates, True), zor(res[3], vec, layout.later), zor(res[4], ten, ht), zor(res[5], qn, True), zor(res[6], kn, True), zor(res[7], rho, need_drho), zor(res[8], cell, need_vir), None, zor(next(it2), g_num, True), zor(next(it2), g_zc, True), zor(next(it2), g_dsum, True))


def _dispatch_fwd(plan, layout, cfg, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z):
    fn = metal_csr_forward if pos.device.type == "mps" else (triton_csr_forward if pos.device.type == "cuda" else (cpu_csr_forward if has_cpu_csr() else _cpu_csr_forward))
    return fn(plan, layout, cfg, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z)


def _dispatch_bwd(plan, layout, cfg, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z, g_num, g_zc, g_dsum, *, need_vir: bool, need_drho: bool, need_dshift: bool):
    fn = metal_csr_backward if pos.device.type == "mps" else (triton_csr_backward if pos.device.type == "cuda" else (cpu_csr_backward if has_cpu_csr() else stream_vjp_csr_backward))
    return fn(plan, layout, cfg, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z, g_num, g_zc, g_dsum, need_vir=need_vir, need_drho=need_drho, need_dshift=need_dshift)


@torch.library.custom_op("lam4::csr_agg", mutates_args=())
def _csr_agg_op(rp: Tensor, si: Tensor, ri: Tensor, sp: Tensor, spm: Tensor, sh_c: Tensor, gr: Tensor, z: Tensor, rmin: Tensor, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor, sh_d: Tensor, later: bool, r_cut: float, r_on: float, floor: bool, has_rho: bool) -> tuple[Tensor, Tensor, Tensor]:
    pl = CSRPlan(int(pos.shape[0]), int(si.shape[0]), rp, si, ri, sp, spm)
    return _dispatch_fwd(pl, LATER if later else FIRST, EdgeCfg(r_cut=float(r_cut), r_on=float(r_on), floor=bool(floor)), pos, poly, gates, vec, ten, qn, kn, rho, sh_d if sh_d.ndim == 2 else sh_c, cell, gr, z)


@_csr_agg_op.register_fake
def _(rp: Tensor, si: Tensor, ri: Tensor, sp: Tensor, spm: Tensor, sh_c: Tensor, gr: Tensor, z: Tensor, rmin: Tensor, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor, sh_d: Tensor, later: bool, r_cut: float, r_on: float, floor: bool, has_rho: bool) -> tuple[Tensor, Tensor, Tensor]:
    return pos.new_empty((pos.shape[0], 408 if later else 360)), pos.new_empty((pos.shape[0], 4)), pos.new_empty((pos.shape[0],))


@torch.library.custom_op("lam4::csr_agg_bwd", mutates_args=())
def _csr_agg_bwd_op(rp: Tensor, si: Tensor, ri: Tensor, sp: Tensor, spm: Tensor, sh: Tensor, gr: Tensor, z: Tensor, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor, gn: Tensor, gz: Tensor, gd: Tensor, later: bool, r_cut: float, r_on: float, floor: bool, has_rho: bool, need_vir: bool, need_drho: bool, need_dshift: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    pl = CSRPlan(int(pos.shape[0]), int(si.shape[0]), rp, si, ri, sp, spm)
    outs = _dispatch_bwd(pl, LATER if later else FIRST, EdgeCfg(r_cut=float(r_cut), r_on=float(r_on), floor=bool(floor)), pos, poly, gates, vec, ten, qn, kn, rho, sh, cell, gr, z, gn, gz, gd, need_vir=need_vir, need_drho=need_drho, need_dshift=need_dshift)
    return tuple(pos.new_empty(0) if g is None else g for g in outs)  # type: ignore[return-value]


@_csr_agg_bwd_op.register_fake
def _(rp: Tensor, si: Tensor, ri: Tensor, sp: Tensor, spm: Tensor, sh: Tensor, gr: Tensor, z: Tensor, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor, gn: Tensor, gz: Tensor, gd: Tensor, later: bool, r_cut: float, r_on: float, floor: bool, has_rho: bool, need_vir: bool, need_drho: bool, need_dshift: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (torch.empty_like(pos), torch.empty_like(gates), torch.empty_like(vec) if (later and vec.ndim == 3) else pos.new_empty(0), torch.empty_like(ten) if (later and ten.ndim == 3) else pos.new_empty(0), torch.empty_like(qn), torch.empty_like(kn), torch.empty_like(rho) if (need_drho and has_rho) else pos.new_empty(0), torch.empty_like(cell) if (need_vir and cell.ndim == 3) else pos.new_empty(0), torch.empty_like(sh) if need_dshift else pos.new_empty(0))


@torch.library.custom_op("lam4::csr_agg_bwd_p", mutates_args=())
def _csr_agg_bwd_p_op(rp: Tensor, si: Tensor, ri: Tensor, sp: Tensor, spm: Tensor, sh: Tensor, gr: Tensor, z: Tensor, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor, gn: Tensor, gz: Tensor, gd: Tensor, later: bool, r_cut: float, r_on: float, floor: bool, has_rho: bool, need_vir: bool, need_drho: bool, need_dshift: bool, need_dpoly: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    pl = CSRPlan(int(pos.shape[0]), int(si.shape[0]), rp, si, ri, sp, spm)
    outs = triton_csr_backward_poly(pl, LATER if later else FIRST, EdgeCfg(r_cut=float(r_cut), r_on=float(r_on), floor=bool(floor)), pos, poly, gates, vec, ten, qn, kn, rho, sh, cell, gr, z, gn, gz, gd, need_vir=need_vir, need_drho=need_drho, need_dshift=need_dshift, need_dpoly=need_dpoly)
    return tuple(pos.new_empty(0) if g is None else g for g in outs)  # type: ignore[return-value]


@_csr_agg_bwd_p_op.register_fake
def _(rp: Tensor, si: Tensor, ri: Tensor, sp: Tensor, spm: Tensor, sh: Tensor, gr: Tensor, z: Tensor, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor, gn: Tensor, gz: Tensor, gd: Tensor, later: bool, r_cut: float, r_on: float, floor: bool, has_rho: bool, need_vir: bool, need_drho: bool, need_dshift: bool, need_dpoly: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (torch.empty_like(pos), torch.empty_like(poly) if need_dpoly else pos.new_empty(0), torch.empty_like(gates), torch.empty_like(vec) if (later and vec.ndim == 3) else pos.new_empty(0), torch.empty_like(ten) if (later and ten.ndim == 3) else pos.new_empty(0), torch.empty_like(qn), torch.empty_like(kn), torch.empty_like(rho) if (need_drho and has_rho) else pos.new_empty(0), torch.empty_like(cell) if (need_vir and cell.ndim == 3) else pos.new_empty(0), torch.empty_like(sh) if need_dshift else pos.new_empty(0))


@torch.library.custom_op("lam4::csr_agg_bwd2", mutates_args=())
def _csr_agg_bwd2_op(rp: Tensor, si: Tensor, ri: Tensor, sp: Tensor, spm: Tensor, sh: Tensor, gr: Tensor, z: Tensor, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor, gn: Tensor, gz: Tensor, gd: Tensor, vp: Tensor, vpo: Tensor, vg: Tensor, vv: Tensor, vt: Tensor, vq: Tensor, vk: Tensor, vr: Tensor, vc: Tensor, later: bool, r_cut: float, r_on: float, floor: bool, has_rho: bool, need_dpoly: bool, need_vir: bool, need_drho: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    pl = CSRPlan(int(pos.shape[0]), int(si.shape[0]), rp, si, ri, sp, spm)
    outs = triton_csr_bwd2(pl, LATER if later else FIRST, EdgeCfg(r_cut=float(r_cut), r_on=float(r_on), floor=bool(floor)), pos, poly, gates, vec, ten, qn, kn, rho, sh, cell, gr, z, gn, gz, gd, vp, vpo, vg, vv, vt, vq, vk, vr, vc, need_dpoly=need_dpoly, need_vir=need_vir, need_drho=need_drho)
    return tuple(pos.new_empty(0) if g is None else g for g in outs)  # type: ignore[return-value]


@_csr_agg_bwd2_op.register_fake
def _(rp: Tensor, si: Tensor, ri: Tensor, sp: Tensor, spm: Tensor, sh: Tensor, gr: Tensor, z: Tensor, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor, gn: Tensor, gz: Tensor, gd: Tensor, vp: Tensor, vpo: Tensor, vg: Tensor, vv: Tensor, vt: Tensor, vq: Tensor, vk: Tensor, vr: Tensor, vc: Tensor, later: bool, r_cut: float, r_on: float, floor: bool, has_rho: bool, need_dpoly: bool, need_vir: bool, need_drho: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (torch.empty_like(pos), torch.empty_like(poly) if need_dpoly else pos.new_empty(0), torch.empty_like(gates), torch.empty_like(vec) if (later and vec.ndim == 3) else pos.new_empty(0), torch.empty_like(ten) if (later and ten.ndim == 3) else pos.new_empty(0), torch.empty_like(qn), torch.empty_like(kn), torch.empty_like(rho) if (need_drho and has_rho) else pos.new_empty(0), torch.empty_like(cell) if (need_vir and cell.ndim == 3) else pos.new_empty(0), pos.new_empty(0), torch.empty_like(gn), torch.empty_like(gz), torch.empty_like(gd))


def _setup_csr_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    rp, si, ri, sp, spm, sh_c, gr, z, rmin, pos, poly, gates, vec, ten, qn, kn, rho, cell, sh_d, later, r_cut, r_on, floor, has_rho = inputs
    ctx.used_shift_diff = bool(sh_d.ndim == 2)
    ctx.later, ctx.r_cut, ctx.r_on, ctx.floor, ctx.has_rho = bool(later), float(r_cut), float(r_on), bool(floor), bool(has_rho)
    ctx.has_vec, ctx.has_ten, ctx.has_cell = bool(later and vec.ndim == 3), bool(later and ten.ndim == 3), bool(cell.ndim == 3)
    ctx.save_for_backward(rp, si, ri, sp, spm, sh_d if ctx.used_shift_diff else sh_c, gr, z, rmin, pos, poly, gates, vec, ten, qn, kn, rho, cell)


def _csr_agg_autograd_bwd(ctx: Any, g_num: Tensor, g_zc: Tensor, g_dsum: Tensor):
    rp, si, ri, sp, spm, sh_in, gr, z, rmin, pos, poly, gates, vec, ten, qn, kn, rho, cell = ctx.saved_tensors
    nds = ctx.needs_input_grad
    need_poly, need_drho, need_vir, need_dshift = bool(nds[10]), bool(nds[16] and ctx.has_rho), bool(nds[17] and ctx.has_cell), bool(nds[18] and ctx.used_shift_diff)
    gn, gz, gd = g_num.contiguous(), g_zc.contiguous(), g_dsum.contiguous()
    use_bwd2_op = has_triton_csr_bwd2(pos.device) and pos.device.type == "cuda" and not need_dshift and (torch.is_grad_enabled() or need_poly)
    use_gpu_bwd_op = (not torch.is_grad_enabled()) and (not need_poly) and (pos.device.type == "mps" or (pos.device.type == "cpu" and has_cpu_csr()) or (pos.device.type == "cuda" and not need_dshift and has_triton_csr_backward(pos.device)))
    if use_bwd2_op:
        d_pos, d_poly, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell, d_shift = _csr_agg_bwd_p_op(rp, si, ri, sp, spm, sh_in, gr, z, pos, poly, gates, vec, ten, qn, kn, rho, cell, gn, gz, gd, ctx.later, ctx.r_cut, ctx.r_on, ctx.floor, ctx.has_rho, need_vir, need_drho, need_dshift, need_poly)
    elif use_gpu_bwd_op:
        d_poly = None
        d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell, d_shift = _csr_agg_bwd_op(rp, si, ri, sp, spm, sh_in, gr, z, pos, poly, gates, vec, ten, qn, kn, rho, cell, gn, gz, gd, ctx.later, ctx.r_cut, ctx.r_on, ctx.floor, ctx.has_rho, need_vir, need_drho, need_dshift)
    elif (not storable(pos)) or need_poly or torch.is_grad_enabled():
        outs = _ref_vjp_csr_backward(rp, si, ri, sp, spm, LATER if ctx.later else FIRST, EdgeCfg(r_cut=ctx.r_cut, r_on=ctx.r_on, floor=ctx.floor), rmin, pos, poly, gates, vec, ten, qn, kn, rho, sh_in, cell, gr, z, gn, gz, gd, need_vir=need_vir, need_drho=need_drho, need_dshift=need_dshift, need_dpoly=need_poly, create_graph=torch.is_grad_enabled(), live_attach=torch.is_grad_enabled())
        d_pos, d_poly, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell, d_shift = outs if need_poly else (outs[0], None, *outs[1:])
    else:
        d_poly = None
        d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell, d_shift = stream_vjp_csr_backward(CSRPlan(int(pos.shape[0]), int(si.shape[0]), rp, si, ri, sp, spm), LATER if ctx.later else FIRST, EdgeCfg(r_cut=ctx.r_cut, r_on=ctx.r_on, floor=ctx.floor), pos, poly, gates, vec, ten, qn, kn, rho, sh_in, cell, gr, z, gn, gz, gd, need_vir=need_vir, need_drho=need_drho, need_dshift=need_dshift)
    return (None, None, None, None, None, None, None, None, None, d_pos if nds[9] else None, d_poly if need_poly else None, d_gates if nds[11] else None, d_vec if (nds[12] and ctx.has_vec) else None, d_ten if (nds[13] and ctx.has_ten) else None, d_qn if nds[14] else None, d_kn if nds[15] else None, d_rho if need_drho else None, d_cell if need_vir else None, d_shift if need_dshift else None, None, None, None, None, None)


def _setup_bwd2_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    ctx.later, ctx.r_cut, ctx.r_on, ctx.floor, ctx.has_rho = bool(inputs[20]), float(inputs[21]), float(inputs[22]), bool(inputs[23]), bool(inputs[24])
    ctx.save_for_backward(*inputs[:20])


def _bwd2_autograd(has_poly: bool, ctx: Any, *vs: Tensor | None):
    rp, si, ri, sp, spm, sh, gr, z, pos, poly, gates, vec, ten, qn, kn, rho, cell, gn, gz, gd = ctx.saved_tensors
    zif = lambda t: pos.new_empty(0) if (t is None or t.numel() == 0) else t.contiguous()
    vp, vpo, vg, vv, vt, vq, vk, vr, vc = [zif(x) for x in vs[:9]] if has_poly else (zif(vs[0]), pos.new_empty(0), zif(vs[1]), zif(vs[2]), zif(vs[3]), zif(vs[4]), zif(vs[5]), zif(vs[6]), zif(vs[7]))
    nds = ctx.needs_input_grad
    r = _csr_agg_bwd2_op(rp, si, ri, sp, spm, sh, gr, z, pos, poly, gates, vec, ten, qn, kn, rho, cell, gn, gz, gd, vp, vpo, vg, vv, vt, vq, vk, vr, vc, ctx.later, ctx.r_cut, ctx.r_on, ctx.floor, ctx.has_rho, bool(nds[9]), bool(nds[16] and cell.ndim == 3), bool(nds[15] and ctx.has_rho))
    og = (r[0] if nds[8] else None, r[1] if nds[9] else None, r[2] if nds[10] else None, r[3] if (nds[11] and ctx.later and vec.ndim == 3) else None, r[4] if (nds[12] and ctx.later and ten.ndim == 3) else None, r[5] if nds[13] else None, r[6] if nds[14] else None, r[7] if (nds[15] and ctx.has_rho) else None, r[8] if (nds[16] and cell.ndim == 3) else None, r[10] if nds[17] else None, r[11] if nds[18] else None, r[12] if nds[19] else None)
    return (None,) * 8 + og + (None,) * (9 if has_poly else 8)


torch.library.register_autograd("lam4::csr_agg", _csr_agg_autograd_bwd, setup_context=_setup_csr_ctx)
torch.library.register_autograd("lam4::csr_agg_bwd_p", lambda ctx, *vs: _bwd2_autograd(True, ctx, *vs), setup_context=_setup_bwd2_ctx)
torch.library.register_autograd("lam4::csr_agg_bwd", lambda ctx, *vs: _bwd2_autograd(False, ctx, *vs), setup_context=_setup_bwd2_ctx)


class CSRAgg(torch.autograd.Function):
    """Atomic-free Dual-CSR edge aggregation returning `(num [N, out_dim], zc [N, 4], dsum [N])`."""

    @staticmethod
    def forward(ctx, plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, shift_const: Tensor, graph: Tensor, z: Tensor, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor, shift_diff: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        shift_in = shift_diff if shift_diff.ndim == 2 else shift_const
        ctx.plan, ctx.layout, ctx.cfg, ctx.used_shift_diff = plan, layout, cfg, bool(shift_diff.ndim == 2)
        ctx.save_for_backward(shift_in, graph, z, pos, poly, gates, vec, ten, qn, kn, rho, cell)
        return _dispatch_fwd(plan, layout, cfg, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z)

    @staticmethod
    def backward(ctx, g_num: Tensor, g_zc: Tensor, g_dsum: Tensor):
        shift_in, graph, z, pos, poly, gates, vec, ten, qn, kn, rho, cell = ctx.saved_tensors
        nds = ctx.needs_input_grad
        need_vir, need_drho, need_dshift = bool(nds[14] and cell.ndim == 3 and cell.numel() > 0), bool(nds[13] and rho.numel() > 0), bool(nds[15] and ctx.used_shift_diff)
        fn = stream_vjp_csr_backward if torch.is_grad_enabled() else _dispatch_bwd
        d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell, d_shift = fn(ctx.plan, ctx.layout, ctx.cfg, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z, g_num.contiguous(), g_zc.contiguous(), g_dsum.contiguous(), need_vir=need_vir, need_drho=need_drho, need_dshift=need_dshift)
        outs = (d_pos, None, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell, d_shift)
        return (None, None, None, None, None, None, *(g if (nd and g is not None) else None for g, nd in zip(outs, nds[6:])))


def edge_aggregate_csr(
    pos: Tensor, recv: Tensor, send: Tensor, shift: Tensor | None, coef: Tensor, gates: Tensor,
    q: Tensor | None, k: Tensor | None, *, layout: StageLayout, vec: Tensor | None = None,
    ten: Tensor | None = None, rho: Tensor | None = None, edge_scale: Tensor | None = None,
    qk_gain: Tensor | None = None, cfg: EdgeCfg = EdgeCfg(), z: Tensor | None = None,
    csr_plan: CSRPlan | None = None, poly: Tensor | None = None, poly4: Tensor | None = None, edge_shift: Tensor | None = None,
    cell: Tensor | None = None, graph: Tensor | None = None, create_graph: bool = False,
    low_memory_fallback: bool = False, allow_bwd2: bool = False, return_dsum: bool = False,
):
    """Fused Dual-CSR edge aggregation (`[N, layout.out_dim]`), with automatic exact PyTorch fallback."""
    _check(layout, pos, gates, coef, vec, q, edge_scale, ten)
    if not isinstance(recv.shape[0], torch.SymInt) and recv.numel() == 0:
        n, od = pos.shape[0], layout.out_dim
        z0 = (pos.sum() * 0.0) if pos.requires_grad else 0.0
        out = finalise(pos.new_zeros((n, od)) + z0, pos.new_zeros((n, 4)), pos.new_zeros(n), layout)
        return (out, pos.new_zeros(n)) if return_dsum else out
    if not can_use_csr(pos, coef, q, k, layout=layout, cfg=cfg, edge_scale=edge_scale, create_graph=create_graph, allow_bwd2=allow_bwd2):
        if shift is None:
            assert edge_shift is not None and cell is not None
            shift = (edge_shift @ cell[0]) if cell.shape[0] == 1 else torch.einsum("ek,ekl->el", edge_shift, cell[graph[recv]])
        kw = dict(layout=layout, vec=vec, ten=ten, rho=rho, edge_scale=edge_scale, qk_gain=qk_gain, cfg=cfg, z=z)
        out = (edge_aggregate_stream if low_memory_fallback else edge_aggregate_ref)(pos, recv, send, shift, coef, gates, q, k, **kw)
        if not return_dsum:
            return out
        from lam4.core.edge_terms import EdgeGraph
        from lam4.core.inv_scale import compute_dsum

        return out, compute_dsum(EdgeGraph(pos=pos, z=z, send=send, recv=recv, shift=shift), cfg)
    assert q is not None and k is not None
    plan = build_csr_plan(recv, send, pos.shape[0]) if csr_plan is None else csr_plan
    poly_c = coef_to_horner_poly(coef, cfg.r_cut) if poly is None else poly.contiguous()
    qn_c, kn_c = (x.contiguous() for x in _gain(normalise_qk(q, k), qk_gain))
    empty_f, empty_i = pos.new_zeros(0), pos.new_zeros(0, dtype=torch.int64)
    pos_c, gates_c = pos.contiguous(), gates.contiguous()
    vec_c = vec.contiguous() if (layout.later and vec is not None and vec.ndim == 3) else empty_f
    ten_c = ten.contiguous() if (layout.later and ten is not None and ten.ndim == 3 and _present(ten)) else empty_f
    has_rho = _present(rho)
    rho_c = rho.contiguous() if (has_rho and rho is not None) else empty_f
    z_c = z.to(torch.int64).contiguous() if (cfg.floor and _present(z) and z is not None) else empty_i
    if cell is not None and edge_shift is not None:
        cell_c, graph_c = cell.contiguous(), (graph.to(torch.int64).contiguous() if (graph is not None and cell.shape[0] > 1) else empty_i)
        sh_const, sh_diff = edge_shift.to(pos.dtype).contiguous(), empty_f
    else:
        assert shift is not None
        cell_c, graph_c, sh_32 = empty_f, empty_i, shift.to(pos.dtype).contiguous()
        sh_const, sh_diff = (empty_f, sh_32) if shift.requires_grad else (sh_32, empty_f)
    use_cuda_fused = (
        pos_c.device.type == "cuda"
        and not create_graph
        and not coef.requires_grad
        and sh_diff.numel() == 0
        and has_csr_custom_op()
        and has_triton_csr_backward(pos_c.device)
        and has_cuda_csr(pos_c.device)
    )
    if use_cuda_fused:
        p4_in = poly4.contiguous() if poly4 is not None else poly_c
        return cuda_csr_aggregate_final(
            plan, layout, cfg, sh_const, graph_c, z_c, pos_c, p4_in,
            gates_c, vec_c, ten_c, qn_c, kn_c, rho_c, cell_c,
            has_rho=has_rho, return_dsum=return_dsum,
        )
    use_op = has_csr_custom_op() and not (not torch.compiler.is_compiling() and pos.device.type == "cuda" and not has_triton_csr_backward(pos.device))
    if use_op:
        rmin_c = (cpu_rmin_table(pos.dtype) if pos.device.type == "cpu" else rmin_table_f32(pos.device)) if cfg.floor else empty_f
        num, zc, dsum = _csr_agg_op(plan.recv_ptr, plan.send_idx, plan.recv_idx, plan.send_ptr, plan.send_perm, sh_const, graph_c, z_c, rmin_c, pos_c, poly_c, gates_c, vec_c, ten_c, qn_c, kn_c, rho_c, cell_c, sh_diff, bool(layout.later), float(cfg.r_cut), float(cfg.r_on), bool(cfg.floor), bool(has_rho))
    else:
        num, zc, dsum = CSRAgg.apply(plan, layout, cfg, sh_const, graph_c, z_c, pos_c, poly_c, gates_c, vec_c, ten_c, qn_c, kn_c, rho_c, cell_c, sh_diff)
    out = finalise(num, zc, dsum, layout)
    return (out, dsum) if return_dsum else out
