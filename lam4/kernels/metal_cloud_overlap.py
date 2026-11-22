"""Dual-CSR Metal custom ops & autograd for CloudOverlapHead on MPS (zero [E, C] edge buffers)."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lam4.kernels.csr_plan import CSRPlan
from lam4.kernels import metal_csr as _metal_csr
from lam4.kernels.metal_cloud_overlap_src import METAL_CLOUD_OV_MSL
from lam4.kernels.metal_node_flags import MetalShader

_CLOUD_OV_SHADER = MetalShader("metal_cloud_overlap", METAL_CLOUD_OV_MSL)
_FLAG_SHIFT = 1
_FLAG_CELL = 2
_FLAG_G1 = 4
_FLAG_VIR = 8
_FLAG_DPOS = 16


def has_metal_cloud_ov() -> bool:
    return bool(_metal_csr.has_metal_csr() and _CLOUD_OV_SHADER.available())


def get_metal_cloud_ov_lib() -> Any:
    return _CLOUD_OV_SHADER.lib()


def can_use_metal_cloud_ov(
    pos: Tensor,
    poly: Tensor,
    k_ex: int,
    k_an: int,
    k_t: int,
    *,
    create_graph: bool = False,
    shift_in: Tensor | None = None,
) -> bool:
    return (
        not create_graph
        and pos.dtype == torch.float32
        and pos.device.type == "mps"
        and has_metal_cloud_ov()
        and not poly.requires_grad
        and not (shift_in is not None and shift_in.requires_grad)
        and (int(k_ex), int(k_an), int(k_t)) == (16, 8, 4)
    )


def _grid_flags(grid: int, shift_in: Tensor, cell: Tensor, *, need_vir: bool = False, need_dpos: bool = False) -> int:
    flags = 0
    if shift_in.numel() > 0:
        flags |= _FLAG_SHIFT
    if cell.numel() > 0:
        flags |= _FLAG_CELL
        if cell.shape[0] == 1:
            flags |= _FLAG_G1
    if need_vir:
        flags |= _FLAG_VIR
    if need_dpos:
        flags |= _FLAG_DPOS
    return (int(grid) << 8) | flags


@torch.library.custom_op("lam4::metal_cloud_ov_fwd", mutates_args=())
def _metal_cloud_ov_fwd_op(
    pos: Tensor, poly: Tensor, a_p: Tensor, c1: Tensor, d1: Tensor, c2: Tensor, d2: Tensor,
    atom_mask_u8: Tensor, recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor, r_cut: float, r_on: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    n, n_e = int(pos.shape[0]), int(send_idx.shape[0])
    dev = pos.device
    out_ap = torch.zeros((n, 16), dtype=torch.float32, device=dev)
    out_ds = torch.zeros((n, 8), dtype=torch.float32, device=dev)
    out_v = torch.zeros((n, 8, 3), dtype=torch.float32, device=dev)
    out_ys = torch.zeros((n, 4), dtype=torch.float32, device=dev)
    out_yt = torch.zeros((n, 4, 5), dtype=torch.float32, device=dev)
    if n == 0 or n_e == 0:
        return out_ap, out_ds, out_v, out_ys, out_yt
    lib = get_metal_cloud_ov_lib()
    df, di = pos.new_zeros(1), graph.new_zeros(1) if graph.numel() else torch.zeros(1, dtype=torch.int64, device=dev)
    grid = int(poly.shape[0])
    gf = _grid_flags(grid, shift_in, cell)
    params = (float(r_cut), float(r_on), 1.0 / (float(r_cut) - float(r_on)), float(grid) / float(r_cut))
    n_tot = ((n + 3) // 4) * 128
    lib.cloud_ov_fwd_recv(
        pos.contiguous(), recv_ptr, send_idx, atom_mask_u8.contiguous(),
        shift_in.contiguous() if shift_in.numel() else df,
        cell.contiguous() if cell.numel() else df,
        graph.contiguous() if graph.numel() else di,
        poly.contiguous(), a_p.contiguous(), c1.contiguous(), d1.contiguous(), c2.contiguous(), d2.contiguous(),
        out_ap, out_ds, out_v, out_ys, out_yt,
        n, gf, params, threads=(n_tot, 1, 1), group_size=(128, 1, 1),
    )
    return out_ap, out_ds, out_v, out_ys, out_yt


@_metal_cloud_ov_fwd_op.register_fake
def _(
    pos: Tensor, poly: Tensor, a_p: Tensor, c1: Tensor, d1: Tensor, c2: Tensor, d2: Tensor,
    atom_mask_u8: Tensor, recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor, r_cut: float, r_on: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    n, dev = pos.shape[0], pos.device
    return (
        torch.empty((n, 16), dtype=torch.float32, device=dev),
        torch.empty((n, 8), dtype=torch.float32, device=dev),
        torch.empty((n, 8, 3), dtype=torch.float32, device=dev),
        torch.empty((n, 4), dtype=torch.float32, device=dev),
        torch.empty((n, 4, 5), dtype=torch.float32, device=dev),
    )


@torch.library.custom_op("lam4::metal_cloud_ov_bwd", mutates_args=())
def _metal_cloud_ov_bwd_op(
    pos: Tensor, poly: Tensor, a_p: Tensor, c1: Tensor, d1: Tensor, c2: Tensor, d2: Tensor,
    atom_mask_u8: Tensor, recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor,
    gp: Tensor, gds: Tensor, gv: Tensor, gys: Tensor, gyt: Tensor,
    r_cut: float, r_on: float, need_dpos: bool, need_dcell: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    n, n_e = int(pos.shape[0]), int(send_idx.shape[0])
    dev = pos.device
    d_pos = torch.zeros((n, 3), dtype=torch.float32, device=dev)
    d_cell = torch.zeros(cell.shape, dtype=torch.float32, device=dev) if (cell.numel() > 0) else torch.zeros((0, 3, 3), dtype=torch.float32, device=dev)
    d_ap = torch.zeros(a_p.shape, dtype=torch.float32, device=dev)
    d_c1 = torch.zeros(c1.shape, dtype=torch.float32, device=dev)
    d_d1 = torch.zeros(d1.shape, dtype=torch.float32, device=dev)
    d_c2 = torch.zeros(c2.shape, dtype=torch.float32, device=dev)
    d_d2 = torch.zeros(d2.shape, dtype=torch.float32, device=dev)
    if n == 0 or n_e == 0:
        return d_pos, d_cell, d_ap, d_c1, d_d1, d_c2, d_d2
    lib = get_metal_cloud_ov_lib()
    df, di = pos.new_zeros(1), graph.new_zeros(1) if graph.numel() else torch.zeros(1, dtype=torch.int64, device=dev)
    grid = int(poly.shape[0])
    need_vir = bool(need_dcell and cell.ndim == 3 and cell.numel() > 0 and shift_in.numel() > 0)
    gf = _grid_flags(grid, shift_in, cell, need_vir=need_vir, need_dpos=need_dpos)
    params = (float(r_cut), float(r_on), 1.0 / (float(r_cut) - float(r_on)), float(grid) / float(r_cut))
    pos_c = pos.contiguous()
    mask_c = atom_mask_u8.contiguous()
    sh_c = shift_in.contiguous() if shift_in.numel() else df
    cell_c = cell.contiguous() if cell.numel() else df
    gr_c = graph.contiguous() if graph.numel() else di
    poly_c = poly.contiguous()
    ap_c, c1_c, d1_c, c2_c, d2_c = a_p.contiguous(), c1.contiguous(), d1.contiguous(), c2.contiguous(), d2.contiguous()
    gp_c, gds_c, gv_c, gys_c, gyt_c = gp.contiguous(), gds.contiguous(), gv.contiguous(), gys.contiguous(), gyt.contiguous()

    n_tot = ((n + 3) // 4) * 128
    vir_atom = torch.empty((n, 9), dtype=torch.float32, device=dev) if need_vir else df
    if need_dpos or need_vir:
        lib.cloud_ov_bwd_recv(
            pos_c, recv_ptr, send_idx, mask_c, sh_c, cell_c, gr_c, poly_c,
            ap_c, c1_c, d1_c, c2_c, d2_c, gp_c, gds_c, gv_c, gys_c, gyt_c,
            d_pos if need_dpos else df, vir_atom,
            n, gf, params, threads=(n_tot, 1, 1), group_size=(128, 1, 1),
        )
    lib.cloud_ov_bwd_send(
        pos_c, send_ptr, send_perm, recv_idx, mask_c, sh_c, cell_c, gr_c, poly_c,
        ap_c, c1_c, d1_c, c2_c, d2_c, gp_c, gds_c, gv_c, gys_c, gyt_c,
        d_pos if need_dpos else df, d_ap, d_c1, d_d1, d_c2, d_d2,
        n, gf, params, threads=(n_tot, 1, 1), group_size=(128, 1, 1),
    )
    if need_vir:
        if cell.shape[0] == 1:
            d_cell = vir_atom.sum(dim=0).view(1, 3, 3)
        else:
            d_cell = vir_atom.new_zeros(cell.shape[0], 9).index_add_(0, graph, vir_atom).view(-1, 3, 3)
    return d_pos, d_cell, d_ap, d_c1, d_d1, d_c2, d_d2


@_metal_cloud_ov_bwd_op.register_fake
def _(
    pos: Tensor, poly: Tensor, a_p: Tensor, c1: Tensor, d1: Tensor, c2: Tensor, d2: Tensor,
    atom_mask_u8: Tensor, recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor,
    gp: Tensor, gds: Tensor, gv: Tensor, gys: Tensor, gyt: Tensor,
    r_cut: float, r_on: float, need_dpos: bool, need_dcell: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty(pos.shape, dtype=torch.float32, device=pos.device),
        torch.empty(cell.shape, dtype=torch.float32, device=cell.device) if cell.numel() else cell.new_empty((0, 3, 3), dtype=torch.float32),
        torch.empty(a_p.shape, dtype=torch.float32, device=a_p.device),
        torch.empty(c1.shape, dtype=torch.float32, device=c1.device),
        torch.empty(d1.shape, dtype=torch.float32, device=d1.device),
        torch.empty(c2.shape, dtype=torch.float32, device=c2.device),
        torch.empty(d2.shape, dtype=torch.float32, device=d2.device),
    )


def _setup_ov_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    ctx.save_for_backward(*inputs[:16])
    ctx.r_cut, ctx.r_on = float(inputs[16]), float(inputs[17])


def _bwd_ov_autograd(ctx: Any, gp: Tensor | None, gds: Tensor | None, gv: Tensor | None, gys: Tensor | None, gyt: Tensor | None):
    saved = ctx.saved_tensors
    pos, poly, a_p, c1, d1, c2, d2 = saved[:7]
    nds = ctx.needs_input_grad
    need_dp = bool(nds[0])
    need_dc = bool(nds[14] and saved[14].ndim == 3 and saved[14].numel() > 0)
    if not (need_dp or need_dc or any(nds[2:7])):
        return (None,) * 18
    gp_ = gp if gp is not None else torch.zeros_like(a_p, dtype=torch.float32)
    gds_ = gds if gds is not None else torch.zeros_like(d1, dtype=torch.float32)
    gv_ = gv if gv is not None else torch.zeros_like(c1, dtype=torch.float32)
    gys_ = gys if gys is not None else torch.zeros_like(d2, dtype=torch.float32)
    gyt_ = gyt if gyt is not None else torch.zeros_like(c2, dtype=torch.float32)
    dp, dc, dap, dc1, dd1, dc2, dd2 = _metal_cloud_ov_bwd_op(
        *saved, gp_, gds_, gv_, gys_, gyt_, ctx.r_cut, ctx.r_on, need_dp, need_dc,
    )
    return (
        dp.to(pos.dtype) if nds[0] else None,
        None,
        dap.to(a_p.dtype) if nds[2] else None,
        dc1.to(c1.dtype) if nds[3] else None,
        dd1.to(d1.dtype) if nds[4] else None,
        dc2.to(c2.dtype) if nds[5] else None,
        dd2.to(d2.dtype) if nds[6] else None,
        None, None, None, None, None, None, None,
        dc.to(saved[14].dtype) if need_dc else None,
        None, None, None,
    )


def _ov_bwd2_unsupported(*_args: Any, **_kw: Any):
    raise NotImplementedError("lam4::metal_cloud_ov_bwd supports 1st-order backward only; route create_graph=True to PyTorch.")


_metal_cloud_ov_fwd_op.register_autograd(_bwd_ov_autograd, setup_context=_setup_ov_ctx)
_metal_cloud_ov_bwd_op.register_autograd(_ov_bwd2_unsupported, setup_context=lambda ctx, inputs, output: None)


def metal_cloud_overlap_csr(
    pos: Tensor, poly: Tensor, a_p: Tensor, c1: Tensor, d1: Tensor, c2: Tensor, d2: Tensor,
    atom_mask: Tensor, csr_plan: CSRPlan,
    edge_shift: Tensor | None, cell: Tensor | None, graph: Tensor | None,
    r_cut: float, r_on: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    empty_f, empty_i = pos.new_zeros(0, dtype=torch.float32), pos.new_zeros(0, dtype=torch.int64)
    cell_t = cell if (cell is not None and cell.numel() > 0) else empty_f
    sh_t = edge_shift.to(torch.float32) if (edge_shift is not None and edge_shift.numel() > 0) else empty_f
    gr_t = graph.to(torch.int64) if (graph is not None and cell_t.numel() > 0 and cell_t.shape[0] > 1) else empty_i
    return _metal_cloud_ov_fwd_op(
        pos.to(torch.float32).contiguous(), poly.to(torch.float32).contiguous(),
        a_p.to(torch.float32).contiguous(), c1.to(torch.float32).contiguous(), d1.to(torch.float32).contiguous(),
        c2.to(torch.float32).contiguous(), d2.to(torch.float32).contiguous(),
        atom_mask.to(torch.uint8).contiguous(),
        csr_plan.recv_ptr, csr_plan.send_idx, csr_plan.recv_idx, csr_plan.send_ptr, csr_plan.send_perm,
        sh_t.contiguous(), cell_t.contiguous(), gr_t.contiguous(), float(r_cut), float(r_on),
    )
