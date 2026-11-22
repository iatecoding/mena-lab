"""Dual-CSR Metal custom ops & autograd for SpinExchangeHead on MPS (zero [E, C] edge buffers)."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lam4.kernels.csr_plan import CSRPlan
from lam4.kernels import metal_csr as _metal_csr
from lam4.kernels.metal_node_flags import MetalShader
from lam4.kernels.metal_spin_exchange_src import METAL_SPIN_EXCH_MSL

_SPIN_EXCH_SHADER = MetalShader("metal_spin_exchange", METAL_SPIN_EXCH_MSL)
_FLAG_SHIFT = 1
_FLAG_CELL = 2
_FLAG_G1 = 4
_FLAG_VIR = 8
_FLAG_DPOS = 16


def has_metal_spin_exch() -> bool:
    return bool(_metal_csr.has_metal_csr() and _SPIN_EXCH_SHADER.available())


def get_metal_spin_exch_lib() -> Any:
    return _SPIN_EXCH_SHADER.lib()


def can_use_metal_spin_exch(
    pos: Tensor,
    w_dir: Tensor,
    w_cov: Tensor,
    rank: int,
    n_rbf: int,
    *,
    create_graph: bool = False,
    shift_in: Tensor | None = None,
) -> bool:
    return (
        not create_graph
        and pos.dtype == torch.float32
        and pos.device.type == "mps"
        and has_metal_spin_exch()
        and not w_dir.requires_grad
        and not w_cov.requires_grad
        and not (shift_in is not None and shift_in.requires_grad)
        and int(rank) == 16
        and int(n_rbf) == 32
    )


def _exch_flags(shift_in: Tensor, cell: Tensor, *, need_vir: bool = False, need_dpos: bool = False) -> int:
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
    return flags


@torch.library.custom_op("lam4::metal_spin_exch_fwd", mutates_args=())
def _metal_spin_exch_fwd_op(
    pos: Tensor, h_dir: Tensor, h_cov: Tensor, m_t: Tensor, u_rad: Tensor,
    w_dir: Tensor, w_cov: Tensor, centres: Tensor, atom_mask_u8: Tensor,
    recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor, r_cut: float, sigma: float,
) -> Tensor:
    n, n_e = int(pos.shape[0]), int(send_idx.shape[0])
    dev = pos.device
    out_pair = torch.zeros(n, dtype=torch.float32, device=dev)
    if n == 0 or n_e == 0:
        return out_pair
    lib = get_metal_spin_exch_lib()
    df = pos.new_zeros(1)
    di = graph.new_zeros(1) if graph.numel() else torch.zeros(1, dtype=torch.int64, device=dev)
    flags = _exch_flags(shift_in, cell)
    params = (float(r_cut), float(r_cut) * float(r_cut), 1.0 / float(sigma), 0.0)
    n_tot = ((n + 3) // 4) * 128
    lib.spin_exch_fwd_recv(
        pos.contiguous(), recv_ptr, send_idx, atom_mask_u8.contiguous(),
        shift_in.contiguous() if shift_in.numel() else df,
        cell.contiguous() if cell.numel() else df,
        graph.contiguous() if graph.numel() else di,
        h_dir.contiguous(), h_cov.contiguous(), m_t.contiguous(), u_rad.contiguous(),
        w_dir.contiguous(), w_cov.contiguous(), centres.contiguous(),
        out_pair, n, flags, params, threads=(n_tot, 1, 1), group_size=(128, 1, 1),
    )
    return out_pair


@_metal_spin_exch_fwd_op.register_fake
def _(
    pos: Tensor, h_dir: Tensor, h_cov: Tensor, m_t: Tensor, u_rad: Tensor,
    w_dir: Tensor, w_cov: Tensor, centres: Tensor, atom_mask_u8: Tensor,
    recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor, r_cut: float, sigma: float,
) -> Tensor:
    return torch.empty(pos.shape[0], dtype=torch.float32, device=pos.device)


@torch.library.custom_op("lam4::metal_spin_exch_bwd", mutates_args=())
def _metal_spin_exch_bwd_op(
    pos: Tensor, h_dir: Tensor, h_cov: Tensor, m_t: Tensor, u_rad: Tensor,
    w_dir: Tensor, w_cov: Tensor, centres: Tensor, atom_mask_u8: Tensor,
    recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor, g_out: Tensor,
    r_cut: float, sigma: float, need_dpos: bool, need_dcell: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    n, n_e = int(pos.shape[0]), int(send_idx.shape[0])
    dev = pos.device
    d_pos = torch.zeros((n, 3), dtype=torch.float32, device=dev)
    d_hdir = torch.zeros(h_dir.shape, dtype=torch.float32, device=dev)
    d_hcov = torch.zeros(h_cov.shape, dtype=torch.float32, device=dev)
    d_m = torch.zeros(m_t.shape, dtype=torch.float32, device=dev)
    d_u = torch.zeros(u_rad.shape, dtype=torch.float32, device=dev)
    d_cell = (
        torch.zeros(cell.shape, dtype=torch.float32, device=dev)
        if cell.numel() > 0
        else torch.zeros((0, 3, 3), dtype=torch.float32, device=dev)
    )
    if n == 0 or n_e == 0:
        return d_pos, d_hdir, d_hcov, d_m, d_u, d_cell
    lib = get_metal_spin_exch_lib()
    df = pos.new_zeros(1)
    di = graph.new_zeros(1) if graph.numel() else torch.zeros(1, dtype=torch.int64, device=dev)
    need_vir = bool(need_dcell and cell.ndim == 3 and cell.numel() > 0 and shift_in.numel() > 0)
    flags = _exch_flags(shift_in, cell, need_vir=need_vir, need_dpos=need_dpos)
    params = (float(r_cut), float(r_cut) * float(r_cut), 1.0 / float(sigma), 0.0)

    pos_c, mask_c = pos.contiguous(), atom_mask_u8.contiguous()
    sh_c = shift_in.contiguous() if shift_in.numel() else df
    cell_c = cell.contiguous() if cell.numel() else df
    gr_c = graph.contiguous() if graph.numel() else di
    hd_c, hc_c = h_dir.contiguous(), h_cov.contiguous()
    m_c, u_c = m_t.contiguous(), u_rad.contiguous()
    wd_c, wc_c, ct_c = w_dir.contiguous(), w_cov.contiguous(), centres.contiguous()
    g_c = g_out.contiguous()

    n_tot = ((n + 3) // 4) * 128
    vir_atom = torch.empty((n, 9), dtype=torch.float32, device=dev) if need_vir else df
    lib.spin_exch_bwd_recv(
        pos_c, recv_ptr, send_idx, mask_c, sh_c, cell_c, gr_c,
        hd_c, hc_c, m_c, u_c, wd_c, wc_c, ct_c, g_c,
        d_pos if need_dpos else df, d_hdir, d_hcov, d_m, d_u, vir_atom,
        n, flags, params, threads=(n_tot, 1, 1), group_size=(128, 1, 1),
    )
    lib.spin_exch_bwd_send(
        pos_c, send_ptr, send_perm, recv_idx, mask_c, sh_c, cell_c, gr_c,
        hd_c, hc_c, m_c, u_c, wd_c, wc_c, ct_c, g_c,
        d_pos if need_dpos else df, d_hdir, d_hcov, d_m, d_u,
        n, flags, params, threads=(n_tot, 1, 1), group_size=(128, 1, 1),
    )
    if need_vir:
        if cell.shape[0] == 1:
            d_cell = vir_atom.sum(dim=0).view(1, 3, 3)
        else:
            d_cell = vir_atom.new_zeros(cell.shape[0], 9).index_add_(0, graph, vir_atom).view(-1, 3, 3)
    return d_pos, d_hdir, d_hcov, d_m, d_u, d_cell


@_metal_spin_exch_bwd_op.register_fake
def _(
    pos: Tensor, h_dir: Tensor, h_cov: Tensor, m_t: Tensor, u_rad: Tensor,
    w_dir: Tensor, w_cov: Tensor, centres: Tensor, atom_mask_u8: Tensor,
    recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor, g_out: Tensor,
    r_cut: float, sigma: float, need_dpos: bool, need_dcell: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty(pos.shape, dtype=torch.float32, device=pos.device),
        torch.empty(h_dir.shape, dtype=torch.float32, device=h_dir.device),
        torch.empty(h_cov.shape, dtype=torch.float32, device=h_cov.device),
        torch.empty(m_t.shape, dtype=torch.float32, device=m_t.device),
        torch.empty(u_rad.shape, dtype=torch.float32, device=u_rad.device),
        torch.empty(cell.shape, dtype=torch.float32, device=cell.device)
        if cell.numel()
        else cell.new_empty((0, 3, 3), dtype=torch.float32),
    )


def _setup_se_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    ctx.save_for_backward(*inputs[:17])
    ctx.r_cut, ctx.sigma = float(inputs[17]), float(inputs[18])


def _bwd_se_autograd(ctx: Any, g_out: Tensor | None):
    saved = ctx.saved_tensors
    pos, h_dir, h_cov, m_t, u_rad = saved[:5]
    nds = ctx.needs_input_grad
    need_dp = bool(nds[0])
    need_dc = bool(nds[15] and saved[15].ndim == 3 and saved[15].numel() > 0)
    if g_out is None or not (need_dp or need_dc or any(nds[1:5])):
        return (None,) * 19
    dp, dhd, dhc, dm, du, dc = _metal_spin_exch_bwd_op(
        *saved, g_out.to(torch.float32), ctx.r_cut, ctx.sigma, need_dp, need_dc,
    )
    return (
        dp.to(pos.dtype) if nds[0] else None,
        dhd.to(h_dir.dtype) if nds[1] else None,
        dhc.to(h_cov.dtype) if nds[2] else None,
        dm.to(m_t.dtype) if nds[3] else None,
        du.to(u_rad.dtype) if nds[4] else None,
        None, None, None, None, None, None, None, None, None, None,
        dc.to(saved[15].dtype) if need_dc else None,
        None, None, None,
    )


def _se_bwd2_unsupported(*_args: Any, **_kw: Any):
    raise NotImplementedError("lam4::metal_spin_exch_bwd supports 1st-order backward only; route create_graph=True to PyTorch.")


_metal_spin_exch_fwd_op.register_autograd(_bwd_se_autograd, setup_context=_setup_se_ctx)
_metal_spin_exch_bwd_op.register_autograd(_se_bwd2_unsupported, setup_context=lambda ctx, inputs, output: None)


def metal_spin_exchange_csr(
    pos: Tensor,
    h_dir: Tensor,
    h_cov: Tensor,
    m_t: Tensor,
    u_rad: Tensor,
    w_dir: Tensor,
    w_cov: Tensor,
    centres: Tensor,
    atom_mask: Tensor,
    csr_plan: CSRPlan,
    edge_shift: Tensor | None,
    cell: Tensor | None,
    graph: Tensor | None,
    r_cut: float,
    sigma: float,
) -> Tensor:
    empty_f, empty_i = pos.new_zeros(0, dtype=torch.float32), pos.new_zeros(0, dtype=torch.int64)
    cell_t = cell if (cell is not None and cell.numel() > 0) else empty_f
    sh_t = edge_shift.to(torch.float32) if (edge_shift is not None and edge_shift.numel() > 0) else empty_f
    gr_t = graph.to(torch.int64) if (graph is not None and cell_t.numel() > 0 and cell_t.shape[0] > 1) else empty_i
    return _metal_spin_exch_fwd_op(
        pos.to(torch.float32).contiguous(),
        h_dir.to(torch.float32).contiguous(),
        h_cov.to(torch.float32).contiguous(),
        m_t.to(torch.float32).contiguous(),
        u_rad.to(torch.float32).contiguous(),
        w_dir.to(torch.float32).contiguous(),
        w_cov.to(torch.float32).contiguous(),
        centres.to(torch.float32).contiguous(),
        atom_mask.to(torch.uint8).contiguous(),
        csr_plan.recv_ptr, csr_plan.send_idx, csr_plan.recv_idx, csr_plan.send_ptr, csr_plan.send_perm,
        sh_t.contiguous(), cell_t.contiguous(), gr_t.contiguous(), float(r_cut), float(sigma),
    )
