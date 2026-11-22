"""Apple Silicon Metal Dual-CSR runtime compiler & dispatcher (`torch.mps.compile_shader`)."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lam4.core.edge_terms import EdgeCfg
from lam4.core.layout import StageLayout
from lam4.kernels.csr_plan import MAX_DOFF_R2, CSRPlan, rmin_table_f32
from lam4.kernels.metal_src import METAL_CSR_SOURCE

FLAG_FLOOR = 1
FLAG_RHO   = 2
FLAG_TEN   = 4
FLAG_CELL  = 8
FLAG_G1    = 16
FLAG_VIR   = 32
FLAG_DRHO  = 64

_METAL_LIB: Any = None
_METAL_FAILED: bool = False
_DUMMY_F32: dict[tuple[str, int | None], Tensor] = {}
_DUMMY_I64: dict[tuple[str, int | None], Tensor] = {}


def has_metal_csr() -> bool:
    if _METAL_FAILED or not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")):
        return False
    if _METAL_LIB is None and not torch.compiler.is_compiling():
        try:
            get_metal_csr_lib()
        except Exception:  # noqa: BLE001
            return False
    return True


def get_metal_csr_lib() -> Any:
    global _METAL_LIB, _METAL_FAILED
    if _METAL_LIB is None:
        try:
            _METAL_LIB = torch.mps.compile_shader(METAL_CSR_SOURCE)
        except Exception:
            _METAL_FAILED = True
            raise
    return _METAL_LIB


def _dummy_f32(dev: torch.device) -> Tensor:
    key = (dev.type, dev.index)
    t = _DUMMY_F32.get(key)
    if t is None:
        t = torch.zeros(1, dtype=torch.float32, device=dev)
        _DUMMY_F32[key] = t
    return t


def _dummy_i64(dev: torch.device) -> Tensor:
    key = (dev.type, dev.index)
    t = _DUMMY_I64.get(key)
    if t is None:
        t = torch.zeros(1, dtype=torch.int64, device=dev)
        _DUMMY_I64[key] = t
    return t


def metal_build_csr_plan(recv: Tensor, send: Tensor, n_atoms: int) -> CSRPlan:
    """O(E) GPU counting-sort Dual-CSR builder on Metal."""
    dev = recv.device
    n_e = int(recv.numel())
    if n_atoms <= 0 or n_e == 0:
        zp = torch.zeros(max(0, n_atoms) + 1, dtype=torch.int32, device=dev)
        zi = torch.zeros(0, dtype=torch.int32, device=dev)
        return CSRPlan(max(0, n_atoms), 0, zp, zi, zi, zp.clone(), zi.clone())
    lib = get_metal_csr_lib()
    recv_i32 = recv.to(torch.int32).contiguous()
    send_i32 = send.to(torch.int32).contiguous()
    recv_ptr = torch.zeros(n_atoms + 1, dtype=torch.int32, device=dev)
    recv_ptr[1:] = torch.searchsorted(recv_i32, torch.arange(n_atoms, dtype=torch.int32, device=dev), right=True).to(torch.int32)

    deg_send = torch.zeros(n_atoms, dtype=torch.int32, device=dev)
    tg = 256
    tot = ((n_e + tg - 1) // tg) * tg
    lib.csr_count_deg(send_i32, deg_send, n_e, threads=(tot, 1, 1), group_size=(tg, 1, 1))
    send_ptr = torch.zeros(n_atoms + 1, dtype=torch.int32, device=dev)
    torch.cumsum(deg_send, dim=0, dtype=torch.int32, out=send_ptr[1:])
    cur_send = send_ptr[:-1].clone()
    send_perm = torch.empty(n_e, dtype=torch.int32, device=dev)
    lib.csr_scatter_send(send_i32, cur_send, send_perm, n_e, threads=(tot, 1, 1), group_size=(tg, 1, 1))
    return CSRPlan(int(n_atoms), n_e, recv_ptr, send_i32, recv_i32, send_ptr, send_perm)


def _pack_flags(cfg: EdgeCfg, z: Tensor, rho: Tensor, ten: Tensor, cell: Tensor, *, later: bool, need_vir: bool = False, need_drho: bool = False) -> int:
    flags = 0
    if cfg.floor and z.numel() > 0:
        flags |= FLAG_FLOOR
    if rho.numel() > 0:
        flags |= FLAG_RHO
    if later and ten.numel() > 0:
        flags |= FLAG_TEN
    if cell.numel() > 0:
        flags |= FLAG_CELL
        if cell.shape[0] == 1:
            flags |= FLAG_G1
    if need_vir:
        flags |= FLAG_VIR
    if need_drho:
        flags |= FLAG_DRHO
    return flags


def metal_csr_forward(
    plan: CSRPlan,
    layout: StageLayout,
    cfg: EdgeCfg,
    pos: Tensor,
    poly: Tensor,
    gates: Tensor,
    vec: Tensor,
    ten: Tensor,
    qn: Tensor,
    kn: Tensor,
    rho: Tensor,
    shift_in: Tensor,
    cell: Tensor,
    graph: Tensor,
    z: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    dev = pos.device
    n = plan.n_atoms
    if n == 0 or plan.n_edges == 0:
        return (
            torch.zeros((n, layout.out_dim), dtype=torch.float32, device=dev),
            torch.zeros((n, 4), dtype=torch.float32, device=dev),
            torch.zeros((n,), dtype=torch.float32, device=dev),
        )
    lib = get_metal_csr_lib()
    df, di = _dummy_f32(dev), _dummy_i64(dev)
    flags = _pack_flags(cfg, z, rho, ten, cell, later=layout.later)
    rmin = rmin_table_f32(dev) if (flags & FLAG_FLOOR) else df
    out_num = torch.empty((n, layout.out_dim), dtype=torch.float32, device=dev)
    out_zc = torch.empty((n, 4), dtype=torch.float32, device=dev)
    out_dsum = torch.empty((n,), dtype=torch.float32, device=dev)
    fn = lib.csr_fwd_later if layout.later else lib.csr_fwd_first
    n_tot = ((n + 3) // 4) * 128
    fn(
        pos, plan.recv_ptr, plan.send_idx, shift_in,
        cell if cell.numel() else df,
        graph if graph.numel() else di,
        z if z.numel() else di,
        rmin,
        rho if rho.numel() else df,
        poly, gates, qn, kn,
        vec if vec.numel() else df,
        ten if ten.numel() else df,
        out_num, out_zc, out_dsum,
        int(n), int(poly.shape[0]), int(flags), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2),
        threads=(n_tot, 1, 1), group_size=(128, 1, 1),
    )
    return out_num, out_zc, out_dsum


def metal_csr_backward(
    plan: CSRPlan,
    layout: StageLayout,
    cfg: EdgeCfg,
    pos: Tensor,
    poly: Tensor,
    gates: Tensor,
    vec: Tensor,
    ten: Tensor,
    qn: Tensor,
    kn: Tensor,
    rho: Tensor,
    shift_in: Tensor,
    cell: Tensor,
    graph: Tensor,
    z: Tensor,
    g_num: Tensor,
    g_zc: Tensor,
    g_dsum: Tensor,
    *,
    need_vir: bool,
    need_drho: bool,
    need_dshift: bool,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    dev = pos.device
    n, n_e = plan.n_atoms, plan.n_edges
    has_ten = layout.later and ten.numel() > 0
    if n == 0 or n_e == 0:
        return (
            torch.zeros_like(pos),
            torch.zeros_like(gates),
            torch.zeros_like(vec) if layout.later else None,
            torch.zeros_like(ten) if has_ten else None,
            torch.zeros_like(qn),
            torch.zeros_like(kn),
            torch.zeros_like(rho) if need_drho else None,
            torch.zeros_like(cell) if need_vir else None,
            torch.zeros_like(shift_in) if need_dshift else None,
        )
    lib = get_metal_csr_lib()
    df, di = _dummy_f32(dev), _dummy_i64(dev)
    flags = _pack_flags(cfg, z, rho, ten, cell, later=layout.later, need_vir=need_vir, need_drho=need_drho)
    rmin = rmin_table_f32(dev) if (flags & FLAG_FLOOR) else df

    d_pos = torch.empty((n, 3), dtype=torch.float32, device=dev)
    d_qn = torch.empty_like(qn)
    edge_grad8 = torch.empty((n_e, 8), dtype=torch.float32, device=dev)
    vir_atom = torch.empty((n, 9), dtype=torch.float32, device=dev) if need_vir else df
    d_rho = torch.empty((n,), dtype=torch.float32, device=dev) if need_drho else df

    n_tot = ((n + 3) // 4) * 128
    fn_recv = lib.csr_bwd_recv_later if layout.later else lib.csr_bwd_recv_first
    fn_recv(
        pos, plan.recv_ptr, plan.send_idx, shift_in,
        cell if cell.numel() else df,
        graph if graph.numel() else di,
        z if z.numel() else di,
        rmin,
        rho if rho.numel() else df,
        poly, gates, qn, kn,
        vec if vec.numel() else df,
        ten if ten.numel() else df,
        g_num, g_zc, g_dsum,
        d_pos, d_qn, edge_grad8, vir_atom, d_rho,
        int(n), int(poly.shape[0]), int(flags), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2),
        threads=(n_tot, 1, 1), group_size=(128, 1, 1),
    )

    d_gates = torch.empty((n, layout.gate_dim), dtype=torch.float32, device=dev)
    d_kn = torch.empty_like(kn)
    d_vec = torch.empty((n, 32, 3), dtype=torch.float32, device=dev) if layout.later else df
    d_ten = torch.empty((n, 16, 5), dtype=torch.float32, device=dev) if has_ten else df
    fn_send = lib.csr_bwd_send_later if layout.later else lib.csr_bwd_send_first
    fn_send(
        pos, plan.send_ptr, plan.send_perm, plan.recv_idx, shift_in,
        cell if cell.numel() else df,
        graph if graph.numel() else di,
        z if z.numel() else di,
        rmin,
        rho if rho.numel() else df,
        poly, qn, kn, g_num, edge_grad8,
        d_pos, d_gates, d_kn, d_vec, d_ten, d_rho,
        int(n), int(poly.shape[0]), int(flags), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2),
        threads=(n_tot, 1, 1), group_size=(128, 1, 1),
    )

    d_cell = None
    if need_vir:
        if cell.shape[0] == 1:
            d_cell = vir_atom.sum(dim=0).view(1, 3, 3)
        else:
            d_cell = cell.new_zeros(cell.shape[0], 9).index_add(0, graph, vir_atom).view_as(cell)
    d_shift = edge_grad8[:, :3].contiguous() if need_dshift else None
    return (
        d_pos,
        d_gates,
        d_vec if layout.later else None,
        d_ten if has_ten else None,
        d_qn,
        d_kn,
        d_rho if need_drho else None,
        d_cell,
        d_shift,
    )
