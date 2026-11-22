"""Native strict-IEEE FP32 CUDA Dual-CSR custom_op (`lam4::cuda_csr_agg_final` + fused `finalise` + 1-pass/2-pass bwd)."""
from __future__ import annotations

import os
from typing import Any
import weakref

import torch
from torch import Tensor

from lam4.core.consts import capturing, storable
from lam4.core.edge_terms import EdgeCfg
from lam4.core.layout import StageLayout
from lam4.kernels.csr_plan import MAX_DOFF_R2, CSRPlan, dummy_f32, dummy_i64, rmin_table_f32
from lam4.kernels.cuda_csr_bwd_src import CUDA_CSR_BWD_SRC
from lam4.kernels.cuda_csr_fwd_src import CUDA_CSR_FWD_SRC
from lam4.kernels.cuda_csr_launch_src import CUDA_CSR_LAUNCH_SRC
from lam4.kernels.cuda_ext import cuda_ext_ready, load_cuda_ext

_CPP_DECLS = r"""
std::vector<torch::Tensor> cuda_csr_fwd_launch(
    torch::Tensor pos, torch::Tensor recv_ptr, torch::Tensor send_idx,
    torch::Tensor shift_in, torch::Tensor cell, torch::Tensor graph,
    torch::Tensor z, torch::Tensor rmin_tab, torch::Tensor rho,
    torch::Tensor poly4, torch::Tensor gates, torch::Tensor qn,
    torch::Tensor kn, torch::Tensor vec, torch::Tensor ten,
    int64_t n_atoms, int64_t grid, int64_t flags, bool later,
    double r_cut, double r_on, double max_doff_r2);

std::vector<torch::Tensor> cuda_csr_bwd_launch(
    torch::Tensor pos, torch::Tensor recv_ptr, torch::Tensor send_idx,
    torch::Tensor send_ptr, torch::Tensor send_perm, torch::Tensor recv_idx,
    torch::Tensor shift_in, torch::Tensor cell, torch::Tensor graph,
    torch::Tensor z, torch::Tensor rmin_tab, torch::Tensor rho,
    torch::Tensor poly4, torch::Tensor gates, torch::Tensor qn,
    torch::Tensor kn, torch::Tensor vec, torch::Tensor ten,
    torch::Tensor agg, torch::Tensor zc, torch::Tensor dsum,
    torch::Tensor g_agg, torch::Tensor g_dsum_ext,
    int64_t n_atoms, int64_t n_edges, int64_t grid, int64_t flags,
    bool later, bool one_pass, double r_cut, double r_on, double max_doff_r2);
"""

_F_FLOOR = 1
_F_RHO = 2
_F_TEN = 4
_F_CELL = 8
_F_G1 = 16
_F_VIR = 32
_F_DRHO = 64
_F_SEND_FEAT = 128
_F_DQN = 256
_F_GDSUM = 512

_POLY4_CACHE: dict[int, tuple[int, tuple[int, ...], torch.device, Tensor]] = {}


def _ext() -> Any | None:
    return load_cuda_ext(
        "lam4_cuda_csr",
        _CPP_DECLS,
        CUDA_CSR_FWD_SRC + "\n" + CUDA_CSR_BWD_SRC + "\n" + CUDA_CSR_LAUNCH_SRC,
        ["cuda_csr_fwd_launch", "cuda_csr_bwd_launch"],
    )


def _require_ext() -> Any:
    ext = _ext()
    if ext is None:
        raise RuntimeError("lam4_cuda_csr CUDA extension unavailable")
    return ext


def has_cuda_csr(dev: torch.device | str | None = None) -> bool:
    return cuda_ext_ready("LAM4_CUDA_CSR", "lam4_cuda_csr", dev, _ext)


def poly_to_poly4(poly: Tensor) -> Tensor:
    """Convert `[G, 4, TD]` Horner table to 16-byte-aligned `[G, TD, 4]` `float4` table (cached with weakref eviction)."""
    if poly.ndim == 3 and poly.shape[2] == 4:
        return poly.contiguous()
    if storable(poly) and not capturing():
        pid = id(poly)
        ver = int(poly._version)
        shp = tuple(int(s) for s in poly.shape)
        entry = _POLY4_CACHE.get(pid)
        if entry is not None and entry[0] == ver and entry[1] == shp and entry[2] == poly.device:
            return entry[3]
        p4 = poly.permute(0, 2, 1).contiguous()
        if len(_POLY4_CACHE) >= 64:
            _POLY4_CACHE.clear()
        _POLY4_CACHE[pid] = (ver, shp, poly.device, p4)
        weakref.finalize(poly, _POLY4_CACHE.pop, pid, None)
        return p4
    return poly.permute(0, 2, 1).contiguous()


def _use_one_pass(n_atoms: int, need_send_feat: bool) -> bool:
    mode = os.environ.get("LAM4_CUDA_CSR_BWD_MODE", "1pass").strip().lower()
    if mode in ("2pass", "two_pass", "atomic_free"):
        return False
    if mode in ("1pass", "one_pass"):
        return True
    # "auto": Stage-0 (no sender feature grads) ALWAYS wins with 1-pass; small N<=128 uses 2-pass to avoid contention.
    return (not need_send_feat) or n_atoms > 128


@torch.library.custom_op("lam4::cuda_csr_agg_final", mutates_args=())
def _cuda_csr_agg_final_op(
    rp: Tensor, si: Tensor, ri: Tensor, sp: Tensor, spm: Tensor, sh_c: Tensor, gr: Tensor, z: Tensor, rmin: Tensor,
    pos: Tensor, poly4: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor,
    later: bool, r_cut: float, r_on: float, floor: bool, has_rho: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    dev, n, n_e = pos.device, int(pos.shape[0]), int(si.shape[0])
    od = 408 if later else 360
    if n == 0 or n_e == 0:
        return (
            torch.zeros((n, od), dtype=torch.float32, device=dev),
            torch.zeros((n, 4), dtype=torch.float32, device=dev),
            torch.zeros((n,), dtype=torch.float32, device=dev),
        )
    has_floor = bool(floor and z.numel() > 0)
    has_r = bool(has_rho and rho.numel() > 0)
    has_ten = bool(later and ten.numel() > 0)
    has_cell = bool(cell.ndim == 3 and cell.numel() > 0)
    is_g1 = bool(has_cell and cell.shape[0] == 1)
    flags = (
        (_F_FLOOR if has_floor else 0)
        | (_F_RHO if has_r else 0)
        | (_F_TEN if has_ten else 0)
        | (_F_CELL if has_cell else 0)
        | (_F_G1 if is_g1 else 0)
    )
    df, di = dummy_f32(dev), dummy_i64(dev)
    res = _require_ext().cuda_csr_fwd_launch(
        pos, rp, si, sh_c, cell if has_cell else df, gr if gr.numel() else di,
        z if has_floor else di, rmin if has_floor else df, rho if has_r else df,
        poly4, gates, qn, kn, vec if vec.numel() else df, ten if has_ten else df,
        n, int(poly4.shape[0]), flags, bool(later), float(r_cut), float(r_on), float(MAX_DOFF_R2),
    )
    return res[0], res[1], res[2]


@_cuda_csr_agg_final_op.register_fake
def _(rp, si, ri, sp, spm, sh_c, gr, z, rmin, pos, poly4, gates, vec, ten, qn, kn, rho, cell, later, r_cut, r_on, floor, has_rho):
    return pos.new_empty((pos.shape[0], 408 if later else 360)), pos.new_empty((pos.shape[0], 4)), pos.new_empty((pos.shape[0],))


@torch.library.custom_op("lam4::cuda_csr_agg_final_bwd", mutates_args=())
def _cuda_csr_agg_final_bwd_op(
    rp: Tensor, si: Tensor, ri: Tensor, sp: Tensor, spm: Tensor, sh_c: Tensor, gr: Tensor, z: Tensor, rmin: Tensor,
    pos: Tensor, poly4: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, cell: Tensor,
    agg: Tensor, zc: Tensor, dsum: Tensor, g_agg: Tensor, g_dsum: Tensor,
    later: bool, r_cut: float, r_on: float, floor: bool, has_rho: bool,
    need_vir: bool, need_drho: bool, need_send_feat: bool, need_dqn: bool, has_gdsum: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dev, n, n_e = pos.device, int(pos.shape[0]), int(si.shape[0])
    has_floor = bool(floor and z.numel() > 0)
    has_r = bool(has_rho and rho.numel() > 0)
    has_ten = bool(later and ten.numel() > 0)
    has_cell = bool(cell.ndim == 3 and cell.numel() > 0)
    is_g1 = bool(has_cell and cell.shape[0] == 1)
    eff_vir = bool(need_vir and has_cell)
    eff_drho = bool(need_drho and has_r)
    flags = (
        (_F_FLOOR if has_floor else 0)
        | (_F_RHO if has_r else 0)
        | (_F_TEN if has_ten else 0)
        | (_F_CELL if has_cell else 0)
        | (_F_G1 if is_g1 else 0)
        | (_F_VIR if eff_vir else 0)
        | (_F_DRHO if eff_drho else 0)
        | (_F_SEND_FEAT if need_send_feat else 0)
        | (_F_DQN if need_dqn else 0)
        | (_F_GDSUM if has_gdsum else 0)
    )
    df, di = dummy_f32(dev), dummy_i64(dev)
    one_pass = _use_one_pass(n, bool(need_send_feat))
    res = _require_ext().cuda_csr_bwd_launch(
        pos, rp, si, sp, spm, ri, sh_c, cell if has_cell else df, gr if gr.numel() else di,
        z if has_floor else di, rmin if has_floor else df, rho if has_r else df,
        poly4, gates, qn, kn, vec if vec.numel() else df, ten if has_ten else df,
        agg, zc, dsum, g_agg.contiguous(), g_dsum.contiguous() if has_gdsum else df,
        n, n_e, int(poly4.shape[0]), flags, bool(later), bool(one_pass),
        float(r_cut), float(r_on), float(MAX_DOFF_R2),
    )
    return tuple(res)  # type: ignore[return-value]


@_cuda_csr_agg_final_bwd_op.register_fake
def _(rp, si, ri, sp, spm, sh_c, gr, z, rmin, pos, poly4, gates, vec, ten, qn, kn, rho, cell,
      agg, zc, dsum, g_agg, g_dsum, later, r_cut, r_on, floor, has_rho,
      need_vir, need_drho, need_send_feat, need_dqn, has_gdsum):
    empty = pos.new_empty(0)
    return (
        torch.empty_like(pos),
        torch.empty_like(gates) if need_send_feat else empty,
        torch.empty_like(vec) if (later and need_send_feat and vec.ndim == 3) else empty,
        torch.empty_like(ten) if (later and need_send_feat and ten.ndim == 3) else empty,
        torch.empty_like(qn) if need_dqn else empty,
        torch.empty_like(kn) if need_send_feat else empty,
        torch.empty_like(rho) if (need_drho and has_rho) else empty,
        torch.empty_like(cell) if (need_vir and cell.ndim == 3) else empty,
    )


def _setup_ctx(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, Tensor, Tensor]) -> None:
    rp, si, ri, sp, spm, sh_c, gr, z, rmin, pos, poly4, gates, vec, ten, qn, kn, rho, cell, later, r_cut, r_on, floor, has_rho = inputs
    agg, zc, dsum = output
    ctx.later, ctx.r_cut, ctx.r_on, ctx.floor, ctx.has_rho = bool(later), float(r_cut), float(r_on), bool(floor), bool(has_rho)
    ctx.has_vec, ctx.has_ten, ctx.has_cell = bool(later and vec.ndim == 3), bool(later and ten.ndim == 3), bool(cell.ndim == 3)
    ctx.save_for_backward(rp, si, ri, sp, spm, sh_c, gr, z, rmin, pos, poly4, gates, vec, ten, qn, kn, rho, cell, agg, zc, dsum)


def _autograd_bwd(ctx: Any, g_agg: Tensor | None, g_zc: Tensor | None, g_dsum: Tensor | None):
    rp, si, ri, sp, spm, sh_c, gr, z, rmin, pos, poly4, gates, vec, ten, qn, kn, rho, cell, agg, zc, dsum = ctx.saved_tensors
    nds = ctx.needs_input_grad
    need_pos = bool(nds[9])
    need_gates = bool(nds[11])
    need_vec = bool(nds[12] and ctx.has_vec)
    need_ten = bool(nds[13] and ctx.has_ten)
    need_qn = bool(nds[14])
    need_kn = bool(nds[15])
    need_drho = bool(nds[16] and ctx.has_rho)
    need_vir = bool(nds[17] and ctx.has_cell)
    need_send_feat = bool(need_gates or need_vec or need_ten or need_kn)
    ga = torch.zeros_like(agg) if g_agg is None else g_agg.contiguous()
    has_gd = g_dsum is not None
    gd = g_dsum.contiguous() if has_gd else pos.new_empty(0)
    d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell = _cuda_csr_agg_final_bwd_op(
        rp, si, ri, sp, spm, sh_c, gr, z, rmin, pos, poly4, gates, vec, ten, qn, kn, rho, cell,
        agg, zc, dsum, ga, gd, ctx.later, ctx.r_cut, ctx.r_on, ctx.floor, ctx.has_rho,
        need_vir, need_drho, need_send_feat, need_qn, has_gd,
    )
    return (
        None, None, None, None, None, None, None, None, None,
        d_pos if need_pos else None, None,
        d_gates if need_gates else None,
        d_vec if need_vec else None,
        d_ten if need_ten else None,
        d_qn if need_qn else None,
        d_kn if need_kn else None,
        d_rho if need_drho else None,
        d_cell if need_vir else None,
        None, None, None, None, None,
    )


torch.library.register_autograd("lam4::cuda_csr_agg_final", _autograd_bwd, setup_context=_setup_ctx)


def cuda_csr_aggregate_final(
    plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg,
    sh_const: Tensor, graph_c: Tensor, z_c: Tensor, pos_c: Tensor, poly_c: Tensor,
    gates_c: Tensor, vec_c: Tensor, ten_c: Tensor, qn_c: Tensor, kn_c: Tensor,
    rho_c: Tensor, cell_c: Tensor, *, has_rho: bool, return_dsum: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    poly4 = poly_to_poly4(poly_c)
    rmin_c = rmin_table_f32(pos_c.device) if cfg.floor else pos_c.new_zeros(0)
    agg, _zc, dsum = _cuda_csr_agg_final_op(
        plan.recv_ptr, plan.send_idx, plan.recv_idx, plan.send_ptr, plan.send_perm,
        sh_const, graph_c, z_c, rmin_c, pos_c, poly4, gates_c, vec_c, ten_c, qn_c, kn_c, rho_c, cell_c,
        bool(layout.later), float(cfg.r_cut), float(cfg.r_on), bool(cfg.floor), bool(has_rho),
    )
    return (agg, dsum) if return_dsum else agg
