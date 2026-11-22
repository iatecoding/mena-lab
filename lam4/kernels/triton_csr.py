"""CUDA / ROCm Triton Dual-CSR 4-warp forward & atomic-free 2-pass backward kernels."""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

from lam4.core.consts import cuda_cap_ok
from lam4.core.edge_terms import EdgeCfg
from lam4.core.layout import StageLayout
from lam4.kernels.csr_plan import MAX_DOFF_R2, CSRPlan, dummy_f32, dummy_i64, rmin_table_f32, select_triton_block_warps, stream_vjp_csr_backward, warm_csr_nlh_caches
from lam4.kernels.triton_nlh import _av3, _csr_geom, _eval_y2_y3, _ld3, _ld5, _ld7, _st3, _st5, _st7, _tanh_half, _vjp_y2_y3

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - optional on macOS / CPU environments
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False

_jit = triton.jit if HAS_TRITON else (lambda fn: fn)


def _cfg_block_warps(n_atoms: int | None = None) -> tuple[int, int]:
    return select_triton_block_warps(n_atoms)


def has_triton_csr() -> bool:
    return bool(HAS_TRITON and cuda_cap_ok())


def has_triton_csr_backward(device: torch.device | str | None = None) -> bool:
    if os.environ.get("LAM4_TRITON_BWD", "1") == "0" or (device is not None and torch.device(device).type != "cuda"):
        return False
    return has_triton_csr()


@_jit
def _horner_v(poly_p, base, td: tl.constexpr, col, tsp, mask):
    a0 = tl.load(poly_p + base + col, mask=mask, other=0.0, eviction_policy="evict_last")
    a1 = tl.load(poly_p + base + td + col, mask=mask, other=0.0, eviction_policy="evict_last")
    a2 = tl.load(poly_p + base + 2 * td + col, mask=mask, other=0.0, eviction_policy="evict_last")
    a3 = tl.load(poly_p + base + 3 * td + col, mask=mask, other=0.0, eviction_policy="evict_last")
    return ((a3 * tsp + a2) * tsp + a1) * tsp + a0


@_jit
def _horner_vd(poly_p, base, td: tl.constexpr, col, tsp, inv_h, mask):
    a0 = tl.load(poly_p + base + col, mask=mask, other=0.0, eviction_policy="evict_last")
    a1 = tl.load(poly_p + base + td + col, mask=mask, other=0.0, eviction_policy="evict_last")
    a2 = tl.load(poly_p + base + 2 * td + col, mask=mask, other=0.0, eviction_policy="evict_last")
    a3 = tl.load(poly_p + base + 3 * td + col, mask=mask, other=0.0, eviction_policy="evict_last")
    return ((a3 * tsp + a2) * tsp + a1) * tsp + a0, ((3.0 * a3 * tsp + 2.0 * a2) * tsp + a1) * inv_h


@_jit
def _bcast_heads(ws, m16, m8, B: tl.constexpr):
    ws41, ws141 = tl.reshape(ws, [B, 4, 1]), tl.reshape(ws, [B, 1, 4, 1])
    w128, w32 = tl.reshape(tl.broadcast_to(ws41, [B, 4, 32]), [B, 128]), tl.reshape(tl.broadcast_to(ws41, [B, 4, 8]), [B, 32])
    return w128, w32, tl.where(m16[None, :], tl.reshape(tl.broadcast_to(ws141, [B, 2, 4, 4]), [B, 32]), 0.0), tl.where(m8[None, :], tl.reshape(tl.broadcast_to(ws141, [B, 4, 4, 2]), [B, 32]), 0.0)


@_jit
def _qk_heads(q, k, chi, poly_p, pbase, TD: tl.constexpr, h4, tsp, act, m16, m8, B: tl.constexpr):
    dots = tl.sum(tl.reshape(q * k, [B, 4, 16]), axis=2)
    ws = tl.where(act[:, None], chi[:, None] * tl.exp(dots + 2.0 * _tanh_half(_horner_v(poly_p, pbase[:, None], TD, h4[None, :], tsp[:, None], act[:, None]))), 0.0)
    ws128, wv, wt, wo = _bcast_heads(ws, m16, m8, B)
    return ws, ws128, wv, wt, wo


@_jit
def _triton_csr_fwd(pos_p, rptr_p, sidx_p, sh_p, cell_p, grp_p, z_p, rmin_p, rho_p, poly_p, gates_p, qn_p, kn_p, vec_p, ten_p, num_p, zc_p, dsum_p, n_atoms, grid, r_cut, r_on, max_doff_r2, LATER: tl.constexpr, HAS_FLOOR: tl.constexpr, HAS_RHO: tl.constexpr, HAS_TEN: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    am, lane, c64, c128, h4 = i < n_atoms, tl.arange(0, 32), tl.arange(0, 64), tl.arange(0, 128), tl.arange(0, 4)
    m16, m8 = lane < 16, lane < 8
    TD: tl.constexpr = 284 if LATER else 188
    OD: tl.constexpr = 408 if LATER else 360
    FO: tl.constexpr = 276 if LATER else 180
    OT: tl.constexpr = 256 if LATER else 224
    OO: tl.constexpr = 352 if LATER else 304
    g_id = tl.zeros([B], tl.int32) if (not HAS_CELL or IS_G1) else tl.load(grp_p + i, mask=am, other=0).to(tl.int32)
    zi_s = (tl.minimum(tl.maximum(tl.load(z_p + i, mask=am, other=0).to(tl.int32), 0), 127) << 7) if HAS_FLOOR else tl.zeros([B], tl.int32)
    rho_i = tl.load(rho_p + i, mask=am, other=0.0) if HAS_RHO else tl.zeros([B], tl.float32)
    px, py, pz = _ld3(pos_p, 3 * i, am)
    q = tl.load(qn_p + 64 * i[:, None] + c64[None, :], mask=am[:, None], other=0.0)
    as_all, az, adsum = tl.zeros([B, 128], tl.float32), tl.zeros([B, 4], tl.float32), tl.zeros([B], tl.float32)
    avx, avy, avz, alon = tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32)
    at0, at1, at2, at3, at4 = tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32)
    alon2, ao0, ao1, ao2 = tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32)
    ao3, ao4, ao5, ao6 = tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32)
    inv_w, inv_h, rcut2 = 1.0 / (r_cut - r_on), grid / r_cut, r_cut * r_cut
    e_beg, e_end = tl.load(rptr_p + i, mask=am, other=0), tl.load(rptr_p + i + 1, mask=am, other=0)
    for step in range(tl.max(tl.maximum(e_end - e_beg, 0), axis=0)):
        e = e_beg + step
        ve = am & (e < e_end)
        j = tl.load(sidx_p + e, mask=ve, other=0)
        pjx, pjy, pjz = _ld3(pos_p, 3 * j, ve)
        ux, uy, uz, _, _, _, _, _, _, _, act, chi, _, _, _, _, tsp, pbase = _csr_geom(px, py, pz, pjx, pjy, pjz, sh_p, cell_p, e, g_id, ve, z_p, rmin_p, j, zi_s, rho_p, j, rho_i, grid, r_cut, r_on, inv_w, inv_h, rcut2, max_doff_r2, TD, HAS_CELL, HAS_FLOOR, HAS_RHO, False, False, False, B)
        k = tl.load(kn_p + 64 * j[:, None] + c64[None, :], mask=act[:, None], other=0.0)
        ws, ws128, wv, wt, wo = _qk_heads(q, k, chi, poly_p, pbase, TD, h4, tsp, act, m16, m8, B)
        az, adsum = az + ws, adsum + chi
        gj, act32, act16, act8 = 184 * j[:, None], act[:, None], act[:, None] & m16[None, :], act[:, None] & m8[None, :]
        as_all += ws128 * _horner_v(poly_p, pbase[:, None], TD, 4 + c128[None, :], tsp[:, None], act32) * tl.load(gates_p + gj + c128[None, :], mask=act32, other=0.0)
        w_dir = _horner_v(poly_p, pbase[:, None], TD, 132 + lane[None, :], tsp[:, None], act32) * tl.load(gates_p + gj + 128 + lane[None, :], mask=act32, other=0.0)
        if LATER:
            bx, by, bz = _ld3(vec_p, 96 * j[:, None] + 3 * lane[None, :], act32)
            st = wv * _horner_v(poly_p, pbase[:, None], TD, 180 + lane[None, :], tsp[:, None], act32)
            w_lon = _horner_v(poly_p, pbase[:, None], TD, 212 + lane[None, :], tsp[:, None], act32) * (bx * ux[:, None] + by * uy[:, None] + bz * uz[:, None])
            sc = wv * (w_dir + w_lon)
            avx, avy, avz, alon = avx + sc * ux[:, None] + st * bx, avy + sc * uy[:, None] + st * by, avz + sc * uz[:, None] + st * bz, alon + wv * w_lon
        else:
            sc = wv * w_dir
            avx, avy, avz = avx + sc * ux[:, None], avy + sc * uy[:, None], avz + sc * uz[:, None]
        y20, y21, y22, y23, y24, y30, y31, y32, y33, y34, y35, y36 = _eval_y2_y3(ux, uy, uz)
        w_ten = _horner_v(poly_p, pbase[:, None], TD, 164 + lane[None, :], tsp[:, None], act16) * tl.load(gates_p + gj + 160 + lane[None, :], mask=act16, other=0.0)
        if HAS_TEN:
            t0, t1, t2_, t3, t4 = _ld5(ten_p, 80 * j[:, None] + 5 * lane[None, :], act16)
            st2 = wt * _horner_v(poly_p, pbase[:, None], TD, 244 + lane[None, :], tsp[:, None], act16)
            w_lon2 = _horner_v(poly_p, pbase[:, None], TD, 260 + lane[None, :], tsp[:, None], act16) * (t0 * y20[:, None] + t1 * y21[:, None] + t2_ * y22[:, None] + t3 * y23[:, None] + t4 * y24[:, None])
            sct = wt * (w_ten + w_lon2)
            at0, at1, at2 = at0 + sct * y20[:, None] + st2 * t0, at1 + sct * y21[:, None] + st2 * t1, at2 + sct * y22[:, None] + st2 * t2_
            at3, at4, alon2 = at3 + sct * y23[:, None] + st2 * t3, at4 + sct * y24[:, None] + st2 * t4, alon2 + wt * w_lon2
        else:
            sct = wt * w_ten
            at0, at1, at2, at3, at4 = at0 + sct * y20[:, None], at1 + sct * y21[:, None], at2 + sct * y22[:, None], at3 + sct * y23[:, None], at4 + sct * y24[:, None]
        sco = wo * _horner_v(poly_p, pbase[:, None], TD, FO + lane[None, :], tsp[:, None], act8) * tl.load(gates_p + gj + 176 + lane[None, :], mask=act8, other=0.0)
        ao0, ao1, ao2, ao3 = ao0 + sco * y30[:, None], ao1 + sco * y31[:, None], ao2 + sco * y32[:, None], ao3 + sco * y33[:, None]
        ao4, ao5, ao6 = ao4 + sco * y34[:, None], ao5 + sco * y35[:, None], ao6 + sco * y36[:, None]
    ob, m32, m16b, m8b = OD * i[:, None], am[:, None], am[:, None] & m16[None, :], am[:, None] & m8[None, :]
    tl.store(num_p + ob + c128[None, :], as_all, mask=m32)
    _st3(num_p, ob + 128 + 3 * lane[None, :], avx, avy, avz, m32)
    if LATER:
        tl.store(num_p + ob + 224 + lane[None, :], alon, mask=m32)
        tl.store(num_p + ob + 336 + lane[None, :], alon2, mask=m16b)
    _st5(num_p, ob + OT + 5 * lane[None, :], at0, at1, at2, at3, at4, m16b)
    _st7(num_p, ob + OO + 7 * lane[None, :], ao0, ao1, ao2, ao3, ao4, ao5, ao6, m8b)
    tl.store(zc_p + 4 * i[:, None] + h4[None, :], az, mask=m32)
    tl.store(dsum_p + i, adsum, mask=am)


@_jit
def _triton_csr_bwd_recv(pos_p, rptr_p, sidx_p, sh_p, cell_p, grp_p, z_p, rmin_p, rho_p, poly_p, gates_p, qn_p, kn_p, vec_p, ten_p, g_num_p, g_zc_p, g_dsum_p, d_pos_p, d_qn_p, eg8_p, d_cell_p, d_rho_p, n_atoms, grid, r_cut, r_on, max_doff_r2, LATER: tl.constexpr, HAS_FLOOR: tl.constexpr, HAS_RHO: tl.constexpr, HAS_TEN: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr, NEED_VIR: tl.constexpr, NEED_DRHO: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    am, lane, c64, c128, h4 = i < n_atoms, tl.arange(0, 32), tl.arange(0, 64), tl.arange(0, 128), tl.arange(0, 4)
    m16, m8 = lane < 16, lane < 8
    TD: tl.constexpr = 284 if LATER else 188
    OD: tl.constexpr = 408 if LATER else 360
    FO: tl.constexpr = 276 if LATER else 180
    OT: tl.constexpr = 256 if LATER else 224
    OO: tl.constexpr = 352 if LATER else 304
    g_id = tl.zeros([B], tl.int32) if (not HAS_CELL or IS_G1) else tl.load(grp_p + i, mask=am, other=0).to(tl.int32)
    zi_s = (tl.minimum(tl.maximum(tl.load(z_p + i, mask=am, other=0).to(tl.int32), 0), 127) << 7) if HAS_FLOOR else tl.zeros([B], tl.int32)
    rho_i = tl.load(rho_p + i, mask=am, other=0.0) if HAS_RHO else tl.zeros([B], tl.float32)
    px, py, pz = _ld3(pos_p, 3 * i, am)
    q = tl.load(qn_p + 64 * i[:, None] + c64[None, :], mask=am[:, None], other=0.0)
    gi, m32 = OD * i[:, None], am[:, None]
    gs = tl.load(g_num_p + gi + c128[None, :], mask=m32, other=0.0)
    gvx, gvy, gvz = _ld3(g_num_p, gi + 128 + 3 * lane[None, :], m32)
    glon = tl.load(g_num_p + gi + 224 + lane[None, :], mask=m32, other=0.0) if LATER else tl.zeros([B, 32], tl.float32)
    gzc, gdsum_i = tl.load(g_zc_p + 4 * i[:, None] + h4[None, :], mask=m32, other=0.0), tl.load(g_dsum_p + i, mask=am, other=0.0)
    dpos_x, dpos_y, dpos_z, drho_i, dq = tl.zeros([B], tl.float32), tl.zeros([B], tl.float32), tl.zeros([B], tl.float32), tl.zeros([B], tl.float32), tl.zeros([B, 64], tl.float32)
    v00, v01, v02 = tl.zeros([B], tl.float32), tl.zeros([B], tl.float32), tl.zeros([B], tl.float32)
    v10, v11, v12 = tl.zeros([B], tl.float32), tl.zeros([B], tl.float32), tl.zeros([B], tl.float32)
    v20, v21, v22 = tl.zeros([B], tl.float32), tl.zeros([B], tl.float32), tl.zeros([B], tl.float32)
    inv_w, inv_h, rcut2 = 1.0 / (r_cut - r_on), grid / r_cut, r_cut * r_cut
    e_beg, e_end = tl.load(rptr_p + i, mask=am, other=0), tl.load(rptr_p + i + 1, mask=am, other=0)
    for step in range(tl.max(tl.maximum(e_end - e_beg, 0), axis=0)):
        e = e_beg + step
        ve = am & (e < e_end)
        j = tl.load(sidx_p + e, mask=ve, other=0)
        pjx, pjy, pjz = _ld3(pos_p, 3 * j, ve)
        ux, uy, uz, inv_r, drfl_x, drfl_y, drfl_z, esx, esy, esz, act, chi, dchi_drfl, dx_drfl, dx_drho, inv_he, tsp, pbase = _csr_geom(px, py, pz, pjx, pjy, pjz, sh_p, cell_p, e, g_id, ve, z_p, rmin_p, j, zi_s, rho_p, j, rho_i, grid, r_cut, r_on, inv_w, inv_h, rcut2, max_doff_r2, TD, HAS_CELL, HAS_FLOOR, HAS_RHO, True, NEED_DRHO, False, B)
        k = tl.load(kn_p + 64 * j[:, None] + c64[None, :], mask=act[:, None], other=0.0)
        dots = tl.sum(tl.reshape(q * k, [B, 4, 16]), axis=2)
        b_raw, db_dx = _horner_vd(poly_p, pbase[:, None], TD, h4[None, :], tsp[:, None], inv_he[:, None], act[:, None])
        th_b = _tanh_half(b_raw)
        ex_l = tl.where(act[:, None], tl.exp(dots + 2.0 * th_b), 0.0)
        ws = chi[:, None] * ex_l
        gj, act32, act16, act8 = 184 * j[:, None], act[:, None], act[:, None] & m16[None, :], act[:, None] & m8[None, :]
        fs, dfs = _horner_vd(poly_p, pbase[:, None], TD, 4 + c128[None, :], tsp[:, None], inv_he[:, None], act32)
        gg = gs * tl.load(gates_p + gj + c128[None, :], mask=act32, other=0.0)
        dw_4 = tl.sum(tl.reshape(gg * fs, [B, 4, 32]), axis=2)
        dx_acc = tl.sum(ws * tl.sum(tl.reshape(gg * dfs, [B, 4, 32]), axis=2), axis=1)
        _, wv, wt, wo = _bcast_heads(ws, m16, m8, B)
        f_dir, df_dir = _horner_vd(poly_p, pbase[:, None], TD, 132 + lane[None, :], tsp[:, None], inv_he[:, None], act32)
        gv_j = tl.load(gates_p + gj + 128 + lane[None, :], mask=act32, other=0.0)
        w_dir, dot_gv_u = f_dir * gv_j, gvx * ux[:, None] + gvy * uy[:, None] + gvz * uz[:, None]
        if LATER:
            bx, by, bz = _ld3(vec_p, 96 * j[:, None] + 3 * lane[None, :], act32)
            f_tr, df_tr = _horner_vd(poly_p, pbase[:, None], TD, 180 + lane[None, :], tsp[:, None], inv_he[:, None], act32)
            f_lon, df_lon = _horner_vd(poly_p, pbase[:, None], TD, 212 + lane[None, :], tsp[:, None], inv_he[:, None], act32)
            dot_bu, dot_gv_b = bx * ux[:, None] + by * uy[:, None] + bz * uz[:, None], gvx * bx + gvy * by + gvz * bz
            w_lon, dw_lon = f_lon * dot_bu, wv * (dot_gv_u + glon)
            dw_v = (w_dir + w_lon) * dot_gv_u + f_tr * dot_gv_b + w_lon * glon
            dx_v = (wv * dot_gv_u * gv_j) * df_dir + (dw_lon * dot_bu) * df_lon + (wv * dot_gv_b) * df_tr
            sc_v, st_v = wv * (w_dir + w_lon), dw_lon * f_lon
            dux, duy, duz = tl.sum(sc_v * gvx + st_v * bx, axis=1), tl.sum(sc_v * gvy + st_v * by, axis=1), tl.sum(sc_v * gvz + st_v * bz, axis=1)
        else:
            dw_v, dx_v, sc_v = w_dir * dot_gv_u, (wv * dot_gv_u * gv_j) * df_dir, wv * w_dir
            dux, duy, duz = tl.sum(sc_v * gvx, axis=1), tl.sum(sc_v * gvy, axis=1), tl.sum(sc_v * gvz, axis=1)
        dw_4 += tl.sum(tl.reshape(dw_v, [B, 4, 8]), axis=2)
        dx_acc += tl.sum(dx_v, axis=1)
        y20, y21, y22, y23, y24, y30, y31, y32, y33, y34, y35, y36 = _eval_y2_y3(ux, uy, uz)
        f_ten, df_ten = _horner_vd(poly_p, pbase[:, None], TD, 164 + lane[None, :], tsp[:, None], inv_he[:, None], act16)
        gt_j = tl.load(gates_p + gj + 160 + lane[None, :], mask=act16, other=0.0)
        gt0, gt1, gt2, gt3, gt4 = _ld5(g_num_p, gi + OT + 5 * lane[None, :], act16)
        w_ten, dot_gt_y2 = f_ten * gt_j, gt0 * y20[:, None] + gt1 * y21[:, None] + gt2 * y22[:, None] + gt3 * y23[:, None] + gt4 * y24[:, None]
        if HAS_TEN:
            glon2 = tl.load(g_num_p + gi + 336 + lane[None, :], mask=act16, other=0.0)
            t0, t1, t2_, t3, t4 = _ld5(ten_p, 80 * j[:, None] + 5 * lane[None, :], act16)
            f_tr2, df_tr2 = _horner_vd(poly_p, pbase[:, None], TD, 244 + lane[None, :], tsp[:, None], inv_he[:, None], act16)
            f_lon2, df_lon2 = _horner_vd(poly_p, pbase[:, None], TD, 260 + lane[None, :], tsp[:, None], inv_he[:, None], act16)
            dot_ty2, dot_gt_tau = t0 * y20[:, None] + t1 * y21[:, None] + t2_ * y22[:, None] + t3 * y23[:, None] + t4 * y24[:, None], gt0 * t0 + gt1 * t1 + gt2 * t2_ + gt3 * t3 + gt4 * t4
            w_lon2, dw_lon2 = f_lon2 * dot_ty2, wt * (dot_gt_y2 + glon2)
            dw_t = (w_ten + w_lon2) * dot_gt_y2 + f_tr2 * dot_gt_tau + w_lon2 * glon2
            dx_t = (wt * dot_gt_y2 * gt_j) * df_ten + (dw_lon2 * dot_ty2) * df_lon2 + (wt * dot_gt_tau) * df_tr2
            sc_t, st_t = wt * (w_ten + w_lon2), dw_lon2 * f_lon2
            gy20, gy21, gy22, gy23, gy24 = tl.sum(sc_t * gt0 + st_t * t0, axis=1), tl.sum(sc_t * gt1 + st_t * t1, axis=1), tl.sum(sc_t * gt2 + st_t * t2_, axis=1), tl.sum(sc_t * gt3 + st_t * t3, axis=1), tl.sum(sc_t * gt4 + st_t * t4, axis=1)
        else:
            dw_t, dx_t, sc_t = w_ten * dot_gt_y2, (wt * dot_gt_y2 * gt_j) * df_ten, wt * w_ten
            gy20, gy21, gy22, gy23, gy24 = tl.sum(sc_t * gt0, axis=1), tl.sum(sc_t * gt1, axis=1), tl.sum(sc_t * gt2, axis=1), tl.sum(sc_t * gt3, axis=1), tl.sum(sc_t * gt4, axis=1)
        dw_4 += tl.sum(tl.sum(tl.reshape(dw_t, [B, 2, 4, 4]), axis=3), axis=1)
        dx_acc += tl.sum(dx_t, axis=1)
        f_o, df_o = _horner_vd(poly_p, pbase[:, None], TD, FO + lane[None, :], tsp[:, None], inv_he[:, None], act8)
        go_j = tl.load(gates_p + gj + 176 + lane[None, :], mask=act8, other=0.0)
        go0, go1, go2, go3, go4, go5, go6 = _ld7(g_num_p, gi + OO + 7 * lane[None, :], act8)
        dot_go_y3 = go0 * y30[:, None] + go1 * y31[:, None] + go2 * y32[:, None] + go3 * y33[:, None] + go4 * y34[:, None] + go5 * y35[:, None] + go6 * y36[:, None]
        w_o, sc_o = f_o * go_j, wo * f_o * go_j
        dw_4 += tl.sum(tl.sum(tl.reshape(w_o * dot_go_y3, [B, 4, 4, 2]), axis=3), axis=1)
        dx_acc += tl.sum((wo * dot_go_y3 * go_j) * df_o, axis=1)
        dux, duy, duz = _vjp_y2_y3(ux, uy, uz, dux, duy, duz, gy20, gy21, gy22, gy23, gy24, tl.sum(sc_o * go0, axis=1), tl.sum(sc_o * go1, axis=1), tl.sum(sc_o * go2, axis=1), tl.sum(sc_o * go3, axis=1), tl.sum(sc_o * go4, axis=1), tl.sum(sc_o * go5, axis=1), tl.sum(sc_o * go6, axis=1))
        dw_tot = tl.where(act[:, None], dw_4 + gzc, 0.0)
        gl = dw_tot * ws
        d_chi = tl.where(act, gdsum_i + tl.sum(dw_tot * ex_l, axis=1), 0.0)
        d_x = dx_acc + tl.sum(gl * (1.0 - th_b * th_b) * db_dx, axis=1)
        d_rfl, dgu = d_chi * dchi_drfl + d_x * dx_drfl, dux * ux + duy * uy + duz * uz
        gdx, gdy, gdz = tl.where(act, (dux - dgu * ux) * inv_r + d_rfl * drfl_x, 0.0), tl.where(act, (duy - dgu * uy) * inv_r + d_rfl * drfl_y, 0.0), tl.where(act, (duz - dgu * uz) * inv_r + d_rfl * drfl_z, 0.0)
        dpos_x, dpos_y, dpos_z = dpos_x - gdx, dpos_y - gdy, dpos_z - gdz
        dq += tl.reshape(tl.reshape(gl, [B, 4, 1]) * tl.reshape(k, [B, 4, 16]), [B, 64])
        drho_e = tl.where(act, d_x * dx_drho, 0.0) if NEED_DRHO else tl.zeros([B], tl.float32)
        drho_i += drho_e
        if NEED_VIR:
            v00, v01, v02 = v00 + esx * gdx, v01 + esx * gdy, v02 + esx * gdz
            v10, v11, v12 = v10 + esy * gdx, v11 + esy * gdy, v12 + esy * gdz
            v20, v21, v22 = v20 + esz * gdx, v21 + esz * gdy, v22 + esz * gdz
        _st3(eg8_p, 8 * e, gdx, gdy, gdz, ve)
        tl.store(eg8_p + 8 * e[:, None] + 3 + h4[None, :], gl, mask=ve[:, None])
        tl.store(eg8_p + 8 * e + 7, drho_e, mask=ve)
    _st3(d_pos_p, 3 * i, dpos_x, dpos_y, dpos_z, am)
    tl.store(d_qn_p + 64 * i[:, None] + c64[None, :], dq, mask=am[:, None])
    if NEED_DRHO:
        tl.store(d_rho_p + i, drho_i, mask=am)
    if NEED_VIR:
        cbo = 9 * g_id
        _av3(d_cell_p, cbo, 0, v00, v01, v02, am, IS_G1)
        _av3(d_cell_p, cbo, 3, v10, v11, v12, am, IS_G1)
        _av3(d_cell_p, cbo, 6, v20, v21, v22, am, IS_G1)


@_jit
def _triton_csr_bwd_send(pos_p, sptr_p, sperm_p, ridx_p, sh_p, cell_p, grp_p, z_p, rmin_p, rho_p, poly_p, qn_p, kn_p, g_num_p, eg8_p, d_pos_p, d_gates_p, d_kn_p, d_vec_p, d_ten_p, d_rho_p, n_atoms, grid, r_cut, r_on, max_doff_r2, LATER: tl.constexpr, HAS_FLOOR: tl.constexpr, HAS_RHO: tl.constexpr, HAS_TEN: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr, NEED_DRHO: tl.constexpr, B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    am, lane, c64, c128, h4 = j < n_atoms, tl.arange(0, 32), tl.arange(0, 64), tl.arange(0, 128), tl.arange(0, 4)
    m16, m8 = lane < 16, lane < 8
    TD: tl.constexpr = 284 if LATER else 188
    OD: tl.constexpr = 408 if LATER else 360
    FO: tl.constexpr = 276 if LATER else 180
    OT: tl.constexpr = 256 if LATER else 224
    OO: tl.constexpr = 352 if LATER else 304
    zj = tl.minimum(tl.maximum(tl.load(z_p + j, mask=am, other=0).to(tl.int32), 0), 127) if HAS_FLOOR else tl.zeros([B], tl.int32)
    rho_j = tl.load(rho_p + j, mask=am, other=0.0) if HAS_RHO else tl.zeros([B], tl.float32)
    pjx, pjy, pjz = _ld3(pos_p, 3 * j, am)
    k = tl.load(kn_p + 64 * j[:, None] + c64[None, :], mask=am[:, None], other=0.0)
    dpos_jx, dpos_jy, dpos_jz, drho_j = tl.zeros([B], tl.float32), tl.zeros([B], tl.float32), tl.zeros([B], tl.float32), tl.zeros([B], tl.float32)
    dk, dgs = tl.zeros([B, 64], tl.float32), tl.zeros([B, 128], tl.float32)
    dgv, dgt, dgo = tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32)
    dbx, dby, dbz = tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32)
    dt0, dt1, dt2, dt3, dt4 = tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32), tl.zeros([B, 32], tl.float32)
    inv_w, inv_h, rcut2 = 1.0 / (r_cut - r_on), grid / r_cut, r_cut * r_cut
    s_beg, s_end = tl.load(sptr_p + j, mask=am, other=0), tl.load(sptr_p + j + 1, mask=am, other=0)
    for step in range(tl.max(tl.maximum(s_end - s_beg, 0), axis=0)):
        kp = s_beg + step
        ve = am & (kp < s_end)
        e = tl.load(sperm_p + kp, mask=ve, other=0)
        i = tl.load(ridx_p + e, mask=ve, other=0)
        g_id = tl.zeros([B], tl.int32) if (not HAS_CELL or IS_G1) else tl.load(grp_p + i, mask=ve, other=0).to(tl.int32)
        pix, piy, piz = _ld3(pos_p, 3 * i, ve)
        ux, uy, uz, _, _, _, _, _, _, _, act, chi, _, _, _, _, tsp, pbase = _csr_geom(pix, piy, piz, pjx, pjy, pjz, sh_p, cell_p, e, g_id, ve, z_p, rmin_p, i, zj, rho_p, i, rho_j, grid, r_cut, r_on, inv_w, inv_h, rcut2, max_doff_r2, TD, HAS_CELL, HAS_FLOOR, HAS_RHO, False, False, True, B)
        egx, egy, egz = _ld3(eg8_p, 8 * e, act)
        dpos_jx, dpos_jy, dpos_jz = dpos_jx + egx, dpos_jy + egy, dpos_jz + egz
        gl = tl.load(eg8_p + 8 * e[:, None] + 3 + h4[None, :], mask=act[:, None], other=0.0)
        if NEED_DRHO:
            drho_j += tl.load(eg8_p + 8 * e + 7, mask=act, other=0.0)
        q = tl.load(qn_p + 64 * i[:, None] + c64[None, :], mask=act[:, None], other=0.0)
        dk += tl.reshape(tl.reshape(gl, [B, 4, 1]) * tl.reshape(q, [B, 4, 16]), [B, 64])
        _, ws128, wv, wt, wo = _qk_heads(q, k, chi, poly_p, pbase, TD, h4, tsp, act, m16, m8, B)
        gi, act32, act16, act8 = OD * i[:, None], act[:, None], act[:, None] & m16[None, :], act[:, None] & m8[None, :]
        dgs += tl.load(g_num_p + gi + c128[None, :], mask=act32, other=0.0) * ws128 * _horner_v(poly_p, pbase[:, None], TD, 4 + c128[None, :], tsp[:, None], act32)
        gvx, gvy, gvz = _ld3(g_num_p, gi + 128 + 3 * lane[None, :], act32)
        dot_gv_u = gvx * ux[:, None] + gvy * uy[:, None] + gvz * uz[:, None]
        dgv += dot_gv_u * wv * _horner_v(poly_p, pbase[:, None], TD, 132 + lane[None, :], tsp[:, None], act32)
        if LATER:
            glon = tl.load(g_num_p + gi + 224 + lane[None, :], mask=act32, other=0.0)
            f_tr = _horner_v(poly_p, pbase[:, None], TD, 180 + lane[None, :], tsp[:, None], act32)
            sl1 = _horner_v(poly_p, pbase[:, None], TD, 212 + lane[None, :], tsp[:, None], act32) * (dot_gv_u + glon)
            dbx, dby, dbz = dbx + wv * (f_tr * gvx + sl1 * ux[:, None]), dby + wv * (f_tr * gvy + sl1 * uy[:, None]), dbz + wv * (f_tr * gvz + sl1 * uz[:, None])
        y20, y21, y22, y23, y24, y30, y31, y32, y33, y34, y35, y36 = _eval_y2_y3(ux, uy, uz)
        gt0, gt1, gt2, gt3, gt4 = _ld5(g_num_p, gi + OT + 5 * lane[None, :], act16)
        dot_gt_y2 = gt0 * y20[:, None] + gt1 * y21[:, None] + gt2 * y22[:, None] + gt3 * y23[:, None] + gt4 * y24[:, None]
        dgt += dot_gt_y2 * wt * _horner_v(poly_p, pbase[:, None], TD, 164 + lane[None, :], tsp[:, None], act16)
        if HAS_TEN:
            glon2 = tl.load(g_num_p + gi + 336 + lane[None, :], mask=act16, other=0.0)
            st2 = wt * _horner_v(poly_p, pbase[:, None], TD, 244 + lane[None, :], tsp[:, None], act16)
            sl2 = wt * _horner_v(poly_p, pbase[:, None], TD, 260 + lane[None, :], tsp[:, None], act16) * (dot_gt_y2 + glon2)
            dt0, dt1, dt2 = dt0 + st2 * gt0 + sl2 * y20[:, None], dt1 + st2 * gt1 + sl2 * y21[:, None], dt2 + st2 * gt2 + sl2 * y22[:, None]
            dt3, dt4 = dt3 + st2 * gt3 + sl2 * y23[:, None], dt4 + st2 * gt4 + sl2 * y24[:, None]
        go0, go1, go2, go3, go4, go5, go6 = _ld7(g_num_p, gi + OO + 7 * lane[None, :], act8)
        dot_go_y3 = go0 * y30[:, None] + go1 * y31[:, None] + go2 * y32[:, None] + go3 * y33[:, None] + go4 * y34[:, None] + go5 * y35[:, None] + go6 * y36[:, None]
        dgo += dot_go_y3 * wo * _horner_v(poly_p, pbase[:, None], TD, FO + lane[None, :], tsp[:, None], act8)
    rp_x, rp_y, rp_z = _ld3(d_pos_p, 3 * j, am)
    _st3(d_pos_p, 3 * j, rp_x + dpos_jx, rp_y + dpos_jy, rp_z + dpos_jz, am)
    if NEED_DRHO:
        tl.store(d_rho_p + j, tl.load(d_rho_p + j, mask=am, other=0.0) + drho_j, mask=am)
    dgj, m32, m16b, m8b = 184 * j[:, None], am[:, None], am[:, None] & m16[None, :], am[:, None] & m8[None, :]
    tl.store(d_gates_p + dgj + c128[None, :], dgs, mask=m32)
    tl.store(d_gates_p + dgj + 128 + lane[None, :], dgv, mask=m32)
    tl.store(d_gates_p + dgj + 160 + lane[None, :], dgt, mask=m16b)
    tl.store(d_gates_p + dgj + 176 + lane[None, :], dgo, mask=m8b)
    tl.store(d_kn_p + 64 * j[:, None] + c64[None, :], dk, mask=m32)
    if LATER:
        _st3(d_vec_p, 96 * j[:, None] + 3 * lane[None, :], dbx, dby, dbz, m32)
        if HAS_TEN:
            _st5(d_ten_p, 80 * j[:, None] + 5 * lane[None, :], dt0, dt1, dt2, dt3, dt4, m16b)


def triton_csr_forward(plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor, cell: Tensor, graph: Tensor, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    dev, n = pos.device, plan.n_atoms
    if n == 0 or plan.n_edges == 0:
        return torch.zeros((n, layout.out_dim), dtype=torch.float32, device=dev), torch.zeros((n, 4), dtype=torch.float32, device=dev), torch.zeros((n,), dtype=torch.float32, device=dev)
    ba, nw = _cfg_block_warps(n)
    has_floor, has_rho, has_ten, has_cell = bool(cfg.floor and z.numel() > 0), bool(rho.numel() > 0), bool(layout.later and ten.numel() > 0), bool(cell.numel() > 0)
    is_g1, df, di = bool(has_cell and cell.shape[0] == 1), dummy_f32(dev), dummy_i64(dev)
    out_num, out_zc, out_dsum = torch.empty((n, layout.out_dim), dtype=torch.float32, device=dev), torch.empty((n, 4), dtype=torch.float32, device=dev), torch.empty((n,), dtype=torch.float32, device=dev)
    _triton_csr_fwd[((n + ba - 1) // ba,)](
        pos, plan.recv_ptr, plan.send_idx, shift_in, cell if has_cell else df, graph if graph.numel() else di, z if has_floor else di, rmin_table_f32(dev) if has_floor else df,
        rho if has_rho else df, poly, gates, qn, kn, vec if vec.numel() else df, ten if has_ten else df, out_num, out_zc, out_dsum,
        n, int(poly.shape[0]), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2),
        LATER=layout.later, HAS_FLOOR=has_floor, HAS_RHO=has_rho, HAS_TEN=has_ten, HAS_CELL=has_cell, IS_G1=is_g1, B=ba, num_warps=nw,
    )
    return out_num, out_zc, out_dsum


def triton_csr_backward(plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor, cell: Tensor, graph: Tensor, z: Tensor, g_num: Tensor, g_zc: Tensor, g_dsum: Tensor, *, need_vir: bool, need_drho: bool, need_dshift: bool) -> tuple[Any, ...]:
    """Atomic-free 2-pass Triton Dual-CSR backward (`_triton_csr_bwd_recv` + `_triton_csr_bwd_send`), with `_StreamVJP` fallback."""
    dev, n, n_e = pos.device, plan.n_atoms, plan.n_edges
    has_ten = bool(layout.later and ten.numel() > 0)
    if n == 0 or n_e == 0:
        return torch.zeros_like(pos), torch.zeros_like(gates), torch.zeros_like(vec) if layout.later else None, torch.zeros_like(ten) if has_ten else None, torch.zeros_like(qn), torch.zeros_like(kn), torch.zeros_like(rho) if need_drho else None, torch.zeros_like(cell) if need_vir else None, torch.zeros_like(shift_in) if need_dshift else None
    if not HAS_TRITON or os.environ.get("LAM4_TRITON_BWD", "1") == "0" or need_dshift:
        return stream_vjp_csr_backward(plan, layout, cfg, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z, g_num, g_zc, g_dsum, need_vir=need_vir, need_drho=need_drho, need_dshift=need_dshift)
    ba, nw = _cfg_block_warps(n)
    has_floor, has_rho, has_cell = bool(cfg.floor and z.numel() > 0), bool(rho.numel() > 0), bool(cell.numel() > 0)
    is_g1, eff_vir, eff_drho = bool(has_cell and cell.shape[0] == 1), bool(need_vir and has_cell), bool(need_drho and has_rho)
    df, di = dummy_f32(dev), dummy_i64(dev)
    rmin, grid = rmin_table_f32(dev) if has_floor else df, ((n + ba - 1) // ba,)
    d_pos, d_qn, edge_grad8 = torch.empty((n, 3), dtype=torch.float32, device=dev), torch.empty_like(qn), torch.empty((n_e, 8), dtype=torch.float32, device=dev)
    d_cell, d_rho = torch.zeros_like(cell) if eff_vir else df, torch.empty((n,), dtype=torch.float32, device=dev) if eff_drho else df
    _triton_csr_bwd_recv[grid](
        pos, plan.recv_ptr, plan.send_idx, shift_in, cell if has_cell else df, graph if graph.numel() else di, z if has_floor else di, rmin,
        rho if has_rho else df, poly, gates, qn, kn, vec if vec.numel() else df, ten if has_ten else df, g_num, g_zc, g_dsum, d_pos, d_qn, edge_grad8, d_cell, d_rho,
        n, int(poly.shape[0]), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2),
        LATER=layout.later, HAS_FLOOR=has_floor, HAS_RHO=has_rho, HAS_TEN=has_ten, HAS_CELL=has_cell, IS_G1=is_g1, NEED_VIR=eff_vir, NEED_DRHO=eff_drho, B=ba, num_warps=nw,
    )
    d_gates, d_kn = torch.empty((n, layout.gate_dim), dtype=torch.float32, device=dev), torch.empty_like(kn)
    d_vec, d_ten = torch.empty((n, 32, 3), dtype=torch.float32, device=dev) if layout.later else df, torch.empty((n, 16, 5), dtype=torch.float32, device=dev) if has_ten else df
    _triton_csr_bwd_send[grid](
        pos, plan.send_ptr, plan.send_perm, plan.recv_idx, shift_in, cell if has_cell else df, graph if graph.numel() else di, z if has_floor else di, rmin,
        rho if has_rho else df, poly, qn, kn, g_num, edge_grad8, d_pos, d_gates, d_kn, d_vec, d_ten, d_rho,
        n, int(poly.shape[0]), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2),
        LATER=layout.later, HAS_FLOOR=has_floor, HAS_RHO=has_rho, HAS_TEN=has_ten, HAS_CELL=has_cell, IS_G1=is_g1, NEED_DRHO=eff_drho, B=ba, num_warps=nw,
    )
    return d_pos, d_gates, d_vec if layout.later else None, d_ten if has_ten else None, d_qn, d_kn, d_rho if eff_drho else None, d_cell if eff_vir else None, None
