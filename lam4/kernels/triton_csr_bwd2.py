"""Analytic 2nd-order Dual-CSR Triton kernels (`O1` double-backward + `O2` `d_poly`, 0 `[E, C]` DRAM buffers)."""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

from lam4.core.edge_terms import EdgeCfg
from lam4.core.layout import StageLayout
from lam4.kernels.csr_plan import MAX_DOFF_R2, CSRPlan, dummy_f32, dummy_i64, rmin_table_f32
from lam4.kernels.triton_csr import HAS_TRITON, _bcast_heads, _cfg_block_warps, _horner_vd, _jit, has_triton_csr_backward, tl
from lam4.kernels.triton_nlh import _av3, _eval_y2_y3, _ld3, _ld5, _ld7, _nlh_shift, _st3, _st5, _st7, _tanh_half, _vjp_y2_y3


def _bwd2_block_warps(n_atoms: int) -> tuple[int, int]:
    ba, nw = os.environ.get("LAM4_BWD2_BLOCK_ATOMS"), os.environ.get("LAM4_BWD2_WARPS")
    dba, dnw = (2, 4) if n_atoms > 256 else (2, 2)
    return (int(ba) if ba else dba, int(nw) if nw else dnw)


def _dp_splits(n_atoms: int) -> int:
    env = os.environ.get("LAM4_DPOLY_SPLITS")
    return max(1, int(env)) if env is not None else (8 if n_atoms >= 256 else 1)


def has_triton_csr_bwd2(device: torch.device | str | None = None) -> bool:
    """True when CUDA Dual-CSR double-backward (`O1/O2`) is enabled."""
    return os.environ.get("LAM4_CSR_BWD2", "1") != "0" and has_triton_csr_backward(device)


@_jit
def _d5(a0, a1, a2, a3, a4, b0, b1, b2, b3, b4): return a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4
@_jit
def _d7(a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6): return a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6
@_jit
def _sum5(a, t0, t1, t2, t3, t4, b, u0, u1, u2, u3, u4, c, v0, v1, v2, v3, v4):
    return tl.sum(a * t0 + b * u0 + c * v0, axis=1), tl.sum(a * t1 + b * u1 + c * v1, axis=1), tl.sum(a * t2 + b * u2 + c * v2, axis=1), tl.sum(a * t3 + b * u3 + c * v3, axis=1), tl.sum(a * t4 + b * u4 + c * v4, axis=1)
@_jit
def _sum7(a, g0, g1, g2, g3, g4, g5, g6):
    return tl.sum(a * g0, axis=1), tl.sum(a * g1, axis=1), tl.sum(a * g2, axis=1), tl.sum(a * g3, axis=1), tl.sum(a * g4, axis=1), tl.sum(a * g5, axis=1), tl.sum(a * g6, axis=1)


@_jit
def _hd(poly_p, vp_p, base, td: tl.constexpr, col, tsp, inv_h, dot_x, mask, HAS_VP: tl.constexpr):
    a0, a1 = tl.load(poly_p + base + col, mask=mask, other=0.0, eviction_policy="evict_last"), tl.load(poly_p + base + td + col, mask=mask, other=0.0, eviction_policy="evict_last")
    a2, a3 = tl.load(poly_p + base + 2 * td + col, mask=mask, other=0.0, eviction_policy="evict_last"), tl.load(poly_p + base + 3 * td + col, mask=mask, other=0.0, eviction_policy="evict_last")
    val, df, d2f = ((a3 * tsp + a2) * tsp + a1) * tsp + a0, ((3.0 * a3 * tsp + 2.0 * a2) * tsp + a1) * inv_h, (6.0 * a3 * tsp + 2.0 * a2) * (inv_h * inv_h)
    vp, dvp = _horner_vd(vp_p, base, td, col, tsp, inv_h, mask) if HAS_VP else (val * 0.0, val * 0.0)
    return val, df, d2f, vp + df * dot_x, dvp


@_jit
def _acc_poly(dp_p, base, td: tl.constexpr, col, tsp, gv, gd, mask, NEED_DP: tl.constexpr, NEED_B2: tl.constexpr):
    if NEED_DP:
        t2, ge = tsp * tsp, (gd if NEED_B2 else (gv * 0.0))
        tl.atomic_add(dp_p + base + col, gv, mask=mask)
        tl.atomic_add(dp_p + base + td + col, gv * tsp + ge, mask=mask)
        tl.atomic_add(dp_p + base + 2 * td + col, gv * t2 + ge * (2.0 * tsp), mask=mask)
        tl.atomic_add(dp_p + base + 3 * td + col, gv * (t2 * tsp) + ge * (3.0 * t2), mask=mask)


@_jit
def _y2_y3_jvp(ux, uy, uz, dx, dy, dz):
    xx, yy, zz, dxx, dyy, dzz = ux * ux, uy * uy, uz * uz, 2.0 * ux * dx, 2.0 * uy * dy, 2.0 * uz * dz
    k2, k4, k5, k6, k7 = 1.4142135623730951, 2.449489742783178, 0.3872983346207417, 0.31622776601683794, 1.224744871391589
    return (k2 * (dx * uy + ux * dy), k2 * (dx * uz + ux * dz), k2 * (dy * uz + uy * dz), (dxx - dyy) * 0.7071067811865475, (2.0 * dzz - dxx - dyy) * 0.4082482904638630, 0.5 * (dy * (3.0 * xx - yy) + uy * (3.0 * dxx - dyy)), k4 * (dx * uy * uz + ux * dy * uz + ux * uy * dz), k5 * (dy * (4.0 * zz - xx - yy) + uy * (4.0 * dzz - dxx - dyy)), k6 * (dz * (2.0 * zz - 3.0 * xx - 3.0 * yy) + uz * (2.0 * dzz - 3.0 * dxx - 3.0 * dyy)), k5 * (dx * (4.0 * zz - xx - yy) + ux * (4.0 * dzz - dxx - dyy)), k7 * (dz * (xx - yy) + uz * (dxx - dyy)), 0.5 * (dx * (xx - 3.0 * yy) + ux * (dxx - 3.0 * dyy)))


@_jit
def _y2_y3_hess_u(ux, uy, uz, dux, duy, duz, cx, cy, cz, g20, g21, g22, g23, g24, g30, g31, g32, g33, g34, g35, g36):
    cx, cy, cz = _vjp_y2_y3(dux, duy, duz, cx, cy, cz, g20, g21, g22, g23, g24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    k4, k5, k6, k7, k5b, k6b, k5c = 2.449489742783178, 0.3872983346207417, 0.31622776601683794, 1.224744871391589, -0.7745966692414834, -1.8973665961010278, 3.0983866769659336
    dxx, dyy, dzz, dxy, dxz, dyz = 2.0 * ux * dux, 2.0 * uy * duy, 2.0 * uz * duz, dux * uy + ux * duy, dux * uz + ux * duz, duy * uz + uy * duz
    return (cx + 3.0 * g30 * dxy + k4 * g31 * dyz + k5b * g32 * dxy + k6b * g33 * dxz + g34 * k5 * (4.0 * dzz - 3.0 * dxx - dyy) + k4 * g35 * dxz + 0.5 * g36 * (3.0 * dxx - 3.0 * dyy), cy + 0.5 * g30 * (3.0 * dxx - 3.0 * dyy) + k4 * g31 * dxz + g32 * k5 * (4.0 * dzz - dxx - 3.0 * dyy) + k6b * g33 * dyz + k5b * g34 * dxy - k4 * g35 * dyz - 3.0 * g36 * dxy, cz + k4 * g31 * dxy + k5c * g32 * dyz + g33 * k6 * (6.0 * dzz - 3.0 * dxx - 3.0 * dyy) + k5c * g34 * dxz + k7 * g35 * (dxx - dyy))


@_jit
def _geom_tan(dx, dy, dz, ddx, ddy, ddz, ve, z_p, rmin_p, z_peer, zi_s, rho_p, rho_peer, rho_self, dot_rho_p, dot_rho_peer, dot_rho_self, grid, r_cut, r_on, inv_w, inv_h, rcut2, max_doff_r2, HAS_FLOOR: tl.constexpr, HAS_RHO: tl.constexpr, HAS_DR: tl.constexpr, NEED_HESS: tl.constexpr, IS_SEND: tl.constexpr, B: tl.constexpr):
    r2 = dx * dx + dy * dy + dz * dz
    ar = ve & (r2 >= 1e-24) & (r2 < rcut2)
    r_raw = tl.sqrt(tl.where(ar, r2, 1.0))
    inv_r = tl.where(ar, 1.0 / r_raw, 0.0)
    ux, uy, uz, dot_r = dx * inv_r, dy * inv_r, dz * inv_r, (dx * ddx + dy * ddy + dz * ddz) * inv_r
    r_fl, drx, dry, drz, ddrx, ddry, ddrz, dot_rfl = r_raw, ux, uy, uz, (ddx - dot_r * ux) * inv_r, (ddy - dot_r * uy) * inv_r, (ddz - dot_r * uz) * inv_r, dot_r
    if HAS_FLOOR:
        nf = ar & (r2 < max_doff_r2)
        zp = tl.minimum(tl.maximum(tl.load(z_p + z_peer, mask=nf, other=0).to(tl.int32), 0), 127)
        r_min = tl.load(rmin_p + (((zp << 7) + zi_s) if IS_SEND else (zi_s + zp)), mask=nf, other=0.0, eviction_policy="evict_last")
        d_off, r_eps = r_min + 0.05, tl.sqrt(tl.where(nf, r2, 0.0) + 1e-8)
        fon = nf & (r_eps < d_off) & (r_min > 0.0)
        e2 = tl.exp(-40.0 * tl.where(fon, d_off - r_eps, 0.0))
        th = (1.0 - e2) / (1.0 + e2)
        fsc, dot_reps = (1.0 - th * th) / r_eps, (dx * ddx + dy * ddy + dz * ddz) / r_eps
        r_fl, dot_rfl, drx, dry, drz = tl.where(fon, d_off - 0.05 * th, r_fl), tl.where(fon, (1.0 - th * th) * dot_reps, dot_rfl), tl.where(fon, dx * fsc, drx), tl.where(fon, dy * fsc, dry), tl.where(fon, dz * fsc, drz)
        if NEED_HESS:
            dfsc = fsc * (40.0 * th - 1.0 / r_eps) * dot_reps
            ddrx, ddry, ddrz = tl.where(fon, ddx * fsc + dx * dfsc, ddrx), tl.where(fon, ddy * fsc + dy * dfsc, ddry), tl.where(fon, ddz * fsc + dz * dfsc, ddrz)
    act = ar & (r_fl < r_cut)
    in_env = act & (r_fl > r_on)
    te = tl.where(in_env, (r_fl - r_on) * inv_w, 0.0)
    t2, om = te * te, 1.0 - te
    chi, dchi = tl.where(act, tl.where(in_env, 1.0 - t2 * t2 * (35.0 + te * (-84.0 + te * (70.0 - 20.0 * te))), 1.0), 0.0), tl.where(in_env, -140.0 * (t2 * te) * (om * om * om) * inv_w, 0.0)
    ddchi = tl.where(in_env, -420.0 * t2 * (om * om) * (1.0 - 2.0 * te) * (inv_w * inv_w), 0.0) if NEED_HESS else tl.zeros([B], tl.float32)
    dot_chi, x, dxf, dxr, ddxf, ddxr, dot_x = dchi * dot_rfl, r_fl, tl.where(act, 1.0, 0.0), tl.zeros([B], tl.float32), tl.zeros([B], tl.float32), tl.zeros([B], tl.float32), dot_rfl
    if HAS_RHO:
        drij_p = tl.load(dot_rho_p + rho_peer, mask=act, other=0.0) if HAS_DR else tl.zeros([B], tl.float32)
        rij, drij = tl.where(act, rho_self + tl.load(rho_p + rho_peer, mask=act, other=1.0), 1.0), tl.where(act, dot_rho_self + drij_p, 0.0)
        sw = 1.52 / rij
        iden = 1.0 / tl.where(act, 1.0 + (sw - 1.0) * (r_fl / r_cut), 1.0)
        x, dxf, dxr = tl.where(act, (sw * r_fl) * iden, 0.0), tl.where(act, sw * (iden * iden), 0.0), tl.where(act, -(1.52 / (rij * rij)) * r_fl * (1.0 - r_fl / r_cut) * (iden * iden), 0.0)
        dot_x = dxf * dot_rfl + dxr * drij
        if NEED_HESS:
            dsw, diden = -(1.52 / (rij * rij)) * drij, -(iden * iden) * (-(1.52 / (rij * rij)) * drij * (r_fl / r_cut) + (sw - 1.0) * (dot_rfl / r_cut))
            ddxf, ddxr = tl.where(act, dsw * (iden * iden) + sw * 2.0 * iden * diden, 0.0), tl.where(act, ((3.04 / (rij * rij * rij)) * drij * r_fl * (1.0 - r_fl / r_cut) - (1.52 / (rij * rij)) * (1.0 - 2.0 * r_fl / r_cut) * dot_rfl) * (iden * iden) - (1.52 / (rij * rij)) * r_fl * (1.0 - r_fl / r_cut) * 2.0 * iden * diden, 0.0)
    inv_he, u_sp = tl.where(act & (x >= 0.0) & (x <= r_cut), inv_h, 0.0), tl.minimum(tl.maximum(x, 0.0), r_cut) * inv_h
    idx = tl.minimum(tl.maximum(u_sp.to(tl.int32), 0), grid - 1)
    return act, inv_r, ux, uy, uz, (ddx - dot_r * ux) * inv_r, (ddy - dot_r * uy) * inv_r, (ddz - dot_r * uz) * inv_r, drx, dry, drz, ddrx, ddry, ddrz, dot_rfl, chi, dchi, ddchi, dot_chi, dxf, dxr, ddxf, ddxr, dot_x, inv_he, u_sp - idx.to(tl.float32), idx


@_jit
def _qk_b4_dual(q, vq, k, vk, chi, dot_chi, poly_p, vp_p, pbase, TD: tl.constexpr, h4, tsp, inv_he, dot_x, act, m16, m8, HAS_VP: tl.constexpr, B: tl.constexpr):
    dots, ddots = tl.sum(tl.reshape(q * k, [B, 4, 16]), axis=2), tl.sum(tl.reshape(vq * k + q * vk, [B, 4, 16]), axis=2)
    b_raw, db_dx, d2b_dx, b_tan, dvp_b = _hd(poly_p, vp_p, pbase[:, None], TD, h4[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act[:, None], HAS_VP)
    th_b = _tanh_half(b_raw)
    sech2, ex_l = 1.0 - th_b * th_b, tl.where(act[:, None], tl.exp(dots + 2.0 * th_b), 0.0)
    ws, eta_tan = chi[:, None] * ex_l, ddots + sech2 * b_tan
    dws = dot_chi[:, None] * ex_l + ws * eta_tan
    w0, w1, w2, w3 = _bcast_heads(ws, m16, m8, B)
    dw0, dw1, dw2, dw3 = _bcast_heads(dws, m16, m8, B)
    return db_dx, d2b_dx, th_b, sech2, ex_l, b_tan, dvp_b, eta_tan, ws, dws, w0, w1, w2, w3, dw0, dw1, dw2, dw3


@_jit
def _triton_csr_bwd2_recv(pos_p, rptr_p, sidx_p, sh_p, cell_p, grp_p, z_p, rmin_p, rho_p, poly_p, gates_p, qn_p, kn_p, vec_p, ten_p, g_num_p, g_zc_p, g_dsum_p, v_pos_p, v_poly_p, v_gates_p, v_qn_p, v_kn_p, v_vec_p, v_ten_p, v_rho_p, v_cell_p, d_pos_p, d_poly_p, d_qn_p, eg16_p, d_cell_p, d_rho_p, d_gnum_p, d_gzc_p, d_gdsum_p, n_atoms, grid, r_cut, r_on, max_doff_r2, LATER: tl.constexpr, HAS_FLOOR: tl.constexpr, HAS_RHO: tl.constexpr, HAS_TEN: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr, HAS_VPOLY: tl.constexpr, HAS_VRHO: tl.constexpr, HAS_VCELL_EFF: tl.constexpr, HAS_VG: tl.constexpr, HAS_VV: tl.constexpr, HAS_VT: tl.constexpr, NEED_DPOLY: tl.constexpr, NEED_BWD2: tl.constexpr, NEED_JVP: tl.constexpr, NEED_V: tl.constexpr, NEED_VIR: tl.constexpr, NEED_DRHO: tl.constexpr, DP_SPLIT: tl.constexpr, B: tl.constexpr):
    i, lane, c64, c128, h4 = tl.program_id(0) * B + tl.arange(0, B), tl.arange(0, 32), tl.arange(0, 64), tl.arange(0, 128), tl.arange(0, 4)
    am, m16, m8 = i < n_atoms, lane < 16, lane < 8
    TD: tl.constexpr = 284 if LATER else 188
    OD: tl.constexpr = 408 if LATER else 360
    FO: tl.constexpr = 276 if LATER else 180
    OT: tl.constexpr = 256 if LATER else 224
    OO: tl.constexpr = 352 if LATER else 304
    if NEED_DPOLY and DP_SPLIT > 1:
        d_poly_p = d_poly_p + (tl.program_id(0) % DP_SPLIT) * (grid * 4 * TD)
    g_id, zi_s = (tl.zeros([B], tl.int32) if (not HAS_CELL or IS_G1) else tl.load(grp_p + i, mask=am, other=0).to(tl.int32)), ((tl.minimum(tl.maximum(tl.load(z_p + i, mask=am, other=0).to(tl.int32), 0), 127) << 7) if HAS_FLOOR else tl.zeros([B], tl.int32))
    rho_i, v_rho_i = (tl.load(rho_p + i, mask=am, other=0.0) if HAS_RHO else tl.zeros([B], tl.float32)), (tl.load(v_rho_p + i, mask=am, other=0.0) if (HAS_RHO and HAS_VRHO) else tl.zeros([B], tl.float32))
    px, py, pz = _ld3(pos_p, 3 * i, am)
    vpx, vpy, vpz = _ld3(v_pos_p, 3 * i, am) if NEED_V else (px * 0.0, px * 0.0, px * 0.0)
    q, vq, gi, m32 = tl.load(qn_p + 64 * i[:, None] + c64[None, :], mask=am[:, None], other=0.0), (tl.load(v_qn_p + 64 * i[:, None] + c64[None, :], mask=am[:, None], other=0.0) if NEED_V else tl.zeros([B, 64], tl.float32)), OD * i[:, None], am[:, None]
    gs, gzc, gdsum_i = tl.load(g_num_p + gi + c128[None, :], mask=m32, other=0.0), tl.load(g_zc_p + 4 * i[:, None] + h4[None, :], mask=m32, other=0.0), tl.load(g_dsum_p + i, mask=am, other=0.0)
    gvx, gvy, gvz = _ld3(g_num_p, gi + 128 + 3 * lane[None, :], m32)
    glon, z0, z32 = (tl.load(g_num_p + gi + 224 + lane[None, :], mask=m32, other=0.0) if LATER else tl.zeros([B, 32], tl.float32)), tl.zeros([B], tl.float32), tl.zeros([B, 32], tl.float32)
    dpos_x, dpos_y, dpos_z, drho_i, dq, v00, v01, v02, v10, v11, v12, v20, v21, v22 = z0, z0, z0, z0, tl.zeros([B, 64], tl.float32), z0, z0, z0, z0, z0, z0, z0, z0, z0
    jas, jaz, jadsum, javx, javy, javz, jalon, jalon2, jat0, jat1, jat2, jat3, jat4, jao0, jao1, jao2, jao3, jao4, jao5, jao6 = tl.zeros([B, 128], tl.float32), tl.zeros([B, 4], tl.float32), z0, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32
    inv_w, inv_h, rcut2, e_beg, e_end = 1.0 / (r_cut - r_on), grid / r_cut, r_cut * r_cut, tl.load(rptr_p + i, mask=am, other=0), tl.load(rptr_p + i + 1, mask=am, other=0)
    for step in range(tl.max(tl.maximum(e_end - e_beg, 0), axis=0)):
        e = e_beg + step
        ve = am & (e < e_end)
        j = tl.load(sidx_p + e, mask=ve, other=0)
        shx, shy, shz, esx, esy, esz = _nlh_shift(sh_p, cell_p, e, g_id, ve, HAS_CELL)
        vshx, vshy, vshz, _, _, _ = _nlh_shift(sh_p, v_cell_p, e, g_id, ve, True) if HAS_VCELL_EFF else (z0, z0, z0, z0, z0, z0)
        pjx, pjy, pjz = _ld3(pos_p, 3 * j, ve)
        vpjx, vpjy, vpjz = _ld3(v_pos_p, 3 * j, ve) if NEED_V else (z0, z0, z0)
        ddx, ddy, ddz = vpjx - vpx + vshx, vpjy - vpy + vshy, vpjz - vpz + vshz
        act, inv_r, ux, uy, uz, dux, duy, duz, drx, dry, drz, ddrx, ddry, ddrz, drfl_tan, chi, dchi, ddchi, dot_chi, dxf, dxr, ddxf, ddxr, dot_x, inv_he, tsp, idx = _geom_tan(pjx - px + shx, pjy - py + shy, pjz - pz + shz, ddx, ddy, ddz, ve, z_p, rmin_p, j, zi_s, rho_p, j, rho_i, v_rho_p, j, v_rho_i, grid, r_cut, r_on, inv_w, inv_h, rcut2, max_doff_r2, HAS_FLOOR, HAS_RHO, HAS_VRHO, NEED_BWD2, False, B)
        pbase, k, vk = (idx * 4) * TD, tl.load(kn_p + 64 * j[:, None] + c64[None, :], mask=act[:, None], other=0.0), (tl.load(v_kn_p + 64 * j[:, None] + c64[None, :], mask=act[:, None], other=0.0) if NEED_V else tl.zeros([B, 64], tl.float32))
        db_dx, d2b_dx, th_b, sech2, ex_l, b_tan, dvp_b, eta_tan, ws, dws, ws128, wv, wt, wo, dws128, dwv, dwt, dwo = _qk_b4_dual(q, vq, k, vk, chi, dot_chi, poly_p, v_poly_p, pbase, TD, h4, tsp, inv_he, dot_x, act, m16, m8, HAS_VPOLY, B)
        gj, act32, act16, act8 = 184 * j[:, None], act[:, None], act[:, None] & m16[None, :], act[:, None] & m8[None, :]
        fs, dfs, d2fs, dfst, dvps = _hd(poly_p, v_poly_p, pbase[:, None], TD, 4 + c128[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act32, HAS_VPOLY)
        gs_j, vgs_j = tl.load(gates_p + gj + c128[None, :], mask=act32, other=0.0), (tl.load(v_gates_p + gj + c128[None, :], mask=act32, other=0.0) if HAS_VG else tl.zeros([B, 128], tl.float32))
        if NEED_JVP:
            jaz, jadsum, jas = jaz + dws, jadsum + dot_chi, jas + dws128 * fs * gs_j + ws128 * (dfst * gs_j + fs * vgs_j)
        gg_s, gvg_s = gs * gs_j, gs * vgs_j
        aw4, bw4, bf_s = tl.sum(tl.reshape(gg_s * fs, [B, 4, 32]), axis=2), tl.sum(tl.reshape(gvg_s * fs + gg_s * dfst, [B, 4, 32]), axis=2), (dws128 * gg_s + ws128 * gvg_s) if NEED_BWD2 else (ws128 * gg_s)
        dx_a, dx_b = tl.sum(ws * tl.sum(tl.reshape(gg_s * dfs, [B, 4, 32]), axis=2), axis=1), tl.sum(bf_s * dfs + (ws128 * gg_s) * (dvps + d2fs * dot_x[:, None]), axis=1)
        _acc_poly(d_poly_p, pbase[:, None], TD, 4 + c128[None, :], tsp[:, None], bf_s, (ws128 * gg_s) * dot_x[:, None] * inv_he[:, None], act32, NEED_DPOLY, NEED_BWD2)
        f_d, df_d, d2f_d, dfdt, dvpd = _hd(poly_p, v_poly_p, pbase[:, None], TD, 132 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act32, HAS_VPOLY)
        gv_j, vgv_j = tl.load(gates_p + gj + 128 + lane[None, :], mask=act32, other=0.0), (tl.load(v_gates_p + gj + 128 + lane[None, :], mask=act32, other=0.0) if HAS_VG else z32)
        dot_gvu, gvu_tan, wd, dwd = gvx * ux[:, None] + gvy * uy[:, None] + gvz * uz[:, None], gvx * dux[:, None] + gvy * duy[:, None] + gvz * duz[:, None], f_d * gv_j, dfdt * gv_j + f_d * vgv_j
        if LATER:
            bx, by, bz = _ld3(vec_p, 96 * j[:, None] + 3 * lane[None, :], act32)
            vbx, vby, vbz = _ld3(v_vec_p, 96 * j[:, None] + 3 * lane[None, :], act32) if HAS_VV else (z32, z32, z32)
            f_tr, df_tr, d2f_tr, dftrt, dvptr = _hd(poly_p, v_poly_p, pbase[:, None], TD, 180 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act32, HAS_VPOLY)
            f_ln, df_ln, d2f_ln, dflnt, dvpln = _hd(poly_p, v_poly_p, pbase[:, None], TD, 212 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act32, HAS_VPOLY)
            dbu, ddbu, dgvb, dvgvb = bx * ux[:, None] + by * uy[:, None] + bz * uz[:, None], vbx * ux[:, None] + vby * uy[:, None] + vbz * uz[:, None] + bx * dux[:, None] + by * duy[:, None] + bz * duz[:, None], gvx * bx + gvy * by + gvz * bz, gvx * vbx + gvy * vby + gvz * vbz
            w_ln, dw_ln, gvl = f_ln * dbu, dflnt * dbu + f_ln * ddbu, dot_gvu + glon
            sc_v, dsc_v, st_v, dst_v = wv * (wd + w_ln), dwv * (wd + w_ln) + wv * (dwd + dw_ln), wv * gvl * f_ln, (dwv * gvl + wv * gvu_tan) * f_ln + wv * gvl * dflnt
            if NEED_JVP:
                javx, javy, javz, jalon = javx + dsc_v * ux[:, None] + sc_v * dux[:, None] + (dwv * f_tr + wv * dftrt) * bx + (wv * f_tr) * vbx, javy + dsc_v * uy[:, None] + sc_v * duy[:, None] + (dwv * f_tr + wv * dftrt) * by + (wv * f_tr) * vby, javz + dsc_v * uz[:, None] + sc_v * duz[:, None] + (dwv * f_tr + wv * dftrt) * bz + (wv * f_tr) * vbz, jalon + dwv * w_ln + wv * dw_ln
            aw_v, bw_v, ax_v = (wd + w_ln) * dot_gvu + f_tr * dgvb + w_ln * glon, (dwd + dw_ln) * dot_gvu + (wd + w_ln) * gvu_tan + dftrt * dgvb + f_tr * dvgvb + dw_ln * glon, (wv * dot_gvu * gv_j) * df_d + (wv * gvl * dbu) * df_ln + (wv * dgvb) * df_tr
            bf_d, bf_tr, bf_ln = ((dwv * dot_gvu + wv * gvu_tan) * gv_j + wv * dot_gvu * vgv_j) if NEED_BWD2 else (wv * dot_gvu * gv_j), (dwv * dgvb + wv * dvgvb) if NEED_BWD2 else (wv * dgvb), ((dwv * gvl + wv * gvu_tan) * dbu + (wv * gvl) * ddbu) if NEED_BWD2 else (wv * gvl * dbu)
            dx_v_b = bf_d * df_d + (wv * dot_gvu * gv_j) * (dvpd + d2f_d * dot_x[:, None]) + bf_tr * df_tr + (wv * dgvb) * (dvptr + d2f_tr * dot_x[:, None]) + bf_ln * df_ln + (wv * gvl * dbu) * (dvpln + d2f_ln * dot_x[:, None])
            cux, cuy, cuz, bux, buy, buz = tl.sum(sc_v * gvx + st_v * bx, axis=1), tl.sum(sc_v * gvy + st_v * by, axis=1), tl.sum(sc_v * gvz + st_v * bz, axis=1), tl.sum(dsc_v * gvx + dst_v * bx + st_v * vbx, axis=1), tl.sum(dsc_v * gvy + dst_v * by + st_v * vby, axis=1), tl.sum(dsc_v * gvz + dst_v * bz + st_v * vbz, axis=1)
            _acc_poly(d_poly_p, pbase[:, None], TD, 132 + lane[None, :], tsp[:, None], bf_d, (wv * dot_gvu * gv_j) * dot_x[:, None] * inv_he[:, None], act32, NEED_DPOLY, NEED_BWD2)
            _acc_poly(d_poly_p, pbase[:, None], TD, 180 + lane[None, :], tsp[:, None], bf_tr, (wv * dgvb) * dot_x[:, None] * inv_he[:, None], act32, NEED_DPOLY, NEED_BWD2)
            _acc_poly(d_poly_p, pbase[:, None], TD, 212 + lane[None, :], tsp[:, None], bf_ln, (wv * gvl * dbu) * dot_x[:, None] * inv_he[:, None], act32, NEED_DPOLY, NEED_BWD2)
        else:
            sc_v, dsc_v = wv * wd, dwv * wd + wv * dwd
            if NEED_JVP:
                javx, javy, javz = javx + dsc_v * ux[:, None] + sc_v * dux[:, None], javy + dsc_v * uy[:, None] + sc_v * duy[:, None], javz + dsc_v * uz[:, None] + sc_v * duz[:, None]
            aw_v, bw_v, ax_v, bf_d = wd * dot_gvu, dwd * dot_gvu + wd * gvu_tan, (wv * dot_gvu * gv_j) * df_d, ((dwv * dot_gvu + wv * gvu_tan) * gv_j + wv * dot_gvu * vgv_j) if NEED_BWD2 else (wv * dot_gvu * gv_j)
            dx_v_b, cux, cuy, cuz, bux, buy, buz = bf_d * df_d + (wv * dot_gvu * gv_j) * (dvpd + d2f_d * dot_x[:, None]), tl.sum(sc_v * gvx, axis=1), tl.sum(sc_v * gvy, axis=1), tl.sum(sc_v * gvz, axis=1), tl.sum(dsc_v * gvx, axis=1), tl.sum(dsc_v * gvy, axis=1), tl.sum(dsc_v * gvz, axis=1)
            _acc_poly(d_poly_p, pbase[:, None], TD, 132 + lane[None, :], tsp[:, None], bf_d, (wv * dot_gvu * gv_j) * dot_x[:, None] * inv_he[:, None], act32, NEED_DPOLY, NEED_BWD2)
        aw4, bw4, dx_a, dx_b = aw4 + tl.sum(tl.reshape(aw_v, [B, 4, 8]), axis=2), bw4 + tl.sum(tl.reshape(bw_v, [B, 4, 8]), axis=2), dx_a + tl.sum(ax_v, axis=1), dx_b + tl.sum(dx_v_b, axis=1)
        y20, y21, y22, y23, y24, y30, y31, y32, y33, y34, y35, y36 = _eval_y2_y3(ux, uy, uz)
        dy20, dy21, dy22, dy23, dy24, dy30, dy31, dy32, dy33, dy34, dy35, dy36 = _y2_y3_jvp(ux, uy, uz, dux, duy, duz)
        f_t, df_t, d2f_t, dftt, dvpt = _hd(poly_p, v_poly_p, pbase[:, None], TD, 164 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act16, HAS_VPOLY)
        gt_j, vgt_j = tl.load(gates_p + gj + 160 + lane[None, :], mask=act16, other=0.0), (tl.load(v_gates_p + gj + 160 + lane[None, :], mask=act16, other=0.0) if HAS_VG else z32)
        gt0, gt1, gt2, gt3, gt4 = _ld5(g_num_p, gi + OT + 5 * lane[None, :], act16)
        dot_gt, gt_tan, wten, dwten = _d5(gt0, gt1, gt2, gt3, gt4, y20[:, None], y21[:, None], y22[:, None], y23[:, None], y24[:, None]), _d5(gt0, gt1, gt2, gt3, gt4, dy20[:, None], dy21[:, None], dy22[:, None], dy23[:, None], dy24[:, None]), f_t * gt_j, dftt * gt_j + f_t * vgt_j
        if HAS_TEN:
            glon2 = tl.load(g_num_p + gi + 336 + lane[None, :], mask=act16, other=0.0)
            t0, t1, t2_, t3, t4 = _ld5(ten_p, 80 * j[:, None] + 5 * lane[None, :], act16)
            vt0, vt1, vt2, vt3, vt4 = _ld5(v_ten_p, 80 * j[:, None] + 5 * lane[None, :], act16) if HAS_VT else (z32, z32, z32, z32, z32)
            f_tr2, df_tr2, d2f_tr2, dftr2t, dvptr2 = _hd(poly_p, v_poly_p, pbase[:, None], TD, 244 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act16, HAS_VPOLY)
            f_ln2, df_ln2, d2f_ln2, dfln2t, dvpln2 = _hd(poly_p, v_poly_p, pbase[:, None], TD, 260 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act16, HAS_VPOLY)
            dty, ddty = _d5(t0, t1, t2_, t3, t4, y20[:, None], y21[:, None], y22[:, None], y23[:, None], y24[:, None]), _d5(vt0, vt1, vt2, vt3, vt4, y20[:, None], y21[:, None], y22[:, None], y23[:, None], y24[:, None]) + _d5(t0, t1, t2_, t3, t4, dy20[:, None], dy21[:, None], dy22[:, None], dy23[:, None], dy24[:, None])
            dgtt, dvgtt, w_ln2, dw_ln2, gvl2 = _d5(gt0, gt1, gt2, gt3, gt4, t0, t1, t2_, t3, t4), _d5(gt0, gt1, gt2, gt3, gt4, vt0, vt1, vt2, vt3, vt4), f_ln2 * dty, dfln2t * dty + f_ln2 * ddty, dot_gt + glon2
            sc_t, dsc_t, st_t, dst_t = wt * (wten + w_ln2), dwt * (wten + w_ln2) + wt * (dwten + dw_ln2), wt * gvl2 * f_ln2, (dwt * gvl2 + wt * gt_tan) * f_ln2 + wt * gvl2 * dfln2t
            if NEED_JVP:
                jat0, jat1, jat2, jat3, jat4 = jat0 + dsc_t * y20[:, None] + sc_t * dy20[:, None] + (dwt * f_tr2 + wt * dftr2t) * t0 + (wt * f_tr2) * vt0, jat1 + dsc_t * y21[:, None] + sc_t * dy21[:, None] + (dwt * f_tr2 + wt * dftr2t) * t1 + (wt * f_tr2) * vt1, jat2 + dsc_t * y22[:, None] + sc_t * dy22[:, None] + (dwt * f_tr2 + wt * dftr2t) * t2_ + (wt * f_tr2) * vt2, jat3 + dsc_t * y23[:, None] + sc_t * dy23[:, None] + (dwt * f_tr2 + wt * dftr2t) * t3 + (wt * f_tr2) * vt3, jat4 + dsc_t * y24[:, None] + sc_t * dy24[:, None] + (dwt * f_tr2 + wt * dftr2t) * t4 + (wt * f_tr2) * vt4
                jalon2 = jalon2 + dwt * w_ln2 + wt * dw_ln2
            aw_t, bw_t = (wten + w_ln2) * dot_gt + f_tr2 * dgtt + w_ln2 * glon2, (dwten + dw_ln2) * dot_gt + (wten + w_ln2) * gt_tan + dftr2t * dgtt + f_tr2 * dvgtt + dw_ln2 * glon2
            dx_a = dx_a + tl.sum((wt * dot_gt * gt_j) * df_t + (wt * gvl2 * dty) * df_ln2 + (wt * dgtt) * df_tr2, axis=1)
            bf_t, bf_tr2, bf_ln2 = ((dwt * dot_gt + wt * gt_tan) * gt_j + wt * dot_gt * vgt_j) if NEED_BWD2 else (wt * dot_gt * gt_j), (dwt * dgtt + wt * dvgtt) if NEED_BWD2 else (wt * dgtt), ((dwt * gvl2 + wt * gt_tan) * dty + (wt * gvl2) * ddty) if NEED_BWD2 else (wt * gvl2 * dty)
            dx_b = dx_b + tl.sum(bf_t * df_t + (wt * dot_gt * gt_j) * (dvpt + d2f_t * dot_x[:, None]) + bf_tr2 * df_tr2 + (wt * dgtt) * (dvptr2 + d2f_tr2 * dot_x[:, None]) + bf_ln2 * df_ln2 + (wt * gvl2 * dty) * (dvpln2 + d2f_ln2 * dot_x[:, None]), axis=1)
            cy20, cy21, cy22, cy23, cy24 = _sum5(sc_t, gt0, gt1, gt2, gt3, gt4, st_t, t0, t1, t2_, t3, t4, z32, z32, z32, z32, z32, z32)
            by20, by21, by22, by23, by24 = _sum5(dsc_t, gt0, gt1, gt2, gt3, gt4, dst_t, t0, t1, t2_, t3, t4, st_t, vt0, vt1, vt2, vt3, vt4)
            _acc_poly(d_poly_p, pbase[:, None], TD, 164 + lane[None, :], tsp[:, None], bf_t, (wt * dot_gt * gt_j) * dot_x[:, None] * inv_he[:, None], act16, NEED_DPOLY, NEED_BWD2)
            _acc_poly(d_poly_p, pbase[:, None], TD, 244 + lane[None, :], tsp[:, None], bf_tr2, (wt * dgtt) * dot_x[:, None] * inv_he[:, None], act16, NEED_DPOLY, NEED_BWD2)
            _acc_poly(d_poly_p, pbase[:, None], TD, 260 + lane[None, :], tsp[:, None], bf_ln2, (wt * gvl2 * dty) * dot_x[:, None] * inv_he[:, None], act16, NEED_DPOLY, NEED_BWD2)
        else:
            sc_t, dsc_t = wt * wten, dwt * wten + wt * dwten
            if NEED_JVP:
                jat0, jat1, jat2, jat3, jat4 = jat0 + dsc_t * y20[:, None] + sc_t * dy20[:, None], jat1 + dsc_t * y21[:, None] + sc_t * dy21[:, None], jat2 + dsc_t * y22[:, None] + sc_t * dy22[:, None], jat3 + dsc_t * y23[:, None] + sc_t * dy23[:, None], jat4 + dsc_t * y24[:, None] + sc_t * dy24[:, None]
            aw_t, bw_t, dx_a, bf_t = wten * dot_gt, dwten * dot_gt + wten * gt_tan, dx_a + tl.sum((wt * dot_gt * gt_j) * df_t, axis=1), ((dwt * dot_gt + wt * gt_tan) * gt_j + wt * dot_gt * vgt_j) if NEED_BWD2 else (wt * dot_gt * gt_j)
            dx_b += tl.sum(bf_t * df_t + (wt * dot_gt * gt_j) * (dvpt + d2f_t * dot_x[:, None]), axis=1)
            cy20, cy21, cy22, cy23, cy24 = _sum5(sc_t, gt0, gt1, gt2, gt3, gt4, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32)
            by20, by21, by22, by23, by24 = _sum5(dsc_t, gt0, gt1, gt2, gt3, gt4, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32)
            _acc_poly(d_poly_p, pbase[:, None], TD, 164 + lane[None, :], tsp[:, None], bf_t, (wt * dot_gt * gt_j) * dot_x[:, None] * inv_he[:, None], act16, NEED_DPOLY, NEED_BWD2)
        aw4, bw4 = aw4 + tl.sum(tl.sum(tl.reshape(aw_t, [B, 2, 4, 4]), axis=3), axis=1), bw4 + tl.sum(tl.sum(tl.reshape(bw_t, [B, 2, 4, 4]), axis=3), axis=1)
        f_o, df_o, d2f_o, dfot, dvpo = _hd(poly_p, v_poly_p, pbase[:, None], TD, FO + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act8, HAS_VPOLY)
        go_j, vgo_j = tl.load(gates_p + gj + 176 + lane[None, :], mask=act8, other=0.0), (tl.load(v_gates_p + gj + 176 + lane[None, :], mask=act8, other=0.0) if HAS_VG else z32)
        go0, go1, go2, go3, go4, go5, go6 = _ld7(g_num_p, gi + OO + 7 * lane[None, :], act8)
        dot_go, go_tan = _d7(go0, go1, go2, go3, go4, go5, go6, y30[:, None], y31[:, None], y32[:, None], y33[:, None], y34[:, None], y35[:, None], y36[:, None]), _d7(go0, go1, go2, go3, go4, go5, go6, dy30[:, None], dy31[:, None], dy32[:, None], dy33[:, None], dy34[:, None], dy35[:, None], dy36[:, None])
        wo_f, dwo_f = f_o * go_j, dfot * go_j + f_o * vgo_j
        sco, dsco = wo * wo_f, dwo * wo_f + wo * dwo_f
        if NEED_JVP:
            jao0, jao1, jao2, jao3, jao4, jao5, jao6 = jao0 + dsco * y30[:, None] + sco * dy30[:, None], jao1 + dsco * y31[:, None] + sco * dy31[:, None], jao2 + dsco * y32[:, None] + sco * dy32[:, None], jao3 + dsco * y33[:, None] + sco * dy33[:, None], jao4 + dsco * y34[:, None] + sco * dy34[:, None], jao5 + dsco * y35[:, None] + sco * dy35[:, None], jao6 + dsco * y36[:, None] + sco * dy36[:, None]
        bf_o = ((dwo * dot_go + wo * go_tan) * go_j + wo * dot_go * vgo_j) if NEED_BWD2 else (wo * dot_go * go_j)
        aw4, bw4 = aw4 + tl.sum(tl.sum(tl.reshape(wo_f * dot_go, [B, 4, 4, 2]), axis=3), axis=1), bw4 + tl.sum(tl.sum(tl.reshape(dwo_f * dot_go + wo_f * go_tan, [B, 4, 4, 2]), axis=3), axis=1)
        dx_a, dx_b = dx_a + tl.sum((wo * dot_go * go_j) * df_o, axis=1), dx_b + tl.sum(bf_o * df_o + (wo * dot_go * go_j) * (dvpo + d2f_o * dot_x[:, None]), axis=1)
        _acc_poly(d_poly_p, pbase[:, None], TD, FO + lane[None, :], tsp[:, None], bf_o, (wo * dot_go * go_j) * dot_x[:, None] * inv_he[:, None], act8, NEED_DPOLY, NEED_BWD2)
        aw_tot, bw_tot = tl.where(act[:, None], aw4 + gzc, 0.0), tl.where(act[:, None], bw4, 0.0)
        beta_h, alpha_h = aw_tot * ws, aw_tot * dws + bw_tot * ws
        d_braw, d_btan = (alpha_h * sech2 - beta_h * (th_b * sech2 * b_tan)) if NEED_BWD2 else (beta_h * sech2), beta_h * sech2
        _acc_poly(d_poly_p, pbase[:, None], TD, h4[None, :], tsp[:, None], d_braw, d_btan * dot_x[:, None] * inv_he[:, None], act[:, None], NEED_DPOLY, NEED_BWD2)
        if NEED_BWD2:
            cy30, cy31, cy32, cy33, cy34, cy35, cy36 = _sum7(sco, go0, go1, go2, go3, go4, go5, go6)
            cux, cuy, cuz = _vjp_y2_y3(ux, uy, uz, cux, cuy, cuz, cy20, cy21, cy22, cy23, cy24, cy30, cy31, cy32, cy33, cy34, cy35, cy36)
            dco0, dco1, dco2, dco3, dco4, dco5, dco6 = _sum7(dsco, go0, go1, go2, go3, go4, go5, go6)
            vyx, vyy, vyz = _vjp_y2_y3(ux, uy, uz, bux, buy, buz, by20, by21, by22, by23, by24, dco0, dco1, dco2, dco3, dco4, dco5, dco6)
            bux, buy, buz = _y2_y3_hess_u(ux, uy, uz, dux, duy, duz, vyx, vyy, vyz, cy20, cy21, cy22, cy23, cy24, cy30, cy31, cy32, cy33, cy34, cy35, cy36)
            dq = dq + tl.reshape(tl.reshape(alpha_h, [B, 4, 1]) * tl.reshape(k, [B, 4, 16]) + tl.reshape(beta_h, [B, 4, 1]) * tl.reshape(vk, [B, 4, 16]), [B, 64])
            d_x_a, d_x_b, d_chi_a, d_chi_b = dx_a + tl.sum(d_btan * db_dx, axis=1), dx_b + tl.sum(d_braw * db_dx + d_btan * (dvp_b + d2b_dx * dot_x[:, None]), axis=1), tl.where(act, gdsum_i + tl.sum(aw_tot * ex_l, axis=1), 0.0), tl.where(act, tl.sum((aw_tot * eta_tan + bw_tot) * ex_l, axis=1), 0.0)
            d_rfl_a, d_rfl_b, dgu_c, dgu_b, dot_r, du_c = d_chi_a * dchi + d_x_a * dxf, d_chi_b * dchi + d_chi_a * (ddchi * drfl_tan) + d_x_b * dxf + d_x_a * ddxf, cux * ux + cuy * uy + cuz * uz, bux * ux + buy * uy + buz * uz, ux * ddx + uy * ddy + uz * ddz, dux * cux + duy * cuy + duz * cuz
            gdx = tl.where(act, (-dot_r * inv_r * ((cux - dgu_c * ux) * inv_r) - (du_c * ux + dgu_c * dux) * inv_r) + (bux - dgu_b * ux) * inv_r + d_rfl_b * drx + d_rfl_a * ddrx, 0.0)
            gdy = tl.where(act, (-dot_r * inv_r * ((cuy - dgu_c * uy) * inv_r) - (du_c * uy + dgu_c * duy) * inv_r) + (buy - dgu_b * uy) * inv_r + d_rfl_b * dry + d_rfl_a * ddry, 0.0)
            gdz = tl.where(act, (-dot_r * inv_r * ((cuz - dgu_c * uz) * inv_r) - (du_c * uz + dgu_c * duz) * inv_r) + (buz - dgu_b * uz) * inv_r + d_rfl_b * drz + d_rfl_a * ddrz, 0.0)
            dpos_x, dpos_y, dpos_z, drho_e = dpos_x - gdx, dpos_y - gdy, dpos_z - gdz, (tl.where(act, d_x_b * dxr + d_x_a * ddxr, 0.0) if NEED_DRHO else z0)
            drho_i = drho_i + drho_e
            if NEED_VIR:
                v00, v01, v02, v10, v11, v12, v20, v21, v22 = v00 + esx * gdx, v01 + esx * gdy, v02 + esx * gdz, v10 + esy * gdx, v11 + esy * gdy, v12 + esy * gdz, v20 + esz * gdx, v21 + esz * gdy, v22 + esz * gdz
            _st3(eg16_p, 16 * e, gdx, gdy, gdz, ve); tl.store(eg16_p + 16 * e + 3, drho_e, mask=ve)
            tl.store(eg16_p + 16 * e[:, None] + 4 + h4[None, :], alpha_h, mask=ve[:, None]); tl.store(eg16_p + 16 * e[:, None] + 8 + h4[None, :], beta_h, mask=ve[:, None])
    if NEED_JVP:
        ob, m16b, m8b = OD * i[:, None], am[:, None] & m16[None, :], am[:, None] & m8[None, :]
        tl.store(d_gnum_p + ob + c128[None, :], jas, mask=m32); _st3(d_gnum_p, ob + 128 + 3 * lane[None, :], javx, javy, javz, m32)
        if LATER: tl.store(d_gnum_p + ob + 224 + lane[None, :], jalon, mask=m32); tl.store(d_gnum_p + ob + 336 + lane[None, :], jalon2, mask=m16b)
        _st5(d_gnum_p, ob + OT + 5 * lane[None, :], jat0, jat1, jat2, jat3, jat4, m16b); _st7(d_gnum_p, ob + OO + 7 * lane[None, :], jao0, jao1, jao2, jao3, jao4, jao5, jao6, m8b)
        tl.store(d_gzc_p + 4 * i[:, None] + h4[None, :], jaz, mask=m32); tl.store(d_gdsum_p + i, jadsum, mask=am)
    if NEED_BWD2:
        _st3(d_pos_p, 3 * i, dpos_x, dpos_y, dpos_z, am); tl.store(d_qn_p + 64 * i[:, None] + c64[None, :], dq, mask=am[:, None])
        if NEED_DRHO: tl.store(d_rho_p + i, drho_i, mask=am)
        if NEED_VIR: _av3(d_cell_p, 9 * g_id, 0, v00, v01, v02, am, IS_G1); _av3(d_cell_p, 9 * g_id, 3, v10, v11, v12, am, IS_G1); _av3(d_cell_p, 9 * g_id, 6, v20, v21, v22, am, IS_G1)


@_jit
def _triton_csr_bwd2_send(pos_p, sptr_p, sperm_p, ridx_p, sh_p, cell_p, grp_p, z_p, rmin_p, rho_p, poly_p, qn_p, kn_p, g_num_p, v_pos_p, v_poly_p, v_qn_p, v_kn_p, v_rho_p, v_cell_p, eg16_p, d_pos_p, d_gates_p, d_kn_p, d_vec_p, d_ten_p, d_rho_p, n_atoms, grid, r_cut, r_on, max_doff_r2, LATER: tl.constexpr, HAS_FLOOR: tl.constexpr, HAS_RHO: tl.constexpr, HAS_TEN: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr, HAS_VPOLY: tl.constexpr, HAS_VRHO: tl.constexpr, HAS_VCELL_EFF: tl.constexpr, NEED_DRHO: tl.constexpr, B: tl.constexpr):
    j, lane, c64, c128, h4 = tl.program_id(0) * B + tl.arange(0, B), tl.arange(0, 32), tl.arange(0, 64), tl.arange(0, 128), tl.arange(0, 4)
    am, m16, m8 = j < n_atoms, lane < 16, lane < 8
    TD: tl.constexpr = 284 if LATER else 188
    OD: tl.constexpr = 408 if LATER else 360
    FO: tl.constexpr = 276 if LATER else 180
    OT: tl.constexpr = 256 if LATER else 224
    OO: tl.constexpr = 352 if LATER else 304
    zj, rho_j, v_rho_j = (tl.minimum(tl.maximum(tl.load(z_p + j, mask=am, other=0).to(tl.int32), 0), 127) if HAS_FLOOR else tl.zeros([B], tl.int32)), (tl.load(rho_p + j, mask=am, other=0.0) if HAS_RHO else tl.zeros([B], tl.float32)), (tl.load(v_rho_p + j, mask=am, other=0.0) if (HAS_RHO and HAS_VRHO) else tl.zeros([B], tl.float32))
    pjx, pjy, pjz = _ld3(pos_p, 3 * j, am); vpjx, vpjy, vpjz = _ld3(v_pos_p, 3 * j, am)
    k, vk, z0, z32 = tl.load(kn_p + 64 * j[:, None] + c64[None, :], mask=am[:, None], other=0.0), tl.load(v_kn_p + 64 * j[:, None] + c64[None, :], mask=am[:, None], other=0.0), tl.zeros([B], tl.float32), tl.zeros([B, 32], tl.float32)
    dpos_jx, dpos_jy, dpos_jz, drho_j, dk, dgs = z0, z0, z0, z0, tl.zeros([B, 64], tl.float32), tl.zeros([B, 128], tl.float32)
    dgv, dgt, dgo, dbx, dby, dbz, dt0, dt1, dt2, dt3, dt4 = z32, z32, z32, z32, z32, z32, z32, z32, z32, z32, z32
    inv_w, inv_h, rcut2, s_beg, s_end = 1.0 / (r_cut - r_on), grid / r_cut, r_cut * r_cut, tl.load(sptr_p + j, mask=am, other=0), tl.load(sptr_p + j + 1, mask=am, other=0)
    for step in range(tl.max(tl.maximum(s_end - s_beg, 0), axis=0)):
        kp = s_beg + step; ve = am & (kp < s_end); e = tl.load(sperm_p + kp, mask=ve, other=0); i = tl.load(ridx_p + e, mask=ve, other=0)
        g_id = tl.zeros([B], tl.int32) if (not HAS_CELL or IS_G1) else tl.load(grp_p + i, mask=ve, other=0).to(tl.int32)
        shx, shy, shz, _, _, _ = _nlh_shift(sh_p, cell_p, e, g_id, ve, HAS_CELL)
        vshx, vshy, vshz, _, _, _ = _nlh_shift(sh_p, v_cell_p, e, g_id, ve, True) if HAS_VCELL_EFF else (z0, z0, z0, z0, z0, z0)
        pix, piy, piz = _ld3(pos_p, 3 * i, ve); vpix, vpiy, vpiz = _ld3(v_pos_p, 3 * i, ve)
        act, _, ux, uy, uz, dux, duy, duz, _, _, _, _, _, _, _, chi, _, _, dot_chi, _, _, _, _, dot_x, inv_he, tsp, idx = _geom_tan(pjx - pix + shx, pjy - piy + shy, pjz - piz + shz, vpjx - vpix + vshx, vpjy - vpiy + vshy, vpjz - vpiz + vshz, ve, z_p, rmin_p, i, zj, rho_p, i, rho_j, v_rho_p, i, v_rho_j, grid, r_cut, r_on, inv_w, inv_h, rcut2, max_doff_r2, HAS_FLOOR, HAS_RHO, HAS_VRHO, False, True, B)
        pbase = (idx * 4) * TD; egx, egy, egz = _ld3(eg16_p, 16 * e, act); dpos_jx, dpos_jy, dpos_jz = dpos_jx + egx, dpos_jy + egy, dpos_jz + egz
        if NEED_DRHO: drho_j = drho_j + tl.load(eg16_p + 16 * e + 3, mask=act, other=0.0)
        q, vq = tl.load(qn_p + 64 * i[:, None] + c64[None, :], mask=act[:, None], other=0.0), tl.load(v_qn_p + 64 * i[:, None] + c64[None, :], mask=act[:, None], other=0.0)
        alpha_h, beta_h = tl.load(eg16_p + 16 * e[:, None] + 4 + h4[None, :], mask=act[:, None], other=0.0), tl.load(eg16_p + 16 * e[:, None] + 8 + h4[None, :], mask=act[:, None], other=0.0)
        dk = dk + tl.reshape(tl.reshape(alpha_h, [B, 4, 1]) * tl.reshape(q, [B, 4, 16]) + tl.reshape(beta_h, [B, 4, 1]) * tl.reshape(vq, [B, 4, 16]), [B, 64])
        _, _, _, _, _, _, _, _, ws, dws, ws128, wv, wt, wo, dws128, dwv, dwt, dwo = _qk_b4_dual(q, vq, k, vk, chi, dot_chi, poly_p, v_poly_p, pbase, TD, h4, tsp, inv_he, dot_x, act, m16, m8, HAS_VPOLY, B)
        gi, act32, act16, act8 = OD * i[:, None], act[:, None], act[:, None] & m16[None, :], act[:, None] & m8[None, :]
        fs, _, _, dfst, _ = _hd(poly_p, v_poly_p, pbase[:, None], TD, 4 + c128[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act32, HAS_VPOLY)
        dgs = dgs + tl.load(g_num_p + gi + c128[None, :], mask=act32, other=0.0) * (dws128 * fs + ws128 * dfst)
        gvx, gvy, gvz = _ld3(g_num_p, gi + 128 + 3 * lane[None, :], act32)
        dot_gvu, gvu_tan = gvx * ux[:, None] + gvy * uy[:, None] + gvz * uz[:, None], gvx * dux[:, None] + gvy * duy[:, None] + gvz * duz[:, None]
        fd, _, _, dfdt, _ = _hd(poly_p, v_poly_p, pbase[:, None], TD, 132 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act32, HAS_VPOLY)
        dgv = dgv + (dwv * dot_gvu + wv * gvu_tan) * fd + (wv * dot_gvu) * dfdt
        if LATER:
            glon = tl.load(g_num_p + gi + 224 + lane[None, :], mask=act32, other=0.0)
            ftr, _, _, dftrt, _ = _hd(poly_p, v_poly_p, pbase[:, None], TD, 180 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act32, HAS_VPOLY)
            fln, _, _, dflnt, _ = _hd(poly_p, v_poly_p, pbase[:, None], TD, 212 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act32, HAS_VPOLY)
            st, s1, sl_d = dwv * ftr + wv * dftrt, (dwv * (dot_gvu + glon) + wv * gvu_tan) * fln + wv * (dot_gvu + glon) * dflnt, wv * (dot_gvu + glon) * fln
            dbx, dby, dbz = dbx + st * gvx + s1 * ux[:, None] + sl_d * dux[:, None], dby + st * gvy + s1 * uy[:, None] + sl_d * duy[:, None], dbz + st * gvz + s1 * uz[:, None] + sl_d * duz[:, None]
        y20, y21, y22, y23, y24, y30, y31, y32, y33, y34, y35, y36 = _eval_y2_y3(ux, uy, uz)
        dy20, dy21, dy22, dy23, dy24, dy30, dy31, dy32, dy33, dy34, dy35, dy36 = _y2_y3_jvp(ux, uy, uz, dux, duy, duz)
        gt0, gt1, gt2, gt3, gt4 = _ld5(g_num_p, gi + OT + 5 * lane[None, :], act16)
        dot_gt, gt_tan = _d5(gt0, gt1, gt2, gt3, gt4, y20[:, None], y21[:, None], y22[:, None], y23[:, None], y24[:, None]), _d5(gt0, gt1, gt2, gt3, gt4, dy20[:, None], dy21[:, None], dy22[:, None], dy23[:, None], dy24[:, None])
        ft, _, _, dftt, _ = _hd(poly_p, v_poly_p, pbase[:, None], TD, 164 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act16, HAS_VPOLY)
        dgt = dgt + (dwt * dot_gt + wt * gt_tan) * ft + (wt * dot_gt) * dftt
        if HAS_TEN:
            glon2 = tl.load(g_num_p + gi + 336 + lane[None, :], mask=act16, other=0.0)
            ftr2, _, _, dftr2t, _ = _hd(poly_p, v_poly_p, pbase[:, None], TD, 244 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act16, HAS_VPOLY)
            fln2, _, _, dfln2t, _ = _hd(poly_p, v_poly_p, pbase[:, None], TD, 260 + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act16, HAS_VPOLY)
            st2, s2, sl2_d = dwt * ftr2 + wt * dftr2t, (dwt * (dot_gt + glon2) + wt * gt_tan) * fln2 + wt * (dot_gt + glon2) * dfln2t, wt * (dot_gt + glon2) * fln2
            dt0, dt1, dt2, dt3, dt4 = dt0 + st2 * gt0 + s2 * y20[:, None] + sl2_d * dy20[:, None], dt1 + st2 * gt1 + s2 * y21[:, None] + sl2_d * dy21[:, None], dt2 + st2 * gt2 + s2 * y22[:, None] + sl2_d * dy22[:, None], dt3 + st2 * gt3 + s2 * y23[:, None] + sl2_d * dy23[:, None], dt4 + st2 * gt4 + s2 * y24[:, None] + sl2_d * dy24[:, None]
        go0, go1, go2, go3, go4, go5, go6 = _ld7(g_num_p, gi + OO + 7 * lane[None, :], act8)
        dot_go, go_tan = _d7(go0, go1, go2, go3, go4, go5, go6, y30[:, None], y31[:, None], y32[:, None], y33[:, None], y34[:, None], y35[:, None], y36[:, None]), _d7(go0, go1, go2, go3, go4, go5, go6, dy30[:, None], dy31[:, None], dy32[:, None], dy33[:, None], dy34[:, None], dy35[:, None], dy36[:, None])
        fo, _, _, dfot, _ = _hd(poly_p, v_poly_p, pbase[:, None], TD, FO + lane[None, :], tsp[:, None], inv_he[:, None], dot_x[:, None], act8, HAS_VPOLY)
        dgo = dgo + (dwo * dot_go + wo * go_tan) * fo + (wo * dot_go) * dfot
    rp_x, rp_y, rp_z = _ld3(d_pos_p, 3 * j, am)
    _st3(d_pos_p, 3 * j, rp_x + dpos_jx, rp_y + dpos_jy, rp_z + dpos_jz, am)
    if NEED_DRHO: tl.store(d_rho_p + j, tl.load(d_rho_p + j, mask=am, other=0.0) + drho_j, mask=am)
    dgj, m32, m16b, m8b = 184 * j[:, None], am[:, None], am[:, None] & m16[None, :], am[:, None] & m8[None, :]
    tl.store(d_gates_p + dgj + c128[None, :], dgs, mask=m32); tl.store(d_gates_p + dgj + 128 + lane[None, :], dgv, mask=m32)
    tl.store(d_gates_p + dgj + 160 + lane[None, :], dgt, mask=m16b); tl.store(d_gates_p + dgj + 176 + lane[None, :], dgo, mask=m8b)
    tl.store(d_kn_p + 64 * j[:, None] + c64[None, :], dk, mask=m32)
    if LATER:
        _st3(d_vec_p, 96 * j[:, None] + 3 * lane[None, :], dbx, dby, dbz, m32)
        if HAS_TEN: _st5(d_ten_p, 80 * j[:, None] + 5 * lane[None, :], dt0, dt1, dt2, dt3, dt4, m16b)


def triton_csr_backward_poly(plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor, cell: Tensor, graph: Tensor, z: Tensor, g_num: Tensor, g_zc: Tensor, g_dsum: Tensor, *, need_vir: bool, need_drho: bool, need_dshift: bool = False, need_dpoly: bool = True) -> tuple[Any, ...]:
    """1st-order Dual-CSR backward with optional `d_poly` (`O2`); PyTorch fallback on CPU/MPS."""
    from lam4.kernels.csr_ops import _ref_vjp_csr_backward
    from lam4.kernels.triton_csr import triton_csr_backward
    if not has_triton_csr_bwd2(pos.device) or need_dshift:
        return _ref_vjp_csr_backward(plan.recv_ptr, plan.send_idx, plan.recv_idx, plan.send_ptr, plan.send_perm, layout, cfg, rmin_table_f32(pos.device), pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z, g_num, g_zc, g_dsum, need_vir=need_vir, need_drho=need_drho, need_dshift=need_dshift, need_dpoly=need_dpoly, create_graph=torch.is_grad_enabled())
    base = triton_csr_backward(plan, layout, cfg, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z, g_num, g_zc, g_dsum, need_vir=need_vir, need_drho=need_drho, need_dshift=need_dshift)
    if not need_dpoly or plan.n_atoms == 0 or plan.n_edges == 0:
        return (base[0], torch.zeros_like(poly) if need_dpoly else None, *base[1:])
    has_ = lambda t: (t is not None and t.numel() > 0)
    dev, n, (ba, nw), sp = pos.device, plan.n_atoms, _cfg_block_warps(plan.n_atoms), _dp_splits(plan.n_atoms)
    dp_buf = torch.zeros((sp, *poly.shape), dtype=poly.dtype, device=dev)
    hf, hr, ht, hc, df, di = bool(cfg.floor and has_(z)), bool(has_(rho)), bool(layout.later and has_(ten)), bool(has_(cell)), dummy_f32(dev), dummy_i64(dev)
    hv = bool(layout.later and has_(vec))
    _triton_csr_bwd2_recv[((n + ba - 1) // ba,)](pos, plan.recv_ptr, plan.send_idx, shift_in, cell if hc else df, graph if has_(graph) else di, z if hf else di, rmin_table_f32(dev) if hf else df, rho if hr else df, poly, gates, qn, kn, vec if hv else df, ten if ht else df, g_num, g_zc, g_dsum, df, df, df, df, df, df, df, df, df, df, dp_buf, df, df, df, df, df, df, df, n, int(poly.shape[0]), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2), LATER=layout.later, HAS_FLOOR=hf, HAS_RHO=hr, HAS_TEN=ht, HAS_CELL=hc, IS_G1=bool(hc and cell.shape[0] == 1), HAS_VPOLY=False, HAS_VRHO=False, HAS_VCELL_EFF=False, HAS_VG=False, HAS_VV=False, HAS_VT=False, NEED_DPOLY=True, NEED_BWD2=False, NEED_JVP=False, NEED_V=False, NEED_VIR=False, NEED_DRHO=False, DP_SPLIT=sp, B=ba, num_warps=nw)
    return (base[0], dp_buf.sum(0) if sp > 1 else dp_buf[0], *base[1:])


def triton_csr_bwd2(plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor, cell: Tensor, graph: Tensor, z: Tensor, g_num: Tensor, g_zc: Tensor, g_dsum: Tensor, v_pos: Tensor, v_poly: Tensor, v_gates: Tensor, v_vec: Tensor, v_ten: Tensor, v_qn: Tensor, v_kn: Tensor, v_rho: Tensor, v_cell: Tensor, *, need_dpoly: bool, need_vir: bool, need_drho: bool) -> tuple[Any, ...]:
    """Analytic 2nd-order Dual-CSR double-backward (`0` `[E, C]` DRAM buffers) + exact CPU autograd fallback."""
    from lam4.kernels.csr_ops import _ref_csr_bwd2_cpu
    has_ = lambda t: (t is not None and t.numel() > 0)
    dev, n, n_e, hv, ht, hr, hc = pos.device, plan.n_atoms, plan.n_edges, bool(layout.later and has_(vec)), bool(layout.later and has_(ten)), bool(has_(rho)), bool(has_(cell))
    if n == 0 or n_e == 0:
        return (torch.zeros_like(pos), torch.zeros_like(poly) if need_dpoly else None, torch.zeros_like(gates), torch.zeros_like(vec) if layout.later else None, torch.zeros_like(ten) if ht else None, torch.zeros_like(qn), torch.zeros_like(kn), torch.zeros_like(rho) if need_drho else None, torch.zeros_like(cell) if need_vir else None, None, torch.zeros_like(g_num), torch.zeros_like(g_zc), torch.zeros_like(g_dsum))
    if not has_triton_csr_bwd2(dev):
        return _ref_csr_bwd2_cpu(plan, layout, cfg, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z, g_num, g_zc, g_dsum, v_pos, v_poly, v_gates, v_vec, v_ten, v_qn, v_kn, v_rho, v_cell, need_dpoly=need_dpoly, need_vir=need_vir, need_drho=need_drho)
    ba, nw = _bwd2_block_warps(n)
    sp = _dp_splits(n) if need_dpoly else 1
    hf, is_g1, evir, edrho = bool(cfg.floor and has_(z)), bool(hc and cell.shape[0] == 1), bool(need_vir and hc), bool(need_drho and hr)
    hvp, hvr, hvc, hvg, hvv, hvt = has_(v_poly), bool(hr and has_(v_rho)), bool(hc and has_(v_cell)), has_(v_gates), bool(hv and has_(v_vec)), bool(ht and has_(v_ten))
    df, di, grid_l, rmin = dummy_f32(dev), dummy_i64(dev), ((n + ba - 1) // ba,), (rmin_table_f32(dev) if hf else dummy_f32(dev))
    vp0, vqn0, vkn0 = (v_pos if has_(v_pos) else torch.zeros_like(pos)), (v_qn if has_(v_qn) else torch.zeros_like(qn)), (v_kn if has_(v_kn) else torch.zeros_like(kn))
    d2_pos, d2_qn, eg16, dp_buf, d2_cell, d2_rho, d2_gn, d2_gz, d2_gd = torch.empty((n, 3), dtype=torch.float32, device=dev), torch.empty_like(qn), torch.empty((n_e, 16), dtype=torch.float32, device=dev), (torch.zeros((sp, *poly.shape), dtype=poly.dtype, device=dev) if need_dpoly else df), (torch.zeros_like(cell) if evir else df), (torch.empty((n,), dtype=torch.float32, device=dev) if edrho else df), torch.empty_like(g_num), torch.empty_like(g_zc), torch.empty_like(g_dsum)
    _triton_csr_bwd2_recv[grid_l](pos, plan.recv_ptr, plan.send_idx, shift_in, cell if hc else df, graph if has_(graph) else di, z if hf else di, rmin, rho if hr else df, poly, gates, qn, kn, vec if hv else df, ten if ht else df, g_num, g_zc, g_dsum, vp0, v_poly if hvp else df, v_gates if hvg else df, vqn0, vkn0, v_vec if hvv else df, v_ten if hvt else df, v_rho if hvr else df, v_cell if hvc else df, d2_pos, dp_buf, d2_qn, eg16, d2_cell, d2_rho, d2_gn, d2_gz, d2_gd, n, int(poly.shape[0]), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2), LATER=layout.later, HAS_FLOOR=hf, HAS_RHO=hr, HAS_TEN=ht, HAS_CELL=hc, IS_G1=is_g1, HAS_VPOLY=hvp, HAS_VRHO=hvr, HAS_VCELL_EFF=hvc, HAS_VG=hvg, HAS_VV=hvv, HAS_VT=hvt, NEED_DPOLY=need_dpoly, NEED_BWD2=True, NEED_JVP=True, NEED_V=True, NEED_VIR=evir, NEED_DRHO=edrho, DP_SPLIT=sp, B=ba, num_warps=nw)
    d2_gates, d2_kn, d2_vec, d2_ten = torch.empty((n, layout.gate_dim), dtype=torch.float32, device=dev), torch.empty_like(kn), (torch.empty((n, 32, 3), dtype=torch.float32, device=dev) if layout.later else df), (torch.empty((n, 16, 5), dtype=torch.float32, device=dev) if ht else df)
    _triton_csr_bwd2_send[grid_l](pos, plan.send_ptr, plan.send_perm, plan.recv_idx, shift_in, cell if hc else df, graph if has_(graph) else di, z if hf else di, rmin, rho if hr else df, poly, qn, kn, g_num, vp0, v_poly if hvp else df, vqn0, vkn0, v_rho if hvr else df, v_cell if hvc else df, eg16, d2_pos, d2_gates, d2_kn, d2_vec, d2_ten, d2_rho, n, int(poly.shape[0]), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2), LATER=layout.later, HAS_FLOOR=hf, HAS_RHO=hr, HAS_TEN=ht, HAS_CELL=hc, IS_G1=is_g1, HAS_VPOLY=hvp, HAS_VRHO=hvr, HAS_VCELL_EFF=hvc, NEED_DRHO=edrho, B=ba, num_warps=nw)
    return (d2_pos, (dp_buf.sum(0) if sp > 1 else dp_buf[0]) if need_dpoly else None, d2_gates, d2_vec if layout.later else None, d2_ten if ht else None, d2_qn, d2_kn, d2_rho if edrho else None, d2_cell if evir else None, None, d2_gn, d2_gz, d2_gd)
