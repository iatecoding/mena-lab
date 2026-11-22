"""O3 + O4 CUDA/Triton Coulomb EFG (`NEED_F2=True`), 2nd-order `create_graph=True` backward (`triton_coulomb_bwd2`), and decoupled NLH repulsion helper (`eval_nlh_decoupled_impl`)."""
from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from lam4.core.consts import cuda_cap_ok, is_fake, proxy_tracing

try:
    import triton, triton.language as tl
    _HAS_TRITON, _jit = True, triton.jit
except ImportError:  # pragma: no cover - CPU/MPS macOS fallback
    _HAS_TRITON, tl, _jit = False, None, (lambda fn: fn)  # type: ignore[assignment]


def has_triton_coulomb_bwd2(device: torch.device | str) -> bool:
    return bool(
        os.environ.get("LAM4_TRITON_COULOMB", "1") != "0"
        and _HAS_TRITON
        and torch.device(device).type == "cuda"
        and cuda_cap_ok(device)
    )


if _HAS_TRITON and hasattr(tl, "erf"):
    @_jit
    def _erf(x):
        return tl.erf(x)
elif _HAS_TRITON and hasattr(getattr(tl, "math", None), "erf"):
    @_jit
    def _erf(x):
        return tl.math.erf(x)
else:
    @_jit
    def _erf(a):
        x = tl.abs(a)
        u, t = x * x, 1.0 / (0.5 * x + 1.0)
        p1 = ((((((6.245970226e-5 * u - 7.440516159e-4) * u + 5.111238764e-3) * u - 2.680452797e-2) * u + 1.128210157e-1) * u - 3.761244680e-1) * u + 1.283791142e-1)
        p2 = ((((((((3.475873057e-2 * t - 1.453217770e-2) * t - 2.472636555e-1) * t + 3.508501074e-1) * t + 2.135339677e-2) * t + 3.011171738e-1) * t + 2.705037481e-1) * t + 2.834916809e-1) * t - 7.315009595e-5)
        r = tl.where(x < 1.25, p1 * x + x, tl.where(x < 3.95, 1.0 - p2 * tl.exp(-u), 1.0))
        return tl.where(a < 0.0, -r, r)


@_jit
def _is_far_tile(bsph_ptr, b, tcx, tcy, tcz, rtg, far_pad, IS_FAST64: tl.constexpr):
    if IS_FAST64:
        tbx = tl.load(bsph_ptr + b * 4 + 0)
        tby = tl.load(bsph_ptr + b * 4 + 1)
        tbz = tl.load(bsph_ptr + b * 4 + 2)
        rtb = tl.load(bsph_ptr + b * 4 + 3)
        thr = rtg + far_pad + rtb
        far = ((tcx - tbx) * (tcx - tbx) + (tcy - tby) * (tcy - tby) + (tcz - tbz) * (tcz - tbz) > thr * thr) & (rtb >= 0.0) & (rtg >= 0.0)
    else:
        far = False
    return far


@_jit
def _ld3(ptr, offs, mask):
    return tl.load(ptr + offs * 3, mask=mask, other=0.0), tl.load(ptr + offs * 3 + 1, mask=mask, other=0.0), tl.load(ptr + offs * 3 + 2, mask=mask, other=0.0)


@_jit
def _st3(ptr, offs, mask, a, b, c):
    tl.store(ptr + offs * 3, a, mask=mask); tl.store(ptr + offs * 3 + 1, b, mask=mask); tl.store(ptr + offs * 3 + 2, c, mask=mask)


@_jit
def _smeared_k_f1_f2_f3(r2, inv_sig, c0, inv_sig2, NEED_F2: tl.constexpr, NEED_F3: tl.constexpr):
    u2 = r2 * inv_sig2
    small = u2 < 0.1
    r2_safe = tl.where(small, 1.0, r2)
    inv_r = tl.rsqrt(r2_safe)
    inv_r2 = inv_r * inv_r
    u = r2_safe * inv_r * inv_sig
    exp_u2 = tl.exp(-u2)
    k_g = _erf(u) * inv_r
    f1_g = (k_g - c0 * exp_u2) * inv_r2
    k = tl.where(small, c0 * (1.0 - u2 * (1.0 / 3.0 - u2 * (0.1 - u2 * (1.0 / 42.0 - u2 / 216.0)))), k_g)
    f1 = tl.where(small, c0 * inv_sig2 * ((2.0 / 3.0) + u2 * (-0.4 + u2 * ((1.0 / 7.0) + u2 * (-(1.0 / 27.0) + u2 / 132.0)))), f1_g)
    inv_sig4 = inv_sig2 * inv_sig2
    f2_g = (2.0 * c0 * inv_sig2 * exp_u2 - 3.0 * f1_g) * inv_r2 if NEED_F2 else f1 * 0.0
    f2 = tl.where(small, c0 * inv_sig4 * (-0.8 + u2 * ((4.0 / 7.0) + u2 * (-(2.0 / 9.0) + u2 * ((2.0 / 33.0) - u2 / 78.0)))), f2_g) if NEED_F2 else f1 * 0.0
    f3 = tl.where(small, c0 * inv_sig4 * inv_sig2 * ((8.0 / 7.0) + u2 * (-(8.0 / 9.0) + u2 * ((4.0 / 11.0) + u2 * (-(4.0 / 39.0) + u2 / 45.0)))), (-4.0 * c0 * inv_sig4 * exp_u2 - 5.0 * f2_g) * inv_r2) if NEED_F3 else f1 * 0.0
    return k, f1, f2, f3


@_jit
def _tile_bounds(t0, n, g_sorted_ptr, group_ptr_ptr, IS_GROUPED: tl.constexpr, BLOCK: tl.constexpr):
    if not IS_GROUPED: return 0, n
    return tl.load(group_ptr_ptr + tl.load(g_sorted_ptr + tl.minimum(t0, n - 1))), tl.load(group_ptr_ptr + tl.load(g_sorted_ptr + tl.minimum(t0 + BLOCK - 1, n - 1)) + 1)


@_jit
def _row_meta(i_offs, rmask, mask_ptr, g_sorted_ptr, IS_FAST64: tl.constexpr, IS_GROUPED: tl.constexpr, BLOCK: tl.constexpr):
    return tl.load(mask_ptr + i_offs, mask=rmask, other=0).to(tl.int32), (tl.load(g_sorted_ptr + i_offs, mask=rmask, other=-1) if IS_GROUPED else tl.zeros((BLOCK,), tl.int32))


@_jit
def _pair_mask(i_offs, j_offs, rmask, cmask, mi, gi, mask_ptr, g_sorted_ptr, n, IS_FAST64: tl.constexpr, IS_GROUPED: tl.constexpr):
    mj = tl.load(mask_ptr + j_offs, mask=cmask, other=0).to(tl.int32)
    active = rmask[:, None] & cmask[None, :] & (i_offs[:, None] != j_offs[None, :]) & (mi[:, None] != 0) & (mj[None, :] != 0)
    return (active & (gi[:, None] == tl.load(g_sorted_ptr + j_offs, mask=cmask, other=-2)[None, :])) if IS_GROUPED else active


@_jit
def _coulomb_efg_kernel(pos_ptr, q_ptr, mask_ptr, g_sorted_ptr, group_ptr_ptr, bsph_ptr, phi_ptr, fld_ptr, efg_ptr, n, inv_sig, c0, inv_sig2, far_pad, IS_FAST64: tl.constexpr, IS_GROUPED: tl.constexpr, BLOCK: tl.constexpr):
    tg = tl.program_id(0); t0 = tg * BLOCK; i_offs = t0 + tl.arange(0, BLOCK); rmask = i_offs < n
    xi, yi, zi = _ld3(pos_ptr, i_offs, rmask); mi, gi = _row_meta(i_offs, rmask, mask_ptr, g_sorted_ptr, IS_FAST64, IS_GROUPED, BLOCK)
    tcx = tl.load(bsph_ptr + tg * 4 + 0) if IS_FAST64 else 0.0
    tcy = tl.load(bsph_ptr + tg * 4 + 1) if IS_FAST64 else 0.0
    tcz = tl.load(bsph_ptr + tg * 4 + 2) if IS_FAST64 else 0.0
    rtg = tl.load(bsph_ptr + tg * 4 + 3) if IS_FAST64 else -1.0
    z0 = tl.zeros((BLOCK,), tl.float32); ap, afx, afy, afz, ae0, ae1, ae2, ae3, ae4 = z0, z0, z0, z0, z0, z0, z0, z0, z0
    col_lo, col_hi = _tile_bounds(t0, n, g_sorted_ptr, group_ptr_ptr, IS_GROUPED, BLOCK)
    for j0 in range(col_lo, col_hi, BLOCK):
        j_offs = j0 + tl.arange(0, BLOCK); cmask = j_offs < col_hi
        xj, yj, zj = _ld3(pos_ptr, j_offs, cmask); qj = tl.load(q_ptr + j_offs, mask=cmask, other=0.0)
        dx, dy, dz = xi[:, None] - xj[None, :], yi[:, None] - yj[None, :], zi[:, None] - zj[None, :]
        r2 = dx * dx + dy * dy + dz * dz
        if _is_far_tile(bsph_ptr, j0 // BLOCK, tcx, tcy, tcz, rtg, far_pad, IS_FAST64):
            inv_r = tl.rsqrt(r2)
            inv_r2 = inv_r * inv_r
            k = inv_r
            f1 = inv_r * inv_r2
            f2 = -3.0 * f1 * inv_r2
        else:
            active = _pair_mask(i_offs, j_offs, rmask, cmask, mi, gi, mask_ptr, g_sorted_ptr, n, IS_FAST64, IS_GROUPED)
            k, f1, f2, _ = _smeared_k_f1_f2_f3(r2, inv_sig, c0, inv_sig2, NEED_F2=True, NEED_F3=False)
            k, f1, f2 = tl.where(active, k, 0.0), tl.where(active, f1, 0.0), tl.where(active, f2, 0.0)
        qf1, qf2 = qj[None, :] * f1, qj[None, :] * f2
        ap, afx, afy, afz = ap + tl.sum(k * qj[None, :], axis=1), afx + tl.sum(qf1 * dx, axis=1), afy + tl.sum(qf1 * dy, axis=1), afz + tl.sum(qf1 * dz, axis=1)
        ae0, ae1, ae2 = ae0 + tl.sum(qf2 * (1.4142135623730951 * dx * dy), axis=1), ae1 + tl.sum(qf2 * (1.4142135623730951 * dx * dz), axis=1), ae2 + tl.sum(qf2 * (1.4142135623730951 * dy * dz), axis=1)
        ae3, ae4 = ae3 + tl.sum(qf2 * (0.7071067811865475 * (dx * dx - dy * dy)), axis=1), ae4 + tl.sum(qf2 * (0.4082482904638630 * (2.0 * dz * dz - dx * dx - dy * dy)), axis=1)
    tl.store(phi_ptr + i_offs, ap, mask=rmask); _st3(fld_ptr, i_offs, rmask, afx, afy, afz)
    tl.store(efg_ptr + i_offs * 5, ae0, mask=rmask); tl.store(efg_ptr + i_offs * 5 + 1, ae1, mask=rmask); tl.store(efg_ptr + i_offs * 5 + 2, ae2, mask=rmask); tl.store(efg_ptr + i_offs * 5 + 3, ae3, mask=rmask); tl.store(efg_ptr + i_offs * 5 + 4, ae4, mask=rmask)


@_jit
def _ld_ge_mat(ge_ptr, offs, mask):
    g0, g1, g2 = tl.load(ge_ptr + offs * 5, mask=mask, other=0.0), tl.load(ge_ptr + offs * 5 + 1, mask=mask, other=0.0), tl.load(ge_ptr + offs * 5 + 2, mask=mask, other=0.0)
    t3, t4 = 0.7071067811865475 * tl.load(ge_ptr + offs * 5 + 3, mask=mask, other=0.0), 0.4082482904638630 * tl.load(ge_ptr + offs * 5 + 4, mask=mask, other=0.0)
    return t3 - t4, -t3 - t4, 2.0 * t4, 0.7071067811865475 * g0, 0.7071067811865475 * g1, 0.7071067811865475 * g2


@_jit
def _coulomb_efg_bwd_kernel(pos_ptr, q_ptr, mask_ptr, g_sorted_ptr, group_ptr_ptr, bsph_ptr, gp_ptr, gf_ptr, ge_ptr, dp_ptr, dq_ptr, n, inv_sig, c0, inv_sig2, far_pad, IS_FAST64: tl.constexpr, IS_GROUPED: tl.constexpr, HAS_GP: tl.constexpr, HAS_GF: tl.constexpr, HAS_GE: tl.constexpr, NEED_DPOS: tl.constexpr, NEED_DQ: tl.constexpr, BLOCK: tl.constexpr):
    tg = tl.program_id(0); t0 = tg * BLOCK; i_offs = t0 + tl.arange(0, BLOCK); rmask, z0 = i_offs < n, tl.zeros((BLOCK,), tl.float32)
    xi, yi, zi = _ld3(pos_ptr, i_offs, rmask); qi, gpi = tl.load(q_ptr + i_offs, mask=rmask, other=0.0), (tl.load(gp_ptr + i_offs, mask=rmask, other=0.0) if HAS_GP else z0)
    gfxi, gfyi, gfzi = _ld3(gf_ptr, i_offs, rmask) if HAS_GF else (z0, z0, z0)
    mixx, miyy, mizz, mixy, mixz, miyz = _ld_ge_mat(ge_ptr, i_offs, rmask) if HAS_GE else (z0, z0, z0, z0, z0, z0)
    mi, gi = _row_meta(i_offs, rmask, mask_ptr, g_sorted_ptr, IS_FAST64, IS_GROUPED, BLOCK); dpx, dpy, dpz, dq = z0, z0, z0, z0
    tcx = tl.load(bsph_ptr + tg * 4 + 0) if IS_FAST64 else 0.0
    tcy = tl.load(bsph_ptr + tg * 4 + 1) if IS_FAST64 else 0.0
    tcz = tl.load(bsph_ptr + tg * 4 + 2) if IS_FAST64 else 0.0
    rtg = tl.load(bsph_ptr + tg * 4 + 3) if IS_FAST64 else -1.0
    col_lo, col_hi = _tile_bounds(t0, n, g_sorted_ptr, group_ptr_ptr, IS_GROUPED, BLOCK)
    for j0 in range(col_lo, col_hi, BLOCK):
        j_offs = j0 + tl.arange(0, BLOCK); cmask = j_offs < col_hi
        xj, yj, zj = _ld3(pos_ptr, j_offs, cmask); qj, gpj = tl.load(q_ptr + j_offs, mask=cmask, other=0.0), (tl.load(gp_ptr + j_offs, mask=cmask, other=0.0) if HAS_GP else z0)
        gfxj, gfyj, gfzj = _ld3(gf_ptr, j_offs, cmask) if HAS_GF else (z0, z0, z0)
        mjxx, mjyy, mjzz, mjxy, mjxz, mjyz = _ld_ge_mat(ge_ptr, j_offs, cmask) if HAS_GE else (z0, z0, z0, z0, z0, z0)
        dx, dy, dz = xi[:, None] - xj[None, :], yi[:, None] - yj[None, :], zi[:, None] - zj[None, :]
        r2 = dx * dx + dy * dy + dz * dz
        if _is_far_tile(bsph_ptr, j0 // BLOCK, tcx, tcy, tcz, rtg, far_pad, IS_FAST64):
            inv_r = tl.rsqrt(r2)
            inv_r2 = inv_r * inv_r
            k = inv_r
            f1 = inv_r * inv_r2
            f2 = (-3.0 * f1 * inv_r2) if (HAS_GF or HAS_GE) else (0.0 * dx)
            f3 = (-5.0 * f2 * inv_r2) if (HAS_GE and NEED_DPOS) else (0.0 * dx)
        else:
            active = _pair_mask(i_offs, j_offs, rmask, cmask, mi, gi, mask_ptr, g_sorted_ptr, n, IS_FAST64, IS_GROUPED)
            k, f1, f2, f3 = _smeared_k_f1_f2_f3(r2, inv_sig, c0, inv_sig2, NEED_F2=(HAS_GF or HAS_GE), NEED_F3=(HAS_GE and NEED_DPOS))
            k, f1, f2, f3 = tl.where(active, k, 0.0), tl.where(active, f1, 0.0), tl.where(active, f2, 0.0), tl.where(active, f3, 0.0)
        z2 = 0.0 * dx; cd = (-(gpi[:, None] * qj[None, :] + gpj[None, :] * qi[:, None]) * f1) if (HAS_GP and NEED_DPOS) else z2
        if HAS_GF:
            if NEED_DQ:
                dq = dq - tl.sum(f1 * (gfxj[None, :] * dx + gfyj[None, :] * dy + gfzj[None, :] * dz), axis=1)
            if NEED_DPOS:
                vgx, vgy, vgz = qj[None, :] * gfxi[:, None] - qi[:, None] * gfxj[None, :], qj[None, :] * gfyi[:, None] - qi[:, None] * gfyj[None, :], qj[None, :] * gfzi[:, None] - qi[:, None] * gfzj[None, :]
                cd = cd + f2 * (vgx * dx + vgy * dy + vgz * dz)
            else:
                vgx, vgy, vgz = z2, z2, z2
        else:
            vgx, vgy, vgz = z2, z2, z2
        if HAS_GP and NEED_DQ:
            dq = dq + tl.sum(gpj[None, :] * k, axis=1)
        if HAS_GE:
            mdjx, mdjy, mdjz = mjxx[None, :] * dx + mjxy[None, :] * dy + mjxz[None, :] * dz, mjxy[None, :] * dx + mjyy[None, :] * dy + mjyz[None, :] * dz, mjxz[None, :] * dx + mjyz[None, :] * dy + mjzz[None, :] * dz
            sj = mdjx * dx + mdjy * dy + mdjz * dz
            if NEED_DQ:
                dq = dq + tl.sum(f2 * sj, axis=1)
            if NEED_DPOS:
                mdix, mdiy, mdiz = mixx[:, None] * dx + mixy[:, None] * dy + mixz[:, None] * dz, mixy[:, None] * dx + miyy[:, None] * dy + miyz[:, None] * dz, mixz[:, None] * dx + miyz[:, None] * dy + mizz[:, None] * dz
                si = mdix * dx + mdiy * dy + mdiz * dz
                cd = cd + f3 * (qj[None, :] * si + qi[:, None] * sj)
                gx, gy, gz = 2.0 * f2 * (qj[None, :] * mdix + qi[:, None] * mdjx), 2.0 * f2 * (qj[None, :] * mdiy + qi[:, None] * mdjy), 2.0 * f2 * (qj[None, :] * mdiz + qi[:, None] * mdjz)
            else:
                gx, gy, gz = z2, z2, z2
        else:
            gx, gy, gz = z2, z2, z2
        if NEED_DPOS:
            dpx, dpy, dpz = dpx + tl.sum(cd * dx + f1 * vgx + gx, axis=1), dpy + tl.sum(cd * dy + f1 * vgy + gy, axis=1), dpz + tl.sum(cd * dz + f1 * vgz + gz, axis=1)
    if NEED_DPOS:
        _st3(dp_ptr, i_offs, rmask, dpx, dpy, dpz)
    if NEED_DQ:
        tl.store(dq_ptr + i_offs, dq, mask=rmask)


@_jit
def _coulomb_bwd2_kernel(pos_ptr, q_ptr, mask_ptr, g_sorted_ptr, group_ptr_ptr, gp_ptr, gf_ptr, up_ptr, uq_ptr, d2p_ptr, d2q_ptr, d2gp_ptr, d2gf_ptr, n, inv_sig, c0, inv_sig2, IS_FAST64: tl.constexpr, IS_GROUPED: tl.constexpr, HAS_GFLD: tl.constexpr, BLOCK: tl.constexpr):
    t0 = tl.program_id(0) * BLOCK; i_offs = t0 + tl.arange(0, BLOCK); rmask, z0 = i_offs < n, tl.zeros((BLOCK,), tl.float32)
    xi, yi, zi = _ld3(pos_ptr, i_offs, rmask); uxi, uyi, uzi = _ld3(up_ptr, i_offs, rmask)
    qi, gpi, uqi = tl.load(q_ptr + i_offs, mask=rmask, other=0.0), tl.load(gp_ptr + i_offs, mask=rmask, other=0.0), tl.load(uq_ptr + i_offs, mask=rmask, other=0.0)
    gexi, geyi, gezi = _ld3(gf_ptr, i_offs, rmask) if HAS_GFLD else (z0, z0, z0)
    mi, gi = _row_meta(i_offs, rmask, mask_ptr, g_sorted_ptr, IS_FAST64, IS_GROUPED, BLOCK); adpx, adpy, adpz, adq, adgp, adgfx, adgfy, adgfz = z0, z0, z0, z0, z0, z0, z0, z0
    col_lo, col_hi = _tile_bounds(t0, n, g_sorted_ptr, group_ptr_ptr, IS_GROUPED, BLOCK)
    for j0 in range(col_lo, col_hi, BLOCK):
        j_offs = j0 + tl.arange(0, BLOCK); cmask = j_offs < col_hi
        xj, yj, zj = _ld3(pos_ptr, j_offs, cmask); uxj, uyj, uzj = _ld3(up_ptr, j_offs, cmask)
        qj, gpj, uqj = tl.load(q_ptr + j_offs, mask=cmask, other=0.0), tl.load(gp_ptr + j_offs, mask=cmask, other=0.0), tl.load(uq_ptr + j_offs, mask=cmask, other=0.0)
        gexj, geyj, gezj = _ld3(gf_ptr, j_offs, cmask) if HAS_GFLD else (z0, z0, z0)
        active = _pair_mask(i_offs, j_offs, rmask, cmask, mi, gi, mask_ptr, g_sorted_ptr, n, IS_FAST64, IS_GROUPED)
        dx, dy, dz, dux, duy, duz = xi[:, None] - xj[None, :], yi[:, None] - yj[None, :], zi[:, None] - zj[None, :], uxi[:, None] - uxj[None, :], uyi[:, None] - uyj[None, :], uzi[:, None] - uzj[None, :]
        k, f1, f2, f3 = _smeared_k_f1_f2_f3(dx * dx + dy * dy + dz * dz, inv_sig, c0, inv_sig2, NEED_F2=True, NEED_F3=True)
        k, f1, f2, f3 = tl.where(active, k, 0.0), tl.where(active, f1, 0.0), tl.where(active, f2, 0.0), tl.where(active, f3, 0.0)
        sud = dux * dx + duy * dy + duz * dz; adgp, gw, gq = adgp + tl.sum(uqj[None, :] * k - qj[None, :] * f1 * sud, axis=1), gpi[:, None] * uqj[None, :] + gpj[None, :] * uqi[:, None], gpi[:, None] * qj[None, :] + gpj[None, :] * qi[:, None]
        if HAS_GFLD:
            adgfx, adgfy, adgfz = adgfx + tl.sum(uqj[None, :] * f1 * dx + qj[None, :] * (f1 * dux + f2 * sud * dx), axis=1), adgfy + tl.sum(uqj[None, :] * f1 * dy + qj[None, :] * (f1 * duy + f2 * sud * dy), axis=1), adgfz + tl.sum(uqj[None, :] * f1 * dz + qj[None, :] * (f1 * duz + f2 * sud * dz), axis=1)
            adq = adq + tl.sum(-gpj[None, :] * f1 * sud - f1 * (dux * gexj[None, :] + duy * geyj[None, :] + duz * gezj[None, :]) - f2 * sud * (dx * gexj[None, :] + dy * geyj[None, :] + dz * gezj[None, :]), axis=1)
            vwx, vwy, vwz = uqj[None, :] * gexi[:, None] - uqi[:, None] * gexj[None, :], uqj[None, :] * geyi[:, None] - uqi[:, None] * geyj[None, :], uqj[None, :] * gezi[:, None] - uqi[:, None] * gezj[None, :]
            vex, vey, vez = qj[None, :] * gexi[:, None] - qi[:, None] * gexj[None, :], qj[None, :] * geyi[:, None] - qi[:, None] * geyj[None, :], qj[None, :] * gezi[:, None] - qi[:, None] * gezj[None, :]
            sed = vex * dx + vey * dy + vez * dz; cd, cu = -gw * f1 - gq * f2 * sud + f2 * (vwx * dx + vwy * dy + vwz * dz + vex * dux + vey * duy + vez * duz) + f3 * sud * sed, -gq * f1 + f2 * sed
            adpx, adpy, adpz = adpx + tl.sum(cd * dx + cu * dux + f1 * vwx + f2 * sud * vex, axis=1), adpy + tl.sum(cd * dy + cu * duy + f1 * vwy + f2 * sud * vey, axis=1), adpz + tl.sum(cd * dz + cu * duz + f1 * vwz + f2 * sud * vez, axis=1)
        else:
            adq, cd, cu = adq + tl.sum(-gpj[None, :] * f1 * sud, axis=1), -gw * f1 - gq * f2 * sud, -gq * f1
            adpx, adpy, adpz = adpx + tl.sum(cd * dx + cu * dux, axis=1), adpy + tl.sum(cd * dy + cu * duy, axis=1), adpz + tl.sum(cd * dz + cu * duz, axis=1)
    _st3(d2p_ptr, i_offs, rmask, adpx, adpy, adpz); tl.store(d2q_ptr + i_offs, adq, mask=rmask); tl.store(d2gp_ptr + i_offs, adgp, mask=rmask)
    if HAS_GFLD: _st3(d2gf_ptr, i_offs, rmask, adgfx, adgfy, adgfz)


def _cpu_pair_mask(pos: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, mode: int) -> Tensor:
    n = int(pos.shape[0])
    if mode == 0: m = mask_work != 0; return m.unsqueeze(1) & m.unsqueeze(0) & ~torch.eye(n, dtype=torch.bool, device=pos.device)
    m = torch.zeros(n, dtype=torch.bool, device=pos.device); m[perm] = (mask_work[:n] != 0)
    pm = m.unsqueeze(1) & m.unsqueeze(0) & ~torch.eye(n, dtype=torch.bool, device=pos.device)
    if mode == 1: return pm
    g = torch.empty(n, dtype=g_sorted.dtype, device=pos.device); g[perm] = g_sorted
    return pm & (g.unsqueeze(1) == g.unsqueeze(0))


def _torch_k_f123(r2: Tensor, sigma: float, pm: Tensor, *, need_f3: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    inv_sig, c0 = 1.0 / float(sigma), 2.0 / (float(sigma) * math.sqrt(math.pi))
    inv_sig2 = inv_sig * inv_sig
    u2 = r2 * inv_sig2
    small = u2 < 0.1
    r2s = torch.where(small, torch.ones_like(r2), r2)
    r = r2s.sqrt()
    u, exp_u2 = r * inv_sig, torch.exp(-u2)
    k_g = torch.erf(u) / r
    f1_g = (k_g - c0 * exp_u2) / r2s
    k0 = torch.where(small, c0 * (1.0 - u2 * (1.0 / 3.0 - u2 * (0.1 - u2 * (1.0 / 42.0 - u2 / 216.0)))), k_g)
    f10 = torch.where(small, c0 * inv_sig2 * (2.0 / 3.0 + u2 * (-0.4 + u2 * ((1.0 / 7.0) + u2 * (-(1.0 / 27.0) + u2 / 132.0)))), f1_g)
    f2_g = (2.0 * c0 * inv_sig2 * exp_u2 - 3.0 * f1_g) / r2s
    f20 = torch.where(small, c0 * inv_sig2 * inv_sig2 * (-0.8 + u2 * (4.0 / 7.0 + u2 * (-(2.0 / 9.0) + u2 * ((2.0 / 33.0) - u2 / 78.0)))), f2_g)
    f30 = (torch.where(small, c0 * inv_sig2 ** 3 * ((8.0 / 7.0) + u2 * (-(8.0 / 9.0) + u2 * ((4.0 / 11.0) + u2 * (-(4.0 / 39.0) + u2 / 45.0)))), (-4.0 * c0 * inv_sig2 ** 2 * exp_u2 - 5.0 * f2_g) / r2s) * pm) if need_f3 else None
    return k0 * pm, f10 * pm, f20 * pm, f30


def _torch_eval_coulomb_efg(pos: Tensor, q: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, mode: int, sigma: float) -> tuple[Tensor, Tensor, Tensor]:
    n = int(pos.shape[0])
    if n == 0: return pos.new_zeros((0,), dtype=torch.float32), pos.new_zeros((0, 3), dtype=torch.float32), pos.new_zeros((0, 5), dtype=torch.float32)
    pm, diff = _cpu_pair_mask(pos, mask_work, perm, g_sorted, mode).to(torch.float32), pos.float().unsqueeze(1) - pos.float().unsqueeze(0)
    k, f1, f2, _ = _torch_k_f123((diff * diff).sum(-1), sigma, pm); dx, dy, dz = diff.unbind(-1)
    sq2, isq2, isq6 = math.sqrt(2.0), math.sqrt(0.5), 1.0 / math.sqrt(6.0)
    y2 = torch.stack((sq2 * dx * dy, sq2 * dx * dz, sq2 * dy * dz, isq2 * (dx * dx - dy * dy), isq6 * (2.0 * dz * dz - dx * dx - dy * dy)), dim=-1)
    qf = q.float()
    return torch.einsum("ij,j->i", k, qf), torch.einsum("ij,ijc,j->ic", f1, diff, qf), torch.einsum("ij,ijm,j->im", f2, y2, qf)


def _torch_coulomb_efg_bwd_cpu(pos: Tensor, q: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, gp: Tensor, gf: Tensor, ge: Tensor, mode: int, sigma: float, has_gp: bool, has_gf: bool, has_ge: bool) -> tuple[Tensor, Tensor]:
    pm, diff, qf = _cpu_pair_mask(pos, mask_work, perm, g_sorted, mode).to(torch.float32), pos.float().unsqueeze(1) - pos.float().unsqueeze(0), q.float()
    k, f1, f2, f3 = _torch_k_f123((diff * diff).sum(-1), sigma, pm, need_f3=has_ge)
    dq, dp = torch.zeros_like(qf), pos.new_zeros((int(pos.shape[0]), 3), dtype=torch.float32)
    cd = -((gp.float()[:, None] * qf[None, :] + gp.float()[None, :] * qf[:, None]) * f1) if has_gp else torch.zeros_like(k)
    if has_gp: dq = dq + torch.einsum("ij,j->i", k, gp.float())
    if has_gf:
        gff = gf.float(); dq = dq - (f1 * torch.einsum("ijc,jc->ij", diff, gff)).sum(1)
        vw = qf[None, :, None] * gff[:, None, :] - qf[:, None, None] * gff[None, :, :]
        cd, dp = cd + f2 * (vw * diff).sum(-1), dp + (f1.unsqueeze(-1) * vw).sum(1)
    if has_ge:
        gef, isq2, isq6 = ge.float(), math.sqrt(0.5), 1.0 / math.sqrt(6.0); m = pos.new_zeros((gef.shape[0], 3, 3), dtype=torch.float32)
        m[:, 0, 0], m[:, 1, 1], m[:, 2, 2] = isq2 * gef[:, 3] - isq6 * gef[:, 4], -isq2 * gef[:, 3] - isq6 * gef[:, 4], 2.0 * isq6 * gef[:, 4]
        m[:, 0, 1] = m[:, 1, 0] = isq2 * gef[:, 0]; m[:, 0, 2] = m[:, 2, 0] = isq2 * gef[:, 1]; m[:, 1, 2] = m[:, 2, 1] = isq2 * gef[:, 2]
        mdi, mdj = torch.einsum("iab,ijb->ija", m, diff), torch.einsum("jab,ijb->ija", m, diff); si, sj = (mdi * diff).sum(-1), (mdj * diff).sum(-1)
        dq, cd = dq + (f2 * sj).sum(1), cd + f3 * (qf[None, :] * si + qf[:, None] * sj)  # type: ignore[operator]
        dp = dp + (2.0 * f2.unsqueeze(-1) * (qf[None, :, None] * mdi + qf[:, None, None] * mdj)).sum(1)
    return dp + (cd.unsqueeze(-1) * diff).sum(1), dq


def _torch_coulomb_bwd2_cpu(pos: Tensor, q: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, g_phi: Tensor, g_fld: Tensor, u_pos: Tensor, u_q: Tensor, mode: int, sigma: float, has_gfld: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    pm, p, qf, gpf, up, uq = _cpu_pair_mask(pos, mask_work, perm, g_sorted, mode).to(torch.float32), pos.float(), q.float(), g_phi.float(), u_pos.float(), u_q.float()
    diff, du = p.unsqueeze(1) - p.unsqueeze(0), up.unsqueeze(1) - up.unsqueeze(0)
    k, f1, f2, f3 = _torch_k_f123((diff * diff).sum(-1), sigma, pm, need_f3=True); assert f3 is not None
    sud, gw, gq = (du * diff).sum(-1), gpf.unsqueeze(1) * uq.unsqueeze(0) + gpf.unsqueeze(0) * uq.unsqueeze(1), gpf.unsqueeze(1) * qf.unsqueeze(0) + gpf.unsqueeze(0) * qf.unsqueeze(1)
    d2gp = (uq.unsqueeze(0) * k - qf.unsqueeze(0) * f1 * sud).sum(1)
    if not has_gfld: return ((-gw * f1 - gq * f2 * sud).unsqueeze(-1) * diff - (gq * f1).unsqueeze(-1) * du).sum(1), (-gpf.unsqueeze(0) * f1 * sud).sum(1), d2gp, pos.new_zeros((0, 3), dtype=torch.float32)
    gf = g_fld.float(); d2gf = ((f1 * uq.unsqueeze(0)).unsqueeze(-1) * diff + (f1 * qf.unsqueeze(0)).unsqueeze(-1) * du + (f2 * qf.unsqueeze(0) * sud).unsqueeze(-1) * diff).sum(1)
    d2q = (-gpf.unsqueeze(0) * f1 * sud - f1 * torch.einsum("ijc,jc->ij", du, gf) - f2 * sud * torch.einsum("ijc,jc->ij", diff, gf)).sum(1)
    vw, ve = uq[None, :, None] * gf[:, None, :] - uq[:, None, None] * gf[None, :, :], qf[None, :, None] * gf[:, None, :] - qf[:, None, None] * gf[None, :, :]
    sed, vwd = (ve * diff).sum(-1), (vw * diff + ve * du).sum(-1); cd = -gw * f1 - gq * f2 * sud + f2 * vwd + f3 * sud * sed
    return (cd.unsqueeze(-1) * diff + (-gq * f1 + f2 * sed).unsqueeze(-1) * du + f1.unsqueeze(-1) * vw + (f2 * sud).unsqueeze(-1) * ve).sum(1), d2q, d2gp, d2gf


def _pack_inputs(mode: int, n: int, perm: Tensor, bsph: Tensor, *ts: Tensor) -> tuple[int, int, int, list[Tensor]]:
    if mode == 1:
        nt, npad = int(bsph.shape[0]), int(bsph.shape[0]) * 64
        out = [torch.zeros((npad,) + t.shape[1:], dtype=torch.float32, device=t.device) for t in ts]
        for buf, t in zip(out, ts): buf[:n] = t.float().index_select(0, perm)
        return nt, npad, 64, out
    if mode == 2: return (n + 31) // 32, n, 32, [t.float().index_select(0, perm).contiguous() for t in ts]
    return (n + 63) // 64, n, 64, [t.float().contiguous() for t in ts]


def _unpack_outputs(mode: int, n: int, perm: Tensor, dev: torch.device, shapes: list[tuple[int, ...]], work: list[Tensor]) -> list[Tensor]:
    return work if mode == 0 else [torch.zeros(sh, dtype=torch.float32, device=dev).index_copy(0, perm, t[:n]) for sh, t in zip(shapes, work)]


def _take_grad(it, fl: bool, like: Tensor) -> Tensor:
    g = next(it) if fl else None
    return torch.zeros_like(like) if g is None else g


def _run_coulomb_efg_triton(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, group_ptr: Tensor, bsph: Tensor, mode: int, sigma: float) -> tuple[Tensor, Tensor, Tensor]:
    n = int(pos.shape[0])
    if n == 0: return pos.new_zeros((0,), dtype=torch.float32), pos.new_zeros((0, 3), dtype=torch.float32), pos.new_zeros((0, 5), dtype=torch.float32)
    inv_sig = 1.0 / float(sigma)
    grid, n_arg, blk, (qw,) = _pack_inputs(mode, n, perm, bsph, q)
    out_w = [torch.empty((n_arg,) + sh, dtype=torch.float32, device=pos.device) for sh in ((), (3,), (5,))]
    _coulomb_efg_kernel[(grid,)](pos_work, qw, mask_work, g_sorted, group_ptr, bsph, *out_w, n_arg, inv_sig, 2.0 * inv_sig / math.sqrt(math.pi), inv_sig * inv_sig, 5.5 * float(sigma), IS_FAST64=(mode == 1), IS_GROUPED=(mode == 2), BLOCK=blk, num_warps=4)
    return tuple(_unpack_outputs(mode, n, perm, pos.device, [(n,), (n, 3), (n, 5)], out_w))  # type: ignore[return-value]


@torch.library.custom_op("lam4::triton_coulomb_bwd2", mutates_args=())
def _triton_coulomb_bwd2_op(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, group_ptr: Tensor, bsph: Tensor, g_phi: Tensor, g_fld: Tensor, u_pos: Tensor, u_q: Tensor, mode: int, sigma: float, has_gfld: bool, need_dpos: bool, need_dq: bool, need_dgphi: bool, need_dgfld: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n = int(pos.shape[0])
    d2p, d2q, d2gp, d2gf = torch.zeros((n, 3), dtype=torch.float32, device=pos.device), torch.zeros(n, dtype=torch.float32, device=pos.device), torch.zeros(n, dtype=torch.float32, device=pos.device), torch.zeros((n, 3) if has_gfld else (0, 3), dtype=torch.float32, device=pos.device)
    if n == 0 or not (need_dpos or need_dq or need_dgphi or (need_dgfld and has_gfld)): return d2p, d2q, d2gp, d2gf
    if not has_triton_coulomb_bwd2(pos.device):
        cp, cq, cgp, cgf = _torch_coulomb_bwd2_cpu(pos, q, mask_work, perm, g_sorted, g_phi, g_fld, u_pos, u_q, mode, sigma, has_gfld)
        return cp if need_dpos else d2p, cq if need_dq else d2q, cgp if need_dgphi else d2gp, (cgf if (has_gfld and need_dgfld) else d2gf)
    inv_sig = 1.0 / float(sigma)
    grid, n_arg, blk, ws = _pack_inputs(mode, n, perm, bsph, *([q, g_phi, u_pos, u_q] + ([g_fld] if has_gfld else [])))
    out_w = [torch.empty(sh, dtype=torch.float32, device=pos.device) for sh in ([(n_arg, 3), (n_arg,), (n_arg,)] + ([(n_arg, 3)] if has_gfld else []))]
    _coulomb_bwd2_kernel[(grid,)](pos_work, ws[0], mask_work, g_sorted, group_ptr, ws[1], ws[4] if has_gfld else pos_work[:1], ws[2], ws[3], out_w[0], out_w[1], out_w[2], out_w[3] if has_gfld else pos_work[:1], n_arg, inv_sig, 2.0 * inv_sig / math.sqrt(math.pi), inv_sig * inv_sig, IS_FAST64=(mode == 1), IS_GROUPED=(mode == 2), HAS_GFLD=has_gfld, BLOCK=blk, num_warps=4)
    res = _unpack_outputs(mode, n, perm, pos.device, [(n, 3), (n,), (n,)] + ([(n, 3)] if has_gfld else []), out_w)
    return res[0], res[1], res[2], (res[3] if has_gfld else d2gf)


@_triton_coulomb_bwd2_op.register_fake
def _(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, group_ptr: Tensor, bsph: Tensor, g_phi: Tensor, g_fld: Tensor, u_pos: Tensor, u_q: Tensor, mode: int, sigma: float, has_gfld: bool, need_dpos: bool, need_dq: bool, need_dgphi: bool, need_dgfld: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return torch.empty(pos.shape, dtype=torch.float32, device=pos.device), torch.empty(q.shape, dtype=torch.float32, device=q.device), torch.empty(g_phi.shape, dtype=torch.float32, device=g_phi.device), torch.empty(pos.shape, dtype=torch.float32, device=pos.device) if has_gfld else pos.new_empty((0, 3), dtype=torch.float32)


def _setup_bwd2_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    ctx.save_for_backward(*inputs[:10]); ctx.mode, ctx.sigma, ctx.has_gfld = int(inputs[10]), float(inputs[11]), bool(inputs[12])


def _bwd2_autograd(ctx: Any, g_dp: Tensor | None, g_dq: Tensor | None):
    pos, q, pos_work, mask_work, perm, g_sorted, group_ptr, bsph, g_phi, g_fld = ctx.saved_tensors
    nds = ctx.needs_input_grad
    up, uq = (g_dp.contiguous() if g_dp is not None else pos.new_zeros((pos.shape[0], 3), dtype=torch.float32)), (g_dq.contiguous() if g_dq is not None else q.new_zeros((q.shape[0],), dtype=torch.float32))
    d2p, d2q, d2gp, d2gf = _triton_coulomb_bwd2_op(pos, q, pos_work, mask_work, perm, g_sorted, group_ptr, bsph, g_phi, g_fld, up, uq, ctx.mode, ctx.sigma, ctx.has_gfld, bool(nds[0]), bool(nds[1]), bool(nds[8]), bool(nds[9] and ctx.has_gfld))
    return (d2p.to(pos.dtype) if nds[0] else None, d2q.to(q.dtype) if nds[1] else None, None, None, None, None, None, None, d2gp.to(g_phi.dtype) if nds[8] else None, d2gf.to(g_fld.dtype) if (nds[9] and ctx.has_gfld) else None, None, None, None, None, None)


def register_coulomb_bwd2() -> None:
    from lam4.kernels import triton_coulomb as _tc
    if getattr(_tc, "HAS_TRITON", False) and not getattr(register_coulomb_bwd2, "_done", False):
        torch.library.register_autograd("lam4::triton_coulomb_bwd", _bwd2_autograd, setup_context=_setup_bwd2_ctx)
        register_coulomb_bwd2._done = True  # type: ignore[attr-defined]


@torch.library.custom_op("lam4::triton_coulomb_efg", mutates_args=())
def _triton_coulomb_efg_op(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, group_ptr: Tensor, bsph: Tensor, mode: int, sigma: float) -> tuple[Tensor, Tensor, Tensor]:
    return _run_coulomb_efg_triton(pos, q, pos_work, mask_work, perm, g_sorted, group_ptr, bsph, mode, sigma) if has_triton_coulomb_bwd2(pos.device) else _torch_eval_coulomb_efg(pos, q, mask_work, perm, g_sorted, mode, sigma)


@_triton_coulomb_efg_op.register_fake
def _(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, group_ptr: Tensor, bsph: Tensor, mode: int, sigma: float) -> tuple[Tensor, Tensor, Tensor]:
    return torch.empty_like(q, dtype=torch.float32), torch.empty_like(pos, dtype=torch.float32), pos.new_empty((pos.shape[0], 5), dtype=torch.float32)


@torch.library.custom_op("lam4::triton_coulomb_efg_bwd", mutates_args=())
def _triton_coulomb_efg_bwd_op(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, group_ptr: Tensor, bsph: Tensor, g_phi: Tensor, g_fld: Tensor, g_efg: Tensor, mode: int, sigma: float, has_gp: bool, has_gf: bool, has_ge: bool, need_dpos: bool, need_dq: bool) -> tuple[Tensor, Tensor]:
    n = int(pos.shape[0])
    if n == 0 or not (need_dpos or need_dq) or not (has_gp or has_gf or has_ge):
        return torch.zeros_like(pos, dtype=torch.float32), torch.zeros_like(q, dtype=torch.float32)
    if not has_triton_coulomb_bwd2(pos.device):
        dp_c, dq_c = _torch_coulomb_efg_bwd_cpu(pos, q, mask_work, perm, g_sorted, g_phi, g_fld, g_efg, mode, sigma, has_gp, has_gf, has_ge)
        return (dp_c if need_dpos else torch.zeros_like(pos, dtype=torch.float32)), (dq_c if need_dq else torch.zeros_like(q, dtype=torch.float32))
    inv_sig = 1.0 / float(sigma)
    pack_list = [q]
    if has_gp: pack_list.append(g_phi)
    if has_gf: pack_list.append(g_fld)
    if has_ge: pack_list.append(g_efg)
    grid, n_arg, blk, packed = _pack_inputs(mode, n, perm, bsph, *pack_list)
    it = iter(packed)
    qw = next(it)
    gpw = next(it) if has_gp else pos_work[:1]
    gfw = next(it) if has_gf else pos_work[:1]
    gew = next(it) if has_ge else pos_work[:1]
    dp_w = torch.empty((n_arg, 3), dtype=torch.float32, device=pos.device) if need_dpos else pos_work[:1]
    dq_w = torch.empty((n_arg,), dtype=torch.float32, device=pos.device) if need_dq else pos_work[:1]
    _coulomb_efg_bwd_kernel[(grid,)](pos_work, qw, mask_work, g_sorted, group_ptr, bsph, gpw, gfw, gew, dp_w, dq_w, n_arg, inv_sig, 2.0 * inv_sig / math.sqrt(math.pi), inv_sig * inv_sig, 5.5 * float(sigma), IS_FAST64=(mode == 1), IS_GROUPED=(mode == 2), HAS_GP=has_gp, HAS_GF=has_gf, HAS_GE=has_ge, NEED_DPOS=need_dpos, NEED_DQ=need_dq, BLOCK=blk, num_warps=4)
    dp = _unpack_outputs(mode, n, perm, pos.device, [(n, 3)], [dp_w])[0] if need_dpos else torch.zeros((n, 3), dtype=torch.float32, device=pos.device)
    dq = _unpack_outputs(mode, n, perm, pos.device, [(n,)], [dq_w])[0] if need_dq else torch.zeros((n,), dtype=torch.float32, device=pos.device)
    return dp, dq


@_triton_coulomb_efg_bwd_op.register_fake
def _(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, group_ptr: Tensor, bsph: Tensor, g_phi: Tensor, g_fld: Tensor, g_efg: Tensor, mode: int, sigma: float, has_gp: bool, has_gf: bool, has_ge: bool, need_dpos: bool, need_dq: bool) -> tuple[Tensor, Tensor]:
    return torch.empty(pos.shape, dtype=torch.float32, device=pos.device), torch.empty(q.shape, dtype=torch.float32, device=q.device)


def _setup_efg_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    ctx.save_for_backward(*inputs[:8]); ctx.mode, ctx.sigma = int(inputs[8]), float(inputs[9])


def _efg_bwd_autograd(ctx: Any, g_phi: Tensor | None, g_fld: Tensor | None, g_efg: Tensor | None):
    pos, q, pw, mw, perm, gs, gptr, bsph = ctx.saved_tensors
    np_, nq_ = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
    if not (np_ or nq_): return (None,) * 10
    gp = g_phi.float().contiguous() if g_phi is not None else q[:1].float()
    gf = g_fld.float().contiguous() if g_fld is not None else pos[:1].float()
    ge = g_efg.float().contiguous() if g_efg is not None else pos[:1].float()
    dp, dq = _triton_coulomb_efg_bwd_op(pos, q, pw, mw, perm, gs, gptr, bsph, gp, gf, ge, ctx.mode, ctx.sigma, g_phi is not None, g_fld is not None, g_efg is not None, np_, nq_)
    return (dp.to(pos.dtype) if np_ else None, dq.to(q.dtype) if nq_ else None) + (None,) * 8


def _efg_bwd2_unsupported(*_args: Any, **_kw: Any):
    raise NotImplementedError("lam4::triton_coulomb_efg_bwd supports 1st-order backward only (`f4` 2nd-order EFG bwd omitted); use `phi_field` (`triton_coulomb_bwd2`) for `create_graph=True`.")


_triton_coulomb_efg_op.register_autograd(_efg_bwd_autograd, setup_context=_setup_efg_ctx)
_triton_coulomb_efg_bwd_op.register_autograd(_efg_bwd2_unsupported, setup_context=lambda ctx, inputs, output: None)


def eval_coulomb_phi_field_efg(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, group_ptr: Tensor, bsph: Tensor, mode: int, sigma: float) -> tuple[Tensor, Tensor, Tensor]:
    """Compute `(phi [N], fld [N, 3], efg_l2 [N, 5])` via `lam4::triton_coulomb_efg` (`O(N)` tiled CUDA fwd+bwd, `make_fx` FakeTensor safe) or eager CPU autograd."""
    if has_triton_coulomb_bwd2(pos.device) or torch.compiler.is_compiling() or proxy_tracing() or is_fake(pos):
        phi, fld, efg = _triton_coulomb_efg_op(pos, q, pos_work, mask_work, perm, g_sorted, group_ptr, bsph, int(mode), float(sigma))
    else:
        phi, fld, efg = _torch_eval_coulomb_efg(pos, q, mask_work, perm, g_sorted, int(mode), float(sigma))
    return phi.to(q.dtype), fld.to(q.dtype), efg.to(q.dtype)


def eval_nlh_decoupled_impl(b, pos: Tensor | None = None, *, csr_plan=None, compute_forces: bool = True, compute_stress: bool = False) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """O3: Evaluate parameter-free `Repulsion` (`E_nlh, F_nlh, stress_nlh`) using 1st-order `triton_nlh` / CPU fallback outside `create_graph=True`."""
    p, c = (b.pos if pos is None else pos).detach(), b.cell.detach()
    do_stress = bool(compute_stress and getattr(b, "pbc", False) and c.ndim == 3)
    if p.device.type == "cuda" and csr_plan is not None:
        from lam4.kernels.triton_nlh import _triton_nlh_bwd_op, _triton_nlh_op, has_triton_nlh
        if has_triton_nlh(p.device):
            g_c, es, z = (b.graph.to(torch.int64).contiguous() if (c.ndim == 3 and c.shape[0] > 1) else p.new_zeros(0, dtype=torch.int64)), b.edge_shift.to(torch.float32).contiguous(), b.z.to(torch.int64).contiguous()
            e_atom = _triton_nlh_op(csr_plan.recv_ptr, csr_plan.send_idx, csr_plan.send_ptr, csr_plan.send_perm, csr_plan.recv_idx, es, g_c, z, p.contiguous(), c.contiguous()).detach()
            if not (compute_forces or do_stress): return e_atom, None, None
            dp, dc = _triton_nlh_bwd_op(csr_plan.recv_ptr, csr_plan.send_idx, csr_plan.send_ptr, csr_plan.send_perm, csr_plan.recv_idx, es, g_c, z, p.contiguous(), c.contiguous(), torch.ones_like(e_atom), do_stress)
            f_nlh = (-dp).detach() if compute_forces else None
            st = (0.5 * (m + m.transpose(-1, -2)) / torch.det(c).abs().clamp(min=1e-12).view(-1, 1, 1)).detach() if (do_stress and (m := torch.bmm(c.transpose(-1, -2), dc).index_add(0, b.graph, p.unsqueeze(2) * dp.unsqueeze(1))) is not None) else None
            return e_atom, f_nlh, st
    from lam4.physics.repulsion import nlh_atom_energy
    if not (compute_forces or do_stress):
        with torch.no_grad(): return nlh_atom_energy(b, p, create_graph=False).detach(), None, None
    with torch.enable_grad():
        pw, sym = p.clone().requires_grad_(compute_forces), (c.new_zeros((b.n_graphs, 3, 3)).requires_grad_(True) if do_stress else None)
        p_in, c_in = (pw + torch.einsum("ni,nij->nj", pw, sym[b.graph]), c + torch.einsum("gki,gij->gkj", c, sym)) if do_stress else (pw, c)
        e_atom = nlh_atom_energy(b if not do_stress else (b.with_cell(c_in) if hasattr(b, "with_cell") else replace(b, cell=c_in)), p_in, create_graph=False)
        it = iter(torch.autograd.grad(e_atom.sum(), ([pw] if compute_forces else []) + ([sym] if do_stress else []), create_graph=False, allow_unused=True))
        return e_atom.detach(), ((-_take_grad(it, compute_forces, p)).detach() if compute_forces else None), ((_take_grad(it, True, sym) / torch.det(c).abs().clamp(min=1e-12).view(-1, 1, 1)).detach() if do_stress else None)
