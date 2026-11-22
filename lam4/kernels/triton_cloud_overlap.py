"""Dual-CSR Triton CUDA custom ops & autograd for CloudOverlapHead (zero [E, C] edge buffers)."""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

from lam4.core.consts import cuda_cap_ok
from lam4.kernels.csr_plan import CSRPlan, dummy_f32, dummy_i64

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - macOS / CPU
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

_jit = triton.jit if _HAS_TRITON else (lambda fn: fn)


def has_triton_cloud_ov(device: torch.device | str | None = None) -> bool:
    if os.environ.get("LAM4_TRITON_CLOUD_OV", "1") == "0":
        return False
    return bool(_HAS_TRITON and cuda_cap_ok(device))


def can_use_triton_cloud_ov(
    pos: Tensor, poly: Tensor, k_ex: int, k_an: int, k_t: int,
    *, create_graph: bool = False, shift_in: Tensor | None = None,
) -> bool:
    return (
        not create_graph and pos.dtype == torch.float32 and pos.device.type == "cuda"
        and has_triton_cloud_ov(pos.device) and not poly.requires_grad
        and not (shift_in is not None and shift_in.requires_grad)
        and (int(k_ex), int(k_an), int(k_t)) == (16, 8, 4)
    )


@_jit
def _ov_c3_env(r, r_on, inv_w):
    t = tl.minimum(tl.maximum((r - r_on) * inv_w, 0.0), 1.0)
    om = 1.0 - t; t2 = t * t; t3 = t2 * t
    chi = tl.where(r <= r_on, 1.0, 1.0 - t3 * t * (35.0 - 84.0 * t + 70.0 * t2 - 20.0 * t3))
    dchi = tl.where(r <= r_on, 0.0, -140.0 * t3 * (om * om * om) * inv_w)
    return chi, dchi


@_jit
def _ov_horner_val(poly_ptr, p_base, col, tf, valid):
    a0 = tl.load(poly_ptr + p_base + col, mask=valid, other=0.0); a1 = tl.load(poly_ptr + p_base + 44 + col, mask=valid, other=0.0)
    a2 = tl.load(poly_ptr + p_base + 88 + col, mask=valid, other=0.0); a3 = tl.load(poly_ptr + p_base + 132 + col, mask=valid, other=0.0)
    return ((a3 * tf + a2) * tf + a1) * tf + a0


@_jit
def _ov_horner_val_der(poly_ptr, p_base, col, tf, inv_h, valid):
    a0 = tl.load(poly_ptr + p_base + col, mask=valid, other=0.0); a1 = tl.load(poly_ptr + p_base + 44 + col, mask=valid, other=0.0)
    a2 = tl.load(poly_ptr + p_base + 88 + col, mask=valid, other=0.0); a3 = tl.load(poly_ptr + p_base + 132 + col, mask=valid, other=0.0)
    return ((a3 * tf + a2) * tf + a1) * tf + a0, ((3.0 * a3 * tf + 2.0 * a2) * tf + a1) * inv_h


@_jit
def _ov_load_shift(shift_ptr, cell_ptr, e, g_id, em, HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr):
    if not HAS_SHIFT:
        z = e.to(tl.float32) * 0.0
        return z, z, z, z, z, z
    sx = tl.load(shift_ptr + e * 3 + 0, mask=em, other=0.0); sy = tl.load(shift_ptr + e * 3 + 1, mask=em, other=0.0); sz = tl.load(shift_ptr + e * 3 + 2, mask=em, other=0.0)
    if not HAS_CELL:
        return sx, sy, sz, sx, sy, sz
    c = cell_ptr + g_id * 9
    dx = sx * tl.load(c + 0) + sy * tl.load(c + 3) + sz * tl.load(c + 6)
    dy = sx * tl.load(c + 1) + sy * tl.load(c + 4) + sz * tl.load(c + 7)
    dz = sx * tl.load(c + 2) + sy * tl.load(c + 5) + sz * tl.load(c + 8)
    return dx, dy, dz, sx, sy, sz


@_jit
def _ov_edge_grad_d(
    dx, dy, dz, r2, valid, r_cut, r_on, inv_w, inv_h, grid,
    poly_ptr, a_p_ptr, c1_ptr, d1_ptr, c2_ptr, d2_ptr,
    gp_ptr, gds_ptr, gv_ptr, gys_ptr, gyt_ptr,
    i_idx, j_idx, BLOCK_E: tl.constexpr,
):
    r = tl.sqrt(r2 + 1e-12)
    inv_r = tl.where(valid, 1.0 / r, 0.0)
    ux, uy, uz = dx * inv_r, dy * inv_r, dz * inv_r
    chi, dchi = _ov_c3_env(r, r_on, inv_w)
    chi, dchi = tl.where(valid, chi, 0.0), tl.where(valid, dchi, 0.0)
    xf = tl.minimum(r, r_cut) * inv_h
    g = tl.minimum(tl.floor(xf).to(tl.int32), grid - 1)
    tf = (xf - g.to(tl.float32))[:, None]
    p_base = (4 * g[:, None]) * 44
    v2 = valid[:, None]

    # l=0 Pauli (16 channels)
    k16 = tl.arange(0, 16)[None, :]
    sp, dsp = _ov_horner_val_der(poly_ptr, p_base, k16, tf, inv_h, v2)
    dfp = dchi[:, None] * sp + chi[:, None] * dsp
    apj = tl.load(a_p_ptr + j_idx[:, None] * 16 + k16, mask=v2, other=0.0)
    gpi = tl.load(gp_ptr + i_idx[:, None] * 16 + k16, mask=v2, other=0.0)
    gr = tl.sum(dfp * gpi * apj, axis=1)

    # l=1 anisotropic (8 channels)
    k8 = tl.arange(0, 8)[None, :]
    s_dir, ds_dir = _ov_horner_val_der(poly_ptr, p_base, 16 + k8, tf, inv_h, v2)
    s_tr, ds_tr = _ov_horner_val_der(poly_ptr, p_base, 24 + k8, tf, inv_h, v2)
    s_lon, ds_lon = _ov_horner_val_der(poly_ptr, p_base, 32 + k8, tf, inv_h, v2)
    fdir, dfdir = chi[:, None] * s_dir, dchi[:, None] * s_dir + chi[:, None] * ds_dir
    dftr = dchi[:, None] * s_tr + chi[:, None] * ds_tr
    flon, dflon = chi[:, None] * s_lon, dchi[:, None] * s_lon + chi[:, None] * ds_lon
    j24 = j_idx[:, None] * 24 + k8 * 3
    i24 = i_idx[:, None] * 24 + k8 * 3
    c1x = tl.load(c1_ptr + j24 + 0, mask=v2, other=0.0)
    c1y = tl.load(c1_ptr + j24 + 1, mask=v2, other=0.0)
    c1z = tl.load(c1_ptr + j24 + 2, mask=v2, other=0.0)
    gvx = tl.load(gv_ptr + i24 + 0, mask=v2, other=0.0)
    gvy = tl.load(gv_ptr + i24 + 1, mask=v2, other=0.0)
    gvz = tl.load(gv_ptr + i24 + 2, mask=v2, other=0.0)
    d1j = tl.load(d1_ptr + j_idx[:, None] * 8 + k8, mask=v2, other=0.0)
    gdsi = tl.load(gds_ptr + i_idx[:, None] * 8 + k8, mask=v2, other=0.0)
    cu = c1x * ux[:, None] + c1y * uy[:, None] + c1z * uz[:, None]
    gvu = gvx * ux[:, None] + gvy * uy[:, None] + gvz * uz[:, None]
    gr += tl.sum(dfdir * (gdsi * cu + d1j * gvu) + dftr * (gvx * c1x + gvy * c1y + gvz * c1z) + dflon * cu * gvu, axis=1)
    wc1 = fdir * gdsi + flon * gvu
    wgv = fdir * d1j + flon * cu
    gux = tl.sum(wc1 * c1x + wgv * gvx, axis=1)
    guy = tl.sum(wc1 * c1y + wgv * gvy, axis=1)
    guz = tl.sum(wc1 * c1z + wgv * gvz, axis=1)

    # l=2 quadrupole (4 channels)
    k4 = tl.arange(0, 4)[None, :]
    sy, dsy = _ov_horner_val_der(poly_ptr, p_base, 40 + k4, tf, inv_h, v2)
    fy2, dfy2 = chi[:, None] * sy, dchi[:, None] * sy + chi[:, None] * dsy
    s2, is2, is6 = 1.4142135623730951, 0.7071067811865475, 0.4082482904638630
    y0, y1, y2v = s2 * ux * uy, s2 * ux * uz, s2 * uy * uz
    y3, y4 = is2 * (ux * ux - uy * uy), is6 * (2.0 * uz * uz - ux * ux - uy * uy)
    j20 = j_idx[:, None] * 20 + k4 * 5
    i20 = i_idx[:, None] * 20 + k4 * 5
    d2j = tl.load(d2_ptr + j_idx[:, None] * 4 + k4, mask=v2, other=0.0)
    gysi = tl.load(gys_ptr + i_idx[:, None] * 4 + k4, mask=v2, other=0.0)
    c20 = tl.load(c2_ptr + j20 + 0, mask=v2, other=0.0); g0 = tl.load(gyt_ptr + i20 + 0, mask=v2, other=0.0)
    c21 = tl.load(c2_ptr + j20 + 1, mask=v2, other=0.0); g1 = tl.load(gyt_ptr + i20 + 1, mask=v2, other=0.0)
    c22 = tl.load(c2_ptr + j20 + 2, mask=v2, other=0.0); g2 = tl.load(gyt_ptr + i20 + 2, mask=v2, other=0.0)
    c23 = tl.load(c2_ptr + j20 + 3, mask=v2, other=0.0); g3 = tl.load(gyt_ptr + i20 + 3, mask=v2, other=0.0)
    c24 = tl.load(c2_ptr + j20 + 4, mask=v2, other=0.0); g4 = tl.load(gyt_ptr + i20 + 4, mask=v2, other=0.0)
    c2y = c20 * y0[:, None] + c21 * y1[:, None] + c22 * y2v[:, None] + c23 * y3[:, None] + c24 * y4[:, None]
    gyty = g0 * y0[:, None] + g1 * y1[:, None] + g2 * y2v[:, None] + g3 * y3[:, None] + g4 * y4[:, None]
    gr += tl.sum(dfy2 * (gysi * c2y + d2j * gyty), axis=1)
    w0 = tl.sum(fy2 * (gysi * c20 + d2j * g0), axis=1)
    w1 = tl.sum(fy2 * (gysi * c21 + d2j * g1), axis=1)
    w2 = tl.sum(fy2 * (gysi * c22 + d2j * g2), axis=1)
    w3 = tl.sum(fy2 * (gysi * c23 + d2j * g3), axis=1)
    w4 = tl.sum(fy2 * (gysi * c24 + d2j * g4), axis=1)
    k6 = 2.0 * is6
    gux += s2 * (w0 * uy + w1 * uz + w3 * ux) - k6 * w4 * ux
    guy += s2 * (w0 * ux + w2 * uz - w3 * uy) - k6 * w4 * uy
    guz += s2 * (w1 * ux + w2 * uy) + 2.0 * k6 * w4 * uz

    gu_u = gux * ux + guy * uy + guz * uz
    gdx = tl.where(valid, (gux - gu_u * ux) * inv_r + gr * ux, 0.0)
    gdy = tl.where(valid, (guy - gu_u * uy) * inv_r + gr * uy, 0.0)
    gdz = tl.where(valid, (guz - gu_u * uz) * inv_r + gr * uz, 0.0)
    return gdx, gdy, gdz


@_jit
def _cloud_ov_fwd_recv_kernel(
    pos_ptr, recv_ptr, send_idx, mask_ptr, shift_ptr, cell_ptr, graph_ptr, poly_ptr,
    a_p_ptr, c1_ptr, d1_ptr, c2_ptr, d2_ptr,
    out_ap, out_ds, out_v, out_ys, out_yt,
    n_atoms, grid, r_cut, r_on, inv_w, inv_h,
    HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr, BLOCK_E: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= n_atoms:
        return
    if tl.load(mask_ptr + i) == 0:
        return
    g_id = 0 if (not HAS_CELL or IS_G1) else tl.load(graph_ptr + i).to(tl.int32)
    pix, piy, piz = tl.load(pos_ptr + i * 3 + 0), tl.load(pos_ptr + i * 3 + 1), tl.load(pos_ptr + i * 3 + 2)
    e0, e1 = tl.load(recv_ptr + i), tl.load(recv_ptr + i + 1)
    r_cut2 = r_cut * r_cut
    k16, k8, k4 = tl.arange(0, 16), tl.arange(0, 8), tl.arange(0, 4)
    acc_p = tl.zeros((16,), dtype=tl.float32)
    acc_ds = tl.zeros((8,), dtype=tl.float32)
    acc_vx, acc_vy, acc_vz = tl.zeros((8,), dtype=tl.float32), tl.zeros((8,), dtype=tl.float32), tl.zeros((8,), dtype=tl.float32)
    acc_ys = tl.zeros((4,), dtype=tl.float32)
    ay0 = tl.zeros((4,), dtype=tl.float32); ay1 = tl.zeros((4,), dtype=tl.float32); ay2 = tl.zeros((4,), dtype=tl.float32)
    ay3 = tl.zeros((4,), dtype=tl.float32); ay4 = tl.zeros((4,), dtype=tl.float32)
    s2, is2, is6 = 1.4142135623730951, 0.7071067811865475, 0.4082482904638630

    for eb in range(e0, e1, BLOCK_E):
        e = eb + tl.arange(0, BLOCK_E)
        em = e < e1
        j = tl.load(send_idx + e, mask=em, other=0)
        mj = tl.load(mask_ptr + j, mask=em, other=0)
        shx, shy, shz, _, _, _ = _ov_load_shift(shift_ptr, cell_ptr, e, g_id, em, HAS_SHIFT, HAS_CELL)
        dx = tl.load(pos_ptr + j * 3 + 0, mask=em, other=0.0) - pix + shx
        dy = tl.load(pos_ptr + j * 3 + 1, mask=em, other=0.0) - piy + shy
        dz = tl.load(pos_ptr + j * 3 + 2, mask=em, other=0.0) - piz + shz
        r2 = dx * dx + dy * dy + dz * dz
        valid = em & (mj != 0) & (r2 > 1e-12) & (r2 < r_cut2)
        r = tl.sqrt(r2 + 1e-12)
        inv_r = tl.where(valid, 1.0 / r, 0.0)
        ux, uy, uz = dx * inv_r, dy * inv_r, dz * inv_r
        chi, _ = _ov_c3_env(r, r_on, inv_w)
        chi = tl.where(valid, chi, 0.0)
        xf = tl.minimum(r, r_cut) * inv_h
        g = tl.minimum(tl.floor(xf).to(tl.int32), grid - 1)
        tf = (xf - g.to(tl.float32))[:, None]
        p_base = (4 * g[:, None]) * 44
        v2 = valid[:, None]

        fp = chi[:, None] * _ov_horner_val(poly_ptr, p_base, k16[None, :], tf, v2)
        acc_p += tl.sum(fp * tl.load(a_p_ptr + j[:, None] * 16 + k16[None, :], mask=v2, other=0.0), axis=0)

        fdir = chi[:, None] * _ov_horner_val(poly_ptr, p_base, 16 + k8[None, :], tf, v2)
        ftr = chi[:, None] * _ov_horner_val(poly_ptr, p_base, 24 + k8[None, :], tf, v2)
        flon = chi[:, None] * _ov_horner_val(poly_ptr, p_base, 32 + k8[None, :], tf, v2)
        j24 = j[:, None] * 24 + k8[None, :] * 3
        c1x = tl.load(c1_ptr + j24 + 0, mask=v2, other=0.0)
        c1y = tl.load(c1_ptr + j24 + 1, mask=v2, other=0.0)
        c1z = tl.load(c1_ptr + j24 + 2, mask=v2, other=0.0)
        d1j = tl.load(d1_ptr + j[:, None] * 8 + k8[None, :], mask=v2, other=0.0)
        ld = c1x * ux[:, None] + c1y * uy[:, None] + c1z * uz[:, None]
        acc_ds += tl.sum(fdir * ld, axis=0)
        cu = fdir * d1j + flon * ld
        acc_vx += tl.sum(cu * ux[:, None] + ftr * c1x, axis=0)
        acc_vy += tl.sum(cu * uy[:, None] + ftr * c1y, axis=0)
        acc_vz += tl.sum(cu * uz[:, None] + ftr * c1z, axis=0)

        fy2 = chi[:, None] * _ov_horner_val(poly_ptr, p_base, 40 + k4[None, :], tf, v2)
        y0, y1, y2v = s2 * ux * uy, s2 * ux * uz, s2 * uy * uz
        y3, y4 = is2 * (ux * ux - uy * uy), is6 * (2.0 * uz * uz - ux * ux - uy * uy)
        j20 = j[:, None] * 20 + k4[None, :] * 5
        d2j = tl.load(d2_ptr + j[:, None] * 4 + k4[None, :], mask=v2, other=0.0)
        c20 = tl.load(c2_ptr + j20 + 0, mask=v2, other=0.0)
        c21 = tl.load(c2_ptr + j20 + 1, mask=v2, other=0.0)
        c22 = tl.load(c2_ptr + j20 + 2, mask=v2, other=0.0)
        c23 = tl.load(c2_ptr + j20 + 3, mask=v2, other=0.0)
        c24 = tl.load(c2_ptr + j20 + 4, mask=v2, other=0.0)
        acc_ys += tl.sum(fy2 * (c20 * y0[:, None] + c21 * y1[:, None] + c22 * y2v[:, None] + c23 * y3[:, None] + c24 * y4[:, None]), axis=0)
        wy = fy2 * d2j
        ay0 += tl.sum(wy * y0[:, None], axis=0); ay1 += tl.sum(wy * y1[:, None], axis=0); ay2 += tl.sum(wy * y2v[:, None], axis=0)
        ay3 += tl.sum(wy * y3[:, None], axis=0); ay4 += tl.sum(wy * y4[:, None], axis=0)

    tl.store(out_ap + i * 16 + k16, acc_p)
    tl.store(out_ds + i * 8 + k8, acc_ds)
    tl.store(out_v + i * 24 + k8 * 3 + 0, acc_vx); tl.store(out_v + i * 24 + k8 * 3 + 1, acc_vy); tl.store(out_v + i * 24 + k8 * 3 + 2, acc_vz)
    tl.store(out_ys + i * 4 + k4, acc_ys)
    tl.store(out_yt + i * 20 + k4 * 5 + 0, ay0); tl.store(out_yt + i * 20 + k4 * 5 + 1, ay1); tl.store(out_yt + i * 20 + k4 * 5 + 2, ay2)
    tl.store(out_yt + i * 20 + k4 * 5 + 3, ay3); tl.store(out_yt + i * 20 + k4 * 5 + 4, ay4)


@_jit
def _cloud_ov_bwd_recv_kernel(
    pos_ptr, recv_ptr, send_idx, mask_ptr, shift_ptr, cell_ptr, graph_ptr, poly_ptr,
    a_p_ptr, c1_ptr, d1_ptr, c2_ptr, d2_ptr, gp_ptr, gds_ptr, gv_ptr, gys_ptr, gyt_ptr,
    d_pos_ptr, vir_atom_ptr,
    n_atoms, grid, r_cut, r_on, inv_w, inv_h,
    HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr,
    NEED_DPOS: tl.constexpr, NEED_VIR: tl.constexpr, BLOCK_E: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= n_atoms or tl.load(mask_ptr + i) == 0:
        return
    g_id = 0 if (not HAS_CELL or IS_G1) else tl.load(graph_ptr + i).to(tl.int32)
    pix, piy, piz = tl.load(pos_ptr + i * 3 + 0), tl.load(pos_ptr + i * 3 + 1), tl.load(pos_ptr + i * 3 + 2)
    e0, e1 = tl.load(recv_ptr + i), tl.load(recv_ptr + i + 1)
    r_cut2 = r_cut * r_cut
    dpx, dpy, dpz = 0.0, 0.0, 0.0
    v00 = 0.0; v01 = 0.0; v02 = 0.0; v10 = 0.0; v11 = 0.0; v12 = 0.0; v20 = 0.0; v21 = 0.0; v22 = 0.0
    i_vec = tl.full((BLOCK_E,), i, dtype=tl.int32)
    for eb in range(e0, e1, BLOCK_E):
        e = eb + tl.arange(0, BLOCK_E)
        em = e < e1
        j = tl.load(send_idx + e, mask=em, other=0)
        mj = tl.load(mask_ptr + j, mask=em, other=0)
        shx, shy, shz, sx, sy, sz = _ov_load_shift(shift_ptr, cell_ptr, e, g_id, em, HAS_SHIFT, HAS_CELL)
        dx = tl.load(pos_ptr + j * 3 + 0, mask=em, other=0.0) - pix + shx
        dy = tl.load(pos_ptr + j * 3 + 1, mask=em, other=0.0) - piy + shy
        dz = tl.load(pos_ptr + j * 3 + 2, mask=em, other=0.0) - piz + shz
        r2 = dx * dx + dy * dy + dz * dz
        valid = em & (mj != 0) & (r2 > 1e-12) & (r2 < r_cut2)
        gdx, gdy, gdz = _ov_edge_grad_d(
            dx, dy, dz, r2, valid, r_cut, r_on, inv_w, inv_h, grid,
            poly_ptr, a_p_ptr, c1_ptr, d1_ptr, c2_ptr, d2_ptr,
            gp_ptr, gds_ptr, gv_ptr, gys_ptr, gyt_ptr, i_vec, j, BLOCK_E,
        )
        if NEED_DPOS:
            dpx -= tl.sum(gdx, axis=0); dpy -= tl.sum(gdy, axis=0); dpz -= tl.sum(gdz, axis=0)
        if NEED_VIR:
            v00 += tl.sum(sx * gdx, axis=0); v01 += tl.sum(sx * gdy, axis=0); v02 += tl.sum(sx * gdz, axis=0)
            v10 += tl.sum(sy * gdx, axis=0); v11 += tl.sum(sy * gdy, axis=0); v12 += tl.sum(sy * gdz, axis=0)
            v20 += tl.sum(sz * gdx, axis=0); v21 += tl.sum(sz * gdy, axis=0); v22 += tl.sum(sz * gdz, axis=0)
    if NEED_DPOS:
        tl.store(d_pos_ptr + i * 3 + 0, dpx); tl.store(d_pos_ptr + i * 3 + 1, dpy); tl.store(d_pos_ptr + i * 3 + 2, dpz)
    if NEED_VIR:
        b = vir_atom_ptr + i * 9
        tl.store(b + 0, v00); tl.store(b + 1, v01); tl.store(b + 2, v02)
        tl.store(b + 3, v10); tl.store(b + 4, v11); tl.store(b + 5, v12)
        tl.store(b + 6, v20); tl.store(b + 7, v21); tl.store(b + 8, v22)


@_jit
def _cloud_ov_bwd_send_kernel(
    pos_ptr, send_ptr, send_perm, recv_idx, mask_ptr, shift_ptr, cell_ptr, graph_ptr, poly_ptr,
    a_p_ptr, c1_ptr, d1_ptr, c2_ptr, d2_ptr, gp_ptr, gds_ptr, gv_ptr, gys_ptr, gyt_ptr,
    d_pos_ptr, d_ap_ptr, d_c1_ptr, d_d1_ptr, d_c2_ptr, d_d2_ptr,
    n_atoms, grid, r_cut, r_on, inv_w, inv_h,
    HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr,
    NEED_DPOS: tl.constexpr, BLOCK_E: tl.constexpr,
):
    j = tl.program_id(0)
    if j >= n_atoms or tl.load(mask_ptr + j) == 0:
        return
    g_id = 0 if (not HAS_CELL or IS_G1) else tl.load(graph_ptr + j).to(tl.int32)
    pjx, pjy, pjz = tl.load(pos_ptr + j * 3 + 0), tl.load(pos_ptr + j * 3 + 1), tl.load(pos_ptr + j * 3 + 2)
    k0, k1 = tl.load(send_ptr + j), tl.load(send_ptr + j + 1)
    r_cut2 = r_cut * r_cut
    k16, k8, k4 = tl.arange(0, 16), tl.arange(0, 8), tl.arange(0, 4)
    dpx, dpy, dpz = 0.0, 0.0, 0.0
    dap = tl.zeros((16,), dtype=tl.float32)
    dc1x, dc1y, dc1z = tl.zeros((8,), dtype=tl.float32), tl.zeros((8,), dtype=tl.float32), tl.zeros((8,), dtype=tl.float32)
    dd1 = tl.zeros((8,), dtype=tl.float32)
    dc20 = tl.zeros((4,), dtype=tl.float32); dc21 = tl.zeros((4,), dtype=tl.float32); dc22 = tl.zeros((4,), dtype=tl.float32)
    dc23 = tl.zeros((4,), dtype=tl.float32); dc24 = tl.zeros((4,), dtype=tl.float32)
    dd2 = tl.zeros((4,), dtype=tl.float32)
    s2, is2, is6 = 1.4142135623730951, 0.7071067811865475, 0.4082482904638630
    j_vec = tl.full((BLOCK_E,), j, dtype=tl.int32)

    for kb in range(k0, k1, BLOCK_E):
        kp = kb + tl.arange(0, BLOCK_E)
        km = kp < k1
        e = tl.load(send_perm + kp, mask=km, other=0)
        i = tl.load(recv_idx + e, mask=km, other=0)
        mi = tl.load(mask_ptr + i, mask=km, other=0)
        shx, shy, shz, _, _, _ = _ov_load_shift(shift_ptr, cell_ptr, e, g_id, km, HAS_SHIFT, HAS_CELL)
        dx = pjx - tl.load(pos_ptr + i * 3 + 0, mask=km, other=0.0) + shx
        dy = pjy - tl.load(pos_ptr + i * 3 + 1, mask=km, other=0.0) + shy
        dz = pjz - tl.load(pos_ptr + i * 3 + 2, mask=km, other=0.0) + shz
        r2 = dx * dx + dy * dy + dz * dz
        valid = km & (mi != 0) & (r2 > 1e-12) & (r2 < r_cut2)
        r = tl.sqrt(r2 + 1e-12)
        inv_r = tl.where(valid, 1.0 / r, 0.0)
        ux, uy, uz = dx * inv_r, dy * inv_r, dz * inv_r
        chi, _ = _ov_c3_env(r, r_on, inv_w)
        chi = tl.where(valid, chi, 0.0)
        xf = tl.minimum(r, r_cut) * inv_h
        g = tl.minimum(tl.floor(xf).to(tl.int32), grid - 1)
        tf = (xf - g.to(tl.float32))[:, None]
        p_base = (4 * g[:, None]) * 44
        v2 = valid[:, None]

        fp = chi[:, None] * _ov_horner_val(poly_ptr, p_base, k16[None, :], tf, v2)
        dap += tl.sum(fp * tl.load(gp_ptr + i[:, None] * 16 + k16[None, :], mask=v2, other=0.0), axis=0)

        fdir = chi[:, None] * _ov_horner_val(poly_ptr, p_base, 16 + k8[None, :], tf, v2)
        ftr = chi[:, None] * _ov_horner_val(poly_ptr, p_base, 24 + k8[None, :], tf, v2)
        flon = chi[:, None] * _ov_horner_val(poly_ptr, p_base, 32 + k8[None, :], tf, v2)
        i24 = i[:, None] * 24 + k8[None, :] * 3
        gvx = tl.load(gv_ptr + i24 + 0, mask=v2, other=0.0)
        gvy = tl.load(gv_ptr + i24 + 1, mask=v2, other=0.0)
        gvz = tl.load(gv_ptr + i24 + 2, mask=v2, other=0.0)
        gdsi = tl.load(gds_ptr + i[:, None] * 8 + k8[None, :], mask=v2, other=0.0)
        gvu = gvx * ux[:, None] + gvy * uy[:, None] + gvz * uz[:, None]
        wc = fdir * gdsi + flon * gvu
        dc1x += tl.sum(wc * ux[:, None] + ftr * gvx, axis=0)
        dc1y += tl.sum(wc * uy[:, None] + ftr * gvy, axis=0)
        dc1z += tl.sum(wc * uz[:, None] + ftr * gvz, axis=0)
        dd1 += tl.sum(fdir * gvu, axis=0)

        fy2 = chi[:, None] * _ov_horner_val(poly_ptr, p_base, 40 + k4[None, :], tf, v2)
        y0, y1, y2v = s2 * ux * uy, s2 * ux * uz, s2 * uy * uz
        y3, y4 = is2 * (ux * ux - uy * uy), is6 * (2.0 * uz * uz - ux * ux - uy * uy)
        i20 = i[:, None] * 20 + k4[None, :] * 5
        gysi = tl.load(gys_ptr + i[:, None] * 4 + k4[None, :], mask=v2, other=0.0)
        wc2 = fy2 * gysi
        dc20 += tl.sum(wc2 * y0[:, None], axis=0); dc21 += tl.sum(wc2 * y1[:, None], axis=0); dc22 += tl.sum(wc2 * y2v[:, None], axis=0)
        dc23 += tl.sum(wc2 * y3[:, None], axis=0); dc24 += tl.sum(wc2 * y4[:, None], axis=0)
        gyty = (
            tl.load(gyt_ptr + i20 + 0, mask=v2, other=0.0) * y0[:, None]
            + tl.load(gyt_ptr + i20 + 1, mask=v2, other=0.0) * y1[:, None]
            + tl.load(gyt_ptr + i20 + 2, mask=v2, other=0.0) * y2v[:, None]
            + tl.load(gyt_ptr + i20 + 3, mask=v2, other=0.0) * y3[:, None]
            + tl.load(gyt_ptr + i20 + 4, mask=v2, other=0.0) * y4[:, None]
        )
        dd2 += tl.sum(fy2 * gyty, axis=0)

        if NEED_DPOS:
            gdx, gdy, gdz = _ov_edge_grad_d(
                dx, dy, dz, r2, valid, r_cut, r_on, inv_w, inv_h, grid,
                poly_ptr, a_p_ptr, c1_ptr, d1_ptr, c2_ptr, d2_ptr,
                gp_ptr, gds_ptr, gv_ptr, gys_ptr, gyt_ptr, i, j_vec, BLOCK_E,
            )
            dpx += tl.sum(gdx, axis=0); dpy += tl.sum(gdy, axis=0); dpz += tl.sum(gdz, axis=0)

    if NEED_DPOS:
        tl.store(d_pos_ptr + j * 3 + 0, tl.load(d_pos_ptr + j * 3 + 0) + dpx)
        tl.store(d_pos_ptr + j * 3 + 1, tl.load(d_pos_ptr + j * 3 + 1) + dpy)
        tl.store(d_pos_ptr + j * 3 + 2, tl.load(d_pos_ptr + j * 3 + 2) + dpz)
    tl.store(d_ap_ptr + j * 16 + k16, dap)
    tl.store(d_c1_ptr + j * 24 + k8 * 3 + 0, dc1x); tl.store(d_c1_ptr + j * 24 + k8 * 3 + 1, dc1y); tl.store(d_c1_ptr + j * 24 + k8 * 3 + 2, dc1z)
    tl.store(d_d1_ptr + j * 8 + k8, dd1)
    tl.store(d_c2_ptr + j * 20 + k4 * 5 + 0, dc20); tl.store(d_c2_ptr + j * 20 + k4 * 5 + 1, dc21); tl.store(d_c2_ptr + j * 20 + k4 * 5 + 2, dc22)
    tl.store(d_c2_ptr + j * 20 + k4 * 5 + 3, dc23); tl.store(d_c2_ptr + j * 20 + k4 * 5 + 4, dc24)
    tl.store(d_d2_ptr + j * 4 + k4, dd2)


@torch.library.custom_op("lam4::triton_cloud_ov_fwd", mutates_args=())
def _triton_cloud_ov_fwd_op(
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
    grid = int(poly.shape[0])
    has_sh, has_cell = shift_in.numel() > 0, cell.numel() > 0
    is_g1 = bool(has_cell and cell.shape[0] == 1)
    df, di = dummy_f32(dev), dummy_i64(dev)
    _cloud_ov_fwd_recv_kernel[(n,)](
        pos.contiguous(), recv_ptr, send_idx, atom_mask_u8.contiguous(),
        shift_in.contiguous() if has_sh else df, cell.contiguous() if has_cell else df,
        graph.contiguous() if graph.numel() else di, poly.contiguous(),
        a_p.contiguous(), c1.contiguous(), d1.contiguous(), c2.contiguous(), d2.contiguous(),
        out_ap, out_ds, out_v, out_ys, out_yt,
        n, grid, float(r_cut), float(r_on), 1.0 / (float(r_cut) - float(r_on)), float(grid) / float(r_cut),
        HAS_SHIFT=has_sh, HAS_CELL=has_cell, IS_G1=is_g1, BLOCK_E=16, num_warps=4,
    )
    return out_ap, out_ds, out_v, out_ys, out_yt


@_triton_cloud_ov_fwd_op.register_fake
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


@torch.library.custom_op("lam4::triton_cloud_ov_bwd", mutates_args=())
def _triton_cloud_ov_bwd_op(
    pos: Tensor, poly: Tensor, a_p: Tensor, c1: Tensor, d1: Tensor, c2: Tensor, d2: Tensor,
    atom_mask_u8: Tensor, recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor,
    gp: Tensor, gds: Tensor, gv: Tensor, gys: Tensor, gyt: Tensor,
    r_cut: float, r_on: float, need_dpos: bool, need_dcell: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    n, n_e = int(pos.shape[0]), int(send_idx.shape[0])
    dev = pos.device
    d_pos = torch.zeros((n, 3), dtype=torch.float32, device=dev)
    d_cell = torch.zeros(cell.shape, dtype=torch.float32, device=dev) if cell.numel() > 0 else torch.zeros((0, 3, 3), dtype=torch.float32, device=dev)
    d_ap = torch.zeros(a_p.shape, dtype=torch.float32, device=dev)
    d_c1 = torch.zeros(c1.shape, dtype=torch.float32, device=dev)
    d_d1 = torch.zeros(d1.shape, dtype=torch.float32, device=dev)
    d_c2 = torch.zeros(c2.shape, dtype=torch.float32, device=dev)
    d_d2 = torch.zeros(d2.shape, dtype=torch.float32, device=dev)
    if n == 0 or n_e == 0:
        return d_pos, d_cell, d_ap, d_c1, d_d1, d_c2, d_d2
    grid = int(poly.shape[0])
    has_sh, has_cell = shift_in.numel() > 0, cell.numel() > 0
    is_g1 = bool(has_cell and cell.shape[0] == 1)
    need_vir = bool(need_dcell and has_cell and has_sh)
    df, di = dummy_f32(dev), dummy_i64(dev)
    pos_c, poly_c, mask_c = pos.contiguous(), poly.contiguous(), atom_mask_u8.contiguous()
    sh_c = shift_in.contiguous() if has_sh else df
    cell_c = cell.contiguous() if has_cell else df
    g_c = graph.contiguous() if graph.numel() else di
    ap_c, c1_c, d1_c, c2_c, d2_c = a_p.contiguous(), c1.contiguous(), d1.contiguous(), c2.contiguous(), d2.contiguous()
    gp_c, gds_c, gv_c, gys_c, gyt_c = gp.contiguous(), gds.contiguous(), gv.contiguous(), gys.contiguous(), gyt.contiguous()
    inv_w, inv_h = 1.0 / (float(r_cut) - float(r_on)), float(grid) / float(r_cut)
    vir_atom = torch.zeros((n, 9), dtype=torch.float32, device=dev) if need_vir else df
    if need_dpos or need_vir:
        _cloud_ov_bwd_recv_kernel[(n,)](
            pos_c, recv_ptr, send_idx, mask_c, sh_c, cell_c, g_c, poly_c,
            ap_c, c1_c, d1_c, c2_c, d2_c, gp_c, gds_c, gv_c, gys_c, gyt_c,
            d_pos, vir_atom, n, grid, float(r_cut), float(r_on), inv_w, inv_h,
            HAS_SHIFT=has_sh, HAS_CELL=has_cell, IS_G1=is_g1, NEED_DPOS=need_dpos, NEED_VIR=need_vir, BLOCK_E=16, num_warps=4,
        )
    _cloud_ov_bwd_send_kernel[(n,)](
        pos_c, send_ptr, send_perm, recv_idx, mask_c, sh_c, cell_c, g_c, poly_c,
        ap_c, c1_c, d1_c, c2_c, d2_c, gp_c, gds_c, gv_c, gys_c, gyt_c,
        d_pos, d_ap, d_c1, d_d1, d_c2, d_d2, n, grid, float(r_cut), float(r_on), inv_w, inv_h,
        HAS_SHIFT=has_sh, HAS_CELL=has_cell, IS_G1=is_g1, NEED_DPOS=need_dpos, BLOCK_E=16, num_warps=4,
    )
    if need_vir:
        v33 = vir_atom.view(n, 3, 3)
        if cell.shape[0] == 1:
            d_cell[0] = v33.sum(dim=0)
        else:
            d_cell.index_add_(0, graph, v33)
    return d_pos, d_cell, d_ap, d_c1, d_d1, d_c2, d_d2


@_triton_cloud_ov_bwd_op.register_fake
def _(
    pos: Tensor, poly: Tensor, a_p: Tensor, c1: Tensor, d1: Tensor, c2: Tensor, d2: Tensor,
    atom_mask_u8: Tensor, recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor,
    gp: Tensor, gds: Tensor, gv: Tensor, gys: Tensor, gyt: Tensor,
    r_cut: float, r_on: float, need_dpos: bool, need_dcell: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dev = pos.device
    d_cell = torch.empty(cell.shape, dtype=torch.float32, device=dev) if cell.numel() > 0 else torch.empty((0, 3, 3), dtype=torch.float32, device=dev)
    return (
        torch.empty(pos.shape, dtype=torch.float32, device=dev), d_cell,
        torch.empty(a_p.shape, dtype=torch.float32, device=dev), torch.empty(c1.shape, dtype=torch.float32, device=dev),
        torch.empty(d1.shape, dtype=torch.float32, device=dev), torch.empty(c2.shape, dtype=torch.float32, device=dev),
        torch.empty(d2.shape, dtype=torch.float32, device=dev),
    )


def _setup_tov_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    ctx.save_for_backward(*inputs[:16])
    ctx.r_cut, ctx.r_on = float(inputs[16]), float(inputs[17])


def _bwd_tov_autograd(ctx: Any, gp: Tensor | None, gds: Tensor | None, gv: Tensor | None, gys: Tensor | None, gyt: Tensor | None):
    saved = ctx.saved_tensors
    pos, poly, a_p, c1, d1, c2, d2, mask_u8, rp, si, ri, sp, pm, sh, cell, graph = saved
    need_dp = bool(ctx.needs_input_grad[0])
    need_dc = bool(ctx.needs_input_grad[14])
    if not (need_dp or need_dc or any(ctx.needs_input_grad[2:7])):
        return (None,) * 18
    gp_ = gp.float().contiguous() if gp is not None else torch.zeros_like(a_p, dtype=torch.float32)
    gds_ = gds.float().contiguous() if gds is not None else torch.zeros_like(d1, dtype=torch.float32)
    gv_ = gv.float().contiguous() if gv is not None else torch.zeros_like(c1, dtype=torch.float32)
    gys_ = gys.float().contiguous() if gys is not None else torch.zeros_like(d2, dtype=torch.float32)
    gyt_ = gyt.float().contiguous() if gyt is not None else torch.zeros_like(c2, dtype=torch.float32)
    d_pos, d_cell, d_ap, d_c1, d_d1, d_c2, d_d2 = _triton_cloud_ov_bwd_op(
        pos, poly, a_p, c1, d1, c2, d2, mask_u8, rp, si, ri, sp, pm, sh, cell, graph,
        gp_, gds_, gv_, gys_, gyt_, ctx.r_cut, ctx.r_on, need_dp, need_dc,
    )
    return (
        d_pos.to(pos.dtype) if need_dp else None,
        None,
        d_ap.to(a_p.dtype) if ctx.needs_input_grad[2] else None,
        d_c1.to(c1.dtype) if ctx.needs_input_grad[3] else None,
        d_d1.to(d1.dtype) if ctx.needs_input_grad[4] else None,
        d_c2.to(c2.dtype) if ctx.needs_input_grad[5] else None,
        d_d2.to(d2.dtype) if ctx.needs_input_grad[6] else None,
        None, None, None, None, None, None, None,
        d_cell.to(cell.dtype) if need_dc else None,
        None, None, None,
    )


def _tov_bwd2_unsupported(*_args: Any, **_kw: Any):
    raise NotImplementedError("lam4::triton_cloud_ov_bwd supports 1st-order backward only; route create_graph=True to PyTorch.")


_triton_cloud_ov_fwd_op.register_autograd(_bwd_tov_autograd, setup_context=_setup_tov_ctx)
_triton_cloud_ov_bwd_op.register_autograd(_tov_bwd2_unsupported, setup_context=lambda ctx, inputs, output: None)


def triton_cloud_overlap_csr(
    pos: Tensor, poly: Tensor, a_p: Tensor, c1: Tensor, d1: Tensor, c2: Tensor, d2: Tensor,
    atom_mask: Tensor, csr_plan: CSRPlan, shift_in: Tensor | None, cell: Tensor | None,
    graph: Tensor | None, r_cut: float, r_on: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dev = pos.device
    sh_t = shift_in.to(dtype=torch.float32) if (shift_in is not None and shift_in.numel() > 0) else torch.empty((0, 3), dtype=torch.float32, device=dev)
    cell_t = cell.to(dtype=torch.float32) if (cell is not None and cell.numel() > 0) else torch.empty((0, 3, 3), dtype=torch.float32, device=dev)
    graph_t = graph.to(dtype=torch.int64) if (graph is not None and graph.numel() > 0) else torch.empty(0, dtype=torch.int64, device=dev)
    return _triton_cloud_ov_fwd_op(
        pos.to(dtype=torch.float32), poly.to(dtype=torch.float32),
        a_p.to(dtype=torch.float32), c1.to(dtype=torch.float32), d1.to(dtype=torch.float32),
        c2.to(dtype=torch.float32), d2.to(dtype=torch.float32),
        atom_mask.to(dtype=torch.uint8),
        csr_plan.recv_ptr, csr_plan.send_idx, csr_plan.recv_idx, csr_plan.send_ptr, csr_plan.send_perm,
        sh_t, cell_t, graph_t, float(r_cut), float(r_on),
    )
