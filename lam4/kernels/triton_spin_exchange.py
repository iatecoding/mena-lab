"""Dual-CSR Triton CUDA custom ops & autograd for SpinExchangeHead (zero [E, C] edge buffers)."""
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


def has_triton_spin_exch(device: torch.device | str | None = None) -> bool:
    if os.environ.get("LAM4_TRITON_SPIN_EXCH", "1") == "0":
        return False
    return bool(_HAS_TRITON and cuda_cap_ok(device))


def can_use_triton_spin_exch(
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
        and pos.device.type == "cuda"
        and has_triton_spin_exch(pos.device)
        and not w_dir.requires_grad
        and not w_cov.requires_grad
        and not (shift_in is not None and shift_in.requires_grad)
        and int(rank) == 16
        and int(n_rbf) == 32
    )


@_jit
def _se_load_shift(shift_ptr, cell_ptr, e, g_id, em, HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr):
    if not HAS_SHIFT:
        z = e.to(tl.float32) * 0.0
        return z, z, z, z, z, z
    sx = tl.load(shift_ptr + e * 3 + 0, mask=em, other=0.0)
    sy = tl.load(shift_ptr + e * 3 + 1, mask=em, other=0.0)
    sz = tl.load(shift_ptr + e * 3 + 2, mask=em, other=0.0)
    if not HAS_CELL:
        return sx, sy, sz, sx, sy, sz
    c = cell_ptr + g_id * 9
    dx = sx * tl.load(c + 0) + sy * tl.load(c + 3) + sz * tl.load(c + 6)
    dy = sx * tl.load(c + 1) + sy * tl.load(c + 4) + sz * tl.load(c + 7)
    dz = sx * tl.load(c + 2) + sy * tl.load(c + 5) + sz * tl.load(c + 8)
    return dx, dy, dz, sx, sy, sz


@_jit
def _spin_exch_fwd_recv_kernel(
    pos_ptr, recv_ptr, send_idx, mask_ptr, shift_ptr, cell_ptr, graph_ptr,
    h_dir_ptr, h_cov_ptr, m_ptr, u_ptr, w_dir_t_ptr, w_cov_t_ptr, centres_ptr,
    out_pair_ptr, n_atoms, r_cut, r_cut2, inv_sig,
    HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr, BLOCK_E: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= n_atoms or tl.load(mask_ptr + i) == 0:
        return
    g_id = 0 if (not HAS_CELL or IS_G1) else tl.load(graph_ptr + i).to(tl.int32)
    pix, piy, piz = tl.load(pos_ptr + i * 3 + 0), tl.load(pos_ptr + i * 3 + 1), tl.load(pos_ptr + i * 3 + 2)
    mi, ui = tl.load(m_ptr + i), tl.load(u_ptr + i)
    k16, b32 = tl.arange(0, 16), tl.arange(0, 32)
    hdi = tl.load(h_dir_ptr + i * 16 + k16)
    hci = tl.load(h_cov_ptr + i * 16 + k16)
    centres = tl.load(centres_ptr + b32)
    wd_t = tl.load(w_dir_t_ptr + b32[:, None] * 16 + k16[None, :])
    wc_t = tl.load(w_cov_t_ptr + b32[:, None] * 16 + k16[None, :])
    e0, e1 = tl.load(recv_ptr + i), tl.load(recv_ptr + i + 1)
    acc = 0.0
    pi_f = 3.14159265358979323846

    for eb in range(e0, e1, BLOCK_E):
        e = eb + tl.arange(0, BLOCK_E)
        em = e < e1
        j = tl.load(send_idx + e, mask=em, other=0)
        mj_m = tl.load(mask_ptr + j, mask=em, other=0)
        shx, shy, shz, _, _, _ = _se_load_shift(shift_ptr, cell_ptr, e, g_id, em, HAS_SHIFT, HAS_CELL)
        dx = tl.load(pos_ptr + j * 3 + 0, mask=em, other=0.0) - pix + shx
        dy = tl.load(pos_ptr + j * 3 + 1, mask=em, other=0.0) - piy + shy
        dz = tl.load(pos_ptr + j * 3 + 2, mask=em, other=0.0) - piz + shz
        r2 = dx * dx + dy * dy + dz * dz
        valid = em & (mj_m != 0) & (r2 > 1e-12) & (r2 < r_cut2)
        r = tl.sqrt(r2 + 1e-12)
        t = tl.minimum(tl.maximum(r / r_cut, 0.0), 1.0)
        chi = tl.where(valid, 0.5 * (1.0 + tl.cos(pi_f * t)), 0.0)
        rc = tl.minimum(r, r_cut)
        zb = (rc[:, None] - centres[None, :]) * inv_sig
        pb = tl.where((tl.abs(zb) < 9.0) & valid[:, None], tl.exp(-0.5 * zb * zb), 0.0)
        wd = tl.dot(pb, wd_t, allow_tf32=False)
        wc = tl.dot(pb, wc_t, allow_tf32=False)
        v2 = valid[:, None]
        hdj = tl.load(h_dir_ptr + j[:, None] * 16 + k16[None, :], mask=v2, other=0.0)
        hcj = tl.load(h_cov_ptr + j[:, None] * 16 + k16[None, :], mask=v2, other=0.0)
        j_dir = tl.sum(wd * (hdi[None, :] * hdj), axis=1)
        j_cov = tl.sum(wc * (hci[None, :] * hcj), axis=1)
        mp = tl.load(m_ptr + j, mask=valid, other=0.0) * mi
        up = tl.load(u_ptr + j, mask=valid, other=0.0) * ui
        acc += tl.sum(chi * (j_dir * mp + j_cov * (mp - up)), axis=0)
    tl.store(out_pair_ptr + i, acc)


@_jit
def _spin_exch_bwd_recv_kernel(
    pos_ptr, recv_ptr, send_idx, mask_ptr, shift_ptr, cell_ptr, graph_ptr,
    h_dir_ptr, h_cov_ptr, m_ptr, u_ptr, w_dir_t_ptr, w_cov_t_ptr, centres_ptr, g_out_ptr,
    d_pos_ptr, d_hdir_ptr, d_hcov_ptr, d_m_ptr, d_u_ptr, vir_atom_ptr,
    n_atoms, r_cut, r_cut2, inv_sig,
    HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr,
    NEED_DPOS: tl.constexpr, NEED_VIR: tl.constexpr, BLOCK_E: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= n_atoms or tl.load(mask_ptr + i) == 0:
        return
    g_id = 0 if (not HAS_CELL or IS_G1) else tl.load(graph_ptr + i).to(tl.int32)
    pix, piy, piz = tl.load(pos_ptr + i * 3 + 0), tl.load(pos_ptr + i * 3 + 1), tl.load(pos_ptr + i * 3 + 2)
    mi, ui, gi = tl.load(m_ptr + i), tl.load(u_ptr + i), tl.load(g_out_ptr + i)
    k16, b32 = tl.arange(0, 16), tl.arange(0, 32)
    hdi = tl.load(h_dir_ptr + i * 16 + k16)
    hci = tl.load(h_cov_ptr + i * 16 + k16)
    centres = tl.load(centres_ptr + b32)
    wd_t = tl.load(w_dir_t_ptr + b32[:, None] * 16 + k16[None, :])
    wc_t = tl.load(w_cov_t_ptr + b32[:, None] * 16 + k16[None, :])
    e0, e1 = tl.load(recv_ptr + i), tl.load(recv_ptr + i + 1)
    dpx, dpy, dpz, dm_i, du_i = 0.0, 0.0, 0.0, 0.0, 0.0
    dhd_i, dhc_i = tl.zeros((16,), dtype=tl.float32), tl.zeros((16,), dtype=tl.float32)
    v00 = 0.0; v01 = 0.0; v02 = 0.0; v10 = 0.0; v11 = 0.0; v12 = 0.0; v20 = 0.0; v21 = 0.0; v22 = 0.0
    pi_f = 3.14159265358979323846

    for eb in range(e0, e1, BLOCK_E):
        e = eb + tl.arange(0, BLOCK_E)
        em = e < e1
        j = tl.load(send_idx + e, mask=em, other=0)
        mj_m = tl.load(mask_ptr + j, mask=em, other=0)
        shx, shy, shz, sx, sy, sz = _se_load_shift(shift_ptr, cell_ptr, e, g_id, em, HAS_SHIFT, HAS_CELL)
        dx = tl.load(pos_ptr + j * 3 + 0, mask=em, other=0.0) - pix + shx
        dy = tl.load(pos_ptr + j * 3 + 1, mask=em, other=0.0) - piy + shy
        dz = tl.load(pos_ptr + j * 3 + 2, mask=em, other=0.0) - piz + shz
        r2 = dx * dx + dy * dy + dz * dz
        valid = em & (mj_m != 0) & (r2 > 1e-12) & (r2 < r_cut2)
        r = tl.sqrt(r2 + 1e-12)
        t = tl.minimum(tl.maximum(r / r_cut, 0.0), 1.0)
        chi = tl.where(valid, 0.5 * (1.0 + tl.cos(pi_f * t)), 0.0)
        dchi = tl.where(valid & (t < 1.0), -0.5 * pi_f / r_cut * tl.sin(pi_f * t), 0.0)
        rc = tl.minimum(r, r_cut)
        zb = (rc[:, None] - centres[None, :]) * inv_sig
        pb = tl.where((tl.abs(zb) < 9.0) & valid[:, None], tl.exp(-0.5 * zb * zb), 0.0)
        wd = tl.dot(pb, wd_t, allow_tf32=False)
        wc = tl.dot(pb, wc_t, allow_tf32=False)
        v2 = valid[:, None]
        hdj = tl.load(h_dir_ptr + j[:, None] * 16 + k16[None, :], mask=v2, other=0.0)
        hcj = tl.load(h_cov_ptr + j[:, None] * 16 + k16[None, :], mask=v2, other=0.0)
        hh_d, hh_c = hdi[None, :] * hdj, hci[None, :] * hcj
        j_dir, j_cov = tl.sum(wd * hh_d, axis=1), tl.sum(wc * hh_c, axis=1)
        mj = tl.load(m_ptr + j, mask=valid, other=0.0)
        uj = tl.load(u_ptr + j, mask=valid, other=0.0)
        mp, up = mj * mi, uj * ui
        c_dir, c_cov = gi * chi * mp, gi * chi * (mp - up)
        dhd_i += tl.sum(c_dir[:, None] * wd * hdj, axis=0)
        dhc_i += tl.sum(c_cov[:, None] * wc * hcj, axis=0)
        dm_i += tl.sum(gi * chi * (j_dir + j_cov) * mj, axis=0)
        du_i += tl.sum(gi * chi * (-j_cov) * uj, axis=0)
        if NEED_DPOS or NEED_VIR:
            dpb = tl.where((r < r_cut)[:, None], -pb * zb * inv_sig, 0.0)
            dj_dir_dr = tl.sum(tl.dot(dpb, wd_t, allow_tf32=False) * hh_d, axis=1)
            dj_cov_dr = tl.sum(tl.dot(dpb, wc_t, allow_tf32=False) * hh_c, axis=1)
            dmsg_dr = dchi * (j_dir * mp + j_cov * (mp - up)) + chi * (dj_dir_dr * mp + dj_cov_dr * (mp - up))
            fac = tl.where(valid, gi * dmsg_dr / r, 0.0)
            gdx, gdy, gdz = fac * dx, fac * dy, fac * dz
            if NEED_DPOS:
                dpx -= tl.sum(gdx, axis=0); dpy -= tl.sum(gdy, axis=0); dpz -= tl.sum(gdz, axis=0)
            if NEED_VIR:
                v00 += tl.sum(sx * gdx, axis=0); v01 += tl.sum(sx * gdy, axis=0); v02 += tl.sum(sx * gdz, axis=0)
                v10 += tl.sum(sy * gdx, axis=0); v11 += tl.sum(sy * gdy, axis=0); v12 += tl.sum(sy * gdz, axis=0)
                v20 += tl.sum(sz * gdx, axis=0); v21 += tl.sum(sz * gdy, axis=0); v22 += tl.sum(sz * gdz, axis=0)

    if NEED_DPOS:
        tl.store(d_pos_ptr + i * 3 + 0, dpx); tl.store(d_pos_ptr + i * 3 + 1, dpy); tl.store(d_pos_ptr + i * 3 + 2, dpz)
    tl.store(d_hdir_ptr + i * 16 + k16, dhd_i)
    tl.store(d_hcov_ptr + i * 16 + k16, dhc_i)
    tl.store(d_m_ptr + i, dm_i)
    tl.store(d_u_ptr + i, du_i)
    if NEED_VIR:
        b = vir_atom_ptr + i * 9
        tl.store(b + 0, v00); tl.store(b + 1, v01); tl.store(b + 2, v02)
        tl.store(b + 3, v10); tl.store(b + 4, v11); tl.store(b + 5, v12)
        tl.store(b + 6, v20); tl.store(b + 7, v21); tl.store(b + 8, v22)


@_jit
def _spin_exch_bwd_send_kernel(
    pos_ptr, send_ptr, send_perm, recv_idx, mask_ptr, shift_ptr, cell_ptr, graph_ptr,
    h_dir_ptr, h_cov_ptr, m_ptr, u_ptr, w_dir_t_ptr, w_cov_t_ptr, centres_ptr, g_out_ptr,
    d_pos_ptr, d_hdir_ptr, d_hcov_ptr, d_m_ptr, d_u_ptr,
    n_atoms, r_cut, r_cut2, inv_sig,
    HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr,
    NEED_DPOS: tl.constexpr, BLOCK_E: tl.constexpr,
):
    j = tl.program_id(0)
    if j >= n_atoms or tl.load(mask_ptr + j) == 0:
        return
    g_id = 0 if (not HAS_CELL or IS_G1) else tl.load(graph_ptr + j).to(tl.int32)
    pjx, pjy, pjz = tl.load(pos_ptr + j * 3 + 0), tl.load(pos_ptr + j * 3 + 1), tl.load(pos_ptr + j * 3 + 2)
    mj, uj = tl.load(m_ptr + j), tl.load(u_ptr + j)
    k16, b32 = tl.arange(0, 16), tl.arange(0, 32)
    hdj = tl.load(h_dir_ptr + j * 16 + k16)
    hcj = tl.load(h_cov_ptr + j * 16 + k16)
    centres = tl.load(centres_ptr + b32)
    wd_t = tl.load(w_dir_t_ptr + b32[:, None] * 16 + k16[None, :])
    wc_t = tl.load(w_cov_t_ptr + b32[:, None] * 16 + k16[None, :])
    k0, k1 = tl.load(send_ptr + j), tl.load(send_ptr + j + 1)
    dpx, dpy, dpz, dm_j, du_j = 0.0, 0.0, 0.0, 0.0, 0.0
    dhd_j, dhc_j = tl.zeros((16,), dtype=tl.float32), tl.zeros((16,), dtype=tl.float32)
    pi_f = 3.14159265358979323846

    for kb in range(k0, k1, BLOCK_E):
        kp = kb + tl.arange(0, BLOCK_E)
        km = kp < k1
        e = tl.load(send_perm + kp, mask=km, other=0)
        i = tl.load(recv_idx + e, mask=km, other=0)
        mi_m = tl.load(mask_ptr + i, mask=km, other=0)
        shx, shy, shz, _, _, _ = _se_load_shift(shift_ptr, cell_ptr, e, g_id, km, HAS_SHIFT, HAS_CELL)
        dx = pjx - tl.load(pos_ptr + i * 3 + 0, mask=km, other=0.0) + shx
        dy = pjy - tl.load(pos_ptr + i * 3 + 1, mask=km, other=0.0) + shy
        dz = pjz - tl.load(pos_ptr + i * 3 + 2, mask=km, other=0.0) + shz
        r2 = dx * dx + dy * dy + dz * dz
        valid = km & (mi_m != 0) & (r2 > 1e-12) & (r2 < r_cut2)
        r = tl.sqrt(r2 + 1e-12)
        t = tl.minimum(tl.maximum(r / r_cut, 0.0), 1.0)
        chi = tl.where(valid, 0.5 * (1.0 + tl.cos(pi_f * t)), 0.0)
        dchi = tl.where(valid & (t < 1.0), -0.5 * pi_f / r_cut * tl.sin(pi_f * t), 0.0)
        rc = tl.minimum(r, r_cut)
        zb = (rc[:, None] - centres[None, :]) * inv_sig
        pb = tl.where((tl.abs(zb) < 9.0) & valid[:, None], tl.exp(-0.5 * zb * zb), 0.0)
        wd = tl.dot(pb, wd_t, allow_tf32=False)
        wc = tl.dot(pb, wc_t, allow_tf32=False)
        v2 = valid[:, None]
        hdi = tl.load(h_dir_ptr + i[:, None] * 16 + k16[None, :], mask=v2, other=0.0)
        hci = tl.load(h_cov_ptr + i[:, None] * 16 + k16[None, :], mask=v2, other=0.0)
        hh_d, hh_c = hdi * hdj[None, :], hci * hcj[None, :]
        j_dir, j_cov = tl.sum(wd * hh_d, axis=1), tl.sum(wc * hh_c, axis=1)
        gi = tl.load(g_out_ptr + i, mask=valid, other=0.0)
        mi = tl.load(m_ptr + i, mask=valid, other=0.0)
        ui = tl.load(u_ptr + i, mask=valid, other=0.0)
        mp, up = mj * mi, uj * ui
        c_dir, c_cov = gi * chi * mp, gi * chi * (mp - up)
        dhd_j += tl.sum(c_dir[:, None] * wd * hdi, axis=0)
        dhc_j += tl.sum(c_cov[:, None] * wc * hci, axis=0)
        dm_j += tl.sum(gi * chi * (j_dir + j_cov) * mi, axis=0)
        du_j += tl.sum(gi * chi * (-j_cov) * ui, axis=0)
        if NEED_DPOS:
            dpb = tl.where((r < r_cut)[:, None], -pb * zb * inv_sig, 0.0)
            dj_dir_dr = tl.sum(tl.dot(dpb, wd_t, allow_tf32=False) * hh_d, axis=1)
            dj_cov_dr = tl.sum(tl.dot(dpb, wc_t, allow_tf32=False) * hh_c, axis=1)
            dmsg_dr = dchi * (j_dir * mp + j_cov * (mp - up)) + chi * (dj_dir_dr * mp + dj_cov_dr * (mp - up))
            fac = tl.where(valid, gi * dmsg_dr / r, 0.0)
            dpx += tl.sum(fac * dx, axis=0); dpy += tl.sum(fac * dy, axis=0); dpz += tl.sum(fac * dz, axis=0)

    if NEED_DPOS:
        tl.store(d_pos_ptr + j * 3 + 0, tl.load(d_pos_ptr + j * 3 + 0) + dpx)
        tl.store(d_pos_ptr + j * 3 + 1, tl.load(d_pos_ptr + j * 3 + 1) + dpy)
        tl.store(d_pos_ptr + j * 3 + 2, tl.load(d_pos_ptr + j * 3 + 2) + dpz)
    tl.store(d_hdir_ptr + j * 16 + k16, tl.load(d_hdir_ptr + j * 16 + k16) + dhd_j)
    tl.store(d_hcov_ptr + j * 16 + k16, tl.load(d_hcov_ptr + j * 16 + k16) + dhc_j)
    tl.store(d_m_ptr + j, tl.load(d_m_ptr + j) + dm_j)
    tl.store(d_u_ptr + j, tl.load(d_u_ptr + j) + du_j)


@torch.library.custom_op("lam4::triton_spin_exch_fwd", mutates_args=())
def _triton_spin_exch_fwd_op(
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
    has_sh, has_cell = shift_in.numel() > 0, cell.numel() > 0
    is_g1 = bool(has_cell and cell.shape[0] == 1)
    df, di = dummy_f32(dev), dummy_i64(dev)
    _spin_exch_fwd_recv_kernel[(n,)](
        pos.contiguous(), recv_ptr, send_idx, atom_mask_u8.contiguous(),
        shift_in.contiguous() if has_sh else df, cell.contiguous() if has_cell else df,
        graph.contiguous() if graph.numel() else di,
        h_dir.contiguous(), h_cov.contiguous(), m_t.contiguous(), u_rad.contiguous(),
        w_dir.transpose(0, 1).contiguous(), w_cov.transpose(0, 1).contiguous(), centres.contiguous(),
        out_pair, n, float(r_cut), float(r_cut) * float(r_cut), 1.0 / float(sigma),
        HAS_SHIFT=has_sh, HAS_CELL=has_cell, IS_G1=is_g1, BLOCK_E=16, num_warps=4,
    )
    return out_pair


@_triton_spin_exch_fwd_op.register_fake
def _(
    pos: Tensor, h_dir: Tensor, h_cov: Tensor, m_t: Tensor, u_rad: Tensor,
    w_dir: Tensor, w_cov: Tensor, centres: Tensor, atom_mask_u8: Tensor,
    recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor, r_cut: float, sigma: float,
) -> Tensor:
    return torch.empty(pos.shape[0], dtype=torch.float32, device=pos.device)


@torch.library.custom_op("lam4::triton_spin_exch_bwd", mutates_args=())
def _triton_spin_exch_bwd_op(
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
    d_cell = torch.zeros(cell.shape, dtype=torch.float32, device=dev) if cell.numel() > 0 else torch.zeros((0, 3, 3), dtype=torch.float32, device=dev)
    if n == 0 or n_e == 0:
        return d_pos, d_hdir, d_hcov, d_m, d_u, d_cell
    has_sh, has_cell = shift_in.numel() > 0, cell.numel() > 0
    is_g1 = bool(has_cell and cell.shape[0] == 1)
    need_vir = bool(need_dcell and has_cell and has_sh)
    df, di = dummy_f32(dev), dummy_i64(dev)
    pos_c, mask_c = pos.contiguous(), atom_mask_u8.contiguous()
    sh_c = shift_in.contiguous() if has_sh else df
    cell_c = cell.contiguous() if has_cell else df
    g_c = graph.contiguous() if graph.numel() else di
    hd_c, hc_c, m_c, u_c = h_dir.contiguous(), h_cov.contiguous(), m_t.contiguous(), u_rad.contiguous()
    wd_t, wc_t, cen_c = w_dir.transpose(0, 1).contiguous(), w_cov.transpose(0, 1).contiguous(), centres.contiguous()
    go_c = g_out.contiguous()
    vir_atom = torch.zeros((n, 9), dtype=torch.float32, device=dev) if need_vir else df
    r_cut_f, r_cut2_f, inv_sig_f = float(r_cut), float(r_cut) * float(r_cut), 1.0 / float(sigma)
    _spin_exch_bwd_recv_kernel[(n,)](
        pos_c, recv_ptr, send_idx, mask_c, sh_c, cell_c, g_c,
        hd_c, hc_c, m_c, u_c, wd_t, wc_t, cen_c, go_c,
        d_pos, d_hdir, d_hcov, d_m, d_u, vir_atom,
        n, r_cut_f, r_cut2_f, inv_sig_f,
        HAS_SHIFT=has_sh, HAS_CELL=has_cell, IS_G1=is_g1, NEED_DPOS=need_dpos, NEED_VIR=need_vir, BLOCK_E=16, num_warps=4,
    )
    _spin_exch_bwd_send_kernel[(n,)](
        pos_c, send_ptr, send_perm, recv_idx, mask_c, sh_c, cell_c, g_c,
        hd_c, hc_c, m_c, u_c, wd_t, wc_t, cen_c, go_c,
        d_pos, d_hdir, d_hcov, d_m, d_u,
        n, r_cut_f, r_cut2_f, inv_sig_f,
        HAS_SHIFT=has_sh, HAS_CELL=has_cell, IS_G1=is_g1, NEED_DPOS=need_dpos, BLOCK_E=16, num_warps=4,
    )
    if need_vir:
        v33 = vir_atom.view(n, 3, 3)
        if cell.shape[0] == 1:
            d_cell[0] = v33.sum(dim=0)
        else:
            d_cell.index_add_(0, graph, v33)
    return d_pos, d_hdir, d_hcov, d_m, d_u, d_cell


@_triton_spin_exch_bwd_op.register_fake
def _(
    pos: Tensor, h_dir: Tensor, h_cov: Tensor, m_t: Tensor, u_rad: Tensor,
    w_dir: Tensor, w_cov: Tensor, centres: Tensor, atom_mask_u8: Tensor,
    recv_ptr: Tensor, send_idx: Tensor, recv_idx: Tensor, send_ptr: Tensor, send_perm: Tensor,
    shift_in: Tensor, cell: Tensor, graph: Tensor, g_out: Tensor,
    r_cut: float, sigma: float, need_dpos: bool, need_dcell: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dev = pos.device
    d_cell = torch.empty(cell.shape, dtype=torch.float32, device=dev) if cell.numel() > 0 else torch.empty((0, 3, 3), dtype=torch.float32, device=dev)
    return (
        torch.empty(pos.shape, dtype=torch.float32, device=dev), torch.empty(h_dir.shape, dtype=torch.float32, device=dev),
        torch.empty(h_cov.shape, dtype=torch.float32, device=dev), torch.empty(m_t.shape, dtype=torch.float32, device=dev),
        torch.empty(u_rad.shape, dtype=torch.float32, device=dev), d_cell,
    )


def _setup_tse_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    ctx.save_for_backward(*inputs[:17])
    ctx.r_cut, ctx.sigma = float(inputs[17]), float(inputs[18])


def _bwd_tse_autograd(ctx: Any, g_out: Tensor | None):
    saved = ctx.saved_tensors
    pos, h_dir, h_cov, m_t, u_rad, w_dir, w_cov, centres, mask_u8, rp, si, ri, sp, pm, sh, cell, graph = saved
    need_dp = bool(ctx.needs_input_grad[0])
    need_dc = bool(ctx.needs_input_grad[15])
    if g_out is None or not (need_dp or need_dc or any(ctx.needs_input_grad[1:5])):
        return (None,) * 19
    d_pos, d_hd, d_hc, d_m, d_u, d_cell = _triton_spin_exch_bwd_op(
        pos, h_dir, h_cov, m_t, u_rad, w_dir, w_cov, centres, mask_u8, rp, si, ri, sp, pm, sh, cell, graph,
        g_out.float().contiguous(), ctx.r_cut, ctx.sigma, need_dp, need_dc,
    )
    return (
        d_pos.to(pos.dtype) if need_dp else None,
        d_hd.to(h_dir.dtype) if ctx.needs_input_grad[1] else None,
        d_hc.to(h_cov.dtype) if ctx.needs_input_grad[2] else None,
        d_m.to(m_t.dtype) if ctx.needs_input_grad[3] else None,
        d_u.to(u_rad.dtype) if ctx.needs_input_grad[4] else None,
        None, None, None, None, None, None, None, None, None, None,
        d_cell.to(cell.dtype) if need_dc else None,
        None, None, None,
    )


def _tse_bwd2_unsupported(*_args: Any, **_kw: Any):
    raise NotImplementedError("lam4::triton_spin_exch_bwd supports 1st-order backward only; route create_graph=True to PyTorch.")


_triton_spin_exch_fwd_op.register_autograd(_bwd_tse_autograd, setup_context=_setup_tse_ctx)
_triton_spin_exch_bwd_op.register_autograd(_tse_bwd2_unsupported, setup_context=lambda ctx, inputs, output: None)


def triton_spin_exchange_csr(
    pos: Tensor, h_dir: Tensor, h_cov: Tensor, m_t: Tensor, u_rad: Tensor,
    w_dir: Tensor, w_cov: Tensor, centres: Tensor, atom_mask: Tensor, csr_plan: CSRPlan,
    shift_in: Tensor | None, cell: Tensor | None, graph: Tensor | None, r_cut: float, sigma: float,
) -> Tensor:
    dev = pos.device
    sh_t = shift_in.to(dtype=torch.float32) if (shift_in is not None and shift_in.numel() > 0) else torch.empty((0, 3), dtype=torch.float32, device=dev)
    cell_t = cell.to(dtype=torch.float32) if (cell is not None and cell.numel() > 0) else torch.empty((0, 3, 3), dtype=torch.float32, device=dev)
    graph_t = graph.to(dtype=torch.int64) if (graph is not None and graph.numel() > 0) else torch.empty(0, dtype=torch.int64, device=dev)
    return _triton_spin_exch_fwd_op(
        pos.to(dtype=torch.float32), h_dir.to(dtype=torch.float32), h_cov.to(dtype=torch.float32),
        m_t.to(dtype=torch.float32), u_rad.to(dtype=torch.float32),
        w_dir.to(dtype=torch.float32), w_cov.to(dtype=torch.float32), centres.to(dtype=torch.float32),
        atom_mask.to(dtype=torch.uint8),
        csr_plan.recv_ptr, csr_plan.send_idx, csr_plan.recv_idx, csr_plan.send_ptr, csr_plan.send_perm,
        sh_t, cell_t, graph_t, float(r_cut), float(sigma),
    )
