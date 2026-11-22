"""Zero-sync CUDA Triton Dual-CSR NLH repulsion kernel (`nlh_atom_energy`) + shared NLH-floor/basis primitives."""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

from lam4.kernels.csr_plan import (
    MAX_DOFF_R2,
    CSRPlan,
    cacheable_tensor,
    dummy_f32,
    dummy_i64,
    norm_device,
    warm_csr_nlh_caches,
)
from lam4.physics.nlh_tables import nlh_tables_f64

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - optional on macOS / CPU environments
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

_jit = triton.jit if _HAS_TRITON else (lambda fn: fn)
_NLH_CPU_F32: tuple[Tensor, Tensor] | None = None
_NLH_TABS: dict[tuple[str, int | None], tuple[Tensor, Tensor]] = {}
NLH_BLOCK_N = 128


def has_triton_nlh(device: torch.device | str | None = None) -> bool:
    if os.environ.get("LAM4_TRITON_NLH", "1") == "0" or (device is not None and torch.device(device).type != "cuda"):
        return False
    from lam4.kernels.triton_csr import has_triton_csr

    return has_triton_csr()


def _nlh_cb_jet_f32(device: torch.device) -> tuple[Tensor, Tensor]:
    global _NLH_CPU_F32
    dev = norm_device(device)
    key = (dev.type, dev.index)
    t = _NLH_TABS.get(key)
    if t is None:
        if _NLH_CPU_F32 is None:
            cb64, _, jet64 = nlh_tables_f64()
            _NLH_CPU_F32 = (
                torch.from_numpy(cb64).to(dtype=torch.float32).contiguous(),
                torch.from_numpy(jet64).to(dtype=torch.float32).contiguous(),
            )
        cb_cpu, jet_cpu = _NLH_CPU_F32
        t = (cb_cpu, jet_cpu) if dev.type == "cpu" else (cb_cpu.to(dev), jet_cpu.to(dev))
        if cacheable_tensor(t[0]) and cacheable_tensor(t[1]):
            _NLH_TABS[key] = t
    return t


@_jit
def _ld3(p, off, m):
    return tl.load(p + off, mask=m, other=0.0), tl.load(p + off + 1, mask=m, other=0.0), tl.load(p + off + 2, mask=m, other=0.0)


@_jit
def _st3(p, off, x, y, z, m):
    tl.store(p + off + 0, x, mask=m)
    tl.store(p + off + 1, y, mask=m)
    tl.store(p + off + 2, z, mask=m)


@_jit
def _ld5(p, off, m):
    x0, x1, x2 = _ld3(p, off, m)
    return x0, x1, x2, tl.load(p + off + 3, mask=m, other=0.0), tl.load(p + off + 4, mask=m, other=0.0)


@_jit
def _st5(p, off, t0, t1, t2, t3, t4, m):
    _st3(p, off, t0, t1, t2, m)
    tl.store(p + off + 3, t3, mask=m)
    tl.store(p + off + 4, t4, mask=m)


@_jit
def _ld7(p, off, m):
    o0, o1, o2, o3, o4 = _ld5(p, off, m)
    return o0, o1, o2, o3, o4, tl.load(p + off + 5, mask=m, other=0.0), tl.load(p + off + 6, mask=m, other=0.0)


@_jit
def _st7(p, off, o0, o1, o2, o3, o4, o5, o6, m):
    _st5(p, off, o0, o1, o2, o3, o4, m)
    tl.store(p + off + 5, o5, mask=m)
    tl.store(p + off + 6, o6, mask=m)


@_jit
def _av3(p, cbo, o: tl.constexpr, a, b, c, am, IS_G1: tl.constexpr):
    if IS_G1:
        tl.atomic_add(p + o + 0, tl.sum(a, axis=0))
        tl.atomic_add(p + o + 1, tl.sum(b, axis=0))
        tl.atomic_add(p + o + 2, tl.sum(c, axis=0))
    else:
        tl.atomic_add(p + cbo + o + 0, a, mask=am)
        tl.atomic_add(p + cbo + o + 1, b, mask=am)
        tl.atomic_add(p + cbo + o + 2, c, mask=am)


@_jit
def _tanh_half(b):
    eab = tl.exp(-tl.abs(b))
    return tl.where(b >= 0.0, (1.0 - eab) / (1.0 + eab), (eab - 1.0) / (1.0 + eab))


@_jit
def _eval_y2_y3(ux, uy, uz):
    xx, yy, zz = ux * ux, uy * uy, uz * uz
    return (
        1.4142135623730951 * ux * uy, 1.4142135623730951 * ux * uz, 1.4142135623730951 * uy * uz,
        (xx - yy) * 0.7071067811865475, (2.0 * zz - xx - yy) * 0.4082482904638630,
        0.5 * uy * (3.0 * xx - yy), 2.449489742783178 * ux * uy * uz,
        0.3872983346207417 * uy * (4.0 * zz - xx - yy), 0.31622776601683794 * uz * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        0.3872983346207417 * ux * (4.0 * zz - xx - yy), 1.224744871391589 * uz * (xx - yy), 0.5 * ux * (xx - 3.0 * yy),
    )


@_jit
def _vjp_y2_y3(ux, uy, uz, dux, duy, duz, g20, g21, g22, g23, g24, g30, g31, g32, g33, g34, g35, g36):
    xx, yy, zz = ux * ux, uy * uy, uz * uz
    k6: tl.constexpr = 0.8164965809277260
    dux += 1.4142135623730951 * (g20 * uy + g21 * uz + g23 * ux) - k6 * g24 * ux + g30 * (3.0 * ux * uy) + g31 * (2.449489742783178 * uy * uz) + g32 * (-0.7745966692414834 * ux * uy) + g33 * (-1.8973665961010278 * ux * uz) + g34 * (0.3872983346207417 * (4.0 * zz - 3.0 * xx - yy)) + g35 * (2.449489742783178 * ux * uz) + g36 * (0.5 * (3.0 * xx - 3.0 * yy))
    duy += 1.4142135623730951 * (g20 * ux + g22 * uz - g23 * uy) - k6 * g24 * uy + g30 * (0.5 * (3.0 * xx - 3.0 * yy)) + g31 * (2.449489742783178 * ux * uz) + g32 * (0.3872983346207417 * (4.0 * zz - xx - 3.0 * yy)) + g33 * (-1.8973665961010278 * uy * uz) + g34 * (-0.7745966692414834 * ux * uy) + g35 * (-2.449489742783178 * uy * uz) + g36 * (-3.0 * ux * uy)
    duz += 1.4142135623730951 * (g21 * ux + g22 * uy) + 2.0 * k6 * g24 * uz + g31 * (2.449489742783178 * ux * uy) + g32 * (3.0983866769659336 * uy * uz) + g33 * (0.31622776601683794 * (6.0 * zz - 3.0 * xx - 3.0 * yy)) + g34 * (3.0983866769659336 * ux * uz) + g35 * (1.224744871391589 * (xx - yy))
    return dux, duy, duz


@_jit
def _nlh_shift(sh_p, cell_p, e, g_id, valid_e, HAS_CELL: tl.constexpr):
    esx, esy, esz = _ld3(sh_p, 3 * e, valid_e)
    if HAS_CELL:
        cb = 9 * g_id
        c00, c01, c02 = _ld3(cell_p, cb + 0, valid_e)
        c10, c11, c12 = _ld3(cell_p, cb + 3, valid_e)
        c20, c21, c22 = _ld3(cell_p, cb + 6, valid_e)
        sx, sy, sz = esx * c00 + esy * c10 + esz * c20, esx * c01 + esy * c11 + esz * c21, esx * c02 + esy * c12 + esz * c22
    else:
        sx, sy, sz = esx, esy, esz
    return sx, sy, sz, esx, esy, esz


@_jit
def _csr_geom(rx, ry, rz, sx_p, sy_p, sz_p, sh_p, cell_p, e, g_id, ve, z_p, rmin_p, z_peer, zi_s, rho_p, rho_peer, rho_self, grid, r_cut, r_on, inv_w, inv_h, rcut2, max_doff_r2, TD: tl.constexpr, HAS_CELL: tl.constexpr, HAS_FLOOR: tl.constexpr, HAS_RHO: tl.constexpr, NEED_GRAD: tl.constexpr, NEED_DRHO: tl.constexpr, IS_SEND: tl.constexpr, B: tl.constexpr):
    shx, shy, shz, esx, esy, esz = _nlh_shift(sh_p, cell_p, e, g_id, ve, HAS_CELL)
    dx, dy, dz = sx_p - rx + shx, sy_p - ry + shy, sz_p - rz + shz
    r2 = dx * dx + dy * dy + dz * dz
    ar = ve & (r2 >= 1e-24) & (r2 < rcut2)
    r_raw = tl.sqrt(tl.where(ar, r2, 1.0))
    inv_r = tl.where(ar, 1.0 / r_raw, 0.0)
    ux, uy, uz = dx * inv_r, dy * inv_r, dz * inv_r
    r_fl, drfl_x, drfl_y, drfl_z = r_raw, ux, uy, uz
    if HAS_FLOOR:
        nf = ar & (r2 < max_doff_r2)
        zp = tl.minimum(tl.maximum(tl.load(z_p + z_peer, mask=nf, other=0).to(tl.int32), 0), 127)
        pidx = ((zp << 7) + zi_s) if IS_SEND else (zi_s + zp)
        r_min = tl.load(rmin_p + pidx, mask=nf, other=0.0, eviction_policy="evict_last")
        d_off, r_eps = r_min + 0.05, tl.sqrt(tl.where(nf, r2, 0.0) + 1e-8)
        fon = nf & (r_eps < d_off) & (r_min > 0.0)
        e2 = tl.exp(-40.0 * tl.where(fon, d_off - r_eps, 0.0))
        th = (1.0 - e2) / (1.0 + e2)
        r_fl = tl.where(fon, d_off - 0.05 * th, r_fl)
        if NEED_GRAD:
            fsc = (1.0 - th * th) / r_eps
            drfl_x, drfl_y, drfl_z = tl.where(fon, dx * fsc, drfl_x), tl.where(fon, dy * fsc, drfl_y), tl.where(fon, dz * fsc, drfl_z)
    act = ar & (r_fl < r_cut)
    in_env = act & (r_fl > r_on)
    te = tl.where(in_env, (r_fl - r_on) * inv_w, 0.0)
    t2, om = te * te, 1.0 - te
    chi = tl.where(act, tl.where(in_env, 1.0 - t2 * t2 * (35.0 + te * (-84.0 + te * (70.0 - 20.0 * te))), 1.0), 0.0)
    dchi_drfl = tl.where(in_env, -140.0 * (t2 * te) * (om * om * om) * inv_w, 0.0) if NEED_GRAD else tl.zeros([B], tl.float32)
    x, dx_drfl, dx_drho = r_fl, (tl.where(act, 1.0, 0.0) if NEED_GRAD else tl.zeros([B], tl.float32)), tl.zeros([B], tl.float32)
    if HAS_RHO:
        rho_ij = tl.where(act, rho_self + tl.load(rho_p + rho_peer, mask=act, other=1.0), 1.0)
        sw = 1.52 / rho_ij
        iden = 1.0 / tl.where(act, 1.0 + (sw - 1.0) * (r_fl / r_cut), 1.0)
        x = tl.where(act, (sw * r_fl) * iden, 0.0)
        if NEED_GRAD:
            dx_drfl = tl.where(act, sw * (iden * iden), 0.0)
            if NEED_DRHO:
                dx_drho = tl.where(act, -(1.52 / (rho_ij * rho_ij)) * r_fl * (1.0 - r_fl / r_cut) * (iden * iden), 0.0)
    inv_he = tl.where(act & (x >= 0.0) & (x <= r_cut), inv_h, 0.0) if NEED_GRAD else tl.zeros([B], tl.float32)
    u_sp = tl.minimum(tl.maximum(x, 0.0), r_cut) * inv_h
    idx = tl.minimum(tl.maximum(u_sp.to(tl.int32), 0), grid - 1)
    return ux, uy, uz, inv_r, drfl_x, drfl_y, drfl_z, esx, esy, esz, act, chi, dchi_drfl, dx_drfl, dx_drho, inv_he, u_sp - idx.to(tl.float32), (idx * 4) * TD


@_jit
def _nlh_cb6(cb_p, pair_idx, act):
    c0 = tl.load(cb_p + 6 * pair_idx + 0, mask=act, other=0.0, eviction_policy="evict_last")
    c1 = tl.load(cb_p + 6 * pair_idx + 1, mask=act, other=0.0, eviction_policy="evict_last")
    c2 = tl.load(cb_p + 6 * pair_idx + 2, mask=act, other=0.0, eviction_policy="evict_last")
    b0 = tl.load(cb_p + 6 * pair_idx + 3, mask=act, other=0.0, eviction_policy="evict_last")
    b1 = tl.load(cb_p + 6 * pair_idx + 4, mask=act, other=0.0, eviction_policy="evict_last")
    b2 = tl.load(cb_p + 6 * pair_idx + 5, mask=act, other=0.0, eviction_policy="evict_last")
    return c0, c1, c2, b0, b1, b2


@_jit
def _nlh_grad_d(dx, dy, dz, r2, pair_idx, short, cb_p, jet_p):
    d_off = tl.load(jet_p + 4 * pair_idx + 0, mask=short, other=0.0, eviction_policy="evict_last")
    r = tl.sqrt(tl.where(short, r2, 0.0) + 1e-8)
    act = short & (d_off > 0.0) & (r < d_off)
    c0, c1, c2, b0, b1, b2 = _nlh_cb6(cb_p, pair_idx, act)
    v1 = tl.load(jet_p + 4 * pair_idx + 2, mask=act, other=0.0, eviction_policy="evict_last")
    v2 = tl.load(jet_p + 4 * pair_idx + 3, mask=act, other=0.0, eviction_policy="evict_last")
    e0, e1, e2 = c0 * tl.exp(-b0 * r), c1 * tl.exp(-b1 * r), c2 * tl.exp(-b2 * r)
    inv_r = 1.0 / r
    de_nlh = -(e0 * (b0 * r + 1.0) + e1 * (b1 * r + 1.0) + e2 * (b2 * r + 1.0)) * (inv_r * inv_r)
    sc = tl.where(act, 0.5 * (de_nlh - (v1 + (r - d_off) * v2)) * inv_r, 0.0)
    return sc * dx, sc * dy, sc * dz


@_jit
def _triton_nlh_fwd(pos_p, rptr_p, sidx_p, sh_p, cell_p, grp_p, z_p, cb_p, jet_p, out_p, n_atoms, max_doff_r2, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr, BLOCK_N: tl.constexpr):
    i = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = i < n_atoms
    zi_s = tl.minimum(tl.maximum(tl.load(z_p + i, mask=mask, other=0).to(tl.int32), 0), 127) << 7
    g_id = tl.zeros([BLOCK_N], tl.int32) if (not HAS_CELL or IS_G1) else tl.load(grp_p + i, mask=mask, other=0).to(tl.int32)
    px, py, pz = _ld3(pos_p, 3 * i, mask)
    acc = tl.zeros([BLOCK_N], tl.float32)
    e_beg, e_end = tl.load(rptr_p + i, mask=mask, other=0), tl.load(rptr_p + i + 1, mask=mask, other=0)
    e_end = tl.where(zi_s > 0, e_end, e_beg)
    for step in range(tl.max(tl.maximum(e_end - e_beg, 0), axis=0)):
        e = e_beg + step
        valid_e = mask & (e < e_end)
        j = tl.load(sidx_p + e, mask=valid_e, other=0)
        sx, sy, sz, _, _, _ = _nlh_shift(sh_p, cell_p, e, g_id, valid_e, HAS_CELL)
        pjx, pjy, pjz = _ld3(pos_p, 3 * j, valid_e)
        dx, dy, dz = pjx - px + sx, pjy - py + sy, pjz - pz + sz
        r2 = dx * dx + dy * dy + dz * dz
        short = valid_e & (r2 < max_doff_r2)
        p_idx = zi_s + tl.minimum(tl.maximum(tl.load(z_p + j, mask=short, other=0).to(tl.int32), 0), 127)
        d_off = tl.load(jet_p + 4 * p_idx + 0, mask=short, other=0.0, eviction_policy="evict_last")
        r = tl.sqrt(tl.where(short, r2, 0.0) + 1e-8)
        act = short & (d_off > 0.0) & (r < d_off)
        c0, c1, c2, b0, b1, b2 = _nlh_cb6(cb_p, p_idx, act)
        v0 = tl.load(jet_p + 4 * p_idx + 1, mask=act, other=0.0, eviction_policy="evict_last")
        v1 = tl.load(jet_p + 4 * p_idx + 2, mask=act, other=0.0, eviction_policy="evict_last")
        v2 = tl.load(jet_p + 4 * p_idx + 3, mask=act, other=0.0, eviction_policy="evict_last")
        e_nlh = (c0 * tl.exp(-b0 * r) + c1 * tl.exp(-b1 * r) + c2 * tl.exp(-b2 * r)) / r
        dr = r - d_off
        acc += tl.where(act, 0.5 * (e_nlh - (v0 + dr * (v1 + 0.5 * dr * v2))), 0.0)
    tl.store(out_p + i, acc, mask=mask)


@_jit
def _triton_nlh_bwd(pos_p, rptr_p, sidx_p, sptr_p, sperm_p, ridx_p, sh_p, cell_p, grp_p, z_p, cb_p, jet_p, g_out_p, d_pos_p, d_cell_p, n_atoms, max_doff_r2, HAS_CELL: tl.constexpr, IS_G1: tl.constexpr, NEED_VIR: tl.constexpr, BLOCK_N: tl.constexpr):
    a = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = a < n_atoms
    za = tl.minimum(tl.maximum(tl.load(z_p + a, mask=mask, other=0).to(tl.int32), 0), 127)
    za_s = za << 7
    g_a = tl.zeros([BLOCK_N], tl.int32) if (not HAS_CELL or IS_G1) else tl.load(grp_p + a, mask=mask, other=0).to(tl.int32)
    pa_x, pa_y, pa_z = _ld3(pos_p, 3 * a, mask)
    ga = tl.load(g_out_p + a, mask=mask, other=0.0)
    dpx, dpy, dpz = tl.zeros([BLOCK_N], tl.float32), tl.zeros([BLOCK_N], tl.float32), tl.zeros([BLOCK_N], tl.float32)
    v00, v01, v02 = tl.zeros([BLOCK_N], tl.float32), tl.zeros([BLOCK_N], tl.float32), tl.zeros([BLOCK_N], tl.float32)
    v10, v11, v12 = tl.zeros([BLOCK_N], tl.float32), tl.zeros([BLOCK_N], tl.float32), tl.zeros([BLOCK_N], tl.float32)
    v20, v21, v22 = tl.zeros([BLOCK_N], tl.float32), tl.zeros([BLOCK_N], tl.float32), tl.zeros([BLOCK_N], tl.float32)
    r_beg, r_end = tl.load(rptr_p + a, mask=mask, other=0), tl.load(rptr_p + a + 1, mask=mask, other=0)
    r_end = tl.where(za > 0, r_end, r_beg)
    for step in range(tl.max(tl.maximum(r_end - r_beg, 0), axis=0)):
        e = r_beg + step
        valid_e = mask & (e < r_end)
        j = tl.load(sidx_p + e, mask=valid_e, other=0)
        sx, sy, sz, esx, esy, esz = _nlh_shift(sh_p, cell_p, e, g_a, valid_e, HAS_CELL)
        pjx, pjy, pjz = _ld3(pos_p, 3 * j, valid_e)
        dx, dy, dz = pjx - pa_x + sx, pjy - pa_y + sy, pjz - pa_z + sz
        r2 = dx * dx + dy * dy + dz * dz
        short = valid_e & (r2 < max_doff_r2)
        zj = tl.minimum(tl.maximum(tl.load(z_p + j, mask=short, other=0).to(tl.int32), 0), 127)
        gx, gy, gz = _nlh_grad_d(dx, dy, dz, r2, za_s + zj, short, cb_p, jet_p)
        gdx, gdy, gdz = ga * gx, ga * gy, ga * gz
        dpx, dpy, dpz = dpx - gdx, dpy - gdy, dpz - gdz
        if NEED_VIR:
            v00, v01, v02 = v00 + esx * gdx, v01 + esx * gdy, v02 + esx * gdz
            v10, v11, v12 = v10 + esy * gdx, v11 + esy * gdy, v12 + esy * gdz
            v20, v21, v22 = v20 + esz * gdx, v21 + esz * gdy, v22 + esz * gdz
    s_beg, s_end = tl.load(sptr_p + a, mask=mask, other=0), tl.load(sptr_p + a + 1, mask=mask, other=0)
    s_end = tl.where(za > 0, s_end, s_beg)
    for step in range(tl.max(tl.maximum(s_end - s_beg, 0), axis=0)):
        kp = s_beg + step
        valid_e = mask & (kp < s_end)
        e = tl.load(sperm_p + kp, mask=valid_e, other=0)
        i = tl.load(ridx_p + e, mask=valid_e, other=0)
        g_i = tl.zeros([BLOCK_N], tl.int32) if (not HAS_CELL or IS_G1) else tl.load(grp_p + i, mask=valid_e, other=0).to(tl.int32)
        sx, sy, sz, _, _, _ = _nlh_shift(sh_p, cell_p, e, g_i, valid_e, HAS_CELL)
        pix, piy, piz = _ld3(pos_p, 3 * i, valid_e)
        dx, dy, dz = pa_x - pix + sx, pa_y - piy + sy, pa_z - piz + sz
        r2 = dx * dx + dy * dy + dz * dz
        short = valid_e & (r2 < max_doff_r2)
        zi = tl.minimum(tl.maximum(tl.load(z_p + i, mask=short, other=0).to(tl.int32), 0), 127)
        gi = tl.load(g_out_p + i, mask=short, other=0.0)
        gx, gy, gz = _nlh_grad_d(dx, dy, dz, r2, (zi << 7) + za, short, cb_p, jet_p)
        dpx, dpy, dpz = dpx + gi * gx, dpy + gi * gy, dpz + gi * gz
    _st3(d_pos_p, 3 * a, dpx, dpy, dpz, mask)
    if NEED_VIR:
        cbo = 9 * g_a
        _av3(d_cell_p, cbo, 0, v00, v01, v02, mask, IS_G1)
        _av3(d_cell_p, cbo, 3, v10, v11, v12, mask, IS_G1)
        _av3(d_cell_p, cbo, 6, v20, v21, v22, mask, IS_G1)


@torch.library.custom_op("lam4::triton_nlh", mutates_args=())
def _triton_nlh_op(recv_ptr: Tensor, send_idx: Tensor, send_ptr: Tensor, send_perm: Tensor, recv_idx: Tensor, edge_shift: Tensor, graph: Tensor, z: Tensor, pos: Tensor, cell: Tensor) -> Tensor:
    from lam4.kernels.triton_csr import HAS_TRITON

    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    dev, n = pos.device, int(pos.shape[0])
    if n == 0 or int(send_idx.shape[0]) == 0:
        return torch.zeros(n, dtype=torch.float32, device=dev)
    cb, jet = _nlh_cb_jet_f32(dev)
    has_cell = bool(cell.ndim == 3 and cell.numel() > 0)
    is_g1, df, di = bool(has_cell and cell.shape[0] == 1), dummy_f32(dev), dummy_i64(dev)
    out = torch.empty(n, dtype=torch.float32, device=dev)
    _triton_nlh_fwd[((n + NLH_BLOCK_N - 1) // NLH_BLOCK_N,)](
        pos, recv_ptr, send_idx, edge_shift, cell if has_cell else df, graph if graph.numel() else di,
        z, cb, jet, out, n, float(MAX_DOFF_R2), HAS_CELL=has_cell, IS_G1=is_g1, BLOCK_N=NLH_BLOCK_N, num_warps=4,
    )
    return out


@_triton_nlh_op.register_fake
def _(recv_ptr: Tensor, send_idx: Tensor, send_ptr: Tensor, send_perm: Tensor, recv_idx: Tensor, edge_shift: Tensor, graph: Tensor, z: Tensor, pos: Tensor, cell: Tensor) -> Tensor:
    return pos.new_empty(pos.shape[0])


@torch.library.custom_op("lam4::triton_nlh_bwd", mutates_args=())
def _triton_nlh_bwd_op(recv_ptr: Tensor, send_idx: Tensor, send_ptr: Tensor, send_perm: Tensor, recv_idx: Tensor, edge_shift: Tensor, graph: Tensor, z: Tensor, pos: Tensor, cell: Tensor, g_out: Tensor, need_vir: bool) -> tuple[Tensor, Tensor]:
    from lam4.kernels.triton_csr import HAS_TRITON

    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    dev, n = pos.device, int(pos.shape[0])
    has_cell = bool(cell.ndim == 3 and cell.numel() > 0)
    eff_vir = bool(need_vir and has_cell)
    if n == 0 or int(send_idx.shape[0]) == 0:
        return torch.zeros_like(pos), torch.zeros_like(cell) if eff_vir else pos.new_empty(0)
    cb, jet = _nlh_cb_jet_f32(dev)
    is_g1, df, di = bool(has_cell and cell.shape[0] == 1), dummy_f32(dev), dummy_i64(dev)
    d_pos, d_cell = torch.empty_like(pos), torch.zeros_like(cell) if eff_vir else df
    _triton_nlh_bwd[((n + NLH_BLOCK_N - 1) // NLH_BLOCK_N,)](
        pos, recv_ptr, send_idx, send_ptr, send_perm, recv_idx, edge_shift, cell if has_cell else df,
        graph if graph.numel() else di, z, cb, jet, g_out, d_pos, d_cell, n, float(MAX_DOFF_R2),
        HAS_CELL=has_cell, IS_G1=is_g1, NEED_VIR=eff_vir, BLOCK_N=NLH_BLOCK_N, num_warps=4,
    )
    return d_pos, d_cell if eff_vir else pos.new_empty(0)


@_triton_nlh_bwd_op.register_fake
def _(recv_ptr: Tensor, send_idx: Tensor, send_ptr: Tensor, send_perm: Tensor, recv_idx: Tensor, edge_shift: Tensor, graph: Tensor, z: Tensor, pos: Tensor, cell: Tensor, g_out: Tensor, need_vir: bool) -> tuple[Tensor, Tensor]:
    eff_vir = bool(need_vir and cell.ndim == 3)
    return torch.empty_like(pos), torch.empty_like(cell) if eff_vir else pos.new_empty(0)


def _setup_nlh_ctx(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    recv_ptr, send_idx, send_ptr, send_perm, recv_idx, edge_shift, graph, z, pos, cell = inputs
    ctx.has_cell = bool(cell.ndim == 3)
    ctx.save_for_backward(recv_ptr, send_idx, send_ptr, send_perm, recv_idx, edge_shift, graph, z, pos, cell)


def _autograd_nlh_bwd(ctx: Any, g_out: Tensor):
    recv_ptr, send_idx, send_ptr, send_perm, recv_idx, edge_shift, graph, z, pos, cell = ctx.saved_tensors
    needs = ctx.needs_input_grad
    need_vir = bool(needs[9] and ctx.has_cell)
    d_pos, d_cell = _triton_nlh_bwd_op(recv_ptr, send_idx, send_ptr, send_perm, recv_idx, edge_shift, graph, z, pos, cell, g_out.contiguous(), need_vir)
    return None, None, None, None, None, None, None, None, d_pos if needs[8] else None, d_cell if need_vir else None


torch.library.register_autograd("lam4::triton_nlh", _autograd_nlh_bwd, setup_context=_setup_nlh_ctx)


def triton_nlh_atom_energy(plan: CSRPlan, pos: Tensor, edge_shift: Tensor, cell: Tensor, graph: Tensor, z: Tensor) -> Tensor:
    if not has_triton_nlh(pos.device):
        raise RuntimeError("Triton NLH CSR is unavailable")
    g_c = graph.to(torch.int64).contiguous() if (cell.ndim == 3 and cell.shape[0] > 1) else pos.new_zeros(0, dtype=torch.int64)
    return _triton_nlh_op(
        plan.recv_ptr, plan.send_idx, plan.send_ptr, plan.send_perm, plan.recv_idx,
        edge_shift.to(torch.float32).contiguous(), g_c, z.to(torch.int64).contiguous(), pos.contiguous(), cell.contiguous(),
    )


def eval_nlh_decoupled(b, pos: Tensor | None = None, *, csr_plan: CSRPlan | None = None, compute_forces: bool = True, compute_stress: bool = False) -> tuple[Tensor, Tensor | None, Tensor | None]:
    from lam4.kernels.triton_coulomb_bwd2 import eval_nlh_decoupled_impl
    return eval_nlh_decoupled_impl(b, pos, csr_plan=csr_plan, compute_forces=compute_forces, compute_stress=compute_stress)
