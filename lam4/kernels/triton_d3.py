"""Atomic-free CUDA Triton kernels for Grimme D3 dispersion (`bj_wb97m` & `zero_pbe`, `LAM4_TRITON_D3=0` disables).
Exposes `triton_d3_grouped` (`O(N)` open molecules) and `triton_d3_csr` (`O(N+E)` 12 A CSR, periodic shifts + Cauchy virial stress).
"""
from __future__ import annotations

import os

import torch
from torch import Tensor

from lam.models.dispersion import ANGSTROM_TO_BOHR, D3BJ, HARTREE_TO_KCAL, WB97M_D3BJ
from lam4.core.consts import capturing, cuda_cap_ok, storable
from lam4.core.radial import norm_device
from lam4.kernels.triton_coulomb import HAS_TRITON, _jit, _rsqrt, build_group_ptr, triton, tl

_D3_TABLES: dict[torch.device, tuple[Tensor, Tensor, Tensor]] = {}
_CN_CUT_BOHR = float(11.2 * ANGSTROM_TO_BOHR)
_R_CUT, _TAPER_ON, _INV_TSPAN = 12.0, 9.0, float(1.0 / (12.0 - 9.0))
_MODE_PARAMS: dict[str, tuple[bool, float, float, float, float]] = {
    "bj_wb97m": (False, float(WB97M_D3BJ["s6"]), float(WB97M_D3BJ["s8"]), float(WB97M_D3BJ["a1"]), float(WB97M_D3BJ["a2"])),
    "zero_pbe": (True, 1.0, 0.722, 1.217, 14.0),
}


def has_triton_d3(dev: torch.device | str | None = None) -> bool:
    return bool(os.environ.get("LAM4_TRITON_D3", "1") != "0" and HAS_TRITON and cuda_cap_ok(dev))


def _get_d3_tables(dev: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    ndev = norm_device(dev)
    tabs = _D3_TABLES.get(ndev)
    if tabs is None:
        if capturing():
            raise RuntimeError("D3 tables not warmed before CUDA stream capture; call warm_d3_tables(dev) first")
        ref = D3BJ()
        tabs = (
            ref.rcov.to(device=ndev, dtype=torch.float32).contiguous(),
            ref.r2r4.to(device=ndev, dtype=torch.float32).contiguous(),
            ref.c6ab_flat.reshape(-1).to(device=ndev, dtype=torch.float32).contiguous(),
        )
        if storable(tabs[0]):
            _D3_TABLES[ndev] = tabs
    return tabs


def warm_d3_tables(dev: torch.device | str | None = None) -> tuple[Tensor, Tensor, Tensor] | None:
    if not (torch.cuda.is_available() and not capturing()):
        return None
    ndev = norm_device("cuda" if dev is None else dev)
    if ndev.type == "cuda":
        from lam4.kernels.cuda_d3 import warm_cuda_d3_tables
        warm_cuda_d3_tables(ndev)
        return _get_d3_tables(ndev)
    return None


@torch.no_grad()
def triton_build_csr_i32(
    recv_i32: Tensor, send_i32: Tensor, n_atoms: int, edge_shift: Tensor | None = None
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Deterministic zero-sync GPU CSR builder `(recv_ptr [N+1] int32, send_sorted [E] int32[, shift_sorted])`."""
    dev, n_a, e_tot = recv_i32.device, int(n_atoms), int(recv_i32.numel())
    if e_tot == 0:
        rp0, s0 = torch.zeros(n_a + 1, dtype=torch.int32, device=dev), torch.empty(0, dtype=torch.int32, device=dev)
        return (rp0, s0) if edge_shift is None else (rp0, s0, edge_shift[:0].contiguous())
    order = torch.argsort(recv_i32, stable=True)
    r_s = recv_i32.index_select(0, order).to(torch.int32).contiguous()
    s_s = send_i32.index_select(0, order).to(torch.int32).contiguous()
    rp = torch.zeros(n_a + 1, dtype=torch.int32, device=dev)
    rp[1:] = torch.searchsorted(r_s, torch.arange(n_a, dtype=torch.int32, device=dev), right=True).to(torch.int32)
    return (rp, s_s) if edge_shift is None else (rp, s_s, edge_shift.index_select(0, order).contiguous())


@_jit
def _load_cell9(cell_ptr, g, LOAD: tl.constexpr):
    if LOAD:
        b = g * 9
        c00, c01, c02 = tl.load(cell_ptr + b + 0), tl.load(cell_ptr + b + 1), tl.load(cell_ptr + b + 2)
        c10, c11, c12 = tl.load(cell_ptr + b + 3), tl.load(cell_ptr + b + 4), tl.load(cell_ptr + b + 5)
        c20, c21, c22 = tl.load(cell_ptr + b + 6), tl.load(cell_ptr + b + 7), tl.load(cell_ptr + b + 8)
    else:
        c00, c01, c02, c10, c11, c12, c20, c21, c22 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return c00, c01, c02, c10, c11, c12, c20, c21, c22


@_jit
def _d3_pair_vec(lane, idx1, i, pix, piy, piz, pos_ptr, mask_ptr, idx_ptr, shift_ptr,
                 c00, c01, c02, c10, c11, c12, c20, c21, c22,
                 IS_CSR: tl.constexpr, HAS_MASK: tl.constexpr, HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr):
    eb = lane < idx1
    oj = tl.load(idx_ptr + lane, mask=eb, other=0) if IS_CSR else lane
    act_j = (eb & (tl.load(mask_ptr + oj, mask=eb, other=0) != 0)) if HAS_MASK else eb
    dx = pix - tl.load(pos_ptr + oj * 3 + 0, mask=act_j, other=0.0)
    dy = piy - tl.load(pos_ptr + oj * 3 + 1, mask=act_j, other=0.0)
    dz = piz - tl.load(pos_ptr + oj * 3 + 2, mask=act_j, other=0.0)
    if IS_CSR and HAS_SHIFT:
        sx, sy, sz = tl.load(shift_ptr + lane * 3 + 0, mask=eb, other=0.0), tl.load(shift_ptr + lane * 3 + 1, mask=eb, other=0.0), tl.load(shift_ptr + lane * 3 + 2, mask=eb, other=0.0)
        if HAS_CELL:
            dx, dy, dz = dx - (sx * c00 + sy * c10 + sz * c20), dy - (sx * c01 + sy * c11 + sz * c21), dz - (sx * c02 + sy * c12 + sz * c22)
        else:
            dx, dy, dz = dx - sx, dy - sy, dz - sz
    d2 = dx * dx + dy * dy + dz * dz
    return oj, dx, dy, dz, d2, act_j & ((d2 > 1.0e-12) if IS_CSR else (oj != i))


@_jit
def _d3_cn_kernel(pos_ptr, z_ptr, mask_ptr, g_ptr, ptr_ptr, idx_ptr, shift_ptr, cell_ptr, rcov_ptr, out_cn_ptr,
                  cn_cut_b, ang_to_bohr, IS_CSR: tl.constexpr, HAS_MASK: tl.constexpr,
                  HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr, MULTI_CELL: tl.constexpr, BLOCK_J: tl.constexpr):
    i = tl.program_id(0)
    if HAS_MASK and (tl.load(mask_ptr + i) == 0):
        tl.store(out_cn_ptr + i, 0.0)
        return
    zi = tl.minimum(tl.maximum(tl.load(z_ptr + i), 0), 94)
    rc_i = tl.load(rcov_ptr + zi)
    pix, piy, piz = tl.load(pos_ptr + i * 3 + 0), tl.load(pos_ptr + i * 3 + 1), tl.load(pos_ptr + i * 3 + 2)
    gi = tl.load(g_ptr + i) if (not IS_CSR or MULTI_CELL) else 0
    row = i if IS_CSR else gi
    idx0, idx1 = tl.load(ptr_ptr + row), tl.load(ptr_ptr + row + 1)
    c00, c01, c02, c10, c11, c12, c20, c21, c22 = _load_cell9(cell_ptr, gi if MULTI_CELL else 0, HAS_SHIFT and HAS_CELL)
    acc = 0.0
    for off in range(idx0, idx1, BLOCK_J):
        oj, _, _, _, d2, valid = _d3_pair_vec(off + tl.arange(0, BLOCK_J), idx1, i, pix, piy, piz, pos_ptr, mask_ptr, idx_ptr, shift_ptr,
                                              c00, c01, c02, c10, c11, c12, c20, c21, c22, IS_CSR, HAS_MASK, HAS_SHIFT, HAS_CELL)
        r_b = tl.sqrt(d2 + 1.0e-8) * ang_to_bohr
        in_cn = valid & (r_b <= cn_cut_b)
        zj = tl.minimum(tl.maximum(tl.load(z_ptr + oj, mask=in_cn, other=0), 0), 94)
        rr = (rc_i + tl.load(rcov_ptr + zj, mask=in_cn, other=0.0)) / tl.maximum(r_b, 1.0e-6)
        acc += tl.sum(tl.where(in_cn, 1.0 / (1.0 + tl.exp(-16.0 * (rr - 1.0))), 0.0), axis=0)
    tl.store(out_cn_ptr + i, acc)


@_jit
def _d3_ef_kernel(pos_ptr, z_ptr, mask_ptr, g_ptr, ptr_ptr, idx_ptr, shift_ptr, cell_ptr, r2r4_ptr, c6_ptr, cn_ptr,
                  out_e_ptr, out_dedcn_ptr, out_f_ptr, out_vir_ptr, s6, s8, p_a1, p_a2, r_cut, taper_on, inv_tspan,
                  ang_to_bohr, hartree_kcal, IS_CSR: tl.constexpr, IS_ZERO: tl.constexpr, HAS_MASK: tl.constexpr,
                  HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr, MULTI_CELL: tl.constexpr,
                  NEED_DERIV: tl.constexpr, NEED_FORCES: tl.constexpr, NEED_STRESS: tl.constexpr, BLOCK_J: tl.constexpr):
    i = tl.program_id(0)
    if HAS_MASK and (tl.load(mask_ptr + i) == 0):
        tl.store(out_e_ptr + i, 0.0)
        if NEED_DERIV:
            tl.store(out_dedcn_ptr + i, 0.0)
        if NEED_FORCES:
            for c in range(3):
                tl.store(out_f_ptr + i * 3 + c, 0.0)
        if NEED_STRESS:
            for m in range(6):
                tl.store(out_vir_ptr + i * 6 + m, 0.0)
        return
    zi = tl.minimum(tl.maximum(tl.load(z_ptr + i), 0), 94)
    r2r4_i, cn_i = tl.load(r2r4_ptr + zi), tl.load(cn_ptr + i)
    pix, piy, piz = tl.load(pos_ptr + i * 3 + 0), tl.load(pos_ptr + i * 3 + 1), tl.load(pos_ptr + i * 3 + 2)
    gi = tl.load(g_ptr + i) if (not IS_CSR or MULTI_CELL) else 0
    row = i if IS_CSR else gi
    idx0, idx1 = tl.load(ptr_ptr + row), tl.load(ptr_ptr + row + 1)
    c00, c01, c02, c10, c11, c12, c20, c21, c22 = _load_cell9(cell_ptr, gi if MULTI_CELL else 0, HAS_SHIFT and HAS_CELL)
    acc_e, acc_de, acc_fx, acc_fy, acc_fz = 0.0, 0.0, 0.0, 0.0, 0.0
    vxx, vyy, vzz, vxy, vxz, vyz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for off in range(idx0, idx1, BLOCK_J):
        oj, dx, dy, dz, d2, valid = _d3_pair_vec(off + tl.arange(0, BLOCK_J), idx1, i, pix, piy, piz, pos_ptr, mask_ptr, idx_ptr, shift_ptr,
                                                 c00, c01, c02, c10, c11, c12, c20, c21, c22, IS_CSR, HAS_MASK, HAS_SHIFT, HAS_CELL)
        r2_eps = d2 + 1.0e-8
        inv_r = _rsqrt(r2_eps)
        r_ang = r2_eps * inv_r
        in_cut = valid & (d2 <= r_cut * r_cut) & (r_ang <= r_cut)
        if tl.sum(in_cut.to(tl.int32), axis=0) > 0:
            zj = tl.minimum(tl.maximum(tl.load(z_ptr + oj, mask=in_cut, other=0), 0), 94)
            cn_j = tl.load(cn_ptr + oj, mask=in_cut, other=0.0)
            pbase = 75 * (zi * 95 + zj)
            s_max = tl.full([BLOCK_J], -1.0e20, dtype=tl.float32)
            for k in range(25):
                c0 = tl.load(c6_ptr + pbase + 3 * k, mask=in_cut, other=0.0)
                di, dj = tl.load(c6_ptr + pbase + 3 * k + 1, mask=in_cut, other=0.0) - cn_i, tl.load(c6_ptr + pbase + 3 * k + 2, mask=in_cut, other=0.0) - cn_j
                s_max = tl.where(c0 > 0.0, tl.maximum(s_max, -4.0 * (di * di + dj * dj)), s_max)
            has_ref = in_cut & (s_max > -1.0e19)
            sw, swc, swd, swcd = tl.zeros([BLOCK_J], dtype=tl.float32), tl.zeros([BLOCK_J], dtype=tl.float32), tl.zeros([BLOCK_J], dtype=tl.float32), tl.zeros([BLOCK_J], dtype=tl.float32)
            for k in range(25):
                c0 = tl.load(c6_ptr + pbase + 3 * k, mask=has_ref, other=0.0)
                di, dj = tl.load(c6_ptr + pbase + 3 * k + 1, mask=has_ref, other=0.0) - cn_i, tl.load(c6_ptr + pbase + 3 * k + 2, mask=has_ref, other=0.0) - cn_j
                w = tl.where(has_ref & (c0 > 0.0), tl.exp(-4.0 * (di * di + dj * dj) - s_max), 0.0)
                sw += w
                swc += w * c0
                if NEED_DERIV:
                    wd = w * di
                    swd += wd
                    swcd += wd * c0
            inv_sw = 1.0 / tl.maximum(sw, 1.0e-30)
            c6 = swc * inv_sw
            q_ij = 3.0 * r2r4_i * tl.load(r2r4_ptr + zj, mask=has_ref, other=0.0)
            r_b = r_ang * ang_to_bohr
            rb2 = r_b * r_b
            rb4 = rb2 * rb2
            rb6, rb8 = rb4 * rb2, rb4 * rb4
            if IS_ZERO:
                r0, rb_s = tl.sqrt(tl.maximum(q_ij, 1.0e-12)), tl.maximum(r_b, 1.0e-6)
                u6, u8 = (p_a1 * r0) / rb_s, r0 / rb_s
                u6_2, u8_2 = u6 * u6, u8 * u8
                u6_4, u8_4 = u6_2 * u6_2, u8_2 * u8_2
                g6, g8 = 6.0 * (u6_4 * u6_4 * u6_4 * u6_2), 6.0 * (u8_4 * u8_4 * u8_4 * u8_4)
                f6, f8 = 1.0 / (1.0 + g6), 1.0 / (1.0 + g8)
                t6_v, t8_v = s6 * f6 / tl.maximum(rb6, 1.0e-12), (s8 * q_ij) * f8 / tl.maximum(rb8, 1.0e-12)
                A = t6_v + t8_v
                if NEED_DERIV:
                    dA_dr = ang_to_bohr * ((t6_v * (14.0 * g6 * f6 - tl.where(rb6 > 1.0e-12, 6.0, 0.0)) + t8_v * (16.0 * g8 * f8 - tl.where(rb8 > 1.0e-12, 8.0, 0.0))) / rb_s)
            else:
                tmp = p_a1 * tl.sqrt(tl.maximum(q_ij, 1.0e-12)) + p_a2
                t2 = tmp * tmp
                t4 = t2 * t2
                e6, e8 = 1.0 / (rb6 + t4 * t2), 1.0 / (rb8 + t4 * t4)
                A = s6 * e6 + (s8 * q_ij) * e8
                if NEED_DERIV:
                    dA_dr = -ang_to_bohr * (6.0 * s6 * (rb4 * r_b) * (e6 * e6) + 8.0 * (s8 * q_ij) * (rb6 * r_b) * (e8 * e8))
            tc = tl.minimum(tl.maximum((r_ang - taper_on) * inv_tspan, 0.0), 1.0)
            taper = 1.0 - tc * tc * tc * (10.0 + tc * (6.0 * tc - 15.0))
            acc_e += tl.sum(tl.where(has_ref, (-0.5 * hartree_kcal) * c6 * A * taper, 0.0), axis=0)
            if NEED_DERIV:
                dc6 = 8.0 * (swcd * inv_sw - c6 * (swd * inv_sw))
                acc_de += tl.sum(tl.where(has_ref, (-hartree_kcal) * dc6 * A * taper, 0.0), axis=0)
                dtap = tl.where((r_ang > taper_on) & (r_ang < r_cut), -30.0 * (tc * tc) * ((1.0 - tc) * (1.0 - tc)) * inv_tspan, 0.0)
                fm = tl.where(has_ref, hartree_kcal * c6 * (dA_dr * taper + A * dtap) * inv_r, 0.0)
                if NEED_FORCES:
                    acc_fx += tl.sum(fm * dx, axis=0)
                    acc_fy += tl.sum(fm * dy, axis=0)
                    acc_fz += tl.sum(fm * dz, axis=0)
                if NEED_STRESS:
                    vm = -0.5 * fm
                    vxx += tl.sum(vm * dx * dx, axis=0)
                    vyy += tl.sum(vm * dy * dy, axis=0)
                    vzz += tl.sum(vm * dz * dz, axis=0)
                    vxy += tl.sum(vm * dx * dy, axis=0)
                    vxz += tl.sum(vm * dx * dz, axis=0)
                    vyz += tl.sum(vm * dy * dz, axis=0)
    tl.store(out_e_ptr + i, acc_e)
    if NEED_DERIV:
        tl.store(out_dedcn_ptr + i, acc_de)
    if NEED_FORCES:
        tl.store(out_f_ptr + i * 3 + 0, acc_fx)
        tl.store(out_f_ptr + i * 3 + 1, acc_fy)
        tl.store(out_f_ptr + i * 3 + 2, acc_fz)
    if NEED_STRESS:
        tl.store(out_vir_ptr + i * 6 + 0, vxx)
        tl.store(out_vir_ptr + i * 6 + 1, vyy)
        tl.store(out_vir_ptr + i * 6 + 2, vzz)
        tl.store(out_vir_ptr + i * 6 + 3, vxy)
        tl.store(out_vir_ptr + i * 6 + 4, vxz)
        tl.store(out_vir_ptr + i * 6 + 5, vyz)


@_jit
def _d3_fcn_kernel(pos_ptr, z_ptr, mask_ptr, g_ptr, ptr_ptr, idx_ptr, shift_ptr, cell_ptr, rcov_ptr, dedcn_ptr,
                   inout_f_ptr, inout_vir_ptr, cn_cut_b, ang_to_bohr, IS_CSR: tl.constexpr, HAS_MASK: tl.constexpr,
                   HAS_SHIFT: tl.constexpr, HAS_CELL: tl.constexpr, MULTI_CELL: tl.constexpr,
                   NEED_FORCES: tl.constexpr, NEED_STRESS: tl.constexpr, BLOCK_J: tl.constexpr):
    i = tl.program_id(0)
    if HAS_MASK and (tl.load(mask_ptr + i) == 0):
        return
    zi = tl.minimum(tl.maximum(tl.load(z_ptr + i), 0), 94)
    rc_i, de_i = tl.load(rcov_ptr + zi), tl.load(dedcn_ptr + i)
    pix, piy, piz = tl.load(pos_ptr + i * 3 + 0), tl.load(pos_ptr + i * 3 + 1), tl.load(pos_ptr + i * 3 + 2)
    gi = tl.load(g_ptr + i) if (not IS_CSR or MULTI_CELL) else 0
    row = i if IS_CSR else gi
    idx0, idx1 = tl.load(ptr_ptr + row), tl.load(ptr_ptr + row + 1)
    c00, c01, c02, c10, c11, c12, c20, c21, c22 = _load_cell9(cell_ptr, gi if MULTI_CELL else 0, HAS_SHIFT and HAS_CELL)
    acc_fx, acc_fy, acc_fz, vxx, vyy, vzz, vxy, vxz, vyz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for off in range(idx0, idx1, BLOCK_J):
        oj, dx, dy, dz, d2, valid = _d3_pair_vec(off + tl.arange(0, BLOCK_J), idx1, i, pix, piy, piz, pos_ptr, mask_ptr, idx_ptr, shift_ptr,
                                                 c00, c01, c02, c10, c11, c12, c20, c21, c22, IS_CSR, HAS_MASK, HAS_SHIFT, HAS_CELL)
        r2_eps = d2 + 1.0e-8
        inv_r = _rsqrt(r2_eps)
        r_b = (r2_eps * inv_r) * ang_to_bohr
        in_cn = valid & (r_b <= cn_cut_b) & (r_b > 1.0e-6)
        zj = tl.minimum(tl.maximum(tl.load(z_ptr + oj, mask=in_cn, other=0), 0), 94)
        rr = (rc_i + tl.load(rcov_ptr + zj, mask=in_cn, other=0.0)) / tl.maximum(r_b, 1.0e-6)
        damp = 1.0 / (1.0 + tl.exp(-16.0 * (rr - 1.0)))
        fm = tl.where(in_cn, (de_i + tl.load(dedcn_ptr + oj, mask=in_cn, other=0.0)) * 16.0 * damp * (1.0 - damp) * rr * (inv_r * inv_r), 0.0)
        if NEED_FORCES:
            acc_fx += tl.sum(fm * dx, axis=0)
            acc_fy += tl.sum(fm * dy, axis=0)
            acc_fz += tl.sum(fm * dz, axis=0)
        if NEED_STRESS:
            vm = -0.5 * fm
            vxx += tl.sum(vm * dx * dx, axis=0)
            vyy += tl.sum(vm * dy * dy, axis=0)
            vzz += tl.sum(vm * dz * dz, axis=0)
            vxy += tl.sum(vm * dx * dy, axis=0)
            vxz += tl.sum(vm * dx * dz, axis=0)
            vyz += tl.sum(vm * dy * dz, axis=0)
    if NEED_FORCES:
        tl.store(inout_f_ptr + i * 3 + 0, tl.load(inout_f_ptr + i * 3 + 0) + acc_fx)
        tl.store(inout_f_ptr + i * 3 + 1, tl.load(inout_f_ptr + i * 3 + 1) + acc_fy)
        tl.store(inout_f_ptr + i * 3 + 2, tl.load(inout_f_ptr + i * 3 + 2) + acc_fz)
    if NEED_STRESS:
        tl.store(inout_vir_ptr + i * 6 + 0, tl.load(inout_vir_ptr + i * 6 + 0) + vxx)
        tl.store(inout_vir_ptr + i * 6 + 1, tl.load(inout_vir_ptr + i * 6 + 1) + vyy)
        tl.store(inout_vir_ptr + i * 6 + 2, tl.load(inout_vir_ptr + i * 6 + 2) + vzz)
        tl.store(inout_vir_ptr + i * 6 + 3, tl.load(inout_vir_ptr + i * 6 + 3) + vxy)
        tl.store(inout_vir_ptr + i * 6 + 4, tl.load(inout_vir_ptr + i * 6 + 4) + vxz)
        tl.store(inout_vir_ptr + i * 6 + 5, tl.load(inout_vir_ptr + i * 6 + 5) + vyz)


def _reduce_e_graph(e_atom: Tensor, graph: Tensor | None, n_graphs: int) -> Tensor:
    ed = e_atom.to(torch.float64)
    if n_graphs <= 1 or graph is None:
        return ed.sum(0, keepdim=True).to(torch.float32)
    return torch.zeros(n_graphs, dtype=torch.float64, device=e_atom.device).index_add_(0, graph.to(torch.int64), ed).to(torch.float32)


def _run_d3(pos_c: Tensor, z_c: Tensor, mask_u8: Tensor, g_c: Tensor, ptr: Tensor, idx: Tensor, sh_c: Tensor, cell_c: Tensor,
            d3_mode: str, is_csr: bool, has_mask: bool, has_shift: bool, has_cell: bool, multi_cell: bool,
            compute_forces: bool, do_stress: bool) -> tuple[Tensor, Tensor | None, Tensor | None]:
    if d3_mode not in _MODE_PARAMS:
        raise ValueError(f"Unsupported d3_mode {d3_mode!r}; expected 'bj_wb97m' or 'zero_pbe'")
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in this environment")
    dev, n_atoms = pos_c.device, int(pos_c.shape[0])
    is_zero, s6, s8, p_a1, p_a2 = _MODE_PARAMS[d3_mode]
    rcov, r2r4, c6_flat = _get_d3_tables(dev)
    need_deriv = bool(compute_forces or do_stress)
    cn, e_atom = torch.empty(n_atoms, dtype=torch.float32, device=dev), torch.empty(n_atoms, dtype=torch.float32, device=dev)
    dedcn = torch.empty(n_atoms, dtype=torch.float32, device=dev) if need_deriv else cn[:1]
    forces = torch.empty((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else pos_c[:1]
    vir = torch.empty((n_atoms, 6), dtype=torch.float32, device=dev) if do_stress else pos_c[:1]
    ab, hk = float(ANGSTROM_TO_BOHR), float(HARTREE_TO_KCAL)
    _d3_cn_kernel[(n_atoms,)](pos_c, z_c, mask_u8, g_c, ptr, idx, sh_c, cell_c, rcov, cn, _CN_CUT_BOHR, ab,
                              IS_CSR=is_csr, HAS_MASK=has_mask, HAS_SHIFT=has_shift, HAS_CELL=has_cell, MULTI_CELL=multi_cell, BLOCK_J=64, num_warps=2)
    _d3_ef_kernel[(n_atoms,)](pos_c, z_c, mask_u8, g_c, ptr, idx, sh_c, cell_c, r2r4, c6_flat, cn, e_atom, dedcn, forces, vir,
                              s6, s8, p_a1, p_a2, _R_CUT, _TAPER_ON, _INV_TSPAN, ab, hk,
                              IS_CSR=is_csr, IS_ZERO=is_zero, HAS_MASK=has_mask, HAS_SHIFT=has_shift, HAS_CELL=has_cell, MULTI_CELL=multi_cell,
                              NEED_DERIV=need_deriv, NEED_FORCES=compute_forces, NEED_STRESS=do_stress, BLOCK_J=64, num_warps=2)
    if need_deriv:
        _d3_fcn_kernel[(n_atoms,)](pos_c, z_c, mask_u8, g_c, ptr, idx, sh_c, cell_c, rcov, dedcn, forces, vir, _CN_CUT_BOHR, ab,
                                   IS_CSR=is_csr, HAS_MASK=has_mask, HAS_SHIFT=has_shift, HAS_CELL=has_cell, MULTI_CELL=multi_cell,
                                   NEED_FORCES=compute_forces, NEED_STRESS=do_stress, BLOCK_J=64, num_warps=2)
    return e_atom, (forces if compute_forces else None), (vir if do_stress else None)


@torch.no_grad()
def triton_d3_grouped(
    z: Tensor, pos: Tensor, graph: Tensor, n_graphs: int, *, d3_mode: str = "bj_wb97m",
    atom_mask: Tensor | None = None, group_ptr: Tensor | None = None, compute_forces: bool = True, compute_stress: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Triton/CUDA D3 over grouped open molecules (`e_graph [G]` fp32 kcal/mol, `forces [N, 3]` fp32 kcal/(mol*A) | None, `None`)."""
    from lam4.kernels.cuda_d3 import cuda_d3_grouped, has_cuda_d3
    if has_cuda_d3(pos.device):
        try:
            e_g, f = cuda_d3_grouped(z, pos, graph, n_graphs, d3_mode=d3_mode, atom_mask=atom_mask, group_ptr=group_ptr, compute_forces=compute_forces)
            return e_g.to(torch.float32), f, None
        except Exception:  # noqa: BLE001
            if not HAS_TRITON:
                raise
    if d3_mode not in _MODE_PARAMS:
        raise ValueError(f"Unsupported d3_mode {d3_mode!r}; expected 'bj_wb97m' or 'zero_pbe'")
    dev, n_atoms, G = pos.device, int(pos.shape[0]), int(n_graphs)
    if n_atoms == 0 or G == 0:
        return torch.zeros(G, dtype=torch.float32, device=dev), (torch.zeros((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else None), None
    pos_c, z_c, g_c = pos.to(torch.float32).contiguous(), z.to(torch.int32).contiguous(), graph.to(torch.int32).contiguous()
    ptr, has_mask = (build_group_ptr(graph, G) if group_ptr is None else group_ptr), (atom_mask is not None)
    e_atom, forces, _ = _run_d3(pos_c, z_c, atom_mask.to(torch.uint8).contiguous() if has_mask else z_c[:1], g_c, ptr, z_c[:1], pos_c[:1], pos_c[:1], d3_mode, False, has_mask, False, False, False, compute_forces, False)
    return _reduce_e_graph(e_atom, graph, G), forces, None


@torch.no_grad()
def triton_d3_csr(
    z: Tensor, pos: Tensor, recv_ptr: Tensor, send_idx: Tensor, *, edge_shift: Tensor | None = None, cell: Tensor | None = None,
    d3_mode: str = "bj_wb97m", atom_mask: Tensor | None = None, graph: Tensor | None = None, n_graphs: int = 1,
    compute_forces: bool = True, compute_stress: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Triton/CUDA D3 over a 12.0 A receiver-sorted CSR neighbor list (open or periodic `edge_shift` + `cell` + `stress`)."""
    from lam4.kernels.cuda_d3 import cuda_d3_csr, has_cuda_d3
    if has_cuda_d3(pos.device):
        try:
            e_g, f, s = cuda_d3_csr(z, pos, recv_ptr, send_idx, edge_shift=edge_shift, cell=cell, d3_mode=d3_mode, atom_mask=atom_mask, graph=graph, n_graphs=n_graphs, compute_forces=compute_forces, compute_stress=compute_stress)
            return e_g.to(torch.float32), f, (None if s is None else s.to(torch.float32))
        except Exception:  # noqa: BLE001
            if not HAS_TRITON:
                raise
    if d3_mode not in _MODE_PARAMS:
        raise ValueError(f"Unsupported d3_mode {d3_mode!r}; expected 'bj_wb97m' or 'zero_pbe'")
    dev, n_atoms, G, do_stress = pos.device, int(pos.shape[0]), int(n_graphs), bool(compute_stress and cell is not None)
    if n_atoms == 0 or G == 0:
        return torch.zeros(G, dtype=torch.float32, device=dev), (torch.zeros((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else None), (torch.zeros((G, 3, 3), dtype=torch.float32, device=dev) if do_stress else None)
    pos_c, z_c, rp_c, si_c = pos.to(torch.float32).contiguous(), z.to(torch.int32).contiguous(), recv_ptr.to(torch.int32).contiguous(), send_idx.to(torch.int32).contiguous()
    has_mask, has_shift = atom_mask is not None, (edge_shift is not None and int(edge_shift.numel()) > 0)
    mask_u8, sh_c = (atom_mask.to(torch.uint8).contiguous() if has_mask else z_c[:1]), (edge_shift.to(torch.float32).contiguous() if has_shift else pos_c[:1])
    has_cell, cell_g = bool(has_shift and cell is not None), ((cell.unsqueeze(0) if cell.ndim == 2 else cell) if cell is not None else None)
    multi_cell = bool(has_cell and cell_g is not None and cell_g.shape[0] > 1 and graph is not None)
    cell_c, g_c = (cell_g.to(torch.float32).contiguous().view(-1, 9) if cell_g is not None else pos_c[:1]), (graph.to(torch.int32).contiguous() if (graph is not None and multi_cell) else z_c[:1])
    e_atom, forces, vir = _run_d3(pos_c, z_c, mask_u8, g_c, rp_c, si_c, sh_c, cell_c, d3_mode, True, has_mask, has_shift, has_cell, multi_cell, compute_forces, do_stress)
    stress = None
    if do_stress and vir is not None and cell_g is not None:
        vd = vir.to(torch.float64)
        vg = vd.sum(0, keepdim=True) if (G <= 1 or graph is None) else torch.zeros((G, 6), dtype=torch.float64, device=dev).index_add_(0, graph.to(torch.int64), vd)
        cg_d = cell_g.to(torch.float64).expand(G, 3, 3) if (cell_g.shape[0] == 1 and G > 1) else cell_g.to(torch.float64)
        a, bb, c = cg_d.unbind(1)
        vol = (a * torch.cross(bb, c, dim=-1)).sum(-1).abs()
        ok = vol > 1.0e-8
        vs = (vg * torch.where(ok, 1.0 / torch.where(ok, vol, torch.ones_like(vol)), torch.zeros_like(vol)).unsqueeze(-1)).to(torch.float32)
        stress = torch.stack([torch.stack([vs[:, 0], vs[:, 3], vs[:, 4]], -1), torch.stack([vs[:, 3], vs[:, 1], vs[:, 5]], -1), torch.stack([vs[:, 4], vs[:, 5], vs[:, 2]], -1)], dim=1)
    return _reduce_e_graph(e_atom, graph, G), forces, stress
