"""Optional CUDA / ROCm Triton Dual-CSR forward & atomic-free backward kernels (`BLOCK_C = 32` lanes = 1 atom)."""
from __future__ import annotations

import os

from typing import Any

import torch
from torch import Tensor

from lam4.core.edge_terms import EdgeCfg
from lam4.core.layout import StageLayout
from lam4.kernels.csr_plan import MAX_DOFF_R2, CSRPlan, rmin_table_f32

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - optional on macOS / CPU environments
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


def has_triton_csr() -> bool:
    if not (HAS_TRITON and torch.cuda.is_available()):
        return False
    try:
        return int(torch.cuda.get_device_capability()[0]) >= 7
    except Exception:
        return False


if HAS_TRITON:

    @triton.jit
    def _horner_v(poly_ptr, base, td: tl.constexpr, col, tsp, mask):
        a0 = tl.load(poly_ptr + base + col, mask=mask, other=0.0)
        a1 = tl.load(poly_ptr + base + td + col, mask=mask, other=0.0)
        a2 = tl.load(poly_ptr + base + 2 * td + col, mask=mask, other=0.0)
        a3 = tl.load(poly_ptr + base + 3 * td + col, mask=mask, other=0.0)
        return ((a3 * tsp + a2) * tsp + a1) * tsp + a0

    @triton.jit
    def _horner_vd(poly_ptr, base, td: tl.constexpr, col, tsp, inv_h, mask):
        a0 = tl.load(poly_ptr + base + col, mask=mask, other=0.0)
        a1 = tl.load(poly_ptr + base + td + col, mask=mask, other=0.0)
        a2 = tl.load(poly_ptr + base + 2 * td + col, mask=mask, other=0.0)
        a3 = tl.load(poly_ptr + base + 3 * td + col, mask=mask, other=0.0)
        fv = ((a3 * tsp + a2) * tsp + a1) * tsp + a0
        df = ((3.0 * a3 * tsp + 2.0 * a2) * tsp + a1) * inv_h
        return fv, df

    @triton.jit
    def _triton_csr_fwd(
        pos_p, rptr_p, sidx_p, sh_p, cell_p, grp_p, z_p, rmin_p, rho_p, poly_p, gates_p, qn_p, kn_p, vec_p, ten_p,
        num_p, zc_p, dsum_p, n_atoms, grid, r_cut, r_on, max_doff_r2,
        LATER: tl.constexpr, HAS_FLOOR: tl.constexpr, HAS_RHO: tl.constexpr, HAS_TEN: tl.constexpr,
        HAS_CELL: tl.constexpr, IS_G1: tl.constexpr,
    ):
        i = tl.program_id(0)
        if i >= n_atoms:
            return
        lane = tl.arange(0, 32)
        m16 = lane < 16
        m8 = lane < 8
        TD: tl.constexpr = 284 if LATER else 188
        OD: tl.constexpr = 408 if LATER else 360
        FO: tl.constexpr = 276 if LATER else 180
        OT: tl.constexpr = 256 if LATER else 224
        OO: tl.constexpr = 352 if LATER else 304

        g_id = 0 if (not HAS_CELL or IS_G1) else tl.load(grp_p + i).to(tl.int32)
        zi = tl.minimum(tl.maximum(tl.load(z_p + i).to(tl.int32), 0), 127) if HAS_FLOOR else 0
        rho_i = tl.load(rho_p + i) if HAS_RHO else 0.0
        px = tl.load(pos_p + 3 * i + 0)
        py = tl.load(pos_p + 3 * i + 1)
        pz = tl.load(pos_p + 3 * i + 2)
        q0 = tl.load(qn_p + 64 * i + lane)
        q1 = tl.load(qn_p + 64 * i + 32 + lane)

        as0 = tl.zeros([32], tl.float32)
        as1 = tl.zeros([32], tl.float32)
        as2 = tl.zeros([32], tl.float32)
        as3 = tl.zeros([32], tl.float32)
        avx = tl.zeros([32], tl.float32)
        avy = tl.zeros([32], tl.float32)
        avz = tl.zeros([32], tl.float32)
        alon = tl.zeros([32], tl.float32)
        at0 = tl.zeros([32], tl.float32)
        at1 = tl.zeros([32], tl.float32)
        at2 = tl.zeros([32], tl.float32)
        at3 = tl.zeros([32], tl.float32)
        at4 = tl.zeros([32], tl.float32)
        alon2 = tl.zeros([32], tl.float32)
        ao0 = tl.zeros([32], tl.float32)
        ao1 = tl.zeros([32], tl.float32)
        ao2 = tl.zeros([32], tl.float32)
        ao3 = tl.zeros([32], tl.float32)
        ao4 = tl.zeros([32], tl.float32)
        ao5 = tl.zeros([32], tl.float32)
        ao6 = tl.zeros([32], tl.float32)
        az0 = 0.0
        az1 = 0.0
        az2 = 0.0
        az3 = 0.0
        adsum = 0.0

        inv_w = 1.0 / (r_cut - r_on)
        inv_h = grid / r_cut
        rcut2 = r_cut * r_cut
        e_beg = tl.load(rptr_p + i)
        e_end = tl.load(rptr_p + i + 1)
        for e in range(e_beg, e_end):
            j = tl.load(sidx_p + e)
            sx = tl.load(sh_p + 3 * e + 0)
            sy = tl.load(sh_p + 3 * e + 1)
            sz = tl.load(sh_p + 3 * e + 2)
            if HAS_CELL:
                cb = 9 * g_id
                esx, esy, esz = sx, sy, sz
                sx = esx * tl.load(cell_p + cb + 0) + esy * tl.load(cell_p + cb + 3) + esz * tl.load(cell_p + cb + 6)
                sy = esx * tl.load(cell_p + cb + 1) + esy * tl.load(cell_p + cb + 4) + esz * tl.load(cell_p + cb + 7)
                sz = esx * tl.load(cell_p + cb + 2) + esy * tl.load(cell_p + cb + 5) + esz * tl.load(cell_p + cb + 8)
            dx = tl.load(pos_p + 3 * j + 0) - px + sx
            dy = tl.load(pos_p + 3 * j + 1) - py + sy
            dz = tl.load(pos_p + 3 * j + 2) - pz + sz
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= 1e-24 and r2 < rcut2:
                r_raw = tl.sqrt(r2)
                ux, uy, uz = dx / r_raw, dy / r_raw, dz / r_raw
                r_fl = r_raw
                if HAS_FLOOR and r2 < max_doff_r2:
                    zj = tl.minimum(tl.maximum(tl.load(z_p + j).to(tl.int32), 0), 127)
                    r_min = tl.load(rmin_p + (zi << 7) + zj)
                    d_off = r_min + 0.05
                    r_eps = tl.sqrt(r2 + 1e-8)
                    if r_eps < d_off and r_min > 0.0:
                        u_th = (d_off - r_eps) * 20.0
                        e2 = tl.exp(-2.0 * u_th)
                        r_fl = d_off - 0.05 * ((1.0 - e2) / (1.0 + e2))
                if r_fl < r_cut:
                    chi = 1.0
                    if r_fl > r_on:
                        te = (r_fl - r_on) * inv_w
                        t2 = te * te
                        chi = 1.0 - t2 * t2 * (35.0 + te * (-84.0 + te * (70.0 - 20.0 * te)))
                    x = r_fl
                    if HAS_RHO:
                        sw = 1.52 / (rho_i + tl.load(rho_p + j))
                        x = (sw * r_fl) / (1.0 + (sw - 1.0) * (r_fl / r_cut))
                    u_sp = tl.minimum(tl.maximum(x, 0.0), r_cut) * inv_h
                    idx = tl.minimum(tl.maximum(u_sp.to(tl.int32), 0), grid - 1)
                    tsp = u_sp - idx.to(tl.float32)
                    pbase = (idx * 4) * TD

                    k0 = tl.load(kn_p + 64 * j + lane)
                    k1 = tl.load(kn_p + 64 * j + 32 + lane)
                    d0 = tl.sum(tl.where(lane < 16, q0 * k0, 0.0), axis=0)
                    d1 = tl.sum(tl.where(lane >= 16, q0 * k0, 0.0), axis=0)
                    d2 = tl.sum(tl.where(lane < 16, q1 * k1, 0.0), axis=0)
                    d3 = tl.sum(tl.where(lane >= 16, q1 * k1, 0.0), axis=0)
                    b0 = _horner_v(poly_p, pbase, TD, 0, tsp, True)
                    b1 = _horner_v(poly_p, pbase, TD, 1, tsp, True)
                    b2 = _horner_v(poly_p, pbase, TD, 2, tsp, True)
                    b3 = _horner_v(poly_p, pbase, TD, 3, tsp, True)
                    w0 = chi * tl.exp(d0 + 2.0 * ((1.0 - tl.exp(-b0)) / (1.0 + tl.exp(-b0))))
                    w1 = chi * tl.exp(d1 + 2.0 * ((1.0 - tl.exp(-b1)) / (1.0 + tl.exp(-b1))))
                    w2 = chi * tl.exp(d2 + 2.0 * ((1.0 - tl.exp(-b2)) / (1.0 + tl.exp(-b2))))
                    w3 = chi * tl.exp(d3 + 2.0 * ((1.0 - tl.exp(-b3)) / (1.0 + tl.exp(-b3))))
                    az0 += w0
                    az1 += w1
                    az2 += w2
                    az3 += w3
                    adsum += chi

                    gj = 184 * j
                    as0 += w0 * _horner_v(poly_p, pbase, TD, 4 + lane, tsp, True) * tl.load(gates_p + gj + lane)
                    as1 += w1 * _horner_v(poly_p, pbase, TD, 36 + lane, tsp, True) * tl.load(gates_p + gj + 32 + lane)
                    as2 += w2 * _horner_v(poly_p, pbase, TD, 68 + lane, tsp, True) * tl.load(gates_p + gj + 64 + lane)
                    as3 += w3 * _horner_v(poly_p, pbase, TD, 100 + lane, tsp, True) * tl.load(gates_p + gj + 96 + lane)

                    hv = lane >> 3
                    wv = tl.where(hv == 0, w0, tl.where(hv == 1, w1, tl.where(hv == 2, w2, w3)))
                    w_dir = _horner_v(poly_p, pbase, TD, 132 + lane, tsp, True) * tl.load(gates_p + gj + 128 + lane)
                    if LATER:
                        vb = 96 * j + 3 * lane
                        bx = tl.load(vec_p + vb + 0)
                        by = tl.load(vec_p + vb + 1)
                        bz = tl.load(vec_p + vb + 2)
                        f_tr = _horner_v(poly_p, pbase, TD, 180 + lane, tsp, True)
                        w_lon = _horner_v(poly_p, pbase, TD, 212 + lane, tsp, True) * (bx * ux + by * uy + bz * uz)
                        sc = wv * (w_dir + w_lon)
                        st = wv * f_tr
                        avx += sc * ux + st * bx
                        avy += sc * uy + st * by
                        avz += sc * uz + st * bz
                        alon += wv * w_lon
                    else:
                        sc = wv * w_dir
                        avx += sc * ux
                        avy += sc * uy
                        avz += sc * uz

                    s2: tl.constexpr = 1.4142135623730951
                    is2: tl.constexpr = 0.7071067811865475
                    is6: tl.constexpr = 0.4082482904638630
                    xx, yy, zz = ux * ux, uy * uy, uz * uz
                    y20, y21, y22 = s2 * ux * uy, s2 * ux * uz, s2 * uy * uz
                    y23, y24 = (xx - yy) * is2, (2.0 * zz - xx - yy) * is6

                    ht = lane >> 2
                    wt = tl.where(ht == 0, w0, tl.where(ht == 1, w1, tl.where(ht == 2, w2, w3)))
                    w_ten = _horner_v(poly_p, pbase, TD, 164 + lane, tsp, m16) * tl.load(gates_p + gj + 160 + lane, mask=m16, other=0.0)
                    if HAS_TEN:
                        tb = 80 * j + 5 * lane
                        t0 = tl.load(ten_p + tb + 0, mask=m16, other=0.0)
                        t1 = tl.load(ten_p + tb + 1, mask=m16, other=0.0)
                        t2_ = tl.load(ten_p + tb + 2, mask=m16, other=0.0)
                        t3 = tl.load(ten_p + tb + 3, mask=m16, other=0.0)
                        t4 = tl.load(ten_p + tb + 4, mask=m16, other=0.0)
                        dty2 = t0 * y20 + t1 * y21 + t2_ * y22 + t3 * y23 + t4 * y24
                        st2 = wt * _horner_v(poly_p, pbase, TD, 244 + lane, tsp, m16)
                        w_lon2 = _horner_v(poly_p, pbase, TD, 260 + lane, tsp, m16) * dty2
                        sct = wt * (w_ten + w_lon2)
                        at0 += sct * y20 + st2 * t0
                        at1 += sct * y21 + st2 * t1
                        at2 += sct * y22 + st2 * t2_
                        at3 += sct * y23 + st2 * t3
                        at4 += sct * y24 + st2 * t4
                        alon2 += wt * w_lon2
                    else:
                        sct = wt * w_ten
                        at0 += sct * y20
                        at1 += sct * y21
                        at2 += sct * y22
                        at3 += sct * y23
                        at4 += sct * y24

                    ho = lane >> 1
                    wo = tl.where(ho == 0, w0, tl.where(ho == 1, w1, tl.where(ho == 2, w2, w3)))
                    sco = wo * _horner_v(poly_p, pbase, TD, FO + lane, tsp, m8) * tl.load(gates_p + gj + 176 + lane, mask=m8, other=0.0)
                    ao0 += sco * (0.5 * uy * (3.0 * xx - yy))
                    ao1 += sco * (2.449489742783178 * ux * uy * uz)
                    ao2 += sco * (0.3872983346207417 * uy * (4.0 * zz - xx - yy))
                    ao3 += sco * (0.31622776601683794 * uz * (2.0 * zz - 3.0 * xx - 3.0 * yy))
                    ao4 += sco * (0.3872983346207417 * ux * (4.0 * zz - xx - yy))
                    ao5 += sco * (1.224744871391589 * uz * (xx - yy))
                    ao6 += sco * (0.5 * ux * (xx - 3.0 * yy))

        ob = OD * i
        tl.store(num_p + ob + lane, as0)
        tl.store(num_p + ob + 32 + lane, as1)
        tl.store(num_p + ob + 64 + lane, as2)
        tl.store(num_p + ob + 96 + lane, as3)
        tl.store(num_p + ob + 128 + 3 * lane + 0, avx)
        tl.store(num_p + ob + 128 + 3 * lane + 1, avy)
        tl.store(num_p + ob + 128 + 3 * lane + 2, avz)
        if LATER:
            tl.store(num_p + ob + 224 + lane, alon)
            tl.store(num_p + ob + 336 + lane, alon2, mask=m16)
        tl.store(num_p + ob + OT + 5 * lane + 0, at0, mask=m16)
        tl.store(num_p + ob + OT + 5 * lane + 1, at1, mask=m16)
        tl.store(num_p + ob + OT + 5 * lane + 2, at2, mask=m16)
        tl.store(num_p + ob + OT + 5 * lane + 3, at3, mask=m16)
        tl.store(num_p + ob + OT + 5 * lane + 4, at4, mask=m16)
        tl.store(num_p + ob + OO + 7 * lane + 0, ao0, mask=m8)
        tl.store(num_p + ob + OO + 7 * lane + 1, ao1, mask=m8)
        tl.store(num_p + ob + OO + 7 * lane + 2, ao2, mask=m8)
        tl.store(num_p + ob + OO + 7 * lane + 3, ao3, mask=m8)
        tl.store(num_p + ob + OO + 7 * lane + 4, ao4, mask=m8)
        tl.store(num_p + ob + OO + 7 * lane + 5, ao5, mask=m8)
        tl.store(num_p + ob + OO + 7 * lane + 6, ao6, mask=m8)
        tl.store(zc_p + 4 * i + 0, az0)
        tl.store(zc_p + 4 * i + 1, az1)
        tl.store(zc_p + 4 * i + 2, az2)
        tl.store(zc_p + 4 * i + 3, az3)
        tl.store(dsum_p + i, adsum)


def triton_csr_forward(
    plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor,
    vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor,
    cell: Tensor, graph: Tensor, z: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    dev = pos.device
    n = plan.n_atoms
    if n == 0 or plan.n_edges == 0:
        return (
            torch.zeros((n, layout.out_dim), dtype=torch.float32, device=dev),
            torch.zeros((n, 4), dtype=torch.float32, device=dev),
            torch.zeros((n,), dtype=torch.float32, device=dev),
        )
    has_floor = bool(cfg.floor and z.numel() > 0)
    has_rho = bool(rho.numel() > 0)
    has_ten = bool(layout.later and ten.numel() > 0)
    has_cell = bool(cell.numel() > 0)
    is_g1 = bool(has_cell and cell.shape[0] == 1)
    df = torch.zeros(1, dtype=torch.float32, device=dev)
    di = torch.zeros(1, dtype=torch.int64, device=dev)
    rmin = rmin_table_f32(dev) if has_floor else df
    out_num = torch.empty((n, layout.out_dim), dtype=torch.float32, device=dev)
    out_zc = torch.empty((n, 4), dtype=torch.float32, device=dev)
    out_dsum = torch.empty((n,), dtype=torch.float32, device=dev)
    _triton_csr_fwd[(n,)](
        pos, plan.recv_ptr, plan.send_idx, shift_in,
        cell if has_cell else df, graph if graph.numel() else di, z if has_floor else di, rmin,
        rho if has_rho else df, poly, gates, qn, kn,
        vec if vec.numel() else df, ten if has_ten else df,
        out_num, out_zc, out_dsum, n, int(poly.shape[0]), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2),
        LATER=layout.later, HAS_FLOOR=has_floor, HAS_RHO=has_rho, HAS_TEN=has_ten, HAS_CELL=has_cell, IS_G1=is_g1,
        num_warps=1,
    )
    return out_num, out_zc, out_dsum


def triton_csr_backward(
    plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor,
    vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor,
    cell: Tensor, graph: Tensor, z: Tensor, g_num: Tensor, g_zc: Tensor, g_dsum: Tensor,
    *, need_vir: bool, need_drho: bool, need_dshift: bool,
) -> tuple[Any, ...]:
    """Low-memory streaming VJP fallback on CUDA when backward is requested under Triton forward."""
    from lam4.core.edge_chunks import make_chunks, stream_chunk_edges
    from lam4.core.edge_stream import _StreamVJP, empty_z

    chunks = plan.chunks if os.environ.get("LAM4_CHUNK_CACHE", "1") != "0" else None   # =0: old behaviour (host make_chunks every backward), for A/B timing
    if chunks is None:   # per edge list, not per backward: make_chunks reads the plan on the host (a device sync per stage and call)
        recv_l, send_l = plan.recv_idx.to(torch.long), plan.send_idx.to(torch.long)
        if plan.n_edges == 0:
            chunks = []   # isolated atoms: nothing to differentiate (make_chunks also yields no chunk for an empty edge list)
        elif plan.n_edges <= stream_chunk_edges(pos.device):   # one chunk covering every atom, built on the device: no host sync, static shapes
            from lam4.core.edge_chunks import Chunk
            chunks = [Chunk(0, plan.n_atoms, 0, plan.n_edges, recv_l, send_l, torch.arange(plan.n_atoms, device=pos.device))]
        else:
            chunks = make_chunks(recv_l, send_l, plan.n_atoms, stream_chunk_edges(pos.device))
        if not torch.cuda.is_current_stream_capturing() and os.environ.get("LAM4_CHUNK_CACHE", "1") != "0":
            plan.chunks = chunks
    empty = pos.new_zeros(0)
    if cell.numel() > 0:
        sh = (shift_in @ cell[0]) if cell.shape[0] == 1 else torch.einsum("ek,ekl->el", shift_in, cell[graph[plan.recv_idx.to(torch.long)]])
    else:
        sh = shift_in
    need_sh_internal = bool(need_vir or need_dshift)
    sh_c, sh_d = (empty, sh) if need_sh_internal else (sh, empty)
    # Reconstruct B-spline window coefficients [G, 4, C] from Horner [a0, a1, a2, a3]:
    a0, a1, a2, a3 = poly[:, 0], poly[:, 1], poly[:, 2], poly[:, 3]
    c1 = a0 - a2 * (1.0 / 3.0)
    c0 = c1 - a1 + a2
    c2 = c1 + a1 + a2
    c3 = 6.0 * a3 + c0 - 3.0 * c1 + 3.0 * c2
    wtab = torch.stack((c0, c1, c2, c3), dim=1).contiguous()
    ins = (pos, wtab, gates, vec if layout.later else empty, qn, kn, rho if rho.numel() else empty,
           empty, ten if (layout.later and ten.numel()) else empty, sh_d)
    needs = (True, False, True, bool(layout.later), True, True, bool(need_drho), False, bool(layout.later and ten.numel()), need_sh_internal)
    grads = _StreamVJP.apply(chunks, layout, cfg, needs, sh_c, empty_z(pos, z if z.numel() else None), *ins, g_num, g_zc, g_dsum)
    d_sh = grads[9] if need_sh_internal else None
    d_cell = None
    if need_vir and d_sh is not None:
        if cell.shape[0] == 1:
            d_cell = (shift_in.T @ d_sh).unsqueeze(0)
        else:
            eg = torch.einsum("ek,el->ekl", shift_in, d_sh).reshape(-1, 9)
            d_cell = cell.new_zeros(cell.shape[0], 9).index_add(0, graph[plan.recv_idx.to(torch.long)], eg).view_as(cell)
    return (
        grads[0], grads[2], grads[3] if layout.later else None,
        grads[8] if (layout.later and ten.numel()) else None,
        grads[4], grads[5], grads[6] if need_drho else None,
        d_cell, d_sh if need_dshift else None,
    )
