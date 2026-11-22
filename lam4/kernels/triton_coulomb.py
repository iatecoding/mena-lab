"""Triton CUDA kernels & custom_op autograd for open-boundary smeared ExactCoulomb (`LAM4_TRITON_COULOMB=0` disables).
Modes: `G > 1` grouped CSR (`O(N)` memory, no `cap=350` drop); `G == 1, N <= 4096` 64x64 tiled; `G == 1, N > 4096` Morton `fast64` spheres.
"""
from __future__ import annotations

import math
import os

import torch
from torch import Tensor

from lam4.core.consts import capturing, cuda_cap_ok, is_fake, proxy_tracing
from lam4.core.radial import norm_device, upcast_fp32

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = tl = None  # type: ignore[assignment]
    HAS_TRITON = False

_jit = triton.jit if HAS_TRITON else (lambda fn: fn)
_PTR_G1_CACHE: dict[tuple[torch.device, int], Tensor] = {}
FAST64_THRESHOLD = 1024


def has_triton_coulomb(dev: torch.device | str | None = None) -> bool:
    return bool(os.environ.get("LAM4_TRITON_COULOMB", "1") != "0" and HAS_TRITON and cuda_cap_ok(dev))


def warm_triton_coulomb(dev: torch.device | str | None = None, n_atoms: int | tuple[int, ...] = ()) -> None:
    if not has_triton_coulomb(dev) or capturing() or proxy_tracing():
        return
    ndev = norm_device("cuda" if dev is None else dev)
    for n in ((n_atoms,) if isinstance(n_atoms, int) else n_atoms):
        ni = int(n)
        if ni >= 0 and (ndev, ni) not in _PTR_G1_CACHE:
            if len(_PTR_G1_CACHE) >= 64:
                _PTR_G1_CACHE.clear()
            _PTR_G1_CACHE[(ndev, ni)] = torch.tensor([0, ni], dtype=torch.int32, device=ndev)


if HAS_TRITON and hasattr(tl, "rsqrt"):
    @_jit
    def _rsqrt(x): return tl.rsqrt(x)
elif HAS_TRITON and hasattr(getattr(tl, "math", None), "rsqrt"):
    @_jit
    def _rsqrt(x): return tl.math.rsqrt(x)
else:
    @_jit
    def _rsqrt(x): return 1.0 / tl.sqrt(x)
if HAS_TRITON and hasattr(tl, "erf"):
    @_jit
    def _erf(a): return tl.erf(a)
elif HAS_TRITON and hasattr(getattr(tl, "math", None), "erf"):
    @_jit
    def _erf(a): return tl.math.erf(a)
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
def _smeared_k_f1_f2(r2, inv_sig, c0, inv_sig2, NEED_F2: tl.constexpr):
    x2 = r2 * inv_sig2
    small = x2 < 2.5e-3
    x4 = x2 * x2
    c0_s2 = c0 * inv_sig2
    k_t = c0 * (1.0 - 0.333333343 * x2 + 0.1 * x4)
    f1_t = c0_s2 * (0.666666687 - 0.4 * x2 + 0.142857149 * x4)
    r2_s = tl.where(small, 1.0, r2)
    inv_r = _rsqrt(r2_s)
    inv_r2 = inv_r * inv_r
    ex = tl.exp(-x2)
    k_g = _erf(r2_s * inv_r * inv_sig) * inv_r
    f1_g = (k_g - c0 * ex) * inv_r2
    k, f1 = tl.where(small, k_t, k_g), tl.where(small, f1_t, f1_g)
    if NEED_F2:
        f2_t = (c0_s2 * inv_sig2) * (-0.8 + 0.571428597 * x2 - 0.222222224 * x4)
        f2 = tl.where(small, f2_t, (2.0 * c0_s2 * ex - 3.0 * f1_g) * inv_r2)
    else:
        f2 = r2 * 0.0
    return k, f1, f2


@_jit
def _is_far_tile(bsph_ptr, b, tcx, tcy, tcz, rtg, far_pad, IS_FAST64: tl.constexpr):
    if IS_FAST64:
        tbx, tby, tbz, rtb = tl.load(bsph_ptr + b * 4 + 0), tl.load(bsph_ptr + b * 4 + 1), tl.load(bsph_ptr + b * 4 + 2), tl.load(bsph_ptr + b * 4 + 3)
        thr = rtg + far_pad + rtb
        far = ((tcx - tbx) * (tcx - tbx) + (tcy - tby) * (tcy - tby) + (tcz - tbz) * (tcz - tbz) > thr * thr) & (rtb >= 0.0) & (rtg >= 0.0)
    else:
        far = False
    return far


@_jit
def _tile_head(tg, pos_ptr, mask_ptr, g_ptr, gptr_ptr, bsph_ptr, n_atoms, IS_FAST64: tl.constexpr, IS_GROUPED: tl.constexpr, BLOCK: tl.constexpr):
    i0 = tg * BLOCK
    oi = i0 + tl.arange(0, BLOCK)
    ib = oi < n_atoms
    if IS_GROUPED:
        g_min, g_max = tl.load(g_ptr + i0), tl.load(g_ptr + tl.minimum(i0 + BLOCK - 1, n_atoms - 1))
        j_start, j_end, gi = tl.load(gptr_ptr + g_min), tl.load(gptr_ptr + g_max + 1), tl.load(g_ptr + oi, mask=ib, other=-1)
    else:
        j_start, j_end, gi = 0, n_atoms, 0
    tcx, tcy = (tl.load(bsph_ptr + tg * 4 + 0) if IS_FAST64 else 0.0), (tl.load(bsph_ptr + tg * 4 + 1) if IS_FAST64 else 0.0)
    tcz, rtg = (tl.load(bsph_ptr + tg * 4 + 2) if IS_FAST64 else 0.0), (tl.load(bsph_ptr + tg * 4 + 3) if IS_FAST64 else -1.0)
    act_i = ib & (tl.load(mask_ptr + oi, mask=ib, other=0) != 0)
    pix, piy, piz = tl.load(pos_ptr + oi * 3 + 0, mask=act_i, other=0.0), tl.load(pos_ptr + oi * 3 + 1, mask=act_i, other=0.0), tl.load(pos_ptr + oi * 3 + 2, mask=act_i, other=0.0)
    return oi, ib, act_i, gi, j_start, j_end, tcx, tcy, tcz, rtg, pix, piy, piz


@_jit
def _coulomb_fwd_kernel(pos_ptr, q_ptr, mask_ptr, g_ptr, gptr_ptr, bsph_ptr, phi_ptr, fld_ptr, n_atoms,
                        inv_sig, c0, inv_sig2, far_pad, IS_FAST64: tl.constexpr, IS_GROUPED: tl.constexpr,
                        NEED_FLD: tl.constexpr, BLOCK: tl.constexpr):
    oi, ib, act_i, gi, j_start, j_end, tcx, tcy, tcz, rtg, pix, piy, piz = _tile_head(
        tl.program_id(0), pos_ptr, mask_ptr, g_ptr, gptr_ptr, bsph_ptr, n_atoms, IS_FAST64, IS_GROUPED, BLOCK)
    ap, afx, afy, afz = tl.zeros([BLOCK], dtype=tl.float32), tl.zeros([BLOCK], dtype=tl.float32), tl.zeros([BLOCK], dtype=tl.float32), tl.zeros([BLOCK], dtype=tl.float32)
    for j_base in range(j_start, j_end, BLOCK):
        oj = j_base + tl.arange(0, BLOCK)
        jb = oj < j_end
        act_j = jb & (tl.load(mask_ptr + oj, mask=jb, other=0) != 0)
        pjx, pjy, pjz, qj = tl.load(pos_ptr + oj * 3 + 0, mask=act_j, other=0.0), tl.load(pos_ptr + oj * 3 + 1, mask=act_j, other=0.0), tl.load(pos_ptr + oj * 3 + 2, mask=act_j, other=0.0), tl.load(q_ptr + oj, mask=act_j, other=0.0)
        dx, dy, dz = pix[:, None] - pjx[None, :], piy[:, None] - pjy[None, :], piz[:, None] - pjz[None, :]
        r2 = tl.maximum(dx * dx + dy * dy + dz * dz, 1.0e-16)
        if _is_far_tile(bsph_ptr, j_base // BLOCK, tcx, tcy, tcz, rtg, far_pad, IS_FAST64):
            ir = _rsqrt(r2)
            ap += tl.sum(qj[None, :] * ir, axis=1)
            if NEED_FLD:
                wf = qj[None, :] * (ir * ir * ir)
                afx, afy, afz = afx + tl.sum(wf * dx, axis=1), afy + tl.sum(wf * dy, axis=1), afz + tl.sum(wf * dz, axis=1)
        else:
            valid = act_i[:, None] & act_j[None, :] & (oi[:, None] != oj[None, :])
            if IS_GROUPED:
                valid = valid & (gi[:, None] == tl.load(g_ptr + oj, mask=jb, other=-2)[None, :])
            k, f1, _ = _smeared_k_f1_f2(r2, inv_sig, c0, inv_sig2, False)
            wj = tl.where(valid, qj[None, :], 0.0)
            ap += tl.sum(wj * k, axis=1)
            if NEED_FLD:
                wf = wj * f1
                afx, afy, afz = afx + tl.sum(wf * dx, axis=1), afy + tl.sum(wf * dy, axis=1), afz + tl.sum(wf * dz, axis=1)
    tl.store(phi_ptr + oi, tl.where(act_i, ap, 0.0), mask=ib)
    if NEED_FLD:
        tl.store(fld_ptr + oi * 3 + 0, tl.where(act_i, afx, 0.0), mask=ib)
        tl.store(fld_ptr + oi * 3 + 1, tl.where(act_i, afy, 0.0), mask=ib)
        tl.store(fld_ptr + oi * 3 + 2, tl.where(act_i, afz, 0.0), mask=ib)


@_jit
def _coulomb_bwd_kernel(pos_ptr, q_ptr, mask_ptr, g_ptr, gptr_ptr, bsph_ptr, gphi_ptr, gfld_ptr, dpos_ptr, dq_ptr, n_atoms,
                        inv_sig, c0, inv_sig2, far_pad, IS_FAST64: tl.constexpr, IS_GROUPED: tl.constexpr, HAS_GFLD: tl.constexpr,
                        NEED_DPOS: tl.constexpr, NEED_DQ: tl.constexpr, BLOCK: tl.constexpr):
    oi, ib, act_i, gi, j_start, j_end, tcx, tcy, tcz, rtg, pix, piy, piz = _tile_head(
        tl.program_id(0), pos_ptr, mask_ptr, g_ptr, gptr_ptr, bsph_ptr, n_atoms, IS_FAST64, IS_GROUPED, BLOCK)
    qi, gpi = tl.load(q_ptr + oi, mask=act_i, other=0.0), tl.load(gphi_ptr + oi, mask=act_i, other=0.0)
    geix = tl.load(gfld_ptr + oi * 3 + 0, mask=act_i, other=0.0) if HAS_GFLD else tl.zeros([BLOCK], dtype=tl.float32)
    geiy = tl.load(gfld_ptr + oi * 3 + 1, mask=act_i, other=0.0) if HAS_GFLD else tl.zeros([BLOCK], dtype=tl.float32)
    geiz = tl.load(gfld_ptr + oi * 3 + 2, mask=act_i, other=0.0) if HAS_GFLD else tl.zeros([BLOCK], dtype=tl.float32)
    dpx, dpy, dpz, dq = tl.zeros([BLOCK], dtype=tl.float32), tl.zeros([BLOCK], dtype=tl.float32), tl.zeros([BLOCK], dtype=tl.float32), tl.zeros([BLOCK], dtype=tl.float32)
    for j_base in range(j_start, j_end, BLOCK):
        oj = j_base + tl.arange(0, BLOCK)
        jb = oj < j_end
        act_j = jb & (tl.load(mask_ptr + oj, mask=jb, other=0) != 0)
        pjx, pjy, pjz = tl.load(pos_ptr + oj * 3 + 0, mask=act_j, other=0.0), tl.load(pos_ptr + oj * 3 + 1, mask=act_j, other=0.0), tl.load(pos_ptr + oj * 3 + 2, mask=act_j, other=0.0)
        qj, gpj = tl.load(q_ptr + oj, mask=act_j, other=0.0), tl.load(gphi_ptr + oj, mask=act_j, other=0.0)
        gejx = tl.load(gfld_ptr + oj * 3 + 0, mask=act_j, other=0.0) if HAS_GFLD else tl.zeros([BLOCK], dtype=tl.float32)
        gejy = tl.load(gfld_ptr + oj * 3 + 1, mask=act_j, other=0.0) if HAS_GFLD else tl.zeros([BLOCK], dtype=tl.float32)
        gejz = tl.load(gfld_ptr + oj * 3 + 2, mask=act_j, other=0.0) if HAS_GFLD else tl.zeros([BLOCK], dtype=tl.float32)
        dx, dy, dz = pix[:, None] - pjx[None, :], piy[:, None] - pjy[None, :], piz[:, None] - pjz[None, :]
        r2 = tl.maximum(dx * dx + dy * dy + dz * dz, 1.0e-16)
        if _is_far_tile(bsph_ptr, j_base // BLOCK, tcx, tcy, tcz, rtg, far_pad, IS_FAST64):
            k = _rsqrt(r2)
            ir2 = k * k
            f1 = k * ir2
            f2 = (-3.0 * f1 * ir2) if HAS_GFLD else (r2 * 0.0)
        else:
            valid = act_i[:, None] & act_j[None, :] & (oi[:, None] != oj[None, :])
            if IS_GROUPED:
                valid = valid & (gi[:, None] == tl.load(g_ptr + oj, mask=jb, other=-2)[None, :])
            k0, f10, f20 = _smeared_k_f1_f2(r2, inv_sig, c0, inv_sig2, HAS_GFLD)
            k, f1, f2 = tl.where(valid, k0, 0.0), tl.where(valid, f10, 0.0), tl.where(valid, f20, 0.0)
        if NEED_DQ:
            if HAS_GFLD:
                dq += tl.sum(gpj[None, :] * k - f1 * (gejx[None, :] * dx + gejy[None, :] * dy + gejz[None, :] * dz), axis=1)
            else:
                dq += tl.sum(gpj[None, :] * k, axis=1)
        if NEED_DPOS:
            s_phi = gpi[:, None] * qj[None, :] + gpj[None, :] * qi[:, None]
            if HAS_GFLD:
                vx, vy, vz = qj[None, :] * geix[:, None] - qi[:, None] * gejx[None, :], qj[None, :] * geiy[:, None] - qi[:, None] * gejy[None, :], qj[None, :] * geiz[:, None] - qi[:, None] * gejz[None, :]
                cd = -s_phi * f1 + f2 * (vx * dx + vy * dy + vz * dz)
                dpx, dpy, dpz = dpx + tl.sum(cd * dx + f1 * vx, axis=1), dpy + tl.sum(cd * dy + f1 * vy, axis=1), dpz + tl.sum(cd * dz + f1 * vz, axis=1)
            else:
                cd = -s_phi * f1
                dpx, dpy, dpz = dpx + tl.sum(cd * dx, axis=1), dpy + tl.sum(cd * dy, axis=1), dpz + tl.sum(cd * dz, axis=1)
    if NEED_DQ:
        tl.store(dq_ptr + oi, tl.where(act_i, dq, 0.0), mask=ib)
    if NEED_DPOS:
        tl.store(dpos_ptr + oi * 3 + 0, tl.where(act_i, dpx, 0.0), mask=ib)
        tl.store(dpos_ptr + oi * 3 + 1, tl.where(act_i, dpy, 0.0), mask=ib)
        tl.store(dpos_ptr + oi * 3 + 2, tl.where(act_i, dpz, 0.0), mask=ib)


def build_group_ptr(graph: Tensor, n_graphs: int) -> Tensor:
    dev = norm_device(graph.device)
    if not (torch.compiler.is_compiling() or proxy_tracing() or is_fake(graph)) and int(n_graphs) <= 1:
        n = int(graph.numel())
        ptr = _PTR_G1_CACHE.get((dev, n))
        if ptr is not None:
            return ptr
        if capturing():
            return torch.arange(2, dtype=torch.int32, device=graph.device) * n
        if len(_PTR_G1_CACHE) >= 64:
            _PTR_G1_CACHE.clear()
        ptr = _PTR_G1_CACHE[(dev, n)] = torch.tensor([0, n], dtype=torch.int32, device=graph.device)
        return ptr
    if n_graphs <= 1:
        return torch.arange(2, dtype=torch.int32, device=graph.device) * graph.shape[0]
    hi = torch.searchsorted(graph.to(torch.int32).contiguous(), torch.arange(n_graphs, dtype=torch.int32, device=graph.device), right=True).to(torch.int32)
    return torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), hi], dim=0)


def build_coulomb_mask(atom_mask: Tensor, graph: Tensor, n_graphs: int, cap: int | None = None) -> Tensor:
    sym = torch.compiler.is_compiling() or proxy_tracing() or is_fake(atom_mask)
    if cap is None or (not sym and (int(atom_mask.numel()) == 0 or int(cap) >= int(atom_mask.numel()))):
        return atom_mask.to(torch.uint8).contiguous()
    real = atom_mask.bool()
    ri = real.long()
    counts = torch.zeros(n_graphs, dtype=torch.long, device=graph.device).index_add(0, graph, ri)
    rank = torch.cumsum(ri, 0) - 1 - (torch.cumsum(counts, 0) - counts)[graph]
    return (real & (rank < cap)).to(torch.uint8).contiguous()


def build_fast64_meta(pos: Tensor, mask_u8: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dev, n = pos.device, pos.shape[0]
    n_tiles = (n + 63) // 64 if (torch.compiler.is_compiling() or proxy_tracing() or is_fake(pos)) else max(1, (int(n) + 63) // 64)
    n_pad = n_tiles * 64
    with torch.no_grad():
        pf, act = upcast_fp32(pos.detach()).to(torch.float32), mask_u8 != 0
        pmin = torch.where(act.unsqueeze(-1), pf, torch.full_like(pf, 1.0e20)).amin(0)
        pmax = torch.maximum(torch.where(act.unsqueeze(-1), pf, torch.full_like(pf, -1.0e20)).amax(0), pmin + 1.0e-3)
        c = ((pf - pmin) / (pmax - pmin) * 1023.0).clamp(0.0, 1023.0).to(torch.int32)

        def _part(x: Tensor) -> Tensor:
            x = (x | (x << 16)) & 0x030000FF
            x = (x | (x << 8)) & 0x0300F00F
            x = (x | (x << 4)) & 0x030C30C3
            return (x | (x << 2)) & 0x09249249

        mkey = torch.where(act, _part(c[:, 0]) | (_part(c[:, 1]) << 1) | (_part(c[:, 2]) << 2), torch.full((n,), 0x7FFFFFFF, dtype=torch.int32, device=dev))
        perm = torch.argsort(mkey, stable=True)
        pos_s, mask_s = pf.index_select(0, perm), torch.zeros(n_pad, dtype=torch.uint8, device=dev)
        mask_s[:n] = mask_u8.index_select(0, perm)
        pos_sp = torch.cat([pos_s, pos_s.new_zeros((n_pad - n, 3))], 0).contiguous()
        tb, tm = pos_sp.view(n_tiles, 64, 3), (mask_s != 0).view(n_tiles, 64)
        cent = (tb.amin(1) + tb.amax(1)) * 0.5
        rad = torch.where(tm.all(1), (tb - cent.unsqueeze(1)).norm(dim=-1).amax(1), torch.full((n_tiles,), -1.0, dtype=torch.float32, device=dev))
        return perm, pos_sp, mask_s, torch.cat([cent, rad.unsqueeze(-1)], -1).contiguous()


if not hasattr(getattr(torch.ops, "lam4", None), "triton_coulomb_fwd"):
    @torch.library.custom_op("lam4::triton_coulomb_fwd", mutates_args=())
    def _triton_coulomb_fwd_op(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor,
                               group_ptr: Tensor, bsph: Tensor, mode: int, sigma: float, need_field: bool) -> tuple[Tensor, Tensor]:
        n = int(pos.shape[0])
        phi, fld = torch.empty(n, dtype=torch.float32, device=pos.device), (torch.empty((n, 3), dtype=torch.float32, device=pos.device) if need_field else pos.new_empty((0, 3), dtype=torch.float32))
        if n == 0:
            return phi, fld
        inv_sig = 1.0 / float(sigma)
        c0, inv_sig2, far_pad, q_c = 2.0 * inv_sig / math.sqrt(math.pi), inv_sig * inv_sig, 5.5 * float(sigma), q.to(torch.float32).contiguous()
        from lam4.kernels.cuda_coulomb import cuda_coulomb_fwd, has_cuda_coulomb
        use_cu = has_cuda_coulomb(pos.device)
        def _run_fwd(pos_k, q_k, fld_k, n_k, *, fast64: bool, grouped: bool, grid_sz: int, block: int):
            if use_cu:
                try:
                    cuda_coulomb_fwd(pos_k, q_k, mask_work, g_sorted, group_ptr, bsph, fld_k[0], fld_k[1], n_k, inv_sig, c0, inv_sig2, far_pad, is_fast64=fast64, is_grouped=grouped, need_fld=need_field)
                    return
                except Exception:  # noqa: BLE001
                    if not HAS_TRITON:
                        raise
            _coulomb_fwd_kernel[(grid_sz,)](pos_k, q_k, mask_work, g_sorted, group_ptr, bsph, fld_k[0], fld_k[1], n_k, inv_sig, c0, inv_sig2, far_pad, IS_FAST64=fast64, IS_GROUPED=grouped, NEED_FLD=need_field, BLOCK=block, num_warps=4)

        if mode == 1:
            n_tiles = int(bsph.shape[0])
            n_pad = n_tiles * 64
            qs = torch.zeros(n_pad, dtype=torch.float32, device=pos.device)
            qs[:n] = q_c.index_select(0, perm)
            phi_s, fld_s = torch.empty(n_pad, dtype=torch.float32, device=pos.device), (torch.empty((n_pad, 3), dtype=torch.float32, device=pos.device) if need_field else pos_work[:1])
            _run_fwd(pos_work, qs, (phi_s, fld_s), n_pad, fast64=True, grouped=False, grid_sz=n_tiles, block=64)
            phi.index_copy_(0, perm, phi_s[:n])
            if need_field:
                fld.index_copy_(0, perm, fld_s[:n])
        elif mode == 2:
            qs = q_c.index_select(0, perm).contiguous()
            phi_s, fld_s = torch.empty_like(phi), (torch.empty_like(fld) if need_field else pos_work[:1])
            _run_fwd(pos_work, qs, (phi_s, fld_s), n, fast64=False, grouped=True, grid_sz=(n + 31) // 32, block=32)
            phi.index_copy_(0, perm, phi_s)
            if need_field:
                fld.index_copy_(0, perm, fld_s)
        else:
            _run_fwd(pos_work, q_c, (phi, fld if need_field else pos_work[:1]), n, fast64=False, grouped=False, grid_sz=(n + 63) // 64, block=64)
        return phi, fld

    @_triton_coulomb_fwd_op.register_fake
    def _triton_coulomb_fwd_fake(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor,
                                 group_ptr: Tensor, bsph: Tensor, mode: int, sigma: float, need_field: bool) -> tuple[Tensor, Tensor]:
        return pos.new_empty(pos.shape[0], dtype=torch.float32), pos.new_empty((pos.shape[0], 3) if need_field else (0, 3), dtype=torch.float32)

    @torch.library.custom_op("lam4::triton_coulomb_bwd", mutates_args=())
    def _triton_coulomb_bwd_op(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, group_ptr: Tensor,
                               bsph: Tensor, g_phi: Tensor, g_fld: Tensor, mode: int, sigma: float, has_gfld: bool, need_dpos: bool, need_dq: bool) -> tuple[Tensor, Tensor]:
        n = int(pos.shape[0])
        dpos, dq = torch.zeros((n, 3), dtype=torch.float32, device=pos.device), torch.zeros(n, dtype=torch.float32, device=pos.device)
        if n == 0 or not (need_dpos or need_dq):
            return dpos, dq
        inv_sig = 1.0 / float(sigma)
        c0, inv_sig2, far_pad = 2.0 * inv_sig / math.sqrt(math.pi), inv_sig * inv_sig, 5.5 * float(sigma)
        q_c, gp_c, gf_c = q.to(torch.float32).contiguous(), g_phi.to(torch.float32).contiguous(), (g_fld.to(torch.float32).contiguous() if has_gfld else pos_work[:1])
        from lam4.kernels.cuda_coulomb import cuda_coulomb_bwd, has_cuda_coulomb
        use_cu = has_cuda_coulomb(pos.device)
        def _run_bwd(q_k, gp_k, gf_k, dp_k, dq_k, n_k, *, fast64: bool, grouped: bool, grid_sz: int, block: int):
            if use_cu:
                try:
                    cuda_coulomb_bwd(pos_work, q_k, mask_work, g_sorted, group_ptr, bsph, gp_k, gf_k, dp_k, dq_k, n_k, inv_sig, c0, inv_sig2, far_pad, is_fast64=fast64, is_grouped=grouped, has_gfld=has_gfld, need_dpos=need_dpos, need_dq=need_dq)
                    return
                except Exception:  # noqa: BLE001
                    if not HAS_TRITON:
                        raise
            _coulomb_bwd_kernel[(grid_sz,)](pos_work, q_k, mask_work, g_sorted, group_ptr, bsph, gp_k, gf_k, dp_k, dq_k, n_k, inv_sig, c0, inv_sig2, far_pad, IS_FAST64=fast64, IS_GROUPED=grouped, HAS_GFLD=has_gfld, NEED_DPOS=need_dpos, NEED_DQ=need_dq, BLOCK=block, num_warps=4)

        if mode == 1:
            n_tiles = int(bsph.shape[0])
            n_pad = n_tiles * 64
            qs, gps = torch.zeros(n_pad, dtype=torch.float32, device=pos.device), torch.zeros(n_pad, dtype=torch.float32, device=pos.device)
            qs[:n], gps[:n] = q_c.index_select(0, perm), gp_c.index_select(0, perm)
            gfs = torch.zeros((n_pad, 3), dtype=torch.float32, device=pos.device) if has_gfld else pos_work[:1]
            if has_gfld:
                gfs[:n] = gf_c.index_select(0, perm)
            dp_s, dq_s = (torch.empty((n_pad, 3), dtype=torch.float32, device=pos.device) if need_dpos else pos_work[:1]), (torch.empty(n_pad, dtype=torch.float32, device=pos.device) if need_dq else q_c[:1])
            _run_bwd(qs, gps, gfs, dp_s, dq_s, n_pad, fast64=True, grouped=False, grid_sz=n_tiles, block=64)
            if need_dpos: dpos.index_copy_(0, perm, dp_s[:n])
            if need_dq: dq.index_copy_(0, perm, dq_s[:n])
        elif mode == 2:
            qs, gps, gfs = q_c.index_select(0, perm).contiguous(), gp_c.index_select(0, perm).contiguous(), (gf_c.index_select(0, perm).contiguous() if has_gfld else pos_work[:1])
            dp_s, dq_s = (torch.empty_like(dpos) if need_dpos else pos_work[:1]), (torch.empty_like(dq) if need_dq else q_c[:1])
            _run_bwd(qs, gps, gfs, dp_s, dq_s, n, fast64=False, grouped=True, grid_sz=(n + 31) // 32, block=32)
            if need_dpos: dpos.index_copy_(0, perm, dp_s)
            if need_dq: dq.index_copy_(0, perm, dq_s)
        else:
            _run_bwd(q_c, gp_c, gf_c, dpos, dq, n, fast64=False, grouped=False, grid_sz=(n + 63) // 64, block=64)
        return dpos, dq

    @_triton_coulomb_bwd_op.register_fake
    def _triton_coulomb_bwd_fake(pos: Tensor, q: Tensor, pos_work: Tensor, mask_work: Tensor, perm: Tensor, g_sorted: Tensor, group_ptr: Tensor,
                                 bsph: Tensor, g_phi: Tensor, g_fld: Tensor, mode: int, sigma: float, has_gfld: bool, need_dpos: bool, need_dq: bool) -> tuple[Tensor, Tensor]:
        return torch.empty_like(pos, dtype=torch.float32), torch.empty_like(q, dtype=torch.float32)

    def _setup_ctx(ctx, inputs, output):
        pos, q, pos_work, mask_work, perm, g_sorted, group_ptr, bsph, mode, sigma, need_field = inputs
        ctx.save_for_backward(pos, q, pos_work, mask_work, perm, g_sorted, group_ptr, bsph)
        ctx.mode, ctx.sigma, ctx.need_field = int(mode), float(sigma), bool(need_field)

    def _bwd_ctx(ctx, g_phi: Tensor | None, g_fld: Tensor | None):
        pos, q, pos_work, mask_work, perm, g_sorted, group_ptr, bsph = ctx.saved_tensors
        need_dp, need_dq = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        if not (need_dp or need_dq):
            return (None,) * 11
        gp, has_gf = (g_phi if g_phi is not None else torch.zeros_like(q, dtype=torch.float32)), bool(ctx.need_field and g_fld is not None)
        dp, dq = torch.ops.lam4.triton_coulomb_bwd.default(pos, q, pos_work, mask_work, perm, g_sorted, group_ptr, bsph, gp, g_fld if has_gf else pos[:1], ctx.mode, ctx.sigma, has_gf, need_dp, need_dq)
        return (dp.to(pos.dtype) if need_dp else None, dq.to(q.dtype) if need_dq else None) + (None,) * 9

    _triton_coulomb_fwd_op.register_autograd(_bwd_ctx, setup_context=_setup_ctx)


class TritonExactCoulomb:
    """Drop-in CUDA Triton replacement for `ExactCoulomb` (`O(N)` memory, cached step geometry across stages + readout)."""
    def __init__(self, pos: Tensor, atom_mask: Tensor, graph: Tensor, n_graphs: int, cap: int | None = None,
                 sigma: float = 2.0, *, group_ptr: Tensor | None = None, mask_u8: Tensor | None = None, fast64_threshold: int = FAST64_THRESHOLD) -> None:
        self.pos, self.n_atoms, self.n_graphs, self.sigma = pos, pos.shape[0], n_graphs, float(sigma)
        dev, raw_mask, pos_f32 = pos.device, (build_coulomb_mask(atom_mask, graph, self.n_graphs, cap) if mask_u8 is None else mask_u8), pos.detach().to(torch.float32).contiguous()
        sym = torch.compiler.is_compiling() or proxy_tracing() or is_fake(pos)
        if self.n_graphs > 1 and (sym or int(self.n_atoms) > 0):
            self.mode, self.perm = 2, torch.argsort(graph, stable=True)
            self.pos_work, self.mask_work = pos_f32.index_select(0, self.perm).contiguous(), raw_mask.index_select(0, self.perm).contiguous()
            self.g_sorted = graph.index_select(0, self.perm).to(torch.int32).contiguous()
            self.group_ptr, self.bsph = (build_group_ptr(self.g_sorted, self.n_graphs) if group_ptr is None else group_ptr), torch.empty((0, 4), dtype=torch.float32, device=dev)
        elif self.n_graphs == 1 and self.n_atoms > int(fast64_threshold):
            self.mode = 1
            self.perm, self.pos_work, self.mask_work, self.bsph = build_fast64_meta(pos, raw_mask)
            self.g_sorted = self.group_ptr = torch.empty(0, dtype=torch.int32, device=dev)
        else:
            self.mode, self.pos_work, self.mask_work = 0, pos_f32, raw_mask
            self.perm, self.g_sorted = torch.empty(0, dtype=torch.int64, device=dev), torch.empty(0, dtype=torch.int32, device=dev)
            self.group_ptr, self.bsph = self.g_sorted, torch.empty((0, 4), dtype=torch.float32, device=dev)

    def phi(self, q: Tensor) -> Tensor:
        if not (torch.compiler.is_compiling() or proxy_tracing() or is_fake(self.pos) or has_triton_coulomb(self.pos.device)):
            return self.phi_field_efg(q)[0]
        return torch.ops.lam4.triton_coulomb_fwd.default(self.pos, q, self.pos_work, self.mask_work, self.perm, self.g_sorted, self.group_ptr, self.bsph, self.mode, self.sigma, False)[0]

    def phi_field(self, q: Tensor) -> tuple[Tensor, Tensor]:
        if not (torch.compiler.is_compiling() or proxy_tracing() or is_fake(self.pos) or has_triton_coulomb(self.pos.device)):
            return self.phi_field_efg(q)[:2]
        return torch.ops.lam4.triton_coulomb_fwd.default(self.pos, q, self.pos_work, self.mask_work, self.perm, self.g_sorted, self.group_ptr, self.bsph, self.mode, self.sigma, True)

    def phi_field_efg(self, q: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        from lam4.kernels.triton_coulomb_bwd2 import eval_coulomb_phi_field_efg
        return eval_coulomb_phi_field_efg(self.pos, q, self.pos_work, self.mask_work, self.perm, self.g_sorted, self.group_ptr, self.bsph, self.mode, self.sigma)

    def phi_field_multipole(self, q: Tensor, p: Tensor | None = None, *, need_efg: bool = False) -> tuple[Tensor, Tensor, Tensor | None]:
        if p is not None:
            raise ValueError("TritonExactCoulomb.phi_field_multipole supports monopole-only (p=None); use OpenMultipoleBackend when sqe_pqeq=True.")
        if need_efg:
            return self.phi_field_efg(q)
        phi, fld = self.phi_field(q)
        return phi, fld, None


if HAS_TRITON:
    from lam4.kernels.triton_coulomb_bwd2 import register_coulomb_bwd2
    register_coulomb_bwd2()
