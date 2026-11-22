"""Dual-CSR neighbor topology plan (`CSRPlan`) and precomputed Horner cubic B-spline tables."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lam4.core.consts import capturing, storable
from lam4.core.edge_chunks import Chunk, chunks_to, make_chunks, stream_chunk_edges
from lam4.core.edge_stream import _StreamVJP, empty_z
from lam4.core.edge_terms import EdgeCfg
from lam4.core.layout import StageLayout
from lam4.core.radial import N_RBF, R_CUT
from lam4.core.spline_tables import DEFAULT_GRID, BSplineTables, windows
from lam4.physics.nlh_tables import FLOOR_WIDTH, nlh_tables_f64

# Exact global ceiling on (r_min + FLOOR_WIDTH)^2 across all Z in 0..127 (max d_off = 1.2749 A -> d_off^2 = 1.6253 A^2).
MAX_DOFF_R2: float = float((nlh_tables_f64()[1].max() + FLOOR_WIDTH + 1e-3) ** 2)

_CPU_RMIN_F32: Tensor = torch.from_numpy(nlh_tables_f64()[1]).to(dtype=torch.float32).contiguous()
_RMIN_F32_CACHE: dict[tuple[str, int | None], Tensor] = {("cpu", None): _CPU_RMIN_F32}
_DUMMY_F32: dict[tuple[str, int | None], Tensor] = {}
_DUMMY_I64: dict[tuple[str, int | None], Tensor] = {}
_P_HORNER64: dict[tuple[int, float, int], Tensor] = {}
_P_HORNER_DEV: dict[tuple[int, float, torch.dtype, torch.device], Tensor] = {}


def norm_device(device: torch.device | str) -> torch.device:
    """Resolve default device index (`cuda` -> `cuda:current`, `mps` -> `mps:0`) for canonical cache keys & `.to()` checks."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        idx = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        return torch.device("cuda", idx)
    if dev.type == "mps" and dev.index is None:
        return torch.device("mps", 0)
    return dev


def is_cuda_capturing(device: torch.device | str | None = None) -> bool:
    if device is not None and torch.device(device).type != "cuda":
        return False
    return capturing()


def raw_proxy_tracing() -> bool:
    if torch.compiler.is_compiling():
        return False
    try:
        from torch.fx.experimental.proxy_tensor import get_proxy_mode
    except ImportError:
        return False
    return get_proxy_mode() is not None


def cacheable_tensor(t: Tensor) -> bool:
    """True iff `t` is a plain eager tensor NOT allocated on a capturing CUDA Graph private pool."""
    return bool(storable(t) and not is_cuda_capturing(t.device))


def select_triton_block_warps(n_atoms: int | None = None) -> tuple[int, int]:
    """A100-calibrated `(BLOCK_ATOMS, num_warps)` selector (`(2,2)` <=1024, `(4,4)` mid/default, `(8,4)` >=2048)."""
    env_ba, env_nw = os.environ.get("LAM4_TRITON_BLOCK_ATOMS"), os.environ.get("LAM4_TRITON_WARPS")
    if env_ba is not None or env_nw is not None:
        ba = int(env_ba) if env_ba is not None else 4
        nw = int(env_nw) if env_nw is not None else (2 if ba <= 2 else 4)
        if ba <= 0 or (ba & (ba - 1)) != 0 or nw <= 0 or (nw & (nw - 1)) != 0:
            raise ValueError(f"LAM4_TRITON_BLOCK_ATOMS ({ba}) and LAM4_TRITON_WARPS ({nw}) must be powers of 2")
        return ba, nw
    if n_atoms is None or (1024 < n_atoms < 2048):
        return 4, 4
    return (2, 2) if n_atoms <= 1024 else (8, 4)


def dummy_f32(device: torch.device) -> Tensor:
    dev = norm_device(device)
    key = (dev.type, dev.index)
    t = _DUMMY_F32.get(key)
    if t is None:
        t = torch.zeros(1, dtype=torch.float32, device=dev)
        if cacheable_tensor(t):
            _DUMMY_F32[key] = t
    return t


def dummy_i64(device: torch.device) -> Tensor:
    dev = norm_device(device)
    key = (dev.type, dev.index)
    t = _DUMMY_I64.get(key)
    if t is None:
        t = torch.zeros(1, dtype=torch.int64, device=dev)
        if cacheable_tensor(t):
            _DUMMY_I64[key] = t
    return t


@dataclass
class CSRPlan:
    """Receiver-CSR + Sender-CSR dual topology in int32 for atomic-free forward & backward passes."""

    n_atoms: int
    n_edges: int
    recv_ptr: Tensor   # [N + 1] int32
    send_idx: Tensor   # [E] int32 (aligned with edge order 0..E-1)
    recv_idx: Tensor   # [E] int32 (aligned with edge order 0..E-1, non-decreasing)
    send_ptr: Tensor   # [N + 1] int32
    send_perm: Tensor  # [E] int32: edge indices e grouped by send_idx[e]
    chunks: list | None = None  # CUDA streaming-VJP chunk plan, built on first backward and reused while the plan (edge list) lives

    def to(self, device: torch.device | str) -> "CSRPlan":
        dev = norm_device(device)
        if norm_device(self.recv_ptr.device) == dev:
            return self
        return CSRPlan(
            self.n_atoms,
            self.n_edges,
            self.recv_ptr.to(dev),
            self.send_idx.to(dev),
            self.recv_idx.to(dev),
            self.send_ptr.to(dev),
            self.send_perm.to(dev),
            chunks=None if self.chunks is None else chunks_to(self.chunks, dev),
        )


def rmin_table_f32(device: torch.device) -> Tensor:
    """Cached [16384] float32 NLH r_min lookup table indexed by `(z_i << 7) + z_j`."""
    dev = norm_device(device)
    key = (dev.type, dev.index)
    t = _RMIN_F32_CACHE.get(key)
    if t is None:
        t = _CPU_RMIN_F32.to(device=dev)
        if cacheable_tensor(t):
            _RMIN_F32_CACHE[key] = t
    return t


def _win_to_horner(win: Tensor) -> Tensor:
    """Convert window coefficients `[G, 4, C]` `(c0, c1, c2, c3)` into Horner `[G, 4, C]` `(a0, a1, a2, a3)`."""
    c0, c1, c2, c3 = win[:, 0], win[:, 1], win[:, 2], win[:, 3]
    a0 = (c0 + 4.0 * c1 + c2) * (1.0 / 6.0)
    a1 = 0.5 * (c2 - c0)
    a2 = 0.5 * (c0 - 2.0 * c1 + c2)
    a3 = (-c0 + 3.0 * c1 - 3.0 * c2 + c3) * (1.0 / 6.0)
    return torch.stack((a0, a1, a2, a3), dim=1).contiguous()


def _default_p_horner(r_cut: float, dtype: torch.dtype, device: torch.device, grid: int = DEFAULT_GRID) -> Tensor:
    dev = norm_device(device)
    key = (int(grid), float(r_cut), dtype, dev)
    c = _P_HORNER_DEV.get(key)
    if c is None:
        k64 = (int(grid), float(r_cut), N_RBF)
        h64 = _P_HORNER64.get(k64)
        if h64 is None:
            p64 = BSplineTables(1, grid, r_cut)._p64
            h64 = _win_to_horner(windows(p64)).detach().contiguous()
            if cacheable_tensor(h64):
                _P_HORNER64[k64] = h64
        c = h64.to(dtype=dtype).to(device=dev).contiguous()
        if cacheable_tensor(c):
            _P_HORNER_DEV[key] = c
    return c


def warm_csr_nlh_caches(
    device: torch.device | str, r_cut: float = R_CUT, dtype: torch.dtype = torch.float32
) -> None:
    """Pre-populate CSR & NLH constant lookup tables on `device` prior to CUDA Graph capture."""
    dev = norm_device(device)
    if is_cuda_capturing(dev) or torch.compiler.is_compiling() or raw_proxy_tracing():
        return
    dummy_f32(dev)
    dummy_i64(dev)
    rmin_table_f32(dev)
    _default_p_horner(r_cut, dtype, dev)
    from lam4.kernels.triton_nlh import _nlh_cb_jet_f32

    _nlh_cb_jet_f32(dev)


def coef_to_horner_poly(coef: Tensor, r_cut: float = R_CUT, *, precise: bool = False) -> Tensor:
    """Build `[G, 4, C]` contiguous Horner cubic B-spline table from factored `[32, C]`, `[G+3, C]`, or `windows` `[G, 4, C]`."""
    if coef.ndim == 2 and coef.shape[0] == N_RBF:
        if precise:
            p_h64 = _default_p_horner(r_cut, torch.float64, torch.device("cpu"))
            w64 = coef.detach().cpu().to(torch.float64)
            out64 = (p_h64.view(-1, N_RBF) @ w64).view(DEFAULT_GRID, 4, coef.shape[1])
            return out64.to(dtype=coef.dtype).to(device=coef.device).contiguous()
        p_h = _default_p_horner(r_cut, coef.dtype, coef.device)
        if coef.device.type == "cuda" and not torch.compiler.is_compiling() and not raw_proxy_tracing():
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                with torch.amp.autocast("cuda", enabled=False):
                    return (p_h.view(-1, N_RBF) @ coef).view(DEFAULT_GRID, 4, coef.shape[1]).contiguous()
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        return (p_h.view(-1, N_RBF) @ coef).view(DEFAULT_GRID, 4, coef.shape[1]).contiguous()
    if precise:
        win64 = windows(coef.detach().cpu().to(torch.float64))
        return _win_to_horner(win64).to(dtype=coef.dtype).to(device=coef.device).contiguous()
    return _win_to_horner(windows(coef))


def build_csr_plan_cpu(
    recv_np: np.ndarray, send_np: np.ndarray, n_atoms: int, *, active_edges: int | None = None,
    edge_mask: np.ndarray | Tensor | None = None,
) -> CSRPlan:
    """Fast NumPy CPU builder (`active_edges`/`edge_mask` zero pad-atom degree while keeping `[n_e]` buffer shapes)."""
    n_e = int(len(recv_np))
    if n_atoms <= 0 or n_e == 0:
        z_ptr = torch.zeros(max(0, n_atoms) + 1, dtype=torch.int32)
        z_idx = torch.zeros(0, dtype=torch.int32)
        return CSRPlan(max(0, n_atoms), 0, z_ptr, z_idx, z_idx, z_ptr.clone(), z_idx.clone())
    r_i32 = np.ascontiguousarray(recv_np, dtype=np.int32)
    s_i32 = np.ascontiguousarray(send_np, dtype=np.int32)
    rptr = np.empty(n_atoms + 1, dtype=np.int32)
    rptr[0] = 0
    sptr = np.empty(n_atoms + 1, dtype=np.int32)
    sptr[0] = 0
    if active_edges is None and edge_mask is not None:
        em = np.asarray(edge_mask.detach().cpu().numpy() if torch.is_tensor(edge_mask) else edge_mask, dtype=bool)
        np.cumsum(np.bincount(r_i32[em], minlength=n_atoms), dtype=np.int32, out=rptr[1:])
        np.cumsum(np.bincount(s_i32[em], minlength=n_atoms), dtype=np.int32, out=sptr[1:])
        sperm = np.ascontiguousarray(np.argsort(np.where(em, s_i32, n_atoms), kind="mergesort").astype(np.int32, copy=False))
    else:
        e_act = n_e if active_edges is None else max(0, min(int(active_edges), n_e))
        r_act, s_act = r_i32[:e_act], s_i32[:e_act]
        np.cumsum(np.bincount(r_act, minlength=n_atoms), dtype=np.int32, out=rptr[1:])
        np.cumsum(np.bincount(s_act, minlength=n_atoms), dtype=np.int32, out=sptr[1:])
        if e_act == n_e:
            sperm = np.ascontiguousarray(np.argsort(s_i32, kind="mergesort").astype(np.int32, copy=False))
        else:
            sperm = np.zeros(n_e, dtype=np.int32)
            if e_act > 0:
                sperm[:e_act] = np.argsort(s_act, kind="mergesort").astype(np.int32, copy=False)
    return CSRPlan(
        n_atoms=int(n_atoms),
        n_edges=n_e,
        recv_ptr=torch.from_numpy(rptr),
        send_idx=torch.from_numpy(s_i32),
        recv_idx=torch.from_numpy(r_i32),
        send_ptr=torch.from_numpy(sptr),
        send_perm=torch.from_numpy(sperm),
    )


def build_csr_plan(
    recv: Tensor, send: Tensor, n_atoms: int, *, active_edges: int | None = None,
    edge_mask: Tensor | None = None,
) -> CSRPlan:
    """Build `CSRPlan` on `recv.device` (`active_edges`/`edge_mask` zero pad-atom degree while keeping `[n_e]` capacity)."""
    dev = recv.device
    n_e = int(recv.numel())
    if n_atoms <= 0 or n_e == 0:
        z_ptr = torch.zeros(max(0, n_atoms) + 1, dtype=torch.int32, device=dev)
        z_idx = torch.zeros(0, dtype=torch.int32, device=dev)
        return CSRPlan(max(0, n_atoms), 0, z_ptr, z_idx, z_idx, z_ptr.clone(), z_idx.clone())
    if dev.type == "cpu":
        return build_csr_plan_cpu(recv.detach().numpy(), send.detach().numpy(), n_atoms, active_edges=active_edges, edge_mask=edge_mask)
    e_act = n_e if active_edges is None else max(0, min(int(active_edges), n_e))
    if dev.type == "mps" and e_act == n_e and edge_mask is None:
        from lam4.kernels.metal_csr import has_metal_csr, metal_build_csr_plan

        if has_metal_csr():
            try:
                return metal_build_csr_plan(recv, send, n_atoms)
            except Exception:
                pass
    recv_i32 = recv if (recv.dtype == torch.int32 and recv.is_contiguous()) else recv.to(torch.int32).contiguous()
    send_i32 = send if (send.dtype == torch.int32 and send.is_contiguous()) else send.to(torch.int32).contiguous()
    atoms_ar = torch.arange(n_atoms, dtype=torch.int32, device=dev)
    recv_ptr = torch.zeros(n_atoms + 1, dtype=torch.int32, device=dev)
    send_ptr = torch.zeros(n_atoms + 1, dtype=torch.int32, device=dev)
    if e_act == 0:
        send_perm = torch.zeros(n_e, dtype=torch.int32, device=dev)
    elif edge_mask is not None and active_edges is None:
        # Pure tensor masking (zero host sync, make_fx/CUDAGraph safe): pad edges map to sentinel n_atoms.
        r_key = torch.where(edge_mask, recv_i32, n_atoms)
        s_key = torch.where(edge_mask, send_i32, n_atoms)
        torch.searchsorted(r_key, atoms_ar, out_int32=True, right=True, out=recv_ptr[1:])
        send_sorted, perm_i64 = torch.sort(s_key, stable=True)
        send_perm = perm_i64.to(torch.int32)
        torch.searchsorted(send_sorted, atoms_ar, out_int32=True, right=True, out=send_ptr[1:])
    elif e_act == n_e:
        torch.searchsorted(recv_i32, atoms_ar, out_int32=True, right=True, out=recv_ptr[1:])
        send_sorted, perm_i64 = torch.sort(send_i32, stable=True)
        send_perm = perm_i64.to(torch.int32)
        torch.searchsorted(send_sorted, atoms_ar, out_int32=True, right=True, out=send_ptr[1:])
    else:
        torch.searchsorted(recv_i32[:e_act], atoms_ar, out_int32=True, right=True, out=recv_ptr[1:])
        send_sorted, perm_i64 = torch.sort(send_i32[:e_act], stable=True)
        send_perm = torch.zeros(n_e, dtype=torch.int32, device=dev)
        send_perm[:e_act] = perm_i64.to(torch.int32)
        torch.searchsorted(send_sorted, atoms_ar, out_int32=True, right=True, out=send_ptr[1:])
    return CSRPlan(int(n_atoms), n_e, recv_ptr, send_i32, recv_i32, send_ptr, send_perm)


def horner_to_win(poly: Tensor) -> Tensor:
    """Invert Horner `[G, 4, C]` `(a0, a1, a2, a3)` back to B-spline window coefficients `(c0, c1, c2, c3)`."""
    a0, a1, a2, a3 = poly[:, 0], poly[:, 1], poly[:, 2], poly[:, 3]
    c1 = a0 - a2 * (1.0 / 3.0)
    c0, c2 = c1 - a1 + a2, c1 + a1 + a2
    return torch.stack((c0, c1, c2, 6.0 * a3 + c0 - 3.0 * c1 + 3.0 * c2), dim=1).contiguous()


def stream_vjp_csr_backward(
    plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor,
    vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor,
    cell: Tensor, graph: Tensor, z: Tensor, g_num: Tensor, g_zc: Tensor, g_dsum: Tensor,
    *, need_vir: bool, need_drho: bool, need_dshift: bool,
) -> tuple[Any, ...]:
    """Differentiable streaming-VJP fallback (`LAM4_TRITON_BWD=0`, `need_dshift=True`, CPU, or double-backward)."""
    chunks = plan.chunks if os.environ.get("LAM4_CHUNK_CACHE", "1") != "0" else None
    if chunks is None:
        capturing = is_cuda_capturing(pos.device)
        e_act = (
            max(0, min(int(plan.recv_ptr[-1].item()), plan.n_edges))
            if (not capturing and storable(pos) and plan.recv_ptr.numel() > 0)
            else plan.n_edges
        )
        recv_l, send_l = plan.recv_idx[:e_act].to(torch.long), plan.send_idx[:e_act].to(torch.long)
        if e_act == 0:
            chunks = []
        elif e_act <= stream_chunk_edges(pos.device):
            chunks = [Chunk(0, plan.n_atoms, 0, e_act, recv_l, send_l, torch.arange(plan.n_atoms, device=pos.device))]
        else:
            chunks = make_chunks(recv_l, send_l, plan.n_atoms, stream_chunk_edges(pos.device))
        if not capturing and os.environ.get("LAM4_CHUNK_CACHE", "1") != "0":
            plan.chunks = chunks
    empty = pos.new_zeros(0)
    has_cell = cell.ndim == 3 and cell.numel() > 0
    ridx_all = plan.recv_idx.to(torch.long).clamp(0, max(0, plan.n_atoms - 1)) if (has_cell and cell.shape[0] > 1) else empty
    sh = ((shift_in @ cell[0]) if cell.shape[0] == 1 else torch.einsum("ek,ekl->el", shift_in, cell[graph[ridx_all]])) if has_cell else shift_in
    need_sh_int = bool((need_vir and has_cell) or need_dshift)
    sh_c, sh_d = (empty, sh) if need_sh_int else (sh, empty)
    has_ten = bool(layout.later and ten.ndim == 3 and ten.numel() > 0)
    ins = (
        pos, horner_to_win(poly), gates, vec if (layout.later and vec.ndim == 3) else empty, qn, kn,
        rho if rho.numel() else empty, empty, ten if has_ten else empty, sh_d,
    )
    needs = (True, False, True, bool(layout.later and vec.ndim == 3), True, True, bool(need_drho), False, has_ten, need_sh_int)
    grads = _StreamVJP.apply(chunks, layout, cfg, needs, sh_c, empty_z(pos, z if z.numel() else None), *ins, g_num, g_zc, g_dsum)
    d_sh = grads[9] if need_sh_int else None
    d_cell = None
    if need_vir and has_cell and d_sh is not None:
        d_cell = (shift_in.T @ d_sh).unsqueeze(0) if cell.shape[0] == 1 else cell.new_zeros(cell.shape[0], 9).index_add(
            0, graph[ridx_all], torch.einsum("ek,el->ekl", shift_in, d_sh).reshape(-1, 9)
        ).view_as(cell)
    return (
        grads[0], grads[2], grads[3] if (layout.later and vec.ndim == 3) else None, grads[8] if has_ten else None,
        grads[4], grads[5], grads[6] if need_drho else None, d_cell, d_sh if need_dshift else None,
    )
