"""Dual-CSR neighbor topology plan (`CSRPlan`) and precomputed Horner cubic B-spline tables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from lam4.core.consts import storable
from lam4.core.radial import N_RBF, R_CUT
from lam4.core.spline_tables import DEFAULT_GRID, BSplineTables, windows
from lam4.physics.nlh_tables import FLOOR_WIDTH, nlh_tables_f64

# Exact global ceiling on (r_min + FLOOR_WIDTH)^2 across all Z in 0..127 (max d_off = 1.2749 A -> d_off^2 = 1.6253 A^2).
MAX_DOFF_R2: float = float((nlh_tables_f64()[1].max() + FLOOR_WIDTH + 1e-3) ** 2)

_RMIN_F32_CACHE: dict[tuple[str, int | None], Tensor] = {}
_P_HORNER64: dict[tuple[int, float, int], Tensor] = {}
_P_HORNER_DEV: dict[tuple[int, float, torch.dtype, torch.device], Tensor] = {}


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
        dev = torch.device(device)
        if self.recv_ptr.device == dev:
            return self
        return CSRPlan(
            self.n_atoms,
            self.n_edges,
            self.recv_ptr.to(dev),
            self.send_idx.to(dev),
            self.recv_idx.to(dev),
            self.send_ptr.to(dev),
            self.send_perm.to(dev),
        )


def rmin_table_f32(device: torch.device) -> Tensor:
    """Cached [16384] float32 NLH r_min lookup table indexed by `(z_i << 7) + z_j`."""
    key = (device.type, device.index)
    t = _RMIN_F32_CACHE.get(key)
    if t is None:
        cpu_t = torch.from_numpy(nlh_tables_f64()[1]).to(dtype=torch.float32).contiguous()
        t = cpu_t.to(device=device)
        if storable(t):
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
    key = (int(grid), float(r_cut), dtype, device)
    c = _P_HORNER_DEV.get(key)
    if c is None:
        k64 = (int(grid), float(r_cut), N_RBF)
        h64 = _P_HORNER64.get(k64)
        if h64 is None:
            p64 = BSplineTables(1, grid, r_cut)._p64
            h64 = _win_to_horner(windows(p64)).detach().contiguous()
            if storable(h64):
                _P_HORNER64[k64] = h64
        c = h64.to(dtype=dtype).to(device=device).contiguous()
        if storable(c):
            _P_HORNER_DEV[key] = c
    return c


def coef_to_horner_poly(coef: Tensor, r_cut: float = R_CUT, *, precise: bool = False) -> Tensor:
    """Build `[G, 4, C]` contiguous Horner cubic B-spline table from factored `[32, C]`, `[G+3, C]`, or `windows` `[G, 4, C]`."""
    if coef.ndim == 2 and coef.shape[0] == N_RBF:
        if precise:
            p_h64 = _default_p_horner(r_cut, torch.float64, torch.device("cpu"))
            w64 = coef.detach().cpu().to(torch.float64)
            out64 = (p_h64.view(-1, N_RBF) @ w64).view(DEFAULT_GRID, 4, coef.shape[1])
            return out64.to(dtype=coef.dtype).to(device=coef.device).contiguous()
        p_h = _default_p_horner(r_cut, coef.dtype, coef.device)
        return (p_h.view(-1, N_RBF) @ coef).view(DEFAULT_GRID, 4, coef.shape[1]).contiguous()
    if precise:
        win64 = windows(coef.detach().cpu().to(torch.float64))
        return _win_to_horner(win64).to(dtype=coef.dtype).to(device=coef.device).contiguous()
    return _win_to_horner(windows(coef))


def build_csr_plan_cpu(recv_np: np.ndarray, send_np: np.ndarray, n_atoms: int) -> CSRPlan:
    """Fast NumPy CPU builder when neighbor indices are already on host (`verlet_neighbor_graph`)."""
    n_e = int(len(recv_np))
    if n_atoms <= 0 or n_e == 0:
        z_ptr = torch.zeros(max(0, n_atoms) + 1, dtype=torch.int32)
        z_idx = torch.zeros(0, dtype=torch.int32)
        return CSRPlan(max(0, n_atoms), 0, z_ptr, z_idx, z_idx, z_ptr.clone(), z_idx.clone())
    r_i32 = np.ascontiguousarray(recv_np, dtype=np.int32)
    s_i32 = np.ascontiguousarray(send_np, dtype=np.int32)
    rc = np.bincount(r_i32, minlength=n_atoms)
    rptr = np.empty(n_atoms + 1, dtype=np.int32)
    rptr[0] = 0
    np.cumsum(rc, dtype=np.int32, out=rptr[1:])
    sc = np.bincount(s_i32, minlength=n_atoms)
    sptr = np.empty(n_atoms + 1, dtype=np.int32)
    sptr[0] = 0
    np.cumsum(sc, dtype=np.int32, out=sptr[1:])
    sperm = np.argsort(s_i32, kind="mergesort").astype(np.int32, copy=False)
    return CSRPlan(
        n_atoms=int(n_atoms),
        n_edges=n_e,
        recv_ptr=torch.from_numpy(rptr),
        send_idx=torch.from_numpy(s_i32),
        recv_idx=torch.from_numpy(r_i32),
        send_ptr=torch.from_numpy(sptr),
        send_perm=torch.from_numpy(np.ascontiguousarray(sperm)),
    )


def build_csr_plan(recv: Tensor, send: Tensor, n_atoms: int) -> CSRPlan:
    """Build `CSRPlan` on `recv.device` ( assumes `recv` is receiver-sorted as required across `lam4`)."""
    dev = recv.device
    n_e = int(recv.numel())
    if n_atoms <= 0 or n_e == 0:
        z_ptr = torch.zeros(max(0, n_atoms) + 1, dtype=torch.int32, device=dev)
        z_idx = torch.zeros(0, dtype=torch.int32, device=dev)
        return CSRPlan(max(0, n_atoms), 0, z_ptr, z_idx, z_idx, z_ptr.clone(), z_idx.clone())
    if dev.type == "cpu":
        return build_csr_plan_cpu(recv.detach().numpy(), send.detach().numpy(), n_atoms)
    if dev.type == "mps":
        from lam4.kernels.metal_csr import has_metal_csr, metal_build_csr_plan

        if has_metal_csr():
            try:
                return metal_build_csr_plan(recv, send, n_atoms)
            except Exception:
                pass
    recv_i32 = recv.to(torch.int32).contiguous()
    send_i32 = send.to(torch.int32).contiguous()
    atoms_ar = torch.arange(n_atoms, dtype=torch.int32, device=dev)
    recv_ptr = torch.zeros(n_atoms + 1, dtype=torch.int32, device=dev)
    recv_ptr[1:] = torch.searchsorted(recv_i32, atoms_ar, right=True).to(torch.int32)
    send_perm = torch.argsort(send_i32, stable=True).to(torch.int32).contiguous()
    send_sorted = send_i32.index_select(0, send_perm)
    send_ptr = torch.zeros(n_atoms + 1, dtype=torch.int32, device=dev)
    send_ptr[1:] = torch.searchsorted(send_sorted, atoms_ar, right=True).to(torch.int32)
    return CSRPlan(int(n_atoms), n_e, recv_ptr, send_i32, recv_i32, send_ptr, send_perm)
