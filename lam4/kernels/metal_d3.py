"""Atomic-free Tier-2 Metal kernels for rank-1 factorized Grimme D3(BJ) & D3(0) dispersion.

Replaces the O(B * N^2 * 25 * 3) PyTorch autograd `D3BJ` path on MPS:
  - Memory: O(N) (`pw [N, 5]`, `dpw [N, 5]`, `dedcn [N]`, `e_atom [N]`, `forces [N, 3]`, `< 1 MB` at N = 32,768).
  - Rank-1 exact factorization: per-atom `p_w[i, 0..4]` and `dp_w[i, 0..4]` are computed ONCE at the end of Pass 1,
    eliminating all `25 * exp()` calls and the two-pass softmax from the inner pair loop (3x smaller L2-resident C6 table).
  - 8 lanes per atom (`simd_shuffle_down`) for high GPU occupancy from N=24 to N=131,072.
  - Supports both `bj_wb97m` and `zero_pbe`, open molecules (`metal_d3_grouped`) and 12 A CSR (`metal_d3_csr`,
    `metal_d3_csr_full` with periodic `edge_shift`, `cell`, and Cauchy virial `stress`).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from lam.models.dispersion import ANGSTROM_TO_BOHR, D3BJ, WB97M_D3BJ
from lam4.kernels import metal_csr as _metal_csr
from lam4.kernels.metal_coulomb import build_group_ptr
from lam4.kernels.metal_d3_src import METAL_D3_MSL

_D3_LIB: Any = None
_D3_FAILED: bool = False
_D3_TABLES: dict[tuple[str, int | None], tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = {}
_R1_NP_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

_MODE_PARAMS: dict[str, tuple[int, tuple[float, float, float, float]]] = {
    "bj_wb97m": (0, (float(WB97M_D3BJ["s6"]), float(WB97M_D3BJ["s8"]), float(WB97M_D3BJ["a1"]), float(WB97M_D3BJ["a2"]))),
    "zero_pbe": (1, (1.0, 0.722, 1.217, 14.0)),
}
_PARAMS1 = (12.0, 9.0, float(11.2 * ANGSTROM_TO_BOHR), float(1.0 / (12.0 - 9.0)))


def d3_rank1_tables_np() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(rcov [95], r2r4 [95], n_ref [95] int32, cn_ref [95*5], c6_mat [95*95*25])` in float64/int32."""
    global _R1_NP_CACHE
    if _R1_NP_CACHE is None:
        ref = D3BJ()
        c6 = ref.c6ab.detach().cpu().numpy().astype(np.float64)  # [95, 95, 5, 5, 3]
        n_ref = np.zeros(95, dtype=np.int32)
        cn_ref = np.zeros((95, 5), dtype=np.float64)
        for zi in range(95):
            va = np.where((c6[zi, :, :, :, 0] > 0.0).any(axis=(0, 2)))[0]
            if len(va):
                n_ref[zi] = int(len(va))
                for a in va:
                    mask = c6[zi, :, a, :, 0] > 0.0
                    cn_ref[zi, a] = float(c6[zi, :, a, :, 1][mask][0])
        c6_mat = np.ascontiguousarray(c6[..., 0].reshape(95 * 95 * 25), dtype=np.float64)
        _R1_NP_CACHE = (
            np.ascontiguousarray(ref.rcov.detach().cpu().numpy(), dtype=np.float64),
            np.ascontiguousarray(ref.r2r4.detach().cpu().numpy(), dtype=np.float64),
            n_ref,
            np.ascontiguousarray(cn_ref.reshape(95 * 5), dtype=np.float64),
            c6_mat,
        )
    return _R1_NP_CACHE


def has_metal_d3() -> bool:
    if _D3_FAILED or not _metal_csr.has_metal_csr():
        return False
    if _D3_LIB is None and not torch.compiler.is_compiling():
        try:
            get_metal_d3_lib()
        except Exception:  # noqa: BLE001
            return False
    return True


def get_metal_d3_lib() -> Any:
    global _D3_LIB, _D3_FAILED
    if _D3_LIB is None:
        try:
            _D3_LIB = torch.mps.compile_shader(METAL_D3_MSL)
        except Exception:
            _D3_FAILED = True
            raise
    return _D3_LIB


def _get_d3_tables(dev: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    key = (dev.type, dev.index)
    tabs = _D3_TABLES.get(key)
    if tabs is None:
        rcov, r2r4, n_ref, cn_ref, c6_mat = d3_rank1_tables_np()
        tabs = (
            torch.from_numpy(rcov).to(device=dev, dtype=torch.float32).contiguous(),
            torch.from_numpy(r2r4).to(device=dev, dtype=torch.float32).contiguous(),
            torch.from_numpy(n_ref).to(device=dev, dtype=torch.int32).contiguous(),
            torch.from_numpy(cn_ref).to(device=dev, dtype=torch.float32).contiguous(),
            torch.from_numpy(c6_mat).to(device=dev, dtype=torch.float32).contiguous(),
        )
        _D3_TABLES[key] = tabs
    return tabs


def _grid_sub8(n_atoms: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    tg = 128  # 16 atoms per threadgroup (4 simdgroups x 4 atoms/simdgroup x 8 lanes/atom)
    tot = ((n_atoms * 8 + tg - 1) // tg) * tg
    return (tot, 1, 1), (tg, 1, 1)


@torch.no_grad()
def metal_d3_grouped(
    z: Tensor,
    pos: Tensor,
    graph: Tensor,
    n_graphs: int,
    *,
    d3_mode: str = "bj_wb97m",
    atom_mask: Tensor | None = None,
    group_ptr: Tensor | None = None,
    compute_forces: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Tier-2 Metal D3 over grouped open molecules (`e_graph [G]` kcal/mol, `forces [N, 3]` kcal/mol/A | None)."""
    if d3_mode not in _MODE_PARAMS:
        raise ValueError(f"Unsupported d3_mode {d3_mode!r}; expected 'bj_wb97m' or 'zero_pbe'")
    dev = pos.device
    n_atoms = int(pos.shape[0])
    G = int(n_graphs)
    if n_atoms == 0 or G == 0:
        e0 = torch.zeros(G, dtype=torch.float32, device=dev)
        f0 = torch.zeros((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else None
        return e0, f0

    lib = get_metal_d3_lib()
    rcov, r2r4, n_ref, cn_ref, c6_mat = _get_d3_tables(dev)
    is_zero, params0 = _MODE_PARAMS[d3_mode]
    pos_c = pos.to(torch.float32).contiguous()
    z_c = z.to(torch.int64).contiguous()
    g_c = graph.to(torch.int64).contiguous()
    ptr = build_group_ptr(g_c, G) if group_ptr is None else group_ptr
    mask_u8 = (
        torch.ones(n_atoms, dtype=torch.uint8, device=dev)
        if atom_mask is None
        else atom_mask.to(torch.uint8).contiguous()
    )

    pw = torch.empty((n_atoms, 5), dtype=torch.float32, device=dev)
    dpw = torch.empty((n_atoms, 5), dtype=torch.float32, device=dev)
    e_atom = torch.empty(n_atoms, dtype=torch.float32, device=dev)
    dedcn = torch.empty(n_atoms, dtype=torch.float32, device=dev) if compute_forces else e_atom[:1]
    forces = torch.empty((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else pos_c[:1]

    threads, group_size = _grid_sub8(n_atoms)
    lib.d3_cn_grouped(pos_c, z_c, mask_u8, g_c, ptr, rcov, n_ref, cn_ref, pw, dpw, n_atoms, _PARAMS1, threads=threads, group_size=group_size)
    lib.d3_ef_grouped(
        pos_c, z_c, mask_u8, g_c, ptr, r2r4, n_ref, c6_mat, pw, dpw, e_atom, dedcn, forces,
        n_atoms, int(is_zero), int(compute_forces), params0, _PARAMS1,
        threads=threads, group_size=group_size,
    )
    if compute_forces:
        lib.d3_fcn_grouped(pos_c, z_c, mask_u8, g_c, ptr, rcov, dedcn, forces, n_atoms, _PARAMS1, threads=threads, group_size=group_size)

    e_graph = e_atom.sum(0, keepdim=True) if G == 1 else torch.zeros(G, dtype=torch.float32, device=dev).index_add_(0, g_c, e_atom)
    return e_graph, (forces if compute_forces else None)


@torch.no_grad()
def metal_d3_csr_full(
    z: Tensor,
    pos: Tensor,
    recv_ptr: Tensor,
    send_idx: Tensor,
    *,
    edge_shift: Tensor | None = None,
    cell: Tensor | None = None,
    d3_mode: str = "bj_wb97m",
    atom_mask: Tensor | None = None,
    graph: Tensor | None = None,
    n_graphs: int = 1,
    compute_forces: bool = True,
    compute_stress: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Tier-2 Metal D3 over a 12.0 A CSR neighbor list (open or periodic `edge_shift` + `cell` + Cauchy virial `stress`)."""
    if d3_mode not in _MODE_PARAMS:
        raise ValueError(f"Unsupported d3_mode {d3_mode!r}; expected 'bj_wb97m' or 'zero_pbe'")
    dev = pos.device
    n_atoms, G = int(pos.shape[0]), int(n_graphs)
    do_stress = bool(compute_stress and cell is not None)
    if n_atoms == 0 or G == 0:
        return (
            torch.zeros(G, dtype=torch.float32, device=dev),
            (torch.zeros((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else None),
            (torch.zeros((G, 3, 3), dtype=torch.float32, device=dev) if do_stress else None),
        )

    lib = get_metal_d3_lib()
    rcov, r2r4, n_ref, cn_ref, c6_mat = _get_d3_tables(dev)
    is_zero, params0 = _MODE_PARAMS[d3_mode]
    pos_c = pos.to(torch.float32).contiguous()
    z_c = z.to(torch.int64).contiguous()
    rp_c = recv_ptr.to(torch.int32).contiguous()
    si_c = send_idx.to(torch.int32).contiguous()

    has_mask = atom_mask is not None
    has_shift = edge_shift is not None and int(edge_shift.numel()) > 0
    cell_g = (cell.unsqueeze(0) if cell.ndim == 2 else cell) if cell is not None else None
    has_cell = bool(has_shift and cell_g is not None)
    multi_cell = bool(has_cell and cell_g is not None and cell_g.shape[0] > 1 and graph is not None)
    need_deriv = bool(compute_forces or do_stress)

    mask_u8 = atom_mask.to(torch.uint8).contiguous() if has_mask else torch.ones(1, dtype=torch.uint8, device=dev)
    g_c = graph.to(torch.int64).contiguous() if (graph is not None and (multi_cell or G > 1)) else z_c[:1]
    sh_c = edge_shift.to(torch.float32).contiguous() if has_shift else pos_c[:1]
    cell_c = cell_g.to(torch.float32).contiguous().view(-1, 9) if cell_g is not None else pos_c[:1]

    flags = (
        int(has_mask)
        | (int(has_shift) << 1)
        | (int(has_cell) << 2)
        | (int(multi_cell) << 3)
        | (int(is_zero) << 4)
        | (int(compute_forces) << 5)
        | (int(do_stress) << 6)
    )

    pw = torch.empty((n_atoms, 5), dtype=torch.float32, device=dev)
    dpw = torch.empty((n_atoms, 5), dtype=torch.float32, device=dev)
    e_atom = torch.empty(n_atoms, dtype=torch.float32, device=dev)
    dedcn = torch.empty(n_atoms, dtype=torch.float32, device=dev) if need_deriv else e_atom[:1]
    forces = torch.empty((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else pos_c[:1]
    vir = torch.empty((n_atoms, 6), dtype=torch.float32, device=dev) if do_stress else pos_c[:1]

    threads, group_size = _grid_sub8(n_atoms)
    lib.d3_cn_csr(pos_c, z_c, mask_u8, g_c, rp_c, si_c, sh_c, cell_c, rcov, n_ref, cn_ref, pw, dpw, n_atoms, flags, _PARAMS1, threads=threads, group_size=group_size)
    lib.d3_ef_csr(
        pos_c, z_c, mask_u8, g_c, rp_c, si_c, sh_c, cell_c, r2r4, n_ref, c6_mat, pw, dpw,
        e_atom, dedcn, forces, vir, n_atoms, flags, params0, _PARAMS1, threads=threads, group_size=group_size,
    )
    if need_deriv:
        lib.d3_fcn_csr(pos_c, z_c, mask_u8, g_c, rp_c, si_c, sh_c, cell_c, rcov, dedcn, forces, vir, n_atoms, flags, _PARAMS1, threads=threads, group_size=group_size)

    if G == 1 or graph is None:
        e_graph = e_atom.sum(0, keepdim=True)
    else:
        e_graph = torch.zeros(G, dtype=torch.float32, device=dev).index_add_(0, g_c, e_atom)

    stress = None
    if do_stress and cell_g is not None:
        vg = vir.sum(0, keepdim=True) if (G <= 1 or graph is None) else torch.zeros((G, 6), dtype=torch.float32, device=dev).index_add_(0, g_c, vir)
        cg = cell_g.to(torch.float32).expand(G, 3, 3) if (cell_g.shape[0] == 1 and G > 1) else cell_g.to(torch.float32)
        a, bb, c = cg.unbind(1)
        vol = (a * torch.cross(bb, c, dim=-1)).sum(-1).abs()
        ok = vol > 1.0e-8
        vs = vg * torch.where(ok, 1.0 / torch.where(ok, vol, torch.ones_like(vol)), torch.zeros_like(vol)).unsqueeze(-1)
        stress = torch.stack(
            [
                torch.stack([vs[:, 0], vs[:, 3], vs[:, 4]], -1),
                torch.stack([vs[:, 3], vs[:, 1], vs[:, 5]], -1),
                torch.stack([vs[:, 4], vs[:, 5], vs[:, 2]], -1),
            ],
            dim=1,
        )
    return e_graph, (forces if compute_forces else None), stress


@torch.no_grad()
def metal_d3_csr(
    z: Tensor,
    pos: Tensor,
    recv_ptr: Tensor,
    send_idx: Tensor,
    *,
    graph: Tensor | None = None,
    n_graphs: int = 1,
    compute_forces: bool = True,
    d3_mode: str = "bj_wb97m",
) -> tuple[Tensor, Tensor | None]:
    """Tier-2 Metal D3 over a receiver-sorted 12.0 A CSR neighbor list (`e_graph [G]`, `forces [N, 3] | None`)."""
    e_g, f_g, _ = metal_d3_csr_full(
        z, pos, recv_ptr, send_idx, d3_mode=d3_mode, graph=graph, n_graphs=n_graphs,
        compute_forces=compute_forces, compute_stress=False,
    )
    return e_g, f_g


@torch.no_grad()
def metal_build_csr_i32(
    recv_i32: Tensor, send_i32: Tensor, n_atoms: int, edge_shift: Tensor | None = None
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Build `(recv_ptr [N+1] int32, send_idx [E] int32[, shift [E, 3]])` on MPS from unsorted `(recv, send)` int32 edges."""
    dev = recv_i32.device
    e_tot = int(recv_i32.numel())
    if e_tot == 0:
        rp0 = torch.zeros(n_atoms + 1, dtype=torch.int32, device=dev)
        s0 = torch.empty(0, dtype=torch.int32, device=dev)
        return (rp0, s0) if edge_shift is None else (rp0, s0, edge_shift[:0].contiguous())
    lib = get_metal_d3_lib()
    deg = torch.zeros(n_atoms, dtype=torch.int32, device=dev)
    tg = 256
    tot = ((e_tot + tg - 1) // tg) * tg
    r_c = recv_i32.contiguous()
    s_c = send_i32.contiguous()
    lib.d3_csr_count_deg(r_c, deg, e_tot, threads=(tot, 1, 1), group_size=(tg, 1, 1))
    rp = torch.zeros(n_atoms + 1, dtype=torch.int32, device=dev)
    rp[1:] = torch.cumsum(deg, 0, dtype=torch.int32)
    cursor = rp[:-1].clone()
    out_send = torch.empty(e_tot, dtype=torch.int32, device=dev)
    if edge_shift is None:
        lib.d3_csr_scatter(r_c, s_c, cursor, out_send, e_tot, threads=(tot, 1, 1), group_size=(tg, 1, 1))
        return rp, out_send
    out_shift = torch.empty((e_tot, 3), dtype=torch.float32, device=dev)
    lib.d3_csr_scatter_shift(r_c, s_c, edge_shift.to(torch.float32).contiguous(), cursor, out_send, out_shift, e_tot, threads=(tot, 1, 1), group_size=(tg, 1, 1))
    return rp, out_send, out_shift
