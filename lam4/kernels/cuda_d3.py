"""Python wrapper for strict-IEEE FP32 rank-1 factorized CUDA Grimme D3(BJ)/D3(0) kernels."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lam.models.dispersion import ANGSTROM_TO_BOHR, WB97M_D3BJ
from lam4.core.consts import capturing, storable
from lam4.core.radial import norm_device
from lam4.kernels.cuda_d3_src import CUDA_D3_SRC
from lam4.kernels.cuda_ext import cuda_ext_ready, load_cuda_ext
from lam4.kernels.metal_d3 import d3_rank1_tables_np
from lam4.kernels.triton_coulomb import build_group_ptr

_CPP_DECLS = r"""
void cuda_d3_launch(
    torch::Tensor pos, torch::Tensor z, torch::Tensor mask, torch::Tensor graph,
    torch::Tensor ptr, torch::Tensor send_idx, torch::Tensor shift, torch::Tensor cell,
    torch::Tensor rcov, torch::Tensor r2r4, torch::Tensor n_ref, torch::Tensor cn_ref, torch::Tensor c6_mat,
    torch::Tensor pw, torch::Tensor dpw, torch::Tensor out_e, torch::Tensor out_dedcn,
    torch::Tensor out_f, torch::Tensor out_vir,
    int64_t n_atoms, int64_t flags, bool is_csr,
    double s6, double s8, double p_a1, double p_a2, double r_cut, double taper_on, double inv_tspan, double cn_cut_b);
"""

_D3_R1_TABS: dict[torch.device, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = {}
_CN_CUT_BOHR = float(11.2 * ANGSTROM_TO_BOHR)
_R_CUT, _TAPER_ON, _INV_TSPAN = 12.0, 9.0, float(1.0 / (12.0 - 9.0))
_MODE_PARAMS: dict[str, tuple[bool, float, float, float, float]] = {
    "bj_wb97m": (False, float(WB97M_D3BJ["s6"]), float(WB97M_D3BJ["s8"]), float(WB97M_D3BJ["a1"]), float(WB97M_D3BJ["a2"])),
    "zero_pbe": (True, 1.0, 0.722, 1.217, 14.0),
}


def _ext() -> Any | None:
    return load_cuda_ext("lam4_cuda_d3", _CPP_DECLS, CUDA_D3_SRC, ["cuda_d3_launch"])


def _require_ext() -> Any:
    ext = _ext()
    if ext is None:
        raise RuntimeError("lam4_cuda_d3 CUDA extension unavailable")
    return ext


def has_cuda_d3(dev: torch.device | str | None = None) -> bool:
    return cuda_ext_ready("LAM4_CUDA_D3", "lam4_cuda_d3", dev, _ext)


def _get_r1_tables(dev: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    ndev = norm_device(dev)
    tabs = _D3_R1_TABS.get(ndev)
    if tabs is None:
        if capturing():
            raise RuntimeError("CUDA D3 rank-1 tables not warmed before stream capture; call warm_cuda_d3_tables(dev)")
        rcov, r2r4, n_ref, cn_ref, c6_mat = d3_rank1_tables_np()
        tabs = (
            torch.from_numpy(rcov).to(device=ndev, dtype=torch.float32).contiguous(),
            torch.from_numpy(r2r4).to(device=ndev, dtype=torch.float32).contiguous(),
            torch.from_numpy(n_ref).to(device=ndev, dtype=torch.int32).contiguous(),
            torch.from_numpy(cn_ref).to(device=ndev, dtype=torch.float32).contiguous(),
            torch.from_numpy(c6_mat).to(device=ndev, dtype=torch.float32).contiguous(),
        )
        if storable(tabs[0]):
            _D3_R1_TABS[ndev] = tabs
    return tabs


def warm_cuda_d3_tables(dev: torch.device | str | None = None) -> None:
    if not (torch.cuda.is_available() and not capturing() and has_cuda_d3(dev)):
        return
    ndev = norm_device("cuda" if dev is None else dev)
    if ndev.type == "cuda":
        _get_r1_tables(ndev)


def _reduce_e(e_atom: Tensor, graph: Tensor | None, n_graphs: int) -> Tensor:
    ed = e_atom.to(torch.float64)
    if n_graphs <= 1 or graph is None:
        return ed.sum(0, keepdim=True)
    return torch.zeros(n_graphs, dtype=torch.float64, device=e_atom.device).index_add_(0, graph, ed)


@torch.no_grad()
def cuda_d3_grouped(
    z: Tensor, pos: Tensor, graph: Tensor, n_graphs: int, *,
    d3_mode: str = "bj_wb97m", atom_mask: Tensor | None = None,
    group_ptr: Tensor | None = None, compute_forces: bool = True,
) -> tuple[Tensor, Tensor | None]:
    if d3_mode not in _MODE_PARAMS:
        raise ValueError(f"Unsupported d3_mode {d3_mode!r}")
    dev, n_atoms, g_cnt = pos.device, int(pos.shape[0]), int(n_graphs)
    if n_atoms == 0 or g_cnt == 0:
        return torch.zeros(g_cnt, dtype=torch.float64, device=dev), (torch.zeros((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else None)
    is_zero, s6, s8, a1, a2 = _MODE_PARAMS[d3_mode]
    rcov, r2r4, n_ref, cn_ref, c6_mat = _get_r1_tables(dev)
    pos_c = pos.to(torch.float32).contiguous()
    z_c = z.to(torch.int64).contiguous()
    g_c = graph.to(torch.int64).contiguous()
    ptr = build_group_ptr(g_c, g_cnt) if group_ptr is None else group_ptr.to(torch.int32).contiguous()
    has_mask = atom_mask is not None
    mask_u8 = atom_mask.to(torch.uint8).contiguous() if has_mask else torch.ones(1, dtype=torch.uint8, device=dev)
    flags = int(has_mask) | (int(is_zero) << 4) | (int(compute_forces) << 5)
    pw = torch.empty((n_atoms, 5), dtype=torch.float32, device=dev)
    dpw = torch.empty((n_atoms, 5), dtype=torch.float32, device=dev)
    e_atom = torch.empty(n_atoms, dtype=torch.float32, device=dev)
    dedcn = torch.empty(n_atoms, dtype=torch.float32, device=dev) if compute_forces else e_atom[:1]
    forces = torch.empty((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else pos_c[:1]
    _require_ext().cuda_d3_launch(
        pos_c, z_c, mask_u8, g_c, ptr, ptr, pos_c[:1], pos_c[:1],
        rcov, r2r4, n_ref, cn_ref, c6_mat, pw, dpw, e_atom, dedcn, forces, pos_c[:1],
        n_atoms, flags, False, s6, s8, a1, a2, _R_CUT, _TAPER_ON, _INV_TSPAN, _CN_CUT_BOHR,
    )
    return _reduce_e(e_atom, g_c, g_cnt), (forces if compute_forces else None)


@torch.no_grad()
def cuda_d3_csr(
    z: Tensor, pos: Tensor, recv_ptr: Tensor, send_idx: Tensor, *,
    edge_shift: Tensor | None = None, cell: Tensor | None = None, d3_mode: str = "bj_wb97m",
    atom_mask: Tensor | None = None, graph: Tensor | None = None, n_graphs: int = 1,
    compute_forces: bool = True, compute_stress: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    if d3_mode not in _MODE_PARAMS:
        raise ValueError(f"Unsupported d3_mode {d3_mode!r}")
    dev, n_atoms, g_cnt = pos.device, int(pos.shape[0]), int(n_graphs)
    do_stress = bool(compute_stress and cell is not None)
    if n_atoms == 0 or g_cnt == 0:
        return (
            torch.zeros(g_cnt, dtype=torch.float64, device=dev),
            torch.zeros((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else None,
            torch.zeros((g_cnt, 3, 3), dtype=torch.float64, device=dev) if do_stress else None,
        )
    is_zero, s6, s8, a1, a2 = _MODE_PARAMS[d3_mode]
    rcov, r2r4, n_ref, cn_ref, c6_mat = _get_r1_tables(dev)
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
    g_c = graph.to(torch.int64).contiguous() if (graph is not None and (multi_cell or g_cnt > 1)) else z_c[:1]
    sh_c = edge_shift.to(torch.float32).contiguous() if has_shift else pos_c[:1]
    cell_c = cell_g.to(torch.float32).contiguous().view(-1, 9) if cell_g is not None else pos_c[:1]
    flags = (
        int(has_mask) | (int(has_shift) << 1) | (int(has_cell) << 2) | (int(multi_cell) << 3)
        | (int(is_zero) << 4) | (int(compute_forces) << 5) | (int(do_stress) << 6)
    )
    pw = torch.empty((n_atoms, 5), dtype=torch.float32, device=dev)
    dpw = torch.empty((n_atoms, 5), dtype=torch.float32, device=dev)
    e_atom = torch.empty(n_atoms, dtype=torch.float32, device=dev)
    dedcn = torch.empty(n_atoms, dtype=torch.float32, device=dev) if need_deriv else e_atom[:1]
    forces = torch.empty((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else pos_c[:1]
    vir = torch.empty((n_atoms, 6), dtype=torch.float32, device=dev) if do_stress else pos_c[:1]
    _require_ext().cuda_d3_launch(
        pos_c, z_c, mask_u8, g_c, rp_c, si_c, sh_c, cell_c,
        rcov, r2r4, n_ref, cn_ref, c6_mat, pw, dpw, e_atom, dedcn, forces, vir,
        n_atoms, flags, True, s6, s8, a1, a2, _R_CUT, _TAPER_ON, _INV_TSPAN, _CN_CUT_BOHR,
    )
    e_graph = _reduce_e(e_atom, g_c if (g_cnt > 1 and graph is not None) else None, g_cnt)
    stress = None
    if do_stress and cell_g is not None:
        v64 = vir.to(torch.float64)
        vg = v64.sum(0, keepdim=True) if (g_cnt <= 1 or graph is None) else torch.zeros((g_cnt, 6), dtype=torch.float64, device=dev).index_add_(0, g_c, v64)
        cg = cell_g.to(torch.float64).expand(g_cnt, 3, 3) if (cell_g.shape[0] == 1 and g_cnt > 1) else cell_g.to(torch.float64)
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
