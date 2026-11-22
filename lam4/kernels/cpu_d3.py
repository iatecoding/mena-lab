"""CPU Grimme D3(BJ) & D3(0) dispersion kernel (C++ / OpenMP, float32 and float64): lazy build, persistent cache.

Exact rank-1 factorized C6 lookup (`w_ab = p_i[a] * p_j[b]` precomputed once per atom), zero `exp()` in the pair loop,
full FP32 & FP64 precision, open molecules (`cpu_d3_grouped` / `cpu_d3_csr`) and periodic molecular crystals (`zero_pbe`
/ `bj_wb97m` with `edge_shift`, `cell`, and exact Cauchy virial stress). Set `LAM4_CPU_D3=0` to disable.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import threading
from typing import Any
import warnings

import torch
from torch import Tensor

from lam.models.dispersion import WB97M_D3BJ
from lam4.kernels.cpu_csr import CFLAGS, cpu_toolchain_ok
from lam4.kernels.cpu_d3_src import CPU_D3_SOURCE
from lam4.kernels.metal_d3 import d3_rank1_tables_np

_EXT: Any = None
_FAILED: str | None = None
_LOCK = threading.Lock()
_TABLES: dict[torch.dtype, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = {}

_MODE_PARAMS: dict[str, tuple[bool, float, float, float, float]] = {
    "bj_wb97m": (False, float(WB97M_D3BJ["s6"]), float(WB97M_D3BJ["s8"]), float(WB97M_D3BJ["a1"]), float(WB97M_D3BJ["a2"])),
    "zero_pbe": (True, 1.0, 0.722, 1.217, 14.0),
}


def _src_hash() -> str:
    return hashlib.sha1((CPU_D3_SOURCE + " ".join(CFLAGS) + torch.__version__).encode()).hexdigest()[:10]


def _build_dir() -> str | None:
    if os.environ.get("TORCH_EXTENSIONS_DIR"):
        return None
    if "mena_app" in sys.modules:
        try:
            from mena_app.paths import data_dir

            d = Path(data_dir()) / "torch_extensions" / f"lam4_cpu_d3_{_src_hash()}"
            d.mkdir(parents=True, exist_ok=True)
            return str(d)
        except Exception:  # noqa: BLE001
            return None
    return None


def _build() -> Any:
    from torch.utils.cpp_extension import load_inline

    tlib = Path(torch.__file__).resolve().parent / "lib"
    ldflags = [f"-L{tlib}", "-lomp", f"-Wl,-rpath,{tlib}"] if sys.platform == "darwin" else ["-fopenmp"]
    cflags = list(CFLAGS) if sys.platform == "darwin" else [f for f in CFLAGS if f != "-Xpreprocessor"]
    return load_inline(
        name=f"lam4_cpu_d3_{_src_hash()}", cpp_sources=[CPU_D3_SOURCE], functions=None,
        extra_cflags=cflags, extra_ldflags=ldflags, build_directory=_build_dir(), verbose=False,
    )


def get_cpu_d3_ext() -> Any:
    """Return compiled CPU D3 extension, or None on failure."""
    global _EXT, _FAILED
    if _EXT is not None or _FAILED is not None:
        return _EXT
    if os.environ.get("LAM4_CPU_D3", "1") == "0":
        _FAILED = "disabled (LAM4_CPU_D3=0)"
        return None
    with _LOCK:
        if _EXT is None and _FAILED is None:
            ok, reason = cpu_toolchain_ok()
            if not ok:
                _FAILED = reason
                return None
            try:
                _EXT = _build()
                _get_tables(torch.float32)
                _get_tables(torch.float64)
            except Exception as exc:  # noqa: BLE001
                _FAILED = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200] if str(exc).strip() else ''}"
                warnings.warn(f"lam4 CPU D3 kernel unavailable ({_FAILED}); using PyTorch D3 fallback.", RuntimeWarning, stacklevel=2)
    return _EXT


def has_cpu_d3() -> bool:
    if _EXT is not None:
        return True
    if _FAILED is not None or torch.compiler.is_compiling():
        return False
    return get_cpu_d3_ext() is not None


def _get_tables(dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    tabs = _TABLES.get(dtype)
    if tabs is None:
        rcov, r2r4, n_ref, cn_ref, c6_mat = d3_rank1_tables_np()
        tabs = (
            torch.from_numpy(rcov).to(dtype=dtype).contiguous(),
            torch.from_numpy(r2r4).to(dtype=dtype).contiguous(),
            torch.from_numpy(n_ref).to(dtype=torch.int32).contiguous(),
            torch.from_numpy(cn_ref).to(dtype=dtype).contiguous(),
            torch.from_numpy(c6_mat).to(dtype=dtype).contiguous(),
        )
        _TABLES[dtype] = tabs
    return tabs


def _group_ptr(graph: Tensor, n_graphs: int) -> Tensor:
    ptr = torch.zeros(n_graphs + 1, dtype=torch.int32)
    if graph.numel() > 0:
        cnt = torch.bincount(graph.to(torch.int64), minlength=n_graphs).to(torch.int32)
        ptr[1:] = torch.cumsum(cnt, dim=0, dtype=torch.int32)
    return ptr


@torch.no_grad()
def cpu_d3_grouped(
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
    """OpenMP CPU D3 over grouped open molecules (`e_graph [G]` float64 kcal/mol, `forces [N, 3]` pos.dtype | None)."""
    if d3_mode not in _MODE_PARAMS:
        raise ValueError(f"Unsupported d3_mode {d3_mode!r}; expected 'bj_wb97m' or 'zero_pbe'")
    dt = pos.dtype
    n_atoms, G = int(pos.shape[0]), int(n_graphs)
    if n_atoms == 0 or G == 0:
        return torch.zeros(G, dtype=torch.float64), (torch.zeros((n_atoms, 3), dtype=dt) if compute_forces else None)
    ext = get_cpu_d3_ext()
    assert ext is not None
    rcov, r2r4, n_ref, cn_ref, c6_mat = _get_tables(dt)
    is_zero, s6, s8, a1, a2 = _MODE_PARAMS[d3_mode]
    pos_c = pos.contiguous()
    z_c = z.to(torch.int64).contiguous()
    g_c = graph.to(torch.int64).contiguous() if G > 1 else torch.empty(0, dtype=torch.int64)
    ptr = (torch.tensor([0, n_atoms], dtype=torch.int32) if (G == 1 and group_ptr is None) else (_group_ptr(g_c, G) if group_ptr is None else group_ptr.to(torch.int32).contiguous()))
    mask_u8 = atom_mask.to(torch.uint8).contiguous() if atom_mask is not None else torch.empty(0, dtype=torch.uint8)
    e_atom, f_out = ext.d3_grouped(
        pos_c, z_c, mask_u8, g_c, ptr, rcov, r2r4, n_ref, cn_ref, c6_mat,
        s6, s8, a1, a2, is_zero, bool(compute_forces),
    )
    e_graph = e_atom.sum(0, keepdim=True) if G == 1 else torch.zeros(G, dtype=torch.float64).index_add_(0, g_c, e_atom)
    return e_graph, (f_out if compute_forces else None)


@torch.no_grad()
def cpu_d3_csr(
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
    """OpenMP CPU D3 over a 12.0 A CSR neighbor list (open or periodic `edge_shift` + `cell` + Cauchy virial `stress`)."""
    if d3_mode not in _MODE_PARAMS:
        raise ValueError(f"Unsupported d3_mode {d3_mode!r}; expected 'bj_wb97m' or 'zero_pbe'")
    dt = pos.dtype
    n_atoms, G = int(pos.shape[0]), int(n_graphs)
    do_stress = bool(compute_stress and cell is not None)
    if n_atoms == 0 or G == 0:
        return (
            torch.zeros(G, dtype=torch.float64),
            (torch.zeros((n_atoms, 3), dtype=dt) if compute_forces else None),
            (torch.zeros((G, 3, 3), dtype=torch.float64) if do_stress else None),
        )
    ext = get_cpu_d3_ext()
    assert ext is not None
    rcov, r2r4, n_ref, cn_ref, c6_mat = _get_tables(dt)
    is_zero, s6, s8, a1, a2 = _MODE_PARAMS[d3_mode]
    pos_c = pos.contiguous()
    z_c = z.to(torch.int64).contiguous()
    rp_c = recv_ptr.to(torch.int32).contiguous()
    si_c = send_idx.to(torch.int32).contiguous()
    has_shift = edge_shift is not None and int(edge_shift.numel()) > 0
    cell_g = (cell.unsqueeze(0) if cell.ndim == 2 else cell) if cell is not None else None
    multi_cell = bool(has_shift and cell_g is not None and cell_g.shape[0] > 1 and graph is not None)
    mask_u8 = atom_mask.to(torch.uint8).contiguous() if atom_mask is not None else torch.empty(0, dtype=torch.uint8)
    g_c = graph.to(torch.int64).contiguous() if (graph is not None and (multi_cell or G > 1)) else torch.empty(0, dtype=torch.int64)
    sh_c = edge_shift.to(dt).contiguous() if has_shift else torch.empty(0, dtype=dt)
    cell_c = cell_g.to(dt).contiguous().view(-1, 9) if cell_g is not None else torch.empty(0, dtype=dt)

    e_atom, f_out, vir = ext.d3_csr(
        pos_c, z_c, mask_u8, g_c, rp_c, si_c, sh_c, cell_c, multi_cell,
        rcov, r2r4, n_ref, cn_ref, c6_mat, s6, s8, a1, a2, is_zero, bool(compute_forces), do_stress,
    )
    e_graph = e_atom.sum(0, keepdim=True) if (G == 1 or graph is None) else torch.zeros(G, dtype=torch.float64).index_add_(0, g_c, e_atom)
    stress = None
    if do_stress and cell_g is not None:
        vg = vir.sum(0, keepdim=True) if (G <= 1 or graph is None) else torch.zeros((G, 6), dtype=torch.float64).index_add_(0, g_c, vir)
        cg_d = cell_g.to(torch.float64).expand(G, 3, 3) if (cell_g.shape[0] == 1 and G > 1) else cell_g.to(torch.float64)
        a, bb, c = cg_d.unbind(1)
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
    return e_graph, (f_out if compute_forces else None), stress
