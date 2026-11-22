"""CPU Dual-CSR edge kernel (C++ / OpenMP, float32 and float64): lazy build, persistent cache, silent fallback.

The extension is compiled on first use with `torch.utils.cpp_extension.load_inline` (-O3, OpenMP from torch's own libomp, no
-ffast-math: clang's default -ffp-contract=on may fuse a*b+c into one FMA, a rounding-level difference only). The build lands in
`TORCH_EXTENSIONS_DIR` when set, else in the Mena data folder (`mena_app.paths.data_dir()/torch_extensions`) when running inside the
app, else torch's default cache; the module name carries a hash of the source, so an edited kernel never loads a stale binary.
If the build or load fails (no compiler, `LAM4_CPU_CSR=0`), `has_cpu_csr()` is False and every caller keeps the PyTorch ref path.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
import threading
from typing import Any
import warnings

import torch
from torch import Tensor

from lam4.core.edge_terms import EdgeCfg
from lam4.core.layout import StageLayout
from lam4.kernels.cpu_csr_src import CPU_CSR_COMMON, CPU_CSR_FWD
from lam4.kernels.cpu_csr_src_bwd import CPU_CSR_BWD
from lam4.kernels.cpu_csr_src_glue import CPU_CSR_GLUE
from lam4.kernels.csr_plan import MAX_DOFF_R2, CSRPlan
from lam4.kernels.metal_csr import FLAG_FLOOR, _pack_flags

CPU_CSR_SOURCE = CPU_CSR_COMMON + CPU_CSR_FWD + CPU_CSR_BWD + CPU_CSR_GLUE
CFLAGS = ("-O3", "-Xpreprocessor", "-fopenmp", "-fno-fast-math", "-ffp-contract=on")
_EXT: Any = None
_FAILED: str | None = None
_LOCK = threading.Lock()
_RMIN: dict[torch.dtype, Tensor] = {}
_TOOLCHAIN_OK: tuple[bool, str] | None = None


def cpu_toolchain_ok() -> tuple[bool, str]:
    """Check if ninja + GCC/Clang OpenMP toolchain is usable (skips GCC-only flags on Windows)."""
    global _TOOLCHAIN_OK
    if _TOOLCHAIN_OK is not None:
        return _TOOLCHAIN_OK
    if sys.platform == "win32":
        res = (False, "skipped on Windows (GCC/OpenMP flags unsupported)")
    else:
        try:
            from torch.utils.cpp_extension import is_ninja_available

            if not is_ninja_available():
                res = (False, "ninja not available")
            else:
                env_cxx = os.environ.get("CXX", "").strip()
                has_cxx = bool((env_cxx and shutil.which(env_cxx)) or any(shutil.which(c) for c in ("c++", "g++", "clang++")))
                res = (True, "") if has_cxx else (False, "C++ compiler not found")
        except Exception as exc:  # noqa: BLE001
            res = (False, f"toolchain check failed: {exc}")
    _TOOLCHAIN_OK = res
    return res


def _src_hash() -> str:
    return hashlib.sha1((CPU_CSR_SOURCE + " ".join(CFLAGS) + torch.__version__).encode()).hexdigest()[:10]


def build_dir() -> str | None:
    """Persistent build folder (None = torch's default, which honours TORCH_EXTENSIONS_DIR)."""
    if os.environ.get("TORCH_EXTENSIONS_DIR"):
        return None
    if "mena_app" in sys.modules:
        try:
            from mena_app.paths import data_dir

            d = Path(data_dir()) / "torch_extensions" / f"lam4_cpu_csr_{_src_hash()}"
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
        name=f"lam4_cpu_csr_{_src_hash()}", cpp_sources=[CPU_CSR_SOURCE], functions=None,
        extra_cflags=cflags, extra_ldflags=ldflags, build_directory=build_dir(), verbose=False,
    )


def get_cpu_csr_ext() -> Any:
    """The compiled extension, or None (one-line warning on the first failure, never raises)."""
    global _EXT, _FAILED
    if _EXT is not None or _FAILED is not None:
        return _EXT
    if os.environ.get("LAM4_CPU_CSR", "1") == "0":
        _FAILED = "disabled (LAM4_CPU_CSR=0)"
        return None
    with _LOCK:
        if _EXT is None and _FAILED is None:
            ok, reason = cpu_toolchain_ok()
            if not ok:
                _FAILED = reason
                return None
            try:
                _EXT = _build()
                rmin_table(torch.float32)
                rmin_table(torch.float64)
            except Exception as exc:  # noqa: BLE001
                _FAILED = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200] if str(exc).strip() else ''}"
                warnings.warn(f"lam4 CPU CSR kernel unavailable ({_FAILED}); using the PyTorch edge path.", RuntimeWarning, stacklevel=2)
    return _EXT


def has_cpu_csr() -> bool:
    """True iff the CPU kernel is (or can be) built. Builds lazily: the first call may take ~30-60 s, later processes hit the cache."""
    if _EXT is not None:
        return True
    if _FAILED is not None or torch.compiler.is_compiling():
        return False
    return get_cpu_csr_ext() is not None


def cpu_csr_status() -> str:
    return "on" if _EXT is not None else (f"off: {_FAILED}" if _FAILED else "not built")


def rmin_table(dtype: torch.dtype) -> Tensor:
    """[16384] NLH r_min table in `dtype` (the oracle's `repulsion._tables(cpu, dtype)[1]`)."""
    t = _RMIN.get(dtype)
    if t is None:
        from lam4.physics.nlh_tables import nlh_tables_f64

        t = torch.from_numpy(nlh_tables_f64()[1]).to(dtype).contiguous()
        _RMIN[dtype] = t
    return t


def _f(t: Tensor, dt: torch.dtype) -> Tensor:
    return t if (t.dtype == dt and t.is_contiguous()) else t.to(dt).contiguous()


def _i32(t: Tensor) -> Tensor:
    return t if (t.dtype == torch.int32 and t.is_contiguous()) else t.to(torch.int32).contiguous()


def _common(plan: CSRPlan, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor, ten: Tensor, qn: Tensor, kn: Tensor,
            rho: Tensor, shift_in: Tensor, cell: Tensor, graph: Tensor, z: Tensor, flags: int) -> list[Tensor]:
    dt = pos.dtype
    e = pos.new_zeros(0)
    el = torch.zeros(0, dtype=torch.int64)
    rmin = rmin_table(dt) if (flags & FLAG_FLOOR) else e
    return [
        pos.contiguous(), _i32(plan.recv_ptr), _i32(plan.send_idx), _f(shift_in, dt) if shift_in.numel() else e,
        _f(cell, dt) if cell.numel() else e, graph.to(torch.int64).contiguous() if graph.numel() else el,
        z.to(torch.int64).contiguous() if z.numel() else el, rmin, _f(rho, dt) if rho.numel() else e, _f(poly, dt),
        _f(gates, dt), _f(qn.reshape(qn.shape[0], -1), dt), _f(kn.reshape(kn.shape[0], -1), dt),
        _f(vec, dt) if vec.numel() else e, _f(ten, dt) if ten.numel() else e,
    ]


def cpu_csr_forward(plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor,
                    ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor, cell: Tensor, graph: Tensor, z: Tensor
                    ) -> tuple[Tensor, Tensor, Tensor]:
    n = plan.n_atoms
    if n == 0 or plan.n_edges == 0:
        return pos.new_zeros((n, layout.out_dim)), pos.new_zeros((n, 4)), pos.new_zeros((n,))
    flags = _pack_flags(cfg, z, rho, ten, cell, later=layout.later)
    ext = get_cpu_csr_ext()
    a = _common(plan, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z, flags)
    num, zc, dsum = ext.csr_fwd(*a, int(flags), bool(layout.later), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2))
    return num, zc, dsum


def cpu_csr_backward(plan: CSRPlan, layout: StageLayout, cfg: EdgeCfg, pos: Tensor, poly: Tensor, gates: Tensor, vec: Tensor,
                     ten: Tensor, qn: Tensor, kn: Tensor, rho: Tensor, shift_in: Tensor, cell: Tensor, graph: Tensor, z: Tensor,
                     g_num: Tensor, g_zc: Tensor, g_dsum: Tensor, *, need_vir: bool, need_drho: bool, need_dshift: bool
                     ) -> tuple[Any, ...]:
    """Same contract as `metal_csr_backward`: (d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell, d_shift)."""
    n, n_e = plan.n_atoms, plan.n_edges
    has_ten = layout.later and ten.numel() > 0
    if n == 0 or n_e == 0:
        return (torch.zeros_like(pos), torch.zeros_like(gates), torch.zeros_like(vec) if layout.later else None,
                torch.zeros_like(ten) if has_ten else None, torch.zeros_like(qn), torch.zeros_like(kn),
                torch.zeros_like(rho) if need_drho else None, torch.zeros_like(cell) if need_vir else None,
                torch.zeros_like(shift_in) if need_dshift else None)
    flags = _pack_flags(cfg, z, rho, ten, cell, later=layout.later, need_vir=need_vir, need_drho=need_drho)
    ext = get_cpu_csr_ext()
    dt = pos.dtype
    a = _common(plan, pos, poly, gates, vec, ten, qn, kn, rho, shift_in, cell, graph, z, flags)
    a[3:3] = [_i32(plan.send_ptr), _i32(plan.send_perm), _i32(plan.recv_idx)]
    d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, vir, eg8 = ext.csr_bwd(
        *a, _f(g_num, dt), _f(g_zc, dt), _f(g_dsum, dt), int(flags), bool(layout.later), float(cfg.r_cut), float(cfg.r_on), float(MAX_DOFF_R2))
    d_cell = None
    if need_vir:
        d_cell = vir.sum(dim=0).view(1, 3, 3) if cell.shape[0] == 1 else cell.new_zeros(cell.shape[0], 9).index_add(0, graph, vir).view_as(cell)
    d_shift = eg8[:, :3].contiguous() if need_dshift else None
    return (d_pos, d_gates, d_vec if layout.later else None, d_ten if has_ten else None, d_qn.view_as(qn), d_kn.view_as(kn),
            d_rho if need_drho else None, d_cell, d_shift)
