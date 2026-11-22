"""Shared cached NVRTC / nvcc inline extension loader for strict-IEEE FP32 CUDA kernels."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from typing import Any, Callable
import warnings

import torch

from lam4.core.consts import capturing, cuda_cap_ok, proxy_tracing

_EXT_CACHE: dict[str, Any] = {}
_EXT_FAILED: dict[str, str] = {}
_LOCK = threading.Lock()
_TOOLCHAIN_OK: bool | None = None


def _has_nvcc_and_headers() -> bool:
    from torch.utils import cpp_extension

    nvcc_name = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or getattr(cpp_extension, "CUDA_HOME", None)
    nvcc_path: Path | None = None
    if cuda_home:
        cand = Path(cuda_home) / "bin" / nvcc_name
        if cand.is_file():
            nvcc_path = cand
    if nvcc_path is None:
        which_nvcc = shutil.which("nvcc")
        if which_nvcc:
            nvcc_path = Path(which_nvcc).resolve()
    if nvcc_path is None:
        return False
    inc_root = Path(cuda_home) if cuda_home else nvcc_path.parent.parent
    return (inc_root / "include" / "cuda_runtime.h").is_file()


def _has_cxx_compiler() -> bool:
    if sys.platform == "win32":
        return shutil.which("cl") is not None
    env_cxx = os.environ.get("CXX", "").strip()
    if env_cxx and shutil.which(env_cxx) is not None:
        return True
    return any(shutil.which(c) is not None for c in ("c++", "g++", "clang++"))


def cuda_toolchain_ok() -> bool:
    """Cached check for ninja + nvcc/CUDA headers + host C++ compiler required by PyTorch CUDA JIT."""
    global _TOOLCHAIN_OK
    if _TOOLCHAIN_OK is not None:
        return _TOOLCHAIN_OK
    with _LOCK:
        if _TOOLCHAIN_OK is None:
            try:
                from torch.utils.cpp_extension import is_ninja_available

                _TOOLCHAIN_OK = bool(is_ninja_available() and _has_nvcc_and_headers() and _has_cxx_compiler())
            except Exception:  # noqa: BLE001
                _TOOLCHAIN_OK = False
    return bool(_TOOLCHAIN_OK)


def cuda_ext_enabled(
    env_var: str = "LAM4_CUDA_EXT", dev: torch.device | str | None = None, *, ext_name: str | None = None
) -> bool:
    if os.environ.get("LAM4_CUDA_EXT", "1").strip().lower() in ("0", "false", "off", "no"):
        return False
    if env_var != "LAM4_CUDA_EXT" and os.environ.get(env_var, "1").strip().lower() in ("0", "false", "off", "no"):
        return False
    if dev is not None and torch.device(dev).type != "cuda":
        return False
    if ext_name is not None and ext_name in _EXT_FAILED:
        return False
    return bool(torch.cuda.is_available() and cuda_cap_ok(dev) and cuda_toolchain_ok())


def cuda_ext_ready(
    env_var: str, ext_name: str, dev: torch.device | str | None, loader: Callable[[], Any]
) -> bool:
    """True when CUDA JIT extension `ext_name` is enabled and loaded (returns False cleanly on compile failure)."""
    if not cuda_ext_enabled(env_var, dev, ext_name=ext_name):
        return False
    if ext_name in _EXT_CACHE:
        return True
    if torch.compiler.is_compiling() or capturing() or proxy_tracing():
        return False
    return loader() is not None


def _build_root() -> Path:
    base = os.environ.get("LAM4_CUDA_BUILD_DIR")
    if base:
        p = Path(base)
    elif "mena_app" in sys.modules:
        try:
            from mena_app.paths import data_dir

            p = Path(data_dir()) / "torch_extensions" / "lam4_cuda_ext"
        except Exception:  # noqa: BLE001
            p = Path.home() / ".cache" / "lam4_cuda_ext"
    elif os.environ.get("MENA_HOME"):
        p = Path(os.environ["MENA_HOME"]) / "torch_extensions" / "lam4_cuda_ext"
    elif os.environ.get("TORCH_EXTENSIONS_DIR"):
        p = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / "lam4_cuda_ext"
    else:
        p = Path.home() / ".cache" / "lam4_cuda_ext"
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except OSError:
        uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
        fb = Path(tempfile.gettempdir()) / f"lam4_cuda_ext_{uid}"
        fb.mkdir(parents=True, exist_ok=True)
        return fb


def load_cuda_ext(name: str, cpp_sources: str, cuda_sources: str, functions: list[str]) -> Any | None:
    """Compile (once, cached by source SHA256 + CUDA arch) a strict-FP32 PyTorch CUDA extension, or None on failure."""
    if name in _EXT_CACHE:
        return _EXT_CACHE[name]
    if name in _EXT_FAILED:
        return None
    with _LOCK:
        if name in _EXT_CACHE:
            return _EXT_CACHE[name]
        if name in _EXT_FAILED:
            return None
        try:
            from torch.utils.cpp_extension import load_inline

            cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (8, 0)
            arch = f"sm_{cc[0]}{cc[1]}"
            h = hashlib.sha256(f"{torch.__version__}|{arch}|{cpp_sources}|{cuda_sources}".encode("utf-8")).hexdigest()[:12]
            mod_name = f"{name}_{arch}_{h}"
            build_dir = _build_root() / mod_name
            build_dir.mkdir(parents=True, exist_ok=True)
            # Strict IEEE FP32: explicitly NO --use_fast_math so sqrtf/expf/tanhf/div match PyTorch FP32 to ~1e-7.
            cuda_flags = [
                "-O3",
                "--expt-relaxed-constexpr",
                f"-gencode=arch=compute_{cc[0]}{cc[1]},code={arch}",
            ]
            cpp_flags = ["/O2"] if sys.platform == "win32" else ["-O3"]
            ext = load_inline(
                name=mod_name,
                cpp_sources=cpp_sources,
                cuda_sources=cuda_sources,
                functions=functions,
                build_directory=str(build_dir),
                extra_cflags=cpp_flags,
                extra_cuda_cflags=cuda_flags,
                verbose=bool(os.environ.get("LAM4_CUDA_VERBOSE", "0") == "1"),
            )
        except Exception as exc:  # noqa: BLE001
            _EXT_FAILED[name] = str(exc)
            warnings.warn(
                f"lam4 CUDA extension {name} unavailable ({type(exc).__name__}: {str(exc).splitlines()[-1][:160] if str(exc).strip() else ''}); falling back.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        _EXT_CACHE[name] = ext
        return ext
