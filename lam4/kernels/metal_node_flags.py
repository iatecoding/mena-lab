"""Switches for the MPS node-side fast paths (fused per-atom work in the compiled whole-model runner).

`LAM4_MPS_NODE_FUSED=0` turns every node-side MPS fast path off (the plain PyTorch code runs instead). The fast paths are
used only when the code is being traced by `torch.compile` for an MPS tensor: eager MPS, CPU, CUDA and FP64 never see them.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any
import warnings

import torch
from torch import Tensor

_EAGER = [False]
_DISABLED = [False]


@contextmanager
def node_fused_eager():
    """Force fused MPS node fast paths enabled inside this block (kept for backwards compatibility with tests)."""
    prev_e, prev_d = _EAGER[0], _DISABLED[0]
    _EAGER[0], _DISABLED[0] = True, False
    try:
        yield
    finally:
        _EAGER[0], _DISABLED[0] = prev_e, prev_d


@contextmanager
def node_fused_disabled():
    """Temporarily disable fused MPS node/SwiGLU kernels (e.g. plain-PyTorch parity reference passes)."""
    prev = _DISABLED[0]
    _DISABLED[0] = True
    try:
        yield
    finally:
        _DISABLED[0] = prev


def mps_shaders_available() -> bool:
    """True when MPS backend and `torch.mps.compile_shader` (PyTorch >= 2.6) are both available."""
    return bool(torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"))


class MetalShader:
    """Lazy, cached `torch.mps.compile_shader` holder with safe fallback state on compile errors."""

    def __init__(self, name: str, source_fn: Any) -> None:
        self.name = name
        self._source_fn = source_fn
        self._lib: Any = None
        self.failed: bool = False

    def lib(self) -> Any:
        if self._lib is not None:
            return self._lib
        if self.failed:
            raise RuntimeError(f"Metal shader {self.name} previously failed to compile")
        if not mps_shaders_available():
            self.failed = True
            raise RuntimeError("torch.mps.compile_shader unavailable")
        try:
            src = self._source_fn() if callable(self._source_fn) else self._source_fn
            self._lib = torch.mps.compile_shader(src)
        except Exception as exc:  # noqa: BLE001
            self.failed = True
            warnings.warn(
                f"lam4 Metal shader {self.name} unavailable ({exc}); falling back to PyTorch.",
                RuntimeWarning,
                stacklevel=2,
            )
            raise
        return self._lib

    def available(self) -> bool:
        if self.failed or not mps_shaders_available():
            return False
        if self._lib is not None:
            return True
        if torch.compiler.is_compiling():
            return False
        try:
            self.lib()
            return True
        except Exception:  # noqa: BLE001
            return False


def node_fused_enabled() -> bool:
    return (not _DISABLED[0]) and os.environ.get("LAM4_MPS_NODE_FUSED", "1").strip().lower() not in ("0", "false", "off", "no")


def compiled_mps(x: Tensor) -> bool:
    """True for MPS FP32 inference tensors (compiled OR eager, outside autocast) when Metal node fast paths are enabled."""
    return (
        x.device.type == "mps"
        and x.dtype == torch.float32
        and mps_shaders_available()
        and not torch.is_autocast_enabled("mps")
        and node_fused_enabled()
    )


def fused_node_device(x: Tensor) -> bool:
    """True for MPS or CUDA FP32 inference tensors (compiled OR eager, outside autocast) when fused node kernels are enabled."""
    if x.dtype != torch.float32 or not node_fused_enabled():
        return False
    if x.device.type == "mps":
        return mps_shaders_available() and not torch.is_autocast_enabled("mps")
    if x.device.type == "cuda":
        from lam4.kernels.cuda_node import has_cuda_node

        return (not torch.is_autocast_enabled("cuda")) and has_cuda_node(x.device)
    return False


def compiled_angular_elemwise(x: Tensor) -> bool:
    """True while torch.compile traces an MPS (FP32) or CPU (FP32/FP64) tensor for elementwise angular contractions."""
    if not node_fused_enabled():
        return False
    if x.device.type == "mps":
        return x.dtype == torch.float32 and (torch.compiler.is_compiling() or _EAGER[0])
    return x.device.type == "cpu" and x.dtype in (torch.float32, torch.float64) and torch.compiler.is_compiling()
