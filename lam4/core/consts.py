"""consts.py: one rule for the module-level constant caches (spline solve matrix, angular bases, NLH tables) under tracing.

The caches cast a float64 CPU constant to the (device, dtype) of the call and keep the result. Inside make_fx (lam4.train.traced), Dynamo or a
FakeTensor mode the cast returns a traced / fake tensor: storing it would hand a FakeTensor to every later eager call, and computing the base
constant there can be impossible (linalg.lstsq has a data-dependent output shape under fake tracing). So base constants are built eagerly, and a
cast result is stored only when it is a plain eager tensor (`storable`).
"""
from __future__ import annotations

import torch
from torch import Tensor


def capturing() -> bool:
    """True while recording a CUDA stream into a CUDAGraph."""
    if torch.compiler.is_compiling() or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except (AssertionError, RuntimeError):
        return False


def proxy_tracing() -> bool:
    """True inside torch.compile / Dynamo or a make_fx proxy trace (NOT CUDA stream capture)."""
    if torch.compiler.is_compiling():
        return True
    try:
        from torch.fx.experimental.proxy_tensor import get_proxy_mode
    except ImportError:   # older torch: fall back to the FakeTensor check in `storable`
        return False
    return get_proxy_mode() is not None


def is_fake(t: Tensor) -> bool:
    """True when `t` is a meta tensor or FakeTensor."""
    if t.is_meta:
        return True
    from torch._subclasses.fake_tensor import FakeTensor
    return isinstance(t, FakeTensor)


def tracing() -> bool:
    """True inside torch.compile / Dynamo, CUDA stream capture, or a make_fx proxy trace."""
    return proxy_tracing() or capturing()


_CUDA_CAP_OK: dict[tuple[int, int], bool] = {}


def storable(t: Tensor) -> bool:
    """A cast constant may be cached across calls only if it is a plain eager tensor."""
    return not tracing() and not is_fake(t)


def cuda_cap_ok(dev: torch.device | str | int | None = None, min_major: int = 7) -> bool:
    """True when `dev` (or current CUDA device) has compute capability >= `min_major`; memoized for stream-capture/compile."""
    if not torch.cuda.is_available():
        return False
    try:
        if isinstance(dev, int):
            arg, idx = dev, dev
        elif dev is None:
            arg, idx = None, -1
        else:
            d = torch.device(dev)
            if d.type != "cuda":
                return False
            arg = d.index
            idx = -1 if arg is None else int(arg)
        if idx < 0:
            try:
                idx = int(torch.cuda.current_device())
            except Exception:
                idx = 0
        key = (idx, min_major)
        tr = tracing()
        if tr and key in _CUDA_CAP_OK:
            return _CUDA_CAP_OK[key]
        if torch.compiler.is_compiling():
            return int(torch.cuda.get_device_properties(idx).major) >= min_major
        cap = torch.cuda.get_device_capability() if arg is None else torch.cuda.get_device_capability(arg)
        ok = int(cap[0]) >= min_major
        if not tr:
            _CUDA_CAP_OK[key] = ok
        return ok
    except Exception:
        return False
