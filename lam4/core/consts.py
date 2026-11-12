"""consts.py: one rule for the module-level constant caches (spline solve matrix, angular bases, NLH tables) under tracing.

The caches cast a float64 CPU constant to the (device, dtype) of the call and keep the result. Inside make_fx (lam4.train.traced), Dynamo or a
FakeTensor mode the cast returns a traced / fake tensor: storing it would hand a FakeTensor to every later eager call, and computing the base
constant there can be impossible (linalg.lstsq has a data-dependent output shape under fake tracing). So base constants are built eagerly, and a
cast result is stored only when it is a plain eager tensor (`storable`).
"""
from __future__ import annotations

import torch
from torch import Tensor


def tracing() -> bool:
    """True inside torch.compile / Dynamo or a make_fx proxy trace."""
    if torch.compiler.is_compiling():
        return True
    try:
        from torch.fx.experimental.proxy_tensor import get_proxy_mode
    except ImportError:   # older torch: fall back to the FakeTensor check in `storable`
        return False
    return get_proxy_mode() is not None


def storable(t: Tensor) -> bool:
    """A cast constant may be cached across calls only if it is a plain eager tensor."""
    if tracing():
        return False
    from torch._subclasses.fake_tensor import FakeTensor
    return not isinstance(t, FakeTensor)
