"""lam4 core: radial basis, spline tables, harmonics, product-only edge ops, contraction calibration."""
from __future__ import annotations

from typing import Any

__all__ = ["calibrate_inv_scale"]


def __getattr__(name: str) -> Any:
    # lazy: calibrate imports Stage, which imports lam4.kernels; an eager import here made `import lam4.kernels.<x>`
    # (before lam4.calc / lam4.model) fail with a circular import
    if name == "calibrate_inv_scale":
        from lam4.core.calibrate import calibrate_inv_scale
        return calibrate_inv_scale
    raise AttributeError(f"module 'lam4.core' has no attribute {name!r}")
