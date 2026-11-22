"""Fused Dual-CSR GPU kernels (`Metal` on Apple Silicon MPS, optional `Triton` on CUDA/ROCm)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lam4.kernels.csr_ops import CSRAgg, can_use_csr, device_supports_csr, edge_aggregate_csr
    from lam4.kernels.csr_plan import CSRPlan, build_csr_plan, build_csr_plan_cpu, coef_to_horner_poly
    from lam4.kernels.metal_coulomb import MetalExactCoulomb, has_metal_coulomb
    from lam4.kernels.metal_d3 import has_metal_d3, metal_d3_csr, metal_d3_grouped

__all__ = [
    "CSRAgg",
    "CSRPlan",
    "MetalExactCoulomb",
    "build_csr_plan",
    "build_csr_plan_cpu",
    "can_use_csr",
    "coef_to_horner_poly",
    "device_supports_csr",
    "edge_aggregate_csr",
    "eval_nlh_decoupled",
    "has_metal_coulomb",
    "has_metal_d3",
    "has_triton_coulomb_bwd2",
    "has_triton_csr_bwd2",
    "metal_d3_csr",
    "metal_d3_grouped",
]


def __getattr__(name: str) -> Any:
    if name in ("CSRPlan", "build_csr_plan", "build_csr_plan_cpu", "coef_to_horner_poly"):
        from lam4.kernels import csr_plan

        return getattr(csr_plan, name)
    if name in ("CSRAgg", "can_use_csr", "device_supports_csr", "edge_aggregate_csr"):
        from lam4.kernels import csr_ops

        return getattr(csr_ops, name)
    if name == "has_triton_csr_bwd2":
        from lam4.kernels import triton_csr_bwd2

        return getattr(triton_csr_bwd2, name)
    if name == "has_triton_coulomb_bwd2":
        from lam4.kernels import triton_coulomb_bwd2

        return getattr(triton_coulomb_bwd2, name)
    if name == "eval_nlh_decoupled":
        from lam4.kernels import triton_nlh

        return getattr(triton_nlh, name)
    if name in ("MetalExactCoulomb", "has_metal_coulomb"):
        from lam4.kernels import metal_coulomb

        return getattr(metal_coulomb, name)
    if name in ("has_metal_d3", "metal_d3_csr", "metal_d3_grouped"):
        from lam4.kernels import metal_d3

        return getattr(metal_d3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
