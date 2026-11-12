"""Runtime compilation & Dual-CSR parity guardrails for MenaCalculator."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
import torch

from lam4.calc.physics import CRYSTAL_EV, MOL_KCAL_TO_EV
from lam4.data.batch import GraphBatch

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator


def clear_stage_compile_attrs(calc: "MenaCalculator") -> None:
    calc._set_compiled_active(False)
    for st in calc.model.trunk.stages:
        for k in ("_compiled_core", "_compiled_finish", "_compiled_agg", "_compiled_post"):
            st.__dict__.pop(k, None)


def _bucket_dynamic(fn, mode: str | None):
    def _sm(s, *a, **kw):
        return fn(s, *a, **kw)

    def _lg(s, *a, **kw):
        return fn(s, *a, **kw)

    c_sm = torch.compile(_sm, dynamic=True, mode=mode)
    c_lg = torch.compile(_lg, dynamic=True, mode=mode)
    return lambda s, *a, **kw: (c_lg if s.shape[0] >= 256 else c_sm)(s, *a, **kw)


def setup_calc_compile(calc: "MenaCalculator", mode: str | None = None) -> None:
    if calc.low_memory:
        calc.compile_status = "off"
        return
    try:
        is_csr = calc.model.trunk.edge_mode == "csr"
        calc._compile_checked_buckets = set()
        for st in calc.model.trunk.stages:
            if is_csr:
                st._eager_post = st.__dict__.get("_eager_post", st.post_plain)
                st._eager_finish = st.__dict__.get("_eager_finish", st.finish_plain)
                st._compiled_post = _bucket_dynamic(st._eager_post, mode)
                st._compiled_finish = _bucket_dynamic(st._eager_finish, mode)
            elif calc.device.type == "mps":
                st._eager_core = st.__dict__.get("_eager_core", st.core_plain)
                st._eager_finish = st.__dict__.get("_eager_finish", st.finish_plain)
                st._compiled_core = torch.compile(st._eager_core, dynamic=True, mode=mode)
                st._compiled_finish = torch.compile(st._eager_finish, dynamic=True, mode=mode)
            else:
                st._eager_agg = st.__dict__.get("_eager_agg", st.aggregate_plain)
                st._eager_post = st.__dict__.get("_eager_post", st.post_plain)
                st._compiled_agg = torch.compile(st._eager_agg, dynamic=True, mode=mode)
                st._compiled_post = torch.compile(st._eager_post, dynamic=True, mode=mode)
        calc._set_compiled_active(True)
        calc.compile_status = "unchecked" if calc.compile_check else "on"
    except Exception as exc:
        warnings.warn(f"MenaCalculator compile setup failed ({exc}); falling back to eager.", RuntimeWarning)
        calc._disable_compile(reason=f"fallback: {type(exc).__name__}")


def _parity_deltas(out_a: dict[str, Any], out_b: dict[str, Any], b: GraphBatch, need_forces: bool, need_stress: bool) -> tuple[float, float, float]:
    scale = (1.0 / CRYSTAL_EV) if b.domain == "crystal" else MOL_KCAL_TO_EV
    de = abs(float((out_a["energy"] - out_b["energy"]).abs().max().item())) * scale / max(1, b.n_atoms)
    df = (
        float((out_a["forces"] - out_b["forces"]).abs().max().item()) * scale
        if (need_forces and out_a.get("forces") is not None and out_b.get("forces") is not None)
        else 0.0
    )
    ds = (
        float((out_a["stress"] - out_b["stress"]).abs().max().item()) * scale
        if (need_stress and out_a.get("stress") is not None and out_b.get("stress") is not None)
        else 0.0
    )
    return de, df, ds


def _fallback_from_csr(calc: "MenaCalculator", reason: str) -> None:
    calc._csr_checked = True
    calc.csr_status = reason
    was_compiled = calc._want_compile
    clear_stage_compile_attrs(calc)
    calc.model.trunk.edge_mode = "stream" if calc.low_memory else "ref"
    calc._nl_cache = None
    calc._active_csr_plan = None
    if was_compiled and not calc.low_memory:
        calc._compile_checked = False
        setup_calc_compile(calc, mode=calc._compile_mode)


def run_model_checked(
    calc: "MenaCalculator", b: GraphBatch, chunks: list | None, csr_plan: Any, *, need_forces: bool, need_stress: bool
) -> dict[str, Any]:
    grad_on = need_forces or need_stress
    kw = dict(
        create_graph=False, compute_forces=need_forces, compute_stress=need_stress,
        chunks=chunks, add_ref=False, return_stage_energies=False, csr_plan=csr_plan,
    )

    out_eager_csr: dict[str, Any] | None = None
    if calc.model.trunk.edge_mode == "csr" and calc.csr_check and not calc._csr_checked:
        was_active = calc._compiled
        if was_active:
            calc._set_compiled_active(False)
        ref_mode = "stream" if (calc.low_memory or b.n_atoms > 4096) else "ref"
        try:
            with torch.set_grad_enabled(grad_on):
                out_eager_csr = calc.model(b, **kw)
        except Exception as exc:
            warnings.warn(f"CSR kernel execution failed ({exc}); falling back to PyTorch {ref_mode}.", RuntimeWarning)
            _fallback_from_csr(calc, f"fallback: {type(exc).__name__}")
            return run_model_checked(calc, b, chunks, None, need_forces=need_forces, need_stress=need_stress)

        calc.model.trunk.edge_mode = ref_mode
        kw_ref = dict(kw, chunks=chunks if ref_mode == "stream" else None, csr_plan=None)
        with torch.set_grad_enabled(grad_on):
            out_ref = calc.model(b, **kw_ref)
        calc.model.trunk.edge_mode = "csr"

        de, df, ds = _parity_deltas(out_eager_csr, out_ref, b, need_forces, need_stress)
        if not np.isfinite(de + df + ds) or de > 1e-4 or df > 1e-3 or ds > 1e-3:
            warnings.warn(
                f"CSR parity mismatch (dE/at={de:.2e} eV, dF={df:.2e} eV/A, dS={ds:.2e} eV/A^3); falling back to PyTorch.",
                RuntimeWarning,
            )
            _fallback_from_csr(calc, f"fallback: parity (dE={de:.1e}, dF={df:.1e})")
            return out_ref
        calc._csr_checked = True
        calc.csr_status = "on"
        if was_active:
            calc._set_compiled_active(True)
        else:
            return out_eager_csr

    bkt = (b.n_atoms >= 256) if calc.model.trunk.edge_mode == "csr" else False
    checked_bkts = getattr(calc, "_compile_checked_buckets", None)
    need_c_chk = calc._compiled and calc.compile_check and (bkt not in checked_bkts if checked_bkts is not None else not calc._compile_checked)
    if need_c_chk:
        try:
            with torch.set_grad_enabled(grad_on):
                out_c = calc.model(b, **kw)
        except Exception as exc:
            warnings.warn(f"Compiled forward failed ({exc}); falling back to eager.", RuntimeWarning)
            calc._disable_compile(reason=f"fallback: {type(exc).__name__}")
            if out_eager_csr is not None:
                return out_eager_csr
            with torch.set_grad_enabled(grad_on):
                return calc.model(b, **kw)
        if out_eager_csr is not None:
            out_e = out_eager_csr
        else:
            calc._set_compiled_active(False)
            with torch.set_grad_enabled(grad_on):
                out_e = calc.model(b, **kw)
        de, df, ds = _parity_deltas(out_c, out_e, b, need_forces, need_stress)
        if not np.isfinite(de + df + ds) or de > 1e-4 or df > 1e-3 or ds > 1e-3:
            warnings.warn(f"Compiled parity mismatch (dE/at={de:.2e} eV, dF={df:.2e} eV/A); staying on eager.", RuntimeWarning)
            calc._set_compiled_active(False)
            calc._compile_checked = True
            calc.compile_status = f"fallback: parity (dE={de:.1e}, dF={df:.1e})"
            return out_e
        calc._set_compiled_active(True)
        calc._compile_checked = True
        if checked_bkts is not None:
            checked_bkts.add(bkt)
        calc.compile_status = "on"
        return out_c

    with torch.set_grad_enabled(grad_on):
        return calc.model(b, **kw)
