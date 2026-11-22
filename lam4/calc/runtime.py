"""Runtime compilation, precision contexts & Dual-CSR parity guardrails for MenaCalculator."""
from __future__ import annotations

from contextlib import contextmanager
import os
from typing import TYPE_CHECKING, Any, Sequence
import warnings

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from lam4.calc.multipole_pack import (  # noqa: F401
    MULTIPOLE_PROPERTIES, append_multipole_flat_parts, slice_multipoles_dict, unpack_multipoles_from_host, wants_multipoles,
)
from lam4.calc.physics import CRYSTAL_EV, MOL_KCAL_TO_EV
from lam4.data.batch import GraphBatch
from lam4.kernels.csr_plan import CSRPlan

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator

CHECK_CROP_ATOMS = 256
CHECK_FULL_PERIODIC_MAX = 2048
WHOLE_COMPILE_MAX_ATOMS = 8192
WHOLE_COMPILE_MAX_MOL_ATOMS = 512
# CPU (edge_mode "ref"): the whole-model graph lets Inductor's C++ backend fuse the per-edge math, readout and physics;
# lam6 2.4-2.8x over eager on molecules (FP32 and FP64). Open molecules compile the dense [N, N] Coulomb, cheap up to ~2k atoms.
CPU_WHOLE_COMPILE_MAX_MOL_ATOMS = 2048


def _has_csr_custom_op() -> bool:
    from lam4.kernels import csr_ops

    return bool(getattr(csr_ops, "has_csr_custom_op", lambda: False)())


def _install_bf16_node_wrappers(model: nn.Module) -> None:
    """Install idempotent forward wrappers on node-heavy stage MLPs gated on `model._bf16_nodes_active`."""
    if getattr(model, "_bf16_wrappers_installed", False):
        return

    def _wrap_mod(mod: nn.Module) -> None:
        if hasattr(mod, "_orig_bf16_forward"):
            return
        orig_fwd = mod.forward
        mod._orig_bf16_forward = orig_fwd  # type: ignore[attr-defined]

        def _fwd(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            if getattr(model, "_bf16_nodes_active", False) and x.dtype == torch.float32 and x.device.type in ("cuda", "cpu", "mps"):
                with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                    return orig_fwd(x, *args, **kwargs).to(dtype=x.dtype)
            return orig_fwd(x, *args, **kwargs)

        mod.forward = _fwd  # type: ignore[method-assign]

    trunk = getattr(model, "trunk", None)
    if trunk is not None:
        for st in trunk.stages:
            if hasattr(st, "unbake_inference"):
                st.unbake_inference()
            for mod in (st.swiglu, st.msg_up, st.inv_mlp[0], st.inv_mlp[2]):
                _wrap_mod(mod)
    model._bf16_wrappers_installed = True  # type: ignore[attr-defined]


@contextmanager
def precision_context(precision: str, device: torch.device, model: nn.Module | None = None):
    """Apply execution-precision controls (`auto` | `fp32_ieee` | `bf16_nodes` | `exact`) around model evaluation."""
    if precision in ("fp32_ieee", "exact"):
        prev_mm, prev_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        prev_f32 = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev_mm, prev_cudnn
            torch.set_float32_matmul_precision(prev_f32)
        return
    if precision == "bf16_nodes" and model is not None:
        _install_bf16_node_wrappers(model)
        for st in getattr(getattr(model, "trunk", None), "stages", ()):
            if getattr(st, "_baked", None) is not None and hasattr(st, "unbake_inference"):
                st.unbake_inference()
        prev_flag = getattr(model, "_bf16_nodes_active", False)
        model._bf16_nodes_active = True  # type: ignore[attr-defined]
        try:
            yield
        finally:
            model._bf16_nodes_active = prev_flag  # type: ignore[attr-defined]
        return
    yield


def _crop_check_batch(calc: "MenaCalculator", b: GraphBatch) -> tuple[GraphBatch, Any, Any] | None:
    """(batch, chunks, csr_plan) for the CSR parity check on a crop of CHECK_CROP_ATOMS atoms."""
    atoms = getattr(calc, "atoms", None)
    if atoms is None or b.n_atoms <= CHECK_CROP_ATOMS or len(atoms) != b.n_atoms:
        return None
    periodic = bool(np.any(atoms.get_pbc()))
    if periodic and b.n_atoms <= CHECK_FULL_PERIODIC_MAX:
        return None
    pos = atoms.get_positions(wrap=periodic)
    idx = np.sort(np.argsort(((pos - pos.mean(axis=0)) ** 2).sum(axis=1), kind="stable")[:CHECK_CROP_ATOMS])
    sub = atoms[idx]
    sub.calc = None
    if periodic:
        sub.positions = pos[idx]
        sub.set_pbc(False); sub.set_cell(np.zeros((3, 3)))
    sub.info = {k: v for k, v in atoms.info.items() if k == "lot"}
    sub.info["charge"] = 0
    keys = ("_nl_cache", "_static_cache", "_active_csr_plan", "_lot_cache")
    saved, q_o, s_o = {k: getattr(calc, k, None) for k in keys}, calc.charge_override, calc.spin_override
    try:
        for k in keys:
            setattr(calc, k, None)
        calc.charge_override = calc.spin_override = None
        cb, c_chunks = calc._single_batch(sub)[:2]
        return cb, c_chunks, calc._active_csr_plan
    except Exception:  # noqa: BLE001
        return None
    finally:
        for k, v in saved.items():
            setattr(calc, k, v)
        calc.charge_override, calc.spin_override = q_o, s_o


def release_device_cache(device: torch.device) -> None:
    """Return the allocator's cached blocks to the system after a one-off large evaluation."""
    try:
        if device.type == "mps":
            torch.mps.synchronize(); torch.mps.empty_cache()
        elif device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.synchronize(); torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _configure_dynamo_limits() -> None:
    cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
    if cfg is not None:
        cfg.recompile_limit = max(64, int(getattr(cfg, "recompile_limit", 8)))
        cfg.cache_size_limit = max(64, int(getattr(cfg, "cache_size_limit", 8)))
    icfg = getattr(getattr(torch, "_inductor", None), "config", None)
    if icfg is not None and hasattr(icfg, "cpp_cache_precompile_headers"):
        # A precompiled header whose source mtime was touched (macOS temp cleaning) makes every CPU kernel build fail with
        # "has been modified since the precompiled header was built"; the calculator then silently ran eager. PCH only
        # shortens C++ build time, so CPU compile stays on and never hits that failure.
        icfg.cpp_cache_precompile_headers = False
    if icfg is not None and hasattr(icfg, "max_fusion_unique_io_buffers") and torch.backends.mps.is_available():
        cur_io = getattr(icfg, "max_fusion_unique_io_buffers", None)
        icfg.max_fusion_unique_io_buffers = 24 if cur_io is None else min(int(cur_io), 24)


_configure_dynamo_limits()


def _specialized_floats(fn):
    """Run a `torch.compile`d callable with `torch._dynamo.config.specialize_float = True` (restored after the call).

    The compiled inputs are tensors and bools only; the floats Dynamo meets are module constants (RMSNorm eps, r_cut,
    embedding norm_type, coulomb sigma). Unspecialized, they become symbolic floats and Dynamo restarts the whole analysis
    to specialize them again: the traced graph is twice as large and the warm-cache first call takes ~2x longer
    (lam6 mol-64 on MPS: 18.3 s -> 8.7 s). Values and kernels are unchanged. Scoped to our calls, not a process-wide flag.
    """
    cfg = torch._dynamo.config
    if not hasattr(cfg, "specialize_float"):
        return fn

    def _run(*a, **kw):
        prev = cfg.specialize_float
        cfg.specialize_float = True
        try:
            return fn(*a, **kw)
        finally:
            cfg.specialize_float = prev

    return _run


def clear_stage_compile_attrs(calc: "MenaCalculator") -> None:
    calc._set_compiled_active(False)
    calc._compiled_whole_model = calc._md_compiled_runner = calc._compile_spec = None
    calc.model.__dict__.pop("_whole_model_runners", None)
    for st in calc.model.trunk.stages:
        for k in ("_compiled_core", "_compiled_finish", "_compiled_agg", "_compiled_post"):
            st.__dict__.pop(k, None)


def _bucket_dynamic(fn, mode: str | None, backend: str | None = None):
    _configure_dynamo_limits()
    kw_c = {"dynamic": True, "mode": mode} if backend is None else {"dynamic": True, "backend": backend}
    c_sm = _specialized_floats(torch.compile(lambda s, *a, **kw: fn(s, *a, **kw), **kw_c))
    c_lg = _specialized_floats(torch.compile(lambda s, *a, **kw: fn(s, *a, **kw), **kw_c))
    return lambda s, *a, **kw: (c_lg if s.shape[0] >= 256 else c_sm)(s, *a, **kw)


def _make_whole_model_runner(model: nn.Module, mode: str | None, backend: str | None = None):
    """Compile whole `Lam4Model.forward` once per `(mode, backend)` over flat tensor inputs (`dynamic=True` `SymInt`s)."""
    _configure_dynamo_limits()
    cache: dict[tuple[str | None, str | None], Any] = model.__dict__.setdefault("_whole_model_runners", {})
    if (mode, backend) in cache:
        return cache[(mode, backend)]
    kw_c = {"dynamic": True, "mode": mode} if backend is None else {"dynamic": True, "backend": backend}

    def _flat_forward(
        z: Tensor, pos: Tensor, graph: Tensor, cell: Tensor, edge_index: Tensor, edge_shift: Tensor,
        charge: Tensor, spin: Tensor, u_flag: Tensor, token: Tensor, ref_id: Tensor, atom_mask: Tensor, graph_mask: Tensor,
        energy: Tensor, forces: Tensor, energy_mask: Tensor, force_mask: Tensor,
        rp: Tensor | None, si: Tensor | None, ri: Tensor | None, sp: Tensor | None, pm: Tensor | None,
        frag_id: Tensor | None, frag_charge: Tensor | None, frag_spin: Tensor | None, xc_id: Tensor | None, paw_id: Tensor | None, u_atom: Tensor | None,
        domain: str, pbc: bool, coulomb_cap: int | None, compute_forces: bool, compute_stress: bool, coulomb_eff_g: int | None = None,
    ) -> dict[str, Any]:
        plan = CSRPlan(z.shape[0], si.shape[0], rp, si, ri, sp, pm) if (rp is not None and si is not None and ri is not None and sp is not None and pm is not None) else None
        batch = GraphBatch(
            z=z, pos=pos, graph=graph, cell=cell, edge_index=edge_index, edge_shift=edge_shift, charge=charge, spin=spin,
            u_flag=u_flag, token=token, ref_id=ref_id, domain=domain, atom_mask=atom_mask, graph_mask=graph_mask,
            energy=energy, forces=forces, energy_mask=energy_mask, force_mask=force_mask, coulomb_cap=coulomb_cap, pbc=pbc,
            frag_id=frag_id, frag_charge=frag_charge, frag_spin=frag_spin, xc_id=xc_id, paw_id=paw_id, u_atom=u_atom,
        )
        if coulomb_eff_g is not None:
            batch._coulomb_eff_g = coulomb_eff_g  # type: ignore[attr-defined]
        return model(
            batch, create_graph=False, compute_forces=compute_forces, compute_stress=compute_stress,
            chunks=None, add_ref=False, return_stage_energies=False, csr_plan=plan,
            return_phi=False, return_aux_heads=False,
        )

    c_fn = _specialized_floats(torch.compile(_flat_forward, **kw_c))

    def _run(b: GraphBatch, csr_plan: CSRPlan | None, *, need_forces: bool, need_stress: bool) -> dict[str, Any]:
        ptrs = (csr_plan.recv_ptr, csr_plan.send_idx, csr_plan.recv_idx, csr_plan.send_ptr, csr_plan.send_perm) if csr_plan is not None else (None,) * 5
        eff_g: int | None = None
        if b.domain == "molecule" and not b.pbc and int(b.n_graphs) > 1:
            from lam4.physics.coulomb_exact import resolve_coulomb_meta
            g_res, _ = resolve_coulomb_meta(b, getattr(getattr(model, "cfg", None), "coulomb_cap", 350))
            if g_res == 1:
                eff_g = 1
        return c_fn(
            b.z, b.pos, b.graph, b.cell, b.edge_index, b.edge_shift, b.charge, b.spin, b.u_flag, b.token, b.ref_id,
            b.atom_mask, b.graph_mask, b.energy, b.forces, b.energy_mask, b.force_mask, *ptrs,
            b.frag_id, b.frag_charge, b.frag_spin, b.xc_id, b.paw_id, b.u_atom,
            b.domain, bool(b.pbc), None, bool(need_forces), bool(need_stress), eff_g,
        )

    cache[(mode, backend)] = _run
    return _run


def setup_calc_compile(calc: "MenaCalculator", mode: str | None = None, *, backend: str | None = None) -> None:
    if calc.low_memory:
        calc.compile_status = "off"
        return
    _configure_dynamo_limits()
    if getattr(calc, "precision", "auto") == "bf16_nodes":
        _install_bf16_node_wrappers(calc.model)
    try:
        is_csr, has_op = calc.model.trunk.edge_mode == "csr", bool(_has_csr_custom_op())
        spec = (calc.model.trunk.edge_mode, calc.device.type, has_op, mode, backend)
        st0_d = calc.model.trunk.stages[0].__dict__
        if getattr(calc, "_compile_spec", None) == spec and (
            calc._compiled_whole_model is not None or any(k in st0_d for k in ("_compiled_core", "_compiled_agg", "_compiled_post"))
        ):
            calc._set_compiled_active(True)
            calc.compile_status = "on" if calc._compile_checked else ("unchecked" if calc.compile_check else "on")
            return
        calc._compile_backend, calc._compile_spec = backend, spec
        calc._compile_checked_buckets = set()
        calc._compiled_whole_model = None
        if (is_csr and has_op) or calc.device.type == "cpu":
            calc._compiled_whole_model = _make_whole_model_runner(calc.model, mode, backend=backend)
        if not (is_csr and has_op):  # CPU also keeps compiled stages for systems above the whole-model caps
            kw_c = {"dynamic": True, "mode": mode} if backend is None else {"dynamic": True, "backend": backend}
            for st in calc.model.trunk.stages:
                if is_csr:
                    st._eager_post = st.__dict__.get("_eager_post", st.post_plain)
                    st._compiled_post = _bucket_dynamic(st._eager_post, mode, backend=backend)
                elif calc.device.type == "mps":
                    st._eager_core = st.__dict__.get("_eager_core", st.core_plain)
                    st._eager_finish = st.__dict__.get("_eager_finish", st.finish_plain)
                    st._compiled_core = _specialized_floats(torch.compile(st._eager_core, **kw_c))
                    st._compiled_finish = _specialized_floats(torch.compile(st._eager_finish, **kw_c))
                else:
                    st._eager_agg = st.__dict__.get("_eager_agg", st.aggregate_plain)
                    st._eager_post = st.__dict__.get("_eager_post", st.post_plain)
                    st._compiled_agg = _specialized_floats(torch.compile(st._eager_agg, **kw_c))
                    st._compiled_post = _specialized_floats(torch.compile(st._eager_post, **kw_c))
        calc._set_compiled_active(True)
        calc.compile_status = "unchecked" if calc.compile_check else "on"
    except Exception as exc:
        warnings.warn(f"MenaCalculator compile setup failed ({exc}); falling back to eager.", RuntimeWarning)
        calc._disable_compile(reason=f"fallback: {type(exc).__name__}")


def _parity_deltas(out_a: dict[str, Any], out_b: dict[str, Any], b: GraphBatch, need_forces: bool, need_stress: bool) -> tuple[float, float, float]:
    scale = (1.0 / CRYSTAL_EV) if b.domain == "crystal" else MOL_KCAL_TO_EV
    de = abs(float((out_a["energy"] - out_b["energy"]).abs().max().item())) * scale / max(1, b.n_atoms)
    df = float((out_a["forces"] - out_b["forces"]).abs().max().item()) * scale if (need_forces and out_a.get("forces") is not None and out_b.get("forces") is not None) else 0.0
    ds = float((out_a["stress"] - out_b["stress"]).abs().max().item()) * scale if (need_stress and out_a.get("stress") is not None and out_b.get("stress") is not None) else 0.0
    return de, df, ds


def _parity_exceeded(out_a: dict[str, Any], out_b: dict[str, Any], b: GraphBatch, need_forces: bool, need_stress: bool) -> tuple[bool, float, float, float]:
    de, df, ds = _parity_deltas(out_a, out_b, b, need_forces, need_stress)
    scale = (1.0 / CRYSTAL_EV) if b.domain == "crystal" else MOL_KCAL_TO_EV
    e_ref = float(out_b["energy"].abs().max().item()) * scale / max(1, b.n_atoms)
    f_ref = float(out_b["forces"].abs().max().item()) * scale if (need_forces and out_b.get("forces") is not None) else 0.0
    s_ref = float(out_b["stress"].abs().max().item()) * scale if (need_stress and out_b.get("stress") is not None) else 0.0
    bad = (not np.isfinite(de + df + ds)) or de > max(1e-4, 1e-5 * e_ref) or df > max(1e-3, 1e-4 * f_ref) or ds > max(1e-3, 1e-4 * s_ref)
    return bad, de, df, ds


def _fallback_from_csr(calc: "MenaCalculator", reason: str) -> None:
    calc._csr_checked, calc.csr_status = True, reason
    was_compiled = calc._want_compile
    clear_stage_compile_attrs(calc)
    calc.model.trunk.edge_mode = "stream" if calc.low_memory else "ref"
    calc._nl_cache = calc._active_csr_plan = None
    if was_compiled and not calc.low_memory:
        calc._compile_checked = False
        setup_calc_compile(calc, mode=calc._compile_mode)


def _max_mol_compile_atoms(dev: torch.device) -> int:
    if dev.type == "cpu":
        return CPU_WHOLE_COMPILE_MAX_MOL_ATOMS
    if dev.type == "cuda":
        from lam4.kernels.triton_coulomb import has_triton_coulomb

        if has_triton_coulomb(dev):
            return WHOLE_COMPILE_MAX_ATOMS
    if dev.type == "mps":
        from lam4.kernels.metal_csr import has_metal_csr

        if has_metal_csr():
            return WHOLE_COMPILE_MAX_ATOMS
    return WHOLE_COMPILE_MAX_MOL_ATOMS


def _max_whole_compile_atoms(calc: "MenaCalculator", dev: torch.device) -> int:
    if dev.type == "mps" and str(getattr(getattr(calc.model, "cfg", None), "ffn_mode", "")).startswith("env_mod"):
        raw = os.environ.get("LAM4_MPS_WHOLE_COMPILE_MAX_ATOMS", "").strip()
        if raw.isdigit():
            return int(raw)
    return WHOLE_COMPILE_MAX_ATOMS


def get_md_compiled_runner(calc: "MenaCalculator", use_compile: bool | str = "auto"):
    """Return a CUDAGraph/Inductor whole-model compiled runner for `MDEngine`, or `None`."""
    flag = str(use_compile).strip().lower() if isinstance(use_compile, str) else ("on" if use_compile else "off")
    if flag in ("false", "off", "0", "none", "eager") or (flag == "auto" and not calc._compiled):
        return None
    if calc.low_memory or not (calc.device.type == "cpu" or (calc.model.trunk.edge_mode == "csr" and _has_csr_custom_op())):
        return None
    if getattr(calc, "precision", "auto") == "bf16_nodes":
        _install_bf16_node_wrappers(calc.model)
    if calc._compiled_whole_model is not None and calc._compile_mode not in ("reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
        return calc._compiled_whole_model
    runner = getattr(calc, "_md_compiled_runner", None)
    if runner is None:
        try:
            runner = _make_whole_model_runner(calc.model, mode=None, backend=getattr(calc, "_compile_backend", None))
            calc._md_compiled_runner = runner
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"MD whole-model compile failed ({exc}); using eager.", RuntimeWarning)
            return None
    return runner


def _can_use_whole_model(calc: "MenaCalculator", b: GraphBatch, chunks: list | None, return_multipoles: bool = False) -> bool:
    return (
        not return_multipoles
        and getattr(calc, "_compiled_whole_model", None) is not None and chunks is None
        and b.n_atoms <= _max_whole_compile_atoms(calc, b.z.device)
        and not (b.domain == "molecule" and not b.pbc and b.n_atoms > _max_mol_compile_atoms(b.z.device))
    )


def _eval_active(calc: "MenaCalculator", b: GraphBatch, kw: dict[str, Any], csr_plan: Any, *, need_forces: bool, need_stress: bool) -> dict[str, Any]:
    if calc._compiled and _can_use_whole_model(calc, b, kw.get("chunks"), return_multipoles=bool(kw.get("return_multipoles", False))):
        return calc._compiled_whole_model(b, csr_plan, need_forces=need_forces, need_stress=need_stress)
    return calc.model(b, **kw)


def run_model_checked(calc: "MenaCalculator", b: GraphBatch, chunks: list | None, csr_plan: Any, *, need_forces: bool, need_stress: bool, return_multipoles: bool = False) -> dict[str, Any]:
    with precision_context(getattr(calc, "precision", "auto"), calc.device, calc.model):
        return _run_model_checked_inner(calc, b, chunks, csr_plan, need_forces=need_forces, need_stress=need_stress, return_multipoles=return_multipoles)


def _run_model_checked_inner(calc: "MenaCalculator", b: GraphBatch, chunks: list | None, csr_plan: Any, *, need_forces: bool, need_stress: bool, return_multipoles: bool = False) -> dict[str, Any]:
    grad_on = need_forces or need_stress
    kw: dict[str, Any] = dict(
        create_graph=False, compute_forces=need_forces, compute_stress=need_stress,
        chunks=chunks, add_ref=False, return_stage_energies=False, csr_plan=csr_plan,
        return_phi=False, return_aux_heads=False,
    )
    if return_multipoles:
        kw["return_multipoles"] = True
    out_eager_csr: dict[str, Any] | None = None
    if calc.model.trunk.edge_mode == "csr" and calc.csr_check and not calc._csr_checked:
        was_active = calc._compiled
        if was_active:
            calc._set_compiled_active(False)
        crop = _crop_check_batch(calc, b)
        cb, c_chunks, c_plan = crop if crop is not None else (b, chunks, csr_plan)
        ref_mode = "ref" if crop is not None else ("stream" if (calc.low_memory or b.n_atoms > 256) else "ref")
        kw_chk = dict(kw, chunks=c_chunks, csr_plan=c_plan)
        try:
            with torch.set_grad_enabled(grad_on):
                out_eager_csr = calc.model(cb, **kw_chk)
        except Exception as exc:
            warnings.warn(f"CSR kernel execution failed ({exc}); falling back to PyTorch {ref_mode}.", RuntimeWarning)
            _fallback_from_csr(calc, f"fallback: {type(exc).__name__}")
            return _run_model_checked_inner(calc, b, chunks, None, need_forces=need_forces, need_stress=need_stress, return_multipoles=return_multipoles)
        calc.model.trunk.edge_mode = ref_mode
        from lam4.kernels.metal_node_flags import node_fused_disabled
        with node_fused_disabled(), torch.set_grad_enabled(grad_on):
            out_ref = calc.model(cb, **dict(kw_chk, chunks=c_chunks if ref_mode == "stream" else None, csr_plan=None))
        calc.model.trunk.edge_mode = "csr"
        bad, de, df, ds = _parity_exceeded(out_eager_csr, out_ref, cb, need_forces, need_stress)
        if bad:
            release_device_cache(b.z.device)
            warnings.warn(f"CSR parity mismatch (dE/at={de:.2e} eV, dF={df:.2e} eV/A, dS={ds:.2e} eV/A^3); falling back to PyTorch.", RuntimeWarning)
            _fallback_from_csr(calc, f"fallback: parity (dE={de:.1e}, dF={df:.1e})")
            return out_ref if crop is None else _run_model_checked_inner(calc, b, chunks, None, need_forces=need_forces, need_stress=need_stress, return_multipoles=return_multipoles)
        calc._csr_checked, calc.csr_status = True, "on"
        del out_ref
        release_device_cache(b.z.device)
        if crop is not None:
            out_eager_csr = None
        if was_active:
            calc._set_compiled_active(True)
        elif crop is None:
            return out_eager_csr

    # REPORT.md:171: MPS Inductor has a large-molecule (>8192 atoms) force parity defect; stay eager on that path.
    if calc._compiled and calc.device.type == "mps" and b.domain == "molecule" and not b.pbc and b.n_atoms > WHOLE_COMPILE_MAX_ATOMS:
        calc._set_compiled_active(False)
        try:
            with torch.set_grad_enabled(grad_on):
                return calc.model(b, **kw)
        finally:
            calc._set_compiled_active(True)
    has_whole = getattr(calc, "_compiled_whole_model", None) is not None
    whole_now = has_whole and _can_use_whole_model(calc, b, chunks, return_multipoles=return_multipoles)
    stage_c = any(k in calc.model.trunk.stages[0].__dict__ for k in ("_compiled_core", "_compiled_agg", "_compiled_post"))
    if calc._compiled and has_whole and not whole_now and not stage_c:
        with torch.set_grad_enabled(grad_on):
            return calc.model(b, **kw)

    if has_whole:
        bkt = False if whole_now else "stages"   # CPU: whole-model graph and the compiled-stage fallback are checked separately
    else:
        bkt = (b.n_atoms >= 256) if calc.model.trunk.edge_mode == "csr" else False
    checked_bkts = getattr(calc, "_compile_checked_buckets", None)
    if calc._compiled and calc.compile_check and (bkt not in checked_bkts if checked_bkts is not None else not calc._compile_checked):
        try:
            with torch.set_grad_enabled(grad_on):
                out_c = _eval_active(calc, b, kw, csr_plan, need_forces=need_forces, need_stress=need_stress)
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
        bad, de, df, ds = _parity_exceeded(out_c, out_e, b, need_forces, need_stress)
        if bad:
            warnings.warn(f"Compiled parity mismatch (dE/at={de:.2e} eV, dF={df:.2e} eV/A); staying on eager.", RuntimeWarning)
            calc._set_compiled_active(False)
            calc._compile_checked, calc.compile_status = True, f"fallback: parity (dE={de:.1e}, dF={df:.1e})"
            return out_e
        calc._set_compiled_active(True)
        calc._compile_checked, calc.compile_status = True, "on"
        if checked_bkts is not None:
            checked_bkts.add(bkt)
        return out_c

    with torch.set_grad_enabled(grad_on):
        if calc._compiled:
            try:
                return _eval_active(calc, b, kw, csr_plan, need_forces=need_forces, need_stress=need_stress)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Compiled forward failed ({exc}); falling back to eager.", RuntimeWarning)
                calc._disable_compile(reason=f"fallback: {type(exc).__name__}")
                return calc.model(b, **kw)
        return calc.model(b, **kw)
