"""Plug-and-play Python API for Mena models: calculator, singlepoint, relax, run_md, available_models.

All heavy dependencies (`torch`, `lam4`, `lam`, `ase`, `numpy`) are imported lazily inside functions so importing
`mena_app` or `mena_app.python_api` remains pure-stdlib and instant.
"""

from __future__ import annotations

import inspect
from typing import Any

_VALID_PRECISIONS = ("auto", "fp32_ieee", "exact")
_F64_ALIASES = ("float64", "fp64", "f64", "double")
_E_ANG_TO_DEBYE = 1.0 / 0.20819434


def available_models(*, installed_only: bool = False) -> list[dict[str, Any]]:
    """Return available (or all declared) model entries without importing PyTorch."""
    from mena_app import models

    return [dict(m) for m in models.all_models() if (not installed_only or m.get("available"))]


def _resolve_auto_device(dtype: str) -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    is_f64 = str(dtype).strip().lower() in _F64_ALIASES
    if not is_f64 and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_precision_and_dtype(precision: str, dtype: str) -> tuple[str, str]:
    prec = str(precision).strip().lower()
    if prec == "bf16_nodes":
        raise ValueError("precision='bf16_nodes' is disabled in mena_app (non-conservative forces); use 'auto', 'fp32_ieee', or 'exact'.")
    if prec not in _VALID_PRECISIONS:
        raise ValueError(f"Unsupported precision {precision!r}; expected one of {_VALID_PRECISIONS}")
    dt_low = str(dtype).strip().lower()
    is_f64 = dt_low in _F64_ALIASES
    if prec == "fp32_ieee" and is_f64:
        raise ValueError(f"precision={precision!r} requires single precision (float32), got dtype={dtype!r}")
    eff_dtype = "float64" if prec == "exact" else dtype
    return prec, eff_dtype


def _infer_domain_from_lot(lot: str) -> str | None:
    if str(lot).strip().lower() == "auto":
        return None
    from mena_app.routing import LOT_ALIASES, MOLECULAR_LOTS

    low = str(lot).strip().lower()
    canon = LOT_ALIASES.get(low, low).lower()
    if any(ml.lower() == canon for ml in MOLECULAR_LOTS):
        return "molecule"
    if canon in ("oc20_rpbe", "oc25_rpbe", "oc22_pbeu"):
        return "slab"
    return "crystal"


def _resolve_model_entry(model: str, domain: str | None, lot: str) -> dict[str, Any]:
    from mena_app import models
    from mena_app.routing import canonical_lot

    if str(model).strip().lower() != "auto":
        entry = next((m for m in models.all_models() if m["id"] == model), None)
        if entry is None:
            raise models.ModelError(f"no model called {model!r} (see `mena models`)")
        return entry

    eff_dom = domain or _infer_domain_from_lot(lot)
    if eff_dom is not None:
        try:
            entry = models.choose("auto", eff_dom)
        except models.ModelError:
            entry = None
        else:
            if lot == "auto" or canonical_lot(lot, entry.get("lots", [])) is not None:
                return entry
    else:
        for fallback_dom in ("molecule", "crystal", "slab"):
            try:
                return models.choose("auto", fallback_dom)
            except models.ModelError:
                continue

    if lot != "auto":
        usable = [
            m for m in models.all_models()
            if models._usable(m)
            and (eff_dom is None or eff_dom in m.get("domains", []))
            and canonical_lot(lot, m.get("lots", [])) is not None
        ]
        if usable:
            rank_dom = eff_dom or "molecule"
            return sorted(usable, key=lambda m: m.get("rank", {}).get(rank_dom, 99))[0]

    return models.choose("auto", eff_dom or "molecule")


def _calculator_with_entry(
    model: str = "auto",
    device: str = "auto",
    dtype: str = "float32",
    lot: str = "auto",
    skin: float = 0.0,
    low_memory: bool = False,
    compile: bool | str = False,
    precision: str = "auto",
    *,
    domain: str | None = None,
    edge_mode: str | None = None,
    add_dispersion: bool | None = None,
    u_flag: int | None = None,
    weights: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    from mena_app import backends, models

    prec, eff_dtype = _normalize_precision_and_dtype(precision, dtype)
    entry = _resolve_model_entry(model, domain, lot)
    if lot != "auto":
        from mena_app.routing import canonical_lot

        canon = canonical_lot(lot, entry.get("lots", []))
        if canon is None:
            raise models.ModelError(f"{entry.get('title', model)} does not support level of theory {lot!r}; "
                                    f"it supports: {', '.join(entry.get('lots', []))}")
        lot = canon
    dev = _resolve_auto_device(eff_dtype) if str(device).strip().lower() == "auto" else str(device)
    if dev.strip().lower() == "mps" and str(eff_dtype).strip().lower() in _F64_ALIASES:
        raise ValueError("Apple MPS does not support float64 / precision='exact'; use device='cpu' or device='auto'.")
    entry = models.ensure_file(entry)
    dec: dict[str, Any] = {
        "device": dev, "dtype": eff_dtype, "lot": lot, "skin": skin,
        "low_memory": low_memory, "compile": compile, "precision": prec,
    }
    if edge_mode is not None:
        dec["edge_mode"] = edge_mode
    if add_dispersion is not None:
        dec["add_dispersion"] = bool(add_dispersion)
    if u_flag is not None:
        dec["u_flag"] = int(u_flag)
    if weights is not None:
        dec["weights"] = str(weights)
    return backends.build(entry, dec), entry


def calculator(model: str = "auto", device: str = "auto", dtype: str = "float32", lot: str = "auto",
               skin: float = 0.0, low_memory: bool = False, compile: bool | str = False, precision: str = "auto",
               *, domain: str | None = None, edge_mode: str | None = None, add_dispersion: bool | None = None,
               u_flag: int | None = None, weights: str | None = None):
    """Build an ASE calculator for a Mena model (`auto`, `lam6-v1`, `lam5-v1`, `lam-400k`, or user-added id).

    With `model='auto'`, pass `domain='crystal'` or `'slab'` (or a periodic `lot`) to pick the materials default
    (`lam6-v1`); `singlepoint()`, `relax()`, and `run_md()` detect `domain` from `atoms` automatically.
    """
    calc, _ = _calculator_with_entry(
        model=model, device=device, dtype=dtype, lot=lot, skin=skin, low_memory=low_memory,
        compile=compile, precision=precision, domain=domain, edge_mode=edge_mode,
        add_dispersion=add_dispersion, u_flag=u_flag, weights=weights,
    )
    return calc


def _prepare_atoms(atoms, *, charge: int | None = None, spin: int | None = None):
    if isinstance(atoms, str) or hasattr(atoms, "__fspath__"):
        from mena_app import structures

        at = structures.load(str(atoms))
    else:
        at = atoms.copy()
    if charge is not None:
        at.info["charge"] = int(charge)
    if spin is not None:
        at.info["spin"] = int(spin)
    return at


def singlepoint(atoms, *, model: str = "auto", device: str = "auto", dtype: str = "float32",
                precision: str = "auto", lot: str = "auto", low_memory: bool = False,
                compile: bool | str = False, charge: int | None = None, spin: int | None = None,
                edge_mode: str | None = None, add_dispersion: bool | None = None,
                u_flag: int | None = None, weights: str | None = None) -> dict[str, Any]:
    """Evaluate energy, forces, stress (when 3D periodic), and molecular/electronic properties."""
    import numpy as np
    from mena_app import structures

    at = _prepare_atoms(atoms, charge=charge, spin=spin)
    dom = structures.domain_of(at) if str(model).strip().lower() == "auto" else None
    calc, entry = _calculator_with_entry(
        model=model, device=device, dtype=dtype, precision=precision, lot=lot,
        low_memory=low_memory, compile=compile, domain=dom, edge_mode=edge_mode,
        add_dispersion=add_dispersion, u_flag=u_flag, weights=weights,
    )
    at.calc = calc
    forces = np.asarray(at.get_forces(), dtype=np.float64)
    energy = float(at.get_potential_energy())
    out: dict[str, Any] = {"energy": energy, "forces": forces, "model": entry["id"]}

    if bool(at.pbc.all()) and "stress" in getattr(calc, "implemented_properties", ()):
        try:
            out["stress"] = np.asarray(at.get_stress(voigt=True), dtype=np.float64)
        except Exception:  # noqa: BLE001
            pass

    res = getattr(calc, "results", {}) or {}
    if res.get("lot_used") is not None:
        out["lot_used"] = str(res["lot_used"])
    if res.get("charges") is not None:
        out["charges"] = np.asarray(res["charges"], dtype=np.float64)
    if res.get("spin_density") is not None:
        out["spin_density"] = np.asarray(res["spin_density"], dtype=np.float64)
    if res.get("energy_decomposition") is not None:
        out["energy_decomposition"] = res["energy_decomposition"]
    if not bool(at.pbc.any()):
        if res.get("dipole") is not None:
            dip = np.asarray(res["dipole"], dtype=np.float64).reshape(-1)
            out["dipole"] = dip
            if res.get("dipole_debye") is None:
                out["dipole_debye"] = dip * _E_ANG_TO_DEBYE
        if res.get("dipole_debye") is not None:
            out["dipole_debye"] = np.asarray(res["dipole_debye"], dtype=np.float64).reshape(-1)
        if res.get("polarizability") is not None:
            out["polarizability"] = np.asarray(res["polarizability"], dtype=np.float64)
        if res.get("polarizability_iso") is not None:
            out["polarizability_iso"] = float(res["polarizability_iso"])
        if res.get("polarizability_aniso") is not None:
            out["polarizability_aniso"] = float(res["polarizability_aniso"])
    return out


def _map_method_kwargs(fn, raw_kwargs: dict[str, Any], aliases: dict[str, str], *, target_sig=None) -> dict[str, Any]:
    sig = inspect.signature(fn).parameters if target_sig is None else target_sig
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
    mapped: dict[str, Any] = {}
    for k, v in raw_kwargs.items():
        if k in sig:
            mapped[k] = v
        elif k in aliases and (aliases[k] in sig or has_var_kw):
            mapped.setdefault(aliases[k], v)
        else:
            mapped[k] = v
    return mapped


def _fallback_relax(calc, at, **relax_kwargs: Any) -> dict[str, Any]:
    import numpy as np
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE, LBFGS

    fmax = float(relax_kwargs.pop("fmax", 0.02))
    steps = int(relax_kwargs.pop("steps", relax_kwargs.pop("max_steps", 500)))
    optimizer = str(relax_kwargs.pop("optimizer", "LBFGS"))
    relax_cell = bool(relax_kwargs.pop("relax_cell", False))
    maxstep = float(relax_kwargs.pop("maxstep", 0.2))
    relax_kwargs.pop("skin", None)
    if relax_kwargs:
        raise TypeError(f"Unexpected relax keyword argument(s): {', '.join(sorted(relax_kwargs))}")
    opt_cls = {"LBFGS": LBFGS, "FIRE": FIRE}.get(optimizer)
    if opt_cls is None:
        raise ValueError(f"Unsupported optimizer {optimizer!r}; expected 'LBFGS' or 'FIRE'")

    at.calc = calc
    e0 = float(at.get_potential_energy())
    target = FrechetCellFilter(at) if (relax_cell and bool(at.pbc.all())) else at
    opt = opt_cls(target, maxstep=maxstep, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    e1 = float(at.get_potential_energy())
    f1 = np.asarray(target.get_forces(), dtype=np.float64)
    fm1 = float(np.linalg.norm(f1, axis=-1).max()) if len(f1) else 0.0
    return {
        "atoms": at,
        "positions": np.asarray(at.get_positions(), dtype=np.float64).copy(),
        "initial_energy": e0,
        "final_energy": e1,
        "energy_ev": e1,
        "fmax": fm1,
        "final_fmax": fm1,
        "steps": int(opt.nsteps),
        "converged": converged,
    }


def relax(atoms, *, model: str = "auto", device: str = "auto", dtype: str = "float32",
          precision: str = "auto", lot: str = "auto", low_memory: bool = False,
          compile: bool | str = False, charge: int | None = None, spin: int | None = None,
          edge_mode: str | None = None, add_dispersion: bool | None = None,
          u_flag: int | None = None, weights: str | None = None,
          **relax_kwargs: Any) -> dict[str, Any]:
    """Relax atomic positions (and optionally unit cell) using `calc.relax` or an ASE optimizer fallback."""
    from mena_app import structures

    at = _prepare_atoms(atoms, charge=charge, spin=spin)
    dom = structures.domain_of(at) if str(model).strip().lower() == "auto" else None
    calc_skin = float(relax_kwargs.get("skin", 0.0))
    calc, entry = _calculator_with_entry(
        model=model, device=device, dtype=dtype, precision=precision, lot=lot,
        skin=calc_skin, low_memory=low_memory, compile=compile, domain=dom,
        edge_mode=edge_mode, add_dispersion=add_dispersion, u_flag=u_flag, weights=weights,
    )
    if hasattr(calc, "relax"):
        sig = inspect.signature(calc.relax).parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
        wants_cell = bool(relax_kwargs.get("relax_cell")) and bool(at.pbc.all())
        wants_custom_opt = str(relax_kwargs.get("optimizer", "LBFGS")) != "LBFGS"
        if (not wants_cell or "relax_cell" in sig or has_var_kw) and (not wants_custom_opt or "optimizer" in sig or has_var_kw):
            raw_kw = dict(relax_kwargs)
            if "skin" not in sig:
                raw_kw.pop("skin", None)
            if not wants_cell and "relax_cell" not in sig and not has_var_kw:
                raw_kw.pop("relax_cell", None)
            if not wants_custom_opt and "optimizer" not in sig and not has_var_kw:
                raw_kw.pop("optimizer", None)
            call_kw = _map_method_kwargs(calc.relax, raw_kw, {"max_steps": "steps"})
            res = calc.relax(at, **call_kw)
            if isinstance(res, dict):
                if "atoms" in res:
                    res.setdefault("positions", res["atoms"].get_positions().copy())
                if "energy_ev" in res:
                    res.setdefault("final_energy", res["energy_ev"])
                elif "final_energy" in res:
                    res.setdefault("energy_ev", res["final_energy"])
                if "fmax" in res:
                    res.setdefault("final_fmax", res["fmax"])
                elif "final_fmax" in res:
                    res.setdefault("fmax", res["final_fmax"])
                res.setdefault("model", entry["id"])
            return res
    res = _fallback_relax(calc, at, **relax_kwargs)
    res.setdefault("model", entry["id"])
    return res


def _fallback_run_md(calc, at, **md_kwargs: Any) -> dict[str, Any]:
    import numpy as np
    from ase.md.velocitydistribution import Stationary

    from mena_app.tasks.common import thermalize
    from mena_app.tasks.md import _has_nonzero_momenta, _integrator, _temperature_K, _zero_rotation_safe

    ensemble = str(md_kwargs.pop("ensemble", "nvt")).strip().lower()
    if ensemble not in ("nvt", "nve", "npt"):
        raise ValueError(f"Unsupported ensemble {ensemble!r}; expected 'nvt', 'nve', or 'npt'")
    steps = int(md_kwargs.pop("steps", 100))
    temp_val = md_kwargs.pop("temperature_K", md_kwargs.pop("temperature", 300.0))
    temperature_K = float(300.0 if temp_val is None else temp_val)
    timestep_fs = float(md_kwargs.pop("timestep_fs", md_kwargs.pop("timestep", 1.0)))
    tdamp_fs = md_kwargs.pop("tdamp_fs", None)
    friction = float(md_kwargs.pop("friction", (1.0 / max(float(tdamp_fs), 1e-4)) if tdamp_fs is not None else 0.01))
    pressure_bar = float(md_kwargs.pop("pressure_bar", md_kwargs.pop("pressure", 1.0)))
    seed = int(md_kwargs.pop("seed", 0))
    maxwell = md_kwargs.pop("maxwell", None)
    save_every = md_kwargs.pop("save_every", None)
    record = bool(md_kwargs.pop("record", False))
    md_kwargs.pop("skin", None)
    md_kwargs.pop("window", None)
    if md_kwargs:
        raise TypeError(f"Unexpected run_md keyword argument(s): {', '.join(sorted(md_kwargs))}")

    if ensemble == "npt" and not bool(at.pbc.all()):
        raise ValueError("NPT constant-pressure MD requires a 3D periodic cell.")
    if int(at.get_number_of_degrees_of_freedom()) <= 0:
        raise ValueError("All atoms are fixed; dynamics needs at least one free degree of freedom.")

    at.calc = calc
    has_cons = bool(getattr(at, "constraints", None))
    if maxwell is True or (maxwell is None and not _has_nonzero_momenta(at)):
        thermalize(at, temperature_K=temperature_K, rng=np.random.default_rng(seed))
        if has_cons:
            at.set_momenta(at.get_momenta(), apply_constraint=True)
        else:
            if len(at) > 1:
                Stationary(at)
            if not at.pbc.any():
                _zero_rotation_safe(at)
    elif has_cons:
        at.set_momenta(at.get_momenta(), apply_constraint=True)

    p = {
        "ensemble": ensemble,
        "temperature": temperature_K,
        "timestep": timestep_fs,
        "friction": friction,
        "pressure": pressure_bar,
        "seed": seed,
    }
    dyn = _integrator(p, at)
    traj: list[dict[str, Any]] = []
    if record or save_every is not None:
        iv = max(1, int(save_every or 1))
        dyn.attach(lambda: traj.append({
            "step": int(dyn.nsteps),
            "positions": np.asarray(at.get_positions(), dtype=np.float64).copy(),
            "energy": float(at.get_potential_energy()),
            "temperature_K": _temperature_K(at),
        }), interval=iv)
    dyn.run(steps)
    ep = float(at.get_potential_energy())
    ek = float(at.get_kinetic_energy())
    out: dict[str, Any] = {
        "atoms": at,
        "backend": "ase",
        "positions": np.asarray(at.get_positions(), dtype=np.float64).copy(),
        "momenta": np.asarray(at.get_momenta(), dtype=np.float64).copy(),
        "energy": ep,
        "energy_ev": ep,
        "kinetic_energy": ek,
        "kinetic_energy_ev": ek,
        "total_energy": ep + ek,
        "total_energy_ev": ep + ek,
        "temperature_K": _temperature_K(at),
        "timestep_fs": timestep_fs,
        "steps": int(dyn.nsteps),
        "ensemble": ensemble,
    }
    if traj:
        out["trajectory"] = traj
    return out


def run_md(atoms, *, model: str = "auto", device: str = "auto", dtype: str = "float32",
           precision: str = "auto", lot: str = "auto", low_memory: bool = False,
           compile: bool | str = False, charge: int | None = None, spin: int | None = None,
           edge_mode: str | None = None, add_dispersion: bool | None = None,
           u_flag: int | None = None, weights: str | None = None,
           **md_kwargs: Any) -> dict[str, Any]:
    """Run molecular dynamics using `calc.run_md` when supported, else clean ASE fallback."""
    from mena_app import structures

    at = _prepare_atoms(atoms, charge=charge, spin=spin)
    dom = structures.domain_of(at) if str(model).strip().lower() == "auto" else None
    calc_skin = float(md_kwargs.get("skin", 0.0))
    calc, entry = _calculator_with_entry(
        model=model, device=device, dtype=dtype, precision=precision, lot=lot,
        skin=calc_skin, low_memory=low_memory, compile=compile, domain=dom,
        edge_mode=edge_mode, add_dispersion=add_dispersion, u_flag=u_flag, weights=weights,
    )
    ensemble = str(md_kwargs.get("ensemble", "nvt")).strip().lower()
    wants_record = bool(md_kwargs.get("record", False))
    if hasattr(calc, "run_md") and ensemble != "npt" and not getattr(at, "constraints", None) and not wants_record:
        from mena_app.tasks.md import _has_nonzero_momenta

        raw_md = dict(md_kwargs)
        raw_md.pop("record", None)
        if "friction" in raw_md and "tdamp_fs" not in raw_md and "friction_per_fs" not in raw_md:
            fr = float(raw_md.pop("friction"))
            raw_md["tdamp_fs"] = 1.0 / max(fr, 1e-4)
        call_kw = _map_method_kwargs(
            calc.run_md,
            raw_md,
            {"temperature": "temperature_K", "timestep": "timestep_fs", "pressure": "pressure_bar", "save_every": "log_interval"},
        )
        sig = inspect.signature(calc.run_md).parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
        if "maxwell" not in call_kw and _has_nonzero_momenta(at) and ("maxwell" in sig or has_var_kw):
            call_kw["maxwell"] = False
        try:
            res = calc.run_md(at, **call_kw)
            if isinstance(res, dict):
                if "atoms" in res:
                    res.setdefault("positions", res["atoms"].get_positions().copy())
                    res.setdefault("momenta", res["atoms"].get_momenta().copy())
                if "energy_ev" in res:
                    res.setdefault("energy", res["energy_ev"])
                if "kinetic_energy_ev" in res:
                    res.setdefault("kinetic_energy", res["kinetic_energy_ev"])
                if "total_energy_ev" in res:
                    res.setdefault("total_energy", res["total_energy_ev"])
                res.setdefault("model", entry["id"])
            return res
        except NotImplementedError:
            pass
    res = _fallback_run_md(calc, at, **md_kwargs)
    res.setdefault("model", entry["id"])
    return res
