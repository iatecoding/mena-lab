"""Plug-and-play MD & single-structure relaxation orchestrator with MDEngine / ASE routing."""
from __future__ import annotations

import math
import time
import warnings
from typing import Any, Callable, Sequence

from ase import Atoms, units
from ase.md import velocitydistribution as _vd
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.verlet import VelocityVerlet
import numpy as np
import torch

from lam4.calc import md_props as _mdp
from lam4.calc.md_nhc import NHCState
from lam4.calc.physics import resolve_lot_and_u
from lam4.calc.precision_hygiene import center_positions_f64, run_safe_ase_relax

_OPTIMIZERS = {"lbfgs": "LBFGS", "bfgs": "BFGS", "fire": "FIRE"}


def _thermalize_and_zero_com(atoms: Atoms, temperature_K: float, seed: int, remove_com: bool) -> None:
    fn = getattr(_vd, "thermalize_momenta", _vd.MaxwellBoltzmannDistribution)
    fn(atoms, temperature_K=float(temperature_K), rng=np.random.default_rng(seed))
    if remove_com and len(atoms) > 1 and not getattr(atoms, "constraints", None) and not np.allclose(atoms.get_momenta(), 0.0, atol=1e-12):
        _vd.Stationary(atoms)
        if not bool(np.any(atoms.get_pbc())):
            _vd.ZeroRotation(atoms)


def init_md_engine_state(
    atoms: Atoms, dev: torch.device, *, ensemble: str = "nvt", temperature_K: float | None = 300.0,
    timestep_fs: float = 0.25, tdamp_fs: float = 25.0, maxwell: bool = True, seed: int = 0,
) -> tuple[str, Atoms, NHCState, float | None, float]:
    """Validate MD ensemble args, optionally thermalize `atoms.copy()`, and build `NHCState` on `dev`."""
    ens = str(ensemble).strip().lower()
    if ens not in ("nve", "nvt"):
        raise ValueError(f"Unsupported MDEngine ensemble {ensemble!r}; expected 'nve' or 'nvt'.")
    dt_fs, td_fs = float(timestep_fs), float(tdamp_fs)
    if dt_fs <= 0.0:
        raise ValueError(f"timestep_fs must be > 0, got {timestep_fs}")
    t_val = None if temperature_K is None else float(temperature_K)
    if t_val is not None and t_val < 0.0:
        raise ValueError(f"temperature_K must be >= 0, got {temperature_K}")
    if ens == "nvt" and (t_val is None or t_val <= 0.0 or td_fs <= 0.0):
        raise ValueError("temperature_K > 0 and tdamp_fs > 0 are required when ensemble='nvt'.")
    at = atoms.copy()
    if maxwell and t_val is not None:
        _thermalize_and_zero_com(at, t_val, seed, False) if t_val > 0.0 else at.set_momenta(np.zeros((len(at), 3), dtype=np.float64))
    at_state = at.copy()
    at_state.set_positions(center_positions_f64(at.get_positions())[0])
    if ens == "nvt":
        assert t_val is not None
        dyn = NoseHooverChainNVT(at_state, timestep=dt_fs * units.fs, temperature_K=t_val, tdamp=td_fs * units.fs)
        return ens, at, NHCState.from_ase(dyn, dev), t_val, td_fs
    return ens, at, NHCState.for_nve(at_state, dt_fs, dev), t_val, td_fs


def retarget_md_engine(
    eng: Any, *, temperature_K: float | None = None, timestep_fs: float | None = None, tdamp_fs: float | None = None,
) -> None:
    """Update timestep and (in NVT only) thermostat parameters on a live `MDEngine`."""
    if eng.ensemble == "nve" and (temperature_K is not None or tdamp_fs is not None):
        raise ValueError("Cannot retarget thermostat parameters (temperature_K, tdamp_fs) when ensemble='nve'.")
    if timestep_fs is not None:
        if float(timestep_fs) <= 0.0:
            raise ValueError(f"timestep_fs must be > 0, got {timestep_fs}")
        eng.st.update_params(timestep_fs=float(timestep_fs))
        eng.graph = None
    if eng.ensemble == "nvt" and (temperature_K is not None or tdamp_fs is not None):
        if temperature_K is not None:
            if float(temperature_K) <= 0.0:
                raise ValueError(f"temperature_K must be > 0 for ensemble='nvt', got {temperature_K}")
            eng._temperature_K = float(temperature_K)
        if tdamp_fs is not None:
            if float(tdamp_fs) <= 0.0:
                raise ValueError(f"tdamp_fs must be > 0, got {tdamp_fs}")
            eng._tdamp_fs = float(tdamp_fs)
        eng.st.update_params(temperature_K=eng._temperature_K, tdamp_fs=eng._tdamp_fs)
        eng.graph = None


def sync_md_atoms(eng: Any, atoms: Atoms | None = None) -> Atoms:
    """Copy `MDEngine` positions & momenta back to `atoms` (default `eng.atoms`) and clear stale ASE cache."""
    target = eng.atoms if atoms is None else atoms
    target.set_positions(eng.positions())
    target.set_momenta(np.asarray(eng.st.p.detach().cpu().to(torch.float64).numpy(), dtype=np.float64))
    calc = getattr(target, "calc", None)
    if calc is not None:
        if hasattr(calc, "results"):
            calc.results = {}
        if hasattr(calc, "atoms"):
            calc.atoms = None
    return target


def can_use_md_engine(calc: Any, atoms: Atoms, ensemble: str = "nvt") -> tuple[bool, str]:
    """Return `(True, '')` when fast GPU/CPU-resident `MDEngine` applies, else `(False, reason)`."""
    ens = str(ensemble).strip().lower()
    if ens == "npt":
        return False, "ensemble 'npt' uses ASE barostat/thermostat"
    if ens not in ("nve", "nvt"):
        return False, f"unsupported ensemble {ensemble!r} for MDEngine"
    if len(atoms) == 0:
        return False, "atoms is empty"
    if getattr(atoms, "constraints", None):
        return False, "atoms has ASE constraints"
    if atoms.arrays.get("frag_id") is not None or atoms.info.get("frag_id") is not None:
        return False, "atoms has per-fragment charges/spins (frag_id) which use ASE MD"
    rk = atoms.info.get("restraint_k", None)
    if rk is not None:
        try:
            if np.any(np.asarray(rk, dtype=np.float64) > 0.0):
                return False, "atoms has harmonic restraints (restraint_k)"
        except (TypeError, ValueError):
            return False, "atoms has invalid restraint_k"
    if calc is None or not hasattr(calc, "model") or not hasattr(calc, "device") or not hasattr(calc, "md_engine"):
        return False, "calculator does not expose MenaCalculator/MDEngine interface"
    if not getattr(calc, "add_dispersion", True):
        return True, ""
    z_np = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    src, _ = calc._resolve_lot_cached(atoms, z_np) if hasattr(calc, "_resolve_lot_cached") else resolve_lot_and_u(atoms, getattr(calc, "lot", "auto"), getattr(calc, "u_override", None))
    d3_mode = src.d3_addback
    if d3_mode is None:
        return True, ""
    dev = calc.device if isinstance(calc.device, torch.device) else torch.device(str(calc.device))
    if dev.type == "cpu":
        return True, ""
    pbc_any, dtype = bool(np.any(atoms.get_pbc())), getattr(calc, "dtype", None)
    if dev.type == "mps":
        if dtype in (torch.float64, "float64"):
            return False, "MPS on-device D3 does not support float64"
        if d3_mode not in ("bj_wb97m", "zero_pbe"):
            return False, f"MPS on-device D3 does not support mode {d3_mode!r}"
        from lam4.kernels.metal_d3 import has_metal_d3
        return (True, "") if has_metal_d3() else (False, "Metal D3 kernel unavailable")
    if dev.type == "cuda":
        if dtype not in (torch.float32, "float32"):
            return False, "CUDA on-device D3 requires float32"
        if d3_mode not in ("bj_wb97m", "zero_pbe"):
            return False, f"CUDA on-device D3 does not support mode {d3_mode!r}"
        from lam4.calc.d3_nl_cache import has_triton_d3_on
        return (True, "") if has_triton_d3_on(dev) else (False, "Triton D3 kernel unavailable")
    return False, f"unsupported device type {dev.type!r} for MDEngine D3"


def _choose_md_window(window: int, steps: int, log_interval: int, need_hooks: bool) -> int:
    w_max = max(1, min(int(window), steps))
    base = math.gcd(steps, max(1, int(log_interval))) if need_hooks else steps
    for w in range(min(w_max, base), 0, -1):
        if base % w == 0:
            return w
    for w in range(w_max, 0, -1):
        if steps % w == 0:
            return w
    return 1


def _emit_step(sink: _mdp.TrajSink, callback: Callable[[int, Atoms, dict[str, Any]], Any] | None,
               step: int, work: Atoms, e: float, f: np.ndarray | None, s: np.ndarray | None) -> None:
    sink.write(work, energy=e, forces=f, stress=s)
    if callback is not None:
        ek = float(work.get_kinetic_energy())
        callback(step, work, {"step": step, "energy_ev": e, "kinetic_energy_ev": ek,
                              "total_energy_ev": e + ek, "temperature_K": float(work.get_temperature())})


def _run_fast_md(
    calc: Any, work: Atoms, ens: str, steps: int, dt_fs: float, temperature_K: float, tdamp_fs: float,
    skin: float | None, window: int, use_graph: bool | str, use_compile: bool | str, seed: int,
    rec_props: tuple[str, ...], sink: _mdp.TrajSink, log_iv: int, callback: Callable[[int, Atoms, dict[str, Any]], Any] | None,
) -> tuple[float, np.ndarray, np.ndarray | None, dict[str, np.ndarray] | None]:
    calc_skin = float(getattr(calc, "skin", 0.0))
    eff_skin = (calc_skin if calc_skin > 0.0 else 0.6) if skin is None else float(skin)
    need_hooks = callback is not None or sink._target is not None
    eff_win = _choose_md_window(window, steps, log_iv, need_hooks)
    eng = calc.md_engine(
        work, ensemble=ens, temperature_K=temperature_K, timestep_fs=dt_fs, tdamp_fs=tdamp_fs, skin=eff_skin,
        window=eff_win, use_graph=use_graph, use_compile=use_compile, seed=seed, maxwell=False, record_properties=rec_props,
    )
    want_pass = _mdp.needs_passthrough_frames(rec_props)
    if not need_hooks:
        eng.run(steps, record=want_pass)
    else:
        for _ in range(steps // eff_win):
            eng.window_step()
            prop_w = _mdp.collect_window_properties(eng) if eng._want_props else None
            if want_pass:
                _mdp.append_recorded_frame(eng, prop_w)
            k = int(eng.stats["steps"])
            if k % log_iv == 0:
                eng.sync_atoms(work)
                _emit_step(sink, callback, k, work, float(eng.energy_ev()), eng.forces(), eng.stress_voigt() if eng.can_stress else None)
    eng.sync_atoms(work)
    e_pot, f_ev = float(eng.energy_ev()), np.asarray(eng.forces(), dtype=np.float64).copy()
    s_ev = np.asarray(eng.stress_voigt(), dtype=np.float64).copy() if eng.can_stress else None
    if sink._target is not None and sink.count == 0:
        sink.write(work, energy=e_pot, forces=f_ev, stress=s_ev)
    return e_pot, f_ev, s_ev, _mdp.finalize_engine_props(eng, rec_props)


def _run_ase_md(
    calc: Any, work: Atoms, ens: str, steps: int, dt_fs: float, temperature_K: float, pressure_GPa: float,
    tdamp_fs: float, pdamp_fs: float, friction_per_fs: float, seed: int, rest_k: np.ndarray | None,
    rest_ref: np.ndarray | None, can_stress: bool, rec_props: tuple[str, ...], sink: _mdp.TrajSink, log_iv: int,
    callback: Callable[[int, Atoms, dict[str, Any]], Any] | None,
) -> tuple[float, np.ndarray, np.ndarray | None, dict[str, np.ndarray] | None]:
    from lam4.calc.relax import _RestrainedCalc

    if rest_k is not None:
        if ens == "npt":
            raise ValueError("Harmonic restraints (restraint_k) are not supported with ensemble='npt'.")
        assert rest_ref is not None
        work.calc = _RestrainedCalc(calc, rest_k, rest_ref)
    dt_ase = dt_fs * units.fs
    if ens == "nve":
        dyn: Any = VelocityVerlet(work, timestep=dt_ase)
    elif ens == "nvt":
        dyn = (
            Langevin(work, timestep=dt_ase, temperature_K=float(temperature_K), friction=float(friction_per_fs) / units.fs,
                     fixcm=False, rng=np.random.default_rng(seed))
            if getattr(work, "constraints", None)
            else NoseHooverChainNVT(work, timestep=dt_ase, temperature_K=float(temperature_K), tdamp=float(tdamp_fs) * units.fs)
        )
    else:
        if getattr(work, "constraints", None):
            raise ValueError("ASE IsotropicMTKNPT does not support constraints; remove constraints for NPT.")
        dyn = IsotropicMTKNPT(work, timestep=dt_ase, temperature_K=float(temperature_K), pressure_au=float(pressure_GPa) * units.GPa,
                              tdamp=float(tdamp_fs) * units.fs, pdamp=float(pdamp_fs) * units.fs)
    hist: dict[str, list[Any]] = {}

    def _obs() -> None:
        k = int(dyn.nsteps)
        if k <= 0:
            return
        if rec_props:
            _mdp.ase_collect_step(hist, k, work, calc, rec_props)
        if k % log_iv == 0:
            e_k = float(work.calc.results["energy"]) if "energy" in work.calc.results else float(work.get_potential_energy())
            _emit_step(sink, callback, k, work, e_k, work.calc.results.get("forces"), calc.results.get("stress") if can_stress else None)

    if rec_props or callback is not None or sink._target is not None:
        dyn.attach(_obs, interval=1)
    dyn.run(steps)
    e_pot = float(work.get_potential_energy())
    f_ev = np.asarray(work.get_forces(), dtype=np.float64).copy()
    s_ev = np.asarray(calc.get_stress(work) if rest_k is not None else work.get_stress(), dtype=np.float64).copy() if can_stress else None
    if sink._target is not None and sink.count == 0:
        sink.write(work, energy=e_pot, forces=f_ev, stress=s_ev)
    if rest_k is not None:
        work.calc = calc
    return e_pot, f_ev, s_ev, ({k: np.asarray(v) for k, v in hist.items()} if (rec_props and hist) else None)


def run_md(
    calc: Any, atoms: Atoms, *, ensemble: str = "nvt", steps: int = 1000, timestep_fs: float | None = None,
    temperature_K: float = 300.0, pressure_GPa: float = 0.0, tdamp_fs: float = 25.0, pdamp_fs: float = 1000.0,
    friction_per_fs: float = 0.01, engine: str = "auto", maxwell: bool | str = "auto", remove_com: bool = True,
    skin: float | None = None, window: int = 10, use_graph: bool | str = "auto", use_compile: bool | str = "auto",
    record_properties: Sequence[str] = (), trajectory: str | Any | None = None, log_interval: int = 10,
    callback: Callable[[int, Atoms, dict[str, Any]], Any] | None = None, seed: int = 0, inplace: bool = True,
) -> dict[str, Any]:
    """Run NVT / NVE / NPT molecular dynamics, routing automatically to fast `MDEngine` or ASE fallback."""
    from lam4.calc.relax import _RestrainedCalc, _parse_restraints

    ens = str(ensemble).strip().lower()
    if ens not in ("nve", "nvt", "npt"):
        raise ValueError(f"Unsupported ensemble {ensemble!r}; expected 'nve', 'nvt', or 'npt'.")
    steps_i, eng_mode = int(steps), str(engine).strip().lower()
    if steps_i < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    if eng_mode not in ("auto", "md_engine", "ase"):
        raise ValueError(f"Unsupported engine {engine!r}; expected 'auto', 'md_engine', or 'ase'.")
    if ens == "npt" and not bool(np.all(atoms.get_pbc())):
        raise ValueError("ensemble='npt' requires 3D periodic boundary conditions (pbc=(True, True, True)).")
    if ens in ("nvt", "npt") and (temperature_K is None or float(temperature_K) <= 0.0):
        raise ValueError(f"temperature_K > 0 is required for ensemble={ens!r}.")
    if getattr(calc, "precision", "auto") == "bf16_nodes":
        warnings.warn("precision='bf16_nodes' is non-conservative (breaks NVE dt^2 scaling and Hessian symmetry); prefer 'auto' or 'fp32_ieee'.", RuntimeWarning, stacklevel=2)
    rec_props = _mdp.validate_record_props(record_properties)
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    dt_fs = (0.5 if (len(masses) > 0 and float(masses.min()) <= 4.5) else 1.0) if timestep_fs is None else float(timestep_fs)
    if dt_fs <= 0.0:
        raise ValueError(f"timestep_fs must be > 0, got {dt_fs}")

    work = atoms if inplace else atoms.copy()
    rest_k, rest_ref = _parse_restraints(work)
    work.calc = calc
    maxw_auto = isinstance(maxwell, str) and maxwell.strip().lower() == "auto"
    if not (isinstance(maxwell, bool) or maxw_auto):
        raise ValueError(f"maxwell must be True, False, or 'auto', got {maxwell!r}")
    if maxwell is True or (maxw_auto and np.allclose(work.get_momenta(), 0.0, atol=1e-12)):
        if temperature_K is None:
            raise ValueError("temperature_K is required to thermalize momenta.")
        t_init = float(temperature_K)
        if t_init > 0.0:
            _thermalize_and_zero_com(work, t_init, seed, remove_com)
        elif maxwell is True:
            work.set_momenta(np.zeros((len(work), 3), dtype=np.float64))

    ok, reason = can_use_md_engine(calc, work, ens)
    if eng_mode == "md_engine" and not ok:
        raise ValueError(reason)
    use_fast = (eng_mode == "md_engine") or (eng_mode == "auto" and ok)
    fallback_reason = None if use_fast else (reason if not ok else "engine='ase' requested")
    sink, t0 = _mdp.TrajSink(trajectory), time.perf_counter()
    can_stress = bool(np.all(work.get_pbc()) and abs(float(np.linalg.det(work.cell.array))) > 1e-8)

    try:
        if steps_i == 0:
            if rest_k is not None:
                assert rest_ref is not None
                work.calc = _RestrainedCalc(calc, rest_k, rest_ref)
            e_pot = float(work.get_potential_energy())
            f_ev = np.asarray(work.get_forces(), dtype=np.float64).copy()
            s_ev = np.asarray(calc.get_stress(work) if rest_k is not None else work.get_stress(), dtype=np.float64).copy() if can_stress else None
            work.calc, props_out = calc, None
        elif use_fast:
            e_pot, f_ev, s_ev, props_out = _run_fast_md(
                calc, work, ens, steps_i, dt_fs, temperature_K, tdamp_fs, skin, window,
                use_graph, use_compile, seed, rec_props, sink, max(1, int(log_interval)), callback,
            )
        else:
            e_pot, f_ev, s_ev, props_out = _run_ase_md(
                calc, work, ens, steps_i, dt_fs, temperature_K, pressure_GPa, tdamp_fs, pdamp_fs,
                friction_per_fs, seed, rest_k, rest_ref, can_stress, rec_props, sink, max(1, int(log_interval)), callback,
            )
    finally:
        sink.close()

    elapsed = time.perf_counter() - t0
    e_kin = float(work.get_kinetic_energy())
    return {
        "atoms": work, "backend": "md_engine" if use_fast else "ase", "fallback_reason": fallback_reason,
        "ensemble": ens, "steps": steps_i, "timestep_fs": float(dt_fs),
        "temperature_K": float(work.get_temperature()) if len(work) > 0 else 0.0,
        "energy_ev": e_pot, "kinetic_energy_ev": e_kin, "total_energy_ev": e_pot + e_kin,
        "forces_ev_a": f_ev, "stress_ev_a3": s_ev, "trajectory_properties": props_out,
        "elapsed_s": float(elapsed),
        "ms_per_step": (1000.0 * elapsed / steps_i) if steps_i > 0 else 0.0,
        "steps_per_s": (steps_i / max(elapsed, 1e-12)) if steps_i > 0 else 0.0,
    }


def relax_atoms(
    calc: Any, atoms: Atoms, *, fmax: float = 0.01, steps: int = 200, maxstep: float = 0.2,
    relax_cell: bool = False, pressure_GPa: float = 0.0, optimizer: str = "lbfgs",
    trajectory: str | Any | None = None, logfile: str | Any | None = None, inplace: bool = True,
) -> dict[str, Any]:
    """Relax a single molecule, crystal, slab (`FixAtoms`), or restrained structure via ASE optimizers."""
    import ase.optimize as _ao
    from ase.filters import FrechetCellFilter
    from lam4.calc.relax import _RestrainedCalc, _parse_restraints

    pbc = tuple(bool(x) for x in atoms.get_pbc())
    pbc_any = any(pbc)
    if relax_cell and not pbc_any:
        raise ValueError("relax_cell=True requires periodic boundary conditions")
    if relax_cell and not all(pbc):
        raise ValueError("relax_cell=True requires 3D periodic boundary conditions (all pbc True) for stress")
    opt_key = str(optimizer).strip().lower()
    if opt_key not in _OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer {optimizer!r}; expected one of {sorted(_OPTIMIZERS)}")
    opt_cls = getattr(_ao, _OPTIMIZERS[opt_key])
    if getattr(calc, "precision", "auto") == "bf16_nodes":
        warnings.warn("precision='bf16_nodes' stalls L-BFGS below ~1e-3 eV/A and can hit y*s==0 NaNs; prefer 'auto', 'fp32_ieee', or 'exact'.", RuntimeWarning, stacklevel=2)

    work = atoms if inplace else atoms.copy()
    rest_k, rest_ref = _parse_restraints(work)
    if relax_cell and rest_k is not None:
        raise ValueError("Harmonic restraints (restraint_k) are not supported with relax_cell=True.")
    work.calc = _RestrainedCalc(calc, rest_k, rest_ref) if (rest_k is not None and rest_ref is not None) else calc
    target: Any = FrechetCellFilter(work, scalar_pressure=float(pressure_GPa) * units.GPa) if relax_cell else work

    t0, sink = time.perf_counter(), _mdp.TrajSink(trajectory)
    try:
        opt = opt_cls(target, maxstep=float(maxstep), logfile=logfile, trajectory=None)
        if trajectory is not None:
            opt.attach(lambda: sink.write(work, energy=float(work.get_potential_energy()), forces=work.get_forces()))
        converged, _ = run_safe_ase_relax(opt, target, fmax=float(fmax), steps=int(steps), calc=calc)
        f_chk = np.asarray(target.get_forces(), dtype=np.float64)
        final_fmax = float(np.sqrt((f_chk ** 2).sum(axis=-1)).max()) if len(f_chk) else 0.0
        energy_ev = float(work.get_potential_energy())
    finally:
        sink.close()
        if rest_k is not None:
            work.calc = calc

    return {
        "atoms": work, "converged": converged, "fmax": final_fmax, "steps": int(opt.nsteps),
        "energy_ev": energy_ev, "cell_a": np.asarray(work.cell.array, dtype=np.float64).copy() if pbc_any else None,
        "elapsed_s": float(time.perf_counter() - t0),
    }
