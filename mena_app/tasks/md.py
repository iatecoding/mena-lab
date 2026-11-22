"""Molecular dynamics: Langevin (NVT), Langevin barostat (NPT, periodic only) or velocity Verlet (NVE)."""

from __future__ import annotations

import inspect
import math
import time

import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet

from mena_app import report as R
from mena_app import structures
from mena_app.routing import md_repeat
from mena_app.runtime import Cancelled, UserError
from mena_app.tasks.common import optimize, setup, thermalize, write_structure, write_trajectory


def _has_nonzero_momenta(atoms) -> bool:
    mom = atoms.arrays.get("momenta")
    return mom is not None and bool(np.any(np.abs(mom) > 1e-12))


def _temperature_K(atoms) -> float:
    dof = int(atoms.get_number_of_degrees_of_freedom())
    if dof <= 0:
        return 0.0
    return float(2.0 * atoms.get_kinetic_energy() / (dof * units.kB))


def _zero_rotation_safe(atoms) -> None:
    """Remove rigid-body rotation on unconstrained molecules without divide-by-zero warnings on linear systems."""
    if atoms.pbc.any() or getattr(atoms, "constraints", None) or len(atoms) < 2:
        return
    Ip, basis = atoms.get_moments_of_inertia(vectors=True)
    active = Ip > 1e-9
    if np.all(active):
        ZeroRotation(atoms)
        return
    if int(np.sum(active)) < 2:
        return
    temp0 = _temperature_K(atoms)
    Lp = np.dot(basis, atoms.get_angular_momentum())
    inv_Ip = np.zeros_like(Ip, dtype=np.float64)
    inv_Ip[active] = 1.0 / Ip[active]
    omega = np.dot(np.linalg.inv(basis), Lp * inv_Ip)
    pos = atoms.get_positions() - atoms.get_center_of_mass()
    atoms.set_velocities(atoms.get_velocities() - np.cross(omega, pos))
    if temp0 > 0.0:
        temp1 = _temperature_K(atoms)
        if temp1 > 0.0:
            atoms.set_momenta(atoms.get_momenta() * math.sqrt(temp0 / temp1))


def _integrator(p, atoms):
    dt = p["timestep"] * units.fs
    rng = np.random.default_rng(p["seed"])
    if p["ensemble"] == "nve":
        return VelocityVerlet(atoms, timestep=dt)
    if p["ensemble"] == "nvt":
        has_cons = bool(getattr(atoms, "constraints", None))
        return Langevin(
            atoms,
            timestep=dt,
            temperature_K=p["temperature"],
            friction=p["friction"] / units.fs,
            fixcm=not (has_cons or len(atoms) <= 1),
            rng=rng,
        )
    if not atoms.pbc.all():
        raise UserError("Constant pressure needs a periodic cell.")
    try:
        from ase.md.langevinbaoab import LangevinBAOAB
        return LangevinBAOAB(
            atoms,
            timestep=dt,
            temperature_K=p["temperature"],
            externalstress=-p["pressure"] * units.bar,
            T_tau=(1.0 / max(p["friction"], 1e-4)) * units.fs,
            P_tau=1000.0 * units.fs,
            hydrostatic=True,
            rng=rng,
        )
    except ImportError:
        from ase.md.nptberendsen import NPTBerendsen
        return NPTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=p["temperature"],
            pressure_au=p["pressure"] * units.bar,
            taut=100 * units.fs,
            taup=1000 * units.fs,
            compressibility_au=4.57e-5 / units.bar,
        )


def _d3_mode_of(atoms, calc) -> str | None:
    if not getattr(calc, "add_dispersion", True):
        return None
    if hasattr(calc, "_resolve_lot_and_u"):
        from lam4.calc.physics import resolve_lot_and_u

        src, _ = resolve_lot_and_u(atoms, getattr(calc, "lot", "auto"), getattr(calc, "u_override", None))
        return src.d3_addback
    return None


def _d3_handled_on_device_reason(atoms, calc, d3_mode: str | None) -> tuple[bool, str]:
    """Return `(on_device, reason)` for `MDEngine` D3 dispersion handling."""
    if d3_mode is None:
        return True, "no D3 dispersion needed"
    raw_dev = getattr(calc, "device", None)
    if raw_dev is None:
        return False, "ASE integrator (calculator device unknown for on-device D3)"
    try:
        import torch

        dev = raw_dev if isinstance(raw_dev, torch.device) else torch.device(str(raw_dev))
    except Exception:
        return False, "ASE integrator (could not resolve torch device for D3)"
    if dev.type == "cuda":
        if getattr(calc, "dtype", None) not in (torch.float32, "float32"):
            return False, "ASE integrator (double-precision CUDA D3 fallback)"
        if d3_mode not in ("bj_wb97m", "zero_pbe"):
            return False, f"ASE integrator (unsupported CUDA on-device D3 mode {d3_mode!r})"
        try:
            from lam4.calc.d3_nl_cache import has_triton_d3_on

            if has_triton_d3_on(dev):
                return True, f"on-device Triton D3 ({d3_mode})"
            return False, "ASE integrator (Triton D3 kernel unavailable on CUDA)"
        except Exception:
            return False, "ASE integrator (Triton D3 probe failed)"
    if dev.type == "mps":
        if getattr(calc, "dtype", None) in (torch.float64, "float64"):
            return False, "ASE integrator (double-precision MPS fallback)"
        if d3_mode not in ("bj_wb97m", "zero_pbe"):
            return False, f"ASE integrator (MPS Metal D3 does not support {d3_mode!r})"
        try:
            from lam4.kernels.metal_d3 import has_metal_d3

            if has_metal_d3():
                return True, f"on-device Metal D3 ({d3_mode})"
            return False, "ASE integrator (Metal D3 kernel unavailable on MPS)"
        except Exception:
            return False, "ASE integrator (Metal D3 probe failed)"
    return False, f"ASE integrator (on {dev.type.upper()} with D3 dispersion {d3_mode})"


def _d3_handled_on_device(atoms, calc, d3_mode: str | None) -> bool:
    """Return True when `MDEngine` keeps D3 dispersion entirely on-device (no per-step CPU roundtrip)."""
    ok, _ = _d3_handled_on_device_reason(atoms, calc, d3_mode)
    return ok


def _md_engine_decision(p, atoms, calc) -> tuple[bool, str]:
    cons = getattr(atoms, "constraints", None)
    if cons:
        return False, f"ASE integrator (atoms have {len(cons)} constraint(s))"
    ens = p["ensemble"]
    if ens == "npt":
        return False, "ASE barostat (NPT variable-cell dynamics)"
    if calc is None or not hasattr(calc, "md_engine"):
        return False, "ASE integrator (calculator has no GPU MDEngine)"
    if ens == "nve":
        try:
            from lam4.calc.md_engine import MDEngine

            if "ensemble" not in inspect.signature(MDEngine.__init__).parameters:
                return False, "ASE VelocityVerlet (MDEngine lacks ensemble='nve' support)"
        except Exception:
            return False, "ASE VelocityVerlet (MDEngine unavailable)"
    elif ens != "nvt":
        return False, f"ASE integrator (ensemble {ens!r})"
    d3_ok, d3_reason = _d3_handled_on_device_reason(atoms, calc, _d3_mode_of(atoms, calc))
    if not d3_ok:
        return False, d3_reason
    return True, f"MDEngine GPU-resident path ({ens.upper()}, {d3_reason})"


def _can_use_md_engine(p, atoms, calc) -> bool:
    ok, _ = _md_engine_decision(p, atoms, calc)
    return ok


def _sync_from_engine(atoms, eng) -> None:
    atoms.set_positions(eng.positions())
    mom = eng.momenta() if hasattr(eng, "momenta") else eng.st.p.detach().cpu().numpy()
    atoms.set_momenta(mom, apply_constraint=False)


def _engine_window(save_every: int, steps: int) -> int:
    g = math.gcd(max(1, int(save_every)), max(1, int(steps)))
    for w in range(min(10, g), 0, -1):
        if g % w == 0:
            return w
    return 1


def run(ctx):
    p = ctx.params
    atoms = setup(ctx)
    if p["ensemble"] == "nve" and len(atoms) < 2 or (not atoms.pbc.any() and len(atoms) < 2):
        raise UserError("Dynamics needs at least two atoms.")
    r = md_repeat(structures.analyze(atoms))
    if r > 1:
        calc = atoms.calc
        atoms = atoms.repeat((r, r, r))
        atoms.calc = calc
        ctx.log(f"cell repeated {r}×{r}×{r} to {len(atoms)} atoms")
    if int(atoms.get_number_of_degrees_of_freedom()) <= 0:
        raise UserError("All atoms are fixed; dynamics needs at least one free degree of freedom.")
    n = len(atoms)
    prerelax_steps = 0
    if p.get("prerelax", False):
        ctx.status("relaxing starting structure before dynamics")
        pre = optimize(ctx, atoms, "LBFGS", fmax=0.05, steps=200, cell=False, chart="prerelax", label="pre-relax")
        prerelax_steps = int(pre["steps"])
        ctx.log(f"pre-relax finished in {prerelax_steps} step(s) (converged={pre['converged']})")
        if pre["cancelled"]:
            files = [write_structure(ctx, atoms, "final")]
            return {
                "headline": f"Stopped during pre-relaxation after {prerelax_steps} steps.",
                "tiles": [], "facts": {"Steps": 0, "Atoms": n, "Pre-relax steps": prerelax_steps}, "tables": [], "charts": [],
                "viewer": R.viewer([R.frame_with_physics(atoms, R.snapshot_physics(atoms), label="pre-relax")], atoms),
                "files": files, "notes": ["Stopped by you during pre-relaxation before dynamics began."], "cancelled": True,
                "raw": {"time_ps": [], "temperature_K": [], "epot_eV": [], "etot_eV": [], "volume_A3": []},
            }
    has_cons = bool(getattr(atoms, "constraints", None))
    if _has_nonzero_momenta(atoms):
        if has_cons:
            atoms.set_momenta(atoms.get_momenta(), apply_constraint=True)
        ctx.log("preserving existing non-zero initial momenta")
    else:
        thermalize(atoms, temperature_K=p["temperature"], rng=np.random.default_rng(p["seed"]))
        if has_cons:
            atoms.set_momenta(atoms.get_momenta(), apply_constraint=True)
        else:
            if n > 1:
                Stationary(atoms)
            if not atoms.pbc.any():
                _zero_rotation_safe(atoms)
    frames, phys, t_ps, temps, epot, etot, vols = [], [], [], [], [], [], []
    save_every = max(1, int(p["save_every"]))
    # speed is timed from the end of the first model call: that call includes compilation (5-30 s) when compiled
    state = {"cancelled": False, "t0": time.perf_counter(), "t_run": None, "e0": None}
    steps = 0

    def _record_step(k: int, ep: float, ek: float, T: float, save_frame: bool, save_phys: bool = True) -> None:
        tt = k * p["timestep"] / 1000.0
        if state["e0"] is None:
            state["e0"] = ep + ek
        t_ps.append(tt); temps.append(T); epot.append(ep); etot.append(ep + ek)
        if atoms.pbc.all():
            vols.append(atoms.get_volume())
        if save_frame:
            frames.append(atoms.copy())
            phys.append(R.snapshot_physics(atoms) if save_phys else {})
        el = time.perf_counter() - state["t_run"] if state["t_run"] is not None else 0.0
        ns_day = (k * p["timestep"] * 1e-6) / max(el, 1e-9) * 86400 if k else 0.0
        ctx.series("md_temperature", "temperature", tt, T)
        ctx.series("md_energy", "total energy", tt, (ep + ek - state["e0"]) / n * 1000)
        if save_frame:
            ctx.frame(atoms, f"{tt:.3f} ps")
        ctx.progress(k, p["steps"], f"{tt:.3f} ps of {p['steps'] * p['timestep'] / 1000:.3f} ps",
                     {"temperature": f"{T:.0f} K", "speed": f"{ns_day:.2f} ns/day"})

    calc = atoms.calc
    use_engine, md_reason = _md_engine_decision(p, atoms, calc)
    ctx.log(f"MD path: {md_reason}")
    if use_engine:
        win = _engine_window(save_every, p["steps"])
        eng_kw = dict(
            temperature_K=p["temperature"],
            timestep_fs=p["timestep"],
            tdamp_fs=1.0 / max(p["friction"], 1e-4),
            skin=max(0.4, float(getattr(calc, "skin", 0.6))),
            window=win,
            seed=p["seed"],
            maxwell=False,
        )
        if p["ensemble"] == "nve":
            eng_kw["ensemble"] = "nve"
        dev_name = str(getattr(calc, "device", "")).lower()
        eng_tag = "fast GPU engine" if dev_name.startswith(("cuda", "mps")) else "fast MDEngine"
        ctx.status(f"running {p['ensemble'].upper()} at {p['temperature']:.0f} K ({eng_tag})")
        eng = calc.md_engine(atoms, **eng_kw)
        ep0 = eng.energy_ev()
        ek0 = float(atoms.get_kinetic_energy())
        T0 = _temperature_K(atoms)
        _record_step(0, ep0, ek0, T0, True, save_phys=bool(getattr(calc, "results", None)))
        if ctx.cancelled:
            state["cancelled"] = True
        else:
            state["t_run"] = time.perf_counter()
            n_win = p["steps"] // win
            try:
                for _ in range(n_win):
                    if ctx.cancelled:
                        raise Cancelled()
                    eng.window_step()
                    k = int(eng.stats["steps"])
                    if k % save_every == 0 or k >= p["steps"]:
                        _sync_from_engine(atoms, eng)
                        ek = float(atoms.get_kinetic_energy())
                        T = _temperature_K(atoms)
                        _record_step(k, eng.energy_ev(), ek, T, k % save_every == 0, save_phys=False)
            except Cancelled:
                state["cancelled"] = True
                _sync_from_engine(atoms, eng)
        steps = int(eng.stats["steps"])
        if frames and not phys[-1]:
            atoms.calc = calc
            try:
                target_fr = atoms if (steps == 0 or steps % save_every == 0) else frames[-1].copy()
                if target_fr is not atoms:
                    target_fr.calc = calc
                calc.calculate(target_fr, properties=("energy", "forces"), system_changes=("positions",))
                phys[-1] = R.snapshot_physics(target_fr)
            except Exception:
                pass
            finally:
                calc.atoms = atoms
        ctx.flush()
    else:
        md = _integrator(p, atoms)

        def observe():
            k = md.nsteps
            ep = atoms.get_potential_energy()
            ek = atoms.get_kinetic_energy()
            T = _temperature_K(atoms)
            _record_step(k, ep, ek, T, k % save_every == 0, save_phys=True)
            if ctx.cancelled:
                raise Cancelled()

        md.attach(observe, interval=save_every)
        ctx.status(f"running {p['ensemble'].upper()} at {p['temperature']:.0f} K")
        try:
            atoms.get_forces()
            state["t_run"] = time.perf_counter()
            md.run(p["steps"])
        except Cancelled:
            state["cancelled"] = True
        steps = md.nsteps
        if not state["cancelled"] and steps > 0 and steps % save_every != 0:
            observe()
        ctx.flush()
    t_end = time.perf_counter()
    sec = t_end - state["t0"]
    startup = (state["t_run"] or t_end) - state["t0"]
    run_sec = t_end - (state["t_run"] or t_end)
    ns_day = steps * p["timestep"] * 1e-6 / max(run_sec, 1e-9) * 86400
    drift = (etot[-1] - etot[0]) / n * 1000 / max(t_ps[-1], 1e-9) if len(etot) > 1 else 0.0
    half = len(temps) // 2
    tiles = [R.tile("sim_time", "Simulated time", t_ps[-1] if t_ps else 0, "ps", 3),
             R.tile("speed", "Speed", ns_day, "ns/day", 3, better="higher"),
             R.tile("ms_step", "Time per step", run_sec / max(steps, 1) * 1000, "ms", 2, better="lower"),
             R.tile("t_mean", "Mean temperature (second half)", float(np.mean(temps[half:])) if temps else None, "K", 1)]
    if p["ensemble"] == "nve":
        tiles.append(R.tile("drift", "Energy drift", drift, "meV/atom/ps", 3, better="lower", hint="Constant-energy runs should keep this near zero."))
    if vols:
        tiles.append(R.tile("dvol", "Volume change", 100 * (vols[-1] / vols[0] - 1), "%", 2))
    charts = [R.chart("md_temperature", "Temperature", [{"name": "temperature", "x": t_ps, "y": temps}], "line", "time (ps)", "K"),
              R.chart("md_energy", "Total energy (per atom, from start)", [{"name": "total energy", "x": t_ps, "y": [(e - etot[0]) / n * 1000 for e in etot]}], "line", "time (ps)", "meV/atom")]
    idx = R.subsample(len(frames), 80)
    files = [write_trajectory(ctx, frames), write_structure(ctx, atoms, "final")]
    facts = {"Steps": steps, "Atoms": n, "Start-up (incl. compiling)": f"{startup:.1f} s"}
    if prerelax_steps:
        facts["Pre-relax steps"] = prerelax_steps
    return {
        "headline": f"{t_ps[-1] if t_ps else 0:.3f} ps of {p['ensemble'].upper()} dynamics at {p['temperature']:.0f} K in {sec:.0f} s ({ns_day:.2f} ns/day).",
        "tiles": tiles, "facts": facts, "tables": [], "charts": charts,
        "viewer": R.viewer([R.frame_with_physics(frames[i], phys[i], label=f"{i * p['save_every'] * p['timestep'] / 1000:.3f} ps") for i in idx], atoms),
        "files": files, "notes": ["Stopped by you; the trajectory covers the time run so far."] if state["cancelled"] else [], "cancelled": state["cancelled"],
        "raw": {"time_ps": t_ps, "temperature_K": temps, "epot_eV": epot, "etot_eV": etot, "volume_A3": vols},
    }
