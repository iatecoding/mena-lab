"""Molecular dynamics: Langevin (NVT), Langevin barostat (NPT, periodic only) or velocity Verlet (NVE)."""

from __future__ import annotations

import time

import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet

from mena_app import report as R
from mena_app import structures
from mena_app.routing import md_repeat
from mena_app.runtime import Cancelled, UserError
from mena_app.tasks.common import setup, write_structure, write_trajectory


def _integrator(p, atoms):
    dt = p["timestep"] * units.fs
    if p["ensemble"] == "nve":
        return VelocityVerlet(atoms, timestep=dt)
    if p["ensemble"] == "nvt":
        return Langevin(atoms, timestep=dt, temperature_K=p["temperature"], friction=p["friction"] / units.fs)
    if not atoms.pbc.all():
        raise UserError("Constant pressure needs a periodic cell.")
    try:
        from ase.md.langevinbaoab import LangevinBAOAB
        return LangevinBAOAB(atoms, timestep=dt, temperature_K=p["temperature"], externalstress=p["pressure"] * units.bar,
                             T_tau=1.0 / p["friction"] * units.fs, P_tau=1000 * units.fs, hydrostatic=True)
    except ImportError:
        from ase.md.nptberendsen import NPTBerendsen
        return NPTBerendsen(atoms, timestep=dt, temperature_K=p["temperature"], pressure_au=p["pressure"] * units.bar,
                            taut=100 * units.fs, taup=1000 * units.fs, compressibility_au=4.57e-5 / units.bar)


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
    n = len(atoms)
    MaxwellBoltzmannDistribution(atoms, temperature_K=p["temperature"], rng=np.random.default_rng(p["seed"]))
    Stationary(atoms)
    if not atoms.pbc.any():
        ZeroRotation(atoms)
    md = _integrator(p, atoms)
    frames, phys, t_ps, temps, epot, etot, vols = [], [], [], [], [], [], []
    # speed is timed from the end of the first model call: that call includes compilation (5-30 s) when compiled
    state = {"cancelled": False, "t0": time.perf_counter(), "t_run": None, "e0": None}

    def observe():
        k = md.nsteps
        ep = atoms.get_potential_energy()
        ek = atoms.get_kinetic_energy()
        T = atoms.get_temperature()
        tt = k * p["timestep"] / 1000.0
        if state["e0"] is None:
            state["e0"] = ep + ek
        t_ps.append(tt); temps.append(T); epot.append(ep); etot.append(ep + ek)
        if atoms.pbc.all():
            vols.append(atoms.get_volume())
        if k % p["save_every"] == 0:
            frames.append(atoms.copy())
            phys.append(R.snapshot_physics(atoms))
        el = time.perf_counter() - state["t_run"] if state["t_run"] is not None else 0.0
        ns_day = (k * p["timestep"] * 1e-6) / max(el, 1e-9) * 86400 if k else 0.0
        ctx.series("md_temperature", "temperature", tt, T)
        ctx.series("md_energy", "total energy", tt, (ep + ek - state["e0"]) / n * 1000)
        ctx.frame(atoms, f"{tt:.3f} ps")
        ctx.progress(k, p["steps"], f"{tt:.3f} ps of {p['steps'] * p['timestep'] / 1000:.3f} ps",
                     {"temperature": f"{T:.0f} K", "speed": f"{ns_day:.2f} ns/day"})
        if ctx.cancelled:
            raise Cancelled()

    md.attach(observe, interval=1)
    ctx.status(f"running {p['ensemble'].upper()} at {p['temperature']:.0f} K")
    try:
        atoms.get_forces()
        state["t_run"] = time.perf_counter()
        md.run(p["steps"])
    except Cancelled:
        state["cancelled"] = True
    ctx.flush()
    t_end = time.perf_counter()
    sec = t_end - state["t0"]
    startup = (state["t_run"] or t_end) - state["t0"]
    steps = md.nsteps
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
    return {
        "headline": f"{t_ps[-1] if t_ps else 0:.3f} ps of {p['ensemble'].upper()} dynamics at {p['temperature']:.0f} K in {sec:.0f} s ({ns_day:.2f} ns/day).",
        "tiles": tiles, "facts": {"Steps": steps, "Atoms": n, "Start-up (incl. compiling)": f"{startup:.1f} s"}, "tables": [], "charts": charts,
        "viewer": R.viewer([R.frame_with_physics(frames[i], phys[i], label=f"{i * p['save_every'] * p['timestep'] / 1000:.3f} ps") for i in idx], atoms),
        "files": files, "notes": ["Stopped by you; the trajectory covers the time run so far."] if state["cancelled"] else [], "cancelled": state["cancelled"],
        "raw": {"time_ps": t_ps, "temperature_K": temps, "epot_eV": epot, "etot_eV": etot, "volume_A3": vols},
    }
