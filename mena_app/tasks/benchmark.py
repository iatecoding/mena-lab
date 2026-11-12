"""Speed and memory benchmark on this computer:
1. `visual_md` ("Live visual MD"): real-time 3D molecular droplet or crystal supercell dynamics, streaming live 3D atomic
   frames, throughput (atoms/s & atoms/ms), step time, peak memory, temperature, and NVE energy conservation.
2. `ladder` ("Scaling ladder"): growing copies of a molecular cluster or crystal supercell, each size in its own process so peak
   memory is honest and an out-of-memory ends only that size, while streaming the growing 3D systems to the viewer."""

from __future__ import annotations

import json
import math
import time

import numpy as np
from ase import Atoms, units
from ase.constraints import FixConstraint
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet

from mena_app import backends, report as R
from mena_app import runner, structures, system
from mena_app.runtime import Cancelled, UserError
from mena_app.tasks.common import prepare_atoms, write_trajectory

RESERVE_GB = 3.0
MAX_VIS_ATOMS = 1800


class _SphericalWall(FixConstraint):
    """Conservative soft spherical wall keeping open-boundary molecular droplets coherent during NVE dynamics."""

    def __init__(self, center: np.ndarray, radius: float, k_spring: float = 0.25) -> None:
        self.center = np.asarray(center, dtype=np.float64).copy()
        self.radius = float(radius)
        self.k_spring = float(k_spring)

    def get_removed_dof(self, atoms) -> int:
        return 0

    def adjust_positions(self, atoms, newpositions) -> None:
        pass

    def adjust_forces(self, atoms, forces) -> None:
        vec = atoms.positions - self.center
        r = np.linalg.norm(vec, axis=1)
        over = r > self.radius
        if np.any(over):
            dr = r[over] - self.radius
            unit_vec = vec[over] / np.maximum(r[over], 1e-12)[:, None]
            forces[over] -= self.k_spring * dr[:, None] * unit_vec

    def adjust_potential_energy(self, atoms) -> float:
        r = np.linalg.norm(atoms.positions - self.center, axis=1)
        over = np.maximum(r - self.radius, 0.0)
        return float(0.5 * self.k_spring * np.sum(over * over))

    def index_shuffle(self, atoms, ind) -> None:
        pass

    def copy(self):
        return _SphericalWall(self.center, self.radius, self.k_spring)


def ladder(n0: int, max_atoms: int) -> list[tuple[int, int, int]]:
    """Repeats (a, b, c) giving roughly doubling sizes from ~100 atoms to max_atoms."""
    out, seen = [], set()
    target = max(n0, 100)
    while target <= max_atoms * 1.0001:
        k = target / n0
        r = [max(1, round(k ** (1 / 3)))] * 3
        i = 0
        while r[0] * r[1] * r[2] * n0 < target * 0.9:
            r[i % 3] += 1
            i += 1
        key = r[0] * r[1] * r[2]
        if key not in seen and key * n0 <= max_atoms * 1.15:
            seen.add(key)
            out.append(tuple(r))
        target *= 2
    if not out or n0 * math.prod(out[-1]) < 0.8 * max_atoms:  # a last size close to the requested maximum
        k = max(1, int((max_atoms / n0) ** (1 / 3)))
        r = [k, k, k]
        for i in range(3):
            if (math.prod(r) // r[i]) * (r[i] + 1) * n0 <= max_atoms * 1.05:
                r[i] += 1
        if math.prod(r) not in seen and math.prod(r) * n0 >= 64:
            out.append(tuple(r))
    return out


def _best_repeat(n0: int, target_atoms: int) -> tuple[int, int, int]:
    """Repeat (a, b, c) giving an isotropic supercell close to target_atoms."""
    k = max(1, round((max(target_atoms, n0) / n0) ** (1 / 3)))
    r = [k, k, k]
    for i in range(3):
        if n0 * math.prod(r) < target_atoms * 0.85:
            r[i] += 1
    return tuple(r)


def _make_vis_extractor(atoms, unit):
    """Return a fast frame-extractor capping viewer atom count at MAX_VIS_ATOMS while tracking whole molecules/cells smoothly."""
    if len(atoms) <= MAX_VIS_ATOMS:
        def _full(cur):
            fr = cur.copy()
            fr.set_constraint()
            return fr
        return _full
    if any(atoms.pbc):
        rep = _best_repeat(len(unit), 1000)
        template = unit.repeat(rep)
        n_sub = len(template)

        def _crys(cur):
            sub = template.copy()
            sub.positions = cur.positions[:n_sub].copy()
            return sub
        return _crys
    n0 = max(len(unit), 1)
    n_mol = len(atoms) // n0
    k_vis = min(n_mol, max(1, 1000 // n0))
    mol_coms = atoms.positions.reshape(n_mol, n0, 3).mean(axis=1)
    center = atoms.positions.mean(axis=0)
    mol_idx = np.sort(np.argsort(np.linalg.norm(mol_coms - center, axis=1))[:k_vis])
    atom_idx = (mol_idx[:, None] * n0 + np.arange(n0)[None, :]).reshape(-1)
    template = Atoms(numbers=atoms.numbers[atom_idx], positions=atoms.positions[atom_idx], pbc=False)

    def _mol(cur):
        sub = template.copy()
        sub.positions = cur.positions[atom_idx].copy()
        return sub
    return _mol


def _used_gb(device: str) -> float:
    import torch

    rss = system.process_rss_bytes() / 1e9
    gpu = (
        torch.mps.driver_allocated_memory() / 1e9
        if device == "mps" and hasattr(torch, "mps")
        else (torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" and torch.cuda.is_available() else 0.0)
    )
    return rss + gpu


def _run_visual_md(ctx, unit):
    import torch

    p, d = ctx.params, ctx.decision
    dev = d["device"]
    n0 = len(unit)
    rep = _best_repeat(n0, p["max_atoms"])
    atoms = structures.tile_system(unit, rep)
    is_mol = not any(atoms.pbc)
    n = len(atoms)
    edges = structures.count_edges(atoms)
    unit_formula = structures.analyze(unit)["formula"]
    n_steps = max(2, p["steps"] if p["steps"] != 5 else (22 if is_mol else 28))
    dt_fs = 0.5 if is_mol else 1.5
    pulse_temp = float(p.get("temperature_K", 600.0))

    sync = (
        torch.mps.synchronize
        if dev == "mps" and hasattr(torch, "mps")
        else (torch.cuda.synchronize if dev == "cuda" and torch.cuda.is_available() else (lambda: None))
    )
    calc_dec = {**d, "dtype": "float32", "skin": 0.6}
    atoms.calc = backends.build(ctx.model, calc_dec)
    sync()
    base_gb = _used_gb(dev)

    rng = np.random.default_rng(0)
    if is_mol:
        default_temp = abs(pulse_temp - 600.0) < 1e-6
        outer_T = 220.0 if default_temp else max(80.0, pulse_temp * 0.65)
        core_T = 360.0 if default_temp else max(150.0, pulse_temp * 1.15)
        center = atoms.positions.mean(axis=0)
        MaxwellBoltzmannDistribution(atoms, temperature_K=outer_T, rng=rng)
        mol_coms = atoms.positions.reshape(-1, n0, 3).mean(axis=1)
        mol_dists = np.linalg.norm(mol_coms - center, axis=1)
        core_mask = np.repeat(mol_dists <= np.percentile(mol_dists, 25.0), n0)
        hot = atoms.copy()
        MaxwellBoltzmannDistribution(hot, temperature_K=core_T, rng=rng)
        mom = atoms.get_momenta()
        mom[core_mask] = hot.get_momenta()[core_mask]
        atoms.set_momenta(mom)
        Stationary(atoms)
        if len(atoms) > 1:
            ZeroRotation(atoms)
        r_max = float(np.linalg.norm(atoms.positions - center, axis=1).max()) if len(atoms) else 5.0
        atoms.set_constraint(_SphericalWall(center, radius=r_max + 2.0, k_spring=0.25))
    else:
        # Initialise a thermal hotspot in the central core (like the 10k CsSnI3 shockwave showcase)
        core_T = max(300.0, pulse_temp * 1.9)
        MaxwellBoltzmannDistribution(atoms, temperature_K=max(100.0, pulse_temp * 0.4), rng=rng)
        center = atoms.cell.sum(axis=0) * 0.5
        dists = np.linalg.norm(atoms.positions - center, axis=1)
        core_cut = np.percentile(dists, 22.0)
        core_mask = dists <= core_cut
        hot = atoms.copy()
        MaxwellBoltzmannDistribution(hot, temperature_K=core_T, rng=rng)
        mom = atoms.get_momenta()
        mom[core_mask] = hot.get_momenta()[core_mask]
        atoms.set_momenta(mom)
        Stationary(atoms)

    vis_of = _make_vis_extractor(atoms, unit)
    grid_lbl = "grid" if is_mol else "cells"
    ctx.status(f"Warming up {unit_formula} ({n:,} atoms, {rep[0]}×{rep[1]}×{rep[2]} {grid_lbl}, {edges:,} edges) on {dev.upper()}…")
    ctx.frame(vis_of(atoms), f"{unit_formula} · {n:,} atoms (warming up)")
    ctx.flush()

    t0_cold = time.perf_counter()
    atoms.get_forces()
    ep0 = atoms.get_potential_energy()
    sync()
    cold_ms = (time.perf_counter() - t0_cold) * 1e3
    e0 = ep0 + atoms.get_kinetic_energy()

    md = VelocityVerlet(atoms, timestep=dt_fs * units.fs)
    frames = [vis_of(atoms)]
    steps_x, speed_s, speed_ms, step_ms_list, temps, drifts, peaks = [], [], [], [], [], [], []

    for k in range(1, n_steps + 1):
        ctx.check()
        t_step = time.perf_counter()
        md.run(1)
        ep = atoms.get_potential_energy()
        sync()
        ms = (time.perf_counter() - t_step) * 1e3
        ek = atoms.get_kinetic_energy()
        T = atoms.get_temperature()
        drift_mev = (ep + ek - e0) / n * 1000.0
        at_ms = n / max(ms, 1e-6)
        at_s = at_ms * 1000.0
        pk = _used_gb(dev)

        steps_x.append(k)
        step_ms_list.append(ms)
        speed_ms.append(at_ms)
        speed_s.append(at_s)
        temps.append(T)
        drifts.append(drift_mev)
        peaks.append(pk)

        vis = vis_of(atoms)
        frames.append(vis)
        ctx.series("bench_live_speed", "throughput", k, round(at_s, 0))
        ctx.series("bench_live_temp", "temperature", round(k * dt_fs, 1), round(T, 1))
        ctx.series("bench_live_energy", "energy drift", round(k * dt_fs, 1), round(drift_mev, 4))
        ctx.frame(vis, f"{unit_formula} · {n:,} atoms · step {k}/{n_steps} · {at_s:,.0f} atoms/s ({ms:.1f} ms)")
        ctx.progress(
            k,
            n_steps,
            f"Step {k}/{n_steps}: {at_s:,.0f} atoms/s ({at_ms:.1f} atoms/ms, {ms:.1f} ms/step) · {pk:.2f} GB · {T:.0f} K",
            {
                "atoms": f"{n:,}",
                "atoms/s": f"{at_s:,.0f}",
                "atoms/ms": f"{at_ms:.1f}",
                "ms/step": f"{ms:.1f} ms",
                "peak": f"{pk:.2f} GB",
            },
            force=(k == 1 or k % 2 == 0 or k == n_steps),
        )

    med_ms = float(np.median(step_ms_list))
    med_at_ms = n / med_ms
    med_at_s = med_at_ms * 1000.0
    peak_gb = float(max(peaks)) if peaks else base_gb
    added_gb = max(peak_gb - base_gb, 0.05)
    mb_per_atom = added_gb * 1000.0 / n
    max_drift = float(np.max(np.abs(drifts))) if drifts else 0.0
    ns_day = 86400.0 / (med_ms / 1000.0) * (dt_fs * 1e-6)
    drift_hint = f"Constant-energy stability ({core_T:.0f} K core droplet)" if is_mol else f"Constant-energy stability with {core_T:.0f} K hotspot"
    temp_title = f"Temperature ({core_T:.0f} K core droplet equilibration)" if is_mol else f"Temperature ({core_T:.0f} K hotspot equilibration)"
    size_key = "Molecular cluster" if is_mol else "Supercell"
    size_val = f"{rep[0]}×{rep[1]}×{rep[2]} grid ({n // n0:,} molecules, {n:,} atoms)" if is_mol else f"{rep[0]}×{rep[1]}×{rep[2]} ({n:,} atoms)"
    tbl_note = (
        f"Measured live during Velocity-Verlet NVE dynamics on a 3D molecular droplet ({core_T:.0f} K core / {outer_T:.0f} K shell)."
        if is_mol
        else f"Measured live during Velocity-Verlet NVE dynamics with a central {core_T:.0f} K thermal hotspot."
    )

    tiles = [
        R.tile("atoms_s", "Throughput (atoms/s)", med_at_s, "atoms/s", 0, better="higher", hint=f"median across {n_steps} NVE steps on {n:,} atoms"),
        R.tile("best_speed", "Throughput (atoms/ms)", med_at_ms, "atoms/ms", 2, better="higher", hint=f"{med_ms:.1f} ms per step on {dev.upper()}"),
        R.tile("ms_step", "Time per step", med_ms, "ms", 1, better="lower", hint=f"first call (warm-up): {cold_ms:.0f} ms"),
        R.tile("ns_day", "MD speed", ns_day, "ns/day", 2, better="higher", hint=f"with {dt_fs:g} fs time step"),
        R.tile("peak", "Peak memory", peak_gb, "GB", 2, better="lower", hint=f"+{mb_per_atom:.2f} MB/atom · {base_gb:.2f} GB model/runtime base"),
        R.tile("drift", "NVE energy drift", max_drift, "meV/atom", 4, status="pass" if max_drift < 1.0 else "warn", better="lower", hint=drift_hint),
    ]
    cols = ["step", "time (fs)", "ms/step", "atoms/ms", "atoms/s", "temperature (K)", "energy drift (meV/atom)", "process GB"]
    table = [
        [k, round(k * dt_fs, 1), round(ms, 2), round(ams, 2), round(ats, 0), round(t, 1), round(dr, 4), round(pk, 2)]
        for k, ms, ams, ats, t, dr, pk in zip(steps_x, step_ms_list, speed_ms, speed_s, temps, drifts, peaks)
    ]
    t_fs = [round(k * dt_fs, 1) for k in steps_x]
    charts = [
        R.chart("bench_live_speed", "Live throughput", [{"name": "throughput", "x": steps_x, "y": speed_s}], "line", "step", "atoms/s", markers=True),
        R.chart("bench_live_temp", temp_title, [{"name": "temperature", "x": t_fs, "y": temps}], "line", "time (fs)", "K"),
        R.chart("bench_live_energy", "NVE total energy drift", [{"name": "energy drift", "x": t_fs, "y": drifts}], "line", "time (fs)", "meV/atom"),
    ]
    files = [write_trajectory(ctx, frames)]
    vis_ref = frames[0]
    return {
        "headline": f"{unit_formula} ({n:,} atoms, {edges:,} neighbour pairs on {dev.upper()}): {med_at_s:,.0f} atoms/s ({med_at_ms:.1f} atoms/ms, {med_ms:.1f} ms/step), {peak_gb:.2f} GB peak memory.",
        "tiles": tiles,
        "facts": {
            "Model": ctx.model.get("title"),
            "Device": dev.upper(),
            size_key: size_val,
            "Neighbour pairs": f"{edges:,} ({edges / n:.0f} per atom)",
            "Neighbour skin": "0.6 Å",
            "Precision": "float32",
        },
        "tables": [R.table("Every MD step", cols, table, tbl_note)],
        "charts": charts,
        "viewer": R.viewer([R.frame(fr, None, f"{i * dt_fs:.1f} fs · {n:,} atoms") for i, fr in enumerate(frames)], vis_ref),
        "files": files,
        "notes": [f"Visual 3D trajectory shows {len(vis_ref):,} atoms so playback stays smooth at 60 FPS."] if len(vis_ref) < n else [],
        "raw": {"n": n, "edges": edges, "median_ms": med_ms, "atoms_per_ms": med_at_ms, "atoms_per_s": med_at_s, "peak_gb": peak_gb},
    }


def run(ctx):
    p, d = ctx.params, ctx.decision
    unit = prepare_atoms(ctx)
    if p.get("mode") == "visual_md":
        return _run_visual_md(ctx, unit)
    n0 = len(unit)
    reps = ladder(n0, p["max_atoms"])
    if not reps:
        raise UserError("The largest system is smaller than one cell.")
    unit_formula = structures.analyze(unit)["formula"]
    modes = [("standard", False)] + ([("low-memory", True)] if p["compare_low_memory"] and ctx.model.get("low_memory") else [])
    rows, best, total = [], {}, len(reps) * len(modes)
    vis_frames = []
    stopped = None
    done = 0
    for mode, low in modes:
        last_gb, last_n = None, None
        for r in reps:
            ctx.check()
            n = n0 * r[0] * r[1] * r[2]
            free = system.available_memory_bytes() / 1e9
            if last_gb and free and last_gb * (n / last_n) > free - min(RESERVE_GB, 0.35 * free):
                stopped = f"Stopped {mode} at {last_n:,} atoms: the next size would need about {last_gb * n / last_n:.1f} GB of the {free:.1f} GB free."
                break
            is_mol = not any(unit.pbc)
            grid_lbl = "grid" if is_mol else "cells"
            vis_cell = structures.tile_system(unit, r if n <= MAX_VIS_ATOMS else _best_repeat(n0, 1000))
            ctx.frame(vis_cell, f"{unit_formula} · {mode}: {n:,} atoms ({r[0]}×{r[1]}×{r[2]} {grid_lbl})")
            ctx.status(f"{mode}: {n:,} atoms ({r[0]}×{r[1]}×{r[2]} {grid_lbl}) on {d['device']}")
            ctx.flush()
            req = {"kind": "bench", "model": ctx.model, "device": d["device"], "structure": ctx.spec["structure"], "repeat": list(r),
                   "low_memory": low, "steps": p["steps"]}
            res = runner.isolated_python(ctx.model, ["-m", "mena_app.calibrate"], json.dumps(req), timeout=1800)
            done += 1
            try:
                pt = json.loads(res.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                err = (res.stderr.strip().splitlines() or ["no output"])[-1]
                stopped = f"{mode} at {n:,} atoms did not finish ({'out of memory' if 'memory' in err.lower() else err[:120]})."
                break
            pt["mode"] = mode
            pt["atoms_per_s"] = pt["atoms_per_ms"] * 1000.0
            rows.append(pt)
            last_gb, last_n = pt["peak_gb"], pt["n"]
            if mode == "standard":
                rng = np.random.default_rng(done)
                disp = vis_cell.copy()
                disp.positions = disp.positions + rng.normal(0, 0.045, disp.positions.shape)
                lbl = f"{unit_formula} · {pt['n']:,} atoms · {pt['atoms_per_s']:,.0f} atoms/s ({pt['peak_gb']:.2f} GB)"
                vis_frames.append((disp, lbl))
                ctx.frame(disp, lbl)
            ctx.series("bench_speed", mode, pt["n"], pt["atoms_per_ms"])
            ctx.series("bench_memory", mode, pt["n"], pt["peak_gb"])
            ctx.series("bench_ms", mode, pt["n"], pt["ms"])
            ctx.flush()
            ctx.progress(
                done,
                total,
                f"{mode}: {pt['n']:,} atoms, {pt['atoms_per_s']:,.0f} atoms/s ({pt['atoms_per_ms']:.1f} atoms/ms, {pt['ms']:.0f} ms/step), {pt['peak_gb']:.2f} GB",
                {
                    "atoms": f"{pt['n']:,}",
                    "atoms/s": f"{pt['atoms_per_s']:,.0f}",
                    "atoms/ms": f"{pt['atoms_per_ms']:.1f}",
                    "ms/step": f"{pt['ms']:.0f} ms",
                    "peak": f"{pt['peak_gb']:.2f} GB",
                },
                force=True,
            )
            if mode not in best or pt["n"] > best[mode]["n"]:
                best[mode] = pt
        if ctx.cancelled:
            raise Cancelled()
    if not rows:
        raise UserError(stopped or "No size finished.")
    top = best["standard"]
    fast = max((r for r in rows if r["mode"] == "standard"), key=lambda r: r["atoms_per_ms"])
    tiles = [R.tile("largest", "Largest system", top["n"], "atoms", 0),
             R.tile("atoms_s", "Best throughput (atoms/s)", fast["atoms_per_ms"] * 1000.0, "atoms/s", 0, better="higher", hint=f"at {fast['n']:,} atoms"),
             R.tile("best_speed", "Best speed (atoms/ms)", fast["atoms_per_ms"], "atoms/ms", 1, better="higher", hint=f"at {fast['n']:,} atoms"),
             R.tile("ns_day", f"MD speed at {top['n']:,} atoms", 86400 / (top["ms"] / 1000) * 1e-6, "ns/day", 3, hint="with a 1 fs time step"),
             R.tile("peak", f"Peak memory at {top['n']:,} atoms", top["peak_gb"], "GB", 2, better="lower", hint="whole process: model, Python and the device"),
             R.tile("mb_atom", "Memory per atom", top["added_gb"] * 1000 / top["n"], "MB/atom", 3, better="lower",
                    hint=f"what each atom adds, above the fixed {top['base_gb']:.2f} GB of PyTorch, the device and the model")]
    if "low-memory" in best:
        lm = best["low-memory"]
        tiles.append(R.tile("low_peak", f"Low-memory mode at {lm['n']:,} atoms", lm["peak_gb"], "GB", 2, better="lower",
                            hint=f"{lm['atoms_per_ms'] * 1000:,.0f} atoms/s ({lm['atoms_per_ms']:.1f} atoms/ms)"))
    cols = ["mode", "atoms", "neighbour pairs", "ms per step", "atoms/ms", "atoms/s", "peak GB (process)", "added GB", "MB/atom added", "first call (ms)"]
    table = [[r["mode"], r["n"], r["edges"], round(r["ms"], 1), round(r["atoms_per_ms"], 2), round(r["atoms_per_ms"] * 1000, 0),
              round(r["peak_gb"], 2), round(r["added_gb"], 2), round(r["added_gb"] * 1000 / r["n"], 3), round(r["cold_ms"], 0)] for r in rows]
    series = lambda key: [{"name": m, "x": [r["n"] for r in rows if r["mode"] == m], "y": [r[key] for r in rows if r["mode"] == m]} for m, _ in modes]
    big = structures.tile_system(structures.load(ctx.spec["structure"]), tuple(max((t for t in reps if n0 * math.prod(t) <= top["n"]), key=math.prod)))
    vis_ref = big if len(big) <= MAX_VIS_ATOMS else (vis_frames[-1][0] if vis_frames else unit)
    return {
        "headline": f"{unit_formula}: up to {top['n']:,} atoms on {d['device'].upper()}, {fast['atoms_per_ms'] * 1000:,.0f} atoms/s ({fast['atoms_per_ms']:.1f} atoms/ms) at best, "
                    f"{top['peak_gb']:.2f} GB at the largest size.",
        "tiles": tiles, "facts": {"Model": ctx.model.get("title"), "Device": d["device"].upper(), "Neighbour skin": "0.6 Å", "Precision": "float32"},
        "tables": [R.table("Every size", cols, table, "Each size ran in a fresh process. Peak is the whole process; added is the peak minus the process just before the first evaluation. "
                           "GPU memory is counted in allocator blocks, so small sizes step up unevenly.")],
        "charts": [R.chart("bench_speed", "Speed", series("atoms_per_ms"), "line", "atoms", "atoms per ms", logx=True, markers=True),
                   R.chart("bench_memory", "Peak memory", series("peak_gb"), "line", "atoms", "GB", logx=True, markers=True),
                   R.chart("bench_ms", "Time per step", series("ms"), "line", "atoms", "ms", logx=True, logy=True, markers=True)],
        "viewer": R.viewer([R.frame(vis_ref, None, f"{top['n']:,} atoms · {fast['atoms_per_ms'] * 1000:,.0f} atoms/s")], vis_ref),
        "files": [], "notes": [stopped] if stopped else [], "raw": {"rows": rows},
    }
