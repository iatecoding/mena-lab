"""Helpers shared by the tasks: build the structure and calculator, run an optimiser with live output, write files."""

from __future__ import annotations

import time

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import FIRE, LBFGS

from mena_app import backends, report as R, structures
from mena_app.catalog import LOT_LABELS  # noqa: F401  (re-exported for the tasks)
from mena_app.runtime import Cancelled, Context, UserError

OPTIMIZERS = {"LBFGS": LBFGS, "FIRE": FIRE}


def thermalize(atoms, temperature_K: float, rng) -> None:
    """Maxwell-Boltzmann momenta: `thermalize_momenta` on ASE >= 3.29, the older name before it."""
    from ase.md import velocitydistribution as vd

    if hasattr(vd, "thermalize_momenta"):
        vd.thermalize_momenta(atoms, temperature_K=temperature_K, rng=rng)
    else:
        vd.MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)


def prepare_atoms(ctx: Context, which: str = "structure"):
    try:
        atoms = structures.load(ctx.spec[which])
    except structures.StructureError as exc:
        raise UserError(str(exc)) from None
    if structures.domain_of(atoms) == "molecule":
        atoms.info["charge"] = int(ctx.system["charge"] if ctx.system.get("charge") is not None else 0)
        atoms.info["spin"] = int(ctx.system["spin"] if ctx.system.get("spin") is not None else 1)
    return atoms


def make_calc(ctx: Context):
    d = ctx.decision
    ctx.status(f"loading {ctx.model.get('title', ctx.model['id'])}")
    t = time.perf_counter()
    calc = backends.build(ctx.model, d)
    ctx.log(f"model {ctx.model['id']} loaded in {time.perf_counter() - t:.2f} s on {d['device']}, {d['dtype']}"
            + (", low-memory" if d["low_memory"] else "") + (f", neighbour skin {d['skin']} Å" if d["skin"] else "") + (", compiled" if d.get("compile") else ""))
    if getattr(calc, "compile_status", None) == "unchecked":
        announce_compile(ctx, calc)
    return calc


def announce_compile(ctx: Context, calc) -> None:
    """Say what the first model call is doing when it compiles (5-30 s, once per run), then how it went."""
    def first_call(*args, **kwargs):
        del calc.calculate  # one shot: later calls go straight to the class method
        ctx.status("compiling the model for this computer: about 15 s the first time, about 5 s on later runs")
        t = time.perf_counter()
        out = calc.calculate(*args, **kwargs)
        dt = time.perf_counter() - t
        if calc.compile_status == "on":
            ctx.status(f"compiled in {dt:.0f} s and checked against the uncompiled model: same energies and forces")
        else:
            ctx.status("running uncompiled: compilation did not pass its check on this computer (results are unaffected)")
            ctx.log(f"compile check: {calc.compile_status}")
        return out

    calc.calculate = first_call


def setup(ctx: Context, which: str = "structure"):
    atoms = prepare_atoms(ctx, which)
    atoms.calc = make_calc(ctx)
    return atoms


def lot_used(calc, atoms) -> str | None:
    """The level of theory the calculator actually used for this structure (the auto choice resolved)."""
    res = getattr(calc, "results", None)
    if isinstance(res, dict) and res.get("lot_used"):
        return str(res["lot_used"])
    try:
        if hasattr(calc, "_resolve_lot_and_u"):
            try:
                lot_name, _ = calc._resolve_lot_and_u(atoms)
                if isinstance(lot_name, str):
                    return lot_name
            except TypeError:
                from lam.models.mena_lam import LOT_ID_TO_NAME
                lid, _, _ = calc._resolve_lot_and_u(atoms.numbers, bool(atoms.pbc.any()), atoms.info)
                return LOT_ID_TO_NAME[lid]
    except Exception:  # noqa: BLE001 - a nicety for the report
        pass
    return "omol_wb97mv" if structures.domain_of(atoms) == "molecule" else None




def fmax_of(forces) -> float:
    return float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0


def make_observer(ctx: Context, atoms, state: dict, target=None, opt=None, steps: int = 1, chart: str = "opt", label: str = "relax"):
    def observe():
        fm = fmax_of((target if target is not None else atoms).get_forces())
        e = atoms.get_potential_energy()
        step = opt.nsteps if opt is not None else len(state["energies"])
        state["energies"].append(e)
        state["fmaxs"].append(fm)
        state["frames"].append(atoms.copy())
        if "phys" in state:
            state["phys"].append(R.snapshot_physics(atoms))
        ctx.series(f"{chart}_energy", "energy", step, e - state["energies"][0])
        ctx.series(f"{chart}_fmax", "largest force", step, fm)
        ctx.frame(atoms, f"{label} step {step}")
        ctx.progress(min(step, steps), steps, f"{label}: step {step}, largest force {fm:.4f} eV/Å", {"step": step, "largest force": f"{fm:.4f} eV/Å"})
        if ctx.cancelled:
            raise Cancelled()

    return observe


def optimize(ctx: Context, atoms, method: str, fmax: float, steps: int, cell: bool = False, chart: str = "opt", label: str = "relax"):
    """Runs an ASE optimiser with live charts. Returns dict(converged, steps, energies, fmaxs, frames, phys, cancelled, seconds)."""
    target = FrechetCellFilter(atoms) if cell else atoms
    opt = OPTIMIZERS[method](target, logfile=None)
    rec = {"energies": [], "fmaxs": [], "frames": [], "phys": [], "cancelled": False}
    t0 = time.perf_counter()
    opt.attach(make_observer(ctx, atoms, rec, target=target, opt=opt, steps=steps, chart=chart, label=label), interval=1)
    converged = False
    try:
        converged = bool(opt.run(fmax=fmax, steps=steps))
    except Cancelled:
        rec["cancelled"] = True
    ctx.flush()
    return {**rec, "converged": converged, "steps": opt.nsteps, "seconds": time.perf_counter() - t0}


def write_structure(ctx: Context, atoms, stem: str, comment: str = "") -> dict:
    """CIF for periodic structures (opens in VESTA, Materials Studio), XYZ for molecules (Avogadro, GaussView)."""
    a = atoms.copy()
    a.calc = None
    a.info.pop("_mena_phys", None)
    if a.pbc.any() and a.cell.rank == 3:
        write(ctx.path(f"{stem}.cif"), a, format="cif")
        return {"name": f"{stem}.cif", "label": "Structure (CIF)"}
    if a.pbc.any():
        write(ctx.path(f"{stem}.extxyz"), a, format="extxyz")
        return {"name": f"{stem}.extxyz", "label": "Structure (extended XYZ)"}
    write(ctx.path(f"{stem}.xyz"), a, format="xyz", comment=comment)
    return {"name": f"{stem}.xyz", "label": "Structure (XYZ)"}


def write_trajectory(ctx: Context, frames, name: str = "trajectory.extxyz", label: str = "Trajectory (extended XYZ)") -> dict:
    out = []
    for f in frames:
        g = f.copy()
        g.calc = None
        g.info.pop("_mena_phys", None)
        out.append(g)
    write(ctx.path(name), out, format="extxyz")
    return {"name": name, "label": label}
