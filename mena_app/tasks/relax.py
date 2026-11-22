"""Relax atoms (and optionally the cell) to the nearest energy minimum."""

from __future__ import annotations

import numpy as np

from mena_app import report as R
from mena_app import structures
from mena_app.tasks.common import LOT_LABELS, lot_used, optimize, setup, write_structure, write_trajectory


def run(ctx):
    p = ctx.params
    atoms = setup(ctx)
    a0 = structures.analyze(atoms)
    cell = bool(p.get("relax_cell")) and bool(atoms.pbc.all())
    v0 = atoms.get_volume() if atoms.pbc.all() else None
    pos0 = atoms.positions.copy()
    ctx.status(f"relaxing with {p['optimizer']}{' (atoms and cell)' if cell else ''}")
    r = optimize(ctx, atoms, p["optimizer"], p["fmax"], p["max_steps"], cell=cell)
    if not r["energies"]:
        return {
            "headline": "Stopped before the first optimization step completed.",
            "tiles": [R.tile("converged", "Converged", "no", status="warn"),
                      R.tile("steps", "Steps", 0, "", 0),
                      R.tile("time", "Time", r["seconds"], "s", 1)],
            "facts": {}, "tables": [], "charts": [],
            "viewer": R.viewer([R.frame_with_physics(atoms, None, label="initial")], atoms),
            "files": [], "notes": ["Stopped by you before the first step finished."],
            "cancelled": True, "raw": {"energies_eV": [], "fmax_eV_A": [], "converged": False, "cell": atoms.cell.array if atoms.pbc.any() else None},
        }
    e0, e1 = r["energies"][0], r["energies"][-1]
    files = [write_structure(ctx, atoms, "relaxed", f"Mena relaxed, E = {e1:.6f} eV"), write_trajectory(ctx, r["frames"])]
    disp = float(np.sqrt(((atoms.positions - pos0) ** 2).sum(1).mean())) if not cell else None
    tiles = [R.tile("converged", "Converged", "yes" if r["converged"] else "no", status="pass" if r["converged"] else "warn"),
             R.tile("steps", "Steps", r["steps"], "", 0),
             R.tile("e_drop", "Energy lowered by", e0 - e1, "eV", 4),
             R.tile("e_final", "Final energy", e1, "eV", 5),
             R.tile("fmax", "Final largest force", r["fmaxs"][-1], "eV/Å", 4, better="lower"),
             R.tile("time", "Time", r["seconds"], "s", 1)]
    if v0:
        tiles.append(R.tile("dv", "Volume change", 100 * (atoms.get_volume() / v0 - 1), "%", 2))
    if disp is not None:
        tiles.append(R.tile("rms_disp", "Atoms moved (rms)", disp, "Å", 3))
    lot_s = lot_used(atoms.calc, atoms)
    facts = {"Level of theory": LOT_LABELS.get(lot_s, lot_s or "model default")}
    if atoms.pbc.all():
        c = atoms.cell.cellpar()
        facts["Cell after"] = "a={:.4f} b={:.4f} c={:.4f} Å, α={:.2f} β={:.2f} γ={:.2f}°".format(*c)
        facts["Cell before"] = "a={:.4f} b={:.4f} c={:.4f} Å, α={:.2f} β={:.2f} γ={:.2f}°".format(*a0["cell"])
    steps = list(range(len(r["energies"])))
    notes = []
    if r["cancelled"]:
        notes.append("Stopped by you: this is the structure at that moment.")
    elif not r["converged"]:
        notes.append("The step limit was reached first. Run again from the relaxed structure, or try the FIRE method.")
    idx = R.subsample(len(r["frames"]), 60)
    phys = r.get("phys") or [None] * len(r["frames"])
    n_fr = max(1, len(r["frames"]) - 1)
    step_scale = r["steps"] / n_fr if len(r["frames"]) > 1 else 1.0
    return {
        "headline": f"{'Converged' if r['converged'] else 'Not converged'} in {r['steps']} steps: energy lowered by {e0 - e1:.4f} eV, largest force {r['fmaxs'][-1]:.4f} eV/Å.",
        "tiles": tiles, "facts": facts, "tables": [],
        "charts": [R.chart("opt_energy", "Energy", [{"name": "energy", "x": steps, "y": [e - e0 for e in r["energies"]]}], "line", "step", "energy change (eV)"),
                   R.chart("opt_fmax", "Largest force", [{"name": "largest force", "x": steps, "y": r["fmaxs"]}], "line", "step", "eV/Å", logy=True)],
        "viewer": R.viewer([R.frame_with_physics(r["frames"][i], phys[i], label=f"step {int(round(i * step_scale))}") for i in idx], atoms),
        "files": files, "notes": notes, "cancelled": r["cancelled"],
        "raw": {"energies_eV": r["energies"], "fmax_eV_A": r["fmaxs"], "converged": r["converged"], "cell": atoms.cell.array if atoms.pbc.any() else None},
    }
