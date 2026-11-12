"""Reaction barrier with the nudged elastic band method (climbing image), between two structures with the same atoms."""

from __future__ import annotations

import numpy as np
from ase.mep import NEB
from ase.optimize import FIRE

from mena_app import backends, report as R
from mena_app.runtime import Cancelled
from mena_app.tasks.common import fmax_of, optimize, prepare_atoms, setup, write_trajectory

_DECOMP_TERMS = (
    ("network", "ΔNetwork", "Many-body network"),
    ("electrostatic", "ΔElectrostatics", "Long-range electrostatics"),
    ("polarization", "ΔPolarization", "Electronic polarization"),
    ("dispersion_d3", "ΔDispersion D3", "London dispersion (DFT-D3)"),
    ("repulsion", "ΔCore repulsion", "Short-range core repulsion"),
)


class _CachedNEB(NEB):
    last_forces: np.ndarray | None = None

    def get_forces(self):
        f = super().get_forces()
        self.last_forces = f
        return f


def _cached_energy(im) -> float:
    res = getattr(im.calc, "results", None) or {}
    return float(res["energy"]) if "energy" in res else float(im.get_potential_energy())


def _ensure_results(im) -> dict:
    res = getattr(im.calc, "results", None)
    if not isinstance(res, dict) or "energy" not in res:
        if "forces" in getattr(im.calc, "implemented_properties", ()):
            im.get_forces()
        im.get_potential_energy()
        res = getattr(im.calc, "results", None)
    return res if isinstance(res, dict) else {}


def _decomp_along_path(results_list: list[dict], dist_list: list[float], top: int):
    decs = [r.get("energy_decomposition") for r in results_list]
    if not decs or not all(isinstance(d, dict) and "kcal_mol" in d for d in decs):
        return None, None, None
    n_im = len(decs)
    tot_vals = [float(decs[i]["kcal_mol"].get("total", 0.0)) for i in range(n_im)]
    dtot = [v - tot_vals[0] for v in tot_vals]
    series = []
    if max(dtot) - min(dtot) > 0.02:
        series.append({"name": "ΔTotal", "x": dist_list, "y": [round(v, 3) for v in dtot]})
    rows = []
    raw_dec = {"total": [round(v, 4) for v in dtot]}
    for key, sname, tlabel in _DECOMP_TERMS:
        vals = [float(decs[i]["kcal_mol"].get(key, 0.0)) for i in range(n_im)]
        dvals = [v - vals[0] for v in vals]
        raw_dec[key] = [round(v, 4) for v in dvals]
        span = max(dvals) - min(dvals)
        if span > 0.02:
            series.append({"name": sname, "x": dist_list, "y": [round(v, 3) for v in dvals]})
        if span > 1e-4:
            rows.append([tlabel, round(vals[top] - vals[0], 3), round(vals[top] - vals[-1], 3), round(vals[-1] - vals[0], 3)])
    rows.append(["Total (ΔE)", round(tot_vals[top] - tot_vals[0], 3), round(tot_vals[top] - tot_vals[-1], 3), round(tot_vals[-1] - tot_vals[0], 3)])
    ch = (R.chart("neb_decomp", "Barrier physical decomposition (kcal/mol)", series, "line", "path length (Å)", "Δenergy (kcal/mol)", markers=True)
          if series else None)
    tbl = R.table(
        "Barrier physical decomposition",
        ["Component", "TS − Reactant (kcal/mol)", "TS − Product (kcal/mol)", "Product − Reactant (kcal/mol)"],
        rows,
        f"Transition state is image {top} of {n_im - 1} (path length {dist_list[top]:.3f} Å). Positive values raise the barrier relative to reactant or product.",
    )
    return ch, tbl, raw_dec


def run(ctx):
    p = ctx.params
    a = setup(ctx)
    calc = a.calc
    b = prepare_atoms(ctx, "structure2")
    b.calc = backends.clone_calculator(calc)
    if p["relax_ends"]:
        for name, at in (("start", a), ("end", b)):
            ctx.status(f"relaxing the {name}")
            r = optimize(ctx, at, "LBFGS", min(p["fmax"], 0.02), 500, chart=f"end_{name}", label=f"relax {name}")
            if r["cancelled"]:
                raise Cancelled()
    e0 = float(_ensure_results(a)["energy"])
    e_end = float(_ensure_results(b)["energy"])
    images = [a] + [a.copy() for _ in range(p["images"])] + [b]
    for im in images[1:-1]:
        c_im = backends.clone_calculator(calc)
        if hasattr(c_im, "skin"):
            c_im.skin = 0.0
        im.calc = c_im
    neb = _CachedNEB(images, climb=p["climb"], method="improvedtangent")
    ctx.status("building the starting path")
    neb.interpolate("idpp")
    opt = FIRE(neb, logfile=None)
    state = {"cancelled": False}

    def observe():
        es = [0.0, *(_cached_energy(im) - e0 for im in images[1:-1]), e_end - e0]
        f_neb = neb.last_forces if neb.last_forces is not None else neb.get_forces()
        fm = fmax_of(f_neb.reshape(-1, 3))
        ctx.series("neb_barrier", "barrier", opt.nsteps, max(es))
        ctx.series("neb_fmax", "largest force", opt.nsteps, fm)
        ctx.progress(min(opt.nsteps, p["max_steps"]), p["max_steps"], f"step {opt.nsteps}: barrier {max(es):.3f} eV, largest force {fm:.3f} eV/Å")
        if ctx.cancelled:
            raise Cancelled()

    opt.attach(observe, interval=1)
    ctx.status("optimising the path")
    converged = False
    try:
        converged = bool(opt.run(fmax=p["fmax"], steps=p["max_steps"]))
    except Cancelled:
        state["cancelled"] = True
    ctx.flush()
    results_list = [_ensure_results(im) for im in images]
    es = np.array([float(res["energy"]) for res in results_list])
    rel = es - es[0]
    top = int(np.argmax(rel))
    dist = np.concatenate([[0.0], np.cumsum([np.linalg.norm(images[i + 1].positions - images[i].positions) for i in range(len(images) - 1)])])
    dist_list = dist.round(4).tolist()
    fwd, rev = float(rel[top]), float(rel[top] - rel[-1])
    tiles = [R.tile("fwd", "Barrier (forward)", fwd, "eV", 3, hint=f"{fwd * 23.0605:.2f} kcal/mol, {fwd * 96.485:.1f} kJ/mol"),
             R.tile("rev", "Barrier (reverse)", rev, "eV", 3, hint=f"{rev * 23.0605:.2f} kcal/mol"),
             R.tile("de", "Reaction energy", float(rel[-1]), "eV", 3, hint="End minus start."),
             R.tile("conv", "Converged", "yes" if converged else "no", status="pass" if converged else "warn"),
             R.tile("steps", "Steps", opt.nsteps, "", 0)]
    charts = [R.chart("profile", "Energy along the path", [{"name": "energy", "x": dist_list, "y": rel.round(5).tolist()}], "line", "path length (Å)", "energy (eV)", markers=True)]
    tables = []
    raw: dict = {"energies_eV": es, "path_A": dist}
    dec_ch, dec_tbl, raw_dec = _decomp_along_path(results_list, dist_list, top)
    if dec_ch is not None:
        charts.append(dec_ch)
    if dec_tbl is not None:
        tables.append(dec_tbl)
    if raw_dec is not None:
        raw["decomp_kcal_mol"] = raw_dec
    if not any(a.pbc):
        dips, dnorms = [], []
        for res in results_list:
            dv = res.get("dipole_debye")
            if dv is None and res.get("dipole") is not None:
                dv = np.asarray(res["dipole"], dtype=float).reshape(3) * R.E_ANG_TO_DEBYE
            if dv is not None:
                vec = np.asarray(dv, dtype=float).reshape(3)
                dips.append(vec.tolist())
                dn = float(res["dipole_norm_debye"]) if res.get("dipole_norm_debye") is not None else float(np.linalg.norm(vec))
                dnorms.append(dn)
        if len(dnorms) == len(images):
            raw["dipole_debye"] = dips
            raw["dipole_norm_debye"] = dnorms
            if max(dnorms) - min(dnorms) > 0.05:
                charts.append(R.chart("neb_dipole", "Dipole along path", [{"name": "|μ| (D)", "x": dist_list, "y": [round(v, 4) for v in dnorms]}], "line", "path length (Å)", "dipole (D)", markers=True))
    files = [write_trajectory(ctx, images, "path.extxyz", "Path images (extended XYZ)")]
    notes = [] if converged else ["The path did not converge within the step limit; the barrier is an estimate. Raise the steps and run again."]
    if 0 < top < len(images) - 1 and not p["climb"]:
        notes.append("Without the climbing image the top of the path lies slightly below the true saddle.")
    return {
        "headline": f"Forward barrier {fwd:.3f} eV ({fwd * 23.0605:.2f} kcal/mol), reaction energy {rel[-1]:+.3f} eV, over {len(images)} images.",
        "tiles": tiles, "facts": {"Highest image": f"{top} of {len(images) - 1}"}, "tables": tables,
        "charts": charts,
        "viewer": R.viewer([R.frame_with_physics(im, results_list[i], include_forces=False, label=f"image {i}: {rel[i]:+.3f} eV") for i, im in enumerate(images)], a),
        "files": files, "notes": notes, "cancelled": state["cancelled"], "raw": raw,
    }
