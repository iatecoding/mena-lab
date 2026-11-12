"""Phonons of a crystal by finite displacements in a supercell (ASE Phonons): bands, density of states, free energy."""

from __future__ import annotations

import shutil

import numpy as np
from ase import units
from ase.phonons import Phonons

from mena_app import report as R
from mena_app import phonon_symmetry, structures
from mena_app.routing import phonon_supercell
from mena_app.runtime import Cancelled
from mena_app.tasks.common import fmax_of, optimize, setup, write_structure

EV_TO_THZ = units._e / units._hplanck / 1e12
EV_TO_CM = 1.0 / units.invcm
trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2 has only trapz


class _LivePhonons(Phonons):
    """Reports progress and honours cancel between displaced supercells. With a symmetry plan only the irreducible
    displacements are evaluated (`measure`), and ASE's loop then reads every displacement from the rebuilt table."""

    def __init__(self, ctx, total, *a, plan=None, **kw):
        super().__init__(*a, **kw)
        self.ctx, self.total, self.done = ctx, total, 0
        self.plan, self._table = plan, None

    def _tick(self) -> None:
        self.done += 1
        self.ctx.progress(self.done, self.total, f"displaced supercell {self.done} of {self.total}")

    def measure(self) -> None:
        """Evaluate the irreducible displacements and rebuild the force of every displacement ASE would make."""
        plan, n = self.plan, len(self.atoms)
        sc = self.atoms * self.supercell
        sc.calc = self.calc
        self.ctx.check()
        f0 = sc.get_forces().copy()
        self._tick()
        p0 = sc.positions.copy()
        rep = {}
        for b in plan.reps:
            meas = []
            for d in plan.directions[b]:
                fs = []
                for sign in (1, -1):
                    self.ctx.check()
                    sc.positions = p0.copy()
                    sc.positions[plan.offset * n + b] += sign * self.delta * d
                    fs.append(sc.get_forces().copy())
                    self._tick()
                meas.append((fs[0] - fs[1]) / (2 * self.delta))
            rep[b] = phonon_symmetry.rep_response(plan, b, np.array(meas))
        sc.positions = p0
        self._table = (f0, phonon_symmetry.all_responses(plan, rep))

    def calculate(self, atoms_N, disp):
        if self._table is not None:  # rebuilt from the irreducible displacements: no model call
            f0, M = self._table
            if disp.sign == 0:
                return {"forces": f0.copy()}
            return {"forces": f0 + disp.sign * self.delta * M[disp.a, :, :, disp.i]}
        self.ctx.check()
        out = super().calculate(atoms_N, disp)
        self._tick()
        return out


def free_energy(e: np.ndarray, w: np.ndarray, T: float, n_modes: int) -> dict:
    """Harmonic vibrational free energy, entropy and heat capacity per cell from a phonon DOS sampled on energies e (eV)."""
    m = e > 1e-5
    e, w = e[m], w[m]
    w = w / trapezoid(w, e) * n_modes
    kT = units.kB * T
    x = e / kT
    f = trapezoid(w * (0.5 * e + kT * np.log1p(-np.exp(-x))), e)
    s = trapezoid(w * units.kB * (x / np.expm1(x) - np.log1p(-np.exp(-x))), e)
    cv = trapezoid(w * units.kB * x ** 2 * np.exp(x) / np.expm1(x) ** 2, e)
    return {"f": f, "s": s, "cv": cv, "zpe": trapezoid(w * 0.5 * e, e)}


def run(ctx):
    p = ctx.params
    atoms = setup(ctx)
    calc = atoms.calc
    notes, files = [], []
    if p["prerelax"]:
        ctx.status("relaxing atoms and cell first")
        r = optimize(ctx, atoms, "LBFGS", p["fmax"], 1000, cell=True, label="pre-relax")
        if r["cancelled"]:
            raise Cancelled()
        if not r["converged"]:
            notes.append("The relaxation did not fully converge, so small imaginary frequencies may be artefacts.")
        files.append(write_structure(ctx, atoms, "relaxed"))
    f_res = fmax_of(atoms.get_forces())
    k = phonon_supercell(ctx.spec, structures.analyze(atoms))
    n = len(atoms)
    work = ctx.path("phonon_cache")
    shutil.rmtree(work, ignore_errors=True)
    prim = atoms.copy()
    prim.calc = None
    plan = phonon_symmetry.symmetry_plan(prim, k)
    full = 6 * n + 1
    total = plan.n_evaluations if plan is not None else full
    ph = _LivePhonons(ctx, total, prim, calc, supercell=(k, k, k), delta=p["displacement"], name=work, plan=plan)
    if plan is not None and ph.offset != plan.offset:  # the plan assumes ASE's reference cell; never guess
        plan, total = None, full
        ph.plan, ph.total = None, full
    if plan is not None:
        ctx.status(f"{k}×{k}×{k} supercell, {n * k ** 3} atoms: {total} evaluations instead of {full}, using the crystal's symmetry")
        ph.measure()
    else:
        ctx.status(f"{k}×{k}×{k} supercell, {n * k ** 3} atoms, {total} evaluations")
    ph.run()
    ph.read(acoustic=True)
    path = prim.cell.bandpath(npoints=220)
    bs = ph.get_band_structure(path, verbose=False)
    bands = bs.energies[0] * EV_TO_THZ  # (nk, 3n)
    x, xs, labels = path.get_linear_kpoint_axis()
    dos = ph.get_dos(kpts=(p["kpoints"],) * 3, verbose=False).sample_grid(npts=600, width=2e-4)
    de, dw = dos.get_energies(), dos.get_weights()
    th = free_energy(de, dw, p["temperature"], 3 * n)
    fmin = float(bands.min())
    unstable = fmin < -0.1
    series = [{"name": f"band {b + 1}", "x": x.tolist(), "y": bands[:, b].round(4).tolist()} for b in range(bands.shape[1])]
    ticks = [{"x": float(v), "label": ("Γ" if lab == "G" else lab)} for v, lab in zip(xs, labels)]
    np.savetxt(ctx.path("bands_THz.csv"), np.column_stack([x, bands]), delimiter=",", header="k_distance," + ",".join(f"band{b + 1}" for b in range(bands.shape[1])), comments="")
    np.savetxt(ctx.path("dos.csv"), np.column_stack([de * EV_TO_THZ, dw]), delimiter=",", header="frequency_THz,dos", comments="")
    files += [{"name": "bands_THz.csv", "label": "Band structure (CSV)"}, {"name": "dos.csv", "label": "Density of states (CSV)"}]
    shutil.rmtree(work, ignore_errors=True)
    tiles = [R.tile("stable", "Dynamically stable", "no" if unstable else "yes", status="warn" if unstable else "pass",
                    hint="No imaginary (negative) frequencies anywhere in the Brillouin zone."),
             R.tile("fmin", "Lowest frequency", fmin, "THz", 3), R.tile("fmax", "Highest frequency", float(bands.max()), "THz", 2),
             R.tile("zpe", "Zero-point energy", th["zpe"] * 1000 / n, "meV/atom", 2),
             R.tile("fvib", f"Vibrational free energy at {p['temperature']:.0f} K", th["f"] * 1000 / n, "meV/atom", 2),
             R.tile("cv", f"Heat capacity at {p['temperature']:.0f} K", th["cv"] / units.kB / n, "k_B/atom", 3, hint="3 k_B/atom is the classical limit.")]
    if unstable:
        notes.append("Imaginary frequencies: the structure may be unstable, or the supercell too small or the relaxation too loose. Check before concluding.")
    width = k * min(structures.analyze(prim)["widths"])
    if k % 2 == 0:
        notes.append("Even supercells can give small spurious imaginary frequencies near Γ with this method; an odd size (3, 5) is safer.")
    if width < 9:
        notes.append(f"The supercell is only {width:.1f} Å across in its shortest direction; 9-10 Å or more is safer.")
    return {
        "headline": f"{'Dynamically stable' if not unstable else 'Imaginary modes found'}: frequencies from {fmin:.2f} to {bands.max():.2f} THz on a {k}×{k}×{k} supercell.",
        "tiles": tiles, "facts": {"Supercell atoms": n * k ** 3, "Evaluations": total if plan is None else f"{total} (symmetry; {full} without)", "Residual force": f"{f_res:.4f} eV/Å", "k-path": path.path},
        "tables": [],
        "charts": [R.chart("bands", "Phonon band structure", series, "line", "", "frequency (THz)", xticks=ticks, legend=False),
                   R.chart("dos", "Phonon density of states", [{"name": "DOS", "x": (de * EV_TO_THZ).round(4).tolist(), "y": dw.round(4).tolist()}], "line", "frequency (THz)", "states")],
        "viewer": R.viewer([R.frame(atoms, None, "relaxed" if p["prerelax"] else "input")], atoms),
        "files": files, "notes": notes, "raw": {"band_min_THz": fmin, "thermo": th},
    }
