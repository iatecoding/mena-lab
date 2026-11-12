"""Energy, physical decomposition, forces, stress, dipole, polarizability and atomic properties for the structure as given."""

from __future__ import annotations

import time

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write
from ase.units import GPa

from mena_app import report as R
from mena_app import structures
from mena_app.tasks.common import LOT_LABELS, backends, fmax_of, lot_used, setup, write_structure

E_ANG_TO_DEBYE = 1.0 / 0.20819434


def _decomp_artifacts(dec: dict | None):
    if not isinstance(dec, dict) or "kcal_mol" not in dec or "eV" not in dec:
        return None, None
    km, ev = dec["kcal_mol"], dec["eV"]
    rows = [
        ["Many-body network (total)", round(km.get("network", 0.0), 3), round(ev.get("network", 0.0), 5),
         "Short- & medium-range covalent/metallic bonding (3-stage equivariant trunk)"],
    ]
    if any(abs(km.get(k, 0.0)) > 1e-6 for k in ("stage0", "stage1", "final", "iso")):
        for key, label in (("stage0", "  · Stage 0 readout"), ("stage1", "  · Stage 1 readout"),
                           ("final", "  · Stage 2 readout"), ("iso", "  · Isolated-atom network shift")):
            if abs(km.get(key, 0.0)) > 1e-6:
                rows.append([label, round(km[key], 3), round(ev[key], 5), ""])
    for key, title, role in (
        ("electrostatic", "Long-range electrostatics", "Screened Coulomb interaction of equilibrated NQE partial charges"),
        ("polarization", "Electronic polarization", "Self-consistent induced atomic polarization response"),
        ("dispersion_d3", "London dispersion (DFT-D3)", "Long-range van der Waals attraction"),
        ("repulsion", "Short-range core repulsion", "Nordlund–Lehtola–Hobler ZBL-like nuclear repulsion"),
        ("interaction", "Residual subtotal (E − E_ref)", "Sum of neural network, electrostatic, polarization, D3 & repulsion terms"),
        ("reference_e0", "Composition reference (E_ref)", "Per-element linear baseline at this level of theory"),
    ):
        if key in ("interaction", "reference_e0") or abs(km.get(key, 0.0)) > 1e-6:
            rows.append([title, round(km.get(key, 0.0), 3), round(ev.get(key, 0.0), 5), role])
    if abs(km.get("reference_charge_spin", 0.0)) > 1e-6:
        rows.append(["Charge / spin reference (E_q,S)", round(km["reference_charge_spin"], 3),
                     round(ev["reference_charge_spin"], 5), "Net molecular charge & spin state reference offset"])
    rows.append(["Total potential energy", round(km.get("total", 0.0), 3), round(ev.get("total", 0.0), 5),
                 "Full DFT target energy"])
    tbl = R.table("Physical energy decomposition", ["Component", "kcal/mol", "eV", "Physical role"], rows,
                  "E_ref is a per-element linear composition baseline (not vacuum isolated atoms); for atomization or reaction energies, subtract total energies of the reactants.")
    bar_terms = [
        ("Network", km.get("network", 0.0)),
        ("Coulomb", km.get("electrostatic", 0.0)),
        ("Polariz.", km.get("polarization", 0.0)),
        ("D3", km.get("dispersion_d3", 0.0)),
        ("Repulsion", km.get("repulsion", 0.0)),
    ]
    active = [(lbl, float(val)) for lbl, val in bar_terms if abs(val) > 1e-4]
    ch = None
    if active:
        ch = R.chart("energy_decomp", "Model residual components (E − E_ref)",
                     [{"name": "kcal/mol", "x": [k for k, _ in active], "y": [round(v, 3) for _, v in active]}],
                     "bar", "", "energy (kcal/mol)", legend=False)
    return tbl, ch


def run(ctx):
    atoms = setup(ctx)
    calc = atoms.calc
    n, periodic = len(atoms), bool(atoms.pbc.any())
    ctx.status("computing")
    t = time.perf_counter()
    f = atoms.get_forces()
    e = atoms.get_potential_energy()
    first = time.perf_counter() - t
    times = []
    for _ in range(3):  # timing repeats: cheap, and researchers want the warm number
        t = time.perf_counter()
        calc.calculate(atoms, ["energy", "forces"], ["positions"])
        times.append(time.perf_counter() - t)
    warm = float(np.median(times))
    fn = np.linalg.norm(f, axis=1)
    lot = lot_used(calc, atoms)
    dec = calc.results.get("energy_decomposition")
    tiles = [R.tile("energy", "Energy", e, "eV", 5, hint="Total energy at the level of theory below. Compare only with the same model and level."),
             R.tile("energy_per_atom", "Energy per atom", e / n, "eV/atom", 5)]
    if isinstance(dec, dict) and "kcal_mol" in dec and "eV" in dec:
        ik, ie = float(dec["kcal_mol"]["interaction"]), float(dec["eV"]["interaction"])
        tiles.append(R.tile("interaction_energy", "Residual energy (E − E_ref)", ik, "kcal/mol", 2,
                            hint=f"{ie:.4f} eV (sum of network, electrostatics, polarization, D3 & repulsion)."))
    tiles += [R.tile("fmax", "Largest force", fmax_of(f), "eV/Å", 4, hint="Close to zero at a relaxed structure."),
              R.tile("frms", "Typical force (rms)", float(np.sqrt((f ** 2).sum(1).mean())), "eV/Å", 4)]
    raw = {"energy_eV": e, "forces_eV_A": f, "lot": lot}
    if dec is not None:
        raw["energy_decomposition"] = dec
    if periodic and backends.has_stress(calc) and atoms.pbc.all():
        s = atoms.get_stress(voigt=True)
        p = -float(s[:3].mean()) / GPa
        tiles.append(R.tile("pressure", "Pressure", p, "GPa", 3, hint="Positive means the cell wants to expand."))
        raw["stress_eV_A3"] = s

    dip_vec = None
    if not periodic:
        if calc.results.get("dipole_debye") is not None:
            dip_vec = np.asarray(calc.results["dipole_debye"], dtype=float).reshape(3)
        elif calc.results.get("dipole") is not None:
            dip_vec = np.asarray(calc.results["dipole"], dtype=float).reshape(3) * E_ANG_TO_DEBYE
        if dip_vec is not None:
            dnorm = float(np.linalg.norm(dip_vec))
            tiles.append(R.tile("dipole", "Dipole moment", dnorm, "D", 3,
                                hint=f"Vector: ({dip_vec[0]:+.3f}, {dip_vec[1]:+.3f}, {dip_vec[2]:+.3f}) D"))
            raw["dipole_debye"] = dip_vec
            raw["dipole_norm_debye"] = dnorm
        pol_iso = calc.results.get("polarizability_iso")
        pol_aniso = calc.results.get("polarizability_aniso")
        if pol_iso is not None and float(pol_iso) > 1e-6:
            hint = (f"Anisotropy γ = {float(pol_aniso):.2f} Å³ · latent intramolecular polarization head"
                    if pol_aniso is not None else "Latent intramolecular polarization head")
            tiles.append(R.tile("polarizability_iso", "Model polarizability", float(pol_iso), "Å³", 2, hint=hint))
            raw["polarizability_iso_A3"] = float(pol_iso)
            if pol_aniso is not None:
                raw["polarizability_aniso_A3"] = float(pol_aniso)
            if calc.results.get("polarizability") is not None:
                raw["polarizability_tensor_A3"] = np.asarray(calc.results["polarizability"], dtype=float)

    q = calc.results.get("charges")
    has_q = q is not None and len(q) == n and float(np.max(np.abs(q))) > 1e-4
    stage_q = calc.results.get("stage_charges")
    dq_equil = None
    if has_q and stage_q is not None and len(stage_q) >= 1:
        q0 = np.asarray(stage_q[0], dtype=float).reshape(-1)
        if len(q0) == n:
            dq_equil = np.asarray(q, dtype=float).reshape(-1) - q0
    spins = calc.results.get("spin_density")
    if spins is None:
        spins = calc.results.get("spins")
    has_spins = spins is not None and len(spins) == n and float(np.max(np.abs(spins))) > 1e-4
    alpha_at = calc.results.get("atomic_polarizabilities")
    has_alpha_at = alpha_at is not None and len(alpha_at) == n and float(np.max(np.abs(alpha_at))) > 1e-4
    e_at = calc.results.get("atomic_energies")
    if e_at is None:
        e_at = calc.results.get("energies")
    has_e_at = e_at is not None and len(e_at) == n

    tiles += [R.tile("time_warm_ms", "Time per evaluation", warm * 1000, "ms", 1, better="lower", hint="After warm-up."),
              R.tile("time_first_ms", "First evaluation", first * 1000, "ms", 1, hint="Includes neighbour search and device warm-up.")]
    sym = atoms.get_chemical_symbols()
    rows = [
        [i + 1, sym[i], *np.round(atoms.positions[i], 4).tolist(), *np.round(f[i], 4).tolist(), round(float(fn[i]), 4)]
        + ([round(float(q[i]), 4)] if has_q else [])
        + ([round(float(dq_equil[i]), 4)] if dq_equil is not None else [])
        + ([round(float(spins[i]), 4)] if has_spins else [])
        + ([round(float(alpha_at[i]), 3)] if has_alpha_at else [])
        + ([round(float(e_at[i]), 5)] if has_e_at else [])
        for i in range(min(n, 2000))
    ]
    cols = (["#", "element", "x (Å)", "y (Å)", "z (Å)", "Fx (eV/Å)", "Fy", "Fz", "|F|"]
            + (["model charge (e)"] if has_q else [])
            + (["Δq_equil (e)"] if dq_equil is not None else [])
            + (["spin (e)"] if has_spins else [])
            + (["α (Å³)"] if has_alpha_at else [])
            + (["E_atom (eV)"] if has_e_at else []))
    if has_q:
        raw["charges_e"] = q
    if dq_equil is not None:
        raw["dq_equil_e"] = dq_equil
    if has_spins:
        raw["spins_e"] = spins
    if has_alpha_at:
        raw["atomic_polarizabilities_A3"] = alpha_at
    if has_e_at:
        raw["atomic_energies_eV"] = e_at

    tables = [R.table("Per-atom values", cols, rows,
                      "Model charges equilibrate across the structure (NQE) and feed electrostatics & dipole." if has_q else "")]
    charts = []
    dec_tbl, dec_chart = _decomp_artifacts(dec)
    if dec_tbl is not None:
        tables.append(dec_tbl)
    if dec_chart is not None:
        charts.append(dec_chart)

    a = structures.analyze(atoms)
    dip_frame = dip_vec.tolist() if (dip_vec is not None and float(np.linalg.norm(dip_vec)) > 0.01) else None
    ev_atoms = atoms.copy()
    ev_atoms.info.pop("_mena_phys", None)
    sp_props: dict = {"energy": float(e), "forces": np.asarray(f, dtype=float)}
    if has_q:
        sp_props["charges"] = np.asarray(q, dtype=float)
    if "stress_eV_A3" in raw:
        sp_props["stress"] = np.asarray(raw["stress_eV_A3"], dtype=float)
    if dip_vec is not None:
        ev_atoms.info["dipole_debye"] = np.round(dip_vec, 6).tolist()
    if lot:
        ev_atoms.info["lot"] = lot
    ev_atoms.calc = SinglePointCalculator(ev_atoms, **sp_props)
    ase_write(ctx.path("evaluated.extxyz"), ev_atoms, format="extxyz")
    files = [
        write_structure(ctx, atoms, "structure", f"Mena singlepoint, E = {e:.6f} eV"),
        {"name": "evaluated.extxyz", "label": "Evaluated (extXYZ)"},
    ]
    return {
        "headline": f"{a['formula']}: E = {e:.4f} eV ({e / n:.4f} eV/atom), largest force {fmax_of(f):.3f} eV/Å.",
        "tiles": tiles,
        "facts": {"Level of theory": LOT_LABELS.get(lot, lot or "model default"), "Atoms": n},
        "tables": tables,
        "charts": charts,
        "viewer": R.viewer([R.frame(atoms, f, a["formula"], dipole=dip_frame, charges=q if has_q else None, spins=spins if has_spins else None, alpha=alpha_at if has_alpha_at else None)], atoms),
        "files": files, "notes": [] if n <= 2000 else ["The table shows the first 2,000 atoms; the JSON and extXYZ downloads have all of them."],
        "raw": raw,
    }
