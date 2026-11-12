"""Vibrations of a molecule: finite-difference Hessian, normal modes, IR & Raman spectra, ZPE and harmonic thermodynamics."""

from __future__ import annotations

import numpy as np
from ase import units

from lam4.calc.physics import harmonic_ir_raman
from mena_app import report as R
from mena_app.runtime import Cancelled
from mena_app.tasks.common import fmax_of, optimize, setup, write_structure

EV_TO_CM = 1.0 / units.invcm
HBAR_EV = units._hbar / units._e  # eV s
BATCH_DISP_CHUNK = 36


def displacements_and_derivatives(ctx, atoms, dx: float) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Central finite-difference Hessian + Cartesian dipole & polarizability Jacobians.
    Uses `calc.eval_batch` in chunks when supported (~4-6x faster on CPU/MPS), with exact serial fallback."""
    p0 = atoms.positions.copy()
    n3 = 3 * len(atoms)
    H = np.zeros((n3, n3), dtype=np.float64)
    dmu_dR = np.zeros((n3, 3), dtype=np.float64)
    dalpha_dR = np.zeros((n3, 3, 3), dtype=np.float64)
    got_dip = False
    got_pol = False
    calc = atoms.calc

    if hasattr(calc, "eval_batch"):
        frames = []
        for k in range(n3):
            i, c = divmod(k, 3)
            for sign in (1.0, -1.0):
                cp = atoms.copy()
                cp.positions[i, c] += sign * dx
                frames.append(cp)
        outs: list[dict] = []
        total = len(frames)
        for start in range(0, total, BATCH_DISP_CHUNK):
            ctx.check()
            chunk = frames[start:start + BATCH_DISP_CHUNK]
            outs.extend(calc.eval_batch(chunk, properties=("energy", "forces", "dipole", "polarizability")))
            done_coords = min(n3, (start + len(chunk)) // 2)
            ctx.progress(done_coords, n3, f"batched displacement {done_coords} of {n3}")
        for k in range(n3):
            rp, rm = outs[2 * k], outs[2 * k + 1]
            H[k] = -(np.asarray(rp["forces"], dtype=np.float64).ravel() - np.asarray(rm["forces"], dtype=np.float64).ravel()) / (2.0 * dx)
            if rp.get("dipole") is not None and rm.get("dipole") is not None:
                dmu_dR[k] = (np.asarray(rp["dipole"], dtype=np.float64) - np.asarray(rm["dipole"], dtype=np.float64)) / (2.0 * dx)
                got_dip = True
            if rp.get("polarizability") is not None and rm.get("polarizability") is not None:
                dalpha_dR[k] = (np.asarray(rp["polarizability"], dtype=np.float64) - np.asarray(rm["polarizability"], dtype=np.float64)) / (2.0 * dx)
                got_pol = True
    else:
        for k in range(n3):
            ctx.check()
            i, c = divmod(k, 3)
            fs, dips, pols = [], [], []
            for sign in (1.0, -1.0):
                atoms.positions = p0.copy()
                atoms.positions[i, c] += sign * dx
                fs.append(atoms.get_forces().ravel())
                res = getattr(calc, "results", {})
                dips.append(res.get("dipole"))
                pols.append(res.get("polarizability"))
            H[k] = -(fs[0] - fs[1]) / (2.0 * dx)
            if dips[0] is not None and dips[1] is not None:
                dmu_dR[k] = (np.asarray(dips[0], dtype=np.float64) - np.asarray(dips[1], dtype=np.float64)) / (2.0 * dx)
                got_dip = True
            if pols[0] is not None and pols[1] is not None:
                dalpha_dR[k] = (np.asarray(pols[0], dtype=np.float64) - np.asarray(pols[1], dtype=np.float64)) / (2.0 * dx)
                got_pol = True
            ctx.progress(k + 1, n3, f"displacement {k + 1} of {n3}")
    atoms.positions = p0
    has_dip = got_dip and float(np.max(np.abs(dmu_dR))) > 1e-10
    has_pol = got_pol and float(np.max(np.abs(dalpha_dR))) > 1e-10
    return 0.5 * (H + H.T), (dmu_dR if has_dip else None), (dalpha_dR if has_pol else None)


def hessian(ctx, atoms, dx: float) -> np.ndarray:
    H, _, _ = displacements_and_derivatives(ctx, atoms, dx)
    return H


def external_basis(atoms) -> np.ndarray:
    """Orthonormal mass-weighted translations and rotations (3N x 6, or x 5 for a linear molecule)."""
    m = atoms.get_masses()
    sm = np.repeat(np.sqrt(m), 3)
    x = atoms.positions - (m[:, None] * atoms.positions).sum(0) / m.sum()
    vecs = []
    for c in range(3):
        t = np.zeros((len(atoms), 3))
        t[:, c] = 1.0
        vecs.append(t.ravel() * sm)
    for c in range(3):
        axis = np.zeros(3)
        axis[c] = 1.0
        vecs.append(np.cross(axis, x).ravel() * sm)
    q, r = np.linalg.qr(np.array(vecs).T)
    # 1e-3 (relative, linear in the bend angle): a molecule within ~0.05 degrees of linear, as a relaxed or rounded CO2
    # usually is, has 5 rigid-body motions; 1e-6 kept a 6th and removed a real bending mode.
    return q[:, np.abs(np.diag(r)) > 1e-3 * np.abs(np.diag(r)).max()]


def normal_modes(atoms, H: np.ndarray, project: bool = True, return_mw: bool = False):
    """Mode energies (eV, negative = imaginary) and displacement patterns. With `project`, translations and rotations are
    removed exactly from the mass-weighted Hessian, as quantum-chemistry codes do."""
    m = np.repeat(atoms.get_masses(), 3)
    Hm = H / np.sqrt(np.outer(m, m))
    if project:
        V = external_basis(atoms)
        P = np.eye(len(m)) - V @ V.T
        Hm = P @ Hm @ P
    lam, vec = np.linalg.eigh(Hm)
    # hbar*omega in eV: lam is eV/(Å² amu)
    s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
    energies = s * np.sign(lam) * np.sqrt(np.abs(lam))
    disp = (vec / np.sqrt(m)[:, None]).T.reshape(-1, len(atoms), 3)
    if return_mw:
        return energies, disp, vec.T
    return energies, disp


def broaden_lorentzian(cm: np.ndarray, intensities: np.ndarray, fwhm: float = 15.0, n_pts: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Broaden real harmonic peaks (cm > 0) with a Lorentzian profile of width `fwhm` cm⁻¹."""
    hi = max(4000.0, float(np.max(cm)) + 200.0) if len(cm) else 4000.0
    grid = np.linspace(0.0, hi, n_pts)
    y = np.zeros_like(grid)
    hwhm = 0.5 * fwhm
    mask = cm > 0.0
    if np.any(mask):
        c_pos = cm[mask][:, None]
        w_pos = intensities[mask][:, None]
        y = np.sum(w_pos * (hwhm ** 2) / ((grid[None, :] - c_pos) ** 2 + hwhm ** 2), axis=0)
    return grid, y


def thermo(e_vib: np.ndarray, T: float) -> dict:
    """Harmonic oscillator: zero-point energy, vibrational internal energy, entropy and free energy (eV, eV/K)."""
    real = np.asarray(e_vib, dtype=float)
    real = real[real > 1e-4]
    if len(real) == 0:
        return {"zpe": 0.0, "u": 0.0, "s": 0.0, "g": 0.0}
    kT = units.kB * T
    x = real / kT
    zpe = float(0.5 * real.sum())
    u = float(zpe + (real / np.expm1(x)).sum())
    s = float(units.kB * (x / np.expm1(x) - np.log1p(-np.exp(-x))).sum())
    return {"zpe": zpe, "u": u, "s": s, "g": u - T * s}


def run(ctx):
    p = ctx.params
    atoms = setup(ctx)
    n = len(atoms)
    notes, files = [], []
    if p["prerelax"]:
        ctx.status("relaxing first")
        r = optimize(ctx, atoms, "LBFGS", p["fmax"], 1000, label="pre-relax")
        if r["cancelled"]:
            raise Cancelled()
        if not r["converged"]:
            notes.append("The relaxation did not fully converge, so the lowest frequencies are less reliable.")
        files.append(write_structure(ctx, atoms, "relaxed"))
    f_res = fmax_of(atoms.get_forces())
    ctx.status("measuring force constants, IR & Raman derivatives")
    H, dmu_dR, dalpha_dR = displacements_and_derivatives(ctx, atoms, p["displacement"])
    raw_energies, _ = normal_modes(atoms, H, project=False)
    V = external_basis(atoms)
    n_ext = V.shape[1]
    energies, disp, vec_mw = normal_modes(atoms, H, return_mw=True)
    order = np.argsort(np.abs(energies))
    vib = np.sort(order[n_ext:])
    e_vib = energies[vib]
    cm = e_vib * EV_TO_CM
    ir_km_mol, raman_a4 = harmonic_ir_raman(vec_mw[vib], atoms.get_masses(), dmu_dR, dalpha_dR)

    imag = cm[cm < 0]
    real = e_vib[e_vib > 1e-4]
    th = thermo(real, p["temperature"])
    tiles = [R.tile("n_imag", "Imaginary modes", len(imag), "", 0, status="pass" if len(imag) == 0 else "warn",
                    hint="Zero at a true minimum; one at a transition state."),
             R.tile("lowest", "Lowest frequency", float(cm.min()) if len(cm) else None, "cm⁻¹", 1),
             R.tile("highest", "Highest frequency", float(cm.max()) if len(cm) else None, "cm⁻¹", 1)]
    real_mask = cm > 0
    if ir_km_mol is not None and np.any(real_mask):
        idx_ir = int(np.argmax(np.where(real_mask, ir_km_mol, -1.0)))
        tiles.append(R.tile("strongest_ir", "Strongest IR band", float(cm[idx_ir]), "cm⁻¹", 1,
                            hint=f"{float(ir_km_mol[idx_ir]):.1f} km/mol (mode {idx_ir + 1})"))
    if raman_a4 is not None and np.any(real_mask):
        idx_ra = int(np.argmax(np.where(real_mask, raman_a4, -1.0)))
        tiles.append(R.tile("strongest_raman", "Strongest Raman band", float(cm[idx_ra]), "cm⁻¹", 1,
                            hint=f"{float(raman_a4[idx_ra]):.1f} Å⁴/amu (mode {idx_ra + 1})"))
    tiles += [R.tile("zpe", "Zero-point energy", th["zpe"], "eV", 4, hint=f"{th['zpe'] * 23.0605:.2f} kcal/mol"),
              R.tile("g_corr", f"Vibrational free energy at {p['temperature']:.0f} K", th["g"], "eV", 4, hint="Harmonic; add to the electronic energy."),
              R.tile("s_vib", "Vibrational entropy", th["s"] * 1000, "meV/K", 4)]
    if len(imag):
        notes.append(f"{len(imag)} imaginary mode(s). If you expected a minimum, relax more tightly (smaller force target) and run again.")
    rot = float(np.sort(np.abs(raw_energies))[:n_ext].max() * EV_TO_CM)
    if rot > 50:
        notes.append(f"Before removing translations and rotations they reached {rot:.0f} cm⁻¹; the structure may not be fully relaxed, "
                     f"so frequencies below about {rot:.0f} cm⁻¹ are uncertain.")

    cols = ["mode", "cm⁻¹"] + (["IR (km/mol)"] if ir_km_mol is not None else []) + (["Raman (Å⁴/amu)"] if raman_a4 is not None else []) + [""]
    rows = []
    modes = []
    for k, (c, i) in enumerate(zip(cm, vib)):
        row = [k + 1, round(float(c), 1)]
        m_entry = {"freq": float(c), "disp": np.round(disp[i] / max(np.abs(disp[i]).max(), 1e-12), 4).tolist()}
        if ir_km_mol is not None:
            row.append(round(float(ir_km_mol[k]), 2))
            m_entry["ir"] = round(float(ir_km_mol[k]), 2)
        if raman_a4 is not None:
            row.append(round(float(raman_a4[k]), 2))
            m_entry["raman"] = round(float(raman_a4[k]), 2)
        row.append("imaginary" if c < 0 else "")
        rows.append(row)
        modes.append(m_entry)

    charts = []
    spectrum_grid = ir_env = raman_env = None
    if ir_km_mol is not None:
        spectrum_grid, ir_env = broaden_lorentzian(cm, ir_km_mol, fwhm=15.0)
        charts.append(R.chart("ir_spectrum", "Infrared (IR) spectrum (FWHM 15 cm⁻¹)",
                              [{"name": "IR intensity", "x": np.round(spectrum_grid, 1).tolist(), "y": np.round(ir_env, 3).tolist()}],
                              "line", "wavenumber (cm⁻¹)", "IR intensity (km/mol)", legend=False))
    if raman_a4 is not None:
        g_ra, raman_env = broaden_lorentzian(cm, raman_a4, fwhm=15.0)
        if spectrum_grid is None:
            spectrum_grid = g_ra
        charts.append(R.chart("raman_spectrum", "Raman activity spectrum (relative, FWHM 15 cm⁻¹)",
                              [{"name": "Raman activity", "x": np.round(g_ra, 1).tolist(), "y": np.round(raman_env, 3).tolist()}],
                              "line", "wavenumber (cm⁻¹)", "Raman activity (Å⁴/amu, latent α)", legend=False))
        notes.append("IR intensities come from DFT-supervised dipole derivatives (km/mol). Raman activities come from the model's latent anisotropic polarization head (relative band strengths and symmetry selection rules).")
    sticks = {"name": "modes", "x": [abs(float(c)) for c in cm], "y": [1.0] * len(cm)}
    charts.append(R.chart("spectrum", "Vibrational frequencies", [sticks], "sticks", "wavenumber (cm⁻¹)", ""))

    np.save(ctx.path("hessian_eV_A2.npy"), H)
    files.append({"name": "hessian_eV_A2.npy", "label": "Hessian (NumPy, eV/Å²)"})
    _write_modes(ctx, cm, ir_km_mol, raman_a4)
    files.append({"name": "frequencies.csv", "label": "Frequencies & IR/Raman (CSV)" if ir_km_mol is not None else "Frequencies (CSV)"})
    if spectrum_grid is not None:
        _write_spectrum_curve(ctx, spectrum_grid, ir_env, raman_env)
        files.append({"name": "ir_raman_spectrum.csv", "label": "Broadened IR/Raman spectrum (CSV)"})

    raw = {"frequencies_cm": cm, "zpe_eV": th["zpe"], "thermo": th}
    if ir_km_mol is not None:
        raw["ir_km_mol"] = ir_km_mol
    if raman_a4 is not None:
        raw["raman_A4_amu"] = raman_a4
    return {
        "headline": f"{len(cm)} vibrations from {abs(cm).min():.0f} to {cm.max():.0f} cm⁻¹, {len(imag)} imaginary. Zero-point energy {th['zpe']:.4f} eV ({th['zpe'] * 23.0605:.2f} kcal/mol).",
        "tiles": tiles, "facts": {"Residual force before displacements": f"{f_res:.4f} eV/Å", "Translations/rotations before projection": f"{rot:.1f} cm⁻¹", "Atoms": n},
        "tables": [R.table("Frequencies", cols, rows, "Click any row to animate that vibrational mode in 3D.")],
        "charts": charts,
        "viewer": R.viewer([R.frame(atoms, None, "relaxed" if p["prerelax"] else "input")], atoms, modes=modes),
        "files": files, "notes": notes, "raw": raw,
    }


def _linear(atoms) -> bool:
    return len(atoms) > 1 and external_basis(atoms).shape[1] == 5


def _write_modes(ctx, cm, ir_km_mol=None, raman_a4=None) -> None:
    with open(ctx.path("frequencies.csv"), "w", encoding="utf-8") as fh:
        cols = ["mode", "frequency_cm-1"]
        if ir_km_mol is not None:
            cols.append("ir_km_mol")
        if raman_a4 is not None:
            cols.append("raman_A4_amu")
        fh.write(",".join(cols) + "\n")
        for k, c in enumerate(cm):
            vals = [str(k + 1), f"{c:.2f}"]
            if ir_km_mol is not None:
                vals.append(f"{float(ir_km_mol[k]):.4f}")
            if raman_a4 is not None:
                vals.append(f"{float(raman_a4[k]):.4f}")
            fh.write(",".join(vals) + "\n")


def _write_spectrum_curve(ctx, grid, ir_env=None, raman_env=None) -> None:
    with open(ctx.path("ir_raman_spectrum.csv"), "w", encoding="utf-8") as fh:
        cols = ["wavenumber_cm-1"] + (["ir_km_mol"] if ir_env is not None else []) + (["raman_A4_amu"] if raman_env is not None else [])
        fh.write(",".join(cols) + "\n")
        for i, w in enumerate(grid):
            vals = [f"{float(w):.2f}"]
            if ir_env is not None:
                vals.append(f"{float(ir_env[i]):.4f}")
            if raman_env is not None:
                vals.append(f"{float(raman_env[i]):.4f}")
            fh.write(",".join(vals) + "\n")
