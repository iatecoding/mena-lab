"""End-to-end and unit tests for Mena-LAM 5 (`lam5-v1` / `lam4` backend) chemist interpretability and catalysis routing in `mena_app`."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mena_app import costmodel, export, models, planner, report, routing, runner, structures, validation


def _have_lam5() -> bool:
    try:
        m = models.choose("lam5-v1", "molecule")
        return bool(m.get("available"))
    except models.ModelError:
        return False


lam5 = pytest.mark.skipif(not _have_lam5(), reason="lam5-v1 release checkpoint not installed")


def _run_job(raw_spec: dict, out_dir: Path):
    spec = validation.validate(raw_spec)
    pl = planner.plan(spec)
    events = []
    res = runner.run(pl, spec, str(out_dir), events.append, threading.Event())
    assert res["status"] == "done", res.get("error")
    return spec, pl, res["result"]


def test_report_frame_backward_compat_and_frame_with_physics():
    h2o = structures.load("water")
    f_raw = np.array([[0.0, 0.0, 0.12], [0.0, 0.05, -0.06], [0.0, -0.05, -0.06]])
    legacy = report.frame(h2o, f_raw, "H2O", [0.0, 0.0, 1.85])
    assert legacy["label"] == "H2O" and len(legacy["forces"]) == 3 and legacy["dipole"] == [0.0, 0.0, 1.85]
    assert "charges" not in legacy and "fmag" not in legacy

    res = {
        "forces": f_raw,
        "charges": np.array([-0.52134, 0.26067, 0.26067]),
        "spin_density": np.array([0.0, 0.0, 0.0]),
        "atomic_polarizabilities": np.array([0.8812, 0.4198, 0.4198]),
        "dipole_debye": np.array([0.0, 0.0, 1.9116]),
    }
    traj_fr = report.frame_with_physics(h2o, res, include_forces=False, label="step 1")
    assert "forces" not in traj_fr
    assert traj_fr["fmag"] == [0.12, 0.0781, 0.0781]
    assert traj_fr["charges"] == [-0.5213, 0.2607, 0.2607]
    assert "spins" not in traj_fr  # closed-shell zeros omitted
    assert traj_fr["alpha"] == [0.8812, 0.4198, 0.4198]
    assert traj_fr["dipole"] == [0.0, 0.0, 1.9116]

    sp_fr = report.frame_with_physics(h2o, res, include_forces=True, label="sp")
    assert len(sp_fr["forces"]) == 3 and "fmag" not in sp_fr

    # Periodic slab omits molecular dipole vector even if present in dict
    slab = structures.load("cu111")
    slab_fr = report.frame_with_physics(slab, {"dipole_debye": [1.0, 0.0, 0.0], "charges": np.zeros(len(slab))})
    assert "dipole" not in slab_fr and "charges" not in slab_fr


def test_catalysis_slab_lot_routing_and_builtin_examples():
    # Built-in reaction & catalysis structures
    pt_co = structures.load("pt111_co")
    an_pt = structures.analyze(pt_co)
    assert an_pt["domain"] == "slab" and an_pt["n_atoms"] == 29
    assert list(pt_co.pbc) == [True, True, False]
    assert an_pt["hubbard_u"] is False and an_pt["warnings"] == []

    mal_l = structures.load("malonaldehyde_l")
    mal_r = structures.load("malonaldehyde_r")
    an_ml = structures.analyze(mal_l)
    an_mr = structures.analyze(mal_r)
    assert an_ml["domain"] == "molecule" and an_ml["n_atoms"] == 9 and an_ml["formula"] == "C3H4O2"
    assert mal_l.get_chemical_symbols() == mal_r.get_chemical_symbols()
    assert np.allclose(mal_l.positions[:, 2], 0.0) and np.allclose(mal_r.positions[:, 2], 0.0)
    assert an_ml["warnings"] == [] and an_mr["warnings"] == []

    all_m = {m["id"]: m for m in models.all_models()}
    lam5_m = all_m["lam5-v1"]
    legacy_m = all_m["lam-400k"]
    cal = costmodel.load()

    for name in ("cu111", "pt111_co"):
        sp = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": name}})
        an = structures.analyze(structures.load(name))
        dec = routing.route(sp, an, ["cpu"], 32.0, cal, lam5_m)
        assert dec["lot"] == "oc20_rpbe"
        assert "RPBE" in dec["reasons"]["lot"] and "OC20" in dec["reasons"]["lot"]

        # Legacy model without oc20_rpbe stays untouched
        dec_old = routing.route(sp, an, ["cpu"], 32.0, cal, legacy_m)
        assert dec_old["lot"] == "auto"

    # Oxide slab with MP Hubbard metal (Fe + O) routes to oc22_pbeU on lam5-v1
    feo_slab = Atoms("Fe2O2", positions=[[0, 0, 4.0], [1.5, 1.5, 4.0], [1.5, 0, 4.8], [0, 1.5, 4.8]],
                     cell=[[3.1, 0, 0], [0, 3.1, 0], [0, 0, 14.0]], pbc=[True, True, False])
    an_feo = structures.analyze(feo_slab)
    assert an_feo["domain"] == "slab" and an_feo["hubbard_u"] is True
    sp_feo = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "cu111"}})
    dec_feo = routing.route(sp_feo, an_feo, ["cpu"], 32.0, cal, lam5_m)
    assert dec_feo["lot"] == "oc22_pbeU" and "OC22" in dec_feo["reasons"]["lot"]

    # Bulk crystal still uses auto MP PBE / PBE+U routing
    sp_si = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "si"}})
    dec_si = routing.route(sp_si, structures.analyze(structures.load("si")), ["cpu"], 32.0, cal, lam5_m)
    assert dec_si["lot"] == "auto"


@lam5
def test_lam5_singlepoint_molecule_interpretability_and_export(tmp_path: Path):
    spec, pl, r = _run_job(
        {
            "task": "singlepoint",
            "structure": {"format": "builtin", "text": "water"},
            "params": {},
            "runtime": {"model": "lam5-v1", "device": "cpu"},
        },
        tmp_path / "sp_water",
    )
    py_code = export.to_python(spec, pl, "test")
    assert "from lam4 import MenaCalculator" in py_code
    assert "MenaCalculator.from_checkpoint(" in py_code

    raw = r["raw"]
    assert 1.65 < raw["dipole_norm_debye"] < 2.25
    assert 0.5 < raw["polarizability_iso_A3"] < 8.0
    dec = raw["energy_decomposition"]
    assert abs(dec["eV"]["total"] - raw["energy_eV"]) < 1e-6
    inter_sum = (
        dec["kcal_mol"]["network"]
        + dec["kcal_mol"]["electrostatic"]
        + dec["kcal_mol"]["polarization"]
        + dec["kcal_mol"]["repulsion"]
        + dec["kcal_mol"]["dispersion_d3"]
    )
    assert abs(dec["kcal_mol"]["interaction"] - inter_sum) < 1e-5

    tile_keys = {t["key"] for t in r["tiles"]}
    assert {"energy", "interaction_energy", "dipole", "polarizability_iso"} <= tile_keys
    table_titles = [t["title"] for t in r["tables"]]
    assert "Per-atom values" in table_titles
    assert "Physical energy decomposition" in table_titles
    assert any(c["id"] == "energy_decomp" and c["kind"] == "bar" for c in r["charts"])

    atom_cols = r["tables"][0]["columns"]
    assert "model charge (e)" in atom_cols
    assert "Δq_equil (e)" in atom_cols
    assert "α (Å³)" in atom_cols
    assert "dq_equil_e" in raw and len(raw["dq_equil_e"]) == 3

    fr0 = r["viewer"]["frames"][0]
    assert "dipole" in fr0 and len(fr0["dipole"]) == 3
    assert np.allclose(fr0["dipole"], raw["dipole_debye"], atol=1e-3)
    assert "charges" in fr0 and len(fr0["charges"]) == 3 and fr0["charges"][0] < -0.15 and abs(sum(fr0["charges"])) < 1e-3
    assert "alpha" in fr0 and len(fr0["alpha"]) == 3 and min(fr0["alpha"]) > 0.1


@lam5
def test_lam5_open_shell_spin_density_column(tmp_path: Path):
    _, _, r = _run_job(
        {
            "task": "singlepoint",
            "structure": {"format": "xyz", "text": "2\ntriplet O2\nO 0.0 0.0 0.0\nO 0.0 0.0 1.21\n"},
            "params": {},
            "system": {"charge": 0, "spin": 3},
            "runtime": {"model": "lam5-v1", "device": "cpu"},
        },
        tmp_path / "sp_o2_triplet",
    )
    cols = r["tables"][0]["columns"]
    assert "spin (e)" in cols
    assert abs(sum(r["raw"]["spins_e"]) - 2.0) < 0.05
    fr0 = r["viewer"]["frames"][0]
    assert "spins" in fr0 and len(fr0["spins"]) == 2 and abs(sum(fr0["spins"]) - 2.0) < 0.05


@lam5
def test_lam5_neb_barrier_decomposition_and_viewer_physics(tmp_path: Path):
    _, _, r = _run_job(
        {
            "task": "neb",
            "structure": {"format": "builtin", "text": "malonaldehyde_l"},
            "structure2": {"format": "builtin", "text": "malonaldehyde_r"},
            "params": {"images": 5, "relax_ends": False, "max_steps": 25, "fmax": 0.05, "climb": True},
            "runtime": {"model": "lam5-v1", "device": "cpu"},
        },
        tmp_path / "neb_malonaldehyde",
    )
    chart_ids = {c["id"]: c for c in r["charts"]}
    assert "profile" in chart_ids
    assert "neb_decomp" in chart_ids
    assert "neb_dipole" in chart_ids

    dec_series = {s["name"]: s["y"] for s in chart_ids["neb_decomp"]["series"]}
    assert "ΔTotal" in dec_series and "ΔNetwork" in dec_series and "ΔElectrostatics" in dec_series
    assert max(dec_series["ΔTotal"]) > 1.0  # non-trivial proton-transfer barrier in kcal/mol

    assert any(t["title"] == "Barrier physical decomposition" for t in r["tables"])
    dec_tbl = next(t for t in r["tables"] if t["title"] == "Barrier physical decomposition")
    comp_sum_ts = sum(row[1] for row in dec_tbl["rows"][:-1])
    tot_ts = dec_tbl["rows"][-1][1]
    assert abs(comp_sum_ts - tot_ts) < 0.02

    # Per-image physics on all 7 frames (charges, alpha, dipole flip, fmag)
    frames = r["viewer"]["frames"]
    assert len(frames) == 7
    for fr in frames:
        assert len(fr["charges"]) == 9
        assert len(fr["alpha"]) == 9
        assert len(fr["fmag"]) == 9
        assert len(fr["dipole"]) == 3
        assert "forces" not in fr
    # Dipole x-component flips sign between left and right tautomers!
    assert frames[0]["dipole"][0] * frames[-1]["dipole"][0] < -0.1


@lam5
def test_lam5_relax_md_and_pt111_co_catalysis_routing(tmp_path: Path):
    # Pt(111)+CO singlepoint auto-routes to oc20_rpbe
    _, pl_pt, r_pt = _run_job(
        {
            "task": "singlepoint",
            "structure": {"format": "builtin", "text": "pt111_co"},
            "params": {},
            "runtime": {"model": "lam5-v1", "device": "cpu"},
        },
        tmp_path / "sp_pt111_co",
    )
    assert pl_pt["decision"]["lot"] == "oc20_rpbe"
    assert r_pt["raw"]["lot"] == "oc20_rpbe"
    pt_fr0 = r_pt["viewer"]["frames"][0]
    assert len(pt_fr0["forces"]) == 29 and "cell" in pt_fr0 and "dipole" not in pt_fr0 and "charges" not in pt_fr0

    # Relax & MD trajectory frames capture zero-overhead charges, alpha, dipole, and fmag
    _, _, r_rel = _run_job(
        {
            "task": "relax",
            "structure": {"format": "builtin", "text": "water"},
            "params": {"max_steps": 3, "fmax": 0.001},
            "runtime": {"model": "lam5-v1", "device": "cpu"},
        },
        tmp_path / "relax_water",
    )
    rel_fr = r_rel["viewer"]["frames"][-1]
    assert "charges" in rel_fr and "alpha" in rel_fr and "dipole" in rel_fr and "fmag" in rel_fr

    _, _, r_md = _run_job(
        {
            "task": "md",
            "structure": {"format": "builtin", "text": "water"},
            "params": {"ensemble": "nvt", "steps": 4, "timestep": 0.5, "save_every": 2},
            "runtime": {"model": "lam5-v1", "device": "cpu"},
        },
        tmp_path / "md_water",
    )
    md_fr = r_md["viewer"]["frames"][-1]
    assert "charges" in md_fr and "alpha" in md_fr and "dipole" in md_fr and "fmag" in md_fr


@lam5
def test_lam5_vibrations_ir_and_raman_spectra(tmp_path: Path):
    out_dir = tmp_path / "vib_water"
    _, _, r = _run_job(
        {
            "task": "vibrations",
            "structure": {"format": "builtin", "text": "water"},
            "params": {"prerelax": True, "fmax": 0.005},
            "runtime": {"model": "lam5-v1", "device": "cpu", "dtype": "float32"},
        },
        out_dir,
    )
    raw = r["raw"]
    assert len(raw["frequencies_cm"]) == 3
    assert len(raw["ir_km_mol"]) == 3 and float(np.max(raw["ir_km_mol"])) > 10.0
    assert len(raw["raman_A4_amu"]) == 3 and float(np.max(raw["raman_A4_amu"])) > 1.0

    chart_ids = [c["id"] for c in r["charts"]]
    assert {"ir_spectrum", "raman_spectrum", "spectrum"} <= set(chart_ids)
    cols = r["tables"][0]["columns"]
    assert "IR (km/mol)" in cols and "Raman (Å⁴/amu)" in cols
    assert "ir" in r["viewer"]["modes"][0] and "raman" in r["viewer"]["modes"][0]

    freq_csv = (out_dir / "frequencies.csv").read_text(encoding="utf-8")
    assert "ir_km_mol" in freq_csv and "raman_A4_amu" in freq_csv
    spec_csv = (out_dir / "ir_raman_spectrum.csv").read_text(encoding="utf-8")
    assert "wavenumber_cm-1,ir_km_mol,raman_A4_amu" in spec_csv
