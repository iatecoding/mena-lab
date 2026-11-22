"""End-to-end and unit tests verifying zero optional runtime extras (pure-Python + NumPy SMILES, symmetry, and YAML)."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.data import covalent_radii
from ase.phonons import Phonons

from lam.data import registry as lot_registry
from mena_app import (
    backends,
    cli,
    crystal_symmetry,
    doctor,
    models,
    phonon_symmetry,
    runtime,
    server,
    smiles,
    smiles_ff,
    structures,
    validation,
    yaml_lite,
)
from mena_app.structures import StructureError
from mena_app.tasks import singlepoint
from mena_app.tasks.phonons import _LivePhonons

REPO_ROOT = Path(__file__).resolve().parents[2]


SMILES_CASES = [
    ("O", "H2O", 0, None),
    ("CCO", "C2H6O", 0, None),
    ("CC(=O)O", "C2H4O2", 0, None),
    ("c1ccccc1", "C6H6", 0, None),
    ("n1ccccc1", "C5H5N", 0, None),
    ("c1cc[nH]c1", "C4H5N", 0, None),
    ("c1ccoc1", "C4H4O", 0, None),
    ("c1ccsc1", "C4H4S", 0, None),
    ("c1cnc[nH]1", "C3H4N2", 0, None),
    ("c1ccc2[nH]ccc2c1", "C8H7N", 0, None),
    ("c1ccc2ccccc2c1", "C10H8", 0, None),
    ("CC(=O)Oc1ccccc1C(=O)O", "C9H8O4", 0, None),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "C13H18O2", 0, None),
    ("Cn1cnc2c1c(=O)n(c(=O)n2C)C", "C8H10N4O2", 0, None),
    ("[NH4+]", "H4N", 1, None),
    ("[OH-]", "HO", -1, None),
    ("[O-]", "O", -1, 2),
    ("CC(=O)[O-]", "C2H3O2", -1, None),
    ("[BH4-]", "H4B", -1, None),
    ("[CH3]", "CH3", 0, 2),
    ("[CH2]", "CH2", 0, 3),
    ("[Na+].[Cl-]", "ClNa", 0, None),
    ("CCO.O", "C2H8O2", 0, None),
    ("C#N", "CHN", 0, None),
    ("O=C=O", "CO2", 0, None),
    ("C1CC1", "C3H6", 0, None),
    ("C1CCCCC1", "C6H12", 0, None),
    ("[13CH4]", "CH4", 0, None),
    ("[Fe+2]", "Fe", 2, None),
    ("[Fe++]", "Fe", 2, None),
    ("[O--]", "O", -2, None),
    ("C12CC3CC(C1)CC(C3)C2", "C10H16", 0, None),
    ("c%10ccccc%10", "C6H6", 0, None),
    ("C=1CCCCC=1", "C6H10", 0, None),
    ("c1ccccc1c2ccccc2", "C12H10", 0, None),
    ("C[N+](=O)[O-]", "CH3NO2", 0, None),
]


@pytest.mark.parametrize("smi,formula,charge,spin", SMILES_CASES)
def test_smiles_to_atoms_formula_charge_spin_and_clash_free_3d(smi, formula, charge, spin):
    a = smiles.smiles_to_atoms(smi)
    assert a.get_chemical_formula("hill") == formula
    assert a.info["charge"] == charge
    assert a.info.get("spin") == spin
    if charge != 0 or "." in smi or "+" in smi or "-" in smi:
        if any(ch in smi for ch in "+-"):
            assert abs(float(a.get_initial_charges().sum()) - charge) < 1e-6
    nelec = int(a.numbers.sum()) - charge
    eff_spin = a.info.get("spin", 1)
    assert nelec % 2 == (eff_spin - 1) % 2
    assert np.all(np.isfinite(a.positions))
    if len(a) > 1:
        rc = covalent_radii[a.numbers]
        d = np.linalg.norm(a.positions[:, None, :] - a.positions[None, :, :], axis=-1) + np.eye(len(a)) * 100.0
        assert float((d / (rc[:, None] + rc[None, :])).min()) >= 0.72


def test_smiles_stereo_fragments_speed_and_errors():
    # Warmup then verify < 40 ms on drug-sized molecules (caffeine & ibuprofen); shared CI runners get a looser bound
    is_shared_runner = bool(os.environ.get("CI") or os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))
    limit_ms = 250.0 if is_shared_runner else 40.0
    smiles.smiles_to_atoms("CCO")
    for drug in ("Cn1cnc2c1c(=O)n(c(=O)n2C)C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"):
        dts = []
        for _ in range(3):
            t0 = time.perf_counter()
            smiles.smiles_to_atoms(drug)
            dts.append((time.perf_counter() - t0) * 1000.0)
        assert min(dts) < limit_ms, f"{drug} took {min(dts):.2f} ms (> {limit_ms:.0f} ms)"

    # E/Z alkene stereochemistry: trans 2-butene vs cis 2-butene 1-4 C..C distance
    at_trans = smiles.smiles_to_atoms("C/C=C/C")
    at_cis = smiles.smiles_to_atoms(r"C/C=C\C")
    ct = [i for i, z in enumerate(at_trans.numbers) if z == 6]
    cc = [i for i, z in enumerate(at_cis.numbers) if z == 6]
    d_trans = float(np.linalg.norm(at_trans.positions[ct[0]] - at_trans.positions[ct[3]]))
    d_cis = float(np.linalg.norm(at_cis.positions[cc[0]] - at_cis.positions[cc[3]]))
    assert d_trans > d_cis + 0.7 and d_trans > 3.6 and d_cis < 3.1

    # Tetrahedral chirality: @ vs @@ opposite signed volume
    def signed_vol(mol: Atoms) -> float:
        by_z = {int(z): i for i, z in enumerate(mol.numbers)}
        p_c = mol.positions[by_z[6]]
        v1, v2, v3, v4 = (mol.positions[by_z[z]] - p_c for z in (9, 1, 17, 35))
        return float(np.dot(v1 - v4, np.cross(v2 - v4, v3 - v4)))

    v_at = signed_vol(smiles.smiles_to_atoms("F[C@H](Cl)Br"))
    v_atat = signed_vol(smiles.smiles_to_atoms("F[C@@H](Cl)Br"))
    assert v_at > 2.0 and v_atat < -2.0 and abs(v_at + v_atat) < 0.2

    # Disconnected fragments placed ~3.0-4.5 A apart
    nacl = smiles.smiles_to_atoms("[Na+].[Cl-]")
    assert 3.0 <= float(np.linalg.norm(nacl.positions[0] - nacl.positions[1])) <= 4.5
    etoh_w = smiles.smiles_to_atoms("CCO.O")
    g1, g2 = [0, 1, 2, 4, 5, 6, 7, 8, 9], [3, 10, 11]
    d_frag = float(np.linalg.norm(etoh_w.positions[g1][:, None, :] - etoh_w.positions[g2][None, :, :], axis=-1).min())
    assert 3.0 <= d_frag <= 4.5

    # Analytical vs central finite-difference gradient check on smiles_ff.make_forcefield
    rng = np.random.default_rng(42)
    x0 = rng.normal(0.0, 1.2, size=(6, 3)).reshape(-1)
    ff = smiles_ff.make_forcefield(
        6,
        np.array([[0, 1], [1, 2], [2, 3], [1, 4], [2, 5]]),
        np.array([1.4, 1.35, 1.5, 1.1, 1.1]),
        np.array([[0, 1, 2], [1, 2, 3], [0, 1, 4], [3, 2, 5]]),
        np.array([2.09, 2.09, 2.09, 3.14159]),
        np.array([False, False, False, True]),
        np.array([[1, 0, 2, 4]]),
        np.array([[0, 2, 4, 5]]),
        np.array([1.0]),
        np.array([[0, 1, 2, 3]]),
        np.array([-1.0]),
        np.array([[4, 1, 2, 5]]),
        np.array([[0, 1, 2, 5]]),
        np.array([[0, 3], [4, 5]]),
        np.array([3.2, 2.6]),
        np.array([2.2, 1.8]),
    )
    _, g_ana = ff(x0)
    eps = 1e-6
    g_fd = np.zeros_like(x0)
    for k in range(len(x0)):
        xp, xm = x0.copy(), x0.copy()
        xp[k] += eps
        xm[k] -= eps
        g_fd[k] = (ff(xp)[0] - ff(xm)[0]) / (2.0 * eps)
    assert float(np.max(np.abs(g_ana - g_fd))) < 1e-5

    # Invalid SMILES must raise StructureError
    for bad in ("", "   ", "CC(", "CC)", "C1CC", "[CH4", "[Zz]", "c1cccc1", "CC(C)(C)(C)C", "CC=", "C11", ".CC", "CC.", "C=1CCC-1"):
        with pytest.raises(StructureError):
            smiles.smiles_to_atoms(bad)


def test_crystal_symmetry_and_phonon_reduction_across_crystal_systems():
    # 1. Diamond Si, rocksalt MgO, cubic perovskite SrTiO3 -> 48 cubic ops (m-3m)
    for name, n_eval_expected in (("si", 3), ("mgo", 5), ("srtio3", 9)):
        a = structures.load(name)
        ds = crystal_symmetry.find_crystal_symmetry(a.cell.array, a.get_scaled_positions(wrap=False), a.numbers)
        assert ds is not None
        assert len(ds["rotations"]) == 48
        assert ds["pointgroup"] == "m-3m"
        assert ds["crystal_system"] == "cubic"
        plan = phonon_symmetry.symmetry_plan(a, 2)
        assert plan is not None and plan.n_evaluations == n_eval_expected
        an = structures.analyze(a)
        assert an["spacegroup"] == ds["label"] and an["phonon_evaluations"] == n_eval_expected

    # 2. Rutile TiO2 -> tetragonal 4/mmm (16 ops)
    tio2 = structures.load("tio2")
    ds_tio2 = crystal_symmetry.find_crystal_symmetry(tio2.cell.array, tio2.get_scaled_positions(wrap=False), tio2.numbers)
    assert ds_tio2 is not None and len(ds_tio2["rotations"]) == 16
    assert ds_tio2["pointgroup"] == "4/mmm" and ds_tio2["crystal_system"] == "tetragonal"

    # 3. Hexagonal hcp Mg (24 ops, 6/mmm) & Cu(111) 1x1 slab cell (12 ops, -3m)
    mg = bulk("Mg", "hcp", a=3.2, c=5.2)
    ds_mg = crystal_symmetry.find_crystal_symmetry(mg.cell.array, mg.get_scaled_positions(wrap=False), mg.numbers)
    assert ds_mg is not None and len(ds_mg["rotations"]) == 24 and ds_mg["crystal_system"] == "hexagonal"

    cu111 = structures.load("cu111")
    cu_3d = cu111.copy()
    cu_3d.pbc = [True, True, True]
    ds_cu = crystal_symmetry.find_crystal_symmetry(cu_3d.cell.array, cu_3d.get_scaled_positions(wrap=False), cu_3d.numbers)
    assert ds_cu is not None and len(ds_cu["rotations"]) >= 12

    # 4. Unimodularly skewed primitive Si cell (det U = 1) still recovers all 48 m-3m operations via lattice reduction
    si = structures.load("si")
    u_skew = np.array([[1, 2, 3], [0, 1, 2], [0, 0, 1]], dtype=float)
    si_skew = Atoms(numbers=si.numbers, positions=si.positions, cell=u_skew @ si.cell.array, pbc=True)
    ds_skew = crystal_symmetry.find_crystal_symmetry(si_skew.cell.array, si_skew.get_scaled_positions(wrap=False), si_skew.numbers)
    assert ds_skew is not None and len(ds_skew["rotations"]) == 48 and ds_skew["pointgroup"] == "m-3m"

    # 5. Centrosymmetric triclinic (-1, 2 ops) vs rattled P1 cell (1 op -> symmetry_plan is None)
    tri_inv = Atoms("Si2", scaled_positions=[[0.13, 0.27, 0.41], [0.87, 0.73, 0.59]],
                    cell=[[4.1, 0.0, 0.0], [0.6, 4.7, 0.0], [0.4, 0.8, 5.3]], pbc=True)
    ds_inv = crystal_symmetry.find_crystal_symmetry(tri_inv.cell.array, tri_inv.get_scaled_positions(wrap=False), tri_inv.numbers)
    assert ds_inv is not None and len(ds_inv["rotations"]) == 2 and ds_inv["pointgroup"] == "-1"

    tri_p1 = Atoms("Si3", scaled_positions=[[0.0, 0.0, 0.0], [0.23, 0.19, 0.31], [0.61, 0.77, 0.44]],
                   cell=[[4.1, 0.0, 0.0], [0.6, 4.7, 0.0], [0.4, 0.8, 5.3]], pbc=True)
    ds_p1 = crystal_symmetry.find_crystal_symmetry(tri_p1.cell.array, tri_p1.get_scaled_positions(wrap=False), tri_p1.numbers)
    assert ds_p1 is not None and len(ds_p1["rotations"]) == 1 and ds_p1["pointgroup"] == "1"
    assert phonon_symmetry.symmetry_plan(tri_p1, 2) is None


def test_yaml_lite_parity_registry_and_cli_without_pyyaml(tmp_path, monkeypatch):
    reg_path = REPO_ROOT / "lam" / "data" / "lot_registry.yaml"
    raw_text = reg_path.read_text(encoding="utf-8")
    parsed = yaml_lite.safe_load(raw_text)
    assert isinstance(parsed, dict)
    assert set(parsed["levels_of_theory"]) == set(lot_registry.LOT_NAME_TO_ID)
    assert len(parsed["datasets"]) >= 8 and "validation_splits" in parsed  # the public registry keeps no split files

    try:
        import yaml as pyyaml
    except ImportError:
        pyyaml = None
    if pyyaml is not None:
        assert parsed == pyyaml.safe_load(raw_text)
        for yml_file in sorted((REPO_ROOT / "lam" / "eval").glob("*.yml")):
            txt = yml_file.read_text(encoding="utf-8")
            assert yaml_lite.safe_load(txt) == pyyaml.safe_load(txt)

    # Inline flow dicts/lists, escapes, `#` inside quotes, null/~, booleans, scientific floats
    sample = (
        "# Job config comment\n"
        "task: singlepoint\n"
        'structure: "smiles:CC(=O)O" # inline comment\n'
        "tags: [alpha, 'beta # not comment', \"line\\n2\", ~, null, true, false, -3, 1.25e-3]\n"
        "nested:\n"
        "  flow: {a: 1, b: [2, 3]}\n"
        "  items:\n"
        "    - name: first\n"
        "      ok: true\n"
        "    - second\n"
    )
    doc = yaml_lite.safe_load(sample)
    assert doc == {
        "task": "singlepoint",
        "structure": "smiles:CC(=O)O",
        "tags": ["alpha", "beta # not comment", "line\n2", None, None, True, False, -3, 1.25e-3],
        "nested": {"flow": {"a": 1, "b": [2, 3]}, "items": [{"name": "first", "ok": True}, "second"]},
    }

    # Block PyYAML and verify custom registry + `mena run job.yaml` work natively
    with monkeypatch.context() as mp:
        mp.setitem(__import__("sys").modules, "yaml", None)
        custom_reg = tmp_path / "custom_lot_registry.yaml"
        custom_reg.write_text(raw_text, encoding="utf-8")
        loaded_reg = lot_registry.load_lot_registry(custom_reg)
        assert loaded_reg == parsed

        fake_mol = {"id": "m", "title": "Mol", "backend": "lam4", "path": __file__, "available": True,
                    "domains": ["molecule"], "lots": ["omol_wb97mv"], "dtypes": ["float32"], "low_memory": True}
        mp.setattr(models, "choose", lambda req, dom: fake_mol)
        job_yaml = tmp_path / "job.yaml"
        job_yaml.write_text("task: singlepoint\nstructure: smiles:CCO\nparams: {}\nsystem: {charge: 0, spin: 1}\n", encoding="utf-8")
        assert cli.main(["run", str(job_yaml), "--dry-run", "--quiet"]) == 0


def test_full_end_to_end_with_rdkit_spglib_and_yaml_blocked(tmp_path, monkeypatch, capsys):
    import sys
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]

    # Verify pyproject.toml declares zero optional runtime extras (only `test`)
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert set(pyproject["project"]["optional-dependencies"].keys()) == {"test"}

    # Block all three formerly-optional packages simultaneously
    monkeypatch.setitem(sys.modules, "rdkit", None)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", None)
    monkeypatch.setitem(sys.modules, "spglib", None)
    monkeypatch.setitem(sys.modules, "yaml", None)

    # 1. Server features & `mena doctor` self-check cleanly with zero "not installed" noise
    feats = server._features()
    assert all(feats[k] is True for k in ("smiles", "symmetry", "yaml", "rdkit", "spglib"))
    fake_model = {"id": "m", "title": "M", "backend": "lam4", "path": __file__, "available": False, "domains": ["molecule"]}
    with monkeypatch.context() as mp:
        mp.setattr(models, "all_models", lambda: [fake_model])
        doctor.main()
    doc_out = capsys.readouterr().out
    assert "builtins ok (smiles, symmetry, yaml)" in doc_out
    assert "not installed" not in doc_out

    # 2. End-to-end SMILES singlepoint execution (caffeine & methyl radical)
    class _HarmonicCalc(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            pos = self.atoms.get_positions()
            disp = pos - pos.mean(axis=0, keepdims=True)
            self.results = {"energy": 0.75 * float((disp ** 2).sum()), "forces": -1.5 * disp}
            if self.atoms.pbc.all():
                self.results["stress"] = np.zeros(6, dtype=float)

    monkeypatch.setattr(backends, "build", lambda model, dec: _HarmonicCalc())
    for smi, exp_q, exp_s in (("Cn1cnc2c1c(=O)n(c(=O)n2C)C", 0, 1), ("[CH3]", 0, 2)):
        spec = validation.validate({"task": "singlepoint", "structure": {"format": "smiles", "text": smi},
                                    "system": {"charge": exp_q, "spin": exp_s}})
        ctx = runtime.Context({"spec": spec, "decision": {"device": "cpu", "dtype": "float32", "low_memory": False, "skin": 0.0},
                               "model": {"id": "m", "title": "M"}, "workdir": str(tmp_path / "sp")}, io.StringIO())
        res = singlepoint.run(ctx)
        assert np.isfinite(res["raw"]["energy_eV"]) and np.all(np.isfinite(res["raw"]["forces_eV_A"]))
        assert {f["name"] for f in res["files"]} == {"structure.xyz", "evaluated.extxyz"}
        from ase.io import read as ase_read
        ev = ase_read(tmp_path / "sp" / "evaluated.extxyz")
        assert np.allclose(ev.get_forces(), res["raw"]["forces_eV_A"]) and np.isclose(ev.get_potential_energy(), res["raw"]["energy_eV"])

    # 3. End-to-end symmetry-reduced phonons on SrTiO3 (9 supercell evaluations vs 31 full!)
    srtio3 = structures.load("srtio3")
    plan = phonon_symmetry.symmetry_plan(srtio3, 2)
    assert plan is not None and plan.n_evaluations == 9

    class _NullCtx:
        def check(self):
            pass

        def progress(self, *a, **k):
            pass

    calc = LennardJones(sigma=2.0, epsilon=0.1, rc=6.0, smooth=True)
    full_ph = Phonons(srtio3, calc, supercell=(2, 2, 2), delta=0.002, name=str(tmp_path / "ph_full"))
    full_ph.run()
    sym_ph = _LivePhonons(_NullCtx(), plan.n_evaluations, srtio3, calc, supercell=(2, 2, 2), delta=0.002,
                          name=str(tmp_path / "ph_sym"), plan=plan)
    sym_ph.measure()
    sym_ph.run()
    assert sym_ph.done == 9
    full_ph.read(acoustic=True)
    sym_ph.read(acoustic=True)
    bp = srtio3.cell.bandpath(npoints=25)
    assert np.abs(full_ph.get_band_structure(bp, verbose=False).energies[0] -
                  sym_ph.get_band_structure(bp, verbose=False).energies[0]).max() < 1e-8

    # 4. End-to-end YAML job loading referencing a SMILES molecule
    monkeypatch.setattr(models, "choose", lambda req, dom: {
        "id": "m", "title": "Mol", "backend": "lam4", "path": __file__, "available": True,
        "domains": ["molecule"], "lots": ["omol_wb97mv"], "dtypes": ["float32"], "low_memory": True,
    })
    job_yml = tmp_path / "aspirin_job.yml"
    job_yml.write_text("task: sp\nstructure: smiles:CC(=O)Oc1ccccc1C(=O)O\nruntime:\n  device: cpu\n", encoding="utf-8")
    assert cli.main(["run", str(job_yml), "--dry-run", "--quiet"]) == 0
