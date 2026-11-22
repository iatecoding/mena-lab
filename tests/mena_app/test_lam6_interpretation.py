"""lam6 in the app: registration and routing, the opt-in interpretation read-out (calculator, single-point task, CLI),
and the SMILES fix for planar radicals. Model-dependent tests skip when the lam6 weights are not on disk."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from mena_app import catalog, cli, models, paths, planner, report, runner, structures, validation
from mena_app.tasks import interpretation

REPO_ROOT = Path(__file__).resolve().parents[2]


def _lam6_path() -> Path | None:
    m = next((x for x in models.all_models() if x["id"] == "lam6-v1"), None)
    return Path(m["path"]) if m and m.get("available") else None


LAM6_PATH = _lam6_path()  # resolved once: other tests drop dummy weight files into the shared test home later
lam6 = pytest.mark.skipif(LAM6_PATH is None, reason="lam6-v1 weights not installed")


def _declared() -> dict:
    return {m["id"]: m for m in json.loads((paths.DATA_DIR / "models.json").read_text(encoding="utf-8"))["models"]}


def test_lam6_entry_is_downloadable_and_clean():
    m = _declared()["lam6-v1"]
    assert m["backend"] == "lam4" and m.get("release") != "preview" and m["interpretable"] is True
    assert m["rank"] == {"crystal": 0, "slab": 0, "molecule": 1}
    assert {"molecule", "crystal", "slab"} <= set(m["domains"])
    assert len(m["urls"]) >= 2
    assert m["urls"][0].startswith("https://github.com/iatecoding/mena-lab/releases/download/v0.1.0/")
    assert "huggingface.co/iatecoding/Mena-LAM-6/" in m["urls"][1]
    assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"])
    assert m["file"] == "mena-lam6-v1.pt"
    # the public build audits every copied file: the shipped list must name no internal paths
    audit = pytest.importorskip("release.build_public_repo").AUDIT  # only in the research tree
    assert not audit.search((paths.DATA_DIR / "models.json").read_text(encoding="utf-8"))


def test_missing_preview_model_explains_itself(monkeypatch):
    m = {
        **_declared()["lam6-v1"],
        "id": "lam6-preview",
        "release": "preview",
        "urls": [],
        "available": False,
        "downloadable": False,
        "path": "/nowhere/mena-lam6-preview.pt",
    }
    monkeypatch.setattr(models, "all_models", lambda: [m])
    with pytest.raises(models.ModelError) as exc:
        models.choose("lam6-preview", "crystal")
    assert "preview" in str(exc.value) and "mena models add lam6-preview" in str(exc.value)


def test_auto_routing_lam6_materials_lam5_molecules(monkeypatch, tmp_path):
    declared = list(_declared().values())
    fake = [{**m, "available": True, "downloadable": False, "path": str(tmp_path / f"{m['id']}.pt")} for m in declared]
    monkeypatch.setattr(models, "all_models", lambda: fake)
    assert models.choose("auto", "crystal")["id"] == "lam6-v1"
    assert models.choose("auto", "slab")["id"] == "lam6-v1"
    assert models.choose("auto", "molecule")["id"] == "lam5-v1"
    assert models.choose("lam6-v1", "molecule")["id"] == "lam6-v1"
    assert models.choose("lam-400k", "crystal")["id"] == "lam-400k"


def test_interpret_param_and_cli_alias():
    sp = catalog.task("singlepoint")
    p = next(x for x in sp["params"] if x["name"] == "interpret")
    assert p["type"] == "bool" and p["default"] is True
    ap = cli.build_parser()
    assert ap.parse_args(["sp", "water", "--explain"]).interpret is True
    assert ap.parse_args(["sp", "water", "--no-interpret"]).interpret is False
    spec = cli.spec_from_args("singlepoint", ap.parse_args(["sp", "water", "--no-explain"]))
    assert spec["params"]["interpret"] is False
    assert validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}})["params"]["interpret"] is True


def _fake_interp(periodic: bool, mult: int = 1, lam6_heads: bool = True) -> dict:
    n = 2 if periodic else 3
    heads = {"charges": True, "spins": True, "pauli_overlap": lam6_heads, "spin_exchange": lam6_heads,
             "homo_lumo_gap": lam6_heads, "quadrupole_l2": lam6_heads, "spin_descriptor": lam6_heads}
    return {
        "domain": "crystal" if periodic else "molecule", "periodic": periodic, "multiplicity": mult, "heads": heads,
        "charges_e": None if periodic else np.array([-0.4, 0.2, 0.2]),
        "spin_density_e": None if periodic else np.array([0.6, 0.2, 0.2]) * (mult - 1),
        "unpaired_u_rad": np.full(n, 0.3) if lam6_heads else None,
        "e_pauli_atom_ev": np.full(n, 0.05) if lam6_heads else None,
        "e_exchange_atom_ev": None if (periodic or not lam6_heads) else np.array([-0.1, 0.0, 0.0]),
        "homo_lumo_gap_ev": None if (periodic or not lam6_heads) else [9.0, 8.0],
    }


def test_interpretation_panel_molecule_crystal_and_lam5():
    water = structures.load("water")
    radical = interpretation.build(_fake_interp(False, mult=2), water, "Mena-LAM-6")
    keys = [it["key"] for it in radical["panel"]["items"]]
    assert keys == ["gap", "unpaired", "charges", "spin", "pauli", "exchange"]
    for it in radical["panel"]["items"]:
        assert it["meaning"] and it["caveat"] and it["short"] and it["trust"] in interpretation.TRUST
    assert radical["panel"]["items"][0]["value"] == 8.0  # open shell: the smaller (β) gap
    assert set(radical["frame"]) == {"unpaired", "pauli"} and set(radical["columns"]) == {"unpaired", "pauli"}

    closed = interpretation.build(_fake_interp(False, mult=1), water, "Mena-LAM-6")
    assert [it["key"] for it in closed["panel"]["items"]] == ["gap", "unpaired", "charges", "pauli"]

    si = bulk("Si", "diamond", a=5.43)
    crys = interpretation.build(_fake_interp(True), si, "Mena-LAM-6")
    assert [it["key"] for it in crys["panel"]["items"]] == ["pauli"]
    assert crys["panel"]["items"][0]["unit"] == "eV/atom"
    assert any("crystals" in n for n in crys["panel"]["notes"])

    lam5 = interpretation.build(_fake_interp(False, lam6_heads=False), water, "Mena-LAM-s")
    assert [it["key"] for it in lam5["panel"]["items"]] == ["charges"]
    assert any("Mena-LAM-6 adds" in n for n in lam5["panel"]["notes"])


def test_report_frame_extra_colourings():
    water = structures.load("water")
    fr = report.frame(water, extra={"unpaired": [0.1, 0.0, 0.0], "pauli": [0.0, 0.0, 0.0]})
    assert fr["unpaired"] == [0.1, 0.0, 0.0] and "pauli" not in fr  # all-zero arrays are dropped


def test_smiles_radical_centres_are_planar():
    a = structures.load("smiles:C=C[CH2]")
    d = a.get_all_distances()
    hs = [i for i in range(len(a)) if a.numbers[i] == 1 and d[2, i] < 1.2]
    for h in hs:
        dih = a.get_dihedral(h, 2, 1, 0) % 180.0
        assert min(dih, 180.0 - dih) < 15.0  # conjugated with the C=C, not twisted 90 degrees
    m = structures.load("smiles:[CH3]")
    assert abs(np.linalg.det(m.positions[1:] - m.positions[0])) < 0.05  # methyl radical is planar


@lam6
def test_calculator_interpretation_is_opt_in():
    from lam4.calc import MenaCalculator

    calc = MenaCalculator(str(LAM6_PATH), device="cpu")
    w = structures.load("water")
    w.info.update(charge=0, spin=1)
    w.calc = calc
    e = w.get_potential_energy()
    assert "interpretation" not in calc.results
    it = calc.interpretation(w)
    assert it["heads"]["homo_lumo_gap"] and it["heads"]["pauli_overlap"]
    assert 8.0 < min(it["homo_lumo_gap_ev"]) < 16.0
    assert abs(float(np.sum(it["charges_e"]))) < 1e-4 and it["charges_e"][0] < 0  # O negative, neutral total
    assert np.all(it["e_pauli_atom_ev"] >= 0)
    assert w.get_potential_energy() == e  # reading the heads does not change the cached result

    calc_i = MenaCalculator(str(LAM6_PATH), device="cpu", interpret=True)
    w2 = w.copy()
    w2.calc = calc_i
    assert abs(w2.get_potential_energy() - e) < 1e-6
    assert calc_i.results["interpretation"]["homo_lumo_gap_ev"] is not None

    si = bulk("Si", "diamond", a=5.43)
    si.calc = calc
    si.get_potential_energy()
    its = calc.interpretation(si)
    assert its["periodic"] and its["homo_lumo_gap_ev"] is None and its["e_exchange_atom_ev"] is None
    assert its["e_pauli_atom_ev"] is not None and len(its["e_pauli_atom_ev"]) == 2


@lam6
@pytest.mark.parametrize("interpret", [True, False])
def test_singlepoint_job_shows_interpretation(tmp_path, interpret):
    spec = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"},
                                "params": {"interpret": interpret}, "runtime": {"model": "lam6-v1", "device": "cpu"}})
    pl = planner.plan(spec)
    pl["model"] = {**pl["model"], "path": str(LAM6_PATH)}
    res = runner.run(pl, spec, str(tmp_path), lambda e: None, threading.Event())
    assert res["status"] == "done", res.get("error")
    r = res["result"]
    if interpret:
        keys = [it["key"] for it in r["interpretation"]["items"]]
        assert "gap" in keys and "pauli" in keys
        assert "unpaired" in r["viewer"]["frames"][0] or "pauli" in r["viewer"]["frames"][0]
        assert "Pauli (eV)" in r["tables"][0]["columns"]
        saved = json.loads((tmp_path / "result.json").read_text()) if (tmp_path / "result.json").exists() else None
        assert saved is None or "interpretation" in saved.get("result", {})
    else:
        assert "interpretation" not in r
