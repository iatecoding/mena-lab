"""Regressions for the pre-release robustness audit: levels of theory, odd inputs, job files, server and cancel."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from ase import Atoms
from ase.build import bcc110, bulk, fcc111, molecule, add_adsorbate
from ase.io import write

from mena_app import cli, export, models, planner, routing, runtime, server, structures, validation
from mena_app.tasks.vibrations import external_basis

LAM = {"id": "lam-400k", "title": "Mena-LAM", "backend": "lam", "path": __file__, "available": True,
       "domains": ["crystal", "slab", "molecule"], "rank": {"crystal": 0, "slab": 0, "molecule": 3},
       "lots": ["mp_pbe", "mp_pbeU", "omat_pbe", "omat_pbeU", "matpes_pbe", "matpes_r2scan", "omol_wb97mv"],
       "dtypes": ["float32", "float64"], "low_memory": True}
LAM5 = {"id": "lam5-v1", "title": "Mena-LAM 5", "backend": "lam4", "path": __file__, "available": True,
        "domains": ["molecule", "crystal", "slab"], "rank": {"molecule": 0, "crystal": 1, "slab": 1},
        "lots": ["mp_pbe", "mp_pbeU", "matpes_r2scan", "oc20_rpbe", "oc22_pbeU", "omc25_pbe", "omol_wb97mv"],
        "dtypes": ["float32", "float64"], "low_memory": True}


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(models, "all_models", lambda: [dict(LAM5), dict(LAM)])


def _xyz(a: Atoms) -> dict:
    buf = io.StringIO()
    write(buf, a, format="extxyz")
    return {"format": "extxyz", "text": buf.getvalue(), "name": "t"}


def _spec(structure, **runtime):
    return validation.validate({"task": "singlepoint", "structure": structure, "runtime": {"device": "cpu", **runtime}})


# ---- levels of theory ----

def test_file_lot_is_a_checked_request_not_a_silent_override(fake_models):
    si = bulk("Si", "diamond", 5.43)
    si.info["lot"] = "oc20_rpbe"
    a = structures.load(_xyz(si))
    assert "lot" not in a.info and a.info["file_lot"] == "oc20_rpbe"  # calculators never see it
    with pytest.raises(planner.PlanError, match=r"oc20_rpbe.*structure file.*--model lam5-v1"):
        planner.plan(_spec(_xyz(si)))  # auto model for a crystal is lam-400k, which lacks oc20_rpbe
    pl = planner.plan(_spec(_xyz(si), model="lam5-v1"))
    assert pl["decision"]["lot"] == "oc20_rpbe" and "structure file" in pl["decision"]["reasons"]["lot"]


def test_explicit_lot_beats_the_file(fake_models):
    si = bulk("Si", "diamond", 5.43)
    si.info["lot"] = "mp_pbe"
    pl = planner.plan(_spec(_xyz(si), lot="mp_pbeU"))
    assert pl["decision"]["lot"] == "mp_pbeU"


def test_lot_must_fit_the_structure_kind_and_is_case_insensitive(fake_models):
    with pytest.raises(planner.PlanError, match="periodic"):
        planner.plan(_spec({"format": "builtin", "text": "water"}, lot="mp_pbe"))
    slab = bcc110("Fe", (2, 2, 3), vacuum=8.0)
    with pytest.raises(planner.PlanError, match="2D slab"):
        planner.plan(_spec(_xyz(slab), lot="omol_wb97mv"))
    assert planner.plan(_spec({"format": "builtin", "text": "si"}, lot="omol_wb97mv"))["decision"]["lot"] == "omol_wb97mv"
    assert planner.plan(_spec({"format": "builtin", "text": "si"}, lot="MP_PBE"))["decision"]["lot"] == "mp_pbe"
    assert routing.canonical_lot("Mp_PbeU", LAM["lots"]) == "mp_pbeU"


def test_shipped_lam5_offers_only_distinct_validated_lots():
    shipped = {m["id"]: m for m in json.loads((models.paths.DATA_DIR / "models.json").read_text(encoding="utf-8"))["models"]}
    lots = shipped["lam5-v1"]["lots"]
    assert not {"omat_pbe", "matpes_pbe", "omat_pbeU", "openrxn_wb97xd", "spice_wb97md3"} & set(lots)
    assert {"mp_pbe", "mp_pbeU", "oc20_rpbe", "oc22_pbeU", "omol_wb97mv"} <= set(lots)


def test_metal_slab_with_adsorbate_is_not_an_oxide():
    fe = bcc110("Fe", (3, 3, 4), vacuum=8.0)
    add_adsorbate(fe, molecule("CO"), 2.0, "ontop")
    assert structures.analyze(fe)["hubbard_u"] is False
    assert structures.analyze(bulk("NiO", "rocksalt", 4.17))["hubbard_u"] is True


# ---- odd inputs ----

def test_vacuum_in_a_fully_periodic_cell_is_recognised():
    slab = fcc111("Cu", (2, 2, 4), vacuum=8.0)
    slab.pbc = True
    a = structures.load(_xyz(slab))
    assert list(a.pbc) == [True, True, False] and structures.analyze(a)["domain"] == "slab"
    w = molecule("H2O")
    w.center(vacuum=12.0)
    w.pbc = True
    assert structures.analyze(structures.load(_xyz(w)))["domain"] == "molecule"
    si = structures.load(_xyz(bulk("Si", "diamond", 5.43, cubic=True)))
    assert si.pbc.all()


def test_elements_without_training_data_are_refused(fake_models):
    og = Atoms("OgH2", positions=[[0, 0, 0], [0, 0, 1.8], [0, 1.8, 0]])
    with pytest.raises(planner.PlanError, match="not trained on Og"):
        planner.plan(_spec(_xyz(og)))
    pu_mol = Atoms("PuO2", positions=[[0, 0, 0], [0, 0, 1.8], [0, 0, -1.8]])
    with pytest.raises(planner.PlanError, match="Pu"):
        planner.plan(_spec(_xyz(pu_mol)))
    planner.plan(_spec(_xyz(bulk("Pu", "fcc", 4.6))))  # actinides up to Pu are covered for materials
    with pytest.raises(structures.StructureError, match="dummy"):
        structures.load(_xyz(Atoms("XH", positions=[[0, 0, 0], [0, 0, 1]])))


def test_overlapping_atoms_are_refused(fake_models):
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0]])
    with pytest.raises(planner.PlanError, match="overlapping"):
        planner.plan(_spec(_xyz(h2)))


def test_files_from_any_editor_load(tmp_path):
    text = "3\ncafé\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n"
    (tmp_path / "latin.xyz").write_bytes(text.encode("latin-1"))
    (tmp_path / "bom.xyz").write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    for name in ("latin.xyz", "bom.xyz"):
        assert len(structures.load(str(tmp_path / name))) == 3


def test_disordered_cif_is_refused():
    cif = """data_x
_cell_length_a 5.6
_cell_length_b 5.6
_cell_length_c 5.6
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0 0 0 0.5
K1 K 0 0 0 0.5
Cl1 Cl 0.5 0.5 0.5 1.0
"""
    with pytest.raises(structures.StructureError, match="partially occupied"):
        structures.load({"format": "cif", "text": cif})


def test_nearly_linear_molecule_keeps_its_bending_modes():
    for ang, n_ext in ((180.0, 5), (179.9999, 5), (179.0, 6), (120.0, 6)):
        t = np.radians(ang / 2)
        co2 = Atoms("OCO", positions=[[-1.16 * np.sin(t), 1.16 * np.cos(t), 0], [0, 0, 0], [1.16 * np.sin(t), 1.16 * np.cos(t), 0]])
        assert external_basis(co2).shape[1] == n_ext, ang


# ---- job files ----

def test_job_files_refuse_typos_and_accept_nested_blocks(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"task": "relax", "structure": "si", "max_stpes": 2}), encoding="utf-8")
    with pytest.raises(validation.ValidationError, match="max_stpes"):
        cli._flat_config(cli._load_config(str(bad)), "relax", str(bad))
    good = {"task": "relax", "structure": "si", "params": {"max_steps": 2}, "runtime": {"device": "cpu"}}
    flat = cli._flat_config(good, "relax", "good.json")
    assert flat["max_steps"] == 2 and flat["device"] == "cpu"


def test_run_folder_job_json_points_at_a_copied_input(tmp_path):
    src = tmp_path / "elsewhere" / "my water.xyz"
    src.parent.mkdir()
    src.write_text("3\nw\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n", encoding="utf-8")
    spec = validation.validate({"task": "singlepoint", "structure": structures.from_path(str(src))})
    run = tmp_path / "run"
    run.mkdir()
    p1, p2 = export.save_inputs(spec, run)
    cfg = json.loads(export.to_json(spec, p1, p2))
    assert p2 is None and cfg["structure"] == "input.xyz" and (run / "input.xyz").read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


# ---- server and cancel ----

def test_server_rejects_non_object_bodies_and_saves_settings_atomically(fake_models, monkeypatch, tmp_path):
    monkeypatch.setenv("MENA_HOME", str(tmp_path / "home"))
    srv, app = server.create("127.0.0.1", port=18961, tries=30)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{app.port}"
    hdr = {"X-Mena": "1", "Origin": base, "Content-Type": "application/json"}

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=hdr, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    try:
        assert post("/api/structure", [1])[0] == 400
        code, _ = post("/api/settings", {"default_model": {"crystal": "lam5-v1", "foo": "lam5-v1"}})
        assert code == 400
        assert models.settings.load()["default_model"] == {}  # nothing saved from the rejected request
        assert post("/api/settings", {"default_model": {"crystal": "lam5-v1"}})[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_cancel_file_stops_the_worker_on_every_os(tmp_path):
    ctx = runtime.Context({"spec": {"params": {}, "system": {}}, "decision": {}, "model": {}, "workdir": str(tmp_path)}, io.StringIO())
    ctx.check()
    (tmp_path / runtime.CANCEL_FILE).write_text("cancel\n", encoding="utf-8")
    assert ctx.cancelled
    with pytest.raises(runtime.Cancelled):
        ctx.check()


def test_windows_line_endings_are_not_doubled(tmp_path):
    src = tmp_path / "crlf.xyz"
    src.write_bytes(b"3\r\nw\r\nO 0 0 0\r\nH 0.96 0 0\r\nH -0.24 0.93 0\r\n")
    spec = validation.validate({"task": "singlepoint", "structure": structures.from_path(str(src))})
    assert "\r" not in spec["structure"]["text"]
    run = tmp_path / "run"
    run.mkdir()
    export.save_inputs(spec, run)
    assert b"\r\r" not in (run / "input.xyz").read_bytes() and len(structures.load(str(run / "input.xyz"))) == 3
