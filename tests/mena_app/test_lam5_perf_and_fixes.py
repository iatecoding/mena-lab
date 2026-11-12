"""Regression and integration tests for Mena-LAM-s Performance tab data, molecular live benchmarks, and mena_app bug fixes."""

from __future__ import annotations

import ast
import io
import json
import os
import shlex
import subprocess
import sys
import threading
import types
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read as ase_read

from mena_app import (
    backends,
    calibrate,
    catalog,
    cli,
    costmodel,
    export,
    jobs,
    models,
    paths,
    planner,
    routing,
    runner,
    runtime,
    server,
    structures,
    validation,
)
from mena_app.tasks import benchmark

REPO_ROOT = Path(__file__).resolve().parents[2]


class PairwiseSpringCalc(Calculator):
    """Fast conservative calculator for molecular clusters: soft harmonic tether around initial geometry."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, k: float = 0.8, skin: float = 0.0):
        super().__init__()
        self.k = k
        self.skin = skin
        self.model = object()
        self._nl_cache = None
        self._last_state = None
        self._p0: np.ndarray | None = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        if self._p0 is None or self._p0.shape != pos.shape:
            self._p0 = pos.copy()
        disp = pos - self._p0
        energy = 0.5 * self.k * float((disp * disp).sum())
        forces = -self.k * disp
        self.results = {"energy": energy, "forces": forces}


def test_benchmarks_json_lam5_molecules_and_architecture_status():
    """Verify ground-truth Mena-LAM-s molecular numbers, the model callout, the crystal speed card, and backward compatibility."""
    b = json.loads((paths.DATA_DIR / "benchmarks.json").read_text(encoding="utf-8"))

    # Backward-compatibility invariants (used by test_pure.py and legacy periodic sections)
    assert "hardware_banner" in b and "dynamat" in b["hardware_banner"] and "diamond" in b["hardware_banner"]
    assert all("atoms_per_s" in r and "cps" in r for r in b["diamond"]["rows"])
    assert "headline" in b and "dynamat" in b and "showcase" in b

    # Architecture & checkpoint status callout
    arch = b["architecture_status"]
    assert arch["molecules"]["status"] == "active"
    assert "lam5-v1" in arch["molecules"]["badge"]
    assert "IR & Raman" in arch["molecules"]["body"]
    assert arch["materials"]["status"] == "preview"
    assert "lam-400k" in arch["materials"]["body"]
    assert "44,670" in arch["materials"]["body"] and "65,536" in arch["materials"]["body"]
    assert "Settings" in arch["materials"]["body"]  # how to choose Mena-LAM-s for crystals

    # Mena-LAM-s molecular section: measured numbers
    mol = b["molecules_lam5"]
    assert len(mol["headline"]) == 4
    plf_rows = mol["plf547"]["rows"]
    assert plf_rows[0]["name"].startswith("Mena-LAM-s") and plf_rows[0]["mae"] == pytest.approx(0.297) and plf_rows[0]["mena"] is True

    t2_by_n = {row["n"]: row for row in mol["tier2_scaling"]["rows"]}
    assert {24, 96, 288, 1024, 2048, 4096, 8192, 16384} <= set(t2_by_n)
    assert t2_by_n[1024]["atoms_per_s"] == 13210 and "0.48 GB" in t2_by_n[1024]["t2_gb"]
    assert t2_by_n[8192]["atoms_per_s"] == 11727 and t2_by_n[8192]["t1_ms"] is None and "4.89 GB" in t2_by_n[8192]["t2_gb"]
    assert t2_by_n[16384]["atoms_per_s"] == 11309 and "5.30 GB" in t2_by_n[16384]["t2_gb"]

    cpu_facts = mol["cpu_and_workflows"]["facts"]
    assert "0.16 MB/atom" in cpu_facts["CPU low-memory mode (3,000 atoms)"]
    assert "+0.47 GB" in cpu_facts["CPU low-memory mode (3,000 atoms)"] and "0.84 GB" in cpu_facts["CPU low-memory mode (3,000 atoms)"]
    assert "+1.31 GB" in cpu_facts["CPU low-memory mode (8,193 atoms)"] and "1.68 GB" in cpu_facts["CPU low-memory mode (8,193 atoms)"]
    assert "30.05×" in cpu_facts["GPU relax_batch (24× L-BFGS, 35 steps)"]
    assert "0.508 s" in cpu_facts["Caffeine Hessian + IR + Raman"]

    acc_facts = mol["accuracy_and_spectroscopy"]["facts"]
    assert "0.205" in acc_facts["TorsionNet500 (DLPNO-CCSD(T)/CBS)"]
    assert "0.252" in acc_facts["Folmsbee 44 conformer sets"]
    assert "4.78" in acc_facts["GMTKN55 v2 (Rowan-20 main-group)"]
    assert "1.640" in acc_facts["OMol25 S2EF val_sub5k & Biomolecules"] and "0.340" in acc_facts["OMol25 S2EF val_sub5k & Biomolecules"]
    assert "0.9997" in acc_facts["Dipole & polarizability (def2-TZVPD)"]
    assert "5.58" in acc_facts["QVib 1,366 conformers & IR spectra"] and "0.8761" in acc_facts["QVib 1,366 conformers & IR spectra"]

    # Mena-LAM-s on crystals card
    mat_prev = b["materials_tier2_preview"]
    assert "44,670 atoms/s" in mat_prev["facts"]["8,192-atom Silicon (E+F+Voigt Stress)"]
    assert "5.80 GB" in mat_prev["facts"]["65,536-atom Silicon supercell"]
    assert "0.2992" in mat_prev["facts"]["PhononDB-103 thermal conductivity"]

    # Verify /api/benchmarks HTTP endpoint serves the updated payload cleanly
    srv, app = server.create("127.0.0.1", port=18941, tries=30)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{app.port}/api/benchmarks", timeout=3) as resp:
            assert resp.status == 200
            served = json.loads(resp.read().decode("utf-8"))
            assert "molecules_lam5" in served and "architecture_status" in served
    finally:
        app.jobs.shutdown()
        srv.shutdown()
        srv.server_close()


def test_tile_system_molecular_grid_non_overlapping_and_lam5_routing(monkeypatch, tmp_path):
    """Non-periodic molecules tile onto a clash-free 3D grid with pbc=False and route to Mena-LAM-s (`lam5-v1`)."""
    assert "molecule" in catalog.task("benchmark")["domains"]

    for name, n_unit in (("water", 3), ("ethanol", 9)):
        unit = structures.load(name)
        tiled = structures.tile_system(unit, (3, 3, 3))
        assert len(tiled) == 27 * n_unit
        assert not any(tiled.pbc)
        assert np.allclose(tiled.cell.array, 0.0)
        assert tiled.info["charge"] == 0 and tiled.info["spin"] == 1
        an = structures.analyze(tiled)
        assert an["domain"] == "molecule"
        assert an["warnings"] == []

        # Check intermolecular distances: every atom in copy m is >= 3.0 A from every atom in copy m' != m
        coords = tiled.positions.reshape(27, n_unit, 3)
        min_inter = min(
            float(np.linalg.norm(coords[i][:, None, :] - coords[j][None, :, :], axis=-1).min())
            for i in range(27)
            for j in range(i + 1, 27)
        )
        assert min_inter >= 3.0, f"{name} intermolecular distance {min_inter:.3f} A < 3.0 A"

    # Charged & open-shell molecular tiling preserves total charge and electron-spin parity
    nh4_grid = structures.tile_system(structures.load("smiles:[NH4+]"), (2, 2, 2))
    assert nh4_grid.info["charge"] == 8 and nh4_grid.info["spin"] == 1
    ch3_grid = structures.tile_system(structures.load("smiles:[CH3]"), (3, 1, 1))
    nelec_ch3 = int(ch3_grid.numbers.sum()) - ch3_grid.info["charge"]
    assert ch3_grid.info["charge"] == 0 and ch3_grid.info["spin"] == 4
    assert nelec_ch3 % 2 == (ch3_grid.info["spin"] - 1) % 2

    # Periodic crystal still uses unit.repeat(r) with full 3D PBC
    si_sup = structures.tile_system(structures.load("si"), (2, 2, 2))
    assert len(si_sup) == 16 and all(si_sup.pbc) and si_sup.get_volume() > 100.0

    # Routing & planner: molecular benchmark routes to lam5-v1 (`lam4` backend) and estimates condensed cluster edges
    monkeypatch.setenv("MENA_HOME", str(tmp_path / "home"))
    declared = json.loads((REPO_ROOT / "mena_app" / "data" / "models.json").read_text(encoding="utf-8"))["models"]
    fake_all = [{**m, "available": True, "path": str(tmp_path / f"{m['id']}.pt")} for m in declared]
    monkeypatch.setattr(models, "all_models", lambda: fake_all)

    sp_bench_w = validation.validate(
        {"task": "benchmark", "structure": {"format": "builtin", "text": "water"}, "params": {"mode": "visual_md", "max_atoms": 600}}
    )
    pl_w = planner.plan(sp_bench_w)
    assert pl_w["analysis"]["domain"] == "molecule"
    assert pl_w["model"]["id"] == "lam5-v1"
    assert pl_w["model"]["backend"] == "lam4"
    assert sp_bench_w["system"] == {"charge": 0, "spin": 1}

    _, x_edges = routing.working_atoms(sp_bench_w, pl_w["analysis"])
    assert x_edges >= 600 * 28  # condensed 3D droplet edge estimate rather than isolated 3-atom water (6 edges)
    assert routing.estimated_calls(sp_bench_w, pl_w["analysis"]) == 23  # 1 cold call + 22 NVE steps

    # With M3 Max calibration data loaded explicitly, 600-atom molecular visual_md picks MPS over CPU
    cal_m3 = json.loads((paths.DATA_DIR / "calibration_m3max.json").read_text(encoding="utf-8"))
    dec_m3 = routing.route(sp_bench_w, pl_w["analysis"], ["cpu", "mps"], 24.0, cal_m3, pl_w["model"])
    assert dec_m3["device"] == "mps"


def test_molecular_benchmark_visual_md_ladder_and_calibrate_bench_point(tmp_path, monkeypatch):
    """Run molecular `visual_md` (small & >MAX_VIS_ATOMS), `ladder`, and `calibrate.bench_point` verifying NVE stability & clash-free coords."""
    monkeypatch.setattr(backends, "build", lambda model, dec: PairwiseSpringCalc(k=0.8, skin=dec.get("skin", 0.0)))

    # 1. Small water droplet visual_md (81 atoms, <= MAX_VIS_ATOMS)
    spec_small = validation.validate(
        {
            "task": "benchmark",
            "structure": {"format": "builtin", "text": "water"},
            "params": {"mode": "visual_md", "max_atoms": 81, "steps": 6},
            "system": {"charge": 0, "spin": 1},
        }
    )
    out_small = tmp_path / "vis_small"
    out_small.mkdir()
    ctx_small = runtime.Context(
        {
            "spec": spec_small,
            "decision": {"device": "cpu", "dtype": "float32", "low_memory": False, "skin": 0.6, "lot": "auto"},
            "model": {"id": "lam5-v1", "title": "Mena-LAM-s", "backend": "lam4"},
            "workdir": str(out_small),
        },
        io.StringIO(),
    )
    res_small = benchmark.run(ctx_small)
    assert res_small["raw"]["n"] == 81
    assert res_small["raw"]["edges"] > 400
    assert "Molecular cluster" in res_small["facts"]
    drift_tile = next(t for t in res_small["tiles"] if t["key"] == "drift")
    assert drift_tile["value"] < 0.5 and drift_tile["status"] == "pass"

    # Saved trajectory.extxyz file must load cleanly via ASE (no private _SphericalWall serialization failure!)
    traj_name = res_small["files"][0]["name"]
    assert traj_name == "trajectory.extxyz"
    traj_frames = ase_read(str(out_small / traj_name), index=":")
    assert len(traj_frames) == 7 and all(len(fr) == 81 and not any(fr.pbc) for fr in traj_frames)

    # 2. Large water droplet visual_md (> MAX_VIS_ATOMS) exercises whole-molecule central subsampling
    spec_big = validation.validate(
        {
            "task": "benchmark",
            "structure": {"format": "builtin", "text": "water"},
            "params": {"mode": "visual_md", "max_atoms": 1950, "steps": 3},
            "system": {"charge": 0, "spin": 1},
        }
    )
    out_big = tmp_path / "vis_big"
    out_big.mkdir()
    ctx_big = runtime.Context(
        {
            "spec": spec_big,
            "decision": {"device": "cpu", "dtype": "float32", "low_memory": False, "skin": 0.6, "lot": "auto"},
            "model": {"id": "lam5-v1", "title": "Mena-LAM-s", "backend": "lam4"},
            "workdir": str(out_big),
        },
        io.StringIO(),
    )
    res_big = benchmark.run(ctx_big)
    assert res_big["raw"]["n"] > benchmark.MAX_VIS_ATOMS
    n_vis = len(res_big["viewer"]["frames"][0]["positions"])
    assert 0 < n_vis <= 1002 and n_vis % 3 == 0  # whole H2O molecules preserved!
    assert any("60 FPS" in n for n in res_big["notes"])

    # 3. Direct calibrate.bench_point + benchmark ladder mode on ethanol
    pt = calibrate.bench_point(
        {
            "kind": "bench",
            "model": {"id": "lam5-v1", "title": "Mena-LAM-s", "backend": "lam4"},
            "device": "cpu",
            "structure": {"format": "builtin", "text": "ethanol"},
            "repeat": [3, 3, 2],
            "low_memory": False,
            "steps": 2,
        }
    )
    assert pt["n"] == 9 * 18 == 162
    assert pt["edges"] > 500
    assert pt["atoms_per_ms"] > 0

    def fake_isolated_python(model, args, stdin_text, timeout=1800):
        req = json.loads(stdin_text)
        out = json.dumps(calibrate.bench_point(req))
        return subprocess.CompletedProcess(args, 0, stdout=out + "\n", stderr="")

    monkeypatch.setattr(runner, "isolated_python", fake_isolated_python)
    spec_ladder = validation.validate(
        {
            "task": "benchmark",
            "structure": {"format": "builtin", "text": "ethanol"},
            "params": {"mode": "ladder", "max_atoms": 200, "steps": 2, "compare_low_memory": False},
            "system": {"charge": 0, "spin": 1},
        }
    )
    out_lad = tmp_path / "ladder_etoh"
    out_lad.mkdir()
    ctx_lad = runtime.Context(
        {
            "spec": spec_ladder,
            "decision": {"device": "cpu", "dtype": "float32", "low_memory": False, "skin": 0.6, "lot": "auto"},
            "model": {"id": "lam5-v1", "title": "Mena-LAM-s", "backend": "lam4", "low_memory": True},
            "workdir": str(out_lad),
        },
        io.StringIO(),
    )
    res_lad = benchmark.run(ctx_lad)
    assert len(res_lad["raw"]["rows"]) >= 1
    assert res_lad["raw"]["rows"][0]["edges"] > 200


def test_export_smiles_cli_python_json_and_neb_slot2(tmp_path, monkeypatch):
    """WebApp SMILES structures (including `name='smiles'`, shell chars, `smiles:` prefix, and NEB slot 2) export valid CLI, JSON & Python."""

    class StubLam4Calc(PairwiseSpringCalc):
        @classmethod
        def from_checkpoint(cls, path, **kw):
            return cls(skin=kw.get("skin", 0.0))

    mod_lam4 = types.ModuleType("lam4")
    mod_lam4.MenaCalculator = StubLam4Calc
    monkeypatch.setitem(sys.modules, "lam4", mod_lam4)

    lam5_model = {
        "id": "lam5-v1",
        "title": "Mena-LAM-s",
        "backend": "lam4",
        "path": "/tmp/mena_lam5_v1_release.pt",
        "lots": models.DEFAULT_LOTS["lam4"],
        "dtypes": ["float32", "float64"],
        "low_memory": True,
    }

    # 1. Redundant `smiles:` prefix and surrounding whitespace normalized cleanly in validation
    sp_prefixed = validation.validate(
        {"task": "singlepoint", "structure": {"format": "smiles", "text": "  smiles:CC(=O)Oc1ccccc1C(=O)O \n", "name": "smiles"}}
    )
    assert sp_prefixed["structure"]["text"] == "CC(=O)Oc1ccccc1C(=O)O"
    with pytest.raises(validation.ValidationError):
        validation.validate({"task": "singlepoint", "structure": {"format": "smiles", "text": "smiles:   "}})

    # 2. Shell-sensitive SMILES (`\`, `(`, `)`, `[`, `+`) round-trips through shlex -> CLI parser -> structures.load
    for smi, q_exp in ((r"C/C=C\C", 0), ("[NH4+]", 1), ("CC(=O)Oc1ccccc1C(=O)O", 0)):
        sp = validation.validate(
            {"task": "singlepoint", "structure": {"format": "smiles", "text": smi, "name": "smiles"}, "system": {"charge": q_exp, "spin": 1}}
        )
        cli_line = export.to_cli(sp)
        tokens = shlex.split(cli_line)
        assert tokens[0] == "mena" and tokens[1] == "singlepoint" and tokens[2] == f"smiles:{smi}"
        ns = cli.build_parser().parse_args(tokens[1:])
        rt_spec = cli.spec_from_args("singlepoint", ns)
        assert rt_spec["structure"]["format"] == "smiles" and rt_spec["structure"]["text"] == smi

        # JSON config round-trip
        cfg = json.loads(export.to_json(sp))
        assert cfg["structure"] == f"smiles:{smi}"

        # Python export uses `load_structure('smiles:...')` (NEVER `ase.io.read('smiles')`) and executes cleanly!
        an = structures.analyze(structures.load(sp["structure"]))
        dec = routing.route(sp, an, ["cpu"], 32.0, {"points": [], "source": "none"}, lam5_model)
        py_code = export.to_python(sp, {"analysis": an, "model": lam5_model, "decision": dec}, "0.1.0")
        assert "from ase.io import read" not in py_code
        assert "from mena_app.structures import load as load_structure" in py_code
        assert f"load_structure({'smiles:' + smi!r})" in py_code
        ast.parse(py_code)
        exec(compile(py_code, "<smiles_sp>", "exec"), {"__name__": "__smiles_sp_test__"})

        # Explicit override_path='smiles:...' also uses load_structure
        py_override = export.to_python(sp, {"analysis": an, "model": lam5_model, "decision": dec}, "0.1.0", path=f"smiles:{smi}")
        assert f"load_structure({'smiles:' + smi!r})" in py_override

    # 3. NEB with SMILES for BOTH slot 1 (trans-2-butene) and slot 2 (cis-2-butene)
    sp_neb = validation.validate(
        {
            "task": "neb",
            "structure": {"format": "smiles", "text": "C/C=C/C", "name": "smiles"},
            "structure2": {"format": "smiles", "text": r"C/C=C\C", "name": "smiles"},
            "params": {"images": 3, "max_steps": 1, "relax_ends": False},
            "system": {"charge": 0, "spin": 1},
        }
    )
    cli_neb = export.to_cli(sp_neb)
    neb_tokens = shlex.split(cli_neb)
    assert neb_tokens[2] == "smiles:C/C=C/C" and neb_tokens[3] == r"smiles:C/C=C\C"

    an_neb = structures.analyze(structures.load(sp_neb["structure"]))
    dec_neb = routing.route(sp_neb, an_neb, ["cpu"], 32.0, {"points": [], "source": "none"}, lam5_model)
    py_neb = export.to_python(sp_neb, {"analysis": an_neb, "model": lam5_model, "decision": dec_neb}, "0.1.0")
    assert "from ase.io import read" not in py_neb
    assert "load_structure('smiles:C/C=C/C')" in py_neb
    assert r"load_structure('smiles:C/C=C\\C')" in py_neb
    ast.parse(py_neb)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        exec(compile(py_neb, "<smiles_neb>", "exec"), {"__name__": "__smiles_neb_test__"})
    finally:
        os.chdir(old_cwd)


def test_jobmanager_load_previous_preserves_analysis_and_cli_malformed_config(tmp_path):
    """`JobManager.load_previous()` restores `analysis` for History table, and `mena run` rejects non-string `structure` cleanly."""

    class FastEngine:
        def run(self, spec, meta, emit, cancel, workdir):
            return {"status": "done", "result": {"headline": "ok", "tiles": []}}

    jobs_root = tmp_path / "jobs"
    jm1 = jobs.JobManager(FastEngine(), max_jobs=1, root=jobs_root)
    sp = validation.validate({"task": "singlepoint", "structure": {"format": "smiles", "text": "CCO"}})
    an = structures.analyze(structures.load(sp["structure"]))
    submitted = jm1.submit(sp, {"test_title": "Energy & forces", "analysis": an, "model": {"id": "lam5-v1", "title": "Mena-LAM-s"}})
    for _ in range(50):
        if jm1.get(submitted.id).status == "done":
            break
        threading.Event().wait(0.02)
    jm1.shutdown()

    jm2 = jobs.JobManager(FastEngine(), max_jobs=1, root=jobs_root)
    try:
        loaded_count = jm2.load_previous()
        assert loaded_count == 1
        restored = jm2.list()[0].summary()
        assert restored["earlier_session"] is True
        assert restored["analysis"]["formula"] == "C2H6O"
        assert restored["analysis"]["n_atoms"] == 9
        assert restored["analysis"]["domain"] == "molecule"
    finally:
        jm2.shutdown()

    # Malformed non-string `structure` in `mena run bad.json` exits with code 2 (ValidationError), not TypeError
    for bad_struct in (None, 123, ["water"]):
        bad_cfg = tmp_path / "bad_job.json"
        bad_cfg.write_text(json.dumps({"task": "singlepoint", "structure": bad_struct}), encoding="utf-8")
        rc = cli.main(["run", str(bad_cfg)])
        assert rc == 2


def _have_lam5() -> bool:
    try:
        return bool(models.choose("lam5-v1", "molecule").get("available"))
    except models.ModelError:
        return False


@pytest.mark.skipif(not _have_lam5(), reason="lam5-v1 release checkpoint not installed")
def test_live_lam5_molecular_visual_md_end_to_end(tmp_path):
    """End-to-end live benchmark on a non-periodic water droplet routed automatically to `lam5-v1` (`Mena-LAM-s`)."""
    spec = validation.validate(
        {
            "task": "benchmark",
            "structure": {"format": "builtin", "text": "water"},
            "params": {"mode": "visual_md", "max_atoms": 24, "steps": 3},
            "runtime": {"device": "cpu"},
        }
    )
    pl = planner.plan(spec)
    assert pl["model"]["id"] == "lam5-v1"
    assert pl["model"]["backend"] == "lam4"
    res = runner.run(pl, spec, str(tmp_path / "lam5_bench"), lambda _: None, threading.Event())
    assert res["status"] == "done", res.get("error")
    out = res["result"]
    assert out["raw"]["n"] == 24
    assert out["raw"]["atoms_per_s"] > 0
    assert out["facts"]["Model"] == "Mena-LAM-s"
    assert "Molecular cluster" in out["facts"]

