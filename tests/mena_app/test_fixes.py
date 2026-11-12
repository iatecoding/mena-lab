"""Fast regression tests for verified release bug fixes (B4, B5, B8, B9, B10, B11, B12, B14, B15, B17, B18)."""

from __future__ import annotations

import ast
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.io import write as ase_write

from mena_app import (
    backends,
    calibrate,
    cli,
    costmodel,
    export,
    jobs,
    models,
    phonon_symmetry,
    planner,
    report,
    routing,
    runner,
    runtime,
    server,
    structures,
    system,
    validation,
)
from mena_app.routing import PlanError
from mena_app.tasks import common, md, neb, singlepoint, vibrations


class HarmonicCalc(Calculator):
    """Instant analytic calculator with optional stress and shared `model` sentinel for fast task tests."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, k: float = 1.5, skin: float = 0.0):
        super().__init__()
        self.k = k
        self.skin = skin
        self.model = object()
        self._nl_cache = {"dummy": 1}
        self._last_state = ("dummy",)
        self.call_log: list[list[str]] = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.call_log.append(list(properties))
        pos = self.atoms.get_positions()
        center = pos.mean(axis=0, keepdims=True)
        disp = pos - center
        energy = 0.5 * self.k * float((disp ** 2).sum())
        forces = -self.k * disp
        self.results = {"energy": energy, "forces": forces}
        if self.atoms.pbc.all():
            vol = max(abs(float(self.atoms.get_volume())), 1.0)
            strain = (vol - 35.0) / 35.0
            energy += 2.0 * (strain ** 2)
            self.results["energy"] = energy
            s = 4.0 * strain / 35.0
            self.results["stress"] = np.array([s, s, s, 0.0, 0.0, 0.0], dtype=float)


# ---- 1. Process Cleanup & Memory Polling Throttle (B4, B18) ----


def test_runner_interrupt_kills_worker_and_throttles_memory_poll(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    spawned: list[subprocess.Popen] = []

    def fake_worker_popen(cmd, **kw):
        if not cmd or cmd[0] != sys.executable:
            return real_popen(cmd, **kw)
        script = (
            "import json, sys, time\n"
            "sys.stdin.read()\n"
            "print(json.dumps({'ev': 'status', 'text': 'ready'}), flush=True)\n"
            "time.sleep(30)\n"
        )
        p = real_popen([sys.executable, "-c", script], **kw)
        spawned.append(p)
        return p

    plan_stub = {"decision": {}, "model": {}}

    ready = threading.Event()

    def on_emit(ev):
        if ev.get("ev") == "status":
            ready.set()

    real_sleep = time.sleep
    interrupted = False

    def interrupt_once_ready(secs):
        nonlocal interrupted
        real_sleep(0.02)
        if ready.is_set() and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    with monkeypatch.context() as mp:
        mp.setattr(subprocess, "Popen", fake_worker_popen)
        mp.setattr(time, "sleep", interrupt_once_ready)
        with pytest.raises(KeyboardInterrupt):
            runner.run(plan_stub, {}, str(tmp_path / "ki"), on_emit, threading.Event())

    assert len(spawned) == 1
    pid = spawned[0].pid
    assert spawned[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)

    # Verify memory polling runs at most once per 2.0 s across a ~0.7 s worker run
    mem_calls = 0

    def count_mem():
        nonlocal mem_calls
        mem_calls += 1
        return 16_000_000_000

    def short_worker_popen(cmd, **kw):
        if not cmd or cmd[0] != sys.executable:
            return real_popen(cmd, **kw)
        script = (
            "import json, sys, time\n"
            "sys.stdin.read()\n"
            "time.sleep(0.65)\n"
            "print(json.dumps({'ev': 'result', 'data': {'cancelled': False}}), flush=True)\n"
        )
        return real_popen([sys.executable, "-c", script], **kw)

    monkeypatch.setattr(subprocess, "Popen", short_worker_popen)
    monkeypatch.setattr(system, "available_memory_bytes", count_mem)
    res = runner.run(plan_stub, {}, str(tmp_path / "mem"), lambda _: None, threading.Event())
    assert res["status"] == "done"
    assert mem_calls == 1


def test_jobmanager_shutdown_cancels_and_joins_threads(tmp_path):
    started = threading.Event()
    cleaned_up = threading.Event()

    class BlockingEngine:
        def run(self, spec, meta, emit, cancel, workdir):
            started.set()
            cancel.wait(timeout=5.0)
            cleaned_up.set()
            return {"status": "cancelled", "result": None}

    jm = jobs.JobManager(BlockingEngine(), max_jobs=1, root=tmp_path / "jobs")
    job = jm.submit({"task": "singlepoint"}, {"test_title": "SP"})
    assert started.wait(timeout=2.0)
    t0 = time.perf_counter()
    jm.shutdown()
    dt = time.perf_counter() - t0
    assert dt < 2.5
    assert job.cancel.is_set()
    assert cleaned_up.is_set()
    assert all(not t.is_alive() for t in jm.threads)


# ---- 2. Monotonic Sparse Edge Cost Model & Uncalibrated GPU Routing (B5) ----


def test_sparse_edges_monotonic_costmodel_and_gpu_routing(monkeypatch, tmp_path):
    monkeypatch.setenv("MENA_HOME", str(tmp_path / "mena_home"))

    # Pure-numpy fallback matches vesin on molecules, 3D crystals, and 2D mixed-PBC slabs with zero c-vector
    slab = Atoms("Cu2", positions=[[0, 0, 0], [1.28, 1.28, 0.5]], cell=[[2.56, 0, 0], [0, 2.56, 0], [0, 0, 0]], pbc=[True, True, False])
    for st in (structures.load("water"), structures.load("si"), slab, calibrate._water_box(180)):
        e_vesin = structures.count_edges(st)
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "vesin", None)
            e_np = structures.count_edges(st)
        assert e_vesin == e_np > 0

    # calibration_m3max.json molecule x values match sparse count_edges and strictly increase across 3..1020 atoms
    cal = json.loads((Path(structures.__file__).resolve().parent / "data" / "calibration_m3max.json").read_text())
    expected_xs = [structures.count_edges(calibrate._water_box(w)) for w in calibrate.MOLECULE_WATERS]
    assert expected_xs == sorted(expected_xs) and len(set(expected_xs)) == len(expected_xs)
    for backend, dev in (("lam", "cpu"), ("lam", "mps"), ("lam4", "cpu"), ("lam4", "mps")):
        xs = [p["x"] for p in cal["points"] if p.get("backend") == backend and p["domain"] == "molecule" and p["device"] == dev]
        assert xs == expected_xs

    # Strictly non-decreasing costmodel.estimate across N up to 5000+ atoms (no N=512 jump!)
    waters = [1, 4, 11, 32, 100, 170, 171, 180, 340, 800, 1670]  # up to 5010 atoms
    edges_seq = [structures.count_edges(calibrate._water_box(w)) for w in waters]
    assert edges_seq == sorted(edges_seq) and len(set(edges_seq)) == len(edges_seq)
    for dev in ("cpu", "mps"):
        ests = [costmodel.estimate(cal, "molecule", dev, ed, n_calls=10) for ed in edges_seq]
        secs = [e["seconds"] for e in ests]
        gbs = [e["peak_gb"] for e in ests]
        assert secs == sorted(secs) and secs[-1] > secs[0]
        assert gbs == sorted(gbs) and gbs[-1] >= gbs[0]

    # Negative noisy calibration slopes are clamped non-decreasing
    noisy = [(100.0, 20.0), (1000.0, 12.0), (10000.0, 50.0)]
    vals = [costmodel._interp(noisy, x) for x in (50, 100, 300, 1000, 3000, 10000, 20000)]
    assert vals == sorted(vals)

    # costmodel.load() only falls back to calibration_m3max.json on darwin + arm64
    with monkeypatch.context() as mp:
        mp.setattr(sys, "platform", "linux")
        loaded = costmodel.load()
        assert loaded["points"] == []

    # Uncalibrated routing prefers CUDA, then MPS (when worthwhile), and never falsely says "The only device available"
    sp_single = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "si"}})
    sp_md = validation.validate({"task": "md", "structure": {"format": "builtin", "text": "cu111"}, "params": {"steps": 100}})
    an_si = structures.analyze(structures.load("si"))
    an_cu = structures.analyze(structures.load("cu111"))
    lam_model = {"id": "lam", "title": "LAM", "backend": "lam", "lots": ["mp_pbe"], "dtypes": ["float32", "float64"], "low_memory": True}
    empty_cal = {"points": [], "source": "no measurements"}

    r_cuda = routing.route(sp_single, an_si, ["cpu", "cuda"], 32.0, empty_cal, lam_model)
    assert r_cuda["device"] == "cuda"
    assert "mena calibrate" in r_cuda["reasons"]["device"]
    assert "only device" not in r_cuda["reasons"]["device"].lower()

    r_mps_md = routing.route(sp_md, an_cu, ["cpu", "mps"], 32.0, empty_cal, lam_model)
    assert r_mps_md["device"] == "mps"
    assert "only device" not in r_mps_md["reasons"]["device"].lower()

    r_only_cpu = routing.route(sp_single, an_si, ["cpu"], 32.0, empty_cal, lam_model)
    assert r_only_cpu["device"] == "cpu" and "only device available" in r_only_cpu["reasons"]["device"].lower()


# ---- 3. Molecular charge / spin Hints & Explicit Overrides (B8) ----


def test_charge_spin_hints_and_explicit_cli_override(tmp_path, monkeypatch):
    # Fake model selection so planner.plan works even without model checkpoints if needed
    fake_mol_model = {"id": "m", "title": "MolModel", "backend": "lam4", "path": __file__, "available": True,
                      "domains": ["molecule"], "lots": ["omol_wb97mv"], "dtypes": ["float32", "float64"], "low_memory": False}
    monkeypatch.setattr(models, "choose", lambda req, dom: fake_mol_model)

    # 1. .extxyz with charge=1 spin=2 in atoms.info
    nh3 = structures.load("nh3_up")
    nh3.info["charge"] = 1
    nh3.info["spin"] = 2
    extxyz_buf = io.StringIO()
    ase_write(extxyz_buf, nh3, format="extxyz")
    extxyz_text = extxyz_buf.getvalue()

    an = structures.analyze(structures.load({"format": "extxyz", "text": extxyz_text}))
    assert an["charge_hint"] == 1 and an["spin_hint"] == 2
    sp_ext = validation.validate({"task": "singlepoint", "structure": {"format": "extxyz", "text": extxyz_text}})
    planner.plan(sp_ext)
    assert sp_ext["system"] == {"charge": 1, "spin": 2}

    # 2. .extxyz with charge=1 and NO spin tag -> auto-doublet (spin=2 for 9 electrons)
    nh3_no_spin = structures.load("nh3_up")
    nh3_no_spin.info["charge"] = 1
    buf2 = io.StringIO()
    ase_write(buf2, nh3_no_spin, format="extxyz")
    sp_auto_doublet = validation.validate({"task": "singlepoint", "structure": {"format": "extxyz", "text": buf2.getvalue()}})
    planner.plan(sp_auto_doublet)
    assert sp_auto_doublet["system"] == {"charge": 1, "spin": 2}

    # 3. SMILES [NH4+] auto-populates charge=1, spin=1, while explicit --charge 0 --spin 2 overrides charge_hint!
    ns_default = cli.build_parser().parse_args(["sp", "smiles:[NH4+]"])
    sp_nh4 = cli.spec_from_args("singlepoint", ns_default)
    planner.plan(sp_nh4)
    assert sp_nh4["system"] == {"charge": 1, "spin": 1}

    ns_override = cli.build_parser().parse_args(["sp", "smiles:[NH4+]", "--charge", "0", "--spin", "2"])
    sp_nh4_override = cli.spec_from_args("singlepoint", ns_override)
    planner.plan(sp_nh4_override)
    assert sp_nh4_override["system"] == {"charge": 0, "spin": 2}

    # 4. Explicit wrong parity on neutral H2O (--charge 0 --spin 2) STILL raises PlanError
    ns_bad = cli.build_parser().parse_args(["sp", "water", "--charge", "0", "--spin", "2"])
    sp_bad = cli.spec_from_args("singlepoint", ns_bad)
    with pytest.raises(PlanError, match="cannot have spin multiplicity"):
        planner.plan(sp_bad)


# ---- 4. NEB Redundant Calls & structure2 Validation (B9) ----


def test_neb_evaluation_count_and_structure2_validation(tmp_path, monkeypatch):
    shared_log: list[list[str]] = []

    class SharedCountingCalc(HarmonicCalc):
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            shared_log.append(list(properties))

    root_calc = SharedCountingCalc()
    cloned = backends.clone_calculator(root_calc)
    assert cloned is not root_calc
    assert cloned.model is root_calc.model
    assert cloned.results == {} and cloned._nl_cache == {} and cloned._last_state is None

    monkeypatch.setattr(backends, "build", lambda model, dec: root_calc)
    spec = validation.validate({
        "task": "neb",
        "structure": {"format": "builtin", "text": "nh3_up"},
        "structure2": {"format": "builtin", "text": "nh3_down"},
        "params": {"images": 5, "max_steps": 2, "fmax": 1e-4, "climb": True, "relax_ends": False},
        "system": {"charge": 0, "spin": 1},
    })
    decision = {"device": "cpu", "dtype": "float32", "low_memory": False, "skin": 0.0, "compile": False, "lot": "auto"}
    ctx = runtime.Context({"spec": spec, "decision": decision, "model": {"id": "mock", "title": "Mock"}, "workdir": str(tmp_path)}, io.StringIO())
    res = neb.run(ctx)
    assert len(res["raw"]["energies_eV"]) == 7
    assert len(shared_log) <= 17, f"Expected <= 17 calculator calls for 5 images x 2 steps, got {len(shared_log)}"

    # structure2 validation in planner.plan
    fake_mol_model = {"id": "m", "title": "MolModel", "backend": "lam4", "path": __file__, "available": True,
                      "domains": ["molecule"], "lots": ["omol_wb97mv"], "dtypes": ["float32"], "low_memory": False}
    monkeypatch.setattr(models, "choose", lambda req, dom: fake_mol_model)

    # Corrupt structure2
    bad_corrupt = validation.validate({
        "task": "neb",
        "structure": {"format": "builtin", "text": "water"},
        "structure2": {"format": "xyz", "text": "not valid xyz"},
    })
    with pytest.raises(PlanError, match="end structure"):
        planner.plan(bad_corrupt)

    # Different atom count
    bad_count = validation.validate({
        "task": "neb",
        "structure": {"format": "builtin", "text": "water"},
        "structure2": {"format": "builtin", "text": "nh3_up"},
    })
    with pytest.raises(PlanError, match="same atoms"):
        planner.plan(bad_count)

    # Same atom count, swapped element order
    swapped_xyz = "3\nswapped\nH 0 0 0\nO 0.96 0 0\nH -0.24 0.93 0\n"
    bad_order = validation.validate({
        "task": "neb",
        "structure": {"format": "builtin", "text": "water"},
        "structure2": {"format": "xyz", "text": swapped_xyz},
    })
    with pytest.raises(PlanError, match="same order"):
        planner.plan(bad_order)


# ---- 5. Python & Config Export AST + Exec Verification (B10) ----


def test_export_python_ast_and_exec_all_tasks(tmp_path, monkeypatch):
    # Preserve structure2 name/source in export.to_config
    end_file = tmp_path / "product_end.xyz"
    end_file.write_text("3\nwater\nO 0 0 0\nH 0.97 0 0\nH -0.24 0.93 0\n")
    start_file = tmp_path / "reactant_start.xyz"
    start_file.write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")

    ns = cli.build_parser().parse_args(["neb", str(start_file), str(end_file)])
    sp_neb_files = cli.spec_from_args("neb", ns)
    cfg = export.to_config(sp_neb_files)
    assert cfg["structure"] == "reactant_start.xyz"
    assert cfg["structure2"] == "product_end.xyz"

    # Stub calculator modules so exported scripts execute end-to-end in milliseconds
    class StubMenaLAM(HarmonicCalc):
        @classmethod
        def from_checkpoint(cls, path, **kw):
            return cls(skin=kw.get("skin", 0.0))

    class StubLam4(HarmonicCalc):
        @classmethod
        def from_checkpoint(cls, path, **kw):
            return cls(skin=kw.get("skin", 0.0))

    mod_mena_lam = types.ModuleType("lam.models.mena_lam")
    mod_mena_lam.MenaLAMCalculator = StubMenaLAM
    mod_lam4 = types.ModuleType("lam4")
    mod_lam4.MenaCalculator = StubLam4

    monkeypatch.setitem(sys.modules, "lam.models.mena_lam", mod_mena_lam)
    monkeypatch.setitem(sys.modules, "lam4", mod_lam4)

    lam_model = {"id": "lam", "title": "LAM", "backend": "lam", "path": "/tmp/mock_lam.pt",
                 "lots": ["mp_pbe", "omol_wb97mv"], "dtypes": ["float32", "float64"], "low_memory": True}
    omol_model = {"id": "mol", "title": "Mol", "backend": "lam4", "path": "/tmp/mock_mol.pt",
                  "lots": ["omol_wb97mv"], "dtypes": ["float32", "float64"], "low_memory": False}

    def make_plan(sp, model):
        a = structures.analyze(structures.load(sp["structure"]))
        if a["domain"] == "molecule":
            sp["system"]["charge"] = sp["system"].get("charge") or 0
            sp["system"]["spin"] = sp["system"].get("spin") or 1
        d = routing.route(sp, a, ["cpu"], 32.0, {"points": [], "source": "none"}, model)
        return {"analysis": a, "model": model, "decision": d}

    cases = [
        # Builtin + lam4 singlepoint (checks load_structure + MenaCalculator + charge/spin)
        (validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}, "system": {"charge": 0, "spin": 1}}), omol_model),
        # File path + lam relax
        (validation.validate({"task": "relax", "structure": structures.from_path(str(start_file)), "params": {"max_steps": 1}}), lam_model),
        # Crystal MD with NVT, NVE, NPT + md_repeat > 1
        (validation.validate({"task": "md", "structure": {"format": "builtin", "text": "si"}, "params": {"ensemble": "nvt", "steps": 1}}), lam_model),
        (validation.validate({"task": "md", "structure": {"format": "builtin", "text": "si"}, "params": {"ensemble": "nve", "steps": 1}}), lam_model),
        (validation.validate({"task": "md", "structure": {"format": "builtin", "text": "si"}, "params": {"ensemble": "npt", "steps": 1}}), lam_model),
        # Vibrations (H2 molecule, prerelax=False)
        (validation.validate({"task": "vibrations", "structure": {"format": "xyz", "text": "2\nh2\nH 0 0 0\nH 0 0 0.74\n", "name": "h2"},
                              "params": {"prerelax": False}}), omol_model),
        # Phonons (Si crystal, supercell=1, prerelax=False)
        (validation.validate({"task": "phonons", "structure": {"format": "builtin", "text": "si"},
                              "params": {"supercell": 1, "prerelax": False, "kpoints": 4}}), lam_model),
        # NEB (builtin endpoints, 3 images, 1 step, relax_ends=False)
        (validation.validate({"task": "neb", "structure": {"format": "builtin", "text": "nh3_up"},
                              "structure2": {"format": "builtin", "text": "nh3_down"},
                              "params": {"images": 3, "max_steps": 1, "relax_ends": False}}), omol_model),
    ]
    cases.append((validation.validate({"task": "singlepoint", "structure": {"format": "smiles", "text": "[NH4+]"},
                                       "system": {"charge": 1, "spin": 1}}), omol_model))

    (tmp_path / "h2.xyz").write_text("2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*thermalize_momenta.*", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"ase\.md.*")
            for idx, (sp, mod) in enumerate(cases):
                pl = make_plan(sp, mod)
                code = export.to_python(sp, pl, "0.1.0")
                ast.parse(code)
                ns_exec: dict = {"__name__": f"__export_test_{idx}__"}
                exec(compile(code, f"<export_{sp['task']}_{idx}>", "exec"), ns_exec)
                if mod["backend"] == "lam4":
                    assert "MenaCalculator" in code
                    assert 'atoms.info["charge"]' in code and 'atoms.info["spin"]' in code
                    assert 'atoms.info["charge"]' in code and 'atoms.info["spin"]' in code
    finally:
        os.chdir(old_cwd)


# ---- 6. 2D Slabs, Input Validation & Server JSON Safety (B11, B12) ----


def test_2d_slabs_validation_and_server_json_safety(monkeypatch):
    # 2D slab with cell[2] = [0, 0, 0] and pbc=[True, True, False]
    slab = Atoms("Cu2", positions=[[0, 0, 0], [1.28, 1.28, 1.0]], cell=[[2.56, 0, 0], [0, 2.56, 0], [0, 0, 0]], pbc=[True, True, False])
    buf = io.StringIO()
    ase_write(buf, slab, format="extxyz")
    slab_text = buf.getvalue()
    loaded_slab = structures.load({"format": "extxyz", "text": slab_text})
    an_slab = structures.analyze(loaded_slab)
    assert an_slab["domain"] == "slab" and an_slab["volume"] is None and an_slab["edges"] > 0

    # Invalid geometries rejected by math_ok / structures.load
    bad_3d = Atoms("Cu", positions=[[0, 0, 0]], cell=[[2.5, 0, 0], [0, 2.5, 0], [0, 0, 0]], pbc=[True, True, True])
    buf_bad = io.StringIO()
    ase_write(buf_bad, bad_3d, format="extxyz")
    with pytest.raises(structures.StructureError):
        structures.load({"format": "extxyz", "text": buf_bad.getvalue()})
    with pytest.raises(structures.StructureError):
        structures.load([1, 2, 3])  # type: ignore[arg-type]

    # structures.py:177 low-density crystal warning only triggers when "H" is actually present
    big_si = Atoms("Si2", positions=[[0, 0, 0], [2.3, 0, 0]], cell=[25.0, 25.0, 25.0], pbc=True)
    big_h2 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[25.0, 25.0, 25.0], pbc=True)
    assert not any("molecule in a box" in w for w in structures.analyze(big_si)["warnings"])
    assert any("molecule in a box" in w for w in structures.analyze(big_h2)["warnings"])

    # validation.py rejects non-string choice values (lists/dicts) with ValidationError instead of TypeError
    with pytest.raises(validation.ValidationError):
        validation.validate({"task": "relax", "structure": {"format": "builtin", "text": "si"}, "params": {"optimizer": ["LBFGS"]}})
    with pytest.raises(validation.ValidationError):
        validation.validate({"task": "relax", "structure": {"format": "builtin", "text": "si"}, "runtime": {"device": {"id": "cpu"}}})

    # routing.py rejects invalid device or incompatible lot with PlanError
    omol_model = {"id": "mol", "title": "Mol", "backend": "lam4", "path": __file__, "available": True,
                  "domains": ["molecule"], "lots": ["omol_wb97mv"], "dtypes": ["float32", "float64"], "low_memory": False}
    an_w = structures.analyze(structures.load("water"))
    sp_bad_lot = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}, "runtime": {"lot": "mp_pbe"}})
    with pytest.raises(PlanError, match="level of theory"):
        routing.route(sp_bad_lot, an_w, ["cpu"], 32.0, {"points": []}, omol_model)
    sp_bad_dev = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}, "runtime": {"device": "tpu"}})
    with pytest.raises(PlanError, match="not available"):
        routing.route(sp_bad_dev, an_w, ["cpu"], 32.0, {"points": []}, omol_model)

    # HTTP server checks: 2D slab -> 200, list-choice / bad lot -> 400, NaN/Inf sanitized to null
    monkeypatch.setattr(models, "choose", lambda req, dom: omol_model)
    srv, app = server.create("127.0.0.1", port=18911, tries=30)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{app.port}"
        headers = {"Content-Type": "application/json", "X-Mena": "1", "Host": f"127.0.0.1:{app.port}", "Connection": "close"}

        req_slab = urllib.request.Request(f"{base}/api/structure", data=json.dumps({"structure": {"format": "extxyz", "text": slab_text}}).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req_slab, timeout=3) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode())
            assert body["analysis"]["domain"] == "slab" and body["analysis"]["volume"] is None

        for bad_payload in (
            {"task": "relax", "structure": {"format": "builtin", "text": "water"}, "params": {"optimizer": ["LBFGS"]}},
            {"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}, "runtime": {"lot": "mp_pbe"}},
        ):
            req_bad = urllib.request.Request(f"{base}/api/plan", data=json.dumps(bad_payload).encode(), headers=headers, method="POST")
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req_bad, timeout=3)
            try:
                assert exc_info.value.code == 400
            finally:
                exc_info.value.close()
    finally:
        app.jobs.shutdown()
        srv.shutdown()
        srv.server_close()

    # Direct NaN/Inf JSON safety check matching Handler.json
    sanitized = json.dumps(runtime.clean({"a": float("nan"), "b": float("inf"), "c": 1.5}), allow_nan=False)
    assert json.loads(sanitized) == {"a": None, "b": None, "c": 1.5}


# ---- 7. Task & CLI Polish (B14, B15, B17, B18) ----


def test_vibrations_near_linear_singlepoint_order_md_t0_cell_relax_models_and_spglib(tmp_path, monkeypatch):
    monkeypatch.setenv("MENA_HOME", str(tmp_path / "mena_home"))

    # 1. Near-linear CO2 (bent by 1e-4 A): external_basis rank and _linear agree, thermo filters <= 1e-4 eV
    co2_near = Atoms("CO2", positions=[[0.0, 0.0, 0.0], [0.0, 1e-4, 1.16], [0.0, 0.0, -1.16]])
    n_ext = vibrations.external_basis(co2_near).shape[1]
    assert vibrations._linear(co2_near) == (n_ext == 5)
    th = vibrations.thermo(np.array([1e-9, 5e-5, 0.082, 0.165, 0.291]), 298.15)
    assert math.isfinite(th["s"]) and th["s"] < 0.005 and th["zpe"] > 0.25

    # 2. singlepoint.run requests forces BEFORE energy on the first cold call
    sp_calc = HarmonicCalc()
    monkeypatch.setattr(backends, "build", lambda model, dec: sp_calc)
    sp_spec = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}, "system": {"charge": 0, "spin": 1}})
    sp_ctx = runtime.Context({"spec": sp_spec, "decision": {"device": "cpu", "dtype": "float32", "low_memory": False, "skin": 0.0},
                              "model": {"id": "m", "title": "M"}, "workdir": str(tmp_path / "sp")}, io.StringIO())
    singlepoint.run(sp_ctx)
    assert "forces" in sp_calc.call_log[0], f"First call should request forces, got {sp_calc.call_log[0]}"

    # 3. md.run has no duplicate t=0.0 in time_ps or emitted series
    (tmp_path / "md").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(backends, "build", lambda model, dec: LennardJones(sigma=2.3, epsilon=0.1, rc=5.5, smooth=True))
    md_spec = validation.validate({"task": "md", "structure": {"format": "builtin", "text": "cu"},
                                   "params": {"ensemble": "nvt", "steps": 3, "timestep": 1.0, "save_every": 1}})
    md_out = io.StringIO()
    md_ctx = runtime.Context({"spec": md_spec, "decision": {"device": "cpu", "dtype": "float32", "low_memory": False, "skin": 1.0},
                              "model": {"id": "m", "title": "M"}, "workdir": str(tmp_path / "md")}, md_out)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"(mena_app\.tasks\.md|ase\.md.*)")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"ase\.md.*")
        md_res = md.run(md_ctx)
    assert md_res["raw"]["time_ps"] == [0.0, 0.001, 0.002, 0.003]
    md_events = [json.loads(line) for line in md_out.getvalue().splitlines()]
    temp_xs = [x for e in md_events if e.get("ev") == "series" and e.get("chart") == "md_temperature" for x in e["x"]]
    assert temp_xs == [0.0, 0.001, 0.002, 0.003]

    # 4. optimize(..., cell=True) reports generalized FrechetCellFilter fmax > 0 on hydrostatically strained crystal + includes cell in frames
    (tmp_path / "opt").mkdir(parents=True, exist_ok=True)
    strained = bulk("Cu", "fcc", a=3.95)  # hydrostatically strained: all atomic forces are 0 by symmetry, only stress != 0!
    strained.calc = LennardJones(sigma=2.3, epsilon=0.2, rc=5.5, smooth=True)
    assert common.fmax_of(strained.get_forces()) < 1e-12
    opt_out = io.StringIO()
    opt_ctx = runtime.Context({"spec": md_spec, "decision": {}, "model": {}, "workdir": str(tmp_path / "opt")}, opt_out)
    rec = common.optimize(opt_ctx, strained, "LBFGS", fmax=1e-3, steps=2, cell=True)
    assert rec["fmaxs"][0] > 0.01  # FrechetCellFilter generalized cell force is non-zero!
    frame_events = [json.loads(line)["frame"] for line in opt_out.getvalue().splitlines() if json.loads(line).get("ev") == "frame"]
    assert frame_events and "cell" in frame_events[0]
    assert "cell" in report.frame(strained)

    # 5. mena models add --backend lam4 sets the lam4 domain and level-of-theory defaults
    fake_ckpt = tmp_path / "omol_custom.pt"
    fake_ckpt.write_bytes(b"pt")
    rc = cli.main(["models", "add", "custom_mol", str(fake_ckpt), "--backend", "lam4"])
    assert rc == 0
    added = next(m for m in models.all_models() if m["id"] == "custom_mol")
    assert added["backend"] == "lam4"
    assert added["domains"] == ["molecule", "crystal", "slab"]
    assert "omol_wb97mv" in added["lots"]
    assert added["dtypes"] == ["float32", "float64"]
    assert added["low_memory"] is True

    # 6. Zero symmetry DeprecationWarnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        si = structures.load("si")
        structures.analyze(si)
        phonon_symmetry.symmetry_plan(si, 2)
    dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warns == []
