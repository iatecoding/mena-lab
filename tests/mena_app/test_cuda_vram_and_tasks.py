"""Unit & regression tests for CUDA VRAM accounting, multi-GPU routing, precision, exports, and accelerated tasks."""
from __future__ import annotations

import inspect
import io
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import mock

from ase.build import bulk, molecule
import numpy as np
import pytest
import torch

from mena_app import (
    backends,
    calibrate,
    export,
    machine,
    planner,
    python_api,
    routing,
    runner,
    structures,
    system,
    validation,
)
from mena_app.tasks import benchmark as benchmark_task
from mena_app.tasks import common as task_common
from mena_app.tasks import md as md_task
from mena_app.tasks import neb as neb_task
from mena_app.tasks import phonons as phonons_task


# ---------------------------------------------------------------------------
# 1. VRAM accounting & Multi-GPU routing / planning
# ---------------------------------------------------------------------------


def test_system_cuda_vram_info_parses_nvidia_smi_and_handles_missing():
    sample = "0, NVIDIA A100-SXM4-80GB, 79500, 81920\n1, NVIDIA A100-SXM4-80GB, 12000, 81920\n"
    ok_res = SimpleNamespace(returncode=0, stdout=sample)
    with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
         mock.patch("subprocess.run", return_value=ok_res):
        info = system.cuda_vram_info()
        assert len(info) == 2
        assert info[0]["index"] == 0 and info[0]["name"] == "NVIDIA A100-SXM4-80GB"
        assert info[1]["vram_free_bytes"] == 12000 * 1024 * 1024
        assert system.available_vram_bytes(1) == 12000 * 1024 * 1024
        assert system.available_vram_bytes(9) is None

    with mock.patch("shutil.which", return_value=None):
        assert system.cuda_vram_info() == []
        assert system.available_vram_bytes(0) is None


def test_machine_snapshot_exposes_per_device_free_gb_without_mixing_ram_and_vram():
    fake_probe = {
        "os": "Linux",
        "arch": "x86_64",
        "ram_gb": 16.0,
        "cpu_cores": 16,
        "devices": ["cpu", "cuda:0", "cuda:1"],
        "gpu": "NVIDIA A100-SXM4-80GB",
        "vram_total_gb": 80.0,
        "vram_free_gb": 78.0,
        "cuda_devices": [
            {"id": "cuda:0", "index": 0, "name": "NVIDIA A100-SXM4-80GB", "vram_total_gb": 80.0, "vram_free_gb": 78.0},
            {"id": "cuda:1", "index": 1, "name": "NVIDIA A100-SXM4-80GB", "vram_total_gb": 80.0, "vram_free_gb": 79.5},
        ],
    }
    live_smi = [
        {"index": 0, "id": "cuda:0", "name": "NVIDIA A100-SXM4-80GB", "vram_free_bytes": int(10 * 1e9), "vram_total_bytes": int(80 * 1e9), "vram_free_gb": 10.0, "vram_total_gb": 80.0},
        {"index": 1, "id": "cuda:1", "name": "NVIDIA A100-SXM4-80GB", "vram_free_bytes": int(72 * 1e9), "vram_total_bytes": int(80 * 1e9), "vram_free_gb": 72.0, "vram_total_gb": 80.0},
    ]
    with mock.patch("mena_app.machine.probe", return_value=fake_probe), \
         mock.patch("mena_app.system.available_memory_bytes", return_value=int(3.2 * 1e9)), \
         mock.patch("mena_app.system.cuda_vram_info", return_value=live_smi):
        snap = machine.snapshot()
    assert snap["available_gb"] == 3.2
    assert snap["device_free_gb"]["cpu"] == 3.2
    assert snap["device_free_gb"]["cuda:0"] == 10.0
    assert snap["device_free_gb"]["cuda:1"] == 72.0
    assert snap["vram_free_gb"] == 10.0


def test_routing_uses_vram_on_cuda_handles_multi_gpu_and_fp32_ieee_phonons():
    lam4_m = {
        "id": "lam5-v1",
        "title": "Mena-LAM 5",
        "backend": "lam4",
        "lots": ["omol_wb97mv", "mp_pbe"],
        "dtypes": ["float32", "float64"],
        "low_memory": True,
    }
    si_at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3))
    an = structures.analyze(si_at)
    sp = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "si"}})

    # Multi-GPU costmodel lookup calibrated on "cuda" applies to "cuda:1" and breaks ties toward higher free VRAM!
    cal = {
        "source": "measured",
        "points": [
            {"domain": "crystal", "device": "cuda", "backend": "lam4", "x": 10000, "cold_ms": 40.0, "warm_ms": 20.0, "peak_gb": 6.0},
            {"domain": "crystal", "device": "cpu", "backend": "lam4", "x": 10000, "cold_ms": 900.0, "warm_ms": 500.0, "peak_gb": 2.0},
        ],
    }
    # Host RAM tight (1.0 GB) while GPU cuda:1 has 72 GB VRAM -> routes to cuda:1 cleanly without false refusal!
    dec_gpu = routing.route(
        sp, an, ["cpu", "cuda:0", "cuda:1"], {"cpu": 1.0, "cuda:0": 4.0, "cuda:1": 72.0}, cal, lam4_m
    )
    assert dec_gpu["device"] == "cuda:1"
    assert dec_gpu["free_gb"] == 72.0
    assert dec_gpu["memory_level"] != "refuse"
    assert dec_gpu["compile"] is False  # singlepoint calls=1 skips compile overhead

    # Host RAM plentiful (64 GB) while single GPU has tiny VRAM (0.8 GB) -> caught as CUDA VRAM refusal!
    sp_pin = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "si"}, "runtime": {"device": "cuda:0"}})
    dec_oom = routing.route(sp_pin, an, ["cpu", "cuda:0"], {"cpu": 64.0, "cuda:0": 0.8}, cal, lam4_m)
    assert dec_oom["memory_level"] == "refuse"

    # Explicit "cuda" maps cleanly to highest-free CUDA GPU on multi-GPU and compiles long MD
    sp_cuda = validation.validate({"task": "md", "structure": {"format": "builtin", "text": "si"}, "runtime": {"device": "cuda"}, "params": {"steps": 5000}})
    dec_alias = routing.route(sp_cuda, an, ["cpu", "cuda:0", "cuda:1"], {"cpu": 32.0, "cuda:0": 8.0, "cuda:1": 70.0}, cal, lam4_m)
    assert dec_alias["device"] == "cuda:1"
    assert dec_alias["compile"] is True

    # Non-existent cuda:9 raises RoutingError cleanly
    sp_bad = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "si"}, "runtime": {"device": "cuda:9"}})
    with pytest.raises(routing.RoutingError, match="cuda:9"):
        routing.route(sp_bad, an, ["cpu", "cuda:0", "cuda:1"], {"cpu": 32.0, "cuda:0": 8.0, "cuda:1": 70.0}, cal, lam4_m)

    # Phonons on CUDA: float32 + precision="fp32_ieee", while explicit exact/float64 selects float64
    ph_spec = validation.validate({"task": "phonons", "structure": {"format": "builtin", "text": "si"}})
    dec_ph = routing.route(ph_spec, an, ["cpu", "cuda:0"], {"cpu": 32.0, "cuda:0": 40.0}, cal, lam4_m)
    assert dec_ph["device"] == "cuda:0" and dec_ph["dtype"] == "float32" and dec_ph["precision"] == "fp32_ieee"

    ph_exact = validation.validate({"task": "phonons", "structure": {"format": "builtin", "text": "si"}, "runtime": {"dtype": "float64"}})
    dec_ph_ex = routing.route(ph_exact, an, ["cpu", "cuda:0"], {"cpu": 32.0, "cuda:0": 40.0}, cal, lam4_m)
    assert dec_ph_ex["dtype"] == "float64" and dec_ph_ex["precision"] == "exact"

    ph_exact_prec = copy_spec_with_precision(ph_spec, "exact")
    dec_ph_ex2 = routing.route(ph_exact_prec, an, ["cpu", "cuda:0"], {"cpu": 32.0, "cuda:0": 40.0}, cal, lam4_m)
    assert dec_ph_ex2["dtype"] == "float64" and dec_ph_ex2["precision"] == "exact"

    # Phonons on CPU still defaults to float64 + exact
    dec_ph_cpu = routing.route(ph_spec, an, ["cpu"], {"cpu": 32.0}, cal, lam4_m)
    assert dec_ph_cpu["dtype"] == "float64" and dec_ph_cpu["precision"] == "exact"


def copy_spec_with_precision(spec: dict, prec: str) -> dict:
    return {**spec, "runtime": {**spec["runtime"], "precision": prec}}


def test_planner_error_reports_vram_on_cuda():
    spec = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "si"}, "runtime": {"device": "cuda:0"}})
    snap = {"os": "Linux", "arch": "x86_64", "ram_gb": 64.0, "available_gb": 50.0, "devices": ["cpu", "cuda:0"], "device_free_gb": {"cpu": 50.0, "cuda:0": 0.2}}
    cal = {"source": "measured", "points": [{"domain": "crystal", "device": "cuda", "backend": "lam4", "x": 1000, "cold_ms": 20.0, "warm_ms": 10.0, "peak_gb": 4.0}]}
    fake_m = {"id": "lam5-v1", "title": "Mena-LAM 5", "backend": "lam4", "lots": ["mp_pbe"], "dtypes": ["float32", "float64"], "low_memory": True, "path": "/tmp/x.pt"}
    with mock.patch("mena_app.planner.machine.snapshot", return_value=snap), \
         mock.patch("mena_app.planner.costmodel.load", return_value=cal), \
         mock.patch("mena_app.planner.models.choose", return_value=fake_m):
        with pytest.raises(planner.PlanError, match="VRAM"):
            planner.plan(spec)


# ---------------------------------------------------------------------------
# 2. Backends precision forwarding, Python API defaults, and Export parity
# ---------------------------------------------------------------------------


def test_backends_forwards_precision_only_when_supported(tmp_path: Path):
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"0")
    model = {"id": "lam5-v1", "backend": "lam4", "path": str(ckpt)}
    dec = {"device": "cuda:1", "dtype": "float32", "precision": "fp32_ieee", "lot": "mp_pbe", "skin": 0.0, "low_memory": False, "compile": False}

    class DummyWithoutPrecision:
        def __init__(self, path, lot="auto", device="auto", dtype="float32", skin=0.0, low_memory=False, compile=False):
            self.kwargs = dict(path=path, device=device, dtype=dtype)

        @classmethod
        def from_checkpoint(cls, path, **kw):
            return cls(path, **kw)

    class DummyWithPrecision(DummyWithoutPrecision):
        def __init__(self, path, lot="auto", device="auto", dtype="float32", precision="auto", skin=0.0, low_memory=False, compile=False):
            super().__init__(path, lot=lot, device=device, dtype=dtype)
            self.kwargs["precision"] = precision

    import lam4.calc as lam4_calc
    with mock.patch.object(lam4_calc, "MenaCalculator", DummyWithoutPrecision):
        c1 = backends.build(model, dec)
        assert c1.kwargs == {"path": str(ckpt), "device": "cuda:1", "dtype": "float32"}

    with mock.patch.object(lam4_calc, "MenaCalculator", DummyWithPrecision):
        c2 = backends.build(model, dec)
        assert c2.kwargs["precision"] == "fp32_ieee"


def test_python_api_defaults_device_auto_and_resolution():
    sig = inspect.signature(python_api.calculator)
    assert sig.parameters["model"].default == "auto"
    assert sig.parameters["device"].default == "auto"
    assert sig.parameters["precision"].default == "auto"
    for fn in (python_api.singlepoint, python_api.relax, python_api.run_md):
        assert inspect.signature(fn).parameters["model"].default == "auto"

    with mock.patch("torch.cuda.is_available", return_value=True):
        assert python_api._resolve_auto_device("float32") == "cuda"
        assert python_api._resolve_auto_device("float64") == "cuda"
    with mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch.object(torch.backends.mps, "is_available", return_value=True):
        assert python_api._resolve_auto_device("float32") == "mps"
        assert python_api._resolve_auto_device("float64") == "cpu"


def test_export_python_npt_uses_langevinbaoab_and_neb_pops_lot():
    dec = {"model": "lam5-v1", "device": "cpu", "dtype": "float32", "lot": "mp_pbe", "skin": 0.6, "low_memory": False, "compile": False}
    model = {"id": "lam5-v1", "backend": "lam4", "path": "/tmp/m.pt"}

    md_job = validation.validate({"task": "md", "structure": {"format": "builtin", "text": "si"}, "params": {"ensemble": "npt", "steps": 50}})
    md_plan = {"decision": dec, "model": model, "analysis": structures.analyze(structures.load("si"))}
    py_md = export.to_python(md_job, md_plan, "0.0.0")
    assert "LangevinBAOAB" in py_md

    xyz_end = "4\n\nN 0 0 0\nH 1 0 0\nH 0 1 0\nH 0 0 -1\n"
    neb_job = validation.validate({
        "task": "neb",
        "structure": {"format": "builtin", "text": "nh3_up"},
        "structure2": {"format": "xyz", "source": "end.xyz", "text": xyz_end},
    })
    neb_plan = {"decision": {**dec, "lot": "omol_wb97mv"}, "model": model, "analysis": structures.analyze(structures.load("nh3_up"))}
    py_neb = export.to_python(neb_job, neb_plan, "0.0.0")
    assert 'end_atoms.info.pop("lot", None)' in py_neb


# ---------------------------------------------------------------------------
# 3. Calibrate, Benchmark & Runner CUDA memory isolation
# ---------------------------------------------------------------------------


def test_calibrate_benchmark_and_runner_cuda_vram_accounting():
    with mock.patch("mena_app.system.process_rss_bytes", return_value=int(4.0e9)), \
         mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.device"), \
         mock.patch("torch.cuda.max_memory_allocated", return_value=int(1.5e9)), \
         mock.patch("torch.cuda.max_memory_reserved", return_value=int(2.2e9)):
        stats = calibrate._mem_stats("cuda:1")
        assert stats["peak_gb"] == pytest.approx(2.2, abs=1e-4)
        assert stats["rss_gb"] == pytest.approx(4.0, abs=1e-4)
        assert stats["gpu_allocated_gb"] == pytest.approx(1.5, abs=1e-4)
        assert stats["gpu_reserved_gb"] == pytest.approx(2.2, abs=1e-4)

        bstats = benchmark_task._mem_breakdown("cuda:1")
        assert bstats["peak_gb"] == pytest.approx(2.2, abs=1e-4)
        assert bstats["host_rss_gb"] == pytest.approx(4.0, abs=1e-4)

    assert calibrate._is_cuda("cuda:1") is True

    with mock.patch("mena_app.system.available_vram_bytes", return_value=int(5.5e9)):
        assert benchmark_task._free_device_gb({"device": "cuda:1"}) == pytest.approx(5.5, abs=1e-4)


def test_runner_vram_watchdog_stops_worker_on_low_cuda_vram(tmp_path: Path):
    plan = {"model": {"id": "lam5-v1", "backend": "lam4", "path": "/tmp/m.pt"}, "decision": {"device": "cuda:1"}}
    events = []
    called_idx = []

    class _FakeProc:
        def __init__(self):
            self.pid = 999999
            self.stdin = mock.Mock()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode if self.returncode is not None else 0

    def _fake_vram(idx):
        called_idx.append(idx)
        return int(0.05 * 1e9)  # below VRAM_KILL_BELOW_GB = 0.15

    with mock.patch("subprocess.Popen", return_value=_FakeProc()), \
         mock.patch("mena_app.runner._signal", side_effect=lambda p, s: p.terminate()), \
         mock.patch("mena_app.system.available_memory_bytes", return_value=int(32e9)), \
         mock.patch("mena_app.system.available_vram_bytes", side_effect=_fake_vram), \
         mock.patch("time.sleep"):
        out = runner.run(plan, {}, str(tmp_path), events.append, threading.Event())
    assert called_idx == [1]
    assert out["status"] == "stopped"
    assert "cuda:1" in out["error"] and "VRAM" in out["error"]


# ---------------------------------------------------------------------------
# 4. Task runtime optimizations: common, md, neb, phonons
# ---------------------------------------------------------------------------


def _dummy_calculator(with_batch: bool = True, with_md_engine: bool = True):
    from ase.calculators.calculator import Calculator, all_changes

    class _Calc(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self):
            super().__init__()
            self.skin = 0.0
            self.dtype = "float32"
            self.calls = 0
            self.batch_calls = 0
            self.md_engine_calls = 0

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            pos = atoms.get_positions()
            self.results = {
                "energy": float(0.5 * np.sum(pos**2)),
                "forces": -pos.copy(),
                "stress": np.zeros(6, dtype=float),
            }

        def eval_batch(self, atoms_list, properties=("energy", "forces"), **kwargs):
            self.batch_calls += 1
            return [{"energy": float(0.5 * np.sum(a.positions**2)), "forces": -a.positions.copy()} for a in atoms_list]

        def md_engine(self, atoms, **kwargs):
            self.md_engine_calls += 1

            class _Eng:
                def __init__(self, at):
                    self._at = at.copy()
                    self.window = int(kwargs.get("window", 10))
                    self.stats = {"steps": 0}
                    self.st = SimpleNamespace(p=torch.zeros((len(at), 3), dtype=torch.float64))

                def window_step(self):
                    return self.run(self.window)

                def run(self, n):
                    self.stats["steps"] += n
                    self._at.positions = self._at.positions + 1e-4 * n
                    return {"steps": n}

                def positions(self):
                    return self._at.positions.copy()

                def energy_ev(self):
                    return float(0.5 * np.sum(self._at.positions**2))

                def forces(self):
                    return -self._at.positions.copy()

            return _Eng(atoms)

    if not with_batch:
        delattr(_Calc, "eval_batch")
    if not with_md_engine:
        delattr(_Calc, "md_engine")
    return _Calc()


class _FakeCtx:
    def __init__(self, out_dir: Path, spec: dict):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.params = spec["params"]
        self.system = spec["system"]
        self.decision = {"device": "cpu", "dtype": "float32", "lot": "auto", "skin": 0.0, "low_memory": False, "compile": False}
        self.model = {"id": "lam5-v1", "title": "Mena-LAM 5", "backend": "lam4"}
        self.cancelled = False
        self.frames = 0

    def status(self, *a, **k): pass
    def log(self, *a, **k): pass
    def check(self): pass
    def frame(self, *a, **k):
        self.frames += 1
    def series(self, *a, **k): pass
    def progress(self, *a, **k): pass
    def flush(self): pass
    def path(self, name: str) -> Path:
        return self.out_dir / name


def test_common_optimize_throttles_observer_cadence(tmp_path: Path):
    at = molecule("H2O")
    at.rattle(0.05, seed=1)
    at.calc = _dummy_calculator(with_batch=False, with_md_engine=False)
    ctx = _FakeCtx(tmp_path, validation.validate({"task": "relax", "structure": {"format": "builtin", "text": "water"}}))
    res = task_common.optimize(ctx, at, "LBFGS", 1e-6, 120, chart="opt", label="relax")
    assert res["steps"] >= 3
    assert len(res["frames"]) < len(res["energies"]) or len(res["frames"]) == 2
    assert ctx.frames == len(res["frames"])


def test_neb_uses_eval_batch_for_intermediates(tmp_path: Path):
    a = molecule("NH3")
    b = a.copy()
    b.positions[:, 2] = -b.positions[:, 2]
    calc = _dummy_calculator(with_batch=True, with_md_engine=False)
    a.calc = calc
    b.calc = calc
    spec = validation.validate({
        "task": "neb",
        "structure": {"format": "builtin", "text": "nh3_up"},
        "structure2": {"format": "builtin", "text": "nh3_down"},
        "params": {"images": 5, "max_steps": 6, "fmax": 0.05, "relax_ends": False},
    })
    ctx = _FakeCtx(tmp_path, spec)
    with mock.patch("mena_app.tasks.neb.setup", side_effect=[a, b]):
        res = neb_task.run(ctx)
    assert calc.batch_calls >= 1
    assert len(res["raw"]["energies_eV"]) == 7


def test_phonons_bumps_skin_and_uses_eval_batch_on_float32(tmp_path: Path):
    at = bulk("Si", "diamond", a=5.43, cubic=True)
    calc = _dummy_calculator(with_batch=True, with_md_engine=False)
    calc.skin = 0.0
    calc.dtype = "float32"
    at.calc = calc
    spec = validation.validate({
        "task": "phonons",
        "structure": {"format": "builtin", "text": "si"},
        "params": {"prerelax": False, "supercell": 1, "kpoints": 4},
    })
    ctx = _FakeCtx(tmp_path, spec)
    with mock.patch("mena_app.tasks.phonons.setup", return_value=at):
        res = phonons_task.run(ctx)
    assert calc.batch_calls >= 1
    assert calc.skin == 0.0  # cleanly restored after bump to >= 0.3
    assert "band_min_THz" in res["raw"]


def test_md_throttles_ase_observer_and_routes_unconstrained_nvt_to_md_engine(tmp_path: Path):
    at = bulk("Si", "diamond", a=5.43, cubic=True)
    spec = validation.validate({
        "task": "md",
        "structure": {"format": "builtin", "text": "si"},
        "params": {"ensemble": "nvt", "steps": 120, "save_every": 20, "temperature": 300.0},
    })

    # 1. MDEngine supported & no D3/constraints -> routes straight to calc.md_engine!
    c1 = _dummy_calculator(with_batch=False, with_md_engine=True)
    a1 = at.copy()
    a1.calc = c1
    ctx1 = _FakeCtx(tmp_path / "mde", spec)
    with mock.patch("mena_app.tasks.md.setup", return_value=a1):
        res1 = md_task.run(ctx1)
    assert c1.md_engine_calls == 1
    assert len(res1["raw"]["time_ps"]) == 7  # step 0 + steps 20..120

    # 2. Without MDEngine -> ASE observer runs cleanly with interval=save_every
    c2 = _dummy_calculator(with_batch=False, with_md_engine=False)
    a2 = at.copy()
    a2.calc = c2
    ctx2 = _FakeCtx(tmp_path / "ase", spec)
    with mock.patch("mena_app.tasks.md.setup", return_value=a2):
        res2 = md_task.run(ctx2)
    assert len(res2["raw"]["time_ps"]) == 7
    assert ctx2.frames == 7  # initial step 0 + steps 20, 40, 60, 80, 100, 120


# ---------------------------------------------------------------------------
# Phase B: MDEngine on-device D3 eligibility, costmodel cuda:N, CUDA_VISIBLE_DEVICES
# ---------------------------------------------------------------------------


def test_md_engine_routes_on_device_d3_and_safely_falls_back():
    from ase import Atoms

    h2o = molecule("H2O")
    h2o.info.update(charge=0, spin=1)
    si = bulk("Si", "diamond", a=5.43, cubic=True)

    def _mk_calc(device: str, dtype=torch.float32, lot: str = "omol_wb97mv", add_dispersion: bool = True):
        return SimpleNamespace(
            device=torch.device(device),
            dtype=dtype,
            lot=lot,
            u_override=None,
            add_dispersion=add_dispersion,
            _resolve_lot_and_u=lambda *a, **k: None,
            md_engine=lambda *a, **k: None,
        )

    p_nvt = {"ensemble": "nvt"}
    with mock.patch("lam4.calc.d3_nl_cache.has_triton_d3_on", return_value=True):
        assert md_task._can_use_md_engine(p_nvt, h2o, _mk_calc("cuda:0", torch.float32)) is True
        assert md_task._can_use_md_engine(p_nvt, si, _mk_calc("cuda:1", torch.float32, lot="omc25_pbe")) is True
        assert md_task._can_use_md_engine(p_nvt, h2o, _mk_calc("cuda:0", torch.float32, lot="spice_wb97md3")) is True
        # float64 CUDA falls back cleanly
        assert md_task._can_use_md_engine(p_nvt, h2o, _mk_calc("cuda:0", torch.float64)) is False
        # Unsupported d3 mode falls back cleanly
        bad_src = SimpleNamespace(d3_addback="unsupported_d4")
        with mock.patch("lam4.calc.physics.resolve_lot_and_u", return_value=(bad_src, 0)):
            assert md_task._can_use_md_engine(p_nvt, h2o, _mk_calc("cuda:0", torch.float32)) is False

    # Triton D3 disabled/unavailable -> D3 LOTS fall back safely, non-D3 mp_pbe still routes
    with mock.patch("lam4.calc.d3_nl_cache.has_triton_d3_on", return_value=False):
        assert md_task._can_use_md_engine(p_nvt, h2o, _mk_calc("cuda:0", torch.float32)) is False
        assert md_task._can_use_md_engine(p_nvt, si, _mk_calc("cuda:0", torch.float32, lot="omc25_pbe")) is False
        assert md_task._can_use_md_engine(p_nvt, si, _mk_calc("cuda:0", torch.float32, lot="mp_pbe")) is True

    # MPS bj_wb97m & periodic zero_pbe (including >1024 via Metal D3 CSR) use Metal D3 when available; disabled or unsupported falls back
    with mock.patch("lam4.kernels.metal_d3.has_metal_d3", return_value=True):
        assert md_task._can_use_md_engine(p_nvt, h2o, _mk_calc("mps", torch.float32)) is True
        assert md_task._can_use_md_engine(p_nvt, si, _mk_calc("mps", torch.float32, lot="omc25_pbe")) is True
        big_mol = Atoms("H" * 1025, positions=np.zeros((1025, 3)))
        assert md_task._can_use_md_engine(p_nvt, big_mol, _mk_calc("mps", torch.float32)) is True
        assert md_task._can_use_md_engine(p_nvt, h2o, _mk_calc("mps", torch.float64)) is False

    with mock.patch("lam4.kernels.metal_d3.has_metal_d3", return_value=False):
        assert md_task._can_use_md_engine(p_nvt, h2o, _mk_calc("mps", torch.float32)) is False
        assert md_task._can_use_md_engine(p_nvt, si, _mk_calc("mps", torch.float32, lot="omc25_pbe")) is False

    # CPU with D3 falls back safely
    assert md_task._can_use_md_engine(p_nvt, h2o, _mk_calc("cpu", torch.float32)) is False


def test_costmodel_normalizes_multi_gpu_cuda_devices():
    from mena_app import costmodel

    assert costmodel.compile_speedup("cuda:1", "lam4") == costmodel.compile_speedup("cuda", "lam4")
    assert costmodel.compiled_seconds(120.0, "cuda:1", "lam4") == pytest.approx(
        costmodel.compiled_seconds(120.0, "cuda", "lam4")
    )
    cal = {
        "source": "measured",
        "points": [
            {"domain": "crystal", "device": "cuda", "backend": "lam4", "x": 10000, "cold_ms": 40.0, "warm_ms": 20.0, "peak_gb": 6.0},
        ],
    }
    c_cuda = costmodel.curve(cal, "crystal", "cuda", 10000, backend="lam4")
    c_gpu1 = costmodel.curve(cal, "crystal", "cuda:1", 10000, backend="lam4")
    assert c_cuda is not None and c_gpu1 == c_cuda


def test_cuda_visible_devices_maps_logical_index_to_physical_vram_and_machine_snapshot(monkeypatch):
    sample = (
        "0, NVIDIA A100-SXM4-80GB, 5000, 81920, GPU-0000\n"
        "1, NVIDIA A100-SXM4-80GB, 12000, 81920, GPU-1111\n"
        "2, NVIDIA A100-SXM4-80GB, 70000, 81920, GPU-2222\n"
    )
    ok_res = SimpleNamespace(returncode=0, stdout=sample)
    with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
         mock.patch("subprocess.run", return_value=ok_res):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,1")
        assert system.available_vram_bytes(0) == 70000 * 1024 * 1024
        assert system.available_vram_bytes(1) == 12000 * 1024 * 1024
        assert system.available_vram_bytes(2) is None

        probe_res = {
            "cuda_devices": [
                {"id": "cuda:0", "index": 0, "name": "NVIDIA A100-SXM4-80GB", "vram_total_gb": 80.0, "vram_free_gb": 1.0},
                {"id": "cuda:1", "index": 1, "name": "NVIDIA A100-SXM4-80GB", "vram_total_gb": 80.0, "vram_free_gb": 1.0},
            ]
        }
        live_devs = machine._live_cuda_devices(probe_res)
        assert live_devs[0]["vram_free_gb"] == round(70000 * 1024 * 1024 / 1e9, 2)
        assert live_devs[1]["vram_free_gb"] == round(12000 * 1024 * 1024 / 1e9, 2)

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-2222")
        assert system.available_vram_bytes(0) == 70000 * 1024 * 1024
        assert system.available_vram_bytes(1) is None

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        assert system.available_vram_bytes(0) is None
