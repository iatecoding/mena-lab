"""Regression tests for lam6 support and bugfixes in mena_app."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from mena_app import calibrate, catalog, costmodel, crystal_symmetry, export, jobs, models, python_api, routing, structures, validation
from mena_app.tasks import benchmark as benchmark_task
from mena_app.tasks import md as md_task
from mena_app.tasks import neb, phonons


def test_python_api_kwarg_aliases_work_with_var_kwargs() -> None:
    def dummy_relax(atoms, *, fmax=0.05, steps=200, **kwargs):
        return fmax, steps, kwargs

    mapped = python_api._map_method_kwargs(
        dummy_relax,
        {"max_steps": 15, "fmax": 0.01, "skin": 0.4},
        {"max_steps": "steps"},
    )
    assert mapped == {"steps": 15, "fmax": 0.01, "skin": 0.4}
    assert dummy_relax(None, **mapped) == (0.01, 15, {"skin": 0.4})

    def dummy_md(atoms, *, temp_K=300.0, dt_fs=1.0, steps=100, friction_ps=0.01, traj_every=10, **kwargs):
        return temp_K, dt_fs, steps, friction_ps, traj_every, kwargs

    mapped_md = python_api._map_method_kwargs(
        dummy_md,
        {"temperature": 350.0, "timestep": 0.5, "friction": 0.02, "save_every": 5, "record": True},
        {"temperature": "temp_K", "timestep": "dt_fs", "friction": "friction_ps", "save_every": "traj_every"},
    )
    assert mapped_md["temp_K"] == 350.0
    assert mapped_md["dt_fs"] == 0.5
    assert mapped_md["friction_ps"] == 0.02
    assert mapped_md["traj_every"] == 5
    assert mapped_md["record"] is True


def test_models_add_lam6_gets_lam6_lots_and_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENA_HOME", str(tmp_path))
    ckpt = tmp_path / "lam6_v1.pt"
    ckpt.write_bytes(b"stub")
    entry = models.add("lam6-v1", str(ckpt))
    assert entry["backend"] == "lam4"
    assert entry["rank"] == {"crystal": 0, "slab": 0, "molecule": 1}
    assert "oc25_rpbe" in entry["lots"]
    assert "mad16_r2scan" in entry["lots"]
    assert "omat_pbeU" in entry["lots"]


def test_routing_lam6_aliases_slab_fallback_cuda_f64_and_advanced_knobs() -> None:
    cal = costmodel.load()
    lam6_m = {
        "id": "lam6-test",
        "title": "Mena-LAM 6",
        "backend": "lam4",
        "domains": ["molecule", "crystal", "slab"],
        "dtypes": ["float32", "float64"],
        "lots": list(models.LAM6_DEFAULT_LOTS),
    }

    # 1. Alias resolution (mp_pbe52 -> mp_pbe) + advanced runtime knobs validation/export
    sp = validation.validate({
        "task": "relax",
        "structure": {"format": "builtin", "text": "si"},
        "runtime": {
            "lot": "mp_pbe52",
            "skin": 0.4,
            "edge_mode": "csr",
            "add_dispersion": "off",
            "u_flag": "0",
        },
    })
    an_si = structures.analyze(structures.load("si"))
    dec = routing.route(sp, an_si, ["cpu"], 32.0, cal, lam6_m)
    assert dec["lot"] == "mp_pbe"
    assert dec["resolved_lot"] == "mp_pbe"
    assert dec["skin"] == 0.4
    assert dec["edge_mode"] == "csr"
    assert dec["add_dispersion"] is False
    assert dec["u_flag"] == 0

    cli_txt = export.to_cli(sp)
    assert "--skin 0.4" in cli_txt
    assert "--edge-mode csr" in cli_txt
    assert "--add-dispersion off" in cli_txt
    assert "--u-flag 0" in cli_txt

    pl = {"analysis": an_si, "decision": dec, "model": {**lam6_m, "path": "/tmp/lam6.pt"}, "model_public": lam6_m}
    py_txt = export.to_python(sp, pl, "test")
    assert "edge_mode='csr'" in py_txt
    assert "add_dispersion=False" in py_txt
    assert "u_flag=0" in py_txt

    # 2. Oxide slab on lam6 (no oc22_pbeU) routes to omat_pbeU (and sets resolved_lot)
    feo_slab = Atoms(
        "Fe2O2",
        positions=[[0, 0, 4.0], [1.5, 1.5, 4.0], [1.5, 0, 4.8], [0, 1.5, 4.8]],
        cell=[[3.1, 0, 0], [0, 3.1, 0], [0, 0, 14.0]],
        pbc=[True, True, False],
    )
    an_feo = structures.analyze(feo_slab)
    sp_slab = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "cu111"}})
    dec_feo = routing.route(sp_slab, an_feo, ["cpu"], 32.0, cal, lam6_m)
    assert dec_feo["lot"] == "omat_pbeU"
    assert dec_feo["resolved_lot"] == "omat_pbeU"

    # 3. Legacy 'lam' backend on CUDA phonons stays float64 (only lam4 has fp32_ieee)
    legacy_m = {
        "id": "lam-400k",
        "title": "Legacy",
        "backend": "lam",
        "domains": ["crystal"],
        "dtypes": ["float32", "float64"],
        "lots": [],
    }
    sp_ph = validation.validate({"task": "phonons", "structure": {"format": "builtin", "text": "si"}})
    dec_ph = routing.route(sp_ph, an_si, ["cuda", "cpu"], 32.0, cal, legacy_m)
    assert dec_ph["dtype"] == "float64"


def test_phonons_all_imaginary_dos_and_neb_terms_and_cuda_family() -> None:
    # All-imaginary phonon DOS must not divide by zero or return NaN
    freqs = np.array([-3.0, -2.0, -1.0, 0.0], dtype=float)
    dos = np.array([1.0, 2.0, 1.0, 0.5], dtype=float)
    th = phonons.free_energy(freqs, dos, 300.0, n_modes=6)
    assert th["f"] == 0.0
    assert th["s"] == 0.0
    assert th["cv"] == 0.0

    # NEB decomposition tracks lam6 cloud_overlap & spin_exchange
    keys = {row[0] for row in neb._DECOMP_TERMS}
    assert "cloud_overlap" in keys
    assert "spin_exchange" in keys

    # Multi-GPU CUDA device string maps to 'cuda'
    assert costmodel._device_family("cuda:1") == "cuda"
    assert costmodel._device_family("mps:0") == "mps"


def test_job_manager_cancel_queued_job_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time

    monkeypatch.setenv("MENA_HOME", str(tmp_path))
    blocker = threading.Event()

    class BlockingEngine:
        def run(self, spec, meta, emit, cancel, workdir):
            blocker.wait(timeout=5.0)
            return {"status": "done", "result": {"ok": True}}

    jm = jobs.JobManager(BlockingEngine())
    try:
        sp = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}})
        j1 = jm.submit(sp, {"test_title": "SP 1"})
        j2 = jm.submit(sp, {"test_title": "SP 2"})
        # Wait briefly until j1 starts running while j2 is still queued
        t0 = time.monotonic()
        while j1.status == "queued" and time.monotonic() - t0 < 2.0:
            time.sleep(0.01)
        assert j2.status == "queued"
        assert jm.cancel(j2.id) is True
        assert j2.status == "cancelled"
        assert (j2.workdir / "job.json").is_file()
        assert '"cancelled"' in (j2.workdir / "job.json").read_text(encoding="utf-8")
    finally:
        blocker.set()
        jm.shutdown()


def test_export_preserves_string_compile_modes() -> None:
    sp = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}})
    an = structures.analyze(structures.load("water"))
    for backend in ("lam4", "lam"):
        for mode, expect in (
            ("reduce-overhead", "compile='reduce-overhead'"),
            ("traced", "compile='traced'"),
            (True, "compile=True"),
            (False, None),
            ("off", None),
        ):
            pl = {
                "analysis": an,
                "decision": {"device": "cpu", "dtype": "float32", "lot": "auto", "skin": 0.0, "low_memory": False, "compile": mode},
                "model": {"id": "m", "title": "M", "backend": backend, "path": "/tmp/m.pt"},
                "model_public": {"id": "m", "title": "M", "backend": backend},
            }
            code = export.to_python(sp, pl, "0.1.0")
            if expect is None:
                assert "compile=" not in code
            else:
                assert expect in code


def test_calibrate_selects_default_model_per_domain_within_shared_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    declared = [
        {"id": "lam6-v1", "backend": "lam4", "domains": ["crystal", "slab", "molecule"], "rank": {"crystal": 0, "slab": 0, "molecule": 1}, "available": True},
        {"id": "lam5-v1", "backend": "lam4", "domains": ["molecule", "crystal", "slab"], "rank": {"molecule": 0, "crystal": 2, "slab": 2}, "available": True},
        {"id": "lam-400k", "backend": "lam", "domains": ["crystal", "slab", "molecule"], "rank": {"crystal": 1, "slab": 1, "molecule": 3}, "available": True},
    ]
    monkeypatch.setattr(models, "all_models", lambda: declared)
    targets = {m["id"]: set(doms) for m, doms in calibrate._select_calibration_targets(None)}
    assert targets == {"lam6-v1": {"crystal"}, "lam5-v1": {"molecule"}, "lam-400k": {"crystal", "molecule"}}
    assert calibrate._select_calibration_targets("lam5-v1") == [(declared[1], ["molecule", "crystal"])]


def test_structures_analyze_calls_crystal_symmetry_once_and_zero_rotation_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    from ase.build import bulk, molecule

    calls = 0
    orig = crystal_symmetry.find_crystal_symmetry

    def _spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(crystal_symmetry, "find_crystal_symmetry", _spy)
    an = structures.analyze(bulk("Si", "diamond", a=5.43, cubic=True))
    assert calls == 1
    assert an["spacegroup"] is not None and an["phonon_evaluations"] == 3

    assert benchmark_task._zero_rotation_safe is md_task._zero_rotation_safe
    co2 = molecule("CO2")
    co2.set_momenta([[0.0, 0.1, 0.0], [0.0, -0.2, 0.0], [0.0, 0.1, 0.0]])
    benchmark_task._zero_rotation_safe(co2)
    assert np.all(np.isfinite(co2.get_momenta()))


def test_interpret_heads_np_cpu_and_mps_order() -> None:
    import torch
    from lam4.calc import interpret_heads

    assert interpret_heads._np(None) is None
    assert interpret_heads._np(123) is None
    out_cpu = interpret_heads._np(torch.tensor([1.5, -2.25], dtype=torch.float32))
    assert out_cpu.dtype == np.float64 and np.allclose(out_cpu, [1.5, -2.25])

    class _FakeMPSTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, data: torch.Tensor):
            return torch.Tensor._make_subclass(cls, data)

        def detach(self):
            return _FakeMPSTensor(self.as_subclass(torch.Tensor).detach())

        def cpu(self):
            return self.as_subclass(torch.Tensor).cpu()

        def to(self, *args, **kwargs):
            if torch.float64 in args or kwargs.get("dtype") == torch.float64:
                raise TypeError("Cannot convert a MPS Tensor to float64 dtype")
            return self.as_subclass(torch.Tensor).to(*args, **kwargs)

    fake_mps = _FakeMPSTensor(torch.tensor([0.75, -0.25], dtype=torch.float32))
    out_sim = interpret_heads._np(fake_mps)
    assert out_sim.dtype == np.float64 and np.allclose(out_sim, [0.75, -0.25])

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        out_mps = interpret_heads._np(torch.tensor([0.75, -0.25], device="mps", dtype=torch.float32))
        assert out_mps.dtype == np.float64 and np.allclose(out_mps, [0.75, -0.25])


def test_cuda_ext_and_md_nhc_fallbacks_when_build_or_compile_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    import torch.utils.cpp_extension as cpp_ext
    from lam4.calc import md_nhc
    from lam4.kernels import cuda_ext

    # 1. cuda_ext toolchain check returns False cleanly when ninja is absent, and load_cuda_ext returns None on error
    with monkeypatch.context() as mp:
        mp.setattr(cuda_ext, "_TOOLCHAIN_OK", None)
        mp.setattr(cpp_ext, "is_ninja_available", lambda: False)
        assert cuda_ext.cuda_toolchain_ok() is False
        mp.setattr(torch.cuda, "is_available", lambda: True)
        assert cuda_ext.cuda_ext_enabled(dev="cuda") is False

    with monkeypatch.context() as mp:
        calls = 0

        def _fail_inline(**_kw):
            nonlocal calls
            calls += 1
            raise RuntimeError("missing nvcc")

        mp.setattr(cuda_ext, "_EXT_CACHE", dict(cuda_ext._EXT_CACHE))
        mp.setattr(cuda_ext, "_EXT_FAILED", dict(cuda_ext._EXT_FAILED))
        mp.setattr(cpp_ext, "load_inline", _fail_inline)
        with pytest.warns(RuntimeWarning, match="CUDA extension test_fail_ext unavailable"):
            assert cuda_ext.load_cuda_ext("test_fail_ext", "", "", []) is None
        assert "test_fail_ext" in cuda_ext._EXT_FAILED
        assert cuda_ext.load_cuda_ext("test_fail_ext", "", "", []) is None
        assert calls == 1  # cached failure avoids retrying load_inline
        assert cuda_ext.cuda_ext_enabled(dev="cuda", ext_name="test_fail_ext") is False

    # 2. md_nhc falls back cleanly to eager both on torch.compile failure and lazy runtime compile failure
    def _nhc_tensors():
        return (
            torch.zeros(2, 3),
            torch.ones(2, 3) * 0.1,
            torch.zeros(2, 3),
            torch.zeros(3),
            torch.zeros(3),
            torch.ones(2, 1),
            torch.ones(3),
            torch.tensor(0.025),
            torch.tensor(0.5),
            torch.tensor(6.0),
        )

    with monkeypatch.context() as mp:
        mp.setattr(md_nhc, "_COMPILED_NHC", None)
        mp.setattr(md_nhc, "_NHC_COMPILE_FAILED", False)
        mp.setattr(md_nhc, "_NHC_COMPILED_OK", False)
        mp.setattr(torch, "compile", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no triton")))
        with pytest.warns(RuntimeWarning, match="eager NHC"):
            pre, post = md_nhc.get_fused_nhc_steps(torch.device("cuda"))
        assert pre is md_nhc._nhc_pre_step_eager and post is md_nhc._nhc_post_step_eager
        assert md_nhc._NHC_COMPILE_FAILED is True

    with monkeypatch.context() as mp:
        mp.setattr(md_nhc, "_COMPILED_NHC", None)
        mp.setattr(md_nhc, "_NHC_COMPILE_FAILED", False)
        mp.setattr(md_nhc, "_NHC_COMPILED_OK", False)
        mp.setattr(torch, "compile", lambda fn, **kw: (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("inductor failed"))))
        pre, _ = md_nhc.get_fused_nhc_steps(torch.device("cuda"))
        with pytest.warns(RuntimeWarning, match="falling back to eager"):
            got = pre(*_nhc_tensors())
        ref = md_nhc._nhc_pre_step_eager(*_nhc_tensors())
        assert all(torch.allclose(a, b) for a, b in zip(got, ref))
        assert md_nhc._NHC_COMPILE_FAILED is True


def test_md_traced_imports_without_lam4_train() -> None:
    import subprocess
    import sys

    probe = (
        "import importlib.abc, sys\n"
        "class _Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'lam4.train' or name.startswith('lam4.train.'): raise ModuleNotFoundError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "import lam4.calc.md_traced as m\n"
        "assert callable(m.strip_autograd_detach_chains) and callable(m.alias_to_detach)\n"
        "assert not any(k == 'lam4.train' or k.startswith('lam4.train.') for k in sys.modules)"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_smiles_planner_notes_and_md_prerelax(tmp_path: Path) -> None:
    import io
    from mena_app import export, planner, runtime, validation
    from mena_app.tasks import md as md_task

    spec_md = validation.validate({"task": "md", "structure": {"format": "smiles", "text": "CCO"}, "params": {"steps": 4, "save_every": 2}})
    pl_md = planner.plan(spec_md)
    assert pl_md["notes"] and "Relax first" in pl_md["notes"][0]

    spec_md_pre = validation.validate(
        {"task": "md", "structure": {"format": "smiles", "text": "CCO"}, "params": {"steps": 4, "save_every": 2, "prerelax": True}}
    )
    pl_md_pre = planner.plan(spec_md_pre)
    assert pl_md_pre["notes"] == []
    assert "LBFGS(atoms).run" in export.to_python(spec_md_pre, pl_md_pre, "0.1.0")

    spec_sp = validation.validate({"task": "singlepoint", "structure": {"format": "smiles", "text": "CCO"}})
    assert any("approximate 3D conformer" in n for n in planner.plan(spec_sp)["notes"])

    spec_builtin = validation.validate({"task": "md", "structure": {"format": "builtin", "text": "si"}, "params": {"steps": 4}})
    assert planner.plan(spec_builtin)["notes"] == []

    if pl_md_pre["model"].get("available"):
        ctx = runtime.Context(
            {
                "spec": spec_md_pre,
                "decision": {**pl_md_pre["decision"], "device": "cpu", "compile": False},
                "model": pl_md_pre["model"],
                "workdir": str(tmp_path),
            },
            io.StringIO(),
        )
        res = md_task.run(ctx)
        assert res["facts"]["Steps"] == 4
        assert res["facts"].get("Pre-relax steps", 0) >= 1
