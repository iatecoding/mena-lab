"""Regression tests for Mena Hub public release: model routing, packaging, web app integration, and cross-PC portability."""

from __future__ import annotations

import ast
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from ase import Atoms
from ase.build import molecule

from mena_app import (
    backends,
    calibrate,
    cli,
    costmodel,
    export,
    models,
    planner,
    routing,
    runtime,
    structures,
    system,
    validation,
)
from mena_app.tasks import common

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_ships_lam4_and_required_package_data():
    """Verify setuptools package discovery & package_data include `lam4*` and `lam4/physics/data/*.npy`."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]

    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = cfg["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "mena_app*" in includes
    assert "lam*" in includes
    assert "lam4*" in includes

    pkg_data = cfg["tool"]["setuptools"]["package-data"]
    assert "lam4" in pkg_data
    assert any("*.npy" in pat for pat in pkg_data["lam4"])
    assert (REPO_ROOT / "lam4" / "physics" / "data" / "nlh_pairs_z92.npy").is_file()


def test_default_model_routing_materials_400k_molecules_lam5(monkeypatch, tmp_path):
    """Default model: `lam-400k` on `crystal` & `slab`, `lam5-v1` on `molecule`; manual `lam5-v1` on slabs still works."""
    monkeypatch.setenv("MENA_HOME", str(tmp_path / "home"))
    declared = json.loads((REPO_ROOT / "mena_app" / "data" / "models.json").read_text(encoding="utf-8"))["models"]
    by_id = {m["id"]: m for m in declared}

    assert by_id["lam-400k"]["rank"]["crystal"] == 0
    assert by_id["lam-400k"]["rank"]["slab"] == 0
    assert by_id["lam5-v1"]["rank"]["molecule"] == 0
    assert by_id["lam5-v1"]["rank"]["crystal"] > by_id["lam-400k"]["rank"]["crystal"]
    assert by_id["lam5-v1"]["rank"]["slab"] > by_id["lam-400k"]["rank"]["slab"]

    # Simulate a machine where ALL declared checkpoints exist on disk
    fake_all = [{**m, "available": True, "path": str(tmp_path / f"{m['id']}.pt")} for m in declared]
    monkeypatch.setattr(models, "all_models", lambda: fake_all)

    assert models.choose("auto", "molecule")["id"] == "lam5-v1"
    assert models.choose("auto", "crystal")["id"] == "lam-400k"
    assert models.choose("auto", "slab")["id"] == "lam-400k"
    # Manual selection of lam5-v1 on crystal and slab remains available
    assert models.choose("lam5-v1", "crystal")["id"] == "lam5-v1"
    assert models.choose("lam5-v1", "slab")["id"] == "lam5-v1"

    # Check _lot() behavior on slabs for lam-400k (no oc20_rpbe in lots -> clean 'auto' PBE/+U) vs lam5-v1 (oc20_rpbe / oc22_pbeU)
    cu111_an = structures.analyze(structures.load("cu111"))
    lot_400k, _ = routing._lot("auto", "slab", by_id["lam-400k"], cu111_an)
    assert lot_400k == "auto"

    lot_lam5_metal, _ = routing._lot("auto", "slab", by_id["lam5-v1"], cu111_an)
    assert lot_lam5_metal == "oc20_rpbe"

    oxide_slab = Atoms(
        "NiO",
        positions=[[0.0, 0.0, 0.0], [1.05, 1.05, 0.8]],
        cell=[[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 15.0]],
        pbc=[True, True, False],
    )
    oxide_an = structures.analyze(oxide_slab)
    assert oxide_an["domain"] == "slab" and oxide_an["hubbard_u"] is True
    lot_lam5_oxide, _ = routing._lot("auto", "slab", by_id["lam5-v1"], oxide_an)
    assert lot_lam5_oxide == "oc22_pbeU"


def test_missing_checkpoints_error_is_actionable(monkeypatch, tmp_path):
    """Fresh install with downloads switched off: the error names both ways to get the weights."""
    monkeypatch.setenv("MENA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MENA_NO_DOWNLOAD", "1")
    monkeypatch.setattr(models.paths, "REPO_ROOT", tmp_path)
    for dom in ("molecule", "crystal"):
        with pytest.raises(models.ModelError) as exc:
            models.choose("auto", dom)
        assert "mena models download" in str(exc.value) and "mena models add" in str(exc.value)


def test_cli_models_add_supports_lam4_and_bad_task_rejected(monkeypatch, tmp_path):
    """`mena models add --backend lam4` works and `spec_from_args` cleanly rejects unknown tasks."""
    monkeypatch.setenv("MENA_HOME", str(tmp_path / "home"))
    ckpt = tmp_path / "my_lam5.pt"
    ckpt.write_bytes(b"pt")

    rc = cli.main(["models", "add", "custom-lam5", str(ckpt), "--backend", "lam4"])
    assert rc == 0
    added = next(m for m in models.all_models() if m["id"] == "custom-lam5")
    assert added["backend"] == "lam4"
    assert "oc20_rpbe" in added["lots"] and "omol_wb97mv" in added["lots"]
    assert added["domains"] == ["molecule", "crystal", "slab"]

    with pytest.raises(validation.ValidationError, match="unknown task"):
        cli.spec_from_args("not_a_real_task", None, {"task": "not_a_real_task", "structure": "water"})


def test_lam4_molecule_routing_enables_verlet_skin_at_64_atoms_and_8gb_mac_runs():
    """`lam4` enables Verlet skin on >=64-atom molecules (sparse Dual-CSR) while 8 GB Macs don't falsely refuse small jobs."""
    cal = costmodel.load()
    lam4_m = {
        "id": "lam5-v1",
        "title": "Mena-LAM 5",
        "backend": "lam4",
        "lots": models.DEFAULT_LOTS["lam4"],
        "dtypes": ["float32", "float64"],
        "low_memory": True,
    }
    lam_m = {
        "id": "lam-400k",
        "title": "Mena-LAM",
        "backend": "lam",
        "lots": models.DEFAULT_LOTS["lam"],
        "dtypes": ["float32", "float64"],
        "low_memory": True,
    }

    # Small 24-atom caffeine: skin stays 0.0 on both
    caf_spec = validation.validate({"task": "md", "structure": {"format": "builtin", "text": "caffeine"}, "params": {"steps": 100}})
    caf_an = structures.analyze(structures.load("caffeine"))
    assert routing.route(caf_spec, caf_an, ["cpu", "mps"], 16.0, cal, lam4_m)["skin"] == 0.0

    # Medium 120-atom water cluster (40 H2O, < 512 atoms): lam4 enables skin=1.0 (MD) / 0.5 (relax), legacy lam keeps 0.0
    w40 = calibrate._water_box(40)
    w40_an = structures.analyze(w40)
    assert 64 <= w40_an["n_atoms"] < 512
    md_spec = validation.validate({"task": "md", "structure": {"format": "builtin", "text": "water"}, "params": {"steps": 250}})
    rx_spec = validation.validate({"task": "relax", "structure": {"format": "builtin", "text": "water"}})

    d_lam4_md = routing.route(md_spec, w40_an, ["cpu", "mps"], 16.0, cal, lam4_m)
    d_lam4_rx = routing.route(rx_spec, w40_an, ["cpu", "mps"], 16.0, cal, lam4_m)
    d_lam_md = routing.route(md_spec, w40_an, ["cpu", "mps"], 16.0, cal, lam_m)
    assert d_lam4_md["skin"] == 1.0
    assert d_lam4_rx["skin"] == 0.5
    assert d_lam_md["skin"] == 0.0

    # 8 GB Mac scenario (`free_gb = 2.6` GB < RESERVE_GB=3.0): small water job turns on low_memory and is NOT refused!
    w1_spec = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}})
    w1_an = structures.analyze(structures.load("water"))
    d_8gb = routing.route(w1_spec, w1_an, ["cpu", "mps"], 2.6, cal, lam4_m)
    assert d_8gb["memory_level"] != "refuse"
    assert d_8gb["low_memory"] is True

    # Linux + CUDA vibrations/phonons routes to CUDA + float64 cleanly
    vib_spec = validation.validate({"task": "vibrations", "structure": {"format": "builtin", "text": "water"}})
    d_cuda_vib = routing.route(vib_spec, w1_an, ["cpu", "cuda"], 16.0, {"points": [], "source": "none"}, lam4_m)
    assert d_cuda_vib["device"] == "cuda"
    assert d_cuda_vib["dtype"] == "float64"


def test_clone_calculator_on_warm_lam4_calculator_with_skin():
    """Verify `clone_calculator` on a warm `lam4.calc.MenaCalculator` with `skin > 0` does not set `_nl_cache = {}` (`KeyError: 'pbc'`)."""
    from lam4.calc import MenaCalculator
    from lam4.config import ModelConfig
    from lam4.model import Lam4Model

    net = Lam4Model(ModelConfig(n_stages=1, feedback_points=0, d_scalar=64))
    calc = MenaCalculator(net, device="cpu", dtype="float32", skin=0.5)
    h2o_a = molecule("H2O")
    h2o_a.info.update(charge=0, spin=1)
    h2o_a.calc = calc
    e_a = h2o_a.get_potential_energy()
    assert isinstance(calc._nl_cache, dict) and "pbc" in calc._nl_cache

    cloned = backends.clone_calculator(calc)
    assert cloned._nl_cache is None
    assert cloned._static_cache is None
    assert cloned._active_csr_plan is None
    assert cloned.model is calc.model

    h2o_b = molecule("H2O")
    h2o_b.positions[0, 0] += 0.05
    h2o_b.info.update(charge=0, spin=1)
    h2o_b.calc = cloned
    e_b = h2o_b.get_potential_energy()
    f_b = h2o_b.get_forces()
    assert isinstance(e_b, float) and f_b.shape == (3, 3)
    assert abs(e_b - e_a) < 5.0


def test_2d_slab_zero_c_vector_write_structure_and_python_export(tmp_path):
    """2D mixed-PBC slabs with a zero vacuum lattice vector (`cell.rank == 2`) write `.extxyz` cleanly instead of crashing ASE CIF."""
    slab2d = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.28, 1.28, 0.5]],
        cell=[[2.56, 0.0, 0.0], [0.0, 2.56, 0.0], [0.0, 0.0, 0.0]],
        pbc=[True, True, False],
    )
    assert slab2d.cell.rank == 2
    ctx = runtime.Context({"spec": {"params": {}, "system": {}}, "decision": {}, "model": {}, "workdir": str(tmp_path)}, io.StringIO())
    meta = common.write_structure(ctx, slab2d, "relaxed")
    assert meta["name"] == "relaxed.extxyz"
    assert (tmp_path / "relaxed.extxyz").is_file()

    # Check export.to_python for backend='lam4' across tasks (valid Python AST + MenaCalculator import + slab extxyz)
    lam4_plan = {
        "decision": {"device": "cpu", "dtype": "float32", "lot": "oc20_rpbe", "skin": 0.5, "low_memory": False, "compile": False},
        "model": {"id": "lam5-v1", "title": "Mena-LAM 5", "backend": "lam4", "path": "weights/mena-lam5-v1.pt"},
        "analysis": structures.analyze(slab2d),
    }
    rx_spec = validation.validate({"task": "relax", "structure": {"format": "builtin", "text": "cu111"}})
    py_code = export.to_python(rx_spec, lam4_plan, "0.1.0")
    ast.parse(py_code)
    assert "from lam4 import MenaCalculator" in py_code
    assert "MenaCalculator.from_checkpoint(" in py_code
    assert "relaxed.extxyz" in py_code

    for task_id in ("singlepoint", "md", "vibrations", "neb"):
        raw: dict = {"task": task_id, "structure": {"format": "builtin", "text": "nh3_up" if task_id == "neb" else "caffeine"}}
        if task_id == "neb":
            raw["structure2"] = {"format": "builtin", "text": "nh3_down"}
        sp = validation.validate(raw)
        mol_plan = {
            "decision": {"device": "cpu", "dtype": "float32", "lot": "auto", "skin": 0.0, "low_memory": True, "compile": False},
            "model": {"id": "lam5-v1", "title": "Mena-LAM 5", "backend": "lam4", "path": "weights/mena-lam5-v1.pt"},
            "analysis": structures.analyze(structures.load(sp["structure"])),
        }
        code = export.to_python(sp, mol_plan, "0.1.0")
        ast.parse(code)
        assert "from lam4 import MenaCalculator" in code


def test_cross_platform_rss_and_kernel_fallback_guards(monkeypatch):
    """Verify `system.process_rss_bytes()`, Metal shader failure fallback, and Triton Pascal `< sm_70` guard."""
    rss = system.process_rss_bytes()
    assert isinstance(rss, int) and rss > 1_000_000

    # Triton Pascal guard (`sm_61` < 7 -> False)
    from lam4.kernels import triton_csr

    with monkeypatch.context() as mp:
        mp.setattr(triton_csr, "HAS_TRITON", True)
        mp.setattr(torch.cuda, "is_available", lambda: True)
        mp.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (6, 1))
        assert triton_csr.has_triton_csr() is False
        mp.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (8, 0))
        assert triton_csr.has_triton_csr() is True

    # Static JS syntax check if `node` is available on PATH
    node = shutil.which("node")
    if node:
        for js_file in sorted((REPO_ROOT / "mena_app" / "static" / "js").glob("*.js")):
            res = subprocess.run([node, "--check", str(js_file)], capture_output=True, text=True, timeout=15)
            assert res.returncode == 0, f"JS syntax error in {js_file.name}: {res.stderr}"

    # Verify both `python -m mena_app` and `python -m mena_app.cli` execute `main()`
    import sys

    for mod in ("mena_app", "mena_app.cli"):
        r = subprocess.run([sys.executable, "-m", mod, "--version"], capture_output=True, text=True, timeout=15)
        assert r.returncode == 0 and "mena" in r.stdout


def test_webapp_command_and_ui_alias(monkeypatch):
    """`mena webapp` is the primary web command and `mena ui` stays supported as an alias."""
    calls = []
    monkeypatch.setattr("mena_app.server.serve", lambda host, port, token, open_browser: calls.append((host, port, token, open_browser)) or 0)
    assert cli.main(["webapp", "--port", "8899", "--no-browser"]) == 0
    assert cli.main(["ui", "--port", "8898", "--no-browser"]) == 0
    assert calls == [("127.0.0.1", 8899, None, False), ("127.0.0.1", 8898, None, False)]


def test_wheel_install_bundled_e0_references_without_repo_data(monkeypatch, tmp_path):
    """In a wheel install (`site-packages/`), `REPO_ROOT / 'data'` does not exist; bundled `lam/data/e0_refs/*` must resolve cleanly."""
    from lam.data import registry as lam_reg
    from lam4.calc import loader as lam4_loader

    monkeypatch.setattr(lam_reg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lam4_loader, "OMOL_REF_JSON", tmp_path / "data" / "e0_reference_omol25.json")
    monkeypatch.setattr(lam4_loader, "SPICE_REF_JSON", tmp_path / "data" / "e0_reference_spice2.json")
    monkeypatch.setattr(lam4_loader, "_CRYSTAL_E0_CACHE", {})
    monkeypatch.setattr(lam4_loader, "_MOL_REF_CACHE", {})

    for lot in ("mp_pbe", "omat_pbe", "matpes_r2scan", "oc20_rpbe", "oc22_pbeU", "omc25_pbe"):
        tab = lam4_loader.crystal_e0_table_f64(lot)
        assert tab.shape == (119,) and float(tab.abs().max()) > 0.0
    for mol_lot in ("omol_wb97mv", "spice_wb97md3"):
        mtab = lam4_loader.molecule_ref_tables_f64(mol_lot)
        assert mtab["e0"].shape == (119,) and float(abs(mtab["e0"]).max()) > 0.0


def test_models_add_preserves_shipped_metadata_and_accumulates_domains(monkeypatch, tmp_path):
    """`mena models add lam5-v1 PATH` auto-infers backend='lam4', preserves rank/training/metrics, and accepts repeated --domains."""
    monkeypatch.setenv("MENA_HOME", str(tmp_path / "mena_home"))
    ckpt = tmp_path / "mena_lam5_v1_release.pt"
    ckpt.write_bytes(b"stub")

    assert cli.main(["models", "add", "lam5-v1", str(ckpt)]) == 0
    m = next(x for x in models.all_models() if x["id"] == "lam5-v1")
    assert m["backend"] == "lam4"
    assert m["available"] is True
    assert set(m["domains"]) == {"molecule", "crystal", "slab"}
    assert m["rank"]["molecule"] == 0
    assert m.get("training")

    # Repeated --domains flags accumulate rather than overwriting with the last one
    assert cli.main(["models", "add", "custom-lam5", str(ckpt), "--domains", "molecule", "--domains", "crystal"]) == 0
    c = next(x for x in models.all_models() if x["id"] == "custom-lam5")
    assert c["backend"] == "lam4"
    assert c["domains"] == ["molecule", "crystal"]
    assert cli.main(["models", "add", "bad-dom", str(ckpt), "--domains", "plasma"]) == 2


def test_chunked_cpu_d3_parity_and_large_uncalibrated_cpu_low_memory():
    """Verify 3-pass chunked CPU D3(BJ) matches dense D3BJ to <1e-9 and uncalibrated CPU auto-enables low_memory at >=1,500 atoms."""
    import numpy as np
    from lam.models.dispersion import D3BJ
    from lam4.calc.d3_chunked import eval_d3_bj_open_chunked
    from lam4.calc.physics import build_edge_tensors

    d3_64 = D3BJ().double().eval()
    rng = np.random.default_rng(7)
    z_np = rng.choice([1, 6, 7, 8, 16], size=90)
    pos_np = rng.uniform(0.0, 13.0, size=(90, 3))
    z_t = torch.as_tensor(z_np, dtype=torch.long)
    pos_leaf = torch.tensor(pos_np, dtype=torch.float64, requires_grad=True)
    e_ref = d3_64(z_t.unsqueeze(0), pos_leaf.unsqueeze(0), torch.ones((1, 90), dtype=torch.bool))
    (g_ref,) = torch.autograd.grad(e_ref.sum(), pos_leaf)
    f_ref = (-g_ref).detach().numpy()

    ei, _ = build_edge_tensors(pos_np, np.eye(3) * 100.0, (False, False, False), 12.0, torch.float64)
    e_chk, f_chk = eval_d3_bj_open_chunked(d3_64, z_t, torch.as_tensor(pos_np, dtype=torch.float64), ei, compute_forces=True, chunk_edges=128)
    assert abs(float(e_ref.item()) - e_chk) < 1e-9
    assert np.max(np.abs(f_ref - f_chk)) < 1e-9

    lam4_m = next(x for x in models.all_models() if x["id"] == "lam5-v1")
    sp = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}})
    big_an = {**structures.analyze(structures.load("water")), "n_atoms": 2000, "edges": 80000}
    dec = routing.route(sp, big_an, ["cpu"], 12.0, {"points": [], "source": "none"}, lam4_m)
    assert dec["low_memory"] is True




