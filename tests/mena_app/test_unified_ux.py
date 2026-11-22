"""Unified plug-and-play UX tests across catalog, CLI, routing, export, web JS labels, tasks (MD/relax), and python_api (CPU/mocked only)."""

from __future__ import annotations

import ast
import io
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms

from mena_app import (
    backends,
    catalog,
    cli,
    export,
    python_api,
    routing,
    runtime,
    structures,
    validation,
)
from mena_app.tasks import md as md_task
from mena_app.tasks import relax as relax_task

REPO_ROOT = Path(__file__).resolve().parents[2]

LAM4_M = {
    "id": "lam5-v1",
    "title": "Mena-LAM-s",
    "backend": "lam4",
    "lots": ["mp_pbe", "mp_pbeU", "oc20_rpbe", "oc22_pbeU", "omol_wb97mv"],
    "dtypes": ["float32", "float64"],
    "low_memory": True,
}
F32_M = {
    "id": "f32",
    "title": "F32",
    "backend": "lam4",
    "lots": ["mp_pbe", "omol_wb97mv"],
    "dtypes": ["float32"],
    "low_memory": False,
}
CAL = {
    "source": "test",
    "points": [
        {"backend": "lam4", "domain": d, "device": dev, "x": x, "cold_ms": c, "warm_ms": w, "peak_gb": 0.5}
        for d in ("crystal", "molecule", "slab")
        for dev, c, k in (("cpu", 30, 1.0), ("mps", 400, 0.15))
        for x, w in ((100, 10 * k + (5 if dev == "mps" else 0)), (100_000, 500 * k))
    ],
}


def _fake_ctx(tmp_path: Path, spec: dict, out_buf: io.StringIO | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return runtime.Context(
        {
            "spec": spec,
            "decision": {"device": "cpu", "dtype": "float32", "precision": "auto", "lot": "auto", "skin": 0.0, "low_memory": False, "compile": False},
            "model": {"id": "lam5-v1", "title": "Mena-LAM-s", "backend": "lam4"},
            "workdir": str(tmp_path),
        },
        out_buf or io.StringIO(),
    )


def test_precision_catalog_cli_roundtrip_and_json_config(tmp_path: Path):
    prec_schema = next(p for p in catalog.RUNTIME if p["name"] == "precision")
    assert prec_schema["type"] == "choice"
    assert prec_schema["default"] == "auto"
    assert prec_schema["options"] == ["auto", "fp32_ieee", "exact"]

    # Default auto is omitted from CLI export
    sp_auto = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}})
    assert sp_auto["runtime"]["precision"] == "auto"
    assert "--precision" not in export.to_cli(sp_auto)

    # Non-default precision modes round-trip through CLI flags and JSON config
    for prec in ("fp32_ieee", "exact"):
        sp = validation.validate(
            {
                "task": "singlepoint",
                "structure": {"format": "builtin", "text": "water"},
                "runtime": {"precision": prec},
            }
        )
        cli_line = export.to_cli(sp)
        assert f"--precision {prec}" in cli_line
        ns = cli.build_parser().parse_args(cli_line.split()[1:])
        got = cli.spec_from_args("singlepoint", ns)
        assert got["runtime"]["precision"] == prec

        job = tmp_path / f"job_{prec}.json"
        job.write_text(export.to_json(sp), encoding="utf-8")
        cfg = cli._flat_config(cli._load_config(str(job)), "singlepoint", str(job))
        ns_run = cli.build_parser().parse_args(["run", str(job)])
        got_cfg = cli.spec_from_args("singlepoint", ns_run, cfg)
        assert got_cfg["runtime"]["precision"] == prec


def test_routing_precision_exact_avoids_mps_conflicts_and_vibrations():
    cu_an = structures.analyze(structures.load("cu111"))
    md_exact = validation.validate(
        {
            "task": "md",
            "structure": {"format": "builtin", "text": "cu111"},
            "params": {"steps": 100_000},
            "runtime": {"precision": "exact"},
        }
    )
    dec = routing.route(md_exact, cu_an, ["cpu", "mps"], 32.0, CAL, LAM4_M)
    assert dec["device"] == "cpu"
    assert dec["dtype"] == "float64"
    assert dec["precision"] == "exact"
    assert "precision" in dec["reasons"] and "exact" in dec["reasons"]["precision"].lower()
    assert "device" in dec["reasons"] and ("apple gpu" in dec["reasons"]["device"].lower() or "double precision" in dec["reasons"]["device"].lower())

    # Even if only MPS is calibrated while CPU is uncalibrated, exact still routes cleanly to CPU
    mps_only_cal = {"source": "partial", "points": [p for p in CAL["points"] if p["device"] == "mps"]}
    dec_partial = routing.route(md_exact, cu_an, ["cpu", "mps"], 32.0, mps_only_cal, LAM4_M)
    assert dec_partial["device"] == "cpu"
    assert dec_partial["dtype"] == "float64"
    assert dec_partial["precision"] == "exact"

    # Explicit conflicts & disabled bf16_nodes raise RoutingError (subclass of PlanError)
    for bad_rt in (
        {"precision": "fp32_ieee", "dtype": "float64"},
        {"precision": "exact", "dtype": "float32"},
    ):
        sp_bad = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}, "runtime": bad_rt})
        with pytest.raises(routing.RoutingError, match="conflicts"):
            routing.route(sp_bad, structures.analyze(structures.load("water")), ["cpu"], 32.0, CAL, LAM4_M)

    with pytest.raises(validation.ValidationError, match="auto, fp32_ieee, exact"):
        validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}, "runtime": {"precision": "bf16_nodes"}})

    sp_bf16 = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}})
    sp_bf16["runtime"]["precision"] = "bf16_nodes"
    with pytest.raises(routing.RoutingError, match="disabled"):
        routing.route(sp_bf16, structures.analyze(structures.load("water")), ["cpu"], 32.0, CAL, LAM4_M)

    # Single-precision-only model rejects explicit precision='exact'
    sp_ex_f32 = validation.validate(
        {"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}, "runtime": {"precision": "exact"}}
    )
    with pytest.raises(routing.RoutingError, match="single precision only"):
        routing.route(sp_ex_f32, structures.analyze(structures.load("water")), ["cpu"], 32.0, CAL, F32_M)

    # Vibrations / phonons preserve float64 off-MPS for precision='auto', while explicit fp32_ieee keeps float32
    w_an = structures.analyze(structures.load("water"))
    vib_auto = validation.validate({"task": "vibrations", "structure": {"format": "builtin", "text": "water"}})
    d_vib = routing.route(vib_auto, w_an, ["cpu", "mps"], 32.0, CAL, LAM4_M)
    assert d_vib["device"] == "cpu" and d_vib["dtype"] == "float64" and d_vib["precision"] == "exact"

    vib_f32 = validation.validate(
        {"task": "vibrations", "structure": {"format": "builtin", "text": "water"}, "runtime": {"precision": "fp32_ieee"}}
    )
    d_vib_f32 = routing.route(vib_f32, w_an, ["cpu", "mps"], 32.0, CAL, LAM4_M)
    assert d_vib_f32["dtype"] == "float32" and d_vib_f32["precision"] == "fp32_ieee"


def test_export_to_python_emits_precision_and_valid_ast():
    sp = validation.validate({"task": "singlepoint", "structure": {"format": "builtin", "text": "water"}})
    an = structures.analyze(structures.load("water"))
    model = {"id": "lam5-v1", "title": "Mena-LAM-s", "backend": "lam4", "path": "/tmp/lam5.pt"}

    for prec, dt in (("fp32_ieee", "float32"), ("exact", "float64")):
        dec = {"device": "cpu", "dtype": dt, "precision": prec, "lot": "omol_wb97mv", "skin": 0.0, "low_memory": False, "compile": False}
        code = export.to_python(sp, {"analysis": an, "model": model, "decision": dec}, "0.1.0")
        ast.parse(code)
        assert f"precision={prec!r}" in code

    dec_auto = {"device": "cpu", "dtype": "float32", "precision": "auto", "lot": "omol_wb97mv", "skin": 0.0, "low_memory": False, "compile": False}
    code_auto = export.to_python(sp, {"analysis": an, "model": model, "decision": dec_auto}, "0.1.0")
    ast.parse(code_auto)
    assert "precision=" not in code_auto


def test_static_js_precision_labels_and_syntax():
    labels_js = (REPO_ROOT / "mena_app" / "static" / "js" / "labels.js").read_text(encoding="utf-8")
    runopts_js = (REPO_ROOT / "mena_app" / "static" / "js" / "runopts.js").read_text(encoding="utf-8")
    plan_js = (REPO_ROOT / "mena_app" / "static" / "js" / "plan.js").read_text(encoding="utf-8")

    for expected in (
        "auto: 'Automatic'",
        "fp32_ieee: 'Strict FP32 (IEEE)'",
        "exact: 'Double (FP64 exact)'",
    ):
        assert expected in labels_js
    assert "bf16_nodes" not in labels_js
    assert "PRECISION" in runopts_js
    assert "PRECISION" in plan_js

    node = shutil.which("node")
    if node:
        for fname in ("labels.js", "runopts.js", "plan.js"):
            res = subprocess.run([node, "--check", str(REPO_ROOT / "mena_app" / "static" / "js" / fname)], capture_output=True, text=True, timeout=10)
            assert res.returncode == 0, res.stderr


def test_md_nve_nvt_npt_constrained_slab_and_user_momenta(tmp_path: Path):
    # 1. Constrained FixAtoms slab in NVT & NVE: fixed atoms stay frozen, ASE fallback logged cleanly
    slab = fcc111("Cu", (2, 2, 3), vacuum=8.0)
    fixed_idx = [0, 1, 2, 3]
    slab.set_constraint(FixAtoms(indices=fixed_idx))
    p0_fixed = slab.positions[fixed_idx].copy()

    for ens in ("nvt", "nve"):
        at = slab.copy()
        at.calc = LennardJones(sigma=2.3, epsilon=0.1, rc=5.5, smooth=True)
        spec = validation.validate(
            {
                "task": "md",
                "structure": {"format": "builtin", "text": "cu111"},
                "params": {"ensemble": ens, "steps": 6, "timestep": 1.0, "save_every": 2, "temperature": 300.0},
            }
        )
        out_buf = io.StringIO()
        ctx = _fake_ctx(tmp_path / f"slab_{ens}", spec, out_buf)
        with mock.patch("mena_app.tasks.md.setup", return_value=at):
            res = md_task.run(ctx)
        events = [json.loads(line) for line in out_buf.getvalue().splitlines() if line.strip()]
        logs = [e["text"] for e in events if e.get("ev") == "log"]
        assert any("constraint" in msg.lower() for msg in logs)
        assert np.allclose(at.positions[fixed_idx], p0_fixed, atol=1e-12)
        assert np.allclose(at.get_momenta()[fixed_idx], 0.0, atol=1e-12)
        assert np.linalg.norm(at.get_momenta()[4:]) > 0.0
        assert len(res["raw"]["time_ps"]) == 4  # step 0, 2, 4, 6

    # Even if calculator supports md_engine, constraints cleanly bypass it and use ASE
    slab_eng_chk = slab.copy()
    lj_chk = LennardJones(sigma=2.3, epsilon=0.1, rc=5.5, smooth=True)
    lj_chk.md_engine = lambda *a, **kw: pytest.fail("must not call md_engine on constrained atoms")
    slab_eng_chk.calc = lj_chk
    assert md_task._can_use_md_engine({"ensemble": "nvt"}, slab_eng_chk, lj_chk) is False

    # 2. NPT on periodic crystal checks negative externalstress sign convention and volume output
    cu_bulk = bulk("Cu", "fcc", a=3.615).repeat((2, 2, 2))
    cu_bulk.calc = LennardJones(sigma=2.3, epsilon=0.1, rc=5.5, smooth=True)
    spec_npt = validation.validate(
        {
            "task": "md",
            "structure": {"format": "builtin", "text": "cu"},
            "params": {"ensemble": "npt", "steps": 4, "timestep": 1.0, "save_every": 2, "temperature": 300.0, "pressure": 5000.0},
        }
    )
    out_npt = io.StringIO()
    ctx_npt = _fake_ctx(tmp_path / "npt", spec_npt, out_npt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with mock.patch("mena_app.tasks.md.setup", return_value=cu_bulk):
            res_npt = md_task.run(ctx_npt)
    assert len(res_npt["raw"]["volume_A3"]) == 3
    assert any("NPT" in json.loads(line).get("text", "") for line in out_npt.getvalue().splitlines() if json.loads(line).get("ev") == "log")

    # Check _integrator passes negative externalstress for LangevinBAOAB
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dyn = md_task._integrator({"ensemble": "npt", "timestep": 1.0, "temperature": 300.0, "pressure": 2500.0, "friction": 0.01, "seed": 0}, cu_bulk)
    if hasattr(dyn, "externalstress"):
        assert float(dyn.externalstress) < 0.0

    # 3. NPT on non-3D-PBC slab cleanly raises UserError
    at_slab_npt = slab.copy()
    at_slab_npt.calc = LennardJones(sigma=2.3, epsilon=0.1, rc=5.5, smooth=True)
    ctx_bad_npt = _fake_ctx(tmp_path / "bad_npt", spec_npt)
    with mock.patch("mena_app.tasks.md.setup", return_value=at_slab_npt):
        with pytest.raises(md_task.UserError, match="periodic cell"):
            md_task.run(ctx_bad_npt)

    # 4. All atoms fixed raises UserError instead of ZeroDivisionError
    at_all_fixed = slab.copy()
    at_all_fixed.set_constraint(FixAtoms(indices=list(range(len(at_all_fixed)))))
    at_all_fixed.calc = LennardJones(sigma=2.3, epsilon=0.1, rc=5.5, smooth=True)
    spec_nve = validation.validate(
        {"task": "md", "structure": {"format": "builtin", "text": "cu111"}, "params": {"ensemble": "nve", "steps": 2}}
    )
    with mock.patch("mena_app.tasks.md.setup", return_value=at_all_fixed):
        with pytest.raises(md_task.UserError, match="All atoms are fixed"):
            md_task.run(_fake_ctx(tmp_path / "all_fixed", spec_nve))

    # 5. User-supplied non-zero initial momenta preserved & linear molecule runs warning-free
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.78]])
    init_p = np.array([[0.0, 0.0, -0.15], [0.0, 0.0, 0.15]])
    h2.set_momenta(init_p)
    h2.calc = LennardJones(sigma=0.7, epsilon=0.05, rc=4.0, smooth=True)
    out_mom = io.StringIO()
    ctx_mom = _fake_ctx(tmp_path / "user_mom", spec_nve, out_mom)
    with mock.patch("mena_app.tasks.md.setup", return_value=h2), \
         mock.patch("mena_app.tasks.md.thermalize", side_effect=AssertionError("must not overwrite user momenta")):
        res_mom = md_task.run(ctx_mom)
    assert len(res_mom["raw"]["time_ps"]) == 2
    assert any("preserving existing non-zero initial momenta" in line for line in out_mom.getvalue().splitlines())

    # Linear H2 without initial momenta avoids ZeroRotation RuntimeWarning
    h2_zero = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.78]])
    h2_zero.calc = LennardJones(sigma=0.7, epsilon=0.05, rc=4.0, smooth=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with mock.patch("mena_app.tasks.md.setup", return_value=h2_zero):
            md_task.run(_fake_ctx(tmp_path / "h2_nve", spec_nve))
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []

    # 6. MDEngine NVE path (with ensemble in MDEngine.__init__) passes ensemble="nve", maxwell=False, and syncs state
    seen_eng_kw: list[dict] = []

    class _FakeCalc(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.skin = 0.5

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            pos = atoms.get_positions()
            self.results = {"energy": float(0.5 * np.sum(pos**2)), "forces": -pos.copy()}

        def md_engine(self, atoms, **kwargs):
            seen_eng_kw.append(dict(kwargs))

            class _Eng:
                def __init__(self, at):
                    self._at = at.copy()
                    self.stats = {"steps": 0}
                    self.st = SimpleNamespace(p=self._at.get_momenta().copy())

                def window_step(self):
                    w = int(kwargs.get("window", 1))
                    self.stats["steps"] += w
                    self._at.positions = self._at.positions + 1e-3 * w
                    self.st.p = self.st.p + 1e-3 * w

                def positions(self):
                    return self._at.positions.copy()

                def momenta(self):
                    return self.st.p.copy()

                def energy_ev(self):
                    return float(0.5 * np.sum(self._at.positions**2))

            return _Eng(atoms)

    class _FakeMDEngineCls:
        def __init__(self, calc, atoms, *, temperature_K=300.0, ensemble="nvt", **kw):
            pass

    si_cell = bulk("Si", "diamond", a=5.43, cubic=True)
    si_cell.calc = _FakeCalc()
    spec_nve_eng = validation.validate(
        {"task": "md", "structure": {"format": "builtin", "text": "si"}, "params": {"ensemble": "nve", "steps": 4, "save_every": 2}}
    )
    out_eng = io.StringIO()
    with mock.patch("mena_app.tasks.md.setup", return_value=si_cell), \
         mock.patch("lam4.calc.md_engine.MDEngine", _FakeMDEngineCls):
        res_eng = md_task.run(_fake_ctx(tmp_path / "md_eng_nve", spec_nve_eng, out_eng))
    assert seen_eng_kw and seen_eng_kw[0]["ensemble"] == "nve" and seen_eng_kw[0]["maxwell"] is False
    assert res_eng["raw"]["time_ps"] == [0.0, 0.002, 0.004]
    assert any("MDEngine GPU-resident path (NVE" in line for line in out_eng.getvalue().splitlines())


def test_relax_constrained_slab_crystal_cell_and_charged_molecule(tmp_path: Path):
    # 1. FixAtoms slab: fixed atoms stay frozen even if relax_cell=True in params (slab has pbc=[T,T,F])
    slab = fcc111("Cu", (2, 2, 3), vacuum=8.0)
    slab.positions[8:, 2] += 0.15  # perturb top layer
    fixed_idx = [0, 1, 2, 3]
    slab.set_constraint(FixAtoms(indices=fixed_idx))
    p0_fixed = slab.positions[fixed_idx].copy()
    slab.calc = LennardJones(sigma=2.3, epsilon=0.1, rc=5.5, smooth=True)

    spec_slab = validation.validate(
        {"task": "relax", "structure": {"format": "builtin", "text": "cu111"}, "params": {"max_steps": 5, "fmax": 0.05, "relax_cell": True}}
    )
    with mock.patch("mena_app.tasks.relax.setup", return_value=slab):
        res_slab = relax_task.run(_fake_ctx(tmp_path / "rx_slab", spec_slab))
    assert np.allclose(slab.positions[fixed_idx], p0_fixed, atol=1e-12)
    assert {f["name"] for f in res_slab["files"]} == {"relaxed.cif", "trajectory.extxyz"}

    # 1b. 2D slab (cell[2] == [0, 0, 0], cell.rank == 2) with FixAtoms writes relaxed.extxyz cleanly
    slab2d = Atoms(
        "Cu4",
        positions=[[0.0, 0.0, 0.0], [1.28, 1.28, 0.0], [0.0, 0.0, 2.0], [1.28, 1.28, 2.2]],
        cell=[[2.56, 0.0, 0.0], [0.0, 2.56, 0.0], [0.0, 0.0, 0.0]],
        pbc=[True, True, False],
    )
    slab2d.set_constraint(FixAtoms(indices=[0, 1]))
    slab2d.calc = LennardJones(sigma=2.3, epsilon=0.1, rc=5.5, smooth=True)
    with mock.patch("mena_app.tasks.relax.setup", return_value=slab2d):
        res_slab2d = relax_task.run(_fake_ctx(tmp_path / "rx_slab2d", spec_slab))
    assert {f["name"] for f in res_slab2d["files"]} == {"relaxed.extxyz", "trajectory.extxyz"}

    # 2. Periodic crystal with relax_cell=True relaxes volume
    cu_strained = bulk("Cu", "fcc", a=3.95)
    cu_strained.calc = LennardJones(sigma=2.3, epsilon=0.2, rc=5.5, smooth=True)
    spec_cell = validation.validate(
        {"task": "relax", "structure": {"format": "builtin", "text": "cu"}, "params": {"max_steps": 4, "fmax": 0.01, "relax_cell": True}}
    )
    with mock.patch("mena_app.tasks.relax.setup", return_value=cu_strained):
        res_cell = relax_task.run(_fake_ctx(tmp_path / "rx_cell", spec_cell))
    assert any(t["key"] == "dv" for t in res_cell["tiles"])
    assert {f["name"] for f in res_cell["files"]} == {"relaxed.cif", "trajectory.extxyz"}

    # 3. Charged molecule preserves charge/spin through setup & relax
    nh4 = structures.load("smiles:[NH4+]")
    nh4.positions[0, 0] += 0.08
    spec_mol = validation.validate(
        {"task": "relax", "structure": {"format": "smiles", "text": "[NH4+]"}, "params": {"max_steps": 3}, "system": {"charge": 1, "spin": 1}}
    )

    class _HarmonicMol(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            disp = atoms.positions - atoms.positions.mean(axis=0, keepdims=True)
            self.results = {
                "energy": float(0.5 * np.sum(disp**2)),
                "forces": -disp.copy(),
                "charges": np.full(len(atoms), 0.25),
            }

    with mock.patch("mena_app.tasks.common.backends.build", return_value=_HarmonicMol()):
        res_mol = relax_task.run(_fake_ctx(tmp_path / "rx_mol", spec_mol))
    assert res_mol["raw"]["energies_eV"][-1] <= res_mol["raw"]["energies_eV"][0]
    assert {f["name"] for f in res_mol["files"]} == {"relaxed.xyz", "trajectory.extxyz"}


def test_python_api_lazy_import_and_unified_entrypoints(monkeypatch):
    # 1. Pure-stdlib lazy import check in fresh Python subprocess
    probe = (
        "import sys, mena_app; "
        "from mena_app import python_api; "
        "models = mena_app.available_models(); "
        "assert isinstance(models, list) and len(models) >= 2; "
        "assert 'torch' not in sys.modules and 'lam4' not in sys.modules"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr

    # 2. calculator() precision="exact" resolves device off MPS and rejects conflicts
    built_calls: list[tuple[dict, dict]] = []

    class _MockCalc(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self):
            super().__init__()
            self.results = {}

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            pos = atoms.get_positions()
            disp = pos - pos.mean(axis=0, keepdims=True)
            self.results = {
                "energy": float(0.5 * np.sum(disp**2)),
                "forces": -disp.copy(),
                "charges": np.zeros(len(atoms)),
                "lot_used": "omol_wb97mv",
            }
            if bool(atoms.pbc.all()):
                self.results["stress"] = np.zeros(6, dtype=np.float64)
            else:
                self.results["dipole"] = np.array([0.0, 0.0, 0.38])
                self.results["dipole_debye"] = np.array([0.0, 0.0, 1.82])
                self.results["polarizability_iso"] = 1.45

    def fake_build(entry, dec):
        built_calls.append((entry, dec))
        return _MockCalc()

    fake_lam6 = {
        "id": "lam6-v1",
        "title": "Mena-LAM-6",
        "backend": "lam4",
        "domains": ["crystal", "slab", "molecule"],
        "rank": {"crystal": 0, "slab": 0, "molecule": 1},
        "path": __file__,
        "available": True,
        "downloadable": False,
        "lots": ["mp_pbe", "omat_pbe", "oc25_rpbe", "omol_wb97mv"],
    }
    fake_model = {
        "id": "lam5-v1",
        "title": "Mena-LAM-s",
        "backend": "lam4",
        "domains": ["molecule", "crystal", "slab"],
        "rank": {"molecule": 0, "crystal": 2, "slab": 2},
        "path": __file__,
        "available": True,
        "downloadable": False,
        "lots": ["mp_pbe", "oc22_pbeU", "omol_wb97mv"],
    }
    monkeypatch.setattr("mena_app.models.all_models", lambda: [fake_lam6, fake_model])
    monkeypatch.setattr("mena_app.backends.build", fake_build)

    import torch
    with mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch.object(torch.backends.mps, "is_available", return_value=True):
        python_api.calculator("lam5-v1", precision="exact")
        assert built_calls[-1][1]["device"] == "cpu"
        assert built_calls[-1][1]["dtype"] == "float64"
        assert built_calls[-1][1]["precision"] == "exact"

    # calculator() with no args works cleanly and respects domain / lot hints
    python_api.calculator(device="cpu")
    assert built_calls[-1][0]["id"] == "lam5-v1"
    python_api.calculator(device="cpu", domain="crystal")
    assert built_calls[-1][0]["id"] == "lam6-v1"
    python_api.calculator(device="cpu", lot="omat_pbe")
    assert built_calls[-1][0]["id"] == "lam6-v1"
    python_api.calculator(device="cpu", lot="oc22_pbeU")
    assert built_calls[-1][0]["id"] == "lam5-v1"

    with pytest.raises(ValueError, match="requires single precision"):
        python_api.calculator("lam5-v1", dtype="float64", precision="fp32_ieee")

    with pytest.raises(ValueError, match="MPS does not support float64"):
        python_api.calculator("lam5-v1", device="mps", precision="exact")

    with pytest.raises(ValueError, match="disabled"):
        python_api.calculator("lam5-v1", device="cpu", precision="bf16_nodes")

    # 3. singlepoint, relax (fallback + calc.relax delegation), and run_md (fallback + calc.run_md delegation)
    import mena_app

    h2o = molecule("H2O")
    sp_res = mena_app.singlepoint(h2o, device="cpu", charge=0, spin=1)
    assert isinstance(sp_res["energy"], float)
    assert sp_res["model"] == "lam5-v1"
    assert sp_res["forces"].shape == (3, 3)
    assert "dipole_debye" in sp_res and sp_res["polarizability_iso"] == pytest.approx(1.45)

    si = bulk("Si", "diamond", a=5.43)
    sp_si = mena_app.singlepoint(si, device="cpu")
    assert sp_si["model"] == "lam6-v1"
    assert sp_si["stress"].shape == (6,)

    # Relax fallback on FixAtoms slab & crystal relax_cell=True
    slab = fcc111("Cu", (2, 2, 3), vacuum=8.0)
    slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    rx_fallback = mena_app.relax(slab, device="cpu", fmax=0.1, steps=3)
    assert rx_fallback["model"] == "lam6-v1"
    assert rx_fallback["steps"] >= 0 and np.allclose(rx_fallback["positions"][:4], slab.positions[:4])

    rx_cell = mena_app.relax(si, device="cpu", relax_cell=True, steps=2)
    assert rx_cell["model"] == "lam6-v1"
    assert "final_energy" in rx_cell and rx_cell["atoms"].pbc.all()

    # Run MD fallback (NVT, NVE, NPT, constrained slab)
    md_nve = mena_app.run_md(h2o, device="cpu", ensemble="nve", steps=3, timestep_fs=0.5, temperature_K=250.0)
    assert md_nve["model"] == "lam5-v1"
    assert md_nve["steps"] == 3 and md_nve["ensemble"] == "nve"
    md_npt = mena_app.run_md(si, device="cpu", ensemble="npt", steps=2, timestep_fs=0.5)
    assert md_npt["model"] == "lam6-v1"
    assert md_npt["steps"] == 2 and md_npt["ensemble"] == "npt"

    # Delegation when calc implements .relax and .run_md
    class _DelegatingCalc(_MockCalc):
        def __init__(self):
            super().__init__()
            self.relax_calls = []
            self.md_calls = []

        def relax(self, atoms, *, fmax=0.01, steps=200):
            self.relax_calls.append((len(atoms), fmax, steps))
            return {"atoms": atoms, "converged": True, "steps": 1, "final_energy": 0.0, "final_fmax": 0.0, "initial_energy": 0.0, "positions": atoms.positions}

        def run_md(self, atoms, *, ensemble="nvt", temperature_K=300.0, timestep_fs=0.5, steps=10, maxwell=True):
            self.md_calls.append((ensemble, temperature_K, timestep_fs, steps, maxwell))
            return {"atoms": atoms, "steps": steps, "ensemble": ensemble}

    del_calc = _DelegatingCalc()
    monkeypatch.setattr("mena_app.backends.build", lambda e, d: del_calc)

    out_rx = mena_app.relax(h2o, device="cpu", fmax=0.02, max_steps=15)
    assert out_rx["converged"] is True and del_calc.relax_calls == [(3, 0.02, 15)]

    # Preserves non-zero user momenta by setting maxwell=False automatically on calc.run_md
    h2o_p = h2o.copy()
    h2o_p.set_momenta([[0.1, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
    out_md = mena_app.run_md(h2o_p, device="cpu", ensemble="nve", temperature=120.0, timestep=0.5, steps=8)
    assert out_md["steps"] == 8
    assert del_calc.md_calls == [("nve", 120.0, 0.5, 8, False)]

    # Constraints bypass calc.run_md and route cleanly to ASE fallback without crashing
    out_slab_md = mena_app.run_md(slab, device="cpu", ensemble="nvt", steps=2, temperature_K=300.0)
    assert out_slab_md["steps"] == 2 and np.allclose(out_slab_md["momenta"][:4], 0.0)
