"""mena_app parts that need no torch and no weights: validation, routing, cost model, export, structures, CLI parsing."""

from __future__ import annotations

import json

import pytest

from mena_app import catalog, cli, costmodel, export, routing, structures, validation

CAL = {"source": "test", "points": [
    {"backend": "lam", "domain": d, "device": dev, "x": x, "cold_ms": c, "warm_ms": w, "peak_gb": g}
    for d in ("crystal", "molecule") for dev, c, k in (("cpu", 30, 1.0), ("mps", 600, 0.2))
    for x, w, g in ((100, 15 * k + (5 if dev == "mps" else 0), 0.4), (100_000, 500 * k, 4.0))]}
LAM = {"id": "lam", "title": "LAM", "backend": "lam", "dtypes": ["float32", "float64"], "low_memory": True}
F32 = {"id": "f32", "title": "F32", "backend": "lam", "dtypes": ["float32"], "low_memory": False}


def spec(task="singlepoint", s="si", **params):
    return validation.validate({"task": task, "structure": {"format": "builtin", "text": s}, "params": params})


def plan(sp, model=LAM, free=30.0, devices=("cpu", "mps")):
    a = structures.analyze(structures.load(sp["structure"]))
    return routing.route(sp, a, list(devices), free, CAL, model)


def test_catalog_params_valid():
    for t in catalog.TASKS:
        sp = validation.validate({"task": t["id"], "structure": {"format": "builtin", "text": "si"},
                                  **({"structure2": {"format": "builtin", "text": "si"}} if t.get("needs_second") else {})})
        assert set(sp["params"]) == {p["name"] for p in t["params"]}


def test_validation_rejects():
    with pytest.raises(validation.ValidationError):
        spec(fmax=-1, task="relax")
    with pytest.raises(validation.ValidationError):
        validation.validate({"task": "relax", "structure": {"format": "pickle", "text": "x"}})
    with pytest.raises(validation.ValidationError):
        validation.validate({"task": "relax", "structure": {"format": "builtin", "text": "si"}, "params": {"bogus": 1}})


def test_short_job_goes_to_cpu_long_job_to_gpu():
    assert plan(spec())["device"] == "cpu"  # one call: the GPU's first-call cost never pays back
    assert plan(spec("md", "cu111", steps=100_000))["device"] == "mps"


def test_vibrations_double_precision_off_the_mac_gpu():
    d = plan(spec("vibrations", "water"))
    assert d["dtype"] == "float64" and d["device"] == "cpu"
    assert plan(spec("vibrations", "water"), model=F32)["dtype"] == "float32"


def test_explicit_float64_never_picks_mps():
    sp = spec("md", "cu111", steps=100_000)
    sp["runtime"]["dtype"] = "float64"
    assert plan(sp)["device"] == "cpu"


def test_low_memory_turns_on_when_tight():
    sp = spec("md", "cu111", steps=100)
    assert plan(sp, free=30.0)["low_memory"] is False
    tight = plan(sp, free=3.6)
    assert tight["low_memory"] is True and tight["reasons"]["low_memory"].startswith("Turned on")


def test_uncalibrated_device_is_said_not_guessed():
    d = routing.route(spec(), structures.analyze(structures.load({"format": "builtin", "text": "si"})), ["cpu"], 30.0, {"points": [], "source": "none"}, LAM)
    assert d["seconds"] is None and "mena calibrate" in d["reasons"]["device"]


def test_costmodel_interpolates_monotonic():
    a = costmodel.curve(CAL, "crystal", "cpu", 1000)["warm_ms"]
    b = costmodel.curve(CAL, "crystal", "cpu", 10000)["warm_ms"]
    assert 15 < a < b < 500


def test_structures_and_edges():
    a = structures.load({"format": "builtin", "text": "tio2"})
    an = structures.analyze(a)
    assert an["domain"] == "crystal" and an["n_atoms"] == 6 and an["edges"] > 100
    assert structures.analyze(structures.load({"format": "builtin", "text": "cu111"}))["domain"] == "slab"
    xyz = "3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n"
    assert structures.analyze(structures.load({"format": "xyz", "text": xyz}))["domain"] == "molecule"
    with pytest.raises(structures.StructureError):
        structures.load({"format": "cif", "text": "not a cif"})


def test_too_close_atoms_warned():
    xyz = "2\nbad\nH 0 0 0\nH 0.1 0 0\n"
    assert structures.analyze(structures.load({"format": "xyz", "text": xyz}))["warnings"]


def test_export_roundtrip_through_cli(tmp_path):
    sp = spec("relax", "cu", fmax=0.01, relax_cell=False)
    line = export.to_cli(sp)
    assert line == "mena relax cu --fmax 0.01 --no-relax-cell"
    ns = cli.build_parser().parse_args(line.split()[1:])
    got = cli.spec_from_args("relax", ns)
    assert {k: got[k] for k in ("task", "params", "runtime", "system")} == {k: sp[k] for k in ("task", "params", "runtime", "system")}
    job = tmp_path / "job.json"
    job.write_text(export.to_json(sp))
    ns = cli.build_parser().parse_args(["run", str(job)])
    cfg = json.loads(job.read_text())
    assert cli.spec_from_args("relax", ns, cfg)["params"] == sp["params"]


def test_benchmark_visual_md_and_data():
    from mena_app import paths
    sp = spec("benchmark", "cssni3", mode="visual_md", max_atoms=500, steps=40, temperature_K=600.0)
    assert sp["params"]["mode"] == "visual_md" and sp["params"]["steps"] == 40
    b = json.loads((paths.DATA_DIR / "benchmarks.json").read_text())
    assert "hardware_banner" in b and "dynamat" in b["hardware_banner"] and "diamond" in b["hardware_banner"]
    assert all("atoms_per_s" in r and "cps" in r for r in b["diamond"]["rows"])
    icon = (paths.STATIC_DIR / "icon.svg").read_text()
    html = (paths.STATIC_DIR / "index.html").read_text()
    assert "<svg" in icon and 'href="/static/icon.svg"' in html



def test_compile_only_when_it_pays_back():
    assert plan(spec())["compile"] is False  # one call: the warm-up never pays back
    long_md = plan(spec("md", "cu111", steps=100_000))
    assert long_md["compile"] is True and long_md["reasons"]["compile"].startswith("Saves about")
    off = spec("md", "cu111", steps=100_000)
    off["runtime"]["compile"] = "off"
    assert plan(off)["compile"] is False and plan(off)["reasons"]["compile"] == "You turned this off."
    assert plan(spec("md", "water", steps=100_000))["compile"] is False  # small molecule: dense path, nothing to compile


def test_relax_uses_a_skin():
    assert plan(spec("relax"))["skin"] == 0.5


def test_first_compiled_call_is_announced():
    from mena_app.tasks.common import announce_compile

    class Ctx:
        def __init__(self):
            self.lines = []

        def status(self, t):
            self.lines.append(t)

        def log(self, t):
            self.lines.append("log: " + t)

    class Calc:
        compile_status = "unchecked"
        calls = 0

        def calculate(self, *a, **k):
            self.calls += 1
            self.compile_status = "on"
            return "r"

    ctx, calc = Ctx(), Calc()
    announce_compile(ctx, calc)
    assert calc.calculate() == "r" and calc.calculate() == "r" and calc.calls == 2
    assert len(ctx.lines) == 2 and ctx.lines[0].startswith("compiling") and "checked" in ctx.lines[1]


def test_phonon_estimate_counts_symmetry():
    a = structures.analyze(structures.load({"format": "builtin", "text": "srtio3"}))
    assert a["phonon_evaluations"] == 9  # 31 without symmetry
    assert plan(spec("phonons", "srtio3", prerelax=False))["calls"] == 9
