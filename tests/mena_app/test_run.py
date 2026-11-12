"""End-to-end jobs through the worker process, for each installed backend. Skipped when no model file is installed."""

from __future__ import annotations

import threading

import pytest

from mena_app import models, planner, runner, validation


def _job(task, s, s2=None, **params):
    raw = {"task": task, "structure": {"format": "builtin", "text": s}, "params": params, "runtime": {"device": "cpu"}}
    if s2:
        raw["structure2"] = {"format": "builtin", "text": s2}
    return validation.validate(raw)


def _run(spec, tmp_path):
    pl = planner.plan(spec)
    events = []
    res = runner.run(pl, spec, str(tmp_path), events.append, threading.Event())
    assert res["status"] == "done", res.get("error")
    return res["result"], events


def _have(domain):
    try:
        models.choose("auto", domain)
        return True
    except models.ModelError:
        return False


crystal = pytest.mark.skipif(not _have("crystal"), reason="no crystal model installed")
molecule = pytest.mark.skipif(not _have("molecule"), reason="no molecule model installed")


@crystal
def test_crystal_singlepoint_and_relax(tmp_path):
    r, _ = _run(_job("singlepoint", "si"), tmp_path / "a")
    assert abs(r["raw"]["energy_eV"] / 2 + 5.4) < 0.1  # PBE silicon is about -5.4 eV/atom
    r, ev = _run(_job("relax", "cu", fmax=0.01), tmp_path / "b")
    assert r["tiles"][0]["value"] == "yes" and any(e["ev"] == "series" for e in ev)


@molecule
def test_molecule_vibrations_have_no_imaginary_modes(tmp_path):
    r, _ = _run(_job("vibrations", "water"), tmp_path)
    f = r["raw"]["frequencies_cm"]
    assert len(f) == 3 and min(f) > 1400 and max(f) < 4200


@molecule
def test_neb_is_symmetric_for_ammonia_inversion(tmp_path):
    r, _ = _run(_job("neb", "nh3_up", "nh3_down", images=5), tmp_path)
    t = {x["key"]: x["value"] for x in r["tiles"]}
    assert 0.05 < t["fwd"] < 0.5 and abs(t["fwd"] - t["rev"]) < 0.01


def test_cancel_returns_partial_result(tmp_path):
    if not _have("crystal"):
        pytest.skip("no crystal model installed")
    spec = _job("md", "cu", steps=100000)
    pl = planner.plan(spec)
    cancel = threading.Event()
    seen = []

    def emit(e):
        seen.append(e)
        if e["ev"] == "progress" and e["done"] > 20:
            cancel.set()

    res = runner.run(pl, spec, str(tmp_path), emit, cancel)
    assert res["status"] == "cancelled"


@crystal
def test_benchmark_visual_md_emits_supercell_frames(tmp_path):
    r, ev = _run(_job("benchmark", "cssni3", mode="visual_md", max_atoms=40, steps=4, temperature_K=500.0), tmp_path)
    frames = [e["frame"] for e in ev if e["ev"] == "frame"]
    assert frames and "symbols" in frames[0] and len(frames[0]["symbols"]) == len(frames[0]["positions"])
    keys = {t["key"]: t["value"] for t in r["tiles"]}
    assert keys["atoms_s"] > 0 and r["raw"]["n"] == 40 and r["raw"]["atoms_per_s"] > 0

