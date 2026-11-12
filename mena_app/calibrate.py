"""`mena calibrate`: measure time and peak memory per model backend, device and system size on this computer.

Each point runs in its own process (honest peak memory, an out-of-memory dies alone). Writes ~/.mena/calibration.json, which
routing.py reads; points are {backend, domain, device, n, x (neighbour pairs), cold_ms, warm_ms, peak_gb}.
"""

from __future__ import annotations

import json
import sys
import time

from mena_app import machine, models, paths, runner, system

MOLECULE_WATERS = (1, 4, 11, 32, 100, 340)
CRYSTAL_REPEATS = (1, 2, 3, 4, 6)


def _water_cluster(n_mol: int):
    import numpy as np
    from ase.build import molecule
    w = molecule("H2O")
    side = int(np.ceil(n_mol ** (1 / 3)))
    pos = [w.positions + 3.1 * np.array([i, j, k]) for i in range(side) for j in range(side) for k in range(side)][:n_mol]
    from ase import Atoms
    return Atoms(numbers=list(w.numbers) * n_mol, positions=np.vstack(pos))


_water_box = _water_cluster


def _system(domain: str, size: int):
    from ase.build import bulk
    if domain == "molecule":
        return _water_cluster(size)
    return bulk("Si", cubic=True).repeat((size, size, size))


def point(req: dict) -> dict:
    """One measurement (runs in a child process)."""
    import numpy as np
    import torch

    from mena_app import backends, structures

    a = _system(req["domain"], req["size"])
    edges = structures.count_edges(a)
    d = {"device": req["device"], "dtype": "float32", "lot": "auto", "skin": 0.0, "low_memory": False}
    calc = backends.build(req["model"], d)
    a.calc = calc
    if req["domain"] == "molecule":
        a.info.update(charge=0, spin=1)
    rng = np.random.default_rng(0)
    p0 = a.positions.copy()
    sync = (torch.mps.synchronize if req["device"] == "mps" else (torch.cuda.synchronize if req["device"] == "cuda" else (lambda: None)))
    t = time.perf_counter()
    a.get_forces()
    sync()
    cold = time.perf_counter() - t
    ts = []
    for _ in range(5):
        a.positions = p0 + rng.normal(0, 0.01, p0.shape)
        t = time.perf_counter()
        a.get_forces()
        sync()
        ts.append(time.perf_counter() - t)
    rss = system.process_rss_bytes() / 1e9
    gpu = (
        torch.mps.driver_allocated_memory() / 1e9
        if req["device"] == "mps" and hasattr(torch, "mps")
        else (torch.cuda.max_memory_reserved() / 1e9 if req["device"] == "cuda" and torch.cuda.is_available() else 0.0)
    )
    return {"backend": req["model"].get("backend", "lam"), "domain": req["domain"], "device": req["device"], "n": len(a),
            "x": edges, "cold_ms": cold * 1e3, "warm_ms": float(np.median(ts)) * 1e3, "peak_gb": rss + gpu}


def main(model_id: str | None = None) -> int:
    devs = machine.probe()["devices"]
    chosen = {}
    for m in models.all_models():
        if m["available"] and (model_id is None or m["id"] == model_id):
            chosen.setdefault(m.get("backend", "lam"), m)
    if not chosen:
        print("no installed model to calibrate (see `mena models`)")
        return 1
    pts, t0 = [], time.time()
    for backend, m in chosen.items():
        for dev in devs:
            for domain in m["domains"]:
                if domain == "slab":
                    continue
                sizes = MOLECULE_WATERS if domain == "molecule" else CRYSTAL_REPEATS
                for size in sizes:
                    req = {"model": m, "device": dev, "domain": domain, "size": size}
                    try:
                        r = runner.isolated_python(m, ["-m", "mena_app.calibrate"], json.dumps(req))
                        p = json.loads(r.stdout.strip().splitlines()[-1])
                    except Exception as exc:  # noqa: BLE001 - one bad point must not stop the rest
                        print(f"  {backend:6} {dev:4} {domain:8} size {size}: failed ({str(exc)[:80]})", flush=True)
                        break
                    pts.append(p)
                    print(f"  {backend:6} {dev:4} {domain:8} {p['n']:6d} atoms  first {p['cold_ms']:8.1f} ms  then {p['warm_ms']:8.1f} ms  peak {p['peak_gb']:.2f} GB", flush=True)
    out = {"created": time.strftime("%Y-%m-%d %H:%M"), "machine": machine.snapshot(), "points": pts}
    (paths.data_dir() / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved {len(pts)} measurements to {paths.data_dir() / 'calibration.json'} in {time.time() - t0:.0f} s")
    return 0




def bench_point(req: dict) -> dict:
    """One benchmark size (child process): structure example repeated, MD-like force calls with a neighbour skin."""
    import numpy as np
    import torch

    from mena_app import backends, structures

    a = structures.tile_system(structures.load(req["structure"]), tuple(req["repeat"]))
    d = {"device": req["device"], "dtype": "float32", "lot": "auto", "skin": 0.6, "low_memory": req["low_memory"]}
    a.calc = backends.build(req["model"], d)
    sync = torch.mps.synchronize if req["device"] == "mps" and hasattr(torch, "mps") else (torch.cuda.synchronize if req["device"] == "cuda" and torch.cuda.is_available() else (lambda: None))

    def used_gb() -> float:
        rss = system.process_rss_bytes() / 1e9
        gpu = (
            torch.mps.driver_allocated_memory() / 1e9
            if req["device"] == "mps" and hasattr(torch, "mps")
            else (torch.cuda.max_memory_reserved() / 1e9 if req["device"] == "cuda" and torch.cuda.is_available() else 0.0)
        )
        return rss + gpu

    base = used_gb()  # PyTorch, the device runtime and the model, before any structure is evaluated
    rng = np.random.default_rng(0)
    p0 = a.positions.copy()
    t = time.perf_counter()
    a.get_forces()
    sync()
    cold = time.perf_counter() - t
    ts = []
    for k in range(req["steps"]):
        a.positions = p0 + rng.normal(0, 0.005 * (k + 1) ** 0.5, p0.shape)  # small drift, like MD between neighbour rebuilds
        t = time.perf_counter()
        a.get_forces()
        sync()
        ts.append(time.perf_counter() - t)
    peak = used_gb()
    ms = float(np.median(ts)) * 1e3
    return {"n": len(a), "edges": structures.count_edges(a), "cold_ms": cold * 1e3, "ms": ms, "atoms_per_ms": len(a) / ms,
            "peak_gb": peak, "base_gb": base, "added_gb": max(peak - base, 0.0)}


if __name__ == "__main__":
    _req = json.loads(sys.stdin.read())
    print(json.dumps(bench_point(_req) if _req.get("kind") == "bench" else point(_req)))
