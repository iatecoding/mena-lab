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


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _sync_for(device: str):
    import torch

    dev = str(device)
    if dev.startswith("mps") and hasattr(torch, "mps"):
        return torch.mps.synchronize
    if _is_cuda(dev) and torch.cuda.is_available():
        def _sync():
            with torch.cuda.device(dev):
                torch.cuda.synchronize(dev)
        return _sync
    return lambda: None


def _reset_peak_cuda(device: str) -> None:
    import torch

    dev = str(device)
    if _is_cuda(dev) and torch.cuda.is_available():
        with torch.cuda.device(dev):
            torch.cuda.reset_peak_memory_stats(dev)


def _mem_stats(device: str) -> dict:
    import torch

    dev = str(device)
    rss = system.process_rss_bytes() / 1e9
    if _is_cuda(dev) and torch.cuda.is_available():
        with torch.cuda.device(dev):
            gpu_alloc = torch.cuda.max_memory_allocated(dev) / 1e9
            gpu_res = torch.cuda.max_memory_reserved(dev) / 1e9
        return {"peak_gb": gpu_res, "rss_gb": rss, "gpu_allocated_gb": gpu_alloc, "gpu_reserved_gb": gpu_res}
    if dev.startswith("mps") and hasattr(torch, "mps"):
        gpu = torch.mps.driver_allocated_memory() / 1e9
        return {"peak_gb": rss + gpu, "rss_gb": rss, "gpu_allocated_gb": gpu, "gpu_reserved_gb": gpu}
    return {"peak_gb": rss, "rss_gb": rss, "gpu_allocated_gb": 0.0, "gpu_reserved_gb": 0.0}


def point(req: dict) -> dict:
    """One measurement (runs in a child process)."""
    import numpy as np

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
    sync = _sync_for(req["device"])
    _reset_peak_cuda(req["device"])
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
    mem = _mem_stats(req["device"])
    cal_dev = "cuda" if _is_cuda(req["device"]) else req["device"]
    return {"backend": req["model"].get("backend", "lam"), "domain": req["domain"], "device": cal_dev, "n": len(a),
            "x": edges, "cold_ms": cold * 1e3, "warm_ms": float(np.median(ts)) * 1e3, **mem}


def _select_calibration_targets(model_id: str | None = None) -> list[tuple[dict, list[str]]]:
    """Select (model_entry, domains) targets so shared backends calibrate the best model per domain."""
    avail = [m for m in models.all_models() if m.get("available")]
    if model_id is not None:
        m = next((x for x in avail if x["id"] == model_id), None)
        if m is None:
            return []
        doms = [d for d in m.get("domains", []) if d != "slab"]
        return [(m, doms)] if doms else []

    by_model: dict[str, tuple[dict, list[str]]] = {}
    for dom in ("crystal", "molecule"):
        by_backend: dict[str, list[dict]] = {}
        for m in avail:
            if dom in m.get("domains", []):
                by_backend.setdefault(m.get("backend", "lam"), []).append(m)
        for cands in by_backend.values():
            best = sorted(cands, key=lambda x: x.get("rank", {}).get(dom, 99))[0]
            by_model.setdefault(best["id"], (best, []))[1].append(dom)
    return list(by_model.values())


def main(model_id: str | None = None) -> int:
    raw_devs = machine.probe()["devices"]
    devs = []
    seen_cuda = False
    for dev in raw_devs:
        if _is_cuda(dev):
            if not seen_cuda:
                devs.append(dev)
                seen_cuda = True
        else:
            devs.append(dev)
    targets = _select_calibration_targets(model_id)
    if not targets:
        print("no installed model to calibrate (see `mena models`)")
        return 1
    pts, t0 = [], time.time()
    for m, doms in targets:
        backend = m.get("backend", "lam")
        for dev in devs:
            for domain in doms:
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

    from mena_app import backends, structures

    a = structures.tile_system(structures.load(req["structure"]), tuple(req["repeat"]))
    d = {"device": req["device"], "dtype": "float32", "lot": "auto", "skin": 0.6, "low_memory": req["low_memory"],
         "compile": bool(req.get("compile", False)) and not bool(req["low_memory"])}
    a.calc = backends.build(req["model"], d)
    sync = _sync_for(req["device"])
    base_mem = _mem_stats(req["device"])
    base = base_mem["peak_gb"]  # PyTorch, the device runtime and the model, before any structure is evaluated
    _reset_peak_cuda(req["device"])
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
    mem = _mem_stats(req["device"])
    peak = mem["peak_gb"]
    ms = float(np.median(ts)) * 1e3
    return {"n": len(a), "edges": structures.count_edges(a), "cold_ms": cold * 1e3, "ms": ms, "atoms_per_ms": len(a) / ms,
            "peak_gb": peak, "base_gb": base, "added_gb": max(peak - base, 0.0),
            "rss_gb": mem["rss_gb"], "gpu_allocated_gb": mem["gpu_allocated_gb"], "gpu_reserved_gb": mem["gpu_reserved_gb"]}


if __name__ == "__main__":
    _req = json.loads(sys.stdin.read())
    print(json.dumps(bench_point(_req) if _req.get("kind") == "bench" else point(_req)))
