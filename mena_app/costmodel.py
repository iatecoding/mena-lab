"""Predicts time and memory for a job from measured curves. No torch.

Curves are points {domain, device, x, cold_ms, warm_ms, peak_gb} where x is the neighbour-pair count. They come from
`mena calibrate` (~/.mena/calibration.json, this computer) or, failing that, data/calibration_m3max.json. With no curve for a
device (for example CUDA before calibrating) `curve` returns None and routing says so rather than guessing a number.
"""

from __future__ import annotations

import json
import math
import platform
import sys

from mena_app import paths

LOW_MEMORY_FACTOR = 0.55     # measured: 3.23 / 6.48 GB at 10k atoms
FLOAT64_MEMORY_FACTOR = 2.0
# torch.compile of the model: one-time cost per process and warm-call speed-up over eager (both with a skin).
# Measured on an M3 Max. Compiled kernels persist in ~/.mena/compile_cache, so the
# warm-up is ~5-6 s on MPS and ~15-50 s on CPU once a model has compiled on this computer (the very first compile:
# 25-40 s MPS, 40-120 s CPU); we plan for a middle value. lam4 backend (lam5, lam6), warm-call speed-ups:
# MPS 1.9-2.6x (64-atom molecules 2.6x, 1k-atom crystals 2.0x); CPU 1.5-2.8x since the whole-model CPU compile
# (2026-11-21; FP64 too). lam (lam-400k): MPS 2.5x, CPU 1.4x (2026-11-12). CUDA is NOT measured here: a conservative guess.
COMPILE = {"mps": {"warmup_s": 10.0, "speedup": 2.5, "by_backend": {"lam4": 2.0}},
           "cpu": {"warmup_s": 45.0, "speedup": 1.4, "by_backend": {"lam4": 1.8}},
           "cuda": {"warmup_s": 30.0, "speedup": 1.5}}


def _device_family(device: str) -> str:
    dev = str(device)
    return dev.split(":", 1)[0] if ":" in dev else dev


def compile_speedup(device: str, backend: str | None = None) -> float | None:
    c = COMPILE.get(_device_family(device))
    if c is None:
        return None
    return float(c.get("by_backend", {}).get(backend, c["speedup"]))


def compiled_seconds(eager_seconds: float, device: str, backend: str | None = None) -> float:
    dev = _device_family(device)
    c = COMPILE.get(dev)
    return eager_seconds if c is None else c["warmup_s"] + eager_seconds / compile_speedup(dev, backend)


def _consistent_edges(pts: list[dict]) -> bool:
    groups: dict[tuple, list[dict]] = {}
    for p in pts:
        if "n" in p and "x" in p:
            groups.setdefault((p.get("backend", "lam"), p.get("domain"), p.get("device")), []).append(p)
    for g in groups.values():
        s = sorted(g, key=lambda r: r["n"])
        if any(a["n"] < b["n"] and a["x"] > b["x"] for a, b in zip(s, s[1:])):
            return False
    return True


def load() -> dict:
    candidates = [(paths.data_dir() / "calibration.json", "calibrated on this computer")]
    if sys.platform == "darwin" and platform.machine() == "arm64":
        candidates.append((paths.DATA_DIR / "calibration_m3max.json", "measured on an Apple M3 Max"))
    for p, label in candidates:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if not _consistent_edges(d.get("points", [])):
                continue
            d["source"] = label
            return d
        except (OSError, ValueError):
            continue
    return {"points": [], "source": "no measurements"}


def _isotonic(ys: list[float]) -> list[float]:
    """Pool-Adjacent-Violators isotonic projection so noisy calibration blips never create downward steps."""
    blocks: list[list[float]] = [[y, 1.0] for y in ys]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    out: list[float] = []
    for s, n in blocks:
        out.extend([s / n] * int(n))
    return out


def _interp(pts: list[tuple[float, float]], x: float) -> float:
    pts = sorted(pts)
    if len(pts) == 1:
        return pts[0][1]
    lx = math.log(max(x, 1.0))
    xs = [math.log(max(a, 1.0)) for a, _ in pts]
    ys = _isotonic([math.log(max(b, 1e-9)) for _, b in pts])
    logs = list(zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(logs, logs[1:]):
        if lx <= x1:
            if lx < x0:
                return math.exp(y0)
            slope = max(0.0, (y1 - y0) / max(x1 - x0, 1e-9))
            return math.exp(y0 + slope * (lx - x0))
    (x0, y0), (x1, y1) = logs[-2], logs[-1]
    slope = max(0.0, (y1 - y0) / max(x1 - x0, 1e-9))
    return math.exp(y1 + slope * (lx - x1))


def curve(cal: dict, domain: str, device: str, edges: int, backend: str = "lam") -> dict | None:
    dom = "crystal" if domain == "slab" else domain
    dev = _device_family(device)
    rows = [p for p in cal["points"] if p["domain"] == dom and _device_family(p["device"]) == dev and p.get("backend", "lam") == backend]
    if not rows and backend == "lam4":
        rows = [p for p in cal["points"] if p["domain"] == dom and _device_family(p["device"]) == dev and p.get("backend", "lam") == "lam"]
    if not rows:
        return None
    return {k: _interp([(p["x"], p[k]) for p in rows], edges) for k in ("cold_ms", "warm_ms", "peak_gb")}


def job_seconds(c: dict, n_calls: int) -> float:
    return (c["cold_ms"] + max(n_calls - 1, 0) * c["warm_ms"]) / 1000.0


def peak_gb(c: dict, dtype: str, low_memory: bool) -> float:
    gb = c["peak_gb"]
    return gb * (FLOAT64_MEMORY_FACTOR if dtype == "float64" else 1.0) * (LOW_MEMORY_FACTOR if low_memory else 1.0)


def estimate(cal: dict, domain: str, device: str, edges: int, n_calls: int = 1, dtype: str = "float32",
             low_memory: bool = False, backend: str = "lam") -> dict | None:
    c = curve(cal, domain, device, edges, backend=backend)
    if c is None:
        return None
    return {**c, "seconds": job_seconds(c, n_calls), "peak_gb": peak_gb(c, dtype, low_memory)}
