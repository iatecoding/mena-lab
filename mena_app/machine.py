"""This computer: memory, cores and compute devices, without importing torch in the calling process (a short probe
subprocess answers once and is cached)."""

from __future__ import annotations

import functools
import json
import subprocess
import sys

from mena_app import system

PROBE = """
import json, torch
devices = ["cpu"]
cuda_devices = []
if torch.cuda.is_available():
    cnt = torch.cuda.device_count()
    for i in range(cnt):
        dev_id = "cuda" if cnt == 1 else f"cuda:{i}"
        devices.append(dev_id)
        free_b = total_b = None
        try:
            with torch.cuda.device(i):
                free_b, total_b = torch.cuda.mem_get_info(i)
        except Exception:
            pass
        cuda_devices.append({
            "index": i,
            "id": f"cuda:{i}",
            "device": dev_id,
            "name": torch.cuda.get_device_name(i),
            "vram_total_gb": round(total_b / 1e9, 2) if total_b is not None else None,
            "vram_free_gb": round(free_b / 1e9, 2) if free_b is not None else None,
        })
if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    devices.append("mps")
print(json.dumps({
    "devices": devices,
    "torch": torch.__version__,
    "cuda_name": cuda_devices[0]["name"] if cuda_devices else None,
    "cuda_devices": cuda_devices,
    "vram_total_gb": cuda_devices[0]["vram_total_gb"] if cuda_devices else None,
    "vram_free_gb": cuda_devices[0]["vram_free_gb"] if cuda_devices else None,
}))
""".strip()


@functools.lru_cache(maxsize=1)
def probe() -> dict:
    try:
        r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {"devices": ["cpu"], "torch": None, "error": f"PyTorch could not start: {exc}"}


def _live_cuda_devices(probe_res: dict) -> list[dict]:
    cached = [dict(c) for c in (probe_res.get("cuda_devices") or [])]
    live = system.cuda_vram_info() if hasattr(system, "cuda_vram_info") else None
    if not live:
        return cached
    if not cached:
        vis = system._visible_cuda_tokens() if hasattr(system, "_visible_cuda_tokens") else None
        if vis is None:
            return live
        out: list[dict] = []
        for i in range(len(vis)):
            row = system.cuda_row_for_logical_index(live, i)
            if row is not None:
                out.append({**row, "index": i, "id": f"cuda:{i}"})
        return out
    by_idx = {c["index"]: c for c in live if "index" in c}
    for c in cached:
        idx = c.get("index")
        row = (
            system.cuda_row_for_logical_index(live, int(idx))
            if (idx is not None and hasattr(system, "cuda_row_for_logical_index"))
            else by_idx.get(idx)
        )
        if row:
            if row.get("vram_total_gb") is not None:
                c["vram_total_gb"] = row["vram_total_gb"]
            if row.get("vram_free_gb") is not None:
                c["vram_free_gb"] = row["vram_free_gb"]
    return cached


def snapshot() -> dict:
    s = system.snapshot()
    p = probe()
    cuda_devs = _live_cuda_devices(p)
    vram_total = cuda_devs[0].get("vram_total_gb") if cuda_devs else p.get("vram_total_gb")
    vram_free = cuda_devs[0].get("vram_free_gb") if cuda_devs else p.get("vram_free_gb")
    dev_free: dict[str, float | None] = {"cpu": s.get("available_gb")}
    if "mps" in p.get("devices", []):
        dev_free["mps"] = s.get("available_gb")
    for cd in cuda_devs:
        if cd.get("id"):
            dev_free[cd["id"]] = cd.get("vram_free_gb")
        if cd.get("device"):
            dev_free[cd["device"]] = cd.get("vram_free_gb")
    if any(str(d).startswith("cuda") for d in p.get("devices", [])) and "cuda" not in dev_free:
        dev_free["cuda"] = vram_free
    return {
        **s,
        "devices": p["devices"],
        "torch": p.get("torch"),
        "cuda_name": p.get("cuda_name"),
        "cuda_devices": cuda_devs,
        "vram_total_gb": vram_total,
        "vram_free_gb": vram_free,
        "device_free_gb": dev_free,
        "probe_error": p.get("error"),
    }
