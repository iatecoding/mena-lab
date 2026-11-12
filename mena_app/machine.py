"""This computer: memory, cores and compute devices, without importing torch in the calling process (a short probe
subprocess answers once and is cached)."""

from __future__ import annotations

import functools
import json
import subprocess
import sys

from mena_app import system

PROBE = ("import json,torch;d=['cpu'];"
         "d+=['cuda'] if torch.cuda.is_available() else [];"
         "d+=['mps'] if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else [];"
         "print(json.dumps({'devices':d,'torch':torch.__version__,'cuda_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))")


@functools.lru_cache(maxsize=1)
def probe() -> dict:
    try:
        r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {"devices": ["cpu"], "torch": None, "error": f"PyTorch could not start: {exc}"}


def snapshot() -> dict:
    s = system.snapshot()
    p = probe()
    return {**s, "devices": p["devices"], "torch": p.get("torch"), "cuda_name": p.get("cuda_name"), "probe_error": p.get("error")}
