"""Worker process: runs ONE job and prints JSON-line events (runtime.py). Started by runner.py, never directly by a browser.

    python -m mena_app.worker < request.json        request = {"spec", "decision", "model", "workdir"}

The model's code_root goes first on sys.path before anything imports `lam`, so each model loads with its own code.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback


def main() -> int:
    out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr  # stray prints from model code must not corrupt the event stream
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    req = json.loads(sys.stdin.read())
    root = req["model"].get("code_root")
    if root:
        sys.path.insert(0, root)
    # Compiled kernels persist in the user's Mena folder: the system temp dir is purged, which turns a ~5 s compiled
    # warm-up back into a 25-40 s one. Set before torch is imported (Inductor reads it once); a user value wins.
    try:
        from mena_app.paths import data_dir
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(data_dir() / "compile_cache"))
    except Exception:  # noqa: BLE001
        pass
    threads = req["decision"].get("threads")
    if threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(int(threads))
    from mena_app.runtime import Cancelled, Context, UserError, install_signal
    from mena_app.tasks import load

    ctx = Context(req, out)
    install_signal(ctx)
    try:
        if threads:
            import torch
            torch.set_num_threads(int(threads))
        report = load(req["spec"]["task"]).run(ctx)
        raw = (report or {}).get("raw") or {}
        if any("energ" in k and isinstance(v, float) and not math.isfinite(v) for k, v in raw.items()):
            ctx.error("user", "The model returned a non-finite energy (NaN). This usually means overlapping or extremely "
                              "close atoms: check the structure.")
            return 2
        ctx.result(report)
        return 0
    except Cancelled:
        ctx.error("cancelled", "stopped by the user")
        return 0
    except UserError as exc:
        ctx.error("user", str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001 - reported with a short traceback
        msg = str(exc)
        kind = "oom" if "out of memory" in msg.lower() or "invalid buffer size" in msg.lower() else "internal"
        ctx.error(kind, f"{type(exc).__name__}: {msg[:400]}\n{traceback.format_exc(limit=6)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
