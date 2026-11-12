"""Worker plumbing: JSON-line events on stdout, cooperative cancel on SIGTERM, and the calculator for this job.

Events, one JSON object per line:
  {"ev": "status",   "text": "..."}                          what the job is doing now
  {"ev": "log",      "text": "..."}                          a line for the log
  {"ev": "progress", "done": 3, "total": 10, "label": "...", "stats": {...}}
  {"ev": "series",   "chart": "energy", "name": "...", "x": [...], "y": [...]}   points for a live chart
  {"ev": "frame",    "frame": {...}}                         latest structure, for the live viewer
  {"ev": "result",   "data": {...}}                          the final report (report.py)
  {"ev": "error",    "kind": "user" | "internal" | "cancelled" | "oom", "message": "..."}
"""

from __future__ import annotations

import json
import math
import os
import signal
import time

import numpy as np


CANCEL_FILE = ".mena-cancel"  # written by runner.py to stop a job; works where SIGTERM cannot be caught (Windows)


class Cancelled(Exception):
    pass


class UserError(Exception):
    """The input cannot be used; shown to the user as it is."""


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        return clean(o.item())
    if isinstance(o, float):
        return None if not math.isfinite(o) else float(f"{o:.9g}")
    return o


class Context:
    def __init__(self, req: dict, out) -> None:
        self.spec, self.decision, self.model = req["spec"], req["decision"], req["model"]
        self.params, self.system = self.spec["params"], self.spec["system"]
        self.workdir = req.get("workdir") or "."
        self._out, self._cancel = out, False
        self._cancel_file = os.path.join(self.workdir, CANCEL_FILE)
        self._buf: dict[tuple[str, str], tuple[list, list]] = {}
        self._last_flush = self._last_progress = self._last_frame = 0.0
        self.t0 = time.perf_counter()

    def _send(self, obj: dict) -> None:
        self._out.write(json.dumps(clean(obj), separators=(",", ":")) + "\n")
        self._out.flush()

    def status(self, text: str) -> None:
        self._send({"ev": "status", "text": text})

    def log(self, text: str) -> None:
        self._send({"ev": "log", "text": text})

    def progress(self, done, total, label: str = "", stats: dict | None = None, force: bool = False) -> None:
        now = time.perf_counter()
        if force or now - self._last_progress >= 0.2 or done >= total:
            self._last_progress = now
            self._send({"ev": "progress", "done": done, "total": total, "label": label, "stats": stats or {}})

    def series(self, chart: str, name: str, x, y) -> None:
        bx, by = self._buf.setdefault((chart, name), ([], []))
        bx.append(x)
        by.append(y)
        if time.perf_counter() - self._last_flush > 0.3:
            self.flush()

    def frame(self, atoms, label: str = "", every_s: float = 0.25) -> None:
        now = time.perf_counter()
        if now - self._last_frame >= every_s and len(atoms) <= 5000:
            self._last_frame = now
            fr = {"positions": np.round(atoms.positions, 3).tolist(), "label": label}
            if any(atoms.pbc):
                fr["cell"] = atoms.cell.tolist()
            sig = (len(atoms), tuple(atoms.numbers[:8].tolist()))
            if getattr(self, "_frame_sig", None) != sig:
                self._frame_sig = sig
                fr["symbols"] = list(atoms.symbols)
            self._send({"ev": "frame", "frame": fr})

    def flush(self) -> None:
        for (chart, name), (x, y) in self._buf.items():
            if x:
                self._send({"ev": "series", "chart": chart, "name": name, "x": x, "y": y})
        self._buf, self._last_flush = {}, time.perf_counter()

    def result(self, data: dict) -> None:
        self.flush()
        self._send({"ev": "result", "data": data})

    def error(self, kind: str, message: str) -> None:
        self.flush()
        self._send({"ev": "error", "kind": kind, "message": message})

    def request_cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        if not self._cancel and os.path.exists(self._cancel_file):  # the cross-platform stop request (runner.py)
            self._cancel = True
        return self._cancel

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled()

    def path(self, name: str) -> str:
        os.makedirs(self.workdir, exist_ok=True)
        return os.path.join(self.workdir, name)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


def install_signal(ctx: Context) -> None:
    signal.signal(signal.SIGTERM, lambda *_: ctx.request_cancel())
    signal.signal(signal.SIGINT, lambda *_: ctx.request_cancel())
