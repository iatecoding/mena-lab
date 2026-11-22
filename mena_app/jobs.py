"""Job manager (UI side): a queue of test runs, live events for the browser, cancel, history. No torch, no model code.

A job owns a folder under MENA_HOME/jobs/<id>/ for its files. The engine does the work; this module records what
happens and lets any number of browser tabs follow it (Server-Sent Events read `events_after`).
"""

from __future__ import annotations

import json
import queue
import secrets
import shutil
import threading
import time
from pathlib import Path

from mena_app import paths

TERMINAL = {"done", "cancelled", "error", "timeout", "stopped"}
MAX_TEXT_IN_LIST = 2000
MAX_HISTORY = 200


def _shorten(o, limit=MAX_TEXT_IN_LIST):
    if isinstance(o, dict):
        return {k: _shorten(v, limit) for k, v in o.items()}
    if isinstance(o, list):
        return [_shorten(v, limit) for v in o]
    if isinstance(o, str) and len(o) > limit:
        return f"<{len(o)} characters of text>"
    return o


class Job:
    def __init__(self, spec: dict, meta: dict, workdir: Path) -> None:
        self.id = secrets.token_hex(6)
        self.spec = spec
        self.test = spec["task"]
        self.meta = meta  # {"test_title", "model": {...}, "preflight": {...}}
        self.workdir = workdir
        self.status = "queued"
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.events: list[dict] = []
        self.seq = 0
        self.cond = threading.Condition()
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self.error_kind: str | None = None
        self.files: list[dict] = []
        self.earlier_session = False

    # ----- events
    def add_event(self, ev: dict) -> None:
        with self.cond:
            self.seq += 1
            e = {"seq": self.seq, "t": round(time.time() - (self.started or self.created), 3), **ev}
            if ev.get("ev") == "progress" and self.events and self.events[-1].get("ev") == "progress":
                self.events.pop()
            self.events.append(e)
            if len(self.events) > 30000:
                del self.events[:5000]
            self.cond.notify_all()

    def events_after(self, seq: int, timeout: float) -> tuple[list[dict], bool]:
        """(new events, finished). Waits up to `timeout` seconds for something new."""
        with self.cond:
            if not (self.events and self.events[-1]["seq"] > seq) and self.status not in TERMINAL:
                self.cond.wait(timeout)
            new = [e for e in self.events if e["seq"] > seq]
            return new, self.status in TERMINAL and not new

    def set_status(self, status: str) -> None:
        with self.cond:
            self.status = status
            self.cond.notify_all()
        self.add_event({"ev": "state", "status": status})

    # ----- views
    def summary(self) -> dict:
        res = self.result or {}
        return {
            "id": self.id, "task": self.test, "test_title": self.meta.get("test_title"), "status": self.status,
            "created": self.created, "started": self.started, "finished": self.finished,
            "duration_s": None if not self.started else round((self.finished or time.time()) - self.started, 2),
            "model": self.meta.get("model"), "spec": _shorten(self.spec), "decision": self.meta.get("decision"),
            "headline": res.get("headline"), "tiles": res.get("tiles"), "cancelled": bool(res.get("cancelled")) or self.status == "cancelled",
            "error": self.error, "error_kind": self.error_kind, "earlier_session": self.earlier_session,
            "preflight": self.meta.get("preflight"), "analysis": self.meta.get("analysis"),
        }

    def full(self) -> dict:
        d = self.summary()
        d["result"] = self.result
        d["files"] = self.files
        d["spec"] = _shorten(self.spec, 300_000)
        return d


class JobManager:
    def __init__(self, engine, max_jobs: int = 1, root: Path | None = None) -> None:
        self.engine = engine
        self.root = Path(root) if root else paths.data_dir() / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.q: queue.Queue[Job | None] = queue.Queue()
        self.threads = [threading.Thread(target=self._worker, daemon=True, name=f"job-runner-{i}") for i in range(max(1, max_jobs))]
        for t in self.threads:
            t.start()

    # ----- persistence of earlier sessions
    def load_previous(self, limit: int = 30) -> int:
        files = sorted(self.root.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        n = 0
        for f in reversed(files):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                spec = d["spec"]
                job = Job(
                    spec,
                    {
                        "test_title": d.get("test_title"),
                        "model": d.get("model"),
                        "preflight": d.get("preflight"),
                        "decision": d.get("decision"),
                        "analysis": d.get("analysis"),
                    },
                    f.parent,
                )
                job.id = d["id"]
                job.created, job.started, job.finished = d["created"], d["started"], d["finished"]
                job.status, job.result, job.error, job.error_kind = d["status"], d.get("result"), d.get("error"), d.get("error_kind")
                job.files = d.get("files", [])
                job.earlier_session = True
                self.jobs[job.id] = job
                self.order.append(job.id)
                n += 1
            except (KeyError, ValueError, OSError):
                continue
        return n

    # ----- API
    def submit(self, spec: dict, meta: dict) -> Job:
        job = Job(spec, meta, Path("."))
        job.workdir = self.root / job.id
        job.workdir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
            self._trim()
        job.add_event({"ev": "state", "status": "queued"})
        self.q.put(job)
        return job

    def _trim(self) -> None:
        while len(self.order) > MAX_HISTORY:
            for jid in self.order:
                if self.jobs[jid].status in TERMINAL:
                    self._remove(jid)
                    break
            else:
                break

    def _remove(self, jid: str) -> None:
        j = self.jobs.pop(jid, None)
        if jid in self.order:
            self.order.remove(jid)
        if j is not None:
            shutil.rmtree(j.workdir, ignore_errors=True)

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    def list(self) -> list[Job]:
        with self.lock:
            return [self.jobs[j] for j in list(self.order) if j in self.jobs]

    def cancel(self, jid: str) -> bool:
        with self.lock:
            job = self.jobs.get(jid)
            if job is None or job.status in TERMINAL:
                return False
            job.cancel.set()
            was_queued = job.status == "queued"
            if was_queued:
                job.finished = time.time()
                job.error = "cancelled before it started"
                job.set_status("cancelled")
        if was_queued:
            self._persist(job, "cancelled")
        return True

    def delete(self, jid: str) -> bool:
        job = self.jobs.get(jid)
        if job is None:
            return False
        if job.status not in TERMINAL:
            self.cancel(jid)
            return False
        with self.lock:
            self._remove(jid)
        return True

    def shutdown(self) -> None:
        active = set()
        for j in self.list():
            if j.status not in TERMINAL:
                if j.thread is not None:
                    active.add(j.thread)
                self.cancel(j.id)
        for _ in self.threads:
            self.q.put(None)
        for t in list(active) + [t for t in self.threads if t not in active]:
            if t is not threading.current_thread() and t.is_alive():
                t.join(timeout=2.5)

    # ----- runner
    def _worker(self) -> None:
        while True:
            job = self.q.get()
            if job is None:
                return
            with self.lock:
                if job.status != "queued":
                    continue
                job.thread = threading.current_thread()
                job.started = time.time()
                job.set_status("running")
            try:
                res = self.engine.run(job.spec, job.meta, job.add_event, job.cancel, str(job.workdir))
            except Exception as exc:  # noqa: BLE001
                res = {"status": "error", "result": None, "error": f"{type(exc).__name__}: {exc}", "error_kind": "internal"}
            finally:
                job.thread = None
            job.result = res.get("result")
            job.error, job.error_kind = res.get("error"), res.get("error_kind")
            job.finished = time.time()
            if job.result:
                job.files = [f for f in job.result.get("files", []) if (job.workdir / f["name"]).is_file()]
                for f in job.files:
                    f["size"] = (job.workdir / f["name"]).stat().st_size
            status = res.get("status", "error")
            self._persist(job, status)
            job.set_status(status if status in TERMINAL else "error")

    def _persist(self, job: Job, status: str) -> None:
        try:
            from mena_app import runtime as _rt

            d = _rt.clean(job.full())
            d["status"] = status
            (job.workdir / "job.json").write_text(json.dumps(d, default=str), encoding="utf-8")
        except (OSError, TypeError, ValueError):
            pass
