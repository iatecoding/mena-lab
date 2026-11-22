"""Runs one planned job in a worker subprocess and relays its events. Used by both the CLI and the web server, so a job
behaves the same wherever it starts. Cancel = SIGTERM (the task returns what it has), then SIGKILL after a grace period;
a watchdog stops the job if free memory falls low enough that the computer would start swapping."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Callable

from mena_app import paths, system
from mena_app.runtime import CANCEL_FILE

CANCEL_GRACE_S = 1.5
WINDOWS_CANCEL_GRACE_S = 6.0  # Windows has no catchable SIGTERM: the worker sees the cancel file at its next step
WINDOWS = sys.platform == "win32"
KILL_BELOW_GB = 1.0
VRAM_KILL_BELOW_GB = 0.15
MEM_POLL_INTERVAL_S = 2.0


def _env(model: dict) -> dict:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(paths.data_dir() / "compile_cache"))
    parts = [p for p in (model.get("code_root"), str(paths.REPO_ROOT), env.get("PYTHONPATH")) if p]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def isolated_python(model: dict, args: list[str], payload: str | None = None, timeout: float = 600) -> subprocess.CompletedProcess:
    """Run python with the model's code first on the path, from a neutral folder: a `-c`/`-m` child puts its working
    directory first on sys.path, so starting it inside a source tree would load that tree's `lam` instead."""
    return subprocess.run([sys.executable, *args], input=payload, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, env=_env(model), cwd=str(paths.data_dir()))


def run(plan: dict, spec: dict, workdir: str, emit: Callable[[dict], None], cancel: threading.Event,
        timeout_s: float | None = None, python: str | None = None) -> dict:
    os.makedirs(workdir, exist_ok=True)
    cancel_file = os.path.join(workdir, CANCEL_FILE)
    if os.path.exists(cancel_file):
        os.remove(cancel_file)
    model = plan["model"]
    if model.get("downloadable") and not model.get("available"):  # first use of a shipped model: fetch its weights once
        from mena_app import models

        emit({"ev": "status", "text": f"Downloading {model.get('title', model['id'])} weights (once, about {model.get('size_mb', '?')} MB)…"})
        try:
            model = plan["model"] = models.ensure_file(model)
        except models.ModelError as exc:
            emit({"ev": "error", "kind": "user", "message": str(exc)})
            return {"status": "error", "result": None, "error": str(exc), "error_kind": "user"}
    req = {"spec": spec, "decision": plan["decision"], "model": model, "workdir": workdir}
    proc = subprocess.Popen([python or sys.executable, "-m", "mena_app.worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", cwd=workdir,
                            env=_env(model), start_new_session=True, bufsize=1)
    out: dict = {"status": "error", "result": None, "error": None, "error_kind": None}
    err_tail: list[str] = []

    def read_out() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("ev") == "result":
                out["result"] = d["data"]
                out["status"] = "cancelled" if d["data"].get("cancelled") else "done"
            elif d.get("ev") == "error":
                out["error"], out["error_kind"] = d.get("message"), d.get("kind")
                out["status"] = "cancelled" if d.get("kind") == "cancelled" else "error"
            emit(d)

    def read_err() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            err_tail.append(line.rstrip())
            del err_tail[:-40]

    readers = [threading.Thread(target=read_out, daemon=True), threading.Thread(target=read_err, daemon=True)]
    reason = None
    try:
        for t in readers:
            t.start()
        try:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(req))
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        dec_dev = str((plan or {}).get("decision", {}).get("device") or "cpu")
        cuda_idx = int(dec_dev.split(":", 1)[1]) if dec_dev.startswith("cuda:") and dec_dev[5:].isdigit() else 0
        watch_vram = dec_dev == "cuda" or (dec_dev.startswith("cuda:") and dec_dev[5:].isdigit())
        mem_reason_msg: str | None = None
        t0, term_at, last_mem_check = time.time(), None, 0.0
        while proc.poll() is None:
            time.sleep(0.2)
            now = time.time()
            if term_at is None:
                if cancel.is_set():
                    reason = "cancelled"
                elif timeout_s and now - t0 > timeout_s:
                    reason = "timeout"
                elif now - last_mem_check >= MEM_POLL_INTERVAL_S:
                    last_mem_check = now
                    free = system.available_memory_bytes()
                    if free and free / 1e9 < KILL_BELOW_GB:
                        reason = "memory_guard"
                        mem_reason_msg = f"Stopped to protect your computer: free memory fell below {KILL_BELOW_GB:.0f} GB."
                    elif watch_vram:
                        vram_free = system.available_vram_bytes(cuda_idx)
                        if vram_free is not None and vram_free / 1e9 < VRAM_KILL_BELOW_GB:
                            reason = "memory_guard"
                            mem_reason_msg = f"Stopped to protect your GPU ({dec_dev}): free VRAM fell below {VRAM_KILL_BELOW_GB:.2f} GB."
                if reason:
                    _request_stop(proc, cancel_file)
                    term_at = now
            elif now - term_at > (WINDOWS_CANCEL_GRACE_S if WINDOWS and reason != "memory_guard" else CANCEL_GRACE_S):
                _signal(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
        proc.wait()
        for t in readers:
            t.join(timeout=3)
        _signal(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
    finally:
        try:
            if proc.poll() is None:
                _terminate_worker(proc)
        finally:
            for t in readers:
                if t.is_alive():
                    t.join(timeout=1.0)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
    if reason == "cancelled":
        out["status"] = "cancelled"
    elif reason == "timeout":
        out.update(status="timeout", error=f"Stopped after {timeout_s:.0f} s.")
    elif reason == "memory_guard":
        out.update(status="stopped", error=mem_reason_msg or f"Stopped to protect your computer: free memory fell below {KILL_BELOW_GB:.0f} GB.")
    elif out["result"] is None and out["error"] is None:
        code = proc.returncode
        why = "killed by the system, probably out of memory" if code in (-9, -6, 137) else f"exit code {code}"
        out.update(status="error", error_kind="crash", error=f"The worker ended without a result ({why}).\n" + "\n".join(err_tail[-8:]))
    return out


def _request_stop(proc: subprocess.Popen, cancel_file: str) -> None:
    """Ask the worker to stop and return what it has: a cancel file it checks every step (all systems), plus SIGTERM where
    it can be caught. On Windows `terminate()` is a hard kill, so it is used only after the grace period."""
    try:
        with open(cancel_file, "w", encoding="utf-8") as fh:
            fh.write("cancel\n")
    except OSError:
        pass
    if not WINDOWS:
        _signal(proc, signal.SIGTERM)


def _terminate_worker(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    _signal(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=1.5)
    except subprocess.TimeoutExpired:
        _signal(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
        proc.wait()
    except BaseException:
        _signal(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
        proc.wait()
        raise


def _signal(proc: subprocess.Popen, sig: int) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, sig)
            return
        except ProcessLookupError:
            return
        except (PermissionError, OSError):
            pass
    try:
        if sig == signal.SIGTERM:
            proc.terminate()
        elif hasattr(signal, "SIGKILL") and sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.send_signal(sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass
