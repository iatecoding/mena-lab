"""Terminal output for `mena`: the plan, a live progress line, and the result, readable without colour or a wide window."""

from __future__ import annotations

import functools
import os
import shutil
import sys

from mena_app import catalog

print = functools.partial(print, flush=True)  # noqa: A001 - keep stdout in step with stderr

for _stream in (sys.stdout, sys.stderr):  # a Windows code page cannot print Å or →: replace instead of crashing
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except (OSError, ValueError):
            pass


def _ansi_ok() -> bool:
    """Colour only on a terminal; on Windows, only if the console accepts ANSI codes (virtual-terminal mode)."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR") is not None:
        return False
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        return bool(k32.GetConsoleMode(h, ctypes.byref(mode)) and k32.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:  # noqa: BLE001
        return False


TTY = _ansi_ok()
B, D, G, Y, R0, X = (("\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m") if TTY else ("",) * 6)


def _dur(s) -> str:
    if s is None:
        return "unknown (run `mena calibrate`)"
    return f"{s:.1f} s" if s < 60 else (f"{s / 60:.1f} min" if s < 3600 else f"{s / 3600:.1f} h")


class Printer:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self._line = False

    def _clear(self) -> None:
        if self._line:
            sys.stderr.write("\r\033[K" if TTY else "\n")
            self._line = False

    def fail(self, msg: str) -> None:
        print(f"{R0}mena: {msg}{X}", file=sys.stderr)

    def plan(self, spec: dict, pl: dict) -> None:
        if self.quiet:
            return
        a, d, m = pl["analysis"], pl["decision"], pl["model_public"]
        t = catalog.task(spec["task"])
        where = a["domain"] + (f", {a['spacegroup']}" if a.get("spacegroup") else "")
        print(f"{B}{t['title']}{X}  {a['formula']}  {D}{a['n_atoms']} atoms, {where}{X}")
        rows = [("model", f"{m['title']}  {D}{m.get('metrics') or ''}{X}"),
                ("device", f"{d['device'].upper()}  {D}{d['reasons'].get('device', '')}{X}"),
                ("precision", f"{d['dtype']}  {D}{d['reasons'].get('dtype', '')}{X}"),
                ("compiled", f"{'on' if d.get('compile') else 'off'}  {D}{d['reasons'].get('compile', '')}{X}"),
                ("estimate", f"{_dur(d['seconds'])}, {d['peak_gb'] if d['peak_gb'] is not None else '?'} GB peak  {D}({d['calls']:,} evaluations; {d['cost_source']}){X}")]
        if d["reasons"].get("size"):
            rows.append(("size", f"{d['atoms']} atoms  {D}{d['reasons']['size']}{X}"))
        if d["low_memory"]:
            rows.append(("memory", f"low-memory mode  {D}{d['reasons']['low_memory']}{X}"))
        for k, v in rows:
            print(f"  {k:<10}{v}")
        for w in a.get("warnings", []):
            print(f"  {Y}! {w}{X}")
        print()

    def event(self, e: dict) -> None:
        if self.quiet:
            return
        ev = e.get("ev")
        if ev == "progress" and e.get("total"):
            pct = 100 * e["done"] / e["total"]
            bar = "█" * int(pct / 5) + "·" * (20 - int(pct / 5))
            stats = "  ".join(f"{k} {v}" for k, v in (e.get("stats") or {}).items())
            msg = f"  {bar} {pct:5.1f}%  {e.get('label', '')}  {D}{stats}{X}"
            if TTY:
                cols = shutil.get_terminal_size((80, 20)).columns
                sys.stderr.write("\r\033[K" + msg[: cols + 20])
                self._line = True
            elif int(pct) % 10 == 0:
                sys.stderr.write(msg + "\n")
        elif ev in ("status", "log"):
            self._clear()
            sys.stderr.write(f"  {D}{e['text']}{X}\n")
        elif ev == "error" and e.get("kind") != "cancelled":
            self._clear()
            sys.stderr.write(f"{R0}  {e['message']}{X}\n")

    def result(self, res: dict, workdir) -> None:
        self._clear()
        r = res.get("result")
        if not r:
            print(f"{R0}{res['status']}: {res.get('error') or ''}{X}")
            return
        print(f"\n{B}{r['headline']}{X}\n")
        for t in r.get("tiles", []):
            v = t["value"]
            if isinstance(v, float):
                v = f"{v:.{t['digits'] if t.get('digits') is not None else 4}f}"
            mark = {"pass": G + "✓ ", "warn": Y + "! ", "fail": R0 + "✗ "}.get(t.get("status") or "", "")
            print(f"  {t['label']:<38}{mark}{v} {t.get('unit') or ''}{X if mark else ''}")
        for k, v in (r.get("facts") or {}).items():
            print(f"  {D}{k:<38}{v}{X}")
        for n in r.get("notes", []):
            print(f"  {Y}! {n}{X}")
        print(f"\n  results in {workdir}/")
        for f in r.get("files", []):
            print(f"    {f['name']:<24}{D}{f['label']}{X}")
        print(f"    {'result.json':<24}{D}everything above, as data{X}\n    {'job.json':<24}{D}re-run with `mena run job.json`{X}")

    def models(self, rows: list[dict]) -> None:
        for m in rows:
            ok = (f"{G}installed{X}" if m["available"] else f"{D}downloads on first use{X}" if m.get("downloadable")
                  else f"{Y}unsupported{X}" if m.get("unsupported") else f"{Y}missing{X}")
            print(f"{B}{m['id']:<14}{X}{m.get('title', '')}  [{ok}]  {D}{', '.join(m.get('domains', []))}{X}")
            if m.get("metrics"):
                print(f"{'':14}{D}{m['metrics']}{X}")
            print(f"{'':14}{D}{m.get('path')}{X}")
