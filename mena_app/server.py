"""`mena webapp`: the local web application. Standard library HTTP server + JSON and Server-Sent-Events API; no torch in this process.

API (all JSON):
  GET  /api/info                       version, machine, models, catalogue, examples, optional features
  GET  /api/settings | POST {default_model: {domain: id|"auto"}}   per-user defaults (settings.py)
  GET  /api/benchmarks                 published speed/memory numbers for the Performance page (data/benchmarks.json)
  POST /api/structure   {structure}    analysis + atoms for the viewer
  POST /api/plan        {job}          the plan (model, device, precision, estimate, reasons) + exports (cli, python, json)
  POST /api/jobs        {job}          start a job -> {id}
  GET  /api/jobs                       history;  GET /api/jobs/<id>  one job with its result
  GET  /api/jobs/<id>/events           live events (SSE): state, status, log, progress, series, frame, result, error
  GET  /api/jobs/<id>/files/<name>     a result file;  POST /api/jobs/<id>/cancel;  DELETE /api/jobs/<id>
A job is {"task", "structure": {"format", "text", "name"}, "structure2"?, "params", "runtime", "system"} (validation.py).

Safety: listens on 127.0.0.1; checks the Host header (DNS rebinding); every state-changing request needs the X-Mena header and
a same-origin Origin; nothing from a request is executed or opened as a path; structure formats are a fixed list of text formats.
"""

from __future__ import annotations

import json
import mimetypes
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from mena_app import __version__, catalog, export, machine, models, paths, planner, runner, runtime, settings, structures, validation
from mena_app.jobs import JobManager

MAX_JSON = 10_000_000
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


MODEL_KEYS = ("id", "title", "role", "domains", "rank", "metrics", "training", "available", "downloadable", "size_mb", "backend",
              "dtypes", "lots", "release", "interpretable")


def _models_public() -> list[dict]:
    return [{k: m.get(k) for k in MODEL_KEYS} for m in models.all_models()]


def fetch_default_models() -> None:
    """Before the page opens: download the weights of each domain's auto model, so the first run starts at once."""
    from mena_app import weights

    if not weights.auto_download_allowed():
        return
    seen = set()
    for dom in settings.DOMAINS:
        try:
            m = models.choose("auto", dom)
        except models.ModelError:
            continue
        if m["id"] in seen or m["available"]:
            continue
        seen.add(m["id"])
        try:
            models.ensure_file(m)
        except models.ModelError as exc:
            print(f"mena: {exc}\n  (it will be tried again when a job needs it)", file=sys.stderr, flush=True)


def _features() -> dict:
    return {"smiles": True, "symmetry": True, "yaml": True, "rdkit": True, "spglib": True}


class Engine:
    """What JobManager needs: run a planned job to the end."""

    def run(self, spec, meta, emit, cancel, workdir):
        return runner.run(meta["_plan"], spec, workdir, emit, cancel)


class App:
    def __init__(self, host, port, token) -> None:
        self.host, self.port, self.token = host, port, token
        self.jobs = JobManager(Engine())
        self.loopback = host in LOOPBACK

    def plan(self, payload) -> tuple[dict, dict]:
        spec = validation.validate(payload)
        try:
            return spec, planner.plan(spec)
        except planner.PlanError as exc:
            raise validation.ValidationError(str(exc)) from None


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"mena/{__version__}"

        def log_message(self, *a):
            pass

        def send(self, code, body: bytes, ctype, extra=None):
            self.send_response(code)
            for k, v in {"Content-Type": ctype, "Content-Length": str(len(body)), "Cache-Control": "no-store",
                         "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", **(extra or {})}.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def json(self, obj, code=200):
            self.send(code, json.dumps(runtime.clean(obj), separators=(",", ":"), allow_nan=False, default=str).encode(), "application/json; charset=utf-8")

        def fail(self, code, msg, field=None):
            self.close_connection = True
            self.send(code, json.dumps({"error": msg, "field": field}).encode(), "application/json; charset=utf-8", {"Connection": "close"})

        def read_raw(self) -> bool:
            """Consume the whole request body up front, on every route and refusal, so a keep-alive connection never carries
            leftover bytes into the next request. A body too large to read closes the connection instead."""
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_JSON:
                self.close_connection = True
                self._raw = None
                return False
            self._raw = self.rfile.read(n) if n else b""
            return True

        def body(self):
            if self._raw is None:
                raise validation.ValidationError("the request is too large (10 MB at most)")
            try:
                return json.loads(self._raw) if self._raw else {}
            except ValueError:
                raise validation.ValidationError("the request is not valid JSON") from None

        def guard(self) -> bool:
            host = self.headers.get("Host", "")
            if app.loopback and host not in {f"127.0.0.1:{app.port}", f"localhost:{app.port}", f"[::1]:{app.port}"}:
                self.fail(403, "unexpected Host header")
                return False
            if self.command not in ("GET", "HEAD"):
                origin = self.headers.get("Origin")
                if self.headers.get("X-Mena") != "1" or (origin and urlparse(origin).netloc != host):
                    self.fail(403, "cross-site request refused")
                    return False
            if app.token and self.path.startswith("/api/"):
                cookie = self.headers.get("Cookie", "")
                if f"mena_token={app.token}" not in [c.strip() for c in cookie.split(";")] and self.headers.get("Authorization") != f"Bearer {app.token}":
                    self.fail(401, "this server needs its access token")
                    return False
            return True

        def handle_any(self):
            self.read_raw()
            if not self.guard():
                return
            u = urlparse(self.path)
            try:
                {"GET": self.get, "HEAD": self.get, "POST": self.post, "DELETE": self.delete}[self.command](u.path, parse_qs(u.query))
            except validation.ValidationError as exc:
                self.fail(400, str(exc), exc.field)
            except structures.StructureError as exc:
                self.fail(400, str(exc), "structure")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"internal error on {self.command} {self.path}: {type(exc).__name__}: {exc}\n")
                self.fail(500, "internal error")

        do_GET = do_HEAD = do_POST = do_DELETE = handle_any

        def get(self, path, q):
            if path in ("/", "/index.html"):
                if app.token and q.get("token") == [app.token]:
                    return self.send(302, b"", "text/plain", {"Location": "/", "Set-Cookie": f"mena_token={app.token}; HttpOnly; SameSite=Strict; Path=/"})
                return self.static("index.html")
            if path == "/favicon.ico":
                return self.static("icon.svg")
            if path.startswith("/static/"):
                return self.static(path[8:])
            if path == "/api/info":
                return self.json({"version": __version__, "machine": machine.snapshot(), "catalog": catalog.catalog(), "examples": structures.builtin_list(),
                                  "models": _models_public(), "settings": settings.load(), "features": _features()})
            if path == "/api/settings":
                return self.json({"settings": settings.load(), "models": _models_public()})
            if path == "/api/benchmarks":
                return self.send(200, (paths.DATA_DIR / "benchmarks.json").read_bytes(), "application/json; charset=utf-8")
            if path == "/api/machine":
                return self.json(machine.snapshot())
            if path == "/api/jobs":
                return self.json({"jobs": [j.summary() for j in app.jobs.list()]})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})(?:/(events|files/([A-Za-z0-9_.\-]{1,80})))?", path)
            if not m:
                return self.fail(404, "not found")
            job = app.jobs.get(m.group(1))
            if job is None:
                return self.fail(404, "no such job")
            if m.group(2) is None:
                return self.json(job.full())
            if m.group(2) == "events":
                return self.sse(job, q)
            name = m.group(3)
            if name not in {f["name"] for f in job.files} and name not in ("result.json",):
                return self.fail(404, "no such file")
            if name == "result.json":
                return self.send(200, json.dumps(runtime.clean(job.full()), indent=1, allow_nan=False, default=str).encode(), "application/json",
                                 {"Content-Disposition": f'attachment; filename="mena_{job.test}_{job.id}.json"'})
            fp = job.workdir / name
            if not fp.is_file():
                return self.fail(404, "file is gone")
            self.send(200, fp.read_bytes(), "application/octet-stream", {"Content-Disposition": f'attachment; filename="{job.test}_{job.id}_{name}"'})

        def static(self, rel):
            target = (paths.STATIC_DIR / rel).resolve()
            if not target.is_relative_to(paths.STATIC_DIR.resolve()) or not target.is_file():
                return self.fail(404, "not found")
            ctype = {".js": "text/javascript", ".svg": "image/svg+xml"}.get(target.suffix) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send(200, target.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))

        def sse(self, job, q):
            try:
                last = int(q.get("from", ["0"])[0] or self.headers.get("Last-Event-ID") or 0)
            except ValueError:
                last = 0
            self.send_response(200)
            for k, v in {"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-store", "Connection": "close"}.items():
                self.send_header(k, v)
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(b"retry: 2000\n\n")
                while True:
                    events, finished = job.events_after(last, 12.0)
                    if not events:
                        if finished:
                            self.wfile.write(b"event: end\ndata: {}\n\n")
                            self.wfile.flush()
                            return
                        self.wfile.write(b": keep-alive\n\n")
                    for e in events:
                        last = e["seq"]
                        self.wfile.write(f"id: {e['seq']}\nevent: {e['ev']}\ndata: {json.dumps(runtime.clean(e), separators=(',', ':'), allow_nan=False, default=str)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def post(self, path, q):
            if path == "/api/structure":
                b = self.body()
                if not isinstance(b, dict):
                    return self.fail(400, "expected a JSON object {\"structure\": ...}")
                s = validation._structure(b.get("structure"), "structure")
                a = structures.load(s)
                return self.json({"analysis": structures.analyze(a), "viewer": structures.to_viewer(a)})
            if path == "/api/plan":
                spec, pl = app.plan(self.body())
                return self.json({"spec": spec, "analysis": pl["analysis"], "decision": pl["decision"], "model": pl["model_public"],
                                  "exports": {"cli": export.to_cli(spec), "python": export.to_python(spec, pl, __version__), "json": export.to_json(spec)}})
            if path == "/api/jobs":
                spec, pl = app.plan(self.body())
                meta = {"test_title": catalog.task(spec["task"])["title"], "model": pl["model_public"], "decision": pl["decision"],
                        "analysis": pl["analysis"], "_plan": pl, "preflight": {"level": pl["decision"]["memory_level"]}}
                job = app.jobs.submit(spec, meta)
                return self.json({"id": job.id}, 201)
            if path == "/api/settings":
                b = self.body()
                dm = b.get("default_model") if isinstance(b, dict) else None
                if not isinstance(dm, dict) or not all(isinstance(v, str) for v in dm.values()):
                    return self.fail(400, "expected {\"default_model\": {\"crystal\": \"model id or auto\"}}")
                try:
                    for dom, mid in dm.items():  # validate everything first: a bad entry changes nothing
                        models.check_default(dom, mid)
                    for dom, mid in dm.items():
                        models.set_default(dom, mid)
                except models.ModelError as exc:
                    return self.fail(400, str(exc))
                return self.json({"settings": settings.load(), "models": _models_public()})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/cancel", path)
            if m:
                ok = app.jobs.cancel(m.group(1))
                return self.json({"cancelled": ok}, 200 if ok else 409)
            self.fail(404, "not found")

        def delete(self, path, q):
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})", path)
            if m and app.jobs.delete(m.group(1)):
                return self.json({"deleted": True})
            self.fail(409, "the job is still running or does not exist")

    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"  # on Windows SO_REUSEADDR lets two servers share a port

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            super().handle_error(request, client_address)


def create(host="127.0.0.1", port=8765, token=None, tries=20):
    if host not in LOOPBACK and not token:
        raise SystemExit("Refusing a non-local address without --token: anyone on your network could run jobs on this computer.")
    for p in range(port, port + tries):
        try:
            srv = Server((host, p), BaseHTTPRequestHandler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port between {port} and {port + tries - 1}")
    app = App(host, srv.server_address[1], token)
    srv.RequestHandlerClass = make_handler(app)
    return srv, app


def serve(host="127.0.0.1", port=8765, token=None, open_browser=True) -> int:
    fetch_default_models()
    srv, app = create(host, port, token)
    n = app.jobs.load_previous()
    url = f"http://{'127.0.0.1' if host in LOOPBACK else host}:{app.port}/" + (f"?token={token}" if token else "")
    threading.Thread(target=machine.probe, daemon=True).start()
    print(f"mena {__version__} is running at {url}  (Ctrl+C to stop)" + (f"; {n} earlier runs in the history" if n else ""), flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        app.jobs.shutdown()
        srv.server_close()
    return 0


_ = time
