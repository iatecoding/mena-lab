"""`mena`: the command line. Flags are generated from catalog.py, so they always match the web page.

    mena relax structure.cif --fmax 0.01            mena md water.xyz --temperature 330 --steps 5000
    mena phonons si --supercell 3                    mena neb nh3_up nh3_down
    mena relax caffeine --dry-run                    show the plan (model, device, precision, time, memory) and stop
    mena run job.json                                a saved job (from --export json or the web page)
    mena webapp                                      the local web application (alias: mena ui)
    mena models | mena models add NAME PATH [--code-root DIR] [--domains molecule,crystal]
    mena calibrate                                   measure this computer once (about two minutes)
    mena doctor                                      check the installation
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from mena_app import __version__, catalog, export, models, structures, validation
from mena_app.cli_out import Printer

ALIASES = {"sp": "singlepoint", "vib": "vibrations"}


def _add_param(ap, p: dict) -> None:
    flag = "--" + p["name"].replace("_", "-")
    h = (p.get("help") or p["label"]) + (f" [{p['unit']}]" if p.get("unit") else "") + (f" (default {p['default']})" if p["default"] is not None else "")
    if p["type"] == "bool":
        ap.add_argument(flag, dest=p["name"], action=argparse.BooleanOptionalAction, default=None, help=h)
    elif p["type"] == "choice" and "options" in p:
        ap.add_argument(flag, dest=p["name"], choices=p["options"], default=None, help=h)
    else:
        ap.add_argument(flag, dest=p["name"], type={"int": int, "float": float}.get(p["type"], str), default=None, help=h)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mena", description="Run Mena interatomic potentials.", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Structures: a .xyz/.extxyz/.cif/.vasp/POSCAR file, smiles:CCO, or an example name "
                                        f"({', '.join(structures.BUILTINS)}).")
    ap.add_argument("--version", action="version", version=f"mena {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="command")
    for t in catalog.TASKS:
        aliases = [k for k, v in ALIASES.items() if v == t["id"]]
        sp = sub.add_parser(t["id"], aliases=aliases, help=t["summary"], description=t["summary"])
        sp.add_argument("structure", help="file, smiles:..., or example name")
        if t.get("needs_second"):
            sp.add_argument("structure2", help="the other end of the reaction")
        for p in t["params"]:
            _add_param(sp, p)
        _common(sp)
    r = sub.add_parser("run", help="run a saved job file (.json, .yaml, or .yml)")
    r.add_argument("config")
    _common(r)
    u = sub.add_parser("webapp", aliases=["ui"], help="open the local web application")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--token", default=None, help="access token; required with a non-local --host")
    u.add_argument("--no-browser", action="store_true")
    m = sub.add_parser("models", help="list, download or add models; set the default model per structure kind")
    msub = m.add_subparsers(dest="mcmd")
    ma = msub.add_parser("add", help="add a model file")
    ma.add_argument("id")
    ma.add_argument("path")
    ma.add_argument("--code-root", help="folder whose lam/ package loads this model (if not this repository)")
    ma.add_argument("--backend", default=None, choices=["lam", "lam4"],
                    help="model backend (inferred for shipped ids and lam5 file names)")
    ma.add_argument("--domains", action="append", default=None,
                    help="comma-separated or repeated domains (default: molecule,crystal,slab)")
    ma.add_argument("--title")
    mr = msub.add_parser("remove", help="remove a model you added")
    mr.add_argument("id")
    md = msub.add_parser("download", help="download model weights now (otherwise they download on first use)")
    md.add_argument("ids", nargs="*", help="model ids (default: every model with a download source)")
    mdf = msub.add_parser("default", help="show or set the model used for molecules, crystals or surfaces when the model is auto")
    mdf.add_argument("domain", nargs="?", choices=["molecule", "crystal", "slab"])
    mdf.add_argument("model", nargs="?", help="model id, or auto for the recommended one")
    c = sub.add_parser("calibrate", help="measure speed and memory on this computer (about two minutes)")
    c.add_argument("--model", default=None, help="model id (default: one per backend)")
    sub.add_parser("doctor", help="check the installation")
    return ap


def _common(sp) -> None:
    g = sp.add_argument_group("how to run")
    for p in catalog.RUNTIME + catalog.SYSTEM:
        _add_param(g, p)
    g.add_argument("--out", default=None, help="folder for results (default: ./mena_runs/<task>-<time>)")
    g.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    g.add_argument("--export", choices=["cli", "python", "json"], help="print the job in this form and stop")
    g.add_argument("--json", action="store_true", help="print the result as JSON (for scripts)")
    g.add_argument("-q", "--quiet", action="store_true", help="no live progress")


def _pick(schema: list[dict], ns: argparse.Namespace | dict) -> dict:
    get = ns.get if isinstance(ns, dict) else (lambda k: getattr(ns, k, None))
    return {p["name"]: get(p["name"]) for p in schema if get(p["name"]) is not None}


def spec_from_args(task: str, ns, cfg: dict | None = None) -> dict:
    if not isinstance(task, str) or catalog.task(task) is None:
        raise validation.ValidationError(f"unknown task {task!r}")
    t = catalog.task(task)
    src = cfg or ns
    s1 = (cfg or {}).get("structure") if (cfg and "structure" in cfg) else getattr(ns, "structure", None)
    if not isinstance(s1, str) or not s1.strip():
        raise validation.ValidationError("structure must be a non-empty string")
    raw = {"task": task, "structure": structures.from_path(s1), "params": _pick(t["params"], src),
           "runtime": _pick(catalog.RUNTIME, src), "system": _pick(catalog.SYSTEM, src)}
    if t.get("needs_second"):
        s2 = (cfg or {}).get("structure2") if (cfg and "structure2" in cfg) else getattr(ns, "structure2", None)
        if not isinstance(s2, str) or not s2.strip():
            raise validation.ValidationError("structure2 must be a non-empty string")
        raw["structure2"] = structures.from_path(s2)
    if ns is not None and cfg is not None:  # command-line flags override the file
        for group, schema in (("params", t["params"]), ("runtime", catalog.RUNTIME), ("system", catalog.SYSTEM)):
            raw[group].update(_pick(schema, ns))
    return validation.validate(raw)


def _flat_config(cfg: dict, task, path: str) -> dict:
    """Accept settings at the top level or inside params / runtime / system blocks; refuse unknown names (a typo would
    otherwise run silently with the default)."""
    t = catalog.task(task) if isinstance(task, str) else None
    if t is None:
        raise validation.ValidationError(f"{path}: unknown or missing task {task!r}")
    known = {p["name"] for p in t["params"]} | {p["name"] for p in catalog.RUNTIME} | {p["name"] for p in catalog.SYSTEM}
    flat = {}
    for k, v in cfg.items():
        if k in ("params", "runtime", "system") and isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    unknown = sorted(set(flat) - known - {"task", "structure", "structure2"})
    if unknown:
        raise validation.ValidationError(f"{path}: unknown setting{'s' if len(unknown) > 1 else ''} {', '.join(unknown)} "
                                         f"(see `mena {task} --help`)")
    return flat


def _load_config(path: str) -> dict:
    try:
        text = structures.read_text_any(Path(path))
    except OSError as exc:
        raise validation.ValidationError(f"{path}: {exc}") from None
    if path.endswith((".yaml", ".yml")):
        from mena_app import yaml_lite

        try:
            cfg = yaml_lite.safe_load(text)
        except ValueError as exc:
            raise validation.ValidationError(f"invalid YAML in {path}: {exc}") from None
        if not isinstance(cfg, dict):
            raise validation.ValidationError(f"expected a YAML mapping in {path}")
        return cfg
    try:
        cfg = json.loads(text)
    except ValueError as exc:
        raise validation.ValidationError(f"invalid JSON in {path}: {exc}") from None
    if not isinstance(cfg, dict):
        raise validation.ValidationError(f"expected a JSON object in {path}")
    return cfg


def cmd_task(ns) -> int:
    from mena_app import planner, runner

    out = Printer(quiet=ns.quiet or ns.json)
    if ns.cmd == "run":
        cfg = _load_config(ns.config)
        task = ALIASES.get(cfg.get("task"), cfg.get("task"))
        cfg = _flat_config(cfg, task, ns.config)
        base = Path(ns.config).resolve().parent
        for k in ("structure", "structure2"):
            if k in cfg and isinstance(cfg[k], str) and not cfg[k].startswith("smiles:") and cfg[k] not in structures.BUILTINS:
                cfg[k] = str((base / cfg[k]).resolve()) if not Path(cfg[k]).is_absolute() else cfg[k]
        spec = spec_from_args(task, ns, cfg)
    else:
        spec = spec_from_args(ALIASES.get(ns.cmd, ns.cmd), ns)
    try:
        pl = planner.plan(spec)
    except planner.PlanError as exc:
        out.fail(str(exc))
        return 2
    if ns.export:
        print({"cli": lambda: export.to_cli(spec), "python": lambda: export.to_python(spec, pl, __version__), "json": lambda: export.to_json(spec)}[ns.export]())
        return 0
    out.plan(spec, pl)
    if ns.dry_run:
        return 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    workdir = Path(ns.out or f"mena_runs/{spec['task']}-{stamp}").resolve()
    cancel = threading.Event()
    try:
        res = runner.run(pl, spec, str(workdir), out.event, cancel)
    except KeyboardInterrupt:
        cancel.set()
        res = {"status": "cancelled", "result": None}
    record = {"mena_version": __version__, "spec": spec, "plan": {k: pl[k] for k in ("analysis", "decision", "model_public")}, **res}
    (workdir / "result.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    p1, p2 = export.save_inputs(spec, workdir)
    (workdir / "job.json").write_text(export.to_json(spec, p1, p2), encoding="utf-8")
    if ns.json:
        print(json.dumps(record, default=str))
    else:
        out.result(res, workdir)
    return 0 if res["status"] in ("done", "cancelled") else 1


def cmd_models(ns) -> int:
    if ns.mcmd == "add":
        doms = [d.strip() for chunk in ns.domains for d in chunk.split(",") if d.strip()] if ns.domains else None
        m = models.add(ns.id, ns.path, ns.code_root, doms, ns.title, backend=ns.backend)
        print(f"added {m['id']} ({'found' if m['available'] else 'FILE MISSING'}): {m['path']}")
        return 0
    if ns.mcmd == "remove":
        if not models.remove(ns.id):
            raise models.ModelError(f"{ns.id!r} is not in your model list ({models.user_file()})")
        print(f"removed {ns.id}")
        return 0
    if ns.mcmd == "download":
        rows = models.all_models()
        todo = [m for m in rows if m["id"] in ns.ids] if ns.ids else [m for m in rows if m.get("downloadable")]
        unknown = set(ns.ids) - {m["id"] for m in todo}
        if unknown:
            raise models.ModelError(f"no model called {', '.join(sorted(unknown))} (see `mena models`)")
        for m in todo:
            if m["available"]:
                print(f"{m['id']}: already installed ({m['path']})")
            else:
                print(f"{m['id']}: {models.ensure_file(m)['path']}")
        return 0
    if ns.mcmd == "default":
        if ns.domain and ns.model:
            models.set_default(ns.domain, ns.model)
        from mena_app import settings

        cur = settings.load()["default_model"]
        for d in settings.DOMAINS:
            if ns.domain in (None, d):
                print(f"{models.PLURAL[d]:10} {cur.get(d, 'auto (recommended)')}")
        return 0
    Printer().models(models.all_models())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    ns = ap.parse_args(argv)
    if ns.cmd is None:
        ap.print_help()
        return 0
    try:
        if ns.cmd in ("webapp", "ui"):
            from mena_app.server import serve
            return serve(ns.host, ns.port, ns.token, not ns.no_browser)
        if ns.cmd == "models":
            return cmd_models(ns)
        if ns.cmd == "calibrate":
            from mena_app.calibrate import main as cal
            return cal(ns.model)
        if ns.cmd == "doctor":
            from mena_app.doctor import main as doc
            return doc()
        return cmd_task(ns)
    except (structures.StructureError, validation.ValidationError, models.ModelError) as exc:
        print(f"mena: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
