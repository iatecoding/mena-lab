"""The model list. No torch. A model is {id, title, path, code_root, domains, ...}; `code_root` is the folder whose `lam/` package
can load it, so architectures that live in different code trees run side by side (each job is its own process).

Sources, later wins: mena_app/data/models.json (shipped, paths relative to the repo), then ~/.mena/models.json (written by
`mena models add`, for your own paths), then $MENA_MODELS. Models whose file is missing stay listed as unavailable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mena_app import paths, settings, weights


class ModelError(ValueError):
    pass


def _read(p: Path) -> list[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("models", [])
    except (OSError, ValueError):
        return []


def user_file() -> Path:
    return paths.data_dir() / "models.json"


def _default_path(entry: dict) -> str | None:
    """Where a shipped model's file lives: the per-user download folder, else `weights/` of a source checkout."""
    if not entry.get("file"):
        return None
    user = weights.local_path(entry)
    repo = paths.REPO_ROOT / "weights" / entry["file"]
    return str(repo if not user.is_file() and repo.is_file() else user)


def _resolve(entry: dict, base: Path) -> dict:
    m = dict(entry)
    if not m.get("path"):
        m["path"] = _default_path(m)
    for key in ("path", "code_root"):
        if m.get(key):
            p = Path(m[key]).expanduser()
            m[key] = str(p if p.is_absolute() else (base / p).resolve())
    m["available"] = bool(m.get("path")) and Path(m["path"]).is_file()
    m["downloadable"] = not m["available"] and weights.downloadable(m) and m.get("path") == _default_path(m)
    return m


def all_models() -> list[dict]:
    found: dict[str, dict] = {}
    srcs = [(paths.DATA_DIR / "models.json", paths.REPO_ROOT), (user_file(), Path.home())]
    if os.environ.get("MENA_MODELS"):
        srcs.append((Path(os.environ["MENA_MODELS"]), Path.home()))
    for f, base in srcs:
        for e in _read(f):
            if not isinstance(e, dict) or not e.get("id"):
                continue
            resolved = _resolve(e, base)
            prev = found.get(e["id"])
            found[e["id"]] = {**prev, **resolved} if prev else resolved
    for m in found.values():  # e.g. an entry left in ~/.mena/models.json by an older version: listed, never chosen
        if m.get("backend", "lam") not in DEFAULT_LOTS:
            m.update(available=False, downloadable=False,
                     unsupported=f"backend {m.get('backend')!r} is not supported by this version; remove it with `mena models remove {m['id']}`")
    return list(found.values())


VALID_DOMAINS = ("molecule", "crystal", "slab")

DEFAULT_LOTS = {
    "lam": ["mp_pbe", "mp_pbeU", "omat_pbe", "omat_pbeU", "matpes_pbe", "matpes_r2scan", "omol_wb97mv"],
    # distinct, trained levels of theory only: omat_pbe / matpes_pbe share mp_pbe's state on lam5, omat_pbeU shares mp_pbeU's,
    # and omc25_pbe / openrxn_wb97xd / spice_wb97md3 were not trained in lam5
    "lam4": ["mp_pbe", "mp_pbeU", "matpes_r2scan", "oc20_rpbe", "oc22_pbeU", "omol_wb97mv"],
}

LAM6_DEFAULT_LOTS = [
    "mp_pbe", "mp_pbeU", "omat_pbe", "omat_pbeU", "matpes_pbe", "matpes_r2scan",
    "oc20_rpbe", "omc25_pbe", "omol_wb97mv", "spice_wb97md3",
]


def _is_lam6_entry(model_id: str, path: str) -> bool:
    low = f"{model_id} {Path(path).name} {Path(path).parent.name}".lower()
    return "lam6" in low


def _infer_backend(model_id: str, path: str, shipped: dict | None) -> str:
    if shipped and shipped.get("backend") in DEFAULT_LOTS:
        return shipped["backend"]
    low = f"{model_id} {Path(path).name} {Path(path).parent.name}".lower()
    if "lam6" in low or "lam5" in low or "lam4" in low:
        return "lam4"
    return "lam"


def add(model_id: str, path: str, code_root: str | None = None, domains: list[str] | None = None,
        title: str | None = None, backend: str | None = None) -> dict:
    shipped = next((x for x in _read(paths.DATA_DIR / "models.json") if x.get("id") == model_id), None)
    eff_backend = backend or _infer_backend(model_id, path, shipped)
    if eff_backend not in DEFAULT_LOTS:
        raise ModelError(f"unknown backend {eff_backend!r}; choose 'lam' or 'lam4'")
    if domains:
        bad = [d for d in domains if d not in VALID_DOMAINS]
        if bad:
            raise ModelError(f"unknown domain(s) {', '.join(bad)}; valid: {', '.join(VALID_DOMAINS)}")
        eff_domains = list(dict.fromkeys(domains))
    elif shipped and shipped.get("domains"):
        eff_domains = list(shipped["domains"])
    else:
        eff_domains = ["molecule", "crystal", "slab"]
    is_lam6 = eff_backend == "lam4" and _is_lam6_entry(model_id, path)
    f = user_file()
    rows = [e for e in _read(f) if e.get("id") != model_id]
    e: dict = dict(shipped) if shipped else {}
    e.update({
        "id": model_id,
        "title": title or (shipped.get("title") if shipped else None) or model_id,
        "path": str(Path(path).expanduser().resolve()),
        "backend": eff_backend,
        "domains": eff_domains,
        "lots": list(LAM6_DEFAULT_LOTS if is_lam6 else DEFAULT_LOTS[eff_backend]),
        "dtypes": ["float32", "float64"],
        "low_memory": True,
    })
    if is_lam6 and "rank" not in e:
        e["rank"] = {"crystal": 0, "slab": 0, "molecule": 1}
    if code_root:
        e["code_root"] = str(Path(code_root).expanduser().resolve())
    elif "code_root" in e:
        del e["code_root"]
    rows.append(e)
    f.write_text(json.dumps({"models": rows}, indent=1), encoding="utf-8")
    return _resolve(e, Path.home())


PLURAL = {"molecule": "molecules", "crystal": "crystals", "slab": "surfaces"}


def _usable(m: dict) -> bool:
    return m["available"] or (m.get("downloadable", False) and weights.auto_download_allowed())


def _missing(m: dict) -> ModelError:
    if m.get("unsupported"):
        return ModelError(f"{m['id']}: {m['unsupported']}")
    if m.get("release") == "preview" and not weights.downloadable(m):
        return ModelError(f"{m.get('title', m['id'])} ({m['id']}) is a preview: its weights are not published yet. "
                          f"Put the file at {m.get('path')} or run `mena models add {m['id']} PATH`.")
    return ModelError(f"the file of model {m['id']!r} is missing: {m.get('path')} "
                      f"(run `mena models download {m['id']}` or `mena models add {m['id']} PATH`)")


def choose(requested: str, domain: str) -> dict:
    """The model for a job, with `choice_reason`. `auto` takes the user's default for this kind of structure
    (`mena models default`, or Settings in the web app) when it is usable, else the best-ranked usable model."""
    models = all_models()
    if requested != "auto":
        m = next((x for x in models if x["id"] == requested), None)
        if m is None:
            raise ModelError(f"no model called {requested!r} (see `mena models`)")
        if not _usable(m):
            raise _missing(m)
        if domain not in m.get("domains", []):
            raise ModelError(f"{m.get('title')} does not cover {PLURAL.get(domain, domain)}")
        return {**m, "choice_reason": "You chose this model."}
    pref = settings.default_model(domain)
    m = next((x for x in models if x["id"] == pref), None) if pref else None
    if m is not None and domain in m.get("domains", []) and _usable(m):
        return {**m, "choice_reason": f"Your default model for {PLURAL[domain]} (Settings)."}
    usable = [m for m in models if _usable(m) and domain in m.get("domains", [])]
    if not usable:
        raise ModelError(f"no installed model covers {PLURAL.get(domain, domain)}. "
                         "Run `mena models download` or add one with `mena models add NAME PATH`.")
    best = sorted(usable, key=lambda m: m.get("rank", {}).get(domain, 99))[0]
    note = f" Your default ({pref}) is not installed." if pref else ""
    return {**best, "choice_reason": f"Recommended for {PLURAL[domain]}; change it in Settings.{note}"}


def ensure_file(model: dict, quiet: bool = False) -> dict:
    """Download the model's weights if they are missing and downloadable. Returns the model marked available."""
    if model["available"]:
        return model
    if not model.get("downloadable"):
        raise _missing(model)
    try:
        path = weights.download(model, quiet=quiet)
    except weights.DownloadError as exc:
        raise ModelError(str(exc)) from None
    return {**model, "path": str(path), "available": True, "downloadable": False}


def remove(model_id: str) -> bool:
    """Remove a model you added (from ~/.mena/models.json). Shipped models cannot be removed; their files can be deleted."""
    f = user_file()
    rows = _read(f)
    keep = [e for e in rows if e.get("id") != model_id]
    if len(keep) == len(rows):
        return False
    f.write_text(json.dumps({"models": keep}, indent=1), encoding="utf-8")
    return True


def set_default(domain: str, model_id: str | None) -> dict:
    check_default(domain, model_id)
    try:
        return settings.set_default_model(domain, model_id)
    except ValueError as exc:
        raise ModelError(str(exc)) from None


def check_default(domain: str, model_id: str | None) -> None:
    """Raise ModelError unless `model_id` (or auto) can be the default model for `domain`."""
    if domain not in settings.DOMAINS:
        raise ModelError(f"unknown structure kind {domain!r}; choose {', '.join(settings.DOMAINS)}")
    if model_id not in (None, "", "auto"):
        m = next((x for x in all_models() if x["id"] == model_id), None)
        if m is None:
            raise ModelError(f"no model called {model_id!r} (see `mena models`)")
        if domain not in m.get("domains", []):
            raise ModelError(f"{m.get('title')} does not cover {PLURAL.get(domain, domain)}")
