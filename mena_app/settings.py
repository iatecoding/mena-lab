"""Per-user preferences in ~/.mena/settings.json (or $MENA_HOME). No torch.

Only one preference today: the default model for each kind of structure, used when the model is "auto". An empty entry
means automatic (the model ranked best for that domain in models.json).
"""

from __future__ import annotations

import json

from mena_app import paths

DOMAINS = ("molecule", "crystal", "slab")


def _file():
    return paths.data_dir() / "settings.json"


def load() -> dict:
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    raw = data.get("default_model") if isinstance(data, dict) else None
    defaults = {d: raw[d] for d in DOMAINS if isinstance(raw, dict) and isinstance(raw.get(d), str) and raw[d] not in ("", "auto")}
    return {"default_model": defaults}


def default_model(domain: str) -> str | None:
    return load()["default_model"].get(domain)


def set_default_model(domain: str, model_id: str | None) -> dict:
    """Set (or with None / "auto" clear) the default model of one domain. The caller checks the model exists."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown structure kind {domain!r}; choose {', '.join(DOMAINS)}")
    cur = load()
    if model_id in (None, "", "auto"):
        cur["default_model"].pop(domain, None)
    else:
        cur["default_model"][domain] = model_id
    _file().write_text(json.dumps(cur, indent=1), encoding="utf-8")
    return cur
