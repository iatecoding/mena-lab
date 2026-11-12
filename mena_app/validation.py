"""Checks a job request against the catalogue and returns a clean spec. Nothing from a request is executed or used as a path."""

from __future__ import annotations

import math

from mena_app import catalog

FORMATS = {"xyz", "extxyz", "cif", "vasp", "builtin", "smiles"}
MAX_TEXT = 8_000_000


class ValidationError(ValueError):
    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


def _value(p: dict, v):
    name, kind = p["label"], p["type"]
    if v is None or v == "":
        if p.get("nullable"):
            return None
        raise ValidationError(f"{name}: a value is needed", p["name"])
    if kind == "bool":
        if not isinstance(v, bool):
            raise ValidationError(f"{name}: expected true or false", p["name"])
        return v
    if kind == "choice":
        if not isinstance(v, str):
            raise ValidationError(f"{name}: expected a string choice", p["name"])
        if "options" in p and v not in p["options"]:
            raise ValidationError(f"{name}: must be one of {', '.join(p['options'])}", p["name"])
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValidationError(f"{name}: expected a number", p["name"])
    if kind == "int":
        if float(v) != int(v):
            raise ValidationError(f"{name}: expected a whole number", p["name"])
        v = int(v)
    else:
        v = float(v)
    if "min" in p and v < p["min"]:
        raise ValidationError(f"{name}: must be at least {p['min']}", p["name"])
    if "max" in p and v > p["max"]:
        raise ValidationError(f"{name}: must be at most {p['max']}", p["name"])
    return v


def _group(schema: list[dict], raw, what: str, use_defaults: bool = True) -> dict:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValidationError(f"{what} must be an object")
    extra = set(raw) - {p["name"] for p in schema}
    if extra:
        raise ValidationError(f"unknown {what}: {', '.join(sorted(extra))}")
    return {p["name"]: _value(p, raw.get(p["name"], p["default"]) if use_defaults else raw.get(p["name"])) for p in schema}


def _structure(s, field: str) -> dict:
    if not isinstance(s, dict) or s.get("format") not in FORMATS:
        raise ValidationError(f"choose a structure ({', '.join(sorted(FORMATS))})", field)
    text = s.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise ValidationError("the structure is empty or larger than 8 MB", field)
    if s["format"] in ("smiles", "builtin"):
        text = text.strip()
        if s["format"] == "smiles" and text.lower().startswith("smiles:"):
            text = text[7:].strip()
        if not text:
            raise ValidationError("the structure is empty or larger than 8 MB", field)
    out = {"format": s["format"], "text": text, "name": str(s.get("name") or "structure")[:80]}
    if isinstance(s.get("kind"), str) and s["kind"] in {"path", "builtin", "smiles"}:
        out["kind"] = s["kind"]
    if isinstance(s.get("source"), str) and s["source"]:
        out["source"] = s["source"][:240]
    return out


def validate(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValidationError("the request must be an object")
    t = catalog.task(raw.get("task"))
    if t is None:
        raise ValidationError("unknown task", "task")
    spec = {"task": t["id"], "structure": _structure(raw.get("structure"), "structure"),
            "params": _group(t["params"], raw.get("params"), "setting"),
            "runtime": _group(catalog.RUNTIME, raw.get("runtime"), "option"),
            "system": _group(catalog.SYSTEM, raw.get("system"), "system value", use_defaults=False)}
    if t.get("needs_second"):
        spec["structure2"] = _structure(raw.get("structure2"), "structure2")
    return spec
