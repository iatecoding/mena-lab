"""From a validated spec to a full plan: structure analysis, model choice, routing. Shared by the CLI and the server so a
job is planned the same way wherever it starts. No torch: device list and free memory come from machine.py."""

from __future__ import annotations

from mena_app import catalog, costmodel, machine, models, routing, structures
from mena_app.routing import PlanError

# Elements with training data (per-element energy references): molecules H-Bi; materials H-Bi and Ac-Pu.
_SUPPORTED_Z = {"molecule": set(range(1, 84)), "crystal": set(range(1, 84)) | set(range(89, 95))}
_SUPPORTED_Z["slab"] = _SUPPORTED_Z["crystal"]
MIN_DISTANCE_A = 0.1


def _check_atoms(atoms, a: dict) -> None:
    bad = sorted({s for s, z in zip(atoms.get_chemical_symbols(), atoms.numbers) if int(z) not in _SUPPORTED_Z[a["domain"]]})
    if bad:
        span = "H to Bi" if a["domain"] == "molecule" else "H to Bi and Ac to Pu"
        raise PlanError(f"The models were not trained on {', '.join(bad)}: {a['domain']}s can contain {span}.")
    if a.get("min_distance") is not None and a["min_distance"] < MIN_DISTANCE_A:
        raise PlanError(f"Two atoms are {a['min_distance']:.3f} Å apart (overlapping). Check the input structure.")


def _lot_hint(spec: dict, a: dict, model: dict) -> str:
    """When the chosen model lacks the requested level of theory, name a model that has it."""
    from mena_app.routing import canonical_lot

    asked = spec["runtime"].get("lot", "auto")
    asked = asked if asked != "auto" else a.get("file_lot")
    if not asked or canonical_lot(asked, model.get("lots", [])):
        return ""
    other = next((m for m in models.all_models() if m["id"] != model["id"] and canonical_lot(asked, m.get("lots", []))
                  and a["domain"] in m.get("domains", []) and (m["available"] or m.get("downloadable"))), None)
    return f" {other['title']} has it: add --model {other['id']} (or choose it under Tune)." if other else ""


def plan(spec: dict) -> dict:
    try:
        atoms = structures.load(spec["structure"])
    except structures.StructureError as exc:
        raise PlanError(str(exc)) from None
    a = structures.analyze(atoms)
    _check_atoms(atoms, a)
    t = catalog.task(spec["task"])
    if a["domain"] not in t["domains"]:
        need = " or ".join(t["domains"])
        raise PlanError(f"{t['title']} needs a {need}; this structure is a {a['domain']}.")
    if a["n_atoms"] < 2 and spec["task"] in ("vibrations", "neb") or (a["domain"] == "molecule" and a["n_atoms"] < 2 and spec["task"] == "md"):
        raise PlanError(f"{t['title']} needs at least two atoms.")
    sys_cfg = spec.setdefault("system", {})
    if a["domain"] == "molecule":
        q = sys_cfg.get("charge")
        if q is None:
            q = a["charge_hint"] if a.get("charge_hint") is not None else 0
        s = sys_cfg.get("spin")
        n_elec = a["electrons"] - int(q)
        if s is None:
            s = a["spin_hint"] if a.get("spin_hint") is not None else (1 if n_elec % 2 == 0 else 2)
        sys_cfg["charge"], sys_cfg["spin"] = int(q), int(s)
        if (a["electrons"] - sys_cfg["charge"]) % 2 != (sys_cfg["spin"] - 1) % 2:
            raise PlanError(f"{a['electrons'] - sys_cfg['charge']} electrons cannot have spin multiplicity {sys_cfg['spin']}. "
                            "Check the total charge and the multiplicity.")
    else:
        if sys_cfg.get("charge") is None:
            sys_cfg["charge"] = 0
        if sys_cfg.get("spin") is None:
            sys_cfg["spin"] = 1
    if spec["task"] == "neb" or "structure2" in spec:
        if "structure2" not in spec:
            raise PlanError("end structure: missing second structure")
        try:
            b = structures.load(spec["structure2"])
        except Exception as exc:
            raise PlanError(f"end structure: {exc}") from exc
        if len(b) != a["n_atoms"] or list(b.numbers) != list(atoms.numbers):
            raise PlanError("The two end structures must have the same atoms in the same order.")
    try:
        model = models.choose(spec["runtime"]["model"], a["domain"])
        m = machine.snapshot()
        decision = routing.route(spec, a, m["devices"], m["available_gb"], costmodel.load(), model)
        decision["reasons"]["model"] = model.get("choice_reason", "")
    except routing.RoutingError as exc:
        raise PlanError(str(exc) + _lot_hint(spec, a, model)) from None
    except models.ModelError as exc:
        raise PlanError(str(exc)) from None
    if decision["memory_level"] == "refuse":
        raise PlanError(f"This job needs about {decision['peak_gb']:.1f} GB but only {m['available_gb']:.1f} GB is free. Use a smaller system or close other programs.")
    public = {k: model.get(k) for k in ("id", "title", "training", "metrics", "backend")}
    return {"analysis": a, "model": model, "model_public": public, "decision": decision, "machine": m}
