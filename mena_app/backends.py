"""Worker side only: build an ASE calculator for a model entry and a routing decision. This is the one file that knows
model code. A new architecture needs one function here and one line in data/models.json; nothing else changes.

Each builder imports from the `lam` / `lam4` package that is first on sys.path. The worker puts the model's `code_root` there
(if the model entry has one), so a checkpoint that needs a different version of the code can load with its own copy.
"""

from __future__ import annotations

import copy
import inspect


def _set_mps_fraction() -> None:
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "set_per_process_memory_fraction"):
            torch.mps.set_per_process_memory_fraction(1.0)  # an overflow becomes an error instead of swapping
    except Exception:  # noqa: BLE001
        pass


def _lam(model: dict, d: dict):
    from lam.models.mena_lam import MenaLAMCalculator

    dev = str(d["device"])
    if dev.startswith("mps"):
        _set_mps_fraction()
    kw = {"device": dev, "dtype": d["dtype"], "lot": d["lot"], "skin": d["skin"], "low_memory": d["low_memory"],
          "compile": bool(d.get("compile", False))}
    sig = inspect.signature(MenaLAMCalculator.__init__).parameters  # older trees take fewer options
    if d.get("precision") is not None and "precision" in sig:
        kw["precision"] = d["precision"]
    return MenaLAMCalculator.from_checkpoint(model["path"], **{k: v for k, v in kw.items() if k in sig or k == "device"})


def _lam4(model: dict, d: dict):
    from lam4.calc import MenaCalculator

    dev = str(d["device"])
    if dev.startswith("mps"):
        _set_mps_fraction()
    comp_raw = d.get("compile", False)
    comp = comp_raw if isinstance(comp_raw, str) else bool(comp_raw)
    kw = {"device": dev, "dtype": d["dtype"], "lot": d["lot"], "skin": d["skin"], "low_memory": d["low_memory"],
          "compile": comp}
    sig = inspect.signature(MenaCalculator.__init__).parameters
    for opt_k in ("precision", "edge_mode", "add_dispersion", "u_flag", "weights", "compute_stress"):
        val = d.get(opt_k)
        if val is not None and opt_k in sig:
            kw[opt_k] = val
    return MenaCalculator.from_checkpoint(model["path"], **kw)


BUILDERS = {"lam": _lam, "lam4": _lam4}


def build(model: dict, decision: dict):
    backend = model.get("backend", "lam")
    if backend not in BUILDERS:
        raise ValueError(f"unknown model backend {backend!r}; known: {', '.join(BUILDERS)}")
    return BUILDERS[backend](model, decision)


def clone_calculator(calc):
    """Shallow-clone an ASE calculator so multiple images share one loaded nn.Module without thrashing ASE caches."""
    c = copy.copy(calc)
    c.atoms = None
    c.results = {}
    if getattr(c, "parameters", None) is not None:
        c.parameters = copy.copy(c.parameters)
    is_lam4 = hasattr(c, "_active_csr_plan")
    for attr in ("_nl_cache", "_verlet_cache", "_static_cache", "_skin_cache"):
        val = getattr(c, attr, None)
        if is_lam4 and attr in ("_nl_cache", "_static_cache"):
            setattr(c, attr, None)
        elif isinstance(val, dict):
            setattr(c, attr, {})
        elif hasattr(c, attr):
            setattr(c, attr, None)
    for attr in ("_last_state", "_last_cell_t", "_last_edge_shift_cart", "_last_info_state", "_last_info_sig", "_active_csr_plan", "_lot_cache"):
        if hasattr(c, attr):
            setattr(c, attr, None)
    if "calculate" in getattr(c, "__dict__", {}):
        del c.calculate
    return c


def has_stress(calc) -> bool:
    return "stress" in getattr(calc, "implemented_properties", [])
