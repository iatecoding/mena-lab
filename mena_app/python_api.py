"""One-line access to a Mena model as an ASE calculator, for scripts and notebooks.

    from mena_app import calculator
    atoms.calc = calculator("lam5-v1")                 # molecules
    atoms.calc = calculator("lam-400k", device="cuda") # crystals

The weights download on first use, exactly as for the command line and the web app.
"""

from __future__ import annotations


def calculator(model: str = "lam5-v1", device: str = "cpu", dtype: str = "float32", lot: str = "auto",
               skin: float = 0.0, low_memory: bool = False, compile: bool = False):
    from mena_app import backends, models

    entry = next((m for m in models.all_models() if m["id"] == model), None)
    if entry is None:
        raise models.ModelError(f"no model called {model!r} (see `mena models`)")
    if lot != "auto":
        from mena_app.routing import canonical_lot

        canon = canonical_lot(lot, entry.get("lots", []))
        if canon is None:
            raise models.ModelError(f"{entry.get('title', model)} does not support level of theory {lot!r}; "
                                    f"it supports: {', '.join(entry.get('lots', []))}")
        lot = canon
    entry = models.ensure_file(entry)
    return backends.build(entry, {"device": device, "dtype": dtype, "lot": lot, "skin": skin,
                                  "low_memory": low_memory, "compile": compile})
