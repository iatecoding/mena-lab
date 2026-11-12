"""Decides how a job runs: model level of theory, device, precision, memory mode, neighbour skin, compilation. Pure: no torch, no I/O.

Every choice comes with a reason in plain words, shown before the run and stored with the result. The rules follow
measured timings, not fixed atom counts: the faster device is the one with the lower
`cold + calls × warm`, because a GPU's first call costs 150-800 ms and only long jobs earn that back.
"""

from __future__ import annotations

import math

from mena_app import costmodel

RESERVE_GB = 3.0
F64_TASKS = {"vibrations", "phonons"}
MAX_F64_SLOWDOWN = 3.0     # above this, a large supercell stays in float32 on the GPU


def estimated_calls(spec: dict, analysis: dict) -> int:
    t, p, n = spec["task"], spec["params"], analysis["n_atoms"]
    if t == "singlepoint":
        return 1
    if t == "relax":
        return min(p["max_steps"], 150)
    if t == "md":
        return p["steps"]
    if t == "vibrations":
        return 6 * n + 1 + (60 if p["prerelax"] else 0)
    if t == "phonons":  # displaced supercells: symmetry-irreducible ones only (6N + 1 without symmetry)
        return analysis.get("phonon_evaluations", 6 * n + 1) + (60 if p["prerelax"] else 0)
    if t == "neb":
        return p["images"] * min(p["max_steps"], 100)
    if t == "benchmark":
        steps = p["steps"]
        if p.get("mode") == "visual_md" and steps == 5:
            steps = 22 if analysis["domain"] == "molecule" else 28
        return steps + 1
    return 1


MD_MIN_ATOMS = 64
PHONON_MIN_WIDTH_A = 9.0


def phonon_supercell(spec: dict, analysis: dict) -> int:
    """The requested supercell, or for 0 the smallest ODD repeat at least 9 Å across. ASE's finite-displacement phonons
    centre the reference cell only in odd supercells; silicon showed -0.7 / -0.4 THz artefacts at 2 and 4 and none at 3."""
    k = spec["params"].get("supercell", 0)
    if k:
        return k
    k = max(1, math.ceil(PHONON_MIN_WIDTH_A / min(analysis.get("widths") or [PHONON_MIN_WIDTH_A]) - 1e-9))
    return k if k % 2 else k + 1


def md_repeat(analysis: dict) -> int:
    """Repeat of a periodic cell per direction so dynamics has at least MD_MIN_ATOMS atoms (thermostats need many atoms)."""
    if analysis["domain"] != "crystal" or analysis["n_atoms"] >= MD_MIN_ATOMS:
        return 1
    return math.ceil((MD_MIN_ATOMS / analysis["n_atoms"]) ** (1 / 3))


def working_atoms(spec: dict, analysis: dict) -> tuple[int, int]:
    """(atoms, neighbour pairs) of the largest system the model will really see: a phonon job runs on the supercell."""
    if spec["task"] == "phonons":
        k = phonon_supercell(spec, analysis) ** 3
        return analysis["n_atoms"] * k, analysis["edges"] * k
    if spec["task"] == "benchmark":
        k = max(1, spec["params"]["max_atoms"] // max(analysis["n_atoms"], 1))
        n_tot = analysis["n_atoms"] * k
        edges = analysis["edges"] * k
        if analysis["domain"] == "molecule" and k > 1:
            edges = max(edges, n_tot * 28)
        return n_tot, edges
    if spec["task"] == "md":
        k = md_repeat(analysis) ** 3
        return analysis["n_atoms"] * k, analysis["edges"] * k
    return analysis["n_atoms"], analysis["edges"]


MOLECULAR_LOTS = {"omol_wb97mv", "openrxn_wb97xd", "spice_wb97md3"}
LOT_DOMAIN_TEXT = {"molecule": "molecules", "crystal": "crystals", "slab": "surfaces"}


def canonical_lot(name: str, lots: list[str]) -> str | None:
    """The model's spelling of a level of theory, matched without regard to case (MP_PBE -> mp_pbe)."""
    return next((x for x in lots if x.lower() == str(name).strip().lower()), None)


def check_lot_domain(lot: str, domain: str) -> None:
    if lot == "auto":
        return
    if domain == "molecule" and lot not in MOLECULAR_LOTS:
        raise RoutingError(f"{lot} is a level of theory for periodic materials; molecules use omol_wb97mv (or auto).")
    if domain != "molecule" and lot in MOLECULAR_LOTS:
        raise RoutingError(f"{lot} is a level of theory for molecules; {LOT_DOMAIN_TEXT[domain]} need a periodic one (or auto).")


def _lot(requested: str, domain: str, model: dict | None = None, analysis: dict | None = None) -> tuple[str, str]:
    if requested != "auto":
        return requested, "You chose this level of theory."
    file_lot = (analysis or {}).get("file_lot")
    if file_lot:
        return file_lot, f"Set in the structure file (lot={file_lot})."
    if domain == "molecule":
        return "auto", "Molecule: ωB97M-V, the level of the OMol25 data."
    lots = (model or {}).get("lots") or []
    if domain == "slab" and "oc20_rpbe" in lots:
        if "oc22_pbeU" in lots and (analysis or {}).get("hubbard_u") is True:
            return "oc22_pbeU", "Oxide surface: PBE+U (OC22 oxide catalysis)."
        return "oc20_rpbe", "Surface slab: RPBE (OC20 heterogeneous catalysis), avoiding PBE CO/adsorbate overbinding."
    return "auto", "Periodic: PBE, with +U chosen from the elements as in Materials Project."


def route(spec: dict, analysis: dict, devices: list[str], free_gb: float | None, cal: dict, model: dict) -> dict:
    rt, task = spec["runtime"], spec["task"]
    avail = [d["id"] if isinstance(d, dict) else d for d in devices]
    if rt["device"] != "auto" and rt["device"] not in avail:
        raise RoutingError(f"Device {rt['device']!r} is not available on this computer ({', '.join(avail) or 'none'}).")
    lots = model.get("lots", [])
    for asked, where in ((rt["lot"], ""), ((analysis.get("file_lot") if rt["lot"] == "auto" else None), " (named in the structure file)")):
        if asked and asked != "auto":
            canon = canonical_lot(asked, lots)
            if canon is None:
                raise RoutingError(f"{model.get('title', model['id'])} does not support level of theory {asked!r}{where}. "
                                   f"It supports: {', '.join(lots)}.")
            check_lot_domain(canon, analysis["domain"])
            if where:
                analysis = {**analysis, "file_lot": canon}
            else:
                rt = {**rt, "lot": canon}
    can_f64 = "float64" in model.get("dtypes", ["float32"])
    can_low = bool(model.get("low_memory"))
    backend = model.get("backend", "lam")
    domain = analysis["domain"]
    n, edges = working_atoms(spec, analysis)
    calls = estimated_calls(spec, analysis)
    reasons: dict[str, str] = {}
    lot, reasons["lot"] = _lot(rt["lot"], domain, model, analysis)

    want_f64 = can_f64 and (rt["dtype"] == "float64" or (rt["dtype"] == "auto" and task in F64_TASKS))
    options = []
    for dev in ([rt["device"]] if rt["device"] != "auto" else avail):
        c = costmodel.curve(cal, domain, dev, edges, backend)
        options.append({"device": dev, "seconds": None if c is None else costmodel.job_seconds(c, calls), "curve": c})
    known = [o for o in options if o["seconds"] is not None]
    uncal_cuda = next((o for o in options if o["device"] == "cuda" and o["seconds"] is None), None)
    if rt["device"] != "auto":
        pick = options[0]
        reasons["device"] = "You chose this device."
    elif not known or (uncal_cuda is not None and rt["dtype"] != "float64"):
        pool = ([o for o in options if o["device"] != "mps"] or options) if rt["dtype"] == "float64" else options
        by_dev = {o["device"]: o for o in pool}
        if "cuda" in by_dev:
            pick = by_dev["cuda"]
            reasons["device"] = "Selected CUDA GPU (run 'mena calibrate' to enable speed and memory estimates for this machine)."
        elif "mps" in by_dev and not want_f64 and (calls >= 5 or n >= 32):
            pick = by_dev["mps"]
            reasons["device"] = "Selected Apple GPU (run 'mena calibrate' to enable speed and memory estimates for this machine)."
        else:
            pick = by_dev.get("cpu", pool[0])
            if len(avail) == 1:
                reasons["device"] = "The only device available (run 'mena calibrate' to enable speed and memory estimates for this machine)."
            else:
                reasons["device"] = f"Selected {pick['device'].upper()} for this short or double-precision job (run 'mena calibrate' to enable speed and memory estimates for this machine)."
    else:
        pool = known
        if rt["dtype"] == "float64":
            pool = [o for o in known if o["device"] != "mps"] or known
        elif want_f64:
            cap = [o for o in known if o["device"] != "mps"]
            best = min(o["seconds"] for o in known)
            if cap and min(o["seconds"] for o in cap) <= MAX_F64_SLOWDOWN * best:
                pool = cap
        pick = min(pool, key=lambda o: o["seconds"])
        others = [o for o in known if o is not pick]
        if others:
            other = min(others, key=lambda o: o["seconds"])
            reasons["device"] = f"Expected {_dur(pick['seconds'])} here against {_dur(other['seconds'])} on {other['device'].upper()} for {calls:,} model call{'s' if calls != 1 else ''}."
        elif len(avail) == 1:
            reasons["device"] = "The only device available."
        else:
            reasons["device"] = f"Measured {_dur(pick['seconds'])} on {pick['device'].upper()} (run 'mena calibrate' to compare all devices)."
    device = pick["device"]

    if rt["dtype"] == "float64" and not can_f64:
        raise RoutingError(f"{model.get('title')} runs in single precision only.")
    if rt["dtype"] != "auto":
        dtype, reasons["dtype"] = rt["dtype"], "You chose this precision."
    elif not can_f64:
        dtype, reasons["dtype"] = "float32", "This model runs in single precision only."
    elif task in F64_TASKS and device != "mps":
        dtype, reasons["dtype"] = "float64", "Force constants come from small force differences. Double precision adds about 15% to the time here."
    elif task in F64_TASKS:
        dtype, reasons["dtype"] = "float32", "This supercell would be much slower in double precision on the CPU. Single precision agreed to 1 part in 100,000 in our checks."
    else:
        dtype, reasons["dtype"] = "float32", "Single precision is accurate enough here and uses half the memory."
    if dtype == "float64" and device == "mps":
        raise RoutingError("A Mac GPU cannot run double precision. Choose float32 or the CPU.")

    c = pick["curve"]
    if rt["low_memory"] == "on" and not can_low:
        raise RoutingError(f"{model.get('title')} has no low-memory mode.")
    low = rt["low_memory"] == "on"
    need = None if c is None else costmodel.peak_gb(c, dtype, low)
    if rt["low_memory"] == "auto":
        reasons["low_memory"] = "Not needed: the job fits comfortably."
        if can_low and need is not None and free_gb and need > 0.5 * (free_gb - RESERVE_GB):
            low, need = True, costmodel.peak_gb(c, dtype, True)
            reasons["low_memory"] = "Turned on so the job fits in free memory. It costs about 15-30% more time and gives the same forces."
        elif can_low and need is None and device == "cpu" and n >= 1500:
            low = True
            reasons["low_memory"] = "Turned on automatically for large CPU systems (≥ 1,500 atoms) to keep memory around 0.16 MB/atom."
    else:
        reasons["low_memory"] = "You chose this."
    refuse_reserve = min(RESERVE_GB, 0.35 * free_gb) if free_gb else RESERVE_GB
    level = "unknown" if need is None or not free_gb else ("refuse" if need > free_gb - refuse_reserve else ("warn" if need > 0.5 * free_gb else "ok"))
    dense_molecule = domain == "molecule" and n < (64 if backend == "lam4" else 512)
    # NEB images jump between calls: a skin only adds pairs. Relaxation steps are small (FIRE, <= 0.2 A): a 0.5 A skin
    # reuses the list for most steps; results are identical (pairs beyond the cutoff weigh exactly zero)
    skin = 1.0 if task == "md" and not dense_molecule else (0.5 if task == "relax" and not dense_molecule else 0.0)
    compile_on, reasons["compile"] = _compile(rt.get("compile", "auto"), backend, device, dense_molecule, low, pick["seconds"], calls)
    seconds = pick["seconds"]
    if compile_on and seconds is not None:
        seconds = costmodel.compiled_seconds(seconds, device)
    if task == "md" and md_repeat(analysis) > 1:
        r = md_repeat(analysis)
        reasons["size"] = f"The cell is repeated {r}×{r}×{r} ({n} atoms): a thermostat needs many atoms to define a temperature."
    return {"model": model["id"], "backend": backend, "device": device, "dtype": dtype, "low_memory": low, "skin": skin, "compile": compile_on, "lot": lot, "threads": rt["threads"], "calls": calls, "atoms": n, "edges": edges,
            "seconds": seconds, "peak_gb": None if need is None else round(need, 2), "free_gb": free_gb, "memory_level": level,
            "reasons": reasons, "cost_source": cal.get("source")}


GPU_COMPILE_MIN_CALLS = 200  # without time measurements: compile GPU jobs at least this long


def _compile(requested: str, backend: str, device: str, dense_molecule: bool, low: bool, eager_s: float | None, calls: int) -> tuple[bool, str]:
    """(compile?, reason). Compile when the time it saves beats its one-time warm-up."""
    if backend not in ("lam", "lam4"):
        return False, "This model has no compiled mode."
    if requested == "off":
        return False, "You turned this off."
    if low:
        return False, "Low-memory mode runs uncompiled."
    if requested == "on":
        return True, "You turned this on."
    if dense_molecule:
        return False, "Small molecules use a path that compilation does not speed up."
    c = costmodel.COMPILE.get(device)
    if c is None:
        return False, "No compiled mode for this device."
    if eager_s is None:
        on = device != "cpu" and calls >= GPU_COMPILE_MIN_CALLS
        return on, (f"A long job ({calls:,} calls): worth a one-time warm-up of about {c['warmup_s']:.0f} s." if on
                    else "A short job: the one-time warm-up would not pay back.")
    saved = eager_s - costmodel.compiled_seconds(eager_s, device)
    if saved > 0:
        return True, f"Saves about {_dur(saved)} after a one-time warm-up of about {c['warmup_s']:.0f} s."
    return False, f"A short job: the one-time warm-up (about {c['warmup_s']:.0f} s) is longer than the time it would save."


class PlanError(ValueError):
    pass


class RoutingError(PlanError):
    pass


def _dur(s: float) -> str:
    if s < 0.1:
        return "under 0.1 s"
    if s < 60:
        return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"
    return f"{s / 60:.0f} min" if s < 3600 else f"{s / 3600:.1f} h"
