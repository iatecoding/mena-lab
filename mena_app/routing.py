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


MOLECULAR_LOTS = {
    "omol_wb97mv", "openrxn_wb97xd", "spice_wb97md3",
    "opoly26_wb97mv", "mad16_r2scan_mol", "t1x_wb97x", "rgd1_b3lyp",
    "qcml_pbe0", "tmqm_tpssh", "splinter_sapt0", "splinter_aug",
}
LOT_ALIASES = {
    "mp_pbe52": "mp_pbe", "mp_pbe52u": "mp_pbeU",
    "omat_pbe54": "omat_pbe", "omat_pbe54u": "omat_pbeU",
    "matpes_pbe64": "matpes_pbe", "r2scan": "matpes_r2scan",
    "rpbe": "oc20_rpbe", "oc20": "oc20_rpbe", "oc25": "oc25_rpbe",
    "mad": "mad16_r2scan", "mad16": "mad16_r2scan",
    "omol": "omol_wb97mv", "omol25": "omol_wb97mv", "wb97mv": "omol_wb97mv",
    "opoly": "opoly26_wb97mv", "opoly26": "opoly26_wb97mv",
    "spice": "spice_wb97md3", "omc": "omc25_pbe", "omc25": "omc25_pbe",
}
LOT_DOMAIN_TEXT = {"molecule": "molecules", "crystal": "crystals", "slab": "surfaces"}


def canonical_lot(name: str, lots: list[str]) -> str | None:
    """The model's spelling of a level of theory, resolving case and common PAW/functional aliases."""
    low = str(name).strip().lower()
    hit = next((x for x in lots if x.lower() == low), None)
    if hit is not None:
        return hit
    aliased = LOT_ALIASES.get(low)
    if aliased is not None:
        return next((x for x in lots if x.lower() == aliased.lower()), None)
    return None


def check_lot_domain(lot: str, domain: str) -> None:
    if lot == "auto":
        return
    if domain == "molecule" and lot not in MOLECULAR_LOTS:
        raise RoutingError(f"{lot} is a level of theory for periodic materials; molecules use omol_wb97mv (or auto).")
    if domain == "slab" and lot in MOLECULAR_LOTS:
        raise RoutingError(f"{lot} requires either an isolated molecule or a 3D periodic cell (not a 2D slab); surfaces need a periodic level of theory (or auto).")


def _lot(requested: str, domain: str, model: dict | None = None, analysis: dict | None = None) -> tuple[str, str]:
    if requested != "auto":
        return requested, "You chose this level of theory."
    file_lot = (analysis or {}).get("file_lot")
    if file_lot:
        return file_lot, f"Set in the structure file (lot={file_lot})."
    if domain == "molecule":
        return "auto", "Molecule: ωB97M-V, the level of the OMol25 data."
    lots = (model or {}).get("lots") or []
    if domain == "slab" and ("oc20_rpbe" in lots or "oc25_rpbe" in lots):
        if (analysis or {}).get("hubbard_u") is True:
            if "oc22_pbeU" in lots:
                return "oc22_pbeU", "Oxide surface: PBE+U (OC22 oxide catalysis)."
            if "omat_pbeU" in lots:
                return "omat_pbeU", "Oxide surface: PBE+U (OMat24 PBE_54)."
            if "mp_pbeU" in lots:
                return "mp_pbeU", "Oxide surface: PBE+U (Materials Project PBE_52)."
        if "oc25_rpbe" in lots and "oc20_rpbe" not in lots:
            return "oc25_rpbe", "Surface slab: RPBE (OC25 catalysis), avoiding PBE CO/adsorbate overbinding."
        return "oc20_rpbe", "Surface slab: RPBE (OC20 heterogeneous catalysis), avoiding PBE CO/adsorbate overbinding."
    return "auto", "Periodic: PBE, with +U chosen from the elements as in Materials Project."


def _is_cuda_device(dev: str) -> bool:
    return dev == "cuda" or (dev.startswith("cuda:") and dev[5:].isdigit())


def _device_family(dev: str) -> str:
    return "cuda" if _is_cuda_device(dev) else dev


def _resolve_requested_device(req: str, avail: list[str], dev_free: dict[str, float | None]) -> str | None:
    if req in avail:
        return req
    if req == "cuda":
        cuda_avail = [d for d in avail if _is_cuda_device(d)]
        if cuda_avail:
            return max(cuda_avail, key=lambda d: dev_free.get(d) if dev_free.get(d) is not None else -1.0)
    if req == "cuda:0" and "cuda" in avail:
        return "cuda:0"
    return None


def _build_device_free_map(devices: list, free_gb: float | dict | None, device_free_gb: dict[str, float | None] | None) -> tuple[dict[str, float | None], bool]:
    """Return (device -> free_gb map, has_explicit_per_device_map)."""
    out: dict[str, float | None] = {}
    has_map = False
    for d in devices:
        if isinstance(d, dict):
            did = d.get("id") or d.get("device")
            if did:
                val = d.get("vram_free_gb") if "vram_free_gb" in d else d.get("free_gb")
                if val is not None or "vram_free_gb" in d or "free_gb" in d:
                    out[did] = val
                    has_map = True
    if isinstance(free_gb, dict):
        out.update(free_gb)
        has_map = True
    if isinstance(device_free_gb, dict):
        out.update(device_free_gb)
        has_map = True
    return out, has_map


def _free_for(dev: str, scalar_free: float | None, dev_free: dict[str, float | None], has_map: bool) -> float | None:
    if dev in dev_free:
        return dev_free[dev]
    if dev == "cuda" and "cuda:0" in dev_free:
        return dev_free["cuda:0"]
    if dev == "cuda:0" and "cuda" in dev_free:
        return dev_free["cuda"]
    if has_map and _is_cuda_device(dev):
        return None
    return scalar_free


def route(spec: dict, analysis: dict, devices: list, free_gb: float | dict | None, cal: dict, model: dict,
          *, device_free_gb: dict[str, float | None] | None = None) -> dict:
    rt, task = spec["runtime"], spec["task"]
    avail = [d["id"] if isinstance(d, dict) else d for d in devices]
    dev_free_map, has_dev_free_map = _build_device_free_map(devices, free_gb, device_free_gb)
    scalar_free = None if isinstance(free_gb, dict) else free_gb
    req_device: str | None = None
    if rt["device"] != "auto":
        req_device = _resolve_requested_device(rt["device"], avail, dev_free_map)
        if req_device is None:
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
    prec_req = str(rt.get("precision") or "auto").strip().lower()
    if prec_req == "bf16_nodes":
        raise RoutingError("precision='bf16_nodes' is disabled in mena_app (breaks force conservation, Hessian symmetry, and NVE integration); use 'auto', 'fp32_ieee', or 'exact'.")
    if prec_req not in ("auto", "fp32_ieee", "exact"):
        raise RoutingError(f"Unsupported precision mode {prec_req!r}; expected auto, fp32_ieee, or exact.")
    if rt["dtype"] == "float64" and prec_req == "fp32_ieee":
        raise RoutingError("precision='fp32_ieee' requires single precision (float32) and conflicts with dtype='float64'.")
    if rt["dtype"] == "float32" and prec_req == "exact":
        raise RoutingError("precision='exact' requires double precision (float64) and conflicts with dtype='float32'.")
    explicit_exact = rt["dtype"] == "float64" or prec_req == "exact"
    if explicit_exact and not can_f64:
        raise RoutingError(f"{model.get('title')} runs in single precision only.")

    want_f64 = can_f64 and (explicit_exact or (rt["dtype"] == "auto" and prec_req == "auto" and task in F64_TASKS))
    options = []
    for dev in ([req_device] if req_device is not None else avail):
        c = costmodel.curve(cal, domain, _device_family(dev), edges, backend)
        options.append({
            "device": dev,
            "seconds": None if c is None else costmodel.job_seconds(c, calls),
            "curve": c,
            "free_gb": _free_for(dev, scalar_free, dev_free_map, has_dev_free_map),
        })
    known = [o for o in options if o["seconds"] is not None]
    cuda_opts = [o for o in options if _is_cuda_device(o["device"])]
    uncal_cuda = next((o for o in cuda_opts if o["seconds"] is None), None)
    if req_device is not None:
        pick = options[0]
        reasons["device"] = "You chose this device."
    elif not known or (uncal_cuda is not None and not explicit_exact):
        pool = ([o for o in options if o["device"] != "mps"] or options) if explicit_exact else options
        cuda_pool = [o for o in pool if _is_cuda_device(o["device"])]
        by_dev = {o["device"]: o for o in pool}
        if cuda_pool:
            pick = max(cuda_pool, key=lambda o: o["free_gb"] if o["free_gb"] is not None else -1.0)
            reasons["device"] = "Selected CUDA GPU (run 'mena calibrate' to enable speed and memory estimates for this machine)."
        elif "mps" in by_dev and not want_f64 and (calls >= 5 or n >= 32):
            pick = by_dev["mps"]
            reasons["device"] = "Selected Apple GPU (run 'mena calibrate' to enable speed and memory estimates for this machine)."
        else:
            pick = by_dev.get("cpu", pool[0])
            if len(avail) == 1:
                reasons["device"] = "The only device available (run 'mena calibrate' to enable speed and memory estimates for this machine)."
            elif explicit_exact and "mps" in avail and pick["device"] != "mps":
                reasons["device"] = f"Selected {pick['device'].upper()} because Apple GPU does not support double precision (run 'mena calibrate' to enable speed and memory estimates for this machine)."
            else:
                reasons["device"] = f"Selected {pick['device'].upper()} for this short or double-precision job (run 'mena calibrate' to enable speed and memory estimates for this machine)."
    else:
        pool = known
        mps_excluded = False
        if explicit_exact:
            non_mps_known = [o for o in known if o["device"] != "mps"]
            mps_excluded = len(non_mps_known) < len(known)
            if non_mps_known:
                pool = non_mps_known
            else:
                non_mps_opts = [o for o in options if o["device"] != "mps"]
                pool = non_mps_opts or known
        elif want_f64:
            cap = [o for o in known if o["device"] != "mps"]
            best = min(o["seconds"] for o in known)
            if cap and min(o["seconds"] for o in cap) <= MAX_F64_SLOWDOWN * best:
                pool = cap
        pick = min(
            pool,
            key=lambda o: (
                o["seconds"] if o["seconds"] is not None else float("inf"),
                -(o["free_gb"] if o["free_gb"] is not None else -1.0),
            ),
        )
        others = [o for o in known if o is not pick]
        if pick["seconds"] is None:
            reasons["device"] = f"Selected {pick['device'].upper()} because Apple GPU does not support double precision (run 'mena calibrate' to enable speed and memory estimates for this machine)."
        elif others:
            other = min(others, key=lambda o: o["seconds"])
            note = " (Apple GPU excluded: double precision requires CPU or CUDA)." if (mps_excluded and pick["device"] != "mps") else "."
            reasons["device"] = f"Expected {_dur(pick['seconds'])} here against {_dur(other['seconds'])} on {other['device'].upper()} for {calls:,} model call{'s' if calls != 1 else ''}{note}"
        elif len(avail) == 1:
            reasons["device"] = "The only device available."
        else:
            reasons["device"] = f"Measured {_dur(pick['seconds'])} on {pick['device'].upper()} (run 'mena calibrate' to compare all devices)."
    device = pick["device"]
    dev_free_gb = pick["free_gb"]
    dev_family = _device_family(device)

    if explicit_exact:
        dtype, precision = "float64", "exact"
        reasons["dtype"] = "You chose double precision." if rt["dtype"] == "float64" else "Double precision (float64) required by precision='exact'."
        reasons["precision"] = _precision_reason("exact", explicit=(prec_req == "exact"))
    elif rt["dtype"] != "auto":
        dtype = rt["dtype"]
        precision = prec_req
        reasons["dtype"] = "You chose this precision."
        reasons["precision"] = _precision_reason(precision, explicit=(prec_req != "auto"), backend=backend)
    elif not can_f64:
        dtype, precision = "float32", prec_req
        reasons["dtype"] = "This model runs in single precision only."
        reasons["precision"] = _precision_reason(precision, explicit=(prec_req != "auto"), backend=backend)
    elif task in F64_TASKS and prec_req == "auto" and _is_cuda_device(device) and backend == "lam4":
        dtype, precision = "float32", "fp32_ieee"
        reasons["dtype"] = "Exact IEEE float32 (TF32 disabled) keeps fused CUDA Dual-CSR kernels active while preserving finite-difference accuracy."
        reasons["precision"] = reasons["dtype"]
    elif task in F64_TASKS and prec_req == "auto" and device != "mps":
        dtype, precision = "float64", "exact"
        reasons["dtype"] = "Force constants come from small force differences. Double precision adds about 15% to the time here."
        reasons["precision"] = "Double precision (FP64 exact) is used automatically for finite-difference force constants off MPS."
    elif task in F64_TASKS and prec_req == "auto":
        dtype, precision = "float32", "auto"
        reasons["dtype"] = "This supercell would be much slower in double precision on the CPU. Single precision agreed to 1 part in 100,000 in our checks."
        reasons["precision"] = "Single precision on Apple GPU avoids a slow CPU fallback."
    else:
        dtype, precision = "float32", prec_req
        reasons["dtype"] = "Single precision is accurate enough here and uses half the memory."
        reasons["precision"] = _precision_reason(precision, explicit=(prec_req != "auto"), backend=backend)
    if dtype == "float64" and device == "mps":
        raise RoutingError("A Mac GPU cannot run double precision. Choose float32 or the CPU.")

    c = pick["curve"]
    if rt["low_memory"] == "on" and not can_low:
        raise RoutingError(f"{model.get('title')} has no low-memory mode.")
    low = rt["low_memory"] == "on"
    need = None if c is None else costmodel.peak_gb(c, dtype, low)
    if rt["low_memory"] == "auto":
        reasons["low_memory"] = "Not needed: the job fits comfortably."
        if can_low and need is not None and dev_free_gb and need > 0.5 * (dev_free_gb - RESERVE_GB):
            low, need = True, costmodel.peak_gb(c, dtype, True)
            reasons["low_memory"] = "Turned on so the job fits in free memory. It costs about 15-30% more time and gives the same forces."
        elif can_low and need is None and dev_free_gb is not None and dev_free_gb < RESERVE_GB + 1.0:
            low = True
            reasons["low_memory"] = ("Turned on because free memory is low and there are no memory measurements for this "
                                     "computer yet (run 'mena calibrate').")
        elif can_low and need is None and device == "cpu" and n >= 1500:
            low = True
            reasons["low_memory"] = "Turned on automatically for large CPU systems (≥ 1,500 atoms) to keep memory around 0.16 MB/atom."
    else:
        reasons["low_memory"] = "You chose this."
    refuse_reserve = min(RESERVE_GB, 0.35 * dev_free_gb) if dev_free_gb else RESERVE_GB
    level = "unknown" if need is None or not dev_free_gb else ("refuse" if need > dev_free_gb - refuse_reserve else ("warn" if need > 0.5 * dev_free_gb else "ok"))
    dense_molecule = domain == "molecule" and n < (64 if backend == "lam4" else 512)
    # NEB images jump between calls: a skin only adds pairs. Relaxation steps are small (FIRE, <= 0.2 A): a 0.5 A skin
    # reuses the list for most steps; results are identical (pairs beyond the cutoff weigh exactly zero)
    auto_skin = 1.0 if task == "md" and not dense_molecule else (0.5 if task == "relax" and not dense_molecule else 0.0)
    skin = float(rt["skin"]) if rt.get("skin") is not None else auto_skin
    compile_on, reasons["compile"] = _compile(rt.get("compile", "auto"), backend, dev_family, dense_molecule, low, pick["seconds"], calls)
    seconds = pick["seconds"]
    if compile_on and seconds is not None:
        seconds = costmodel.compiled_seconds(seconds, dev_family, backend)
    if task == "md" and md_repeat(analysis) > 1:
        r = md_repeat(analysis)
        reasons["size"] = f"The cell is repeated {r}×{r}×{r} ({n} atoms): a thermostat needs many atoms to define a temperature."
    resolved_lot = lot
    if lot == "auto":
        if domain == "molecule":
            resolved_lot = "omol_wb97mv"
        else:
            resolved_lot = "mp_pbeU" if analysis.get("hubbard_u") is True else "mp_pbe"
    dec_out = {
        "model": model["id"], "backend": backend, "device": device, "dtype": dtype, "precision": precision,
        "low_memory": low, "skin": skin, "compile": compile_on, "lot": lot, "resolved_lot": resolved_lot,
        "threads": rt["threads"], "calls": calls,
        "atoms": n, "edges": edges, "seconds": seconds, "peak_gb": None if need is None else round(need, 2),
        "free_gb": dev_free_gb, "memory_level": level, "reasons": reasons, "cost_source": cal.get("source"),
    }
    if rt.get("edge_mode") and rt["edge_mode"] != "auto":
        dec_out["edge_mode"] = str(rt["edge_mode"])
    if rt.get("add_dispersion") in ("on", "off"):
        dec_out["add_dispersion"] = rt["add_dispersion"] == "on"
    if rt.get("u_flag") in ("0", "1", 0, 1):
        dec_out["u_flag"] = int(rt["u_flag"])
    return dec_out


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
    if dense_molecule and backend == "lam":
        return False, "Small molecules use a path that compilation does not speed up."
    c = costmodel.COMPILE.get(device)
    if c is None:
        return False, "No compiled mode for this device."
    if eager_s is None:
        on = device != "cpu" and calls >= GPU_COMPILE_MIN_CALLS
        return on, (f"A long job ({calls:,} calls): worth a one-time warm-up of about {c['warmup_s']:.0f} s." if on
                    else "A short job: the one-time warm-up would not pay back.")
    saved = eager_s - costmodel.compiled_seconds(eager_s, device, backend)
    if saved > 0:
        return True, f"Saves about {_dur(saved)} after a one-time warm-up of about {c['warmup_s']:.0f} s."
    return False, f"A short job: the one-time warm-up (about {c['warmup_s']:.0f} s) is longer than the time it would save."


class PlanError(ValueError):
    pass


class RoutingError(PlanError):
    pass


def _precision_reason(precision: str, explicit: bool = False, backend: str = "lam4") -> str:
    if precision == "exact":
        return "You chose double precision (FP64 exact)." if explicit else "Double precision (FP64 exact)."
    if precision == "fp32_ieee":
        if backend != "lam4":
            return "Strict IEEE FP32 requested (standard float32 used on legacy backend)."
        return "You chose strict IEEE FP32 (TF32 disabled)." if explicit else "Strict IEEE FP32 (TF32 disabled)."
    return "Automatic execution precision for this device and task."


def _dur(s: float) -> str:
    if s < 0.1:
        return "under 0.1 s"
    if s < 60:
        return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"
    return f"{s / 60:.0f} min" if s < 3600 else f"{s / 3600:.1f} h"
