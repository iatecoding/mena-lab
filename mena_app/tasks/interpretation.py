"""The "Interpretation" section of a single-point result: what the model's physics heads say about this structure,
in plain words, each with how far it can be trusted. Wording and trust levels come from probes against known chemistry
(conjugation, radicals, ions, bond breaking, compression of crystals).

Input is the dict from `calc.interpretation(atoms)` (lam4 backend). Output: a panel for the web page and the terminal,
extra per-atom columns, extra viewer colourings (`unpaired`, `pauli`) and raw arrays for the JSON download.
"""

from __future__ import annotations

import numpy as np

MAX_ATOMS = 20000  # one extra model call; beyond this the panel is skipped (say so) to keep big jobs fast

TRUST = {
    "trend": "Reliable trend",
    "qualitative": "Qualitative",
    "care": "Use with care",
}


def available(calc) -> bool:
    return callable(getattr(calc, "interpretation", None))


SHORT = {  # one line for the terminal; the web page and result.json carry the full meaning and caveat
    "gap": "ranks molecules by conjugation; an orbital gap, not an optical gap",
    "unpaired": "finds radical sites; relative index, runs away for bonds stretched past ~2.5 Å",
    "charges": "right signs and symmetry; smaller than Mulliken/NPA; ion pairs stay ionic",
    "spin": "smoothed; misses negative spin (spin polarisation)",
    "pauli": "compare between structures; not an energy decomposition",
    "exchange": "physical near equilibrium; wrong for stretched singlet bonds",
}


def _item(key, label, value, unit, digits, trust, meaning, caveat, detail=""):
    return {"key": key, "label": label, "value": value, "unit": unit, "digits": digits, "trust": trust,
            "trust_label": TRUST[trust], "meaning": meaning, "caveat": caveat, "short": SHORT[key], "detail": detail}


def _top(arr, sym, k=3, sign=1.0, fmt="{:+.2f}"):
    if arr is None or len(arr) == 0:
        return ""
    order = np.argsort(-sign * np.asarray(arr))[:k]
    return ", ".join(f"{sym[i]}{i + 1} {fmt.format(arr[i])}" for i in order)


def build(interp: dict, atoms, model_title: str) -> dict:
    """{'panel', 'columns', 'frame', 'raw'} for a single-point result."""
    sym = atoms.get_chemical_symbols()
    n = len(sym)
    heads = interp.get("heads", {})
    periodic = bool(interp.get("periodic"))
    open_shell = int(interp.get("multiplicity", 1)) > 1
    items, notes = [], []
    cols: dict[str, np.ndarray] = {}
    frame: dict[str, np.ndarray] = {}

    gap = interp.get("homo_lumo_gap_ev")
    if gap is not None:
        g = float(min(gap)) if open_shell else float(np.mean(gap))
        items.append(_item(
            "gap", "HOMO–LUMO gap (estimate)", g, "eV", 2, "trend",
            "How hard it is to excite or ionise the molecule at the ωB97M-V level. Smaller means more conjugated, "
            "more reactive, more colourful.",
            "Use it to rank molecules: it falls with conjugation length (ethylene 11.6 → octatetraene 8.2 eV, benzene 10.5 → "
            "tetracene 6.5 eV). It is an orbital gap, not an optical or fundamental gap.",
            f"α {gap[0]:.2f} eV, β {gap[1]:.2f} eV" if open_shell else ""))

    u = interp.get("unpaired_u_rad")
    if u is not None and heads.get("spin_exchange") and not periodic:  # crystals: diamond reads 0.997, unusable
        total = float(np.sum(u))
        trust = "care" if (not open_shell and total > 1.0) else "qualitative"
        items.append(_item(
            "unpaired", "Unpaired-electron character", total, "", 2, trust,
            "Where electrons are unpaired: the reactive sites of a radical, or the two ends of a bond that is breaking. "
            "Near zero for an ordinary closed-shell molecule. Colour the structure by it in the viewer.",
            "A relative index, not an electron count. It finds the right sites (benzyl, phenoxyl: ortho and para, not meta; "
            "allyl: both ends) but grows without limit when a bond is pulled past ~2.5 Å (H₂).",
            "largest: " + _top(u, sym, 3, fmt="{:.2f}")))
        cols["unpaired"] = u
        frame["unpaired"] = u
        if not open_shell and total > 1.0:
            notes.append("A closed-shell structure with large unpaired-electron character usually has a stretched or broken "
                         "bond: check the spin multiplicity (a triplet or broken-symmetry state may be the right description).")

    q = interp.get("charges_e")
    if q is not None and not periodic and float(np.max(np.abs(q))) > 1e-4:
        items.append(_item(
            "charges", "Partial charges", float(np.min(q)), "e", 2, "qualitative",
            "How the model shares electrons between atoms; they set the dipole and the electrostatics. "
            "The value shown is the most negative atom.",
            "Signs and symmetry follow chemistry (acetate O equal, phenolate charge on O plus ortho/para C, a leaving group "
            "takes the charge along an SN2 path), but values are smaller than Mulliken or NPA, a crowded cation can come out "
            "wrong (N in NH₄⁺ is positive), and pulled-apart ion pairs stay ionic.",
            "most negative: " + _top(q, sym, 2, sign=-1.0) + " · most positive: " + _top(q, sym, 2)))

    s = interp.get("spin_density_e")
    if s is not None and open_shell and not periodic:
        items.append(_item(
            "spin", "Spin density", float(np.sum(s)), "e", 2, "care",
            "Where the net spin sits; the total is the number of unpaired electrons (2S).",
            "Smoothed by the model: it spreads spin over neighbours and misses spin polarisation (negative spin on meta "
            "carbons or on methyl hydrogens). For radical sites prefer the unpaired-electron colouring.",
            "largest: " + _top(s, sym, 3)))

    pauli = interp.get("e_pauli_atom_ev")
    if pauli is not None:
        tot = float(np.sum(pauli))
        per = tot / max(n, 1)
        items.append(_item(
            "pauli", "Pauli (overlap) repulsion", per if periodic else tot, "eV/atom" if periodic else "eV", 3, "qualitative",
            "Repulsion from electron clouds pressed together: steric clashes, short contacts, compressed lattices. "
            "Colour the structure by it to see where.",
            "A model term, not an energy decomposition: every structure has a baseline, and the rest of the model takes most "
            "of the repulsion. Compare the same atoms between two structures (strained vs relaxed, compressed vs not); it "
            "rises steeply under compression (Si, Cu ×16 at −15 % lattice constant).",
            "largest: " + _top(pauli, sym, 3, fmt="{:.3f} eV")))
        cols["pauli"] = pauli
        frame["pauli"] = pauli

    ex = interp.get("e_exchange_atom_ev")
    if ex is not None and open_shell:
        items.append(_item(
            "exchange", "Spin-exchange energy", float(np.sum(ex)), "eV", 3, "care",
            "Energy from aligning unpaired spins (Hund's rule); negative stabilises the high-spin state "
            "(triplet O₂ −0.32 eV).",
            "Physical near equilibrium geometries; wrong for stretched singlet bonds."))

    if periodic and heads.get("homo_lumo_gap"):
        notes.append("For crystals and surfaces the model predicts no band gap, charges or spin exchange (those heads run on "
                     "molecules only), and its unpaired-electron index is not meaningful there. The Pauli map and the energy "
                     "decomposition are available.")
    if not heads.get("pauli_overlap"):
        notes.append(f"{model_title} has charges and spins only. Mena-LAM-6 adds the HOMO–LUMO gap, unpaired-electron and "
                     "Pauli-repulsion maps.")
    raw = {k: v for k, v in interp.items() if v is not None and k != "heads"}
    panel = {"title": "Interpretation", "model": model_title, "items": items, "notes": notes,
             "intro": "What the model's physics heads say about this structure. Each line says what it means and how far "
                      "to trust it."}
    return {"panel": panel, "columns": cols, "frame": frame, "raw": raw}


def skipped(n: int) -> dict:
    return {"title": "Interpretation", "items": [], "intro": "",
            "notes": [f"Interpretation skipped: {n:,} atoms is above {MAX_ATOMS:,}. Run it on a smaller cut-out."]}
