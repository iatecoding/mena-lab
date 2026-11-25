"""Opt-in read-out of the physics heads (charges, spins, Pauli overlap, exchange, gap, l=2 moments) for one structure.

The fast path of `MenaCalculator.calculate` never asks the model for its auxiliary heads. `interpret_heads` runs one extra
eager forward pass with `return_aux_heads=True` (no forces, no autograd) and captures the per-atom outputs of the
CloudOverlap and SpinExchange heads with forward hooks, so no model code changes. Energies come back in eV.

Heads that a checkpoint does not have, or that the model switches off for this kind of structure, are reported as None
(e.g. the HOMO-LUMO gap head and the exchange head only run on open molecules; charges are not used on bulk crystals).
"""
from __future__ import annotations

from typing import Any

from ase import Atoms
import numpy as np
import torch

from lam4.calc.physics import CRYSTAL_EV, MOL_KCAL_TO_EV
from lam4.calc.runtime import precision_context

MOLECULE_ONLY = ("homo_lumo_gap_ev", "e_exchange_atom_ev", "unpaired_u_rad", "spin_bias_m_raw")


def _np(x: Any) -> np.ndarray | None:
    if x is None or not torch.is_tensor(x):
        return None
    return x.detach().cpu().to(torch.float64).numpy()


def _hooks(model: Any, cap: dict[str, Any]) -> list[Any]:
    phys = getattr(model, "lam6_phys", None)
    if phys is None:
        return []
    out = []
    if getattr(phys, "cloud_head", None) is not None:
        out.append(phys.cloud_head.register_forward_hook(lambda _m, _i, o: cap.__setitem__("overlap", o)))
    if getattr(phys, "exch_head", None) is not None:
        out.append(phys.exch_head.register_forward_hook(lambda _m, _i, o: cap.__setitem__("exchange", o)))
    return out


def heads_present(model: Any) -> dict[str, bool]:
    """Which interpretable heads this checkpoint carries (lam5 has charges/spins only; lam6 adds the physics heads)."""
    phys = getattr(model, "lam6_phys", None)
    has = lambda name: phys is not None and getattr(phys, name, None) is not None  # noqa: E731
    return {
        "charges": True, "spins": True,
        "pauli_overlap": has("cloud_head"), "spin_exchange": has("exch_head"),
        "homo_lumo_gap": has("gap_head"), "quadrupole_l2": has("quad_l2_head"),
        "spin_descriptor": getattr(model.trunk, "spin_desc_head", None) is not None,
    }


def _per_atom(arr: np.ndarray | None, n: int) -> np.ndarray | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return a if a.shape[0] == n else None


def interpret_heads(calc: Any, atoms: Atoms, *, include_invalid: bool = False) -> dict[str, Any]:
    """One eager forward with the auxiliary heads on; returns plain numpy arrays (per atom) and floats (eV, e).

    Heads that are meaningless under PBC (`homo_lumo_gap_ev`, `e_exchange_atom_ev`, `unpaired_u_rad`,
    `spin_bias_m_raw`) are masked to `None` when `periodic=True` unless `include_invalid=True`.
    """
    model = calc.model
    n = len(atoms)
    b, chunks, _ref, _sc, src, _u, q_val, s_val, *_ = calc._prepare_single_batch(atoms)
    to_ev = (1.0 / CRYSTAL_EV) if src.domain == "crystal" else MOL_KCAL_TO_EV
    cap: dict[str, Any] = {}
    handles = _hooks(model, cap)
    was_compiled = bool(getattr(calc, "_compiled", False))
    if was_compiled:
        calc._set_compiled_active(False)
    try:
        with precision_context(getattr(calc, "precision", "auto"), calc.device, model), torch.no_grad():
            out = model(b, create_graph=False, compute_forces=False, compute_stress=False, chunks=chunks, add_ref=False,
                        return_stage_energies=False, csr_plan=calc._active_csr_plan, return_phi=False, return_aux_heads=True)
    finally:
        for h in handles:
            h.remove()
        if was_compiled:
            calc._set_compiled_active(True)

    periodic = bool(b.pbc)
    ov = cap.get("overlap") if isinstance(cap.get("overlap"), dict) else None
    ex = cap.get("exchange") if torch.is_tensor(cap.get("exchange")) else None
    gap = _np(out.get("homo_gap"))
    quad = _np(out.get("quad_l2_atom"))
    hp = heads_present(model)
    raw_u_rad = _per_atom(_np(out.get("u_rad")), n)
    raw_m_raw = _per_atom(_np(out.get("m_raw")), n)
    allow_mol_only = (not periodic) or bool(include_invalid)
    res: dict[str, Any] = {
        "domain": src.domain, "periodic": periodic, "lot": src.name, "charge": float(q_val), "multiplicity": int(s_val),
        "heads": hp,
        "charges_e": _per_atom(_np(out.get("charges")), n),
        "spin_density_e": _per_atom(_np(out.get("spin_density")), n),
        "stage_charges_e": [a for a in (_per_atom(_np(x), n) for x in (out.get("stage_charges") or [])) if a is not None],
        "unpaired_u_rad": raw_u_rad if allow_mol_only else None,
        "spin_bias_m_raw": raw_m_raw if allow_mol_only else None,
        "e_pauli_atom_ev": None, "e_aniso_atom_ev": None, "e_overlap_atom_ev": None, "e_exchange_atom_ev": None,
        "quad_l2_norm": None, "homo_lumo_gap_ev": None,
        "e_network_atom_ev": _per_atom(_np(out.get("energy_atom")), n),
    }
    if res["e_network_atom_ev"] is not None:
        res["e_network_atom_ev"] = res["e_network_atom_ev"] * to_ev
    if ov is not None:
        for key, name in (("e_pauli_atom", "e_pauli_atom_ev"), ("e_aniso_atom", "e_aniso_atom_ev"), ("energy_atom", "e_overlap_atom_ev")):
            arr = _per_atom(_np(ov.get(key)), n)
            res[name] = None if arr is None else arr * to_ev
    if ex is not None and allow_mol_only:
        arr = _per_atom(_np(ex), n)
        res["e_exchange_atom_ev"] = None if arr is None else arr * to_ev
    if quad is not None and quad.shape[0] == n and float(np.abs(quad).max()) > 1e-6:
        res["quad_l2_norm"] = np.linalg.norm(quad, axis=-1)
    if gap is not None and allow_mol_only and gap.size >= 2:
        res["homo_lumo_gap_ev"] = [float(gap.reshape(-1)[0]), float(gap.reshape(-1)[1])]
    res["head_validity"] = {
        "charges_e": "molecule_only" if periodic else ("valid" if res["charges_e"] is not None else "unavailable"),
        "spin_density_e": "molecule_only" if periodic else ("valid" if res["spin_density_e"] is not None else "unavailable"),
        "unpaired_u_rad": "molecule_only" if (periodic and hp["spin_descriptor"]) else ("valid" if raw_u_rad is not None else "unavailable"),
        "spin_bias_m_raw": "molecule_only" if (periodic and hp["spin_descriptor"]) else ("valid" if raw_m_raw is not None else "unavailable"),
        "e_pauli_atom_ev": "valid" if res["e_pauli_atom_ev"] is not None else "unavailable",
        "e_exchange_atom_ev": "molecule_only" if (periodic and hp["spin_exchange"]) else ("valid" if ex is not None else "unavailable"),
        "homo_lumo_gap_ev": "molecule_only" if (periodic and hp["homo_lumo_gap"]) else ("valid" if (gap is not None and gap.size >= 2) else "unavailable"),
        "quad_l2_norm": "valid" if res["quad_l2_norm"] is not None else "unavailable",
    }
    return res


def summarize(interp: dict[str, Any]) -> dict[str, float]:
    """Scalar summary (totals, extremes) for tables and tests."""
    out: dict[str, float] = {}
    for key in ("e_pauli_atom_ev", "e_aniso_atom_ev", "e_overlap_atom_ev", "e_exchange_atom_ev"):
        if interp.get(key) is not None:
            out[key.replace("_atom", "")] = float(np.sum(interp[key]))
    if interp.get("unpaired_u_rad") is not None:
        out["unpaired_total"] = float(np.sum(interp["unpaired_u_rad"]))
    if interp.get("spin_density_e") is not None:
        out["spin_total"] = float(np.sum(interp["spin_density_e"]))
    if interp.get("homo_lumo_gap_ev") is not None:
        out["gap_alpha_ev"], out["gap_beta_ev"] = interp["homo_lumo_gap_ev"]
    return out
