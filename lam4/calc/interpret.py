"""Quantum Dipole & Electric-Field X-Ray interpretability utilities for protein-ligand binding."""
from __future__ import annotations

import math
from typing import Any, Sequence

from ase import Atoms
import numpy as np
import torch

from lam4.calc.implicit_solvent import K_COUL, compute_fragment_ddg_solv
from lam4.calc.interpret_groups import (
    EA2_PER_MV_CM,
    EA2_PER_VA,
    MV_CM_PER_EA2,
    VA_PER_EA2,
    assign_groups_rdkit,
    assign_groups_topology,
    cluster_functional_groups,
    covalent_adj,
    dipole_to_debye,
    field_to_mv_cm,
    field_to_va,
)
from lam4.calc.physics import MOL_KCAL_TO_EV
from lam4.physics.coulomb_exact import SIGMA


def _to_bool_mask(mask_or_idx: np.ndarray | Sequence[bool] | Sequence[int], n_atoms: int, name: str) -> np.ndarray:
    arr = np.asarray(mask_or_idx)
    if arr.dtype == bool:
        if arr.shape != (n_atoms,):
            raise ValueError(f"{name} bool mask must have shape ({n_atoms},), got {arr.shape}")
        return arr.copy()
    out = np.zeros(n_atoms, dtype=bool)
    out[arr.astype(np.int64)] = True
    return out


def _get_frag_id(atoms: Atoms) -> np.ndarray | None:
    """Read per-atom fragment IDs from `atoms.arrays['frag_id']` or `atoms.info['frag_id']`."""
    fid = atoms.arrays.get("frag_id")
    if fid is None:
        fid = atoms.info.get("frag_id")
    return None if fid is None else np.asarray(fid, dtype=np.int64)


def _kabsch_rotation(src_pos: np.ndarray, dst_pos: np.ndarray) -> np.ndarray:
    """Return optimal 3x3 proper rotation R such that (src - mean) @ R.T ~= (dst - mean)."""
    if len(src_pos) < 3:
        return np.eye(3, dtype=np.float64)
    p = src_pos - src_pos.mean(axis=0, keepdims=True)
    q = dst_pos - dst_pos.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(p.T @ q)
    diag = np.diag([1.0, 1.0, -1.0 if np.linalg.det(vt.T @ u.T) < 0.0 else 1.0])
    return (vt.T @ diag @ u.T).astype(np.float64)


def _smeared_coulomb_field_from_sources(
    src_pos: np.ndarray,
    src_q: np.ndarray,
    dst_pos: np.ndarray,
    *,
    sigma: float = SIGMA,
) -> np.ndarray:
    """Compute exact `lam4` erf-smeared electric field (e/A^2) at `dst_pos` due to `src_q` at `src_pos`."""
    if len(src_pos) == 0 or len(dst_pos) == 0:
        return np.zeros((len(dst_pos), 3), dtype=np.float64)
    diff = dst_pos[:, None, :] - src_pos[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    r_safe = np.maximum(r, 1e-12)
    u = r / float(sigma)
    erf_u = torch.erf(torch.from_numpy(u)).numpy()
    radial = (erf_u - (2.0 / math.sqrt(math.pi)) * u * np.exp(-u * u)) / (r_safe ** 3)
    radial = np.where(r > 1e-8, radial, 0.0)
    return np.einsum("ij,ijc,j->ic", radial, diff, src_q, dtype=np.float64)


def _infer_subset_charge_spin(
    cpx_atoms: Atoms,
    mask: np.ndarray,
    charge_override: float | None,
    spin_override: int | None,
    *,
    default_charge: float,
) -> tuple[float, int]:
    if charge_override is not None and spin_override is not None:
        return float(charge_override), int(spin_override)
    info = cpx_atoms.info
    q_val = charge_override
    fids = _get_frag_id(cpx_atoms)
    if q_val is None and fids is not None and "frag_charge" in info:
        fchg = np.asarray(info["frag_charge"], dtype=np.float64)
        sub_fids = np.unique(fids[mask])
        if len(sub_fids) > 0 and int(sub_fids.max()) < len(fchg):
            if all(bool(np.all(mask[fids == fid])) for fid in sub_fids):
                q_val = float(sum(fchg[int(fid)] for fid in sub_fids))
    if q_val is None:
        q_val = float(default_charge)
    s_val = int(spin_override) if spin_override is not None else 1
    return float(q_val), int(s_val)


def _extract_sub_atoms(
    cpx_atoms: Atoms,
    mask: np.ndarray,
    *,
    charge: float,
    spin: int,
    preserve_segments: bool = False,
) -> Atoms:
    sub = cpx_atoms[mask].copy()
    sub.arrays.pop("frag_id", None)
    lot = cpx_atoms.info.get("lot", "auto")
    sub.info = {"lot": lot, "charge": float(charge), "spin": int(spin)}
    fids_full = _get_frag_id(cpx_atoms)
    if preserve_segments and fids_full is not None and "frag_charge" in cpx_atoms.info:
        fids = fids_full[mask]
        fchg = np.asarray(cpx_atoms.info["frag_charge"], dtype=np.float64)
        fspn = np.asarray(cpx_atoms.info.get("frag_spin", np.ones_like(fchg)), dtype=np.float64)
        uniq, inv = np.unique(fids, return_inverse=True)
        if len(uniq) > 1 and int(uniq.max()) < len(fchg):
            sub.arrays["frag_id"] = inv.astype(np.int64)
            sub.info["frag_charge"] = fchg[uniq].astype(np.float64)
            sub.info["frag_spin"] = fspn[uniq].astype(np.float64)
            sub.info["charge"] = float(np.sum(fchg[uniq]))
    return sub


def _eval_with_multipoles(calc: Any, atoms: Atoms) -> dict[str, Any]:
    work = atoms.copy()
    work.info = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in atoms.info.items()}
    if "frag_id" in atoms.arrays:
        work.arrays["frag_id"] = atoms.arrays["frag_id"].copy()
    calc.calculate(work, properties=("energy", "multipoles"))
    r = calc.results
    return {
        "energy_ev": float(r["energy"]),
        "charges": np.asarray(r["charges"], dtype=np.float64).copy(),
        "atomic_dipoles": np.asarray(r["atomic_dipoles"], dtype=np.float64).copy(),
        "atomic_dipoles_perm": np.asarray(r["atomic_dipoles_perm"], dtype=np.float64).copy(),
        "atomic_dipoles_ind": np.asarray(r["atomic_dipoles_ind"], dtype=np.float64).copy(),
        "electric_field": np.asarray(r["electric_field"], dtype=np.float64).copy(),
        "polarization_atom": np.asarray(r["polarization_atom"], dtype=np.float64).copy(),
    }


def analyze_binding_multipoles(
    calc: Any,
    cpx_atoms: Atoms,
    pkt_mask: np.ndarray | Sequence[bool] | Sequence[int],
    lig_mask: np.ndarray | Sequence[bool] | Sequence[int],
    *,
    lig_free_atoms: Atoms | None = None,
    lig_charge: float | None = None,
    lig_spin: int | None = None,
    pkt_charge: float | None = None,
    pkt_spin: int | None = None,
    rdkit_mol: Any | None = None,
    lig_smiles: str | None = None,
    compute_desolvation: bool = True,
    eval_pocket_fragment: bool = True,
    gbsa_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Quantum Dipole & Electric-Field X-Ray analysis on a protein-ligand complex.

    Computes per ligand atom and per functional group:
      - ``E_pkt_i = fld_cpx_i - fld_lig_bound_i`` (in ``e/A^2``, ``V/A``, and ``MV/cm``),
        plus ``E_pkt_direct_i`` (exact smeared Coulomb field due solely to ``q_cpx[pkt]``).
      - Binding-induced dipole shift ``dp_bind_i = p_tot_cpx_i - p_tot_lig_bound_i`` (Debye),
        and optional Kabsch-aligned conformational dipole shift ``p_tot_lig_bound_i - p_tot_lig_free_i``.
      - Dipole-Field Alignment Score ``E_align_i = -K_COUL * (p_tot_cpx_i . E_pkt_i)`` (kcal/mol;
        ``< 0`` stabilizing/aligned, ``> 0`` electrostatically frustrated).
      - Per-atom multipole desolvation penalty ``ddG_solv_i`` (monopole + GK dipole + SASA, kcal/mol).
    """
    n_cpx = len(cpx_atoms)
    pkt_b = _to_bool_mask(pkt_mask, n_cpx, "pkt_mask")
    lig_b = _to_bool_mask(lig_mask, n_cpx, "lig_mask")
    if np.any(pkt_b & lig_b):
        raise ValueError("pkt_mask and lig_mask overlap.")
    lig_idx = np.flatnonzero(lig_b)
    if len(lig_idx) == 0:
        raise ValueError("lig_mask selects zero atoms.")

    q_lig, s_lig = _infer_subset_charge_spin(cpx_atoms, lig_b, lig_charge, lig_spin, default_charge=0.0)
    cpx_tot_q = float(cpx_atoms.info.get("charge", 0.0))
    fids_cpx = _get_frag_id(cpx_atoms)
    if "frag_charge" in cpx_atoms.info and fids_cpx is not None:
        fchg = np.asarray(cpx_atoms.info["frag_charge"], dtype=np.float64)
        cpx_tot_q = float(sum(fchg[int(fid)] for fid in np.unique(fids_cpx)))
    q_pkt, s_pkt = _infer_subset_charge_spin(cpx_atoms, pkt_b, pkt_charge, pkt_spin, default_charge=cpx_tot_q - q_lig)

    cpx_res = _eval_with_multipoles(calc, cpx_atoms)
    lig_bound_atoms = _extract_sub_atoms(cpx_atoms, lig_b, charge=q_lig, spin=s_lig, preserve_segments=False)
    lig_res = _eval_with_multipoles(calc, lig_bound_atoms)

    q_cpx, p_cpx, fld_cpx = cpx_res["charges"], cpx_res["atomic_dipoles"], cpx_res["electric_field"]
    q_lig_b, p_lig_b, fld_lig_b = lig_res["charges"], lig_res["atomic_dipoles"], lig_res["electric_field"]
    p_cpx_l, q_cpx_l, fld_cpx_l = p_cpx[lig_b], q_cpx[lig_b], fld_cpx[lig_b]

    e_pkt_ea2 = fld_cpx_l - fld_lig_b
    cpx_pos = cpx_atoms.get_positions()
    sigma = float(getattr(calc.model.cfg, "coulomb_sigma", SIGMA)) if hasattr(calc, "model") else SIGMA
    e_pkt_direct_ea2 = _smeared_coulomb_field_from_sources(cpx_pos[pkt_b], q_cpx[pkt_b], cpx_pos[lig_b], sigma=sigma)
    e_pkt_va, e_pkt_mv_cm = field_to_va(e_pkt_ea2), field_to_mv_cm(e_pkt_ea2)
    e_pkt_norm_va, e_pkt_norm_mv_cm = np.linalg.norm(e_pkt_va, axis=-1), np.linalg.norm(e_pkt_mv_cm, axis=-1)

    dp_bind_ea = p_cpx_l - p_lig_b
    dp_bind_deb = dipole_to_debye(dp_bind_ea)
    dp_bind_norm_deb = np.linalg.norm(dp_bind_deb, axis=-1)

    e_align_kcal = -K_COUL * np.sum(p_cpx_l * e_pkt_ea2, axis=-1)
    denom = np.linalg.norm(p_cpx_l, axis=-1) * np.linalg.norm(e_pkt_ea2, axis=-1)
    e_align_cos = np.where(denom > 1e-10, np.sum(p_cpx_l * e_pkt_ea2, axis=-1) / np.maximum(denom, 1e-10), 0.0)

    pol_cpx_ev, pol_lig_ev = cpx_res["polarization_atom"][lig_b], lig_res["polarization_atom"]
    de_pol_bind_kcal = (pol_cpx_ev - pol_lig_ev) / MOL_KCAL_TO_EV

    p_lig_free_deb = dp_conf_deb = dp_conf_norm_deb = None
    if lig_free_atoms is not None:
        if len(lig_free_atoms) != len(lig_bound_atoms):
            raise ValueError(f"lig_free_atoms length {len(lig_free_atoms)} != lig_mask length {len(lig_bound_atoms)}")
        free_copy = lig_free_atoms.copy()
        free_copy.arrays.pop("frag_id", None)
        free_copy.info = dict(lig_free_atoms.info)
        free_copy.info.setdefault("lot", cpx_atoms.info.get("lot", "auto"))
        free_copy.info.setdefault("charge", float(q_lig))
        free_copy.info.setdefault("spin", int(s_lig))
        free_res = _eval_with_multipoles(calc, free_copy)
        rot = _kabsch_rotation(free_copy.get_positions(), lig_bound_atoms.get_positions())
        p_free_aligned_ea = free_res["atomic_dipoles"] @ rot.T
        p_lig_free_deb = dipole_to_debye(p_free_aligned_ea)
        dp_conf_deb = dipole_to_debye(p_lig_b - p_free_aligned_ea)
        dp_conf_norm_deb = np.linalg.norm(dp_conf_deb, axis=-1)

    ddg_solv_at = ddg_gb_at = ddg_mono_at = ddg_dip_at = ddg_np_at = np.zeros(len(lig_idx), dtype=np.float64)
    solv_summary = {k: 0.0 for k in ("ddg_solv_total_kcal", "ddg_gb_total_kcal", "ddg_mono_total_kcal", "ddg_dip_total_kcal", "ddg_np_total_kcal")}
    if compute_desolvation:
        gb_kw = dict(gbsa_kwargs or {})
        frag_ids = np.full(n_cpx, 2, dtype=np.int64)
        frag_ids[pkt_b] = 0
        frag_ids[lig_b] = 1
        frag_q_map: dict[int, np.ndarray] = {1: q_lig_b}
        frag_p_map: dict[int, np.ndarray] = {1: p_lig_b}
        if eval_pocket_fragment and np.any(pkt_b):
            pkt_atoms = _extract_sub_atoms(cpx_atoms, pkt_b, charge=q_pkt, spin=s_pkt, preserve_segments=True)
            pkt_res = _eval_with_multipoles(calc, pkt_atoms)
            frag_q_map[0] = pkt_res["charges"]
            frag_p_map[0] = pkt_res["atomic_dipoles"]
        solv_res = compute_fragment_ddg_solv(
            cpx_atoms.get_atomic_numbers(), cpx_pos, q_cpx, frag_ids,
            atomic_dipoles=p_cpx, fragment_charges=frag_q_map, fragment_dipoles=frag_p_map,
            return_per_atom=True, **gb_kw,
        )
        cpx_s, lig_s = solv_res["complex"], solv_res["fragments"][1]
        ddg_solv_at = np.asarray(solv_res["ddg_solv_atom"][lig_b], dtype=np.float64)
        ddg_gb_at = np.asarray(solv_res["ddg_gb_atom"][lig_b], dtype=np.float64)
        ddg_mono_at = np.asarray(cpx_s["dg_mono_atom"][lig_b] - lig_s["dg_mono_atom"], dtype=np.float64)
        ddg_dip_at = np.asarray(solv_res["ddg_dip_atom"][lig_b], dtype=np.float64)
        ddg_np_at = np.asarray(cpx_s["dg_np_atom"][lig_b] - lig_s["dg_np_atom"], dtype=np.float64)
        solv_summary = {
            "ddg_solv_total_kcal": float(solv_res["ddg_solv"]),
            "ddg_gb_total_kcal": float(solv_res["ddg_gb"]),
            "ddg_mono_total_kcal": float(solv_res["ddg_mono"]),
            "ddg_dip_total_kcal": float(solv_res["ddg_dip"]),
            "ddg_np_total_kcal": float(solv_res["ddg_np"]),
        }

    atom_labels = assign_groups_rdkit(lig_bound_atoms, rdkit_mol, lig_smiles) or assign_groups_topology(lig_bound_atoms)
    clusters = cluster_functional_groups(lig_bound_atoms, atom_labels, covalent_adj(lig_bound_atoms.get_atomic_numbers(), lig_bound_atoms.get_positions()))
    lig_pos, lig_syms = lig_bound_atoms.get_positions(), lig_bound_atoms.get_chemical_symbols()
    fg_list: list[dict[str, Any]] = []
    for gname, idxs in clusters:
        idx_arr = np.asarray(idxs, dtype=np.int64)
        r_grp = lig_pos[idx_arr]
        q_grp = q_cpx_l[idx_arr]
        grp_dip_deb = float(np.linalg.norm(dipole_to_debye(np.sum(q_grp[:, None] * (r_grp - r_grp.mean(axis=0, keepdims=True)) + p_cpx_l[idx_arr], axis=0))))
        e_al = float(np.sum(e_align_kcal[idx_arr]))
        ddg_g = float(np.sum(ddg_gb_at[idx_arr]))
        fg_list.append({
            "name": gname, "atom_indices_lig": idxs, "atom_indices_cpx": [int(lig_idx[k]) for k in idxs],
            "symbols": [lig_syms[k] for k in idxs], "n_atoms": len(idxs), "net_charge_cpx": float(np.sum(q_grp)),
            "group_dipole_debye": grp_dip_deb, "mean_dp_bind_debye": float(np.mean(dp_bind_norm_deb[idx_arr])),
            "sum_dp_bind_debye": float(np.sum(dp_bind_norm_deb[idx_arr])),
            "mean_E_pkt_mv_cm": float(np.mean(e_pkt_norm_mv_cm[idx_arr])),
            "max_E_pkt_mv_cm": float(np.max(e_pkt_norm_mv_cm[idx_arr])),
            "E_align_kcal": e_al, "ddg_solv_atom_kcal": float(np.sum(ddg_solv_at[idx_arr])),
            "ddg_gb_atom_kcal": ddg_g, "ddg_mono_atom_kcal": float(np.sum(ddg_mono_at[idx_arr])),
            "ddg_dip_atom_kcal": float(np.sum(ddg_dip_at[idx_arr])), "net_elec_score_kcal": e_al + ddg_g,
            "status": "stabilizing" if e_al < -0.25 else ("frustrated" if e_al > 0.25 else "neutral"),
        })

    return {
        "lig_indices": lig_idx, "symbols": lig_syms, "atomic_numbers": lig_bound_atoms.get_atomic_numbers(),
        "pos": lig_pos, "charges_cpx": q_cpx_l, "charges_lig_bound": q_lig_b, "dq_bind": q_cpx_l - q_lig_b,
        "atomic_dipoles_cpx": p_cpx_l, "atomic_dipoles_lig_bound": p_lig_b,
        "atomic_dipoles_cpx_debye": dipole_to_debye(p_cpx_l), "atomic_dipoles_lig_bound_debye": dipole_to_debye(p_lig_b),
        "dp_bind_ea": dp_bind_ea, "dp_bind_debye": dp_bind_deb, "dp_bind_norm_debye": dp_bind_norm_deb,
        "atomic_dipoles_lig_free_debye": p_lig_free_deb, "dp_conf_debye": dp_conf_deb, "dp_conf_norm_debye": dp_conf_norm_deb,
        "electric_field_cpx_ea2": fld_cpx_l, "electric_field_lig_bound_ea2": fld_lig_b,
        "E_pkt_ea2": e_pkt_ea2, "E_pkt_direct_ea2": e_pkt_direct_ea2,
        "E_pkt_va": e_pkt_va, "E_pkt_norm_va": e_pkt_norm_va,
        "E_pkt_mv_cm": e_pkt_mv_cm, "E_pkt_norm_mv_cm": e_pkt_norm_mv_cm,
        "E_align_kcal": e_align_kcal, "E_align_cos": e_align_cos, "E_align_total_kcal": float(np.sum(e_align_kcal)),
        "polarization_atom_cpx_ev": pol_cpx_ev, "polarization_atom_lig_bound_ev": pol_lig_ev,
        "dE_pol_bind_kcal": de_pol_bind_kcal, "ddg_solv_atom_kcal": ddg_solv_at,
        "ddg_gb_atom_kcal": ddg_gb_at, "ddg_mono_atom_kcal": ddg_mono_at,
        "ddg_dip_atom_kcal": ddg_dip_at, "ddg_np_atom_kcal": ddg_np_at, **solv_summary,
        "atom_groups": atom_labels, "functional_groups": fg_list,
        "lig_charge_used": q_lig, "pkt_charge_used": q_pkt,
        "cpx_energy_ev": cpx_res["energy_ev"], "lig_bound_energy_ev": lig_res["energy_ev"],
    }
