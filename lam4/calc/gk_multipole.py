"""Analytical Multipole Generalized Kirkwood / Born-Onsager Solvation (Monopole + Dipole)."""
from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from lam4.calc.implicit_solvent import (
    ArrayLike,
    GBMethod,
    GBSAContext,
    K_COUL,
    RadiiSet,
    _backend,
    _clamp_min,
    _eye,
    _pairwise_distances,
    _to_float,
    build_gbsa_context,
    compute_gbsa,
    eval_gbsa_from_context,
)


def _select_frag_override(overrides: Sequence[Any] | Mapping[int, Any] | None, fid: int, idx: int) -> Any | None:
    if overrides is None:
        return None
    if isinstance(overrides, Mapping):
        return overrides.get(fid)
    return overrides[idx] if idx < len(overrides) else None


def _sum_axis1(x: Any, is_t: bool) -> Any:
    return x.sum(dim=1) if is_t else np.sum(x, axis=1)


def _sum_all(x: Any, is_t: bool) -> Any:
    return x.sum() if is_t else float(np.sum(x))


def _gk_dielectric_factors(eps_in: float, eps_solv: float) -> tuple[float, float]:
    """Exact Kirkwood spherical-cavity reaction-field factors c_0 (monopole) and c_1 (dipole)."""
    ei, es = float(eps_in), float(eps_solv)
    c0 = -(1.0 / ei - 1.0 / es)
    c1 = -(1.0 / ei) * (2.0 * (es - ei)) / (2.0 * es + ei)
    return c0, c1


def eval_multipole_gbsa_from_context(
    ctx: GBSAContext,
    q: ArrayLike | Sequence[float],
    atomic_dipoles: ArrayLike | None = None,
    *,
    pos: ArrayLike | None = None,
    eps_solv: float = 78.4,
    eps_in: float = 1.0,
    gamma_sasa: float = 0.00542,
    sasa_offset: float = 0.0,
    gk_kc: float = 2.455,
    dipole_cross_terms: bool = True,
    qp_prefactor_mode: Literal["c1", "c0", "mean"] = "c1",
    return_per_atom: bool = False,
) -> dict[str, Any]:
    """Evaluate zero-parameter OBC2 monopole + Generalized Kirkwood atomic-dipole solvation (kcal/mol).

    Equations (zero fitted parameters):
      - Monopole OBC2: ``0.5 * K_COUL * c_0 * sum_{i,j} q_i q_j / f_GB(r_ij)``
      - Self-dipole Onsager/Kirkwood cavity:
        ``0.5 * K_COUL * c_1 * sum_i |p_i|^2 / alpha_i^3``, where
        ``c_1 = -(1/eps_in) * 2*(eps_solv - eps_in)/(2*eps_solv + eps_in)``.
      - Pairwise Generalized Kirkwood cross terms on
        ``f_GK(r_ij) = sqrt(r_ij^2 + alpha_i alpha_j exp(-r_ij^2 / (gk_kc * alpha_i alpha_j)))``
        (Schnieders & Ponder, JCTC 2007, ``gk_kc = 2.455`` default):
        * Charge-dipole: ``K_COUL * c_qp * sum_{i != j} q_i (p_j . grad_j f_GK^-1)``
        * Dipole-dipole: ``0.5 * K_COUL * c_1 * sum_{i != j} p_i . [grad_i grad_j f_GK^-1] . p_j``.
    """
    is_t, dev, dt = _backend(ctx.inv_f_gb, q, atomic_dipoles, pos)
    inv_f = _to_float(ctx.inv_f_gb, is_t, dev, dt)
    q_arr = _to_float(q, is_t, dev, dt)
    alpha = _to_float(ctx.alpha, is_t, dev, dt)
    base = eval_gbsa_from_context(
        ctx, q_arr, eps_solv=eps_solv, eps_in=eps_in, gamma_sasa=gamma_sasa, sasa_offset=sasa_offset,
    )
    dg_mono, dg_np, sasa = base["dg_gb"], base["dg_np"], base["sasa_A2"]
    n = int(q_arr.shape[0])
    c_0, c_1 = _gk_dielectric_factors(eps_in, eps_solv)
    if qp_prefactor_mode == "c1":
        c_qp = c_1
    elif qp_prefactor_mode == "c0":
        c_qp = c_0
    elif qp_prefactor_mode == "mean":
        c_qp = 0.5 * (c_0 + c_1)
    else:
        raise ValueError(f"Unknown qp_prefactor_mode={qp_prefactor_mode!r}")

    zero = inv_f.new_zeros(()) if is_t else 0.0
    zeros_n = inv_f.new_zeros((n,)) if is_t else np.zeros(n, dtype=np.float64)

    if atomic_dipoles is None or n == 0:
        dg_dip_self = dg_qp = dg_pp = zero
        e_dip_self_at = e_qp_at = e_pp_at = zeros_n
    else:
        p_arr = _to_float(atomic_dipoles, is_t, dev, dt)
        if p_arr.shape != (n, 3):
            raise ValueError(f"atomic_dipoles must have shape ({n}, 3), got {tuple(p_arr.shape)}")
        p_norm2 = (p_arr * p_arr).sum(dim=-1) if is_t else np.sum(p_arr * p_arr, axis=-1)
        inv_a3 = 1.0 / _clamp_min(alpha ** 3, 1e-12)
        e_dip_self_at = (0.5 * K_COUL * c_1) * p_norm2 * inv_a3
        dg_dip_self = _sum_all(e_dip_self_at, is_t)

        if dipole_cross_terms and n > 1:
            pos_raw = ctx.pos if pos is None else pos
            if pos_raw is None:
                raise ValueError("pos is required for GK dipole cross terms (pass pos= or build ctx via build_gbsa_context).")
            pos_arr = _to_float(pos_raw, is_t, dev, dt)
            pos_c = pos_arr - (pos_arr.mean(dim=0, keepdim=True) if is_t else np.mean(pos_arr, axis=0, keepdims=True))
            if ctx.r_mat is not None and pos is None:
                r_arr = _to_float(ctx.r_mat, is_t, dev, dt)
                r2 = r_arr * r_arr
            else:
                r_arr = _pairwise_distances(pos_c)
                r2 = r_arr * r_arr
            aa = alpha[:, None] * alpha[None, :]
            kc = float(gk_kc)
            exp_fn, sqrt_fn = (torch.exp, torch.sqrt) if is_t else (np.exp, np.sqrt)
            exp_ij = exp_fn(-r2 / _clamp_min(kc * aa, 1e-8))
            f_gk = sqrt_fn(_clamp_min(r2 + aa * exp_ij, 1e-12))
            gp = 1.0 - exp_ij / kc
            off_diag = (~_eye(n, pos_c)).to(dt) if is_t else (~_eye(n, pos_c)).astype(np.float64)
            inv_f3 = (1.0 / (f_gk ** 3)) * off_diag
            inv_f5 = (1.0 / (f_gk ** 5)) * off_diag
            s_ij = gp * inv_f3
            c_ij = (2.0 * exp_ij / _clamp_min((kc ** 2) * aa, 1e-8)) * inv_f3 - 3.0 * (gp ** 2) * inv_f5

            rp = pos_c @ p_arr.T
            rp_diag = (pos_c * p_arr).sum(dim=-1) if is_t else np.sum(pos_c * p_arr, axis=-1)
            d_dot_pj = rp - rp_diag[None, :]
            d_dot_pi = -d_dot_pj.T

            qp_1 = q_arr * _sum_axis1(s_ij * d_dot_pj, is_t)
            qp_2 = -_sum_axis1(s_ij * q_arr[None, :] * d_dot_pi, is_t)
            e_qp_at = (0.5 * K_COUL * c_qp) * (qp_1 + qp_2)
            e_pp_at = (0.5 * K_COUL * c_1) * _sum_axis1(s_ij * (p_arr @ p_arr.T) + c_ij * d_dot_pi * d_dot_pj, is_t)
            dg_qp = _sum_all(e_qp_at, is_t)
            dg_pp = _sum_all(e_pp_at, is_t)
        else:
            dg_qp = dg_pp = zero
            e_qp_at = e_pp_at = zeros_n

    dg_dip = dg_dip_self + dg_qp + dg_pp
    dg_gb = dg_mono + dg_dip
    out: dict[str, Any] = {
        "dg_solv": dg_gb + dg_np,
        "dg_gb": dg_gb,
        "dg_mono": dg_mono,
        "dg_dip": dg_dip,
        "dg_dip_self": dg_dip_self,
        "dg_qp": dg_qp,
        "dg_pp": dg_pp,
        "dg_np": dg_np,
        "sasa_A2": sasa,
        "alpha_min": base["alpha_min"],
        "alpha_max": base["alpha_max"],
    }
    if return_per_atom:
        e_mono_at = (0.5 * K_COUL * c_0) * q_arr * (inv_f @ q_arr) if n > 0 else zeros_n
        if ctx.sasa_atom_A2 is not None and n > 0:
            sasa_at = _to_float(ctx.sasa_atom_A2, is_t, dev, dt)
            e_np_at = float(gamma_sasa) * sasa_at + (float(sasa_offset) / n)
        else:
            e_np_at = zeros_n + ((float(sasa_offset) / n) if n > 0 else 0.0)
        e_gb_at = e_mono_at + e_dip_self_at + e_qp_at + e_pp_at
        out["dg_mono_atom"] = e_mono_at
        out["dg_dip_self_atom"] = e_dip_self_at
        out["dg_qp_atom"] = e_qp_at
        out["dg_pp_atom"] = e_pp_at
        out["dg_gb_atom"] = e_gb_at
        out["dg_np_atom"] = e_np_at
        out["dg_solv_atom"] = e_gb_at + e_np_at
    return out


def compute_multipole_gbsa(
    z: ArrayLike | Sequence[int],
    pos: ArrayLike,
    q: ArrayLike | Sequence[float],
    atomic_dipoles: ArrayLike | None = None,
    *,
    method: GBMethod = "obc2",
    radii_set: RadiiSet = "mbondi2",
    eps_solv: float = 78.4,
    eps_in: float = 1.0,
    gamma_sasa: float = 0.00542,
    sasa_offset: float = 0.0,
    sasa_n_pts: int = 194,
    probe_radius: float = 1.4,
    radius_scale: float = 1.0,
    dielectric_offset: float = 0.09,
    sr_use_rho0: bool | None = None,
    compute_sasa_term: bool = True,
    use_mps: bool = True,
    gk_kc: float = 2.455,
    dipole_cross_terms: bool = True,
    qp_prefactor_mode: Literal["c1", "c0", "mean"] = "c1",
    return_per_atom: bool = False,
) -> dict[str, Any]:
    """Compute OBC2 + Generalized Kirkwood multipole solvation free energy (kcal/mol)."""
    ctx = build_gbsa_context(
        z, pos, method=method, radii_set=radii_set, radius_scale=radius_scale, dielectric_offset=dielectric_offset,
        sr_use_rho0=sr_use_rho0, sasa_n_pts=sasa_n_pts, probe_radius=probe_radius, compute_sasa_term=compute_sasa_term,
        use_mps=use_mps,
    )
    return eval_multipole_gbsa_from_context(
        ctx, q, atomic_dipoles, eps_solv=eps_solv, eps_in=eps_in, gamma_sasa=gamma_sasa, sasa_offset=sasa_offset,
        gk_kc=gk_kc, dipole_cross_terms=dipole_cross_terms, qp_prefactor_mode=qp_prefactor_mode,
        return_per_atom=return_per_atom,
    )


def compute_fragment_ddg_solv(
    z: ArrayLike | Sequence[int], pos: ArrayLike, q: ArrayLike | Sequence[float], frag_id: ArrayLike | Sequence[int],
    *, atomic_dipoles: ArrayLike | None = None,
    fragment_charges: Sequence[ArrayLike | Sequence[float]] | Mapping[int, ArrayLike | Sequence[float]] | None = None,
    fragment_dipoles: Sequence[ArrayLike] | Mapping[int, ArrayLike] | None = None,
    fragment_positions: Sequence[ArrayLike] | Mapping[int, ArrayLike] | None = None,
    method: GBMethod = "obc2", radii_set: RadiiSet = "mbondi2", eps_solv: float = 78.4, eps_in: float = 1.0,
    gamma_sasa: float = 0.00542, sasa_offset: float = 0.0, sasa_n_pts: int = 194, probe_radius: float = 1.4,
    radius_scale: float = 1.0, dielectric_offset: float = 0.09, sr_use_rho0: bool | None = None,
    compute_sasa_term: bool = True, use_mps: bool = True, gk_kc: float = 2.455,
    dipole_cross_terms: bool = True, qp_prefactor_mode: Literal["c1", "c0", "mean"] = "c1",
    return_per_atom: bool = False,
) -> dict[str, Any]:
    """Compute complex-minus-fragments desolvation `ddG_solv` using `frag_id` (`[receptor, ligand, cofactors...]`)."""
    use_multipoles = (atomic_dipoles is not None) or (fragment_dipoles is not None) or return_per_atom
    is_t = any(torch.is_tensor(x) for x in (pos, q, atomic_dipoles))
    fid_np = frag_id.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(frag_id) else np.asarray(frag_id, dtype=np.int64)
    n_atoms = int(pos.shape[0]) if torch.is_tensor(pos) else len(pos)
    if fid_np.ndim != 1 or len(fid_np) != n_atoms:
        raise ValueError(f"frag_id must be 1D of length {n_atoms}, got shape {fid_np.shape}")
    if n_atoms > 0 and int(fid_np.min()) < 0:
        raise ValueError(f"frag_id must be non-negative, got min={int(fid_np.min())}")

    gbsa_kw = dict(
        method=method, radii_set=radii_set, eps_solv=eps_solv, eps_in=eps_in, gamma_sasa=gamma_sasa,
        sasa_offset=sasa_offset, sasa_n_pts=sasa_n_pts, probe_radius=probe_radius, radius_scale=radius_scale,
        dielectric_offset=dielectric_offset, sr_use_rho0=sr_use_rho0, compute_sasa_term=compute_sasa_term, use_mps=use_mps,
    )
    if use_multipoles:
        mp_kw = dict(gk_kc=gk_kc, dipole_cross_terms=dipole_cross_terms, qp_prefactor_mode=qp_prefactor_mode, return_per_atom=return_per_atom)
        cpx_res = compute_multipole_gbsa(z, pos, q, atomic_dipoles, **gbsa_kw, **mp_kw)
    else:
        mp_kw = {}
        cpx_res = compute_gbsa(z, pos, q, **gbsa_kw)

    unique_fids = [int(x) for x in np.unique(fid_np)]
    frag_res: dict[int, dict[str, Any]] = {}
    dev = next((x.device for x in (pos, q, atomic_dipoles) if torch.is_tensor(x)), None)
    for idx, f in enumerate(unique_fids):
        mask_np = fid_np == f
        p_ov = _select_frag_override(fragment_positions, f, idx)
        q_ov = _select_frag_override(fragment_charges, f, idx)
        d_ov = _select_frag_override(fragment_dipoles, f, idx)
        if is_t:
            mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=dev)
            z_f = z[mask_t] if torch.is_tensor(z) else np.asarray(z, dtype=np.int64)[mask_np]
            pos_f = p_ov if p_ov is not None else (pos[mask_t] if torch.is_tensor(pos) else np.asarray(pos, dtype=np.float64)[mask_np])  # type: ignore[index]
            q_f = q_ov if q_ov is not None else (q[mask_t] if torch.is_tensor(q) else np.asarray(q, dtype=np.float64)[mask_np])  # type: ignore[index]
            d_f = d_ov if d_ov is not None else (
                (atomic_dipoles[mask_t] if torch.is_tensor(atomic_dipoles) else np.asarray(atomic_dipoles, dtype=np.float64)[mask_np])
                if atomic_dipoles is not None else None
            )  # type: ignore[index]
        else:
            z_f = np.asarray(z, dtype=np.int64)[mask_np]
            pos_f = np.asarray(p_ov, dtype=np.float64) if p_ov is not None else np.asarray(pos, dtype=np.float64)[mask_np]
            q_f = np.asarray(q_ov, dtype=np.float64) if q_ov is not None else np.asarray(q, dtype=np.float64)[mask_np]
            d_f = (
                np.asarray(d_ov, dtype=np.float64) if d_ov is not None
                else (np.asarray(atomic_dipoles, dtype=np.float64)[mask_np] if atomic_dipoles is not None else None)
            )
        frag_res[f] = (
            compute_multipole_gbsa(z_f, pos_f, q_f, d_f, **gbsa_kw, **mp_kw)
            if use_multipoles else compute_gbsa(z_f, pos_f, q_f, **gbsa_kw)
        )

    sum_gb = sum(frag_res[f]["dg_gb"] for f in unique_fids) if unique_fids else (0.0 * cpx_res["dg_gb"])
    sum_np = sum(frag_res[f]["dg_np"] for f in unique_fids) if unique_fids else (0.0 * cpx_res["dg_np"])
    sum_sasa = sum(frag_res[f]["sasa_A2"] for f in unique_fids) if unique_fids else (0.0 * cpx_res["sasa_A2"])
    ddg_gb, ddg_np = cpx_res["dg_gb"] - sum_gb, cpx_res["dg_np"] - sum_np
    res: dict[str, Any] = {
        "ddg_solv": ddg_gb + ddg_np, "ddg_gb": ddg_gb, "ddg_np": ddg_np,
        "dsasa_A2": cpx_res["sasa_A2"] - sum_sasa, "complex": cpx_res, "fragments": frag_res,
    }
    if use_multipoles:
        for k in ("mono", "dip", "dip_self", "qp", "pp"):
            sum_k = sum(frag_res[f][f"dg_{k}"] for f in unique_fids) if unique_fids else (0.0 * cpx_res[f"dg_{k}"])
            res[f"ddg_{k}"] = cpx_res[f"dg_{k}"] - sum_k
    if return_per_atom and n_atoms > 0:
        ddg_atom = cpx_res["dg_solv_atom"].clone() if is_t else cpx_res["dg_solv_atom"].copy()
        ddg_gb_atom = cpx_res["dg_gb_atom"].clone() if is_t else cpx_res["dg_gb_atom"].copy()
        ddg_dip_atom_cpx = cpx_res["dg_dip_self_atom"] + cpx_res["dg_qp_atom"] + cpx_res["dg_pp_atom"]
        ddg_dip_atom = ddg_dip_atom_cpx.clone() if is_t else ddg_dip_atom_cpx.copy()
        for f in unique_fids:
            mask_np = fid_np == f
            f_dip_at = frag_res[f]["dg_dip_self_atom"] + frag_res[f]["dg_qp_atom"] + frag_res[f]["dg_pp_atom"]
            if is_t:
                mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=dev)
                ddg_atom[mask_t] -= frag_res[f]["dg_solv_atom"]
                ddg_gb_atom[mask_t] -= frag_res[f]["dg_gb_atom"]
                ddg_dip_atom[mask_t] -= f_dip_at
            else:
                ddg_atom[mask_np] -= frag_res[f]["dg_solv_atom"]
                ddg_gb_atom[mask_np] -= frag_res[f]["dg_gb_atom"]
                ddg_dip_atom[mask_np] -= f_dip_at
        res["ddg_solv_atom"] = ddg_atom
        res["ddg_gb_atom"] = ddg_gb_atom
        res["ddg_dip_atom"] = ddg_dip_atom
    return res
