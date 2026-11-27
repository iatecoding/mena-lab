"""Polarizability, energy decomposition, and harmonic IR/Raman helpers for MenaCalculator."""
from __future__ import annotations

from typing import Any

import ase.units
import numpy as np
from torch import Tensor

E_ANG_TO_DEBYE = float(1.0 / ase.units.Debye)
# 1 (Debye / (A * sqrt(amu)))^2 = 42.2561 km/mol (CODATA harmonic IR prefactor)
IR_KM_MOL_FACTOR = float((E_ANG_TO_DEBYE ** 2) * 42.2561)


def polarizability_from_sums(iso_sum: float, an_sum: np.ndarray | None) -> tuple[np.ndarray, float, float]:
    """Build [3, 3] polarizability tensor (A^3), isotropic mean, and anisotropy gamma from CPU float64 sums."""
    mat = np.eye(3, dtype=np.float64) * iso_sum
    if an_sum is not None:
        r2, r6 = 0.7071067811865475, 0.408248290463863
        t0, t1, t2, t3, t4 = float(an_sum[0]), float(an_sum[1]), float(an_sum[2]), float(an_sum[3]), float(an_sum[4])
        xx, yy, zz = t3 * r2 - t4 * r6, -t3 * r2 - t4 * r6, 2.0 * r6 * t4
        xy, xz, yz = t0 * r2, t1 * r2, t2 * r2
        mat[0, 0] += xx
        mat[1, 1] += yy
        mat[2, 2] += zz
        mat[0, 1] = mat[1, 0] = xy
        mat[0, 2] = mat[2, 0] = xz
        mat[1, 2] = mat[2, 1] = yz
    mean_iso = float((mat[0, 0] + mat[1, 1] + mat[2, 2]) / 3.0)
    dxy, dyz, dzx = mat[0, 0] - mat[1, 1], mat[1, 1] - mat[2, 2], mat[2, 2] - mat[0, 0]
    off = mat[0, 1] ** 2 + mat[0, 2] ** 2 + mat[1, 2] ** 2
    gamma = float(np.sqrt(max(0.0, 0.5 * (dxy * dxy + dyz * dyz + dzx * dzx + 6.0 * off))))
    return mat, mean_iso, gamma


def polarizability_tensor(alpha_iso: Tensor | None, alpha_aniso: Tensor | None) -> tuple[np.ndarray, float, float]:
    """Build full [3, 3] molecular polarizability tensor (A^3), isotropic mean, and anisotropy gamma (A^3)."""
    if alpha_iso is None:
        return np.zeros((3, 3), dtype=np.float64), 0.0, 0.0
    iso_sum = float(alpha_iso.detach().cpu().double().sum().item())
    an_np = alpha_aniso.detach().cpu().double().sum(dim=0).numpy() if alpha_aniso is not None else None
    return polarizability_from_sums(iso_sum, an_np)


def build_energy_decomposition(
    *,
    kcal_to_ev: float,
    net_kcal: float,
    coul_kcal: float,
    pol_kcal: float,
    rep_kcal: float,
    d3_kcal: float,
    e0_kcal: float,
    eqs_kcal: float,
    term_sums_kcal: dict[str, float],
    ov_kcal: float = 0.0,
    exch_kcal: float = 0.0,
) -> dict[str, Any]:
    """Structured energy breakdown in both eV and kcal/mol (total DFT vs interaction residual)."""
    inter_kcal = net_kcal + coul_kcal + pol_kcal + rep_kcal + ov_kcal + exch_kcal + d3_kcal
    tot_kcal = inter_kcal + e0_kcal + eqs_kcal
    kcal = {
        "total": tot_kcal,
        "interaction": inter_kcal,
        "network": net_kcal,
        "stage0": term_sums_kcal.get("stage0", 0.0),
        "stage1": term_sums_kcal.get("stage1", 0.0),
        "final": term_sums_kcal.get("final", 0.0),
        "iso": term_sums_kcal.get("iso", 0.0),
        "iso_atom": term_sums_kcal.get("iso_atom", 0.0),
        "electrostatic": coul_kcal,
        "polarization": pol_kcal,
        "repulsion": rep_kcal,
        "cloud_overlap": ov_kcal,
        "spin_exchange": exch_kcal,
        "dispersion_d3": d3_kcal,
        "reference_e0": e0_kcal,
        "reference_charge_spin": eqs_kcal,
    }
    return {"eV": {k: float(v * kcal_to_ev) for k, v in kcal.items()}, "kcal_mol": {k: float(v) for k, v in kcal.items()}}


def harmonic_ir_raman(
    eigvecs_mw: np.ndarray,
    masses_amu: np.ndarray,
    dmu_dR: np.ndarray | None,
    dalpha_dR: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Project Cartesian dipole & polarizability Jacobians onto mass-weighted normal modes."""
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(np.maximum(np.asarray(masses_amu, dtype=np.float64), 1e-12), 3))
    q_cart = eigvecs_mw * inv_sqrt_m[None, :]
    ir = None
    if dmu_dR is not None:
        dmu_dq = q_cart @ np.asarray(dmu_dR, dtype=np.float64)
        ir = IR_KM_MOL_FACTOR * np.sum(dmu_dq ** 2, axis=-1)
    raman = None
    if dalpha_dR is not None:
        da_dq = np.einsum("mk,kij->mij", q_cart, np.asarray(dalpha_dR, dtype=np.float64))
        a_bar = (da_dq[:, 0, 0] + da_dq[:, 1, 1] + da_dq[:, 2, 2]) / 3.0
        dxy = da_dq[:, 0, 0] - da_dq[:, 1, 1]
        dyz = da_dq[:, 1, 1] - da_dq[:, 2, 2]
        dzx = da_dq[:, 2, 2] - da_dq[:, 0, 0]
        off = (
            0.5 * (da_dq[:, 0, 1] + da_dq[:, 1, 0]) ** 2
            + 0.5 * (da_dq[:, 0, 2] + da_dq[:, 2, 0]) ** 2
            + 0.5 * (da_dq[:, 1, 2] + da_dq[:, 2, 1]) ** 2
        )
        gamma2 = 0.5 * (dxy ** 2 + dyz ** 2 + dzx ** 2 + 6.0 * off)
        raman = 45.0 * (a_bar ** 2) + 7.0 * gamma2
    return ir, raman
