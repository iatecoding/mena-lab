"""nlh_tables.py: float64 pair tables of the NLH short-range repulsion (numpy, built once, no torch).

Source: Nordlund-Lehtola-Hobler, Phys. Rev. A 111, 032818 (2025): E(r) = k_C Z_i Z_j / r * sum_k a_k exp(-b_k r) for every pair Z = 1..92
(4278 pairs), stored exactly (integers x 1e-5) in `data/nlh_pairs_z92.npy` [4278, 6] = [a0 a1 a2 b0 b1 b2] of the upper triangle (np.triu_indices(92)
order). This replaces the base85 blob of lam/models/nlh_core.py (same integers: tests/lam4/test_repulsion.py compares against it).
Z = 93..127 continue Z = 92 with the Thomas-Fermi factor ((Z_i^0.23 + Z_j^0.23) / (Z_i'^0.23 + Z_j'^0.23)) on b (exactly 1 for Z <= 92). Row / column 0
(pad atoms, Z = 0) is all zero: prefactor 0 and d_off 0, so a pad edge is a physical no-op (exact 0 in value and gradient).

Derived tables, flat index (Z_i << 7) | Z_j, shape [16384, .], built exactly as lam/models/nlh_core._build_tables_f64 (64-step float64 bisection):
  cb   [16384, 6]  k_C Z_i Z_j a_k (kcal/mol A), b_k (1/A)      k_C = 332.06371 kcal A / (mol e^2)
  rmin [16384]     r_min: SO3LR switched NLH reaches 5 eV: w_sw(r; 0, 1.5 A) E_NLH(r) = 5 eV (the switch is used ONLY to solve r_min)
  jet  [16384, 4]  d_off = r_min + 0.05 A, and V, V', V'' of E_NLH at d_off (the 2nd-order Taylor jet that the c2_shift variant subtracts)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

COULOMB_K = 332.06371          # kcal A / (mol e^2), as lam.models.nlh_core and lam4.physics.coulomb_exact
EV_TO_KCAL = 23.060548
FLOOR_ENERGY_EV = 5.0
FLOOR_WIDTH = 0.05             # A, tanh floor width and d_off = r_min + width
R_OFF = 1.5                    # A, SO3LR switch radius (only used to define r_min)
MAX_Z = 92                     # table size; Z in 93..127 use the Thomas-Fermi continuation
N_Z = 128                      # Z = 0..127
_DATA = Path(__file__).resolve().parent / "data" / "nlh_pairs_z92.npy"


def _decode_z92() -> tuple[np.ndarray, np.ndarray]:
    """Symmetric (92, 92, 3) float64 arrays (a, b)."""
    triu = np.load(_DATA).astype(np.float64).reshape(4278, 6) / 1e5
    aa, bb = np.zeros((92, 92, 3)), np.zeros((92, 92, 3))
    iu, iu1 = np.triu_indices(92), np.triu_indices(92, k=1)
    aa[iu], bb[iu] = triu[:, :3], triu[:, 3:]
    aa[iu1[1], iu1[0]], bb[iu1[1], iu1[0]] = aa[iu1], bb[iu1]
    return aa, bb


def _switch(r: np.ndarray) -> np.ndarray:
    """SO3LR C-infinity switch w(r; 0, R_OFF): 1 below 1e-3 R_OFF, 0 above (1 - 1e-3) R_OFF."""
    c = r / R_OFF
    out = np.zeros_like(c)
    out[c <= 1e-3] = 1.0
    m = (c > 1e-3) & (c < 1.0 - 1e-3)
    cm = c[m]
    out[m] = 1.0 / (1.0 + np.exp(-np.clip((1.0 - 2.0 * cm) / (cm * (1.0 - cm)), -700.0, 700.0)))
    return out


def _solve_rmin(c_pref: np.ndarray, bb: np.ndarray) -> np.ndarray:
    target = FLOOR_ENERGY_EV * EV_TO_KCAL
    lo, hi = np.full((N_Z, N_Z), 0.01), np.full((N_Z, N_Z), R_OFF)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        with np.errstate(over="ignore", invalid="ignore"):
            e = np.sum(c_pref * np.exp(-bb * mid[..., None]), axis=-1) / mid * _switch(mid)
        above = np.nan_to_num(e, nan=0.0) > target
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    rmin = 0.5 * (lo + hi)
    rmin[0, :] = rmin[:, 0] = 0.0
    return rmin


def _jet(c_pref: np.ndarray, bb: np.ndarray, d_off: np.ndarray) -> np.ndarray:
    """[16384, 4]: d_off, V(d_off), V'(d_off), V''(d_off) of V(r) = sum_k c_k exp(-b_k r) / r (kcal/mol, 1/A, 1/A^2)."""
    d = np.where(d_off > 0.0, d_off, 1.0)
    ex = np.exp(-bb * d[..., None])
    f0, f1, f2 = np.sum(c_pref * ex, -1), -np.sum(c_pref * bb * ex, -1), np.sum(c_pref * (bb * bb) * ex, -1)
    v0 = np.where(d_off > 0.0, f0 / d, 0.0)
    v1 = np.where(d_off > 0.0, (f1 - v0) / d, 0.0)
    v2 = np.where(d_off > 0.0, (f2 - 2.0 * v1) / d, 0.0)
    return np.stack([d_off, v0, v1, v2], axis=-1).reshape(N_Z * N_Z, 4)


@lru_cache(maxsize=1)
def nlh_tables_f64() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cb [16384, 6], rmin [16384], jet [16384, 4]) float64 numpy, cached."""
    aa92, bb92 = _decode_z92()
    z = np.arange(1, N_Z)
    zc = np.minimum(z, MAX_Z) - 1
    zi, zj = np.meshgrid(zc, zc, indexing="ij")
    lo_, hi_ = np.minimum(zi, zj), np.maximum(zi, zj)
    zf, zfc = z.astype(np.float64), np.minimum(z, MAX_Z).astype(np.float64)
    tf = (zf[:, None] ** 0.23 + zf[None, :] ** 0.23) / (zfc[:, None] ** 0.23 + zfc[None, :] ** 0.23)
    aa, bb = np.zeros((N_Z, N_Z, 3)), np.zeros((N_Z, N_Z, 3))
    aa[1:, 1:], bb[1:, 1:] = aa92[lo_, hi_], bb92[lo_, hi_] * tf[..., None]
    za = np.arange(N_Z, dtype=np.float64)
    c_pref = (COULOMB_K * (za[:, None] * za[None, :]))[..., None] * aa
    rmin = _solve_rmin(c_pref, bb)
    d_off = np.where(rmin > 0.0, rmin + FLOOR_WIDTH, 0.0)
    cb = np.concatenate([c_pref, bb], axis=-1).reshape(N_Z * N_Z, 6)
    return cb, rmin.reshape(N_Z * N_Z), _jet(c_pref, bb, d_off)
