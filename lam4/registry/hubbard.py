"""hubbard.py: Physical (coordination-aware) Materials Project DFT+U detection.

Pure numpy + ASE covalent radii (+ optional `vesin` with pure-NumPy fallback;
never imports torch), so both `lam4` (training/inference) and `mena_app`
(CLI/web analysis) share one single source of truth even in zero-extra envs.
"""
from __future__ import annotations

import sys
import threading
from typing import Any, Sequence

import numpy as np
from ase.data import covalent_radii

# Materials Project MP2020 +U transition metals and anionic chalcogen/halogen triggers.
MP_HUBBARD_METALS = frozenset({23, 24, 25, 26, 27, 28, 42, 74})  # V, Cr, Mn, Fe, Co, Ni, Mo, W
MP_HUBBARD_ANIONS = frozenset({8, 9})                             # O, F

# Metallic host / alloy elements (d-block & post-TM metals; excludes Group 1/2/Ln/Ac ionic cations).
METALLIC_HOST_ELEMENTS = frozenset(
    {13, 31, 49, 50, 81, 82, 83}
    | set(range(21, 31))
    | set(range(39, 49))
    | set(range(72, 81))
)

# First-shell covalent bond tolerances (A) around standard Cordero et al. radii.
ANION_BOND_TOL_A = 0.55
METAL_BOND_TOL_A = 0.35
NEIGHBOR_CUTOFF_A = 3.85

_COV_RADII = np.pad(np.asarray(covalent_radii, dtype=np.float64), (0, 20), constant_values=1.5)
_VESIN_NL: Any = None
_VESIN_NL_TLS = threading.local()


def _get_vesin_nl() -> Any | None:
    global _VESIN_NL
    if "vesin" in sys.modules and sys.modules["vesin"] is None:
        return None
    if _VESIN_NL is False:
        return None
    nl = getattr(_VESIN_NL_TLS, "nl", None)
    if nl is None:
        try:
            import vesin

            nl = vesin.NeighborList(cutoff=NEIGHBOR_CUTOFF_A, full_list=True)
            _VESIN_NL_TLS.nl = nl
        except ImportError:
            _VESIN_NL = False
            return None
    return nl


def _normalize_pbc(pbc: bool | Sequence[bool] | np.ndarray | None) -> tuple[bool, bool, bool]:
    if pbc is None:
        return (False, False, False)
    if isinstance(pbc, (bool, np.bool_)):
        b = bool(pbc)
        return (b, b, b)
    arr = np.asarray(pbc, dtype=bool).reshape(-1)
    if arr.size == 1:
        b = bool(arr[0])
        return (b, b, b)
    return (bool(arr[0]), bool(arr[1]), bool(arr[2]))


def _sanitized_box(
    pos: np.ndarray,
    cell: np.ndarray | None,
    pbc_t: tuple[bool, bool, bool],
    cutoff: float = NEIGHBOR_CUTOFF_A,
) -> np.ndarray:
    if not any(pbc_t):
        return np.zeros((3, 3), dtype=np.float64)
    box = np.zeros((3, 3), dtype=np.float64) if cell is None else np.asarray(cell, dtype=np.float64).copy()
    if all(pbc_t) and abs(float(np.linalg.det(box))) >= 1e-8:
        return box
    span = float(np.ptp(pos)) if len(pos) > 0 else 0.0
    pad_len = max(span + 2.0 * cutoff + 4.0, 20.0)
    basis = [box[ax] / max(float(np.linalg.norm(box[ax])), 1e-12) for ax in range(3) if pbc_t[ax] and np.linalg.norm(box[ax]) >= 1e-6]
    for ax in range(3):
        if not pbc_t[ax] or np.linalg.norm(box[ax]) < 1e-6:
            if len(basis) == 2:
                v = np.cross(basis[0], basis[1])
            elif len(basis) == 1:
                e = np.eye(3)[int(np.argmin(np.abs(basis[0])))]
                v = e - float(np.dot(e, basis[0])) * basis[0]
            else:
                v = np.eye(3)[ax]
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            basis.append(v)
            box[ax] = v * pad_len
    return box


def _edges_numpy(
    points: np.ndarray,
    box: np.ndarray,
    pbc_t: tuple[bool, bool, bool],
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy directed neighbor list `(ii, jj, dd)` within `cutoff` A across open, 3D, and mixed PBC."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0 or (n == 1 and not any(pbc_t)):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    r2 = float(cutoff * cutoff)
    ii_parts: list[np.ndarray] = []
    jj_parts: list[np.ndarray] = []
    dd_parts: list[np.ndarray] = []

    if not any(pbc_t):
        for s in range(0, n, 512):
            e = min(s + 512, n)
            d2 = np.sum((pts[s:e, None, :] - pts[None, :, :]) ** 2, axis=-1)
            idx = np.arange(s, e)
            d2[idx - s, idx] = np.inf
            li, lj = np.nonzero(d2 <= r2)
            if len(li):
                ii_parts.append(li + s)
                jj_parts.append(lj)
                dd_parts.append(np.sqrt(d2[li, lj]))
    else:
        inv_box = np.linalg.inv(box)
        frac = pts @ inv_box
        for ax in range(3):
            if pbc_t[ax]:
                frac[:, ax] %= 1.0
        pts_w = frac @ box
        vol = abs(float(np.linalg.det(box)))
        ranges = []
        for ax in range(3):
            if not pbc_t[ax]:
                ranges.append(np.array([0], dtype=np.int64))
            else:
                area = float(np.linalg.norm(np.cross(box[(ax + 1) % 3], box[(ax + 2) % 3])))
                h = vol / max(area, 1e-12)
                m = int(np.ceil(cutoff / h)) + 1
                ranges.append(np.arange(-m, m + 1, dtype=np.int64))
        shifts = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
        for sv, sh in zip(shifts @ box, shifts):
            zero_shift = bool((sh == 0).all())
            shifted = pts_w + sv
            for s in range(0, n, 512):
                e = min(s + 512, n)
                d2 = np.sum((pts_w[s:e, None, :] - shifted[None, :, :]) ** 2, axis=-1)
                if zero_shift:
                    idx = np.arange(s, e)
                    d2[idx - s, idx] = np.inf
                li, lj = np.nonzero(d2 <= r2)
                if len(li):
                    ii_parts.append(li + s)
                    jj_parts.append(lj)
                    dd_parts.append(np.sqrt(d2[li, lj]))

    if not ii_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return (
        np.concatenate(ii_parts).astype(np.int64, copy=False),
        np.concatenate(jj_parts).astype(np.int64, copy=False),
        np.concatenate(dd_parts).astype(np.float64, copy=False),
    )


def _count_edges_numpy(
    points: np.ndarray,
    box: np.ndarray | None,
    pbc: bool | Sequence[bool] | np.ndarray | None,
    cutoff: float = NEIGHBOR_CUTOFF_A,
) -> int:
    """Pure-NumPy directed neighbor count within `cutoff` A (used when `vesin` is unavailable)."""
    pts = np.asarray(points, dtype=np.float64)
    pbc_t = _normalize_pbc(pbc)
    box_s = _sanitized_box(pts, box, pbc_t, cutoff=float(cutoff))
    return int(len(_edges_numpy(pts, box_s, pbc_t, float(cutoff))[0]))


def mp_hubbard_u_flag(
    z_arr: np.ndarray | Sequence[int],
    positions: np.ndarray | None = None,
    cell: np.ndarray | None = None,
    pbc: bool | Sequence[bool] | np.ndarray | None = None,
) -> int:
    """Return 1 iff the structure represents an oxidized MP DFT+U (O/F + {V,Cr,Mn,Fe,Co,Ni,Mo,W}) system.

    When `positions` is provided, requires local M-(O/F) coordination rather than mere formula co-occurrence,
    preventing metal/alloy catalyst slabs with adsorbed oxygenates (e.g. Ni(100)+CO, Fe(110)+H2O, CuFe+O*)
    or organometallic carbonyls from triggering +U reference shifts and violating the slab+gas separable limit.
    """
    z = np.asarray(z_arr, dtype=np.int64)
    zs = {int(v) for v in z.tolist() if v > 0}
    if not ((zs & MP_HUBBARD_ANIONS) and (zs & MP_HUBBARD_METALS)):
        return 0
    if positions is None:
        return 1

    pos = np.asarray(positions, dtype=np.float64)
    if len(z) == 0 or pos.shape[0] != len(z):
        return 0
    pbc_t = _normalize_pbc(pbc)
    box = _sanitized_box(pos, cell, pbc_t, cutoff=NEIGHBOR_CUTOFF_A)
    nl = _get_vesin_nl()
    if nl is not None:
        if any(pbc_t):
            i_raw, j_raw, d_raw = nl.compute(points=pos, box=box, periodic=pbc_t, quantities="ijd")
        else:
            i_raw, j_raw, d_raw = nl.compute(points=pos, box=box, periodic=False, quantities="ijd")
        ii = np.asarray(i_raw, dtype=np.int64)
        jj = np.asarray(j_raw, dtype=np.int64)
        dd = np.asarray(d_raw, dtype=np.float64)
    else:
        ii, jj, dd = _edges_numpy(pos, box, pbc_t, NEIGHBOR_CUTOFF_A)

    u_mask = np.isin(z, tuple(MP_HUBBARD_METALS))
    if len(ii) == 0:
        return 0

    zi, zj = z[ii], z[jj]
    max_z = len(_COV_RADII) - 1
    rc_sum = _COV_RADII[np.clip(zi, 0, max_z)] + _COV_RADII[np.clip(zj, 0, max_z)]
    is_u_center = np.isin(zi, tuple(MP_HUBBARD_METALS))

    anion_bonds = is_u_center & np.isin(zj, tuple(MP_HUBBARD_ANIONS)) & (dd <= rc_sum + ANION_BOND_TOL_A)
    metal_bonds = is_u_center & np.isin(zj, tuple(METALLIC_HOST_ELEMENTS)) & (dd <= rc_sum + METAL_BOND_TOL_A)

    cn_anion = np.bincount(ii[anion_bonds], minlength=len(z))
    cn_metal = np.bincount(ii[metal_bonds], minlength=len(z))

    u_anion = cn_anion[u_mask]
    u_metal = cn_metal[u_mask]
    frac_coordinated = float((u_anion > 0).mean())
    mean_cn_anion = float(u_anion.mean())
    mean_cn_metal = float(u_metal.mean())

    return 1 if (frac_coordinated >= 0.5 and (mean_cn_anion >= 2.0 or (mean_cn_anion >= 1.0 and mean_cn_anion >= mean_cn_metal))) else 0
