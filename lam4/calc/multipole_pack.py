"""Packing & slicing helpers for per-atom multipole arrays in MenaCalculator."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from torch import Tensor


MULTIPOLE_PROPERTIES: tuple[str, ...] = (
    "multipoles",
    "atomic_dipoles",
    "atomic_dipoles_perm",
    "atomic_dipoles_ind",
    "electric_field",
    "polarization_atom",
)

_MULTIPOLE_VEC_KEYS: tuple[str, ...] = (
    "atomic_dipoles",
    "atomic_dipoles_perm",
    "atomic_dipoles_ind",
    "electric_field",
)


def wants_multipoles(properties: Sequence[str]) -> bool:
    """Return True if any requested ASE property requires per-atom dipoles/fields."""
    return any(p in properties for p in MULTIPOLE_PROPERTIES)


def append_multipole_flat_parts(flat_parts: list[Tensor], out: dict[str, Any]) -> bool:
    """Append flattened multipole tensors to a single D2H packing list; returns True if present."""
    ad = out.get("atomic_dipoles")
    if ad is None:
        return False
    n = ad.shape[0]
    for k in _MULTIPOLE_VEC_KEYS:
        t = out.get(k)
        flat_parts.append(ad.new_zeros((n, 3)).reshape(-1) if t is None else t.reshape(-1))
    pa = out.get("polarization_atom")
    flat_parts.append(ad.new_zeros(n) if pa is None else pa.reshape(-1))
    return True


def unpack_multipoles_from_host(
    host_buf: np.ndarray,
    cur: int,
    n_atoms: int,
    *,
    has_mp: bool,
    kcal_to_ev: float,
) -> tuple[dict[str, np.ndarray], int]:
    """Unpack per-atom dipoles (e*A), internal field (e/A^2), and polarization energy (scaled by kcal_to_ev)."""
    if not has_mp:
        mp = {k: np.zeros((n_atoms, 3), dtype=np.float64) for k in _MULTIPOLE_VEC_KEYS}
        mp["polarization_atom"] = np.zeros(n_atoms, dtype=np.float64)
        return mp, cur
    mp = {}
    for k in _MULTIPOLE_VEC_KEYS:
        mp[k] = host_buf[cur : cur + 3 * n_atoms].reshape(n_atoms, 3).copy()
        cur += 3 * n_atoms
    mp["polarization_atom"] = host_buf[cur : cur + n_atoms].copy() * kcal_to_ev
    cur += n_atoms
    return mp, cur


def slice_multipoles_dict(
    mp_all: dict[str, np.ndarray],
    s0: int,
    s1: int,
    *,
    kcal_to_ev: float,
) -> dict[str, np.ndarray]:
    """Slice one frame `[s0:s1]` from unpacked microbatch multipole arrays and scale polarization_atom."""
    out = {k: mp_all[k][s0:s1].copy() for k in _MULTIPOLE_VEC_KEYS}
    out["polarization_atom"] = mp_all["polarization_atom"][s0:s1].copy() * kcal_to_ev
    return out
