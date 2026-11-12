"""units.py: Leaf module for physical unit constants and graph construction helpers.

Separates `EV`, `MEV`, `cell_volume`, and `build_graph` from `lam.models.mena_lam` so that
importing `lam.models` or `lam.data` in any order never triggers a circular import.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

EV = 23.060548  # kcal/mol per eV
MEV = 1000.0 / EV  # kcal/mol -> meV
KCAL_TO_EV = 1.0 / EV  # kcal/mol -> eV
EV_TO_KCAL = EV  # eV -> kcal/mol
EV_TO_MEV = 1000.0  # eV -> meV
KCAL_TO_MEV = MEV  # kcal/mol -> meV


def convert_energy_forces_stress(
    energy: Tensor | np.ndarray,
    forces: Tensor | np.ndarray | None = None,
    stress: Tensor | np.ndarray | None = None,
    src_units: str = "kcal",
    dst_units: str = "ev",
):
    """Convert (energy, forces, stress) between 'kcal' (kcal/mol, kcal/mol/A, kcal/mol/A^3)
    and 'ev' (eV, eV/A, eV/A^3). Dipoles (e*A) are unit-invariant across both systems."""
    src = src_units.lower()
    dst = dst_units.lower()
    if src not in ("kcal", "ev") or dst not in ("kcal", "ev"):
        raise ValueError(f"Unsupported units transformation {src_units!r} -> {dst_units!r}")
    if src == dst:
        scale = 1.0
    elif src == "kcal" and dst == "ev":
        scale = KCAL_TO_EV
    else:
        scale = EV_TO_KCAL
    e_out = energy * scale
    f_out = forces * scale if forces is not None else None
    s_out = stress * scale if stress is not None else None
    return e_out, f_out, s_out


def cell_volume(cell: Tensor) -> Tensor:
    """|det(cell)| [B] as the triple product a . (b x c)."""
    return (cell[:, 0] * torch.linalg.cross(cell[:, 1], cell[:, 2])).sum(-1).abs()


def build_graph(pos: np.ndarray, cell: np.ndarray, pbc: np.ndarray, cutoff: float):
    """(i, j, S) directed pairs within the cutoff, S integer image offsets (vesin)."""
    import vesin

    nl = vesin.NeighborList(cutoff=float(cutoff), full_list=True)
    p = tuple(bool(x) for x in np.atleast_1d(pbc)) if np.size(pbc) == 3 else (bool(pbc),) * 3
    box = np.asarray(cell, np.float64) if any(p) else np.zeros((3, 3))
    i, j, S = nl.compute(points=np.asarray(pos, np.float64), box=box, periodic=p, quantities="ijS")
    return np.asarray(i, np.int64), np.asarray(j, np.int64), np.asarray(S, np.int64)
