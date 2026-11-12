"""omol.py: OMol25 flats (lam.data.omol_buckets.OmolFlat, the reader) as labelled structures in size buckets for the lam4 loader.

ENERGY TARGET = the molecular training energy space, in float64 on the host, cast to float32 at the end (kcal/mol):
    E_res = E_DFT - E_D3(BJ, wB97M) - sum_i e0[Z_i] - E_Q[Q] - E_S[S]
e0 [Z], E_Q [Q = -10..10], E_S [S = 1..11] from data/e0_reference_omol25.json (the table the molecular model trained with: its fit residual MAE is 103.5 kcal/mol, which this reproduces
on train_4M_sub20k). The model is called with add_ref=False and predicts E_res; the isolated-atom part of e0 is also learnable in the readout (`iso`), E_Q + E_S are the frozen group
reference `lam4.physics.GroupReference` (`add_ref=True` adds exactly them). One charge group per structure (total Q, S), as before, without --multi-island fragments.
FORCE TARGET = F_DFT - F_D3 (kcal/mol/A); D3 labels (energy + forces) from `lam.data.omol_buckets.d3_labels` (cache `d3_<sig>.pt` next to the flat, or computed).
Dipole label (e A) when the flat has dipole.npy, else NaN (masked in the loss). Structures with non-finite energy or more than the largest bucket cap atoms are dropped.
Adapted from the earlier Mena data pipeline (reference_energy, OmolBuckets) without the dense padded device buckets: rows are gathered per batch from the mmap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lam.data.omol_buckets import OmolFlat, d3_labels, d3_sig, default_subset, gather_atoms
from lam4.physics.group_context import DEFAULT_REF

EDGES = (24, 48, 80, 128, 192, 350)


def load_reference(path: str | Path = DEFAULT_REF) -> dict[str, np.ndarray]:
    """e0 [119], e_q [21] (index Q + 10), e_s [11] (index S - 1), float64 kcal/mol."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    e0, e_q, e_s = np.zeros(119), np.zeros(21), np.zeros(11)
    for z, v in d["e0_kcal"].items():
        e0[int(z)] = v
    for q, v in d["charge_kcal"].items():
        e_q[int(q) + 10] = v
    for s, v in d["spin_kcal"].items():
        e_s[int(s) - 1] = v
    return {"e0": e0, "e_q": e_q, "e_s": e_s}


def reference_energy(z_atoms: np.ndarray, starts: np.ndarray, n_atoms: np.ndarray, charge: np.ndarray, spin: np.ndarray, ref: dict) -> np.ndarray:
    """sum_i e0[Z_i] + E_Q[Q] + E_S[S] per structure, float64. z_atoms: atoms of the structures concatenated, starts: first atom of each."""
    e = np.add.reduceat(ref["e0"][z_atoms.astype(np.int64)], starts) if len(z_atoms) else np.zeros(len(n_atoms))
    q, s = np.asarray(charge).astype(np.int64), np.asarray(spin).astype(np.int64)
    eq = np.where((q >= -10) & (q <= 10), ref["e_q"][np.clip(q + 10, 0, 20)], 0.0)
    es = np.where((s >= 1) & (s <= 11), ref["e_s"][np.clip(s - 1, 0, 10)], 0.0)
    return e + eq + es


class OmolSource:
    """Labelled OMol structures in size buckets. `rows` of the flat are indexed by POSITION p in `self.idx`; `buckets[cap]` = positions with n_atoms in (prev cap, cap]."""

    def __init__(self, path: str | Path, *, indices: np.ndarray | None = None, max_structs: int = 0, seed: int = 0, edges: tuple[int, ...] = EDGES,
                 ref_json: str | Path = DEFAULT_REF, d3_cache: str | Path | None = None, d3_compute: bool = True,
                 device: torch.device = torch.device("cpu")) -> None:
        self.flat = flat = OmolFlat(path)
        self.edges = tuple(int(e) for e in edges)
        if indices is None:
            idx = default_subset(flat, max_structs, seed, self.edges[-1])
        else:
            idx = np.asarray(indices, np.int64)
            idx = idx[np.isfinite(np.asarray(flat.energy[idx])) & (flat.n_atoms[idx] <= self.edges[-1])]
            if max_structs and len(idx) > max_structs:
                idx = np.sort(np.random.default_rng(seed + 777).choice(idx, max_structs, replace=False))
        self.idx = idx
        self.n_at = flat.n_atoms[idx].astype(np.int64)
        full = default_subset(flat, 0, 0, self.edges[-1])   # a subset (or a val split) reuses the labels cached for the whole flat
        reuse = [(full, flat.path / f"d3_{d3_sig(flat, full)}.pt")]
        self.d3_e, self.d3_f, self.d3_off = d3_labels(flat, idx, device, d3_cache, reuse, compute=d3_compute)
        self.charge = np.asarray(flat.charge[idx]).astype(np.float64)
        self.spin = np.asarray(flat.spin[idx]).astype(np.float64)
        atoms, starts = gather_atoms(flat, idx)
        e_ref = reference_energy(np.asarray(flat.z[atoms]), starts, self.n_at, self.charge, self.spin, load_reference(ref_json))
        self.e_res = np.asarray(flat.energy[idx], np.float64) - np.asarray(self.d3_e, np.float64) - e_ref   # float64 on the host, cast per batch
        self.dipole = None if flat.dipole is None else np.asarray(flat.dipole[idx], np.float32)
        which = np.searchsorted(np.asarray(self.edges), self.n_at)
        self.buckets = {c: np.where(which == b)[0] for b, c in enumerate(self.edges) if (which == b).any()}
        self._groups: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.idx)

    def group_ids(self) -> np.ndarray:
        """Composition group (element counts, Q, S) of every position, for the group-relative energy term."""
        if self._groups is None:
            from lam.data.omol_buckets import composition_key
            self._groups = np.unique(composition_key(self.flat, self.idx), return_inverse=True)[1].ravel().astype(np.int64)
        return self._groups

    def gather(self, cap: int, pos: np.ndarray) -> dict[str, np.ndarray]:
        """Dense host arrays of the structures at positions `pos` (all with n_atoms <= cap): z [K, cap] (0 = pad), pos [K, cap, 3], forces [K, cap, 3] (D3 removed),
        n [K], energy [K] float64, charge, spin [K], dipole [K, 3] (NaN without labels)."""
        pos = np.asarray(pos, np.int64)
        n = self.n_at[pos]
        atoms, starts = gather_atoms(self.flat, self.idx[pos])
        d3_atoms = np.repeat(self.d3_off[pos] - starts, n) + np.arange(int(n.sum()))
        valid = np.arange(cap)[None, :] < n[:, None]
        z = np.zeros((len(pos), cap), np.int64)
        p = np.zeros((len(pos), cap, 3), np.float32)
        f = np.zeros((len(pos), cap, 3), np.float32)
        z[valid] = np.asarray(self.flat.z[atoms])
        p[valid] = np.asarray(self.flat.pos[atoms])
        f[valid] = np.asarray(self.flat.forces[atoms], np.float32) - self.d3_f[d3_atoms]
        dip = np.full((len(pos), 3), np.nan, np.float32) if self.dipole is None else self.dipole[pos]
        return {"z": z, "pos": p, "forces": f, "n": n, "energy": self.e_res[pos], "charge": self.charge[pos], "spin": self.spin[pos], "dipole": dip}
