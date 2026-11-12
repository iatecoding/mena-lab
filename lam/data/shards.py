"""shards.py: Memory-mapped flat layout and fast pre-allocated collation for periodic crystal datasets."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from lam.units import EV, build_graph

U_ELEMENTS = frozenset({23, 24, 25, 26, 27, 28, 42, 74})
O_OR_F = frozenset({8, 9})


def compute_u_flag(atomic_numbers: Sequence[int] | np.ndarray | Tensor) -> int:
    """u = 1 if structure contains (O or F) AND any of {Co, Cr, Fe, Mn, Mo, Ni, V, W}, else 0."""
    if isinstance(atomic_numbers, Tensor):
        nums = set(atomic_numbers.detach().cpu().numpy().tolist())
    elif isinstance(atomic_numbers, np.ndarray):
        nums = set(atomic_numbers.tolist())
    else:
        nums = set(int(x) for x in atomic_numbers)
    return 1 if (nums & O_OR_F) and (nums & U_ELEMENTS) else 0


def voigt_from_3x3(stress_3x3: np.ndarray) -> np.ndarray:
    """Converts [..., 3, 3] stress tensor to ASE Voigt 6-vector [..., 6] ([xx, yy, zz, yz, xz, xy])."""
    s = stress_3x3
    return np.stack(
        [s[..., 0, 0], s[..., 1, 1], s[..., 2, 2], s[..., 1, 2], s[..., 0, 2], s[..., 0, 1]],
        axis=-1,
    )


class _ConcatMmap:
    """Zero-copy virtual concatenation of two memory-mapped arrays along axis 0."""

    def __init__(self, a: np.ndarray, b: np.ndarray) -> None:
        self.a = a
        self.b = b
        self.n_a = len(a)
        self.shape = (len(a) + len(b), *a.shape[1:])
        self.dtype = a.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            i = int(idx)
            if i < 0:
                i += self.shape[0]
            return self.a[i] if i < self.n_a else self.b[i - self.n_a]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            if step == 1:
                if stop <= self.n_a:
                    return self.a[start:stop]
                if start >= self.n_a:
                    return self.b[start - self.n_a : stop - self.n_a]
                return np.concatenate([self.a[start:], self.b[: stop - self.n_a]], axis=0)
            idx = np.arange(start, stop, step, dtype=np.int64)
        idx_arr = np.asarray(idx, dtype=np.int64)
        m = idx_arr < self.n_a
        if m.all():
            return self.a[idx_arr]
        if not m.any():
            return self.b[idx_arr - self.n_a]
        out = np.empty((len(idx_arr), *self.a.shape[1:]), dtype=self.dtype)
        out[m] = self.a[idx_arr[m]]
        out[~m] = self.b[idx_arr[~m] - self.n_a]
        return out


class MatFlat:
    """Memory-mapped flat materials dataset for high-throughput training and evaluation."""

    def __init__(self, path: str | Path) -> None:
        self._sub_flats: list[MatFlat] = []
        if isinstance(path, str) and "," in path:
            parts = [p.strip() for p in path.split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError("Comma-separated MatFlat currently supports exactly 2 directories ('dir1,dir2')")
            f1, f2 = MatFlat(parts[0]), MatFlat(parts[1])
            self._sub_flats = [f1, f2]
            self.path = f1.path
            self.manifest = {"sub_dirs": parts}
            self.pos = _ConcatMmap(f1.pos, f2.pos)
            self.z = _ConcatMmap(f1.z, f2.z)
            self.forces = _ConcatMmap(f1.forces, f2.forces)
            self.cell = _ConcatMmap(f1.cell, f2.cell)
            self.pbc = _ConcatMmap(f1.pbc, f2.pbc)
            self.energy = _ConcatMmap(f1.energy, f2.energy)
            self.stress = _ConcatMmap(f1.stress, f2.stress)
            self.stress_voigt = (
                _ConcatMmap(f1.stress_voigt, f2.stress_voigt)
                if (f1.stress_voigt is not None and f2.stress_voigt is not None)
                else None
            )
            self.u_flag = _ConcatMmap(f1.u_flag, f2.u_flag)
            self.mp_id = _ConcatMmap(f1.mp_id, f2.mp_id)
            self.offsets = np.concatenate([f1.offsets, int(f1.offsets[-1]) + f2.offsets[1:]])
            self.n_atoms = np.concatenate([f1.n_atoms, f2.n_atoms])
            self.num_structures = len(self.offsets) - 1
            return

        self.path = Path(path)
        if not (self.path / "manifest.json").exists() and not (self.path / "pos.npy").exists():
            raise FileNotFoundError(f"No manifest.json or pos.npy found in {self.path}")

        self.manifest = (
            json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
            if (self.path / "manifest.json").exists()
            else {}
        )

        ld = lambda k: np.load(self.path / f"{k}.npy", mmap_mode="r")  # noqa: E731
        self.pos = ld("pos")
        self.z = ld("z")
        self.forces = ld("forces")
        self.cell = ld("cell")
        self.pbc = ld("pbc")
        self.energy = ld("energy")
        self.stress = ld("stress")
        self.stress_voigt = ld("stress_voigt") if (self.path / "stress_voigt.npy").exists() else None
        self.u_flag = ld("u_flag")
        self.mp_id = ld("mp_id") if (self.path / "mp_id.npy").exists() else None

        self.offsets = np.load(self.path / "offsets.npy")
        self.n_atoms = np.load(self.path / "n_atoms.npy") if (self.path / "n_atoms.npy").exists() else np.diff(self.offsets)
        self.num_structures = len(self.offsets) - 1

    def __len__(self) -> int:
        return self.num_structures

    def __getitem__(self, idx: int) -> dict[str, Tensor | int | str]:
        if idx < 0:
            idx += self.num_structures
        if idx < 0 or idx >= self.num_structures:
            raise IndexError(f"Index {idx} out of range [0, {self.num_structures})")

        a, b = int(self.offsets[idx]), int(self.offsets[idx + 1])
        if self.mp_id is not None:
            mp_id_raw = self.mp_id[idx]
            mp_id_str = mp_id_raw.decode("ascii") if isinstance(mp_id_raw, bytes) else str(mp_id_raw)
        else:
            mp_id_str = f"frame-{idx}"

        return {
            "pos": torch.from_numpy(np.array(self.pos[a:b], dtype=np.float32)),
            "z": torch.from_numpy(np.array(self.z[a:b], dtype=np.int64)),
            "forces": torch.from_numpy(np.array(self.forces[a:b], dtype=np.float32)),
            "cell": torch.from_numpy(np.array(self.cell[idx], dtype=np.float32)),
            "pbc": torch.from_numpy(np.array(self.pbc[idx], dtype=bool)),
            "energy": torch.tensor(float(self.energy[idx]), dtype=torch.float32),
            "stress": torch.from_numpy(np.array(self.stress[idx], dtype=np.float32)),
            "stress_voigt": (
                torch.from_numpy(np.array(self.stress_voigt[idx], dtype=np.float32))
                if self.stress_voigt is not None
                else torch.zeros(6)
            ),
            "u_flag": torch.tensor(int(self.u_flag[idx]), dtype=torch.int64),
            "n_atoms": int(self.n_atoms[idx]),
            "mp_id": mp_id_str,
        }

    def close(self) -> None:
        for sf in getattr(self, "_sub_flats", ()):
            sf.close()
        for attr in ("pos", "z", "forces", "cell", "pbc", "energy", "stress", "stress_voigt", "u_flag", "mp_id"):
            m = getattr(self, attr, None)
            if hasattr(m, "_mmap") and m._mmap is not None:
                try:
                    m._mmap.close()
                except Exception:
                    pass

    def __enter__(self) -> MatFlat:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def collate_legacy(
    flat: MatFlat,
    idx: np.ndarray,
    cutoff: float,
    mu_kcal: np.ndarray,
    ev: float = EV,
    lot_id_by_u: tuple[int, int] | dict[int, int] | str | int | None = None,
) -> dict:
    """Original list-append + concatenate collate from mat/mena/periodic.py (kept as benchmark baseline)."""
    from lam.data.registry import resolve_lot_id_by_u

    zs, ps, fs, bs, ei, es = [], [], [], [], [], []
    off = 0
    for b, s in enumerate(idx):
        a0, a1 = int(flat.offsets[s]), int(flat.offsets[s + 1])
        pos = np.asarray(flat.pos[a0:a1], np.float64)
        i, j, S = build_graph(pos, np.asarray(flat.cell[s]), np.asarray(flat.pbc[s]), cutoff)
        zs.append(np.asarray(flat.z[a0:a1], np.int64))
        ps.append(pos)
        fs.append(np.asarray(flat.forces[a0:a1]))
        bs.append(np.full(a1 - a0, b, np.int64))
        ei.append(np.stack([i, j]) + off)
        es.append(S)
        off += a1 - a0
    idx = np.asarray(idx)
    zc, bc = np.concatenate(zs), np.concatenate(bs)
    u = np.asarray(flat.u_flag[idx], np.int64)
    id0, id1 = resolve_lot_id_by_u(lot_id_by_u)
    lot_id_arr = np.where(np.clip(u, 0, 1) == 1, id1, id0).astype(np.int64)
    ref = np.bincount(bc, weights=mu_kcal[np.clip(zc, 0, len(mu_kcal) - 1), u[bc]], minlength=len(idx))
    e_res = np.asarray(flat.energy[idx], np.float64) * ev - ref
    return {
        "e_res": torch.from_numpy(e_res.astype(np.float32)),
        "z": torch.from_numpy(np.concatenate(zs)),
        "pos": torch.from_numpy(np.concatenate(ps).astype(np.float32)),
        "forces": torch.from_numpy((np.concatenate(fs) * ev).astype(np.float32)),
        "batch": torch.from_numpy(np.concatenate(bs)),
        "edge_index": torch.from_numpy(np.concatenate(ei, 1)),
        "edge_shift": torch.from_numpy(np.concatenate(es)),
        "cell": torch.from_numpy(np.asarray(flat.cell[idx], np.float32)),
        "energy": torch.from_numpy(np.asarray(flat.energy[idx], np.float64) * ev),
        "stress": torch.from_numpy((np.asarray(flat.stress[idx]) * ev).astype(np.float32)),
        "u": torch.from_numpy(u),
        "lot_id": torch.from_numpy(lot_id_arr),
        "n_atoms": torch.from_numpy(np.asarray(flat.offsets)[idx + 1] - np.asarray(flat.offsets)[idx]),
    }


def collate(
    flat: MatFlat,
    idx: np.ndarray,
    cutoff: float,
    mu_kcal: np.ndarray | Tensor,
    ev: float = EV,
    pin_memory: bool = False,
    lot_id: int | tuple[int, int] | dict[int, int] | str | np.ndarray | Tensor | None = None,
    lot_id_by_u: tuple[int, int] | dict[int, int] | str | int | None = None,
    units: str | None = None,
    mu_in_kcal: bool | None = None,
) -> dict:
    """Fast pre-allocated contiguous NumPy collation with single-instance vesin.NeighborList,
    per-structure `lot_id` stamped from `u_flag` via `lot_id_by_u`, float64 per-LoT E0 subtraction,
    precomputed Cartesian edge shifts (`edge_shift_cart`), optional `units="ev" | "kcal"`, and optional pinned memory."""
    import vesin
    from lam.data.registry import resolve_lot_id_by_u

    if units is not None:
        u_norm = units.lower()
        if u_norm not in ("ev", "kcal"):
            raise ValueError(f"Unsupported units={units!r}; expected 'ev' or 'kcal'")
        ev_mult = 1.0 if u_norm == "ev" else float(EV)
    else:
        ev_mult = float(ev)
        u_norm = "ev" if abs(ev_mult - 1.0) < 1e-6 else "kcal"

    idx = np.asarray(idx, dtype=np.int64)
    B = len(idx)
    a0_arr = flat.offsets[idx]
    a1_arr = flat.offsets[idx + 1]
    n_atoms = (a1_arr - a0_arr).astype(np.int64)
    total_atoms = int(n_atoms.sum())

    z_arr = np.empty(total_atoms, dtype=np.int64)
    pos_arr = np.empty((total_atoms, 3), dtype=np.float32)
    forces_arr = np.empty((total_atoms, 3), dtype=np.float32)
    batch_arr = np.repeat(np.arange(B, dtype=np.int64), n_atoms)

    cells_f64 = np.asarray(flat.cell[idx], dtype=np.float64)
    cells_f32 = cells_f64.astype(np.float32)
    pbcs = np.asarray(flat.pbc[idx])

    nl = vesin.NeighborList(cutoff=float(cutoff), full_list=True)
    i_list, j_list, s_list = [], [], []
    edge_counts = np.empty(B, dtype=np.int64)
    off = 0
    zero_box = np.zeros((3, 3), dtype=np.float64)
    for b in range(B):
        a0, a1 = int(a0_arr[b]), int(a1_arr[b])
        na = a1 - a0
        pos_b = flat.pos[a0:a1]
        z_arr[off : off + na] = flat.z[a0:a1]
        pos_arr[off : off + na] = pos_b
        forces_arr[off : off + na] = flat.forces[a0:a1]
        pbc_b = pbcs[b]
        p = (bool(pbc_b[0]), bool(pbc_b[1]), bool(pbc_b[2]))
        box = cells_f64[b] if (p[0] or p[1] or p[2]) else zero_box
        i, j, S = nl.compute(points=np.asarray(pos_b, np.float64), box=box, periodic=p, quantities="ijS")
        i_i64 = np.asarray(i, np.int64)
        j_i64 = np.asarray(j, np.int64)
        if off:
            i_i64 = i_i64 + off
            j_i64 = j_i64 + off
        i_list.append(i_i64)
        j_list.append(j_i64)
        s_list.append(np.asarray(S, np.int64))
        edge_counts[b] = len(i_i64)
        off += na

    forces_arr *= float(ev_mult)

    total_edges = int(edge_counts.sum())
    edge_index = np.empty((2, total_edges), dtype=np.int64)
    edge_shift = np.empty((total_edges, 3), dtype=np.int64)
    edge_shift_cart = np.empty((total_edges, 3), dtype=np.float32)
    e_off = 0
    for b in range(B):
        ne = int(edge_counts[b])
        if ne:
            sb = s_list[b]
            edge_index[0, e_off : e_off + ne] = i_list[b]
            edge_index[1, e_off : e_off + ne] = j_list[b]
            edge_shift[e_off : e_off + ne] = sb
            edge_shift_cart[e_off : e_off + ne] = sb.astype(np.float32) @ cells_f32[b]
            e_off += ne

    u = np.asarray(flat.u_flag[idx], dtype=np.int64)
    u_clipped = np.clip(u, 0, 1)
    if lot_id_by_u is not None:
        id0, id1 = resolve_lot_id_by_u(lot_id_by_u)
        lot_id_arr = np.where(u_clipped == 1, id1, id0).astype(np.int64)
    elif lot_id is None:
        id0, id1 = resolve_lot_id_by_u(None)
        lot_id_arr = np.where(u_clipped == 1, id1, id0).astype(np.int64)
    elif isinstance(lot_id, (str, dict, tuple, list)):
        id0, id1 = resolve_lot_id_by_u(lot_id)
        lot_id_arr = np.where(u_clipped == 1, id1, id0).astype(np.int64)
    elif isinstance(lot_id, (int, np.integer)):
        id0, id1 = resolve_lot_id_by_u(int(lot_id))
        lot_id_arr = np.where(u_clipped == 1, id1, id0).astype(np.int64)
    else:
        lot_arr_raw = lot_id.detach().cpu().numpy() if torch.is_tensor(lot_id) else np.asarray(lot_id)
        if lot_arr_raw.ndim == 0 or (lot_arr_raw.size == 1 and B > 1):
            id0, id1 = resolve_lot_id_by_u(int(lot_arr_raw.item()))
            lot_id_arr = np.where(u_clipped == 1, id1, id0).astype(np.int64)
        else:
            lot_id_arr = lot_arr_raw.astype(np.int64).reshape(B)

    mu_f64 = (
        mu_kcal.detach().cpu().to(torch.float64).numpy()
        if torch.is_tensor(mu_kcal)
        else np.asarray(mu_kcal, dtype=np.float64)
    )
    if units is not None:
        if mu_in_kcal is not None:
            is_kcal = bool(mu_in_kcal)
        elif mu_f64.size == 0:
            is_kcal = True
        elif mu_f64.ndim == 2 and mu_f64.shape[0] >= 6 and mu_f64.shape[1] >= 9:
            # Multi-LoT [num_lots, Z_max] table: inspect crystal rows (0..5) so row 6 (omol_wb97mv) does not skew eV detection.
            # Crystal E0 tables in eV have max_abs <= 87 eV (r2SCAN actinides), whereas kcal/mol tables have max_abs >= 309 kcal/mol.
            is_kcal = float(np.nanmax(np.abs(mu_f64[:6]))) > 120.0
        else:
            is_kcal = float(np.nanmax(np.abs(mu_f64))) > 120.0
        if u_norm == "ev" and is_kcal:
            mu_f64 = mu_f64 / float(EV)
        elif u_norm == "kcal" and not is_kcal:
            mu_f64 = mu_f64 * float(EV)

    if mu_f64.ndim == 2 and mu_f64.shape[0] <= 32 and mu_f64.shape[1] >= 95:
        # Full [num_lots, Z_max] float64 E0 table (e.g. model.crystal_e0 in kcal/mol)
        z_clip = np.clip(z_arr, 0, mu_f64.shape[1] - 1)
        lot_clip = np.clip(lot_id_arr[batch_arr], 0, mu_f64.shape[0] - 1)
        atom_e0 = mu_f64[lot_clip, z_clip]
    elif mu_f64.ndim == 2 and mu_f64.shape[1] == 2:
        # [Z_max, 2] table indexed by (z, u_flag)
        z_clip = np.clip(z_arr, 0, mu_f64.shape[0] - 1)
        atom_e0 = mu_f64[z_clip, u_clipped[batch_arr]]
    elif mu_f64.ndim == 2:
        # [Z_max, num_lots] table
        z_clip = np.clip(z_arr, 0, mu_f64.shape[0] - 1)
        lot_clip = np.clip(lot_id_arr[batch_arr], 0, mu_f64.shape[1] - 1)
        atom_e0 = mu_f64[z_clip, lot_clip]
    else:
        z_clip = np.clip(z_arr, 0, len(mu_f64) - 1)
        atom_e0 = mu_f64[z_clip]

    ref = np.bincount(batch_arr, weights=atom_e0, minlength=B)
    energy_f64 = np.asarray(flat.energy[idx], dtype=np.float64) * float(ev_mult)
    e_res = (energy_f64 - ref).astype(np.float32)
    stress_f32 = (np.asarray(flat.stress[idx], dtype=np.float32) * float(ev_mult))

    out = {
        "e_res": torch.from_numpy(e_res),
        "z": torch.from_numpy(z_arr),
        "pos": torch.from_numpy(pos_arr),
        "forces": torch.from_numpy(forces_arr),
        "batch": torch.from_numpy(batch_arr),
        "edge_index": torch.from_numpy(edge_index),
        "edge_shift": torch.from_numpy(edge_shift),
        "edge_shift_cart": torch.from_numpy(edge_shift_cart),
        "cell": torch.from_numpy(cells_f32),
        "energy": torch.from_numpy(energy_f64),
        "stress": torch.from_numpy(stress_f32),
        "u": torch.from_numpy(u),
        "lot_id": torch.from_numpy(lot_id_arr),
        "n_atoms": torch.from_numpy(n_atoms),
        "units": u_norm,
        "energy_unit": "eV" if u_norm == "ev" else "kcal/mol",
    }
    if pin_memory and torch.cuda.is_available():
        out = {k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in out.items()}
    return out

