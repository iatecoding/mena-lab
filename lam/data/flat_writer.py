"""flat_writer.py: shared MatFlat writer + per-pool E0 fit for the LAM-2 extra crystal pools (OC20, OC22, OMC25).

MatFlat layout (read by `lam.data.shards.MatFlat`):
  pos f32 [A,3], z int16 [A], forces f32 [A,3], offsets int64 [N+1], n_atoms int32 [N], cell f32 [N,3,3], pbc bool [N,3],
  energy f32 [N] (eV, total DFT energy), stress f32 [N,3,3] + stress_voigt f32 [N,6] (eV/A^3, ASE tensile-positive;
  NaN when the source has no stress: the trainer masks NaN stress), u_flag uint8 [N], mp_id S32 [N] (source frame id).
Frames are buffered in RAM per chunk and spilled to `work_dir/chunk_XXXXX/*.npy`; `finalize()` concatenates the chunks
into open_memmap arrays (no full-dataset RAM copy) and writes manifest.json last.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

ARRAYS_ATOM = {"pos": (np.float32, (3,)), "z": (np.int16, ()), "forces": (np.float32, (3,))}
ARRAYS_FRAME = {
    "cell": (np.float32, (3, 3)),
    "pbc": (np.bool_, (3,)),
    "energy": (np.float32, ()),
    "stress": (np.float32, (3, 3)),
    "stress_voigt": (np.float32, (6,)),
    "u_flag": (np.uint8, ()),
    "mp_id": ("S32", ()),
}


def voigt_from_3x3(s: np.ndarray) -> np.ndarray:
    """[3,3] -> ASE Voigt [xx, yy, zz, yz, xz, xy]."""
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]], dtype=np.float64)


def frame_reject_reason(n: int, energy: float, forces: np.ndarray, fmax_cap: float = 100.0,
                        epa_range: tuple[float, float] = (-120.0, 20.0)) -> str | None:
    """Same sanity filters as the MatPES flat-file builder. Returns None if the frame is kept."""
    if n <= 0:
        return "empty"
    if not np.isfinite(energy) or not (epa_range[0] <= energy / n <= epa_range[1]):
        return "energy"
    if forces.shape != (n, 3) or not np.all(np.isfinite(forces)):
        return "forces_nonfinite"
    if float(np.linalg.norm(forces, axis=1).max()) > fmax_cap:
        return "fmax"
    return None


@dataclass
class FlatWriter:
    out_dir: Path
    chunk_frames: int = 20_000
    fmax_cap: float = 100.0
    n_frames: int = 0
    n_atoms_total: int = 0
    rejects: dict[str, int] = field(default_factory=dict)
    _buf: dict[str, list] = field(default_factory=dict)
    _chunks: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.work = self.out_dir / "_chunks"
        self.work.mkdir(parents=True, exist_ok=True)
        self._reset()

    def _reset(self) -> None:
        self._buf = {k: [] for k in list(ARRAYS_ATOM) + list(ARRAYS_FRAME) + ["n_atoms"]}

    def add(self, *, z, pos, forces, cell, pbc, energy: float, frame_id: str, stress=None, u_flag: int = 0) -> bool:
        z = np.asarray(z, dtype=np.int64).reshape(-1)
        n = len(z)
        forces = np.asarray(forces, dtype=np.float64).reshape(-1, 3) if forces is not None else np.full((n, 3), np.nan)
        why = frame_reject_reason(n, float(energy), forces, self.fmax_cap)
        if why is not None:
            self.rejects[why] = self.rejects.get(why, 0) + 1
            return False
        s33 = np.full((3, 3), np.nan) if stress is None else np.asarray(stress, dtype=np.float64)
        if s33.shape == (6,):  # ASE Voigt -> 3x3
            xx, yy, zz, yz, xz, xy = s33
            s33 = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        b = self._buf
        b["z"].append(z.astype(np.int16))
        b["pos"].append(np.asarray(pos, dtype=np.float32).reshape(n, 3))
        b["forces"].append(forces.astype(np.float32))
        b["cell"].append(np.asarray(cell, dtype=np.float32).reshape(3, 3))
        b["pbc"].append(np.asarray(pbc, dtype=bool).reshape(3))
        b["energy"].append(np.float32(energy))
        b["stress"].append(s33.astype(np.float32))
        b["stress_voigt"].append(voigt_from_3x3(s33).astype(np.float32))
        b["u_flag"].append(np.uint8(u_flag))
        b["mp_id"].append(str(frame_id)[:32].encode("ascii", "replace"))
        b["n_atoms"].append(np.int32(n))
        self.n_frames += 1
        self.n_atoms_total += n
        if len(b["n_atoms"]) >= self.chunk_frames:
            self._spill()
        return True

    def _spill(self) -> None:
        b = self._buf
        if not b["n_atoms"]:
            return
        d = self.work / f"chunk_{len(self._chunks):05d}"
        d.mkdir(parents=True, exist_ok=True)
        for k in ARRAYS_ATOM:
            np.save(d / f"{k}.npy", np.concatenate(b[k], axis=0))
        for k, (dt, _) in ARRAYS_FRAME.items():
            np.save(d / f"{k}.npy", np.asarray(b[k], dtype=dt) if dt == "S32" else np.stack(b[k]).astype(dt))
        np.save(d / "n_atoms.npy", np.asarray(b["n_atoms"], dtype=np.int32))
        self._chunks.append(d)
        self._reset()

    def finalize(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Concatenates chunks into the final MatFlat arrays in out_dir; manifest.json is written last."""
        self._spill()
        N, A = self.n_frames, self.n_atoms_total
        if N == 0:
            raise RuntimeError(f"FlatWriter: no frames kept (rejects {self.rejects})")
        mm = {}
        for k, (dt, sh) in {**ARRAYS_ATOM}.items():
            mm[k] = np.lib.format.open_memmap(self.out_dir / f"{k}.npy", mode="w+", dtype=dt, shape=(A, *sh))
        for k, (dt, sh) in ARRAYS_FRAME.items():
            mm[k] = np.lib.format.open_memmap(self.out_dir / f"{k}.npy", mode="w+", dtype=dt, shape=(N, *sh))
        nat = np.empty(N, dtype=np.int32)
        a0 = f0 = 0
        for d in self._chunks:
            na = np.load(d / "n_atoms.npy")
            nf, at = len(na), int(na.sum())
            for k in ARRAYS_ATOM:
                mm[k][a0:a0 + at] = np.load(d / f"{k}.npy")
            for k in ARRAYS_FRAME:
                mm[k][f0:f0 + nf] = np.load(d / f"{k}.npy")
            nat[f0:f0 + nf] = na
            a0 += at
            f0 += nf
        assert a0 == A and f0 == N, (a0, A, f0, N)
        for m in mm.values():
            m.flush()
        del mm
        np.save(self.out_dir / "n_atoms.npy", nat)
        offsets = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(nat, out=offsets[1:])
        np.save(self.out_dir / "offsets.npy", offsets)
        shutil.rmtree(self.work, ignore_errors=True)
        man = dict(manifest)
        man.update({"num_structures": int(N), "total_atoms": int(A), "rejects": dict(self.rejects),
                    "dtypes": {k: str(np.dtype(v[0])) for k, v in {**ARRAYS_ATOM, **ARRAYS_FRAME}.items()}})
        (self.out_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        return man


def fit_e0_single_lot(flat_dir: Path, train_idx: np.ndarray, max_z: int = 100, chunk: int = 50_000,
                      out_path: Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Float64 per-element least squares E ~ sum_i mu[Z_i] on train_idx (per-atom weighted, as build_matpes_flat.py).
    Saves {"mu": [max_z+1, 2] f32, "mu_f64", "stats"} with both columns equal (registry e0_column 0)."""
    flat_dir = Path(flat_dir)
    z_all = np.load(flat_dir / "z.npy", mmap_mode="r")
    off = np.load(flat_dir / "offsets.npy")
    energy = np.load(flat_dir / "energy.npy", mmap_mode="r")
    P = max_z + 1
    ata = np.zeros((P, P))
    ate = np.zeros(P)
    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))

    def design(c_idx):
        na = (off[c_idx + 1] - off[c_idx]).astype(np.int64)
        zc = np.concatenate([np.asarray(z_all[off[i]:off[i + 1]], dtype=np.int64) for i in c_idx])
        sid = np.repeat(np.arange(len(c_idx)), na)
        A_c = np.bincount(sid * P + np.clip(zc, 0, max_z), minlength=len(c_idx) * P).reshape(len(c_idx), P)
        return A_c.astype(np.float64), na, np.asarray(energy[c_idx], dtype=np.float64)

    for s in range(0, len(train_idx), chunk):
        A_c, na, e = design(train_idx[s:s + chunk])
        w = 1.0 / np.maximum(na, 1)
        Aw = A_c * w[:, None]
        ata += Aw.T @ Aw
        ate += Aw.T @ (e * w)
    active = np.where(np.diag(ata) > 0)[0]
    mu = np.zeros(P)
    mu[active] = np.linalg.lstsq(ata[np.ix_(active, active)], ate[active], rcond=1e-10)[0]
    sa = sq = 0.0
    for s in range(0, len(train_idx), chunk):
        A_c, na, e = design(train_idx[s:s + chunk])
        r = (e - A_c @ mu) / np.maximum(na, 1)
        sa += float(np.abs(r).sum())
        sq += float((r ** 2).sum())
    n = max(len(train_idx), 1)
    stats = {"residual_mae_ev_per_atom": sa / n, "residual_rmse_ev_per_atom": float(np.sqrt(sq / n)),
             "num_structures": int(len(train_idx)), "active_elements": [int(z) for z in active]}
    table = np.stack([mu, mu], axis=1)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mu": torch.from_numpy(table).float(), "mu_f64": torch.from_numpy(table), "stats": stats,
                    "max_z": max_z}, out_path)
    return table, stats


def sync_dir_to_drive(src: Path, dst: Path) -> None:
    """Copies every file (fsync, tmp + rename) with manifest.json LAST, so a reader never sees a partial flat."""
    src, dst = Path(src), Path(dst)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    files = [p for p in files if p.name != "manifest.json"] + [p for p in files if p.name == "manifest.json"]
    for p in files:
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        shutil.copyfile(p, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        tmp.replace(out)
