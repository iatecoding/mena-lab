"""crystal.py: MatFlat crystal pools (lam.data.shards.MatFlat, the reader: MPtrj, sAlex, OMat24, MatPES, OC20, OC22) as static-shape periodic `GraphBatch`es.

Labels in kcal/mol, kcal/mol/A, kcal/mol/A^3 (units `lam.units.EV` = 23.060548 as lam1: shards.py collate, `flat.energy * ev - sum mu[Z]`, `flat.stress * ev`):
  E_res = E_eV * EV - sum_i mu[Z_i]   float64 per-element table of the registry LoT (`load_lot_e0_table_f64(lot, in_kcal=True)`, the table lam1's `model.crystal_e0` row of that LoT);
          u_flag 1 structures use the second source of the pool (mp_pbe / mp_pbeU, omat_pbe / omat_pbeU; MatPES: one LoT twice), which fixes the token and the reference id.
  F = F_eV * EV
  S = stress_eV_A3 * EV   ASE tensile-positive (1/V dE/d eps), NO sign flip: lam1's model computes sym(dE/d eps) / V (lam/models/mena_lam.py:1101-1105) against the same label
      (lam/data/shards.py:231); lam4's `compute_stress` is the same quantity. NaN rows (a source without stress, slabs) are masked (extra["stress_mask"]).
Neighbours with vesin on the host (integer shifts, receiver-sorted, `lam4.data.neighbors.vesin_edges`), pbc True, domain "crystal", no Coulomb cap. Static shapes: n_pad atoms, k graph
slots, e_pad edges (calibrated from probe packs x 1.15, multiple of 4096); a batch keeps the longest prefix of structures that fits, the rest carries over to the next batch.
`CrystalStream` has the stream interface of the sampler (`next() -> Packed`, `state()`, `load()`). One `CrystalSource` may span several flat dirs (OMat24 parts p0..p3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lam.data.registry import load_lot_e0_table_f64
from lam.data.shards import MatFlat
from lam.units import EV
from lam4.data.batch import GraphBatch
from lam4.data.buckets import Packed, source_ids
from lam4.data.neighbors import vesin_edges
from lam4.data.padding import pad_batch
from lam4.registry.sources import _OVERLAY, reference_key_of


class CrystalSource:
    """Structures of one or several MatFlat dirs. `indices` (one dir only) = rows of the flat; positions p enumerate the kept structures (finite energy, <= max_atoms atoms)."""

    def __init__(self, paths: str | Path | list, sources: tuple[str, str] = ("mp_pbe", "mp_pbeU"), indices: np.ndarray | None = None, max_atoms: int = 400) -> None:
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        if indices is not None and len(paths) != 1:
            raise ValueError("indices select rows of ONE flat")
        self.flats = [MatFlat(p) for p in paths]
        self.sources = sources
        self.fixed_u = int(_OVERLAY[sources[0]]["u"]) if sources[0] == sources[1] and sources[0] in _OVERLAY and _OVERLAY[sources[0]]["u"] is not None else None
        fid, idx = [], []
        for k, flat in enumerate(self.flats):
            ix = np.arange(len(flat)) if indices is None else np.asarray(indices, np.int64)
            ix = ix[(ix >= 0) & (ix < len(flat))]   # the registry splits were built on slightly longer flats (MatPES: 116 filtered rows): clipped as lam1 does
            ok = np.isfinite(np.asarray(flat.energy[ix], np.float64)) & (flat.n_atoms[ix] <= max_atoms)
            idx.append(ix[ok])
            fid.append(np.full(int(ok.sum()), k, np.int64))
        self.idx, self.fid = np.concatenate(idx), np.concatenate(fid)
        self.n_at = np.concatenate([f.n_atoms[i] for f, i in zip(self.flats, idx)]).astype(np.int64)
        self.mu = [load_lot_e0_table_f64(reference_key_of(s), num_elements=119, in_kcal=True).numpy() for s in sources]

    def __len__(self) -> int:
        return len(self.idx)

    def structure(self, p: int, r_cut: float) -> dict:
        """One structure at position p: atoms, labels (kcal) and its radius graph."""
        f, i = self.flats[int(self.fid[p])], int(self.idx[p])
        a, b = int(f.offsets[i]), int(f.offsets[i + 1])
        z = np.asarray(f.z[a:b], np.int64)
        pos = np.asarray(f.pos[a:b], np.float64)
        u = int(np.clip(f.u_flag[i], 0, 1)) if self.fixed_u is None else self.fixed_u
        cell = np.asarray(f.cell[i], np.float64)
        snd, rcv, shift = vesin_edges(pos, cell, np.asarray(f.pbc[i]), r_cut)
        e_res = float(f.energy[i]) * EV - float(self.mu[u][np.clip(z, 0, 118)].sum())
        return {"z": z, "pos": pos.astype(np.float32), "forces": (np.asarray(f.forces[a:b], np.float64) * EV).astype(np.float32), "cell": cell.astype(np.float32),
                "snd": snd, "rcv": rcv, "shift": shift, "e": e_res, "u": u, "stress": (np.asarray(f.stress[i], np.float64) * EV).astype(np.float32)}


def pack_crystals(structs: list[dict], source: CrystalSource, n_pad: int, e_pad: int | None, k: int, r_cut: float, device, dtype=torch.float32) -> Packed:
    n_at = np.array([len(s["z"]) for s in structs])
    start = np.cumsum(n_at) - n_at
    snd = np.concatenate([s["snd"] + o for s, o in zip(structs, start)])
    rcv = np.concatenate([s["rcv"] + o for s, o in zip(structs, start)])
    G, N = len(structs), int(n_at.sum())
    u = np.array([s["u"] for s in structs])
    ids = [source_ids(source.sources[x]) for x in u]
    t = lambda a, dt=None, **kw: torch.as_tensor(np.asarray(a), dtype=dt, **kw).to(device)  # noqa: E731
    b = GraphBatch(
        z=t(np.concatenate([s["z"] for s in structs])), pos=t(np.concatenate([s["pos"] for s in structs]), dtype), graph=t(np.repeat(np.arange(G), n_at)),
        cell=t(np.stack([s["cell"] for s in structs]), dtype), edge_index=t(np.stack([snd, rcv])),
        edge_shift=t(np.concatenate([s["shift"] for s in structs]), dtype), charge=torch.zeros(G, dtype=dtype, device=device),
        spin=torch.ones(G, dtype=dtype, device=device), u_flag=t(u), token=t([i[0] for i in ids]), ref_id=t([i[1] for i in ids]), domain="crystal",
        atom_mask=torch.ones(N, dtype=torch.bool, device=device), graph_mask=torch.ones(G, dtype=torch.bool, device=device),
        energy=t([s["e"] for s in structs], dtype), forces=t(np.concatenate([s["forces"] for s in structs]), dtype),
        energy_mask=torch.ones(G, dtype=torch.bool, device=device), force_mask=torch.ones(N, dtype=torch.bool, device=device), coulomb_cap=None, pbc=True)
    if e_pad is not None:
        b = pad_batch(b, n_pad, e_pad, k + 1, r_cut)
    g = b.n_graphs
    st = np.zeros((g, 3, 3), np.float32)
    st[:G] = np.stack([s["stress"] for s in structs])
    st_ok = np.zeros(g, bool)
    st_ok[:G] = np.isfinite(st[:G]).all((1, 2))
    extra = {"dipole": torch.zeros(g, 3, dtype=dtype, device=device), "dipole_mask": torch.zeros(g, dtype=torch.bool, device=device), "seg": torch.arange(g, device=device),
             "stress": t(np.nan_to_num(st), dtype), "stress_mask": t(st_ok)}
    return Packed(b, extra, cap=0, used=G, n_real=N, domain="crystal")


class CrystalStream:
    domain = "crystal"

    def __init__(self, src: CrystalSource, n_pad: int = 8192, k: int = 128, seed: int = 0, r_cut: float = 6.0, device="cpu", e_pad: int | None = None,
                 margin: float = 1.15, n_probe: int = 8) -> None:
        self.src, self.n_pad, self.k, self.r_cut, self.device = src, n_pad, k, r_cut, device
        self.rng = np.random.default_rng(seed)
        self.epoch, self.skipped, self.e_pad = 0, 0, e_pad
        self._new_epoch()
        self.e_pad = e_pad or self._calibrate(margin, n_probe)

    def _new_epoch(self) -> None:
        self.epoch += 1
        self.perm, self.ptr, self.pending = self.rng.permutation(len(self.src)), 0, []   # pending: structures read but not yet used (consecutive in perm)

    def _read(self) -> dict | None:
        if self.pending:
            return self.pending.pop(0)
        while self.ptr < len(self.perm):
            p = int(self.perm[self.ptr])
            self.ptr += 1
            if self.src.n_at[p] <= self.n_pad - 2:
                s = self.src.structure(p, self.r_cut)
                if self.e_pad is not None and len(s["snd"]) > self.e_pad:   # above e_pad on its own: dropped, not a crash mid-run
                    self.skipped += 1
                    continue
                return s
        return None

    def _take(self) -> list[dict]:
        out, n = [], 0
        while len(out) < self.k:
            s = self._read()
            if s is None:
                if out:
                    break
                self._new_epoch()
                continue
            if n + len(s["z"]) > self.n_pad - 2:
                self.pending.insert(0, s)
                break
            out.append(s)
            n += len(s["z"])
        return out

    def _calibrate(self, margin: float, n_probe: int) -> int:
        top = max(sum(len(s["snd"]) for s in self._take()) for _ in range(n_probe))
        self.ptr, self.pending = 0, []
        return int(np.ceil(top * margin / 4096) * 4096)

    def next(self) -> Packed:
        structs = self._take()
        if self.e_pad is not None and len(structs) > 1:   # the longest prefix that fits e_pad; the rest goes back in front, in order
            keep = max(1, int(np.searchsorted(np.cumsum([len(s["snd"]) for s in structs]), self.e_pad, side="right")))
            self.pending = structs[keep:] + self.pending
            structs = structs[:keep]
        return pack_crystals(structs, self.src, self.n_pad, self.e_pad, self.k, self.r_cut, self.device)

    def state(self) -> dict:
        return {"epoch": self.epoch, "rng": self.rng.bit_generator.state, "perm": self.perm.copy(), "ptr": self.ptr - len(self.pending)}

    def load(self, st: dict) -> None:
        self.epoch = st["epoch"]
        self.rng.bit_generator.state = st["rng"]
        perm = np.asarray(st["perm"])
        if len(perm) == len(self.src):
            self.perm, self.ptr, self.pending = perm, st["ptr"], []
        else:
            self.perm, self.ptr, self.pending = self.rng.permutation(len(self.src)), 0, []
