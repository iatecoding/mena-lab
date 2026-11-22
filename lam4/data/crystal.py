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
import threading

import numpy as np
import torch

from lam.data.registry import load_lot_e0_table_f64
from lam.data.shards import MatFlat
from lam.units import EV
from lam4.data.batch import GraphBatch
from lam4.data.buckets import Packed, source_ids
from lam4.data.neighbors import vesin_edges
from lam4.data.padding import pad_batch
from lam4.registry.hubbard import (
    _COV_RADII,
    ANION_BOND_TOL_A,
    MP_HUBBARD_ANIONS,
    MP_HUBBARD_METALS,
    NEIGHBOR_CUTOFF_A,
    mp_hubbard_u_flag,
)
import lam4.registry.tokens as _tokens
from lam4.registry.sources import _OVERLAY, reference_key_of

_XC_TOKENS = getattr(_tokens, "XC_TOKENS", (
    "pbe", "r2scan", "rpbe", "wb97mv", "wb97xd", "wb97m_d3bj",
    "wb97x", "b3lyp_d3", "pbe0", "tpssh_d3bj", "sapt0",
))
_PAW_BASES = getattr(_tokens, "PAW_BASES", (
    "vasp_pbe52", "vasp_pbe54", "vasp_pbe64", "aims_tight", "def2_tzvpd",
    "def2_tzvppd", "pop_631gd", "def2_tzvp", "def2_svp", "aug_cc_pvddz", "jun_cc_pvdz",
))
XC_ID = getattr(_tokens, "XC_ID", {t: i for i, t in enumerate(_XC_TOKENS)})
PAW_ID = getattr(_tokens, "PAW_ID", {b: i for i, b in enumerate(_PAW_BASES)})

_EXTENDED_OVERLAY: dict[str, dict] = {
    "oc25_rpbe": dict(token="rpbe", u=0, ref="oc25_rpbe", fallback_src="oc20_rpbe", xc="rpbe", paw="vasp_pbe54"),
    "mad16_r2scan": dict(token="r2scan", u=0, ref="mad16_r2scan", fallback_src="matpes_r2scan", xc="r2scan", paw="aims_tight"),
    "mad16_pbe": dict(token="pbe", u=0, ref="mad16_pbe", fallback_src="matpes_pbe", xc="pbe", paw="aims_tight"),
}
_SOURCE_PAW = {
    "mp_pbe": "vasp_pbe52", "mp_pbeU": "vasp_pbe52",
    "omat_pbe": "vasp_pbe54", "omat_pbeU": "vasp_pbe54",
    "oc20_rpbe": "vasp_pbe54", "oc22_pbeU": "vasp_pbe54",
    "oc25_rpbe": "vasp_pbe54", "omc25_pbe": "vasp_pbe54",
    "matpes_pbe": "vasp_pbe64", "matpes_r2scan": "vasp_pbe64",
    "mad16_r2scan": "aims_tight", "mad16_pbe": "aims_tight",
}


def _meta_of(name: str) -> tuple[int | None, str, int, int]:
    if name in _OVERLAY:
        ov = _OVERLAY[name]
        xc = XC_ID.get(ov["token"], 0)
        paw = PAW_ID.get(_SOURCE_PAW.get(name, "vasp_pbe54"), 1)
        return ov["u"], reference_key_of(name), xc, paw
    ext = _EXTENDED_OVERLAY[name]
    return ext["u"], ext["ref"], XC_ID.get(ext["xc"], 0), PAW_ID.get(ext["paw"], 1)


def _safe_source_ids(name: str) -> tuple[int, int]:
    try:
        return source_ids(name)
    except KeyError:
        fb = _EXTENDED_OVERLAY.get(name, {}).get("fallback_src")
        if fb:
            return source_ids(fb)
        return (0, 0)


_TM_LUT = np.isin(np.arange(len(_COV_RADII)), tuple(MP_HUBBARD_METALS))
_AN_LUT = np.isin(np.arange(len(_COV_RADII)), tuple(MP_HUBBARD_ANIONS))
_UCACHE_CHUNK = 65536
_UCACHE_DEFAULT_BYTES = 256 * 1024 * 1024


def _default_ucache_bytes() -> int:
    from lam4.data.omol import _mem_available_gb
    avail_gb = _mem_available_gb()
    return min(_UCACHE_DEFAULT_BYTES, max(16 * 1024 * 1024, int(avail_gb * 0.04 * (1024 ** 3)))) if avail_gb > 0.0 else _UCACHE_DEFAULT_BYTES


class _UAtomCache:
    """Lazy per-chunk bit-packed cache of per-atom uint8 0/1 masks, bounded by a byte budget."""

    def __init__(self, n_structs: int, max_bytes: int) -> None:
        self.n_structs, self.max_bytes, self.used_bytes, self.rcut = int(n_structs), max(0, int(max_bytes)), 0, None
        self.chunks: list[tuple[np.ndarray, bytearray] | None] = [None] * ((self.n_structs + _UCACHE_CHUNK - 1) // _UCACHE_CHUNK)
        self._lock, self.hits, self.misses = threading.Lock(), 0, 0

    def get_or_compute(self, p: int, n: int, r_cut: float, fn) -> np.ndarray:
        c, r = divmod(p, _UCACHE_CHUNK)
        rc, raw = float(r_cut), None
        with self._lock:
            if self.rcut is not None and self.rcut != rc:
                self.chunks, self.used_bytes = [None] * len(self.chunks), 0
            self.rcut = rc
            ch = self.chunks[c]
            o = -1 if ch is None else int(ch[0][r])
            if o != -1:
                self.hits += 1
                if o == -2:
                    return np.zeros(n, dtype=np.uint8)
                raw = bytes(ch[1][o : o + (n + 7) // 8])
        if raw is not None:
            return np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little", count=n)
        arr = fn()
        with self._lock:
            self.misses += 1
            if self.rcut != rc:
                return arr
            ch = self.chunks[c]
            if ch is None:
                clen = min(_UCACHE_CHUNK, self.n_structs - c * _UCACHE_CHUNK)
                if self.used_bytes + clen * 4 > self.max_bytes:
                    return arr
                ch = (np.full(clen, -1, dtype=np.int32), bytearray())
                self.chunks[c], self.used_bytes = ch, self.used_bytes + clen * 4
            if int(ch[0][r]) == -1:
                if not arr.any():
                    ch[0][r] = -2
                elif self.used_bytes + (nb := (len(arr) + 7) // 8) <= self.max_bytes:
                    ch[0][r] = len(ch[1])
                    ch[1].extend(np.packbits(arr, bitorder="little").tobytes())
                    self.used_bytes += nb
        return arr


def mp_hubbard_u_atoms(z: np.ndarray, pos: np.ndarray, cell: np.ndarray, pbc, struct_u: int, *,
                       edges: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, r_cut: float | None = None) -> np.ndarray:
    """Per-atom +U mask [N] uint8: only MP_HUBBARD_METALS directly bonded to O/F when struct_u == 1."""
    out = np.zeros(len(z), dtype=np.uint8)
    if struct_u <= 0 or len(z) == 0:
        return out
    zc = np.clip(z, 0, len(_COV_RADII) - 1)
    is_tm, is_an = _TM_LUT[zc], _AN_LUT[zc]
    if not (is_tm.any() and is_an.any()):
        return out
    if edges is not None and (r_cut is None or float(r_cut) >= NEIGHBOR_CUTOFF_A):
        snd, rcv, shift = edges
    else:
        snd, rcv, shift = vesin_edges(pos, cell, pbc, NEIGHBOR_CUTOFF_A)
    if len(snd) == 0:
        return out
    m = is_tm[snd] & is_an[rcv]
    if not m.any():
        return out
    si, ri, sh = snd[m], rcv[m], shift[m]
    dr = (pos[si] + sh @ cell) - pos[ri]
    dist = np.linalg.norm(dr, axis=-1)
    max_r = np.minimum(_COV_RADII[zc[si]] + _COV_RADII[zc[ri]] + ANION_BOND_TOL_A, NEIGHBOR_CUTOFF_A)
    bond = dist <= max_r
    if bond.any():
        out[np.unique(si[bond])] = 1
    return out


def _load_crystal_lsq_pt(pt_path: Path, col: int = 0) -> np.ndarray:
    """Load per-element E0 table [119] in kcal/mol from a MatFlat `e0_lsq.pt` (`mu_f64` / `mu` stored in eV)."""
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    m = (d.get("mu_f64", d.get("mu")) if isinstance(d, dict) else d).to(torch.float64).numpy()
    out = np.zeros(119, dtype=np.float64)
    n = min(119, m.shape[0])
    out[:n] = (m[:n, col] if m.ndim == 2 else m[:n]) * EV
    return out


class CrystalSource:
    """Structures of one or several MatFlat dirs. `indices` (one dir only) = rows of the flat; positions p enumerate the kept structures (finite energy, <= max_atoms atoms)."""

    def __init__(self, paths: str | Path | list, sources: tuple[str, str] = ("mp_pbe", "mp_pbeU"), indices: np.ndarray | None = None, max_atoms: int = 400,
                 derive_u: bool | None = None, *, ref_keys: tuple[str, str] | None = None, e0_tables: tuple[np.ndarray, np.ndarray] | None = None,
                 derive_u_atoms: bool = True, cache_u_atoms: bool = True, u_atom_cache_bytes: int | None = None) -> None:
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        if indices is not None and len(paths) != 1:
            raise ValueError("indices select rows of ONE flat")
        self.flats = [MatFlat(p) for p in paths]
        self.sources = sources
        self.derive_u_atoms = bool(derive_u_atoms)
        u0, r0, xc0, paw0 = _meta_of(sources[0])
        _, r1, _, _ = _meta_of(sources[1])
        self.fixed_u = int(u0) if sources[0] == sources[1] and u0 is not None else None
        self.lot_meta = [(_meta_of(s)[2], _meta_of(s)[3]) for s in sources]
        for flat in self.flats:
            man_xc = flat.manifest.get("xc")
            man_paw = flat.manifest.get("paw_basis")
            if man_xc is not None and XC_ID.get(man_xc, xc0) != xc0:
                raise ValueError(f"{flat.path} manifest xc={man_xc!r} disagrees with sources={sources}")
            if man_paw is not None and PAW_ID.get(man_paw, paw0) != paw0:
                raise ValueError(f"{flat.path} manifest paw_basis={man_paw!r} disagrees with sources={sources}")
        # Fallback for legacy OC22 flats built before build_lam2_pool_flat populated u_flag.npy (where u_flag is all zero).
        self._derive_u = [
            (bool(derive_u) if derive_u is not None else (self.fixed_u is None and sources[0] != sources[1] and sources[1] == "oc22_pbeU" and not bool(np.asarray(flat.u_flag).any())))
            for flat in self.flats
        ]
        self._magmom = [np.load(f.path / "magmom.npy", mmap_mode="r") if (f.path / "magmom.npy").exists() else None for f in self.flats]
        self._u_atom = [np.load(f.path / "u_atom.npy", mmap_mode="r") if (f.path / "u_atom.npy").exists() else None for f in self.flats]
        fid, idx = [], []
        for k, flat in enumerate(self.flats):
            ix = np.arange(len(flat)) if indices is None else np.asarray(indices, np.int64)
            ix = ix[(ix >= 0) & (ix < len(flat))]   # the registry splits were built on slightly longer flats (MatPES: 116 filtered rows): clipped as lam1 does
            ok = np.isfinite(np.asarray(flat.energy[ix], np.float64)) & (flat.n_atoms[ix] <= max_atoms)
            idx.append(ix[ok])
            fid.append(np.full(int(ok.sum()), k, np.int64))
        self.idx, self.fid = np.concatenate(idx), np.concatenate(fid)
        self.n_at = np.concatenate([f.n_atoms[i] for f, i in zip(self.flats, idx)]).astype(np.int64)
        keys = ref_keys if ref_keys is not None else (r0, r1)
        if e0_tables is not None:
            self.mu = [np.asarray(t, dtype=np.float64) for t in e0_tables]
        else:
            root_data = Path(__file__).resolve().parents[2] / "data"
            mu_list: list[np.ndarray] = []
            for s_name, rk in zip(sources, keys):
                if s_name in _EXTENDED_OVERLAY:
                    p0 = self.flats[0].path
                    cands = [p0 / "e0_lsq.pt", p0.parent / "train" / "e0_lsq.pt", root_data / f"e0_reference_{rk}_peratom.pt"]
                    hit = next((c for c in cands if c.exists()), None)
                    if hit is None:
                        raise FileNotFoundError(f"no own LSQ E0 table found for {s_name!r} at {cands}; convert with LSQ calibration first (never fall back to alien PAW)")
                    mu_list.append(_load_crystal_lsq_pt(hit))
                else:
                    mu_list.append(load_lot_e0_table_f64(rk, num_elements=119, in_kcal=True).numpy())
            self.mu = mu_list
        if len(self.idx) >= 50_000:
            from lam4.data.omol import pretouch_memmaps
            pretouch_memmaps([arr for f in self.flats for arr in (f.z, f.pos, f.forces, f.offsets)])
        budget = 0 if not cache_u_atoms else (_default_ucache_bytes() if u_atom_cache_bytes is None else int(u_atom_cache_bytes))
        need_cache = budget > 0 and self.derive_u_atoms and (self.fixed_u is None or self.fixed_u > 0) and any(ua is None for ua in self._u_atom)
        self._ucache = _UAtomCache(len(self.idx), budget) if need_cache else None
        self._u_flag_cache = np.full(len(self.idx), -1, dtype=np.int8) if (cache_u_atoms and any(self._derive_u)) else None
        self._pair_index: tuple[np.ndarray, dict] | None = None

    def pair_index(self, **kw) -> tuple[np.ndarray, dict]:
        """Lazily build (or load cached) close secant crystal pairs `(pairs [M, 2], stats)`."""
        if self._pair_index is None:
            from lam4.data.pairs import build_crystal_pairs
            self._pair_index = build_crystal_pairs(self, **kw)
        return self._pair_index

    def __len__(self) -> int:
        return len(self.idx)

    def structure(self, p: int, r_cut: float) -> dict:
        """One structure at position p: atoms, labels (kcal) and its radius graph."""
        p = int(p) + (len(self.idx) if p < 0 else 0)
        fid_i = int(self.fid[p])
        f, i = self.flats[fid_i], int(self.idx[p])
        a, b = int(f.offsets[i]), int(f.offsets[i + 1])
        z = np.asarray(f.z[a:b], np.int64)
        pos = np.asarray(f.pos[a:b], np.float64)
        cell = np.asarray(f.cell[i], np.float64)
        pbc = np.asarray(f.pbc[i])
        if self.fixed_u is not None:
            u = self.fixed_u
        elif self._derive_u[fid_i]:
            uc = self._u_flag_cache
            u = int(uc[p]) if (uc is not None and uc[p] >= 0) else mp_hubbard_u_flag(z, pos, cell, pbc)
            if uc is not None and uc[p] < 0:
                uc[p] = u
        else:
            u = int(np.clip(f.u_flag[i], 0, 1))
        snd, rcv, shift = vesin_edges(pos, cell, pbc, r_cut)
        if self._u_atom[fid_i] is not None:
            u_atom = np.asarray(self._u_atom[fid_i][a:b], np.uint8)
        elif not self.derive_u_atoms:
            u_atom = np.full(len(z), u, dtype=np.uint8)
        elif u <= 0 or len(z) == 0:
            u_atom = np.zeros(len(z), dtype=np.uint8)
        else:
            fn = lambda: mp_hubbard_u_atoms(z, pos, cell, pbc, u, edges=(snd, rcv, shift), r_cut=r_cut)  # noqa: E731
            u_atom = fn() if self._ucache is None else self._ucache.get_or_compute(p, len(z), r_cut, fn)
        mag = np.full(len(z), np.nan, np.float32) if self._magmom[fid_i] is None else np.asarray(self._magmom[fid_i][a:b], np.float32)
        e_res = float(f.energy[i]) * EV - float(self.mu[u][np.clip(z, 0, 118)].sum())
        return {"z": z, "pos": pos.astype(np.float32), "forces": (np.asarray(f.forces[a:b], np.float64) * EV).astype(np.float32), "cell": cell.astype(np.float32),
                "snd": snd, "rcv": rcv, "shift": shift, "e": e_res, "u": u, "u_atom": u_atom, "magmom": mag, "pbc": bool(np.any(pbc)),
                "stress": (np.asarray(f.stress[i], np.float64) * EV).astype(np.float32)}


def pack_crystals(structs: list[dict], source: CrystalSource, n_pad: int, e_pad: int | None, k: int, r_cut: float, device, dtype=torch.float32) -> Packed:
    n_at = np.array([len(s["z"]) for s in structs])
    start = np.cumsum(n_at) - n_at
    snd = np.concatenate([s["snd"] + o for s, o in zip(structs, start)])
    rcv = np.concatenate([s["rcv"] + o for s, o in zip(structs, start)])
    n_real_edges = int(len(snd))
    G, N = len(structs), int(n_at.sum())
    u = np.array([s["u"] for s in structs])
    ids = [_safe_source_ids(source.sources[x]) for x in u]
    cells_np = np.stack([s["cell"] for s in structs])
    has_pbc = any(bool(s.get("pbc", True)) for s in structs)
    if has_pbc and float(np.abs(np.linalg.det(cells_np)).min()) < 1e-8:
        raise ValueError("pack_crystals received singular/zero cell inside pbc=True batch; split pbc=False structures to OmolFlat")
    meta = getattr(source, "lot_meta", [(0, 1), (0, 1)])
    xc_id_g = np.array([meta[x][0] for x in u], dtype=np.int64)
    paw_id_g = np.array([meta[x][1] for x in u], dtype=np.int64)
    u_atom_n = np.concatenate([s.get("u_atom", np.full(len(s["z"]), s["u"])) for s in structs]).astype(np.int64)
    t = lambda a, dt=None, **kw: torch.as_tensor(np.asarray(a), dtype=dt, **kw).to(device)  # noqa: E731
    b = GraphBatch(
        z=t(np.concatenate([s["z"] for s in structs])), pos=t(np.concatenate([s["pos"] for s in structs]), dtype), graph=t(np.repeat(np.arange(G), n_at)),
        cell=t(cells_np, dtype), edge_index=t(np.stack([snd, rcv])),
        edge_shift=t(np.concatenate([s["shift"] for s in structs]), dtype), charge=torch.zeros(G, dtype=dtype, device=device),
        spin=torch.ones(G, dtype=dtype, device=device), u_flag=t(u), token=t([i[0] for i in ids]), ref_id=t([i[1] for i in ids]), domain="crystal",
        atom_mask=torch.ones(N, dtype=torch.bool, device=device), graph_mask=torch.ones(G, dtype=torch.bool, device=device),
        energy=t([s["e"] for s in structs], dtype), forces=t(np.concatenate([s["forces"] for s in structs]), dtype),
        energy_mask=torch.ones(G, dtype=torch.bool, device=device), force_mask=torch.ones(N, dtype=torch.bool, device=device), coulomb_cap=None, pbc=has_pbc,
        xc_id=t(xc_id_g), paw_id=t(paw_id_g), u_atom=t(u_atom_n))
    if e_pad is not None:
        b = pad_batch(b, n_pad, e_pad, k + 1, r_cut)
    g, n_tot = b.n_graphs, len(b.z)
    st = np.zeros((g, 3, 3), np.float32)
    st[:G] = np.stack([s["stress"] for s in structs])
    st_ok = np.zeros(g, bool)
    st_ok[:G] = np.isfinite(st[:G]).all((1, 2))
    mag = np.full(n_tot, np.nan, np.float32)
    mag[:N] = np.concatenate([s.get("magmom", np.full(len(s["z"]), np.nan, np.float32)) for s in structs])
    mag_ok = np.isfinite(mag)
    from lam4.data.pairs import build_pair_tensors_crystal
    extra = {"dipole": torch.zeros(g, 3, dtype=dtype, device=device), "dipole_mask": torch.zeros(g, dtype=torch.bool, device=device), "seg": torch.arange(g, device=device),
             "stress": t(np.nan_to_num(st), dtype), "stress_mask": t(st_ok), "u_atom": b.u_atom, "magmom": t(np.nan_to_num(mag), dtype), "magmom_mask": t(mag_ok),
             "xc_id": b.xc_id, "paw_id": b.paw_id, **build_pair_tensors_crystal(structs, start, n_tot, g, device)}
    return Packed(b, extra, cap=0, used=G, n_real=N, domain="crystal", n_real_edges=n_real_edges)


class CrystalStream:
    domain = "crystal"

    def __init__(self, src: CrystalSource, n_pad: int = 8192, k: int = 128, seed: int = 0, r_cut: float = 6.0, device="cpu", e_pad: int | None = None,
                 margin: float = 1.15, n_probe: int = 8, workers: int = 4, pair_frac: float = 0.0) -> None:
        self.src, self.n_pad, self.k, self.r_cut, self.device = src, n_pad, k, r_cut, device
        self.pair_frac = float(pair_frac)
        self.rng = np.random.default_rng(seed)
        self._pair_rng = np.random.default_rng(seed + 99991)
        self.epoch, self.skipped, self.e_pad = 0, 0, e_pad
        self._workers = max(1, int(workers))
        self._pool = None
        if self._workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=self._workers)
        self._pairs = np.zeros((0, 2), dtype=np.int64)
        self.pair_stats: dict = {}
        self._pair_perm = np.zeros(0, dtype=np.int64)
        self._pair_ptr = 0
        self._pair_tok = 0
        if self.pair_frac > 0.0:
            self._pairs, self.pair_stats = self.src.pair_index()
            if len(self._pairs) > 0:
                self._pair_perm = self._pair_rng.permutation(len(self._pairs))
        self._new_epoch()
        self.e_pad = e_pad or self._calibrate(margin, n_probe)

    def _new_epoch(self) -> None:
        self.epoch += 1
        self.perm, self.ptr, self.pending = self.rng.permutation(len(self.src)), 0, []   # pending: structures read but not yet used (consecutive in perm)

    def _take_pairs(self, cap_n: int) -> tuple[list[dict], int]:
        if self.pair_frac <= 0.0 or len(self._pairs) == 0 or self.k < 2:
            return [], 0
        target_n = min(cap_n, int(round(self.pair_frac * cap_n)))
        if target_n <= 0:
            return [], 0
        cand_pairs: list[tuple[int, int]] = []
        n_est = 0
        max_tries = min(len(self._pairs), max(32, self.k))
        tries = 0
        while n_est < target_n and (len(cand_pairs) * 2 + 2) <= self.k and tries < max_tries:
            if self._pair_ptr >= len(self._pair_perm):
                self._pair_perm = self._pair_rng.permutation(len(self._pairs))
                self._pair_ptr = 0
            pa, pb = int(self._pairs[self._pair_perm[self._pair_ptr], 0]), int(self._pairs[self._pair_perm[self._pair_ptr], 1])
            self._pair_ptr += 1
            tries += 1
            na, nb = int(self.src.n_at[pa]), int(self.src.n_at[pb])
            if na != nb or na <= 0 or (na + nb) > cap_n:
                continue
            if n_est + na + nb > target_n and cand_pairs:
                break
            if n_est + na + nb > cap_n:
                continue
            cand_pairs.append((pa, pb))
            n_est += na + nb
        if not cand_pairs:
            return [], 0
        flat_idx = [p for ab in cand_pairs for p in ab]
        if self._pool is not None and len(flat_idx) >= 8:
            built = list(self._pool.map(lambda idx: self.src.structure(idx, self.r_cut), flat_idx))
        else:
            built = [self.src.structure(idx, self.r_cut) for idx in flat_idx]
        out: list[dict] = []
        n = 0
        e_est = 0
        for j in range(0, len(built), 2):
            sa, sb = built[j], built[j + 1]
            ea, eb = len(sa["snd"]), len(sb["snd"])
            if self.e_pad is not None and (ea > self.e_pad or eb > self.e_pad or ea + eb > self.e_pad):
                continue
            if self.e_pad is not None and (e_est + ea + eb) > self.e_pad:
                break
            self._pair_tok += 1
            sa["_pair_id"] = self._pair_tok
            sb["_pair_id"] = self._pair_tok
            out.append(sa)
            out.append(sb)
            n += len(sa["z"]) + len(sb["z"])
            e_est += ea + eb
        return out, n

    def _take(self) -> list[dict]:
        cap_n = self.n_pad - 2
        out, n = self._take_pairs(cap_n)
        while len(out) < self.k:
            if self.pending:
                s = self.pending.pop(0)
                if n + len(s["z"]) > cap_n:
                    self.pending.insert(0, s)
                    break
                out.append(s)
                n += len(s["z"])
                continue
            rem_k = self.k - len(out)
            cand: list[int] = []
            n_est = n
            while len(cand) < rem_k and self.ptr < len(self.perm):
                p = int(self.perm[self.ptr])
                na = int(self.src.n_at[p])
                if na > cap_n:
                    self.ptr += 1
                    continue
                if n_est + na > cap_n:
                    break
                self.ptr += 1
                cand.append(p)
                n_est += na
            if not cand:
                if self.ptr >= len(self.perm):
                    self._new_epoch()
                    continue
                if out:
                    break
                self._new_epoch()
                continue
            if self._pool is not None and len(cand) >= 8:
                built = list(self._pool.map(lambda idx: self.src.structure(idx, self.r_cut), cand))
            else:
                built = [self.src.structure(idx, self.r_cut) for idx in cand]
            for s in built:
                if self.e_pad is not None and len(s["snd"]) > self.e_pad:
                    self.skipped += 1
                    continue
                out.append(s)
                n += len(s["z"])
        return out

    def _calibrate(self, margin: float, n_probe: int) -> int:
        saved_ptr, saved_rng = self._pair_ptr, self._pair_rng.bit_generator.state
        top = max(sum(len(s["snd"]) for s in self._take()) for _ in range(n_probe))
        self.ptr, self.pending = 0, []
        self._pair_ptr, self._pair_rng.bit_generator.state = saved_ptr, saved_rng
        return int(np.ceil(top * margin / 4096) * 4096)

    def next(self) -> Packed:
        structs = self._take()
        if self.e_pad is not None and len(structs) > 1:   # the longest prefix that fits e_pad; the rest goes back in front, in order
            keep = max(1, int(np.searchsorted(np.cumsum([len(s["snd"]) for s in structs]), self.e_pad, side="right")))
            # Avoid splitting a matched pair across the e_pad cut boundary.
            if keep < len(structs) and keep > 1:
                pid_l = structs[keep - 1].get("_pair_id")
                if pid_l is not None and pid_l == structs[keep].get("_pair_id"):
                    keep -= 1
            dropped = [s for s in structs[keep:] if s.pop("_pair_id", None) is None]
            self.pending = dropped + self.pending
            structs = structs[:keep]
        return pack_crystals(structs, self.src, self.n_pad, self.e_pad, self.k, self.r_cut, self.device)

    def state(self) -> dict:
        st = {"epoch": self.epoch, "rng": self.rng.bit_generator.state, "perm": self.perm.copy(), "ptr": max(0, self.ptr - len(self.pending))}
        if self.pair_frac > 0.0 and len(self._pairs) > 0:
            st["pair_rng"] = self._pair_rng.bit_generator.state
            st["pair_ptr"] = int(self._pair_ptr)
        return st

    def load(self, st: dict) -> None:
        self.epoch = st["epoch"]
        self.rng.bit_generator.state = st["rng"]
        perm = np.asarray(st["perm"])
        if len(perm) == len(self.src):
            self.perm, self.ptr, self.pending = perm, max(0, int(st["ptr"])), []
        else:
            self.perm, self.ptr, self.pending = self.rng.permutation(len(self.src)), 0, []
        if self.pair_frac > 0.0 and len(self._pairs) > 0 and "pair_rng" in st:
            self._pair_rng.bit_generator.state = st["pair_rng"]
            self._pair_ptr = int(st.get("pair_ptr", 0)) % len(self._pairs)
