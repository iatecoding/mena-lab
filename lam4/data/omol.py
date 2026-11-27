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
import lam4.registry.tokens as _tokens

EDGES = (24, 48, 80, 128, 192, 350)
_ROOT = Path(__file__).resolve().parents[2]
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

OMOL_SOURCE_CONFIGS: dict[str, dict] = {
    "omol_wb97mv": dict(xc="wb97mv", paw="def2_tzvpd", d3_strip=True, default_ref=DEFAULT_REF, zero_ref=False),
    "opoly26_wb97mv": dict(xc="wb97mv", paw="def2_tzvpd", d3_strip=True, default_ref=DEFAULT_REF, zero_ref=False),
    "spice_wb97md3": dict(xc="wb97m_d3bj", paw="def2_tzvppd", d3_strip=True, default_ref=_ROOT / "data" / "e0_reference_spice2.json", zero_ref=False),
    "qcml_pbe0": dict(xc="pbe0", paw="aims_tight", d3_strip=False, default_ref=None, zero_ref=False),
    "t1x_wb97x": dict(xc="wb97x", paw="pop_631gd", d3_strip=False, default_ref=None, zero_ref=False),
    "mad16_r2scan_mol": dict(xc="r2scan", paw="aims_tight", d3_strip=False, default_ref=None, zero_ref=False),
    "splinter_sapt0": dict(xc="sapt0", paw="jun_cc_pvdz", d3_strip=False, default_ref=None, zero_ref=True),
}


def load_reference(path: str | Path = DEFAULT_REF) -> dict[str, np.ndarray]:
    """e0 [119], e_q [21] (index Q + 10), e_s [11] (index S - 1), float64 kcal/mol (JSON or calibrated .pt)."""
    p = Path(path)
    if p.suffix == ".pt":
        d = torch.load(p, map_location="cpu", weights_only=False)
        e0 = np.asarray(d.get("e0_kcal", np.zeros(119)), dtype=np.float64)[:119]
        if len(e0) < 119:
            e0 = np.pad(e0, (0, 119 - len(e0)))
        e_q = np.asarray(d.get("e_q_kcal", np.zeros(21)), dtype=np.float64)[:21]
        e_s = np.asarray(d.get("e_s_kcal", np.zeros(11)), dtype=np.float64)[:11]
        return {"e0": e0, "e_q": e_q, "e_s": e_s}
    d = json.loads(p.read_text(encoding="utf-8"))
    e0, e_q, e_s = np.zeros(119), np.zeros(21), np.zeros(11)
    for z, v in d["e0_kcal"].items():
        e0[int(z)] = v
    for q, v in d.get("charge_kcal", {}).items():
        e_q[int(q) + 10] = v
    for s, v in d.get("spin_kcal", {}).items():
        e_s[int(s) - 1] = v
    return {"e0": e0, "e_q": e_q, "e_s": e_s}


def load_exact_reference(path: str | Path = DEFAULT_REF, *, xc: str | None = None, paw: str | None = None) -> dict[str, np.ndarray]:
    """Load per-(xc, paw) LSQ reference table (raises loudly on missing or mismatched LoT; never silent-falls back)."""
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        if xc is not None and "xc" in d and d["xc"] != xc:
            raise ValueError(f"reference {p} has xc={d['xc']!r}, expected {xc!r}")
        if paw is not None and "paw_basis" in d and d["paw_basis"] != paw:
            raise ValueError(f"reference {p} has paw_basis={d['paw_basis']!r}, expected {paw!r}")
        return load_reference(p)
    d = torch.load(p, map_location="cpu", weights_only=False)
    if "by_lot" in d:
        if xc is None:
            if len(d["by_lot"]) == 1:
                sub = next(iter(d["by_lot"].values()))
            else:
                raise ValueError(f"multi-LoT reference {p} requires explicit xc= (keys: {sorted(d['by_lot'])})")
        else:
            key = f"{xc}|{paw}" if paw else xc
            sub = d["by_lot"].get(key) or (d["by_lot"].get(xc) if paw is None else None)
            if sub is None:
                raise KeyError(f"LoT {key!r} missing from reference bundle {p} (available: {sorted(d['by_lot'])})")
        e0 = np.pad(np.asarray(sub["e0_kcal"], np.float64), (0, max(0, 119 - len(sub["e0_kcal"]))))[:119]
        e_q = np.asarray(sub.get("e_q_kcal", np.zeros(21)), np.float64)[:21]
        e_s = np.asarray(sub.get("e_s_kcal", np.zeros(11)), np.float64)[:11]
        return {"e0": e0, "e_q": e_q, "e_s": e_s}
    if xc is not None and "xc" in d and d["xc"] != xc:
        raise ValueError(f"reference {p} has xc={d['xc']!r}, expected {xc!r}")
    if paw is not None and "paw_basis" in d and d["paw_basis"] != paw:
        raise ValueError(f"reference {p} has paw_basis={d['paw_basis']!r}, expected {paw!r}")
    return load_reference(p)


def reference_energy(z_atoms: np.ndarray, starts: np.ndarray, n_atoms: np.ndarray, charge: np.ndarray, spin: np.ndarray, ref: dict) -> np.ndarray:
    """sum_i e0[Z_i] + E_Q[Q] + E_S[S] per structure, float64. z_atoms: atoms of the structures concatenated, starts: first atom of each."""
    e = np.add.reduceat(ref["e0"][z_atoms.astype(np.int64)], starts) if len(z_atoms) else np.zeros(len(n_atoms))
    q, s = np.asarray(charge).astype(np.int64), np.asarray(spin).astype(np.int64)
    eq = np.where((q >= -10) & (q <= 10), ref["e_q"][np.clip(q + 10, 0, 20)], 0.0)
    es = np.where((s >= 1) & (s <= 11), ref["e_s"][np.clip(s - 1, 0, 10)], 0.0)
    return e + eq + es


def _opt_mmap(path: Path, name: str):
    f = path / f"{name}.npy"
    return np.load(f, mmap_mode="r") if f.exists() else None


def _mem_available_gb() -> float:
    p = Path("/proc/meminfo")
    if not p.exists():
        return 0.0
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / (1024.0 * 1024.0)
    return 0.0


def pretouch_memmaps(arrays: list[np.ndarray | None], *, min_avail_gb: float = 48.0) -> int:
    """Sequentially fault one entry per 4KB page into Linux OS cache when ample RAM is free. Returns count of arrays touched."""
    if _mem_available_gb() < min_avail_gb:
        return 0
    touched = 0
    for arr in arrays:
        if arr is None or len(arr) == 0:
            continue
        row_cols = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
        step = max(1, 4096 // max(1, row_cols * int(arr.itemsize)))
        sl = arr[::step, 0] if arr.ndim > 1 else arr[::step]
        np.count_nonzero(sl)
        touched += 1
    return touched


class OmolSource:
    """Labelled molecular structures in size buckets. Thread `source=` cleanly so each physical LoT uses its own LSQ table & D3 policy."""

    def __init__(self, path: str | Path, *, source: str = "omol_wb97mv", indices: np.ndarray | None = None, max_structs: int = 0, seed: int = 0,
                 edges: tuple[int, ...] = EDGES, ref_json: str | Path | None = None, d3_cache: str | Path | None = None, d3_compute: bool = True,
                 d3_strip: bool | None = None, device: torch.device = torch.device("cpu")) -> None:
        if source not in OMOL_SOURCE_CONFIGS:
            raise KeyError(f"unknown OmolSource source={source!r}; known: {sorted(OMOL_SOURCE_CONFIGS)}")
        cfg = OMOL_SOURCE_CONFIGS[source]
        self.flat = flat = OmolFlat(path)
        man_xc = flat.manifest.get("xc")
        man_paw = flat.manifest.get("paw_basis")
        if man_xc is not None and man_xc != cfg["xc"]:
            raise ValueError(f"{flat.path} manifest declares xc={man_xc!r}, incompatible with source={source!r} (xc={cfg['xc']!r})")
        if man_paw is not None and man_paw != cfg["paw"]:
            raise ValueError(f"{flat.path} manifest declares paw_basis={man_paw!r}, incompatible with source={source!r} (paw={cfg['paw']!r})")
        self.source = source
        self.xc = str(cfg["xc"])
        self.paw_basis = str(cfg["paw"])
        self.xc_id = int(XC_ID[self.xc])
        self.paw_id = int(PAW_ID[self.paw_basis])
        self.has_forces = bool(flat.manifest.get("has_forces", True))
        strip_d3 = bool(cfg["d3_strip"]) if d3_strip is None else bool(d3_strip)
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
        if strip_d3:
            full = default_subset(flat, 0, 0, self.edges[-1])   # a subset (or a val split) reuses the labels cached for the whole flat
            reuse = [(full, flat.path / f"d3_{d3_sig(flat, full)}.pt")]
            self.d3_e, self.d3_f, self.d3_off = d3_labels(flat, idx, device, d3_cache, reuse, compute=d3_compute)
        else:
            self.d3_e = np.zeros(len(idx), dtype=np.float64)
            self.d3_f = np.zeros((int(self.n_at.sum()), 3), dtype=np.float32)
            off = np.zeros(len(idx), dtype=np.int64)
            if len(idx) > 1:
                np.cumsum(self.n_at[:-1], out=off[1:])
            self.d3_off = off
        self.charge = np.asarray(flat.charge[idx]).astype(np.float64)
        self.spin = np.asarray(flat.spin[idx]).astype(np.float64)
        if cfg["zero_ref"] and ref_json is None:
            ref = {"e0": np.zeros(119), "e_q": np.zeros(21), "e_s": np.zeros(11)}
        else:
            cands = [Path(ref_json)] if ref_json is not None else [
                flat.path / "e0_lsq.pt", flat.path / "e0_lsq.json",
                *([Path(cfg["default_ref"])] if cfg["default_ref"] is not None else []),
            ]
            chosen = next((c for c in cands if c.exists()), None)
            if chosen is None:
                raise FileNotFoundError(f"no LSQ E0 reference table found for source={source!r} at {cands}; pass explicit ref_json= or run build_exact_e0.py")
            ref = load_exact_reference(chosen, xc=self.xc, paw=self.paw_basis)
        atoms, starts = gather_atoms(flat, idx)
        e_ref = reference_energy(np.asarray(flat.z[atoms]), starts, self.n_at, self.charge, self.spin, ref)
        self.e_res = np.asarray(flat.energy[idx], np.float64) - np.asarray(self.d3_e, np.float64) - e_ref   # float64 on the host, cast per batch
        self.dipole = None if flat.dipole is None else np.asarray(flat.dipole[idx], np.float32)
        self._alpha_mol = _opt_mmap(flat.path, "alpha_iso")
        self._alpha_atom = _opt_mmap(flat.path, "alpha_atom")
        self._sapt = _opt_mmap(flat.path, "sapt")
        self._pop_charges = _opt_mmap(flat.path, "pop_charges")
        self._sidecar_manifest_mtime = 0.0
        self._load_sidecars(require_manifest=False)
        which = np.searchsorted(np.asarray(self.edges), self.n_at)
        self.buckets = {c: np.where(which == b)[0] for b, c in enumerate(self.edges) if (which == b).any()}
        self._groups: np.ndarray | None = None
        self._pair_index: tuple[dict[int, np.ndarray], dict] | None = None
        if max_structs == 0 and len(idx) >= 50_000:
            self._pretouch_all()

    def pair_index(self, **kw) -> tuple[dict[int, np.ndarray], dict]:
        """Lazily build (or load cached) close aligned molecular pairs per size bucket `(pairs_by_cap, stats)`."""
        if self._pair_index is None:
            from lam4.data.pairs import build_omol_pairs
            self._pair_index = build_omol_pairs(self, **kw)
        return self._pair_index

    def _sidecar_arrays(self) -> list:
        return [self._spins_lowdin, self._frag_id_545, self._unrestricted, self._s_squared_dev,
                self._homo_gap, self._quadrupole, self._n_islands, self._frag_q_lowdin]

    def _pretouch_all(self) -> None:
        pretouch_memmaps([self.flat.z, self.flat.pos, self.flat.forces, self.flat.n_atoms, self.flat.offsets, self.flat.dipole, *self._sidecar_arrays()])

    def _load_sidecars(self, *, require_manifest: bool = False) -> bool:
        man_p = self.flat.path / "sidecars_manifest.json"
        if require_manifest and not man_p.is_file():
            return False
        mt = man_p.stat().st_mtime if man_p.is_file() else 0.0
        if require_manifest and self._sidecar_manifest_mtime > 0.0 and mt <= self._sidecar_manifest_mtime:
            return False
        self._spins_lowdin = _opt_mmap(self.flat.path, "spins_lowdin")
        self._frag_id_545 = _opt_mmap(self.flat.path, "frag_id_545")
        self._unrestricted = _opt_mmap(self.flat.path, "unrestricted")
        self._s_squared_dev = _opt_mmap(self.flat.path, "s_squared_dev")
        self._homo_gap = _opt_mmap(self.flat.path, "homo_gap")
        self._quadrupole = _opt_mmap(self.flat.path, "quadrupole")
        self._n_islands = _opt_mmap(self.flat.path, "n_islands")
        self._frag_q_lowdin = _opt_mmap(self.flat.path, "frag_q_lowdin")
        self._sidecar_manifest_mtime = mt
        return any(x is not None for x in self._sidecar_arrays())

    def reload_sidecars(self) -> bool:
        """Reload Canonical Sidecar Schema v1 mmaps once `sidecars_manifest.json` appears/updates (guards against partial copies)."""
        if not self._load_sidecars(require_manifest=True):
            return False
        if len(self.idx) >= 50_000:
            pretouch_memmaps(self._sidecar_arrays())
        return True

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
        n [K], energy [K] float64, charge, spin [K], dipole [K, 3] (NaN without labels), plus Canonical Sidecar Schema v1 fields."""
        pos = np.asarray(pos, np.int64)
        rows = self.idx[pos]
        k, n = len(pos), self.n_at[pos]
        atoms, starts = gather_atoms(self.flat, rows)
        d3_atoms = np.repeat(self.d3_off[pos] - starts, n) + np.arange(int(n.sum()))
        valid = np.arange(cap)[None, :] < n[:, None]
        z = np.zeros((k, cap), np.int64)
        p = np.zeros((k, cap, 3), np.float32)
        f = np.zeros((k, cap, 3), np.float32)
        f_mask = np.zeros((k, cap), dtype=bool)
        z[valid] = np.asarray(self.flat.z[atoms])
        p[valid] = np.asarray(self.flat.pos[atoms])
        if self.has_forces:
            f_raw = np.asarray(self.flat.forces[atoms], np.float32) - self.d3_f[d3_atoms]
            fin = np.isfinite(f_raw).all(axis=-1)
            f_mask[valid] = fin
            f[valid] = np.where(fin[:, None], f_raw, 0.0)
        dip = np.full((k, 3), np.nan, np.float32) if self.dipole is None else self.dipole[pos]
        spins_low = np.full((k, cap), np.nan, np.float32)
        if self._spins_lowdin is not None:
            spins_low[valid] = np.asarray(self._spins_lowdin[atoms], np.float32)
        frag_id = np.zeros((k, cap), np.uint8)
        if self._frag_id_545 is not None:
            frag_id[valid] = np.asarray(self._frag_id_545[atoms], np.uint8)
        return {
            "z": z, "pos": p, "forces": f, "force_mask": f_mask, "n": n, "energy": self.e_res[pos], "charge": self.charge[pos], "spin": self.spin[pos], "dipole": dip,
            "spins_lowdin": spins_low, "frag_id_545": frag_id,
            "unrestricted": np.full(k, 255, np.uint8) if self._unrestricted is None else np.asarray(self._unrestricted[rows], np.uint8),
            "s_squared_dev": np.full(k, np.nan, np.float32) if self._s_squared_dev is None else np.asarray(self._s_squared_dev[rows], np.float32),
            "homo_gap": np.full((k, 2), np.nan, np.float32) if self._homo_gap is None else np.asarray(self._homo_gap[rows], np.float32),
            "quadrupole": np.full((k, 6), np.nan, np.float32) if self._quadrupole is None else np.asarray(self._quadrupole[rows], np.float32),
            "n_islands": np.zeros(k, np.uint8) if self._n_islands is None else np.asarray(self._n_islands[rows], np.uint8),
            "frag_q_lowdin": np.full((k, 8), np.nan, np.float32) if self._frag_q_lowdin is None else np.asarray(self._frag_q_lowdin[rows], np.float32),
        }

    def gather_aux(self, cap: int, pos: np.ndarray) -> dict[str, np.ndarray]:
        """Optional LAM-6 auxiliary targets ([K] / [K, cap]) with boolean validity masks."""
        pos = np.asarray(pos, np.int64)
        rows = self.idx[pos]
        K, n = len(pos), self.n_at[pos]
        alpha_mol = np.full(K, np.nan, np.float32) if self._alpha_mol is None else np.asarray(self._alpha_mol[rows], np.float32)
        sapt = np.full((K, 5), np.nan, np.float32) if self._sapt is None else np.asarray(self._sapt[rows], np.float32)
        alpha_atom = np.full((K, cap), np.nan, np.float32)
        pop = np.full((K, cap, 2), np.nan, np.float32)
        if self._alpha_atom is not None or self._pop_charges is not None:
            atoms, _ = gather_atoms(self.flat, rows)
            valid = np.arange(cap)[None, :] < n[:, None]
            if self._alpha_atom is not None:
                alpha_atom[valid] = np.asarray(self._alpha_atom[atoms], np.float32)
                if self._alpha_mol is None:
                    alpha_mol = np.nansum(np.where(valid, alpha_atom, 0.0), axis=1).astype(np.float32)
                    alpha_mol[~np.isfinite(alpha_atom).any(axis=1)] = np.nan
            if self._pop_charges is not None:
                pop[valid] = np.asarray(self._pop_charges[atoms], np.float32)
        return {
            "alpha_iso": np.nan_to_num(alpha_mol), "alpha_mask": np.isfinite(alpha_mol),
            "alpha_atom": np.nan_to_num(alpha_atom), "alpha_atom_mask": np.isfinite(alpha_atom),
            "sapt": np.nan_to_num(sapt), "sapt_mask": np.isfinite(sapt).all(axis=-1),
            "pop_charges": np.nan_to_num(pop), "pop_mask": np.isfinite(pop),
        }


def resolve_omol_dir_source(spec: str) -> tuple[str, str]:
    """Parse `path[@source]` and infer `OMOL_SOURCE_CONFIGS` key from `manifest.json` (`(xc, paw_basis)` or `lot`)."""
    if "@" in spec:
        p, src = spec.rsplit("@", 1)
        if src not in OMOL_SOURCE_CONFIGS:
            raise KeyError(f"unknown OmolSource source={src!r}; known: {sorted(OMOL_SOURCE_CONFIGS)}")
        return p, src
    man_p = Path(spec) / "manifest.json"
    if man_p.is_file():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        xc_m, paw_m, lot_m = man.get("xc"), man.get("paw_basis"), man.get("lot")
        if xc_m is not None or paw_m is not None:
            for k, c in OMOL_SOURCE_CONFIGS.items():
                if c["xc"] == xc_m and (paw_m is None or c["paw"] == paw_m):
                    return spec, k
            raise ValueError(f"{spec} manifest declares unknown OmolSource (xc={xc_m!r}, paw_basis={paw_m!r})")
        if lot_m is not None:
            if lot_m in OMOL_SOURCE_CONFIGS:
                return spec, str(lot_m)
            raise ValueError(f"{spec} manifest declares unknown OmolSource (lot={lot_m!r})")
    return spec, "omol_wb97mv"
