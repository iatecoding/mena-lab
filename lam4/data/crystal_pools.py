"""crystal_pools.py: the crystal domain of the lam4 loader = several MatFlat pools, one `CrystalStream` each, mixed by `DomainSampler` with the pool shares (shares are within the
crystal domain; the domain share against molecules is the outer sampler's). All pools share ONE (n_pad, k, e_pad), the max over the pools' calibrations, so the crystal batches are a
single static shape (one traced graph). Also the registry facts per pool (LoT pair by u_flag, train split) and the validation sets (split file -> flat dir -> pool sources).

Pool spec (CLI `--crystal-pools`): `name:dir[+dir...]:share,...`, share optional (default below). Names: mptrj, mptrj_salex, omat24, matpes_pbe, matpes_r2scan.
LoT of a structure: u_flag 0/1 -> (mp_pbe, mp_pbeU) for mptrj / sAlex, (omat_pbe, omat_pbeU) for OMat24, one LoT for MatPES (lam/data/lot_registry.yaml `lot_id_by_u_flag`).
Train rows: the registry train split when it exists (data/splits/*_train_idx.npy, clipped to the flat), else every row (OMat24 parts: val is a separate flat).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lam.data.shards import MatFlat
from lam4.data.crystal import CrystalSource, CrystalStream
from lam4.data.sampler import DomainSampler

SPLITS = Path(__file__).resolve().parents[2] / "data" / "splits"
MPTRJ_N_TOTAL = 1_580_395   # rows 0..MPTRJ_N_TOTAL-1 of the merged mptrj_salex_flat are mptrj_flat in the same row order
POOL_SOURCES = {"mptrj": ("mp_pbe", "mp_pbeU"), "mptrj_salex": ("mp_pbe", "mp_pbeU"), "omat24": ("omat_pbe", "omat_pbeU"),
                "matpes_pbe": ("matpes_pbe", "matpes_pbe"), "matpes_r2scan": ("matpes_r2scan", "matpes_r2scan"),
                "oc20": ("oc20_rpbe", "oc20_rpbe"), "oc22": ("mp_pbe", "oc22_pbeU"), "omc25": ("omc25_pbe", "omc25_pbe"),
                "oc25": ("oc25_rpbe", "oc25_rpbe"), "mad16_r2scan": ("mad16_r2scan", "mad16_r2scan"), "mad16_pbe": ("mad16_pbe", "mad16_pbe")}
POOL_TRAIN_SPLIT = {"mptrj": "mptrj_train_idx.npy", "mptrj_salex": "salex_train_idx.npy", "matpes_pbe": "matpes_pbe_train_idx.npy", "matpes_r2scan": "matpes_r2scan_train_idx.npy"}
DEFAULT_SHARES = {"mptrj_salex": 0.38, "omat24": 0.30, "matpes_pbe": 0.085, "matpes_r2scan": 0.085, "oc20": 0.075, "oc22": 0.075, "mptrj": 1.0, "omc25": 0.05,
                  "oc25": 0.14, "mad16_r2scan": 0.035, "mad16_pbe": 0.015}
# Periodic G4 100h epoch-capped shares (sum=1.0; at 20.24B periodic atom-steps: MAD-1.6 95.1x<=115x, MatPES-r2SCAN 132.3x<=150x, MatPES-PBE 78.2x<=100x)
LAM6_PERIODIC_SHARES = {"omat24": 0.355, "mptrj_salex": 0.300, "oc25": 0.140, "omc25": 0.075, "oc20": 0.060, "mad16_r2scan": 0.035, "matpes_r2scan": 0.020, "matpes_pbe": 0.015}
# crystal dirs under a data root (a launcher may copy each <flat> folder to local disk first)
ROOT_POOLS = {"mptrj_salex": ["mptrj_salex_flat"], "omat24": [f"omat24_sub_p{i}" for i in range(4)], "matpes_pbe": ["matpes_pbe_flat"], "matpes_r2scan": ["matpes_r2scan_flat"],
              "oc20": ["lam2_pools/oc20_s2ef2m_flat"], "oc22": ["lam2_pools/oc22_s2eftotal_cut2m_flat"]}
LAM6_ROOT_POOLS = {"omat24": [f"omat24_sub_p{i}" for i in range(4)], "mptrj_salex": ["mptrj_salex_flat"], "mad16_r2scan": ["mad16_r2scan_flat/train"],
                   "matpes_r2scan": ["matpes_r2scan_flat"], "matpes_pbe": ["matpes_pbe_flat"], "oc25": ["oc25_flat/train"], "oc20": ["lam2_pools/oc20_s2ef2m_flat"], "omc25": ["omc25_flat/train"]}
# validation: split file -> (pool for the LoT pair, flat dir under the root); salex / omat24 val sets live in their own val flats
VAL_SETS = {"mptrj_val_2k": ("mptrj", "mptrj_flat"), "salex_val_2k": ("mptrj_salex", "salex_val_flat"), "omat24_val_stratified_2100": ("omat24", "omat24_val_flat"),
            "matpes_pbe_val_1k": ("matpes_pbe", "matpes_pbe_flat"), "matpes_r2scan_val_1k": ("matpes_r2scan", "matpes_r2scan_flat"),
            "oc20_val": ("oc20", "lam2_pools/oc20_s2ef2m_flat"), "oc22_val": ("oc22", "lam2_pools/oc22_s2eftotal_cut2m_flat"),
            "omc25_val": ("omc25", "lam2_pools/omc25_val_thin10_flat"),
            "oc25_val": ("oc25", "oc25_flat/val"), "mad16_r2scan_val": ("mad16_r2scan", "mad16_r2scan_flat/val")}
# sets whose flat IS the held-out set (every row is validation): sampled from the flat itself. salex_val_2k.npy indexes the old 647,849-row sAlex subset,
# not salex_val_flat (applied to it, 17% of the sampled rows were dropped and the held-out-formula metadata did not hold); salex_val_flat = official sAlex val.
WHOLE_FLAT_VAL = {"salex_val_2k", "oc25_val", "mad16_r2scan_val", "omc25_val"}


def parse_pools(spec: str = "", root: str = "", *, profile: str = "lam5") -> list[tuple[str, list[str], float]]:
    """-> [(name, dirs, share)]. spec empty: the default pools found under `root`."""
    shares = LAM6_PERIODIC_SHARES if profile == "lam6" else DEFAULT_SHARES
    roots = LAM6_ROOT_POOLS if profile == "lam6" else ROOT_POOLS
    out = []
    if spec:
        for item in [x for x in spec.split(",") if x.strip()]:
            f = item.split(":")
            name, dirs = f[0].strip(), [d for d in f[1].split("+") if d]
            if name not in POOL_SOURCES:
                raise ValueError(f"unknown crystal pool {name!r} (known: {sorted(POOL_SOURCES)})")
            out.append((name, dirs, float(f[2]) if len(f) > 2 and f[2] else shares.get(name, DEFAULT_SHARES[name])))
    elif root:
        for name, subs in roots.items():
            dirs = [str(Path(root) / s) for s in subs if (Path(root) / s).exists()]
            if len(dirs) == len(subs):
                out.append((name, dirs, shares[name]))
    return out


def build_source(name: str, dirs: list[str], max_atoms: int = 400, *, e0_tables: tuple[np.ndarray, np.ndarray] | None = None,
                 derive_u_atoms: bool = True, cache_u_atoms: bool = True, u_atom_cache_bytes: int | None = None) -> CrystalSource:
    idx = None
    if len(dirs) == 1:
        n_total = len(MatFlat(dirs[0]))
        mp_sp = SPLITS / "mptrj_train_idx.npy"
        if name == "mptrj_salex" and n_total >= MPTRJ_N_TOTAL and mp_sp.exists():
            mp_tr = np.asarray(np.load(mp_sp), dtype=np.int64)
            idx = np.concatenate([mp_tr, np.arange(MPTRJ_N_TOTAL, n_total, dtype=np.int64)])
        elif name in POOL_TRAIN_SPLIT and (SPLITS / POOL_TRAIN_SPLIT[name]).exists():
            idx = np.load(SPLITS / POOL_TRAIN_SPLIT[name])
        elif (Path(dirs[0]) / "splits" / "train_idx.npy").exists():
            idx = np.load(Path(dirs[0]) / "splits" / "train_idx.npy")
    kw = {}
    if e0_tables is not None:
        kw["e0_tables"] = e0_tables
    if not derive_u_atoms:
        kw["derive_u_atoms"] = False
    if not cache_u_atoms:
        kw["cache_u_atoms"] = False
    if u_atom_cache_bytes is not None:
        kw["u_atom_cache_bytes"] = int(u_atom_cache_bytes)
    return CrystalSource(dirs, POOL_SOURCES[name], indices=idx, max_atoms=max_atoms, **kw)


def build_crystal_domain(pools: list[tuple[str, list[str], float]], n_pad: int, seed: int, device, plan: dict | None = None, k: int = 128, log=print, *,
                         pair_frac: float = 0.0) -> tuple[DomainSampler, dict]:
    """One stream per pool, mixed by the pool shares. plan = {"n_pad", "k", "e_pad"} from a previous run (plans.json), else e_pad = the max over the pools' calibrations."""
    if not pools:
        raise ValueError("no crystal pools")
    k = plan["k"] if plan else k
    streams = {}
    for i, (name, dirs, _) in enumerate(pools):
        src = build_source(name, dirs)
        streams[name] = CrystalStream(src, n_pad=n_pad, k=k, seed=seed + 100 + i, device=device, e_pad=plan["e_pad"] if plan else None, pair_frac=pair_frac)
        uc = getattr(src, "_ucache", None)
        cache_note = f", u_atom_cache_cap={uc.max_bytes // (1024 * 1024)}MiB" if uc is not None else ""
        ps = streams[name].pair_stats
        pair_note = f", close_pairs={ps['n_pairs']:,} ({'cached' if ps.get('cached') else str(ps.get('build_s', 0)) + 's'})" if ps else ""
        log(f"crystal pool {name}: {len(src):,} structures from {len(dirs)} dir(s), e_pad {streams[name].e_pad}{cache_note}{pair_note}")
    uc_streams = [s for s in streams.values() if getattr(s.src, "_ucache", None) is not None]
    if len(uc_streams) > 1:
        from lam4.data.omol import _mem_available_gb
        avail_gb = _mem_available_gb()
        if avail_gb > 0.0:
            tot_cap = min(sum(s.src._ucache.max_bytes for s in uc_streams), max(32 * 1024 * 1024, int(avail_gb * 0.08 * (1024 ** 3))))
            tot_n = max(1, sum(len(s.src) for s in uc_streams))
            for s in uc_streams:
                s.src._ucache.max_bytes = max(4 * 1024 * 1024, int(tot_cap * (len(s.src) / tot_n)))
    e_pad = plan["e_pad"] if plan else max(s.e_pad for s in streams.values())
    for s in streams.values():
        s.e_pad = e_pad
    return DomainSampler(streams, {n: sh for n, _, sh in pools}), {"n_pad": n_pad, "k": k, "e_pad": int(e_pad)}


def val_sets(root: str, cap: int, seed: int = 0, max_atoms: int = 400, *, profile: str = "lam5") -> dict[str, CrystalSource]:
    """The registry validation splits found under `root`, each capped to `cap` random structures (0 = all)."""
    out = {}
    for split, (pool, sub) in VAL_SETS.items():
        if profile == "lam6" and split == "oc22_val":
            continue
        d = Path(root) / sub
        if not d.exists():
            for alt in (Path(root) / pool, Path(root) / "lam2_pool_slices" / pool):
                if alt.exists():
                    d = alt
                    break
        sp = SPLITS / f"{split}.npy"
        if not sp.exists():
            sp = d / "splits" / "val_idx.npy"
        if not (d.exists() and (sp.exists() or split in WHOLE_FLAT_VAL)):
            continue
        flat = MatFlat(d)
        n_flat = len(flat)
        idx = np.arange(n_flat, dtype=np.int64) if split in WHOLE_FLAT_VAL else np.asarray(np.load(sp), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_flat)]
        idx = idx[np.isfinite(np.asarray(flat.energy[idx], np.float64)) & (np.asarray(flat.n_atoms[idx]) <= max_atoms)]   # usable rows BEFORE the cap: n == cap
        if cap and len(idx) > cap:
            idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
        out[split] = CrystalSource([str(d)], POOL_SOURCES[pool], indices=idx, max_atoms=max_atoms)
    return out
