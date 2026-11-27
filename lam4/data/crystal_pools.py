"""crystal_pools.py: the crystal domain of the lam4 loader = several MatFlat pools, one `CrystalStream` each, mixed by `DomainSampler` with the pool shares (shares are within the
crystal domain; the domain share against molecules is the outer sampler's). All pools share ONE (n_pad, k, e_pad), the max over the pools' calibrations, so the crystal batches are a
single static shape (one traced graph). Also the registry facts per pool (LoT pair by u_flag, train split) and the validation sets (split file -> flat dir -> pool sources).

Pool spec (CLI `--crystal-pools`): `name:dir[+dir...]:share,...`, share optional (default below). Names: mptrj, mptrj_salex, omat24, matpes_pbe, matpes_r2scan.
LoT of a structure: u_flag 0/1 -> (mp_pbe, mp_pbeU) for mptrj / sAlex, (omat_pbe, omat_pbeU) for OMat24, one LoT for MatPES (lam/data/lot_registry.yaml `lot_id_by_u_flag`).
Train rows: the registry train split when it exists (data/splits/*_train_idx.npy, clipped to the flat), else every row (OMat24 parts: val is a separate flat).
"""
from __future__ import annotations

import hashlib
import json
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
POOL_VAL_SPLIT = {"mptrj": "mptrj_val_2k.npy", "mptrj_salex": "salex_val_2k.npy", "matpes_pbe": "matpes_pbe_val_1k.npy", "matpes_r2scan": "matpes_r2scan_val_1k.npy"}
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_split_disjointness(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    pool: str = "",
    train_path: str | Path = "",
    val_path: str | Path = "",
) -> None:
    """Raise `ValueError` if train and validation row indices share any element."""
    tr = np.asarray(train_idx, dtype=np.int64).ravel()
    vl = np.asarray(val_idx, dtype=np.int64).ravel()
    overlap = np.intersect1d(tr, vl)
    if overlap.size > 0:
        raise ValueError(
            f"train/val split leakage detected for pool={pool!r}: {overlap.size} overlapping row indices "
            f"(first 5: {overlap[:5].tolist()}) between {train_path} and {val_path}"
        )


def verify_splits_dir(
    splits_dir: str | Path = SPLITS,
    *,
    manifest_path: str | Path | None = None,
    require_all: bool = True,
) -> dict[str, dict]:
    """Verify presence, non-emptiness, train/val disjointness, and optional SHA-256 checksums of registry splits."""
    sp_dir = Path(splits_dir)
    manifest_splits: dict = {}
    if manifest_path is not None:
        manifest_splits = json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("splits", {})
    manifest_key_by_pool = {
        "mptrj": "mp_pbe",
        "mptrj_salex": "mptrj_salex_pbe",
        "matpes_pbe": "matpes_pbe",
        "matpes_r2scan": "matpes_r2scan",
    }
    summary: dict[str, dict] = {}
    for pool, tr_name in POOL_TRAIN_SPLIT.items():
        tr_p = sp_dir / tr_name
        vl_name = POOL_VAL_SPLIT.get(pool)
        vl_p = (sp_dir / vl_name) if vl_name else None
        if not tr_p.is_file():
            if require_all:
                raise FileNotFoundError(f"missing required registry train split {tr_p} for pool={pool!r}")
            continue
        tr_idx = np.asarray(np.load(tr_p), dtype=np.int64)
        if tr_idx.size == 0:
            raise ValueError(f"empty train split {tr_p} for pool={pool!r}")
        tr_sha = _sha256_file(tr_p)
        vl_count = 0
        vl_sha = None
        if vl_p is not None and vl_p.is_file():
            vl_idx = np.asarray(np.load(vl_p), dtype=np.int64)
            verify_split_disjointness(tr_idx, vl_idx, pool=pool, train_path=tr_p, val_path=vl_p)
            vl_count = int(vl_idx.size)
            vl_sha = _sha256_file(vl_p)
        if pool == "mptrj" and (sp_dir / "val_idx.npy").is_file():
            verify_split_disjointness(tr_idx, np.load(sp_dir / "val_idx.npy"), pool=pool, train_path=tr_p, val_path=sp_dir / "val_idx.npy")
        m_entry = manifest_splits.get(manifest_key_by_pool.get(pool, ""), {})
        if m_entry.get("train_sha256") and tr_sha != m_entry["train_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {tr_p}: got {tr_sha}, expected {m_entry['train_sha256']}")
        if vl_sha is not None and m_entry.get("val_sha256") and vl_sha != m_entry["val_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {vl_p}: got {vl_sha}, expected {m_entry['val_sha256']}")
        summary[pool] = {
            "train_count": int(tr_idx.size),
            "val_count": vl_count,
            "overlap": 0,
            "train_sha256": tr_sha,
            "val_sha256": vl_sha,
        }
    return summary


def build_source(name: str, dirs: list[str], max_atoms: int = 400, *, e0_tables: tuple[np.ndarray, np.ndarray] | None = None,
                 derive_u_atoms: bool = True, cache_u_atoms: bool = True, u_atom_cache_bytes: int | None = None,
                 splits_dir: str | Path | None = None, allow_unsplit: bool = False) -> CrystalSource:
    idx = None
    sp_dir = SPLITS if splits_dir is None else Path(splits_dir)
    if len(dirs) == 1:
        n_total = len(MatFlat(dirs[0]))
        if name == "mptrj_salex" and n_total >= MPTRJ_N_TOTAL:
            mp_sp = sp_dir / "mptrj_train_idx.npy"
            if not mp_sp.is_file():
                if not allow_unsplit:
                    raise FileNotFoundError(
                        f"missing required registry train split {mp_sp} for pool {name!r} (n_total={n_total:,}); "
                        f"refusing silent fallback to unsplit flat (pass allow_unsplit=True only for synthetic smoke tests)"
                    )
            else:
                mp_tr = np.asarray(np.load(mp_sp), dtype=np.int64)
                if mp_tr.size == 0:
                    raise ValueError(f"empty train split {mp_sp} for pool {name!r}")
                for val_name in ("mptrj_val_2k.npy", "val_idx.npy"):
                    val_p = sp_dir / val_name
                    if val_p.is_file():
                        verify_split_disjointness(mp_tr, np.load(val_p), pool=name, train_path=mp_sp, val_path=val_p)
                idx = np.concatenate([mp_tr, np.arange(MPTRJ_N_TOTAL, n_total, dtype=np.int64)])
        elif name in POOL_TRAIN_SPLIT:
            tr_sp = sp_dir / POOL_TRAIN_SPLIT[name]
            if not tr_sp.is_file():
                if not allow_unsplit:
                    raise FileNotFoundError(
                        f"missing required registry train split {tr_sp} for pool {name!r}; "
                        f"refusing silent fallback to <flat>/splits/train_idx.npy or unsplit flat "
                        f"(pass allow_unsplit=True only for synthetic smoke tests)"
                    )
            else:
                idx = np.asarray(np.load(tr_sp), dtype=np.int64)
                if idx.size == 0:
                    raise ValueError(f"empty train split {tr_sp} for pool {name!r}")
                val_name = POOL_VAL_SPLIT.get(name)
                if val_name and (sp_dir / val_name).is_file():
                    verify_split_disjointness(idx, np.load(sp_dir / val_name), pool=name, train_path=tr_sp, val_path=sp_dir / val_name)
                if name == "mptrj" and (sp_dir / "val_idx.npy").is_file():
                    verify_split_disjointness(idx, np.load(sp_dir / "val_idx.npy"), pool=name, train_path=tr_sp, val_path=sp_dir / "val_idx.npy")
        elif (Path(dirs[0]) / "splits" / "train_idx.npy").is_file():
            local_tr = Path(dirs[0]) / "splits" / "train_idx.npy"
            idx = np.asarray(np.load(local_tr), dtype=np.int64)
            local_val = Path(dirs[0]) / "splits" / "val_idx.npy"
            if local_val.is_file():
                verify_split_disjointness(idx, np.load(local_val), pool=name, train_path=local_tr, val_path=local_val)
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
                         pair_frac: float = 0.0, splits_dir: str | Path | None = None, allow_unsplit: bool = False) -> tuple[DomainSampler, dict]:
    """One stream per pool, mixed by the pool shares. plan = {"n_pad", "k", "e_pad"} from a previous run (plans.json), else e_pad = the max over the pools' calibrations."""
    if not pools:
        raise ValueError("no crystal pools")
    k = plan["k"] if plan else k
    streams = {}
    for i, (name, dirs, _) in enumerate(pools):
        src = build_source(name, dirs, splits_dir=splits_dir, allow_unsplit=allow_unsplit)
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


def val_sets(root: str, cap: int, seed: int = 0, max_atoms: int = 400, *, profile: str = "lam5", splits_dir: str | Path | None = None) -> dict[str, CrystalSource]:
    """The registry validation splits found under `root`, each capped to `cap` random structures (0 = all)."""
    sp_dir = SPLITS if splits_dir is None else Path(splits_dir)
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
        sp = sp_dir / f"{split}.npy"
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
