"""sampler.py: Temperature-balanced multi-dataset sampling and final-phase alignment schedule for Mena-LAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lam.data.registry import (
    ID_TO_LOT_NAME,
    LOT_ID_TO_NAME,
    LOT_NAMES,
    LOT_NAME_TO_ID,
    LOT_TO_ID,
    resolve_lot_id_by_u,
)


DEFAULT_CRYSTAL_FLOORS: dict[str, float] = {
    "mptrj": 0.10,
    "mptrj_salex": 0.20,
    "salex": 0.15,
    "omat24_sub": 0.15,
    "omat24": 0.15,
    "matpes_pbe": 0.05,
    "matpes_r2scan": 0.05,
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    stream: str  # "crystal" or "molecule"
    lot_name: str
    size: int
    weight_override: float | None = None
    min_share: float = 0.0

    @property
    def lot_id(self) -> int:
        return LOT_NAME_TO_ID[self.lot_name]

    @property
    def lot_id_by_u(self) -> tuple[int, int]:
        try:
            return resolve_lot_id_by_u(self.name)
        except KeyError:
            return resolve_lot_id_by_u(self.lot_name)


def _project_with_floors(weights: np.ndarray, floors: np.ndarray) -> np.ndarray:
    """Normalize positive `weights` onto the simplex subject to `p_i >= floors_i` while preserving
    relative ratios among unconstrained entries."""
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-12)
    f = np.clip(np.asarray(floors, dtype=np.float64), 0.0, 1.0)
    f_sum = float(f.sum())
    if f_sum >= 1.0 - 1e-9:
        return f / max(f_sum, 1e-12)

    fixed = np.zeros(n, dtype=bool)
    p = np.zeros(n, dtype=np.float64)
    for _ in range(n + 1):
        rem_mass = max(0.0, 1.0 - float(f[fixed].sum()))
        free = ~fixed
        if not free.any():
            break
        w_free_sum = float(w[free].sum())
        p[fixed] = f[fixed]
        p[free] = rem_mass * (w[free] / max(w_free_sum, 1e-12))
        violated = free & (p < f - 1e-12)
        if not violated.any():
            break
        fixed |= violated
    p[fixed] = f[fixed]
    return p / max(float(p.sum()), 1e-12)


class MultiDatasetSampler:
    """Temperature-balanced dataset sampler:
        p_{D | stream} \\propto N_D^\\alpha   (default \\alpha = 0.5)
    with per-dataset probability floors (`min_crystal_floor` / `crystal_floors` / `DatasetSpec.min_share`),
    explicit stream probability split (`p_cryst = 0.65`, `p_mol = 0.35` when both streams are present),
    and final-phase MP alignment (final `align_frac = 0.15` of steps shifts `mp_pbe` / `mp_pbeU`
    to >= `min_mp_crystal_share = 0.70` of crystal draws while keeping `omol_wb97mv` active).
    """

    def __init__(
        self,
        specs: Sequence[DatasetSpec],
        alpha: float = 0.5,
        align_frac: float = 0.15,
        align_boost: float = 2.0,
        min_mp_crystal_share: float = 0.70,
        min_crystal_floor: float = 0.05,
        crystal_floors: dict[str, float] | None = None,
        p_cryst: float = 0.65,
        p_mol: float = 0.35,
        seed: int = 42,
    ) -> None:
        if not specs:
            raise ValueError("specs must be non-empty")
        self.specs = list(specs)
        self.alpha = float(alpha)
        self.align_frac = float(align_frac)
        self.align_boost = float(align_boost)
        self.min_mp_crystal_share = float(min_mp_crystal_share)
        self.min_crystal_floor = float(min_crystal_floor)
        self.crystal_floors = (
            dict(crystal_floors) if crystal_floors is not None else dict(DEFAULT_CRYSTAL_FLOORS)
        )
        self.p_cryst = float(p_cryst)
        self.p_mol = float(p_mol)
        self.rng = np.random.default_rng(seed)

    def _raw_weights(self) -> np.ndarray:
        return np.array(
            [
                s.weight_override if s.weight_override is not None else max(1, int(s.size)) ** self.alpha
                for s in self.specs
            ],
            dtype=np.float64,
        )

    def _crystal_floors_for_specs(self, cryst_specs: Sequence[DatasetSpec]) -> np.ndarray:
        if len(cryst_specs) <= 1:
            return np.zeros(len(cryst_specs), dtype=np.float64)
        floors = []
        for s in cryst_specs:
            f_val = max(
                float(s.min_share),
                float(self.crystal_floors.get(s.name, 0.0)),
                float(self.min_crystal_floor),
            )
            floors.append(f_val)
        return np.asarray(floors, dtype=np.float64)

    def probabilities(self, progress: float = 0.0) -> np.ndarray:
        """Compute normalized sampling probabilities at training progress in [0, 1]."""
        raw = self._raw_weights()
        cryst_mask = np.array([s.stream == "crystal" for s in self.specs], dtype=bool)
        mol_mask = np.array([s.stream == "molecule" for s in self.specs], dtype=bool)
        mp_mask = np.array(
            [s.stream == "crystal" and s.lot_name in ("mp_pbe", "mp_pbeU") for s in self.specs],
            dtype=bool,
        )
        non_mp_cryst_mask = cryst_mask & ~mp_mask

        if cryst_mask.any():
            cryst_specs = [s for s in self.specs if s.stream == "crystal"]
            cryst_floors = self._crystal_floors_for_specs(cryst_specs)
            raw[cryst_mask] = _project_with_floors(raw[cryst_mask], cryst_floors)

        if progress >= (1.0 - self.align_frac) and self.align_frac > 0.0 and mp_mask.any():
            ramp = min(1.0, max(0.0, (progress - (1.0 - self.align_frac)) / self.align_frac))
            if non_mp_cryst_mask.any():
                mp_base = float(raw[mp_mask].sum())
                non_mp_base = float(raw[non_mp_cryst_mask].sum())
                base_share = mp_base / max(mp_base + non_mp_base, 1e-12)
                boosted_mp = mp_base * self.align_boost
                boosted_share = boosted_mp / max(boosted_mp + non_mp_base, 1e-12)
                end_share = max(boosted_share, self.min_mp_crystal_share)
                target_share = base_share + (end_share - base_share) * ramp
                target_share = min(max(target_share, 0.0), 1.0 - 1e-9)
                raw[mp_mask] *= (target_share / max(base_share, 1e-12)) * (
                    non_mp_base / max((1.0 - target_share) * (mp_base + non_mp_base), 1e-12)
                )
            else:
                raw[mp_mask] *= 1.0 + (self.align_boost - 1.0) * ramp

        if cryst_mask.any() and mol_mask.any():
            tot_stream = max(self.p_cryst + self.p_mol, 1e-12)
            wc = self.p_cryst / tot_stream
            wm = self.p_mol / tot_stream
            probs = np.zeros_like(raw)
            probs[cryst_mask] = wc * raw[cryst_mask] / max(float(raw[cryst_mask].sum()), 1e-12)
            probs[mol_mask] = wm * raw[mol_mask] / max(float(raw[mol_mask].sum()), 1e-12)
            return probs / probs.sum()

        return raw / raw.sum()

    def crystal_probabilities(self, progress: float = 0.0) -> dict[str, float]:
        """Return normalized probabilities restricted to the crystal stream."""
        probs = self.probabilities(progress)
        cryst = [(s.name, float(probs[i])) for i, s in enumerate(self.specs) if s.stream == "crystal"]
        tot = sum(p for _, p in cryst)
        return {name: p / max(tot, 1e-12) for name, p in cryst}

    def sample(self, progress: float = 0.0) -> DatasetSpec:
        probs = self.probabilities(progress)
        idx = int(self.rng.choice(len(self.specs), p=probs))
        return self.specs[idx]

    def sample_domain(self, progress: float = 0.0) -> str:
        """Sample either 'crystal' or 'molecule' according to stream probabilities at `progress`."""
        return self.sample(progress).stream

    def sample_crystal(self, progress: float = 0.0) -> DatasetSpec:
        """Sample a crystal dataset spec according to normalized crystal probabilities at `progress`."""
        cryst_specs = [s for s in self.specs if s.stream == "crystal"]
        if not cryst_specs:
            raise ValueError("No crystal datasets registered in MultiDatasetSampler")
        probs = self.probabilities(progress)
        cryst_p = np.array([probs[i] for i, s in enumerate(self.specs) if s.stream == "crystal"], dtype=np.float64)
        cryst_p = cryst_p / max(float(cryst_p.sum()), 1e-12)
        idx = int(self.rng.choice(len(cryst_specs), p=cryst_p))
        return cryst_specs[idx]

    def sample_molecule(self, progress: float = 0.0) -> DatasetSpec:
        """Sample a molecule dataset spec according to normalized molecule probabilities at `progress`."""
        mol_specs = [s for s in self.specs if s.stream == "molecule"]
        if not mol_specs:
            raise ValueError("No molecule datasets registered in MultiDatasetSampler")
        probs = self.probabilities(progress)
        mol_p = np.array([probs[i] for i, s in enumerate(self.specs) if s.stream == "molecule"], dtype=np.float64)
        mol_p = mol_p / max(float(mol_p.sum()), 1e-12)
        idx = int(self.rng.choice(len(mol_specs), p=mol_p))
        return mol_specs[idx]

    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "align_frac": self.align_frac,
            "align_boost": self.align_boost,
            "min_mp_crystal_share": self.min_mp_crystal_share,
            "min_crystal_floor": self.min_crystal_floor,
            "crystal_floors": dict(self.crystal_floors),
            "p_cryst": self.p_cryst,
            "p_mol": self.p_mol,
            "specs": [
                {
                    "name": s.name,
                    "stream": s.stream,
                    "lot_name": s.lot_name,
                    "size": int(s.size),
                    "weight_override": s.weight_override,
                    "min_share": float(s.min_share),
                }
                for s in self.specs
            ],
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if "alpha" in state:
            self.alpha = float(state["alpha"])
        if "align_frac" in state:
            self.align_frac = float(state["align_frac"])
        if "align_boost" in state:
            self.align_boost = float(state["align_boost"])
        if "min_mp_crystal_share" in state:
            self.min_mp_crystal_share = float(state["min_mp_crystal_share"])
        if "min_crystal_floor" in state:
            self.min_crystal_floor = float(state["min_crystal_floor"])
        if "crystal_floors" in state and state["crystal_floors"] is not None:
            self.crystal_floors = dict(state["crystal_floors"])
        if "p_cryst" in state:
            self.p_cryst = float(state["p_cryst"])
        if "p_mol" in state:
            self.p_mol = float(state["p_mol"])
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]


MultiLoTSampler = MultiDatasetSampler


@dataclass(frozen=True)
class PoolSpec:
    """One extra training pool (`--crystal-pool`): a MatFlat directory under one LoT token, optionally restricted to
    selected rows and with a seeded held-out validation slice."""

    name: str
    lot: str
    path: str
    rows: str | None = None
    floor: float = 0.0
    val: int = 1000
    weight: float | None = None
    valrows: str | None = None  # held-out GROUP rows (the flat's splits/val_idx.npy): excluded from training entirely


def parse_pool_spec(text: str) -> PoolSpec:
    """`name=oc20,lot=oc20_rpbe,path=/data/oc20_flat[,rows=sel.npy][,floor=0.10][,val=1000][,weight=1.0]
    [,valrows=/data/oc20_flat/splits/val_idx.npy]`."""
    kv = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"pool spec {text!r}: expected key=value, got {part!r}")
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    missing = {"name", "lot", "path"} - set(kv)
    if missing:
        raise ValueError(f"pool spec {text!r}: missing {sorted(missing)}")
    unknown = set(kv) - {"name", "lot", "path", "rows", "floor", "val", "weight", "valrows"}
    if unknown:
        raise ValueError(f"pool spec {text!r}: unknown keys {sorted(unknown)}")
    if kv["lot"] not in LOT_NAME_TO_ID:
        raise ValueError(f"pool spec {text!r}: LoT {kv['lot']!r} is not registered")
    return PoolSpec(
        name=kv["name"], lot=kv["lot"], path=kv["path"], rows=kv.get("rows"), floor=float(kv.get("floor", 0.0)),
        val=int(kv.get("val", 1000)), weight=float(kv["weight"]) if "weight" in kv else None, valrows=kv.get("valrows"),
    )


@dataclass(frozen=True)
class MolExtraSpec:
    name: str
    lot: str
    dirs: tuple[str, ...]
    e0: str
    share: float
    val: str | None = None
    val_max: int = 2000


def parse_mol_extra(text: str) -> MolExtraSpec:
    """`name=spice2,lot=spice_wb97md3,dirs=/c/spice/train[;/c/spice/train_p1],e0=data/e0_reference_spice2.json,share=0.05
    [,val=/c/spice/val][,val_max=2000]`: an extra molecular stream with its own LoT token, E0 and validation set."""
    kv = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"mol-extra spec {text!r}: expected key=value, got {part!r}")
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    missing = {"name", "lot", "dirs", "e0", "share"} - set(kv)
    if missing:
        raise ValueError(f"mol-extra spec {text!r}: missing {sorted(missing)}")
    unknown = set(kv) - {"name", "lot", "dirs", "e0", "share", "val", "val_max"}
    if unknown:
        raise ValueError(f"mol-extra spec {text!r}: unknown keys {sorted(unknown)}")
    if kv["lot"] not in LOT_NAME_TO_ID:
        raise ValueError(f"mol-extra spec {text!r}: LoT {kv['lot']!r} is not registered")
    share = float(kv["share"])
    if not 0.0 <= share < 1.0:
        raise ValueError(f"mol-extra spec {text!r}: share must be in [0, 1)")
    dirs = tuple(d for d in kv["dirs"].split(";") if d)
    if not dirs:
        raise ValueError(f"mol-extra spec {text!r}: empty dirs")
    return MolExtraSpec(kv["name"], kv["lot"], dirs, kv["e0"], share, kv.get("val"), int(kv.get("val_max", 2000)))


def pick_mol_stream(shares: list[float], seed: int, step: int) -> int:
    """Index of the extra molecular stream drawn at `step` (-1: the main OMol stream). Deterministic in (seed, step), so a
    resumed run replays the same choice and the main stream's own order is untouched when every share is 0."""
    if not shares or sum(shares) <= 0.0:
        return -1
    u = float(np.random.default_rng([int(seed), 0x4D4F4C, int(step)]).random())
    c = 0.0
    for i, w in enumerate(shares):
        c += w
        if u < c:
            return i
    return -1


def split_pool_rows(rows: np.ndarray, n_val: int, seed: int,
                    val_pool: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(train, val): a seeded held-out slice of n_val rows; train keeps the input order of the rest.
    With `val_pool` (rows of held-out groups, e.g. whole trajectories): every val_pool row leaves training and val is a
    seeded sample of at most n_val of them, so near-duplicate frames of one trajectory never sit on both sides."""
    rows = np.asarray(rows, np.int64)
    if val_pool is not None:
        vp = np.asarray(val_pool, np.int64)
        tr = rows[~np.isin(rows, vp)]
        k = min(int(n_val), len(vp))
        va = np.sort(np.random.default_rng(seed).choice(vp, size=k, replace=False)) if k > 0 else vp[:0]
        return tr, va
    n_val = min(int(n_val), len(rows) // 10)
    val = np.sort(np.random.default_rng(seed).choice(rows, size=n_val, replace=False)) if n_val > 0 else rows[:0]
    return rows[~np.isin(rows, val)], val

