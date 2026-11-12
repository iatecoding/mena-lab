"""Mena-LAM Dataset & Level-of-Theory (LoT) Registry Loader.

Enforces the invariant:
  Every dataset and Level of Theory must be registered in `lam/data/lot_registry.yaml`.
  A dataset without a registry entry cannot be loaded by the trainer.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch

REGISTRY_PATH = Path(__file__).resolve().parent / "lot_registry.yaml"
BUNDLED_E0_DIR = Path(__file__).resolve().parent / "e0_refs"
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_e0_path(rel_path: str | Path) -> Path:
    """Resolve an E0 reference file path across repo checkouts (`REPO_ROOT/data/...`) and wheel installs (`lam/data/e0_refs/...`)."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    repo_candidate = REPO_ROOT / p
    if repo_candidate.is_file():
        return repo_candidate
    bundled_candidate = BUNDLED_E0_DIR / p.name
    if bundled_candidate.is_file():
        return bundled_candidate
    return repo_candidate

LOT_NAME_TO_ID: dict[str, int] = {
    "mp_pbe": 0,
    "mp_pbeU": 1,
    "omat_pbe": 2,
    "omat_pbeU": 3,
    "matpes_pbe": 4,
    "matpes_r2scan": 5,
    "omol_wb97mv": 6,
    "oc20_rpbe": 7,
    "oc22_pbeU": 8,
    "omc25_pbe": 9,
    "openrxn_wb97xd": 10,
    "spice_wb97md3": 11,
}
LOT_ID_TO_NAME: dict[int, str] = {v: k for k, v in LOT_NAME_TO_ID.items()}
ID_TO_LOT_NAME: dict[int, str] = LOT_ID_TO_NAME
LOT_NAMES: tuple[str, ...] = tuple(LOT_ID_TO_NAME[i] for i in range(len(LOT_ID_TO_NAME)))
LOT_TO_ID: dict[str, int] = LOT_NAME_TO_ID

_FALLBACK_LEVELS_OF_THEORY: dict[str, dict[str, Any]] = {
    "mp_pbe": {
        "lot_id": 0,
        "domain": "bulk",
        "pbc": True,
        "electrostatics": False,
        "functional": "PBE",
        "u_flag": 0,
        "d3_policy": "none",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "eV",
        "force_unit": "eV/A",
        "ev_to_kcal": 23.060548,
        "e0_file": "data/e0_reference_peratom.pt",
        "e0_column": 0,
    },
    "mp_pbeU": {
        "lot_id": 1,
        "domain": "bulk",
        "pbc": True,
        "electrostatics": False,
        "functional": "PBE+U",
        "u_flag": 1,
        "d3_policy": "none",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "eV",
        "force_unit": "eV/A",
        "ev_to_kcal": 23.060548,
        "e0_file": "data/e0_reference_peratom.pt",
        "e0_column": 1,
    },
    "omat_pbe": {
        "lot_id": 2,
        "domain": "bulk",
        "pbc": True,
        "electrostatics": False,
        "functional": "PBE",
        "u_flag": 0,
        "d3_policy": "none",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "eV",
        "force_unit": "eV/A",
        "ev_to_kcal": 23.060548,
        "e0_file": "data/e0_reference_omat24_peratom.pt",
        "e0_column": 0,
    },
    "omat_pbeU": {
        "lot_id": 3,
        "domain": "bulk",
        "pbc": True,
        "electrostatics": False,
        "functional": "PBE+U",
        "u_flag": 1,
        "d3_policy": "none",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "eV",
        "force_unit": "eV/A",
        "ev_to_kcal": 23.060548,
        "e0_file": "data/e0_reference_omat24_peratom.pt",
        "e0_column": 1,
    },
    "matpes_pbe": {
        "lot_id": 4,
        "domain": "bulk",
        "pbc": True,
        "electrostatics": False,
        "functional": "PBE",
        "u_flag": 0,
        "d3_policy": "none",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "eV",
        "force_unit": "eV/A",
        "ev_to_kcal": 23.060548,
        "e0_file": "data/e0_reference_matpes_pbe_peratom.pt",
        "e0_column": 0,
    },
    "matpes_r2scan": {
        "lot_id": 5,
        "domain": "bulk",
        "pbc": True,
        "electrostatics": False,
        "functional": "r2SCAN",
        "u_flag": 0,
        "d3_policy": "none",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "eV",
        "force_unit": "eV/A",
        "ev_to_kcal": 23.060548,
        "e0_file": "data/e0_reference_matpes_r2scan_peratom.pt",
        "e0_column": 0,
    },
    "omol_wb97mv": {
        "lot_id": 6,
        "domain": "molecules",
        "pbc": False,
        "electrostatics": True,
        "functional": "wB97M-V",
        "u_flag": 0,
        "d3_policy": "subtract",
        "d3_params": {
            "functional": "WB97M_D3BJ",
            "s6": 1.0,
            "s8": 0.3908,
            "a1": 0.566,
            "a2_bohr": 3.128,
            "r_on_angstrom": 9.0,
            "r_cut_angstrom": 12.0,
        },
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "kcal/mol",
        "force_unit": "kcal/mol/A",
        "ev_to_kcal": 23.060547830619026,
        "e0_file": "data/e0_reference_omol25.pt",
        "e0_json_file": "data/e0_reference_omol25.json",
    },
    "oc20_rpbe": {
        "lot_id": 7,
        "domain": "catalysis",
        "pbc": True,
        "electrostatics": False,
        "functional": "RPBE",
        "u_flag": 0,
        "d3_policy": "none",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "eV",
        "force_unit": "eV/A",
        "ev_to_kcal": 23.060548,
        "e0_file": "data/e0_reference_oc20_rpbe_branchcal.pt",
        "e0_column": 0,
    },
    "oc22_pbeU": {
        "lot_id": 8,
        "domain": "catalysis",
        "pbc": True,
        "electrostatics": False,
        "functional": "PBE+U",
        "u_flag": 0,
        "d3_policy": "none",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "eV",
        "force_unit": "eV/A",
        "ev_to_kcal": 23.060548,
        "e0_file": "data/e0_reference_oc22_pbeU_branchcal.pt",
        "e0_column": 0,
    },
    "omc25_pbe": {
        "lot_id": 9,
        "domain": "molecular_crystals",
        "pbc": True,
        "electrostatics": True,
        "functional": "PBE",
        "u_flag": 0,
        "d3_policy": "subtract",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "eV",
        "force_unit": "eV/A",
        "ev_to_kcal": 23.060548,
        "e0_file": "data/e0_reference_omc25_pbe_omatcal.pt",
        "e0_column": 0,
    },
    "openrxn_wb97xd": {
        "lot_id": 10,
        "domain": "molecules",
        "pbc": False,
        "electrostatics": True,
        "functional": "wB97X-D",
        "u_flag": 0,
        "d3_policy": "none",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "kcal/mol",
        "force_unit": "kcal/mol/A",
        "ev_to_kcal": 23.060547830619026,
        "e0_file": None,
    },
    "spice_wb97md3": {
        "lot_id": 11,
        "domain": "molecules",
        "pbc": False,
        "electrostatics": True,
        "functional": "wB97M-D3(BJ)",
        "u_flag": 0,
        "d3_policy": "subtract",
        "r_cut": 6.0,
        "r_on": 5.0,
        "c3_taper": True,
        "energy_unit": "kcal/mol",
        "force_unit": "kcal/mol/A",
        "ev_to_kcal": 23.060547830619026,
        "e0_file": None,
        "e0_json_file": "data/e0_reference_spice2.json",
    },
}


def _load_raw(reg_path: Path) -> dict[str, Any]:
    """Load raw registry dict from `reg_path`, using `yaml_lite` (or fallback LoTs) if PyYAML is not installed."""
    if not reg_path.exists():
        raise FileNotFoundError(f"Mena-LAM LoT registry not found at {reg_path}")
    try:
        import yaml

        with open(reg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    try:
        from mena_app import yaml_lite

        return yaml_lite.safe_load(reg_path.read_text(encoding="utf-8"))
    except Exception:
        if reg_path.resolve() == REGISTRY_PATH.resolve():
            return {
                "schema_version": "1.0",
                "levels_of_theory": copy.deepcopy(_FALLBACK_LEVELS_OF_THEORY),
                "datasets": {},
                "validation_splits": {},
            }
        raise


def load_lot_registry(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate `lot_registry.yaml`."""
    reg_path = Path(path) if path is not None else REGISTRY_PATH
    data = _load_raw(reg_path)
    if not isinstance(data, dict) or "levels_of_theory" not in data or "datasets" not in data:
        raise ValueError(f"Malformed LoT registry in {reg_path}")
    yaml_lots = {name: int(cfg["lot_id"]) for name, cfg in data["levels_of_theory"].items()}
    if yaml_lots != LOT_NAME_TO_ID:
        raise ValueError(f"lot_registry.yaml LoT IDs {yaml_lots} do not match LOT_NAME_TO_ID {LOT_NAME_TO_ID}")
    return data


def lot_by_name(lot_name: str | int, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Alias for `get_lot_entry`."""
    return get_lot_entry(lot_name, registry=registry)


def get_lot_entry(lot_name: str | int, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the Level-of-Theory entry for `lot_name` (or integer `lot_id`), raising KeyError if unregistered."""
    reg = load_lot_registry() if registry is None else registry
    if isinstance(lot_name, int) and not isinstance(lot_name, bool):
        if lot_name not in LOT_ID_TO_NAME:
            raise KeyError(f"Unregistered Level of Theory ID {lot_name}. Registered IDs: {sorted(LOT_ID_TO_NAME)}")
        lot_name = LOT_ID_TO_NAME[lot_name]
    lots = reg["levels_of_theory"]
    if lot_name not in lots:
        raise KeyError(
            f"Unregistered Level of Theory {lot_name!r}. "
            f"Registered LoTs: {sorted(lots.keys())}"
        )
    return lots[lot_name]


def get_dataset_entry(
    dataset_name: str,
    registry: dict[str, Any] | None = None,
    verify_e0: bool = False,
) -> dict[str, Any]:
    """Return the dataset registry entry for `dataset_name`.

    Raises KeyError if `dataset_name` is not registered in `lot_registry.yaml`.
    Optionally verifies that the local E0 reference file exists and matches its recorded MD5.
    """
    reg = load_lot_registry() if registry is None else registry
    datasets = reg["datasets"]
    if dataset_name not in datasets:
        raise KeyError(
            f"Dataset {dataset_name!r} is not registered in lot_registry.yaml and cannot be loaded. "
            f"Registered datasets: {sorted(datasets.keys())}"
        )
    entry = datasets[dataset_name]
    if verify_e0 and entry.get("e0_file"):
        e0_path = resolve_e0_path(entry["e0_file"])
        if not e0_path.exists():
            raise FileNotFoundError(f"Missing E0 file {e0_path} for dataset {dataset_name!r}")
        expected_md5 = entry.get("e0_file_md5")
        if expected_md5 and not str(expected_md5).startswith("pending"):
            actual_md5 = hashlib.md5(e0_path.read_bytes()).hexdigest()
            if actual_md5 != expected_md5:
                raise ValueError(
                    f"E0 MD5 mismatch for {dataset_name!r} ({e0_path}): "
                    f"expected {expected_md5}, got {actual_md5}"
                )
    return entry


def resolve_lot_id_by_u(
    spec: tuple[int, int] | list[int] | dict[int, int] | str | int | None,
    registry: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Resolve `spec` (dataset name, LoT name, `(id_u0, id_u1)` tuple, `{0: id_u0, 1: id_u1}` dict, or `int`)
    into a `(lot_id_u0, lot_id_u1)` pair mapping `u_flag in {0, 1}` -> `lot_id`."""
    if spec is None:
        return (LOT_NAME_TO_ID["mp_pbe"], LOT_NAME_TO_ID["mp_pbeU"])
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        v = int(spec)
        if v == LOT_NAME_TO_ID["mp_pbe"]:
            return (LOT_NAME_TO_ID["mp_pbe"], LOT_NAME_TO_ID["mp_pbeU"])
        if v == LOT_NAME_TO_ID["omat_pbe"]:
            return (LOT_NAME_TO_ID["omat_pbe"], LOT_NAME_TO_ID["omat_pbeU"])
        return (v, v)
    if isinstance(spec, dict):
        id0 = int(spec[0] if 0 in spec else spec["0"])
        id1 = int(spec.get(1, spec.get("1", id0)))
        return (id0, id1)
    if isinstance(spec, (tuple, list)):
        if len(spec) == 1:
            return (int(spec[0]), int(spec[0]))
        if len(spec) == 2:
            return (int(spec[0]), int(spec[1]))
        raise ValueError(f"lot_id_by_u tuple/list must have length 1 or 2, got {spec!r}")
    if isinstance(spec, str):
        reg = load_lot_registry() if registry is None else registry
        if spec in reg["datasets"]:
            ds = reg["datasets"][spec]
            if "lot_id_by_u_flag" in ds:
                m = ds["lot_id_by_u_flag"]
                return (int(m[0] if 0 in m else m["0"]), int(m[1] if 1 in m else m["1"]))
            if "lot_id" in ds:
                v = int(ds["lot_id"])
                return (v, v)
        if spec in LOT_NAME_TO_ID or spec == "omat24":
            if spec in ("mp_pbe", "mp_pbeU"):
                return (LOT_NAME_TO_ID["mp_pbe"], LOT_NAME_TO_ID["mp_pbeU"]) if spec == "mp_pbe" else (1, 1)
            if spec in ("omat_pbe", "omat_pbeU", "omat24"):
                return (LOT_NAME_TO_ID["omat_pbe"], LOT_NAME_TO_ID["omat_pbeU"]) if spec != "omat_pbeU" else (3, 3)
            v = LOT_NAME_TO_ID[spec]
            return (v, v)
        raise KeyError(f"Unknown dataset or LoT name {spec!r} in resolve_lot_id_by_u")
    raise TypeError(f"Unsupported lot_id_by_u specification: {spec!r}")


def load_lot_e0_table_f64(
    lot_name: str,
    registry: dict[str, Any] | None = None,
    num_elements: int | None = None,
    in_kcal: bool = False,
) -> torch.Tensor:
    """Load the frozen float64 per-element E0 reference vector (in eV by default, or kcal/mol if `in_kcal=True`)
    for `lot_name`. If `num_elements` is given (e.g. 119), pads/slices to `[num_elements]`."""
    lot = get_lot_entry(lot_name, registry=registry)
    if lot.get("e0_file") is None:  # not fitted yet: zeros (the trainer refuses active LoTs with an all-zero row)
        return torch.zeros(int(num_elements) if num_elements is not None else 119, dtype=torch.float64)
    e0_path = resolve_e0_path(lot["e0_file"])
    ckpt = torch.load(e0_path, map_location="cpu", weights_only=False)
    ev_scale = float(lot.get("ev_to_kcal", 23.060548))
    if isinstance(ckpt, dict) and ("e0_kcal" in ckpt or "e0_ev" in ckpt) and lot.get("domain") == "molecules":
        if in_kcal and "e0_kcal" in ckpt:
            vec = ckpt["e0_kcal"].to(torch.float64).contiguous()
        elif not in_kcal and "e0_ev" in ckpt:
            vec = ckpt["e0_ev"].to(torch.float64).contiguous()
        else:
            base = ckpt["e0_kcal"].to(torch.float64)
            vec = (base if in_kcal else base / ev_scale).contiguous()
    else:
        mu_table = ckpt.get("mu_f64", ckpt["mu"]).to(torch.float64) if isinstance(ckpt, dict) else ckpt.to(torch.float64)
        col = int(lot.get("e0_column", 0))
        vec = mu_table[:, col].contiguous()
        if in_kcal:
            vec = vec * ev_scale
    if num_elements is not None:
        out = torch.zeros(int(num_elements), dtype=torch.float64)
        n_copy = min(int(num_elements), int(vec.numel()))
        out[:n_copy] = vec[:n_copy]
        return out
    return vec


def load_all_lot_e0_tables_f64(
    num_lots: int = 8,
    num_elements: int = 119,
    in_kcal: bool = True,
    registry: dict[str, Any] | None = None,
    z_max: int | None = None,
) -> torch.Tensor:
    """Load the full `[num_lots, num_elements]` float64 E0 table across all registered LoTs with id < num_lots."""
    if z_max is not None:
        num_elements = int(z_max)
    reg = load_lot_registry() if registry is None else registry
    table = torch.zeros((int(num_lots), int(num_elements)), dtype=torch.float64)
    for lot_name, lot_id in LOT_NAME_TO_ID.items():
        if lot_id < num_lots:
            table[lot_id] = load_lot_e0_table_f64(
                lot_name, registry=reg, num_elements=num_elements, in_kcal=in_kcal
            )
    return table


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_validation_split_entry(
    split_name: str,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the validation split entry for `split_name` from `lot_registry.yaml`."""
    reg = load_lot_registry() if registry is None else registry
    splits = reg.get("validation_splits", {})
    if split_name not in splits:
        raise KeyError(
            f"Validation split {split_name!r} is not registered in lot_registry.yaml. "
            f"Registered splits: {sorted(splits.keys())}"
        )
    return splits[split_name]


def load_and_verify_validation_split(
    split_name: str,
    registry: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    verify_sha256: bool = True,
    verify_disjoint: bool = True,
) -> Any:
    """Load a registered validation split `.npy` array, verifying SHA-256 and train-split disjointness."""
    import numpy as np

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    entry = get_validation_split_entry(split_name, registry=registry)
    split_path = root / entry["path"]
    if not split_path.exists():
        raise FileNotFoundError(f"Validation split file {split_path} not found for {split_name!r}")

    if verify_sha256 and entry.get("sha256"):
        actual_sha = _sha256_path(split_path)
        if actual_sha != entry["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for validation split {split_name!r} ({split_path}): "
                f"expected {entry['sha256']}, got {actual_sha}"
            )

    val_arr = np.load(split_path)
    expected_n = entry.get("num_structures")
    if expected_n is not None and len(val_arr) != int(expected_n):
        raise ValueError(
            f"Structure count mismatch for validation split {split_name!r}: "
            f"expected {expected_n}, got {len(val_arr)}"
        )

    if verify_disjoint and entry.get("train_split_path"):
        train_path = root / entry["train_split_path"]
        if train_path.exists():
            if verify_sha256 and entry.get("train_split_sha256"):
                tr_sha = _sha256_path(train_path)
                if tr_sha != entry["train_split_sha256"]:
                    raise ValueError(
                        f"SHA-256 mismatch for train split {train_path}: "
                        f"expected {entry['train_split_sha256']}, got {tr_sha}"
                    )
            train_arr = np.load(train_path)
            overlap = len(set(train_arr.tolist()) & set(val_arr.tolist()))
            if overlap > 0:
                raise ValueError(
                    f"Train/val leakage detected for {split_name!r}: "
                    f"{overlap} overlapping indices between {train_path} and {split_path}"
                )

    return val_arr


