"""Checkpoint & reference-table loading for MenaCalculator (`lam4` / `lam5` models)."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from lam.data.registry import load_lot_e0_table_f64, resolve_e0_path
from lam4.config import ModelConfig
from lam4.kernels.csr_ops import device_supports_csr
from lam4.model import Lam4Model
from lam4.registry.sources import reference_key_of

REPO_ROOT = Path(__file__).resolve().parents[2]
OMOL_REF_JSON = resolve_e0_path("data/e0_reference_omol25.json")
SPICE_REF_JSON = resolve_e0_path("data/e0_reference_spice2.json")

KNOWN_CFG_HASHES: dict[str, Callable[[], ModelConfig]] = {
    "ed15546b9b4c": lambda: ModelConfig.lam5(edge_gain_max=1.0),
    "2890592651d2": ModelConfig.lam5,
}

_CRYSTAL_E0_CACHE: dict[str, Tensor] = {}
_MOL_REF_CACHE: dict[str, dict[str, np.ndarray]] = {}


def resolve_device(device: str | torch.device = "auto", dtype: str | torch.dtype = "float32") -> torch.device:
    dt = resolve_dtype(dtype)
    if isinstance(device, torch.device):
        if device.type == "mps" and dt == torch.float64:
            return torch.device("cpu")
        return device
    dev = str(device).strip().lower()
    if dev == "auto":
        if dt == torch.float64:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if dev == "mps" and dt == torch.float64:
        return torch.device("cpu")
    return torch.device(dev)


def resolve_dtype(dtype: str | torch.dtype = "float32") -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).strip().lower()
    if key in ("float32", "fp32", "f32", "single"):
        return torch.float32
    if key in ("float64", "fp64", "f64", "double"):
        return torch.float64
    raise ValueError(f"Unsupported dtype {dtype!r}; expected 'float32' or 'float64'")


def _cfg_from_dict(d: dict) -> ModelConfig:
    clean = {}
    for k, v in d.items():
        clean[k] = tuple(v) if isinstance(v, list) else v
    return ModelConfig(**clean)


def resolve_lam4_config(ck: dict, state_dict: dict[str, Tensor] | None = None) -> ModelConfig:
    sd = ck.get("model", ck) if state_dict is None else state_dict
    if isinstance(ck.get("cfg"), dict):
        cfg = _cfg_from_dict(ck["cfg"])
    elif isinstance(ck.get("cfg"), ModelConfig):
        cfg = ck["cfg"]
    elif ck.get("cfg_hash") in KNOWN_CFG_HASHES:
        cfg = KNOWN_CFG_HASHES[str(ck["cfg_hash"])]()
    else:
        has_inv_norm = "trunk.stages.0.inv_scale" in sd
        has_aniso = "polarization.aniso_head.weight" in sd
        cfg = ModelConfig.lam5() if (has_inv_norm or has_aniso) else ModelConfig()
        if not has_aniso and cfg.aniso_polarizability:
            cfg = replace(cfg, aniso_polarizability=False)
    has_force_head = any(k.startswith("force_head.") for k in sd)
    if cfg.direct_force_head and not has_force_head:
        cfg = replace(cfg, direct_force_head=False)
    return cfg


def _raw_load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def resolve_edge_mode(edge_mode: str, dev: torch.device, dt: torch.dtype, cfg: ModelConfig, low_memory: bool) -> str:
    em = str(edge_mode).strip().lower()
    if em not in ("auto", "csr", "ref", "stream"):
        raise ValueError(f"edge_mode must be in {{'auto', 'csr', 'ref', 'stream'}}, got {edge_mode!r}")
    csr_ok = dt == torch.float32 and device_supports_csr(dev) and cfg.r_cut_near <= 0.0 and cfg.n_heads == 4 and cfg.head_dim == 16
    if em in ("auto", "csr"):
        return "csr" if csr_ok else ("stream" if low_memory else "ref")
    return em


def load_lam4_checkpoint(
    checkpoint: str | Path | Lam4Model,
    *,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype = "float32",
    weights: str = "ema",
    low_memory: bool = False,
    edge_mode: str = "auto",
    bake: bool = True,
) -> tuple[Lam4Model, torch.device, torch.dtype, dict]:
    """Load a trained or slimmed `Lam4Model` ready for inference."""
    dt = resolve_dtype(dtype)
    dev = resolve_device(device, dt)
    remat = "block" if low_memory else "none"

    if isinstance(checkpoint, Lam4Model):
        model = checkpoint
        eff_mode = resolve_edge_mode(edge_mode, dev, dt, model.cfg, low_memory)
        model.trunk.edge_mode = eff_mode
        model.set_remat(remat)
        model.readout.chunk_nodes = 4096 if low_memory else 0
        model = model.to(device=dev, dtype=dt)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if bake:
            model.bake_for_inference()
        return model, dev, dt, {"cfg_hash": model.cfg.hash(), "weights": "in_memory", "edge_mode": eff_mode}

    if weights not in ("ema", "live"):
        raise ValueError(f"weights must be 'ema' or 'live', got {weights!r}")

    path = Path(checkpoint)
    ck = _raw_load_checkpoint(path)
    sd_raw = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    cfg = resolve_lam4_config(ck if isinstance(ck, dict) else {}, sd_raw)
    eff_mode = resolve_edge_mode(edge_mode, dev, dt, cfg, low_memory)

    model = Lam4Model(cfg, remat=remat, edge_mode=eff_mode)
    model.load_state_dict(sd_raw, strict=True)

    used_weights = "baked" if (isinstance(ck, dict) and ck.get("inference_baked")) else "live"
    if isinstance(ck, dict) and weights == "ema" and not ck.get("inference_baked") and isinstance(ck.get("ema"), dict):
        ema_params = ck["ema"].get("params")
        with torch.no_grad():
            if isinstance(ema_params, dict):
                for name, param in model.named_parameters():
                    if name in ema_params:
                        param.copy_(ema_params[name].to(dtype=param.dtype, device=param.device))
                used_weights = "ema"
            elif isinstance(ema_params, (list, tuple)):
                params = list(model.parameters())
                if len(ema_params) != len(params):
                    raise ValueError(f"EMA param count {len(ema_params)} != model param count {len(params)}")
                for param, p_ema in zip(params, ema_params):
                    param.copy_(p_ema.to(dtype=param.dtype, device=param.device))
                used_weights = "ema"

    if model.force_head is not None:
        model.force_head = None
        model.cfg = replace(model.cfg, direct_force_head=False)

    model = model.to(device=dev, dtype=dt)
    model.readout.chunk_nodes = 4096 if low_memory else 0
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if bake:
        model.bake_for_inference()
    meta = {
        "path": str(path),
        "step": ck.get("step") if isinstance(ck, dict) else None,
        "cfg_hash": ck.get("cfg_hash", cfg.hash()) if isinstance(ck, dict) else cfg.hash(),
        "weights": used_weights,
        "edge_mode": eff_mode,
    }
    return model, dev, dt, meta


def crystal_e0_table_f64(lot: str) -> Tensor:
    """Per-element crystal atomic reference E0 [119] float64 on CPU in kcal/mol."""
    ref_key = reference_key_of(lot)
    tab = _CRYSTAL_E0_CACHE.get(ref_key)
    if tab is None:
        tab = load_lot_e0_table_f64(ref_key, num_elements=119, in_kcal=True).detach().cpu().to(torch.float64)
        _CRYSTAL_E0_CACHE[ref_key] = tab
    return tab


def molecule_ref_tables_f64(lot: str) -> dict[str, np.ndarray]:
    """Molecular E0 [119], E_Q [-10..10], E_S [1..11] float64 arrays in kcal/mol."""
    key = "spice" if lot == "spice_wb97md3" else "omol25"
    tab = _MOL_REF_CACHE.get(key)
    if tab is None:
        raw_path = SPICE_REF_JSON if key == "spice" else OMOL_REF_JSON
        path = raw_path if raw_path.is_file() else resolve_e0_path(raw_path.name)
        raw = json.loads(path.read_text(encoding="utf-8"))
        e0 = np.zeros(119, dtype=np.float64)
        e0_map = raw.get("e0_kcal") or raw.get("atomic_e0_kcal") or {}
        for z_str, val in e0_map.items():
            z = int(z_str)
            if 0 <= z < 119:
                e0[z] = float(val)
        e_q = np.zeros(21, dtype=np.float64)
        for q_str, val in raw.get("charge_kcal", {}).items():
            q = int(q_str)
            if -10 <= q <= 10:
                e_q[q + 10] = float(val)
        e_s = np.zeros(11, dtype=np.float64)
        for s_str, val in raw.get("spin_kcal", {}).items():
            s = int(s_str)
            if 1 <= s <= 11:
                e_s[s - 1] = float(val)
        tab = {"e0": e0, "e_q": e_q, "e_s": e_s}
        _MOL_REF_CACHE[key] = tab
    return tab
