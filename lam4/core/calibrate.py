"""calibrate.py: data-driven calibration of Stage.inv_scale from a few batches at initialisation (lam5 Contraction v2).

`calibrate_inv_scale(model, batches, max_batches=4, eps=1e-12)` runs a `torch.no_grad()` forward pass over up to `max_batches`
batches, accumulates the per-feature RMS of `all_inv` over real atoms (`b.atom_mask`) at every `Stage` with `inv_norm=True`,
and writes `rms.clamp_min(eps)` into `stage.inv_scale` in-place.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch import Tensor

from lam4.core.stage import Stage


@torch.no_grad()
def calibrate_inv_scale(model: nn.Module, batches: Iterable, max_batches: int = 4, eps: float = 1e-12) -> list[Tensor]:
    """Calibrate `stage.inv_scale` (or `CoordInvScale` when `coord_inv_scale=True`) over `batches[:max_batches]`."""
    batch_list = list(itertools.islice(batches, int(max_batches)))
    if any(isinstance(m, Stage) and m.inv_norm and getattr(m, "coord_inv_scale", False) for m in model.modules()):
        from lam4.core.inv_scale import calibrate_coord_inv_scale
        calibrate_coord_inv_scale(model, batch_list, max_batches=max_batches, eps=eps)
    stages = [m for m in model.modules() if isinstance(m, Stage) and m.inv_norm and not getattr(m, "coord_inv_scale", False)]
    if not stages:
        return []
    sum_sq = [torch.zeros(st.inv_scale.shape, dtype=torch.float64, device=st.inv_scale.device) for st in stages]
    counts = [0 for _ in stages]
    cur_mask: list[Tensor | None] = [None]

    def make_hook(idx: int):
        def hook(all_inv: Tensor) -> None:
            m = cur_mask[0]
            x = all_inv if m is None else all_inv[m]
            if x.numel() > 0:
                sum_sq[idx] += x.detach().to(torch.float64).pow(2).sum(dim=0)
                counts[idx] += int(x.shape[0])
        return hook

    was_training = model.training
    model.eval()
    try:
        for i, st in enumerate(stages):
            st._inv_hook = make_hook(i)
        for b in batch_list:
            cur_mask[0] = getattr(b, "atom_mask", None)
            try:
                model(b, compute_forces=False)
            except TypeError:
                model(b, b.pos, b.domain)
    finally:
        for st in stages:
            st._inv_hook = None
        if was_training:
            model.train()

    out: list[Tensor] = []
    for st, sq, n in zip(stages, sum_sq, counts):
        if n > 0:
            rms = (sq / n).sqrt().clamp_min(eps).to(dtype=st.inv_scale.dtype, device=st.inv_scale.device)
            st.inv_scale.copy_(rms)
        out.append(st.inv_scale.detach().clone())
    return out


def _coerce_n1_table(src: Any) -> Any:
    if not isinstance(src, (str, Path, list, tuple)):
        return src
    paths = [Path(src)] if isinstance(src, (str, Path)) else [Path(p) for p in src]
    table: dict[tuple[str, str, int], Tensor] = {}
    for p in paths:
        if not p.is_file():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or "delta_kcal" not in ck:
            continue
        val = ck["delta_kcal"].to(torch.float32) * (ck["valid"].to(torch.float32) if "valid" in ck else 1.0)
        xc_s, paw_s = str(ck.get("xc", "pbe")), str(ck.get("paw_basis", "vasp_pbe52"))
        table[(xc_s, paw_s, 0)] = val
        table[(xc_s, paw_s, 1)] = val
    return table or None


def find_n1_delta_paths(*roots: Any) -> list[Path]:
    """Return unique `*n1*delta*.pt` paths found across `roots` (and `n1`/`omol` subdirs)."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root:
            continue
        base = Path(root)
        if base.is_file():
            rp = base.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(base)
            continue
        if not base.is_dir():
            continue
        for sub in (base, base / "n1", base / "omol", base / "omol_np"):
            if not sub.is_dir():
                continue
            for p in sorted(sub.glob("*n1*delta*.pt")):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    found.append(p)
    return found


@torch.no_grad()
def calibrate_lam6_step0(model: nn.Module, batches: Iterable, *, max_batches: int = 16, n1_delta: Any | None = None) -> None:
    """Load optional `N=1` spin-matched deltas and run step-0 Ridge LSQ calibration for `iso_a`/`iso_b`."""
    from lam4.heads.iso_calib import calibrate_iso_reference
    from lam4.heads.readout import Readout

    ro = next((m for m in model.modules() if isinstance(m, Readout)), None)
    if ro is None:
        return
    if n1_delta is not None and ro.iso_atom_delta is not None:
        coerced = _coerce_n1_table(n1_delta)
        if coerced is not None:
            ro.load_iso_atom_delta(coerced)
    if getattr(getattr(model, "cfg", None), "iso_atom_n1", False) and ro.iso_a is not None and ro.iso_b is not None:
        batch_list = list(itertools.islice(batches, int(max_batches)))
        if batch_list:
            calibrate_iso_reference(model, batch_list, max_batches=max_batches)
