"""calibrate.py: data-driven calibration of Stage.inv_scale from a few batches at initialisation (lam5 Contraction v2).

`calibrate_inv_scale(model, batches, max_batches=4, eps=1e-12)` runs a `torch.no_grad()` forward pass over up to `max_batches`
batches, accumulates the per-feature RMS of `all_inv` over real atoms (`b.atom_mask`) at every `Stage` with `inv_norm=True`,
and writes `rms.clamp_min(eps)` into `stage.inv_scale` in-place.
"""
from __future__ import annotations

import itertools
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from lam4.core.stage import Stage


@torch.no_grad()
def calibrate_inv_scale(model: nn.Module, batches: Iterable, max_batches: int = 4, eps: float = 1e-12) -> list[Tensor]:
    """Calibrate `stage.inv_scale` for all `inv_norm=True` stages in `model` over `batches[:max_batches]` (real atoms only)."""
    stages = [m for m in model.modules() if isinstance(m, Stage) and m.inv_norm]
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
        for b in itertools.islice(batches, int(max_batches)):
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
