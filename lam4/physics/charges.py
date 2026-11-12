"""charges.py: NQE charge head and spin-density head, one charge group per graph row.

    q_i = q~_i + f_i (Q_g - sum_{j in g} q~_j) / sum_{j in g} f_j      f_i = softplus(w(s_i)) > 0,  q~ = q_max tanh(x / q_max)

so sum_{i in g} q_i = Q_g exactly for every group that has at least one real atom. The spin-density head is the same
construction with total M_g - 1 (unpaired electrons, M = 2S+1), so sum_{i in g} m_i = M_g - 1.
A group is a graph (`batch.graph`): multi-island rows are one group, as in the earlier Mena models. Pad atoms get q = 0 exactly
and take no share of any group's charge; pad graphs (Q = 0, M = 1) stay at exactly 0.

Signatures (all tensor-in / tensor-out, no domain gate: charges and spin densities exist for both domains)
    ChargeHead(d_model, q_max=2.0)
        forward(s [N, d], graph [N], charge [G], atom_mask [N], *, normed=False) -> q [N]
    SpinDensityHead(d_model, q_max=2.0)
        forward(s [N, d], graph [N], spin [G] (multiplicity), atom_mask [N], *, normed=False) -> m [N]
    equilibrate(q_raw [N], f [N], graph [N], total [G]) -> q [N]
`normed=True` means s is already rms-normalised (the caller shares one F.rms_norm across heads).

Key map from an earlier-generation state dict (identical sub-names, drop the prefix):
    charge_heads.{k}.{mlp.0,mlp.2,nqe}.{weight,bias}  ->  ChargeHead.{mlp.0,mlp.2,nqe}.{weight,bias}
    spin_heads.{k}.<same>                              ->  SpinDensityHead.<same>
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ChargeHead(nn.Module):
    def __init__(self, d_model: int, q_max: float | None = 2.0) -> None:
        super().__init__()
        self.q_max = None if q_max is None else float(q_max)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1))
        nn.init.normal_(self.mlp[-1].weight, std=0.01)  # small random: q^2-type gradients need q != 0
        nn.init.zeros_(self.mlp[-1].bias)
        self.nqe = nn.Linear(d_model, 1)
        nn.init.zeros_(self.nqe.weight)
        nn.init.constant_(self.nqe.bias, math.log(math.e - 1.0))  # f = 1: uniform sharing at the start

    def forward(self, s: Tensor, graph: Tensor, charge: Tensor, atom_mask: Tensor, *, normed: bool = False) -> Tensor:
        h = s if normed else F.rms_norm(s, (s.shape[-1],), eps=1e-6)
        x = self.mlp(h).squeeze(-1)
        if self.q_max is not None:
            x = self.q_max * torch.tanh(x / self.q_max)
        f = F.softplus(self.nqe(h).squeeze(-1))
        m = atom_mask.to(x.dtype)
        return equilibrate(x * m, f * m, graph, charge)


class SpinDensityHead(ChargeHead):
    """Latent spin densities m_i with sum over each group = M_g - 1 (same parameters as ChargeHead)."""

    def forward(self, s: Tensor, graph: Tensor, spin: Tensor, atom_mask: Tensor, *, normed: bool = False) -> Tensor:
        return super().forward(s, graph, spin.to(s.dtype) - 1.0, atom_mask, normed=normed)


def equilibrate(q_raw: Tensor, f: Tensor, graph: Tensor, total: Tensor) -> Tensor:
    """Shift q_raw within each group by f-weighted shares so that the group sums equal `total` [G].
    Masked atoms must come in with q_raw = f = 0 (they then stay 0). A group without real atoms keeps q = 0."""
    G = total.shape[0]
    if G == 1:
        s_q = q_raw.sum(0, keepdim=True)
        s_f = f.sum(0, keepdim=True).clamp_min(1e-6)
        return q_raw + f * ((total.to(q_raw.dtype) - s_q) / s_f)[0]
    s_q = q_raw.new_zeros(G).index_add(0, graph, q_raw)
    s_f = f.new_zeros(G).index_add(0, graph, f).clamp_min(1e-6)
    corr = (total.to(q_raw.dtype) - s_q) / s_f
    return q_raw + f * corr.index_select(0, graph)
