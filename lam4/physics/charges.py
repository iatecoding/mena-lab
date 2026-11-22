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

from lam4.core.radial import fp32_guard, upcast_fp32
from lam4.core.remat import can_chunk_nodes, chunked_node_apply, node_chunk_size


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

    def _local_raw(self, s: Tensor, m: Tensor, *, normed: bool) -> tuple[Tensor, Tensor]:
        s = upcast_fp32(s)
        if normed:
            h = s
        else:
            with fp32_guard(s.device):
                h = F.rms_norm(s, (s.shape[-1],), eps=1e-6)
        x_raw = self.mlp(h).squeeze(-1)
        w_raw = self.nqe(h).squeeze(-1)
        with fp32_guard(s.device):
            x = upcast_fp32(x_raw).to(s.dtype)
            if self.q_max is not None:
                x = self.q_max * torch.tanh(x / self.q_max)
            f = F.softplus(upcast_fp32(w_raw).to(s.dtype))
            mc = m.to(x.dtype)
            return x * mc, f * mc

    def forward(self, s: Tensor, graph: Tensor, charge: Tensor, atom_mask: Tensor, *, normed: bool = False) -> Tensor:
        ps = tuple(self.parameters())
        cn = node_chunk_size(s.device)
        if not self.training and can_chunk_nodes(s.shape[0], ps, min_atoms=cn):
            xm, fm = chunked_node_apply(
                lambda _a0, _a1, s_, m_: self._local_raw(s_, m_, normed=normed),
                (s, atom_mask), params=ps, chunk_atoms=cn,
            )
            return equilibrate(xm, fm, graph, charge)
        xm, fm = self._local_raw(s, atom_mask, normed=normed)
        return equilibrate(xm, fm, graph, charge)


class SpinDensityHead(ChargeHead):
    """Latent spin densities m_i with sum over each group = M_g - 1 (same parameters as ChargeHead)."""

    def forward(self, s: Tensor, graph: Tensor, spin: Tensor, atom_mask: Tensor, *, normed: bool = False) -> Tensor:
        return super().forward(s, graph, spin.to(s.dtype) - 1.0, atom_mask, normed=normed)


def equilibrate(q_raw: Tensor, f: Tensor, graph: Tensor, total: Tensor) -> Tensor:
    """Shift q_raw within each group by f-weighted shares so that the group sums equal `total` [G].
    Masked atoms must come in with q_raw = f = 0 (they then stay 0). A group without real atoms keeps q = 0."""
    with fp32_guard(q_raw.device):
        qf, ff = upcast_fp32(q_raw), upcast_fp32(f)
        acc_dt = torch.float64 if qf.device.type != "mps" else qf.dtype
        qa = qf.to(acc_dt) if qf.dtype != acc_dt else qf
        fa = ff.to(acc_dt) if ff.dtype != acc_dt else ff
        G = total.shape[0]
        if G == 1:
            s_q = qa.sum(0, keepdim=True)
            s_f = fa.sum(0, keepdim=True).clamp_min(1e-6)
            corr = ((total.to(acc_dt) - s_q) / s_f)[0].to(qf.dtype)
            return qf + ff * corr
        s_q = qa.new_zeros(G).index_add(0, graph, qa)
        s_f = fa.new_zeros(G).index_add(0, graph, fa).clamp_min(1e-6)
        corr = ((total.to(acc_dt) - s_q) / s_f).to(qf.dtype)
        return qf + ff * corr.index_select(0, graph)
