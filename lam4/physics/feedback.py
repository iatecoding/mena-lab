"""feedback.py: how the latent charges, fields, EFG tensors, and spin densities reach the network.

FieldFeedback: potential phi_i (l=0), field E_i (l=1), and optional traceless EFG G_i (l=2, when ten_rank > 0):
    s += W_phi rbf(phi_i / phi_scale)
    v += w_E(RMSNorm(s)) (x) clip(E_i)
    ten += w_G(RMSNorm(s)) (x) clip(G_i)   (optional LAM-6 l=2 feedback; omitted when ten_rank=0 for lam5 parity)
DensityFeedback: s += MLP([q_i, m_i, (u_rad_i)]), zero-initialised output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lam4.core.remat import can_chunk_nodes, chunked_node_apply, node_chunk_size


def soft_clip_vec(x: Tensor, scale: float) -> Tensor:
    return x * torch.rsqrt(1.0 + x.pow(2).sum(-1, keepdim=True) / (scale * scale))


class FieldFeedback(nn.Module):
    def __init__(self, d_model: int, vector_rank: int, phi_basis: int = 8, phi_scale: float = 1.0,
                 field_scale: float = 1.0, *, ten_rank: int = 0, efg_scale: float = 1.0) -> None:
        super().__init__()
        self.phi_scale, self.field_scale, self.efg_scale = phi_scale, field_scale, float(efg_scale)
        self.register_buffer("centers", torch.linspace(-3.0, 3.0, phi_basis), persistent=False)
        self.width = 6.0 / (phi_basis - 1)
        self.phi_proj = nn.Linear(phi_basis, d_model, bias=False)
        nn.init.zeros_(self.phi_proj.weight)
        self.field_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.field_gate = nn.Linear(d_model, vector_rank)
        nn.init.zeros_(self.field_gate.weight)
        nn.init.zeros_(self.field_gate.bias)
        self.efg_gate = nn.Linear(d_model, int(ten_rank)) if ten_rank > 0 else None
        if self.efg_gate is not None:
            nn.init.zeros_(self.efg_gate.weight)
            nn.init.zeros_(self.efg_gate.bias)

    def _forward_plain(self, s: Tensor, v: Tensor, phi: Tensor, field: Tensor,
                       ten: Tensor | None = None, efg: Tensor | None = None):
        x = 3.0 * torch.tanh(phi / (3.0 * self.phi_scale))
        rbf = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.width).pow(2))
        s = s + self.phi_proj(rbf)
        sn = self.field_norm(s)
        e = soft_clip_vec(field, self.field_scale)
        v = v + self.field_gate(sn).unsqueeze(-1) * e.unsqueeze(1)
        if ten is None:
            return s, v
        if self.efg_gate is not None and efg is not None:
            g = soft_clip_vec(efg, self.efg_scale)
            ten = ten + self.efg_gate(sn).unsqueeze(-1) * g.unsqueeze(1)
        return s, v, ten

    def forward(self, s: Tensor, v: Tensor, phi: Tensor, field: Tensor, *,
                ten: Tensor | None = None, efg: Tensor | None = None):
        ps = tuple(self.parameters())
        cn = node_chunk_size(s.device)
        if not self.training and can_chunk_nodes(s.shape[0], ps, min_atoms=cn):
            if ten is None:
                return chunked_node_apply(
                    lambda _a0, _a1, s_, v_, p_, f_: self._forward_plain(s_, v_, p_, f_),
                    (s, v, phi, field), params=ps, chunk_atoms=cn,
                )
            if efg is not None:
                return chunked_node_apply(
                    lambda _a0, _a1, s_, v_, p_, f_, t_, g_: self._forward_plain(s_, v_, p_, f_, t_, g_),
                    (s, v, phi, field, ten, efg), params=ps, chunk_atoms=cn,
                )
        return self._forward_plain(s, v, phi, field, ten, efg)


class DensityFeedback(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64, spin_density: bool = True,
                 radical_density: bool = False) -> None:
        super().__init__()
        self.radical_density = bool(radical_density)
        n_in = (2 if spin_density else 1) + (1 if radical_density else 0)
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, d_model))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s: Tensor, q: Tensor, m: Tensor | None = None, u_rad: Tensor | None = None) -> Tensor:
        ps = tuple(self.parameters())
        cols = [q] + ([] if m is None else [m]) + ([u_rad] if (self.radical_density and u_rad is not None) else [])
        x = q.unsqueeze(-1) if len(cols) == 1 else torch.stack(cols, dim=-1)
        cn = node_chunk_size(s.device)
        if not self.training and can_chunk_nodes(s.shape[0], ps, min_atoms=cn):
            return chunked_node_apply(lambda _a0, _a1, s_, x_: s_ + self.mlp(x_), (s, x), params=ps, chunk_atoms=cn)[0]
        return s + self.mlp(x)
