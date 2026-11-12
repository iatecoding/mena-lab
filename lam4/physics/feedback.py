"""feedback.py: how the latent charges and spin densities of stage k reach the network.

FieldFeedback (earlier Mena trunk): potential phi_i and field E_i of this stage's charges are injected,
    s += W_phi rbf(phi_i / phi_scale)                    scalar channel (phi soft-clipped to +-3 phi_scale)
    v += w_E(RMSNorm(s)) (x) clip(E_i)                   vector channel, clip(x) = x / sqrt(1 + |x|^2 / field_scale^2)
both with zero-initialised output weights (the module starts as the identity).
DensityFeedback (omol4): s += MLP([q_i, m_i]), zero-initialised output.

Signatures (tensor-in / tensor-out; pad atoms must come in with phi = field = 0 and q = m = 0, which ExactCoulomb and
the charge heads guarantee, then they receive an exactly zero update from phi/E and a bias-only update from the MLP,
so the caller keeps its usual `state * atom_mask`)
    FieldFeedback(d_model, vector_rank, phi_basis=8, phi_scale=1.0, field_scale=1.0)
        forward(s [N, d], v [N, K, 3], phi [N], field [N, 3]) -> (s, v)
    DensityFeedback(d_model, hidden=64, spin_density=True)
        forward(s [N, d], q [N], m [N] | None) -> s
    soft_clip_vec(x [..., 3], scale) -> [..., 3]

Key map from an earlier-generation state dict (identical sub-names, drop the prefix):
    fb_field.{k}.{phi_proj.weight, field_norm.weight, field_gate.weight, field_gate.bias}  ->  FieldFeedback.<same>
    fb_density.{k}.mlp.{0,2}.{weight,bias}                                                  ->  DensityFeedback.mlp.{0,2}.<same>
(the `centers` buffer is non-persistent and rebuilt)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def soft_clip_vec(x: Tensor, scale: float) -> Tensor:
    return x * torch.rsqrt(1.0 + x.pow(2).sum(-1, keepdim=True) / (scale * scale))


class FieldFeedback(nn.Module):
    def __init__(self, d_model: int, vector_rank: int, phi_basis: int = 8, phi_scale: float = 1.0,
                 field_scale: float = 1.0) -> None:
        super().__init__()
        self.phi_scale, self.field_scale = phi_scale, field_scale
        self.register_buffer("centers", torch.linspace(-3.0, 3.0, phi_basis), persistent=False)
        self.width = 6.0 / (phi_basis - 1)
        self.phi_proj = nn.Linear(phi_basis, d_model, bias=False)
        nn.init.zeros_(self.phi_proj.weight)
        self.field_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.field_gate = nn.Linear(d_model, vector_rank)
        nn.init.zeros_(self.field_gate.weight)
        nn.init.zeros_(self.field_gate.bias)

    def forward(self, s: Tensor, v: Tensor, phi: Tensor, field: Tensor) -> tuple[Tensor, Tensor]:
        x = 3.0 * torch.tanh(phi / (3.0 * self.phi_scale))
        rbf = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.width).pow(2))
        s = s + self.phi_proj(rbf)
        e = soft_clip_vec(field, self.field_scale)
        v = v + self.field_gate(self.field_norm(s)).unsqueeze(-1) * e.unsqueeze(1)
        return s, v


class DensityFeedback(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64, spin_density: bool = True) -> None:
        super().__init__()
        n_in = 2 if spin_density else 1
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, d_model))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s: Tensor, q: Tensor, m: Tensor | None = None) -> Tensor:
        x = q.unsqueeze(-1) if m is None else torch.stack([q, m], -1)
        return s + self.mlp(x)
