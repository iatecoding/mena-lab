"""force_head.py: the non-conservative direct-force head of the direct-force training phase.

    F_i = sum_k w_ik u_ik,   u = bound_vectors(vec, vec_c) [N, K, 3],   w = Linear(SiLU(Linear(RMSNorm(s_i)))) [N, K]   (kcal/mol/A)

w is invariant (it reads the scalar state) and u is the l=1 stream, so F is rotation-equivariant and translation-invariant by construction. It reads the
FINAL trunk state (after feedback, group context and state_norm), i.e. the same features as the energy readout, so a direct step trains the trunk
the energy readout uses. The last layer is zero-init: F = 0 at init, and the loss reaches the head's first layer after one step. Pad atoms give exactly 0.
Cost: one node MLP, no edge work, no double backward. The final model is conservative (by design): this head is a training device only and is
never used by the calculator.

    DirectForceHead(d_model, k_vec, vec_c)
        forward(s [N, d], vec [N, K, 3], atom_mask [N]) -> F [N, 3]
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from lam4.core.vector_bound import bound_vectors


class DirectForceHead(nn.Module):
    def __init__(self, d_model: int, k_vec: int, vec_c: float) -> None:
        super().__init__()
        self.vec_c = float(vec_c)
        self.norm = nn.RMSNorm(d_model, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, k_vec))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s: Tensor, vec: Tensor, atom_mask: Tensor) -> Tensor:
        w = self.mlp(self.norm(s))
        f = (w.unsqueeze(-1) * bound_vectors(vec, self.vec_c)).sum(1)
        return f * atom_mask.to(f.dtype).unsqueeze(-1)
