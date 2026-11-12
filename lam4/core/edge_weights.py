"""Per-stage 4-head q / k projections of the bounded-QK edge weight (edge_terms.normalise_qk turns them into unit vectors per head).

logit^h_ij = <qn_i^h, kn_j^h> + 2 tanh(table bias^h / 2); the op needs q, k [N, n_heads * head_dim] from the stage's normed (and FiLM'd) state.

ZERO-INIT through a per-head gain, not through zero projections: logit^h = tanh(raw_gain_h) <qn, kn> with raw_gain = 0, so the logit is exactly 0 at
init (the op equals the unweighted earlier Mena sum) while q_proj and k_proj keep their normal random init. (An earlier port zeroed q_proj: the
normalisation has a Jacobian ~1e3 at q = 0, so the first optimiser step made the logit jump to O(1). With the gain, d logit / d raw_gain = <qn, kn>
is O(1) from step 0 and q, k grow their gradients smoothly as the gain opens.) |tanh| < 1 keeps the QK logit bounded in (-1, 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EdgeHeads(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, gain_max: float = 1.0) -> None:
        super().__init__()
        self.gain_max = float(gain_max)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.raw_gain = nn.Parameter(torch.zeros(n_heads))

    def gain(self) -> Tensor:
        return self.gain_max * torch.tanh(self.raw_gain / self.gain_max)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return self.q_proj(h), self.k_proj(h)
