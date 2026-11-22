"""vector_swiglu.py: Per-atom equivariant Vector SwiGLU update."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import lam.models.edge_ops as eo


def _mix(x: Tensor, lin: nn.Linear) -> Tensor:
    """Channel mixing of an equivariant feature [..., C, 3] with a bias-free Linear (contiguous input: N tiny matmuls otherwise)."""
    return lin(x.transpose(-1, -2).contiguous()).transpose(-1, -2)


class VectorSwiGLU(nn.Module):
    """Per-atom equivariant counterpart of the scalar SwiGLU :

        h_a = W_a v,  h_b = W_b v                  each [..., H, 3] (bias-free channel mixes)
        q   = <h_a, h_b>                           [..., H] per-channel inner product (O(3) invariant)
        g   = gate([RMSNorm(s); q])                [..., H]
        out = W_o (h_a * SiLU(g))                  [..., K, 3], with W_o zero-initialised
        v  <- v + out

    Submodule names match §2.5 (`w_a`, `w_b`, `gate`, `w_o`), and `RMSNorm(s)` uses parameterless `F.rms_norm(..., eps=1e-6)`
    so only `{w_a, w_b, gate, w_o}` appear in the state_dict under `core.stages_list.{i}.vswiglu.*`.
    """

    def __init__(self, k: int, d_model: int, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.k = int(k)
        self.d_model = int(d_model)
        self.hidden = int(hidden)
        if self.k <= 0 or self.d_model <= 0 or self.hidden <= 0:
            raise ValueError(f"k ({self.k}), d_model ({self.d_model}), and hidden ({self.hidden}) must be > 0")
        self.eps = float(eps)
        self.w_a = nn.Linear(self.k, self.hidden, bias=False)
        self.w_b = nn.Linear(self.k, self.hidden, bias=False)
        self.gate = nn.Linear(self.d_model + self.hidden, self.hidden)
        self.w_o = nn.Linear(self.hidden, self.k, bias=False)
        nn.init.zeros_(self.w_o.weight)

    def forward(self, v: Tensor, s: Tensor) -> Tensor:
        """v [..., K, 3], s [..., D] -> v + out [..., K, 3]."""
        if v.ndim == s.ndim - 1 and s.shape[-1] == 3:
            v, s = s, v
        # Equivariant RMSNorm on v across the K vector channels so h_a, h_b are O(1) instead of O(||v||),
        # and RMSNorm on the O(3)-invariant inner product q = <h_a, h_b> so the stage update is bounded O(1)
        # instead of an unnormalised cubic O(||v||^3) across stages.
        v_rms_inv = torch.rsqrt((v * v).sum(dim=-1, keepdim=True).mean(dim=-2, keepdim=True) + self.eps)
        v_norm = v * v_rms_inv
        vt = v_norm.transpose(-1, -2).contiguous()
        hab = F.linear(vt, torch.cat([self.w_a.weight, self.w_b.weight], dim=0)).transpose(-1, -2)
        h_a, h_b = hab.split(self.hidden, dim=-2)
        q = eo.dot_c(h_a, h_b)
        q_norm = F.rms_norm(q, (q.shape[-1],), eps=self.eps)
        s_norm = F.rms_norm(s, (s.shape[-1],), eps=self.eps)
        g = F.silu(self.gate(torch.cat([s_norm, q_norm], dim=-1)))
        out = self.w_o(eo.scale(h_a, g).transpose(-1, -2).contiguous()).transpose(-1, -2)
        return v + out
