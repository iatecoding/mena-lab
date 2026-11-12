"""Node SwiGLU with ONE EXPERT PER DOMAIN, hard-switched per batch (domains are separate batches, so the choice is free).

SwiGLU (earlier Mena `MinimalSwiGLU`): x -> ffn_out(silu(gate) * val), [gate, val] = ffn_in(RMSNorm(x)), hidden = max(16, round(d_model * ratio)),
ratio = ModelConfig.swiglu_ratio (2.0; earlier Mena used 1.0), both Linear bias-free. The stage adds it residually: s <- s + SwiGLU(s).
DomainExperts(d_model, ratio, domains): an nn.ModuleDict with one SwiGLU per domain ("crystal", "molecule"); forward(x, domain) runs ONLY the
chosen expert (the other parameters exist but see no compute and get no gradient in that step; with DDP set find_unused_parameters or
all-reduce zeros). The domain is a Python string per batch (GraphBatch.domain), so there is no router and no host sync. The regime field
later decides the string; this module only maps it to an expert.
State keys: <prefix>.{crystal,molecule}.{norm.weight, ffn_in.weight, ffn_out.weight} (earlier Mena swiglu.* maps onto one of them).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.ffn_in = nn.Linear(d_model, 2 * hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, val = self.ffn_in(self.norm(x)).chunk(2, dim=-1)
        return self.ffn_out(F.silu(gate) * val)


class DomainExperts(nn.ModuleDict):
    def __init__(self, d_model: int, ratio: float, domains: tuple[str, ...]) -> None:
        super().__init__({d: SwiGLU(d_model, max(16, int(round(d_model * ratio)))) for d in domains})

    def forward(self, x: Tensor, domain: str) -> Tensor:
        if domain not in self:
            raise ValueError(f"no expert for domain {domain!r}; experts: {list(self.keys())}")
        return self[domain](x)
