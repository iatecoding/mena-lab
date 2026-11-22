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

from lam4.cond.env_mod import EnvModulatedSwiGLU
from lam4.config import ModelConfig
from lam4.kernels.metal_swiglu import swiglu_act_fwd, swiglu_fused_ok


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.ffn_in = nn.Linear(d_model, 2 * hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor, *, allow_fused: bool = False) -> Tensor:
        x_norm = self.norm(x)
        gv = self.ffn_in(x_norm)
        if swiglu_fused_ok(self, gv, allow_fused, self.hidden):
            return self.ffn_out(swiglu_act_fwd(gv))
        gate, val = gv.chunk(2, dim=-1)
        return self.ffn_out(F.silu(gate) * val)


class DomainExperts(nn.ModuleDict):
    def __init__(self, d_model: int, ratio: float, domains: tuple[str, ...]) -> None:
        super().__init__({d: SwiGLU(d_model, max(16, int(round(d_model * ratio)))) for d in domains})

    def forward(self, x: Tensor, domain: str, *, allow_fused: bool = False) -> Tensor:
        if domain not in self:
            raise ValueError(f"no expert for domain {domain!r}; experts: {list(self.keys())}")
        return self[domain](x, allow_fused=allow_fused)


def build_ffn(cfg: ModelConfig) -> nn.Module:
    """Build either `DomainExperts` (Design A / lam4 / lam5) or unified `EnvModulatedSwiGLU` (Design C)."""
    if cfg.ffn_mode == "domain_experts":
        return DomainExperts(cfg.d_scalar, cfg.swiglu_ratio, cfg.domains)
    ratio = 2.0 if cfg.ffn_mode == "env_mod_768" else 3.0
    hidden = max(16, int(round(cfg.d_scalar * ratio)))
    return EnvModulatedSwiGLU(cfg.d_scalar, hidden, cfg.cond_dim, cfg.env_mod_rank)
