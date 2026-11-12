from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def taper_envelope(r: Tensor, r_on: float, r_cut: float, c3: bool = False) -> Tensor:
    """Flat-top taper functional: 1 for r <= r_on, quintic (C^2) or septic (C^3) smoothstep down to 0 at r_cut."""
    t = ((r - r_on) / (r_cut - r_on)).clamp(min=0.0, max=1.0)
    if c3:
        t2 = t * t
        return 1.0 - t2 * t2 * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)))
    return 1.0 - t * t * t * (10.0 + t * (6.0 * t - 15.0))


class QuinticCutoff(nn.Module):
    """Smooth C^2 quintic polynomial cutoff ensuring continuous second derivatives:
    f(r) = 1 - 10(r/rc)^3 + 15(r/rc)^4 - 6(r/rc)^5 for r <= rc.
    Guarantees both potential energy and force derivatives vanish smoothly at r_cut.
    """

    def __init__(self, r_cut: float = 5.0) -> None:
        super().__init__()
        self.r_cut = float(r_cut)

    def forward(self, r: Tensor) -> Tensor:
        u = (r / self.r_cut).clamp(min=0.0, max=1.0)
        # Horner polynomial form: 1 - u^3 * (10 + u * (6u - 15))
        f = 1.0 - u * u * u * (10.0 + u * (6.0 * u - 15.0))
        return torch.where(r <= self.r_cut, f, torch.zeros_like(r))


class TaperCutoff(nn.Module):
    """Flat-top taper: 1 for r <= r_on, quintic (C^2) or septic (C^3) smoothstep down to 0 at r_cut."""

    def __init__(self, r_on: float, r_cut: float, c3: bool = False) -> None:
        super().__init__()
        self.r_on, self.r_cut, self.c3 = float(r_on), float(r_cut), bool(c3)

    def forward(self, r: Tensor) -> Tensor:
        return taper_envelope(r, self.r_on, self.r_cut, self.c3)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff function ensuring interactions vanish smoothly at r_cut."""

    def __init__(self, r_cut: float = 5.0) -> None:
        super().__init__()
        self.r_cut = float(r_cut)

    def forward(self, r: Tensor) -> Tensor:
        cut = 0.5 * (torch.cos(r * (math.pi / self.r_cut)) + 1.0)
        return torch.where(r <= self.r_cut, cut, torch.zeros_like(r))


class GaussianRBF(nn.Module):
    """Gaussian Radial Basis Function expansion of interatomic distances with C^2 envelope."""

    def __init__(self, num_rbf: int = 32, r_cut: float = 5.0, envelope: bool = True, r_min: float = 0.0) -> None:
        """r_min > 0: centres span [r_min, r_cut] (the data range) instead of [0, r_cut]; width = span / num_rbf."""
        super().__init__()
        self.r_cut = r_cut
        self.envelope = envelope
        self.cutoff = QuinticCutoff(r_cut)
        centers = torch.linspace(r_min, r_cut, num_rbf)
        self.register_buffer("centers", centers)
        gamma = 0.5 / (((r_cut - r_min) / num_rbf) ** 2)
        self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, r: Tensor) -> Tensor:
        diff = r.unsqueeze(-1) - self.centers
        rbf = torch.exp(-self.gamma * (diff ** 2))
        return rbf * self.cutoff(r).unsqueeze(-1) if self.envelope else rbf


class ChemicalEdgeWeights(nn.Module):
    """Degree-preserving state-conditioned edge modulation."""

    def __init__(self, width: int, smooth_degree: bool = False) -> None:
        super().__init__()
        self.qk = nn.Linear(width, 32, bias=False)
        self.scale = nn.Parameter(torch.zeros(()))
        self.smooth_degree = bool(smooth_degree)

    def forward(self, state: Tensor, support: Tensor, smooth_degree: bool | None = None) -> Tensor:
        norm_state = F.rms_norm(state, (state.shape[-1],), eps=1e-6)
        key, query = self.qk(norm_state).chunk(2, -1)
        pair = torch.bmm(query, key.transpose(1, 2)) * 0.25
        modulation = torch.tanh(self.scale * pair)
        raw_deg = support.sum(dim=-1, keepdim=True)
        sd = self.smooth_degree if smooth_degree is None else bool(smooth_degree)
        degree = (raw_deg + 1.0) if sd else raw_deg.clamp_min(1.0)
        centered = modulation - (support * modulation).sum(dim=-1, keepdim=True) / degree
        return support * (1.0 + 0.49 * centered)


class MinimalSwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network with RMSNorm and custom hidden ratio (default 2/3).
    Supports optional additive `lot_bias` [..., 2 * hidden_dim] before gate/value split."""

    def __init__(self, d_model: int, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.ffn_in = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.ffn_out = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor, lot_bias: Tensor | None = None) -> Tensor:
        h = self.norm(x)
        uv = self.ffn_in(h)
        if lot_bias is not None:
            uv = uv + lot_bias
        gate, val = uv.chunk(2, dim=-1)
        return self.ffn_out(F.silu(gate) * val)

    def forward_pair(self, x: Tensor, lot_bias: Tensor) -> tuple[Tensor, Tensor]:
        """(forward(x), forward(x, lot_bias) - forward(x)) sharing the norm and ffn_in: the token-free output (same ops,
        so bit-identical to forward(x)) and the token's correction (lot_energy_stages, MenaACE)."""
        uv = self.ffn_in(self.norm(x))
        gate, val = uv.chunk(2, dim=-1)
        a = F.silu(gate) * val
        free = self.ffn_out(a)
        gate_l, val_l = (uv + lot_bias).chunk(2, dim=-1)
        return free, self.ffn_out(F.silu(gate_l) * val_l - a)  # ffn_out is bias-free: linear in its input
