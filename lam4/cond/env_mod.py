"""Per-atom bounded environment-modulated SwiGLU for LAM-6 Design C unified trunk.

Replaces batch-level binary `DomainExperts` (`crystal` vs `molecule`) with ONE shared high-capacity
SwiGLU whose hidden gate and value projections are smoothly modulated per atom by local state,
factorized LoT conditioning `(xc, paw, u_i)`, and continuous coordination `log(1 + D_i)`:

    dg, dv = env_out(silu(env_in([RMSNorm(x_i), c_lot_i, log1p(D_i)]))).chunk(2, dim=-1)
    m_g = 1.0 + 0.5 * tanh(dg)    in (0.5, 1.5),  exact 1.0 at zero-init
    m_v = 1.0 + 0.5 * tanh(dv)    in (0.5, 1.5),  exact 1.0 at zero-init
    out_i = ffn_out(silu(gate_i * m_g) * (val_i * m_v))

Zero binary `domain` dependence; preserves O(3), permutation, and supercell extensivity.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.core.consts import tracing
from lam4.core.remat import RematSpec, remat_apply
from lam4.kernels.metal_swiglu import env_swiglu_proj_fwd, swiglu_fused_ok

_TLS = threading.local()


@contextmanager
def suppress_env_remat():
    """Disable inner `EnvModulatedSwiGLU` remat when an outer block/chunk remat is active."""
    prev = int(getattr(_TLS, "depth", 0))
    _TLS.depth = prev + 1
    try:
        yield
    finally:
        _TLS.depth = prev


class EnvModulatedSwiGLU(nn.Module):
    """Unified SwiGLU with bounded zero-init rank-r per-atom environment modulation."""

    def __init__(self, d_model: int, hidden: int, cond_dim: int, rank: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.hidden = int(hidden)
        self.cond_dim = int(cond_dim)
        self.rank = int(rank)
        self.norm = nn.RMSNorm(self.d_model, eps=eps)
        self.ffn_in = nn.Linear(self.d_model, 2 * self.hidden, bias=False)
        self.ffn_out = nn.Linear(self.hidden, self.d_model, bias=False)
        self.env_in = nn.Linear(self.d_model + self.cond_dim + 1, self.rank, bias=False)
        self.env_act = nn.SiLU()
        self.env_out = nn.Linear(self.rank, 2 * self.hidden, bias=True)
        nn.init.zeros_(self.env_out.weight)
        nn.init.zeros_(self.env_out.bias)

    def _modulation(self, x_norm: Tensor, c_lot: Tensor | None, log1p_d: Tensor | None) -> tuple[Tensor, Tensor]:
        n = x_norm.shape[0]
        c_t = x_norm.new_zeros(n, self.cond_dim) if c_lot is None else c_lot.to(dtype=x_norm.dtype)
        if log1p_d is None:
            ld = x_norm.new_zeros(n, 1)
        else:
            ld = (log1p_d.unsqueeze(-1) if log1p_d.ndim == 1 else log1p_d).to(dtype=x_norm.dtype)
        ctx = torch.cat((x_norm, c_t, ld), dim=-1)
        dg, dv = self.env_out(self.env_act(self.env_in(ctx))).chunk(2, dim=-1)
        return 1.0 + 0.5 * torch.tanh(dg), 1.0 + 0.5 * torch.tanh(dv)

    def _eval(self, x: Tensor, c_lot: Tensor | None, log1p_d: Tensor | None, *, allow_fused: bool = False,
              env_wt: Tensor | None = None) -> Tensor:
        x_norm = self.norm(x)
        gv = self.ffn_in(x_norm)
        if swiglu_fused_ok(self, gv, allow_fused, self.hidden, self.rank):
            n = x_norm.shape[0]
            c_t = x_norm.new_zeros(n, self.cond_dim) if c_lot is None else c_lot.to(dtype=x_norm.dtype)
            ld = x_norm.new_zeros(n, 1) if log1p_d is None else (log1p_d.unsqueeze(-1) if log1p_d.ndim == 1 else log1p_d).to(dtype=x_norm.dtype)
            h_env = self.env_act(self.env_in(torch.cat((x_norm, c_t, ld), dim=-1)))
            wt = env_wt if env_wt is not None else self.env_out.weight.t().contiguous()
            return self.ffn_out(env_swiglu_proj_fwd(gv, h_env, wt, self.env_out.bias))
        gate, val = gv.chunk(2, dim=-1)
        m_g, m_v = self._modulation(x_norm, c_lot, log1p_d)
        return self.ffn_out(F.silu(gate * m_g) * (val * m_v))

    def forward(
        self,
        x: Tensor,
        c_lot: Tensor | None = None,
        log1p_d: Tensor | None = None,
        domain: str | None = None,
        *,
        allow_fused: bool = False,
        env_wt: Tensor | None = None,
    ) -> Tensor:
        """Apply bounded per-atom modulated SwiGLU; `domain` is accepted only for signature uniformity and ignored."""
        _ = domain
        n_thresh = 1024 if x.device.type == "mps" else (65536 if (x.device.type == "cuda" and allow_fused) else 2048)
        if (
            not self.training
            and torch.is_grad_enabled()
            and int(getattr(_TLS, "depth", 0)) == 0
            and not torch.compiler.is_compiling()
            and not tracing()
            and not torch.is_autocast_enabled(x.device.type)
            and x.shape[0] > n_thresh
            and not any(p.requires_grad for p in self.parameters())
        ):
            has_c, has_d = c_lot is not None, log1p_d is not None
            ins = (x,) + ((c_lot,) if has_c else ()) + ((log1p_d,) if has_d else ())

            def _fn(x_: Tensor, *rest: Tensor) -> tuple[Tensor, ...]:
                c_ = rest[0] if has_c else None
                d_ = rest[int(has_c)] if has_d else None
                return (self._eval(x_, c_, d_, allow_fused=allow_fused, env_wt=env_wt),)

            return remat_apply(RematSpec(_fn, ()), ins)[0]
        return self._eval(x, c_lot, log1p_d, allow_fused=allow_fused, env_wt=env_wt)
