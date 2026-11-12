"""Bounded equivariant vector path and the state norm (numerically earlier Mena `vector_bound.bound_vectors` / `F.rms_norm`, tests at 1e-12).

u = q(v) = v / sqrt(1 + r^2 / c^2),  r^2 = mean_k |v_k|^2  (per atom, shared by the K channels), c = ModelConfig.vec_c = 2.
q is C-infinity, equivariant, q(0) = 0 with Jacobian I and Jacobian norm <= 1, so every consumer of u sees |u_k| <= c sqrt(K). The stream
v itself is additive and unbounded (it grows at most linearly over the stages); only the READERS (transport, invariants, readout) take u.

state_norm: rms normalisation without weight, eps 1e-6, applied to the scalar state at the end of every stage (the earlier models' state
rms grew to 300-1450 without it).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RMS_EPS = 1e-6


def bound_vectors(v: Tensor, c: float) -> Tensor:
    """v [N, K, M] -> u [N, K, M] (M = 3 for l=1 vectors, M = 5 for l=2 quadrupoles: |T_ik|_F^2 = sum_m T_ikm^2)."""
    r2 = v.pow(2).sum(-1).mean(-1)
    return v * (1.0 + r2 / (c * c)).rsqrt().view(-1, 1, 1)


def state_norm(s: Tensor) -> Tensor:
    """rms-normalise the last axis, no weight."""
    return F.rms_norm(s, (s.shape[-1],), eps=RMS_EPS)


def mix(x: Tensor, lin: nn.Linear) -> Tensor:
    """Channel mixing of an equivariant feature [..., C, M] (M = 3 or 5 components) with a bias-free Linear on the channel axis."""
    return lin(x.transpose(-1, -2)).transpose(-1, -2)
