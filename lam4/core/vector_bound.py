"""Bounded equivariant vector path and the state norm (numerically earlier Mena `vector_bound.bound_vectors` / `F.rms_norm`, tests at 1e-12).

u = q(v) = v / sqrt(1 + r^2 / c^2),  r^2 = mean_k |v_k|^2  (per atom, shared by the K channels), c = ModelConfig.vec_c = 2.
q is C-infinity, equivariant, q(0) = 0 with Jacobian I and Jacobian norm <= 1, so every consumer of u sees |u_k| <= c sqrt(K). The stream
v itself is additive and unbounded (it grows at most linearly over the stages); only the READERS (transport, invariants, readout) take u.

state_norm: rms normalisation without weight, eps 1e-6, applied to the scalar state at the end of every stage (the earlier models' state
rms grew to 300-1450 without it).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RMS_EPS = 1e-6


def bound_vectors(v: Tensor, c: float) -> Tensor:
    """v [N, K, M] -> u [N, K, M] (M = 3 for l=1 vectors, M = 5 for l=2 quadrupoles: |T_ik|_F^2 = sum_m T_ikm^2)."""
    r2 = v.pow(2).sum((-2, -1)) / v.shape[-2]  # = sum over xyz, mean over K; one reduction (5x faster on MPS than sum(-1).mean(-1))
    return v * (1.0 + r2 / (c * c)).rsqrt().view(-1, 1, 1)


def state_norm(s: Tensor) -> Tensor:
    """rms-normalise the last axis, no weight."""
    return F.rms_norm(s, (s.shape[-1],), eps=RMS_EPS)


def dot_last(a: Tensor, b: Tensor) -> Tensor:
    """(a * b).sum(-1) over the small component axis (M = 3, 5, 7, 9) of [..., C, M] features.

    On MPS a reduction over a size-3/5 last axis is a slow kernel (1.2-1.8 ms for [4096, 128, 3]); unrolled adds take 0.13 ms and a
    matmul with ones 0.33-0.5 ms. The CPU and CUDA keep the plain reduction (fastest there; Inductor fuses it when compiling)."""
    if a.device.type != "mps" or torch.compiler.is_compiling():
        return (a * b).sum(-1)
    m = a.shape[-1]
    if m <= 5:
        aa, bb = a.unbind(-1), b.unbind(-1)
        out = aa[0] * bb[0]
        for i in range(1, m):
            out = out + aa[i] * bb[i]
        return out
    return (a * b) @ a.new_ones(m)


class _OuterFlat(torch.autograd.Function):
    """p [..., A] (x) q [..., B] -> [..., A B] with a backward made of unrolled products (no reduction over a size-3/9 axis, slow on MPS).

    The backward uses differentiable ops only, so create_graph / double backward still work."""

    @staticmethod
    def forward(ctx, p: Tensor, q: Tensor) -> Tensor:
        ctx.save_for_backward(p, q)
        return (p.unsqueeze(-1) * q.unsqueeze(-2)).flatten(-2)

    @staticmethod
    def backward(ctx, g: Tensor):
        p, q = ctx.saved_tensors
        na, nb = p.shape[-1], q.shape[-1]
        g3 = g.unflatten(-1, (na, nb))
        gp = gq = None
        if ctx.needs_input_grad[0]:
            gp = g3[..., 0] * q[..., :1]
            for j in range(1, nb):
                gp = gp + g3[..., j] * q[..., j:j + 1]
        if ctx.needs_input_grad[1]:
            gq = g3[..., 0, :] * p[..., :1]
            for i in range(1, na):
                gq = gq + g3[..., i, :] * p[..., i:i + 1]
        return gp, gq


def outer_flat(p: Tensor, q: Tensor) -> Tensor:
    """Per-channel outer product p [..., A] (x) q [..., B] flattened to [..., A B] (row-major: index a B + b)."""
    if p.device.type == "mps" and not torch.compiler.is_compiling() and torch.is_grad_enabled() and (p.requires_grad or q.requires_grad):
        return _OuterFlat.apply(p, q)
    return (p.unsqueeze(-1) * q.unsqueeze(-2)).flatten(-2)


def chan_linear(x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
    """F.linear on the channel axis of an equivariant feature [..., C, M] -> [..., O, M].

    The input is made contiguous first: a linear on the strided transposed view runs as N tiny batched matmuls, measured 20x slower
    on MPS and 80x slower on the CPU (N = 4096, C = 32, O = 128) than one [N M, C] x [C, O] matmul. Same values to float rounding."""
    return F.linear(x.transpose(-1, -2).contiguous(), w, b).transpose(-1, -2)


def mix(x: Tensor, lin: nn.Linear) -> Tensor:
    """Channel mixing of an equivariant feature [..., C, M] (M = 3 or 5 components) with a bias-free Linear on the channel axis."""
    return chan_linear(x, lin.weight, lin.bias)
