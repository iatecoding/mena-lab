"""edge_ops.py: per-edge equivariant products with fast backward AND double backward on MPS.

Why: on Apple MPS a reduction over a small trailing dimension is pathologically slow: sum over the last axis of an
[E, 32, 3] tensor takes 36 ms at E = 400k (8k MPtrj atoms), while the same arithmetic as three strided multiply-adds
takes ~2 ms. Autograd emits exactly that reduction for every broadcast product of
an [E, K] channel weight with an [E, 3] (or [E, 5]) geometric factor, in the backward and again in the double backward
of force training. These Functions compute the same products, and their derivatives are expressed through the same
five Functions, so every order of differentiation stays on the fast kernels. The maths is unchanged (only the float
summation order of the 3- or 5-term sums differs from torch's reduction).

Shapes: x, y [E, K, C]; w, f [E, K]; u [E, C]; C = 3 (vectors) or 5 (l = 2 components).
  outer(w, u)[e, k, c] = w[e, k] u[e, c]
  contract_c(x, u)[e, k] = sum_c x[e, k, c] u[e, c]
  contract_k(x, w)[e, c] = sum_k x[e, k, c] w[e, k]
  scale(x, f)[e, k, c] = x[e, k, c] f[e, k]
  dot_c(x, y)[e, k] = sum_c x[e, k, c] y[e, k, c]
"""
from __future__ import annotations

import torch
from torch import Tensor


def _contract_c(x: Tensor, u: Tensor) -> Tensor:
    r = x[..., 0] * u[..., 0:1]
    for c in range(1, x.shape[-1]):
        r = torch.addcmul(r, x[..., c], u[..., c:c + 1])
    return r


def _dot_c(x: Tensor, y: Tensor) -> Tensor:
    r = x[..., 0] * y[..., 0]
    for c in range(1, x.shape[-1]):
        r = torch.addcmul(r, x[..., c], y[..., c])
    return r


def _contract_k(x: Tensor, w: Tensor) -> Tensor:
    return (x * w.unsqueeze(-1)).sum(-2)


class _Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u):
        ctx.save_for_backward(w, u)
        return w.unsqueeze(-1) * u.unsqueeze(-2)

    @staticmethod
    def backward(ctx, g):
        w, u = ctx.saved_tensors
        gw = contract_c(g, u) if ctx.needs_input_grad[0] else None
        gu = contract_k(g, w) if ctx.needs_input_grad[1] else None
        return gw, gu


class _ContractC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u):
        ctx.save_for_backward(x, u)
        return _contract_c(x, u)

    @staticmethod
    def backward(ctx, g):
        x, u = ctx.saved_tensors
        gx = outer(g, u) if ctx.needs_input_grad[0] else None
        gu = contract_k(x, g) if ctx.needs_input_grad[1] else None
        return gx, gu


class _ContractK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return _contract_k(x, w)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        gx = outer(w, g) if ctx.needs_input_grad[0] else None
        gw = contract_c(x, g) if ctx.needs_input_grad[1] else None
        return gx, gw


class _Scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.save_for_backward(x, f)
        return x * f.unsqueeze(-1)

    @staticmethod
    def backward(ctx, g):
        x, f = ctx.saved_tensors
        gx = scale(g, f) if ctx.needs_input_grad[0] else None
        gf = dot_c(g, x) if ctx.needs_input_grad[1] else None
        return gx, gf


class _DotC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return _dot_c(x, y)

    @staticmethod
    def backward(ctx, g):
        x, y = ctx.saved_tensors
        gx = scale(y, g) if ctx.needs_input_grad[0] else None
        gy = scale(x, g) if ctx.needs_input_grad[1] else None
        return gx, gy


def outer(w: Tensor, u: Tensor) -> Tensor:
    return _Outer.apply(w, u)


def contract_c(x: Tensor, u: Tensor) -> Tensor:
    return _ContractC.apply(x, u)


def contract_k(x: Tensor, w: Tensor) -> Tensor:
    return _ContractK.apply(x, w)


def scale(x: Tensor, f: Tensor) -> Tensor:
    return _Scale.apply(x, f)


def dot_c(x: Tensor, y: Tensor) -> Tensor:
    return _DotC.apply(x, y)
