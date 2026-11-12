"""Generic rematerialisation with double backward: `remat_apply(spec, inputs)` runs `spec.fn(*inputs)` WITHOUT a graph and keeps only its inputs.

Why not torch.utils.checkpoint: the stock non-reentrant checkpoint also supports create_graph, but the double-backward graph then holds every
recomputed activation of the unit until the second backward ends (the full activation memory comes back in the force-training step). This is the
same two-level pattern as `edge_stream`: ONE autograd node per unit, and the vjp of that node is itself a node that recomputes again in the
second backward, so a force-training step never holds more than one unit's activations. Third order raises under `.backward()` (`once_differentiable`); NOTE `torch.autograd.grad(..., inputs)` prunes the error node and would silently drop
the unit's third-order terms, so never differentiate a force-training loss a third time (same for edge_stream).

    spec = RematSpec(fn, params, vjp=None)
      fn(*inputs) -> tuple[Tensor]            pure function of the input tensors and the module parameters (everything else is a constant of the closure)
      params                                  the parameter tensors fn reads (they get their gradients; unused ones get None)
      vjp(ins, gs, wrt, create_graph) -> list  optional faster vjp: gradients of <fn(ins), gs> w.r.t. the tensors `wrt` (a subset of ins + params), with
                                              graph iff create_graph; default = recompute fn with grad and torch.autograd.grad
    outs = remat_apply(spec, inputs)          tuple of outputs (fresh tensors, never aliases of an input)

Level 1 `_Remat`: forward = fn without grad, saves the inputs. backward -> `_RematVJP.apply`.
Level 2 `_RematVJP`: forward = the vjp computed from detached input leaves (outputs are gradients, no graph, an empty tensor where nothing is needed or
the parameter is unused). backward (double backward) = recompute the vjp with create_graph, differentiate it w.r.t. inputs, parameters and cotangents.
No host sync, shapes static (only Python-side requires_grad flags are read).
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


class RematSpec:
    def __init__(self, fn: Callable[..., tuple[Tensor, ...]], params: tuple[Tensor, ...], vjp: Callable | None = None) -> None:
        self.fn, self.params, self.vjp = fn, tuple(params), vjp

    def grads(self, ins: list[Tensor], gs: list[Tensor], wrt: list[Tensor], create_graph: bool) -> list[Tensor | None]:
        """Gradients of <fn(ins), gs> w.r.t. `wrt`; call under enable_grad with ins / wrt requiring grad where needed."""
        if self.vjp is not None:
            return list(self.vjp(ins, gs, wrt, create_graph))
        outs = self.fn(*ins)
        return list(torch.autograd.grad(outs, wrt, gs, create_graph=create_graph, allow_unused=True))


def _live(g: Tensor | None, like: Tensor) -> Tensor | None:
    """An empty placeholder tensor (or None) means 'no gradient'."""
    return None if g is None or (g.numel() == 0 and like.numel() != 0) else g


def _split(spec: RematSpec, n_in: int, tensors) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    n_p = len(spec.params)
    return list(tensors[:n_in]), list(tensors[n_in:n_in + n_p]), list(tensors[n_in + n_p:])


class _RematVJP(torch.autograd.Function):
    """vjp of a remat unit: (inputs, params, output cotangents) -> gradients of the inputs and params that need them; differentiable once more."""

    @staticmethod
    def forward(ctx, spec, n_in, needs, *tensors):
        ctx.spec, ctx.n_in, ctx.needs = spec, n_in, needs
        ctx.save_for_backward(*tensors)
        ins, ps, gs = _split(spec, n_in, tensors)
        wrt_ps = [p for p, nd in zip(spec.params, needs[n_in:]) if nd]
        with torch.enable_grad():
            leaves = [x.detach().requires_grad_(bool(nd)) for x, nd in zip(ins, needs[:n_in])]
            wrt = [x for x, nd in zip(leaves, needs[:n_in]) if nd] + wrt_ps
            got = iter(spec.grads(leaves, gs, wrt, False) if wrt else ())
        like = ins + list(spec.params)
        return tuple(_empty_if_none(next(got), x) if nd else x.new_zeros(0) for x, nd in zip(like, needs))

    @staticmethod
    @once_differentiable
    def backward(ctx, *gg):
        spec, n_in, needs = ctx.spec, ctx.n_in, ctx.needs
        ins, ps, gs = _split(spec, n_in, ctx.saved_tensors)
        n_p = len(spec.params)
        nig = ctx.needs_input_grad[3:]                      # per (inputs, params, cotangents): does the second order reach it
        d_ins = [None] * len(ins)
        d_gs = [None] * len(gs)
        d_ps = [None] * n_p
        with torch.enable_grad():
            leaves = [x.detach().requires_grad_(bool(nd or m)) for x, nd, m in zip(ins, nig[:n_in], needs[:n_in])]
            cs = [g.detach().requires_grad_(bool(nd)) for g, nd in zip(gs, nig[n_in + n_p:])]
            wrt1 = [x for x, nd in zip(leaves, needs[:n_in]) if nd] + [p for p, nd in zip(spec.params, needs[n_in:]) if nd]
            if not wrt1:
                return (None,) * (3 + len(ins) + n_p + len(gs))
            grads = spec.grads(leaves, cs, wrt1, True)
            cot = [c for c, nd in zip(gg, needs) if nd]        # cotangents of the gradients that were returned (same order as wrt1)
            pairs = [(g, c) for g, c in zip(grads, cot) if g is not None and c is not None and c.numel() == g.numel()]
            sel_in = [i for i in range(n_in) if nig[i]]
            sel_p = [j for j in range(n_p) if nig[n_in + j]]
            sel_g = [k for k in range(len(gs)) if nig[n_in + n_p + k]]
            wrt2 = [leaves[i] for i in sel_in] + [spec.params[j] for j in sel_p] + [cs[k] for k in sel_g]
            if pairs and wrt2:
                second = torch.autograd.grad([g for g, _ in pairs], wrt2, [c for _, c in pairs], allow_unused=True)
            else:
                second = (None,) * len(wrt2)
        it = iter(second)
        for i in sel_in:
            d_ins[i] = next(it)
        for j in sel_p:
            d_ps[j] = next(it)
        for k in sel_g:
            d_gs[k] = next(it)
        return (None, None, None, *d_ins, *d_ps, *d_gs)


def _empty_if_none(g: Tensor | None, like: Tensor) -> Tensor:
    return like.new_zeros(0) if g is None else g


class _Remat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spec, n_in, *tensors):
        ctx.spec, ctx.n_in = spec, n_in
        ctx.save_for_backward(*tensors)
        with torch.enable_grad():  # same kernels as the plain path (e.g. rms_norm picks its kernel by requires_grad): bit-identical outputs
            leaves = [x.detach().requires_grad_(x.requires_grad) for x in tensors[:n_in]]
            outs = tuple(o.clone() if any(o is x for x in leaves) else o.detach() for o in spec.fn(*leaves))
        return outs

    @staticmethod
    def backward(ctx, *gouts):
        spec, n_in = ctx.spec, ctx.n_in
        tensors = ctx.saved_tensors
        needs = tuple(bool(n) for n in ctx.needs_input_grad[2:])
        if not any(needs):
            return (None, None) + (None,) * len(tensors)
        grads = _RematVJP.apply(spec, n_in, needs, *tensors, *gouts)
        like = list(tensors)
        return (None, None, *[_live(g, x) if nd else None for g, x, nd in zip(grads, like, needs)])


def remat_apply(spec: RematSpec, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Outputs of spec.fn(*inputs); under no autograd tracking (no input / parameter needs grad) it is a plain call."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (*inputs, *spec.params))):
        return tuple(spec.fn(*inputs))
    return _Remat.apply(spec, len(inputs), *inputs, *spec.params)
