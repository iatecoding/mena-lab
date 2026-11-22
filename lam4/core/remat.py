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


NODE_CHUNK_ATOMS = 8192
MPS_NODE_CHUNK_ATOMS = 2048
MPS_ENV_CHUNK_ATOMS = 1024


def node_chunk_size(device: torch.device | None = None, *, env_mod: bool = False) -> int:
    """Chunk size keeping MPS intermediates under Metal's 10 MiB SmallHeap threshold."""
    if device is not None and device.type == "mps":
        return MPS_ENV_CHUNK_ATOMS if env_mod else MPS_NODE_CHUNK_ATOMS
    return 2048 if env_mod else NODE_CHUNK_ATOMS


def can_chunk_nodes(
    n_atoms: int,
    params: tuple[Tensor, ...] = (),
    *,
    min_atoms: int = NODE_CHUNK_ATOMS,
    create_graph: bool = False,
) -> bool:
    """True when large-graph inference node chunking applies without disturbing training, compile, or second-order graphs."""
    return (
        n_atoms > min_atoms
        and not create_graph
        and not torch.compiler.is_compiling()
        and not any(p.requires_grad for p in params)
    )


class _ChunkedNodeFn(torch.autograd.Function):
    """Fused node-chunked autograd wrapper: eliminates N/C SliceBackward + Remat + CatBackward fanout."""

    @staticmethod
    def forward(ctx, fn, n_atoms: int, chunk_atoms: int, n_sliced: int, *tensors: Tensor):
        sliced = tensors[:n_sliced]
        shared = tensors[n_sliced:]
        ctx.fn = fn
        ctx.n_atoms = n_atoms
        ctx.chunk_atoms = chunk_atoms
        ctx.n_sliced = n_sliced
        ctx.save_for_backward(*tensors)
        outs: list[Tensor] | None = None
        single = n_atoms <= chunk_atoms
        with torch.enable_grad():
            for a0 in range(0, n_atoms, chunk_atoms):
                a1 = min(a0 + chunk_atoms, n_atoms)
                s_c = tuple(x[a0:a1].detach().requires_grad_(x.requires_grad) for x in sliced)
                u_c = tuple(x.detach().requires_grad_(x.requires_grad) for x in shared)
                ys = fn(a0, a1, *s_c, *u_c)
                if not isinstance(ys, tuple):
                    ys = (ys,)
                ys_d = tuple(
                    y.clone() if any(y is x for x in (*s_c, *u_c)) else y.detach()
                    for y in ys
                )
                if single:
                    return ys_d
                if outs is None:
                    outs = [y.new_empty((n_atoms, *y.shape[1:])) for y in ys_d]
                for out, y in zip(outs, ys_d):
                    out[a0:a1].copy_(y)
        assert outs is not None
        return tuple(outs)

    @staticmethod
    def backward(ctx, *gouts: Tensor | None):
        fn, n_atoms, chunk_atoms, n_sliced = ctx.fn, ctx.n_atoms, ctx.chunk_atoms, ctx.n_sliced
        tensors = ctx.saved_tensors
        needs = tuple(ctx.needs_input_grad[4:])
        if not any(needs):
            return (None, None, None, None) + (None,) * len(tensors)
        if torch.is_grad_enabled():
            raise RuntimeError("_ChunkedNodeFn does not support create_graph=True")
        sliced = tensors[:n_sliced]
        shared = tensors[n_sliced:]
        need_s = needs[:n_sliced]
        need_u = needs[n_sliced:]
        d_sliced: list[Tensor | None] = [None] * n_sliced
        d_shared: list[Tensor | None] = [None] * len(shared)
        with torch.enable_grad():
            for a0 in range(0, n_atoms, chunk_atoms):
                a1 = min(a0 + chunk_atoms, n_atoms)
                s_c = [x[a0:a1].detach().requires_grad_(nd) for x, nd in zip(sliced, need_s)]
                u_c = [x.detach().requires_grad_(nd) for x, nd in zip(shared, need_u)]
                ys = fn(a0, a1, *s_c, *u_c)
                if not isinstance(ys, tuple):
                    ys = (ys,)
                pairs = [(y, g[a0:a1]) for y, g in zip(ys, gouts) if g is not None and y.requires_grad]
                if not pairs:
                    for buf in d_sliced:
                        if buf is not None:
                            buf[a0:a1].zero_()
                    continue
                wrt_s = [(i, t) for i, (t, nd) in enumerate(zip(s_c, need_s)) if nd]
                wrt_u = [(j, t) for j, (t, nd) in enumerate(zip(u_c, need_u)) if nd]
                wrt = [t for _, t in wrt_s] + [t for _, t in wrt_u]
                if not wrt:
                    continue
                grads = torch.autograd.grad([y for y, _ in pairs], wrt, [g for _, g in pairs], allow_unused=True)
                it = iter(grads)
                for i, _ in wrt_s:
                    gi = next(it)
                    buf = d_sliced[i]
                    if gi is None:
                        if buf is not None:
                            buf[a0:a1].zero_()
                        continue
                    if buf is None:
                        buf = sliced[i].new_zeros(sliced[i].shape) if a0 > 0 else sliced[i].new_empty(sliced[i].shape)
                        d_sliced[i] = buf
                    buf[a0:a1].copy_(gi)
                for j, _ in wrt_u:
                    gj = next(it)
                    if gj is not None:
                        d_shared[j] = gj.detach() if d_shared[j] is None else (d_shared[j] + gj.detach())
        return (None, None, None, None, *d_sliced, *d_shared)


def chunked_node_apply(
    fn,
    sliced: tuple[Tensor, ...],
    shared: tuple[Tensor, ...] = (),
    params: tuple[Tensor, ...] = (),
    *,
    chunk_atoms: int = NODE_CHUNK_ATOMS,
    create_graph: bool = False,
) -> tuple[Tensor, ...]:
    """Run `fn(a0, a1, *s_chunk, *shared)` in `[chunk_atoms]` slices without materialising `[N]` intermediates."""
    n_atoms = sliced[0].shape[0]
    if not torch.is_grad_enabled():
        if n_atoms <= chunk_atoms:
            res = fn(0, n_atoms, *sliced, *shared)
            return res if isinstance(res, tuple) else (res,)
        outs: list[Tensor] | None = None
        for a0 in range(0, n_atoms, chunk_atoms):
            a1 = min(a0 + chunk_atoms, n_atoms)
            ys = fn(a0, a1, *(x[a0:a1] for x in sliced), *shared)
            if not isinstance(ys, tuple):
                ys = (ys,)
            if outs is None:
                outs = [y.new_empty((n_atoms, *y.shape[1:])) for y in ys]
            for out, y in zip(outs, ys):
                out[a0:a1].copy_(y)
        assert outs is not None
        return tuple(outs)
    if not create_graph and not any(p.requires_grad for p in params):
        return _ChunkedNodeFn.apply(fn, n_atoms, chunk_atoms, len(sliced), *sliced, *shared)
    n_s = len(sliced)
    ins_all = tuple(sliced)
    parts = []
    for a0 in range(0, n_atoms, chunk_atoms):
        a1 = min(a0 + chunk_atoms, n_atoms)
        def _remat_chunk(*xs, _a0=a0, _a1=a1):
            ys = fn(_a0, _a1, *xs[:n_s], *xs[n_s:])
            return ys if isinstance(ys, tuple) else (ys,)
        parts.append(remat_apply(RematSpec(_remat_chunk, params), (*(x[a0:a1] for x in ins_all), *shared)))
    return tuple(torch.cat([p[k] for p in parts], dim=0) for k in range(len(parts[0])))
