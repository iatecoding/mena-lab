"""Streaming edge aggregation: ONE autograd node for the whole op, chunked over receiver ranges, recomputing edge tensors.

Forward (`StreamAgg.forward`) loops over the chunks of an `edge_chunks.make_chunks` plan, computes each chunk's partial sums for its
own receivers (no atomics) and writes them into [N, .] outputs. Nothing edge-sized is saved: only the op's INPUTS.
Backward (`_StreamVJP.forward`) recomputes every chunk's edge terms and differentiates them. `_StreamVJP.backward` (double backward,
needed for force training) recomputes again and differentiates the chunk's vjp. Third order raises.

Row restriction: a chunk reads receiver rows a:b (pos, qn, rho) and the chunk's unique sender rows (pos, gates, vec, kn, rho) only, as
fresh leaves of that size; each chunk's gradient (and double-backward cotangent) has the size of those leaves and is scattered into ONE
full-size accumulator per input (`_scatter`: slice add for receivers, index_add_ for senders). Table and edge_scale grads are the
table-sized tensor and the chunk's edge slice.

Input order after (plan, layout, cfg, shift, z): pos, coef, gates, vec, qn, kn, rho, esc, ten  (vec, qn, kn, rho, esc, ten: empty tensor = absent).
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from lam4.core.edge_chunks import Chunk
from lam4.core.edge_terms import EdgeCfg, edge_terms
from lam4.core.layout import HEADS, StageLayout

N_IN = 10


def empty_z(like: Tensor, z: Tensor | None) -> Tensor:
    """Atomic numbers [N] long for the descriptor floor, or an empty long tensor (= no floor) as the op's constant argument."""
    return like.new_zeros(0, dtype=torch.long) if z is None else z
# leaf k of a chunk <- (input index, rows): r = receiver range, s = unique senders, e = the chunk's edges, f = whole tensor
LEAF = ((0, "r"), (0, "s"), (1, "f"), (2, "s"), (3, "s"), (4, "r"), (5, "s"), (6, "r"), (6, "s"), (7, "e"), (8, "s"), (9, "e"))


def _restrict(ch: Chunk, t: Tensor, kind: str) -> Tensor:
    if t.numel() == 0 or kind == "f":
        return t
    if kind == "r":
        return t[ch.a:ch.b]
    return t.index_select(0, ch.uniq) if kind == "s" else t[ch.e0:ch.e1]


def _scatter(ch: Chunk, acc: Tensor, g: Tensor, kind: str) -> None:
    if kind == "r":
        acc[ch.a:ch.b] += g
    elif kind == "s":
        acc.index_add_(0, ch.uniq, g)
    elif kind == "e":
        acc[ch.e0:ch.e1] += g
    else:
        acc += g


def _leaves(ch: Chunk, ins, needs, grad: bool) -> list[Tensor]:
    out = []
    for i, kind in LEAF:
        t = _restrict(ch, ins[i], kind).detach()
        out.append(t.requires_grad_(True) if (grad and needs[i]) else t)
    return out


def _edges(ch: Chunk, lv, shift: Tensor, zc: Tensor, layout: StageLayout, cfg: EdgeCfg):
    """Partial sums (num [n, out], z [n, H], dsum [n]) of the chunk for its n = b - a receivers."""
    pr, ps, coef, gs, vs, qr, ks, rr, rs, esc, ts, dsh = lv
    opt = lambda t: t if t.numel() else None  # noqa: E731
    zr, zs = (None, None) if zc.numel() == 0 else (zc[ch.a:ch.b], zc.index_select(0, ch.uniq))
    sh = dsh if dsh.numel() else shift[ch.e0:ch.e1]
    msg, zw, chi = edge_terms(pr, ps, ch.recv, ch.send, sh, coef, gs, opt(vs), opt(qr), opt(ks), opt(rr), opt(rs),
                              opt(esc), layout, cfg, zr, zs, opt(ts))
    n, od = ch.b - ch.a, msg.shape[1]
    packed = torch.cat((msg, zw, chi.unsqueeze(-1)), dim=-1)
    acc = packed.new_zeros(n, od + HEADS + 1).index_add(0, ch.recv, packed)
    return acc[:, :od], acc[:, od:od + HEADS], acc[:, od + HEADS]


class _StreamVJP(torch.autograd.Function):
    """vjp of the op: (inputs, output cotangents) -> input gradients (empty tensor where not needed); differentiable once more."""

    @staticmethod
    def forward(ctx, plan, layout, cfg, needs, shift, zc, *tensors):
        ctx.plan, ctx.layout, ctx.cfg, ctx.needs = plan, layout, cfg, needs
        ctx.save_for_backward(shift, zc, *tensors)
        ins, gs = tensors[:N_IN], tensors[N_IN:]
        acc = [torch.zeros_like(x) if nd else x.new_zeros(0) for x, nd in zip(ins, needs)]
        idx = [k for k, (i, _) in enumerate(LEAF) if needs[i]]
        for ch in plan if idx else ():
            with torch.enable_grad():
                lv = _leaves(ch, ins, needs, True)
                grads = torch.autograd.grad(_edges(ch, lv, shift, zc, layout, cfg), [lv[k] for k in idx],
                                            [g[ch.a:ch.b] for g in gs], allow_unused=True)
            for k, g in zip(idx, grads):
                if g is not None:
                    _scatter(ch, acc[LEAF[k][0]], g, LEAF[k][1])
        return tuple(acc)

    @staticmethod
    @once_differentiable
    def backward(ctx, *gg):
        plan, layout, cfg, needs = ctx.plan, ctx.layout, ctx.cfg, ctx.needs
        shift, zc, *tensors = ctx.saved_tensors
        ins, gs = tensors[:N_IN], tensors[N_IN:]
        d_in = [torch.zeros_like(x) if nd else None for x, nd in zip(ins, needs)]
        d_g = [torch.zeros_like(g) for g in gs]
        idx = [k for k, (i, _) in enumerate(LEAF) if needs[i]]
        for ch in plan if idx else ():
            with torch.enable_grad():
                lv = _leaves(ch, ins, needs, True)
                cs = [g[ch.a:ch.b].detach().requires_grad_(True) for g in gs]
                sel = [lv[k] for k in idx]
                grads = torch.autograd.grad(_edges(ch, lv, shift, zc, layout, cfg), sel, cs, create_graph=True, allow_unused=True)
                pairs = [(k, g) for k, g in zip(idx, grads) if g is not None and gg[LEAF[k][0]] is not None]
                if not pairs:
                    continue
                cots = [_restrict(ch, gg[LEAF[k][0]], LEAF[k][1]) for k, _ in pairs]
                second = torch.autograd.grad([g for _, g in pairs], sel + cs, cots, allow_unused=True)
            for k, s in zip(idx, second):
                if s is not None:
                    _scatter(ch, d_in[LEAF[k][0]], s, LEAF[k][1])
            for j, s in enumerate(second[len(sel):]):
                if s is not None:
                    d_g[j][ch.a:ch.b] += s
        return (None, None, None, None, None, None, *d_in, *d_g)


class StreamAgg(torch.autograd.Function):
    """(num [N, out_dim], z [N, H], dsum [N]) of the whole edge list; saves only its inputs."""

    @staticmethod
    def forward(ctx, plan, layout, cfg, shift, zc, *ins):
        ctx.plan, ctx.layout, ctx.cfg = plan, layout, cfg
        ctx.save_for_backward(shift, zc, *ins)
        n, like = ins[0].shape[0], ins[0]
        num, z, dsum = like.new_zeros(n, layout.out_dim), like.new_zeros(n, HEADS), like.new_zeros(n)
        needs = (False,) * N_IN
        for ch in plan:
            num[ch.a:ch.b], z[ch.a:ch.b], dsum[ch.a:ch.b] = _edges(ch, _leaves(ch, ins, needs, False), shift, zc, layout, cfg)
        return num, z, dsum

    @staticmethod
    def backward(ctx, g_num, g_z, g_d):
        needs = tuple(bool(n) for n in ctx.needs_input_grad[5:])
        shift, zc, *ins = ctx.saved_tensors
        grads = _StreamVJP.apply(ctx.plan, ctx.layout, ctx.cfg, needs, shift, zc, *ins, g_num, g_z, g_d)
        return (None, None, None, None, None, *[g if nd else None for g, nd in zip(grads, needs)])
