"""Rematerialisation wiring of `Stage` (see remat.py for the mechanism). Two modes, chosen by the Stage constructor flag `remat`:

    "stage"  one unit per call: core (and finish) saves only its inputs (s [N, d], vec [N, 32, 3], pos, rho, cond) and recomputes everything in backward,
             the streaming edge op included (one extra edge forward per stage in backward: the edge forward is the dominant cost).
    "block"  core = A + B. A = node-side prologue (norm, FiLM, gate_proj, q / k, tables, transported vector) + the streaming edge op, whose vjp is
             written by hand: recompute the prologue, call the edge vjp (`_StreamVJP`, which recomputes the edge terms itself) and backpropagate the
             prologue, so the edge forward is NOT repeated. A returns the RAW sums (num, z, dsum); B = finalise + invariants + angular + SwiGLU + vector
             gate (+ finish when the caller asked for it) saves (s, vec, num, z, dsum). Per stage ~ d + 96 + out_dim + 5 floats instead of ~15k.
For the reference edge path (eg.mode == "ref") A is the generic unit (the oracle materialises edge tensors anyway).

Differentiable tensors of the unit: s, vec, eg.pos, eg.rho, eg.edge_scale, eg.shift (only if it requires grad), nc.cond; every other field of
eg / nc (indices, masks, z, charge, spin, chunk plan) is a constant of the closure.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from lam4.core.edge_chunks import DEFAULT_CHUNK_EDGES, make_chunks
from lam4.core.edge_ops import _check, _gain
from lam4.core.edge_stream import StreamAgg, _StreamVJP, empty_z
from lam4.core.edge_terms import finalise, normalise_qk
from lam4.core.remat import RematSpec, remat_apply
from lam4.core.spline_tables import windows

MODES = ("none", "stage", "block")


def check_mode(remat) -> str:
    mode = {False: "none", None: "none", True: "stage"}.get(remat, remat)
    if mode not in MODES:
        raise ValueError(f"remat must be a bool or one of {MODES}, got {remat!r}")
    return mode


def pack(eg, nc):
    """(tensors, rebuild): the differentiable tensors of (eg, nc) and a function that puts (possibly replaced) tensors back."""
    has_g = eg.geom is not None
    names = ([n for n in ("edge_scale", "phi") if getattr(eg, n) is not None]) if has_g else (
        ["pos"] + [n for n in ("rho", "edge_scale") if getattr(eg, n) is not None]
        + (["shift"] if eg.shift.requires_grad else [])
        + (["cell"] if (eg.cell is not None and eg.cell.requires_grad) else [])
    )
    has_cond = nc is not None and nc.cond is not None
    ts = [getattr(eg, n) for n in names] + (list(eg.geom) if has_g else []) + ([nc.cond] if has_cond else [])

    def rebuild(*t):
        k = len(names)
        g = tuple(t[k:k + 5]) if has_g else None
        return replace(eg, geom=g, **dict(zip(names, t[:k]))), (replace(nc, cond=t[-1]) if has_cond else nc)
    return tuple(ts), rebuild


def stream_ins(st, eg, coef, gates, q, k, b, tau=None):
    """The 10 inputs of StreamAgg (empty tensor = absent), as edge_aggregate_stream builds them."""
    _check(st.layout, eg.pos, gates, coef, b, q, eg.edge_scale, tau)
    empty = eg.pos.new_zeros(0)
    qn, kn = (empty, empty) if eg.edge_scale is not None else _gain(normalise_qk(q, k), st.heads.gain())
    opt = lambda t: empty if t is None else t  # noqa: E731
    sh_d = eg.shift if eg.shift.requires_grad else empty
    return (eg.pos, windows(coef), gates, opt(b), qn, kn, opt(eg.rho), opt(eg.edge_scale), opt(tau if st.layout.later else None), sh_d)


def unit_stage(st, s, vec, eg, nc, domain: str, fin: bool, ten=None, return_ten: bool = False):
    ts, rb = pack(eg, nc)
    has_t = ten is not None

    def fn(s_, v_, *rest):
        t_, t = (rest[0], rest[1:]) if has_t else (None, rest)
        e, n = rb(*t)
        out = st.core_plain(s_, v_, e, n, domain, t_, return_ten=return_ten)
        return ((st.finish_plain(out[0], n, domain) if fin else out[0]), *out[1:])
    return remat_apply(RematSpec(fn, tuple(st.parameters())), (s, vec, *([ten] if has_t else []), *ts))


def unit_finish(st, s, nc, domain: str):
    return remat_apply(RematSpec(lambda s_: (st.finish_plain(s_, nc, domain),), tuple(st.parameters())), (s,))[0]


def _unit_a(st, s, vec, eg, nc, ten=None):
    """A: prologue + edge op -> (num, z, dsum) for the stream path, (out,) for the ref / csr path."""
    if eg.mode == "csr" and not eg.create_graph and not any(p.requires_grad for p in st.parameters()):
        return (st.aggregate_plain(s, vec, eg, nc, ten=ten),), False
    ts, rb = pack(eg, nc)
    has_t = ten is not None
    inputs = (s, vec, *([ten] if has_t else []), *ts)
    if eg.mode != "stream":
        def fn_ref(s_, v_, *rest):
            t_, t = (rest[0], rest[1:]) if has_t else (None, rest)
            return (st.aggregate_plain(s_, v_, *rb(*t), ten=t_),)
        return remat_apply(RematSpec(fn_ref, tuple(st.parameters())), inputs), False
    plan = eg.chunks if eg.chunks is not None else make_chunks(eg.recv, eg.send, eg.pos.shape[0], DEFAULT_CHUNK_EDGES)

    def ins_of(s_, v_, rest):
        t_, t = (rest[0], rest[1:]) if has_t else (None, rest)
        e, n = rb(*t)
        return e, stream_ins(st, e, *st.prep(s_, v_, e, n, t_))

    def fn(s_, v_, *rest):
        e, ins = ins_of(s_, v_, rest)
        sh_c = e.pos.new_zeros(0) if e.shift.requires_grad else e.shift
        return StreamAgg.apply(plan, st.layout, st.edge_cfg, sh_c, empty_z(e.pos, e.z), *ins)

    def vjp(ins, gs, wrt, create_graph):
        e, sins = ins_of(ins[0], ins[1], ins[2:])
        needs = tuple(bool(t.requires_grad) for t in sins)
        sh_c = e.pos.new_zeros(0) if e.shift.requires_grad else e.shift
        with torch.set_grad_enabled(create_graph):
            gi = _StreamVJP.apply(plan, st.layout, st.edge_cfg, needs, sh_c, empty_z(e.pos, e.z), *(sins if create_graph else [t.detach() for t in sins]), *gs)
        pairs = [(t, g) for t, g, nd in zip(sins, gi, needs) if nd]
        return torch.autograd.grad([t for t, _ in pairs], wrt, [g for _, g in pairs], create_graph=create_graph, allow_unused=True)
    return remat_apply(RematSpec(fn, tuple(st.parameters()), vjp), inputs), True


def post_chunked_crystal(st, s, vec, agg, nc, domain: str, fin: bool, ten=None, return_ten: bool = False, *, raw: bool = False, chunk_atoms: int = 4096):
    has_t = ten is not None
    ps = tuple(st.parameters())
    n_tot = s.shape[0]

    def fn_slice(a0: int, a1: int):
        nc_s = None if nc is None else replace(nc, atom_mask=None if nc.atom_mask is None else nc.atom_mask[a0:a1], graph=None)

        def _fn(s_, v_, *rest):
            t_, a = (rest[0], rest[1:]) if has_t else (None, rest)
            out = finalise(*a, st.layout) if raw else a[0]
            res = st.post_plain(s_, v_, out, domain, t_, return_ten=return_ten)
            return ((st.finish_plain(res[0], nc_s, domain) if fin else res[0]), *res[1:])
        ins_s = (s[a0:a1], vec[a0:a1], *([ten[a0:a1]] if has_t else []), *(x[a0:a1] for x in agg))
        return remat_apply(RematSpec(_fn, ps), ins_s)

    parts = [fn_slice(a0, min(a0 + chunk_atoms, n_tot)) for a0 in range(0, n_tot, chunk_atoms)]
    return tuple(torch.cat([p[k] for p in parts], dim=0) for k in range(len(parts[0])))


def unit_block(st, s, vec, eg, nc, domain: str, fin: bool, ten=None, return_ten: bool = False):
    agg, raw = _unit_a(st, s, vec, eg, nc, ten)
    if s.shape[0] > 4096 and domain == "crystal" and st._inv_hook is None and not torch.compiler.is_compiling():
        return post_chunked_crystal(st, s, vec, agg, nc, domain, fin, ten, return_ten=return_ten, raw=raw)
    has_t = ten is not None
    ps = tuple(st.parameters())

    def fn(s_, v_, *rest):
        t_, a = (rest[0], rest[1:]) if has_t else (None, rest)
        out = finalise(*a, st.layout) if raw else a[0]
        res = st.post_plain(s_, v_, out, domain, t_, return_ten=return_ten)
        return ((st.finish_plain(res[0], nc, domain) if fin else res[0]), *res[1:])
    return remat_apply(RematSpec(fn, ps), (s, vec, *([ten] if has_t else []), *agg))
