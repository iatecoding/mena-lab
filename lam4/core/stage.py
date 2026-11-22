"""One trunk stage of lam4 = the earlier Mena stage (bounded vectors, additive vector stream, angular branch) on the product-only edge op.

`Stage(cfg, layout, index)`: layout FIRST for index 0, LATER for index >= 1 (layout.py). Two entry points, so the trunk can put the readout and
the feedback between them (earlier Mena model order):
    core(s, vec, eg, nc, domain)   -> (s, vec_out)    the earlier Mena stage: s [N, d], vec [N, 32, 3] (raw additive stream; zeros before stage 0)
    finish(s, nc, domain)          -> s               GroupContext (molecule only) -> state_norm (rms, no weight) -> atom mask
    forward(...) = finish(core(...))                  (no readout / feedback in between)
`eg` is an EdgeGraph, `nc` a NodeContext (stage_graph.py); `domain` is the per-batch string that picks the SwiGLU expert and gates the group context.

core, in the order of earlier Mena trunk.py (one deviation marked FiLM / HEADS):
 1  h = norm_in(s); FiLM: h <- h (1 + gamma(c_g)) + beta(c_g), (gamma, beta) = Linear(cond_dim, 2d)(c) zero-init, c the per-graph conditioning (nc.cond)
 2  gates = gate_proj(h) [N, 184] = scalar | vec | ten | o (the earlier Mena trunk: scalar_pre, vec_gate, ten_gate, ang.gate3 fused); q, k = EdgeHeads(h) (HEADS)
 3  coef = tables() (C2 B-spline of Phi(x) W, once per forward); LATER: vec_tr = transport_mix(bound(vec)) is the sender vector b_j
 4  agg = edge_aggregate_{ref,stream}(...) -> split_output: s, v, (lon), t, o already carry the degree normaliser (the earlier Mena trunk's `* norm`)
 5  v_new = vec + v;  u_new = bound(v_new) (vec_c); T = t, T3 = o
 6  invariants: nu2_v = <U1 u, V1 u>, nu2_t = <U2 T, V2 T>, nu3 = <T3w T, l2_pair(P3 u, Q3 u)>, LATER + [lon2, cross_v, cross_t, lon] (cross_v/t bound both streams)
 7  h = inv_mlp[0](inv) + ang.out(ang(T, u_new, T3));  s <- out_norm(s + msg_up(agg_s) + inv_mlp[2](silu(h)))  (post-norm)
 8  s <- s + experts[domain](s)                     (node SwiGLU, one expert per domain, hard switch)
 9  vec_out = v_new + mix(u_new, vec_self) * tanh(vec_out_gate(vec_gate_norm(s)))   (additive stream, gate zero-init)
earlier Mena surprises replicated: the vector gate reads the POST-update s (step 9 uses s after the SwiGLU); the post-norm is a weighted RMSNorm (out_norm)
while the stage-end state_norm has no weight; the lon scalar enters inv last (its sign convention is lam4's, see layout.py).
Constructor flag `remat` ("none" | "stage" | "block" | bool, default "none"): node-side rematerialisation with double backward (stage_remat.py, remat.py):
backward recomputes the node math from a small boundary (s, vec, + the raw edge sums for "block") instead of saving ~55 kB/atom of activations per stage.
Same values bit for bit, same gradients to rounding; `core_plain / post_plain / finish_plain / prep` are the plain pieces the units are built from.
Zero-init (as earlier Mena + lam4): inv_mlp[2], ang.out, vec_out_gate, film, heads.raw_gain (q_proj / k_proj keep the normal init). At init the op is the unweighted earlier Mena sum (HEADS zero logit).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.cond.expert import DomainExperts, build_ffn
from lam4.config import ModelConfig
from lam4.core.angular import AngularBranch, l2_pair_mm
from lam4.core.consts import tracing
from lam4.core.edge_ops import edge_aggregate_ref, edge_aggregate_stream
from lam4.core.edge_terms import EdgeCfg, _near_mask
from lam4.core.edge_weights import EdgeHeads
from lam4.core.inv_scale import CoordInvScale, build_stage_baked, compute_dsum, default_inv_scale, post_vector_tensor_update, stage_dispatch
from lam4.core.layout import HEADS, StageLayout, split_output
from lam4.core.radial import R_CUT, fp32_guard, resolve_baked, upcast_fp32
from lam4.core.remat import NODE_CHUNK_ATOMS, chunked_node_apply, node_chunk_size
from lam4.core.spline_tables import DEFAULT_GRID, BSplineTables
from lam4.core.stage_graph import EdgeGraph, NodeContext
from lam4.core.stage_remat import check_mode, unit_finish
from lam4.core.vector_bound import bound_vectors, chan_linear, dot_last, mix, state_norm
from lam4.kernels.csr_ops import edge_aggregate_csr
from lam4.kernels.metal_node import fused_inv, fused_vec_update, node_post_ok
from lam4.physics.group_context import GroupContext


class Stage(nn.Module):
    def __init__(self, cfg: ModelConfig, layout: StageLayout, index: int, remat: bool | str = "none") -> None:
        super().__init__()
        self.remat = check_mode(remat)
        d, k, k2, k3 = cfg.d_scalar, cfg.k_vec, cfg.k2, cfg.k3
        r1, r2, r3 = cfg.r1_vec, cfg.r2_ten, cfg.r3_tvv
        if cfg.n_heads != HEADS or (layout.k_s, layout.k_v, layout.k_t, layout.k_o) != (cfg.msg_dim, k, k2, k3):
            raise ValueError("ModelConfig (n_heads, msg_dim, k_vec, k2, k3) must match the stage layout (4, 128, 32, 16, 8)")
        self.layout, self.index, self.vec_c = layout, int(index), cfg.vec_c
        self.edge_cfg = EdgeCfg(cfg.r_cut, cfg.r_on, cfg.descriptor_floor, cfg.r_cut_near, cfg.r_on_near, cfg.k_s_far, cfg.k_v_far)
        self.use_state_norm, self.inv_norm = cfg.state_norm, bool(cfg.inv_norm)
        self.contract_cumulative_t = bool(cfg.contract_cumulative_t)
        self.coord_inv_scale = bool(cfg.coord_inv_scale)
        self.env_modulated = cfg.ffn_mode != "domain_experts"
        self._inv_hook = None
        self._baked: dict[str, Tensor] | None = None
        self.norm_in = nn.RMSNorm(d, eps=1e-6)
        self.film = nn.Linear(cfg.cond_dim, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.gate_proj = nn.Linear(d, layout.gate_dim)
        self.tables = BSplineTables(layout.table_dim, cfg.spline_grid, cfg.r_cut)
        self._init_tables()
        self.heads = EdgeHeads(d, HEADS, cfg.head_dim, cfg.edge_gain_max)
        self.transport_mix = nn.Linear(k, k, bias=False) if layout.later else None
        self.transport_ten = nn.Linear(k2, k2, bias=False) if layout.later else None
        lin = lambda i, o: nn.Linear(i, o, bias=False)  # noqa: E731
        self.u1, self.v1, self.u2, self.v2 = lin(k, r1), lin(k, r1), lin(k2, r2), lin(k2, r2)
        self.p3, self.q3, self.t3 = lin(k, r3), lin(k, r3), lin(k2, r3)
        self.u_cv, self.v_cv = (lin(k, k), lin(k, k)) if layout.later else (None, None)
        self.u_ct, self.v_ct = (lin(k2, k2), lin(k2, k2)) if layout.later else (None, None)
        ang_dim = cfg.ttt_mix + cfg.vttv_rank + 2 * cfg.k3_read
        inv_dim = r1 + r2 + r3 + (k + k2 + k + k2 if layout.later else 0) + (ang_dim if self.inv_norm else 0)
        self.inv_mlp = nn.Sequential(nn.Linear(inv_dim, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.inv_mlp[-1].weight)
        nn.init.zeros_(self.inv_mlp[-1].bias)
        self.ang = AngularBranch(d, k, k2, k3, cfg.ttt_mix, cfg.vttv_rank, cfg.k3_read, has_out=not self.inv_norm)
        self.msg_up = lin(cfg.msg_dim, d)
        self.out_norm = nn.RMSNorm(d, eps=1e-6)
        self.swiglu = build_ffn(cfg)
        self.vec_self, self.ten_self = lin(k, k), lin(k2, k2)
        self.vec_gate_norm = nn.RMSNorm(d, eps=1e-6)
        self.vec_out_gate = nn.Linear(d, k)
        self.ten_out_gate = nn.Linear(d, k2)
        for g in (self.vec_out_gate, self.ten_out_gate):
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)
        self.group_ctx = GroupContext(d) if cfg.group_context else None
        if self.inv_norm:
            if self.coord_inv_scale:
                self.coord_scale: CoordInvScale | None = CoordInvScale(
                    inv_dim, default_inv_scale(cfg, layout), cfg.coord_alpha_max, signed=cfg.coord_alpha_signed
                )
                self.inv_scale = None
            else:
                self.coord_scale = None
                self.register_buffer("inv_scale", default_inv_scale(cfg, layout))
            self.inv_state_gate = nn.Linear(d, d)
            self.tv_t, self.tv_v, self.tv_gate = lin(k2, k), lin(k, k), nn.Linear(d, k)
            self.vv_p, self.vv_q, self.vv_gate = lin(k, k2), lin(k, k2), nn.Linear(d, k2)
            for g in (self.inv_state_gate, self.tv_gate, self.vv_gate):
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)
        else:
            self.coord_scale = None
        self.aggregate_plain = self._agg_plain_later if layout.later else self._agg_plain_first
        self.post_plain = self._post_plain_later if layout.later else self._post_plain_first
        self.core_plain = self._core_plain_later if layout.later else self._core_plain_first

    @torch.no_grad()
    def _init_tables(self) -> None:
        """the earlier Mena trunk's filters are bias-free nn.Linear(32, .): uniform(+-1/sqrt(32)); the 4 head-bias columns start at 0 (logit bias 0)."""
        w = self.tables.weight
        w.copy_((torch.rand_like(w) * 2.0 - 1.0) / w.shape[0] ** 0.5)
        w[:, :HEADS] = 0.0

    def _condition(self, h: Tensor, nc: NodeContext | None, *, a0: int | None = None, a1: int | None = None,
                   cond_atom: Tensor | None = None) -> Tensor:
        ca = cond_atom if cond_atom is not None else (None if nc is None else getattr(nc, "cond_atom", None))
        if ca is not None:
            ca_sl = ca if (a0 is None or ca.shape[0] == h.shape[0]) else ca[a0:a1]
            gamma, beta = self.film(ca_sl).chunk(2, dim=-1)
            return h * (1.0 + gamma) + beta
        if nc is None or nc.cond is None:
            return h
        c = self.film(nc.cond)
        if c.shape[0] == 1:
            gamma, beta = c.chunk(2, dim=-1)
        else:
            g_idx = nc.graph if a0 is None else nc.graph[a0:a1]
            gamma, beta = c.index_select(0, g_idx).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta

    def train(self, mode: bool = True):
        if mode:
            self._baked = None
        return super().train(mode)

    def _apply(self, fn):
        had_baked = self._baked is not None
        self._baked = None
        out = super()._apply(fn)
        if had_baked and not self.training and not self.u1.weight.is_meta and not tracing():
            self.bake_for_inference()
        return out

    @torch.no_grad()
    def bake_for_inference(self) -> None:
        self.ang.bake_for_inference()
        if self.group_ctx is not None:
            self.group_ctx.bake_for_inference()
        if self.edge_cfg.r_cut_near > 0.0:
            _near_mask(self.layout, self.edge_cfg, self.u1.weight)
        self._baked = build_stage_baked(self)

    def unbake_inference(self) -> None:
        self._baked = None
        self.ang._baked = None
        if self.group_ctx is not None:
            self.group_ctx.unbake_inference()

    def _get_baked(self, like: Tensor) -> dict[str, Tensor] | None:
        return resolve_baked(self, lambda bk: bk["w_uvpq"], self.u1.weight, like)

    def _agg_impl(self, eg: EdgeGraph, coef: Tensor, gates: Tensor, q: Tensor | None, k: Tensor | None, b: Tensor | None,
                  tau: Tensor | None = None, *, return_dsum: bool = False):
        bk = self._get_baked(gates)
        gain = None if q is None else (bk["gain"] if bk is not None else self.heads.gain())
        kw = dict(layout=self.layout, vec=b, ten=tau, rho=eg.rho, edge_scale=eg.edge_scale, cfg=self.edge_cfg, z=eg.z, qk_gain=gain)
        if eg.mode == "csr":
            p4 = bk.get("poly4") if (bk is not None and int(bk["poly"]._version) == bk.get("_poly_ver", -1)) else None
            return edge_aggregate_csr(eg.pos, eg.recv, eg.send, eg.shift if eg.shift.ndim == 2 else None, coef, gates, q, k,
                                      csr_plan=eg.csr_plan, poly=bk["poly"] if bk is not None else None,
                                      poly4=p4, edge_shift=eg.edge_shift,
                                      cell=eg.cell, graph=eg.graph, create_graph=eg.create_graph, allow_bwd2=True, return_dsum=return_dsum, **kw)
        if eg.mode == "stream":
            out = edge_aggregate_stream(eg.pos, eg.recv, eg.send, eg.shift, coef, gates, q, k, chunks=eg.chunks, **kw)
        else:
            out = edge_aggregate_ref(eg.pos, eg.recv, eg.send, eg.shift, coef, gates, q, k, geom=eg.geom, phi=eg.phi, **kw)
        return (out, compute_dsum(eg, self.edge_cfg)) if return_dsum else out

    def _aggregate_first(self, eg: EdgeGraph, coef: Tensor, gates: Tensor, q: Tensor | None, k: Tensor | None, *, return_dsum: bool = False):
        return self._agg_impl(eg, coef, gates, q, k, None, None, return_dsum=return_dsum)

    def _aggregate_later(self, eg: EdgeGraph, coef: Tensor, gates: Tensor, q: Tensor | None, k: Tensor | None,
                         b: Tensor | None, tau: Tensor | None = None, *, return_dsum: bool = False):
        return self._agg_impl(eg, coef, gates, q, k, b, tau, return_dsum=return_dsum)

    def _aggregate(self, eg: EdgeGraph, coef: Tensor, gates: Tensor, q: Tensor | None, k: Tensor | None, b: Tensor | None,
                   tau: Tensor | None = None, *, return_dsum: bool = False):
        return self._aggregate_later(eg, coef, gates, q, k, b, tau, return_dsum=return_dsum) if self.layout.later else self._aggregate_first(eg, coef, gates, q, k, return_dsum=return_dsum)

    def _prep_atoms(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, ten: Tensor | None = None,
                    *, a0: int | None = None, a1: int | None = None, cond_atom: Tensor | None = None) -> tuple[Tensor, ...]:
        h = self._condition(self.norm_in(s), nc, a0=a0, a1=a1, cond_atom=cond_atom)
        bk = self._get_baked(h)
        if bk is not None and eg.edge_scale is None:
            hd = self.heads.q_proj.out_features
            gates, q, k = F.linear(h, bk["w_gqk"], bk["b_gqk"]).split([self.layout.gate_dim, hd, hd], dim=-1)
        else:
            gates = self.gate_proj(h)
            q, k = (None, None) if eg.edge_scale is not None else self.heads(h)
        b = mix(bound_vectors(upcast_fp32(vec), self.vec_c), self.transport_mix) if self.layout.later else None
        tau = mix(bound_vectors(upcast_fp32(ten), self.vec_c), self.transport_ten) if (self.layout.later and ten is not None) else None
        out_dt = torch.float64 if s.dtype == torch.float64 else torch.float32
        return tuple(x.to(out_dt) for x in (gates, q, k, b, tau) if x is not None)

    def prep(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, ten: Tensor | None = None,
             *, cond_atom: Tensor | None = None) -> tuple:
        """Steps 1-3: (coef, gates, q, k, b, tau), the node-side inputs of the edge op."""
        has_qk = eg.edge_scale is None
        has_b = bool(self.layout.later)
        has_tau = bool(self.layout.later and ten is not None)
        tab = self.tables.weight if (self.tables.grid == DEFAULT_GRID and self.tables.r_cut == R_CUT) else self.tables()
        if self._use_csr_node_chunks(s, eg, "molecule"):
            ca_full = cond_atom if cond_atom is not None else (None if nc is None else getattr(nc, "cond_atom", None))
            has_c = ca_full is None and nc is not None and nc.cond is not None
            sliced = (s, vec) + ((ten,) if ten is not None else ()) + ((ca_full,) if ca_full is not None else ())
            shared = (nc.cond,) if has_c else ()

            def _fn(a0: int, a1: int, *xs):
                idx = 2
                t_c = xs[idx] if ten is not None else None
                idx += int(ten is not None)
                c_a = xs[idx] if ca_full is not None else None
                nc_c = None if not has_c else NodeContext(nc.graph, nc.charge, nc.spin, None, xs[-1])
                return self._prep_atoms(xs[0], xs[1], eg, nc_c, t_c, a0=a0, a1=a1, cond_atom=c_a)

            ca = 4096 if self.remat != "none" else (node_chunk_size(s.device) if s.device.type == "mps" else NODE_CHUNK_ATOMS)
            outs = chunked_node_apply(_fn, sliced, shared, params=tuple(self.parameters()), chunk_atoms=ca, create_graph=eg.create_graph)
        else:
            outs = self._prep_atoms(s, vec, eg, nc, ten, cond_atom=cond_atom)
        it = iter(outs)
        gates = next(it)
        q, k = (next(it), next(it)) if has_qk else (None, None)
        b = next(it) if has_b else None
        tau = next(it) if has_tau else None
        return tab, gates, q, k, b, tau

    def _agg_plain_first(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, ten: Tensor | None = None,
                         *, cond_atom: Tensor | None = None, return_dsum: bool = False):
        tab, gates, q, k, _, _ = self.prep(s, vec, eg, nc, None, cond_atom=cond_atom)
        if "_aggregate" in self.__dict__ and not return_dsum:
            return self._aggregate(eg, tab, gates, q, k, None, None)
        return self._aggregate_first(eg, tab, gates, q, k, return_dsum=return_dsum)

    def _agg_plain_later(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, ten: Tensor | None = None,
                         *, cond_atom: Tensor | None = None, return_dsum: bool = False):
        if "_aggregate" in self.__dict__ and not return_dsum:
            return self._aggregate(eg, *self.prep(s, vec, eg, nc, ten, cond_atom=cond_atom))
        return self._aggregate_later(eg, *self.prep(s, vec, eg, nc, ten, cond_atom=cond_atom), return_dsum=return_dsum)

    def aggregate_plain(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, ten: Tensor | None = None,
                        *, cond_atom: Tensor | None = None, return_dsum: bool = False):
        if return_dsum:
            return self._agg_impl(eg, *self.prep(s, vec, eg, nc, ten if self.layout.later else None, cond_atom=cond_atom), return_dsum=True)
        return (self._agg_plain_later if self.layout.later else self._agg_plain_first)(s, vec, eg, nc, ten, cond_atom=cond_atom)

    def _plain_inv(self, vec: Tensor, agg: Tensor, ten: Tensor | None, bk: dict[str, Tensor] | None):
        """Steps 5-6 (plain PyTorch): split, v_new / u_new / t_new / t_c and the invariant list (without the angular features)."""
        L = self.layout
        with fp32_guard(vec.device):
            p = split_output(agg, L)
            v_new = vec + p["v"]
            u_new = bound_vectors(v_new, self.vec_c)
            t = p["t"]
            t_new = t if ten is None else ten + t
            t_c = bound_vectors(t_new, self.vec_c) if self.contract_cumulative_t else t
            w_uvpq = bk["w_uvpq"] if bk is not None else upcast_fp32(torch.cat([self.u1.weight, self.v1.weight, self.p3.weight, self.q3.weight], 0))
            w_uvt = bk["w_uvt"] if bk is not None else upcast_fp32(torch.cat([self.u2.weight, self.v2.weight, self.t3.weight], 0))
            u1, v1, p3, q3 = chan_linear(u_new, w_uvpq).split([self.u1.out_features] * 2 + [self.p3.out_features] * 2, dim=-2)
            u2, v2, t3 = chan_linear(t_c, w_uvt).split([self.u2.out_features] * 2 + [self.t3.out_features], dim=-2)
            inv = [dot_last(u1, v1), dot_last(u2, v2), dot_last(t3, l2_pair_mm(p3, q3))]
            if L.later:
                cross_v = dot_last(mix(bound_vectors(vec, self.vec_c), self.u_cv), mix(bound_vectors(p["v"], self.vec_c), self.v_cv))
                cross_t = cross_v.new_zeros(vec.shape[0], L.k_t) if ten is None else dot_last(
                    mix(bound_vectors(ten, self.vec_c), self.u_ct), mix(bound_vectors(t, self.vec_c), self.v_ct))
                inv += [p["lon2"], cross_v, cross_t, p["lon"]]
        return p, v_new, u_new, t_new, t_c, inv

    def _post_impl(self, s: Tensor, vec: Tensor, agg: Tensor, domain: str, ten: Tensor | None = None, return_ten: bool = False,
                   *, cond_lot_atom: Tensor | None = None, dsum: Tensor | None = None, return_inv: bool = False,
                   allow_fused: bool = False):
        """Steps 5-9 from the (finalised) edge output agg [N, out_dim]. allow_fused (no create_graph): compiled MPS runs the fused
        node kernels (lam4/kernels/metal_node.py) for the invariants and the vector / tensor update; same math, FP32 rounding."""
        L = self.layout
        s_dt, v_dt = (torch.float64 if s.dtype == torch.float64 else torch.float32), (torch.float64 if vec.dtype == torch.float64 else torch.float32)
        s, vec, agg = upcast_fp32(s), upcast_fp32(vec), upcast_fp32(agg)
        ten = None if ten is None else upcast_fp32(ten)
        bk = self._get_baked(s)
        fused = node_post_ok(self, s, bk, allow_fused)
        if fused:
            p, all_inv = split_output(agg, L), fused_inv(self, vec, agg, ten, bk)
        else:
            p, v_new, u_new, t_new, t_c, inv = self._plain_inv(vec, agg, ten, bk)
        scaled_inv: Tensor | None = None
        if self.inv_norm:
            if not fused:
                all_inv = torch.cat([*inv, self.ang.features(t_c, u_new, p["o"])], dim=-1)
            if self._inv_hook is not None:
                self._inv_hook(all_inv)
            if self.coord_inv_scale:
                ds_f = s.new_zeros(s.shape[0]) if dsum is None else dsum
                scaled_inv = all_inv * self.coord_scale.multiplier(ds_f).to(all_inv.dtype)
                inv_h = self.inv_mlp[0](scaled_inv)
            else:
                scaled_inv = all_inv / self.inv_scale.to(all_inv.dtype)
                use_bk_inv0 = bk is not None and type(self.inv_mlp[0]) is nn.Linear and not self.inv_mlp[0]._forward_hooks and not self.inv_mlp[0]._forward_pre_hooks
                inv_h = F.linear(all_inv, bk["w_inv0"], bk["b_inv0"]) if use_bk_inv0 else self.inv_mlp[0](scaled_inv)
            hm = inv_h * (1.0 + self.inv_state_gate(state_norm(s)))
        else:
            hm = self.inv_mlp[0](torch.cat(inv, dim=-1)) + self.ang(t_c, u_new, p["o"])
        s = self.out_norm((s + self.msg_up(p["s"]) + self.inv_mlp[2](self.inv_mlp[1](hm))).to(s_dt))
        if self.env_modulated:
            ld = None if dsum is None else torch.log1p(upcast_fp32(dsum).clamp_min(0.0)).to(s_dt)
            env_wt = bk.get("env_out_wt") if bk is not None else None
            s = (s + self.swiglu(s, cond_lot_atom, ld, allow_fused=allow_fused, env_wt=env_wt)).to(s_dt)
        else:
            s = (s + self.swiglu(s, domain, allow_fused=allow_fused)).to(s_dt)
        if fused:
            v_out, t_out = fused_vec_update(self, s, vec, agg, ten, bk)
            out = (s, v_out, t_out) if (return_ten or ten is not None) else (s, v_out)
            return (*out, scaled_inv) if return_inv else out
        return post_vector_tensor_update(self, s, v_new, u_new, t_new, ten, bk, v_dt, return_ten, scaled_inv, return_inv)

    def _post_plain_first(self, s: Tensor, vec: Tensor, agg: Tensor, domain: str, ten: Tensor | None = None, return_ten: bool = False,
                          **kw):
        return self._post_impl(s, vec, agg, domain, None, return_ten=return_ten, **kw)

    def _post_plain_later(self, s: Tensor, vec: Tensor, agg: Tensor, domain: str, ten: Tensor | None = None, return_ten: bool = False,
                          **kw):
        return self._post_impl(s, vec, agg, domain, ten, return_ten=return_ten, **kw)

    def post_plain(self, s: Tensor, vec: Tensor, agg: Tensor, domain: str, ten: Tensor | None = None, return_ten: bool = False, **kw):
        return self._post_impl(s, vec, agg, domain, ten, return_ten=return_ten, **kw)

    def _resolve_aux(self, s: Tensor, eg: EdgeGraph, nc: NodeContext | None, cond_atom: Tensor | None,
                     cond_lot_atom: Tensor | None, dsum: Tensor | None) -> tuple[Tensor | None, Tensor | None]:
        dsum_eff = dsum if dsum is not None else (None if nc is None else getattr(nc, "dsum", None))
        if dsum_eff is None and (self.coord_inv_scale or self.env_modulated):
            dsum_eff = compute_dsum(eg, self.edge_cfg)
        cl_a = cond_lot_atom if cond_lot_atom is not None else (None if nc is None else getattr(nc, "cond_lot_atom", None))
        if cl_a is None and self.env_modulated:
            cl_a = cond_atom if cond_atom is not None else (None if nc is None else getattr(nc, "cond_atom", None))
            if cl_a is None and nc is not None and nc.cond is not None:
                cl_a = nc.cond.expand(s.shape[0], -1) if nc.cond.shape[0] == 1 else nc.cond.index_select(0, nc.graph)
        return cl_a, dsum_eff

    def _run_agg_and_aux(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None,
                         ten: Tensor | None, cond_atom: Tensor | None, cond_lot_atom: Tensor | None,
                         dsum: Tensor | None, return_dsum: bool) -> tuple[Tensor, Tensor | None, Tensor | None]:
        dsum_in = dsum if dsum is not None else (None if nc is None else getattr(nc, "dsum", None))
        can_csr_dsum = (
            dsum_in is None and eg.mode == "csr" and not eg.create_graph
            and "_aggregate" not in self.__dict__
            and (return_dsum or self.coord_inv_scale or self.env_modulated)
        )
        if can_csr_dsum:
            agg, dsum_eff = self.aggregate_plain(s, vec, eg, nc, ten, cond_atom=cond_atom, return_dsum=True)
            cl_a, _ = self._resolve_aux(s, eg, nc, cond_atom, cond_lot_atom, dsum_eff)
        else:
            cl_a, dsum_eff = self._resolve_aux(s, eg, nc, cond_atom, cond_lot_atom, dsum_in)
            if dsum_eff is None and return_dsum:
                dsum_eff = compute_dsum(eg, self.edge_cfg)
            agg = self.aggregate_plain(s, vec, eg, nc, ten, cond_atom=cond_atom)
        return agg, cl_a, dsum_eff

    def _core_plain_first(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, domain: str,
                          ten: Tensor | None = None, return_ten: bool = False, *, cond_atom: Tensor | None = None,
                          cond_lot_atom: Tensor | None = None, dsum: Tensor | None = None, return_inv: bool = False,
                          return_dsum: bool = False):
        agg, cl_a, dsum_eff = self._run_agg_and_aux(s, vec, eg, nc, None, cond_atom, cond_lot_atom, dsum, return_dsum)
        out = self.post_plain(s, vec, agg, domain, None, return_ten=return_ten, cond_lot_atom=cl_a, dsum=dsum_eff, return_inv=return_inv,
                              allow_fused=not eg.create_graph)
        return (*out, dsum_eff) if return_dsum else out

    def _core_plain_later(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, domain: str,
                          ten: Tensor | None = None, return_ten: bool = False, *, cond_atom: Tensor | None = None,
                          cond_lot_atom: Tensor | None = None, dsum: Tensor | None = None, return_inv: bool = False,
                          return_dsum: bool = False):
        agg, cl_a, dsum_eff = self._run_agg_and_aux(s, vec, eg, nc, ten, cond_atom, cond_lot_atom, dsum, return_dsum)
        out = self.post_plain(s, vec, agg, domain, ten, return_ten=return_ten, cond_lot_atom=cl_a, dsum=dsum_eff, return_inv=return_inv,
                              allow_fused=not eg.create_graph)
        return (*out, dsum_eff) if return_dsum else out

    def core_plain(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, domain: str,
                   ten: Tensor | None = None, return_ten: bool = False, *, cond_atom: Tensor | None = None,
                   cond_lot_atom: Tensor | None = None, dsum: Tensor | None = None, return_inv: bool = False,
                   return_dsum: bool = False):
        agg, cl_a, dsum_eff = self._run_agg_and_aux(s, vec, eg, nc, ten, cond_atom, cond_lot_atom, dsum, return_dsum)
        out = self.post_plain(s, vec, agg, domain, ten, return_ten=return_ten, cond_lot_atom=cl_a, dsum=dsum_eff, return_inv=return_inv,
                              allow_fused=not eg.create_graph)
        return (*out, dsum_eff) if return_dsum else out

    def finish_plain(self, s: Tensor, nc: NodeContext | None, domain: str) -> Tensor:
        """Group context (molecule only, identity for crystals), state_norm, atom mask. The trunk calls readouts / feedback before this."""
        s = upcast_fp32(s)
        if self.group_ctx is not None and nc is not None and nc.graph is not None:
            s = self.group_ctx(s, nc.graph, nc.atom_mask, nc.charge, nc.spin, domain)
        if self.use_state_norm:
            with fp32_guard(s.device):
                s = state_norm(upcast_fp32(s))
        return s if nc is None or nc.atom_mask is None else s * nc.atom_mask.to(s.dtype).unsqueeze(-1)

    def _use_csr_node_chunks(self, s: Tensor, eg: EdgeGraph, domain: str) -> bool:
        if s.device.type == "cuda" and self.remat == "none":
            thresh = 65536
        elif self.env_modulated and s.device.type == "mps":
            thresh = 4096
        else:
            thresh = 8192
        return (not torch.compiler.is_compiling() and eg.mode == "csr" and s.shape[0] > thresh
                and not eg.create_graph and self._inv_hook is None and not self.training and not self.u1.weight.requires_grad)

    def _use_post_node_chunks(self, s: Tensor, eg: EdgeGraph) -> bool:
        if s.device.type == "cuda" and self.remat == "none":
            thresh = 65536
        elif self.env_modulated and s.device.type == "mps":
            thresh = 2048
        else:
            thresh = 4096 if self.env_modulated else NODE_CHUNK_ATOMS
        return (not torch.compiler.is_compiling() and s.shape[0] > thresh
                and not eg.create_graph and self._inv_hook is None and not self.training and not self.u1.weight.requires_grad)

    def core(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None = None, domain: str = "molecule",
             *, ten: Tensor | None = None, return_ten: bool = False, cond_atom: Tensor | None = None,
             cond_lot_atom: Tensor | None = None, dsum: Tensor | None = None, return_inv: bool = False,
             return_dsum: bool = False):
        return stage_dispatch(self, s, vec, eg, nc, domain, False, ten=ten, return_ten=return_ten,
                              cond_atom=cond_atom, cond_lot_atom=cond_lot_atom, dsum=dsum, return_inv=return_inv,
                              return_dsum=return_dsum)

    def finish(self, s: Tensor, nc: NodeContext | None = None, domain: str = "molecule") -> Tensor:
        return self.finish_plain(s, nc, domain) if self.remat == "none" else unit_finish(self, s, nc, domain)

    def forward(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None = None, domain: str = "molecule",
                *, ten: Tensor | None = None, return_ten: bool = False, cond_atom: Tensor | None = None,
                cond_lot_atom: Tensor | None = None, dsum: Tensor | None = None, return_inv: bool = False,
                return_dsum: bool = False):
        return stage_dispatch(self, s, vec, eg, nc, domain, True, ten=ten, return_ten=return_ten,
                              cond_atom=cond_atom, cond_lot_atom=cond_lot_atom, dsum=dsum, return_inv=return_inv,
                              return_dsum=return_dsum)
