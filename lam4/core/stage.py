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

from lam4.cond.expert import DomainExperts
from lam4.config import ModelConfig
from lam4.core.angular import AngularBranch, l2_mat6, l2_matvec, l2_pair_mm
from lam4.core.edge_ops import edge_aggregate_ref, edge_aggregate_stream
from lam4.core.edge_terms import EdgeCfg, _near_mask
from lam4.core.edge_weights import EdgeHeads
from lam4.core.layout import HEADS, StageLayout, split_output
from lam4.core.radial import R_CUT
from lam4.core.spline_tables import DEFAULT_GRID, BSplineTables
from lam4.core.stage_graph import EdgeGraph, NodeContext
from lam4.core.stage_remat import check_mode, post_chunked_crystal, unit_block, unit_finish, unit_stage
from lam4.core.vector_bound import bound_vectors, mix, state_norm
from lam4.kernels.csr_ops import edge_aggregate_csr
from lam4.kernels.csr_plan import coef_to_horner_poly
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
        self.swiglu = DomainExperts(d, cfg.swiglu_ratio, cfg.domains)
        self.vec_self, self.ten_self = lin(k, k), lin(k2, k2)
        self.vec_gate_norm = nn.RMSNorm(d, eps=1e-6)
        self.vec_out_gate = nn.Linear(d, k)
        self.ten_out_gate = nn.Linear(d, k2)
        for g in (self.vec_out_gate, self.ten_out_gate):
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)
        self.group_ctx = GroupContext(d) if cfg.group_context else None
        if self.inv_norm:
            self.register_buffer("inv_scale", self._default_inv_scale(cfg))
            self.inv_state_gate = nn.Linear(d, d)
            self.tv_t, self.tv_v, self.tv_gate = lin(k2, k), lin(k, k), nn.Linear(d, k)
            self.vv_p, self.vv_q, self.vv_gate = lin(k, k2), lin(k, k2), nn.Linear(d, k2)
            for g in (self.inv_state_gate, self.tv_gate, self.vv_gate):
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)

    def _default_inv_scale(self, cfg: ModelConfig) -> Tensor:
        L = self.layout
        segs = [(cfg.r1_vec + cfg.r2_ten, 1e-3), (cfg.r3_tvv, 1e-5)]
        if L.later:
            segs.append((L.k_lon2 + L.k_v + L.k_t + L.k_lon, 1e-2))
        segs += [(cfg.ttt_mix, 1e-5), (cfg.vttv_rank, 1e-6), (cfg.k3_read, 1e-3), (cfg.k3_read, 1e-5)]
        return torch.cat([torch.full((n,), val, dtype=torch.float32) for n, val in segs])

    @torch.no_grad()
    def _init_tables(self) -> None:
        """the earlier Mena trunk's filters are bias-free nn.Linear(32, .): uniform(+-1/sqrt(32)); the 4 head-bias columns start at 0 (logit bias 0)."""
        w = self.tables.weight
        w.copy_((torch.rand_like(w) * 2.0 - 1.0) / w.shape[0] ** 0.5)
        w[:, :HEADS] = 0.0

    def _condition(self, h: Tensor, nc: NodeContext | None) -> Tensor:
        if nc is None or nc.cond is None:
            return h
        gamma, beta = self.film(nc.cond).index_select(0, nc.graph).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta

    def train(self, mode: bool = True):
        if mode:
            self._baked = None
        return super().train(mode)

    @torch.no_grad()
    def bake_for_inference(self) -> None:
        self.ang.bake_for_inference()
        if self.group_ctx is not None:
            self.group_ctx.bake_for_inference()
        if self.edge_cfg.r_cut_near > 0.0:
            _near_mask(self.layout, self.edge_cfg, self.u1.weight)
        hd = self.heads.q_proj.out_features
        w_tab = self.tables.weight
        coef64 = self.tables._p64 @ w_tab.detach().cpu().to(torch.float64)
        bk = {
            "w_gqk": torch.cat([self.gate_proj.weight, self.heads.q_proj.weight, self.heads.k_proj.weight], 0).detach().contiguous(),
            "b_gqk": F.pad(self.gate_proj.bias, (0, 2 * hd)).detach().contiguous(),
            "gain": self.heads.gain().detach().contiguous(),
            "w_uvpq": torch.cat([self.u1.weight, self.v1.weight, self.p3.weight, self.q3.weight], 0).detach().contiguous(),
            "w_uvt": torch.cat([self.u2.weight, self.v2.weight, self.t3.weight], 0).detach().contiguous(),
            "poly": coef_to_horner_poly(coef64, self.edge_cfg.r_cut, precise=True).to(dtype=w_tab.dtype, device=w_tab.device).contiguous(),
        }
        if self.inv_norm:
            w0 = self.inv_mlp[0].weight
            bk["w_inv0"] = (w0 / self.inv_scale.to(w0.dtype).unsqueeze(0)).detach().contiguous()
            bk["b_inv0"] = self.inv_mlp[0].bias.detach().contiguous()
            bk["w_gates4"] = torch.cat([self.vec_out_gate.weight, self.tv_gate.weight, self.ten_out_gate.weight, self.vv_gate.weight], 0).detach().contiguous()
            bk["b_gates4"] = torch.cat([self.vec_out_gate.bias, self.tv_gate.bias, self.ten_out_gate.bias, self.vv_gate.bias], 0).detach().contiguous()
            bk["w_u_out"] = torch.cat([self.vec_self.weight, self.tv_v.weight, self.vv_p.weight, self.vv_q.weight], 0).detach().contiguous()
            bk["w_tb"] = torch.cat([self.ten_self.weight, self.tv_t.weight], 0).detach().contiguous()
        self._baked = bk

    def unbake_inference(self) -> None:
        self._baked = None
        self.ang._baked = None
        if self.group_ctx is not None:
            self.group_ctx.unbake_inference()

    def _get_baked(self, like: Tensor) -> dict[str, Tensor] | None:
        if self._baked is None or self.training or self.u1.weight.requires_grad:
            return None
        if torch.compiler.is_compiling():
            return None if like.device.type == "cpu" else self._baked
        if self._baked["w_uvpq"].device != like.device or self._baked["w_uvpq"].dtype != like.dtype:
            self.bake_for_inference()
        return self._baked

    def _aggregate(self, eg: EdgeGraph, coef: Tensor, gates: Tensor, q: Tensor | None, k: Tensor | None, b: Tensor | None,
                   tau: Tensor | None = None) -> Tensor:
        bk = self._get_baked(gates)
        gain = None if q is None else (bk["gain"] if bk is not None else self.heads.gain())
        kw = dict(layout=self.layout, vec=b, ten=tau, rho=eg.rho, edge_scale=eg.edge_scale, cfg=self.edge_cfg, z=eg.z, qk_gain=gain)
        if eg.mode == "csr":
            return edge_aggregate_csr(
                eg.pos, eg.recv, eg.send, eg.shift if eg.shift.numel() > 0 else None, coef, gates, q, k,
                csr_plan=eg.csr_plan, poly=bk["poly"] if bk is not None else None,
                edge_shift=eg.edge_shift, cell=eg.cell, graph=eg.graph, create_graph=eg.create_graph, **kw,
            )
        if eg.mode == "stream":
            return edge_aggregate_stream(eg.pos, eg.recv, eg.send, eg.shift, coef, gates, q, k, chunks=eg.chunks, **kw)
        return edge_aggregate_ref(eg.pos, eg.recv, eg.send, eg.shift, coef, gates, q, k, geom=eg.geom, phi=eg.phi, **kw)

    def prep(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, ten: Tensor | None = None) -> tuple:
        """Steps 1-3: (coef, gates, q, k, b, tau), the node-side inputs of the edge op."""
        h = self._condition(self.norm_in(s), nc)
        bk = self._get_baked(h)
        if bk is not None and eg.edge_scale is None:
            hd = self.heads.q_proj.out_features
            gates, q, k = F.linear(h, bk["w_gqk"], bk["b_gqk"]).split([self.layout.gate_dim, hd, hd], dim=-1)
        else:
            gates = self.gate_proj(h)
            q, k = (None, None) if eg.edge_scale is not None else self.heads(h)
        b = mix(bound_vectors(vec, self.vec_c), self.transport_mix) if self.layout.later else None
        tau = mix(bound_vectors(ten, self.vec_c), self.transport_ten) if (self.layout.later and ten is not None) else None
        tab = self.tables.weight if (self.tables.grid == DEFAULT_GRID and self.tables.r_cut == R_CUT) else self.tables()
        return tab, gates, q, k, b, tau

    def aggregate_plain(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, ten: Tensor | None = None) -> Tensor:
        return self._aggregate(eg, *self.prep(s, vec, eg, nc, ten))

    def post_plain(self, s: Tensor, vec: Tensor, agg: Tensor, domain: str, ten: Tensor | None = None, return_ten: bool = False):
        """Steps 5-9 from the (finalised) edge output agg [N, out_dim]."""
        L = self.layout
        bk = self._get_baked(s)
        p = split_output(agg, L)
        v_new = vec + p["v"]
        u_new = bound_vectors(v_new, self.vec_c)
        t = p["t"]
        t_new = t if ten is None else ten + t
        w_uvpq = bk["w_uvpq"] if bk is not None else torch.cat([self.u1.weight, self.v1.weight, self.p3.weight, self.q3.weight], 0)
        w_uvt = bk["w_uvt"] if bk is not None else torch.cat([self.u2.weight, self.v2.weight, self.t3.weight], 0)
        u1, v1, p3, q3 = F.linear(u_new.transpose(-1, -2), w_uvpq).transpose(-1, -2).split([self.u1.out_features] * 2 + [self.p3.out_features] * 2, dim=-2)
        u2, v2, t3 = F.linear(t.transpose(-1, -2), w_uvt).transpose(-1, -2).split([self.u2.out_features] * 2 + [self.t3.out_features], dim=-2)
        inv = [(u1 * v1).sum(-1), (u2 * v2).sum(-1), (t3 * l2_pair_mm(p3, q3)).sum(-1)]
        if L.later:
            cross_v = (mix(bound_vectors(vec, self.vec_c), self.u_cv) * mix(bound_vectors(p["v"], self.vec_c), self.v_cv)).sum(-1)
            cross_t = cross_v.new_zeros(vec.shape[0], L.k_t) if ten is None else (
                mix(bound_vectors(ten, self.vec_c), self.u_ct) * mix(bound_vectors(t, self.vec_c), self.v_ct)).sum(-1)
            inv += [p["lon2"], cross_v, cross_t, p["lon"]]
        if self.inv_norm:
            all_inv = torch.cat([*inv, self.ang.features(t, u_new, p["o"])], dim=-1)
            if self._inv_hook is not None:
                self._inv_hook(all_inv)
            inv_h = F.linear(all_inv, bk["w_inv0"], bk["b_inv0"]) if bk is not None else self.inv_mlp[0](all_inv / self.inv_scale.to(all_inv.dtype))
            hm = inv_h * (1.0 + self.inv_state_gate(state_norm(s)))
        else:
            hm = self.inv_mlp[0](torch.cat(inv, dim=-1)) + self.ang(t, u_new, p["o"])
        s = self.out_norm(s + self.msg_up(p["s"]) + self.inv_mlp[2](self.inv_mlp[1](hm)))
        s = s + self.swiglu(s, domain)
        s_gn = self.vec_gate_norm(s)
        if bk is not None and self.inv_norm:
            kv, kt2 = L.k_v, L.k_t
            gv, gtv, gt, gvv = torch.tanh(F.linear(s_gn, bk["w_gates4"], bk["b_gates4"])).split([kv, kv, kt2, kt2], dim=-1)
            tb = bound_vectors(t_new, self.vec_c)
            if not return_ten and ten is None:
                u_self, u_tvv = F.linear(u_new.transpose(-1, -2), bk["w_u_out"][:2 * kv]).transpose(-1, -2).split([kv, kv], dim=-2)
                tv_vec = l2_matvec(l2_mat6(mix(tb, self.tv_t)), u_tvv)
                return s, v_new + u_self * gv.unsqueeze(-1) + tv_vec * gtv.unsqueeze(-1)
            u_self, u_tvv, u_vvp, u_vvq = F.linear(u_new.transpose(-1, -2), bk["w_u_out"]).transpose(-1, -2).split([kv, kv, kt2, kt2], dim=-2)
            t_self, t_tvt = F.linear(tb.transpose(-1, -2), bk["w_tb"]).transpose(-1, -2).split([kt2, kv], dim=-2)
            tv_vec = l2_matvec(l2_mat6(t_tvt), u_tvv)
            v_out = v_new + u_self * gv.unsqueeze(-1) + tv_vec * gtv.unsqueeze(-1)
            t_out = t_new + t_self * gt.unsqueeze(-1) + l2_pair_mm(u_vvp, u_vvq) * gvv.unsqueeze(-1)
            return s, v_out, t_out
        v_out = v_new + mix(u_new, self.vec_self) * torch.tanh(self.vec_out_gate(s_gn)).unsqueeze(-1)
        if self.inv_norm:
            tv_vec = l2_matvec(l2_mat6(mix(bound_vectors(t_new, self.vec_c), self.tv_t)), mix(u_new, self.tv_v))
            v_out = v_out + tv_vec * torch.tanh(self.tv_gate(s_gn)).unsqueeze(-1)
        if not return_ten and ten is None:
            return s, v_out
        t_out = t_new + mix(bound_vectors(t_new, self.vec_c), self.ten_self) * torch.tanh(self.ten_out_gate(s_gn)).unsqueeze(-1)
        if self.inv_norm:
            vv_ten = l2_pair_mm(mix(u_new, self.vv_p), mix(u_new, self.vv_q))
            t_out = t_out + vv_ten * torch.tanh(self.vv_gate(s_gn)).unsqueeze(-1)
        return s, v_out, t_out

    def core_plain(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, domain: str,
                   ten: Tensor | None = None, return_ten: bool = False):
        return self.post_plain(s, vec, self.aggregate_plain(s, vec, eg, nc, ten), domain, ten, return_ten=return_ten)

    def finish_plain(self, s: Tensor, nc: NodeContext | None, domain: str) -> Tensor:
        """Group context (molecule only, identity for crystals), state_norm, atom mask. The trunk calls readouts / feedback before this."""
        if self.group_ctx is not None and nc is not None and nc.graph is not None:
            s = self.group_ctx(s, nc.graph, nc.atom_mask, nc.charge, nc.spin, domain)
        if self.use_state_norm:
            s = state_norm(s)
        return s if nc is None or nc.atom_mask is None else s * nc.atom_mask.to(s.dtype).unsqueeze(-1)

    def _use_csr_node_chunks(self, s: Tensor, eg: EdgeGraph, domain: str) -> bool:
        return (
            eg.mode == "csr" and s.shape[0] > 8192 and domain == "crystal"
            and not eg.create_graph and self._inv_hook is None
            and not torch.compiler.is_compiling() and not self.training and not self.u1.weight.requires_grad
        )

    def core(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None = None, domain: str = "molecule",
             *, ten: Tensor | None = None, return_ten: bool = False):
        if self.remat == "none":
            if self._use_csr_node_chunks(s, eg, domain):
                agg = self.aggregate_plain(s, vec, eg, nc, ten)
                return post_chunked_crystal(self, s, vec, (agg,), nc, domain, False, ten, return_ten=return_ten)
            return self.core_plain(s, vec, eg, nc, domain, ten, return_ten=return_ten)
        return (unit_stage if self.remat == "stage" else unit_block)(self, s, vec, eg, nc, domain, False, ten, return_ten=return_ten)

    def finish(self, s: Tensor, nc: NodeContext | None = None, domain: str = "molecule") -> Tensor:
        return self.finish_plain(s, nc, domain) if self.remat == "none" else unit_finish(self, s, nc, domain)

    def forward(self, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None = None, domain: str = "molecule",
                *, ten: Tensor | None = None, return_ten: bool = False):
        if self.remat == "none":
            if self._use_csr_node_chunks(s, eg, domain):
                agg = self.aggregate_plain(s, vec, eg, nc, ten)
                return post_chunked_crystal(self, s, vec, (agg,), nc, domain, True, ten, return_ten=return_ten)
            out = self.core_plain(s, vec, eg, nc, domain, ten, return_ten=return_ten)
            return (self.finish_plain(out[0], nc, domain), *out[1:])
        return (unit_stage if self.remat == "stage" else unit_block)(self, s, vec, eg, nc, domain, True, ten, return_ten=return_ten)
