"""readout.py: the molecular energy readout of lam4: stage readouts + the final 3-layer SiLU MLP, plus two lam4 additions.

the earlier Mena model's forward pass, reproduced term by term:
    stage l < S-1:   atomic += stage_readouts[l](RMSNorm(s_l))                 s_l = the stage OUTPUT (core), BEFORE feedback / group context / state_norm
    final:           u = bound_vectors(v, vec_c);  s_ro = RMSNorm(s)           s = the state AFTER feedback, group context and state_norm (a second, idempotent-ish rms)
                     x = [s_ro, <U u, V u>]  (cat on the last axis, [N, d + K], <U u, V u> = (mix(u, read_u) * mix(u, read_v)).sum(-1) over xyz)
                     ro = Linear(d/2, 1)( SiLU( Linear(d, d/2)( SiLU( Linear(d + K, d)(x) ) ) ) )
    atomic = (atomic + ro) * atom_mask;   E_net[g] = sum over the atoms of g
Surprises replicated: stage readouts are zero-init (weight AND bias) and read the PRE-feedback state, the final readout is NOT zero-init (last weight N(0, 0.02), bias 0, the
other Linear layers keep torch's default init), the pair invariant uses TWO bias-free K x K mixes, and the mask multiplies the SUM (pad atoms give exactly 0 energy and 0 gradient).
pair_read (omol4c) is not part of this module (off in run 1; it was a separate zero-init head).

lam4 additions (both zero-init, so a fresh Readout is the earlier molecular readout bit for bit):
  ref_id    `ref_emb` [n_ref, d] zero-init, contracted with a deterministic per-element cosine basis `_z_basis` [N_Z, d] (pad row Z=0 is 0) and added AFTER the final SiLU MLP:
            a position-independent per-(ref_id, Z) reference offset `<ref_emb[ref_id_g], _z_basis[Z_i]>` that absorbs dataset reference conventions (MPtrj PAW-PBE vs OMat24 PBE_54)
            without entering the SiLU nonlinearities, so forces and harmonic Hessians are 100% invariant to `ref_id` (fixes earlier models' `lot_readout_proj` stiffness softening bug).
  iso       isolated-atom reference e_iso(Z, c) = a[Z] + <B[Z], c_g>, c_g = lam4.cond.Conditioning.functional(token, u) of the atom's graph (functional and +U ONLY),
            a [119] and B [119, cond_dim] zero-init. Per-atom, so periodic crystals stay strictly extensive. Q and S are NOT inputs: a per-atom term
            conditioned on the graph charge sums to n_g * f(Q, S) and gives every charged / open-shell molecule an offset linear in its size, the shape of the
            N > 192, |Q| >= 2 energy bias (spec section 5); the intensive charge / spin energy belongs to E_Q / E_S, the group context and the NQE charges.

API (everything is [N] per atom unless noted; s [N, d], vec [N, K, 3] raw additive vector stream, b a GraphBatch, cond [G, cond_dim] = Conditioning.functional or None)
    Readout(cfg, n_ref=N_REF, iso_reference=True)
    .stage(l, s, atom_mask=None, normed=False) -> [N]                  stage readout l (0 <= l < n_stages - 1); normed=True if the caller already took RMSNorm(s)
    .final(s, vec, b, cond=None, stage_terms=None) -> (atomic, terms)   atomic = sum of all terms * atom_mask; terms: {"stage0": ..., "final": ..., "iso": ...} (each masked)
    .graph_energy(atomic, b) -> [G]                                     index_add over atoms
Key map from an earlier-generation state dict (names are identical): stage_readouts.{l}.{weight,bias}, read_u.weight, read_v.weight, readout.{0,2,4}.{weight,bias}
-> the same keys here. NOT in earlier Mena (stay at their zero init after a graft): ref_emb.weight, iso_a.weight, iso_b.weight.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.config import ModelConfig
from lam4.core.consts import tracing
from lam4.core.radial import fp32_guard, resolve_baked, upcast_fp32
from lam4.core.remat import NODE_CHUNK_ATOMS, RematSpec, can_chunk_nodes, chunked_node_apply, node_chunk_size, remat_apply
from lam4.core.vector_bound import bound_vectors, chan_linear, dot_last, mix, state_norm
from lam4.data.batch import GraphBatch
from lam4.heads.iso_calib import apply_iso_atom_table, iso_smoothstep
from lam4.registry.sources import _OVERLAY

N_REF = len(_OVERLAY)   # one reference id per registry source (12 today)
N_Z = 119               # Z = 0 (pad) .. 118


def pair_inv(u: Tensor, read_u: nn.Linear, read_v: nn.Linear) -> Tensor:
    """<U u, V u> over the xyz axis: [N, K, 3] -> [N, K] (earlier Mena mena_ace._pair_inv)."""
    return dot_last(mix(u, read_u), mix(u, read_v))


class DirectInvBypass(nn.Module):
    """Zero-init low-rank bypass mapping Stage-2 scaled invariants directly into Readout pre-activation `h`."""

    def __init__(self, inv_dim: int, rank: int, d_scalar: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(inv_dim), int(rank), bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(int(rank), int(d_scalar), bias=False)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, inv2: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(inv2)))


class SpinDescriptorHead(nn.Module):
    """Two-channel spin descriptor: symmetric unpaired density `u_rad_i >= 0` and signed spin bias `m_raw_i`."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = max(16, int(d_model) // 2)
        self.trunk = nn.Sequential(nn.Linear(int(d_model), hidden), nn.SiLU())
        self.u_rad_out = nn.Linear(hidden, 1)
        self.m_out = nn.Linear(hidden, 1)
        for out in (self.u_rad_out, self.m_out):
            nn.init.normal_(out.weight, std=0.01)
            nn.init.zeros_(out.bias)

    def _forward_plain(self, s: Tensor, atom_mask: Tensor | None = None, *, normed: bool = False) -> tuple[Tensor, Tensor]:
        h = s if normed else state_norm(upcast_fp32(s))
        z = self.trunk(h)
        u_rad = F.softplus(self.u_rad_out(z).squeeze(-1))
        m_raw = self.m_out(z).squeeze(-1)
        if atom_mask is not None:
            mf = atom_mask.to(u_rad.dtype)
            u_rad, m_raw = u_rad * mf, m_raw * mf
        return u_rad, m_raw

    def forward(self, s: Tensor, atom_mask: Tensor | None = None, *, normed: bool = False) -> tuple[Tensor, Tensor]:
        ps = tuple(self.parameters())
        cn = node_chunk_size(s.device) if s.device.type == "mps" else NODE_CHUNK_ATOMS
        if not self.training and can_chunk_nodes(s.shape[0], ps, min_atoms=cn):
            sliced = (s,) if atom_mask is None else (s, atom_mask)
            return chunked_node_apply(
                lambda _a0, _a1, *xs: self._forward_plain(xs[0], xs[1] if len(xs) > 1 else None, normed=normed),
                sliced, params=ps, chunk_atoms=cn,
            )
        return self._forward_plain(s, atom_mask, normed=normed)


class Readout(nn.Module):
    def __init__(self, cfg: ModelConfig, n_ref: int = N_REF, iso_reference: bool = True) -> None:
        super().__init__()
        d, k, k2 = cfg.d_scalar, cfg.k_vec, cfg.k2
        self.d, self.vec_c, self.n_stages, self.n_ref = d, cfg.vec_c, cfg.n_stages, int(n_ref)
        self.factorized = bool(cfg.factorized_lot)
        self.iso_atom_n1 = bool(cfg.iso_atom_n1)
        self.iso_d_lo, self.iso_d_hi = float(cfg.iso_atom_d_lo), float(cfg.iso_atom_d_hi)
        self.n_xc_tokens, self.n_paw_bases, self.s_max = int(cfg.n_xc_tokens), int(cfg.n_paw_bases), int(cfg.s_max)
        self.stage_readouts = nn.ModuleList([nn.Linear(d, 1) for _ in range(cfg.n_stages - 1)])
        for lin in self.stage_readouts:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.read_u, self.read_v = nn.Linear(k, k, bias=False), nn.Linear(k, k, bias=False)
        self.read_tu, self.read_tv = nn.Linear(k2, k2, bias=False), nn.Linear(k2, k2, bias=False)
        self.read_t_proj = nn.Linear(k2, d, bias=False)
        self.readout = nn.Sequential(nn.Linear(d + k, d), nn.SiLU(), nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))
        nn.init.normal_(self.readout[-1].weight, std=0.02)
        nn.init.zeros_(self.readout[-1].bias)
        ang_dim = cfg.ttt_mix + cfg.vttv_rank + 2 * cfg.k3_read
        later_cross = (2 * k + 2 * k2) if cfg.n_stages >= 2 else 0
        inv_dim = cfg.r1_vec + cfg.r2_ten + cfg.r3_tvv + later_cross + (ang_dim if cfg.inv_norm else 0)
        self.inv_bypass = DirectInvBypass(inv_dim, cfg.direct_inv_rank, d) if cfg.direct_inv_bypass else None
        self.ro_film = nn.Linear(cfg.cond_dim, 2 * d) if self.factorized else None
        if self.ro_film is not None:
            nn.init.zeros_(self.ro_film.weight)
            nn.init.zeros_(self.ro_film.bias)
        self.ref_emb = nn.Embedding(self.n_ref, d)
        nn.init.zeros_(self.ref_emb.weight)
        if cfg.inv_norm:
            self.ref_emb.weight.requires_grad_(False)
        z_idx = torch.arange(N_Z, dtype=torch.float32).unsqueeze(1)
        k_idx = torch.arange(d, dtype=torch.float32).unsqueeze(0)
        z_basis = (2.0 / d) ** 0.5 * torch.cos(math.pi * (z_idx + 0.5) * (k_idx + 0.5) / N_Z)
        z_basis[0] = 0.0
        self.register_buffer("_z_basis", z_basis, persistent=False)
        self.iso_a = nn.Embedding(N_Z, 1) if iso_reference else None
        self.iso_b = nn.Embedding(N_Z, cfg.cond_dim) if iso_reference else None
        for emb in (self.iso_a, self.iso_b):
            if emb is not None:
                nn.init.zeros_(emb.weight)
        if self.iso_atom_n1:
            self.register_buffer(
                "iso_atom_delta",
                torch.zeros(self.n_xc_tokens * self.n_paw_bases * 2, N_Z, self.s_max, dtype=torch.float32),
            )
        else:
            self.iso_atom_delta = None
        self._baked: dict[str, Tensor | bool] | None = None
        self._cond_ref: list[nn.Module] = []
        self.chunk_nodes: int = 0

    @torch.no_grad()
    def load_iso_atom_delta(self, table) -> Tensor:
        """Load spin-matched N=1 exact-minus-LSQ deltas `[n_cells, N_Z, s_max]` or dict into `iso_atom_delta`."""
        if self.iso_atom_delta is None:
            raise RuntimeError("iso_atom_delta buffer is disabled (set ModelConfig.iso_atom_n1=True)")
        return apply_iso_atom_table(
            self.iso_atom_delta, table, n_xc=self.n_xc_tokens, n_paw=self.n_paw_bases, s_max=self.s_max,
        )

    def train(self, mode: bool = True):
        if mode:
            self._baked = None
        return super().train(mode)

    def _apply(self, fn):
        had_baked = self._baked is not None
        self._baked = None
        out = super()._apply(fn)
        if had_baked and not self.training and not self.read_u.weight.is_meta and not tracing():
            self.bake_for_inference()
        return out

    @torch.no_grad()
    def bake_for_inference(self, conditioning: nn.Module | None = None) -> None:
        if conditioning is not None:
            self._cond_ref = [conditioning]
        elif self._cond_ref:
            conditioning = self._cond_ref[0]
        w_ref = self.ref_emb.weight
        ref_zero = bool(w_ref.abs().max() == 0)
        bk: dict[str, Tensor | bool] = {
            "w_ruv": torch.cat([self.read_u.weight, self.read_v.weight], 0).detach().contiguous(),
            "w_rtuv": torch.cat([self.read_tu.weight, self.read_tv.weight], 0).detach().contiguous(),
            "ref_zero": ref_zero,
            "iso_atom_zero": bool(self.iso_atom_delta is None or self.iso_atom_delta.abs().max() == 0),
        }
        if not ref_zero:
            bk["ref_tab"] = (w_ref @ self._z_basis.to(device=w_ref.device, dtype=w_ref.dtype).t()).detach().contiguous()
        if self.iso_a is not None and conditioning is not None and not getattr(conditioning, "factorized", False) and not conditioning.emb_token.weight.is_meta:  # type: ignore[attr-defined]
            n_tok = conditioning.emb_token.num_embeddings  # type: ignore[attr-defined]
            c_dev = conditioning.emb_token.weight.device  # type: ignore[attr-defined]
            dev, dt = self.iso_a.weight.device, self.iso_a.weight.dtype
            toks = torch.arange(n_tok, device=c_dev).repeat_interleave(2)
            us = torch.tensor([0, 1] * n_tok, device=c_dev)
            c_all = conditioning.functional(toks, us).to(device=dev, dtype=dt)
            bk["iso_tab"] = (self.iso_a.weight.squeeze(-1).unsqueeze(0) + c_all @ self.iso_b.weight.t()).view(n_tok, 2, N_Z).detach().contiguous()
        self._baked = bk

    def unbake_inference(self) -> None:
        self._baked = None

    def _get_baked(self, like: Tensor) -> dict[str, Tensor | bool] | None:
        return resolve_baked(self, lambda bk: bk["w_ruv"], self.read_u.weight, like)

    def _ro_mlp(self, s: Tensor, vec: Tensor, ten: Tensor | None, inv2: Tensor | None = None,
                cond_ro: Tensor | None = None, *, return_phi: bool = False):
        s, vec = upcast_fp32(s), upcast_fp32(vec)
        ten = None if ten is None else upcast_fp32(ten)
        bk = self._get_baked(s)
        with fp32_guard(s.device):
            u = bound_vectors(vec, self.vec_c)
            if bk is not None:
                ru, rv = chan_linear(u, bk["w_ruv"]).chunk(2, dim=-2)  # type: ignore[arg-type]
                pinv = dot_last(ru, rv)
            else:
                pinv = pair_inv(u, self.read_u, self.read_v)
            s_n = state_norm(s)
            tinv = None
            if ten is not None:
                tb = bound_vectors(ten, self.vec_c)
                if bk is not None:
                    tu, tv = chan_linear(tb, bk["w_rtuv"]).chunk(2, dim=-2)  # type: ignore[arg-type]
                    tinv = dot_last(tu, tv)
                else:
                    tinv = pair_inv(tb, self.read_tu, self.read_tv)
        h = self.readout[0](torch.cat([s_n, pinv], dim=-1))
        if tinv is not None:
            h = h + self.read_t_proj(tinv)
        if self.inv_bypass is not None and inv2 is not None:
            h = h + self.inv_bypass(upcast_fp32(inv2))
        if self.ro_film is not None and cond_ro is not None:
            gamma, beta = self.ro_film(upcast_fp32(cond_ro)).chunk(2, dim=-1)
            h = h * (1.0 + gamma) + beta
        phi = self.readout[3](self.readout[2](self.readout[1](h)))
        out = upcast_fp32(self.readout[4](phi).squeeze(-1)).to(s.dtype)
        return (out, phi.to(s.dtype)) if return_phi else out

    @property
    def term_names(self) -> tuple[str, ...]:
        bk = self._baked
        use_bk = bk is not None and not self.training and not self.read_u.weight.requires_grad
        has_iso_atom = bool(self.iso_atom_n1 and not (use_bk and bk.get("iso_atom_zero", False)))
        base = tuple(f"stage{l}" for l in range(self.n_stages - 1)) + ("final",) + (("iso",) if self.iso_a is not None else ())
        return base + (("iso_atom",) if has_iso_atom else ())

    def stage(self, l: int, s: Tensor, atom_mask: Tensor | None = None, normed: bool = False) -> Tensor:
        s = upcast_fp32(s)
        lin = self.stage_readouts[l]
        cn = node_chunk_size(s.device) if s.device.type == "mps" else NODE_CHUNK_ATOMS
        if not normed and can_chunk_nodes(s.shape[0], tuple(lin.parameters()), min_atoms=cn):
            sliced = (s,) if atom_mask is None else (s, atom_mask)

            def _fn(_a0: int, _a1: int, *xs):
                with fp32_guard(xs[0].device):
                    sn = state_norm(xs[0])
                e = upcast_fp32(lin(sn).squeeze(-1)).to(xs[0].dtype)
                return e if len(xs) == 1 else e * xs[1].to(e.dtype)

            return chunked_node_apply(_fn, sliced, params=tuple(lin.parameters()), chunk_atoms=cn)[0]
        if normed:
            sn = s
        else:
            with fp32_guard(s.device):
                sn = state_norm(s)
        e = upcast_fp32(lin(sn).squeeze(-1)).to(s.dtype)
        return e if atom_mask is None else e * atom_mask.to(e.dtype)

    def final(self, s: Tensor, vec: Tensor, b: GraphBatch, cond: Tensor | None = None,
              stage_terms: dict[str, Tensor] | None = None, ten: Tensor | None = None,
              *, inv2: Tensor | None = None, cond_atom: Tensor | None = None,
              cond_ro: Tensor | None = None, dsum: Tensor | None = None, return_phi: bool = False):
        s, vec = upcast_fp32(s), upcast_fp32(vec)
        ten = None if ten is None else upcast_fp32(ten)
        mf = b.atom_mask.to(s.dtype)
        z = b.z.clamp(0, N_Z - 1)
        cro = cond_ro if cond_ro is not None else cond_atom
        if cro is None and self.ro_film is not None and cond is not None:
            cro = cond.expand(s.shape[0], -1) if cond.shape[0] == 1 else cond.index_select(0, b.graph)
        ps = tuple(self.parameters())
        cn_def = node_chunk_size(s.device) if s.device.type == "mps" else NODE_CHUNK_ATOMS
        cn = self.chunk_nodes or (cn_def if can_chunk_nodes(s.shape[0], ps, min_atoms=cn_def) else 0)
        if cn > 0 and s.shape[0] > cn and not torch.compiler.is_compiling():
            flags = (ten is not None, inv2 is not None, cro is not None)
            sliced = [s, vec] + ([ten] if flags[0] else []) + ([inv2] if flags[1] else []) + ([cro] if flags[2] else [])

            def _fn(_a0: int, _a1: int, *xs):
                it = iter(xs)
                s_, v_ = next(it), next(it)
                t_ = next(it) if flags[0] else None
                i_ = next(it) if flags[1] else None
                c_ = next(it) if flags[2] else None
                return self._ro_mlp(s_, v_, t_, i_, c_, return_phi=return_phi)

            res = chunked_node_apply(_fn, tuple(sliced), params=ps, chunk_atoms=cn)
            ro, phi = (res[0], res[1]) if return_phi else (res[0], None)
        else:
            res = self._ro_mlp(s, vec, ten, inv2, cro, return_phi=return_phi)
            ro, phi = res if return_phi else (res, None)
        with fp32_guard(s.device):
            bk = self._get_baked(s)
            if bk is not None and bk["ref_zero"]:
                fin = ro * mf
            elif bk is not None and "ref_tab" in bk:
                rt = bk["ref_tab"]  # type: ignore[assignment]
                rid_m = (b.ref_id >= 0) & (b.ref_id < rt.shape[0])
                rid_c = b.ref_id.clamp(0, rt.shape[0] - 1)
                ref_off = (rt.index_select(0, rid_c[:1]).squeeze(0).index_select(0, z) * rid_m[:1].to(s.dtype) if b.n_graphs == 1
                           else rt.reshape(-1).index_select(0, rid_c.index_select(0, b.graph) * N_Z + z) * rid_m.index_select(0, b.graph).to(s.dtype))
                fin = (ro + ref_off) * mf
            else:
                n_ref_emb = self.ref_emb.num_embeddings
                rid_m = ((b.ref_id >= 0) & (b.ref_id < n_ref_emb)).to(s.dtype)
                rid_c = b.ref_id.clamp(0, n_ref_emb - 1)
                ref_off = ((upcast_fp32(self.ref_emb(rid_c)) * rid_m.unsqueeze(-1)).index_select(0, b.graph) * self._z_basis.to(s.dtype).index_select(0, z)).sum(-1)
                fin = (ro + ref_off.to(s.dtype)) * mf
            terms = {k: upcast_fp32(v).to(s.dtype) * mf for k, v in (stage_terms or {}).items()}
            terms["final"] = fin
            if self.iso_a is not None:
                if cond_atom is None and cond is None:
                    raise ValueError("iso_reference needs the per-graph functional conditioning cond [G, cond_dim] (pass Conditioning.functional(token, u), "
                                     "or build Readout(iso_reference=False))")
                if cond_atom is not None:
                    cond_f = upcast_fp32(cond_atom).to(s.dtype)
                    iso_e = upcast_fp32(self.iso_a(z)).squeeze(-1).to(s.dtype) + (upcast_fp32(self.iso_b(z)).to(s.dtype) * cond_f).sum(-1)
                    terms["iso"] = iso_e * mf
                elif bk is not None and "iso_tab" in bk:
                    it = bk["iso_tab"]  # type: ignore[assignment]
                    tu = b.token * 2 + b.u_flag.clamp(0, 1)
                    iso_e = (it.reshape(-1, N_Z).index_select(0, tu[:1]).squeeze(0).index_select(0, z) if b.n_graphs == 1
                             else it.reshape(-1).index_select(0, tu.index_select(0, b.graph) * N_Z + z))
                    terms["iso"] = iso_e * mf
                else:
                    cond_f = upcast_fp32(cond).to(s.dtype)
                    iso_e = upcast_fp32(self.iso_a(z)).squeeze(-1).to(s.dtype) + (upcast_fp32(self.iso_b(z)).to(s.dtype) * cond_f.index_select(0, b.graph)).sum(-1)
                    terms["iso"] = iso_e * mf
            if self.iso_atom_delta is not None and dsum is not None and not (bk is not None and bk.get("iso_atom_zero", False)):
                xc_g = (getattr(b, "xc_id", None) if self.factorized else None)
                xc_g = b.token if xc_g is None else xc_g
                paw_g = getattr(b, "paw_id", None) if self.factorized else None
                u_a = getattr(b, "u_atom", None) if self.factorized else None
                if u_a is None:
                    u_a = b.u_flag.index_select(0, b.graph)
                paw_a = torch.zeros_like(u_a) if paw_g is None else paw_g.index_select(0, b.graph)
                cell_a = (xc_g.index_select(0, b.graph).clamp(0, self.n_xc_tokens - 1) * self.n_paw_bases + paw_a.clamp(0, self.n_paw_bases - 1)) * 2 + u_a.long().clamp(0, 1)
                spin_a = b.frag_s.round().long().clamp(1, self.s_max).sub(1).index_select(0, b.frag_group)
                flat_idx = (cell_a * N_Z + z) * self.s_max + spin_a
                delta_e = self.iso_atom_delta.reshape(-1).index_select(0, flat_idx).to(s.dtype)
                sw = (1.0 - iso_smoothstep(dsum, self.iso_d_lo, self.iso_d_hi)).to(s.dtype)
                terms["iso_atom"] = delta_e * sw * mf
            atomic = sum(terms.values())
        return (atomic, terms, phi * mf.unsqueeze(-1)) if return_phi else (atomic, terms)

    @staticmethod
    def graph_energy_f64(atomic: Tensor, b: GraphBatch) -> Tensor:
        """Per-graph energy sum in float64 on CPU/CUDA (float32 on MPS which lacks float64)."""
        with fp32_guard(atomic.device):
            out_dt = torch.float32 if atomic.dtype in (torch.float16, torch.bfloat16) else atomic.dtype
            acc_dt = torch.float64 if atomic.device.type != "mps" else out_dt
            a = atomic.to(acc_dt) if atomic.dtype != acc_dt else atomic
            return a.sum(0, keepdim=True) if b.n_graphs == 1 else a.new_zeros(b.n_graphs).index_add(0, b.graph, a)

    @staticmethod
    def graph_energy(atomic: Tensor, b: GraphBatch) -> Tensor:
        with fp32_guard(atomic.device):
            out_dt = torch.float32 if atomic.dtype in (torch.float16, torch.bfloat16) else atomic.dtype
            out = Readout.graph_energy_f64(atomic, b)
            return out.to(out_dt) if out.dtype != out_dt else out
