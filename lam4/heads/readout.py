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
from lam4.core.remat import RematSpec, remat_apply
from lam4.core.vector_bound import bound_vectors, mix, state_norm
from lam4.data.batch import GraphBatch
from lam4.registry.sources import _OVERLAY

N_REF = len(_OVERLAY)   # one reference id per registry source (12 today)
N_Z = 119               # Z = 0 (pad) .. 118


def pair_inv(u: Tensor, read_u: nn.Linear, read_v: nn.Linear) -> Tensor:
    """<U u, V u> over the xyz axis: [N, K, 3] -> [N, K] (earlier Mena mena_ace._pair_inv)."""
    return (mix(u, read_u) * mix(u, read_v)).sum(-1)


class Readout(nn.Module):
    def __init__(self, cfg: ModelConfig, n_ref: int = N_REF, iso_reference: bool = True) -> None:
        super().__init__()
        d, k, k2 = cfg.d_scalar, cfg.k_vec, cfg.k2
        self.d, self.vec_c, self.n_stages, self.n_ref = d, cfg.vec_c, cfg.n_stages, int(n_ref)
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
        self._baked: dict[str, Tensor | bool] | None = None
        self._cond_ref: list[nn.Module] = []
        self.chunk_nodes: int = 0

    def train(self, mode: bool = True):
        if mode:
            self._baked = None
        return super().train(mode)

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
        }
        if not ref_zero:
            bk["ref_tab"] = (w_ref @ self._z_basis.to(device=w_ref.device, dtype=w_ref.dtype).t()).detach().contiguous()
        if self.iso_a is not None and conditioning is not None:
            n_tok = conditioning.emb_token.num_embeddings  # type: ignore[attr-defined]
            dev, dt = self.iso_a.weight.device, self.iso_a.weight.dtype
            toks = torch.arange(n_tok, device=dev).repeat_interleave(2)
            us = torch.tensor([0, 1] * n_tok, device=dev)
            c_all = conditioning.functional(toks, us).to(dt)
            bk["iso_tab"] = (self.iso_a.weight.squeeze(-1).unsqueeze(0) + c_all @ self.iso_b.weight.t()).view(n_tok, 2, N_Z).detach().contiguous()
        self._baked = bk

    def unbake_inference(self) -> None:
        self._baked = None

    def _get_baked(self, like: Tensor) -> dict[str, Tensor | bool] | None:
        if self._baked is None or self.training or self.read_u.weight.requires_grad:
            return None
        w = self._baked["w_ruv"]
        if not torch.compiler.is_compiling() and isinstance(w, Tensor) and (w.device != like.device or w.dtype != like.dtype):
            self.bake_for_inference()
        return self._baked

    def _ro_mlp(self, s: Tensor, vec: Tensor, ten: Tensor | None) -> Tensor:
        bk = self._get_baked(s)
        u = bound_vectors(vec, self.vec_c)
        if bk is not None:
            ru, rv = F.linear(u.transpose(-1, -2), bk["w_ruv"]).transpose(-1, -2).chunk(2, dim=-2)  # type: ignore[arg-type]
            pinv = (ru * rv).sum(-1)
        else:
            pinv = pair_inv(u, self.read_u, self.read_v)
        h = self.readout[0](torch.cat([state_norm(s), pinv], dim=-1))
        if ten is not None:
            tb = bound_vectors(ten, self.vec_c)
            if bk is not None:
                tu, tv = F.linear(tb.transpose(-1, -2), bk["w_rtuv"]).transpose(-1, -2).chunk(2, dim=-2)  # type: ignore[arg-type]
                tinv = (tu * tv).sum(-1)
            else:
                tinv = pair_inv(tb, self.read_tu, self.read_tv)
            h = h + self.read_t_proj(tinv)
        return self.readout[4](self.readout[3](self.readout[2](self.readout[1](h)))).squeeze(-1)

    @property
    def term_names(self) -> tuple[str, ...]:
        return tuple(f"stage{l}" for l in range(self.n_stages - 1)) + ("final",) + (("iso",) if self.iso_a is not None else ())

    def stage(self, l: int, s: Tensor, atom_mask: Tensor | None = None, normed: bool = False) -> Tensor:
        e = self.stage_readouts[l](s if normed else state_norm(s)).squeeze(-1)
        return e if atom_mask is None else e * atom_mask.to(e.dtype)

    def final(self, s: Tensor, vec: Tensor, b: GraphBatch, cond: Tensor | None = None,
              stage_terms: dict[str, Tensor] | None = None, ten: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        mf = b.atom_mask.to(s.dtype)
        z = b.z.clamp(0, N_Z - 1)
        cn = self.chunk_nodes or (4096 if (s.shape[0] > 8192 and not self.training and not self.read_u.weight.requires_grad) else 0)
        if cn > 0 and s.shape[0] > cn and not torch.compiler.is_compiling():
            ps = tuple(self.parameters())
            parts = []
            for a0 in range(0, s.shape[0], cn):
                a1 = min(a0 + cn, s.shape[0])
                if ten is None:
                    parts.append(remat_apply(RematSpec(lambda s_, v_: (self._ro_mlp(s_, v_, None),), ps), (s[a0:a1], vec[a0:a1]))[0])
                else:
                    parts.append(remat_apply(RematSpec(lambda s_, v_, t_: (self._ro_mlp(s_, v_, t_),), ps), (s[a0:a1], vec[a0:a1], ten[a0:a1]))[0])
            ro = torch.cat(parts, dim=0)
        else:
            ro = self._ro_mlp(s, vec, ten)
        bk = self._get_baked(s)
        if bk is not None and bk["ref_zero"]:
            fin = ro * mf
        elif bk is not None and "ref_tab" in bk:
            rt = bk["ref_tab"]  # type: ignore[assignment]
            ref_off = rt[b.ref_id[0], z] if b.n_graphs == 1 else rt[b.ref_id.index_select(0, b.graph), z]  # type: ignore[index]
            fin = (ro + ref_off) * mf
        else:
            ref_off = (self.ref_emb(b.ref_id).index_select(0, b.graph) * self._z_basis.to(s.dtype).index_select(0, z)).sum(-1)
            fin = (ro + ref_off) * mf
        terms = {k: v * mf for k, v in (stage_terms or {}).items()}
        terms["final"] = fin
        if self.iso_a is not None:
            if cond is None:
                raise ValueError("iso_reference needs the per-graph functional conditioning cond [G, cond_dim] (pass Conditioning.functional(token, u), "
                                 "or build Readout(iso_reference=False))")
            if bk is not None and "iso_tab" in bk:
                it = bk["iso_tab"]  # type: ignore[assignment]
                u_cl = b.u_flag.clamp(0, 1)
                iso_e = it[b.token[0], u_cl[0], z] if b.n_graphs == 1 else it[b.token.index_select(0, b.graph), u_cl.index_select(0, b.graph), z]  # type: ignore[index]
                terms["iso"] = iso_e * mf
            else:
                terms["iso"] = (self.iso_a(z).squeeze(-1) + (self.iso_b(z) * cond.index_select(0, b.graph)).sum(-1)) * mf
        atomic = sum(terms.values())
        return atomic, terms

    @staticmethod
    def graph_energy(atomic: Tensor, b: GraphBatch) -> Tensor:
        if b.n_graphs == 1:
            return atomic.sum(0, keepdim=True)
        return atomic.new_zeros(b.n_graphs).index_add(0, b.graph, atomic)
