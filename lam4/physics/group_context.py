"""group_context.py: whole-molecule channel (pooled group context, charge table, E_Q / E_S references).

HARD GATE. Everything in this module is a molecule-only mechanism: for `domain == "crystal"` each entry point is an
exact identity (GroupContext, ChargeEmbed return their input tensor object) or exactly zero (GroupReference), and
n_g^(-1/3) is never computed (group_scalars raises). A periodic supercell therefore stays strictly extensive.
`group_gate(domain)` is the single decision function; every gated module calls it first.

GroupContext (omol4): after a stage, s_i <- s_i + W_o(SiLU(a) * b),
    [a, b] = W_c [ctx_g, Q_g/4, n_g^(-1/3), (M_g-1)/4] + W_s RMSNorm(s_i),   ctx_g = [mean, attention] pool over group g
with n_g the REAL atom count (pad atoms are excluded from the means, the attention softmax and the count). Zero-init out.
ChargeEmbed (omol4 `charge_embed`): s_i += E[Q_g] at the model input (zero-init table, Q clamped to +-max_charge).
GroupReference (earlier Mena References.group_terms): E_Q[Q_g] + E_S[S_g] in float64 (kcal/mol, buffers, no gradient;
a charge outside -10..10 or a spin outside 1..11 adds 0). The per-element E0 is a registry matter, not part of this module.

Signatures (graph [N] is the group id of each atom, charge [G], spin [G] multiplicity, atom_mask [N], domain str)
    group_gate(domain) -> bool;  group_scalars(charge, spin, graph, atom_mask, domain) -> [N, 3]
    GroupContext(d_model, dim=32, heads=4, hidden=64).forward(s [N, d], graph, atom_mask, charge, spin, domain) -> s
    ChargeEmbed(d_model, max_charge=10).forward(s [N, d], graph, charge, atom_mask, domain) -> s
    GroupReference(tables=None).forward(charge, spin, graph_mask, domain) -> [G] float64

Key map from an earlier-generation state dict (drop the prefix, identical sub-names):
    group_ctx.{k}.{in_norm.weight, mean_val.*, att_logit.*, att_val.*, ctx_norm.weight, ctx_in.*, atom_in.weight, out.*}
        -> GroupContext.<same>      (ctx_in is Linear(2*dim + 3, 2*hidden): omol4 built with ic_spin=True)
    charge_embedding.weight -> ChargeEmbed.charge_embedding.weight
    references.{e_q, e_s}   -> GroupReference.{e_q, e_s}      (references.e0 is not ported)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.core.consts import tracing
from lam4.core.radial import fp32_guard, resolve_baked, stop_grad, upcast_fp32
from lam4.core.remat import NODE_CHUNK_ATOMS, can_chunk_nodes, chunked_node_apply, node_chunk_size

NEG = -1.0e9
CHARGES = tuple(range(-10, 11))
SPINS = tuple(range(1, 12))


def _default_ref_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    repo_cand = root / "data" / "e0_reference_omol25.json"
    if repo_cand.is_file():
        return repo_cand
    bundled_cand = root / "lam" / "data" / "e0_refs" / "e0_reference_omol25.json"
    return bundled_cand if bundled_cand.is_file() else repo_cand


DEFAULT_REF = _default_ref_path()


def group_gate(domain: str) -> bool:
    """True when the group-level physics applies (molecules); False for crystals; error on anything else."""
    if domain == "molecule":
        return True
    if domain == "crystal":
        return False
    raise ValueError(f"unknown domain {domain!r}; expected 'molecule' or 'crystal'")


def group_domain(domain: str, pbc: bool = False, has_frag: bool = False) -> str:
    """The domain string every group-level module reads (GroupContext, Conditioning's Q / S, GroupReference). Group physics needs FINITE groups:
    an open molecule (the graph), or a periodic cell whose fragments are given (frag_id). A periodic batch without fragments (OMC25 before the
    fragment pipeline, box-augmented molecules) gets "crystal" here, i.e. no group physics, so E(2x2x2) = 8 E(1x1x1) holds; the node-MLP expert still
    follows `domain`. Spec section 12 (regime / boundary / fragment split)."""
    group_gate(domain)
    return "molecule" if domain == "molecule" and (not pbc or has_frag) else "crystal"


def group_scalars(charge: Tensor, spin: Tensor, graph: Tensor, atom_mask: Tensor, domain: str) -> Tensor:
    """[N, 3] = [Q_g/4, n_g^(-1/3), (M_g-1)/4] of each atom's group; n_g counts real atoms (clamped to [1, 350]). Molecule domain only."""
    if not group_gate(domain):
        raise ValueError("group_scalars is molecule-only: n_g^(-1/3) is not extensive and is never used for crystals")
    with fp32_guard(charge.device):
        cf = upcast_fp32(charge)
        G = cf.shape[0]
        mf = atom_mask.to(cf.dtype)
        n_g = (mf.sum(0, keepdim=True) if G == 1 else cf.new_zeros(G).index_add(0, graph, mf)).clamp(1.0, 350.0)
        cols = [cf / 4.0, n_g.pow(-1.0 / 3.0), (spin.to(cf.dtype) - 1.0) / 4.0]
        row = torch.stack(cols, -1)
        return row.expand(graph.shape[0], -1) if G == 1 else row.index_select(0, graph)


def _group_mean(x: Tensor, graph: Tensor, m: Tensor, G: int) -> Tensor:
    with fp32_guard(x.device):
        xf = upcast_fp32(x)
        mc = m.to(xf.dtype)
        if G == 1:
            s = (xf * mc.unsqueeze(-1)).sum(0, keepdim=True)
            c = mc.sum(0, keepdim=True).clamp_min(1.0)
            return (s / c.unsqueeze(-1)).expand(xf.shape[0], -1)
        s = xf.new_zeros(G, xf.shape[-1]).index_add(0, graph, xf * mc.unsqueeze(-1))
        c = xf.new_zeros(G).index_add(0, graph, mc).clamp_min(1.0)
        return (s / c.unsqueeze(-1))[graph]


def _group_softmax_pool(logits: Tensor, values: Tensor, graph: Tensor, m: Tensor, G: int) -> Tensor:
    """sum_j softmax_j(logits) values_j per group over real atoms. logits [N, H], values [N, H, Dh] -> [N, H*Dh]."""
    with fp32_guard(logits.device):
        logits, values = upcast_fp32(logits), upcast_fp32(values)
        H = logits.shape[-1]
        lg = logits.masked_fill(~m.unsqueeze(-1), NEG)
        if G == 1 and lg.shape[0] > 0:
            mx = stop_grad(lg.amax(0, keepdim=True))
            e = torch.exp(lg - mx) * m.unsqueeze(-1).to(logits.dtype)
            z = e.sum(0, keepdim=True).clamp_min(1e-12)
            num = (e.unsqueeze(-1) * values).sum(0, keepdim=True)
            return (num / z.unsqueeze(-1)).flatten(-2).expand(logits.shape[0], -1)
        mx = stop_grad(lg.new_full((G, H), NEG).scatter_reduce(0, graph.unsqueeze(-1).expand(-1, H), lg, reduce="amax", include_self=True))
        e = torch.exp(lg - mx[graph]) * m.unsqueeze(-1).to(logits.dtype)
        z = logits.new_zeros(G, H).index_add(0, graph, e).clamp_min(1e-12)
        num = values.new_zeros(G, *values.shape[1:]).index_add(0, graph, e.unsqueeze(-1) * values)
        return (num / z.unsqueeze(-1))[graph].flatten(-2)


class _ChunkedGroupPoolFn(torch.autograd.Function):
    """Exact single-pass online pool + exact VJP for GroupContext mean & softmax attention above NODE_CHUNK_ATOMS."""

    @staticmethod
    def forward(ctx, mod: "GroupContext", graph: Tensor, atom_mask: Tensor, G: int, chunk_atoms: int, s: Tensor):
        d_m, H, Dh = mod.mean_val.out_features, mod.heads, mod.dh
        n_atoms = s.shape[0]
        mv_sum = s.new_zeros(G, d_m)
        cnt = s.new_zeros(G)
        mx = s.new_full((G, H), NEG)
        z = s.new_zeros(G, H)
        num = s.new_zeros(G, H, Dh)
        with torch.enable_grad():
            for a0 in range(0, n_atoms, chunk_atoms):
                a1 = min(a0 + chunk_atoms, n_atoms)
                s_c = s[a0:a1].detach().requires_grad_(s.requires_grad)
                mv_c, al_c, av_c = mod._pool_proj(s_c)
                with torch.no_grad():
                    m_c = atom_mask[a0:a1]
                    mc = m_c.to(s.dtype)
                    av_h = av_c.detach().unflatten(-1, (H, Dh))
                    lg_c = al_c.detach().masked_fill(~m_c.unsqueeze(-1), NEG)
                    if G == 1:
                        mv_sum = mv_sum + (mv_c.detach() * mc.unsqueeze(-1)).sum(0, keepdim=True)
                        cnt = cnt + mc.sum(0, keepdim=True)
                        mx_c = lg_c.amax(0, keepdim=True)
                        e_c = torch.exp(lg_c - mx_c) * mc.unsqueeze(-1)
                        z_c = e_c.sum(0, keepdim=True)
                        num_c = (e_c.unsqueeze(-1) * av_h).sum(0, keepdim=True)
                        mx_new = torch.maximum(mx, mx_c)
                        alpha = torch.exp(mx - mx_new)
                        beta = torch.exp(mx_c - mx_new)
                        z = z * alpha + z_c * beta
                        num = num * alpha.unsqueeze(-1) + num_c * beta.unsqueeze(-1)
                        mx = mx_new
                    else:
                        g_c = graph[a0:a1]
                        mv_sum.index_add_(0, g_c, mv_c.detach() * mc.unsqueeze(-1))
                        cnt.index_add_(0, g_c, mc)
                        mx_c = lg_c.new_full((G, H), NEG).scatter_reduce(
                            0, g_c.unsqueeze(-1).expand(-1, H), lg_c, reduce="amax", include_self=True
                        )
                        mx_new = torch.maximum(mx, mx_c)
                        e_c = torch.exp(lg_c - mx_c[g_c]) * mc.unsqueeze(-1)
                        z_c = lg_c.new_zeros(G, H).index_add(0, g_c, e_c)
                        num_c = av_h.new_zeros(G, H, Dh).index_add(0, g_c, e_c.unsqueeze(-1) * av_h)
                        alpha = torch.exp(mx - mx_new)
                        beta = torch.exp(mx_c - mx_new)
                        z = z * alpha + z_c * beta
                        num = num * alpha.unsqueeze(-1) + num_c * beta.unsqueeze(-1)
                        mx = mx_new
        c_safe = cnt.clamp_min(1.0)
        mean = mv_sum / c_safe.unsqueeze(-1)
        z_safe = z.clamp_min(1e-12)
        att = (num / z_safe.unsqueeze(-1)).flatten(-2)
        if ctx is not None:  # no-grad inference path calls forward(None, ...) directly
            ctx.mod = mod
            ctx.G = G
            ctx.chunk_atoms = chunk_atoms
            ctx.save_for_backward(s, graph, atom_mask, c_safe, mx, z_safe, att)
        return mean.detach(), att.detach(), cnt.detach()

    @staticmethod
    def backward(ctx, d_mean: Tensor | None, d_att: Tensor | None, _d_cnt: Tensor | None):
        if not ctx.needs_input_grad[5] or (d_mean is None and d_att is None):
            return None, None, None, None, None, None
        if torch.is_grad_enabled():
            raise RuntimeError("_ChunkedGroupPoolFn does not support create_graph=True")
        mod, G, chunk_atoms = ctx.mod, ctx.G, ctx.chunk_atoms
        s, graph, atom_mask, c_safe, mx, z_safe, att = ctx.saved_tensors
        H, Dh = mod.heads, mod.dh
        n_atoms = s.shape[0]
        d_s = s.new_empty(s.shape)
        att_h = att.unflatten(-1, (H, Dh))
        d_att_h = None if d_att is None else d_att.unflatten(-1, (H, Dh))
        with torch.enable_grad():
            for a0 in range(0, n_atoms, chunk_atoms):
                a1 = min(a0 + chunk_atoms, n_atoms)
                s_c = s[a0:a1].detach().requires_grad_(True)
                mv_c, al_c, av_c = mod._pool_proj(s_c)
                m_c = atom_mask[a0:a1]
                mc = m_c.to(s.dtype)
                lg_c = al_c.masked_fill(~m_c.unsqueeze(-1), NEG)
                surr = s_c.new_zeros(())
                if G == 1:
                    if d_mean is not None:
                        surr = surr + ((mv_c * mc.unsqueeze(-1) / c_safe.unsqueeze(-1)) * d_mean).sum()
                    if d_att_h is not None:
                        w_c = torch.exp(lg_c - mx) * (mc.unsqueeze(-1) / z_safe)
                        surr = surr + (w_c.unsqueeze(-1) * (av_c.unflatten(-1, (H, Dh)) - att_h) * d_att_h).sum()
                else:
                    g_c = graph[a0:a1]
                    if d_mean is not None:
                        surr = surr + ((mv_c * mc.unsqueeze(-1) / c_safe[g_c].unsqueeze(-1)) * d_mean[g_c]).sum()
                    if d_att_h is not None:
                        w_c = torch.exp(lg_c - mx[g_c]) * (mc.unsqueeze(-1) / z_safe[g_c])
                        surr = surr + (w_c.unsqueeze(-1) * (av_c.unflatten(-1, (H, Dh)) - att_h[g_c]) * d_att_h[g_c]).sum()
                (ds_c,) = torch.autograd.grad(surr, s_c)
                d_s[a0:a1].copy_(ds_c)
        return None, None, None, None, None, d_s


class GroupContext(nn.Module):
    N_SCALARS = 3

    def __init__(self, d_model: int, dim: int = 32, heads: int = 4, hidden: int = 64) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim ({dim}) must be a multiple of heads ({heads})")
        self.heads, self.dh = heads, dim // heads
        self.in_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.mean_val = nn.Linear(d_model, dim)
        self.att_logit = nn.Linear(d_model, heads)
        self.att_val = nn.Linear(d_model, dim)
        self.ctx_norm = nn.RMSNorm(2 * dim, eps=1e-6)
        self.ctx_in = nn.Linear(2 * dim + self.N_SCALARS, 2 * hidden)
        self.atom_in = nn.Linear(d_model, 2 * hidden, bias=False)
        self.out = nn.Linear(hidden, d_model)
        for lin in (self.att_logit, self.out):  # attention starts as mean pooling; the module starts as the identity
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._baked: tuple[Tensor, Tensor] | None = None

    def train(self, mode: bool = True):
        if mode:
            self._baked = None
        return super().train(mode)

    def _apply(self, fn):
        had_baked = self._baked is not None
        self._baked = None
        out = super()._apply(fn)
        if had_baked and not self.training and not self.mean_val.weight.is_meta and not tracing():
            self.bake_for_inference()
        return out

    @torch.no_grad()
    def bake_for_inference(self) -> None:
        w = torch.cat([self.mean_val.weight, self.att_logit.weight, self.att_val.weight, self.atom_in.weight], 0).detach().contiguous()
        b = F.pad(torch.cat([self.mean_val.bias, self.att_logit.bias, self.att_val.bias], 0), (0, self.atom_in.out_features)).detach().contiguous()
        self._baked = (w, b)

    def unbake_inference(self) -> None:
        self._baked = None

    def _get_baked(self, like: Tensor) -> tuple[Tensor, Tensor] | None:
        return resolve_baked(self, lambda bk: bk[0], self.mean_val.weight, like)

    def _pool_proj(self, s: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n = self.in_norm(s)
        bk = self._get_baked(n)
        if bk is not None:
            d_m, h = self.mean_val.out_features, self.heads
            n_pool = 2 * d_m + h
            return F.linear(n, bk[0][:n_pool], bk[1][:n_pool]).split([d_m, h, d_m], dim=-1)
        return self.mean_val(n), self.att_logit(n), self.att_val(n)

    def _chunked_forward(self, s: Tensor, graph: Tensor, atom_mask: Tensor, charge: Tensor, spin: Tensor) -> Tensor:
        G = charge.shape[0]
        ps = tuple(self.parameters())
        self._get_baked(s)
        cn = node_chunk_size(s.device) if s.device.type == "mps" else NODE_CHUNK_ATOMS
        if not torch.is_grad_enabled():
            mean, att, cnt = _ChunkedGroupPoolFn.forward(None, self, graph, atom_mask, G, cn, s)  # type: ignore[arg-type]
        else:
            mean, att, cnt = _ChunkedGroupPoolFn.apply(self, graph, atom_mask, G, cn, s)
        ctx = self.ctx_norm(torch.cat([mean, att], -1))
        scal = torch.stack([charge.to(s.dtype) / 4.0, cnt.clamp(1.0, 350.0).pow(-1.0 / 3.0), (spin.to(s.dtype) - 1.0) / 4.0], -1)
        ctx_in_g = self.ctx_in(torch.cat([ctx, scal], -1))

        def _apply_chunk(a0: int, a1: int, s_c: Tensor, cg_in: Tensor) -> Tensor:
            n_c = self.in_norm(s_c)
            an_c = self.atom_in(n_c)
            cg_c = cg_in if G == 1 else cg_in.index_select(0, graph[a0:a1])
            a_c, b_c = (cg_c + an_c).chunk(2, -1)
            mc_c = atom_mask[a0:a1].to(s_c.dtype).unsqueeze(-1)
            return s_c + self.out(F.silu(a_c) * b_c).to(s_c.dtype) * mc_c

        return chunked_node_apply(_apply_chunk, (s,), (ctx_in_g,), params=ps, chunk_atoms=cn)[0]

    def forward(self, s: Tensor, graph: Tensor, atom_mask: Tensor, charge: Tensor, spin: Tensor, domain: str) -> Tensor:
        if not group_gate(domain):
            return s
        s_dt = torch.float64 if s.dtype == torch.float64 else torch.float32
        s = upcast_fp32(s)
        ps = tuple(self.parameters())
        cn = node_chunk_size(s.device) if s.device.type == "mps" else NODE_CHUNK_ATOMS
        if not self.training and can_chunk_nodes(s.shape[0], ps, min_atoms=cn):
            return self._chunked_forward(s, graph, atom_mask, charge, spin).to(s_dt)
        G = charge.shape[0]
        n = self.in_norm(s)
        bk = self._get_baked(n)
        if bk is not None:
            d_m, h, d_a = self.mean_val.out_features, self.heads, self.atom_in.out_features
            mv, al, av, an = F.linear(n, bk[0], bk[1]).split([d_m, h, d_m, d_a], dim=-1)
        else:
            mv, al, av, an = self.mean_val(n), self.att_logit(n), self.att_val(n), self.atom_in(n)
        if G == 1 and s.shape[0] > 0:
            with fp32_guard(s.device):
                mvf, alf, avf = upcast_fp32(mv), upcast_fp32(al), upcast_fp32(av)
                mc = atom_mask.to(s_dt)
                c = mc.sum(0, keepdim=True).clamp_min(1.0)
                mean_1 = (mvf * mc.unsqueeze(-1)).sum(0, keepdim=True) / c.unsqueeze(-1)
                lg = alf.masked_fill(~atom_mask.unsqueeze(-1), NEG)
                e = torch.exp(lg - stop_grad(lg.amax(0, keepdim=True))) * mc.unsqueeze(-1)
                att_1 = ((e.unsqueeze(-1) * avf.unflatten(-1, (self.heads, self.dh))).sum(0, keepdim=True)
                         / e.sum(0, keepdim=True).clamp_min(1e-12).unsqueeze(-1)).flatten(-2)
                scal_1 = torch.stack([charge.to(s_dt) / 4.0, c.clamp(1.0, 350.0).pow(-1.0 / 3.0), (spin.to(s_dt) - 1.0) / 4.0], -1)
            ctx_1 = self.ctx_norm(torch.cat([mean_1, att_1], -1))
            a, b = (self.ctx_in(torch.cat([ctx_1.to(scal_1.dtype), scal_1], -1)) + an).chunk(2, -1)
            return s + self.out(F.silu(a) * b).to(s_dt) * mc.unsqueeze(-1)
        scal = group_scalars(charge.to(s_dt), spin, graph, atom_mask, domain)
        mean = _group_mean(mv, graph, atom_mask, G)
        att = _group_softmax_pool(al, av.unflatten(-1, (self.heads, self.dh)), graph, atom_mask, G)
        ctx = self.ctx_norm(torch.cat([mean, att], -1))
        a, b = (self.ctx_in(torch.cat([ctx.to(scal.dtype), scal], -1)) + an).chunk(2, -1)
        return s + self.out(F.silu(a) * b).to(s_dt) * atom_mask.to(s_dt).unsqueeze(-1)


class ChargeEmbed(nn.Module):
    def __init__(self, d_model: int, max_charge: int = 10) -> None:
        super().__init__()
        self.max_charge = int(max_charge)
        self.charge_embedding = nn.Embedding(2 * self.max_charge + 1, d_model)
        nn.init.zeros_(self.charge_embedding.weight)

    def forward(self, s: Tensor, graph: Tensor, charge: Tensor, atom_mask: Tensor, domain: str) -> Tensor:
        if not group_gate(domain):
            return s
        mc = self.max_charge
        e = self.charge_embedding(charge.round().long().clamp(-mc, mc).add(mc).index_select(0, graph))
        return s + e * atom_mask.to(s.dtype).unsqueeze(-1)


def load_group_tables(path: str | Path = DEFAULT_REF) -> dict[str, np.ndarray]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    e_q, e_s = np.zeros(len(CHARGES)), np.zeros(len(SPINS))
    for q, v in d["charge_kcal"].items():
        e_q[int(q) + 10] = v
    for s, v in d["spin_kcal"].items():
        e_s[int(s) - 1] = v
    return {"e_q": e_q, "e_s": e_s}


class GroupReference(nn.Module):
    def __init__(self, tables: dict[str, np.ndarray] | None = None) -> None:
        super().__init__()
        t = tables if tables is not None else load_group_tables()
        self.register_buffer("e_q", torch.as_tensor(t["e_q"], dtype=torch.float64))
        self.register_buffer("e_s", torch.as_tensor(t["e_s"], dtype=torch.float64))

    def forward(self, charge: Tensor, spin: Tensor, graph_mask: Tensor, domain: str) -> Tensor:
        """E_Q[Q_g] + E_S[S_g] per group [G] float64; exactly 0 for crystals and for pad graphs."""
        if not group_gate(domain):
            return torch.zeros(charge.shape[0], dtype=torch.float64, device=charge.device)
        q = charge.round().long()
        s = spin.round().long()
        ok_q = (q >= -10) & (q <= 10)
        ok_s = (s >= 1) & (s <= 11)
        e_q = self.e_q if self.e_q.device == charge.device else self.e_q.to(device=charge.device)
        e_s = self.e_s if self.e_s.device == spin.device else self.e_s.to(device=spin.device)
        eq = e_q[(q + 10).clamp(0, 20)] * ok_q
        es = e_s[(s - 1).clamp(0, 10)] * ok_s
        return (eq + es) * graph_mask.to(torch.float64)
