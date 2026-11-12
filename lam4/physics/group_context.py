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
    G = charge.shape[0]
    mf = atom_mask.to(charge.dtype)
    n_g = (mf.sum(0, keepdim=True) if G == 1 else charge.new_zeros(G).index_add(0, graph, mf)).clamp(1.0, 350.0)
    cols = [charge / 4.0, n_g.pow(-1.0 / 3.0), (spin.to(charge.dtype) - 1.0) / 4.0]
    row = torch.stack(cols, -1)
    return row.expand(graph.shape[0], -1) if G == 1 else row.index_select(0, graph)


def _group_mean(x: Tensor, graph: Tensor, m: Tensor, G: int) -> Tensor:
    mc = m.to(x.dtype)
    if G == 1:
        s = (x * mc.unsqueeze(-1)).sum(0, keepdim=True)
        c = mc.sum(0, keepdim=True).clamp_min(1.0)
        return (s / c.unsqueeze(-1)).expand(x.shape[0], -1)
    s = x.new_zeros(G, x.shape[-1]).index_add(0, graph, x * mc.unsqueeze(-1))
    c = x.new_zeros(G).index_add(0, graph, mc).clamp_min(1.0)
    return (s / c.unsqueeze(-1))[graph]


def _group_softmax_pool(logits: Tensor, values: Tensor, graph: Tensor, m: Tensor, G: int) -> Tensor:
    """sum_j softmax_j(logits) values_j per group over real atoms. logits [N, H], values [N, H, Dh] -> [N, H*Dh]."""
    H = logits.shape[-1]
    lg = logits.masked_fill(~m.unsqueeze(-1), NEG)
    if G == 1 and lg.shape[0] > 0:
        mx = lg.amax(0, keepdim=True).detach()
        e = torch.exp(lg - mx) * m.unsqueeze(-1).to(logits.dtype)
        z = e.sum(0, keepdim=True).clamp_min(1e-12)
        num = (e.unsqueeze(-1) * values).sum(0, keepdim=True)
        return (num / z.unsqueeze(-1)).flatten(-2).expand(logits.shape[0], -1)
    mx = lg.new_full((G, H), NEG).scatter_reduce(0, graph.unsqueeze(-1).expand(-1, H), lg, reduce="amax",
                                                 include_self=True).detach()
    e = torch.exp(lg - mx[graph]) * m.unsqueeze(-1).to(logits.dtype)
    z = logits.new_zeros(G, H).index_add(0, graph, e).clamp_min(1e-12)
    num = values.new_zeros(G, *values.shape[1:]).index_add(0, graph, e.unsqueeze(-1) * values)
    return (num / z.unsqueeze(-1))[graph].flatten(-2)


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

    @torch.no_grad()
    def bake_for_inference(self) -> None:
        w = torch.cat([self.mean_val.weight, self.att_logit.weight, self.att_val.weight, self.atom_in.weight], 0).detach().contiguous()
        b = F.pad(torch.cat([self.mean_val.bias, self.att_logit.bias, self.att_val.bias], 0), (0, self.atom_in.out_features)).detach().contiguous()
        self._baked = (w, b)

    def unbake_inference(self) -> None:
        self._baked = None

    def forward(self, s: Tensor, graph: Tensor, atom_mask: Tensor, charge: Tensor, spin: Tensor, domain: str) -> Tensor:
        if not group_gate(domain):
            return s
        G = charge.shape[0]
        n = self.in_norm(s)
        bk = self._baked if (self._baked is not None and not self.training and not self.mean_val.weight.requires_grad) else None
        if bk is not None:
            if not torch.compiler.is_compiling() and (bk[0].device != n.device or bk[0].dtype != n.dtype):
                self.bake_for_inference()
                bk = self._baked
            d_m, h, d_a = self.mean_val.out_features, self.heads, self.atom_in.out_features
            mv, al, av, an = F.linear(n, bk[0], bk[1]).split([d_m, h, d_m, d_a], dim=-1)
        else:
            mv, al, av, an = self.mean_val(n), self.att_logit(n), self.att_val(n), self.atom_in(n)
        if G == 1 and s.shape[0] > 0:
            mc = atom_mask.to(s.dtype)
            c = mc.sum(0, keepdim=True).clamp_min(1.0)
            mean_1 = (mv * mc.unsqueeze(-1)).sum(0, keepdim=True) / c.unsqueeze(-1)
            lg = al.masked_fill(~atom_mask.unsqueeze(-1), NEG)
            e = torch.exp(lg - lg.amax(0, keepdim=True).detach()) * mc.unsqueeze(-1)
            att_1 = ((e.unsqueeze(-1) * av.unflatten(-1, (self.heads, self.dh))).sum(0, keepdim=True)
                     / e.sum(0, keepdim=True).clamp_min(1e-12).unsqueeze(-1)).flatten(-2)
            ctx_1 = self.ctx_norm(torch.cat([mean_1, att_1], -1))
            scal_1 = torch.stack([charge.to(s.dtype) / 4.0, c.clamp(1.0, 350.0).pow(-1.0 / 3.0), (spin.to(s.dtype) - 1.0) / 4.0], -1)
            a, b = (self.ctx_in(torch.cat([ctx_1, scal_1], -1)) + an).chunk(2, -1)
            return s + self.out(F.silu(a) * b) * mc.unsqueeze(-1)
        scal = group_scalars(charge.to(s.dtype), spin, graph, atom_mask, domain)
        mean = _group_mean(mv, graph, atom_mask, G)
        att = _group_softmax_pool(al, av.unflatten(-1, (self.heads, self.dh)), graph, atom_mask, G)
        ctx = self.ctx_norm(torch.cat([mean, att], -1))
        a, b = (self.ctx_in(torch.cat([ctx, scal], -1)) + an).chunk(2, -1)
        return s + self.out(F.silu(a) * b) * atom_mask.to(s.dtype).unsqueeze(-1)


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
        eq = self.e_q[(q + 10).clamp(0, 20)] * ok_q
        es = self.e_s[(s - 1).clamp(0, 10)] * ok_s
        return (eq + es) * graph_mask.to(torch.float64)
