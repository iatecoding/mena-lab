"""Conditioning vector c [G, cond_dim] = emb(token) + emb(u) + emb(Q) + emb(S), one per graph.

token: trunk token id (functional, lam4.registry.tokens); u: +U flag 0/1; Q: total charge (clamped to +-q_max); S: spin multiplicity (1..s_max).
Q and S enter ONLY as this per-graph vector and ONLY for molecules (`group_gate(domain)`): there is no per-atom spin embedding, and a crystal
gets emb(token) + emb(u) alone (the Q / S tables are not even read, so a crystal batch carrying a charge or a spin cannot change). The
consumers are the per-stage FiLM layers (lam4.core.stage), whose output projections are zero-init: c has the default N(0, 1) embedding
init (a zero embedding behind a zero projection would never get a gradient), and the trunk is unconditioned at step 0.
Pad graphs (token 0, u 0, Q 0, S 1) produce a finite c that only pad atoms read.
`functional(token, u)` = emb(token) + emb(u) is the part that the per-atom isolated-atom reference reads (Readout iso): Q and S must not
enter a per-atom term, because summed over n_g atoms it becomes an offset linear in n_g for every charged / open-shell molecule, while
the charge and spin energies are intensive (E_Q / E_S, group context, NQE charges).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lam4.config import ModelConfig
from lam4.physics.group_context import group_gate


class Conditioning(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.q_max, self.s_max = cfg.q_max, cfg.s_max
        self.factorized = bool(cfg.factorized_lot)
        n_tok = cfg.n_xc_tokens if self.factorized else cfg.n_tokens
        self.emb_token = nn.Embedding(n_tok, cfg.cond_dim)
        self.emb_paw = nn.Embedding(cfg.n_paw_bases, cfg.cond_dim) if self.factorized else None
        self.emb_u = nn.Embedding(2, cfg.cond_dim)
        self.emb_q = nn.Embedding(2 * cfg.q_max + 1, cfg.cond_dim)
        self.emb_s = nn.Embedding(cfg.s_max, cfg.cond_dim)

    def _u_lookup(self, u: Tensor) -> Tensor:
        if u.is_floating_point():
            uf = u.clamp(0.0, 1.0).to(self.emb_u.weight.dtype).unsqueeze(-1)
            return (1.0 - uf) * self.emb_u.weight[0] + uf * self.emb_u.weight[1]
        return self.emb_u(u.long().clamp(0, 1))

    def functional(self, token: Tensor, u_flag: Tensor, paw_id: Tensor | None = None) -> Tensor:
        """[G, cond_dim] = emb(xc/token) [+ emb(paw)] + emb(u): functional/PAW/+U only (isolated-atom reference)."""
        c = self.emb_token(token) + self._u_lookup(u_flag)
        if self.emb_paw is not None:
            pid = torch.zeros_like(token) if paw_id is None else paw_id
            c = c + self.emb_paw(pid)
        return c

    def functional_atoms(
        self,
        token: Tensor,
        u_flag: Tensor,
        graph: Tensor,
        *,
        paw_id: Tensor | None = None,
        u_atom: Tensor | None = None,
    ) -> Tensor:
        """[N, cond_dim]: per-atom functional/PAW/+U conditioning supporting per-atom `u_atom [N]` or graph `u_flag [G]`."""
        if not self.factorized:
            return self.functional(token, u_flag).index_select(0, graph)
        c_g = self.emb_token(token)
        if self.emb_paw is not None:
            pid = torch.zeros_like(token) if paw_id is None else paw_id
            c_g = c_g + self.emb_paw(pid)
        u_emb = self._u_lookup(u_flag).index_select(0, graph) if u_atom is None else self._u_lookup(u_atom)
        return c_g.index_select(0, graph) + u_emb

    def forward(
        self,
        token: Tensor,
        u_flag: Tensor,
        charge: Tensor,
        spin: Tensor,
        domain: str = "molecule",
        *,
        paw_id: Tensor | None = None,
        apply_qs: bool | None = None,
    ) -> Tensor:
        """token, u_flag [G] long; charge, spin [G] float -> [G, cond_dim]."""
        c = self.functional(token, u_flag, paw_id=paw_id)
        use_qs = group_gate(domain) if apply_qs is None else bool(apply_qs)
        if use_qs:
            c = c + self.emb_q(charge.round().long().clamp(-self.q_max, self.q_max) + self.q_max)
            c = c + self.emb_s(spin.round().long().clamp(1, self.s_max) - 1)
        return c

    def per_atom(
        self,
        token: Tensor,
        u_flag: Tensor,
        charge: Tensor,
        spin: Tensor,
        domain: str,
        graph: Tensor,
        *,
        paw_id: Tensor | None = None,
        u_atom: Tensor | None = None,
        apply_qs: bool | None = None,
        charge_graph: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return `(cond_full_atom [N, cond_dim], cond_func_atom [N, cond_dim])` without graph-vs-atom ambiguity when G == N."""
        c_func = self.functional_atoms(token, u_flag, graph, paw_id=paw_id, u_atom=u_atom)
        use_qs = group_gate(domain) if apply_qs is None else bool(apply_qs)
        if not use_qs:
            return c_func, c_func
        cg = graph if charge_graph is None else charge_graph
        qs = (
            self.emb_q(charge.round().long().clamp(-self.q_max, self.q_max) + self.q_max)
            + self.emb_s(spin.round().long().clamp(1, self.s_max) - 1)
        ).index_select(0, cg)
        return c_func + qs, c_func
