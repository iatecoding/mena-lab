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

import torch.nn as nn
from torch import Tensor

from lam4.config import ModelConfig
from lam4.physics.group_context import group_gate


class Conditioning(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.q_max, self.s_max = cfg.q_max, cfg.s_max
        self.emb_token = nn.Embedding(cfg.n_tokens, cfg.cond_dim)
        self.emb_u = nn.Embedding(2, cfg.cond_dim)
        self.emb_q = nn.Embedding(2 * cfg.q_max + 1, cfg.cond_dim)
        self.emb_s = nn.Embedding(cfg.s_max, cfg.cond_dim)

    def functional(self, token: Tensor, u_flag: Tensor) -> Tensor:
        """[G, cond_dim] = emb(token) + emb(u): functional and +U only (the isolated-atom reference input)."""
        return self.emb_token(token) + self.emb_u(u_flag)

    def forward(self, token: Tensor, u_flag: Tensor, charge: Tensor, spin: Tensor, domain: str) -> Tensor:
        """token, u_flag [G] long; charge, spin [G] float; domain "crystal" | "molecule" -> [G, cond_dim]."""
        c = self.functional(token, u_flag)
        if group_gate(domain):
            c = c + self.emb_q(charge.round().long().clamp(-self.q_max, self.q_max) + self.q_max)
            c = c + self.emb_s(spin.round().long().clamp(1, self.s_max) - 1)
        return c
