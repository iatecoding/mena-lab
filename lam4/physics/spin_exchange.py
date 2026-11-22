"""spin_exchange.py: Size-consistent Hund + Heisenberg spin-exchange head for open molecules.

Returns strictly 0 on periodic batches and crystal domains (`allow_mol = (domain == "molecule") and (not b.pbc)`).
At graft (`gate_raw = 0.0`), `E_exch == 0` and `F_exch == 0` identically while `I_H` and `(J_dir, J_cov)` stay
non-zero so `dE_exch / d(gate_raw) != 0` immediately at step 1 without a zero-shock step discontinuity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.core.consts import is_fake, proxy_tracing
from lam4.core.radial import GaussianBasis, N_RBF, R_CUT, fp32_guard, upcast_fp32
from lam4.core.remat import NODE_CHUNK_ATOMS, can_chunk_nodes, chunked_node_apply, node_chunk_size


_EXCH_EDGE_CHUNK = 16384


class SpinExchangeHead(nn.Module):
    """Size-consistent one-site Hund + pairwise Heisenberg exchange energy head."""

    def __init__(
        self,
        d_model: int,
        *,
        r_cut: float = R_CUT,
        rank: int = 16,
        n_rbf: int = N_RBF,
        charge_bound: float = 2.0,
    ) -> None:
        super().__init__()
        self.r_cut = float(r_cut)
        self.rank = int(rank)
        self.charge_bound = float(charge_bound)
        self.basis = GaussianBasis(n_rbf, self.r_cut)
        self.hund_emb = nn.Embedding(119, 1)
        nn.init.constant_(self.hund_emb.weight, 0.5)
        self.hund_lin = nn.Linear(int(d_model), 1)
        nn.init.normal_(self.hund_lin.weight, std=0.02)
        nn.init.zeros_(self.hund_lin.bias)
        self.proj_dir = nn.Linear(int(d_model), self.rank, bias=False)
        self.proj_cov = nn.Linear(int(d_model), self.rank, bias=False)
        self.rad_dir = nn.Linear(int(n_rbf), self.rank, bias=False)
        self.rad_cov = nn.Linear(int(n_rbf), self.rank, bias=False)
        for lin in (self.proj_dir, self.proj_cov):
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(max(1, d_model)))
        for lin in (self.rad_dir, self.rad_cov):
            nn.init.normal_(lin.weight, std=0.1)
        self.gate_raw = nn.Parameter(torch.zeros(()))

    def _eval_edges_torch(
        self,
        pos: Tensor,
        send: Tensor,
        recv: Tensor,
        cart_shift: Tensor | None,
        atom_mask: Tensor,
        h_dir: Tensor,
        h_cov: Tensor,
        m_t: Tensor,
        urf: Tensor,
    ) -> Tensor:
        n = pos.shape[0]
        sym = torch.compiler.is_compiling() or proxy_tracing() or is_fake(send)
        e_cnt = 0 if sym else int(send.shape[0])
        if not sym and e_cnt == 0:
            return pos.new_zeros(n)
        chunk = _EXCH_EDGE_CHUNK if (not sym and e_cnt > _EXCH_EDGE_CHUNK) else max(1, e_cnt)
        has_sh = cart_shift is not None and cart_shift.numel() > 0
        w_dir = upcast_fp32(self.rad_dir.weight).to(pos.dtype)
        w_cov = upcast_fp32(self.rad_cov.weight).to(pos.dtype)
        acc = pos.new_zeros(n)
        steps = ((0, 0),) if e_cnt == 0 else ((s0, min(e_cnt, s0 + chunk)) for s0 in range(0, e_cnt, chunk))
        for s0, s1 in steps:
            se = send if e_cnt == 0 else send[s0:s1]
            re = recv if e_cnt == 0 else recv[s0:s1]
            sh = (cart_shift if e_cnt == 0 else cart_shift[s0:s1]).to(pos.dtype) if has_sh else 0.0
            d_e = pos.index_select(0, se) - pos.index_select(0, re) + sh
            r2 = d_e.pow(2).sum(-1)
            real_e = (atom_mask.index_select(0, re) & atom_mask.index_select(0, se) & (r2 > 1e-12) & (r2 < self.r_cut * self.r_cut)).to(pos.dtype)
            r = torch.sqrt(r2 + 1e-12)
            chi = 0.5 * (1.0 + torch.cos(math.pi * (r / self.r_cut).clamp(0.0, 1.0))) * real_e
            phi_e = self.basis(r.clamp(max=self.r_cut))
            j_dir = (F.linear(phi_e, w_dir) * (h_dir.index_select(0, se) * h_dir.index_select(0, re))).sum(-1)
            j_cov = (F.linear(phi_e, w_cov) * (h_cov.index_select(0, se) * h_cov.index_select(0, re))).sum(-1)
            m_prod = m_t.index_select(0, se) * m_t.index_select(0, re)
            u_prod = urf.index_select(0, se) * urf.index_select(0, re)
            pair_msg = chi * (j_dir * m_prod + j_cov * (m_prod - u_prod))
            acc = acc + pos.new_zeros(n).index_add(0, re, pair_msg)
        return acc

    def _node_prologue(
        self, z: Tensor, sn: Tensor, u_rad: Tensor, m_spin: Tensor | None, m_raw: Tensor | None,
        atom_mask: Tensor, dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        with fp32_guard(sn.device, disable_tf32=True):
            mf = atom_mask.to(dtype)
            snf = upcast_fp32(sn)
            urf = upcast_fp32(u_rad).to(dtype) * mf
            c = self.charge_bound
            if m_spin is not None and m_raw is not None:
                m_t = 0.5 * (upcast_fp32(m_spin).to(dtype) + c * torch.tanh(upcast_fp32(m_raw).to(dtype) / c)) * mf
            elif m_spin is not None:
                m_t = upcast_fp32(m_spin).to(dtype) * mf
            elif m_raw is not None:
                m_t = c * torch.tanh(upcast_fp32(m_raw).to(dtype) / c) * mf
            else:
                m_t = sn.new_zeros(sn.shape[0], dtype=dtype)
            i_hund = F.softplus(upcast_fp32(self.hund_emb(z)).squeeze(-1).to(dtype) + upcast_fp32(self.hund_lin(snf)).squeeze(-1).to(dtype))
            e_hund = -0.5 * i_hund * urf.pow(2) * mf
            h_dir = upcast_fp32(self.proj_dir(snf)).to(dtype)
            h_cov = upcast_fp32(self.proj_cov(snf)).to(dtype)
            return e_hund, h_dir, h_cov, m_t, urf

    def forward(
        self,
        z: Tensor,
        sn: Tensor,
        u_rad: Tensor,
        m_spin: Tensor | None,
        m_raw: Tensor | None,
        pos: Tensor,
        send: Tensor | None,
        recv: Tensor | None,
        cart_shift: Tensor | None,
        atom_mask: Tensor,
        *,
        allow_mol: bool = True,
        csr_plan=None,
        cell: Tensor | None = None,
        edge_shift: Tensor | None = None,
        graph: Tensor | None = None,
        create_graph: bool = False,
    ) -> Tensor:
        n = pos.shape[0]
        if not allow_mol:
            return pos.new_zeros(n)
        with fp32_guard(pos.device, disable_tf32=True):
            mf = atom_mask.to(pos.dtype)
            ps = tuple(self.parameters())
            cn = node_chunk_size(pos.device) if pos.device.type == "mps" else NODE_CHUNK_ATOMS
            if not self.training and can_chunk_nodes(n, ps, min_atoms=cn, create_graph=create_graph):
                has_ms, has_mr = m_spin is not None, m_raw is not None
                sliced = [z, sn, u_rad, atom_mask] + ([m_spin] if has_ms else []) + ([m_raw] if has_mr else [])

                def _fn(_a0: int, _a1: int, *xs):
                    ms_ = xs[4] if has_ms else None
                    mr_ = xs[4 + int(has_ms)] if has_mr else None
                    return self._node_prologue(xs[0], xs[1], xs[2], ms_, mr_, xs[3], pos.dtype)

                e_hund, h_dir, h_cov, m_t, urf = chunked_node_apply(_fn, tuple(sliced), params=ps, chunk_atoms=cn, create_graph=create_graph)
            else:
                e_hund, h_dir, h_cov, m_t, urf = self._node_prologue(z, sn, u_rad, m_spin, m_raw, atom_mask, pos.dtype)

            sh_in = edge_shift if (cell is not None and cell.numel() > 0) else (cart_shift if (cart_shift is not None and cart_shift.numel() > 0) else None)
            use_metal = False
            use_triton = False
            if csr_plan is not None and pos.device.type == "mps":
                from lam4.kernels.metal_spin_exchange import can_use_metal_spin_exch, metal_spin_exchange_csr

                use_metal = can_use_metal_spin_exch(
                    pos, self.rad_dir.weight, self.rad_cov.weight, self.rank, self.basis.n_rbf,
                    create_graph=create_graph, shift_in=sh_in,
                )
            elif csr_plan is not None and pos.device.type == "cuda":
                from lam4.kernels.triton_spin_exchange import can_use_triton_spin_exch, triton_spin_exchange_csr

                use_triton = can_use_triton_spin_exch(
                    pos, self.rad_dir.weight, self.rad_cov.weight, self.rank, self.basis.n_rbf,
                    create_graph=create_graph, shift_in=sh_in,
                )
            if use_metal or use_triton:
                exch_fn = metal_spin_exchange_csr if use_metal else triton_spin_exchange_csr
                pair_sum = exch_fn(
                    pos, h_dir, h_cov, m_t, urf,
                    self.rad_dir.weight, self.rad_cov.weight, self.basis.centres(pos),
                    atom_mask, csr_plan, sh_in, cell, graph, self.r_cut, self.basis.sigma,
                )
            else:
                assert send is not None and recv is not None
                pair_sum = self._eval_edges_torch(pos, send, recv, cart_shift, atom_mask, h_dir, h_cov, m_t, urf)
            e_pair = -0.5 * pair_sum * mf

            gate = upcast_fp32(self.gate_raw).to(pos.dtype)
            return gate * (e_hund + e_pair) * mf
