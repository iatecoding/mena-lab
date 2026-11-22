"""cloud_overlap.py: abstract learned E(3)-equivariant electron-cloud overlap & Pauli repulsion (6.0 A).

Replaces hardcoded 1/r^5 quadrupoles and hardcoded Born-Mayer walls with:
1. Strictly non-negative l=0 Pauli overlap E_Pauli >= 0 using cached non-negative cubic B-spline knot weights
   (Phi_pos @ softplus(W_raw), bypassing collocation inversion M^-1) and C3 envelope at r_cut = 6.0 A.
2. Learned l=1,2 anisotropic cloud overlap (dir/lon/tr/y2 gathers; optional `dual_csr_gather_fn` hook for 0 MB edge
   memory; pure PyTorch `index_add_` fallback materializes [E, C] intermediates: ~153 floats/edge -> ~0.68 GB fwd
   and ~1.37 GB under create_graph=True at G4 N=32k, E=1.12M in fp32).
3. Macroscopic smeared EFG coupling -1/3 k_C <D_i^(2), G_i^lr> in the l=2 STF basis.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.core.consts import is_fake, proxy_tracing, tracing
from lam4.core.harmonics import y2
from lam4.core.radial import GaussianBasis, N_RBF, R_CUT, R_ON, c3_envelope, fp32_guard, resolve_baked, stop_grad, upcast_fp32
from lam4.core.remat import NODE_CHUNK_ATOMS, can_chunk_nodes, chunked_node_apply, node_chunk_size
from lam4.core.spline_tables import _weights
from lam4.core.vector_bound import bound_vectors, chan_linear
from lam4.physics.coulomb_exact import COULOMB_K


def _pos_knot_basis(grid: int, n_rbf: int, r_cut: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Evaluate positive Gaussian basis directly at spline knots [G+3, n_rbf] >= 0 (no collocation overshoot)."""
    h = float(r_cut) / float(grid)
    knots = (torch.arange(-1, grid + 2, dtype=torch.float64) * h).clamp(0.0, float(r_cut))
    basis = GaussianBasis(n_rbf, r_cut)
    return basis.gaussians(knots).to(device=device, dtype=dtype)


def _eval_pos_bspline(x: Tensor, coef: Tensor, r_cut: float, grid: int) -> Tensor:
    """Evaluate C2 cubic B-spline with knot coefficients [G+3, C] cleanly with exact derivatives."""
    h = float(r_cut) / float(grid)
    xc = x.clamp(0.0, float(r_cut))
    xf = xc / h
    g = stop_grad(xf.floor()).to(torch.long).clamp(0, grid - 1)
    t = xf - g.to(xc.dtype)
    w0, w1, w2, w3 = _weights(t, 0)
    return (
        w0.unsqueeze(-1) * coef.index_select(0, g)
        + w1.unsqueeze(-1) * coef.index_select(0, g + 1)
        + w2.unsqueeze(-1) * coef.index_select(0, g + 2)
        + w3.unsqueeze(-1) * coef.index_select(0, g + 3)
    )


class CloudOverlapHead(nn.Module):
    """Full-6.0 A learned equivariant Pauli (>=0) + l=1,2 anisotropic overlap + EFG head."""

    def __init__(self, d_model: int, k_vec: int, k_ten: int, *, k_ex: int = 16, k_an: int = 8, k_t: int = 4,
                 spline_grid: int = 128, r_on: float = R_ON, r_cut: float = R_CUT, vec_c: float = 2.0,
                 dual_csr_gather_fn=None) -> None:
        super().__init__()
        self.k_ex, self.k_an, self.k_t = int(k_ex), int(k_an), int(k_t)
        self.spline_grid, self.r_on, self.r_cut, self.vec_c = int(spline_grid), float(r_on), float(r_cut), float(vec_c)
        self.dual_csr_gather_fn = dual_csr_gather_fn
        self.register_buffer("knot_b", _pos_knot_basis(self.spline_grid, N_RBF, self.r_cut, torch.float32, torch.device("cpu")), persistent=False)
        n_in = d_model + 2  # [s_norm, q0, u_rad]

        # l=0 non-negative Pauli branch
        self.w_pauli_raw = nn.Parameter(torch.full((N_RBF, self.k_ex), -1.0))
        self.pauli_amp = nn.Sequential(nn.Linear(n_in, 64), nn.SiLU(), nn.Linear(64, self.k_ex))
        nn.init.constant_(self.pauli_amp[-1].bias, -3.0)

        # l=1,2 anisotropic cloud overlap filters (4 channel groups matching Dual-CSR gathers)
        n_an_filt = self.k_an * 3 + self.k_t  # dir, tr, lon (each k_an) + y2 (k_t)
        self.w_aniso_raw = nn.Parameter(torch.randn(N_RBF, n_an_filt) * 1e-2)
        self.d0_head = nn.Sequential(nn.Linear(n_in, 64), nn.SiLU(), nn.Linear(64, self.k_an + self.k_t))
        self.c1_proj = nn.Linear(k_vec, self.k_an, bias=False)
        self.c1_gate = nn.Sequential(nn.Linear(n_in, 32), nn.SiLU(), nn.Linear(32, self.k_an))
        self.c2_proj = nn.Linear(k_ten, self.k_t, bias=False)
        self.c2_gate = nn.Sequential(nn.Linear(n_in, 32), nn.SiLU(), nn.Linear(32, self.k_t))
        nn.init.normal_(self.d0_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.d0_head[-1].bias)
        nn.init.zeros_(self.c1_gate[-1].weight)
        nn.init.zeros_(self.c1_gate[-1].bias)
        nn.init.zeros_(self.c2_gate[-1].weight)
        nn.init.zeros_(self.c2_gate[-1].bias)

        # Macroscopic l=2 EFG coupling D_i^(2) : G_i^lr
        self.efg_proj = nn.Linear(k_ten, 1, bias=False)
        self.efg_gate = nn.Sequential(nn.Linear(n_in, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.efg_gate[-1].weight)
        nn.init.zeros_(self.efg_gate[-1].bias)
        self._baked: dict[str, Tensor] | None = None

    def train(self, mode: bool = True):
        if mode:
            self._baked = None
        return super().train(mode)

    def _apply(self, fn):
        had_baked = self._baked is not None
        self._baked = None
        out = super()._apply(fn)
        if had_baked and not self.training and not self.w_pauli_raw.is_meta and not tracing():
            self.bake_for_inference()
        return out

    @torch.no_grad()
    def bake_for_inference(self) -> None:
        if self.w_pauli_raw.is_meta:
            return
        from lam4.kernels.csr_plan import coef_to_horner_poly

        w_p = F.softplus(upcast_fp32(self.w_pauli_raw))
        w_a = upcast_fp32(self.w_aniso_raw)
        w_all = torch.cat([w_p, w_a], dim=-1)
        knot_b = upcast_fp32(self.knot_b).to(device=w_all.device, dtype=w_all.dtype)
        coef_all = (knot_b @ w_all).detach().contiguous()
        poly = coef_to_horner_poly(coef_all, self.r_cut).detach().contiguous()
        self._baked = {"coef_all": coef_all, "poly": poly}

    def unbake_inference(self) -> None:
        self._baked = None

    def _get_baked(self, like: Tensor) -> dict[str, Tensor] | None:
        return resolve_baked(self, lambda bk: bk["poly"], self.w_pauli_raw, like)

    def _eval_edges_torch(
        self, pos: Tensor, send: Tensor, recv: Tensor, cart_shift: Tensor, atom_mask: Tensor,
        coef_all: Tensor, a_p: Tensor, c1: Tensor, d1: Tensor, c2: Tensor, d2: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        n, ke, ka, kt = pos.shape[0], self.k_ex, self.k_an, self.k_t
        sym = torch.compiler.is_compiling() or proxy_tracing() or is_fake(send)
        e_cnt = 0 if sym else int(send.shape[0])
        if not sym and e_cnt == 0:
            return (
                pos.new_zeros(n, ke),
                pos.new_zeros(n, ka),
                pos.new_zeros(n, ka, 3),
                pos.new_zeros(n, kt),
                pos.new_zeros(n, kt, 5),
            )
        chunk = 32768 if (not sym and e_cnt > 65536) else max(1, e_cnt)
        agg_p = pos.new_zeros(n, ke)
        m_dir_sca = pos.new_zeros(n, ka)
        m_vec = pos.new_zeros(n, ka, 3)
        m_y2_s = pos.new_zeros(n, kt)
        m_y2_t = pos.new_zeros(n, kt, 5)
        has_sh = cart_shift.numel() > 0
        steps = ((0, 0),) if e_cnt == 0 else ((s0, min(e_cnt, s0 + chunk)) for s0 in range(0, e_cnt, chunk))
        for s0, s1 in steps:
            se = send if e_cnt == 0 else send[s0:s1]
            re = recv if e_cnt == 0 else recv[s0:s1]
            sh = (cart_shift[s0:s1] if e_cnt != 0 else cart_shift).to(pos.dtype) if has_sh else 0.0
            d_e = pos.index_select(0, se) - pos.index_select(0, re) + sh
            r2 = d_e.pow(2).sum(-1)
            real_e = (atom_mask.index_select(0, re) & atom_mask.index_select(0, se) & (r2 > 1e-12)).to(pos.dtype)
            r = torch.sqrt(r2 + 1e-12)
            u_e = d_e / r.unsqueeze(-1)
            chi = c3_envelope(r, self.r_on, self.r_cut) * real_e
            rad_all = _eval_pos_bspline(r, coef_all, self.r_cut, self.spline_grid) * chi.unsqueeze(-1)
            r_pauli, f_dir, f_tr, f_lon, f_y2 = rad_all.split([ke, ka, ka, ka, kt], dim=-1)
            c1_s, d1_s, d2_s, c2_s = c1.index_select(0, se), d1.index_select(0, se), d2.index_select(0, se), c2.index_select(0, se)
            y2_e = y2(u_e)
            lon_dot = (c1_s * u_e.unsqueeze(1)).sum(-1)
            msg_sca = torch.cat([
                r_pauli * a_p.index_select(0, se),
                f_dir * lon_dot,
                f_y2 * (c2_s * y2_e.unsqueeze(1)).sum(-1),
            ], dim=-1)
            m_sca = pos.new_zeros(n, ke + ka + kt).index_add(0, re, msg_sca)
            ap_c, ds_c, ys_c = m_sca.split([ke, ka, kt], dim=-1)
            agg_p, m_dir_sca, m_y2_s = agg_p + ap_c, m_dir_sca + ds_c, m_y2_s + ys_c
            msg_vec = (f_dir * d1_s + f_lon * lon_dot).unsqueeze(-1) * u_e.unsqueeze(1) + f_tr.unsqueeze(-1) * c1_s
            m_vec = m_vec + pos.new_zeros(n, ka, 3).index_add(0, re, msg_vec)
            m_y2_t = m_y2_t + pos.new_zeros(n, kt, 5).index_add(0, re, (f_y2 * d2_s).unsqueeze(-1) * y2_e.unsqueeze(1))
        return agg_p, m_dir_sca, m_vec, m_y2_s, m_y2_t

    def _node_prologue(
        self, sn: Tensor, vec: Tensor, ten: Tensor, q0: Tensor, u_rad: Tensor, mf: Tensor, dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        with fp32_guard(sn.device, disable_tf32=True):
            snf = upcast_fp32(sn)
            q0f, urf = upcast_fp32(q0).to(dtype) * mf, upcast_fp32(u_rad).to(dtype) * mf
            feats = torch.cat([snf, q0f.unsqueeze(-1), urf.unsqueeze(-1)], dim=-1)
            a_p = F.softplus(self.pauli_amp(feats)) * mf.unsqueeze(-1)
            u_vec = bound_vectors(upcast_fp32(vec), self.vec_c)
            u_ten = bound_vectors(upcast_fp32(ten), self.vec_c)
            d_all = self.d0_head(feats) * mf.unsqueeze(-1)
            d1, d2 = d_all[:, :self.k_an], d_all[:, self.k_an:]
            c1 = chan_linear(u_vec, upcast_fp32(self.c1_proj.weight).to(dtype)) * self.c1_gate(feats).unsqueeze(-1) * mf.view(-1, 1, 1)
            c2 = chan_linear(u_ten, upcast_fp32(self.c2_proj.weight).to(dtype)) * self.c2_gate(feats).unsqueeze(-1) * mf.view(-1, 1, 1)
            return a_p, c1, d1, c2, d2

    def _efg_term(
        self, sn: Tensor, ten: Tensor, q0: Tensor, u_rad: Tensor, mf: Tensor, efg_lr: Tensor, dtype: torch.dtype,
    ) -> Tensor:
        with fp32_guard(sn.device, disable_tf32=True):
            snf = upcast_fp32(sn)
            q0f, urf = upcast_fp32(q0).to(dtype) * mf, upcast_fp32(u_rad).to(dtype) * mf
            feats = torch.cat([snf, q0f.unsqueeze(-1), urf.unsqueeze(-1)], dim=-1)
            u_ten = bound_vectors(upcast_fp32(ten), self.vec_c)
            d2_efg = chan_linear(u_ten, upcast_fp32(self.efg_proj.weight).to(dtype)).squeeze(-2)
            d2_efg = d2_efg * self.efg_gate(feats) * mf.unsqueeze(-1)
            return -(COULOMB_K / 3.0) * (d2_efg * upcast_fp32(efg_lr).to(dtype)).sum(-1) * mf

    def forward(self, sn: Tensor, vec: Tensor, ten: Tensor, q0: Tensor, u_rad: Tensor,
                pos: Tensor, send: Tensor | None, recv: Tensor | None, cart_shift: Tensor | None, atom_mask: Tensor,
                efg_lr: Tensor | None = None, *, csr_plan=None, cell: Tensor | None = None,
                edge_shift: Tensor | None = None, graph: Tensor | None = None, create_graph: bool = False) -> dict[str, Tensor]:
        """Return per-atom e_pauli_atom (>=0), e_aniso_atom, e_efg_atom, and combined energy_atom."""
        with fp32_guard(pos.device, disable_tf32=True):
            mf = atom_mask.to(pos.dtype)
            n = pos.shape[0]
            bk = self._get_baked(sn)
            if bk is not None:
                coef_all = bk["coef_all"].to(dtype=pos.dtype)
                poly = bk["poly"].to(dtype=pos.dtype)
            else:
                knot_b = upcast_fp32(self.knot_b).to(device=pos.device, dtype=pos.dtype)
                w_all = torch.cat([F.softplus(upcast_fp32(self.w_pauli_raw).to(pos.dtype)), upcast_fp32(self.w_aniso_raw).to(pos.dtype)], dim=-1)
                coef_all = knot_b @ w_all
                poly = None

            ps = tuple(self.parameters())
            cn = node_chunk_size(pos.device) if pos.device.type == "mps" else NODE_CHUNK_ATOMS
            use_chunks = (not self.training) and can_chunk_nodes(n, ps, min_atoms=cn, create_graph=create_graph)
            if use_chunks:
                a_p, c1, d1, c2, d2 = chunked_node_apply(
                    lambda _a0, _a1, sn_, v_, t_, q_, u_, m_: self._node_prologue(sn_, v_, t_, q_, u_, m_, pos.dtype),
                    (sn, vec, ten, q0, u_rad, mf), params=ps, chunk_atoms=cn, create_graph=create_graph,
                )
            else:
                a_p, c1, d1, c2, d2 = self._node_prologue(sn, vec, ten, q0, u_rad, mf, pos.dtype)

            ka, kt, ke = self.k_an, self.k_t, self.k_ex
            sh_in = edge_shift if (cell is not None and cell.numel() > 0) else (cart_shift if (cart_shift is not None and cart_shift.numel() > 0) else None)
            use_metal = False
            use_triton = False
            if csr_plan is not None and self.dual_csr_gather_fn is None and pos.device.type == "mps":
                from lam4.kernels.csr_plan import coef_to_horner_poly
                from lam4.kernels.metal_cloud_overlap import can_use_metal_cloud_ov, metal_cloud_overlap_csr

                use_metal = can_use_metal_cloud_ov(pos, coef_all, ke, ka, kt, create_graph=create_graph, shift_in=sh_in)
            elif csr_plan is not None and self.dual_csr_gather_fn is None and pos.device.type == "cuda":
                from lam4.kernels.csr_plan import coef_to_horner_poly
                from lam4.kernels.triton_cloud_overlap import can_use_triton_cloud_ov, triton_cloud_overlap_csr

                use_triton = can_use_triton_cloud_ov(pos, coef_all, ke, ka, kt, create_graph=create_graph, shift_in=sh_in)
            if use_metal or use_triton:
                if poly is None:
                    poly = coef_to_horner_poly(coef_all, self.r_cut)
                csr_fn = metal_cloud_overlap_csr if use_metal else triton_cloud_overlap_csr
                agg_p, m_dir_sca, m_vec, m_y2_s, m_y2_t = csr_fn(
                    pos, poly, a_p, c1, d1, c2, d2, atom_mask, csr_plan, sh_in, cell, graph, self.r_cut, self.r_on,
                )
                e_pauli = 0.5 * (a_p * agg_p).sum(-1) * mf
                e_aniso = 0.5 * (
                    (c1 * m_vec).sum((-2, -1))
                    - (d1 * m_dir_sca).sum(-1)
                    + (c2 * m_y2_t).sum((-2, -1))
                    + (d2 * m_y2_s).sum(-1)
                ) * mf
            elif self.dual_csr_gather_fn is not None:
                assert send is not None and recv is not None
                sh = cart_shift.to(pos.dtype) if (cart_shift is not None and cart_shift.numel() > 0) else 0.0
                d_e = pos.index_select(0, send) - pos.index_select(0, recv) + sh
                r2 = d_e.pow(2).sum(-1)
                real_e = (atom_mask.index_select(0, recv) & atom_mask.index_select(0, send) & (r2 > 1e-12)).to(pos.dtype)
                r = torch.sqrt(r2 + 1e-12)
                u_e = d_e / r.unsqueeze(-1)
                chi = c3_envelope(r, self.r_on, self.r_cut) * real_e
                r_pauli = _eval_pos_bspline(r, coef_all[:, :ke], self.r_cut, self.spline_grid) * chi.unsqueeze(-1)
                agg_p = pos.new_zeros(n, ke).index_add(0, recv, r_pauli * a_p.index_select(0, send))
                e_pauli = 0.5 * (a_p * agg_p).sum(-1) * mf
                m_dir_vec, m_dir_sca, m_11_tr, m_11_lon, m_y2_t, m_y2_s = self.dual_csr_gather_fn(
                    r, u_e, chi, coef_all[:, ke:], c1, d1, c2, d2, send, recv)
                e_aniso = 0.5 * (
                    (c1 * m_dir_vec).sum((-2, -1))
                    - (d1 * m_dir_sca).sum(-1)
                    + (c1 * (m_11_tr + m_11_lon)).sum((-2, -1))
                    + (c2 * m_y2_t).sum((-2, -1))
                    + (d2 * m_y2_s).sum(-1)
                ) * mf
            else:
                assert send is not None and recv is not None
                sh_t = cart_shift if cart_shift is not None else pos.new_zeros(0)
                agg_p, m_dir_sca, m_vec, m_y2_s, m_y2_t = self._eval_edges_torch(
                    pos, send, recv, sh_t, atom_mask, coef_all, a_p, c1, d1, c2, d2,
                )
                e_pauli = 0.5 * (a_p * agg_p).sum(-1) * mf
                e_aniso = 0.5 * (
                    (c1 * m_vec).sum((-2, -1))
                    - (d1 * m_dir_sca).sum(-1)
                    + (c2 * m_y2_t).sum((-2, -1))
                    + (d2 * m_y2_s).sum(-1)
                ) * mf

            # 3. EFG coupling
            if efg_lr is not None:
                if use_chunks:
                    e_efg = chunked_node_apply(
                        lambda _a0, _a1, sn_, t_, q_, u_, m_, eg_: self._efg_term(sn_, t_, q_, u_, m_, eg_, pos.dtype),
                        (sn, ten, q0, u_rad, mf, efg_lr), params=ps, chunk_atoms=cn, create_graph=create_graph,
                    )[0]
                else:
                    e_efg = self._efg_term(sn, ten, q0, u_rad, mf, efg_lr, pos.dtype)
            else:
                e_efg = pos.new_zeros(n)

            return {"energy_atom": e_pauli + e_aniso + e_efg, "e_pauli_atom": e_pauli, "e_aniso_atom": e_aniso, "e_efg_atom": e_efg}
