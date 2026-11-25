"""lam6_physics_wire.py: modular Track 2C LAM-6 physics wiring for Lam4Model.

Keeps `lam4/model/model.py` compact (<= 400 LOC) while connecting:
  - `ScreeningGate` (branchless CUDAGraph-safe modulation on periodic batches)
  - `OpenMultipoleBackend` (open molecules) / `BatchedSPME` (periodic crystals)
  - `SQEPQEqHead` (variational fixed-K_fix implicit-adjoint SQE + Thole PQEq)
  - `CloudOverlapHead` (learned E_Pauli >= 0 + l=1,2 anisotropic overlap + EFG coupling)
  - `EDA_CalibHead` & `EmpiricalBayesOOD` & X-Ray virial / force decomposition
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.config import ModelConfig
from lam4.kernels.triton_cloud_overlap import has_triton_cloud_ov
from lam4.kernels.triton_spin_exchange import has_triton_spin_exch
from lam4.core.angular import l2_mat6
from lam4.core.consts import proxy_tracing
from lam4.core.radial import fp32_guard, stop_grad, upcast_fp32
from lam4.core.remat import can_chunk_nodes, chunked_node_apply, node_chunk_size
from lam4.core.vector_bound import bound_vectors, chan_linear, state_norm
from lam4.data.batch import GraphBatch
from lam4.physics import (
    COULOMB_K,
    BatchedSPME,
    CloudOverlapHead,
    EDA_CalibHead,
    EmpiricalBayesOOD,
    OpenMultipoleBackend,
    SQEPQEqHead,
    ScreeningGate,
    SpinExchangeHead,
    atomic_strain_metrics,
    compute_eda_forces,
    fragment_com_positions,
    select_grid_bucket,
)


class GapHead(nn.Module):
    """Intensive attention-pooled + softmin HOMO-LUMO gap [alpha, beta] predictor (eV, strictly 0 on crystals)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hid = min(128, max(32, (int(d_model) // 2) & ~1))
        self.in_proj = nn.Linear(int(d_model), hid)
        self.att_logits = nn.Linear(hid, 2)
        self.att_val = nn.Linear(hid, hid)
        self.local_gap = nn.Linear(hid, 2)
        self.out_proj = nn.Linear(hid, 2)
        nn.init.constant_(self.local_gap.bias, 3.0)
        nn.init.constant_(self.out_proj.bias, 2.0)

    def forward(self, sn: Tensor, graph: Tensor, atom_mask: Tensor, graph_mask: Tensor, n_graphs: int, *, allow_mol: bool) -> Tensor:
        if not allow_mol:
            return sn.new_zeros(n_graphs, 2)
        with fp32_guard(sn.device, disable_tf32=True):
            snf = upcast_fp32(sn)
            h = F.silu(self.in_proj(snf))
            mf = atom_mask.to(snf.dtype).unsqueeze(-1)
            lg = self.att_logits(h).masked_fill(~atom_mask.unsqueeze(-1), -1.0e9)
            mx = stop_grad(lg.new_full((n_graphs, 2), -1.0e9).scatter_reduce(0, graph.unsqueeze(-1).expand(-1, 2), lg, reduce="amax", include_self=True))
            ew = torch.exp(lg - mx.index_select(0, graph)) * mf
            z_w = ew.new_zeros(n_graphs, 2).index_add(0, graph, ew).clamp_min(1e-12)
            dh = h.shape[-1] // 2
            val = self.att_val(h).unflatten(-1, (2, dh))
            g_att = (val.new_zeros(n_graphs, 2, dh).index_add(0, graph, ew.unsqueeze(-1) * val) / z_w.unsqueeze(-1)).flatten(-2)

            beta = 2.0
            g_loc = F.softplus(self.local_gap(h))
            xl = (-beta * g_loc).masked_fill(~atom_mask.unsqueeze(-1), -1.0e9)
            mxl = stop_grad(xl.new_full((n_graphs, 2), -1.0e9).scatter_reduce(0, graph.unsqueeze(-1).expand(-1, 2), xl, reduce="amax", include_self=True))
            se = xl.new_zeros(n_graphs, 2).index_add(0, graph, torch.exp(xl - mxl.index_select(0, graph)) * mf).clamp_min(1e-12)
            ng = xl.new_zeros(n_graphs, 1).index_add(0, graph, mf).clamp_min(1.0)
            g_softmin = -(mxl + torch.log(se) - torch.log(ng)) / beta
            return F.softplus(self.out_proj(g_att) + g_softmin) * graph_mask.to(snf.dtype).unsqueeze(-1)


class Lam6PhysicsHeads(nn.Module):
    """Parameter container for LAM-6 unified physics + X-Ray heads (routed to AdamW)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_scalar
        self.screening_gate = (
            ScreeningGate(d, bias_crystal=-4.2, bias_molecule=3.0, skip_thresh=cfg.charge_gate_skip)
            if (cfg.spme_batched or cfg.sqe_pqeq)
            else None
        )
        self.sqe_head = (
            SQEPQEqHead(
                d, cfg.k_vec, cfg.k2,
                n_lots=cfg.n_xc_tokens,
                sigma=cfg.coulomb_sigma,
                k_fix=cfg.sqe_k_fix,
                vec_c=cfg.vec_c,
            )
            if cfg.sqe_pqeq
            else None
        )
        self.cloud_head = (
            CloudOverlapHead(d, cfg.k_vec, cfg.k2, vec_c=cfg.vec_c)
            if cfg.cloud_overlap
            else None
        )
        self.eda_head = EDA_CalibHead(d, n_terms=5) if cfg.eda_calib_head else None
        self.ood_head = EmpiricalBayesOOD() if cfg.eda_calib_head else None
        self.exch_head = SpinExchangeHead(d, r_cut=cfg.r_cut, charge_bound=cfg.charge_bound) if cfg.spin_exchange else None
        self.quad_l2_head = nn.Linear(cfg.k2, 1, bias=False) if cfg.quad_l2_head else None
        if self.quad_l2_head is not None:
            nn.init.zeros_(self.quad_l2_head.weight)
        self.gap_head = GapHead(d) if cfg.gap_head else None


def eval_lam6_physics(
    heads: Lam6PhysicsHeads,
    cfg: ModelConfig,
    trunk,
    t,
    b: GraphBatch,
    pos: Tensor,
    atomic: Tensor,
    rep_atom: Tensor,
    use_ro_chunks: bool,
    *,
    domain: str | None = None,
    q_precomputed: Tensor | None = None,
    return_multipoles: bool = False,
    return_eda: bool = False,
    return_xray: bool = False,
    return_sqe_residual: bool = False,
    return_aux_heads: bool = True,
    create_graph: bool = False,
) -> dict[str, Any]:
    """Evaluate LAM-6 SQE/PQEq, BatchedSPME, CloudOverlap, SpinExchange, QuadL2, GapHead, and EDA/X-Ray terms."""
    mf = b.atom_mask.to(pos.dtype)
    eff_dom = b.domain if domain is None else domain
    allow_mol = (eff_dom == "molecule") and (not b.pbc)
    allow_exch = (eff_dom == "molecule") and ((not b.pbc) or bool(getattr(b, "periodic_electro", False)))
    already_normed = bool(cfg.state_norm)
    s_ro = t.state if (already_normed or use_ro_chunks) else state_norm(t.state)
    sn = t.state if already_normed else (state_norm(t.state) if use_ro_chunks else s_ro)
    q0 = pos.new_zeros(b.n_atoms)
    if heads.sqe_head is not None or heads.cloud_head is not None or heads.eda_head is not None:
        if q_precomputed is not None:
            q0 = q_precomputed
        else:
            q_grp = b.graph if getattr(b, "frag_group", None) is None else b.frag_group
            q_tot = (b.charge if getattr(b, "frag_q", None) is None else b.frag_q).to(pos.dtype)
            q0 = trunk.charge_heads[-1](s_ro, q_grp, q_tot, b.atom_mask, normed=(already_normed or not use_ro_chunks))
    u_rad = t.spin_desc[0] if t.spin_desc is not None else (t.spin_density.abs() if t.spin_density is not None else pos.new_zeros(b.n_atoms))
    m_raw = t.spin_desc[1] if t.spin_desc is not None else None
    csr_base = (t.csr_plan is not None and not create_graph and pos.dtype == torch.float32)
    can_csr_ex = csr_base and (
        pos.device.type == "mps"
        or (pos.device.type == "cuda" and has_triton_spin_exch(pos.device))
    )
    can_csr_ov = csr_base and (
        pos.device.type == "mps"
        or (pos.device.type == "cuda" and has_triton_cloud_ov(pos.device))
    )
    exch_needs_edges = (
        heads.exch_head is not None
        and allow_exch
        and (
            not can_csr_ex
            or heads.exch_head.rad_dir.weight.requires_grad
            or heads.exch_head.rad_cov.weight.requires_grad
            or heads.exch_head.rank != 16
            or heads.exch_head.basis.n_rbf != 32
        )
    )
    ov_needs_edges = (
        heads.cloud_head is not None
        and (
            not can_csr_ov
            or heads.cloud_head.dual_csr_gather_fn is not None
            or heads.cloud_head.w_pauli_raw.requires_grad
            or heads.cloud_head.w_aniso_raw.requires_grad
            or (heads.cloud_head.k_ex, heads.cloud_head.k_an, heads.cloud_head.k_t) != (16, 8, 4)
        )
    )
    need_pair_tensors = bool(heads.sqe_head is not None or exch_needs_edges or ov_needs_edges)
    if t.eg_edges is not None:
        send, recv, cart_shift = t.eg_edges
    elif need_pair_tensors:
        send, recv, cart_shift = b.edge_index[0], b.edge_index[1], b.cart_shift()
    else:
        send = recv = cart_shift = None

    g_at = g_mean = None
    if b.pbc and heads.screening_gate is not None:
        g_at, g_mean = heads.screening_gate(sn, b.atom_mask, pbc=True)

    be = None
    if heads.sqe_head is not None:
        if not b.pbc:
            be = t.backend if isinstance(t.backend, OpenMultipoleBackend) else OpenMultipoleBackend.from_batch(
                b, pos=pos, sigma=cfg.coulomb_sigma
            )
        elif cfg.spme_batched and cfg.long_range_crystal != "none":
            if b.spme_grid is not None:
                grid = b.spme_grid
            elif not proxy_tracing():
                grid = select_grid_bucket(b.cell, cfg.mesh_spacing)
            else:
                grid = (12, 12, 12)
            spme_active = True
            if (not heads.training) and heads.screening_gate is not None and g_mean is not None and (not proxy_tracing()) and (not torch.compiler.is_compiling()):
                if not (b.pos.is_cuda and torch.cuda.is_current_stream_capturing()):
                    spme_active = not heads.screening_gate.should_skip(g_mean)
            be = BatchedSPME(
                pos, b.cell, b.graph, b.atom_mask, b.graph_mask,
                sigma=cfg.coulomb_sigma, grid_shape=grid, order=4, active=spme_active,
            )

    e_sqe_atom = pos.new_zeros(b.n_atoms)
    e_perm_atom = pos.new_zeros(b.n_atoms)
    e_ind_atom = pos.new_zeros(b.n_atoms)
    e_ov_atom = pos.new_zeros(b.n_atoms)
    q_out = q0
    p_tot_out = p_perm_out = p_ind_out = fld_out = efg_lr = alpha_t = sqe_res_out = sqe_rel_out = None

    if heads.sqe_head is not None and be is not None:
        tok_xc = getattr(b, "xc_id", None)
        lot_id = (b.token if tok_xc is None else tok_xc).clamp(0, max(cfg.n_xc_tokens - 1, 0))
        frag_com = (
            fragment_com_positions(pos, b.frag_group, b.atom_mask)
            if getattr(b, "frag_group", None) is not None and getattr(b, "frag_id", None) is not None
            else None
        )
        sqe = heads.sqe_head(
            be, pos, b.cell if b.pbc else None, b.graph, b.atom_mask,
            send, recv, cart_shift, sn, t.vec, t.ten, q0, lot_id,
            e_ext=getattr(b, "e_ext", None), frag_com_pos=frag_com,
            compute_alpha=bool(return_multipoles), n_graphs=b.n_graphs,
            return_residual=bool(return_sqe_residual),
        )
        gate = g_at if (b.pbc and g_at is not None) else 1.0
        e_sqe_atom = gate * sqe["energy_atom"]
        e_perm_atom = gate * sqe["e_perm_atom"]
        e_ind_atom = gate * sqe["e_ind_atom"]
        q_out = sqe["charges"]
        p_tot_out = sqe["atomic_dipoles"]
        p_perm_out = sqe["atomic_dipoles_perm"]
        p_ind_out = sqe["atomic_dipoles_ind"]
        fld_out = sqe["field"]
        alpha_t = sqe["alpha_tensor"]
        sqe_res_out = sqe.get("sqe_residual")
        sqe_rel_out = sqe.get("sqe_rel_residual")
        if sqe.get("efg_l2") is not None:
            efg_lr = (g_at.unsqueeze(-1) * sqe["efg_l2"]) if (b.pbc and g_at is not None) else sqe["efg_l2"]

    if heads.cloud_head is not None:
        ov = heads.cloud_head(
            sn, t.vec, t.ten, q0, u_rad, pos, send, recv, cart_shift, b.atom_mask, efg_lr=efg_lr,
            csr_plan=t.csr_plan, cell=(b.cell if b.pbc else None), edge_shift=(b.edge_shift if b.pbc else None),
            graph=b.graph, create_graph=create_graph,
        )
        e_ov_atom = ov["energy_atom"]

    e_exch_atom = (
        heads.exch_head(
            b.z, sn, u_rad, t.spin_density, m_raw, pos, send, recv, cart_shift, b.atom_mask,
            allow_mol=True, csr_plan=t.csr_plan, cell=(b.cell if b.pbc else None),
            edge_shift=(b.edge_shift if b.pbc else None), graph=b.graph, create_graph=create_graph,
        )
        if (heads.exch_head is not None and allow_exch)
        else pos.new_zeros(b.n_atoms)
    )
    quad_l2_atom = None
    if return_aux_heads and heads.quad_l2_head is not None and t.ten is not None:
        def _quad_fn(_a0: int, _a1: int, ten_c: Tensor, mf_c: Tensor) -> Tensor:
            with fp32_guard(ten_c.device, disable_tf32=True):
                q5 = chan_linear(bound_vectors(upcast_fp32(ten_c), cfg.vec_c), upcast_fp32(heads.quad_l2_head.weight)).squeeze(-2)
                return torch.stack(l2_mat6(q5), dim=-1).to(pos.dtype) * mf_c.unsqueeze(-1)

        q_ps = tuple(heads.quad_l2_head.parameters())
        q_cn = node_chunk_size(pos.device)
        if not heads.training and can_chunk_nodes(b.n_atoms, q_ps, min_atoms=q_cn, create_graph=create_graph):
            quad_l2_atom = chunked_node_apply(_quad_fn, (t.ten, mf), params=q_ps, chunk_atoms=q_cn, create_graph=create_graph)[0]
        else:
            quad_l2_atom = _quad_fn(0, b.n_atoms, t.ten, mf)
    homo_gap = None
    if return_aux_heads and heads.gap_head is not None:
        homo_gap = heads.gap_head(sn, b.graph, b.atom_mask, b.graph_mask, b.n_graphs, allow_mol=True) if allow_mol else pos.new_zeros(b.n_graphs, 2)

    dipole = None
    if cfg.sqe_pqeq:
        if p_tot_out is None:
            p_perm_out = pos.new_zeros(b.n_atoms, 3)
            p_tot_out = p_perm_out
            p_ind_out = p_perm_out
        dip_atom = (q_out.unsqueeze(-1) * pos + p_tot_out) * mf.unsqueeze(-1)
        dipole = dip_atom.sum(0, keepdim=True) if b.n_graphs == 1 else pos.new_zeros(b.n_graphs, 3).index_add(0, b.graph, dip_atom)

    e_eda = delta_eda = eda_raw_terms = eda_m_proj = xray_out = None
    if heads.eda_head is not None and (return_eda or return_xray):
        e_phys = torch.stack([atomic, e_perm_atom, e_ind_atom, e_ov_atom, rep_atom], dim=-1)
        eda_out = heads.eda_head(sn, e_phys, b.atom_mask, detach_inputs=(not return_xray), return_terms=True)
        e_eda, delta_eda = eda_out["eda_calib_terms_5"], eda_out["delta"]
        eda_raw_terms, eda_m_proj = eda_out["eda_raw_terms_5"], eda_out["m_proj"]
        if return_xray:
            eda_f = compute_eda_forces(e_eda, pos, b.atom_mask, retain_graph=True)
            w_lr = be.virial_atom(q_out, p_tot_out) if be is not None else pos.new_zeros(b.n_atoms, 3, 3)
            xray_out = {
                "e_phys": e_phys,
                "e_eda": e_eda,
                "delta_eda": delta_eda,
                "eda_raw_terms_5": eda_raw_terms,
                "eda_forces": eda_f["eda_forces"],
                "eda_force_cancel_ratio": eda_f["eda_force_cancel_ratio"],
                "w_lr": w_lr,
                "strain": atomic_strain_metrics(w_lr, eda_f["forces_sum"], e_phys, b.atom_mask),
            }

    return {
        "e_sqe_atom": e_sqe_atom,
        "e_ov_atom": e_ov_atom,
        "e_exch_atom": e_exch_atom,
        "e_perm_atom": e_perm_atom,
        "e_ind_atom": e_ind_atom,
        "q": q_out,
        "dipole": dipole,
        "alpha_tensor": alpha_t,
        "p_tot": p_tot_out,
        "p_perm": p_perm_out,
        "p_ind": p_ind_out,
        "fld": fld_out,
        "efg_lr": efg_lr,
        "quad_l2_atom": quad_l2_atom,
        "homo_gap": homo_gap,
        "g_at": g_at,
        "g_mean": g_mean,
        "e_eda": e_eda,
        "delta_eda": delta_eda,
        "eda_raw_terms_5": eda_raw_terms,
        "eda_m_proj": eda_m_proj,
        "xray": xray_out,
        "sqe_residual": sqe_res_out,
        "sqe_rel_residual": sqe_rel_out,
    }
