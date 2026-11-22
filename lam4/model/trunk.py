"""trunk.py: the lam4 trunk = embedding + conditioning + 3 stages with the readout / feedback / group-context points of the earlier Mena model's forward pass.

Loop (earlier Mena model.py order, one stage l):
    s, vec = stage.core(s, vec, eg, nc, domain)                      edge op, invariants, angular branch, SwiGLU expert, vector gate
    sn = rms(s)                                                      ONE normalisation shared by the readout and the charge heads
    l < S-1:               atomic term stage_l = Readout.stage(l, sn)                   reads the PRE-feedback state
    l < feedback_points    q_l = ChargeHead_l(sn);  m_l = SpinDensityHead_l(sn)         (electrostatic path only, see below)
                           phi, E = ExactCoulomb.phi_field(q_l);  s, vec = FieldFeedback_l(s, vec, phi, E);  s = DensityFeedback_l(s, q_l, m_l)
    s = stage.finish(s)                                              GroupContext (molecule) -> state_norm -> atom mask
ELECTROSTATIC PATH (`electro`): domain == "molecule", an OPEN batch (b.pbc False) and cfg.long_range_molecule == "exact". Crystals and every periodic
batch run the strictly extensive trunk: no charges, no feedback, no Coulomb (ExactCoulomb has no images; gated SPME is a later work package, `long_range_crystal`
is not read here). Group physics (Q / S conditioning, group context) follows `group_domain(domain, pbc, frag_id given)`: periodic molecules get it only per
finite fragment; the node-MLP expert always follows `domain`.
Pad atoms: the embedding is masked, every stage ends with the atom mask, charge heads and ExactCoulomb give pad atoms exactly 0.
Differences to the earlier Mena trunk, all justified: the element table is a free nn.Embedding (the earlier Mena trunk: fixed descriptors + residual: `embed.weight = phi W^T + resid` maps exactly);
spin / charge enter through the per-stage FiLM of `Conditioning` (the earlier Mena trunk: additive input embeddings; the earlier Mena trunk's charge embedding is off in the the earlier molecular model preset);
the per-atom edge weight is the 4-head bounded QK (the earlier Mena trunk's ChemicalEdgeWeights only through the `edge_scale` COMPAT argument: a tensor [E] or a callable pos -> [E],
because the earlier Mena trunk's scale depends on the positions through chi and the forces need that gradient).
Edge path: `edge_mode` "ref" (oracle, materialises [E, ~700]) | "stream" (chunked; pass `chunks` from the data pipeline, otherwise the plan is built here with a host sync: tests only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from lam4.cond import Conditioning
from lam4.kernels.triton_cloud_overlap import has_triton_cloud_ov
from lam4.kernels.triton_spin_exchange import has_triton_spin_exch
from lam4.config import ModelConfig
from lam4.core import edge_terms
from lam4.core.consts import is_fake, proxy_tracing
from lam4.core.edge_chunks import DEFAULT_CHUNK_EDGES, make_chunks
from lam4.core.edge_terms import build_edge_geom, edge_basis
from lam4.core.inv_scale import compute_dsum
from lam4.core.layout import FIRST, LATER
from lam4.core.radial import PairWarp, R_CUT
from lam4.core.spline_tables import DEFAULT_GRID
from lam4.core.remat import NODE_CHUNK_ATOMS, node_chunk_size
from lam4.core.stage import Stage
from lam4.core.stage_graph import EdgeGraph, NodeContext
from lam4.core.stage_remat import check_mode
from lam4.core.vector_bound import state_norm
from lam4.data.batch import GraphBatch
from lam4.heads import Readout, SpinDescriptorHead
from lam4.kernels import csr_ops
from lam4.kernels.csr_ops import device_supports_csr
from lam4.kernels.triton_csr_bwd2 import has_triton_csr_bwd2
from lam4.kernels.csr_plan import build_csr_plan
from lam4.kernels.triton_coulomb import TritonExactCoulomb, has_triton_coulomb
from lam4.physics import (ChargeHead, DensityFeedback, ExactCoulomb, FieldFeedback, OpenMultipoleBackend,
                          SpinDensityHead, group_domain, make_slots_static)
from lam4.physics.coulomb_exact import ChunkedExactCoulomb, resolve_coulomb_meta

N_ELEMENTS = 119


@dataclass
class TrunkOut:
    state: Tensor                      # [N, d] after the last stage (feedback, group context, state_norm, mask applied)
    vec: Tensor                        # [N, K, 3] raw additive vector stream
    cond: Tensor                       # [G, cond_dim] full conditioning (token, u, Q, S) of the stage FiLMs
    stage_terms: dict[str, Tensor]     # {"stage{l}": [N]} stage readouts (masked)
    stage_charges: list[Tensor] = field(default_factory=list)
    stage_spins: list[Tensor] = field(default_factory=list)
    spin_density: Tensor | None = None
    backend: ExactCoulomb | None = None
    ten: Tensor | None = None          # [N, K2, 5] raw additive quadrupole stream
    cond_iso: Tensor | None = None     # [G, cond_dim] Conditioning.functional(token, u): the isolated-atom reference input (no Q, S)
    csr_plan: Any = None
    inv2: Tensor | None = None         # [N, inv_dim] scaled Stage-(S-1) invariants for DirectInvBypass
    cond_atom: Tensor | None = None    # [N, cond_dim] per-atom full conditioning (with Q, S when applicable)
    cond_lot_atom: Tensor | None = None  # [N, cond_dim] per-atom factorized LoT conditioning (no Q, S)
    spin_desc: tuple[Tensor, Tensor] | None = None  # (u_rad [N] >= 0, m_raw [N])
    eg_edges: tuple[Tensor, Tensor, Tensor] | None = None  # (send, recv, cart_shift) from EdgeGraph
    dsum: Tensor | None = None         # [N] neighbor envelope sum for coord_inv_scale / env_mod / iso_atom_n1


class Trunk(nn.Module):
    def __init__(self, cfg: ModelConfig, remat: bool | str = "none", edge_mode: str = "ref") -> None:
        super().__init__()
        d, k, fp = cfg.d_scalar, cfg.k_vec, cfg.feedback_points
        self.cfg, self.edge_mode = cfg, edge_mode
        self.embed = nn.Embedding(N_ELEMENTS, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.pair_warp = PairWarp(N_ELEMENTS, cfg.r_cut)
        self.conditioning = Conditioning(cfg)
        self.stages = nn.ModuleList([Stage(cfg, FIRST if i == 0 else LATER, i, remat) for i in range(cfg.n_stages)])
        self.readout = Readout(cfg)
        self.charge_heads = nn.ModuleList([ChargeHead(d, cfg.charge_bound) for _ in range(fp + 1)])   # the last one is the final charge head
        self.spin_heads = nn.ModuleList([SpinDensityHead(d, cfg.charge_bound) for _ in range(fp)])
        self.fb_field = nn.ModuleList([FieldFeedback(d, k, cfg.phi_basis, ten_rank=(cfg.k2 if cfg.use_l2_efg else 0)) for _ in range(fp)])
        self.fb_density = nn.ModuleList([DensityFeedback(d, cfg.fb_hidden, radical_density=cfg.two_channel_spin) for _ in range(fp)])
        self.spin_desc_head = SpinDescriptorHead(d) if cfg.two_channel_spin else None

    def set_remat(self, remat: bool | str) -> None:
        mode = check_mode(remat)
        for st in self.stages:
            st.remat = mode

    def electro(self, domain: str, pbc: bool = False) -> bool:
        return domain == "molecule" and not pbc and self.cfg.long_range_molecule == "exact"

    def _can_csr(self, pos: Tensor, edge_scale, create_graph: bool) -> bool:
        # LAM-6 O1..O9: CUDA Triton CSR bwd2 lifts the create_graph / trainable-table restrictions (double backward in-kernel).
        bwd2 = pos.device.type == "cuda" and has_triton_csr_bwd2(pos.device) and getattr(csr_ops, "has_csr_custom_op", lambda: False)()
        return (
            (not create_graph or bwd2)
            and edge_scale is None
            and csr_ops.dtype_supports_csr(pos)
            and device_supports_csr(pos.device)
            and self.cfg.r_cut_near <= 0.0
            and self.cfg.n_heads == 4
            and self.cfg.head_dim == 16
            and (not self.stages[0].tables.weight.requires_grad or bwd2)
            and (not torch.compiler.is_compiling() or getattr(csr_ops, "has_csr_custom_op", lambda: False)())
            and edge_terms.bspline_eval is edge_terms._ORIG_BSPLINE_EVAL
        )

    def edge_graph(self, b: GraphBatch, pos: Tensor, chunks: list | None, edge_scale, *, csr_plan=None, create_graph: bool = False) -> EdgeGraph:
        if self.edge_mode == "csr":
            if self._can_csr(pos, edge_scale, create_graph):
                need_build = csr_plan is None or csr_plan.n_atoms != b.n_atoms or csr_plan.n_edges != b.n_edges or csr_plan.recv_ptr.device != pos.device
                if not (need_build and torch.compiler.is_compiling() and pos.device.type in ("mps", "cpu")):
                    if need_build:
                        em = b.atom_mask.index_select(0, b.edge_index[1]) & b.atom_mask.index_select(0, b.edge_index[0])
                        csr_plan = build_csr_plan(b.edge_index[1], b.edge_index[0], b.n_atoms, edge_mask=em)
                    return EdgeGraph.from_batch(b, self.pair_warp.node_rho(b.z), None, "csr", pos=pos, csr_plan=csr_plan, create_graph=create_graph)
            eff_mode = "stream" if chunks is not None else "ref"
        else:
            eff_mode = self.edge_mode
        if eff_mode == "stream" and chunks is None:
            chunks = make_chunks(b.edge_index[1], b.edge_index[0], b.n_atoms, DEFAULT_CHUNK_EDGES)   # host sync: pass `chunks` from the data pipeline
        eg = EdgeGraph.from_batch(b, self.pair_warp.node_rho(b.z), chunks, eff_mode, pos=pos, create_graph=create_graph)
        eg.edge_scale = edge_scale(pos) if callable(edge_scale) else edge_scale   # COMPAT: the earlier Mena trunk's per-edge scale depends on the positions (through chi)
        if eff_mode == "ref":
            eg.geom = build_edge_geom(pos, eg.recv, eg.send, eg.shift, eg.rho, self.stages[0].edge_cfg, eg.z)
            if self.cfg.spline_grid == DEFAULT_GRID and self.cfg.r_cut == R_CUT:
                eg.phi = edge_basis(eg.geom[1], self.cfg.r_cut)
        return eg

    def _electro_backend(self, b: GraphBatch, pos: Tensor, *, create_graph: bool = False):
        """Build per-step electrostatic backend once, caching Morton/CSR/slot metadata across Stage 0, Stage 1, and Readout."""
        cfg, mask = self.cfg, b.atom_mask
        if cfg.sqe_pqeq or (cfg.use_l2_efg and create_graph):
            return OpenMultipoleBackend.from_batch(b, pos=pos, sigma=cfg.coulomb_sigma)
        eff_g, eff_cap = resolve_coulomb_meta(b, cfg.coulomb_cap)
        eff_graph = torch.zeros_like(b.graph) if (eff_g == 1 and b.n_graphs > 1) else b.graph
        exp_cap = getattr(b, "coulomb_cap", None)
        if not create_graph and pos.dtype in (torch.float32, torch.float16, torch.bfloat16) and has_triton_coulomb(pos.device):
            return TritonExactCoulomb(pos, mask, eff_graph, eff_g, exp_cap, cfg.coulomb_sigma)
        if (
            self.edge_mode == "csr"
            and not create_graph
            and pos.dtype == torch.float32
            and pos.device.type == "mps"
            and device_supports_csr(pos.device)
        ):
            from lam4.kernels.metal_coulomb import MetalExactCoulomb, has_metal_coulomb

            if has_metal_coulomb():
                return MetalExactCoulomb(pos, mask, eff_graph, eff_g, exp_cap, cfg.coulomb_sigma)
        sym = torch.compiler.is_compiling() or proxy_tracing() or is_fake(pos)
        if not create_graph and not sym and (eff_cap > 512 or (eff_g > 1 and eff_g * eff_cap * eff_cap > 262_144)):
            return ChunkedExactCoulomb(pos, mask, cfg.coulomb_sigma, graph=eff_graph, n_graphs=eff_g, cap=exp_cap)
        if cfg.use_l2_efg:
            return OpenMultipoleBackend.from_batch(b, pos=pos, sigma=cfg.coulomb_sigma)
        return ExactCoulomb(pos, make_slots_static(mask, eff_graph, eff_g, eff_cap), cfg.coulomb_sigma)

    def forward(self, b: GraphBatch, pos: Tensor, domain: str, *, chunks: list | None = None, edge_scale=None,
                csr_plan=None, create_graph: bool = False) -> TrunkOut:
        cfg, mask = self.cfg, b.atom_mask
        w_dt = torch.float64 if pos.dtype == torch.float64 else torch.float32
        mf = mask.to(w_dt)
        gdom = group_domain(domain, b.pbc, b.frag_id is not None)
        if cfg.factorized_lot:
            tok_xc = getattr(b, "xc_id", None)
            tok_xc = b.token if tok_xc is None else tok_xc
            paw_id = getattr(b, "paw_id", None)
            u_atom = getattr(b, "u_atom", None)
            apply_qs = (not b.pbc or b.frag_id is not None) if cfg.ffn_mode != "domain_experts" else (gdom == "molecule")
            cond = self.conditioning(tok_xc, b.u_flag, b.charge.to(w_dt), b.spin.to(w_dt), gdom, paw_id=paw_id, apply_qs=apply_qs)
            c_full_a, c_func_a = self.conditioning.per_atom(
                tok_xc, b.u_flag, b.frag_q.to(w_dt), b.frag_s.to(w_dt), gdom, b.graph,
                paw_id=paw_id, u_atom=u_atom, apply_qs=apply_qs, charge_graph=b.frag_group,
            )
            cond_iso = self.conditioning.functional(tok_xc, b.u_flag, paw_id=paw_id)
            nc = NodeContext.from_batch(b, cond)
        else:
            c_full_a = c_func_a = None
            cond = self.conditioning(b.token, b.u_flag, b.charge.to(w_dt), b.spin.to(w_dt), gdom)
            if b.frag_id is None:
                nc_cond = cond
            else:
                fg = torch.arange(b.frag_q.shape[0], device=b.graph.device, dtype=torch.long).clamp_max(b.n_graphs - 1).index_put_((b.frag_group,), b.graph)
                nc_cond = self.conditioning(b.token.index_select(0, fg), b.u_flag.index_select(0, fg), b.frag_q.to(w_dt), b.frag_s.to(w_dt), gdom)
            cond_iso = self.conditioning.functional(b.token, b.u_flag)
            nc = NodeContext.from_batch(b, nc_cond)
        eg = self.edge_graph(b, pos, chunks, edge_scale, csr_plan=csr_plan, create_graph=create_graph)
        need_dsum = bool(cfg.coord_inv_scale or cfg.ffn_mode != "domain_experts" or cfg.iso_atom_n1)
        use_stage0_dsum = bool(
            need_dsum and eg.mode == "csr" and not create_graph
            and self.stages[0].remat in ("none", "block")
            and "_compiled_core" not in self.stages[0].__dict__
            and "_compiled_agg" not in self.stages[0].__dict__
        )
        dsum = None if use_stage0_dsum else (compute_dsum(eg, self.stages[0].edge_cfg) if need_dsum else None)
        electro = self.electro(domain, b.pbc)
        backend = self._electro_backend(b, pos, create_graph=create_graph) if electro else None
        s = self.embed(b.z).to(w_dt) * mf.unsqueeze(-1)
        vec = s.new_zeros(b.n_atoms, cfg.k_vec, 3)
        ten = inv2 = None
        terms: dict[str, Tensor] = {}
        q_stage: list[Tensor] = []
        m_stage: list[Tensor] = []
        m_k = None
        cn_ro = node_chunk_size(pos.device) if pos.device.type == "mps" else NODE_CHUNK_ATOMS
        use_ro_chunks = (
            not torch.compiler.is_compiling()
            and not create_graph
            and not self.training
            and b.n_atoms > cn_ro
            and not any(p.requires_grad for p in self.parameters())
        )
        use_ext = bool(cfg.factorized_lot or cfg.coord_inv_scale or cfg.ffn_mode != "domain_experts" or cfg.direct_inv_bypass)
        for l, st in enumerate(self.stages):
            want_inv = bool(l == cfg.n_stages - 1 and cfg.direct_inv_bypass)
            want_s0_dsum = bool(l == 0 and use_stage0_dsum)
            if use_ext:
                out_st = st.core(s, vec, eg, nc, domain, ten=ten, return_ten=True,
                                 cond_atom=c_full_a, cond_lot_atom=c_func_a, dsum=dsum, return_inv=want_inv,
                                 return_dsum=want_s0_dsum)
                if want_s0_dsum:
                    dsum = out_st[-1]
                    out_st = out_st[:-1]
                if want_inv:
                    s, vec, ten, inv2 = out_st
                else:
                    s, vec, ten = out_st
            else:
                out_st = st.core(s, vec, eg, nc, domain, ten=ten, return_ten=True, return_dsum=want_s0_dsum)
                if want_s0_dsum:
                    dsum = out_st[-1]
                    out_st = out_st[:-1]
                s, vec, ten = out_st
            need_ro, need_fb = l < cfg.n_stages - 1, electro and l < cfg.feedback_points
            if need_ro or need_fb:
                if not need_fb and gdom == "crystal" and st.use_state_norm:
                    s = st.finish(s, nc, gdom)
                    if need_ro:
                        terms[f"stage{l}"] = self.readout.stage(l, s, mask, normed=True)
                    continue
                sn = s if use_ro_chunks else state_norm(s)
                if need_ro:
                    terms[f"stage{l}"] = self.readout.stage(l, sn, mask, normed=not use_ro_chunks)
                if need_fb:
                    q = self.charge_heads[l](sn, b.frag_group, b.frag_q.to(w_dt), mask, normed=not use_ro_chunks)
                    m_k = self.spin_heads[l](sn, b.frag_group, b.frag_s.to(w_dt), mask, normed=not use_ro_chunks)
                    if self.cfg.use_l2_efg and ten is not None and hasattr(backend, "phi_field_multipole"):
                        phi, fld, efg = backend.phi_field_multipole(q, need_efg=True)
                        s, vec, ten = self.fb_field[l](s, vec, phi, fld, ten=ten, efg=efg)
                    else:
                        phi, fld = backend.phi_field(q)
                        s, vec = self.fb_field[l](s, vec, phi, fld)
                    u_rad_fb = self.spin_desc_head(sn, mask, normed=not use_ro_chunks)[0] if (self.cfg.two_channel_spin and self.spin_desc_head is not None) else None
                    s = self.fb_density[l](s, q, m_k, u_rad_fb)
                    q_stage.append(q)
                    m_stage.append(m_k)
            s = st.finish(s, nc, gdom)
        spin_desc = self.spin_desc_head(s, mask, normed=cfg.state_norm) if self.spin_desc_head is not None else None
        allow_mol = (domain == "molecule") and (not b.pbc)
        csr_base = (
            eg.mode == "csr"
            and not create_graph
            and pos.dtype == torch.float32
            and not self.stages[0].tables.weight.requires_grad
        )
        can_csr_ov = csr_base and (
            pos.device.type == "mps"
            or (pos.device.type == "cuda" and has_triton_cloud_ov(pos.device))
        )
        can_csr_ex = csr_base and (
            pos.device.type == "mps"
            or (pos.device.type == "cuda" and has_triton_spin_exch(pos.device))
        )
        ov_needs_edges = bool(cfg.cloud_overlap and not can_csr_ov)
        exch_needs_edges = bool(cfg.spin_exchange and allow_mol and not can_csr_ex)
        need_edges = bool(cfg.sqe_pqeq or cfg.spme_batched or cfg.eda_calib_head or exch_needs_edges or ov_needs_edges)
        eg_edges = (eg.send, eg.recv, b.cart_shift() if (eg.mode == "csr" or eg.shift.numel() == 0) else eg.shift) if need_edges else None
        return TrunkOut(s, vec, cond, terms, q_stage, m_stage, m_k, backend, ten, cond_iso, eg.csr_plan, inv2, c_full_a, c_func_a, spin_desc, eg_edges, dsum)
