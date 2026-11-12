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
from lam4.config import ModelConfig
from lam4.core import edge_terms
from lam4.core.edge_chunks import DEFAULT_CHUNK_EDGES, make_chunks
from lam4.core.edge_terms import build_edge_geom, edge_basis
from lam4.core.layout import FIRST, LATER
from lam4.core.radial import PairWarp, R_CUT
from lam4.core.spline_tables import DEFAULT_GRID
from lam4.core.stage import Stage
from lam4.core.stage_graph import EdgeGraph, NodeContext
from lam4.core.stage_remat import check_mode
from lam4.core.vector_bound import state_norm
from lam4.data.batch import GraphBatch
from lam4.heads import Readout
from lam4.kernels.csr_ops import device_supports_csr
from lam4.kernels.csr_plan import build_csr_plan
from lam4.physics import ChargeHead, DensityFeedback, ExactCoulomb, FieldFeedback, SpinDensityHead, group_domain, make_slots_static

N_ELEMENTS = 119


@dataclass
class TrunkOut:
    state: Tensor                      # [N, d] after the last stage (feedback, group context, state_norm, mask applied)
    vec: Tensor                        # [N, K, 3] raw additive vector stream
    cond: Tensor                       # [G, cond_dim] full conditioning (token, u, Q, S) of the stage FiLMs
    stage_terms: dict[str, Tensor]     # {"stage{l}": [N]} stage readouts (masked)
    stage_charges: list[Tensor] = field(default_factory=list)
    spin_density: Tensor | None = None
    backend: ExactCoulomb | None = None
    ten: Tensor | None = None          # [N, K2, 5] raw additive quadrupole stream
    cond_iso: Tensor | None = None     # [G, cond_dim] Conditioning.functional(token, u): the isolated-atom reference input (no Q, S)
    csr_plan: Any = None


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
        self.fb_field = nn.ModuleList([FieldFeedback(d, k, cfg.phi_basis) for _ in range(fp)])
        self.fb_density = nn.ModuleList([DensityFeedback(d, cfg.fb_hidden) for _ in range(fp)])

    def set_remat(self, remat: bool | str) -> None:
        mode = check_mode(remat)
        for st in self.stages:
            st.remat = mode

    def electro(self, domain: str, pbc: bool = False) -> bool:
        return domain == "molecule" and not pbc and self.cfg.long_range_molecule == "exact"

    def _can_csr(self, pos: Tensor, edge_scale, create_graph: bool) -> bool:
        return (
            not create_graph
            and edge_scale is None
            and pos.dtype == torch.float32
            and device_supports_csr(pos.device)
            and self.cfg.r_cut_near <= 0.0
            and self.cfg.n_heads == 4
            and self.cfg.head_dim == 16
            and not self.stages[0].tables.weight.requires_grad
            and not torch.compiler.is_compiling()
            and edge_terms.bspline_eval is edge_terms._ORIG_BSPLINE_EVAL
        )

    def edge_graph(self, b: GraphBatch, pos: Tensor, chunks: list | None, edge_scale, *, csr_plan=None, create_graph: bool = False) -> EdgeGraph:
        if self.edge_mode == "csr":
            if self._can_csr(pos, edge_scale, create_graph):
                if csr_plan is None or csr_plan.n_atoms != b.n_atoms or csr_plan.n_edges != b.n_edges or csr_plan.recv_ptr.device != pos.device:
                    csr_plan = build_csr_plan(b.edge_index[1], b.edge_index[0], b.n_atoms)
                return EdgeGraph.from_batch(b, self.pair_warp.node_rho(b.z), None, "csr", pos=pos, csr_plan=csr_plan, create_graph=False)
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

    def forward(self, b: GraphBatch, pos: Tensor, domain: str, *, chunks: list | None = None, edge_scale=None,
                csr_plan=None, create_graph: bool = False) -> TrunkOut:
        cfg, mask = self.cfg, b.atom_mask
        mf = mask.to(pos.dtype)
        gdom = group_domain(domain, b.pbc, b.frag_id is not None)
        cond = self.conditioning(b.token, b.u_flag, b.charge.to(pos.dtype), b.spin.to(pos.dtype), gdom)
        if b.frag_id is None:
            nc_cond = cond
        else:
            fg = torch.arange(b.frag_q.shape[0], device=b.graph.device, dtype=torch.long).clamp_max(b.n_graphs - 1).index_put_((b.frag_group,), b.graph)
            nc_cond = self.conditioning(b.token.index_select(0, fg), b.u_flag.index_select(0, fg), b.frag_q.to(pos.dtype), b.frag_s.to(pos.dtype), gdom)
        nc = NodeContext.from_batch(b, nc_cond)
        eg = self.edge_graph(b, pos, chunks, edge_scale, csr_plan=csr_plan, create_graph=create_graph)
        electro = self.electro(domain, b.pbc)
        cap = b.coulomb_cap if getattr(b, "coulomb_cap", None) is not None else (
            max(cfg.coulomb_cap, b.n_atoms) if b.n_graphs == 1 else cfg.coulomb_cap)
        if not electro:
            backend = None
        elif (
            self.edge_mode == "csr"
            and not create_graph
            and pos.dtype == torch.float32
            and pos.device.type == "mps"
            and not torch.compiler.is_compiling()
            and device_supports_csr(pos.device)
        ):
            from lam4.kernels.metal_coulomb import MetalExactCoulomb

            backend = MetalExactCoulomb(pos, mask, b.graph, b.n_graphs, cap, cfg.coulomb_sigma)
        elif not create_graph and b.n_graphs == 1 and cap > 512 and not torch.compiler.is_compiling():
            from lam4.physics.coulomb_exact import ChunkedExactCoulomb

            backend = ChunkedExactCoulomb(pos, mask, cfg.coulomb_sigma)
        else:
            backend = ExactCoulomb(pos, make_slots_static(mask, b.graph, b.n_graphs, cap), cfg.coulomb_sigma)
        s = self.embed(b.z) * mf.unsqueeze(-1)
        vec = s.new_zeros(b.n_atoms, cfg.k_vec, 3)
        ten = None
        terms: dict[str, Tensor] = {}
        q_stage: list[Tensor] = []
        m_k = None
        for l, st in enumerate(self.stages):
            s, vec, ten = st.core(s, vec, eg, nc, domain, ten=ten, return_ten=True)
            need_ro, need_fb = l < cfg.n_stages - 1, electro and l < cfg.feedback_points
            if need_ro or need_fb:
                sn = state_norm(s)
                if need_ro:
                    terms[f"stage{l}"] = self.readout.stage(l, sn, mask, normed=True)
                if need_fb:
                    q = self.charge_heads[l](sn, b.frag_group, b.frag_q.to(pos.dtype), mask, normed=True)
                    m_k = self.spin_heads[l](sn, b.frag_group, b.frag_s.to(pos.dtype), mask, normed=True)
                    phi, fld = backend.phi_field(q)
                    s, vec = self.fb_field[l](s, vec, phi, fld)
                    s = self.fb_density[l](s, q, m_k)
                    q_stage.append(q)
            s = st.finish(s, nc, gdom)
        return TrunkOut(s, vec, cond, terms, q_stage, m_k, backend, ten, self.conditioning.functional(b.token, b.u_flag), eg.csr_plan)
