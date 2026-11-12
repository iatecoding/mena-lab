"""model.py: `Lam4Model` = Trunk + final readout + charges / Coulomb / polarisation (molecules) + NLH repulsion, energy and conservative forces.

    out = model(batch, domain=None, create_graph=None, compute_forces=True, chunks=None, edge_scale=None, add_ref=False, direct_forces=False,
                compute_stress=False)

Energy (kcal/mol, the earlier Mena target space E_DFT - E_D3 - E_ref; the isolated-atom reference is learned inside the readout), per atom then per graph:
    e_i = readout terms (stage readouts + final MLP + iso reference, all masked)
        + 1/2 k_C q_i phi_i                      Coulomb of the final NQE charges, ExactCoulomb sigma = 2           (molecule, long_range_molecule == "exact")
        + E_pol,i                                -k_C (gamma p_i . E_i + alpha_i |E_i|^2 / 2)                       (molecule, cfg.polarization)
        + NLH repulsion (receiver sum)           c2_shift_rmin, the same tables as the 5 eV descriptor floor of the edge geometry   (both domains)
    E_g = sum_{i in g} e_i. Periodic batches (b.pbc) never take the open-boundary terms (no ExactCoulomb, group physics only per given fragment).
    The CRYSTAL domain is strictly extensive: trunk + readout + NLH only (no group context, no charges, no Coulomb;
    E(2x2x2) = 8 E(1x1x1) is tested). `add_ref=True` adds the frozen group reference E_Q[Q_g] + E_S[S_g] (float64, molecule only; earlier Mena `References` minus the
    per-element E0, which the learned iso term and the registry replace).
Forces = -dE/dpos by autograd; `create_graph` defaults to `self.training` (training needs the double backward; the remat units and the streaming op support it).
compute_stress=True (periodic batches, edge_mode "ref"): a symmetric strain eps [G, 3, 3] (zero, differentiable) maps pos -> pos (1 + eps_g) and
cell -> cell (1 + eps_g), so every edge vector (positions and the integer-shift images) is strained; out["stress"] [G, 3, 3] = (1/V_g) dE/d eps
in kcal/mol/A^3 (ASE convention, sigma = (1/V) dE/d eps), 0 for graphs with zero cell volume. Forces and stress come from ONE autograd call.
direct_forces=True adds out["forces_direct"] [N, 3] from the training-only DirectForceHead (no autograd force; pass compute_forces=False for a direct step:
energy + direct forces cost one forward + one first-order backward, no double backward).
Returns a dict: energy [G], energy_atom [N], forces [N, 3] | None, network_energy / electrostatic_energy / polarization_energy / repulsion_energy [G],
charges [N] | None (final NQE charges), dipole [G, 3] | None (sum_{i in g} (q_i r_i + p_i + p_ind,i)), alpha [N] | None (atomic polarisability A^3),
stage_charges, spin_density, terms {name: [N]} (the readout terms: feed `lam4.heads.cancellation_diagnostic`).
Static shapes, no host sync (ExactCoulomb is [G, cap, cap]: `check_batch` is the eager cap guard for the loader; `chunks` comes from the data pipeline).
"""
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
from torch import Tensor

from lam4.config import ModelConfig
from lam4.core.vector_bound import bound_vectors, state_norm
from lam4.data.batch import GraphBatch
from lam4.heads.force_head import DirectForceHead
from lam4.model.trunk import Trunk
from lam4.physics import COULOMB_K, GroupReference, Polarization, assert_capacity, group_domain
from lam4.physics.repulsion import nlh_atom_energy


class Lam4Model(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, *, remat: bool | str = "none", edge_mode: str = "ref") -> None:
        super().__init__()
        cfg = ModelConfig() if cfg is None else cfg
        self.cfg = cfg
        self.trunk = Trunk(cfg, remat, edge_mode)
        self.dipole_head = nn.Linear(cfg.k_vec, 1, bias=False)   # atomic dipole p_i = sum_k w_k u_ik, read by the polarisation (gamma zero-init)
        self.polarization = Polarization(cfg.d_scalar, cfg.k2, cfg.vec_c, cfg.aniso_polarizability) if cfg.polarization else None
        self.force_head = DirectForceHead(cfg.d_scalar, cfg.k_vec, cfg.vec_c) if cfg.direct_force_head else None
        self._ref = [GroupReference()]   # float64 tables outside the module tree: model.float() must not round them

    def set_remat(self, remat: bool | str) -> None:
        self.trunk.set_remat(remat)

    def train(self, mode: bool = True):
        if mode:
            self.unbake_inference()
        return super().train(mode)

    def bake_for_inference(self) -> "Lam4Model":
        self.eval()
        for st in self.trunk.stages:
            st.bake_for_inference()
        self.readout.bake_for_inference(self.trunk.conditioning)
        return self

    def unbake_inference(self) -> "Lam4Model":
        for st in self.trunk.stages:
            st.unbake_inference()
        self.readout.unbake_inference()
        return self

    @property
    def readout(self):
        return self.trunk.readout

    def check_batch(self, b: GraphBatch) -> None:
        """Eager guard for the data pipeline (host sync): no molecule above the ExactCoulomb cap, which would silently drop its overflow atoms."""
        if self.trunk.electro(b.domain, b.pbc):
            assert_capacity(b.atom_mask, b.graph, b.n_graphs, self.cfg.coulomb_cap)

    def group_reference(self, b: GraphBatch, domain: str | None = None) -> Tensor:
        """[G] float64 E_Q[Q_g] + E_S[S_g] of open molecules (exactly 0 for crystals, periodic batches and pad graphs)."""
        return self._ref[0](b.charge, b.spin, b.graph_mask, group_domain(b.domain if domain is None else domain, b.pbc))

    def forward(self, b: GraphBatch, domain: str | None = None, *, create_graph: bool | None = None, compute_forces: bool = True,
                chunks: list | None = None, edge_scale=None, add_ref: bool = False, direct_forces: bool = False, compute_stress: bool = False,
                return_stage_energies: bool = True, csr_plan=None) -> dict:
        domain = b.domain if domain is None else domain
        cg = self.training if create_graph is None else bool(create_graph)
        leaf = b.pos.detach().requires_grad_(True) if (compute_forces and not b.pos.requires_grad) else b.pos
        pos, eps, cell0 = leaf, None, b.cell
        if compute_stress:
            eps = leaf.new_zeros(b.n_graphs, 3, 3).requires_grad_(True)
            sym = 0.5 * (eps + eps.transpose(1, 2))
            pos = leaf + torch.einsum("ni,nij->nj", leaf, sym.index_select(0, b.graph))
            b = replace(b, pos=pos, cell=b.cell + torch.einsum("gki,gij->gkj", b.cell, sym))
        mf = b.atom_mask.to(pos.dtype)
        t = self.trunk(b, pos, domain, chunks=chunks, edge_scale=edge_scale, csr_plan=csr_plan, create_graph=cg)
        atomic, terms = self.readout.final(t.state, t.vec, b, t.cond_iso, t.stage_terms, ten=t.ten)
        e_atom = atomic
        zeros = atomic.new_zeros(b.n_graphs)
        e_coul = e_pol = e_rep = zeros
        q = dipole = alpha = alpha_aniso = None
        if t.backend is not None:
            s_ro = state_norm(t.state)
            q_grp = b.graph if getattr(b, "frag_group", None) is None else b.frag_group
            q_tot = (b.charge if getattr(b, "frag_q", None) is None else b.frag_q).to(pos.dtype)
            q = self.trunk.charge_heads[-1](s_ro, q_grp, q_tot, b.atom_mask, normed=True)
            u = bound_vectors(t.vec, self.cfg.vec_c)
            p = torch.einsum("nkc,k->nc", u, self.dipole_head.weight[0])
            if self.polarization is not None:
                phi, fld = t.backend.phi_field(q)
                pol_atom, p_ind, alpha, alpha_aniso = self.polarization(s_ro, p, fld, mf, normed=True, return_alpha=True, ten=t.ten, return_aniso=True)
                p_tot = p + p_ind
                e_pol = self.readout.graph_energy(pol_atom, b)
                e_atom = e_atom + pol_atom
            else:
                phi = t.backend.phi(q)
                p_tot = p
            dip_atom = (q.unsqueeze(-1) * pos + p_tot) * mf.unsqueeze(-1)
            dipole = dip_atom.sum(0, keepdim=True) if b.n_graphs == 1 else pos.new_zeros(b.n_graphs, 3).index_add(0, b.graph, dip_atom)
            coul_atom = (0.5 * COULOMB_K) * q * (phi * mf)
            e_coul = self.readout.graph_energy(coul_atom, b)
            e_atom = e_atom + coul_atom
        if self.cfg.repulsion != "none":
            rep_atom = nlh_atom_energy(b, pos, csr_plan=t.csr_plan, create_graph=cg)
            e_rep = self.readout.graph_energy(rep_atom, b)
            e_atom = e_atom + rep_atom
        energy = self.readout.graph_energy(e_atom, b)
        ref_e = self.group_reference(b, domain) if add_ref else None
        if ref_e is not None:
            energy = (energy.double() + ref_e).to(pos.dtype)
        stage_energies = None
        if return_stage_energies and self.cfg.stage_e_weight > 0 and t.stage_terms:
            cum = e_atom - atomic + terms.get("iso", 0.0)
            stage_e = []
            for l in range(self.cfg.n_stages - 1):
                cum = cum + terms[f"stage{l}"]
                stage_e.append(self.readout.graph_energy(cum, b))
            stage_energies = torch.stack(stage_e, dim=-1)
            if ref_e is not None:
                stage_energies = (stage_energies.double() + ref_e.unsqueeze(-1)).to(pos.dtype)
        forces = f_direct = None
        if direct_forces:
            if self.force_head is None:
                raise ValueError("direct_forces=True needs ModelConfig.direct_force_head")
            f_direct = self.force_head(t.state, t.vec, b.atom_mask)
        stress = None
        if compute_forces or compute_stress:
            wrt = ([leaf] if compute_forces else []) + ([eps] if compute_stress else [])
            grads = torch.autograd.grad(energy.sum(), wrt, create_graph=cg, retain_graph=cg)
            if compute_forces:
                forces = -grads[0]
            if compute_stress:
                a, bb, c = cell0.unbind(1)
                vol = (a * torch.cross(bb, c, dim=-1)).sum(-1).abs()   # unstrained volume (eps = 0); 0 for open and pad graphs
                ok = vol > 0
                stress = grads[-1] * (ok.to(vol.dtype) / torch.where(ok, vol, torch.ones_like(vol))).view(-1, 1, 1)
        return {"energy": energy, "energy_atom": e_atom, "forces": forces, "forces_direct": f_direct, "stress": stress, "network_energy": self.readout.graph_energy(atomic, b),
                "electrostatic_energy": e_coul, "polarization_energy": e_pol, "repulsion_energy": e_rep, "charges": q,
                "dipole": dipole, "alpha": alpha, "alpha_aniso": alpha_aniso, "stage_energies": stage_energies,
                "stage_charges": t.stage_charges, "spin_density": t.spin_density, "terms": terms}
