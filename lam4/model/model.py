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
from lam4.core.consts import cuda_cap_ok, tracing
from lam4.core.remat import NODE_CHUNK_ATOMS, can_chunk_nodes, chunked_node_apply, node_chunk_size
from lam4.core.vector_bound import bound_vectors, state_norm
from lam4.data.batch import GraphBatch
from lam4.heads.force_head import DirectForceHead
from lam4.kernels.triton_coulomb import warm_triton_coulomb
from lam4.kernels.triton_d3 import warm_d3_tables
from lam4.model.lam6_physics_wire import Lam6PhysicsHeads, eval_lam6_physics
from lam4.model.trunk import Trunk
from lam4.physics import COULOMB_K, GroupReference, Polarization, assert_capacity, group_domain
from lam4.physics.repulsion import nlh_atom_energy, warm_device_tables

if hasattr(torch._dynamo.config, "trace_autograd_ops"):
    torch._dynamo.config.trace_autograd_ops = True


@torch._dynamo.disable
def _make_grad_leaf(pos: Tensor) -> Tensor:
    return pos.detach().requires_grad_(True)


class Lam4Model(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, *, remat: bool | str = "none", edge_mode: str = "ref") -> None:
        super().__init__()
        if hasattr(torch._dynamo.config, "trace_autograd_ops"):
            torch._dynamo.config.trace_autograd_ops = True
        cfg = ModelConfig() if cfg is None else cfg
        self.cfg = cfg
        self.trunk = Trunk(cfg, remat, edge_mode)
        self.dipole_head = nn.Linear(cfg.k_vec, 1, bias=False)   # atomic dipole p_i = sum_k w_k u_ik, read by the polarisation (gamma zero-init)
        self.polarization = Polarization(cfg.d_scalar, cfg.k2, cfg.vec_c, cfg.aniso_polarizability) if (cfg.polarization and not cfg.sqe_pqeq) else None
        self.force_head = DirectForceHead(cfg.d_scalar, cfg.k_vec, cfg.vec_c) if cfg.direct_force_head else None
        use_l6_phys = bool(cfg.sqe_pqeq or cfg.spme_batched or cfg.cloud_overlap or cfg.eda_calib_head or cfg.spin_exchange or cfg.quad_l2_head or cfg.gap_head)
        self.lam6_phys = Lam6PhysicsHeads(cfg) if use_l6_phys else None
        self._ref = [GroupReference()]   # float64 tables outside the module tree: model.float() must not round them

    def set_remat(self, remat: bool | str) -> None:
        self.trunk.set_remat(remat)

    def train(self, mode: bool = True):
        if mode:
            self.unbake_inference()
        return super().train(mode)

    def _apply(self, fn):
        out = super()._apply(fn)
        w0 = self.trunk.embed.weight
        if w0.device.type not in ("meta", "mps") and hasattr(self, "_ref") and self._ref:
            self._ref[0].to(device=w0.device)
        if not self.training and self.trunk.stages[0]._baked is not None and not tracing():
            self.warm_caches()
        return out

    @torch.no_grad()
    def warm_caches(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None,
                    n_atoms: int | tuple[int, ...] = ()) -> "Lam4Model":
        w0 = self.trunk.embed.weight
        dev = w0.device if device is None else torch.device(device)
        dt = w0.dtype if dtype is None else dtype
        if dev.type == "meta" or tracing():
            return self
        warm_device_tables(dev, dt)
        if dev.type == "cuda" and torch.cuda.is_available():
            cuda_cap_ok(dev)
            warm_d3_tables(dev)
            warm_triton_coulomb(dev, n_atoms)
            try:
                from lam4.kernels.triton_nlh import _nlh_cb_jet_f32
                _nlh_cb_jet_f32(dev)
            except Exception:
                pass
        return self

    def bake_for_inference(self) -> "Lam4Model":
        self.eval()
        for st in self.trunk.stages:
            st.bake_for_inference()
        self.readout.bake_for_inference(self.trunk.conditioning)
        if self.lam6_phys is not None and self.lam6_phys.cloud_head is not None:
            self.lam6_phys.cloud_head.bake_for_inference()
        return self.warm_caches()

    def unbake_inference(self) -> "Lam4Model":
        for st in self.trunk.stages:
            st.unbake_inference()
        self.readout.unbake_inference()
        if self.lam6_phys is not None and self.lam6_phys.cloud_head is not None:
            self.lam6_phys.cloud_head.unbake_inference()
        return self

    @property
    def readout(self):
        return self.trunk.readout

    def check_batch(self, b: GraphBatch) -> None:
        """Eager guard for the data pipeline (host sync): no molecule above the ExactCoulomb cap, which would silently drop its overflow atoms."""
        if self.trunk.electro(b.domain, b.pbc):
            assert_capacity(b.atom_mask, b.graph, b.n_graphs, self.cfg.coulomb_cap)

    def group_reference(self, b: GraphBatch, domain: str | None = None) -> Tensor:
        """[G] float64 E_Q[Q_g] + E_S[S_g] of open/fragmented molecules (float32 on MPS; 0 for crystals, unfragmented periodic, pad graphs)."""
        dom = b.domain if domain is None else domain
        dev = b.charge.device
        ref = self._ref[0]

        def _call(q: Tensor, s: Tensor, gm: Tensor, gd: str) -> Tensor:
            if dev.type == "mps":
                return ref(q.cpu(), s.cpu(), gm.cpu(), gd).to(dtype=torch.float32).to(device=dev)
            return ref(q, s, gm, gd)

        if b.frag_id is None:
            return _call(b.charge, b.spin, b.graph_mask, group_domain(dom, b.pbc))
        gdom = group_domain(dom, b.pbc, True)
        fg = torch.arange(b.frag_q.shape[0], device=b.graph.device, dtype=torch.long).clamp_max(b.n_graphs - 1).index_put_((b.frag_group,), b.graph)
        frag_ref = _call(b.frag_q, b.frag_s, b.graph_mask.index_select(0, fg), gdom)
        return torch.zeros(b.n_graphs, dtype=frag_ref.dtype, device=frag_ref.device).index_add(0, fg, frag_ref)

    def forward(self, b: GraphBatch, domain: str | None = None, *, create_graph: bool | None = None, compute_forces: bool = True,
                chunks: list | None = None, edge_scale=None, add_ref: bool = False, direct_forces: bool = False, compute_stress: bool = False,
                return_stage_energies: bool = True, csr_plan=None, return_multipoles: bool = False,
                return_eda: bool = False, return_xray: bool = False, return_sqe_residual: bool = False,
                return_phi: bool | None = None, return_aux_heads: bool = True) -> dict:
        domain = b.domain if domain is None else domain
        cg = self.training if create_graph is None else bool(create_graph)
        if compute_forces and not b.pos.requires_grad:
            leaf = _make_grad_leaf(b.pos) if (cg and torch.compiler.is_compiling()) else b.pos.detach().requires_grad_(True)
        else:
            leaf = b.pos
        pos, eps, cell0 = leaf, None, b.cell
        if compute_stress:
            eps = leaf.new_zeros(b.n_graphs, 3, 3).requires_grad_(True)
            sym = 0.5 * (eps + eps.transpose(1, 2))
            pos = leaf + torch.einsum("ni,nij->nj", leaf, sym.index_select(0, b.graph))
            b = replace(b, pos=pos, cell=b.cell + torch.einsum("gki,gij->gkj", b.cell, sym))
        mf = b.atom_mask.to(pos.dtype)
        t = self.trunk(b, pos, domain, chunks=chunks, edge_scale=edge_scale, csr_plan=csr_plan, create_graph=cg)
        want_phi = bool(self.cfg.expose_readout_phi if return_phi is None else (self.cfg.expose_readout_phi and return_phi))
        if t.inv2 is not None or t.cond_lot_atom is not None or want_phi:
            ro_res = self.readout.final(
                t.state, t.vec, b, t.cond_iso, t.stage_terms, ten=t.ten,
                inv2=t.inv2, cond_atom=t.cond_lot_atom, cond_ro=t.cond_lot_atom,
                return_phi=want_phi, dsum=t.dsum,
            )
            if want_phi:
                atomic, terms, ro_phi = ro_res
            else:
                atomic, terms = ro_res
                ro_phi = None
        else:
            atomic, terms = self.readout.final(t.state, t.vec, b, t.cond_iso, t.stage_terms, ten=t.ten, dsum=t.dsum)
            ro_phi = None
        e_atom = atomic
        zeros = atomic.new_zeros(b.n_graphs)
        e_coul = e_pol = e_rep = zeros
        rep_atom = pos.new_zeros(b.n_atoms)
        if self.cfg.repulsion != "none":
            rep_atom = nlh_atom_energy(b, pos, csr_plan=t.csr_plan, create_graph=cg)
            e_rep = self.readout.graph_energy(rep_atom, b)
            e_atom = e_atom + rep_atom
        q = dipole = alpha = alpha_aniso = alpha_tensor = efg_lr = e_eda = delta_eda = eda_raw_terms = eda_m_proj = xray_out = sqe_res_out = sqe_rel_out = None
        p_tot_out = p_perm_out = p_ind_out = fld_out = pol_atom_out = quad_l2_out = homo_gap_out = None
        if t.backend is not None and not self.cfg.sqe_pqeq:
            ro_ps = tuple(self.trunk.charge_heads[-1].parameters()) + (self.dipole_head.weight,)
            ro_cn = node_chunk_size(pos.device) if pos.device.type == "mps" else NODE_CHUNK_ATOMS
            use_ro_chunks = can_chunk_nodes(b.n_atoms, ro_ps, min_atoms=ro_cn, create_graph=cg) and not self.training
            already_normed = bool(self.cfg.state_norm)
            s_ro = t.state if (already_normed or use_ro_chunks) else state_norm(t.state)
            q_grp = b.graph if getattr(b, "frag_group", None) is None else b.frag_group
            q_tot = (b.charge if getattr(b, "frag_q", None) is None else b.frag_q).to(pos.dtype)
            q = self.trunk.charge_heads[-1](s_ro, q_grp, q_tot, b.atom_mask, normed=(already_normed or not use_ro_chunks))
            if use_ro_chunks:
                p = chunked_node_apply(
                    lambda _a0, _a1, v_: torch.einsum("nkc,k->nc", bound_vectors(v_, self.cfg.vec_c), self.dipole_head.weight[0]),
                    (t.vec,), params=(self.dipole_head.weight,), chunk_atoms=ro_cn,
                )[0]
            else:
                u = bound_vectors(t.vec, self.cfg.vec_c)
                p = torch.einsum("nkc,k->nc", u, self.dipole_head.weight[0])
            if self.polarization is not None:
                phi, fld = t.backend.phi_field(q)
                pol_atom, p_ind, alpha, alpha_aniso = self.polarization(s_ro, p, fld, mf, normed=(already_normed or not use_ro_chunks), return_alpha=True, ten=t.ten, return_aniso=True)
                p_tot = p + p_ind
                e_pol = self.readout.graph_energy(pol_atom, b)
                e_atom = e_atom + pol_atom
            else:
                if return_multipoles:
                    phi, fld = t.backend.phi_field(q)
                    p_ind = torch.zeros_like(p)
                    pol_atom = torch.zeros_like(q)
                else:
                    phi = t.backend.phi(q)
                p_tot = p
            dip_atom = (q.unsqueeze(-1) * pos + p_tot) * mf.unsqueeze(-1)
            dipole = dip_atom.sum(0, keepdim=True) if b.n_graphs == 1 else pos.new_zeros(b.n_graphs, 3).index_add(0, b.graph, dip_atom)
            coul_atom = (0.5 * COULOMB_K) * q * (phi * mf)
            e_coul = self.readout.graph_energy(coul_atom, b)
            e_atom = e_atom + coul_atom
            p_tot_out = p_tot * mf.unsqueeze(-1)
            if return_multipoles:
                mf_3 = mf.unsqueeze(-1)
                p_perm_out = p * mf_3
                p_ind_out = p_ind * mf_3
                fld_out = fld * mf_3
                pol_atom_out = pol_atom * mf
        e_ov_out = e_exch_out = None
        if self.lam6_phys is not None:
            l6_cn = node_chunk_size(pos.device) if pos.device.type == "mps" else NODE_CHUNK_ATOMS
            l6_chunks = can_chunk_nodes(b.n_atoms, tuple(self.lam6_phys.parameters()), min_atoms=l6_cn, create_graph=cg) and not self.training
            lp = eval_lam6_physics(
                self.lam6_phys, self.cfg, self.trunk, t, b, pos, atomic, rep_atom, l6_chunks,
                domain=domain, q_precomputed=q, return_multipoles=return_multipoles, return_eda=return_eda,
                return_xray=return_xray, return_sqe_residual=return_sqe_residual,
                return_aux_heads=return_aux_heads, create_graph=cg,
            )
            e_atom = e_atom + lp["e_sqe_atom"] + lp["e_ov_atom"] + lp["e_exch_atom"]
            if (not self.cfg.sqe_pqeq) and self.cfg.cloud_overlap:
                e_ov_out = self.readout.graph_energy(lp["e_ov_atom"], b)
            if self.cfg.spin_exchange:
                e_exch_out = self.readout.graph_energy(lp["e_exch_atom"], b)
            quad_l2_out = lp.get("quad_l2_atom")
            homo_gap_out = lp.get("homo_gap")
            if self.cfg.sqe_pqeq:
                q, dipole, alpha_tensor = lp["q"], lp["dipole"], lp["alpha_tensor"]
                e_coul = self.readout.graph_energy(lp["e_sqe_atom"], b)
                e_pol = self.readout.graph_energy(lp["e_ov_atom"], b)
                efg_lr = lp["efg_lr"]
                p_tot_out = lp["p_tot"] * mf.unsqueeze(-1)
                if return_multipoles:
                    mf_3 = mf.unsqueeze(-1)
                    p_perm_out, p_ind_out = lp["p_perm"] * mf_3, lp["p_ind"] * mf_3
                    fld_out = None if lp["fld"] is None else lp["fld"] * mf_3
            e_eda, delta_eda, xray_out = lp["e_eda"], lp["delta_eda"], lp["xray"]
            eda_raw_terms, eda_m_proj, sqe_res_out, sqe_rel_out = lp["eda_raw_terms_5"], lp["eda_m_proj"], lp.get("sqe_residual"), lp.get("sqe_rel_residual")
        energy_f64 = self.readout.graph_energy_f64(e_atom, b)
        ref_e = self.group_reference(b, domain) if add_ref else None
        if ref_e is not None:
            energy_f64 = energy_f64 + (ref_e.to(dtype=energy_f64.dtype).to(device=energy_f64.device) if (ref_e.dtype != energy_f64.dtype or ref_e.device != energy_f64.device) else ref_e)
        energy = energy_f64.to(pos.dtype) if energy_f64.dtype != pos.dtype else energy_f64
        stage_energies = None
        if return_stage_energies and self.cfg.stage_e_weight > 0 and t.stage_terms:
            cum = e_atom - atomic + terms.get("iso", 0.0)
            stage_e = []
            for l in range(self.cfg.n_stages - 1):
                cum = cum + terms[f"stage{l}"]
                stage_e.append(self.readout.graph_energy_f64(cum, b))
            se_f64 = torch.stack(stage_e, dim=-1)
            if ref_e is not None:
                se_f64 = se_f64 + ref_e.to(device=se_f64.device, dtype=se_f64.dtype).unsqueeze(-1)
            stage_energies = se_f64.to(pos.dtype) if se_f64.dtype != pos.dtype else se_f64
        forces = f_direct = None
        if direct_forces:
            if self.force_head is None:
                raise ValueError("direct_forces=True needs ModelConfig.direct_force_head")
            f_direct = self.force_head(t.state, t.vec, b.atom_mask)
        stress = None
        if compute_forces or compute_stress:
            wrt = ([leaf] if compute_forces else []) + ([eps] if compute_stress else [])
            grads = torch.autograd.grad(energy_f64.sum(), wrt, create_graph=cg, retain_graph=cg)
            if compute_forces:
                forces = -grads[0]
            if compute_stress:
                a, bb, c = cell0.unbind(1)
                vol = (a * torch.cross(bb, c, dim=-1)).sum(-1).abs()   # unstrained volume (eps = 0); 0 for open and pad graphs
                ok = vol > 0
                stress = grads[-1] * (ok.to(vol.dtype) / torch.where(ok, vol, torch.ones_like(vol))).view(-1, 1, 1)
        detach_out = (compute_forces or compute_stress) and not cg
        net_e = self.readout.graph_energy(atomic.detach() if detach_out else atomic, b)
        u_rad = m_raw = None
        if t.spin_desc is not None:
            u_rad, m_raw = t.spin_desc
        if detach_out:
            det = lambda x: None if x is None else x.detach()  # noqa: E731
            energy, energy_f64, e_atom, forces, f_direct, stress = det(energy), det(energy_f64), det(e_atom), det(forces), det(f_direct), det(stress)
            e_coul, e_pol, e_rep, q, dipole = det(e_coul), det(e_pol), det(e_rep), det(q), det(dipole)
            alpha, alpha_aniso, stage_energies, spin_d = det(alpha), det(alpha_aniso), det(stage_energies), det(t.spin_density)
            stage_q, stage_s = [x.detach() for x in t.stage_charges], [x.detach() for x in t.stage_spins]
            terms = {k: v.detach() for k, v in terms.items()}
            ro_phi, u_rad, m_raw = det(ro_phi), det(u_rad), det(m_raw)
            alpha_tensor, efg_lr, e_eda, delta_eda = det(alpha_tensor), det(efg_lr), det(e_eda), det(delta_eda)
            eda_raw_terms, eda_m_proj, sqe_res_out, sqe_rel_out = det(eda_raw_terms), det(eda_m_proj), det(sqe_res_out), det(sqe_rel_out)
            p_tot_out, quad_l2_out, homo_gap_out = det(p_tot_out), det(quad_l2_out), det(homo_gap_out)
            e_ov_out, e_exch_out = det(e_ov_out), det(e_exch_out)
            if xray_out is not None:
                xray_out = {k: ({sk: det(sv) for sk, sv in v.items()} if isinstance(v, dict) else det(v)) for k, v in xray_out.items()}
            if return_multipoles:
                p_perm_out, p_ind_out = det(p_perm_out), det(p_ind_out)
                fld_out, pol_atom_out = det(fld_out), det(pol_atom_out)
        else:
            spin_d, stage_q, stage_s = t.spin_density, t.stage_charges, t.stage_spins
        out = {"energy": energy, "energy_f64": energy_f64, "energy_atom": e_atom, "forces": forces, "forces_direct": f_direct, "stress": stress, "network_energy": net_e,
                "electrostatic_energy": e_coul, "polarization_energy": e_pol, "repulsion_energy": e_rep, "charges": q,
                "dipole": dipole, "alpha": alpha, "alpha_aniso": alpha_aniso, "stage_energies": stage_energies,
                "stage_charges": stage_q, "stage_spins": stage_s, "spin_density": spin_d, "terms": terms}
        if p_tot_out is not None:
            out["atomic_dipoles_tot"] = p_tot_out
        if e_ov_out is not None:
            out["overlap_energy"] = e_ov_out
        if e_exch_out is not None:
            out["spin_exchange_energy"] = e_exch_out
        if quad_l2_out is not None:
            out["quad_l2_atom"] = quad_l2_out
        if homo_gap_out is not None:
            out["homo_gap"] = homo_gap_out
        if ro_phi is not None:
            out["phi"] = ro_phi
        if u_rad is not None:
            out["u_rad"] = u_rad
            out["m_raw"] = m_raw
        if alpha_tensor is not None:
            out["alpha_tensor"] = alpha_tensor
        if efg_lr is not None:
            out["efg_lr"] = efg_lr
        if e_eda is not None:
            out["e_eda"] = e_eda
            out["delta_eda"] = delta_eda
            if eda_raw_terms is not None:
                out["eda_raw_terms_5"] = eda_raw_terms
            if eda_m_proj is not None:
                out["eda_m_proj"] = eda_m_proj
        if xray_out is not None:
            out["xray"] = xray_out
        if sqe_res_out is not None:
            out["sqe_residual"] = sqe_res_out
        if sqe_rel_out is not None:
            out["sqe_rel_residual"] = sqe_rel_out
        if return_multipoles:
            out["atomic_dipoles"] = p_tot_out
            out["atomic_dipoles_perm"] = p_perm_out
            out["atomic_dipoles_ind"] = p_ind_out
            out["electric_field"] = fld_out
            out["polarization_atom"] = pol_atom_out
        return out
