"""xray_eda.py: Tier B #4 non-degrading Orthogonalized EDA, Empirical-Bayes OOD, and Cauchy Virial X-Ray.

1. EDA_CalibHead: zero-column-sum mixing in 1^perp over stop_grad inputs during auxiliary SAPT/EDA training
   (0.000000% gradient into E_tot/F_tot) and live pos at inference (sum_c e_EDA == E_tot, sum_c F_EDA == F_tot).
2. EmpiricalBayesOOD: 4-level hierarchical covariance shrinkage (Z, LoT) -> (Z) -> group(Z) -> global for
   post-hoc Last-Layer Laplace variance and Stage-2 Mahalanobis d_M^2 without ensemble cost.
3. Per-atom Cauchy virial (Hardy SR edge virial + LR/SPME reciprocal virial summing to V * stress_cell),
   hydrostatic pressure, von Mises deviatoric shear strain, and local strain/cancellation alarms.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lam4.core.radial import fp32_guard, stop_grad, upcast_fp32


def _element_group_table() -> Tensor:
    """Map Z in 0..118 to 8 chemical groups for rare-(Z, LoT) Empirical-Bayes shrinkage."""
    grp = torch.zeros(119, dtype=torch.long)
    for z in (1,): grp[z] = 1                                         # H
    for z in (2, 10, 18, 36, 54, 86, 118): grp[z] = 2                 # Noble gases
    for z in (3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88): grp[z] = 3  # Alkali / AlkEarth
    for z in range(21, 31): grp[z] = 4                                # 3d TM
    for z in (*range(39, 49), *range(72, 81)): grp[z] = 5             # 4d/5d TM
    for z in (*range(57, 72), *range(89, 104)): grp[z] = 6            # Lanthanides/Actinides
    for z in (9, 17, 35, 53, 85): grp[z] = 7                          # Halogens
    return grp  # 0 = C/N/O/P/S/main-group


class EDA_CalibHead(nn.Module):
    """Nullspace (1^perp) SAPT/EDA calibration head guaranteeing 0.0% degradation of total E/F."""

    def __init__(self, d_model: int, n_terms: int = 5, hidden: int = 64,
                 mix_cap: float = 0.5, res_cap: float = 2.0) -> None:
        super().__init__()
        self.n_terms = int(n_terms)
        self.mix_cap, self.res_cap = float(mix_cap), float(res_cap)
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.SiLU(), nn.Linear(hidden, n_terms * n_terms + n_terms))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def mix_matrix(self, sn: Tensor, *, detach_inputs: bool | None = None) -> Tensor:
        """Return zero-column-sum residual mixer M(s_i) in 1^perp [N, T, T] (calibrated = (I + M) raw + res)."""
        det = self.training if detach_inputs is None else bool(detach_inputs)
        sn_in = stop_grad(upcast_fp32(sn)) if det else upcast_fp32(sn)
        t = self.n_terms
        m_raw = self.mix_cap * torch.tanh(self.net(sn_in)[:, :t * t].view(-1, t, t))
        return m_raw - m_raw.mean(dim=-2, keepdim=True)

    def forward(self, sn: Tensor, e_phys: Tensor, atom_mask: Tensor, *,
                detach_inputs: bool | None = None, return_terms: bool = False):
        """Return (e_eda [N, T], delta [N, T]) where sum_t delta_t === 0, or dict with eda_raw_terms_5 / eda_calib_terms_5."""
        det = self.training if detach_inputs is None else bool(detach_inputs)
        sn_in = stop_grad(upcast_fp32(sn)) if det else upcast_fp32(sn)
        ep_in = stop_grad(upcast_fp32(e_phys)) if det else upcast_fp32(e_phys)
        mf = atom_mask.to(ep_in.dtype)

        raw = self.net(sn_in)
        t = self.n_terms
        m_raw = (self.mix_cap * torch.tanh(raw[:, :t * t].view(-1, t, t))).double()
        # Project columns onto 1^perp in fp64 so sum_c M_{c, c'} === 0 cleanly even at large e_phys
        m_proj = (m_raw - m_raw.mean(dim=-2, keepdim=True)).to(ep_in.dtype)
        res_raw = (self.res_cap * torch.tanh(raw[:, t * t:])).double()
        res_proj = res_raw - res_raw.mean(dim=-1, keepdim=True)
        delta_64 = (torch.einsum("ncd,nd->nc", m_proj.double(), ep_in.double()) + res_proj) * mf.double().unsqueeze(-1)
        delta = delta_64.to(ep_in.dtype)
        e_raw = ep_in * mf.unsqueeze(-1)
        e_eda = (e_raw.double() + delta.double()).to(ep_in.dtype)
        if det and e_eda.shape[-1] > 0:
            ulps = (e_eda.double().sum(-1) - e_raw.double().sum(-1)).to(e_eda.dtype)
            e_eda = torch.cat((e_eda[:, :1] - ulps.unsqueeze(-1), e_eda[:, 1:]), dim=-1)
        if return_terms:
            return {"eda_raw_terms_5": e_raw, "eda_calib_terms_5": e_eda, "delta": delta, "m_proj": m_proj}
        return e_eda, delta

    def regularization_loss(self, delta: Tensor, atom_mask: Tensor, *, m_proj: Tensor | None = None,
                            sn: Tensor | None = None, w_delta: float = 1e-2, w_param: float = 1e-2,
                            w_m: float = 1e-2) -> Tensor:
        """Zero-drift regularizer penalizing delta, weights, and ||M(s_i)||_F^2 identity proximity."""
        mf = atom_mask.to(delta.dtype)
        denom = mf.sum().clamp_min(1.0)
        d_l2 = (delta.pow(2).sum(-1) * mf).sum() / denom
        if m_proj is None and sn is not None:
            m_proj = self.mix_matrix(sn, detach_inputs=self.training)
        m_l2 = ((m_proj.pow(2).sum((-2, -1)) * mf).sum() / denom) if m_proj is not None else delta.new_zeros(())
        p_l2 = sum(p.pow(2).sum() for p in self.parameters())
        return w_delta * d_l2 + w_m * m_l2 + w_param * p_l2


@torch.enable_grad()
def compute_eda_forces(e_eda: Tensor, pos: Tensor, atom_mask: Tensor, *, retain_graph: bool = False) -> dict[str, Tensor]:
    """Differentiate each of the T EDA terms wrt pos at inference: sum_c F_c === F_tot."""
    t = e_eda.shape[-1]
    mf = atom_mask.to(pos.dtype)
    f_terms = []
    for c in range(t):
        g = torch.autograd.grad((e_eda[:, c] * mf).sum(), pos, retain_graph=retain_graph or (c < t - 1), allow_unused=True)[0]
        f_terms.append(torch.zeros_like(pos) if g is None else -g)
    f_stack = torch.stack(f_terms, dim=0)  # [T, N, 3]
    f_tot = f_stack.sum(dim=0)
    f_tot_norm = f_tot.norm(dim=-1)
    f_gross = f_stack.norm(dim=-1).sum(dim=0).clamp_min(1e-8)
    cancel_ratio = f_tot_norm / f_gross
    return {"eda_forces": f_stack, "forces_sum": f_tot, "eda_force_cancel_ratio": cancel_ratio}


class EmpiricalBayesOOD(nn.Module):
    """Post-hoc 4-level hierarchical shrinkage OOD sensor: (Z, LoT) -> (Z) -> group(Z) -> global."""

    def __init__(self, prior_reg: float = 1e-2) -> None:
        super().__init__()
        self.prior_reg = float(prior_reg)
        self.register_buffer("grp_table", _element_group_table(), persistent=False)
        self.lap_prec: dict[tuple[int, int], Tensor] = {}
        self.maha_mean: dict[tuple[int, int], Tensor] = {}
        self.maha_prec: dict[tuple[int, int], Tensor] = {}
        self.lap_med: dict[tuple[int, int], float] = {}
        self.maha_med: dict[tuple[int, int], float] = {}
        self._d_phi: int = 0
        self._d_s: int = 0
        self._fitted: bool = False

    @staticmethod
    def _uncentered_cov(x: Tensor) -> Tensor:
        n = x.shape[0]
        if n == 0:
            return x.new_zeros(x.shape[1], x.shape[1])
        return (x.t() @ x) / float(n)

    @staticmethod
    def _mean_centered_cov(x: Tensor) -> tuple[Tensor, Tensor]:
        n = x.shape[0]
        if n == 0:
            return x.new_zeros(x.shape[1]), x.new_zeros(x.shape[1], x.shape[1])
        mu = x.mean(dim=0)
        xc = x - mu
        return mu, (xc.t() @ xc) / float(max(1, n))

    def _shrink_cov(self, c_sub: Tensor | None, n_sub: int, c_parent: Tensor, dim: int) -> Tensor:
        if c_sub is None or n_sub == 0:
            return c_parent
        alpha = float(n_sub) / float(n_sub + dim)
        return alpha * c_sub + (1.0 - alpha) * c_parent

    @torch.no_grad()
    def fit(self, phi: Tensor, s_norm: Tensor, z: Tensor, lot_atom: Tensor, atom_mask: Tensor) -> "EmpiricalBayesOOD":
        """Fit hierarchical Laplace precision & state Mahalanobis covariance without altering model weights."""
        mask = atom_mask.bool()
        pf = phi[mask].to(torch.float64)
        sf = s_norm[mask].to(torch.float64)
        zf = z[mask].to(torch.long)
        lf = lot_atom[mask].to(torch.long)
        gf = self.grp_table.to(zf.device).index_select(0, zf.clamp(0, 118))
        dp, ds = pf.shape[1], sf.shape[1]
        self._d_phi, self._d_s = dp, ds

        # Level 3: global pooled
        eye_p = torch.eye(dp, dtype=torch.float64, device=pf.device)
        eye_s = torch.eye(ds, dtype=torch.float64, device=sf.device)
        c_lap_glob = self._uncentered_cov(pf) + self.prior_reg * eye_p
        mu_s_glob, c_s_glob = self._mean_centered_cov(sf)
        c_s_glob = c_s_glob + self.prior_reg * eye_s

        # Level 2: chemical groups
        c_lap_grp, mu_s_grp, c_s_grp = {}, {}, {}
        for g_id in torch.unique(gf).tolist():
            sel = gf == g_id
            ng = int(sel.sum())
            c_lap_grp[g_id] = self._shrink_cov(self._uncentered_cov(pf[sel]), ng, c_lap_glob, dp)
            mg, cg = self._mean_centered_cov(sf[sel])
            a_g = float(ng) / float(ng + ds)
            mu_s_grp[g_id] = a_g * mg + (1.0 - a_g) * mu_s_glob
            c_s_grp[g_id] = self._shrink_cov(cg, ng, c_s_glob, ds)

        # Level 1: element (Z)
        c_lap_z, mu_s_z, c_s_z, z_to_g = {}, {}, {}, {}
        for zv in torch.unique(zf).tolist():
            sel = zf == zv
            nz = int(sel.sum())
            gid = int(self.grp_table[min(max(zv, 0), 118)].item())
            z_to_g[zv] = gid
            c_lap_z[zv] = self._shrink_cov(self._uncentered_cov(pf[sel]), nz, c_lap_grp.get(gid, c_lap_glob), dp)
            mz, cz = self._mean_centered_cov(sf[sel])
            az = float(nz) / float(nz + ds)
            mu_s_z[zv] = az * mz + (1.0 - az) * mu_s_grp.get(gid, mu_s_glob)
            c_s_z[zv] = self._shrink_cov(cz, nz, c_s_grp.get(gid, c_s_glob), ds)

        # Store backoff levels: (-1,-1)=global, (-2-g,-1)=chemical group, (Z,-1)=element, (Z,LoT)
        self.lap_prec.clear(); self.maha_mean.clear(); self.maha_prec.clear()
        self.lap_med.clear(); self.maha_med.clear()
        self.lap_prec[(-1, -1)] = torch.linalg.inv(self.prior_reg * eye_p + float(dp) * c_lap_glob).to(phi.dtype)
        self.maha_mean[(-1, -1)] = mu_s_glob.to(s_norm.dtype)
        self.maha_prec[(-1, -1)] = torch.linalg.inv(c_s_glob + 1e-6 * eye_s).to(s_norm.dtype)
        for gid, cg_lap in c_lap_grp.items():
            gk = (-2 - gid, -1)
            self.lap_prec[gk] = torch.linalg.inv(self.prior_reg * eye_p + float(dp) * cg_lap).to(phi.dtype)
            self.maha_mean[gk] = mu_s_grp[gid].to(s_norm.dtype)
            self.maha_prec[gk] = torch.linalg.inv(c_s_grp[gid] + 1e-6 * eye_s).to(s_norm.dtype)
        for zv in z_to_g:
            self.lap_prec[(zv, -1)] = torch.linalg.inv(self.prior_reg * eye_p + float(dp) * c_lap_z[zv]).to(phi.dtype)
            self.maha_mean[(zv, -1)] = mu_s_z[zv].to(s_norm.dtype)
            self.maha_prec[(zv, -1)] = torch.linalg.inv(c_s_z[zv] + 1e-6 * eye_s).to(s_norm.dtype)

        pairs = set(zip(zf.tolist(), lf.tolist()))
        for zv, lv in pairs:
            sel = (zf == zv) & (lf == lv)
            nzl = int(sel.sum())
            c_l = self._shrink_cov(self._uncentered_cov(pf[sel]), nzl, c_lap_z[zv], dp)
            lam_mat = self.prior_reg * eye_p + float(nzl + dp) * c_l
            prec_lap = torch.linalg.inv(lam_mat).to(phi.dtype)
            mzl, czl = self._mean_centered_cov(sf[sel])
            azl = float(nzl) / float(nzl + ds)
            mu_f = (azl * mzl + (1.0 - azl) * mu_s_z[zv]).to(s_norm.dtype)
            cov_s = self._shrink_cov(czl, nzl, c_s_z[zv], ds) + 1e-6 * eye_s
            prec_maha = torch.linalg.inv(cov_s).to(s_norm.dtype)
            self.lap_prec[(zv, lv)], self.maha_mean[(zv, lv)], self.maha_prec[(zv, lv)] = prec_lap, mu_f, prec_maha
            lv_vals = torch.einsum("ni,ij,nj->n", pf[sel].to(phi.dtype), prec_lap, pf[sel].to(phi.dtype))
            dm = sf[sel].to(s_norm.dtype) - mu_f
            mv_vals = torch.einsum("ni,ij,nj->n", dm, prec_maha, dm)
            self.lap_med[(zv, lv)] = max(1e-12, float(lv_vals.median().item()))
            self.maha_med[(zv, lv)] = max(1e-12, float(mv_vals.median().item()))
        self._fitted = True
        return self

    def _resolve_key(self, zv: int, lv: int) -> tuple[int, int]:
        if (zv, lv) in self.lap_prec:
            return (zv, lv)
        if (zv, -1) in self.lap_prec:
            return (zv, -1)
        gid = int(self.grp_table[min(max(zv, 0), 118)].item())
        return (-2 - gid, -1) if (-2 - gid, -1) in self.lap_prec else (-1, -1)

    @torch.no_grad()
    def score(self, phi: Tensor, s_norm: Tensor, z: Tensor, lot_atom: Tensor, atom_mask: Tensor) -> dict[str, Tensor]:
        """Compute per-atom Laplace variance, Mahalanobis d_M^2, and normalized OOD ratios (batched by key)."""
        n = phi.shape[0]
        lap_var, maha_d2 = phi.new_zeros(n), s_norm.new_zeros(n)
        lap_norm, maha_norm = phi.new_zeros(n), s_norm.new_zeros(n)
        if not self._fitted:
            return {"laplace_var": lap_var, "mahalanobis_d2": maha_d2, "ood_norm_lap": lap_norm, "ood_norm_maha": maha_norm}
        real_idx = torch.nonzero(atom_mask.bool(), as_tuple=True)[0]
        if real_idx.numel() == 0:
            return {"laplace_var": lap_var, "mahalanobis_d2": maha_d2, "ood_norm_lap": lap_norm, "ood_norm_maha": maha_norm}
        groups: dict[tuple[int, int], list[int]] = {}
        for idx, zv, lv in zip(real_idx.tolist(), z.index_select(0, real_idx).tolist(), lot_atom.index_select(0, real_idx).tolist()):
            groups.setdefault(self._resolve_key(int(zv), int(lv)), []).append(idx)
        for key, idxs in groups.items():
            sel = torch.tensor(idxs, dtype=torch.long, device=phi.device)
            lp = self.lap_prec[key].to(device=phi.device, dtype=phi.dtype)
            mp = self.maha_prec[key].to(device=s_norm.device, dtype=s_norm.dtype)
            mu = self.maha_mean[key].to(device=s_norm.device, dtype=s_norm.dtype)
            ph, dx = phi.index_select(0, sel), s_norm.index_select(0, sel) - mu
            lv_v = torch.einsum("ni,ij,nj->n", ph, lp, ph).clamp_min(0.0)
            mv_v = torch.einsum("ni,ij,nj->n", dx, mp, dx).clamp_min(0.0)
            lap_var.index_copy_(0, sel, lv_v)
            maha_d2.index_copy_(0, sel, mv_v)
            lap_norm.index_copy_(0, sel, lv_v / self.lap_med.get(key, 1.0))
            maha_norm.index_copy_(0, sel, mv_v / self.maha_med.get(key, 1.0))
        return {"laplace_var": lap_var, "mahalanobis_d2": maha_d2, "ood_norm_lap": lap_norm, "ood_norm_maha": maha_norm}


def hardy_edge_virial(e_sr_total: Tensor, edge_vec: Tensor, send: Tensor, recv: Tensor,
                      atom_mask: Tensor) -> Tensor:
    """Exact symmetric per-atom Cauchy virial W_sr [N, 3, 3] summing to +dE_SR/d_eps = V * stress_SR."""
    with fp32_guard(edge_vec.device, disable_tf32=True):
        grad_de = torch.autograd.grad(e_sr_total.sum(), edge_vec, retain_graph=True)[0]
        n = atom_mask.shape[0]
        w_e = 0.5 * torch.einsum("ea,eb->eab", grad_de, edge_vec)
        w_e = 0.5 * (w_e + w_e.transpose(-1, -2))
        w_at = edge_vec.new_zeros(n, 3, 3).index_add(0, recv, w_e).index_add(0, send, w_e)
        return w_at * atom_mask.to(w_at.dtype).view(-1, 1, 1)


def atomic_strain_metrics(w_atom: Tensor, forces: Tensor, e_phys: Tensor, atom_mask: Tensor,
                          e_atom: Tensor | None = None, e_eq_atom: Tensor | None = None) -> dict[str, Tensor]:
    """Per-atom pressure, von Mises deviatoric stress, cancellation ratio, and strain excess."""
    mf = atom_mask.to(w_atom.dtype)
    tr = w_atom[:, 0, 0] + w_atom[:, 1, 1] + w_atom[:, 2, 2]
    pressure = -(1.0 / 3.0) * tr * mf
    eye3 = torch.eye(3, dtype=w_atom.dtype, device=w_atom.device).unsqueeze(0)
    dev = w_atom - (tr / 3.0).view(-1, 1, 1) * eye3
    von_mises = torch.sqrt((1.5 * dev.pow(2).sum((-2, -1))).clamp_min(1e-24)) * mf
    f_norm = torch.sqrt((forces.pow(2).sum(-1)).clamp_min(1e-24)) * mf
    gross = e_phys.abs().sum(-1).clamp_min(1e-12)
    cancel_ratio = (e_phys.sum(-1).abs() / gross) * mf
    out = {
        "virial_atom_3x3": w_atom, "pressure_atom": pressure,
        "von_mises_stress_atom": von_mises, "force_norm_atom": f_norm, "cancellation_ratio": cancel_ratio,
    }
    if e_atom is not None and e_eq_atom is not None:
        out["strain_excess_atom"] = (e_atom - e_eq_atom) * mf
    return out


def eda_term_shares_and_stability(e_raw_5: Tensor, e_calib_5: Tensor, atom_mask: Tensor,
                                  graph: Tensor | None = None, *, m_proj: Tensor | None = None,
                                  ref_calib_shares: Tensor | None = None) -> dict[str, Tensor]:
    """Compute raw vs calibrated per-term shares, |M|_F identity proximity, and C2->C6 share stability."""
    mf = atom_mask.to(e_raw_5.dtype)
    if graph is None:
        raw_g = (e_raw_5 * mf.unsqueeze(-1)).sum(0, keepdim=True)
        cal_g = (e_calib_5 * mf.unsqueeze(-1)).sum(0, keepdim=True)
    else:
        n_g = int(graph.max().item()) + 1 if not e_raw_5.shape[0] == 0 else 1
        raw_g = e_raw_5.new_zeros(n_g, e_raw_5.shape[-1]).index_add(0, graph, e_raw_5 * mf.unsqueeze(-1))
        cal_g = e_calib_5.new_zeros(n_g, e_calib_5.shape[-1]).index_add(0, graph, e_calib_5 * mf.unsqueeze(-1))
    raw_abs = raw_g.abs().mean(0)
    cal_abs = cal_g.abs().mean(0)
    raw_shares = raw_abs / raw_abs.sum().clamp_min(1e-12)
    cal_shares = cal_abs / cal_abs.sum().clamp_min(1e-12)
    m_frob = (torch.sqrt((m_proj.pow(2).sum((-2, -1)) * mf).sum() / mf.sum().clamp_min(1.0))
              if m_proj is not None else e_raw_5.new_zeros(()))
    drift_l1 = (cal_shares - ref_calib_shares.to(cal_shares)).abs().sum() if ref_calib_shares is not None else e_raw_5.new_zeros(())
    return {
        "eda_raw_terms_5": raw_g,
        "eda_calib_terms_5": cal_g,
        "raw_shares": raw_shares,
        "calib_shares": cal_shares,
        "m_frob_rms": m_frob,
        "share_drift_l1": drift_l1,
    }


def evaluate_sapt_ood_calibration(eda_head: EDA_CalibHead, sn: Tensor, e_phys: Tensor,
                                  atom_mask: Tensor, graph: Tensor, e_sapt_5: Tensor) -> dict[str, Tensor]:
    """Evaluate zero-leak EDA_CalibHead on out-of-distribution SAPT sets (e.g. S66 / X40) without total E drift."""
    with torch.no_grad():
        res = eda_head(sn, e_phys, atom_mask, detach_inputs=True, return_terms=True)
        raw_64 = e_phys.new_zeros(e_sapt_5.shape, dtype=torch.float64).index_add(0, graph, res["eda_raw_terms_5"].double())
        cal_64 = e_phys.new_zeros(e_sapt_5.shape, dtype=torch.float64).index_add(0, graph, res["eda_calib_terms_5"].double())
        raw_g, cal_g = raw_64.to(e_phys.dtype), cal_64.to(e_phys.dtype)
        stats = eda_term_shares_and_stability(res["eda_raw_terms_5"], res["eda_calib_terms_5"], atom_mask, graph, m_proj=res["m_proj"])
        tot_err = (cal_64.sum(-1) - raw_64.sum(-1)).abs().max().to(e_phys.dtype)
        return {
            "raw_mae_per_term": (raw_g - e_sapt_5).abs().mean(0),
            "calib_mae_per_term": (cal_g - e_sapt_5).abs().mean(0),
            "total_conservation_max_err": tot_err,
            **stats,
        }
