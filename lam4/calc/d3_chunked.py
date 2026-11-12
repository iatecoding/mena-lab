"""Memory-bounded 3-pass half-edge D3(BJ) evaluation for CPU and non-Metal devices."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from lam.models.dispersion import ANGSTROM_TO_BOHR, D3_K1, D3_K3, D3BJ, HARTREE_TO_KCAL

D3_EDGE_CHUNK = 32_768


def eval_d3_bj_open_chunked(
    d3: D3BJ,
    z_t: Tensor,
    pos_t: Tensor,
    ei_full: Tensor,
    *,
    compute_forces: bool = True,
    chunk_edges: int = D3_EDGE_CHUNK,
) -> tuple[float, np.ndarray | None]:
    """Exact 3-pass half-edge (`i < j`) D3(BJ) energy (kcal/mol) & forces (kcal/mol/A) in O(chunk_edges) memory."""
    n = int(z_t.shape[0])
    keep = ei_full[0] < ei_full[1]
    idx_i = ei_full[0, keep].contiguous()
    idx_j = ei_full[1, keep].contiguous()
    m = int(idx_i.numel())
    if m == 0:
        return 0.0, (np.zeros((n, 3), dtype=np.float64) if compute_forces else None)

    zc = z_t.clamp(max=d3.rcov.shape[0] - 1)
    cn_cut_bohr = float(d3.cn_cut * ANGSTROM_TO_BOHR)
    n_elem = int(d3.c6ab.shape[1])

    with torch.no_grad():
        cn = pos_t.new_zeros(n)
        for c0 in range(0, m, chunk_edges * 2):
            sl = slice(c0, min(c0 + chunk_edges * 2, m))
            ii, jj = idx_i[sl], idx_j[sl]
            diff = pos_t.index_select(0, ii) - pos_t.index_select(0, jj)
            r_ang = (diff.pow(2).sum(-1) + 1e-8).sqrt()
            rb = r_ang * ANGSTROM_TO_BOHR
            rr = (d3.rcov.index_select(0, zc.index_select(0, ii)) + d3.rcov.index_select(0, zc.index_select(0, jj))) / rb.clamp_min(1e-6)
            damp = (1.0 / (1.0 + torch.exp(-D3_K1 * (rr - 1.0)))) * (rb <= cn_cut_bohr)
            cn.index_add_(0, ii, damp)
            cn.index_add_(0, jj, damp)

    e_kcal = 0.0
    forces = pos_t.new_zeros((n, 3)) if compute_forces else None
    de_dcn = pos_t.new_zeros(n) if compute_forces else None

    for c0 in range(0, m, chunk_edges):
        sl = slice(c0, min(c0 + chunk_edges, m))
        ii, jj = idx_i[sl], idx_j[sl]
        with torch.no_grad():
            diff = pos_t.index_select(0, ii) - pos_t.index_select(0, jj)
            r0 = (diff.pow(2).sum(-1) + 1e-8).sqrt()
            zi = zc.index_select(0, ii)
            zj = zc.index_select(0, jj)
            refs = d3.c6ab_flat.index_select(0, zi * n_elem + zj)
            cn0 = refs[..., 0]
            r2r4_prod = d3.r2r4.index_select(0, zi) * d3.r2r4.index_select(0, zj)

        with torch.set_grad_enabled(compute_forces):
            r_c = r0.detach().requires_grad_(compute_forces)
            cni = cn.index_select(0, ii).detach().requires_grad_(compute_forces)
            cnj = cn.index_select(0, jj).detach().requires_grad_(compute_forces)
            rb_c = r_c * ANGSTROM_TO_BOHR
            r2 = (refs[..., 1] - cni.unsqueeze(-1)).pow(2) + (refs[..., 2] - cnj.unsqueeze(-1)).pow(2)
            scores = torch.where(cn0 > 0.0, D3_K3 * r2, torch.full_like(r2, -1.0e20))
            c6 = (torch.softmax(scores, dim=-1) * cn0).sum(dim=-1)
            c8 = 3.0 * c6 * r2r4_prod
            tmp = d3.a1 * torch.sqrt(c8 / c6.clamp_min(1e-12)) + d3.a2
            r2_b = rb_c.pow(2)
            r6 = r2_b.pow(3)
            r8 = r6 * r2_b
            e6 = 1.0 / (r6 + tmp.pow(6))
            e8 = 1.0 / (r8 + tmp.pow(8))
            t = ((r_c - d3.taper_on) / max(1e-6, d3.r_cut - d3.taper_on)).clamp(0.0, 1.0)
            taper = 1.0 - t * t * t * (10.0 + t * (6.0 * t - 15.0))
            ep = -1.0 * (d3.s6 * c6 * e6 + d3.s8 * c8 * e8) * taper * (r_c <= d3.r_cut)
            e_chunk = ep.sum() * HARTREE_TO_KCAL
            e_kcal += float(e_chunk.detach().cpu().double().item())
            if compute_forces:
                gr, gcni, gcnj = torch.autograd.grad(e_chunk, (r_c, cni, cnj))
                f_pair = -(gr / r0).unsqueeze(-1) * diff
                assert forces is not None and de_dcn is not None
                forces.index_add_(0, ii, f_pair)
                forces.index_add_(0, jj, -f_pair)
                de_dcn.index_add_(0, ii, gcni)
                de_dcn.index_add_(0, jj, gcnj)

    if compute_forces:
        assert forces is not None and de_dcn is not None
        for c0 in range(0, m, chunk_edges * 2):
            sl = slice(c0, min(c0 + chunk_edges * 2, m))
            ii, jj = idx_i[sl], idx_j[sl]
            with torch.no_grad():
                diff = pos_t.index_select(0, ii) - pos_t.index_select(0, jj)
                r0 = (diff.pow(2).sum(-1) + 1e-8).sqrt()
                rcov_sum = d3.rcov.index_select(0, zc.index_select(0, ii)) + d3.rcov.index_select(0, zc.index_select(0, jj))
                cot = de_dcn.index_select(0, ii) + de_dcn.index_select(0, jj)
            with torch.enable_grad():
                r_c = r0.detach().requires_grad_(True)
                rb = r_c * ANGSTROM_TO_BOHR
                rr = rcov_sum / rb.clamp_min(1e-6)
                damp = (1.0 / (1.0 + torch.exp(-D3_K1 * (rr - 1.0)))) * (rb <= cn_cut_bohr)
                (gr_cn,) = torch.autograd.grad(damp, r_c, cot)
            with torch.no_grad():
                f_cn = -(gr_cn / r0).unsqueeze(-1) * diff
                forces.index_add_(0, ii, f_cn)
                forces.index_add_(0, jj, -f_cn)
        return e_kcal, forces.detach().cpu().to(torch.float64).numpy()

    return e_kcal, None
