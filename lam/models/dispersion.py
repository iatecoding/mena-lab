"""dispersion.py: Reference Grimme D3(BJ) dispersion with coordination-number interpolation.

Fixes the defects of the legacy hand-rolled dispersion in this repo:
  - correct reference C6 tables (a.u.) with CN interpolation (k3=-4 Gaussian weights)
  - correct BJ damping with a2 in Bohr
  - C8 = 3 * C6 * sqrt(r2r4_i * r2r4_j)
  - defaults: wB97M-D3(BJ) parameters (s6=1.0, s8=0.3908, a1=0.566, a2=3.128 Bohr),
    the exact level of theory of the SPICE 1/2 datasets.

Math mirrors torch-dftd `functions/dftd3.py` (validated numerically against it).

Energies in kcal/mol, distances in Angstrom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

HARTREE_TO_KCAL = 627.509474
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
D3_K1 = 16.0
D3_K3 = -4.0
WB97M_D3BJ = {"s6": 1.0, "s8": 0.3908, "a1": 0.566, "a2": 3.128}


def _params_path() -> Path:
    local_path = Path(__file__).resolve().parent / "params" / "dftd3_params.npz"
    if local_path.exists():
        return local_path
    try:
        import torch_dftd
        return Path(torch_dftd.__file__).resolve().parent / "nn" / "params" / "dftd3_params.npz"
    except ImportError:
        return local_path


class D3BJ(nn.Module):
    """Differenced D3(BJ) dispersion (energy in kcal/mol).

    Args:
        z:    [B, N] atomic numbers (padded)
        pos:  [B, N, 3] Angstrom
        mask: [B, N] valid-atom mask
    Returns:
        e_disp: [B] kcal/mol
    """

    def __init__(
        self,
        s6: float = WB97M_D3BJ["s6"],
        s8: float = WB97M_D3BJ["s8"],
        a1: float = WB97M_D3BJ["a1"],
        a2: float = WB97M_D3BJ["a2"],
        r_cut: float = 12.0,
        taper_on: float = 9.0,
        cn_cut: float = 11.2,
        max_z: int = 95,
        chunk: int = 64,
    ) -> None:
        super().__init__()
        tables = np.load(_params_path(), allow_pickle=True)
        c6ab_t = torch.from_numpy(tables["c6ab"]).float()[:max_z, :max_z]
        self.register_buffer("c6ab", c6ab_t, persistent=False)
        self.register_buffer("c6ab_flat", c6ab_t.reshape(-1, 25, 3).contiguous(), persistent=False)
        self.register_buffer("rcov", torch.from_numpy(tables["rcov"]).float()[:max_z], persistent=False)
        self.register_buffer("r2r4", torch.from_numpy(tables["r2r4"]).float()[:max_z], persistent=False)
        self.s6, self.s8 = float(s6), float(s8)
        self.a1, self.a2 = float(a1), float(a2)
        self.r_cut, self.taper_on, self.cn_cut = float(r_cut), float(taper_on), float(cn_cut)
        self.chunk = int(chunk)

    def forward(self, z: Tensor, pos: Tensor, mask: Tensor) -> Tensor:
        B, N = z.shape
        z = z.clamp(max=self.rcov.shape[0] - 1) * mask.long()
        e_total = pos.new_zeros(B)

        eye = torch.eye(N, dtype=torch.bool, device=z.device).unsqueeze(0)
        pair = (mask.unsqueeze(2) & mask.unsqueeze(1)) & ~eye

        cn_cut_bohr = self.cn_cut * ANGSTROM_TO_BOHR
        for c0 in range(0, B, self.chunk):
            sl = slice(c0, min(c0 + self.chunk, B))
            zc, pc, mc = z[sl], pos[sl], pair[sl]
            b = zc.shape[0]

            diff = pc.unsqueeze(2) - pc.unsqueeze(1)
            r_ang = (diff.pow(2).sum(-1) + 1e-8).sqrt()
            r_bohr = r_ang * ANGSTROM_TO_BOHR

            zi = zc.unsqueeze(2).expand(b, N, N)
            zj = zc.unsqueeze(1).expand(b, N, N)

            # Coordination numbers (Bohr units, reference damping k1=16)
            rr = (self.rcov[zi] + self.rcov[zj]) / r_bohr.clamp_min(1e-6)
            damp = 1.0 / (1.0 + torch.exp(-D3_K1 * (rr - 1.0)))
            damp = damp * mc * (r_bohr <= cn_cut_bohr)
            cn = damp.sum(dim=2)  # [b, N]

            # CN-interpolated C6 from reference table entries (95*95, 25, 3)
            flat = self.c6ab_flat
            idx = (zi * self.c6ab.shape[1] + zj).reshape(-1)
            refs = flat.index_select(0, idx).view(b, N, N, 25, 3)
            cn0 = refs[..., 0]
            r2 = (refs[..., 1] - cn.unsqueeze(2).unsqueeze(-1)) ** 2 + (
                refs[..., 2] - cn.unsqueeze(1).unsqueeze(-1)
            ) ** 2
            scores = torch.where(cn0 > 0.0, D3_K3 * r2, torch.full_like(r2, -1.0e20))
            weights = torch.softmax(scores, dim=-1)
            c6 = (weights * cn0).sum(dim=-1)  # a.u.

            c8 = 3.0 * c6 * self.r2r4[zi] * self.r2r4[zj]  # a.u.

            tmp = self.a1 * torch.sqrt(c8 / c6.clamp_min(1e-12)) + self.a2  # Bohr
            r2_b = r_bohr.pow(2)
            r6 = r2_b.pow(3)
            r8 = r6 * r2_b
            e6 = 1.0 / (r6 + tmp.pow(6))
            e8 = 1.0 / (r8 + tmp.pow(8))

            # Smooth taper (Angstrom) 9 -> 12 A, zero beyond r_cut
            t = ((r_ang - self.taper_on) / max(1e-6, self.r_cut - self.taper_on)).clamp(0.0, 1.0)
            taper = 1.0 - t * t * t * (10.0 + t * (6.0 * t - 15.0))
            valid = mc & (r_ang <= self.r_cut)

            e_pair = -0.5 * (self.s6 * c6 * e6 + self.s8 * c8 * e8) * taper * valid
            e_total[sl] = e_pair.sum(dim=(1, 2)) * HARTREE_TO_KCAL

        return e_total * mask.any(dim=-1).float()

    def forward_sparse(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        n_graphs: int,
        edge_index: Tensor,
        edge_shift: Tensor | None = None,
        chunk_edges: int = 2_000_000,
        edge_diff: Tensor | None = None,
        half_edges: bool = False,
    ) -> Tensor:
        """Edge-list D3(BJ): same math as `forward`, over a neighbour list that covers r_cut (and cn_cut).

        z [N], pos [N, 3] Angstrom, batch [N], edge_index [2, E] (receiver i, sender j), edge_shift [E, 3] Angstrom.
        When `half_edges=True`, `edge_index` holds undirected pairs (i < j) once; coordination numbers accumulate into
        both i and j and pair energies carry weight -1.0 instead of -0.5.
        Pairs beyond r_cut / cn_cut contribute exactly zero, as in the dense version. Returns [n_graphs] kcal/mol.
        """
        N = z.shape[0]
        z = z.clamp(max=self.rcov.shape[0] - 1)
        ei, ej = edge_index[0], edge_index[1]
        if edge_diff is not None:
            diff = edge_diff
        else:
            diff = pos[ei] - pos[ej]
            if edge_shift is not None:
                diff = diff - edge_shift
        r_ang = (diff.pow(2).sum(-1) + 1e-8).sqrt()
        r_bohr = r_ang * ANGSTROM_TO_BOHR
        zi, zj = z[ei], z[ej]

        # Coordination numbers (hard cn_cut, as in the dense path)
        rr = (self.rcov[zi] + self.rcov[zj]) / r_bohr.clamp_min(1e-6)
        damp = 1.0 / (1.0 + torch.exp(-D3_K1 * (rr - 1.0)))
        damp = damp * (r_bohr <= self.cn_cut * ANGSTROM_TO_BOHR)
        cn = damp.new_zeros(N).index_add_(0, ei, damp)
        if half_edges:
            cn = cn.index_add_(0, ej, damp)

        pair_scale = -1.0 if half_edges else -0.5
        flat = self.c6ab_flat
        e_total = damp.new_zeros(int(n_graphs))
        for c0 in range(0, ei.shape[0], chunk_edges):
            sl = slice(c0, c0 + chunk_edges)
            zi_c, zj_c, r_c, rb_c = zi[sl], zj[sl], r_ang[sl], r_bohr[sl]
            refs = flat.index_select(0, zi_c * self.c6ab.shape[1] + zj_c)  # [e, 25, 3]
            cn0 = refs[..., 0]
            r2 = (refs[..., 1] - cn[ei[sl]].unsqueeze(-1)) ** 2 + (refs[..., 2] - cn[ej[sl]].unsqueeze(-1)) ** 2
            scores = torch.where(cn0 > 0.0, D3_K3 * r2, torch.full_like(r2, -1.0e20))
            c6 = (torch.softmax(scores, dim=-1) * cn0).sum(dim=-1)
            c8 = 3.0 * c6 * self.r2r4[zi_c] * self.r2r4[zj_c]
            tmp = self.a1 * torch.sqrt(c8 / c6.clamp_min(1e-12)) + self.a2
            r2_b = rb_c.pow(2)
            r6 = r2_b.pow(3)
            r8 = r6 * r2_b
            e6 = 1.0 / (r6 + tmp.pow(6))
            e8 = 1.0 / (r8 + tmp.pow(8))
            t = ((r_c - self.taper_on) / max(1e-6, self.r_cut - self.taper_on)).clamp(0.0, 1.0)
            taper = 1.0 - t * t * t * (10.0 + t * (6.0 * t - 15.0))
            e_pair = pair_scale * (self.s6 * c6 * e6 + self.s8 * c8 * e8) * taper * (r_c <= self.r_cut)
            e_total = e_total.index_add(0, batch[ei[sl]], e_pair)
        return e_total * HARTREE_TO_KCAL
