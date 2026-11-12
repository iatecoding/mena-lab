"""d3zero_pbc.py: GPU D3 zero-damping (D3(0)) energy / forces / stress for periodic cells, for stripping the OMC25 labels.

Why not torch-dftd directly: its neighbour list is built on the CPU (ASE / pymatgen); at VASP's 50.2 A pair cutoff that
is ~1 s per 35-atom frame on the M3. Here pairs are a dense (i, j, image) tensor on the GPU, cut at `pair_cut`
(default 20 A, exact), plus an analytic uniform-density tail for pair_cut..tail_cut (default 50.2 A, VASP VDW_RADIUS):
  E_tail = -1/2 * sum_ij [s6 C6_ij I6 + s8 C8_ij I8] / V,  I_n = 4 pi (pair_cut^(3-n) - tail_cut^(3-n)) / (n - 3)
(the damping f_n is 1 to < 1e-9 beyond 20 A). Forces = -dE/dx and ASE stress = (1/V) dE/d(strain) by autograd.
Math and tables follow torch-dftd / lam.models.dispersion (CN with k1 = 16 on pre-scaled rcov, k3 = -4 Gaussian C6
interpolation, C8 = 3 C6 r2r4_i r2r4_j, f_n = 1 / (1 + 6 (r / (s_rn R0ab))^(-alpha_n)), alpha8 = alpha6 + 2).
Validated against torch-dftd (tests/test_lam2_pool_converters.py).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from lam.models.dispersion import _params_path

BOHR = 0.529177210903  # Angstrom
HARTREE = 27.211386245988  # eV
PBE_ZERO = {"s6": 1.0, "s8": 0.722, "sr6": 1.217, "sr8": 1.0, "alpha": 14.0}  # VASP IVDW=11 defaults for PBE


class D3ZeroPBC:
    def __init__(self, device: str = "cuda", pair_cut: float = 20.0, cn_cut: float = 20.0, tail_cut: float | None = 50.2,
                 params: dict | None = None, dtype=torch.float64, pair_chunk: int = 4_000_000):
        t = np.load(_params_path(), allow_pickle=True)
        self.dev, self.dt = torch.device(device), dtype
        self.c6ab = torch.as_tensor(t["c6ab"], dtype=dtype, device=self.dev)  # [95,95,5,5,3]
        self.r0ab = torch.as_tensor(t["r0ab"], dtype=dtype, device=self.dev)  # Bohr
        self.rcov = torch.as_tensor(t["rcov"], dtype=dtype, device=self.dev)  # Bohr, pre-scaled by k2
        self.r2r4 = torch.as_tensor(t["r2r4"], dtype=dtype, device=self.dev)
        self.p = dict(PBE_ZERO if params is None else params)
        self.pair_cut, self.cn_cut, self.tail_cut = float(pair_cut), float(cn_cut), tail_cut
        self.pair_chunk = int(pair_chunk)

    def _shifts(self, cell: torch.Tensor, cut: float) -> torch.Tensor:
        c = cell.detach().cpu().double().numpy()
        vol = abs(np.linalg.det(c))
        nrep = [int(math.ceil(cut / (vol / np.linalg.norm(np.cross(c[(k + 1) % 3], c[(k + 2) % 3]))))) for k in range(3)]
        g = [torch.arange(-n, n + 1, dtype=self.dt, device=self.dev) for n in nrep]
        return torch.cartesian_prod(*g)  # [T,3] integer multiples

    def energy(self, z: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """eV; z long [n], pos [n,3] A, cell [3,3] A (fully periodic)."""
        p = self.p
        n = z.shape[0]
        cut = max(self.pair_cut, self.cn_cut)
        S = self._shifts(cell, cut) @ cell  # [T,3] A
        d = pos[None, :, None, :] - pos[:, None, None, :] + S[None, None]  # [i,j,T,3]
        r2 = (d * d).sum(-1)
        self_mask = r2 < 1e-12
        r = torch.sqrt(torch.where(self_mask, torch.ones_like(r2), r2)) / BOHR  # Bohr
        zi, zj = z[:, None], z[None, :]
        rc = (self.rcov[zi] + self.rcov[zj])[..., None]
        cn_term = torch.sigmoid(16.0 * (rc / r - 1.0))
        cn_term = torch.where(self_mask | (r > self.cn_cut / BOHR), torch.zeros_like(cn_term), cn_term)
        cn = cn_term.sum((1, 2))  # [n]
        ref = self.c6ab[zi, zj].reshape(n, n, 25, 3)
        c6r, cni, cnj = ref[..., 0], ref[..., 1], ref[..., 2]
        sc = -4.0 * ((cni - cn[:, None, None]) ** 2 + (cnj - cn[None, :, None]) ** 2)
        sc = torch.where(c6r > 0, sc, torch.full_like(sc, -1e20))
        c6 = (torch.softmax(sc, -1) * c6r).sum(-1)  # [n,n] a.u.
        c8 = 3.0 * c6 * self.r2r4[zi] * self.r2r4[zj]
        r0 = self.r0ab[zi, zj][..., None]
        f6 = 1.0 / (1.0 + 6.0 * (r / (p["sr6"] * r0)) ** (-p["alpha"]))
        f8 = 1.0 / (1.0 + 6.0 * (r / (p["sr8"] * r0)) ** (-(p["alpha"] + 2.0)))
        e = p["s6"] * c6[..., None] * f6 / r ** 6 + p["s8"] * c8[..., None] * f8 / r ** 8
        e = torch.where(self_mask | (r > self.pair_cut / BOHR), torch.zeros_like(e), e)
        E = -0.5 * e.sum()
        if self.tail_cut is not None and self.tail_cut > self.pair_cut:
            vol = torch.abs(torch.linalg.det(cell)) / BOHR ** 3
            a, b = self.pair_cut / BOHR, self.tail_cut / BOHR
            i6 = 4.0 * math.pi * (a ** -3 - b ** -3) / 3.0
            i8 = 4.0 * math.pi * (a ** -5 - b ** -5) / 5.0
            E = E - 0.5 * (p["s6"] * c6.sum() * i6 + p["s8"] * c8.sum() * i8) / vol
        return E * HARTREE

    def __call__(self, z, pos, cell) -> tuple[float, np.ndarray, np.ndarray]:
        """(E eV, F [n,3] eV/A, ASE Voigt stress [6] eV/A^3 tensile-positive)."""
        z = torch.as_tensor(np.asarray(z), dtype=torch.long, device=self.dev)
        pos0 = torch.as_tensor(np.asarray(pos), dtype=self.dt, device=self.dev)
        cell0 = torch.as_tensor(np.asarray(cell), dtype=self.dt, device=self.dev)
        eps = torch.zeros(3, 3, dtype=self.dt, device=self.dev, requires_grad=True)
        pos_ = pos0.clone().requires_grad_(True)
        defm = torch.eye(3, dtype=self.dt, device=self.dev) + eps
        E = self.energy(z, pos_ @ defm, cell0 @ defm)
        gpos, geps = torch.autograd.grad(E, (pos_, eps))
        vol = float(torch.abs(torch.linalg.det(cell0)))
        s = (geps + geps.T).detach().cpu().numpy() / (2.0 * vol)
        voigt = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        return float(E.detach()), (-gpos).detach().cpu().numpy(), voigt
