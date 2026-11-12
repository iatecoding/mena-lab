"""Radial geometry for lam4, numerically the earlier Mena geometry (GaussianRBF(envelope=False),
taper_envelope(c3=True), pair_log_radius endpoint warp); tests compare at 1e-12.

Gaussian basis: N_RBF = 32 centres mu_k = linspace(0, r_cut, 32) (spacing 0.1935 A at r_cut = 6), width sigma = r_cut / N_RBF = 0.1875 A
(gamma = 0.5 / sigma^2). Phi_k(x) = exp(-(x - mu_k)^2 / (2 sigma^2)) evaluated at the WARPED distance x. No envelope inside the table basis.

Envelope chi (flat-top septic smoothstep), evaluated on the UNWARPED distance r exactly as earlier Mena does:
    t = clamp((r - r_on) / (r_cut - r_on), 0, 1),  chi = 1 - S(t),  S(t) = t^4 (35 - 84 t + 70 t^2 - 20 t^3)
    S'(t) = 140 t^3 (1 - t)^3,  S''(t) = 420 t^2 (1 - t)^2 (1 - 2 t)
S and its first three derivatives vanish at t = 1, so chi is C3 at r_cut and chi = chi' = 0 exactly at r = r_cut and beyond. Default r_on = 5,
r_cut = 6. The per-edge message carries the factor chi(r); chi(r) also enters the degree sum D and the head-weight normaliser (edge_terms.py).

Pair warp (as earlier Mena trunk): rho_i = exp(log_rho[Z_i]) (init: ase covalent radii), s_ij = 1.52 / (rho_i + rho_j),
    x = s_ij r / (1 + (s_ij - 1) r / r_cut)       (Moebius endpoint warp)
Near r = 0, x ~ s_ij r; the warp is strictly increasing and x(r_cut) = r_cut for every pair. Only the Gaussians see x; chi sees r. For r >= r_cut we
return r_cut exactly (the Moebius form can round 1 ulp away from r_cut). Padding contract: an edge at r = r_cut has chi = chi' = 0 exactly, so its
message and every gradient through it are exactly 0 whatever the table holds there.
The ZBL / NLH clamp is not part of this module.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

N_RBF = 32
R_CUT = 6.0
R_ON = 5.0
WARP_NUM = 1.52  # s_ij = WARP_NUM / (rho_i + rho_j): identity warp for rho_i + rho_j = 1.52 (e.g. C-C)
FALLBACK_RADIUS = 0.76  # Angstrom, for elements without an ase covalent radius


def _smoothstep_t(x: Tensor, r_on: float, r_cut: float) -> Tensor:
    return ((x - r_on) / (r_cut - r_on)).clamp(min=0.0, max=1.0)


def c3_envelope(x: Tensor, r_on: float = R_ON, r_cut: float = R_CUT) -> Tensor:
    """env(x): 1 for x <= r_on, 0 for x >= r_cut, C3 in between."""
    t = _smoothstep_t(x, r_on, r_cut)
    t2 = t * t
    return 1.0 - t2 * t2 * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)))


def c3_envelope_d(x: Tensor, r_on: float = R_ON, r_cut: float = R_CUT, order: int = 1) -> Tensor:
    """Analytic d^order env / dx^order for order 1 or 2 (zero outside (r_on, r_cut))."""
    t = _smoothstep_t(x, r_on, r_cut)
    w = r_cut - r_on
    inside = ((x > r_on) & (x < r_cut)).to(x.dtype)
    if order == 1:
        return -140.0 * t ** 3 * (1.0 - t) ** 3 / w * inside
    if order == 2:
        return -420.0 * t ** 2 * (1.0 - t) ** 2 * (1.0 - 2.0 * t) / (w * w) * inside
    raise ValueError("order must be 1 or 2")


class GaussianBasis(nn.Module):
    """Phi(x) [..., 32] = the 32 Gaussians at x and their exact x-derivative. No parameters, no envelope."""

    def __init__(self, n_rbf: int = N_RBF, r_cut: float = R_CUT) -> None:
        super().__init__()
        self.n_rbf, self.r_cut = int(n_rbf), float(r_cut)
        self.sigma = self.r_cut / self.n_rbf

    def centres(self, like: Tensor) -> Tensor:
        """The 32 fixed centres in the dtype/device of `like` (computed in float64 first; no buffers, so MPS .to() works)."""
        return torch.linspace(0.0, self.r_cut, self.n_rbf, dtype=torch.float64).to(device=like.device, dtype=like.dtype)

    def gaussians(self, x: Tensor) -> Tensor:
        z = (x.unsqueeze(-1) - self.centres(x)) / self.sigma
        return torch.exp(-0.5 * z * z)

    def forward(self, x: Tensor) -> Tensor:
        return self.gaussians(x)

    def derivative(self, x: Tensor) -> Tensor:
        """dPhi/dx, analytic."""
        return -self.gaussians(x) * (x.unsqueeze(-1) - self.centres(x)) / (self.sigma ** 2)


def covalent_radii(num_elements: int) -> Tensor:
    """[num_elements] covalent radii (Angstrom, ase table, index = Z) with the earlier Mena fallback for missing / non-positive entries."""
    from ase.data import covalent_radii as ase_rad
    rad = torch.tensor(ase_rad[:num_elements], dtype=torch.get_default_dtype())
    if rad.numel() < num_elements:
        rad = torch.cat([rad, rad.new_full((num_elements - rad.numel(),), FALLBACK_RADIUS)])
    return torch.where(torch.isfinite(rad) & (rad > 0), rad, torch.full_like(rad, FALLBACK_RADIUS))


class PairWarp(nn.Module):
    """Learnable per-element radius rho_Z = exp(log_rho[Z]) (init: covalent radii); the edge op forms rho_ij = rho_i + rho_j."""

    def __init__(self, num_elements: int, r_cut: float = R_CUT) -> None:
        super().__init__()
        self.r_cut = float(r_cut)
        self.log_rho = nn.Parameter(covalent_radii(num_elements).log())

    def node_rho(self, z: Tensor) -> Tensor:
        """Per-atom rho_i = exp(log_rho[Z_i]) [N]; the edge op takes rho_ij = rho(recv) + rho(send) from it."""
        return self.log_rho.exp().index_select(0, z)

    def forward(self, r: Tensor, rho_i: Tensor, rho_j: Tensor) -> Tensor:
        return warp_distance(r, rho_i + rho_j, self.r_cut)


def warp_distance(r: Tensor, rho_ij: Tensor | None, r_cut: float = R_CUT) -> Tensor:
    """x(r) for rho_ij = rho_i + rho_j > 0 (None = identity). Returns exactly r_cut where r >= r_cut."""
    if rho_ij is None:
        return r
    s = WARP_NUM / rho_ij
    warped = s * r / (1.0 + (s - 1.0) * (r / r_cut))
    return torch.where(r < r_cut, warped, torch.full_like(r, r_cut))
