"""Cartesian harmonics of a unit vector u = (x, y, z) for l = 1, 2, 3, numerically identical to the earlier Mena trunk's
`l2_pair(u, u)` and `l3_harmonic(u)` (lam/models/mena_ace.py), including their normalisation (tests compare at 1e-12).

l = 1: u itself (|y1|^2 = 1).
l = 2 (5), components of u^T E_m u in an orthonormal basis of symmetric traceless 3x3 matrices (|y2|^2 = 2/3):
    sqrt2 xy, sqrt2 xz, sqrt2 yz, (x^2 - y^2) / sqrt2, (2 z^2 - x^2 - y^2) / sqrt6.
l = 3 (7), coordinates B_m . (u u u) in an orthonormal basis of rank-3 symmetric traceless tensors (|y3|^2 = 2/5), the harmonic
polynomials  y(3x^2 - y^2), xyz, y(4z^2 - x^2 - y^2), z(2z^2 - 3x^2 - 3y^2), x(4z^2 - x^2 - y^2), z(x^2 - y^2), x(x^2 - 3y^2)
each times 1 / ||tensor|| = 1/2, sqrt6, sqrt(3/20), 1/sqrt10, sqrt(3/20), sqrt(3/2), 1/2.
Any rotation R acts as an orthogonal block D_l(R) on each; `rotation_matrix_for_l` recovers it numerically (tests use it).
Pure elementwise arithmetic, so autograd and double backward are cheap and exact.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

_S2, _S6, _S10 = math.sqrt(2.0), math.sqrt(6.0), math.sqrt(10.0)
_C3 = (0.5, _S6, math.sqrt(3.0 / 20.0), 1.0 / _S10, math.sqrt(3.0 / 20.0), math.sqrt(1.5), 0.5)  # 1 / ||B_m|| of the 7 polynomials
Y2_NORM2 = 2.0 / 3.0  # |y2(u)|^2 for a unit vector
Y3_NORM2 = 2.0 / 5.0  # |y3(u)|^2 for a unit vector


def y2(u: Tensor) -> Tensor:
    x, y, z = u.unbind(-1)
    return torch.stack((_S2 * x * y, _S2 * x * z, _S2 * y * z, (x * x - y * y) / _S2, (2.0 * z * z - x * x - y * y) / _S6), dim=-1)


def y3(u: Tensor) -> Tensor:
    x, y, z = u.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    c = _C3
    return torch.stack((
        c[0] * y * (3.0 * xx - yy), c[1] * x * y * z, c[2] * y * (4.0 * zz - xx - yy), c[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        c[4] * x * (4.0 * zz - xx - yy), c[5] * z * (xx - yy), c[6] * x * (xx - 3.0 * yy)), dim=-1)


def harmonics(u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(y1 [..., 3], y2 [..., 5], y3 [..., 7]) of unit vectors u [..., 3]."""
    return u, y2(u), y3(u)


def rotation_matrix_for_l(l: int, rot: Tensor, n_probe: int = 64, seed: int = 0) -> Tensor:
    """D_l(R) [2l+1, 2l+1] with y_l(R u) = D_l(R) y_l(u), by least squares on random unit vectors (float64)."""
    if l == 1:
        return rot
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n_probe, 3, generator=g, dtype=rot.dtype)
    u = u / u.norm(dim=-1, keepdim=True)
    fn = {2: y2, 3: y3}[l]
    a, b = fn(u), fn(u @ rot.T)  # b = a D^T
    return torch.linalg.lstsq(a, b).solution.T
