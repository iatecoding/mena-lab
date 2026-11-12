"""Per-stage radial filter tables: R(x) = Phi(x) W tabulated once per forward as a C2 CUBIC B-SPLINE, evaluated per edge.

R_c(x) = sum_k Phi_k(x) W[k, c] with Phi = the 32 Gaussians (radial.py; NO envelope: chi(r) multiplies the edge message in edge_terms.py), W a
learned [32, C] matrix. Uniform knots x_g = g h, g = 0..G, h = r_cut / G. The spline S(x) = sum_m c_m B3(x / h - m + 1), m = -1..G+1, has
G + 3 coefficients: the table is `coef` [G + 3, C], ONE float per knot per channel (index a = m + 1). On [x_g, x_g + h], t = (x - x_g) / h:

    S = b0(t) c[g] + b1(t) c[g+1] + b2(t) c[g+2] + b3(t) c[g+3]
    b0 = (1-t)^3/6, b1 = (3t^3 - 6t^2 + 4)/6, b2 = (-3t^3 + 3t^2 + 3t + 1)/6, b3 = t^3/6      (the 4 weights: once per edge)
    S(x_g) = (c[g] + 4 c[g+1] + c[g+2]) / 6,   S'(x_g) = (c[g+2] - c[g]) / (2h),   S''(x_g) = (c[g] - 2 c[g+1] + c[g+2]) / h^2

Coefficients (clamped spline): solve the collocation system  S(x_g) = Phi(x_g) W (G+1 equations) with the EXACT end derivatives
S'(0) = Phi'(0) W, S'(r_cut) = Phi'(r_cut) W (analytic Phi', 2 equations): (G+3) x (G+3) tridiagonal M c = rhs(W). rhs is linear in W, so
c = P W with a constant [G+3, 32] matrix P = M^-1 R built once in float64: the per-forward cost is one [G+3, 32] x [32, C] matmul, it is
differentiable in W, and nothing is solved at run time. A clamped cubic spline is C2 at every knot and its error is O(h^4) (value) /
O(h^3) (derivative); autograd differentiates the piecewise polynomial in t (the knot index is a constant), so the force is the exact
gradient of the interpolated energy.

x is clamped to [0, r_cut]: beyond r_cut the table returns its value at r_cut (constant, zero derivative). That is harmless: the message carries
chi(r), exactly 0 with zero derivative for r >= r_cut, so padded edges at r = r_cut stay exact no-ops (the table is finite everywhere).

Grid size: DEFAULT_GRID from `interpolation_error`. The 32 Gaussians have a
constant width, so a uniform grid has uniform error.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lam4.core.consts import storable
from lam4.core.radial import GaussianBasis, N_RBF, R_CUT

# h = 1/128 A (exact in binary, so x / h is exact in float32). Smallest G whose float32 relative error is < 1e-5 for value AND first
# derivative over every one of 10 random W draws (worst 5.4e-6; G = 512 reaches 1.1e-5 on some draw). Beyond
# ~G = 700 the float32 derivative is limited by cancellation in sum(w' c) / h, not by the spline (float64: 1.7e-6 at 768).
DEFAULT_GRID = 768
_P_WIN64: dict[tuple[int, float, int], Tensor] = {}
_DEFAULT_P_WIN: dict = {}


def _default_p_windows(r_cut: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    key = (float(r_cut), dtype, device)
    c = _DEFAULT_P_WIN.get(key)
    if c is None:
        k64 = (DEFAULT_GRID, float(r_cut), N_RBF)
        w64 = _P_WIN64.get(k64)
        if w64 is None:
            w64 = windows(BSplineTables(1, DEFAULT_GRID, r_cut)._p64).detach().contiguous()
            if storable(w64):
                _P_WIN64[k64] = w64
        c = w64.to(dtype=dtype).to(device=device)
        if storable(c):
            _DEFAULT_P_WIN[key] = c
    return c


def _weights(t: Tensor, order: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The 4 cubic B-spline weights (order 0) or their t-derivatives (order 1, 2) at t in [0, 1]."""
    if order == 0:
        t2 = t * t
        t3 = t2 * t
        return (1.0 - t) ** 3 / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0
    if order == 1:
        t2 = t * t
        return -0.5 * (1.0 - t) ** 2, 1.5 * t2 - 2.0 * t, -1.5 * t2 + t + 0.5, 0.5 * t2
    if order == 2:
        return 1.0 - t, 3.0 * t - 2.0, 1.0 - 3.0 * t, t
    raise ValueError("order must be 0, 1 or 2")


def windows(coef: Tensor) -> Tensor:
    """coef [G+3, C] -> [G, 4, C] (identity if already 3-D or factored [32, C]): row g holds (c[g], c[g+1], c[g+2], c[g+3]); built once per stage."""
    if coef.ndim == 3 or (coef.ndim == 2 and coef.shape[0] == N_RBF):
        return coef
    g, c = coef.shape[0] - 3, coef.shape[1]
    return torch.cat([coef[i:i + g] for i in range(4)], dim=1).view(g, 4, c)


def default_bspline_basis(x: Tensor, r_cut: float = R_CUT, order: int = 0) -> Tensor:
    """Spline-interpolated Gaussian basis [E, 32] on DEFAULT_GRID; shared across stages in `ref` mode via `phi @ W`."""
    return bspline_eval(x, _default_p_windows(r_cut, x.dtype, x.device), r_cut, order)


def bspline_eval(x: Tensor, coef: Tensor, r_cut: float = R_CUT, order: int = 0) -> Tensor:
    """x [E] -> d^order S / dx^order [E, C]. `coef` is [G+3, C], pre-windowed `windows(coef)` [G, 4, C], or factored `W` [32, C]."""
    if coef.ndim == 2 and coef.shape[0] == N_RBF:
        return default_bspline_basis(x, r_cut, order) @ coef
    wtab = windows(coef)
    g = wtab.shape[0]
    h = r_cut / g
    u = x.clamp(0.0, r_cut) / h
    idx = u.detach().floor().clamp(0, g - 1).to(torch.long)
    t = u - idx.to(u.dtype)
    w = _weights(t, order)
    win = wtab.index_select(0, idx)
    out = w[0].unsqueeze(-1) * win[:, 0] + w[1].unsqueeze(-1) * win[:, 1] + w[2].unsqueeze(-1) * win[:, 2] + w[3].unsqueeze(-1) * win[:, 3]
    if order:
        out = out / h ** order
    return out


class BSplineTables(nn.Module):
    """W [32, n_out] and the constant float64 solve matrix P; forward() returns coef [G + 3, n_out] (rebuild once per forward pass)."""

    def __init__(self, n_out: int, grid: int = DEFAULT_GRID, r_cut: float = R_CUT, n_rbf: int = N_RBF,
                 init_scale: float | None = None) -> None:
        super().__init__()
        self.n_out, self.grid, self.r_cut = int(n_out), int(grid), float(r_cut)
        self.basis = GaussianBasis(n_rbf, r_cut)
        self._p64 = self._solve_matrix()
        if storable(self._p64):
            _P_WIN64.setdefault((self.grid, self.r_cut, int(n_rbf)), windows(self._p64).detach().contiguous())
        self._cache: dict = {}
        scale = (1.0 / n_rbf) ** 0.5 if init_scale is None else init_scale
        self.weight = nn.Parameter(torch.randn(n_rbf, self.n_out) * scale)

    def _solve_matrix(self) -> Tensor:
        """P [G+3, 32] float64 with coef = P W: clamped-spline collocation at the knots with exact end derivatives."""
        g, h = self.grid, self.r_cut / self.grid
        xg = torch.linspace(0.0, self.r_cut, g + 1, dtype=torch.float64)
        phi, dphi = self.basis(xg), self.basis.derivative(xg)
        m = torch.zeros(g + 3, g + 3, dtype=torch.float64)
        rhs = torch.zeros(g + 3, phi.shape[1], dtype=torch.float64)
        m[0, 0], m[0, 2], rhs[0] = -1.0, 1.0, 2.0 * h * dphi[0]
        for i in range(g + 1):  # c[i] + 4 c[i+1] + c[i+2] = 6 S(x_i)
            m[i + 1, i:i + 3] = torch.tensor([1.0, 4.0, 1.0], dtype=torch.float64)
            rhs[i + 1] = 6.0 * phi[i]
        m[g + 2, g], m[g + 2, g + 2], rhs[g + 2] = -1.0, 1.0, 2.0 * h * dphi[g]
        return torch.linalg.solve(m, rhs)

    def solve_matrix(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        key = (dtype, device)  # float64 kept out of the buffers (MPS cannot hold float64): cast per (dtype, device) on use
        c = self._cache.get(key)
        if c is None:
            c = self._p64.to(dtype=dtype).to(device=device)
            if storable(c):   # never cache a traced / fake cast (lam4.core.consts)
                self._cache[key] = c
        return c

    def coefficients(self, w: Tensor | None = None) -> Tensor:
        w = self.weight if w is None else w
        return self.solve_matrix(w.dtype, w.device) @ w

    def forward(self) -> Tensor:
        return self.coefficients()

    def exact(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Exact R(x) and R'(x) (no interpolation) for x [E] in the dtype of x; the oracle for the error table."""
        w = self.weight.to(x.dtype)
        return self.basis(x) @ w, self.basis.derivative(x) @ w


def interpolation_error(grid: int, n_samples: int = 200_000, seed: int = 0, dtype: torch.dtype = torch.float32,
                        n_out: int = 252) -> dict[str, float]:
    """Max and relative (max / max|exact|) error of the spline value and first derivative vs exact Phi(x) W.

    P is always built in float64; `dtype` is the evaluation precision of coef and x (float32 includes round-off).
    The derivative is the analytic d/dx of the interpolant (equal to its autograd derivative, tested), compared with Phi' W.
    n_out = 252 = 4 head biases + the 248 filter channels of the LATER layout.
    """
    torch.manual_seed(seed)
    tables = BSplineTables(n_out, grid).double()
    coef = tables().detach().to(dtype)
    xe = (torch.rand(n_samples, dtype=torch.float64) * R_CUT).to(dtype)  # exact oracle at the SAME rounded x
    with torch.no_grad():
        ex_val, ex_der = tables.exact(xe.double())
    err_v = (bspline_eval(xe, coef).double() - ex_val).abs()
    err_d = (bspline_eval(xe, coef, order=1).double() - ex_der).abs()
    return {
        "grid": grid, "h": R_CUT / grid, "max_abs_val": float(err_v.max()), "rel_val": float(err_v.max() / ex_val.abs().max()),
        "max_abs_der": float(err_d.max()), "rel_der": float(err_d.max() / ex_der.abs().max()),
    }
