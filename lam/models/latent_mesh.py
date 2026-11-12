"""latent_mesh.py: long-range operators of Mena-LAM on a particle mesh ("latent PME").

Every long-range operator of the model is a radial convolution of per-atom channels,
    y_i = sum_j sum_n K(|r_i - r_j + n L|) x_j          (j = i, n = 0 included: the self term K(0) x_i),
with K a smooth radial kernel: the island-context windows (C3 tapers of >= 3 A width) and the long-range part
erf(r / sigma) / r of Coulomb. Smooth particle-mesh Ewald (Essmann et al. 1995, J. Chem. Phys. 103, 8577) evaluates
that in O(N + M log M) with no long-range neighbour list. This module is the SPME machinery for a general kernel:
    1. fractional coordinates u = M * (r cell^-1), cardinal B-splines of order p spread x onto the M1 x M2 x M3 grid;
    2. real FFT, multiply by G(m) = (N_grid / V) * K_hat(|k_m|) * |b(m)|^2, inverse FFT;
    3. interpolate back with the same B-spline weights.
K_hat(k) = 4 pi int_0^R K(r) r^2 sin(kr)/(kr) dr is tabulated once per kernel (numerical radial transform), or given
analytically. Forces come from autograd through the spline weights (inference path; training uses the dense exact form).

Open boundaries: a compact kernel (support R) evaluated in a box padded by >= R + p h has no image contributions, so
the periodic result equals the open-boundary sum exactly up to mesh error. Coulomb in vacuum therefore uses the long-range
part with a C3 truncation beyond the system diameter (`erf_coulomb_kernel(sigma, r_trunc=...)`).

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
from torch import Tensor


def c3_taper(r: Tensor, r_on: float, r_cut: float) -> Tensor:
    """1 for r <= r_on, 0 for r >= r_cut, septic C3 step in between (same polynomial as mena_ace.taper_envelope)."""
    x = ((r - r_on) / (r_cut - r_on)).clamp(0.0, 1.0)
    return 1.0 - x.pow(4) * (35.0 - 84.0 * x + 70.0 * x.pow(2) - 20.0 * x.pow(3))


@dataclass
class RadialKernel:
    """A smooth radial kernel K(r) with support r <= r_max (math.inf for none) and its 3-D Fourier transform.

    real(r): K at distances r (any shape). fourier(k): K_hat at |k| (any shape), from `fourier_fn` when given
    (analytic), else from a table built by a radial transform on first use (float64, cached per device)."""

    real_fn: Callable[[Tensor], Tensor]
    r_max: float
    name: str = ""
    fourier_fn: Callable[[Tensor], Tensor] | None = None
    dk: float = 2.0e-3
    dr: float = 2.0e-3
    _table: dict = field(default_factory=dict, repr=False)

    def real(self, r: Tensor) -> Tensor:
        return self.real_fn(r)

    def fourier(self, k: Tensor) -> Tensor:
        if self.fourier_fn is not None:
            return self.fourier_fn(k)
        k_hi = float(k.max().detach()) if k.numel() else 0.0
        key = (k.device, k.dtype)
        tab = self._table.get(key)
        if tab is None or tab[1] < k_hi:
            tab = self._build_table(max(k_hi * 1.25, 1.0), k.device, k.dtype)
            self._table[key] = tab
        vals, k_top = tab
        x = (k / self.dk).clamp(max=vals.shape[0] - 1.000001)
        i0 = x.floor().long()
        t = x - i0.to(x.dtype)
        return vals[i0] * (1.0 - t) + vals[i0 + 1] * t

    def _build_table(self, k_max: float, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, float]:
        if not math.isfinite(self.r_max):
            raise ValueError(f"kernel {self.name!r}: numerical transform needs compact support (r_max finite)")
        n_r = int(math.ceil(self.r_max / self.dr)) + 1
        n_r += (n_r + 1) % 2  # odd point count for Simpson
        r = torch.linspace(0.0, self.r_max, n_r, dtype=torch.float64)
        w = torch.ones(n_r, dtype=torch.float64)
        w[1:-1:2], w[2:-1:2] = 4.0, 2.0
        w *= (r[1] - r[0]) / 3.0
        f = self.real_fn(r) * r.pow(2) * w  # [n_r]
        n_k = int(math.ceil(k_max / self.dk)) + 2
        k = torch.arange(n_k, dtype=torch.float64) * self.dk
        vals = torch.empty(n_k, dtype=torch.float64)
        for a in range(0, n_k, 512):  # sinc(kr) in blocks: [512, n_r]
            kr = k[a:a + 512, None] * r[None, :]
            vals[a:a + 512] = 4.0 * math.pi * (torch.sinc(kr / math.pi) * f).sum(-1)
        return vals.to(device=device, dtype=dtype), k_max


def window_kernel(r_on: float, r_cut: float) -> RadialKernel:
    """The island-context window: 1 to r_on, C3 taper to 0 at r_cut. Mesh-friendly when r_cut - r_on >= ~3 A."""
    return RadialKernel(lambda r: c3_taper(r, r_on, r_cut), float(r_cut), name=f"window({r_on},{r_cut})")


def erf_coulomb_kernel(sigma: float, r_trunc: float | None = None, trunc_width: float = 4.0) -> RadialKernel:
    """Long-range Coulomb part erf(r / sigma) / r (finite at r = 0: 2 / (sigma sqrt(pi))).
    Periodic (r_trunc None): analytic K_hat = 4 pi exp(-k^2 sigma^2 / 4) / k^2, with the k = 0 term set to 0 (neutralising
    background, as in Ewald). Open boundary: truncated by a C3 step over [r_trunc - trunc_width, r_trunc]; choose
    r_trunc - trunc_width above the system diameter so no real pair feels the truncation."""
    s = float(sigma)
    c0 = 2.0 / (s * math.sqrt(math.pi))

    def k_real(r: Tensor) -> Tensor:
        safe = r.clamp_min(1e-6)
        return torch.where(r < 1e-6, torch.full_like(r, c0), torch.erf(safe / s) / safe)

    if r_trunc is None:
        def k_hat(k: Tensor) -> Tensor:
            k2 = k.pow(2)
            return torch.where(k2 > 0, 4.0 * math.pi * torch.exp(-k2 * s * s / 4.0) / k2.clamp_min(1e-30),
                               torch.zeros_like(k2))
        return RadialKernel(k_real, math.inf, name=f"erf_coulomb({s})", fourier_fn=k_hat)
    rt = float(r_trunc)
    return RadialKernel(lambda r: k_real(r) * c3_taper(r, rt - trunc_width, rt), rt, name=f"erf_coulomb({s},{rt})")


def _bspline_weights(w: Tensor, order: int) -> Tensor:
    """Cardinal B-spline weights M_p(w + k), k = 0..p-1, for fractional offsets w in [0, 1): [..., p].
    theta[..., k] is the weight of grid point floor(u) - k. Recursion M_n(x) = (x M_{n-1}(x) + (n - x) M_{n-1}(x - 1)) / (n - 1)
    from M_2(x) = 1 - |x - 1| (Essmann et al. 1995, eq. 4.1)."""
    cur = [w, 1.0 - w]  # cur[k] = M_2(w + k), k = 0, 1  (M_2(x) = 1 - |x - 1| on [0, 2])
    for n in range(3, order + 1):
        nxt = []
        for k in range(n):
            x = w + k
            left = cur[k] if k < n - 1 else torch.zeros_like(w)
            right = cur[k - 1] if k >= 1 else torch.zeros_like(w)
            nxt.append((x * left + (n - x) * right) / (n - 1))
        cur = nxt
    return torch.stack(cur, dim=-1)


def _bspline_moduli(M: int, order: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """|b(m)|^2 for m = 0..M-1 (Essmann eq. 4.4): 1 / |sum_{k=0}^{p-2} M_p(k + 1) exp(2 pi i m k / M)|^2."""
    knots = _bspline_weights(torch.zeros(1, dtype=torch.float64), order)[0]  # M_p(0 + k), k = 0..p-1
    mp = knots[1:]  # M_p(k + 1), k = 0..p-2
    m = torch.arange(M, dtype=torch.float64)
    k = torch.arange(order - 1, dtype=torch.float64)
    ang = 2.0 * math.pi * m[:, None] * k[None, :] / M
    re = (mp * torch.cos(ang)).sum(-1)
    im = (mp * torch.sin(ang)).sum(-1)
    den = re.pow(2) + im.pow(2)
    out = torch.where(den > 1e-10, 1.0 / den.clamp_min(1e-30), torch.zeros_like(den))
    if order % 2 == 1:  # odd orders vanish at the Nyquist frequency; interpolate (standard fix)
        for i in range(M):
            if den[i] <= 1e-10:
                out[i] = 0.5 * (out[(i - 1) % M] + out[(i + 1) % M])
    return out.to(device=device, dtype=dtype)


def _good_fft_size(n: int) -> int:
    n = max(int(n), 4)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


@dataclass
class MeshGrid:
    """Grid geometry for one structure: cell [3, 3] (rows = lattice vectors), sizes (M1, M2, M3), spline order."""

    cell: Tensor
    sizes: tuple[int, int, int]
    order: int = 6

    @classmethod
    def for_cell(cls, cell: Tensor, spacing: float = 1.0, order: int = 6) -> "MeshGrid":
        lengths = cell.detach().norm(dim=-1)
        sizes = tuple(_good_fft_size(math.ceil(float(L) / spacing)) for L in lengths)
        return cls(cell, sizes, order)

    @classmethod
    def open_boundary(cls, pos: Tensor, r_kernel: float, spacing: float = 1.0, order: int = 6) -> tuple["MeshGrid", Tensor]:
        """Orthorhombic box around pos padded so no periodic image is within r_kernel; returns (grid, shifted pos)."""
        lo = pos.detach().min(0).values
        hi = pos.detach().max(0).values
        pad = float(r_kernel) + order * spacing + 1.0
        L = (hi - lo) + pad
        cell = torch.diag(L).to(pos)
        return cls.for_cell(cell, spacing, order), pos - lo + 0.5 * order * spacing


def mesh_convolve(pos: Tensor, x: Tensor, grid: MeshGrid, kernels: Sequence[RadialKernel],
                  channel_kernel: Sequence[int] | None = None) -> Tensor:
    """y_i = sum_j sum_images K_c(|r_i - r_j + n|) x_jc for every channel c (self term included), on the mesh.

    pos [N, 3], x [N, C]; kernels: the distinct kernels; channel_kernel[c] = index of channel c's kernel (default:
    all channels use kernels[0]). Differentiable in pos and x (autograd through the B-spline weights)."""
    N, C = x.shape
    if channel_kernel is None:
        channel_kernel = [0] * C
    dtype, device = pos.dtype, pos.device
    M = torch.tensor(grid.sizes, device=device)
    p = grid.order
    cell = grid.cell.to(dtype)
    inv = torch.linalg.inv(cell)
    u = (pos @ inv) * M.to(dtype)  # [N, 3] grid coordinates
    fl = u.detach().floor()
    w = u - fl  # [N, 3] in [0, 1)
    theta = _bspline_weights(w, p)  # [N, 3, p]: weight of grid point floor(u) - k
    base = fl.long()
    offs = torch.arange(p, device=device)
    idx = (base.unsqueeze(-1) - offs) % M.view(1, 3, 1)  # [N, 3, p]
    # flattened grid index and weight of each of the p^3 support points: [N, p^3]
    M1, M2, M3 = grid.sizes
    flat = (idx[:, 0, :, None, None] * (M2 * M3) + idx[:, 1, None, :, None] * M3 + idx[:, 2, None, None, :]).reshape(N, -1)
    wt = (theta[:, 0, :, None, None] * theta[:, 1, None, :, None] * theta[:, 2, None, None, :]).reshape(N, -1)

    Q = x.new_zeros(M1 * M2 * M3, C).index_add_(0, flat.reshape(-1), (wt.unsqueeze(-1) * x.unsqueeze(1)).reshape(-1, C))
    Qh = torch.fft.rfftn(Q.T.reshape(C, M1, M2, M3), dim=(1, 2, 3))  # [C, M1, M2, M3//2+1]

    # reciprocal vectors k = 2 pi (m1 a1* + m2 a2* + m3 a3*), a* = rows of inv^T
    f1 = torch.fft.fftfreq(M1, d=1.0 / M1, device=device).to(dtype)
    f2 = torch.fft.fftfreq(M2, d=1.0 / M2, device=device).to(dtype)
    f3 = torch.fft.rfftfreq(M3, d=1.0 / M3, device=device).to(dtype)
    rec = inv.T  # rows: reciprocal lattice vectors / (2 pi)
    kvec = 2.0 * math.pi * (f1[:, None, None, None] * rec[0] + f2[None, :, None, None] * rec[1]
                            + f3[None, None, :, None] * rec[2])
    kmag = kvec.norm(dim=-1)  # [M1, M2, M3//2+1]
    b2 = (_bspline_moduli(M1, p, dtype, device)[:, None, None] * _bspline_moduli(M2, p, dtype, device)[None, :, None]
          * _bspline_moduli(M3, p, dtype, device)[None, None, : M3 // 2 + 1])
    vol = torch.linalg.det(cell).abs()
    pref = (M1 * M2 * M3) / vol
    groups: dict[int, list[int]] = {}
    for c, kidx in enumerate(channel_kernel):
        groups.setdefault(int(kidx), []).append(c)
    Y = Qh.new_zeros(Qh.shape)
    for kidx, chans in groups.items():
        G = pref * kernels[kidx].fourier(kmag) * b2
        Y[chans] = Qh[chans] * G
    phi = torch.fft.irfftn(Y, s=(M1, M2, M3), dim=(1, 2, 3)).reshape(C, -1).T  # [M_tot, C]
    return (phi[flat] * wt.unsqueeze(-1)).sum(1)  # [N, C]


def dense_convolve(pos: Tensor, x: Tensor, kernels: Sequence[RadialKernel], channel_kernel: Sequence[int] | None = None,
                   cell: Tensor | None = None, n_images: int = 0) -> Tensor:
    """Exact reference of mesh_convolve: all pairs (and periodic images |n_a| <= n_images when cell is given)."""
    N, C = x.shape
    if channel_kernel is None:
        channel_kernel = [0] * C
    shifts = [torch.zeros(3, dtype=pos.dtype, device=pos.device)]
    if cell is not None and n_images > 0:
        rng = range(-n_images, n_images + 1)
        shifts = [i * cell[0] + j * cell[1] + k * cell[2] for i in rng for j in rng for k in rng]
    y = x.new_zeros(N, C)
    for s in shifts:
        r = (pos[:, None, :] - pos[None, :, :] + s).pow(2).sum(-1).add(1e-30).sqrt()  # [N, N]
        for c in range(C):
            y[:, c] = y[:, c] + kernels[channel_kernel[c]].real(r) @ x[:, c]
    return y


class MeshContext:
    """Per-structure mesh geometry for a batch (atoms of structure b: batch == b). Periodic structures use their cell;
    open-boundary structures get a padded orthorhombic box per kernel range (cached by range)."""

    def __init__(self, pos: Tensor, batch: Tensor, n_graphs: int, cell: Tensor | None, pbc: bool,
                 spacing: float = 0.8, order: int = 6) -> None:
        self.pos, self.pbc, self.spacing, self.order = pos, bool(pbc), float(spacing), int(order)
        self.idx = [torch.nonzero(batch == b).squeeze(-1) for b in range(int(n_graphs))]
        self.cell = cell
        self._open: dict[tuple[int, float], tuple[MeshGrid, Tensor]] = {}
        if self.pbc and cell is None:
            raise ValueError("MeshContext(pbc=True) needs the cell [B, 3, 3]")

    def extent(self, b: int) -> float:
        p = self.pos[self.idx[b]].detach()
        return float((p.max(0).values - p.min(0).values).norm()) if p.shape[0] > 1 else 0.0

    def _grid(self, b: int, r_range: float) -> tuple[MeshGrid, Tensor]:
        if self.pbc:
            return MeshGrid.for_cell(self.cell[b], self.spacing, self.order), self.pos[self.idx[b]]
        key = (b, float(r_range))
        if key not in self._open:
            self._open[key] = MeshGrid.open_boundary(self.pos[self.idx[b]], r_range, self.spacing, self.order)
        return self._open[key]

    def conv(self, x: Tensor, kernels: Sequence[RadialKernel], channel_kernel: Sequence[int] | None = None,
             kernel_for: Callable[[int], Sequence[RadialKernel]] | None = None) -> Tensor:
        """y_i = sum_j K_c(r_ij) x_jc within each structure (self term included). x [N] or [N, C].
        kernel_for(b) overrides `kernels` per structure (open-boundary Coulomb truncation depends on the size)."""
        squeeze = x.ndim == 1
        xx = x.unsqueeze(-1) if squeeze else x
        out = torch.zeros_like(xx)
        for b, ib in enumerate(self.idx):
            if ib.numel() == 0:
                continue
            ks = kernel_for(b) if kernel_for is not None else kernels
            r_range = max(k.r_max for k in ks)
            grid, pb = self._grid(b, r_range)
            out = out.index_put((ib,), mesh_convolve(pb, xx[ib], grid, ks, channel_kernel))
        return out.squeeze(-1) if squeeze else out


class MeshWindowPool:
    """IslandPool-compatible window pooling on the mesh: sum_j w(r_ij) x_j with w(0) = 1 (self included)."""

    kind = "mesh"

    def __init__(self, ctx: MeshContext, kernel: RadialKernel) -> None:
        self.ctx, self.kernel = ctx, kernel
        self._count: Tensor | None = None

    def sum(self, x: Tensor) -> Tensor:
        return self.ctx.conv(x, [self.kernel])

    def count(self, dtype: torch.dtype) -> Tensor:
        if self._count is None or self._count.dtype != dtype:
            ones = torch.ones(self.ctx.pos.shape[0], dtype=dtype, device=self.ctx.pos.device)
            self._count = self.ctx.conv(ones, [self.kernel]).clamp_min(1.0)
        return self._count

    def mean(self, x: Tensor) -> Tensor:
        c = self.count(x.dtype)
        return self.sum(x) / (c.unsqueeze(-1) if x.ndim > c.ndim else c)
