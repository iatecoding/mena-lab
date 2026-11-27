"""spme_batched.py: static-shape batched Smooth Particle-Mesh Ewald (SPME) for periodic LAM-6 electrostatics.

Evaluates smeared monopole+dipole Coulomb potential phi [N], field E [N, 3], optional traceless EFG [N, 5],
and per-atom reciprocal Cauchy virial W_recip [N, 3, 3] across packed periodic GraphBatches with zero dynamic
Python loops over graphs, analytic cardinal B-spline derivatives (M_p), live cell[b] d_cell virial stress,
and per-atom metallic screening gate g_i (skipping reciprocal FFTs when mean(g_i) < gate_skip).
"""
from __future__ import annotations

import math
import weakref

import torch
import torch.nn as nn
from torch import Tensor

from lam4.core.consts import capturing, is_fake, proxy_tracing, storable
from lam4.core.radial import fp32_guard, norm_device, stop_grad, upcast_fp32
from lam4.physics.coulomb_exact import COULOMB_K, SIGMA

_MODULI_CACHE: dict[tuple[int, int, torch.dtype, torch.device], Tensor] = {}
_SPME_MESH_CACHE: dict[tuple, tuple] = {}

DEFAULT_GRID_BUCKETS: tuple[tuple[int, int, int], ...] = (
    (12, 12, 12),
    (16, 16, 16),
    (24, 24, 24),
    (16, 16, 48),
)


def _register_spme_fft_ops() -> None:
    if hasattr(torch.ops.lam4, "spme_fft_conv"):
        return

    @torch.library.custom_op("lam4::spme_fft_conv", mutates_args=())
    def _spme_fft_conv(q_grid: Tensor, mult: Tensor) -> Tensor:
        qh = torch.fft.rfftn(q_grid, dim=(1, 2, 3))
        return torch.fft.irfftn(qh.unsqueeze(1) * mult, s=q_grid.shape[1:], dim=(2, 3, 4))

    @_spme_fft_conv.register_fake
    def _conv_fake(q_grid: Tensor, mult: Tensor) -> Tensor:
        return q_grid.new_empty(q_grid.shape[0], mult.shape[1], *q_grid.shape[1:])

    @torch.library.custom_op("lam4::spme_fft_adj_q", mutates_args=())
    def _spme_fft_adj_q(gy: Tensor, mult: Tensor) -> Tensor:
        gh = torch.fft.rfftn(gy, dim=(2, 3, 4))
        return torch.fft.irfftn((gh * mult).sum(1), s=gy.shape[2:], dim=(1, 2, 3))

    @_spme_fft_adj_q.register_fake
    def _adj_q_fake(gy: Tensor, mult: Tensor) -> Tensor:
        return gy.new_empty(gy.shape[0], *gy.shape[2:])

    @torch.library.custom_op("lam4::spme_fft_dm", mutates_args=())
    def _spme_fft_dm(gy: Tensor, q_grid: Tensor) -> Tensor:
        m3 = q_grid.shape[-1]
        w = gy.new_full((m3 // 2 + 1,), 2.0)
        w[0] = 1.0
        if m3 % 2 == 0:
            w[-1] = 1.0
        qh = torch.view_as_real(torch.fft.rfftn(q_grid, dim=(1, 2, 3))).unsqueeze(1)
        gh = torch.view_as_real(torch.fft.rfftn(gy, dim=(2, 3, 4)))
        return (gh * qh).sum(-1) * (w / float(math.prod(q_grid.shape[1:])))

    @_spme_fft_dm.register_fake
    def _dm_fake(gy: Tensor, q_grid: Tensor) -> Tensor:
        return gy.new_empty(gy.shape[0], gy.shape[1], gy.shape[2], gy.shape[3], gy.shape[4] // 2 + 1)

    def _save(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], inputs[1])

    _spme_fft_conv.register_autograd(
        lambda ctx, gy: (
            torch.ops.lam4.spme_fft_adj_q.default(gy, ctx.saved_tensors[1]) if ctx.needs_input_grad[0] else None,
            torch.ops.lam4.spme_fft_dm.default(gy, ctx.saved_tensors[0]) if ctx.needs_input_grad[1] else None,
        ),
        setup_context=_save,
    )
    _spme_fft_adj_q.register_autograd(
        lambda ctx, gq: (
            torch.ops.lam4.spme_fft_conv.default(gq, ctx.saved_tensors[1]) if ctx.needs_input_grad[0] else None,
            torch.ops.lam4.spme_fft_dm.default(ctx.saved_tensors[0], gq) if ctx.needs_input_grad[1] else None,
        ),
        setup_context=_save,
    )
    _spme_fft_dm.register_autograd(
        lambda ctx, gm: (
            torch.ops.lam4.spme_fft_conv.default(ctx.saved_tensors[1], gm) if ctx.needs_input_grad[0] else None,
            torch.ops.lam4.spme_fft_adj_q.default(ctx.saved_tensors[0], gm) if ctx.needs_input_grad[1] else None,
        ),
        setup_context=_save,
    )


_register_spme_fft_ops()


def good_fft_size(n: int, min_size: int = 8) -> int:
    """Smallest even 2^a * 3^b * 5^c >= max(n, min_size) for fast radix-2/3/5 real FFTs."""
    v = max(int(n), int(min_size))
    if v % 2 != 0:
        v += 1
    while True:
        m = v
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return v
        v += 2


def normalize_spme_grid(grid_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate and normalize a 3-tuple of SPME FFT mesh dimensions."""
    if len(grid_shape) != 3:
        raise ValueError(f"fixed_grid_shape must be a 3-tuple (m1, m2, m3), got {grid_shape!r}")
    m = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
    if min(m) < 2:
        raise ValueError(f"SPME grid dimensions must be >= 2, got {m!r}")
    return m


def select_spme_grid(cells: Tensor, spacing: float = 0.8, min_size: int = 8, *,
                     fixed_grid_shape: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    """Select smooth even 2^a 3^b 5^c FFT grid for arbitrary orthorhombic or triclinic periodic cells."""
    if fixed_grid_shape is not None:
        return normalize_spme_grid(fixed_grid_shape)
    c_det = cells.detach()
    if c_det.ndim == 2:
        c_det = c_det.unsqueeze(0)
    a, b, c = c_det.unbind(-2)
    lengths = c_det.norm(dim=-1)
    vol = torch.linalg.det(c_det).abs()
    valid = vol > 1e-6
    if not bool(valid.any()):
        ms = good_fft_size(min_size, min_size=min_size)
        return (ms, ms, ms)
    areas = torch.stack(
        (
            torch.linalg.cross(b, c, dim=-1).norm(dim=-1),
            torch.linalg.cross(c, a, dim=-1).norm(dim=-1),
            torch.linalg.cross(a, b, dim=-1).norm(dim=-1),
        ),
        dim=-1,
    )
    heights = vol.unsqueeze(-1) / areas.clamp_min(1e-12)
    # Skewness factor |a_i| / h_i (>= 1.0, == 1.0 for orthorhombic, invariant under uniform supercells).
    skew = (lengths / heights.clamp_min(1e-6)).clamp(1.0, 2.0)
    eff_len = (lengths * skew)[valid].max(0).values
    req = torch.ceil(eff_len / float(spacing)).to(torch.long)
    return (
        good_fft_size(int(req[0]), min_size=min_size),
        good_fft_size(int(req[1]), min_size=min_size),
        good_fft_size(int(req[2]), min_size=min_size),
    )


def select_grid_bucket(cells: Tensor, spacing: float = 0.8,
                       buckets: tuple[tuple[int, int, int], ...] = DEFAULT_GRID_BUCKETS, *,
                       fixed_grid_shape: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    """Select the smallest static FFT grid bucket guaranteeing effective spacing <= spacing outside trace."""
    if fixed_grid_shape is not None:
        return normalize_spme_grid(fixed_grid_shape)
    lengths = cells.detach().norm(dim=-1)
    valid = lengths.sum(-1) > 0
    if not bool(valid.any()):
        return buckets[0]
    req = torch.ceil(lengths[valid].max(0).values / float(spacing)).to(torch.long)
    need = (int(req[0]), int(req[1]), int(req[2]))
    best, best_vol = buckets[-1], math.prod(buckets[-1])
    for b in buckets:
        if all(b[a] >= need[a] for a in range(3)) and math.prod(b) < best_vol:
            best, best_vol = b, math.prod(b)
    return best


def _bspline_w_dw(w: Tensor, order: int) -> tuple[Tensor, Tensor]:
    """Cardinal B-spline weights M_p(w + k) and w-derivatives M_p'(w + k) for k = 0..p-1, w in [0, 1)."""
    cur = [w, 1.0 - w]  # order 2
    for n in range(3, order):
        nxt = []
        for k in range(n):
            x = w + k
            left = cur[k] if k < n - 1 else torch.zeros_like(w)
            right = cur[k - 1] if k >= 1 else torch.zeros_like(w)
            nxt.append((x * left + (n - x) * right) / (n - 1))
        cur = nxt
    theta, dtheta = [], []
    for k in range(order):
        x = w + k
        left = cur[k] if k < order - 1 else torch.zeros_like(w)
        right = cur[k - 1] if k >= 1 else torch.zeros_like(w)
        theta.append((x * left + (order - x) * right) / (order - 1))
        dtheta.append(left - right)
    return torch.stack(theta, dim=-1), torch.stack(dtheta, dim=-1)


def _bspline_knots_py(order: int) -> list[float]:
    """Pure-Python fp64 cardinal B-spline knot values M_p(1..p-1) (MPS torch.compile safe: no float64 tensors)."""
    cur = [0.0, 1.0]
    for n in range(3, order + 1):
        nxt = []
        for k in range(n):
            left = cur[k] if k < n - 1 else 0.0
            right = cur[k - 1] if k >= 1 else 0.0
            nxt.append((k * left + (n - k) * right) / (n - 1))
        cur = nxt
    return cur[1:]


def _bspline_moduli(m: int, order: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """|b(freq)|^-2 Essmann eq. 4.4 moduli for grid size m and spline order (cached by device/dtype)."""
    can_cache = (device.type != "meta") and (not capturing()) and (not proxy_tracing()) and (not torch.compiler.is_compiling())
    key = (int(m), int(order), dtype, norm_device(device)) if can_cache else None
    if key is not None and key in _MODULI_CACHE:
        return _MODULI_CACHE[key]
    mp = _bspline_knots_py(int(order))
    two_pi_over_m = 2.0 * math.pi / int(m)
    vals: list[float] = []
    for freq in range(int(m)):
        re = 0.0
        im = 0.0
        for k, mk in enumerate(mp):
            ang = two_pi_over_m * freq * k
            re += mk * math.cos(ang)
            im += mk * math.sin(ang)
        den = re * re + im * im
        vals.append((1.0 / den) if den > 1e-10 else 0.0)
    out = torch.tensor(vals, dtype=dtype, device=device)
    if key is not None and storable(out):
        if len(_MODULI_CACHE) >= 64:
            _MODULI_CACHE.clear()
        _MODULI_CACHE[key] = out
    return out


class ScreeningGate(nn.Module):
    """Per-atom screening gate g_i in (0, 1) disabling long-range mesh on bulk metals."""

    def __init__(self, d_model: int, hidden: int = 32, bias_crystal: float = -4.2,
                 bias_molecule: float = 3.0, skip_thresh: float = 0.02) -> None:
        super().__init__()
        self.skip_thresh = float(skip_thresh)
        self.bias_crystal = float(bias_crystal)
        self.bias_molecule = float(bias_molecule)
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def should_skip(self, g_mean: Tensor) -> bool:
        """Return True outside trace/compile/CUDAGraph when g_mean < skip_thresh."""
        if proxy_tracing() or torch.compiler.is_compiling() or is_fake(g_mean):
            return False
        return bool(float(g_mean.detach()) < self.skip_thresh)

    def forward(self, s_norm: Tensor, atom_mask: Tensor, pbc: bool = True) -> tuple[Tensor, Tensor]:
        bias = self.bias_crystal if pbc else self.bias_molecule
        mf = atom_mask.to(s_norm.dtype)
        g = torch.sigmoid(self.net(s_norm).squeeze(-1) + bias) * mf
        g_mean = g.sum() / mf.sum().clamp_min(1.0)
        return g, g_mean


class BatchedSPME:
    """Static-grid batched SPME backend sharing the ExactCoulomb-compatible multipole interface.

    Supports packed batches [G, 3, 3] of periodic cells with exact live d_cell virial stress,
    dipole spreading, analytical B-spline fields (no nested autograd.grad), k-space traceless EFG,
    and exact per-atom reciprocal Cauchy virial W_recip (sum_i W_recip_i == V * stress_recip).
    """

    def __init__(self, pos: Tensor, cell: Tensor, graph: Tensor, atom_mask: Tensor, graph_mask: Tensor,
                 sigma: float = SIGMA, grid_shape: tuple[int, int, int] = (16, 16, 16), order: int = 4,
                 active: bool = True, fixed_grid_shape: tuple[int, int, int] | None = None) -> None:
        with fp32_guard(pos.device, disable_tf32=True):
            self.pos = upcast_fp32(pos)
            self.cell = upcast_fp32(cell)
            self.graph = graph
            self.atom_mask = atom_mask.bool()
            self.graph_mask = graph_mask.bool()
            self.sigma = float(sigma)
            self.fixed_grid_shape = None if fixed_grid_shape is None else normalize_spme_grid(fixed_grid_shape)
            eff_grid = self.fixed_grid_shape if self.fixed_grid_shape is not None else grid_shape
            self.grid_shape = (int(eff_grid[0]), int(eff_grid[1]), int(eff_grid[2]))
            self.order = int(order)
            self.active = bool(active)
            self.n_graphs = int(cell.shape[0])
            self.n_atoms = int(pos.shape[0])

            dt, dev = self.pos.dtype, self.pos.device
            self.k0 = 2.0 / (self.sigma * math.sqrt(math.pi))
            self.f1_0 = 4.0 / (3.0 * math.sqrt(math.pi) * (self.sigma ** 3))
            if not self.active:
                self.valid_g = torch.zeros_like(self.graph_mask)
                self.active_atom = torch.zeros_like(self.atom_mask)
                self.af = self.active_atom.to(dt)
                return
            det = torch.linalg.det(self.cell)
            valid_g = self.graph_mask & (det.abs() > 1e-6)
            can_cache = (
                (dev.type != "meta") and (not self.pos.requires_grad) and (not self.cell.requires_grad)
                and (not capturing()) and (not proxy_tracing()) and (not torch.compiler.is_compiling())
                and (not is_fake(pos)) and (not is_fake(cell))
            )
            mkey = (
                norm_device(dev), dt, self.grid_shape, self.order, self.sigma,
                tuple(pos.shape), pos.data_ptr(), pos._version, tuple(cell.shape), cell.data_ptr(), cell._version,
                graph.data_ptr(), graph._version, atom_mask.data_ptr(), atom_mask._version, graph_mask.data_ptr(), graph_mask._version,
            ) if can_cache else None
            if mkey is not None and (cached := _SPME_MESH_CACHE.get(mkey)) is not None:
                if all(w() is not None for w in cached[0]):
                    (self._inv_cell64, self.inv_cell, self.vol, self.active_atom, self.af, self.valid_g,
                     self.flat, self.wt, self.grad_r_wt, self.g_eff, self.kvec, self.k2_safe) = cached[1]
                    self._mesh_cache_hit = True
                    return
                _SPME_MESH_CACHE.pop(mkey, None)
            self._mesh_cache_hit = False
            eye = torch.eye(3, dtype=dt, device=dev).unsqueeze(0).expand(self.n_graphs, -1, -1)
            safe_cell = torch.where(valid_g.view(-1, 1, 1), self.cell, eye)
            if dev.type != "mps":
                inv_64 = torch.linalg.inv(safe_cell.double())
                self._inv_cell64 = inv_64
                self.inv_cell = inv_64.to(dt)
            else:
                self._inv_cell64 = None
                self.inv_cell = torch.linalg.inv(safe_cell)
            self.vol = torch.linalg.det(safe_cell).abs().clamp_min(1e-6)
            self.active_atom = self.atom_mask & valid_g.index_select(0, self.graph)
            self.af = self.active_atom.to(dt)
            self.valid_g = valid_g

            self._prepare_mesh(dt, dev)
            if mkey is not None and storable(self.wt):
                if len(_SPME_MESH_CACHE) >= 8:
                    _SPME_MESH_CACHE.clear()
                wrefs = tuple(weakref.ref(x.untyped_storage()) for x in (pos, cell, graph, atom_mask, graph_mask))
                _SPME_MESH_CACHE[mkey] = (
                    wrefs,
                    (self._inv_cell64, self.inv_cell, self.vol, self.active_atom, self.af, self.valid_g,
                     self.flat, self.wt, self.grad_r_wt, self.g_eff, self.kvec, self.k2_safe),
                )

    def _prepare_mesh(self, dt: torch.dtype, dev: torch.device) -> None:
        m1, m2, m3 = self.grid_shape
        p = self.order
        m_vec = torch.tensor((m1, m2, m3), dtype=dt, device=dev)
        inv_atom = self.inv_cell.index_select(0, self.graph)
        # Wrap fractional coordinates modulo 1.0 in fp64 on CPU/CUDA (keeps full precision at +10k A shifts);
        # on Apple MPS (no float64), wrap in fp32 (calculator recenters positions around origin in fp64).
        if self._inv_cell64 is not None:
            frac = torch.einsum("ni,nij->nj", self.pos.double(), self._inv_cell64.index_select(0, self.graph))
            u = (frac - stop_grad(frac.floor())).to(dt) * m_vec
        else:
            frac = torch.einsum("ni,nij->nj", self.pos, inv_atom)
            u = (frac - stop_grad(frac.floor())) * m_vec
        fl = stop_grad(u.floor())
        w = u - fl
        theta, dtheta = _bspline_w_dw(w, p)  # [N, 3, p]

        offs = torch.arange(p, device=dev)
        m_long = torch.tensor((m1, m2, m3), dtype=torch.long, device=dev)
        idx = (fl.long().unsqueeze(-1) - offs) % m_long.view(1, 3, 1)
        local_flat = (
            idx[:, 0, :, None, None] * (m2 * m3)
            + idx[:, 1, None, :, None] * m3
            + idx[:, 2, None, None, :]
        ).reshape(self.n_atoms, p ** 3)
        self.flat = self.graph.unsqueeze(-1) * (m1 * m2 * m3) + local_flat

        t0, t1, t2 = theta[:, 0], theta[:, 1], theta[:, 2]
        dt0, dt1, dt2 = dtheta[:, 0], dtheta[:, 1], dtheta[:, 2]
        self.wt = (t0[:, :, None, None] * t1[:, None, :, None] * t2[:, None, None, :]).reshape(self.n_atoms, -1)
        dw0 = (dt0[:, :, None, None] * t1[:, None, :, None] * t2[:, None, None, :]).reshape(self.n_atoms, -1)
        dw1 = (t0[:, :, None, None] * dt1[:, None, :, None] * t2[:, None, None, :]).reshape(self.n_atoms, -1)
        dw2 = (t0[:, :, None, None] * t1[:, None, :, None] * dt2[:, None, None, :]).reshape(self.n_atoms, -1)
        grad_u = torch.stack((dw0, dw1, dw2), dim=-1) * m_vec  # [N, p^3, 3]
        self.grad_r_wt = torch.einsum("nsc,nac->nsa", grad_u, inv_atom)  # (grad_u * M) @ inv^T

        f1 = torch.fft.fftfreq(m1, d=1.0 / m1, device=dev).to(dt)
        f2 = torch.fft.fftfreq(m2, d=1.0 / m2, device=dev).to(dt)
        f3 = torch.fft.rfftfreq(m3, d=1.0 / m3, device=dev).to(dt)
        rec = self.inv_cell.transpose(-1, -2)  # [G, 3, 3]: rows = b_a / (2 pi)
        kvec = 2.0 * math.pi * (
            f1[None, :, None, None, None] * rec[:, 0, None, None, None, :]
            + f2[None, None, :, None, None] * rec[:, 1, None, None, None, :]
            + f3[None, None, None, :, None] * rec[:, 2, None, None, None, :]
        )  # [G, M1, M2, M3//2+1, 3]
        k2 = kvec.pow(2).sum(-1)
        nonzero = k2 > 1e-24
        k2_safe = torch.where(nonzero, k2, torch.ones_like(k2))
        b2 = (
            _bspline_moduli(m1, p, dt, dev)[:, None, None]
            * _bspline_moduli(m2, p, dt, dev)[None, :, None]
            * _bspline_moduli(m3, p, dt, dev)[None, None, : m3 // 2 + 1]
        )
        pref = (m1 * m2 * m3) / self.vol.view(-1, 1, 1, 1)
        g_eff = pref * (4.0 * math.pi * torch.exp(-0.25 * self.sigma * self.sigma * k2_safe) / k2_safe) * b2
        self.g_eff = torch.where(nonzero & self.valid_g.view(-1, 1, 1, 1), g_eff, torch.zeros_like(g_eff))
        self.kvec = kvec
        self.k2_safe = k2_safe
        self.k0 = 2.0 / (self.sigma * math.sqrt(math.pi))
        self.f1_0 = 4.0 / (3.0 * math.sqrt(math.pi) * (self.sigma ** 3))

    def _grid_q(self, q: Tensor, p: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
        m1, m2, m3 = self.grid_shape
        qm = upcast_fp32(q).to(self.pos.dtype) * self.af
        src = qm.unsqueeze(-1) * self.wt
        pm = (upcast_fp32(p).to(self.pos.dtype) * self.af.unsqueeze(-1)) if p is not None else None
        if pm is not None:
            src = src + torch.einsum("nc,nsc->ns", pm, self.grad_r_wt)
        q_grid = qm.new_zeros(self.n_graphs * m1 * m2 * m3)
        q_grid.index_add_(0, self.flat.reshape(-1), src.reshape(-1))
        return q_grid.view(self.n_graphs, m1, m2, m3), qm, pm

    def phi(self, q: Tensor) -> Tensor:
        return self.phi_field_multipole(q, None, need_efg=False)[0]

    def phi_field(self, q: Tensor) -> tuple[Tensor, Tensor]:
        phi, fld, _ = self.phi_field_multipole(q, None, need_efg=False)
        return phi, fld

    def phi_field_multipole(self, q: Tensor, p: Tensor | None = None, *,
                            need_efg: bool = False) -> tuple[Tensor, Tensor, Tensor | None]:
        """Evaluate pair (phi [N], field [N, 3], optional traceless EFG [N, 5]) on periodic mesh."""
        if not self.active:
            z = self.pos.new_zeros(self.n_atoms)
            return z, self.pos.new_zeros(self.n_atoms, 3), (self.pos.new_zeros(self.n_atoms, 5) if need_efg else None)
        with fp32_guard(self.pos.device, disable_tf32=True):
            q_grid, qm, pm = self._grid_q(q, p)
            g1 = self.g_eff.unsqueeze(1)
            if need_efg:
                kx, ky, kz = self.kvec[..., 0], self.kvec[..., 1], self.kvec[..., 2]
                r2, r6 = 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)
                k_l2 = torch.cat([t.unsqueeze(1) for t in (
                    math.sqrt(2.0) * kx * ky, math.sqrt(2.0) * kx * kz, math.sqrt(2.0) * ky * kz,
                    (kx * kx - ky * ky) * r2, (2.0 * kz * kz - kx * kx - ky * ky) * r6,
                )], dim=1)
                mult = torch.cat((g1, g1 * k_l2), dim=1)
            else:
                mult = g1
            grids = torch.ops.lam4.spme_fft_conv.default(q_grid, mult)
            phi_grid = grids[:, 0].reshape(-1)
            gathered = phi_grid.index_select(0, self.flat.reshape(-1)).view(self.n_atoms, -1)
            phi = ((gathered * self.wt).sum(-1) - self.k0 * qm) * self.af
            fld = -torch.bmm(gathered.unsqueeze(1), self.grad_r_wt).squeeze(1)
            if pm is not None:
                fld = fld + self.f1_0 * pm
            fld = fld * self.af.unsqueeze(-1)
            efg = None
            if need_efg:
                efg_flat = grids[:, 1:].permute(0, 2, 3, 4, 1).reshape(-1, 5)
                efg_g = efg_flat.index_select(0, self.flat.reshape(-1)).view(self.n_atoms, -1, 5)
                efg = torch.bmm(self.wt.unsqueeze(1), efg_g).squeeze(1) * self.af.unsqueeze(-1)
            return phi, fld, efg

    def virial_atom(self, q: Tensor, p: Tensor | None = None) -> Tensor:
        """Exact per-atom reciprocal Cauchy virial W_recip [N, 3, 3] summing to dU_mesh/d_eps = V * stress_recip."""
        if not self.active:
            return self.pos.new_zeros(self.n_atoms, 3, 3)
        with fp32_guard(self.pos.device, disable_tf32=True):
            q_grid, qm, pm = self._grid_q(q, p)
            kx, ky, kz = self.kvec[..., 0], self.kvec[..., 1], self.kvec[..., 2]
            fac = 2.0 * (1.0 / self.k2_safe + 0.25 * self.sigma * self.sigma)
            s_mult = self.g_eff.unsqueeze(1) * torch.cat([t.unsqueeze(1) for t in (
                -1.0 + fac * kx * kx, -1.0 + fac * ky * ky, -1.0 + fac * kz * kz,
                fac * kx * ky, fac * kx * kz, fac * ky * kz,
            )], dim=1)
            mult = torch.cat((s_mult, self.g_eff.unsqueeze(1)), dim=1) if pm is not None else s_mult
            grids = torch.ops.lam4.spme_fft_conv.default(q_grid, mult)
            s_flat = grids[:, :6].permute(0, 2, 3, 4, 1).reshape(-1, 6)
            src = qm.unsqueeze(-1) * self.wt
            if pm is not None:
                src = src + torch.einsum("nc,nsc->ns", pm, self.grad_r_wt)
            w6 = (0.5 * COULOMB_K) * (s_flat.index_select(0, self.flat.reshape(-1)).view(self.n_atoms, -1, 6) * src.unsqueeze(-1)).sum(1)
            wxx, wyy, wzz, wxy, wxz, wyz = w6[:, 0], w6[:, 1], w6[:, 2], w6[:, 3], w6[:, 4], w6[:, 5]
            w_mat = torch.cat((
                torch.cat((wxx.unsqueeze(-1), wxy.unsqueeze(-1), wxz.unsqueeze(-1)), dim=-1).unsqueeze(-2),
                torch.cat((wxy.unsqueeze(-1), wyy.unsqueeze(-1), wyz.unsqueeze(-1)), dim=-1).unsqueeze(-2),
                torch.cat((wxz.unsqueeze(-1), wyz.unsqueeze(-1), wzz.unsqueeze(-1)), dim=-1).unsqueeze(-2),
            ), dim=-2)
            if pm is not None:
                e_mesh = -(grids[:, 6].reshape(-1).index_select(0, self.flat.reshape(-1)).view(self.n_atoms, -1).unsqueeze(-1) * self.grad_r_wt).sum(1)
                pe = COULOMB_K * (pm.unsqueeze(-1) * e_mesh.unsqueeze(-2))
                w_mat = w_mat + 0.5 * (pe + pe.transpose(-1, -2))
            return w_mat * self.af.view(-1, 1, 1)
