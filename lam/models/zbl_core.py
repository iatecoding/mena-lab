"""zbl_core.py: Analytic Ziegler-Biersack-Littmark (ZBL) short-range repulsive core.

Universal screened Coulomb potential (Ziegler, Biersack & Littmark 1985, *The Stopping and Range of Ions in Matter*)
switched on smoothly below r_out = scale * (r_cov,i + r_cov,j) (default scale = 0.6), while the neural network sees
distances r_tilde smoothly C3-clamped below r_out so that the NN part has strictly zero radial derivative below r_in.

Units: Angstrom, kcal/mol (matching MenaACE internal units; not multiplied by output_scale).
Zero learnable parameters and zero state_dict keys (all tables are module-level non-persistent constants).
"""

from __future__ import annotations

import torch
from torch import Tensor

COULOMB_K = 332.06371  # kcal * A / (mol * e^2), matching MenaACE.coulomb_k
BOHR_A = 0.529177210903  # Bohr radius in Angstrom (CODATA 2018)
ZBL_A_PREFACTOR = 0.8854 * BOHR_A  # 0.4685335025335162 A
DEFAULT_R_OUT_SCALE = 0.6
DEFAULT_R_IN_RATIO = 0.9
MAX_HEAVY_ZBL_COV_RAD_A = 0.75  # caps Z >= 19 covalent radii so r_out <= 0.906 A on metal bonds (OMat24/MatPES/OC20 >= 1.15 A)


def _build_tables_f64() -> tuple[Tensor, Tensor, Tensor]:
    """Build float64 CPU lookup tables [128] for covalent radii (A, capped at MAX_HEAVY_ZBL_COV_RAD_A for Z >= 19), Z^0.23, and Z.
    Row 0 (padding) has r_cov = 0, Z = 0."""
    from ase.data import covalent_radii

    rad = torch.tensor(covalent_radii[:128], dtype=torch.float64)
    if rad.numel() < 128:
        rad = torch.cat([rad, rad.new_full((128 - rad.numel(),), MAX_HEAVY_ZBL_COV_RAD_A)])
    rad = torch.where(torch.isfinite(rad) & (rad > 0), rad, torch.full_like(rad, MAX_HEAVY_ZBL_COV_RAD_A))
    rad[19:] = rad[19:].clamp_max(MAX_HEAVY_ZBL_COV_RAD_A)
    rad[0] = 0.0

    z_flt = torch.arange(128, dtype=torch.float64)
    z_pow = z_flt.clamp_min(1.0).pow(0.23)
    return rad, z_pow, z_flt


_RAD_F64, _ZPOW_F64, _ZFLT_F64 = _build_tables_f64()
_TABLE_CACHE: dict[tuple[torch.device, torch.dtype], tuple[Tensor, Tensor, Tensor]] = {}


def _zbl_tables(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    """Return (cov_radii [128], z_pow_023 [128], z_float [128]) on (device, dtype), cached without state_dict keys."""
    key = (device, dtype)
    tabs = _TABLE_CACHE.get(key)
    if tabs is None:
        tabs = (
            _RAD_F64.to(device=device, dtype=dtype),
            _ZPOW_F64.to(device=device, dtype=dtype),
            _ZFLT_F64.to(device=device, dtype=dtype),
        )
        if not torch.compiler.is_compiling() and not isinstance(tabs[0], torch._subclasses.FakeTensor):
            _TABLE_CACHE[key] = tabs
    return tabs


class _ZBLClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r: Tensor, r_in: Tensor, r_out: Tensor) -> Tensor:
        ctx.save_for_backward(r, r_in, r_out)
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - r) / w).clamp(0.0, 1.0)
        u2 = u * u
        u4 = u2 * u2
        ig = u - u4 * (2.5 + u * (-3.0 + u))
        clamped = r_out - w * ig
        return torch.where((r >= r_out) | (r_out <= 0.0), r, clamped)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        r, r_in, r_out = ctx.saved_tensors
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - r) / w).clamp(0.0, 1.0)
        u2 = u * u
        dr = torch.where(
            (r >= r_out) | (r_out <= 0.0),
            torch.ones_like(r),
            torch.where(r <= r_in, torch.zeros_like(r), 1.0 - u2 * u * (10.0 + u * (-15.0 + 6.0 * u))),
        )
        return grad_out * dr, None, None


class _ZBLSwitch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r: Tensor, r_in: Tensor, r_out: Tensor) -> Tensor:
        ctx.save_for_backward(r, r_in, r_out)
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - r) / w).clamp(0.0, 1.0)
        u2 = u * u
        u4 = u2 * u2
        s = u4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u)))
        return torch.where(r_out > 0.0, s, torch.zeros_like(s))

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        r, r_in, r_out = ctx.saved_tensors
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - r) / w).clamp(0.0, 1.0)
        om = 1.0 - u
        ds_dr = torch.where(
            (r >= r_out) | (r <= r_in) | (r_out <= 0.0),
            torch.zeros_like(r),
            (-140.0 / w) * (u * u * u) * (om * om * om),
        )
        return grad_out * ds_dr, None, None


def zbl_clamp(r: Tensor, r_in: Tensor, r_out: Tensor) -> Tensor:
    """Smooth C3 distance clamp r_tilde(r):
      - For r >= r_out: r_tilde(r) = r  (d1 = 1, d2 = d3 = 0 at r = r_out).
      - For r_in < r < r_out: smooth C3 transition with d(r_tilde)/dr = 1 - u^3 (10 - 15 u + 6 u^2), u = (r_out - r)/(r_out - r_in).
      - For r <= r_in: r_tilde(r) = (r_in + r_out) / 2 = const  (d1 = d2 = d3 = 0 for all r <= r_in).
    When r_out <= 0 (padding atoms z = 0), returns r unchanged."""
    return _ZBLClamp.apply(r, r_in, r_out)


def zbl_switch(r: Tensor, r_in: Tensor, r_out: Tensor) -> Tensor:
    """Septic C3 switch s(r) in [0, 1]: 0 for r >= r_out, 1 for r <= r_in, with 3 vanishing derivatives at both ends."""
    return _ZBLSwitch.apply(r, r_in, r_out)


def zbl_pair_energy(zi_flt: Tensor, zj_flt: Tensor, zi_p23: Tensor, zj_p23: Tensor, r_safe: Tensor) -> Tensor:
    """Analytic ZBL screened Coulomb energy E_ZBL(r) in kcal/mol for pair (Z_i, Z_j) at distance r_safe > 0 (Angstrom)."""
    a_ij = ZBL_A_PREFACTOR / (zi_p23 + zj_p23)
    x = r_safe / a_ij
    phi = (
        0.1818 * torch.exp(-3.2 * x)
        + 0.5099 * torch.exp(-0.9423 * x)
        + 0.2802 * torch.exp(-0.4029 * x)
        + 0.02817 * torch.exp(-0.2016 * x)
    )
    return COULOMB_K * (zi_flt * zj_flt) * phi / r_safe


def _zbl_pair_energy_and_grad(
    zi_flt: Tensor, zj_flt: Tensor, zi_p23: Tensor, zj_p23: Tensor, r_safe: Tensor
) -> tuple[Tensor, Tensor]:
    a_ij = ZBL_A_PREFACTOR / (zi_p23 + zj_p23)
    x = r_safe / a_ij
    e1 = torch.exp(-3.2 * x)
    e2 = torch.exp(-0.9423 * x)
    e3 = torch.exp(-0.4029 * x)
    e4 = torch.exp(-0.2016 * x)
    phi = 0.1818 * e1 + 0.5099 * e2 + 0.2802 * e3 + 0.02817 * e4
    dphi_dx = -(0.1818 * 3.2) * e1 - (0.5099 * 0.9423) * e2 - (0.2802 * 0.4029) * e3 - (0.02817 * 0.2016) * e4
    pref = COULOMB_K * (zi_flt * zj_flt) / r_safe
    e_zbl = pref * phi
    de_zbl_dr = pref * (dphi_dx / a_ij - phi / r_safe)
    return e_zbl, de_zbl_dr


class _ZBLClampDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dist: Tensor, zc: Tensor, pmf: Tensor, scale: float, r_in_ratio: float) -> Tensor:
        ctx.save_for_backward(dist, zc, pmf)
        ctx.scale = float(scale)
        ctx.r_in_ratio = float(r_in_ratio)
        rad, _, _ = _zbl_tables(zc.device, dist.dtype)
        rad_z = rad[zc]
        valid = (zc > 0).unsqueeze(2) & (zc > 0).unsqueeze(1) & (pmf > 0.0)
        r_out = ctx.scale * (rad_z.unsqueeze(2) + rad_z.unsqueeze(1)) * valid.to(dist.dtype)
        r_in = ctx.r_in_ratio * r_out
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - dist) / w).clamp(0.0, 1.0)
        u2 = u * u
        u4 = u2 * u2
        ig = u - u4 * (2.5 + u * (-3.0 + u))
        clamped = r_out - w * ig
        return torch.where((dist >= r_out) | (r_out <= 0.0), dist, clamped) * pmf

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        dist, zc, pmf = ctx.saved_tensors
        rad, _, _ = _zbl_tables(zc.device, dist.dtype)
        rad_z = rad[zc]
        valid = (zc > 0).unsqueeze(2) & (zc > 0).unsqueeze(1) & (pmf > 0.0)
        r_out = ctx.scale * (rad_z.unsqueeze(2) + rad_z.unsqueeze(1)) * valid.to(dist.dtype)
        r_in = ctx.r_in_ratio * r_out
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - dist) / w).clamp(0.0, 1.0)
        u2 = u * u
        dr = torch.where(
            (dist >= r_out) | (r_out <= 0.0),
            torch.ones_like(dist),
            torch.where(dist <= r_in, torch.zeros_like(dist), 1.0 - u2 * u * (10.0 + u * (-15.0 + 6.0 * u))),
        )
        return grad_out * dr * pmf, None, None, None, None


class _ZBLClampSparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dist: Tensor, zc: Tensor, recv: Tensor, send: Tensor, scale: float, r_in_ratio: float) -> Tensor:
        ctx.save_for_backward(dist, zc, recv, send)
        ctx.scale = float(scale)
        ctx.r_in_ratio = float(r_in_ratio)
        rad, _, _ = _zbl_tables(zc.device, dist.dtype)
        zr, zs = zc[recv], zc[send]
        valid = (zr > 0) & (zs > 0)
        r_out = ctx.scale * (rad[zr] + rad[zs]) * valid.to(dist.dtype)
        r_in = ctx.r_in_ratio * r_out
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - dist) / w).clamp(0.0, 1.0)
        u2 = u * u
        u4 = u2 * u2
        ig = u - u4 * (2.5 + u * (-3.0 + u))
        clamped = r_out - w * ig
        return torch.where((dist >= r_out) | (r_out <= 0.0), dist, clamped)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        dist, zc, recv, send = ctx.saved_tensors
        rad, _, _ = _zbl_tables(zc.device, dist.dtype)
        zr, zs = zc[recv], zc[send]
        valid = (zr > 0) & (zs > 0)
        r_out = ctx.scale * (rad[zr] + rad[zs]) * valid.to(dist.dtype)
        r_in = ctx.r_in_ratio * r_out
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - dist) / w).clamp(0.0, 1.0)
        u2 = u * u
        dr = torch.where(
            (dist >= r_out) | (r_out <= 0.0),
            torch.ones_like(dist),
            torch.where(dist <= r_in, torch.zeros_like(dist), 1.0 - u2 * u * (10.0 + u * (-15.0 + 6.0 * u))),
        )
        return grad_out * dr, None, None, None, None, None


class _ZBLEnergyDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dist_raw: Tensor, zc: Tensor, pmf: Tensor, scale: float, r_in_ratio: float) -> Tensor:
        ctx.save_for_backward(dist_raw, zc, pmf)
        ctx.scale = float(scale)
        ctx.r_in_ratio = float(r_in_ratio)
        rad, z_pow, z_flt = _zbl_tables(zc.device, dist_raw.dtype)
        rad_z = rad[zc]
        valid = (zc > 0).unsqueeze(2) & (zc > 0).unsqueeze(1) & (pmf > 0.0)
        valid_f = valid.to(dist_raw.dtype)
        r_out = ctx.scale * (rad_z.unsqueeze(2) + rad_z.unsqueeze(1)) * valid_f
        r_in = ctx.r_in_ratio * r_out
        r_eval = torch.where(valid, dist_raw.clamp_min(1e-4), torch.ones_like(dist_raw))
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - r_eval) / w).clamp(0.0, 1.0)
        u2 = u * u
        u4 = u2 * u2
        sw = torch.where(r_out > 0.0, u4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u))), torch.zeros_like(u)) * valid_f
        zi_f = z_flt[zc].unsqueeze(2)
        zj_f = z_flt[zc].unsqueeze(1)
        zi_p = z_pow[zc].unsqueeze(2)
        zj_p = z_pow[zc].unsqueeze(1)
        e_pair = sw * zbl_pair_energy(zi_f, zj_f, zi_p, zj_p, r_eval) * pmf
        return 0.5 * e_pair.sum(dim=(-1, -2))

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        dist_raw, zc, pmf = ctx.saved_tensors
        rad, z_pow, z_flt = _zbl_tables(zc.device, dist_raw.dtype)
        rad_z = rad[zc]
        valid = (zc > 0).unsqueeze(2) & (zc > 0).unsqueeze(1) & (pmf > 0.0)
        valid_f = valid.to(dist_raw.dtype)
        r_out = ctx.scale * (rad_z.unsqueeze(2) + rad_z.unsqueeze(1)) * valid_f
        r_in = ctx.r_in_ratio * r_out
        r_eval = torch.where(valid, dist_raw.clamp_min(1e-4), torch.ones_like(dist_raw))
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - r_eval) / w).clamp(0.0, 1.0)
        u2 = u * u
        u4 = u2 * u2
        sw = torch.where(r_out > 0.0, u4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u))), torch.zeros_like(u)) * valid_f
        om = 1.0 - u
        dsw_dr = torch.where(
            (r_eval >= r_out) | (r_eval <= r_in) | (r_out <= 0.0),
            torch.zeros_like(r_eval),
            (-140.0 / w) * (u2 * u) * (om * om * om),
        ) * valid_f
        zi_f = z_flt[zc].unsqueeze(2)
        zj_f = z_flt[zc].unsqueeze(1)
        zi_p = z_pow[zc].unsqueeze(2)
        zj_p = z_pow[zc].unsqueeze(1)
        e_zbl, de_zbl_dr = _zbl_pair_energy_and_grad(zi_f, zj_f, zi_p, zj_p, r_eval)
        de_dr = 0.5 * (dsw_dr * e_zbl + sw * de_zbl_dr) * pmf * (dist_raw >= 1e-4).to(dist_raw.dtype)
        return grad_out.view(-1, 1, 1) * de_dr, None, None, None, None


class _ZBLEnergySparse(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dist_raw: Tensor,
        zc: Tensor,
        recv: Tensor,
        send: Tensor,
        batch: Tensor,
        n_graphs: int,
        scale: float,
        r_in_ratio: float,
    ) -> Tensor:
        ctx.save_for_backward(dist_raw, zc, recv, send, batch)
        ctx.scale = float(scale)
        ctx.r_in_ratio = float(r_in_ratio)
        rad, z_pow, z_flt = _zbl_tables(zc.device, dist_raw.dtype)
        zr, zs = zc[recv], zc[send]
        valid = (zr > 0) & (zs > 0)
        valid_f = valid.to(dist_raw.dtype)
        r_out = ctx.scale * (rad[zr] + rad[zs]) * valid_f
        r_in = ctx.r_in_ratio * r_out
        r_eval = torch.where(valid, dist_raw.clamp_min(1e-4), torch.ones_like(dist_raw))
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - r_eval) / w).clamp(0.0, 1.0)
        u2 = u * u
        u4 = u2 * u2
        sw = torch.where(r_out > 0.0, u4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u))), torch.zeros_like(u)) * valid_f
        e_edge = 0.5 * sw * zbl_pair_energy(z_flt[zr], z_flt[zs], z_pow[zr], z_pow[zs], r_eval)
        return dist_raw.new_zeros(int(n_graphs)).index_add_(0, batch[recv], e_edge)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        dist_raw, zc, recv, send, batch = ctx.saved_tensors
        rad, z_pow, z_flt = _zbl_tables(zc.device, dist_raw.dtype)
        zr, zs = zc[recv], zc[send]
        valid = (zr > 0) & (zs > 0)
        valid_f = valid.to(dist_raw.dtype)
        r_out = ctx.scale * (rad[zr] + rad[zs]) * valid_f
        r_in = ctx.r_in_ratio * r_out
        r_eval = torch.where(valid, dist_raw.clamp_min(1e-4), torch.ones_like(dist_raw))
        w = (r_out - r_in).clamp_min(1e-12)
        u = ((r_out - r_eval) / w).clamp(0.0, 1.0)
        u2 = u * u
        u4 = u2 * u2
        sw = torch.where(r_out > 0.0, u4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u))), torch.zeros_like(u)) * valid_f
        om = 1.0 - u
        dsw_dr = torch.where(
            (r_eval >= r_out) | (r_eval <= r_in) | (r_out <= 0.0),
            torch.zeros_like(r_eval),
            (-140.0 / w) * (u2 * u) * (om * om * om),
        ) * valid_f
        e_zbl, de_zbl_dr = _zbl_pair_energy_and_grad(z_flt[zr], z_flt[zs], z_pow[zr], z_pow[zs], r_eval)
        de_dr = 0.5 * (dsw_dr * e_zbl + sw * de_zbl_dr) * (dist_raw >= 1e-4).to(dist_raw.dtype)
        return grad_out[batch[recv]] * de_dr, None, None, None, None, None, None, None


def zbl_clamp_dense(
    z: Tensor,
    dist: Tensor,
    pmf: Tensor,
    scale: float = DEFAULT_R_OUT_SCALE,
    r_in_ratio: float = DEFAULT_R_IN_RATIO,
) -> Tensor:
    """Dense [B, N, N] smooth C3 distance clamp applied in MenaACE.forward geometry featurisation."""
    return _ZBLClampDense.apply(dist, z.clamp(0, 127), pmf, scale, r_in_ratio)


def zbl_clamp_sparse(
    z: Tensor,
    dist: Tensor,
    recv: Tensor,
    send: Tensor,
    scale: float = DEFAULT_R_OUT_SCALE,
    r_in_ratio: float = DEFAULT_R_IN_RATIO,
) -> Tensor:
    """Sparse [E] smooth C3 distance clamp applied in MenaACE.forward_sparse geometry featurisation."""
    return _ZBLClampSparse.apply(dist, z.clamp(0, 127), recv, send, scale, r_in_ratio)


def zbl_energy_dense(
    z: Tensor,
    dist_raw: Tensor,
    pmf: Tensor,
    scale: float = DEFAULT_R_OUT_SCALE,
    r_in_ratio: float = DEFAULT_R_IN_RATIO,
) -> Tensor:
    """Total ZBL short-range energy [B] in kcal/mol for dense batches: 0.5 * sum_{i != j} s(r_ij) E_ZBL(r_ij)."""
    return _ZBLEnergyDense.apply(dist_raw, z.clamp(0, 127), pmf, scale, r_in_ratio)


def zbl_energy_sparse(
    z: Tensor,
    dist_raw: Tensor,
    recv: Tensor,
    send: Tensor,
    batch: Tensor,
    n_graphs: int,
    scale: float = DEFAULT_R_OUT_SCALE,
    r_in_ratio: float = DEFAULT_R_IN_RATIO,
) -> Tensor:
    """Total ZBL short-range energy [B] in kcal/mol for sparse edge lists: 0.5 * sum_e s(r_e) E_ZBL(r_e)."""
    return _ZBLEnergySparse.apply(dist_raw, z.clamp(0, 127), recv, send, batch, int(n_graphs), scale, r_in_ratio)
