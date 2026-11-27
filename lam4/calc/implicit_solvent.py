"""Vectorized NumPy & PyTorch Implicit Solvent (Still-HCT, OBC1/OBC2, Fibonacci Shrake-Rupley SASA)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, TypeAlias
import warnings

import numpy as np
import torch
from torch import Tensor

GBMethod: TypeAlias = Literal["still_legacy", "still_std", "obc2", "obc1"]
RadiiSet: TypeAlias = Literal["bondi", "mbondi2", "bondi_hn130"]
ArrayLike: TypeAlias = np.ndarray | Tensor

K_COUL = 332.06371  # kcal * A / (mol * e^2)
_DEFAULT_RADIUS, _DEFAULT_HCT_SCALE, _MAX_Z_LUT = 1.70, 0.80, 119

# Bondi (1964) / Mantina et al. (JPCA 2009) / Alvarez (Dalton Trans. 2013) vdW radii (A)
BONDI_RADII: dict[int, float] = {
    1: 1.20, 3: 1.82, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    11: 2.27, 12: 1.73, 13: 1.84, 15: 1.80, 16: 1.80, 17: 1.75,
    19: 2.75, 20: 2.31, 22: 2.15, 23: 2.07, 24: 2.06, 25: 2.05, 26: 2.04,
    27: 2.00, 28: 1.63, 29: 1.40, 30: 1.39, 35: 1.85, 37: 3.03, 38: 2.49,
    42: 2.17, 46: 1.63, 47: 1.72, 48: 1.58, 53: 1.98, 55: 3.43, 56: 2.68,
    78: 1.75, 79: 1.66, 80: 1.55,
}
# AMBER mbondi2 base radii (H bonded to N gets 1.30 A dynamically) + Born-calibrated hydrated metal ion radii (A)
MBONDI2_BASE: dict[int, float] = {
    1: 1.20, 3: 1.40, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.50, 9: 1.50,
    11: 1.76, 12: 1.50, 13: 1.35, 15: 1.85, 16: 1.80, 17: 1.70,
    19: 2.14, 20: 1.82, 22: 1.55, 23: 1.52, 24: 1.50, 25: 1.52, 26: 1.50,
    27: 1.46, 28: 1.44, 29: 1.40, 30: 1.39, 35: 1.85, 37: 2.28, 38: 2.08,
    42: 1.65, 46: 1.63, 47: 1.72, 48: 1.58, 53: 1.98, 55: 2.50, 56: 2.28,
    78: 1.75, 79: 1.66, 80: 1.55,
}
# AMBER / OpenMM HCT screening scale factors (0.85 standard for metal ions)
HCT_SCALE: dict[int, float] = {
    1: 0.85, 3: 0.85, 5: 0.80, 6: 0.72, 7: 0.79, 8: 0.85, 9: 0.88,
    11: 0.85, 12: 0.85, 13: 0.85, 15: 0.86, 16: 0.96, 17: 0.80,
    19: 0.85, 20: 0.85, 22: 0.85, 23: 0.85, 24: 0.85, 25: 0.85, 26: 0.85,
    27: 0.85, 28: 0.85, 29: 0.85, 30: 0.85, 35: 0.80, 37: 0.85, 38: 0.85,
    42: 0.85, 46: 0.85, 47: 0.85, 48: 0.85, 53: 0.80, 55: 0.85, 56: 0.85,
    78: 0.85, 79: 0.85, 80: 0.85,
}
OBC2_PARAMS: tuple[float, float, float] = (1.0, 0.8, 4.85)
OBC1_PARAMS: tuple[float, float, float] = (0.8, 0.0, 2.909125)


def _build_lut(table: Mapping[int, float], default: float) -> np.ndarray:
    lut = np.full(_MAX_Z_LUT, float(default), dtype=np.float64)
    for zi, val in table.items():
        if 0 <= int(zi) < _MAX_Z_LUT:
            lut[int(zi)] = float(val)
    return lut


def _build_known_lut(table: Mapping[int, float]) -> np.ndarray:
    known = np.zeros(_MAX_Z_LUT, dtype=bool)
    for zi in table:
        if 0 <= int(zi) < _MAX_Z_LUT:
            known[int(zi)] = True
    return known


_BONDI_LUT = _build_lut(BONDI_RADII, _DEFAULT_RADIUS)
_MBONDI2_LUT = _build_lut(MBONDI2_BASE, _DEFAULT_RADIUS)
_HCT_LUT = _build_lut(HCT_SCALE, _DEFAULT_HCT_SCALE)
_BONDI_KNOWN_LUT = _build_known_lut(BONDI_RADII)
_MBONDI2_KNOWN_LUT = _build_known_lut(MBONDI2_BASE)
_WARNED_FALLBACK_Z: set[int] = set()


def _warn_if_unknown_radii(z_arr: Any, known_lut: np.ndarray, radii_set: str) -> None:
    """Emit a deduplicated warning once per unknown atomic number falling back to `_DEFAULT_RADIUS`."""
    if torch.is_tensor(z_arr):
        if getattr(torch, "compiler", None) is not None and torch.compiler.is_compiling():
            return
        z_np = z_arr.detach().cpu().numpy()
    else:
        z_np = np.asarray(z_arr, dtype=np.int64)
    if z_np.size == 0:
        return
    in_b = (z_np >= 0) & (z_np < len(known_lut))
    safe = np.where(in_b, z_np, 0)
    unk_mask = (~in_b) | (~known_lut[safe])
    if not bool(np.any(unk_mask)):
        return
    unseen = sorted({int(zi) for zi in np.unique(z_np[unk_mask]) if int(zi) not in _WARNED_FALLBACK_Z})
    if unseen:
        _WARNED_FALLBACK_Z.update(unseen)
        warnings.warn(
            f"implicit_solvent ({radii_set}): no calibrated Born radius for Z={unseen}; "
            f"falling back to default R={_DEFAULT_RADIUS:.2f} A (HCT scale={_DEFAULT_HCT_SCALE:.2f}).",
            UserWarning,
            stacklevel=3,
        )


def _backend(*objs: Any) -> tuple[bool, torch.device | None, Any]:
    is_t = any(torch.is_tensor(x) for x in objs)
    if not is_t:
        return False, None, np.float64
    dev = next(x.device for x in objs if torch.is_tensor(x))
    dt = next((x.dtype for x in objs if torch.is_tensor(x) and x.is_floating_point()), torch.float64)
    return True, dev, dt


def _to_float(x: Any, is_t: bool, dev: torch.device | None, dt: Any) -> Any:
    if not is_t:
        return np.asarray(x, dtype=np.float64)
    return x.to(device=dev, dtype=dt) if torch.is_tensor(x) else torch.as_tensor(x, dtype=dt, device=dev)


def _to_long(x: Any, is_t: bool, dev: torch.device | None) -> Any:
    if not is_t:
        return np.asarray(x, dtype=np.int64)
    return x.to(device=dev, dtype=torch.long) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.long, device=dev)


def _where(cond: Any, a: Any, b: Any) -> Any:
    return torch.where(cond, a, b) if torch.is_tensor(cond) else np.where(cond, a, b)


def _clamp_min(x: Any, val: float) -> Any:
    return x.clamp_min(val) if torch.is_tensor(x) else np.maximum(x, val)


def _eye(n: int, ref: Any) -> Any:
    return torch.eye(n, dtype=torch.bool, device=ref.device) if torch.is_tensor(ref) else np.eye(n, dtype=bool)


def _lookup_lut(lut_np: np.ndarray, z_arr: Any, default: float, dtype: Any) -> Any:
    in_bounds = (z_arr >= 0) & (z_arr < len(lut_np))
    if torch.is_tensor(z_arr):
        lut = torch.as_tensor(lut_np, dtype=dtype, device=z_arr.device)
        safe_z = torch.where(in_bounds, z_arr, torch.zeros_like(z_arr))
        return torch.where(in_bounds, lut[safe_z], torch.full_like(safe_z, float(default), dtype=dtype))
    safe_z = np.where(in_bounds, z_arr, 0)
    return np.where(in_bounds, lut_np[safe_z], float(default))


def _pairwise_distances(pos: Any) -> Any:
    n = pos.shape[0]
    if n == 0:
        return pos.new_zeros((0, 0)) if torch.is_tensor(pos) else np.zeros((0, 0), dtype=np.float64)
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = (diff * diff).sum(-1)
    eye = _eye(n, pos)
    # Keep diagonal positive before sqrt so PyTorch autograd never sees d(sqrt(0))/dx = NaN.
    sqrt_fn = torch.sqrt if torch.is_tensor(pos) else np.sqrt
    return _where(eye, 0.0, sqrt_fn(_where(eye, 1.0, r2)))


@lru_cache(maxsize=16)
def fibonacci_sphere(n_pts: int = 194) -> np.ndarray:
    """Deterministic golden-spiral points on the unit sphere."""
    if n_pts <= 0:
        raise ValueError(f"n_pts must be positive, got {n_pts}")
    idx = np.arange(n_pts, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n_pts)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * idx
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)


def get_radii(
    z: ArrayLike | Sequence[int], pos: ArrayLike, radii_set: RadiiSet = "mbondi2", radius_scale: float = 1.0
) -> np.ndarray | Tensor:
    """Assign atomic GB/SASA radii (`bondi`, `mbondi2`, or `bondi_hn130` where H bonded to N = 1.30 A)."""
    if radii_set not in ("bondi", "mbondi2", "bondi_hn130"):
        raise ValueError(f"Unknown radii_set {radii_set!r}; expected 'bondi', 'mbondi2', or 'bondi_hn130'")
    base_lut = _BONDI_LUT if radii_set in ("bondi", "bondi_hn130") else _MBONDI2_LUT
    known_lut = _BONDI_KNOWN_LUT if radii_set in ("bondi", "bondi_hn130") else _MBONDI2_KNOWN_LUT
    use_hn130 = radii_set in ("mbondi2", "bondi_hn130")
    is_t, dev, dt = _backend(z, pos)
    z_arr, pos_arr = _to_long(z, is_t, dev), _to_float(pos, is_t, dev, dt)
    _warn_if_unknown_radii(z_arr, known_lut, radii_set)
    radii = _lookup_lut(base_lut, z_arr, _DEFAULT_RADIUS, dt)

    if use_hn130 and pos_arr.shape[0] > 0:
        h_idx = torch.nonzero(z_arr == 1, as_tuple=True)[0] if is_t else np.where(z_arr == 1)[0]
        n_idx = torch.nonzero(z_arr == 7, as_tuple=True)[0] if is_t else np.where(z_arr == 7)[0]
        if not is_t:
            radii = radii.copy()
        if (h_idx.numel() > 0 and n_idx.numel() > 0) if is_t else (len(h_idx) > 0 and len(n_idx) > 0):
            d2 = ((pos_arr[h_idx][:, None, :] - pos_arr[n_idx][None, :, :]) ** 2).sum(-1)
            hn_mask = (d2.min(dim=1).values if is_t else np.min(d2, axis=1)) < (1.35 ** 2)
            if (hn_mask.any() if is_t else np.any(hn_mask)):
                if is_t:
                    radii = radii.clone()
                radii[h_idx[hn_mask]] = 1.30
    return radii * float(radius_scale)


def _hct_term(r: Any, lower: Any, upper: Any, sr: Any) -> Any:
    log_fn = torch.log if torch.is_tensor(r) else np.log
    return 0.5 * (
        (1.0 / lower - 1.0 / upper)
        + 0.25 * (sr ** 2 / r - r) * (1.0 / (lower ** 2) - 1.0 / (upper ** 2))
        + (0.5 / r) * log_fn(lower / upper)
    )


def _hct_integral_sum(r: Any, rho0: Any, sr_vec: Any, *, legacy_half_case4: bool = False) -> Any:
    n = rho0.shape[0]
    if n == 0:
        return rho0.new_zeros((0,)) if torch.is_tensor(rho0) else np.zeros((0,), dtype=np.float64)
    eye = _eye(n, r)
    r_safe = _where(eye, 1e6, _clamp_min(r, 1e-4))
    rho_i, sr_j = rho0[:, None], sr_vec[None, :]
    abs_fn = torch.abs if torch.is_tensor(r) else np.abs
    m1 = (~eye) & (r_safe > rho_i + sr_j)
    m2 = (~eye) & (r_safe <= rho_i + sr_j) & (r_safe > abs_fn(rho_i - sr_j))
    m4 = (~eye) & (r_safe + rho_i <= sr_j)

    # Clamp denominators/log args on masked-out entries for NaN-free autograd & zero-warning NumPy.
    l1, u_all = _clamp_min(r_safe - sr_j, 1e-6), _clamp_min(r_safe + sr_j, 1e-6)
    ri_safe, sr_mr = _clamp_min(rho_i, 1e-6), _clamp_min(sr_j - r_safe, 1e-4)
    val1 = _hct_term(r_safe, l1, u_all, sr_j)
    val2 = _hct_term(r_safe, ri_safe, u_all, sr_j)
    val4 = (0.5 if legacy_half_case4 else 1.0) * (1.0 / ri_safe - 1.0 / sr_mr) + _hct_term(r_safe, sr_mr, u_all, sr_j)
    integral = _where(m1, val1, _where(m2, val2, _where(m4, val4, 0.0)))
    return integral.sum(dim=1) if torch.is_tensor(integral) else np.sum(integral, axis=1)


def compute_born_radii(
    z: ArrayLike | Sequence[int], pos: ArrayLike, *, method: GBMethod = "obc2", radii_set: RadiiSet = "mbondi2",
    radius_scale: float = 1.0, dielectric_offset: float = 0.09, sr_use_rho0: bool | None = None,
    r_mat: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[Tensor, Tensor, Tensor]:
    """Compute effective Born radii `alpha`, intrinsic `radii`, and pairwise `r_mat` (NumPy or PyTorch)."""
    if method not in ("still_legacy", "still_std", "obc2", "obc1"):
        raise ValueError(f"Unknown GB method: {method!r}")
    use_rho0 = (method == "still_legacy") if sr_use_rho0 is None else bool(sr_use_rho0)
    is_t, dev, dt = _backend(z, pos, r_mat)
    z_arr, pos_arr = _to_long(z, is_t, dev), _to_float(pos, is_t, dev, dt)
    radii = get_radii(z_arr, pos_arr, radii_set=radii_set, radius_scale=radius_scale)
    rho0 = _clamp_min(radii - float(dielectric_offset), 0.5)
    s_arr = _lookup_lut(_HCT_LUT, z_arr, _DEFAULT_HCT_SCALE, dt)
    r_arr = _to_float(r_mat, is_t, dev, dt) if r_mat is not None else _pairwise_distances(pos_arr)

    i_sum = _hct_integral_sum(r_arr, rho0, s_arr * (rho0 if use_rho0 else radii), legacy_half_case4=(method == "still_legacy"))
    if method in ("still_legacy", "still_std"):
        inv_alpha = _clamp_min(1.0 / rho0 - i_sum, 1.0 / 30.0)
    else:
        a_obc, b_obc, g_obc = OBC2_PARAMS if method == "obc2" else OBC1_PARAMS
        psi = rho0 * i_sum
        arg = a_obc * psi - b_obc * (psi ** 2) + g_obc * (psi ** 3)
        tanh_val = torch.tanh(arg.clamp(-20.0, 20.0)) if is_t else np.tanh(np.clip(arg, -20.0, 20.0))
        inv_alpha = _clamp_min(1.0 / rho0 - tanh_val / radii, 1.0 / 50.0)
    return 1.0 / inv_alpha, radii, r_arr


def _build_sasa_nbrs(nbr_mask: Any, max_deg: int) -> tuple[Any, Any]:
    if torch.is_tensor(nbr_mask):
        nbr_valid_i8, nbr_idx = torch.topk(nbr_mask.to(torch.int8), k=max_deg, dim=1, sorted=False)
        return nbr_idx, nbr_valid_i8.bool()
    n = nbr_mask.shape[0]
    nbr_idx = np.zeros((n, max_deg), dtype=np.int64)
    nbr_valid = np.zeros((n, max_deg), dtype=bool)
    for i in range(n):
        js = np.where(nbr_mask[i])[0]
        if len(js) > 0:
            nbr_idx[i, : len(js)] = js
            nbr_valid[i, : len(js)] = True
    return nbr_idx, nbr_valid


def compute_sasa(
    pos: ArrayLike, radii: ArrayLike, *, probe_radius: float = 1.4, n_pts: int = 194,
    r_mat: ArrayLike | None = None, use_mps: bool = True, return_per_atom: bool = False,
) -> float | Tensor | np.ndarray:
    """Vectorized Fibonacci Shrake-Rupley SASA (A^2); step-function wrt coordinates."""
    sphere = fibonacci_sphere(n_pts)
    is_t, dev, dt = _backend(pos, radii, r_mat)
    pos_arr, rad_arr = _to_float(pos, is_t, dev, dt), _to_float(radii, is_t, dev, dt)
    n = int(pos_arr.shape[0])
    if n == 0:
        if return_per_atom:
            return pos_arr.new_zeros((0,)) if is_t else np.zeros(0, dtype=np.float64)
        return pos_arr.new_zeros(()) if is_t else 0.0

    r_p = rad_arr + float(probe_radius)
    r_arr = _to_float(r_mat, is_t, dev, dt) if r_mat is not None else _pairwise_distances(pos_arr)
    nbr_mask = (~_eye(n, pos_arr)) & (r_arr < (r_p[:, None] + r_p[None, :]))
    max_deg = int((nbr_mask.sum(dim=1).max().item()) if is_t else np.max(np.sum(nbr_mask, axis=1)))
    if max_deg == 0:
        area_atom = 4.0 * math.pi * (r_p ** 2)
        if return_per_atom:
            return area_atom if is_t else np.asarray(area_atom, dtype=np.float64)
        return area_atom.sum() if is_t else float(np.sum(area_atom))

    nbr_idx, nbr_valid = _build_sasa_nbrs(nbr_mask, max_deg)
    d_ij = pos_arr[:, None, :] - pos_arr[nbr_idx]
    r2_ij = (r_arr.gather(1, nbr_idx) if is_t else np.take_along_axis(r_arr, nbr_idx, axis=1)) ** 2
    rp_i, rp_j = r_p[:, None], r_p[nbr_idx]
    thresh = _where(nbr_valid, (rp_j ** 2 - rp_i ** 2 - r2_ij) / (2.0 * rp_i), -1e9)

    if is_t:
        sphere_t = torch.as_tensor(sphere.T, dtype=dt, device=dev)
        buried = (torch.matmul(d_ij, sphere_t) < thresh.unsqueeze(-1)).any(dim=1)
        sasa_atom = 4.0 * math.pi * (r_p ** 2) * (~buried).to(dt).mean(dim=1)
        return sasa_atom if return_per_atom else sasa_atom.sum()
    if use_mps and torch.backends.mps.is_available() and n >= 128:
        mps = torch.device("mps")
        buried = torch.any(
            torch.matmul(torch.as_tensor(d_ij, dtype=torch.float32, device=mps),
                         torch.as_tensor(sphere.T, dtype=torch.float32, device=mps))
            < torch.as_tensor(thresh[:, :, None], dtype=torch.float32, device=mps),
            dim=1,
        )
        frac = (~buried).float().mean(dim=1).cpu().numpy().astype(np.float64)
    else:
        frac = np.mean(~np.any(np.einsum("nkd,pd->nkp", d_ij, sphere) < thresh[:, :, None], axis=1), axis=1)
    sasa_atom_np = np.asarray(4.0 * math.pi * (r_p ** 2) * frac, dtype=np.float64)
    return sasa_atom_np if return_per_atom else float(np.sum(sasa_atom_np))


@dataclass
class GBSAContext:
    """Precomputed geometry-only GB kernel (`inv_f_gb`) and SASA for fast multi-charge evaluation."""

    inv_f_gb: np.ndarray | Tensor
    sasa_A2: float | Tensor
    alpha: np.ndarray | Tensor
    radii: np.ndarray | Tensor
    pos: np.ndarray | Tensor | None = None
    r_mat: np.ndarray | Tensor | None = None
    sasa_atom_A2: np.ndarray | Tensor | None = None


def build_gbsa_context(
    z: ArrayLike | Sequence[int], pos: ArrayLike, *, method: GBMethod = "obc2", radii_set: RadiiSet = "mbondi2",
    radius_scale: float = 1.0, dielectric_offset: float = 0.09, sr_use_rho0: bool | None = None,
    sasa_n_pts: int = 194, probe_radius: float = 1.4, compute_sasa_term: bool = True, use_mps: bool = True,
) -> GBSAContext:
    """Precompute Born radii, pairwise `1 / f_GB` matrix, and SASA (NumPy or PyTorch)."""
    alpha, radii, r_mat = compute_born_radii(
        z, pos, method=method, radii_set=radii_set, radius_scale=radius_scale,
        dielectric_offset=dielectric_offset, sr_use_rho0=sr_use_rho0,
    )
    is_t = torch.is_tensor(alpha)
    pos_arr = _to_float(pos, is_t, alpha.device if is_t else None, alpha.dtype if is_t else None)
    aa, r2 = alpha[:, None] * alpha[None, :], r_mat * r_mat
    exp_fn, sqrt_fn = (torch.exp, torch.sqrt) if is_t else (np.exp, np.sqrt)
    inv_f_gb = 1.0 / sqrt_fn(r2 + aa * exp_fn(-r2 / _clamp_min(4.0 * aa, 1e-8)))
    sasa_atom = (
        compute_sasa(pos_arr, radii, probe_radius=probe_radius, n_pts=sasa_n_pts, r_mat=r_mat, use_mps=use_mps, return_per_atom=True)
        if compute_sasa_term else None
    )
    if sasa_atom is not None:
        sasa: float | Tensor = sasa_atom.sum() if is_t else float(np.sum(sasa_atom))
    else:
        sasa = alpha.new_zeros(()) if is_t else 0.0
    return GBSAContext(
        inv_f_gb=inv_f_gb, sasa_A2=sasa, alpha=alpha, radii=radii,
        pos=pos_arr, r_mat=r_mat, sasa_atom_A2=sasa_atom,
    )


def eval_gbsa_from_context(
    ctx: GBSAContext, q: ArrayLike | Sequence[float], *, eps_solv: float = 78.4, eps_in: float = 1.0,
    gamma_sasa: float = 0.00542, sasa_offset: float = 0.0,
) -> dict[str, Any]:
    """Evaluate GBSA solvation free energy (kcal/mol) from a precomputed `GBSAContext`."""
    pref = -0.5 * K_COUL * (1.0 / float(eps_in) - 1.0 / float(eps_solv))
    is_t, dev, dt = _backend(ctx.inv_f_gb, q)
    inv_f, q_arr, alpha = _to_float(ctx.inv_f_gb, is_t, dev, dt), _to_float(q, is_t, dev, dt), _to_float(ctx.alpha, is_t, dev, dt)
    sasa = _to_float(ctx.sasa_A2, is_t, dev, dt) if is_t else float(ctx.sasa_A2)
    n = int(q_arr.shape[0])
    if is_t:
        quad = torch.dot(q_arr, torch.mv(inv_f, q_arr)) if n > 0 else inv_f.new_zeros(())
        zero = inv_f.new_zeros(())
        a_min, a_max = (alpha.min(), alpha.max()) if n > 0 else (zero, zero)
    else:
        quad = float(q_arr @ (inv_f @ q_arr)) if n > 0 else 0.0
        a_min, a_max = (float(np.min(alpha)), float(np.max(alpha))) if n > 0 else (0.0, 0.0)
    dg_gb, dg_np = pref * quad, float(gamma_sasa) * sasa + float(sasa_offset)
    return {"dg_solv": dg_gb + dg_np, "dg_gb": dg_gb, "dg_np": dg_np, "sasa_A2": sasa, "alpha_min": a_min, "alpha_max": a_max}


def compute_gbsa(
    z: ArrayLike | Sequence[int], pos: ArrayLike, q: ArrayLike | Sequence[float], *, method: GBMethod = "obc2",
    radii_set: RadiiSet = "mbondi2", eps_solv: float = 78.4, eps_in: float = 1.0, gamma_sasa: float = 0.00542,
    sasa_offset: float = 0.0, sasa_n_pts: int = 194, probe_radius: float = 1.4, radius_scale: float = 1.0,
    dielectric_offset: float = 0.09, sr_use_rho0: bool | None = None, compute_sasa_term: bool = True,
    use_mps: bool = True,
) -> dict[str, Any]:
    """Compute GBSA solvation free energy and diagnostics (kcal/mol) for a molecule or complex."""
    ctx = build_gbsa_context(
        z, pos, method=method, radii_set=radii_set, radius_scale=radius_scale, dielectric_offset=dielectric_offset,
        sr_use_rho0=sr_use_rho0, sasa_n_pts=sasa_n_pts, probe_radius=probe_radius, compute_sasa_term=compute_sasa_term,
        use_mps=use_mps,
    )
    return eval_gbsa_from_context(ctx, q, eps_solv=eps_solv, eps_in=eps_in, gamma_sasa=gamma_sasa, sasa_offset=sasa_offset)


if TYPE_CHECKING:
    from lam4.calc.gk_multipole import (
        compute_fragment_ddg_solv as compute_fragment_ddg_solv,
        compute_multipole_gbsa as compute_multipole_gbsa,
        eval_multipole_gbsa_from_context as eval_multipole_gbsa_from_context,
    )


def __getattr__(name: str) -> Any:
    if name in {
        "compute_fragment_ddg_solv",
        "compute_multipole_gbsa",
        "eval_multipole_gbsa_from_context",
    }:
        from lam4.calc import gk_multipole

        return getattr(gk_multipole, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
