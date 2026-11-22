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

from lam4.core.consts import capturing, is_fake, proxy_tracing, tracing

N_RBF = 32
R_CUT = 6.0
R_ON = 5.0
WARP_NUM = 1.52  # s_ij = WARP_NUM / (rho_i + rho_j): identity warp for rho_i + rho_j = 1.52 (e.g. C-C)
FALLBACK_RADIUS = 0.76  # Angstrom, for elements without an ase covalent radius


_CENTRES_CACHE: dict[tuple[float, int, str, int | None, torch.dtype], Tensor] = {}

if not hasattr(getattr(torch.ops, "lam4", None), "stop_grad"):

    @torch.library.custom_op("lam4::stop_grad", mutates_args=())
    def _stop_grad_op(x: Tensor) -> Tensor:
        return x.detach().clone()

    @_stop_grad_op.register_fake
    def _stop_grad_fake(x: Tensor) -> Tensor:
        return torch.empty_like(x)

    _stop_grad_op.register_autograd(lambda _ctx, _g: None)


def stop_grad(x: Tensor) -> Tensor:
    """Explicit stop-gradient that never decomposes into `aten.alias.default` during `make_fx`."""
    return torch.ops.lam4.stop_grad.default(x) if proxy_tracing() else x.detach()


def norm_device(dev: torch.device | str) -> torch.device:
    """Resolve `index=None` on CUDA/MPS so module-level caches never duplicate `cuda` vs `cuda:0`."""
    d = torch.device(dev)
    if d.index is not None:
        return d
    if d.type == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if d.type == "mps":
        return torch.device("mps", 0)
    return d


def upcast_fp32(x: Tensor | None) -> Tensor | None:
    return None if x is None else (x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x)


def resolve_baked(mod: nn.Module, w_fn, param: Tensor, like: Tensor):
    """Return `mod._baked` iff safe for `like` (never leaking stale device/dtype or real-into-FakeTensor)."""
    bk = getattr(mod, "_baked", None)
    if bk is None or mod.training or param.requires_grad:
        return None
    if not torch.compiler.is_compiling() and (proxy_tracing() or is_fake(like)):
        return None
    w = w_fn(bk)
    if w.device != like.device or w.dtype != like.dtype:
        if (torch.compiler.is_compiling() or tracing() or like.is_meta or param.is_meta
                or param.device != like.device or param.dtype != like.dtype):
            return None
        mod.bake_for_inference()
        bk = getattr(mod, "_baked", None)
    return bk


class _NoopGuard:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, _exc_type, _exc, _tb):
        return False


_NOOP_GUARD = _NoopGuard()


class _Fp32Guard:
    __slots__ = ("_ac_ctx", "_prev_tf32", "_prev_prec")

    def __init__(self, dt: str, ac: bool, prev_tf32: bool, prev_prec: str) -> None:
        self._ac_ctx = torch.amp.autocast(device_type=dt, enabled=False) if ac else None
        self._prev_tf32 = prev_tf32
        self._prev_prec = prev_prec

    def __enter__(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if self._ac_ctx is not None:
            self._ac_ctx.__enter__()
        return None

    def __exit__(self, exc_type, exc, tb):
        if self._ac_ctx is not None:
            self._ac_ctx.__exit__(exc_type, exc, tb)
        torch.backends.cuda.matmul.allow_tf32 = self._prev_tf32
        torch.set_float32_matmul_precision(self._prev_prec)
        return False


def fp32_guard(device: torch.device, *, disable_tf32: bool = False):
    """Disable BF16/FP16 autocast (and optionally CUDA TF32 matmul) with a zero-allocation no-op fast path."""
    dt = device.type
    ac = dt in ("cuda", "cpu", "mps") and torch.is_autocast_enabled(dt)
    if disable_tf32 and dt == "cuda" and not torch.compiler.is_compiling() and not capturing():
        prev_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        prev_prec = torch.get_float32_matmul_precision()
        if prev_tf32 or prev_prec != "highest":
            return _Fp32Guard(dt, ac, prev_tf32, prev_prec)
    return torch.amp.autocast(device_type=dt, enabled=False) if ac else _NOOP_GUARD


def _smoothstep_t(x: Tensor, r_on: float, r_cut: float) -> Tensor:
    x = upcast_fp32(x)
    return ((x - r_on) / (r_cut - r_on)).clamp(min=0.0, max=1.0)


def c3_envelope(x: Tensor, r_on: float = R_ON, r_cut: float = R_CUT) -> Tensor:
    """env(x): 1 for x <= r_on, 0 for x >= r_cut, C3 in between."""
    with fp32_guard(x.device):
        t = _smoothstep_t(x, r_on, r_cut)
        t2 = t * t
        return 1.0 - t2 * t2 * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)))


def c3_envelope_d(x: Tensor, r_on: float = R_ON, r_cut: float = R_CUT, order: int = 1) -> Tensor:
    """Analytic d^order env / dx^order for order 1 or 2 (zero outside (r_on, r_cut))."""
    with fp32_guard(x.device):
        xf = upcast_fp32(x)
        t = _smoothstep_t(xf, r_on, r_cut)
        w = r_cut - r_on
        inside = ((xf > r_on) & (xf < r_cut)).to(xf.dtype)
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
        """The 32 fixed centres in the dtype/device of `like` (no buffers, so MPS .to() works)."""
        dt = torch.float64 if like.dtype == torch.float64 else torch.float32
        if capturing() or torch.compiler.is_compiling() or proxy_tracing() or is_fake(like):
            if dt == torch.float32:
                return torch.arange(self.n_rbf, device=like.device, dtype=torch.float32) * (float(self.r_cut) / max(1, self.n_rbf - 1))
            return torch.linspace(0.0, self.r_cut, self.n_rbf, device=like.device, dtype=dt)
        key = (float(self.r_cut), int(self.n_rbf), like.device.type, like.device.index, dt)
        c = _CENTRES_CACHE.get(key)
        if c is None:
            c = torch.linspace(0.0, self.r_cut, self.n_rbf, dtype=torch.float64).to(device=like.device, dtype=dt)
            if len(_CENTRES_CACHE) > 32:
                _CENTRES_CACHE.clear()
            _CENTRES_CACHE[key] = c
        return c

    def gaussians(self, x: Tensor) -> Tensor:
        with fp32_guard(x.device):
            xf = upcast_fp32(x)
            z = (xf.unsqueeze(-1) - self.centres(xf)) / self.sigma
            return torch.exp(-0.5 * z * z)

    def forward(self, x: Tensor) -> Tensor:
        return self.gaussians(x)

    def derivative(self, x: Tensor) -> Tensor:
        """dPhi/dx, analytic."""
        with fp32_guard(x.device):
            xf = upcast_fp32(x)
            return -self.gaussians(xf) * (xf.unsqueeze(-1) - self.centres(xf)) / (self.sigma ** 2)


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
        with fp32_guard(self.log_rho.device):
            return upcast_fp32(self.log_rho).exp().index_select(0, z)

    def forward(self, r: Tensor, rho_i: Tensor, rho_j: Tensor) -> Tensor:
        return warp_distance(r, rho_i + rho_j, self.r_cut)


def warp_distance(r: Tensor, rho_ij: Tensor | None, r_cut: float = R_CUT) -> Tensor:
    """x(r) for rho_ij = rho_i + rho_j > 0 (None = identity). Returns exactly r_cut where r >= r_cut."""
    if rho_ij is None:
        return upcast_fp32(r)
    with fp32_guard(r.device):
        rf = upcast_fp32(r)
        rh = upcast_fp32(rho_ij)
        s = WARP_NUM / rh
        warped = s * rf / (1.0 + (s - 1.0) * (rf / r_cut))
        return torch.where(rf < r_cut, warped, torch.full_like(rf, r_cut))
