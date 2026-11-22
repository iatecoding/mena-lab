"""Angular branch of one stage (earlier Mena `AngularBranch`, mena_ace.py): invariants of order 3 and 4 from the l=1,2,3 densities.

Inputs per atom: T [N, K2, 5] (l=2 density, already normalised: the edge op's "t"), v [N, K, 3] (bounded vectors u_new), T3 [N, k3, 7] (the
edge op's "o"). The l=3 density is NOT built here (the earlier Mena trunk's `ang.density` + `radial3` + `gate3`): it comes from the edge op, whose table
has the f3 columns and whose gates have the gate_o columns. Parameters (names as earlier Mena `ang.*`, minus radial3 / gate3):
    ttt_c  K2 -> ttt_mix        nu3_ttt_c  = tr(M(C_c T)^3) = 3 det M(C_c T)
    vttv_a, vttv_b  K2 -> r;  vttv_p, vttv_q  K -> r      nu4_vttv_c = (P_c v)^T M(A_c T) M(B_c T) (Q_c v)
    u3, v3, w3  k3 -> k3;  p3  K -> k3;  a3  K2 -> k3      nu2_3 = <U3 T3, V3 T3>,  nu3_vt3 = <W3 T3, (P3 v) (x) M(A3 T)>
    out    (ttt_mix + r + 2 k3) -> d_model, bias-free, ZERO-INIT (the module starts as the identity); the others keep the default init.
M(t) = sum_m t_m E_m is the symmetric traceless 3x3 matrix of the l=2 components, E_m the orthonormal basis of `l2_pair`.

Constants (built once per dtype/device from float64): E9 [9, 5] (outer product of two vectors -> l2_pair) and B27 [27, 7] (u (x) u (x) u -> y3,
the orthonormal STF basis of lam4.core.harmonics.y3, obtained by solving the 10 cubic monomial coefficients, equal to earlier Mena `_l3_tensors` at 1e-12).
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.core.consts import storable, tracing
from lam4.core.harmonics import y3 as _y3
from lam4.core.radial import fp32_guard, norm_device, resolve_baked, upcast_fp32
from lam4.core.vector_bound import chan_linear, dot_last, outer_flat
from lam4.kernels.metal_node_flags import compiled_angular_elemwise

_R2, _R6 = 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)
_MONOS = ((3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1))


def l2_pair(p: Tensor, q: Tensor) -> Tensor:
    """Components p^T E_m q of sym(p q^T) in the orthonormal basis E_m of symmetric traceless 3x3 matrices: [..., 3] x [..., 3] -> [..., 5]."""
    with fp32_guard(p.device):
        px, py, pz = upcast_fp32(p).unbind(-1)
        qx, qy, qz = upcast_fp32(q).unbind(-1)
        return torch.stack(((px * qy + py * qx) * _R2, (px * qz + pz * qx) * _R2, (py * qz + pz * qy) * _R2, (px * qx - py * qy) * _R2,
                            (2.0 * pz * qz - px * qx - py * qy) * _R6), dim=-1)


def _e9() -> Tensor:
    eye = torch.eye(3, dtype=torch.float64)
    return l2_pair(eye.unsqueeze(1).expand(3, 3, 3), eye.unsqueeze(0).expand(3, 3, 3)).reshape(9, 5)


def _b27() -> Tensor:
    g = torch.Generator().manual_seed(0)
    u = torch.randn(80, 3, generator=g, dtype=torch.float64)  # y3 is a homogeneous cubic: the monomial coefficients are exact for any u
    mono = torch.stack([u[:, 0] ** a * u[:, 1] ** b * u[:, 2] ** c for a, b, c in _MONOS], dim=-1)
    coef = torch.linalg.lstsq(mono, _y3(u)).solution  # [10, 7]
    out = torch.zeros(3, 3, 3, 7, dtype=torch.float64)
    for idx in itertools.product(range(3), repeat=3):  # a monomial's coefficient is shared by its 3!/(a! b! c!) index orders
        e = tuple(idx.count(ax) for ax in range(3))
        out[idx] = coef[_MONOS.index(e)] / (6 // (math.factorial(e[0]) * math.factorial(e[1]) * math.factorial(e[2])))
    return out.reshape(27, 7)


_CONST: dict = {}
_BASE: dict = {}   # float64 CPU bases, built eagerly on first use OUTSIDE any trace (b27 needs linalg.lstsq: no FakeTensor support)


def _base(name: str) -> Tensor:
    b = _BASE.get(name)
    if b is None:
        if tracing():
            raise RuntimeError("angular constants must be built eagerly before tracing (call lam4.core.angular.warm_constants())")
        b = _BASE[name] = {"e9": _e9, "b27": _b27}[name]()
    return b


def warm_constants() -> None:
    for name in ("e9", "b27"):
        _base(name)
    _y3_terms()


def _const(name: str, like: Tensor) -> Tensor:
    dev = norm_device(like.device)
    dt = torch.float64 if like.dtype == torch.float64 else torch.float32
    key = (name, dt, dev)
    c = _CONST.get(key)
    if c is None:
        c = _base(name).to(dtype=dt).to(device=dev)
        if storable(c):
            _CONST[key] = c
    return c


def l2_pair_mm(p: Tensor, q: Tensor) -> Tensor:
    """l2_pair as one outer product and one matmul (3 ops instead of ~24; small autograd graph).

    Compiled on MPS the explicit elementwise form is used instead: Inductor fuses it (and its backward) into the surrounding
    kernels, while the [.., 9] x [9, 5] matmul became a 0.03 TFLOP/s bmm (1.3 ms per call at 8k atoms)."""
    if compiled_angular_elemwise(p):
        return l2_pair(p, q)
    with fp32_guard(p.device, disable_tf32=True):
        pf, qf = upcast_fp32(p), upcast_fp32(q)
        return outer_flat(pf, qf) @ _const("e9", pf)


_Y3_TERMS: list | None = None


def _y3_terms() -> list:
    """Nonzero terms of y3_j(v, M) = sum_abc v_a M_bc B27[abc, j] over the 6 unique entries of the symmetric M (l2_mat6 order):
    per output j a list of (a, p, coef) with coef = sum of B27 over the (b, c) orders of entry p. Python floats from the float64 table."""
    global _Y3_TERMS
    if _Y3_TERMS is None:
        if tracing():
            raise RuntimeError("y3 terms must be built eagerly before tracing (lam4.core.angular.warm_constants())")
        b = _base("b27").reshape(3, 3, 3, 7)
        pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        terms = []
        for j in range(7):
            tj = []
            for a in range(3):
                for p, (i, k) in enumerate(pairs):
                    c = float(b[a, i, k, j]) + (float(b[a, k, i, j]) if i != k else 0.0)
                    if abs(c) > 1e-14:
                        tj.append((a, p, c))
            terms.append(tj)
        _Y3_TERMS = terms
    return _Y3_TERMS


def y3_vm(v: Tensor, m6: tuple[Tensor, ...]) -> Tensor:
    """sum_abc v_a M_bc B27[abc, :] as elementwise sums: v [..., 3], M given by its 6 entries -> [..., 7] (= outer_flat(v, M9) @ b27)."""
    va = v.unbind(-1)
    prod: dict = {}
    outs = []
    for tj in _y3_terms():
        acc = None
        for a, p, c in tj:
            x = prod.get((a, p))
            if x is None:
                x = prod[(a, p)] = va[a] * m6[p]
            acc = x * c if acc is None else acc + x * c
        outs.append(acc)
    return torch.stack(outs, dim=-1)


def l2_mat6(t: Tensor) -> tuple[Tensor, ...]:
    """The six entries (xx, yy, zz, xy, xz, yz) of M(t) as elementwise expressions of t [..., 5] (no 3x3 matmul, MPS-friendly)."""
    with fp32_guard(t.device):
        t0, t1, t2, t3, t4 = upcast_fp32(t).unbind(-1)
        return t3 * _R2 - t4 * _R6, -t3 * _R2 - t4 * _R6, 2.0 * _R6 * t4, t0 * _R2, t1 * _R2, t2 * _R2


def l2_matvec(m6: tuple[Tensor, ...], p: Tensor) -> Tensor:
    with fp32_guard(p.device):
        xx, yy, zz, xy, xz, yz = (upcast_fp32(x) for x in m6)
        px, py, pz = upcast_fp32(p).unbind(-1)
        return torch.stack((xx * px + xy * py + xz * pz, xy * px + yy * py + yz * pz, xz * px + yz * py + zz * pz), dim=-1)


def l2_det(m6: tuple[Tensor, ...]) -> Tensor:
    """det M; for traceless M (Cayley-Hamilton) tr(M^3) = 3 det M."""
    with fp32_guard(m6[0].device):
        xx, yy, zz, xy, xz, yz = (upcast_fp32(x) for x in m6)
        return xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)


class AngularBranch(nn.Module):
    def __init__(self, d_model: int, k_v: int, k_t: int, k_o: int, ttt_mix: int, vttv_rank: int,
                 k3_read: int | None = None, *, has_out: bool = True) -> None:
        super().__init__()
        self.kt, self.kr, self.k3, self.ko = int(ttt_mix), int(vttv_rank), int(k_o if k3_read is None else k3_read), int(k_o)
        lin = lambda i, o: nn.Linear(i, o, bias=False)  # noqa: E731
        self.ttt_c = lin(k_t, self.kt)
        self.vttv_a, self.vttv_b = lin(k_t, self.kr), lin(k_t, self.kr)
        self.vttv_p, self.vttv_q = lin(k_v, self.kr), lin(k_v, self.kr)
        self.u3, self.v3, self.w3 = lin(self.ko, self.k3), lin(self.ko, self.k3), lin(self.ko, self.k3)
        self.p3, self.a3 = lin(k_v, self.k3), lin(k_t, self.k3)
        self.n_out = self.kt + self.kr + 2 * self.k3
        self.out = lin(self.n_out, d_model) if has_out else None
        if self.out is not None:
            nn.init.zeros_(self.out.weight)
        self._baked: tuple[Tensor, Tensor, Tensor] | None = None

    def train(self, mode: bool = True):
        if mode:
            self._baked = None
        return super().train(mode)

    def _apply(self, fn):
        had_baked = self._baked is not None
        self._baked = None
        out = super()._apply(fn)
        if had_baked and not self.training and not self.ttt_c.weight.is_meta and not tracing():
            self.bake_for_inference()
        return out

    @torch.no_grad()
    def bake_for_inference(self) -> None:
        _const("e9", self.ttt_c.weight)
        _const("b27", self.ttt_c.weight)
        self._baked = (
            torch.cat([self.ttt_c.weight, self.vttv_a.weight, self.vttv_b.weight, self.a3.weight], 0).detach().contiguous(),
            torch.cat([self.vttv_p.weight, self.vttv_q.weight, self.p3.weight], 0).detach().contiguous(),
            torch.cat([self.u3.weight, self.v3.weight, self.w3.weight], 0).detach().contiguous(),
        )

    def _get_baked(self, like: Tensor) -> tuple[Tensor, Tensor, Tensor] | None:
        return resolve_baked(self, lambda bk: bk[0], self.ttt_c.weight, like)

    def features(self, t: Tensor, v: Tensor, t3: Tensor) -> Tensor:
        """t [N, K2, 5], v [N, K, 3] (bounded), t3 [N, k3, 7] -> invariants [N, n_out]."""
        with fp32_guard(t.device, disable_tf32=True):
            tf, vf, t3f = upcast_fp32(t), upcast_fp32(v), upcast_fp32(t3)
            bk = self._get_baked(tf)
            if bk is not None:
                w_yt, w_yv, w_u3 = bk
            else:
                w_yt = upcast_fp32(torch.cat([self.ttt_c.weight, self.vttv_a.weight, self.vttv_b.weight, self.a3.weight], 0))
                w_yv = upcast_fp32(torch.cat([self.vttv_p.weight, self.vttv_q.weight, self.p3.weight], 0))
                w_u3 = upcast_fp32(torch.cat([self.u3.weight, self.v3.weight, self.w3.weight], 0))
            yt = chan_linear(tf, w_yt)  # [N, kt + 2 kr + k3, 5]
            yv = chan_linear(vf, w_yv)
            o = self.kt
            ma, mb = l2_mat6(yt[..., o:o + self.kr, :]), l2_mat6(yt[..., o + self.kr:o + 2 * self.kr, :])
            feats = [3.0 * l2_det(l2_mat6(yt[..., :self.kt, :])),
                     dot_last(l2_matvec(ma, yv[..., :self.kr, :]), l2_matvec(mb, yv[..., self.kr:2 * self.kr, :]))]
            u3_t, v3_t, w3_t = chan_linear(t3f, w_u3).split([self.k3] * 3, dim=-2)
            feats.append(dot_last(u3_t, v3_t))
            if compiled_angular_elemwise(tf):  # elementwise form: fused by Inductor (the [k3, 5] x [5, 9] and [27] x [27, 7] matmuls were slow bmm/mm)
                feats.append(dot_last(w3_t, y3_vm(yv[..., 2 * self.kr:, :], l2_mat6(yt[..., o + 2 * self.kr:, :]))))
            else:
                m9 = yt[..., o + 2 * self.kr:, :] @ _const("e9", tf).t()  # [N, k3, 9]: M(A3 T) row-major
                vm = outer_flat(yv[..., 2 * self.kr:, :], m9)  # [N, k3, 27]: v_a M_bc
                feats.append(dot_last(w3_t, vm @ _const("b27", vm)))
            return torch.cat(feats, dim=-1)

    def forward(self, t: Tensor, v: Tensor, t3: Tensor) -> Tensor:
        """The additive pre-activation contribution out(features) [N, d_model]."""
        feats = self.features(t, v, t3)
        if not torch.is_autocast_enabled(feats.device.type) and feats.dtype != self.out.weight.dtype:
            feats = feats.to(self.out.weight.dtype)
        return self.out(feats)


warm_constants()
