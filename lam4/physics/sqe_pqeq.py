"""sqe_pqeq.py: singularity-free scaled SQE + variational fixed-K_fix implicit-adjoint PCG-PQEq."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.core.angular import l2_mat6, l2_pair
from lam4.core.consts import capturing, is_fake, proxy_tracing
from lam4.core.radial import c3_envelope, fp32_guard, upcast_fp32
from lam4.core.vector_bound import bound_vectors, chan_linear
from lam4.physics.coulomb_exact import COULOMB_K, DEFAULT_CAP, SIGMA, make_slots_static, resolve_coulomb_meta
from lam4.physics.spme_batched import BatchedSPME


def _smeared_k_f1_f2(r2: Tensor, pm: Tensor, sigma: float, *, need_f3: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Taylor-safe (x^2 < 2.5e-3) smeared Coulomb kernel and derivatives k, f1, f2, optional f3."""
    inv_sig = 1.0 / float(sigma); inv_sig2, c0 = inv_sig * inv_sig, 2.0 * inv_sig / math.sqrt(math.pi)
    c0_s2, x2, small, x4 = c0 * inv_sig2, r2 * inv_sig2, r2 * inv_sig2 < 2.5e-3, (r2 * inv_sig2) ** 2
    r2_s = torch.where(pm & ~small, r2, torch.ones_like(r2)); inv_r = torch.rsqrt(r2_s); inv_r2, ex = inv_r * inv_r, torch.exp(-x2)
    k_g = torch.erf(r2_s * inv_r * inv_sig) * inv_r; f1_g = (k_g - c0 * ex) * inv_r2
    f2_g = (2.0 * c0_s2 * ex - 3.0 * f1_g) * inv_r2; pmf = pm.to(r2.dtype)
    f3 = None
    if need_f3:
        f3_g = (-4.0 * c0_s2 * inv_sig2 * ex - 5.0 * f2_g) * inv_r2
        f3 = torch.where(small, (c0_s2 * inv_sig2 * inv_sig2) * (8.0 / 7.0 - (8.0 / 9.0) * x2 + (4.0 / 11.0) * x4), f3_g) * pmf
    return (torch.where(small, c0 * (1.0 - (1.0 / 3.0) * x2 + 0.1 * x4), k_g) * pmf,
            torch.where(small, c0_s2 * ((2.0 / 3.0) - 0.4 * x2 + (1.0 / 7.0) * x4), f1_g) * pmf,
            torch.where(small, (c0_s2 * inv_sig2) * (-0.8 + (4.0 / 7.0) * x2 - (2.0 / 9.0) * x4), f2_g) * pmf,
            f3)


_TRITON_FNS: dict[str, object] = {}


def register_triton_multipole_fn(name: str, fn) -> str:
    """Register a differentiable Triton multipole callback for torch.ops.lam4.sqe_pqeq_open."""
    _TRITON_FNS[str(name)] = fn; return str(name)


class OpenMultipoleBackend:
    """Static-slot smeared monopole+dipole backend (r -> 0 smooth; lazy dense skip when Triton active)."""

    def __init__(self, pos: Tensor, atom_mask: Tensor, graph: Tensor, n_graphs: int, cap: int | None = None,
                 sigma: float = SIGMA, triton_multipole_fn=None, triton_key: str = "") -> None:
        with fp32_guard(pos.device, disable_tf32=True):
            self.pos, self.sigma, self.atom_mask, self.graph, self.n_graphs = upcast_fp32(pos), float(sigma), atom_mask, graph, int(n_graphs)
            self.cap = max(1, int(cap)) if cap is not None else max(int(DEFAULT_CAP), int(pos.shape[0]))
            self.triton_fn = triton_multipole_fn if triton_multipole_fn is not None else _TRITON_FNS.get(str(triton_key))
            if self.triton_fn is not None and not triton_key:
                triton_key = f"auto_{id(self.triton_fn)}"; _TRITON_FNS[triton_key] = self.triton_fn
            self.triton_key, self.layout = str(triton_key), None
            self.pm = self.diff = self.k = self.f1 = self.f2 = self.f3 = self._m = self._m44 = None
            if self.triton_fn is None: self._ensure_dense()

    def _ensure_dense(self) -> None:
        if self.k is not None: return
        self.layout = make_slots_static(self.atom_mask, self.graph, self.n_graphs, self.cap)
        slots, n = self.layout.slots, self.pos.shape[0]; valid, c = slots < n, slots.shape[1]
        p_slot = torch.cat([self.pos, self.pos.new_zeros(1, 3)], 0).index_select(0, slots.reshape(-1)).view(*slots.shape, 3)
        self.pm = valid.unsqueeze(2) & valid.unsqueeze(1) & ~torch.eye(c, dtype=torch.bool, device=self.pos.device).unsqueeze(0)
        self.diff = p_slot.unsqueeze(2) - p_slot.unsqueeze(1)
        self.r2 = self.diff.pow(2).sum(-1)
        self.k, self.f1, self.f2, self.f3 = _smeared_k_f1_f2(self.r2, self.pm, self.sigma, need_f3=False)
        self._m = self.layout.atom_mask.to(self.pos.dtype)

    def _ensure_f3(self) -> Tensor:
        if self.f3 is not None: return self.f3
        self._ensure_dense()
        inv_sig = 1.0 / self.sigma; inv_sig2, c0 = inv_sig * inv_sig, 2.0 * inv_sig / math.sqrt(math.pi)
        c0_s2, x2, small, x4 = c0 * inv_sig2, self.r2 * inv_sig2, self.r2 * inv_sig2 < 2.5e-3, (self.r2 * inv_sig2) ** 2
        r2_s = torch.where(self.pm & ~small, self.r2, torch.ones_like(self.r2))
        f3_g = (-4.0 * c0_s2 * inv_sig2 * torch.exp(-x2) - 5.0 * self.f2) / r2_s
        self.f3 = torch.where(small, (c0_s2 * inv_sig2 * inv_sig2) * (8.0 / 7.0 - (8.0 / 9.0) * x2 + (4.0 / 11.0) * x4), f3_g) * self.pm.to(self.r2.dtype)
        return self.f3

    def _get_m44(self) -> Tensor | None:
        if self._m44 is not None: return self._m44
        self._ensure_dense(); b, c = self.k.shape[:2]
        if b * c * c > 3_000_000: return None
        top_r, eye3 = (self.f1.unsqueeze(-1) * self.diff).reshape(b, c, 3 * c), torch.eye(3, dtype=self.k.dtype, device=self.k.device)
        t5 = self.f1[..., None, None] * eye3 + self.f2[..., None, None] * (self.diff.unsqueeze(-1) * self.diff.unsqueeze(-2))
        self._m44 = torch.cat((torch.cat((self.k, top_r), -1), torch.cat((top_r.transpose(-1, -2), t5.permute(0, 1, 3, 2, 4).reshape(b, 3 * c, 3 * c)), -1)), -2)
        return self._m44

    @classmethod
    def from_batch(cls, batch, pos: Tensor | None = None, sigma: float = SIGMA, triton_multipole_fn=None, triton_key: str = ""):
        eff_g, eff_cap = resolve_coulomb_meta(batch, DEFAULT_CAP)
        eff_graph = torch.zeros_like(batch.graph) if (eff_g == 1 and batch.n_graphs > 1) else batch.graph
        return cls(batch.pos if pos is None else pos, batch.atom_mask, eff_graph, eff_g, eff_cap, sigma, triton_multipole_fn, triton_key)

    def _to_slots(self, x: Tensor) -> Tensor:
        return torch.cat([x, x.new_zeros((1,) + tuple(x.shape[1:]))], 0).index_select(0, self.layout.slots.reshape(-1)).view(*self.layout.slots.shape, *x.shape[1:])

    def _from_slots(self, xs: Tensor) -> Tensor:
        return xs.reshape(-1, *xs.shape[2:]).index_select(0, self.layout.atom_slot) * (self._m if xs.ndim == 2 else self._m.unsqueeze(-1))

    def phi(self, q: Tensor) -> Tensor: return self.phi_field_multipole(q, None, need_efg=False)[0]
    def phi_field(self, q: Tensor) -> tuple[Tensor, Tensor]: return self.phi_field_multipole(q, None, need_efg=False)[:2]

    def phi_field_multipole(self, q: Tensor, p: Tensor | None = None, *, need_efg: bool = False) -> tuple[Tensor, Tensor, Tensor | None]:
        if self.triton_fn is not None:
            return self.triton_fn(self.pos, q, p, self.atom_mask, self.graph, self.n_graphs, self.sigma, need_efg)
        self._ensure_dense()
        with fp32_guard(self.pos.device, disable_tf32=True):
            qs = self._to_slots(upcast_fp32(q).to(self.pos.dtype) * self._m)
            ps = self._to_slots(upcast_fp32(p).to(self.pos.dtype) * self._m.unsqueeze(-1)) if p is not None else None
            m44 = self._get_m44() if ps is not None else None
            if m44 is not None and ps is not None:
                b, c = qs.shape
                y = torch.bmm(m44, torch.cat((qs, ps.reshape(b, 3 * c)), dim=-1).unsqueeze(-1)).squeeze(-1)
                phi_res, fld_res = self._from_slots(y[:, :c]), self._from_slots(-y[:, c:].reshape(b, c, 3))
                if not need_efg:
                    return phi_res, fld_res, None
                d_dot_p = torch.einsum("bijc,bjc->bij", self.diff, ps)
                f3 = self._ensure_f3()
                efg = self._from_slots(
                    torch.einsum("bij,bijm,bj->bim", self.f2, l2_d := l2_pair(self.diff, self.diff), qs)
                    - 2.0 * torch.einsum("bij,bijm->bim", self.f2, l2_pair(self.diff, ps.unsqueeze(1)))
                    - torch.einsum("bij,bij,bijm->bim", f3, d_dot_p, l2_d))
                return phi_res, fld_res, efg
            phi_s, fld_s = torch.bmm(self.k, qs.unsqueeze(-1)).squeeze(-1), torch.einsum("bijc,bij,bj->bic", self.diff, self.f1, qs)
            if ps is not None:
                d_dot_p = torch.einsum("bijc,bjc->bij", self.diff, ps)
                phi_s = phi_s + torch.einsum("bij,bij->bi", self.f1, d_dot_p)
                fld_s = fld_s - torch.bmm(self.f1, ps) - torch.einsum("bij,bij,bijc->bic", self.f2, d_dot_p, self.diff)
            efg = (self._from_slots(
                torch.einsum("bij,bijm,bj->bim", self.f2, l2_d := l2_pair(self.diff, self.diff), qs)
                - (0.0 if ps is None else 2.0 * torch.einsum("bij,bijm->bim", self.f2, l2_pair(self.diff, ps.unsqueeze(1)))
                   + torch.einsum("bij,bij,bijm->bim", self._ensure_f3(), d_dot_p, l2_d))) if need_efg else None)
            return self._from_slots(phi_s), self._from_slots(fld_s), efg

    def virial_atom(self, q: Tensor, p: Tensor | None = None) -> Tensor:
        """Exact open-boundary pair Cauchy virial W_i [N, 3, 3] summing to dU/d_eps."""
        self._ensure_dense()
        with fp32_guard(self.pos.device, disable_tf32=True):
            qs = self._to_slots(upcast_fp32(q).to(self.pos.dtype) * self._m)
            with torch.enable_grad():
                d_ij = self.diff.detach().requires_grad_(True)
                k, f1, f2, _ = _smeared_k_f1_f2(d_ij.pow(2).sum(-1), self.pm, self.sigma)
                u_pair = 0.5 * COULOMB_K * (qs.unsqueeze(2) * qs.unsqueeze(1) * k)
                if p is not None:
                    ps = self._to_slots(upcast_fp32(p).to(self.pos.dtype) * self._m.unsqueeze(-1))
                    dp_j, dp_i = torch.einsum("bijc,bjc->bij", d_ij, ps), torch.einsum("bijc,bic->bij", d_ij, ps)
                    u_pair = u_pair + 0.5 * COULOMB_K * (2.0 * qs.unsqueeze(2) * dp_j * f1 + torch.einsum("bic,bjc->bij", ps, ps) * f1 + dp_i * dp_j * f2)
                grad_d = torch.autograd.grad(u_pair.sum(), d_ij)[0]
            w_slot = torch.einsum("gija,gijc->giac", grad_d, self.diff)
            return self._from_slots((0.5 * (w_slot + w_slot.transpose(-1, -2))).reshape(*w_slot.shape[:2], 9)).view(-1, 3, 3) * self._m.view(-1, 1, 1)


def fragment_com_positions(pos: Tensor, frag_group: Tensor, atom_mask: Tensor) -> Tensor:
    """Geometric centroid per fragment/graph [N, 3] for translation-invariant ion E_ext gauge."""
    mf = atom_mask.to(pos.dtype)
    n_f = int(frag_group.max().item()) + 1 if not proxy_tracing() else pos.shape[0]
    cnt = pos.new_zeros(n_f).index_add(0, frag_group, mf).clamp_min(1.0)
    return (pos.new_zeros(n_f, 3).index_add(0, frag_group, pos * mf.unsqueeze(-1)) / cnt.unsqueeze(-1)).index_select(0, frag_group)


def _div_split(beta: Tensor, s_tilde: Tensor, send: Tensor, recv: Tensor, n_atoms: int) -> Tensor: return beta.new_zeros(n_atoms).index_add(0, recv, flow := beta * s_tilde).index_add(0, send, -flow)
def _grad_split(beta: Tensor, x: Tensor, send: Tensor, recv: Tensor) -> Tensor: return beta * (x.index_select(0, recv) - x.index_select(0, send))
def _apply_sym6(a6: Tensor, v: Tensor) -> Tensor:
    xx, yy, zz, xy, xz, yz = a6[:, 0], a6[:, 1], a6[:, 2], a6[:, 3], a6[:, 4], a6[:, 5]; vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    return torch.cat([t.unsqueeze(-1) for t in (xx * vx + xy * vy + xz * vz, xy * vx + yy * vy + yz * vz, xz * vx + yz * vy + zz * vz)], dim=-1)
def _inv_sym3x3(a6: Tensor) -> Tensor:
    xx, yy, zz, xy, xz, yz = a6[:, 0], a6[:, 1], a6[:, 2], a6[:, 3], a6[:, 4], a6[:, 5]
    c00, c01, c02 = yy * zz - yz * yz, xz * yz - xy * zz, xy * yz - xz * yy; c11, c12, c22 = xx * zz - xz * xz, xy * xz - xx * yz, xx * yy - xy * xy
    return torch.cat([t.unsqueeze(-1) for t in (c00, c11, c22, c01, c02, c12)], dim=-1) / (xx * c00 + xy * c01 + xz * c02).clamp_min(1e-12).unsqueeze(-1)
def _matvec_sym(backend, dq: Tensor, vp: Tensor, eta: Tensor, alpha_inv6: Tensor) -> tuple[Tensor, Tensor]:
    phi_v, fld_v, _ = backend.phi_field_multipole(dq, vp, need_efg=False); return eta * dq + COULOMB_K * phi_v, COULOMB_K * (_apply_sym6(alpha_inv6, vp) - fld_v)


def _to_slots_lay(x: Tensor, lay) -> Tensor:
    return torch.cat([x, x.new_zeros((1,) + tuple(x.shape[1:]))], 0).index_select(0, lay.slots.reshape(-1)).view(*lay.slots.shape, *x.shape[1:])


def _from_slots_lay(xs: Tensor, lay, mf_slot: Tensor) -> Tensor:
    return xs.reshape(-1, *xs.shape[2:]).index_select(0, lay.atom_slot) * (mf_slot if xs.ndim == 2 else mf_slot.unsqueeze(-1))


def _build_lbeta(lay, beta: Tensor, send: Tensor, recv: Tensor, dt: torch.dtype, dev: torch.device) -> Tensor:
    b, c = lay.slots.shape; keep = lay.atom_mask
    bu, bv = lay.atom_slot.index_select(0, recv), lay.atom_slot.index_select(0, send)
    ok = ((bu // c) == (bv // c)) & keep.index_select(0, recv) & keep.index_select(0, send)
    l_flat, b_dt = torch.zeros(b * c * c, dtype=dt, device=dev), beta.to(dt)
    w = torch.where(ok, b_dt * b_dt, torch.zeros(beta.shape, dtype=dt, device=dev))
    u_loc, v_loc, b_off = bu % c, bv % c, (bu // c) * (c * c)
    l_flat.index_add_(0, b_off + u_loc * (c + 1), w).index_add_(0, b_off + v_loc * (c + 1), w).index_add_(0, bu * c + v_loc, -w).index_add_(0, bv * c + u_loc, -w)
    return l_flat.view(b, c, c)


def _safe_chol_inv(mat: Tensor, mask_ok: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Zero-sync per-graph cholesky inverse returning (inv_sym, ok); failed slices zeroed."""
    eye = torch.eye(mat.shape[1], dtype=mat.dtype, device=mat.device).unsqueeze(0)
    l, info = torch.linalg.cholesky_ex(mat, check_errors=False)
    ok = (info == 0) & torch.isfinite(mat).flatten(1).all(-1)
    if mask_ok is not None:
        ok = ok & mask_ok
    inv = torch.cholesky_solve(eye.expand_as(mat), torch.where(ok.view(-1, 1, 1), l, eye))
    ok = ok & torch.isfinite(inv).flatten(1).all(-1)
    return torch.where(ok.view(-1, 1, 1), 0.5 * (inv + inv.transpose(-1, -2)), torch.zeros_like(inv)), ok


def _make_precond(backend, eta: Tensor, alpha_inv6: Tensor, beta: Tensor, send: Tensor, recv: Tensor):
    """Build zero-sync Cholesky-Woodbury with precomputed fp64 m_node (torch.bmm in CG) or SPD Jacobi."""
    n, dt, dev = eta.shape[0], eta.dtype, eta.device
    m_inv_p6 = _inv_sym3x3(alpha_inv6) / COULOMB_K
    c_all = m_node = lay = mf_slot = None
    if isinstance(backend, OpenMultipoleBackend) and backend.triton_fn is None:
        backend._ensure_dense(); lay = backend.layout; b, c = lay.slots.shape; mf_slot = backend._m
        if b * c * c <= 2_500_000 and (m44 := backend._get_m44()) is not None:
            h44 = float(COULOMB_K) * m44.to(dt).clone()
            eta_s = torch.where(lay.slots < n, _to_slots_lay(eta, lay), eta.new_ones(b, c))
            a6_def = alpha_inv6.new_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            a6_s = torch.where((lay.slots < n).unsqueeze(-1), _to_slots_lay(alpha_inv6, lay), a6_def.view(1, 1, 6))
            h44[:, :c, :c].diagonal(dim1=-2, dim2=-1).add_(eta_s)
            for r0, c0, idx6 in ((0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 1, 3), (1, 0, 3), (0, 2, 4), (2, 0, 4), (1, 2, 5), (2, 1, 5)):
                h44[:, c + r0::3, c + c0::3].diagonal(dim1=-2, dim2=-1).add_(float(COULOMB_K) * a6_s[..., idx6])
            c_all, ok44 = _safe_chol_inv(h44)
            eye_p = torch.eye(4 * c, dtype=dt, device=dev).unsqueeze(0).clone()
            eye_p[:, :c, :c].zero_()
            c_all = torch.where(ok44.view(-1, 1, 1), c_all, eye_p)
            del h44
            l_mat64 = _build_lbeta(lay, beta, send, recv, torch.float64, dev)
            eye_c64 = torch.eye(c, dtype=torch.float64, device=dev).unsqueeze(0).expand(b, -1, -1)
            m_node, _ = _safe_chol_inv(c_all[:, :c, :c].double() + l_mat64 + 1e-10 * eye_c64, ok44)
        elif b * c * c <= 6_000_000:
            eta_s = torch.where(lay.slots < n, _to_slots_lay(eta, lay), eta.new_ones(b, c)).double()
            a_inv64, ok_a = _safe_chol_inv(torch.diag_embed(eta_s) + float(COULOMB_K) * backend.k.double())
            l_mat64 = _build_lbeta(lay, beta, send, recv, torch.float64, dev)
            eye_c64 = torch.eye(c, dtype=torch.float64, device=dev).unsqueeze(0).expand(b, -1, -1)
            m_node, _ = _safe_chol_inv(a_inv64 + l_mat64 + 1e-10 * eye_c64, ok_a)
    inv_eta, w_loc = 1.0 / eta.clamp_min(1e-6), (beta * beta).to(dt)
    d_loc_inv = 1.0 / (inv_eta + eta.new_zeros(n).index_add(0, recv, w_loc).index_add(0, send, w_loc)).clamp_min(1e-6)
    return m_inv_p6, c_all, m_node, lay, mf_slot, inv_eta, d_loc_inv, beta, send, recv, n


def _apply_precond(pre, rs64: Tensor, rp64: Tensor) -> tuple[Tensor, Tensor]:
    m_inv_p6, c_all, m_node, lay, mf_slot, inv_eta, d_loc_inv, beta, send, recv, n = pre
    beta64, dq_r64 = beta.double(), _div_split(beta.double(), rs64, send, recv, n)
    if c_all is not None:
        b, c, dt = lay.slots.shape[0], lay.slots.shape[1], c_all.dtype
        rps = _to_slots_lay(rp64.to(dt), lay).reshape(b, 3 * c)
        rhs_y = _to_slots_lay(dq_r64, lay) - torch.bmm(c_all[:, :c, c:], rps.unsqueeze(-1)).squeeze(-1).double()
        y_s = torch.bmm(m_node, rhs_y.unsqueeze(-1)).squeeze(-1)
        qp_s = torch.bmm(c_all, torch.cat((y_s.to(dt), rps), dim=-1).unsqueeze(-1)).squeeze(-1)
        zs64 = rs64 - _grad_split(beta64, _from_slots_lay(y_s, lay, mf_slot.double()), send, recv)
        return zs64, _from_slots_lay(qp_s[:, c:].reshape(b, c, 3), lay, mf_slot).double()
    dt = beta.dtype
    if m_node is not None:
        y_s = torch.bmm(m_node, _to_slots_lay(dq_r64, lay).unsqueeze(-1)).squeeze(-1)
        ys64 = _from_slots_lay(y_s, lay, mf_slot.double())
    else:
        ys, dq_r = dq_r64.new_zeros(n, dtype=dt), dq_r64.to(dt)
        for _ in range(6):
            ys = ys + 0.85 * d_loc_inv * (dq_r - inv_eta * ys - _div_split(beta, _grad_split(beta, ys, send, recv), send, recv, n))
        ys64 = ys.double()
    return rs64 - _grad_split(beta64, ys64, send, recv), _apply_sym6(m_inv_p6, rp64.to(dt)).double()


def _dot_g(a: Tensor, b: Tensor, idx: Tensor | None, ng: int, dim1: bool = False) -> Tensor:
    p = (a * b).sum(-1) if dim1 else (a * b)
    return p.sum() if (ng <= 1 or idx is None) else p.new_zeros(ng).index_add(0, idx, p)


def _pcg_solve(backend, eta: Tensor, alpha_inv6: Tensor, beta: Tensor, send: Tensor, recv: Tensor,
               rhs_s: Tensor, rhs_p: Tensor, k_fix: int, precond=None) -> tuple[Tensor, Tensor]:
    """Fixed-K_fix Woodbury+block-Jacobi PCG with per-graph guarded fp64 Krylov inner products."""
    pre = _make_precond(backend, eta, alpha_inv6, beta, send, recv) if precond is None else precond
    n, c_all, dt = pre[-1], pre[1], rhs_s.dtype
    ng, g_at = int(getattr(backend, "n_graphs", 1)), getattr(backend, "graph", None)
    g_ed = g_at.index_select(0, recv) if (ng > 1 and g_at is not None) else None
    g_a = g_at if (ng > 1 and g_at is not None) else None
    k_eff = min(int(k_fix), 8) if c_all is not None else int(k_fix)
    xs, xp = torch.zeros_like(rhs_s, dtype=torch.float64), torch.zeros_like(rhs_p, dtype=torch.float64)
    rs, rp = rhs_s.double(), rhs_p.double()
    zs, zp = _apply_precond(pre, rs, rp)
    ps, pp = zs, zp
    rz_old = _dot_g(rs, zs, g_ed, ng) + _dot_g(rp, zp, g_a, ng, True)
    for _ in range(k_eff):
        oq, op = _matvec_sym(backend, _div_split(beta, ps.to(dt), send, recv, n), pp.to(dt), eta, alpha_inv6)
        qs, op_d = (ps.to(dt) + _grad_split(beta, oq, send, recv)).double(), op.double()
        pq = _dot_g(ps, qs, g_ed, ng) + _dot_g(pp, op_d, g_a, ng, True)
        ok_s = (rz_old > 1e-30) & (pq > 1e-30) & torch.isfinite(rz_old) & torch.isfinite(pq)
        step = torch.where(ok_s, rz_old / pq.clamp_min(1e-30), 0.0)
        se, sa = (step, step) if g_ed is None else (step.index_select(0, g_ed), step.index_select(0, g_a).unsqueeze(-1))
        ok_se, ok_sa = (ok_s, ok_s) if g_ed is None else (ok_s.index_select(0, g_ed), ok_s.index_select(0, g_a).unsqueeze(-1))
        xs, xp = torch.where(ok_se, xs + se * ps, xs), torch.where(ok_sa, xp + sa * pp, xp)
        rs, rp = torch.where(ok_se, rs - se * qs, rs), torch.where(ok_sa, rp - sa * op_d, rp)
        zs, zp = _apply_precond(pre, rs, rp)
        rz_new = _dot_g(rs, zs, g_ed, ng) + _dot_g(rp, zp, g_a, ng, True)
        ok_b = (rz_old > 1e-30) & (rz_new > 1e-30) & torch.isfinite(rz_new) & torch.isfinite(rz_old)
        b_cg = torch.where(ok_b, rz_new / rz_old.clamp_min(1e-30), 0.0)
        be, ba = (b_cg, b_cg) if g_ed is None else (b_cg.index_select(0, g_ed), b_cg.index_select(0, g_a).unsqueeze(-1))
        ok_be, ok_ba = (ok_b, ok_b) if g_ed is None else (ok_b.index_select(0, g_ed), ok_b.index_select(0, g_a).unsqueeze(-1))
        ps, pp = torch.where(ok_be, zs + be * ps, zs), torch.where(ok_ba, zp + ba * pp, zp)
        rz_old = torch.where(torch.isfinite(rz_new), rz_new, 0.0)
    return xs.to(dt), xp.to(dt)


def _adjoint_vjp(backend, eta: Tensor, a6: Tensor, beta: Tensor, send: Tensor, recv: Tensor,
                 b_q: Tensor, s_star: Tensor, p_star: Tensor, gs: Tensor, gp: Tensor, k_fix: int, solve_fn=None):
    """Implicit adjoint VJP wrt (eta, alpha_inv6, beta, b_q, b_s_ext, b_p) + returned (dq_star, p_star, dw, wp)."""
    n = eta.shape[0]
    ws, wp = _pcg_solve(backend, eta, a6, beta, send, recv, gs, gp, k_fix) if solve_fn is None else solve_fn(gs, gp)
    dq_star, dw = _div_split(beta, s_star, send, recv, n), _div_split(beta, ws, send, recv, n)
    oq_star, _ = _matvec_sym(backend, dq_star, p_star, eta, a6); oq_w, _ = _matvec_sym(backend, dw, wp, eta, a6)
    wpx, wpy, wpz = wp[:, 0], wp[:, 1], wp[:, 2]; psx, psy, psz = p_star[:, 0], p_star[:, 1], p_star[:, 2]
    d_a6 = -COULOMB_K * torch.cat([t.unsqueeze(-1) for t in (wpx * psx, wpy * psy, wpz * psz, wpx * psy + wpy * psx, wpx * psz + wpz * psx, wpy * psz + wpz * psy)], dim=-1)
    oq_tot = oq_star + b_q
    d_beta = -(ws * (oq_tot.index_select(0, recv) - oq_tot.index_select(0, send)) + s_star * (oq_w.index_select(0, recv) - oq_w.index_select(0, send)))
    return (-dw * dq_star, d_a6, d_beta, -dw, ws, -wp), (dq_star, p_star, dw, wp)


def _dh_dgeom(make_be, dq_star: Tensor, p_star: Tensor, dw: Tensor, wp: Tensor, pos: Tensor, cell: Tensor | None = None) -> tuple[Tensor, ...]:
    """Exact implicit d(-w^T H z*)/d(pos, cell) = -k_C d(dw*phi - wp.E)/d(pos, cell) inside enable_grad."""
    with torch.enable_grad():
        p_l = pos.detach().requires_grad_(True)
        vars_t = (p_l,) if cell is None else (p_l, c_l := cell.detach().requires_grad_(True))
        phi_h, fld_h, _ = make_be(*vars_t).phi_field_multipole(dq_star, p_star, need_efg=False)
        gs = torch.autograd.grad(-COULOMB_K * (dw * phi_h - (wp * fld_h).sum(-1)).sum(), vars_t, create_graph=torch.is_grad_enabled(), allow_unused=True)
    return tuple(torch.zeros_like(v) if g is None else g for g, v in zip(gs, vars_t))


def _register_custom_solve_ops():
    if hasattr(getattr(torch.ops, "lam4", None), "sqe_pqeq_open"):
        return
    _nz = lambda g, ref: torch.zeros_like(ref) if g is None else g

    @torch.library.custom_op("lam4::sqe_pqeq_open", mutates_args=())
    def _open_solve_op(pos: Tensor, atom_mask: Tensor, graph: Tensor, n_graphs: int, cap: int,
                       send: Tensor, recv: Tensor, eta: Tensor, alpha_inv6: Tensor, beta: Tensor,
                       b_q: Tensor, b_s_ext: Tensor, b_p: Tensor, sigma: float, k_fix: int, triton_key: str = "") -> tuple[Tensor, Tensor]:
        with fp32_guard(pos.device, disable_tf32=True):
            return _pcg_solve(OpenMultipoleBackend(pos, atom_mask, graph, n_graphs, cap, sigma, triton_key=triton_key),
                              eta, alpha_inv6, beta, send, recv, -_grad_split(beta, b_q, send, recv) + b_s_ext, -b_p, k_fix)

    def _open_setup(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:3], *inputs[5:11], output[0], output[1])
        ctx.meta = (int(inputs[3]), int(inputs[4]), float(inputs[13]), int(inputs[14]), str(inputs[15]))

    def _open_bwd(ctx, gs: Tensor | None, gp: Tensor | None):
        pos, am, gr, send, recv, eta, a6, beta, b_q, s_star, p_star = ctx.saved_tensors
        ng, cap, sig, kf, tk = ctx.meta
        with fp32_guard(pos.device, disable_tf32=True):
            s_fn = lambda g_s, g_p: torch.ops.lam4.sqe_pqeq_open.default(pos, am, gr, ng, cap, send, recv, eta, a6, beta, beta.new_zeros(eta.shape), g_s, -g_p, sig, kf, tk)
            res, st = _adjoint_vjp(OpenMultipoleBackend(pos, am, gr, ng, cap, sig, triton_key=tk), eta, a6, beta, send, recv, b_q, s_star, p_star, _nz(gs, beta), _nz(gp, p_star), kf, s_fn)
            dp = _dh_dgeom(lambda p: OpenMultipoleBackend(p, am, gr, ng, cap, sig, triton_key=tk), *st, pos)[0] if ctx.needs_input_grad[0] else None
        return (dp,) + (None,) * 6 + tuple(r if ctx.needs_input_grad[i + 7] else None for i, r in enumerate(res)) + (None, None, None)

    @torch.library.custom_op("lam4::sqe_pqeq_spme", mutates_args=())
    def _spme_solve_op(pos: Tensor, cell: Tensor, graph: Tensor, atom_mask: Tensor, graph_mask: Tensor,
                       send: Tensor, recv: Tensor, eta: Tensor, alpha_inv6: Tensor, beta: Tensor,
                       b_q: Tensor, b_s_ext: Tensor, b_p: Tensor, sigma: float, m1: int, m2: int, m3: int, order: int, k_fix: int) -> tuple[Tensor, Tensor]:
        with fp32_guard(pos.device, disable_tf32=True):
            return _pcg_solve(BatchedSPME(pos, cell, graph, atom_mask, graph_mask, sigma, (m1, m2, m3), order),
                              eta, alpha_inv6, beta, send, recv, -_grad_split(beta, b_q, send, recv) + b_s_ext, -b_p, k_fix)

    def _spme_setup(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:11], output[0], output[1])
        ctx.meta = (float(inputs[13]), int(inputs[14]), int(inputs[15]), int(inputs[16]), int(inputs[17]), int(inputs[18]))

    def _spme_bwd(ctx, gs: Tensor | None, gp: Tensor | None):
        pos, cell, gr, am, gm, send, recv, eta, a6, beta, b_q, s_star, p_star = ctx.saved_tensors
        sig, m1, m2, m3, ord_, kf = ctx.meta
        with fp32_guard(pos.device, disable_tf32=True):
            s_fn = lambda g_s, g_p: torch.ops.lam4.sqe_pqeq_spme.default(pos, cell, gr, am, gm, send, recv, eta, a6, beta, beta.new_zeros(eta.shape), g_s, -g_p, sig, m1, m2, m3, ord_, kf)
            res, st = _adjoint_vjp(BatchedSPME(pos, cell, gr, am, gm, sig, (m1, m2, m3), ord_), eta, a6, beta, send, recv, b_q, s_star, p_star, _nz(gs, beta), _nz(gp, p_star), kf, s_fn)
            dp, dc = _dh_dgeom(lambda p, c: BatchedSPME(p, c, gr, am, gm, sig, (m1, m2, m3), ord_), *st, pos, cell) if (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]) else (None, None)
        return (dp if ctx.needs_input_grad[0] else None, dc if ctx.needs_input_grad[1] else None) + (None,) * 5 + tuple(r if ctx.needs_input_grad[i + 7] else None for i, r in enumerate(res)) + (None,) * 6

    _open_solve_op.register_fake(lambda *a: (torch.empty_like(a[9]), torch.empty_like(a[12])))
    _spme_solve_op.register_fake(lambda *a: (torch.empty_like(a[9]), torch.empty_like(a[12])))
    _open_solve_op.register_autograd(_open_bwd, setup_context=_open_setup)
    _spme_solve_op.register_autograd(_spme_bwd, setup_context=_spme_setup)


_register_custom_solve_ops()


class _ImplicitGenericSolveFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, be, eta: Tensor, a6: Tensor, beta: Tensor, send: Tensor, recv: Tensor, b_q: Tensor, b_s_ext: Tensor, b_p: Tensor, k_fix: int) -> tuple[Tensor, Tensor]:
        with torch.no_grad(): s_star, p_star = _pcg_solve(be, eta, a6, beta, send, recv, -_grad_split(beta, b_q, send, recv) + b_s_ext, -b_p, k_fix)
        ctx.be, ctx.k_fix = be, int(k_fix); ctx.save_for_backward(eta, a6, beta, send, recv, b_q, s_star, p_star)
        return s_star, p_star

    @staticmethod
    def backward(ctx, gs: Tensor | None, gp: Tensor | None):
        eta, a6, beta, send, recv, b_q, s_star, p_star = ctx.saved_tensors
        with torch.no_grad(): res, _ = _adjoint_vjp(ctx.be, eta, a6, beta, send, recv, b_q, s_star, p_star, torch.zeros_like(beta) if gs is None else gs, torch.zeros_like(p_star) if gp is None else gp, ctx.k_fix)
        return (None, *res[:3], None, None, *res[3:], None)


class SQEPQEqHead(nn.Module):
    """Variational scaled-SQE + anisotropic Thole PQEq head with Gram SPD floor and exact E_ext closure."""

    def __init__(self, d_model: int, k_vec: int, k_ten: int, *, n_lots: int = 16, sigma: float = SIGMA,
                 k_fix: int = 12, vec_c: float = 2.0, eta_min: float = 20.0, alpha_min_inv: float = 0.05, r_ct_on: float = 2.2, r_ct_cut: float = 4.5) -> None:
        super().__init__()
        self.sigma, self.k_fix, self.vec_c = float(sigma), int(k_fix), float(vec_c)
        self.eta_min, self.alpha_min_inv, self.r_ct_on, self.r_ct_cut = float(eta_min), float(alpha_min_inv), float(r_ct_on), float(r_ct_cut)
        self.k0, self.f1_0 = 2.0 / (self.sigma * math.sqrt(math.pi)), 4.0 / (3.0 * math.sqrt(math.pi) * (self.sigma ** 3))
        self.chi_head, self.eta_head, self.alpha_iso_head = (nn.Sequential(nn.Linear(d_model + 1, 64), nn.SiLU(), nn.Linear(64, 1)) for _ in range(3))
        self.lot_alpha_scale = nn.Parameter(torch.zeros(n_lots))
        self.p0_head, self.v_pol, self.t_pol = nn.Linear(k_vec, 1, bias=False), nn.Linear(k_vec, 2, bias=False), nn.Linear(k_ten, 1, bias=False)
        self.ct_edge_head = nn.Sequential(nn.Linear(d_model * 2 + 2, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.p0_head.weight); nn.init.constant_(self.alpha_iso_head[-1].bias, 0.0); nn.init.constant_(self.ct_edge_head[-1].bias, -1.5)

    def _alpha_inv6(self, sn: Tensor, q0: Tensor, u: Tensor, tb: Tensor, lot_atom: Tensor, mf: Tensor) -> Tensor:
        iso_ex, lot_s = F.softplus(self.alpha_iso_head(torch.cat([sn, q0.unsqueeze(-1)], -1)).squeeze(-1)), F.softplus(self.lot_alpha_scale.index_select(0, lot_atom))
        vp, (txx, tyy, tzz, txy, txz, tyz) = chan_linear(u, self.v_pol.weight), l2_mat6(chan_linear(tb, self.t_pol.weight).squeeze(-2))
        t_mat = torch.stack((torch.stack((txx, txy, txz), -1), torch.stack((txy, tyy, tyz), -1), torch.stack((txz, tyz, tzz), -1)), -2)
        l_mat = torch.cat([vp.transpose(-1, -2), t_mat], dim=-1)
        g, floor = lot_s.view(-1, 1, 1) * torch.bmm(l_mat, l_mat.transpose(-1, -2)), self.f1_0 + self.alpha_min_inv
        diag = floor + lot_s * (iso_ex + 1e-6)
        a6 = torch.stack((diag + g[:, 0, 0], diag + g[:, 1, 1], diag + g[:, 2, 2], g[:, 0, 1], g[:, 0, 2], g[:, 1, 2]), dim=-1)
        return torch.where(mf.bool().unsqueeze(-1), a6, a6.new_tensor([floor, floor, floor, 0.0, 0.0, 0.0]).unsqueeze(0))

    def _solve(self, backend, pos: Tensor, cell: Tensor | None, graph: Tensor, atom_mask: Tensor,
               send: Tensor, recv: Tensor, eta: Tensor, alpha_inv6: Tensor, beta: Tensor,
               b_q: Tensor, b_s_ext: Tensor, b_p: Tensor, implicit_pos: bool = False) -> tuple[Tensor, Tensor]:
        if isinstance(backend, OpenMultipoleBackend) and not implicit_pos:
            return torch.ops.lam4.sqe_pqeq_open.default(pos, atom_mask, graph, backend.n_graphs, backend.cap, send, recv, eta, alpha_inv6, beta, b_q, b_s_ext, b_p, self.sigma, self.k_fix, backend.triton_key)
        if isinstance(backend, BatchedSPME) and not implicit_pos:
            assert cell is not None
            return torch.ops.lam4.sqe_pqeq_spme.default(pos, cell, graph, atom_mask, backend.valid_g, send, recv, eta, alpha_inv6, beta, b_q, b_s_ext, b_p, self.sigma, *backend.grid_shape, backend.order, self.k_fix)
        return _ImplicitGenericSolveFn.apply(backend, eta, alpha_inv6, beta, send, recv, b_q, b_s_ext, b_p, self.k_fix)

    def forward(self, backend, pos: Tensor, cell: Tensor | None, graph: Tensor, atom_mask: Tensor,
                send: Tensor, recv: Tensor, cart_shift: Tensor, sn: Tensor, vec: Tensor, ten: Tensor,
                q0: Tensor, lot_id: Tensor, *, e_ext: Tensor | None = None, frag_com_pos: Tensor | None = None,
                compute_alpha: bool = False, n_graphs: int | None = None, implicit_pos: bool = False,
                return_residual: bool = False) -> dict[str, Tensor | None]:
        """Solve scaled SQE + PQEq and evaluate exact conservative energy, multipoles, and polarizability."""
        with fp32_guard(pos.device, disable_tf32=True):
            mf, q0f = atom_mask.to(pos.dtype), upcast_fp32(q0).to(pos.dtype) * atom_mask.to(pos.dtype)
            sq_in = torch.cat([upcast_fp32(sn), q0f.unsqueeze(-1)], -1)
            chi, eta_base = self.chi_head(sq_in).squeeze(-1) * mf, COULOMB_K * self.k0 + self.eta_min
            eta = (eta_base + F.softplus(self.eta_head(sq_in).squeeze(-1))) * mf + (1.0 - mf) * eta_base
            u, tb = bound_vectors(upcast_fp32(vec), self.vec_c), bound_vectors(upcast_fp32(ten), self.vec_c)
            p0 = chan_linear(u, self.p0_head.weight).squeeze(-2) * mf.unsqueeze(-1)
            alpha_inv6 = self._alpha_inv6(upcast_fp32(sn), q0f, u, tb, lot_id.index_select(0, graph), mf)
            sh = cart_shift.to(pos.dtype) if cart_shift.numel() > 0 else 0.0
            d_e = pos.index_select(0, send) - pos.index_select(0, recv) + sh
            r_e = torch.sqrt(d_e.pow(2).sum(-1) + 1e-12)
            both_real = atom_mask.index_select(0, recv) & atom_mask.index_select(0, send) & (r_e > 1e-6)
            _l0, _l1, _dn = self.ct_edge_head[0], self.ct_edge_head[2], sq_in.shape[-1]
            _hr, _hs = F.linear(sq_in, _l0.weight[:, :_dn], _l0.bias), F.linear(sq_in, _l0.weight[:, _dn:])
            if (not torch.is_grad_enabled()) and not (proxy_tracing() or torch.compiler.is_compiling() or is_fake(pos)):
                act = both_real & (r_e < self.r_ct_cut)
                send, recv, d_e, r_e = send[act], recv[act], d_e[act], r_e[act]
                both_real_f = 1.0
            else:
                both_real_f = both_real.to(pos.dtype)
            raw_beta = F.softplus(_l1(F.silu(_hr.index_select(0, recv) + _hs.index_select(0, send))).squeeze(-1)) * (c3_envelope(r_e, self.r_ct_on, self.r_ct_cut) * both_real_f).clamp_min(0.0)
            beta = torch.where(raw_beta <= 64.0, raw_beta, 64.0 + 64.0 * torch.tanh((raw_beta - 64.0).clamp_min(0.0) / 64.0))
            phi0, fld0, _ = backend.phi_field_multipole(q0f, p0, need_efg=False)
            b_q, b_p, r_ref = chi + eta * q0f + COULOMB_K * phi0, -COULOMB_K * fld0, (pos if frag_com_pos is None else (pos - frag_com_pos))
            e_at = (e_ext.index_select(0, graph) if e_ext.ndim == 2 else e_ext.unsqueeze(0).expand_as(pos)) if e_ext is not None else None
            b_s_ext = (-COULOMB_K * beta * (d_e * e_at.index_select(0, recv)).sum(-1)) if e_at is not None else beta.new_zeros(beta.shape)
            b_p = b_p - COULOMB_K * e_at * mf.unsqueeze(-1) if e_at is not None else b_p
            s_star, p_ind = self._solve(backend, pos, cell, graph, atom_mask, send, recv, eta, alpha_inv6, beta, b_q, b_s_ext, b_p, implicit_pos)
            sqe_res = sqe_rel = None
            if return_residual and (not proxy_tracing()):
                with torch.no_grad():
                    oq_r, op_r = _matvec_sym(backend, _div_split(beta, s_star, send, recv, pos.shape[0]), p_ind, eta, alpha_inv6)
                    rhs_s_r = -_grad_split(beta, b_q, send, recv) + b_s_ext
                    rs_r = (s_star + _grad_split(beta, oq_r, send, recv)) - rhs_s_r
                    sqe_res = torch.cat((rs_r.reshape(-1), (op_r + b_p).reshape(-1)))
                    sqe_rel = (sqe_res.double().norm() / torch.cat((rhs_s_r.reshape(-1), (-b_p).reshape(-1))).double().norm().clamp_min(1e-12)).to(pos.dtype)
            dq, p_ind = _div_split(beta, s_star, send, recv, pos.shape[0]) * mf, p_ind * mf.unsqueeze(-1)
            q_star, p_star = q0f + dq, p0 + p_ind
            phi_s, fld_s, efg_s = backend.phi_field_multipole(q_star, p_star, need_efg=True)
            e_perm_atom = (chi * q0f + 0.5 * eta * q0f.pow(2) + 0.5 * COULOMB_K * (q0f * phi0 - (p0 * fld0).sum(-1))) * mf
            s2_half = 0.25 * s_star.pow(2)
            e_edge_atom = pos.new_zeros(pos.shape[0]).index_add(0, recv, s2_half).index_add(0, send, s2_half) * mf
            e_ind_atom = (chi * dq + 0.5 * eta * (q_star.pow(2) - q0f.pow(2)) + 0.5 * COULOMB_K * ((q_star * phi_s - q0f * phi0) - ((p_star * fld_s).sum(-1) - (p0 * fld0).sum(-1)) + (p_ind * _apply_sym6(alpha_inv6, p_ind)).sum(-1))) * mf + e_edge_atom
            e_field_atom = pos.new_zeros(pos.shape[0])
            if e_at is not None:
                e_flow_f = -COULOMB_K * (beta * s_star) * (-d_e * e_at.index_select(0, recv)).sum(-1)
                e_field_atom = (-COULOMB_K * ((q0f.unsqueeze(-1) * r_ref + p_star) * e_at).sum(-1) + e_field_atom.index_add(0, recv, 0.5 * e_flow_f).index_add(0, send, 0.5 * e_flow_f)) * mf
            alpha_t = self._solve_polarizability(backend, pos, cell, atom_mask, d_e, graph, eta, alpha_inv6, beta, send, recv, mf, n_graphs, implicit_pos) if compute_alpha else None
            return {"energy_atom": e_perm_atom + e_ind_atom + e_field_atom, "e_perm_atom": e_perm_atom, "e_ind_atom": e_ind_atom, "q0": q0f,
                    "alpha_inv6": alpha_inv6, "charges": q_star, "delta_q_ct": dq, "atomic_dipoles": p_star, "atomic_dipoles_perm": p0,
                    "atomic_dipoles_ind": p_ind, "phi": phi_s, "field": fld_s, "efg_l2": efg_s, "alpha_tensor": alpha_t,
                    "sqe_residual": sqe_res, "sqe_rel_residual": sqe_rel}

    def _solve_polarizability(self, backend, pos: Tensor, cell: Tensor | None, atom_mask: Tensor, d_e: Tensor, graph: Tensor,
                              eta: Tensor, alpha_inv6: Tensor, beta: Tensor, send: Tensor, recv: Tensor, mf: Tensor,
                              n_graphs: int | None, implicit_pos: bool) -> Tensor:
        n_g, eye3, zero_q, cols = (int(n_graphs) if n_graphs is not None else getattr(backend, "n_graphs", 1)), torch.eye(3, dtype=pos.dtype, device=pos.device), beta.new_zeros(eta.shape), []
        for ax in range(3):
            xs, xp = self._solve(backend, pos, cell, graph, atom_mask, send, recv, eta, alpha_inv6, beta, zero_q, -beta * d_e[:, ax], -eye3[ax].unsqueeze(0).expand_as(pos) * mf.unsqueeze(-1), implicit_pos)
            mu_s = 0.5 * (-d_e) * (beta * xs).unsqueeze(-1)
            cols.append(xp.new_zeros(n_g, 3).index_add(0, graph, xp.index_add(0, recv, mu_s).index_add(0, send, mu_s) * mf.unsqueeze(-1)))
        alpha_raw = torch.stack(cols, dim=-1) * COULOMB_K
        return 0.5 * (alpha_raw + alpha_raw.transpose(-1, -2))
