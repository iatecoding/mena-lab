"""Per-edge math shared by the oracle (edge_ops.edge_aggregate_ref) and the chunked stream (edge_stream.py).

Edge (receiver i, sender j), exactly as earlier Mena (geometry.py + trunk.py): d = x_j - x_i + shift, r = |d|, u = d / r, chi = env(r) on the UNWARPED
distance, x = warp(r) (radial.py), tab = R(x) = Phi(x) W (B-spline table of Gaussians at the warped distance, NO envelope inside; layout.py
column order: HEADS biases then filters). With sender gates g (gates [N, 184] = s | v | t | o), the message is multilinear in (radial profile)
x (sender gate / vector) x (angular), no activation on the edge, and carries the factor chi:

    scalar   chi f_s * g_s
    l=1      chi (f_dir * g_v [+ f_lon * lon]) (x) u [+ chi f_tr * b]              LATER only: b = vec_j [32, 3], lon = <b, u> per channel
    lon      chi f_lon * lon                                                       LATER only (a scalar output)
    l=2      chi (f_ten * g_t [+ f_lon2 * lon2]) (x) Y2(u) [+ chi f_tr2 * tau]     LATER only: tau = ten_j [16, 5], lon2 = <tau, Y2(u)> per channel
    lon2     chi f_lon2 * lon2                                                     LATER only (a scalar output)
    l=3      chi (f3 * g_o) (x) Y3(u)

multiplied by a per-channel head weight: head h scales the h-th contiguous quarter of the channels of every group. Default head
weight exp(logit^h), logit^h = <qn_i^h, kn_j^h> + 2 tanh(tab_bias^h / 2) (qn, kn unit norm per head, so |logit| <= 3, no running max).
COMPAT mode (edge_scale given): every head weight is the per-edge number edge_scale (the earlier Mena trunk's 1 + 0.49 (mod - mean_mod)) and the normaliser sees
no modulation. In both modes the edge carries chi * (head weight): chi is in the message, in the degree sum D and in Z.

finalise: out_i = N_i * [(D_i + 1) / (Z_i^h + 1)] * rsqrt(D_i + 1) = N_i sqrt(D_i + 1) / (Z_i^h + 1), with N = sum msg, D = sum chi,
Z^h = sum chi exp(logit^h) (the +1 is a virtual self neighbour): the chi-weighted mean of exp(logit) is normalised to 1, then the earlier Mena degree
normaliser 1 / sqrt(D + 1) is applied. At logit 0 (zero-init q, k, bias columns) Z = D and the op is exactly the earlier Mena
sum_j chi_ij filt_ij gate_j Y(u_ij) / sqrt(sum_j chi_ij + 1). In compat mode Z := D, so out = N / sqrt(D + 1).

Padding contract: an edge at r = r_cut has chi = chi' = 0 exactly, so msg, Z, D and every gradient through the edge (positions, gates, vec, q, k,
table, rho) are exactly 0 whatever the (finite) table values there.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lam4.core.consts import storable
from lam4.core.harmonics import y2, y3
from lam4.core.layout import HEADS, StageLayout
from lam4.core.radial import N_RBF, R_CUT, R_ON, c3_envelope, warp_distance
from lam4.core.spline_tables import bspline_eval, default_bspline_basis
from lam4.physics.repulsion import floored_length

_ORIG_BSPLINE_EVAL = bspline_eval


def edge_basis(x: Tensor, r_cut: float = R_CUT) -> Tensor:
    if bspline_eval is _ORIG_BSPLINE_EVAL:
        return default_bspline_basis(x, r_cut)
    return bspline_eval(x, torch.eye(N_RBF, dtype=x.dtype, device=x.device), r_cut)

QK_EPS = 1e-6
BIAS_BOUND = 2.0  # |radial logit bias| <= 2, so |logit| <= 3: exp needs no running max and the +1 self term cannot dominate
_NEAR_BASE: dict = {}
_NEAR_MASK: dict = {}


@dataclass(frozen=True)
class EdgeCfg:
    r_cut: float = R_CUT
    r_on: float = R_ON
    floor: bool = False  # 5 eV NLH descriptor floor on the distance that chi and the warp read (needs z; Stage sets it from ModelConfig.descriptor_floor)
    r_cut_near: float = 0.0
    r_on_near: float = 3.0
    k_s_far: int = 32
    k_v_far: int = 16


def _near_base(layout: StageLayout, k_s_far: int, k_v_far: int) -> Tensor:
    key = (layout.name, layout.filt_seg, k_s_far, k_v_far)
    b = _NEAR_BASE.get(key)
    if b is None:
        far = {"s": k_s_far, "dir": k_v_far, "tr": k_v_far, "lon": k_v_far}
        vals = [v for n, k in zip(layout.filt_names, layout.filt_seg)
                for _ in range(HEADS) for v in (0.0,) * (far.get(n, 0) // HEADS) + (1.0,) * ((k - far.get(n, 0)) // HEADS)]
        b = torch.tensor(vals, dtype=torch.float64)
        if storable(b):
            _NEAR_BASE[key] = b
    return b


def _near_mask(layout: StageLayout, cfg: EdgeCfg, like: Tensor) -> Tensor:
    key = (layout.name, layout.filt_seg, cfg.k_s_far, cfg.k_v_far, like.dtype, like.device)
    m = _NEAR_MASK.get(key)
    if m is None:
        m = _near_base(layout, cfg.k_s_far, cfg.k_v_far).to(dtype=like.dtype).to(device=like.device)
        if storable(m):
            _NEAR_MASK[key] = m
    return m


def normalise_qk(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """q, k [N, H*Dh] -> unit-norm per head [N, H, Dh] (RMS-normalised then / sqrt(Dh)); <qn, kn> in [-1, 1]. Zero in, zero out."""
    n, hd = q.shape
    dh = hd // HEADS

    def unit(x: Tensor) -> Tensor:
        x = x.reshape(n, HEADS, dh)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + QK_EPS) / dh ** 0.5
    return unit(q), unit(k)


def head_scale(x: Tensor, w: Tensor, seg: tuple[int, ...]) -> Tensor:
    """x [..., sum(seg)] times w [..., H], head h on the h-th contiguous quarter of every segment."""
    outs, o = [], 0
    for n in seg:
        blk = x[..., o:o + n].reshape(*x.shape[:-1], HEADS, n // HEADS) * w.unsqueeze(-1)
        outs.append(blk.reshape(*x.shape[:-1], n))
        o += n
    return torch.cat(outs, dim=-1)


def _per_channel(ex: Tensor, k: int) -> Tensor:
    """Head weights [e, H] -> one per channel [e, k] (head h on the h-th contiguous quarter)."""
    return ex.unsqueeze(-1).expand(ex.shape[0], HEADS, k // HEADS).reshape(ex.shape[0], k)


def _dot3(b: Tensor, u: Tensor) -> Tensor:
    """sum_c b[e, k, c] u[e, c] as 3 multiply-adds (a reduction over the size-3 axis is pathologically slow on MPS)."""
    return b[..., 0] * u[:, 0:1] + b[..., 1] * u[:, 1:2] + b[..., 2] * u[:, 2:3]


def _dot5(tau: Tensor, y: Tensor) -> Tensor:
    """sum_m tau[e, k, m] y[e, m] as 5 multiply-adds (a reduction over the size-5 axis is slow on MPS)."""
    return (tau[..., 0] * y[:, 0:1] + tau[..., 1] * y[:, 1:2] + tau[..., 2] * y[:, 2:3]
            + tau[..., 3] * y[:, 3:4] + tau[..., 4] * y[:, 4:5])


def _edge_r_u_x(d: Tensor, rho: Tensor | None, cfg: EdgeCfg, z_recv: Tensor | None = None, z_send: Tensor | None = None
                ) -> tuple[Tensor, Tensor, Tensor]:
    r = d.pow(2).sum(-1).sqrt()
    u = d / r.unsqueeze(-1)
    if cfg.floor and z_recv is not None:
        r = floored_length(d, z_recv, z_send, r)
    return r, u, warp_distance(r, rho, cfg.r_cut)


def edge_geometry(d: Tensor, rho: Tensor | None, cfg: EdgeCfg, z_recv: Tensor | None = None, z_send: Tensor | None = None
                  ) -> tuple[Tensor, Tensor, Tensor]:
    """Edge vectors d [e, 3] -> (u [e, 3], x [e], chi [e]): unit vector, warped distance (Gaussian argument), envelope on the raw r.

    DESCRIPTOR FLOOR (earlier Mena geometry.py: `dist = zbl_clamp(dist_raw)`; here the 5 eV NLH tanh floor of lam4.physics.repulsion instead of the earlier Mena trunk's ZBL clamp):
    with cfg.floor and the species of both ends (z_recv, z_send [e] long; None = no floor), chi and the warp read the FLOORED length r~ = |d~|, while
    the direction u stays that of the raw d (the earlier Mena trunk: `unit = diff / dist_raw`). Edges above their floor are untouched (r~ = |d| exactly), pad edges (Z = 0) too."""
    r, u, x = _edge_r_u_x(d, rho, cfg, z_recv, z_send)
    return u, x, c3_envelope(r, cfg.r_on, cfg.r_cut)


def build_edge_geom(pos: Tensor, recv: Tensor, send: Tensor, shift: Tensor, rho: Tensor | None, cfg: EdgeCfg,
                    z: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Precompute stage-invariant per-edge geometry (u, x, chi, y2(u), y3(u)) once across all stages in `ref` mode."""
    d = pos.index_select(0, send) - pos.index_select(0, recv) + shift
    rho_e = None if rho is None else rho.index_select(0, recv) + rho.index_select(0, send)
    zz = (None, None) if z is None else (z.index_select(0, recv), z.index_select(0, send))
    r, u, x = _edge_r_u_x(d, rho_e, cfg, *zz)
    chi = c3_envelope(r, cfg.r_on, cfg.r_cut)
    if cfg.r_cut_near > 0.0:
        chi = torch.stack((chi, c3_envelope(r, cfg.r_on_near, cfg.r_cut_near)), dim=-1)
    return u, x, chi, y2(u), y3(u)


def edge_terms(pos_r: Tensor, pos_s: Tensor, recv: Tensor, send: Tensor, shift: Tensor, coef: Tensor, gates_s: Tensor,
               vec_s: Tensor | None, qn_r: Tensor, kn_s: Tensor, rho_r: Tensor | None, rho_s: Tensor | None,
               esc: Tensor | None, layout: StageLayout, cfg: EdgeCfg, z_r: Tensor | None = None, z_s: Tensor | None = None,
               ten_s: Tensor | None = None, geom: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None,
               phi: Tensor | None = None
               ) -> tuple[Tensor, Tensor, Tensor]:
    """Unnormalised per-edge terms: (msg [e, out_dim] incl. chi and head weight, chi * head weight [e, H], chi [e]).

    The node tensors are the receiver set (pos_r, qn_r, rho_r) and the sender set (pos_s, gates_s, vec_s, ten_s, kn_s, rho_s); `recv` / `send`
    index into them (both are the full node arrays in the oracle, a receiver range and the unique chunk senders in the stream).
    vec_s [., 32, 3] is required for LATER layouts and ignored for FIRST; ten_s [., 16, 5] is optional for LATER; rho_* are PairWarp.node_rho or None;
    esc [e] or None; z_r, z_s: atomic numbers of the receiver / sender sets (long) for the descriptor floor, or None.
    """
    chi_near = None
    if geom is None:
        d = pos_s.index_select(0, send) - pos_r.index_select(0, recv) + shift
        rho = None if rho_r is None else rho_r.index_select(0, recv) + rho_s.index_select(0, send)
        zz = (None, None) if z_r is None else (z_r.index_select(0, recv), z_s.index_select(0, send))
        r, u, x = _edge_r_u_x(d, rho, cfg, *zz)
        chi = c3_envelope(r, cfg.r_on, cfg.r_cut)
        if cfg.r_cut_near > 0.0:
            chi_near = c3_envelope(r, cfg.r_on_near, cfg.r_cut_near)
        y2_u, y3_u = y2(u), y3(u)
    else:
        u, x, chi, y2_u, y3_u = geom
        if chi.ndim == 2:
            chi, chi_near = chi.unbind(-1)
        elif cfg.r_cut_near > 0.0:
            d = pos_s.index_select(0, send) - pos_r.index_select(0, recv) + shift
            zz = (None, None) if z_r is None else (z_r.index_select(0, recv), z_s.index_select(0, send))
            r, _, _ = _edge_r_u_x(d, None, cfg, *zz)
            chi_near = c3_envelope(r, cfg.r_on_near, cfg.r_cut_near)
    tab = (phi @ coef) if (phi is not None and coef.ndim == 2 and coef.shape[0] == N_RBF) else bspline_eval(x, coef, cfg.r_cut)
    if esc is None:
        bias = BIAS_BOUND * torch.tanh(tab[:, :HEADS] / BIAS_BOUND)
        ex = torch.exp((qn_r.index_select(0, recv) * kn_s.index_select(0, send)).sum(-1) + bias)  # [e, H]
        zc = w = chi.unsqueeze(-1) * ex
    else:
        w = chi.unsqueeze(-1) * esc.unsqueeze(-1)
        zc = chi.unsqueeze(-1).expand(-1, HEADS)
    filt = tab[:, HEADS:]
    if chi_near is not None:
        m = _near_mask(layout, cfg, filt)
        filt = filt * (1.0 + (chi_near.unsqueeze(-1) - 1.0) * m)
    f = {n: filt[:, s] for n, s in layout.filt_slices().items()}
    gs = gates_s.index_select(0, send)
    g = {n: gs[:, s] for n, s in layout.gate_slices().items()}
    hw = lambda k: _per_channel(w, k)  # noqa: E731  (head weight times chi)
    e = u.shape[0]
    parts = [f["s"] * g["s"] * hw(layout.k_s)]
    w_dir = f["dir"] * g["v"]
    w_ten = f["ten"] * g["t"]
    if layout.later:
        b = vec_s.index_select(0, send)  # [e, 32, 3]
        w_lon = f["lon"] * _dot3(b, u)
        vmsg = ((w_dir + w_lon) * hw(layout.k_v)).unsqueeze(-1) * u.unsqueeze(1) + (f["tr"] * hw(layout.k_v)).unsqueeze(-1) * b
        if ten_s is not None and ten_s.numel() > 0:
            tau = ten_s.index_select(0, send)  # [e, 16, 5]
            w_lon2 = f["lon2"] * _dot5(tau, y2_u)
            tmsg = ((w_ten + w_lon2) * hw(layout.k_t)).unsqueeze(-1) * y2_u.unsqueeze(1) + (f["tr2"] * hw(layout.k_t)).unsqueeze(-1) * tau
        else:
            w_lon2 = w_ten.new_zeros(e, layout.k_lon2)
            tmsg = (w_ten * hw(layout.k_t)).unsqueeze(-1) * y2_u.unsqueeze(1)
        parts += [vmsg.reshape(e, -1), w_lon * hw(layout.k_lon), tmsg.reshape(e, -1), w_lon2 * hw(layout.k_lon2)]
    else:
        parts.append(((w_dir * hw(layout.k_v)).unsqueeze(-1) * u.unsqueeze(1)).reshape(e, -1))
        parts.append(((w_ten * hw(layout.k_t)).unsqueeze(-1) * y2_u.unsqueeze(1)).reshape(e, -1))
    parts.append(((f["o"] * g["o"] * hw(layout.k_o)).unsqueeze(-1) * y3_u.unsqueeze(1)).reshape(e, -1))
    return torch.cat(parts, dim=-1), zc, chi


def finalise(num: Tensor, z: Tensor, dsum: Tensor, layout: StageLayout) -> Tensor:
    """out = num * sqrt(D + 1) / (Z^h + 1) per head: head-weight normalisation (Z) times the degree normaliser 1 / sqrt(D + 1)."""
    return head_scale(num, torch.sqrt(dsum.unsqueeze(-1) + 1.0) / (z + 1.0), layout.out_seg)
