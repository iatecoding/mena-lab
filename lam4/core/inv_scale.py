"""Continuous coordination-dependent invariant scaling, dsum geometry & calibration for LAM-6.

Design C (`#24`, `#27`, `#37`):
    scale_i = exp(log_scale_0 + alpha * log1p(dsum_i))
    alpha = alpha_max * tanh(alpha_raw) in (-alpha_max, alpha_max) if signed else alpha_max * sigmoid(alpha_raw) in (0, alpha_max)
    inv_scaled_i = all_inv_i * exp(-(log_scale_0 + alpha * log1p(dsum_i)))

where `dsum_i = sum_j chi_ij >= 0` is the strictly non-negative unweighted C3 cutoff sum (independent of signed edge
attention gains), differentiable w.r.t. positions so conservative forces remain exact, and `log_scale_0` / `alpha_raw`
are 1D parameters (`ndim == 1`) routed automatically to AdamW rather than BatchedMuon (`lam4/train/optim.py:24`).
"""
from __future__ import annotations

from dataclasses import replace
import itertools
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.cond.env_mod import suppress_env_remat
from lam4.config import ModelConfig
from lam4.core.angular import l2_mat6, l2_matvec, l2_pair_mm
from lam4.kernels.metal_node import pack_node_weights
from lam4.core.edge_terms import EdgeCfg, finalise
from lam4.core.layout import HEADS, StageLayout
from lam4.core.radial import c3_envelope, fp32_guard, upcast_fp32
from lam4.core.remat import NODE_CHUNK_ATOMS, RematSpec, chunked_node_apply, node_chunk_size, remat_apply
from lam4.core.spline_tables import warm_spline_windows
from lam4.core.stage_graph import EdgeGraph, NodeContext
from lam4.core.stage_remat import _unit_a_ext, pack, post_chunked_crystal, unit_block, unit_finish, unit_stage
from lam4.core.vector_bound import bound_vectors, chan_linear, mix
from lam4.kernels.csr_plan import coef_to_horner_poly
from lam4.physics.repulsion import floored_length


def default_inv_scale(cfg: ModelConfig, layout: StageLayout) -> Tensor:
    """Initial static per-invariant scale vector matching lam5 `_default_inv_scale`."""
    segs = [(cfg.r1_vec + cfg.r2_ten, 1e-3), (cfg.r3_tvv, 1e-5)]
    if layout.later:
        segs.append((layout.k_lon2 + layout.k_v + layout.k_t + layout.k_lon, 1e-2))
    segs += [(cfg.ttt_mix, 1e-5), (cfg.vttv_rank, 1e-6), (cfg.k3_read, 1e-3), (cfg.k3_read, 1e-5)]
    return torch.cat([torch.full((n,), val, dtype=torch.float32) for n, val in segs])


class CoordInvScale(nn.Module):
    """Learnable continuous per-atom invariant scaling via `log(1 + dsum_i)`."""

    def __init__(self, inv_dim: int, init_scale: Tensor, alpha_max: float = 2.5, signed: bool = False) -> None:
        super().__init__()
        self.inv_dim = int(inv_dim)
        self.alpha_max = float(alpha_max)
        self.signed = bool(signed)
        self.log_scale_0 = nn.Parameter(init_scale.clamp_min(1e-12).log().to(torch.float32))
        init_sig = min(0.9, max(0.05, 0.25 / self.alpha_max))
        init_raw = math.atanh(init_sig) if self.signed else math.log(init_sig / (1.0 - init_sig))
        self.alpha_raw = nn.Parameter(torch.full((self.inv_dim,), init_raw, dtype=torch.float32))

    def alpha(self) -> Tensor:
        if self.signed:
            return self.alpha_max * torch.tanh(self.alpha_raw)
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    def log_scale(self, dsum: Tensor) -> Tensor:
        dt = torch.float64 if (dsum.dtype == torch.float64 or self.log_scale_0.dtype == torch.float64) else torch.float32
        ld = torch.log1p(dsum.to(dt).clamp_min(0.0)).unsqueeze(-1)
        return self.log_scale_0.to(dt) + self.alpha().to(dt) * ld

    def multiplier(self, dsum: Tensor) -> Tensor:
        """Return `exp(-(log_scale_0 + alpha * log1p(dsum_i)))` for multiplying `all_inv`."""
        return torch.exp(-self.log_scale(dsum))

    def scale(self, dsum: Tensor) -> Tensor:
        """Return `exp(log_scale_0 + alpha * log1p(dsum_i))` (`all_inv / scale == all_inv * multiplier`)."""
        return torch.exp(self.log_scale(dsum))


def compute_dsum(eg: EdgeGraph, cfg: EdgeCfg) -> Tensor:
    """Strictly non-negative unweighted cutoff sum `dsum_i = sum_j chi_ij >= 0`, differentiable wrt `eg.pos` & `eg.cell`."""
    pos = upcast_fp32(eg.pos)
    n = pos.shape[0]
    with fp32_guard(pos.device):
        if eg.geom is not None:
            chi = upcast_fp32(eg.geom[2])
            chi_main = chi[..., 0] if chi.ndim == 2 else chi
            if not isinstance(eg.recv.shape[0], torch.SymInt) and eg.recv.numel() == 0:
                return pos.new_zeros(n) + ((pos.sum() * 0.0) if pos.requires_grad else 0.0)
            return chi_main.new_zeros(n).index_add(0, eg.recv, chi_main)
        if not isinstance(eg.recv.shape[0], torch.SymInt) and eg.recv.numel() == 0:
            return pos.new_zeros(n) + ((pos.sum() * 0.0) if pos.requires_grad else 0.0)
        if eg.shift.ndim == 2 and eg.shift.shape[0] == eg.recv.shape[0]:
            sh = upcast_fp32(eg.shift)
        elif eg.edge_shift is not None and eg.cell is not None and eg.cell.ndim == 3:
            cell = upcast_fp32(eg.cell)
            es = upcast_fp32(eg.edge_shift)
            sh = es @ cell[0] if cell.shape[0] == 1 else torch.einsum("ek,ekl->el", es, cell[eg.graph[eg.recv]])
        else:
            sh = pos.new_zeros(eg.recv.shape[0], 3)
        d = pos.index_select(0, eg.send) - pos.index_select(0, eg.recv) + sh
        r = d.pow(2).sum(-1).sqrt()
        if cfg.floor and eg.z is not None:
            r = floored_length(d, eg.z.index_select(0, eg.recv), eg.z.index_select(0, eg.send), r)
        chi = c3_envelope(r, cfg.r_on, cfg.r_cut)
        return chi.new_zeros(n).index_add(0, eg.recv, chi)


def build_stage_baked(stage) -> dict[str, Tensor]:
    """Pack fused inference tensors for `Stage.bake_for_inference()`."""
    hd = stage.heads.q_proj.out_features
    w_tab = stage.tables.weight
    if not w_tab.is_meta:
        warm_spline_windows(stage.tables.r_cut, w_tab.dtype, w_tab.device)
        stage.tables.solve_matrix(w_tab.dtype, w_tab.device)
        if w_tab.device.type != "mps":
            stage.tables.solve_matrix(torch.float64, w_tab.device)
    coef64 = stage.tables._p64.detach().cpu().to(torch.float64) @ w_tab.detach().cpu().to(torch.float64)
    bk = {
        "w_gqk": torch.cat([stage.gate_proj.weight, stage.heads.q_proj.weight, stage.heads.k_proj.weight], 0).detach().contiguous(),
        "b_gqk": torch.nn.functional.pad(stage.gate_proj.bias, (0, 2 * hd)).detach().contiguous(),
        "gain": stage.heads.gain().detach().contiguous(),
        "w_uvpq": torch.cat([stage.u1.weight, stage.v1.weight, stage.p3.weight, stage.q3.weight], 0).detach().contiguous(),
        "w_uvt": torch.cat([stage.u2.weight, stage.v2.weight, stage.t3.weight], 0).detach().contiguous(),
        "poly": coef_to_horner_poly(coef64, stage.edge_cfg.r_cut, precise=True).to(dtype=w_tab.dtype, device=w_tab.device).contiguous(),
    }
    if stage.inv_norm:
        if not stage.coord_inv_scale:
            w0 = stage.inv_mlp[0].weight
            bk["w_inv0"] = (w0 / stage.inv_scale.to(w0.dtype).unsqueeze(0)).detach().contiguous()
            bk["b_inv0"] = stage.inv_mlp[0].bias.detach().contiguous()
        bk["w_gates4"] = torch.cat([stage.vec_out_gate.weight, stage.tv_gate.weight, stage.ten_out_gate.weight, stage.vv_gate.weight], 0).detach().contiguous()
        bk["b_gates4"] = torch.cat([stage.vec_out_gate.bias, stage.tv_gate.bias, stage.ten_out_gate.bias, stage.vv_gate.bias], 0).detach().contiguous()
        bk["w_u_out"] = torch.cat([stage.vec_self.weight, stage.tv_v.weight, stage.vv_p.weight, stage.vv_q.weight], 0).detach().contiguous()
        bk["w_tb"] = torch.cat([stage.ten_self.weight, stage.tv_t.weight], 0).detach().contiguous()
        if w_tab.device.type in ("mps", "cuda") and not w_tab.is_meta:
            packs = pack_node_weights(stage)   # fused MPS / CUDA node kernels (lam4/kernels/{metal,cuda}_node.py)
            if packs is not None:
                bk["node_wa"], bk["node_wb"] = packs
            if getattr(stage, "env_modulated", False):
                bk["env_out_wt"] = stage.swiglu.env_out.weight.detach().float().t().contiguous()
    if w_tab.device.type == "cuda" and not w_tab.is_meta:
        bk["poly4"] = bk["poly"].permute(0, 2, 1).contiguous()
        bk["_poly_ver"] = int(bk["poly"]._version)
    return bk


def post_chunked_stage(
    stage, s: Tensor, vec: Tensor, agg: Tensor | tuple[Tensor, ...], domain: str, ten: Tensor | None,
    cond_lot_atom: Tensor | None, dsum: Tensor | None, *, return_ten: bool, return_inv: bool,
    raw: bool = False, nc: NodeContext | None = None, fin: bool = False,
    chunk_atoms: int | None = None, create_graph: bool = False,
):
    """Chunked node-side stage post-processing for eager / block-remat large-atom inference."""
    ps = tuple(stage.parameters())
    agg_t = agg if isinstance(agg, tuple) else (agg,)
    n_agg = len(agg_t)
    t_eff = ten if stage.layout.later else None
    flags = (t_eff is not None, cond_lot_atom is not None, dsum is not None)
    sliced: list[Tensor] = [s, vec, *agg_t]
    if t_eff is not None:
        sliced.append(t_eff)
    if cond_lot_atom is not None:
        sliced.append(cond_lot_atom)
    if dsum is not None:
        sliced.append(dsum)
    ca = NODE_CHUNK_ATOMS if chunk_atoms is None else int(chunk_atoms)

    def _fn(a0: int, a1: int, *xs):
        agg_c = finalise(*xs[2:2 + n_agg], stage.layout) if raw else xs[2]
        it = iter(xs[2 + n_agg:])
        t_ = next(it) if flags[0] else None
        cl_ = next(it) if flags[1] else None
        ds_ = next(it) if flags[2] else None
        with suppress_env_remat():
            res = stage._post_impl(xs[0], xs[1], agg_c, domain, t_, return_ten=return_ten, cond_lot_atom=cl_, dsum=ds_, return_inv=return_inv, allow_fused=not create_graph)
        if not fin:
            return res
        nc_s = None if nc is None else replace(nc, atom_mask=None if nc.atom_mask is None else nc.atom_mask[a0:a1], graph=None)
        return (stage.finish_plain(res[0], nc_s, domain), *res[1:])

    return chunked_node_apply(_fn, tuple(sliced), params=ps, chunk_atoms=ca, create_graph=create_graph)


@torch.no_grad()
def calibrate_coord_inv_scale(model: nn.Module, batches: Iterable, max_batches: int = 8, eps: float = 1e-12) -> list[tuple[Tensor, Tensor]]:
    """Pre-step-0 OLS fit of `(log_scale_0, alpha)` per stage on `log|inv| ~ log_scale_0 + alpha * log1p(dsum)`."""
    from lam4.core.stage import Stage
    from lam4.core.stage_graph import EdgeGraph

    stages = [m for m in model.modules() if isinstance(m, Stage) and m.inv_norm and m.coord_inv_scale]
    if not stages:
        return []
    dim_info = [st.coord_scale.inv_dim for st in stages]
    acc = [dict(n=0.0, sx=0.0, sx2=0.0, sy=torch.zeros(d, dtype=torch.float64),
                sxy=torch.zeros(d, dtype=torch.float64), sy2=torch.zeros(d, dtype=torch.float64)) for d in dim_info]
    cur: dict[str, Tensor | None] = {"mask": None, "ld": None}

    def make_hook(idx: int):
        def hook(all_inv: Tensor) -> None:
            m, ld = cur["mask"], cur["ld"]
            if ld is None:
                return
            x = all_inv if m is None else all_inv[m]
            lx = ld if m is None else ld[m]
            if x.numel() == 0:
                return
            xf = x.detach().to(torch.float64)
            lf = lx.detach().to(torch.float64)
            # Fit on log(sqrt(x^2 + eps^2)) so scale captures typical RMS amplitude smoothly vs log1p(dsum).
            ly = 0.5 * torch.log(xf.pow(2) + eps * eps)
            a = acc[idx]
            a["n"] += float(lf.shape[0])
            a["sx"] += float(lf.sum())
            a["sx2"] += float(lf.pow(2).sum())
            a["sy"] += ly.sum(0).cpu()
            a["sxy"] += (lf.unsqueeze(-1) * ly).sum(0).cpu()
            a["sy2"] += ly.pow(2).sum(0).cpu()
        return hook

    was_training = model.training
    model.eval()
    try:
        for i, st in enumerate(stages):
            st._inv_hook = make_hook(i)
        for b in itertools.islice(batches, int(max_batches)):
            cur["mask"] = getattr(b, "atom_mask", None)
            eg = EdgeGraph.from_batch(b, mode="ref")
            cur["ld"] = torch.log1p(compute_dsum(eg, stages[0].edge_cfg).clamp_min(0.0))
            try:
                model(b, compute_forces=False)
            except TypeError:
                model(b, b.pos, b.domain)
    finally:
        for st in stages:
            st._inv_hook = None
        if was_training:
            model.train()

    out: list[tuple[Tensor, Tensor]] = []
    for st, a in zip(stages, acc):
        cs = st.coord_scale
        if a["n"] > 1.0:
            n = a["n"]
            mx = a["sx"] / n
            vx = max(a["sx2"] / n - mx * mx, 1e-8)
            my = a["sy"] / n
            cov = a["sxy"] / n - mx * my
            lo = min(0.02, 0.5 * cs.alpha_max)
            hi = cs.alpha_max - lo
            alpha = (cov / vx).clamp(-hi if cs.signed else lo, hi)
            # Shift log_scale_0 upward slightly by 0.5*E[log^2] residual variance so RMS ~ 1 rather than exp(E[log]) < 1.
            var_res = (a["sy2"] / n - my.pow(2) - (cov.pow(2) / vx)).clamp_min(0.0)
            l0 = my - alpha * mx + 0.5 * var_res
            if cs.signed:
                raw = torch.atanh((alpha / cs.alpha_max).clamp(-1.0 + 1e-3, 1.0 - 1e-3))
            else:
                sig = (alpha / cs.alpha_max).clamp(1e-3, 1.0 - 1e-3)
                raw = torch.log(sig / (1.0 - sig))
            cs.log_scale_0.copy_(l0.to(dtype=cs.log_scale_0.dtype, device=cs.log_scale_0.device))
            cs.alpha_raw.copy_(raw.to(dtype=cs.alpha_raw.dtype, device=cs.alpha_raw.device))
        out.append((cs.log_scale_0.detach().clone(), cs.alpha().detach().clone()))
    return out


def post_vector_tensor_update(st, s: Tensor, v_new: Tensor, u_new: Tensor, t_new: Tensor, ten: Tensor | None,
                              bk: dict[str, Tensor] | None, v_dt: torch.dtype, return_ten: bool,
                              scaled_inv: Tensor | None, return_inv: bool):
    """Steps 9a-9b of Stage: gated vector and tensor stream updates (baked or unbaked)."""
    L = st.layout
    s_gn = st.vec_gate_norm(s)
    if bk is not None and st.inv_norm:
        kv, kt2 = L.k_v, L.k_t
        gv, gtv, gt, gvv = torch.tanh(F.linear(s_gn, bk["w_gates4"], bk["b_gates4"]).to(v_dt)).split([kv, kv, kt2, kt2], dim=-1)
        with fp32_guard(s.device):
            tb = bound_vectors(t_new, st.vec_c)
            if not return_ten and ten is None:
                u_self, u_tvv = chan_linear(u_new, bk["w_u_out"][:2 * kv]).split([kv, kv], dim=-2)
                tv_vec = l2_matvec(l2_mat6(mix(tb, st.tv_t)), u_tvv)
                v_out = (v_new + u_self * gv.unsqueeze(-1) + tv_vec * gtv.unsqueeze(-1)).to(v_dt)
                return (s, v_out, scaled_inv) if return_inv else (s, v_out)
            u_self, u_tvv, u_vvp, u_vvq = chan_linear(u_new, bk["w_u_out"]).split([kv, kv, kt2, kt2], dim=-2)
            t_self, t_tvt = chan_linear(tb, bk["w_tb"]).split([kt2, kv], dim=-2)
            tv_vec = l2_matvec(l2_mat6(t_tvt), u_tvv)
            v_out = (v_new + u_self * gv.unsqueeze(-1) + tv_vec * gtv.unsqueeze(-1)).to(v_dt)
            t_out = (t_new + t_self * gt.unsqueeze(-1) + l2_pair_mm(u_vvp, u_vvq) * gvv.unsqueeze(-1)).to(v_dt)
            return (s, v_out, t_out, scaled_inv) if return_inv else (s, v_out, t_out)
    gv = torch.tanh(st.vec_out_gate(s_gn).to(v_dt)).unsqueeze(-1)
    gtv = torch.tanh(st.tv_gate(s_gn).to(v_dt)).unsqueeze(-1) if st.inv_norm else None
    with fp32_guard(s.device):
        v_out = v_new + mix(u_new, st.vec_self) * gv
        if st.inv_norm:
            v_out = v_out + l2_matvec(l2_mat6(mix(bound_vectors(t_new, st.vec_c), st.tv_t)), mix(u_new, st.tv_v)) * gtv
    v_out = v_out.to(v_dt)
    if not return_ten and ten is None:
        return (s, v_out, scaled_inv) if return_inv else (s, v_out)
    gt = torch.tanh(st.ten_out_gate(s_gn).to(v_dt)).unsqueeze(-1)
    gvv = torch.tanh(st.vv_gate(s_gn).to(v_dt)).unsqueeze(-1) if st.inv_norm else None
    with fp32_guard(s.device):
        t_out = t_new + mix(bound_vectors(t_new, st.vec_c), st.ten_self) * gt
        if st.inv_norm:
            t_out = t_out + l2_pair_mm(mix(u_new, st.vv_p), mix(u_new, st.vv_q)) * gvv
    t_out = t_out.to(v_dt)
    return (s, v_out, t_out, scaled_inv) if return_inv else (s, v_out, t_out)


def remat_stage_ext(st, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, domain: str, fin: bool,
                    ten: Tensor | None, return_ten: bool, cond_atom: Tensor | None, cond_lot_atom: Tensor | None,
                    dsum: Tensor | None, return_inv: bool):
    """Stage-level rematerialisation preserving optional lam6 per-atom tensors and `return_inv`."""
    ts, rb = pack(eg, nc)
    opt_items = [(k, v) for k, v in (("ten", ten), ("ca", cond_atom), ("cl", cond_lot_atom), ("ds", dsum)) if v is not None]
    k_opt = len(opt_items)

    def fn(s_, v_, *rest):
        opt_d = {k: rest[i] for i, (k, _) in enumerate(opt_items)}
        e, n = rb(*rest[k_opt:])
        with suppress_env_remat():
            out = st.core_plain(s_, v_, e, n, domain, opt_d.get("ten"), return_ten=return_ten,
                                cond_atom=opt_d.get("ca"), cond_lot_atom=opt_d.get("cl"), dsum=opt_d.get("ds"), return_inv=return_inv)
            return ((st.finish_plain(out[0], n, domain) if fin else out[0]), *out[1:])

    return remat_apply(RematSpec(fn, tuple(st.parameters())), (s, vec, *(v for _, v in opt_items), *ts))


def unit_block_ext(st, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, domain: str, fin: bool,
                   ten: Tensor | None, return_ten: bool, cond_atom: Tensor | None, cond_lot_atom: Tensor | None,
                   dsum: Tensor | None, return_inv: bool, return_dsum: bool):
    """Block-level (`edge` + chunked `post`) rematerialisation supporting extended LAM-6 tensors."""
    t_eff = ten if st.layout.later else None
    dsum_in = dsum if dsum is not None else (None if nc is None else getattr(nc, "dsum", None))
    can_csr_dsum = (
        dsum_in is None and eg.mode == "csr" and not eg.create_graph
        and "_aggregate" not in st.__dict__
        and (return_dsum or st.coord_inv_scale or st.env_modulated)
    )
    a_out, raw = _unit_a_ext(st, s, vec, eg, nc, ten=t_eff, cond_atom=cond_atom, want_dsum=can_csr_dsum)
    agg_tup = a_out if raw else (a_out[0],)
    dsum_a = (a_out[2] if raw else a_out[1]) if (raw or can_csr_dsum) and dsum_in is None else dsum_in
    cl_a, dsum_eff = st._resolve_aux(s, eg, nc, cond_atom, cond_lot_atom, dsum_a)
    if return_dsum and dsum_eff is None:
        dsum_eff = compute_dsum(eg, st.edge_cfg)
    ca = node_chunk_size(s.device, env_mod=st.env_modulated) if (st.env_modulated or s.device.type == "mps") else 4096
    if s.shape[0] > ca and st._inv_hook is None and not torch.compiler.is_compiling():
        res = post_chunked_stage(
            st, s, vec, agg_tup, domain, t_eff, cl_a, dsum_eff, return_ten=return_ten, return_inv=return_inv,
            raw=raw, nc=nc, fin=fin and domain == "crystal", chunk_atoms=ca, create_graph=eg.create_graph,
        )
        if fin and domain != "crystal":
            res = (unit_finish(st, res[0], nc, domain), *res[1:])
        return (*res, dsum_eff) if return_dsum else res
    n_agg = len(agg_tup)
    opts = [(k, v) for k, v in (("ten", t_eff), ("cl", cl_a), ("ds", dsum_eff)) if v is not None]
    ps = tuple(st.parameters())

    def fn(s_, v_, *rest):
        agg_c = finalise(*rest[:n_agg], st.layout) if raw else rest[0]
        d = {k: rest[n_agg + i] for i, (k, _) in enumerate(opts)}
        with suppress_env_remat():
            res = st._post_impl(s_, v_, agg_c, domain, d.get("ten"), return_ten=return_ten,
                               cond_lot_atom=d.get("cl"), dsum=d.get("ds"), return_inv=return_inv, allow_fused=not eg.create_graph)
            return ((st.finish_plain(res[0], nc, domain) if fin else res[0]), *res[1:])

    res = remat_apply(RematSpec(fn, ps), (s, vec, *agg_tup, *(v for _, v in opts)))
    return (*res, dsum_eff) if return_dsum else res


def stage_dispatch(st, s: Tensor, vec: Tensor, eg: EdgeGraph, nc: NodeContext | None, domain: str, fin: bool,
                   *, ten: Tensor | None = None, return_ten: bool = False, cond_atom: Tensor | None = None,
                   cond_lot_atom: Tensor | None = None, dsum: Tensor | None = None, return_inv: bool = False,
                   return_dsum: bool = False):
    """Shared `Stage.core` / `Stage.forward` execution dispatcher."""
    ext = cond_atom is not None or cond_lot_atom is not None or dsum is not None or return_inv
    if st.remat == "none":
        ca_post = node_chunk_size(s.device, env_mod=st.env_modulated) if s.device.type == "mps" else (16384 if s.device.type == "cuda" else (4096 if st.env_modulated else NODE_CHUNK_ATOMS))
        if st._use_csr_node_chunks(s, eg, domain):
            can_csr_dsum = (dsum is None and not eg.create_graph and (return_dsum or st.coord_inv_scale or st.env_modulated))
            if can_csr_dsum:
                agg, dsum_eff = st.aggregate_plain(s, vec, eg, nc, ten, cond_atom=cond_atom, return_dsum=True)
                cl_a, _ = st._resolve_aux(s, eg, nc, cond_atom, cond_lot_atom, dsum_eff)
                out = post_chunked_stage(st, s, vec, agg, domain, ten, cl_a, dsum_eff, return_ten=return_ten, return_inv=return_inv, chunk_atoms=ca_post)
                out = (st.finish_plain(out[0], nc, domain), *out[1:]) if fin else out
                return (*out, dsum_eff) if return_dsum else out
            agg = st.aggregate_plain(s, vec, eg, nc, ten, cond_atom=cond_atom)
            if ext or st.coord_inv_scale or st.env_modulated:
                cl_a, dsum_eff = st._resolve_aux(s, eg, nc, cond_atom, cond_lot_atom, dsum)
                out = post_chunked_stage(st, s, vec, agg, domain, ten, cl_a, dsum_eff, return_ten=return_ten, return_inv=return_inv, chunk_atoms=ca_post)
                out = (st.finish_plain(out[0], nc, domain), *out[1:]) if fin else out
                return (*out, dsum_eff) if return_dsum else out
            out = post_chunked_crystal(st, s, vec, (agg,), nc, domain, fin, ten, return_ten=return_ten)
            return (*out, compute_dsum(eg, st.edge_cfg)) if return_dsum else out
        if st._use_post_node_chunks(s, eg):
            agg, cl_a, dsum_eff = st._run_agg_and_aux(s, vec, eg, nc, ten if st.layout.later else None, cond_atom, cond_lot_atom, dsum, return_dsum)
            out = post_chunked_stage(st, s, vec, agg, domain, ten, cl_a, dsum_eff, return_ten=return_ten, return_inv=return_inv, chunk_atoms=ca_post)
            out = (st.finish_plain(out[0], nc, domain), *out[1:]) if fin else out
            return (*out, dsum_eff) if return_dsum else out
        out = st.core_plain(s, vec, eg, nc, domain, ten, return_ten=return_ten,
                            cond_atom=cond_atom, cond_lot_atom=cond_lot_atom, dsum=dsum, return_inv=return_inv,
                            return_dsum=return_dsum)
        if return_dsum:
            head = st.finish_plain(out[0], nc, domain) if fin else out[0]
            return (head, *out[1:])
        return (st.finish_plain(out[0], nc, domain), *out[1:]) if fin else out
    if ext or st.coord_inv_scale or st.env_modulated or return_dsum:
        if st.remat == "block":
            return unit_block_ext(st, s, vec, eg, nc, domain, fin, ten, return_ten, cond_atom, cond_lot_atom, dsum, return_inv, return_dsum)
        dsum_eff = dsum if (dsum is not None or not return_dsum) else compute_dsum(eg, st.edge_cfg)
        out = remat_stage_ext(st, s, vec, eg, nc, domain, fin, ten, return_ten, cond_atom, cond_lot_atom, dsum_eff, return_inv)
        return (*out, dsum_eff) if return_dsum else out
    return (unit_stage if st.remat == "stage" else unit_block)(st, s, vec, eg, nc, domain, fin, ten, return_ten=return_ten)
