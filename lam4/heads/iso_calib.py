"""Step-0 closed-form Ridge LSQ calibration of Readout `iso_a`/`iso_b` and N=1 smoothstep delta helpers."""
from __future__ import annotations

import itertools
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch import Tensor

from lam4.core.radial import upcast_fp32
from lam4.registry.sources import resolve_lot_cell
from lam4.registry.tokens import XC_ID, lot_cell_id, xc_id

N_Z = 119


def iso_smoothstep(dsum: Tensor, d_lo: float = 0.05, d_hi: float = 0.5) -> Tensor:
    """Standard C1 smoothstep on `dsum in [d_lo, d_hi]`; `1 - smoothstep == 1` at N=1 (`dsum=0`) and `0` for `dsum >= 0.5`."""
    t = ((upcast_fp32(dsum) - float(d_lo)) / float(d_hi - d_lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@torch.no_grad()
def apply_iso_atom_table(buf: Tensor, table: Any, *, n_xc: int, n_paw: int, s_max: int, zero_first: bool = True) -> Tensor:
    """Populate `buf [n_cells, N_Z, s_max]` from a Tensor or dict keyed by source name / `(xc, paw, u)` / bare XC."""
    if zero_first:
        buf.zero_()
    n_cells = buf.shape[0]
    if torch.is_tensor(table):
        src = table.to(device=buf.device, dtype=buf.dtype)
        if src.ndim == 5:
            src = src.reshape(-1, src.shape[-2], src.shape[-1])
        c_end = min(n_cells, src.shape[0])
        z_end = min(N_Z, src.shape[1])
        s_end = min(s_max, src.shape[2])
        buf[:c_end, :z_end, :s_end].copy_(src[:c_end, :z_end, :s_end])
        buf[:, 0, :].zero_()
        return buf
    if not isinstance(table, dict):
        raise TypeError(f"iso_atom_delta table must be a Tensor or dict, got {type(table).__name__}")

    def _row_tensor(val: Any) -> Tensor:
        if isinstance(val, dict):
            out = buf.new_zeros(N_Z, s_max)
            for zk, ev in val.items():
                if isinstance(zk, tuple):
                    zi, si = int(zk[0]), int(zk[1]) - 1
                    if 1 <= zi < N_Z and 0 <= si < s_max:
                        out[zi, si] = float(ev)
                elif 1 <= int(zk) < N_Z:
                    out[int(zk), :] = float(ev)
            return out
        t = val if torch.is_tensor(val) else torch.as_tensor(val, dtype=buf.dtype)
        t = t.to(device=buf.device, dtype=buf.dtype)
        out = buf.new_zeros(N_Z, s_max)
        if t.ndim == 1:
            out[: min(N_Z, t.shape[0]), :] = t[:N_Z].unsqueeze(-1)
        elif t.ndim == 2:
            out[: min(N_Z, t.shape[0]), : min(s_max, t.shape[1])] = t[:N_Z, :s_max]
        else:
            raise ValueError(f"expected 1D [Z] or 2D [Z, S] iso_atom row, got {tuple(t.shape)}")
        out[0, :].zero_()
        return out

    broad = [(k, v) for k, v in table.items() if isinstance(k, str) and k in XC_ID]
    specific = [(k, v) for k, v in table.items() if not (isinstance(k, str) and k in XC_ID)]
    for k, v in broad:
        xi = xc_id(k)
        row = _row_tensor(v)
        for pi in range(n_paw):
            for ui in (0, 1):
                cid = lot_cell_id(xi, pi, ui, n_paw)
                if cid < n_cells:
                    buf[cid].copy_(row)
    for k, v in specific:
        cid = int(k) if isinstance(k, int) else resolve_lot_cell(k, n_paw=n_paw)
        if 0 <= cid < n_cells:
            buf[cid].copy_(_row_tensor(v))
    buf[:, 0, :].zero_()
    return buf


def _functional_atoms(cond_mod: nn.Module | None, b, cond_dim: int, like: Tensor) -> Tensor:
    if cond_mod is None:
        return like.new_zeros(b.n_atoms, cond_dim)
    if getattr(cond_mod, "factorized", False) and hasattr(cond_mod, "functional_atoms"):
        xc = getattr(b, "xc_id", None)
        xc = b.token if xc is None else xc
        paw = getattr(b, "paw_id", None)
        u_atom = getattr(b, "u_atom", None)
        return cond_mod.functional_atoms(xc, b.u_flag, b.graph, paw_id=paw, u_atom=u_atom)
    return cond_mod.functional(b.token, b.u_flag).index_select(0, b.graph)


@torch.no_grad()
def calibrate_iso_reference(
    model: nn.Module,
    batches: Iterable,
    *,
    max_batches: int = 16,
    ridge_a: float = 1e-2,
    ridge_b: float = 1.0,
    cg_iters: int = 80,
) -> tuple[Tensor, Tensor]:
    """Closed-form step-0 Ridge LSQ solve for `(readout.iso_a, readout.iso_b)` after frozen LSQ target subtraction."""
    from lam4.heads.readout import Readout

    readout = next((m for m in model.modules() if isinstance(m, Readout) and m.iso_a is not None and m.iso_b is not None), None)
    if readout is None:
        raise ValueError("calibrate_iso_reference requires a model containing Readout with iso_reference=True")
    cond_mod = next((m for m in model.modules() if hasattr(m, "functional")), None)
    cd = int(readout.iso_b.embedding_dim)
    dev, dt = readout.iso_a.weight.device, readout.iso_a.weight.dtype

    snap_a = readout.iso_a.weight.detach().clone()
    snap_b = readout.iso_b.weight.detach().clone()
    readout.iso_a.weight.zero_()
    readout.iso_b.weight.zero_()
    readout._baked = None

    was_training = model.training
    model.eval()
    zs, gs, cs, rs = [], [], [], []
    d_a = torch.full((N_Z,), float(ridge_a), dtype=torch.float64, device=dev)
    d_b = torch.full((N_Z, cd), float(ridge_b), dtype=torch.float64, device=dev)
    g_off = 0
    try:
        for b in itertools.islice(batches, int(max_batches)):
            e_tgt = b.energy.to(device=dev, dtype=torch.float64)
            valid_g = b.graph_mask.to(dev) & b.energy_mask.to(dev) & torch.isfinite(e_tgt)
            g_local = torch.nonzero(valid_g, as_tuple=False).squeeze(-1)
            if g_local.numel() == 0:
                continue
            try:
                out = model(b, compute_forces=False)
            except TypeError:
                out = model(b, b.pos, b.domain)
            e_pred = (out["energy"] if isinstance(out, dict) else out).to(device=dev, dtype=torch.float64)
            r_b = e_tgt - e_pred
            c_full = _functional_atoms(cond_mod, b, cd, r_b).to(device=dev, dtype=torch.float64)

            remap = torch.full((b.n_graphs,), -1, dtype=torch.long, device=dev)
            remap[g_local] = torch.arange(g_local.numel(), device=dev, dtype=torch.long)
            ag = remap.index_select(0, b.graph.to(dev))
            a_ok = b.atom_mask.to(dev) & (ag >= 0) & (b.z.to(dev) > 0)
            if a_ok.any():
                z_b = b.z.to(dev)[a_ok].clamp(0, N_Z - 1).to(torch.long)
                g_b = ag[a_ok]
                c_b = c_full[a_ok]
                ng_b = int(g_local.numel())
                gz = g_b * N_Z + z_b
                cnt_b = d_a.new_zeros(ng_b * N_Z).index_add(0, gz, d_a.new_ones(z_b.numel())).view(ng_b, N_Z)
                sumc_b = d_b.new_zeros(ng_b * N_Z, cd).index_add(0, gz, c_b).view(ng_b, N_Z, cd)
                d_a.add_(cnt_b.pow(2).sum(0))
                d_b.add_(sumc_b.pow(2).sum(0))
                zs.append(z_b)
                gs.append(g_b + g_off)
                cs.append(c_b)
            rs.append(r_b[g_local])
            g_off += int(g_local.numel())
    except BaseException:
        readout.iso_a.weight.copy_(snap_a)
        readout.iso_b.weight.copy_(snap_b)
        if was_training:
            model.train()
        raise

    if g_off == 0 or not zs:
        readout.iso_a.weight.copy_(snap_a)
        readout.iso_b.weight.copy_(snap_b)
        if was_training:
            model.train()
        return readout.iso_a.weight.detach().clone(), readout.iso_b.weight.detach().clone()

    z_all, g_all, c_all, r_vec = torch.cat(zs), torch.cat(gs), torch.cat(cs), torch.cat(rs)

    def _mv(va: Tensor, vb: Tensor) -> tuple[Tensor, Tensor]:
        pa = va.index_select(0, z_all) + (vb.index_select(0, z_all) * c_all).sum(-1)
        pg = r_vec.new_zeros(g_off).index_add(0, g_all, pa)
        ra = pg.index_select(0, g_all)
        return (d_a.new_zeros(N_Z).index_add(0, z_all, ra) + ridge_a * va,
                d_b.new_zeros(N_Z, cd).index_add(0, z_all, ra.unsqueeze(-1) * c_all) + ridge_b * vb)

    r_at = r_vec.index_select(0, g_all)
    ra_v = d_a.new_zeros(N_Z).index_add(0, z_all, r_at)
    rb_v = d_b.new_zeros(N_Z, cd).index_add(0, z_all, r_at.unsqueeze(-1) * c_all)
    ra_v[0] = 0.0
    rb_v[0].zero_()
    xa, xb = torch.zeros_like(ra_v), torch.zeros_like(rb_v)
    za_v, zb_v = ra_v / d_a, rb_v / d_b
    pa_v, pb_v = za_v.clone(), zb_v.clone()
    rz = (ra_v * za_v).sum() + (rb_v * zb_v).sum()
    for _ in range(int(cg_iters)):
        apa, apb = _mv(pa_v, pb_v)
        apa[0] = 0.0
        apb[0].zero_()
        denom = ((pa_v * apa).sum() + (pb_v * apb).sum()).clamp_min(1e-30)
        alpha = rz / denom
        xa, xb = xa + alpha * pa_v, xb + alpha * pb_v
        ra_v, rb_v = ra_v - alpha * apa, rb_v - alpha * apb
        za_v, zb_v = ra_v / d_a, rb_v / d_b
        rz_new = (ra_v * za_v).sum() + (rb_v * zb_v).sum()
        beta = rz_new / rz.clamp_min(1e-30)
        pa_v, pb_v = za_v + beta * pa_v, zb_v + beta * pb_v
        rz = rz_new

    xa[0] = 0.0
    xb[0].zero_()
    readout.iso_a.weight[:, 0].copy_(xa.to(device=dev, dtype=dt))
    readout.iso_b.weight.copy_(xb.to(device=dev, dtype=dt))
    readout._baked = None
    if was_training:
        model.train()
    return readout.iso_a.weight.detach().clone(), readout.iso_b.weight.detach().clone()
