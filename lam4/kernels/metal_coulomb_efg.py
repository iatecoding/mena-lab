"""O(N) Metal custom ops and autograd for smeared Coulomb (phi, field, L=2 EFG) on MPS."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from lam4.kernels import metal_csr as _metal_csr
from lam4.kernels.metal_coulomb_efg_src import METAL_COULOMB_EFG_MSL
from lam4.kernels.metal_coulomb_src import METAL_COULOMB_MSL
from lam4.kernels.metal_node_flags import MetalShader

_EFG_SHADER = MetalShader("metal_coulomb_efg", lambda: METAL_COULOMB_MSL + METAL_COULOMB_EFG_MSL)
_R2 = 1.0 / math.sqrt(2.0)
_R6 = 1.0 / math.sqrt(6.0)


def has_metal_coulomb_efg() -> bool:
    return bool(_metal_csr.has_metal_csr() and _EFG_SHADER.available())


def get_metal_coulomb_efg_lib() -> Any:
    return _EFG_SHADER.lib()


def _decode_ge_sorted(g_efg: Tensor, perm: Tensor, mf: Tensor, n_pad: int) -> tuple[Tensor, Tensor]:
    g0_s = torch.zeros((n_pad, 4), dtype=torch.float32, device=g_efg.device)
    g1_s = torch.zeros((n_pad, 2), dtype=torch.float32, device=g_efg.device)
    n = int(perm.numel())
    ge = (g_efg.contiguous() * mf.unsqueeze(-1)).index_select(0, perm)
    t3 = _R2 * ge[:, 3]
    t4 = _R6 * ge[:, 4]
    g0_s[:n, 0], g0_s[:n, 1], g0_s[:n, 2], g0_s[:n, 3] = t3 - t4, -t3 - t4, 2.0 * t4, _R2 * ge[:, 0]
    g1_s[:n, 0], g1_s[:n, 1] = _R2 * ge[:, 1], _R2 * ge[:, 2]
    return g0_s, g1_s


@torch.library.custom_op("lam4::metal_coulomb_efg_fwd", mutates_args=())
def _metal_coulomb_efg_fwd_op(
    pos: Tensor, q: Tensor, mask_u8: Tensor, graph_i64: Tensor, group_ptr: Tensor,
    perm: Tensor, bsph: Tensor, pos_s: Tensor, mask_s: Tensor,
    n_graphs: int, sigma: float,
) -> tuple[Tensor, Tensor, Tensor]:
    from lam4.kernels.metal_coulomb import _build_fast64_meta, _coulomb_params

    n_atoms = int(pos.shape[0])
    pos_c, q_c = pos.to(torch.float32).contiguous(), q.to(torch.float32).contiguous()
    phi = torch.empty(n_atoms, dtype=torch.float32, device=pos.device)
    fld = torch.empty((n_atoms, 3), dtype=torch.float32, device=pos.device)
    efg = torch.empty((n_atoms, 5), dtype=torch.float32, device=pos.device)
    if n_atoms == 0:
        return phi, fld, efg
    lib = get_metal_coulomb_efg_lib()
    params = _coulomb_params(sigma)
    use_fast = int(n_graphs) == 1 and n_atoms >= 1024
    if use_fast:
        if perm.numel() != n_atoms:
            perm, bsph, pos_s, mask_s, n_pad, n_tiles = _build_fast64_meta(pos_c, mask_u8)
        else:
            n_pad, n_tiles = int(mask_s.shape[0]), int(bsph.shape[0])
        pq_s = torch.zeros((n_pad, 4), dtype=torch.float32, device=pos.device)
        pq_s[:n_atoms, :3] = pos_s
        pq_s[:n_atoms, 3] = (q_c * (mask_u8 != 0).to(torch.float32)).index_select(0, perm)
        out_p = torch.empty(n_pad, dtype=torch.float32, device=pos.device)
        out_f = torch.empty(n_pad * 3, dtype=torch.float32, device=pos.device)
        out_e = torch.empty(n_pad * 5, dtype=torch.float32, device=pos.device)
        lib.coulomb_efg_fwd_fast64(
            pq_s, mask_s, bsph, out_p, out_f, out_e, params, n_tiles,
            threads=(n_pad, 1, 1), group_size=(64, 1, 1),
        )
        phi.index_copy_(0, perm, out_p[:n_atoms])
        fld.index_copy_(0, perm, out_f[:n_atoms * 3].view(n_atoms, 3))
        efg.index_copy_(0, perm, out_e[:n_atoms * 5].view(n_atoms, 5))
        return phi, fld, efg
    tg = 256
    tot = ((n_atoms + tg - 1) // tg) * tg
    if int(n_graphs) == 1 and n_atoms >= 192:
        lib.coulomb_efg_fwd_tiled(
            pos_c, q_c, mask_u8, phi, fld, efg, n_atoms, params,
            threads=(tot, 1, 1), group_size=(tg, 1, 1),
        )
    else:
        lib.coulomb_efg_fwd_grouped(
            pos_c, q_c, mask_u8, graph_i64, group_ptr, phi, fld, efg, n_atoms, params,
            threads=(tot, 1, 1), group_size=(tg, 1, 1),
        )
    return phi, fld, efg


@_metal_coulomb_efg_fwd_op.register_fake
def _(
    pos: Tensor, q: Tensor, mask_u8: Tensor, graph_i64: Tensor, group_ptr: Tensor,
    perm: Tensor, bsph: Tensor, pos_s: Tensor, mask_s: Tensor,
    n_graphs: int, sigma: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(q, dtype=torch.float32),
        torch.empty((pos.shape[0], 3), dtype=torch.float32, device=pos.device),
        torch.empty((pos.shape[0], 5), dtype=torch.float32, device=pos.device),
    )


@torch.library.custom_op("lam4::metal_coulomb_efg_bwd", mutates_args=())
def _metal_coulomb_efg_bwd_op(
    pos: Tensor, q: Tensor, mask_u8: Tensor, graph_i64: Tensor, group_ptr: Tensor,
    perm: Tensor, bsph: Tensor, pos_s: Tensor, mask_s: Tensor,
    g_phi: Tensor, g_fld: Tensor, g_efg: Tensor,
    n_graphs: int, sigma: float,
    has_gp: bool, has_gf: bool, has_ge: bool, need_dpos: bool, need_dq: bool,
) -> tuple[Tensor, Tensor]:
    from lam4.kernels.metal_coulomb import _build_fast64_meta, _coulomb_params

    n_atoms = int(pos.shape[0])
    if n_atoms == 0 or not (need_dpos or need_dq) or not (has_gp or has_gf or has_ge):
        return (
            torch.zeros((n_atoms, 3), dtype=torch.float32, device=pos.device),
            torch.zeros(n_atoms, dtype=torch.float32, device=pos.device),
        )
    lib = get_metal_coulomb_efg_lib()
    params = _coulomb_params(sigma)
    pos_c, q_c = pos.to(torch.float32).contiguous(), q.to(torch.float32).contiguous()
    use_fast = int(n_graphs) == 1 and n_atoms >= 1024
    if use_fast:
        if perm.numel() != n_atoms:
            perm, bsph, pos_s, mask_s, n_pad, n_tiles = _build_fast64_meta(pos_c, mask_u8)
        else:
            n_pad, n_tiles = int(mask_s.shape[0]), int(bsph.shape[0])
        mf = (mask_u8 != 0).to(torch.float32)
        pq_s = torch.zeros((n_pad, 4), dtype=torch.float32, device=pos.device)
        pq_s[:n_atoms, :3] = pos_s
        pq_s[:n_atoms, 3] = (q_c * mf).index_select(0, perm)
        gf_s = torch.zeros((n_pad, 4), dtype=torch.float32, device=pos.device)
        if has_gf:
            gf_s[:n_atoms, :3] = (g_fld.contiguous() * mf.unsqueeze(-1)).index_select(0, perm)
        if has_gp:
            gf_s[:n_atoms, 3] = (g_phi.contiguous() * mf).index_select(0, perm)
        if has_ge:
            g0_s, g1_s = _decode_ge_sorted(g_efg, perm, mf, n_pad)
        else:
            g0_s = torch.zeros((n_pad, 4), dtype=torch.float32, device=pos.device)
            g1_s = torch.zeros((n_pad, 2), dtype=torch.float32, device=pos.device)
        dp_s = torch.empty(n_pad * 3, dtype=torch.float32, device=pos.device) if need_dpos else pq_s[:1, 0]
        dq_s = torch.empty(n_pad, dtype=torch.float32, device=pos.device) if need_dq else pq_s[:1, 0]
        lib.coulomb_efg_bwd_fast64(
            pq_s, mask_s, bsph, gf_s, g0_s, g1_s, dp_s, dq_s, params, n_tiles,
            int(has_gp), int(has_gf), int(has_ge), int(need_dpos), int(need_dq),
            threads=(n_pad, 1, 1), group_size=(64, 1, 1),
        )
        d_pos_out = torch.zeros((n_atoms, 3), dtype=torch.float32, device=pos.device)
        d_q_out = torch.zeros(n_atoms, dtype=torch.float32, device=pos.device)
        if need_dpos:
            d_pos_out.index_copy_(0, perm, dp_s[:n_atoms * 3].view(n_atoms, 3))
        if need_dq:
            d_q_out.index_copy_(0, perm, dq_s[:n_atoms])
        return d_pos_out, d_q_out

    gp_c = g_phi.contiguous() if has_gp else q_c[:1]
    gf_c = g_fld.contiguous() if has_gf else pos_c[:1]
    ge_c = g_efg.contiguous() if has_ge else pos_c[:1]
    d_pos_buf = torch.empty_like(pos_c) if need_dpos else pos_c[:1]
    d_q_buf = torch.empty_like(q_c) if need_dq else q_c[:1]
    if int(n_graphs) == 1 and n_atoms >= 192:
        tg = 128
        tot = ((n_atoms + tg - 1) // tg) * tg
        lib.coulomb_efg_bwd_tiled(
            pos_c, q_c, mask_u8, gp_c, gf_c, ge_c, d_pos_buf, d_q_buf, n_atoms,
            int(has_gp), int(has_gf), int(has_ge), int(need_dpos), int(need_dq), params,
            threads=(tot, 1, 1), group_size=(tg, 1, 1),
        )
    else:
        tg = 256
        tot = ((n_atoms + tg - 1) // tg) * tg
        lib.coulomb_efg_bwd_grouped(
            pos_c, q_c, mask_u8, graph_i64, group_ptr, gp_c, gf_c, ge_c, d_pos_buf, d_q_buf, n_atoms,
            int(has_gp), int(has_gf), int(has_ge), int(need_dpos), int(need_dq), params,
            threads=(tot, 1, 1), group_size=(tg, 1, 1),
        )
    return (
        d_pos_buf if need_dpos else torch.zeros_like(pos_c),
        d_q_buf if need_dq else torch.zeros_like(q_c),
    )


@_metal_coulomb_efg_bwd_op.register_fake
def _(
    pos: Tensor, q: Tensor, mask_u8: Tensor, graph_i64: Tensor, group_ptr: Tensor,
    perm: Tensor, bsph: Tensor, pos_s: Tensor, mask_s: Tensor,
    g_phi: Tensor, g_fld: Tensor, g_efg: Tensor,
    n_graphs: int, sigma: float,
    has_gp: bool, has_gf: bool, has_ge: bool, need_dpos: bool, need_dq: bool,
) -> tuple[Tensor, Tensor]:
    return torch.empty(pos.shape, dtype=torch.float32, device=pos.device), torch.empty(q.shape, dtype=torch.float32, device=q.device)


def _setup_efg_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    ctx.save_for_backward(*inputs[:9])
    ctx.n_graphs, ctx.sigma = int(inputs[9]), float(inputs[10])


def _bwd_efg_autograd(ctx: Any, g_phi: Tensor | None, g_fld: Tensor | None, g_efg: Tensor | None):
    pos, q, mask_u8, graph_i64, group_ptr, perm, bsph, pos_s, mask_s = ctx.saved_tensors
    need_dp, need_dq = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
    if not (need_dp or need_dq):
        return (None,) * 11
    gp = g_phi.contiguous() if g_phi is not None else q[:1]
    gf = g_fld.contiguous() if g_fld is not None else pos[:1]
    ge = g_efg.contiguous() if g_efg is not None else pos[:1]
    dp, dq = _metal_coulomb_efg_bwd_op(
        pos, q, mask_u8, graph_i64, group_ptr, perm, bsph, pos_s, mask_s, gp, gf, ge,
        ctx.n_graphs, ctx.sigma, g_phi is not None, g_fld is not None, g_efg is not None, need_dp, need_dq,
    )
    return (dp.to(pos.dtype) if need_dp else None, dq.to(q.dtype) if need_dq else None) + (None,) * 9


def _efg_bwd2_unsupported(*_args: Any, **_kw: Any):
    raise NotImplementedError("lam4::metal_coulomb_efg_bwd supports 1st-order backward only; route create_graph=True to OpenMultipoleBackend.")


_metal_coulomb_efg_fwd_op.register_autograd(_bwd_efg_autograd, setup_context=_setup_efg_ctx)
_metal_coulomb_efg_bwd_op.register_autograd(_efg_bwd2_unsupported, setup_context=lambda ctx, inputs, output: None)


def eval_metal_coulomb_efg(
    pos: Tensor, q: Tensor, mask_u8: Tensor, graph_i64: Tensor, group_ptr: Tensor,
    n_graphs: int, sigma: float, fast_meta: tuple[Tensor, Tensor, Tensor, Tensor, int, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if fast_meta is not None:
        perm, bsph, pos_s, mask_s, _, _ = fast_meta
    else:
        perm = torch.empty(0, dtype=torch.int64, device=pos.device)
        bsph = torch.empty((0, 4), dtype=torch.float32, device=pos.device)
        pos_s = torch.empty((0, 3), dtype=torch.float32, device=pos.device)
        mask_s = torch.empty(0, dtype=torch.uint8, device=pos.device)
    return _metal_coulomb_efg_fwd_op(pos, q, mask_u8, graph_i64, group_ptr, perm, bsph, pos_s, mask_s, int(n_graphs), float(sigma))
