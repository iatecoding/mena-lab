"""Atomic-free Tier-2 Metal kernel & custom autograd for open-boundary smeared ExactCoulomb.

Eliminates the O(G * cap^2 * 3) pair-matrix and autograd tape blowup on large molecules and
heterogeneous molecular batches:
  - Memory: O(N) (`group_ptr [G+1]` + `mask [N]`, zero pair buffers even at N = 32,768).
  - Work:   Exact O(sum_g n_g^2) per molecule (zero max_n padding waste across batches).
  - Speed:  Threadgroup shared-memory tiling for large single molecules (G=1, N >= 256) and
            register-resident group loops for multi-molecule batches (G > 1).
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from lam4.core.consts import is_fake, proxy_tracing
from lam4.kernels import metal_csr as _metal_csr
from lam4.kernels.metal_coulomb_fast_src import METAL_COULOMB_FAST_MSL
from lam4.kernels.metal_coulomb_src import METAL_COULOMB_MSL
from lam4.kernels.metal_node_flags import MetalShader

_COULOMB_SHADER = MetalShader("metal_coulomb", lambda: METAL_COULOMB_MSL + METAL_COULOMB_FAST_MSL)
_PTR_G1_CACHE: dict[tuple[str, int | None, int], Tensor] = {}


def has_metal_coulomb() -> bool:
    return bool(_metal_csr.has_metal_csr() and _COULOMB_SHADER.available())


def get_metal_coulomb_lib() -> Any:
    return _COULOMB_SHADER.lib()


def _coulomb_params(sigma: float) -> tuple[float, float, float, float]:
    inv_sig = 1.0 / float(sigma)
    c0 = 2.0 * inv_sig / math.sqrt(math.pi)
    return (inv_sig, c0, inv_sig * inv_sig, 5.5 * float(sigma))


def _morton_perm_i32(pos: Tensor, mask_u8: Tensor | None = None) -> Tensor:
    pos_f = pos.to(torch.float32)
    if mask_u8 is not None:
        act = mask_u8 != 0
        has_act = act.any()
        act3 = act.unsqueeze(-1)
        pmin_a = torch.where(act3, pos_f, pos_f.new_full((), 1e30)).amin(0)
        pmax_a = torch.where(act3, pos_f, pos_f.new_full((), -1e30)).amax(0)
        pmin = torch.where(has_act, pmin_a, pos_f.amin(0))
        pmax = torch.where(has_act, pmax_a, pos_f.amax(0))
    else:
        act = None
        pmin = pos_f.amin(0)
        pmax = pos_f.amax(0)
    c = ((pos_f - pmin) / (pmax - pmin).clamp_min(1e-3) * 1023.0).clamp(0.0, 1023.0).to(torch.int32)

    def part(x: Tensor) -> Tensor:
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        return (x | (x << 2)) & 0x09249249

    code = part(c[:, 0]) | (part(c[:, 1]) << 1) | (part(c[:, 2]) << 2)
    if act is not None:
        code = torch.where(act, code, torch.full_like(code, 0x3FFFFFFF))
    return torch.argsort(code)


def _build_fast64_meta(pos: Tensor, mask_u8: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, int, int]:
    """Spatial Morton order + per-64-atom bounding spheres computed once per geometry for `coulomb_fast64`."""
    dev = pos.device
    n = int(pos.shape[0])
    n_tiles = (n + 63) // 64
    n_pad = n_tiles * 64
    with torch.no_grad():
        pos_f = pos.detach().to(torch.float32)
        perm = _morton_perm_i32(pos_f, mask_u8)
        pos_s = pos_f.index_select(0, perm)
        mask_s = torch.zeros(n_pad, dtype=torch.uint8, device=dev)
        mask_s[:n] = mask_u8.index_select(0, perm)
        pos_sp = torch.cat([pos_s, torch.full((n_pad - n, 3), 1e6, dtype=torch.float32, device=dev)], 0) if n_pad > n else pos_s
        tb = pos_sp.view(n_tiles, 64, 3)
        c = (tb.min(1).values + tb.max(1).values) * 0.5
        r = (tb - c.unsqueeze(1)).norm(dim=-1).max(1).values
        tile_full = (mask_s.view(n_tiles, 64) != 0).all(dim=1)
        r = torch.where(tile_full, r, torch.full_like(r, -1.0))
        bsph = torch.cat([c, r.unsqueeze(-1)], -1).contiguous()
    return perm, bsph, pos_s.contiguous(), mask_s, n_pad, n_tiles


@torch.library.custom_op("lam4::metal_coulomb_fast64_meta", mutates_args=())
def _metal_coulomb_fast64_meta_op(pos: Tensor, mask_u8: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n = int(pos.shape[0])
    n_tiles = (n + 63) // 64
    n_pad = n_tiles * 64
    dev = pos.device
    if n < 1024:
        return (
            torch.empty(n, dtype=torch.int64, device=dev),
            torch.full((n_tiles, 4), -2.0, dtype=torch.float32, device=dev),
            torch.empty((n, 3), dtype=torch.float32, device=dev),
            torch.zeros(n_pad, dtype=torch.uint8, device=dev),
        )
    perm, bsph, pos_s, mask_s, _, _ = _build_fast64_meta(pos, mask_u8)
    return perm, bsph, pos_s, mask_s


@_metal_coulomb_fast64_meta_op.register_fake
def _(pos: Tensor, mask_u8: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n = pos.shape[0]
    n_tiles = (n + 63) // 64
    n_pad = n_tiles * 64
    dev = pos.device
    return (
        torch.empty(n, dtype=torch.int64, device=dev),
        torch.empty((n_tiles, 4), dtype=torch.float32, device=dev),
        torch.empty((n, 3), dtype=torch.float32, device=dev),
        torch.empty(n_pad, dtype=torch.uint8, device=dev),
    )


def build_group_ptr(graph: Tensor, n_graphs: int) -> Tensor:
    """Zero-sync O(log N) GPU group pointer builder `[G + 1]` int32 for sorted `graph [N]`."""
    dev = graph.device
    sym = torch.compiler.is_compiling() or proxy_tracing() or is_fake(graph)
    if sym:
        if n_graphs <= 1:
            return torch.stack([
                torch.zeros((), dtype=torch.int32, device=dev),
                graph.new_zeros((), dtype=torch.int32) + graph.shape[0],
            ])
        z0 = torch.zeros((1,), dtype=torch.int32, device=dev)
        end = torch.searchsorted(graph.to(torch.int32).contiguous(), torch.arange(n_graphs, dtype=torch.int32, device=dev), right=True).to(torch.int32)
        return torch.cat([z0, end], dim=0)
    G = int(n_graphs)
    N = int(graph.numel())
    if G <= 1:
        key = (dev.type, dev.index, N)
        ptr = _PTR_G1_CACHE.get(key)
        if ptr is None:
            ptr = torch.tensor([0, N], dtype=torch.int32, device=dev)
            if len(_PTR_G1_CACHE) > 64:
                _PTR_G1_CACHE.clear()
            _PTR_G1_CACHE[key] = ptr
        return ptr
    ptr = torch.zeros(G + 1, dtype=torch.int32, device=dev)
    if N > 0:
        g_i32 = graph.to(torch.int32).contiguous()
        ptr[1:] = torch.searchsorted(g_i32, torch.arange(G, dtype=torch.int32, device=dev), right=True).to(torch.int32)
    return ptr


def build_coulomb_mask(atom_mask: Tensor, graph: Tensor, n_graphs: int, cap: int | None) -> Tensor:
    """Active-atom uint8 mask matching `make_slots_static` (`atom_mask & (rank < cap)`)."""
    if cap is None:
        return atom_mask.to(torch.uint8).contiguous()
    if not (torch.compiler.is_compiling() or proxy_tracing() or is_fake(atom_mask)):
        N = int(atom_mask.numel())
        if N == 0 or int(cap) >= N:
            return atom_mask.to(torch.uint8).contiguous()
    real = atom_mask.bool()
    ri = real.long()
    counts = torch.zeros(n_graphs, dtype=torch.long, device=graph.device).index_add(0, graph, ri)
    start = torch.cumsum(counts, 0) - counts
    rank = torch.cumsum(ri, 0) - 1 - start[graph]
    return (real & (rank < cap)).to(torch.uint8).contiguous()


@torch.library.custom_op("lam4::metal_coulomb_fwd", mutates_args=())
def _metal_coulomb_fwd_op(
    pos: Tensor, q: Tensor, mask_u8: Tensor, graph_i64: Tensor, group_ptr: Tensor,
    perm: Tensor, bsph: Tensor, pos_s: Tensor, mask_s: Tensor,
    n_graphs: int, sigma: float, need_field: bool,
) -> tuple[Tensor, Tensor]:
    n_atoms = int(pos.shape[0])
    pos_c, q_c = pos.to(torch.float32).contiguous(), q.to(torch.float32).contiguous()
    phi = torch.empty(n_atoms, dtype=torch.float32, device=pos.device)
    fld = torch.empty((n_atoms, 3), dtype=torch.float32, device=pos.device) if need_field else pos_c.new_empty((0, 3))
    if n_atoms == 0:
        return phi, fld
    lib = get_metal_coulomb_lib()
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
        out_phi = torch.empty(n_pad, dtype=torch.float32, device=pos.device)
        out_fld = torch.empty(n_pad * 3, dtype=torch.float32, device=pos.device) if need_field else pos_c[:1]
        lib.coulomb_fast64_fwd(
            pq_s, mask_s, bsph, out_phi, out_fld, params, n_tiles, int(need_field),
            threads=(n_pad, 1, 1), group_size=(64, 1, 1),
        )
        phi.index_copy_(0, perm, out_phi[:n_atoms])
        if need_field:
            fld.index_copy_(0, perm, out_fld[:n_atoms * 3].view(n_atoms, 3))
        return phi, fld
    tg = 256
    tot = ((n_atoms + tg - 1) // tg) * tg
    dummy_fld = fld if need_field else pos_c[:1]
    if int(n_graphs) == 1 and n_atoms >= 192:
        lib.coulomb_fwd_tiled(pos_c, q_c, mask_u8, phi, dummy_fld, n_atoms, int(need_field), params, threads=(tot, 1, 1), group_size=(tg, 1, 1))
    else:
        lib.coulomb_fwd_grouped(pos_c, q_c, mask_u8, graph_i64, group_ptr, phi, dummy_fld, n_atoms, int(need_field), params, threads=(tot, 1, 1), group_size=(tg, 1, 1))
    return phi, fld


@_metal_coulomb_fwd_op.register_fake
def _(
    pos: Tensor, q: Tensor, mask_u8: Tensor, graph_i64: Tensor, group_ptr: Tensor,
    perm: Tensor, bsph: Tensor, pos_s: Tensor, mask_s: Tensor,
    n_graphs: int, sigma: float, need_field: bool,
) -> tuple[Tensor, Tensor]:
    fld = torch.empty((pos.shape[0], 3), dtype=torch.float32, device=pos.device) if need_field else pos.new_empty((0, 3), dtype=torch.float32)
    return torch.empty_like(q, dtype=torch.float32), fld


@torch.library.custom_op("lam4::metal_coulomb_bwd", mutates_args=())
def _metal_coulomb_bwd_op(
    pos: Tensor, q: Tensor, mask_u8: Tensor, graph_i64: Tensor, group_ptr: Tensor,
    perm: Tensor, bsph: Tensor, pos_s: Tensor, mask_s: Tensor,
    g_phi: Tensor, g_fld: Tensor,
    n_graphs: int, sigma: float, need_field: bool, has_gp: bool, has_gf: bool, need_dpos: bool, need_dq: bool,
) -> tuple[Tensor, Tensor]:
    n_atoms = int(pos.shape[0])
    if n_atoms == 0 or not (need_dpos or need_dq) or not (has_gp or (need_field and has_gf)):
        return torch.zeros((n_atoms, 3), dtype=torch.float32, device=pos.device), torch.zeros(n_atoms, dtype=torch.float32, device=pos.device)
    lib = get_metal_coulomb_lib()
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
        g_s = torch.zeros((n_pad, 4), dtype=torch.float32, device=pos.device)
        if need_field and has_gf:
            g_s[:n_atoms, :3] = (g_fld.contiguous() * mf.unsqueeze(-1)).index_select(0, perm)
        if has_gp:
            g_s[:n_atoms, 3] = (g_phi.contiguous() * mf).index_select(0, perm)
        dp_s = torch.empty(n_pad * 3, dtype=torch.float32, device=pos.device) if need_dpos else pq_s[:1, 0]
        dq_s = torch.empty(n_pad, dtype=torch.float32, device=pos.device) if need_dq else pq_s[:1, 0]
        lib.coulomb_fast64_bwd(
            pq_s, g_s, mask_s, bsph, dp_s, dq_s, params, n_tiles, int(need_dpos), int(need_dq),
            threads=(n_pad, 1, 1), group_size=(64, 1, 1),
        )
        d_pos_out = torch.zeros((n_atoms, 3), dtype=torch.float32, device=pos.device)
        d_q_out = torch.zeros(n_atoms, dtype=torch.float32, device=pos.device)
        if need_dpos:
            d_pos_out.index_copy_(0, perm, dp_s[:n_atoms * 3].view(n_atoms, 3))
        if need_dq:
            d_q_out.index_copy_(0, perm, dq_s[:n_atoms])
        return d_pos_out, d_q_out
    gp_c = g_phi.contiguous() if has_gp else torch.zeros_like(q_c)
    has_gf_eff = int(need_field and has_gf)
    gf_c = g_fld.contiguous() if has_gf_eff else pos_c[:1]
    d_pos_buf = torch.empty_like(pos_c) if need_dpos else pos_c[:1]
    d_q_buf = torch.empty_like(q_c) if need_dq else q_c[:1]
    tg = 256
    tot = ((n_atoms + tg - 1) // tg) * tg
    if int(n_graphs) == 1 and n_atoms >= 192:
        lib.coulomb_bwd_tiled(pos_c, q_c, mask_u8, gp_c, gf_c, d_pos_buf, d_q_buf, n_atoms, has_gf_eff, int(need_dpos), int(need_dq), params, threads=(tot, 1, 1), group_size=(tg, 1, 1))
    else:
        lib.coulomb_bwd_grouped(pos_c, q_c, mask_u8, graph_i64, group_ptr, gp_c, gf_c, d_pos_buf, d_q_buf, n_atoms, has_gf_eff, int(need_dpos), int(need_dq), params, threads=(tot, 1, 1), group_size=(tg, 1, 1))
    return (d_pos_buf if need_dpos else torch.zeros_like(pos_c)), (d_q_buf if need_dq else torch.zeros_like(q_c))


@_metal_coulomb_bwd_op.register_fake
def _(
    pos: Tensor, q: Tensor, mask_u8: Tensor, graph_i64: Tensor, group_ptr: Tensor,
    perm: Tensor, bsph: Tensor, pos_s: Tensor, mask_s: Tensor,
    g_phi: Tensor, g_fld: Tensor,
    n_graphs: int, sigma: float, need_field: bool, has_gp: bool, has_gf: bool, need_dpos: bool, need_dq: bool,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(pos, dtype=torch.float32), torch.empty_like(q, dtype=torch.float32)


def _setup_mc_ctx(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    ctx.save_for_backward(*inputs[:9])
    ctx.n_graphs, ctx.sigma, ctx.need_field = int(inputs[9]), float(inputs[10]), bool(inputs[11])


def _bwd_mc_autograd(ctx: Any, g_phi: Tensor | None, g_fld: Tensor | None):
    pos, q, mask_u8, graph_i64, group_ptr, perm, bsph, pos_s, mask_s = ctx.saved_tensors
    need_dp, need_dq = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
    if not (need_dp or need_dq):
        return (None,) * 12
    gp = g_phi.contiguous() if g_phi is not None else q[:1]
    gf = g_fld.contiguous() if (ctx.need_field and g_fld is not None) else pos[:1]
    dp, dq = _metal_coulomb_bwd_op(
        pos, q, mask_u8, graph_i64, group_ptr, perm, bsph, pos_s, mask_s, gp, gf,
        ctx.n_graphs, ctx.sigma, ctx.need_field, g_phi is not None, g_fld is not None, need_dp, need_dq,
    )
    return (dp.to(pos.dtype) if need_dp else None, dq.to(q.dtype) if need_dq else None) + (None,) * 10


def _mc_bwd2_unsupported(*_args: Any, **_kw: Any):
    raise NotImplementedError("lam4::metal_coulomb_bwd supports 1st-order backward only; route create_graph=True to ExactCoulomb.")


_metal_coulomb_fwd_op.register_autograd(_bwd_mc_autograd, setup_context=_setup_mc_ctx)
_metal_coulomb_bwd_op.register_autograd(_mc_bwd2_unsupported, setup_context=lambda ctx, inputs, output: None)


class MetalExactCoulomb:
    """Drop-in Tier-2 replacement for `ExactCoulomb` on MPS (`O(N)` memory, zero padding waste)."""

    def __init__(
        self,
        pos: Tensor,
        atom_mask: Tensor,
        graph: Tensor,
        n_graphs: int,
        cap: int | None,
        sigma: float = 2.0,
        *,
        group_ptr: Tensor | None = None,
        mask_u8: Tensor | None = None,
    ) -> None:
        self.pos = pos
        sym = torch.compiler.is_compiling() or proxy_tracing() or is_fake(pos)
        self.n_atoms = pos.shape[0] if sym else int(pos.shape[0])
        self.n_graphs = n_graphs if sym else int(n_graphs)
        self.sigma = float(sigma)
        self.graph_i64 = graph.contiguous()
        self.group_ptr = build_group_ptr(self.graph_i64, self.n_graphs) if group_ptr is None else group_ptr
        self.mask_u8 = build_coulomb_mask(atom_mask, self.graph_i64, self.n_graphs, cap) if mask_u8 is None else mask_u8
        if not sym and self.n_graphs == 1 and self.n_atoms >= 1024:
            self._fast_meta = _build_fast64_meta(pos, self.mask_u8)
        elif sym and self.n_graphs == 1:
            perm, bsph, pos_s, mask_s = _metal_coulomb_fast64_meta_op(pos.detach(), self.mask_u8)
            self._fast_meta = (perm, bsph, pos_s, mask_s, 0, 0)
        else:
            self._fast_meta = None

    def _fast_tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self._fast_meta is not None:
            perm, bsph, pos_s, mask_s, _, _ = self._fast_meta
            return perm, bsph, pos_s, mask_s
        dev = self.pos.device
        return (
            torch.empty(0, dtype=torch.int64, device=dev),
            torch.empty((0, 4), dtype=torch.float32, device=dev),
            torch.empty((0, 3), dtype=torch.float32, device=dev),
            torch.empty(0, dtype=torch.uint8, device=dev),
        )

    def phi(self, q: Tensor) -> Tensor:
        phi, _ = _metal_coulomb_fwd_op(
            self.pos, q, self.mask_u8, self.graph_i64, self.group_ptr,
            *self._fast_tensors(), self.n_graphs, self.sigma, False,
        )
        return phi

    def phi_field(self, q: Tensor) -> tuple[Tensor, Tensor]:
        return _metal_coulomb_fwd_op(
            self.pos, q, self.mask_u8, self.graph_i64, self.group_ptr,
            *self._fast_tensors(), self.n_graphs, self.sigma, True,
        )

    def phi_field_efg(self, q: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        from lam4.kernels.metal_coulomb_efg import eval_metal_coulomb_efg, has_metal_coulomb_efg

        if has_metal_coulomb_efg():
            return eval_metal_coulomb_efg(
                self.pos, q, self.mask_u8, self.graph_i64, self.group_ptr,
                self.n_graphs, self.sigma, self._fast_meta,
            )
        from lam4.physics.coulomb_exact import ChunkedExactCoulomb

        return ChunkedExactCoulomb(
            self.pos, self.mask_u8 != 0, self.sigma,
            graph=(None if self.n_graphs <= 1 else self.graph_i64), n_graphs=self.n_graphs,
        ).phi_field_efg(q)

    def phi_field_multipole(self, q: Tensor, p: Tensor | None = None, *, need_efg: bool = False) -> tuple[Tensor, Tensor, Tensor | None]:
        if p is not None:
            raise ValueError("MetalExactCoulomb.phi_field_multipole supports monopole-only (p=None); use OpenMultipoleBackend when sqe_pqeq=True.")
        if need_efg:
            return self.phi_field_efg(q)
        phi, fld = self.phi_field(q)
        return phi, fld, None
