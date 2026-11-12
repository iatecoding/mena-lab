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
from torch.autograd.function import once_differentiable

from lam4.kernels.metal_coulomb_src import METAL_COULOMB_MSL
from lam4.kernels.metal_csr import has_metal_csr

_COULOMB_LIB: Any = None
_PTR_G1_CACHE: dict[tuple[str, int | None, int], Tensor] = {}


def has_metal_coulomb() -> bool:
    return has_metal_csr()


def get_metal_coulomb_lib() -> Any:
    global _COULOMB_LIB
    if _COULOMB_LIB is None:
        _COULOMB_LIB = torch.mps.compile_shader(METAL_COULOMB_MSL)
    return _COULOMB_LIB


def _coulomb_params(sigma: float) -> tuple[float, float, float, float]:
    inv_sig = 1.0 / float(sigma)
    c0 = 2.0 * inv_sig / math.sqrt(math.pi)
    return (inv_sig, c0, inv_sig * inv_sig, 0.0)


def build_group_ptr(graph: Tensor, n_graphs: int) -> Tensor:
    """Zero-sync O(log N) GPU group pointer builder `[G + 1]` int32 for sorted `graph [N]`."""
    dev = graph.device
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


def build_coulomb_mask(atom_mask: Tensor, graph: Tensor, n_graphs: int, cap: int) -> Tensor:
    """Active-atom uint8 mask matching `make_slots_static` (`atom_mask & (rank < cap)`)."""
    N = int(atom_mask.numel())
    if N == 0 or cap >= N:
        return atom_mask.to(torch.uint8).contiguous()
    real = atom_mask.bool()
    ri = real.long()
    counts = torch.zeros(int(n_graphs), dtype=torch.long, device=graph.device).index_add(0, graph, ri)
    start = torch.cumsum(counts, 0) - counts
    rank = torch.cumsum(ri, 0) - 1 - start[graph]
    return (real & (rank < cap)).to(torch.uint8).contiguous()


class _MetalCoulombPhiFieldFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pos: Tensor,
        q: Tensor,
        mask_u8: Tensor,
        graph_i64: Tensor,
        group_ptr: Tensor,
        n_graphs: int,
        sigma: float,
        need_field: bool,
    ) -> tuple[Tensor, Tensor]:
        n_atoms = int(pos.shape[0])
        pos_c = pos.contiguous()
        q_c = q.contiguous()
        phi = torch.empty(n_atoms, dtype=torch.float32, device=pos.device)
        fld = torch.empty((n_atoms, 3), dtype=torch.float32, device=pos.device) if need_field else pos.new_empty((0, 3))
        if n_atoms > 0:
            lib = get_metal_coulomb_lib()
            tg = 256
            tot = ((n_atoms + tg - 1) // tg) * tg
            params = _coulomb_params(sigma)
            dummy_fld = fld if need_field else pos_c[:1]
            if n_graphs == 1 and n_atoms >= 192:
                lib.coulomb_fwd_tiled(
                    pos_c, q_c, mask_u8, phi, dummy_fld, n_atoms, int(need_field), params,
                    threads=(tot, 1, 1), group_size=(tg, 1, 1),
                )
            else:
                lib.coulomb_fwd_grouped(
                    pos_c, q_c, mask_u8, graph_i64, group_ptr, phi, dummy_fld, n_atoms, int(need_field), params,
                    threads=(tot, 1, 1), group_size=(tg, 1, 1),
                )
        ctx.save_for_backward(pos_c, q_c, mask_u8, graph_i64, group_ptr)
        ctx.n_graphs = int(n_graphs)
        ctx.sigma = float(sigma)
        ctx.need_field = bool(need_field)
        return phi, fld

    @staticmethod
    @once_differentiable
    def backward(ctx, g_phi: Tensor | None, g_fld: Tensor | None):
        pos_c, q_c, mask_u8, graph_i64, group_ptr = ctx.saved_tensors
        need_dpos = bool(ctx.needs_input_grad[0])
        need_dq = bool(ctx.needs_input_grad[1])
        n_atoms = int(pos_c.shape[0])
        if not (need_dpos or need_dq) or n_atoms == 0:
            d_pos = torch.zeros_like(pos_c) if need_dpos else None
            d_q = torch.zeros_like(q_c) if need_dq else None
            return d_pos, d_q, None, None, None, None, None, None

        gp_c = g_phi.contiguous() if g_phi is not None else torch.zeros_like(q_c)
        has_gf = int(ctx.need_field and g_fld is not None)
        gf_c = g_fld.contiguous() if has_gf else pos_c[:1]
        d_pos_buf = torch.empty_like(pos_c) if need_dpos else pos_c[:1]
        d_q_buf = torch.empty_like(q_c) if need_dq else q_c[:1]

        lib = get_metal_coulomb_lib()
        tg = 256
        tot = ((n_atoms + tg - 1) // tg) * tg
        params = _coulomb_params(ctx.sigma)
        if ctx.n_graphs == 1 and n_atoms >= 192:
            lib.coulomb_bwd_tiled(
                pos_c, q_c, mask_u8, gp_c, gf_c, d_pos_buf, d_q_buf,
                n_atoms, has_gf, int(need_dpos), int(need_dq), params,
                threads=(tot, 1, 1), group_size=(tg, 1, 1),
            )
        else:
            lib.coulomb_bwd_grouped(
                pos_c, q_c, mask_u8, graph_i64, group_ptr, gp_c, gf_c, d_pos_buf, d_q_buf,
                n_atoms, has_gf, int(need_dpos), int(need_dq), params,
                threads=(tot, 1, 1), group_size=(tg, 1, 1),
            )
        return (
            d_pos_buf if need_dpos else None,
            d_q_buf if need_dq else None,
            None, None, None, None, None, None,
        )


class MetalExactCoulomb:
    """Drop-in Tier-2 replacement for `ExactCoulomb` on MPS (`O(N)` memory, zero padding waste)."""

    def __init__(
        self,
        pos: Tensor,
        atom_mask: Tensor,
        graph: Tensor,
        n_graphs: int,
        cap: int,
        sigma: float = 2.0,
        *,
        group_ptr: Tensor | None = None,
        mask_u8: Tensor | None = None,
    ) -> None:
        self.pos = pos
        self.n_atoms = int(pos.shape[0])
        self.n_graphs = int(n_graphs)
        self.sigma = float(sigma)
        self.graph_i64 = graph.contiguous()
        self.group_ptr = build_group_ptr(self.graph_i64, self.n_graphs) if group_ptr is None else group_ptr
        self.mask_u8 = build_coulomb_mask(atom_mask, self.graph_i64, self.n_graphs, cap) if mask_u8 is None else mask_u8

    def phi(self, q: Tensor) -> Tensor:
        phi, _ = _MetalCoulombPhiFieldFn.apply(
            self.pos, q, self.mask_u8, self.graph_i64, self.group_ptr, self.n_graphs, self.sigma, False
        )
        return phi

    def phi_field(self, q: Tensor) -> tuple[Tensor, Tensor]:
        return _MetalCoulombPhiFieldFn.apply(
            self.pos, q, self.mask_u8, self.graph_i64, self.group_ptr, self.n_graphs, self.sigma, True
        )
