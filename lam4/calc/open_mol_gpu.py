"""GPU-resident open-molecule neighbor-list tiling and batched D3(BJ)/D3(0) dispatch."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

OPEN_MOL_GPU_NL_MAX_ATOMS = 2048
OPEN_MOL_GPU_D3_CSR_MAX_ATOMS = 4096


def _open_mol_edges_single(
    pos_g: Tensor, start_g: int, r_cut2: float, pair_budget: int,
    send_parts: list[Tensor], recv_parts: list[Tensor],
) -> None:
    """Append receiver-sorted intra-graph edges for one open molecule using bounded-memory row tiles."""
    c_g = int(pos_g.shape[0])
    if c_g <= 1:
        return
    dev = pos_g.device
    row_block = max(1, min(c_g, pair_budget // c_g))
    cols = torch.arange(c_g, dtype=torch.long, device=dev).unsqueeze(0)
    for i0 in range(0, c_g, row_block):
        i1 = min(i0 + row_block, c_g)
        d2 = (pos_g[i0:i1].unsqueeze(1) - pos_g.unsqueeze(0)).pow(2).sum(dim=-1)
        rows = torch.arange(i0, i1, dtype=torch.long, device=dev).unsqueeze(1)
        valid = (d2 < r_cut2) & (rows != cols)
        r_loc, c_loc = torch.nonzero(valid, as_tuple=True)
        if r_loc.numel() > 0:
            recv_parts.append(r_loc + (start_g + i0))
            send_parts.append(c_loc + start_g)


def _open_mol_edges_small_batch(
    pos_det: Tensor, graph: Tensor, sub_counts: list[int], st0: int, end0: int,
    g0: int, r_cut2: float, send_parts: list[Tensor], recv_parts: list[Tensor],
) -> None:
    """Append receiver-sorted intra-graph edges for a contiguous slice `[g0, g0 + len(sub_counts))` of small graphs."""
    dev = pos_det.device
    b_g = len(sub_counts)
    sub_max = max(sub_counts)
    counts_t = torch.as_tensor(sub_counts, dtype=torch.long, device=dev)
    starts_sub = torch.cumsum(counts_t, dim=0) - counts_t
    g_sub = graph[st0:end0] - g0
    loc_sub = torch.arange(end0 - st0, dtype=torch.long, device=dev) - starts_sub.index_select(0, g_sub)
    p_pad = pos_det.new_zeros((b_g, sub_max, 3))
    p_pad[g_sub, loc_sub] = pos_det[st0:end0]
    ar = torch.arange(sub_max, dtype=torch.long, device=dev)
    mask = ar.unsqueeze(0) < counts_t.unsqueeze(1)
    d2 = (p_pad.unsqueeze(2) - p_pad.unsqueeze(1)).pow(2).sum(dim=-1)
    eye = torch.eye(sub_max, dtype=torch.bool, device=dev).unsqueeze(0)
    valid = mask.unsqueeze(2) & mask.unsqueeze(1) & (~eye) & (d2 < r_cut2)
    g_idx, i_loc, j_loc = torch.nonzero(valid, as_tuple=True)
    if g_idx.numel() > 0:
        st_g = starts_sub.index_select(0, g_idx) + st0
        recv_parts.append(st_g + i_loc)
        send_parts.append(st_g + j_loc)


@torch.no_grad()
def build_open_mol_edges_gpu(
    pos_cat: Tensor, graph: Tensor, counts: list[int] | np.ndarray, r_cut: float, dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Build receiver-sorted `(edge_index [2, E], edge_shift [E, 3])` on GPU for open molecules via bounded-memory row tiling."""
    dev = pos_cat.device
    counts_list = [int(c) for c in counts]
    G = len(counts_list)
    N = int(pos_cat.shape[0])
    max_n = max(counts_list, default=0)
    if N == 0 or max_n <= 1 or r_cut <= 0.0:
        return torch.zeros((2, 0), dtype=torch.long, device=dev), torch.zeros((0, 3), dtype=dtype, device=dev)

    pos_det = pos_cat.detach()
    r_cut2 = float(r_cut * r_cut)
    # Transient `(diff, diff^2, d2, valid)` uses ~7 * elem_size + 2 bytes per pair; cap at ~16 MiB.
    bytes_per_pair = 7 * max(1, int(pos_det.element_size())) + 2
    pair_budget = max(65_536, (16 * 1024 * 1024) // bytes_per_pair)

    starts_list: list[int] = []
    cur = 0
    for c in counts_list:
        starts_list.append(cur)
        cur += c

    send_parts: list[Tensor] = []
    recv_parts: list[Tensor] = []
    small_g0: int | None = None
    small_max_c = 0
    small_sum_c2 = 0

    def _flush_small(g_end: int) -> None:
        nonlocal small_g0, small_max_c, small_sum_c2
        if small_g0 is None:
            return
        g0 = small_g0
        if g_end - g0 == 1:
            st0 = starts_list[g0]
            _open_mol_edges_single(pos_det[st0 : st0 + counts_list[g0]], st0, r_cut2, pair_budget, send_parts, recv_parts)
        else:
            st0 = starts_list[g0]
            end0 = starts_list[g_end - 1] + counts_list[g_end - 1]
            _open_mol_edges_small_batch(pos_det, graph, counts_list[g0:g_end], st0, end0, g0, r_cut2, send_parts, recv_parts)
        small_g0, small_max_c, small_sum_c2 = None, 0, 0

    for g, c_g in enumerate(counts_list):
        if c_g <= 1:
            _flush_small(g)
            continue
        if c_g <= 256:
            if small_g0 is None:
                small_g0, small_max_c, small_sum_c2 = g, c_g, c_g * c_g
                continue
            cand_len = g - small_g0 + 1
            cand_max = max(small_max_c, c_g)
            cand_sum2 = small_sum_c2 + c_g * c_g
            if cand_len * cand_max * cand_max <= min(pair_budget, 4 * max(cand_sum2, 65_536)):
                small_max_c, small_sum_c2 = cand_max, cand_sum2
                continue
            _flush_small(g)
            small_g0, small_max_c, small_sum_c2 = g, c_g, c_g * c_g
        else:
            _flush_small(g)
            st_g = starts_list[g]
            _open_mol_edges_single(pos_det[st_g : st_g + c_g], st_g, r_cut2, pair_budget, send_parts, recv_parts)

    _flush_small(G)
    if not recv_parts:
        return torch.zeros((2, 0), dtype=torch.long, device=dev), torch.zeros((0, 3), dtype=dtype, device=dev)
    recv = recv_parts[0] if len(recv_parts) == 1 else torch.cat(recv_parts, dim=0)
    send = send_parts[0] if len(send_parts) == 1 else torch.cat(send_parts, dim=0)
    return torch.stack([send, recv], dim=0), torch.zeros((recv.numel(), 3), dtype=dtype, device=dev)


@torch.no_grad()
def eval_batched_open_mol_d3(
    z: Tensor,
    pos: Tensor,
    graph: Tensor,
    counts: Sequence[int] | np.ndarray,
    d3_mode: str | None,
    *,
    group_ptr: Tensor | None = None,
    compute_forces: bool = True,
) -> tuple[Tensor | None, Tensor | None]:
    """Evaluate batched open-molecule D3 on GPU (`triton_d3` on CUDA, `metal_d3` on MPS) or return `(None, None)`."""
    if d3_mode is None:
        return None, None
    counts_list = [int(c) for c in counts]
    G = len(counts_list)
    if G == 0:
        return None, None
    dev, dt = pos.device, pos.dtype
    n_atoms = int(pos.shape[0])
    if n_atoms == 0:
        return torch.zeros(G, dtype=dt, device=dev), (torch.zeros((0, 3), dtype=dt, device=dev) if compute_forces else None)
    if dt != torch.float32:
        return None, None
    max_n = max(counts_list, default=0)

    if dev.type == "cuda" and d3_mode in ("bj_wb97m", "zero_pbe"):
        from lam4.calc import physics as _phys

        td3 = _phys._get_triton_d3()
        if td3 is not None and getattr(td3, "has_triton_d3", lambda *a: False)(dev):
            try:
                if max_n <= OPEN_MOL_GPU_NL_MAX_ATOMS:
                    ed3, fd3, _ = td3.triton_d3_grouped(
                        z, pos, graph, G, d3_mode=d3_mode, group_ptr=group_ptr,
                        compute_forces=compute_forces, compute_stress=False,
                    )
                    return ed3, (fd3 if (compute_forces and fd3 is not None) else None)
                if max_n <= OPEN_MOL_GPU_D3_CSR_MAX_ATOMS and hasattr(td3, "triton_build_csr_i32") and hasattr(td3, "triton_d3_csr"):
                    ei, _ = build_open_mol_edges_gpu(pos, graph, counts_list, 12.0, dt)
                    rp, send_i32 = td3.triton_build_csr_i32(ei[1].to(torch.int32), ei[0].to(torch.int32), n_atoms)
                    ed3, fd3, _ = td3.triton_d3_csr(
                        z, pos, rp, send_i32, d3_mode=d3_mode, graph=graph, n_graphs=G,
                        compute_forces=compute_forces, compute_stress=False,
                    )
                    return ed3, (fd3 if (compute_forces and fd3 is not None) else None)
            except Exception:  # noqa: BLE001
                pass

    if dev.type == "mps" and d3_mode in ("bj_wb97m", "zero_pbe") and max_n <= OPEN_MOL_GPU_D3_CSR_MAX_ATOMS:
        from lam4.kernels.metal_d3 import has_metal_d3, metal_build_csr_i32, metal_d3_csr, metal_d3_grouped

        if has_metal_d3():
            try:
                if max_n <= 1024:
                    ed3, fd3 = metal_d3_grouped(
                        z, pos, graph, G, d3_mode=d3_mode, group_ptr=group_ptr, compute_forces=compute_forces,
                    )
                    return ed3, (fd3 if (compute_forces and fd3 is not None) else None)
                ei, _ = build_open_mol_edges_gpu(pos, graph, counts_list, 12.0, dt)
                rp, send_i32 = metal_build_csr_i32(ei[1].to(torch.int32), ei[0].to(torch.int32), n_atoms)
                ed3, fd3 = metal_d3_csr(
                    z, pos, rp, send_i32, d3_mode=d3_mode, graph=graph, n_graphs=G, compute_forces=compute_forces,
                )
                return ed3, (fd3 if (compute_forces and fd3 is not None) else None)
            except Exception:  # noqa: BLE001
                pass

    return None, None
