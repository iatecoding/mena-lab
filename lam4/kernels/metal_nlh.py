"""Zero-sync Metal Dual-CSR kernel for NLH short-range repulsion (`nlh_atom_energy` + forces + virial)."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from lam4.core.consts import storable
from lam4.kernels.csr_plan import MAX_DOFF_R2, CSRPlan
from lam4.kernels.metal_csr import FLAG_CELL, FLAG_G1, FLAG_VIR, _dummy_f32, _dummy_i64, has_metal_csr
from lam4.physics.nlh_tables import nlh_tables_f64

_NLH_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant int FLAG_CELL = 8;
constant int FLAG_G1   = 16;
constant int FLAG_VIR  = 32;

inline float3 nlh_shift(device const float* shift_in, device const float* cell, int e, int g_id, int flags) {
    float3 es = float3(shift_in[3 * e], shift_in[3 * e + 1], shift_in[3 * e + 2]);
    if ((flags & FLAG_CELL) == 0) return es;
    device const float* c = cell + 9 * g_id;
    return float3(es.x * c[0] + es.y * c[3] + es.z * c[6],
                  es.x * c[1] + es.y * c[4] + es.z * c[7],
                  es.x * c[2] + es.y * c[5] + es.z * c[8]);
}

inline float3 nlh_grad_d(float3 d, float r2, int pair_idx, device const float* cb, device const float* jet) {
    float d_off = jet[4 * pair_idx + 0];
    if (d_off <= 0.0f) return float3(0.0f);
    float r = sqrt(r2 + 1e-8f);
    if (r >= d_off) return float3(0.0f);
    device const float* p = cb + 6 * pair_idx;
    float e0 = p[0] * exp(-p[3] * r), e1 = p[1] * exp(-p[4] * r), e2 = p[2] * exp(-p[5] * r);
    float inv_r = 1.0f / r, dr = r - d_off;
    float de_nlh = -(e0 * (p[3] * r + 1.0f) + e1 * (p[4] * r + 1.0f) + e2 * (p[5] * r + 1.0f)) * (inv_r * inv_r);
    float de_dr = 0.5f * (de_nlh - (jet[4 * pair_idx + 2] + dr * jet[4 * pair_idx + 3]));
    return (de_dr * inv_r) * d;
}

kernel void csr_nlh_fwd(
    device const float* pos [[buffer(0)]], device const int* recv_ptr [[buffer(1)]],
    device const int* send_idx [[buffer(2)]], device const float* shift_in [[buffer(3)]],
    device const float* cell [[buffer(4)]], device const long* graph [[buffer(5)]],
    device const long* z [[buffer(6)]], device const float* cb [[buffer(7)]],
    device const float* jet [[buffer(8)]], device float* out_e [[buffer(9)]],
    constant int& n_atoms [[buffer(10)]], constant int& flags [[buffer(11)]],
    constant float& max_doff_r2 [[buffer(12)]], uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms) return;
    int zi_s = int(clamp(z[i], 0L, 127L)) << 7;
    int g_id = ((flags & FLAG_CELL) != 0 && (flags & FLAG_G1) == 0) ? int(graph[i]) : 0;
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float acc = 0.0f;
    for (int e = recv_ptr[i], e_end = recv_ptr[i + 1]; e < e_end; ++e) {
        int j = send_idx[e];
        float3 d = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]) - pi + nlh_shift(shift_in, cell, e, g_id, flags);
        float r2 = dot(d, d);
        if (r2 >= max_doff_r2) continue;
        int pair_idx = zi_s + int(clamp(z[j], 0L, 127L));
        float d_off = jet[4 * pair_idx + 0];
        if (d_off <= 0.0f) continue;
        float r = sqrt(r2 + 1e-8f);
        if (r >= d_off) continue;
        device const float* p = cb + 6 * pair_idx;
        float e_nlh = (p[0] * exp(-p[3] * r) + p[1] * exp(-p[4] * r) + p[2] * exp(-p[5] * r)) / r;
        float dr = r - d_off;
        acc += 0.5f * (e_nlh - (jet[4 * pair_idx + 1] + dr * (jet[4 * pair_idx + 2] + 0.5f * dr * jet[4 * pair_idx + 3])));
    }
    out_e[i] = acc;
}

kernel void csr_nlh_bwd(
    device const float* pos [[buffer(0)]], device const int* recv_ptr [[buffer(1)]],
    device const int* send_idx [[buffer(2)]], device const int* send_ptr [[buffer(3)]],
    device const int* send_perm [[buffer(4)]], device const int* recv_idx [[buffer(5)]],
    device const float* shift_in [[buffer(6)]], device const float* cell [[buffer(7)]],
    device const long* graph [[buffer(8)]], device const long* z [[buffer(9)]],
    device const float* cb [[buffer(10)]], device const float* jet [[buffer(11)]],
    device const float* g_out [[buffer(12)]], device float* d_pos [[buffer(13)]],
    device float* vir_atom [[buffer(14)]], constant int& n_atoms [[buffer(15)]],
    constant int& flags [[buffer(16)]], constant float& max_doff_r2 [[buffer(17)]],
    uint tid [[thread_position_in_grid]]
) {
    int a = int(tid);
    if (a >= n_atoms) return;
    int za = int(clamp(z[a], 0L, 127L));
    int za_s = za << 7;
    int g_a = ((flags & FLAG_CELL) != 0 && (flags & FLAG_G1) == 0) ? int(graph[a]) : 0;
    float3 pa = float3(pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]);
    float ga = g_out[a];
    float3 dp = float3(0.0f);
    bool need_vir = (flags & FLAG_VIR) != 0;
    float vir[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int e = recv_ptr[a], e_end = recv_ptr[a + 1]; e < e_end; ++e) {
        int j = send_idx[e];
        float3 d = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]) - pa + nlh_shift(shift_in, cell, e, g_a, flags);
        float r2 = dot(d, d);
        if (r2 >= max_doff_r2) continue;
        float3 gd = ga * nlh_grad_d(d, r2, za_s + int(clamp(z[j], 0L, 127L)), cb, jet);
        dp -= gd;
        if (need_vir) {
            float es0 = shift_in[3 * e], es1 = shift_in[3 * e + 1], es2 = shift_in[3 * e + 2];
            vir[0] += es0 * gd.x; vir[1] += es0 * gd.y; vir[2] += es0 * gd.z;
            vir[3] += es1 * gd.x; vir[4] += es1 * gd.y; vir[5] += es1 * gd.z;
            vir[6] += es2 * gd.x; vir[7] += es2 * gd.y; vir[8] += es2 * gd.z;
        }
    }
    for (int kp = send_ptr[a], k_end = send_ptr[a + 1]; kp < k_end; ++kp) {
        int e = send_perm[kp], i = recv_idx[e];
        int g_i = ((flags & FLAG_CELL) != 0 && (flags & FLAG_G1) == 0) ? int(graph[i]) : 0;
        float3 d = pa - float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) + nlh_shift(shift_in, cell, e, g_i, flags);
        float r2 = dot(d, d);
        if (r2 >= max_doff_r2) continue;
        dp += g_out[i] * nlh_grad_d(d, r2, (int(clamp(z[i], 0L, 127L)) << 7) + za, cb, jet);
    }
    d_pos[3 * a + 0] = dp.x; d_pos[3 * a + 1] = dp.y; d_pos[3 * a + 2] = dp.z;
    if (need_vir) {
        for (int m = 0; m < 9; ++m) vir_atom[9 * a + m] = vir[m];
    }
}
"""

_NLH_LIB: Any = None
_NLH_TABS: dict[tuple[str, int | None], tuple[Tensor, Tensor]] = {}


def _nlh_lib() -> Any:
    global _NLH_LIB
    if _NLH_LIB is None:
        _NLH_LIB = torch.mps.compile_shader(_NLH_MSL)
    return _NLH_LIB


def _nlh_cb_jet_f32(dev: torch.device) -> tuple[Tensor, Tensor]:
    key = (dev.type, dev.index)
    t = _NLH_TABS.get(key)
    if t is None:
        cb64, _, jet64 = nlh_tables_f64()
        cb = torch.from_numpy(cb64).to(dtype=torch.float32).contiguous().to(dev)
        jet = torch.from_numpy(jet64).to(dtype=torch.float32).contiguous().to(dev)
        t = (cb, jet)
        if storable(cb) and storable(jet):
            _NLH_TABS[key] = t
    return t


class _MetalNLHAgg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan: CSRPlan, edge_shift: Tensor, graph: Tensor, z: Tensor, pos: Tensor, cell: Tensor) -> Tensor:
        dev = pos.device
        n = plan.n_atoms
        if n == 0 or plan.n_edges == 0:
            ctx.empty = True
            ctx.save_for_backward(pos, cell)
            return torch.zeros(n, dtype=torch.float32, device=dev)
        ctx.empty = False
        ctx.plan = plan
        ctx.save_for_backward(edge_shift, graph, z, pos, cell)
        cb, jet = _nlh_cb_jet_f32(dev)
        flags = FLAG_CELL | (FLAG_G1 if cell.shape[0] == 1 else 0)
        out = torch.empty(n, dtype=torch.float32, device=dev)
        tg = 128
        tot = ((n + tg - 1) // tg) * tg
        _nlh_lib().csr_nlh_fwd(
            pos, plan.recv_ptr, plan.send_idx, edge_shift, cell,
            graph if graph.numel() else _dummy_i64(dev), z, cb, jet, out,
            int(n), int(flags), float(MAX_DOFF_R2), threads=(tot, 1, 1), group_size=(tg, 1, 1),
        )
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, g_out: Tensor):
        if ctx.empty:
            pos, cell = ctx.saved_tensors
            return (
                None, None, None, None,
                torch.zeros_like(pos) if ctx.needs_input_grad[4] else None,
                torch.zeros_like(cell) if ctx.needs_input_grad[5] else None,
            )
        edge_shift, graph, z, pos, cell = ctx.saved_tensors
        plan = ctx.plan
        dev, n = pos.device, plan.n_atoms
        need_vir = bool(ctx.needs_input_grad[5])
        cb, jet = _nlh_cb_jet_f32(dev)
        flags = FLAG_CELL | (FLAG_G1 if cell.shape[0] == 1 else 0) | (FLAG_VIR if need_vir else 0)
        d_pos = torch.empty_like(pos)
        vir_atom = torch.empty((n, 9), dtype=torch.float32, device=dev) if need_vir else _dummy_f32(dev)
        tg = 128
        tot = ((n + tg - 1) // tg) * tg
        _nlh_lib().csr_nlh_bwd(
            pos, plan.recv_ptr, plan.send_idx, plan.send_ptr, plan.send_perm, plan.recv_idx,
            edge_shift, cell, graph if graph.numel() else _dummy_i64(dev), z, cb, jet,
            g_out.contiguous(), d_pos, vir_atom, int(n), int(flags), float(MAX_DOFF_R2),
            threads=(tot, 1, 1), group_size=(tg, 1, 1),
        )
        d_cell = None
        if need_vir:
            if cell.shape[0] == 1:
                d_cell = vir_atom.sum(dim=0).view(1, 3, 3)
            else:
                d_cell = cell.new_zeros(cell.shape[0], 9).index_add(0, graph, vir_atom).view_as(cell)
        return None, None, None, None, d_pos if ctx.needs_input_grad[4] else None, d_cell


def metal_nlh_atom_energy(plan: CSRPlan, pos: Tensor, edge_shift: Tensor, cell: Tensor, graph: Tensor, z: Tensor) -> Tensor:
    if not has_metal_csr():
        raise RuntimeError("Metal CSR is unavailable")
    g_c = graph.to(torch.int64).contiguous() if cell.shape[0] > 1 else pos.new_zeros(0, dtype=torch.int64)
    return _MetalNLHAgg.apply(
        plan, edge_shift.to(torch.float32).contiguous(), g_c, z.to(torch.int64).contiguous(), pos.contiguous(), cell.contiguous()
    )
