"""Fused vectorized Metal SwiGLU & EnvModulatedSwiGLU kernels for compiled MPS FP32 inference.

    lam4::env_swiglu_proj_fwd(gv [N, 2H], h_env [N, 32], w_t [32, 2H], bias [2H]) -> y [N, H]
    lam4::env_swiglu_proj_bwd(gv, h_env, w_t, bias, gy [N, H]) -> (g_gv [N, 2H], g_h_env [N, 32])
    lam4::swiglu_act_fwd(gv [N, 2H]) -> y [N, H]
    lam4::swiglu_act_bwd(gv [N, 2H], gy [N, H]) -> g_gv [N, 2H]

Fusing the rank-32 env_out projection (`h_env @ W^T + bias`) directly into the SwiGLU activation eliminates BOTH
the `[N, 32] x [32, 2H]` forward/backward GEMMs AND the `[N, 2H]` intermediate/gradient DRAM traffic (~295 MB/stage
at N=8000). First-order autograd only: used when `allow_fused` is True (no create_graph), eval mode, frozen weights,
and `compiled_mps(x)` holds (`LAM4_MPS_NODE_FUSED != 0`).
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lam4.kernels.metal_node_flags import MetalShader, fused_node_device

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// 1 threadgroup (128 threads = 4 simdgroups) handles B=4 atoms across all H channels in tiles of 128.
// Loads h_env[4, 32] once into threadgroup memory and reads W_t[32, 2H] coalesced across all 128 threads, reused 4x.
kernel void env_swiglu_proj_fwd_k(
    device const float* gv [[buffer(0)]], device const float* h_env [[buffer(1)]],
    device const float* w_t [[buffer(2)]], device const float* bias [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant int& N [[buffer(5)]], constant int& H [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    int i0 = int(gid) * 4;
    if (i0 >= N) return;
    int nb = min(4, N - i0);
    threadgroup float Hs[4][32];
    int b_ld = int(tid) >> 5, r_ld = int(tid) & 31;
    Hs[b_ld][r_ld] = (b_ld < nb) ? h_env[(long)(i0 + b_ld) * 32 + r_ld] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int twoH = 2 * H;
    for (int j = int(tid); j < H; j += 128) {
        float bg = bias[j], bv = bias[H + j];
        float dg0 = bg, dg1 = bg, dg2 = bg, dg3 = bg;
        float dv0 = bv, dv1 = bv, dv2 = bv, dv3 = bv;
        for (int r = 0; r < 32; ++r) {
            float wg = w_t[r * twoH + j];
            float wv = w_t[r * twoH + H + j];
            float h0 = Hs[0][r], h1 = Hs[1][r], h2 = Hs[2][r], h3 = Hs[3][r];
            dg0 = fma(h0, wg, dg0); dv0 = fma(h0, wv, dv0);
            dg1 = fma(h1, wg, dg1); dv1 = fma(h1, wv, dv1);
            dg2 = fma(h2, wg, dg2); dv2 = fma(h2, wv, dv2);
            dg3 = fma(h3, wg, dg3); dv3 = fma(h3, wv, dv3);
        }
        float dgs[4] = {dg0, dg1, dg2, dg3};
        float dvs[4] = {dv0, dv1, dv2, dv3};
        for (int b = 0; b < nb; ++b) {
            long row2 = (long)(i0 + b) * twoH;
            float g = gv[row2 + j], v = gv[row2 + H + j];
            float u = g * (1.0f + 0.5f * tanh(dgs[b]));
            float sig = 1.0f / (1.0f + exp(-u));
            out[(long)(i0 + b) * H + j] = (u * sig) * (v * (1.0f + 0.5f * tanh(dvs[b])));
        }
    }
}

kernel void env_swiglu_proj_bwd_k(
    device const float* gv [[buffer(0)]], device const float* h_env [[buffer(1)]],
    device const float* w_t [[buffer(2)]], device const float* bias [[buffer(3)]],
    device const float* gy [[buffer(4)]], device float* g_gv [[buffer(5)]],
    device float* g_h_env [[buffer(6)]],
    constant int& N [[buffer(7)]], constant int& H [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    int i0 = int(gid) * 4;
    if (i0 >= N) return;
    int nb = min(4, N - i0);
    threadgroup float Hs[4][32];
    threadgroup float GH[4][4][32];  // [sg][b][r]
    int b_ld = int(tid) >> 5, r_ld = int(tid) & 31;
    Hs[b_ld][r_ld] = (b_ld < nb) ? h_env[(long)(i0 + b_ld) * 32 + r_ld] : 0.0f;
    for (int s = 0; s < 4; ++s) GH[s][b_ld][r_ld] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int twoH = 2 * H;
    for (int j0 = 0; j0 < H; j0 += 128) {
        int j = j0 + int(tid);
        bool valid = j < H;
        float gdg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float gdv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (valid) {
            float bg = bias[j], bv = bias[H + j];
            float dg0 = bg, dg1 = bg, dg2 = bg, dg3 = bg;
            float dv0 = bv, dv1 = bv, dv2 = bv, dv3 = bv;
            for (int r = 0; r < 32; ++r) {
                float wg = w_t[r * twoH + j];
                float wv = w_t[r * twoH + H + j];
                float h0 = Hs[0][r], h1 = Hs[1][r], h2 = Hs[2][r], h3 = Hs[3][r];
                dg0 = fma(h0, wg, dg0); dv0 = fma(h0, wv, dv0);
                dg1 = fma(h1, wg, dg1); dv1 = fma(h1, wv, dv1);
                dg2 = fma(h2, wg, dg2); dv2 = fma(h2, wv, dv2);
                dg3 = fma(h3, wg, dg3); dv3 = fma(h3, wv, dv3);
            }
            float dgs[4] = {dg0, dg1, dg2, dg3};
            float dvs[4] = {dv0, dv1, dv2, dv3};
            for (int b = 0; b < nb; ++b) {
                long row2 = (long)(i0 + b) * twoH;
                float g = gv[row2 + j], v = gv[row2 + H + j];
                float g_out = gy[(long)(i0 + b) * H + j];
                float tg = tanh(dgs[b]), tv = tanh(dvs[b]);
                float mg = 1.0f + 0.5f * tg, mv = 1.0f + 0.5f * tv;
                float u = g * mg;
                float sig = 1.0f / (1.0f + exp(-u));
                float s = u * sig;
                float ds = g_out * (v * mv);
                float dv_eff = g_out * s;
                float du = ds * (sig * (1.0f + u * (1.0f - sig)));
                g_gv[row2 + j] = du * mg;
                g_gv[row2 + H + j] = dv_eff * mv;
                gdg[b] = (du * g) * (0.5f * (1.0f - tg * tg));
                gdv[b] = (dv_eff * v) * (0.5f * (1.0f - tv * tv));
            }
        }
        // Reduce g_h_env[b, r] += sum_j (gdg[b]*wg + gdv[b]*wv) per simdgroup
        for (int r = 0; r < 32; ++r) {
            float wg = valid ? w_t[r * twoH + j] : 0.0f;
            float wv = valid ? w_t[r * twoH + H + j] : 0.0f;
            float p0 = simd_sum(fma(gdg[0], wg, gdv[0] * wv));
            float p1 = simd_sum(fma(gdg[1], wg, gdv[1] * wv));
            float p2 = simd_sum(fma(gdg[2], wg, gdv[2] * wv));
            float p3 = simd_sum(fma(gdg[3], wg, gdv[3] * wv));
            if (lane == 0) {
                GH[sg][0][r] += p0; GH[sg][1][r] += p1; GH[sg][2][r] += p2; GH[sg][3][r] += p3;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (b_ld < nb) {
        g_h_env[(long)(i0 + b_ld) * 32 + r_ld] =
            GH[0][b_ld][r_ld] + GH[1][b_ld][r_ld] + GH[2][b_ld][r_ld] + GH[3][b_ld][r_ld];
    }
}

kernel void swiglu_fwd_v4(
    device const float4* gv [[buffer(0)]], device float4* out [[buffer(1)]],
    constant int& N [[buffer(2)]], constant int& H4 [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    int total = N * H4;
    if (int(tid) >= total) return;
    int i = int(tid) / H4;
    int j = int(tid) - i * H4;
    int row2 = i * (2 * H4);
    float4 g = gv[row2 + j];
    float4 v = gv[row2 + H4 + j];
    float4 sig = 1.0f / (1.0f + exp(-g));
    out[tid] = (g * sig) * v;
}

kernel void swiglu_bwd_v4(
    device const float4* gv [[buffer(0)]], device const float4* gy [[buffer(1)]],
    device float4* g_gv [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& H4 [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    int total = N * H4;
    if (int(tid) >= total) return;
    int i = int(tid) / H4;
    int j = int(tid) - i * H4;
    int row2 = i * (2 * H4);
    float4 g = gv[row2 + j];
    float4 v = gv[row2 + H4 + j];
    float4 g_out = gy[tid];
    float4 sig = 1.0f / (1.0f + exp(-g));
    float4 s = g * sig;
    float4 ds = g_out * v;
    g_gv[row2 + j] = ds * (sig * (1.0f + g * (1.0f - sig)));
    g_gv[row2 + H4 + j] = g_out * s;
}
"""

_SWIGLU_SHADER = MetalShader("metal_swiglu", _SRC)


def has_metal_swiglu() -> bool:
    return _SWIGLU_SHADER.available()


def _lib() -> Any:
    return _SWIGLU_SHADER.lib()


def swiglu_fused_ok(mod: torch.nn.Module, gv: Tensor, allow: bool, hidden: int, rank: int = 0) -> bool:
    return bool(
        allow and hidden % 4 == 0 and (rank == 0 or rank == 32) and fused_node_device(gv) and not mod.training
        and not any(p.requires_grad for p in mod.parameters())
        and (gv.device.type != "mps" or has_metal_swiglu())
    )


# ---------------------------------------------------------------- EnvModulatedSwiGLU (rank-32 fused proj + act)
@torch.library.custom_op("lam4::env_swiglu_proj_fwd", mutates_args=())
def env_swiglu_proj_fwd(gv: Tensor, h_env: Tensor, w_t: Tensor, bias: Tensor) -> Tensor:
    if gv.device.type == "cuda":
        from lam4.kernels.cuda_node import cuda_env_swiglu_proj_fwd
        return cuda_env_swiglu_proj_fwd(gv, h_env, w_t, bias)
    n, two_h = int(gv.shape[0]), int(gv.shape[1])
    h = two_h // 2
    out = gv.new_empty((n, h))
    if n:
        nb = (n + 3) // 4
        _lib().env_swiglu_proj_fwd_k(gv.contiguous(), h_env.contiguous(), w_t.contiguous(), bias.contiguous(), out,
                                     n, h, threads=(nb * 128, 1, 1), group_size=(128, 1, 1))
    return out


@env_swiglu_proj_fwd.register_fake
def _(gv, h_env, w_t, bias):
    return gv.new_empty((gv.shape[0], gv.shape[1] // 2))


@torch.library.custom_op("lam4::env_swiglu_proj_bwd", mutates_args=())
def env_swiglu_proj_bwd(gv: Tensor, h_env: Tensor, w_t: Tensor, bias: Tensor, gy: Tensor) -> tuple[Tensor, Tensor]:
    if gv.device.type == "cuda":
        from lam4.kernels.cuda_node import cuda_env_swiglu_proj_bwd
        return cuda_env_swiglu_proj_bwd(gv, h_env, w_t, bias, gy)
    n, two_h = int(gv.shape[0]), int(gv.shape[1])
    h = two_h // 2
    g_gv, g_h = gv.new_empty((n, two_h)), h_env.new_empty((n, 32))
    if n:
        nb = (n + 3) // 4
        _lib().env_swiglu_proj_bwd_k(gv.contiguous(), h_env.contiguous(), w_t.contiguous(), bias.contiguous(),
                                     gy.contiguous(), g_gv, g_h, n, h, threads=(nb * 128, 1, 1), group_size=(128, 1, 1))
    return g_gv, g_h


@env_swiglu_proj_bwd.register_fake
def _(gv, h_env, w_t, bias, gy):
    return gv.new_empty(gv.shape), h_env.new_empty(h_env.shape)


def _env_setup(ctx, inputs, output):
    gv, h_env, w_t, bias = inputs
    ctx.save_for_backward(gv, h_env, w_t, bias)


def _env_backward(ctx, gy):
    gv, h_env, w_t, bias = ctx.saved_tensors
    g_gv, g_h = env_swiglu_proj_bwd(gv, h_env, w_t, bias, gy)
    return g_gv, g_h, None, None


torch.library.register_autograd("lam4::env_swiglu_proj_fwd", _env_backward, setup_context=_env_setup)


# ---------------------------------------------------------------- plain SwiGLU (lam5)
@torch.library.custom_op("lam4::swiglu_act_fwd", mutates_args=())
def swiglu_act_fwd(gv: Tensor) -> Tensor:
    if gv.device.type == "cuda":
        from lam4.kernels.cuda_node import cuda_swiglu_fwd
        return cuda_swiglu_fwd(gv)
    n, two_h = int(gv.shape[0]), int(gv.shape[1])
    h = two_h // 2
    out = gv.new_empty((n, h))
    if n:
        h4 = h // 4
        tot = n * h4
        tg = 256
        grid = ((tot + tg - 1) // tg) * tg
        _lib().swiglu_fwd_v4(gv.contiguous(), out, n, h4, threads=(grid, 1, 1), group_size=(tg, 1, 1))
    return out


@swiglu_act_fwd.register_fake
def _(gv):
    return gv.new_empty((gv.shape[0], gv.shape[1] // 2))


@torch.library.custom_op("lam4::swiglu_act_bwd", mutates_args=())
def swiglu_act_bwd(gv: Tensor, gy: Tensor) -> Tensor:
    if gv.device.type == "cuda":
        from lam4.kernels.cuda_node import cuda_swiglu_bwd
        return cuda_swiglu_bwd(gv, gy)
    n, two_h = int(gv.shape[0]), int(gv.shape[1])
    h = two_h // 2
    g_gv = gv.new_empty((n, two_h))
    if n:
        h4 = h // 4
        tot = n * h4
        tg = 256
        grid = ((tot + tg - 1) // tg) * tg
        _lib().swiglu_bwd_v4(gv.contiguous(), gy.contiguous(), g_gv, n, h4, threads=(grid, 1, 1), group_size=(tg, 1, 1))
    return g_gv


@swiglu_act_bwd.register_fake
def _(gv, gy):
    return gv.new_empty(gv.shape)


def _plain_setup(ctx, inputs, output):
    (gv,) = inputs
    ctx.save_for_backward(gv)


def _plain_backward(ctx, gy):
    (gv,) = ctx.saved_tensors
    return (swiglu_act_bwd(gv, gy),)


torch.library.register_autograd("lam4::swiglu_act_fwd", _plain_backward, setup_context=_plain_setup)
