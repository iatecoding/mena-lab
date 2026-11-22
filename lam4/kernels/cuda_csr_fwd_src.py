"""CUDA C++ common device math & warp-per-atom forward Dual-CSR kernel (`poly4` float4 + fused `finalise`)."""
from __future__ import annotations

CUDA_CSR_FWD_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int F_FLOOR     = 1;
constexpr int F_RHO       = 2;
constexpr int F_TEN       = 4;
constexpr int F_CELL      = 8;
constexpr int F_G1        = 16;
constexpr int F_VIR       = 32;
constexpr int F_DRHO      = 64;
constexpr int F_SEND_FEAT = 128;
constexpr int F_DQN       = 256;
constexpr int F_GDSUM     = 512;

constexpr float S2     = 1.4142135623730951f;
constexpr float INV_S2 = 0.7071067811865475f;
constexpr float INV_S6 = 0.4082482904638630f;
constexpr float C3_0   = 0.5f;
constexpr float C3_1   = 2.449489742783178f;
constexpr float C3_2   = 0.3872983346207417f;
constexpr float C3_3   = 0.31622776601683794f;
constexpr float C3_4   = 0.3872983346207417f;
constexpr float C3_5   = 1.224744871391589f;
constexpr float C3_6   = 0.5f;

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ float warp_sum16(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ float warp_sum8(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ float warp_sum4(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ void qk_dots4(
    float q0, float q1, float k0, float k1,
    float& d0, float& d1, float& d2, float& d3
) {
    float p0 = q0 * k0, p1 = q1 * k1;
    p0 += __shfl_xor_sync(0xffffffffu, p0, 8);
    p0 += __shfl_xor_sync(0xffffffffu, p0, 4);
    p0 += __shfl_xor_sync(0xffffffffu, p0, 2);
    p0 += __shfl_xor_sync(0xffffffffu, p0, 1);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 8);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 4);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 2);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 1);
    d0 = __shfl_sync(0xffffffffu, p0, 0);
    d1 = __shfl_sync(0xffffffffu, p0, 16);
    d2 = __shfl_sync(0xffffffffu, p1, 0);
    d3 = __shfl_sync(0xffffffffu, p1, 16);
}

__device__ __forceinline__ float horner4_val(const float4* __restrict__ p4, int col, float t) {
    float4 a = __ldg(p4 + col);
    return fmaf(fmaf(fmaf(a.w, t, a.z), t, a.y), t, a.x);
}

__device__ __forceinline__ void horner4_vd(
    const float4* __restrict__ p4, int col, float t, float inv_h, float& fv, float& df
) {
    float4 a = __ldg(p4 + col);
    fv = fmaf(fmaf(fmaf(a.w, t, a.z), t, a.y), t, a.x);
    df = fmaf(fmaf(3.0f * a.w, t, 2.0f * a.z), t, a.y) * inv_h;
}

__device__ __forceinline__ float3 load_shift(
    const float* __restrict__ shift_in, const float* __restrict__ cell,
    int e, int g_id, int flags, float3& es_out
) {
    es_out = make_float3(__ldg(shift_in + 3 * e), __ldg(shift_in + 3 * e + 1), __ldg(shift_in + 3 * e + 2));
    if ((flags & F_CELL) == 0) return es_out;
    const float* c = cell + 9 * g_id;
    return make_float3(
        fmaf(es_out.x, __ldg(c + 0), fmaf(es_out.y, __ldg(c + 3), es_out.z * __ldg(c + 6))),
        fmaf(es_out.x, __ldg(c + 1), fmaf(es_out.y, __ldg(c + 4), es_out.z * __ldg(c + 7))),
        fmaf(es_out.x, __ldg(c + 2), fmaf(es_out.y, __ldg(c + 5), es_out.z * __ldg(c + 8)))
    );
}

__device__ __forceinline__ void eval_y2(float3 u, float (&y2)[5]) {
    float x = u.x, y = u.y, z = u.z;
    y2[0] = S2 * x * y;
    y2[1] = S2 * x * z;
    y2[2] = S2 * y * z;
    y2[3] = (x * x - y * y) * INV_S2;
    y2[4] = (2.0f * z * z - x * x - y * y) * INV_S6;
}

__device__ __forceinline__ void eval_y3(float3 u, float (&y3)[7]) {
    float x = u.x, y = u.y, z = u.z;
    float xx = x * x, yy = y * y, zz = z * z;
    y3[0] = C3_0 * y * (3.0f * xx - yy);
    y3[1] = C3_1 * x * y * z;
    y3[2] = C3_2 * y * (4.0f * zz - xx - yy);
    y3[3] = C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    y3[4] = C3_4 * x * (4.0f * zz - xx - yy);
    y3[5] = C3_5 * z * (xx - yy);
    y3[6] = C3_6 * x * (xx - 3.0f * yy);
}

__device__ __forceinline__ float3 grad_y2_u(float3 u, const float (&g)[5]) {
    float x = u.x, y = u.y, z = u.z, k6 = 2.0f * INV_S6;
    return make_float3(
        S2 * (g[0] * y + g[1] * z + g[3] * x) - k6 * g[4] * x,
        S2 * (g[0] * x + g[2] * z - g[3] * y) - k6 * g[4] * y,
        S2 * (g[1] * x + g[2] * y) + 2.0f * k6 * g[4] * z
    );
}

__device__ __forceinline__ float3 grad_y3_u(float3 u, const float (&g)[7]) {
    float x = u.x, y = u.y, z = u.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float gx = g[0] * (C3_0 * 6.0f * x * y) + g[1] * (C3_1 * y * z) + g[2] * (-2.0f * C3_2 * x * y)
             + g[3] * (-6.0f * C3_3 * x * z) + g[4] * (C3_4 * (4.0f * zz - 3.0f * xx - yy))
             + g[5] * (2.0f * C3_5 * x * z) + g[6] * (C3_6 * (3.0f * xx - 3.0f * yy));
    float gy = g[0] * (C3_0 * (3.0f * xx - 3.0f * yy)) + g[1] * (C3_1 * x * z)
             + g[2] * (C3_2 * (4.0f * zz - xx - 3.0f * yy)) + g[3] * (-6.0f * C3_3 * y * z)
             + g[4] * (-2.0f * C3_4 * x * y) + g[5] * (-2.0f * C3_5 * y * z) + g[6] * (-6.0f * C3_6 * x * y);
    float gz = g[1] * (C3_1 * x * y) + g[2] * (8.0f * C3_2 * y * z)
             + g[3] * (C3_3 * (6.0f * zz - 3.0f * xx - 3.0f * yy))
             + g[4] * (8.0f * C3_4 * x * z) + g[5] * (C3_5 * (xx - yy));
    return make_float3(gx, gy, gz);
}

__device__ __forceinline__ int clamp_z(long z) {
    return (z < 0L) ? 0 : ((z > 127L) ? 127 : int(z));
}

template <bool LATER, int WPA = 1>
__global__ void cuda_csr_fwd_final_k(
    const float* __restrict__ pos, const int* __restrict__ recv_ptr, const int* __restrict__ send_idx,
    const float* __restrict__ shift_in, const float* __restrict__ cell, const long* __restrict__ graph,
    const long* __restrict__ z, const float* __restrict__ rmin_tab, const float* __restrict__ rho,
    const float4* __restrict__ poly4, const float* __restrict__ gates, const float* __restrict__ qn,
    const float* __restrict__ kn, const float* __restrict__ vec, const float* __restrict__ ten,
    float* __restrict__ out_agg, float* __restrict__ out_zc, float* __restrict__ out_dsum,
    int n_atoms, int grid, int flags, float r_cut, float r_on, float max_doff_r2
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    constexpr int ATOMS_PER_CTA = 4 / WPA;
    int local_atom = warp_id / WPA;
    int sub_w = warp_id - local_atom * WPA;
    int i = blockIdx.x * ATOMS_PER_CTA + local_atom;
    if constexpr (WPA == 1) {
        if (i >= n_atoms) return;
    }

    constexpr int TD = LATER ? 284 : 188;
    constexpr int OD = LATER ? 408 : 360;
    constexpr int F_O_OFF = LATER ? 276 : 180;
    constexpr int OUT_T_OFF = LATER ? 256 : 224;
    constexpr int OUT_O_OFF = LATER ? 352 : 304;

    bool has_floor = (flags & F_FLOOR) != 0;
    bool has_rho = (flags & F_RHO) != 0;
    bool has_ten = LATER && ((flags & F_TEN) != 0);
    float acc_s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_vx = 0.0f, acc_vy = 0.0f, acc_vz = 0.0f;
    float acc_lon = 0.0f, acc_lon2 = 0.0f, acc_zc = 0.0f, acc_dsum = 0.0f;
    float acc_t[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float acc_o[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    if (i < n_atoms) {
        int g_id = ((flags & F_CELL) != 0 && (flags & F_G1) == 0) ? int(graph[i]) : 0;
        int zi_shift = has_floor ? (clamp_z(z[i]) << 7) : 0;
        float rho_i = has_rho ? __ldg(rho + i) : 0.0f;
        float3 pi = make_float3(__ldg(pos + 3 * i), __ldg(pos + 3 * i + 1), __ldg(pos + 3 * i + 2));
        float q0 = __ldg(qn + 64 * i + lane);
        float q1 = __ldg(qn + 64 * i + 32 + lane);
        float r_cut2 = r_cut * r_cut;
        float inv_w_env = 1.0f / (r_cut - r_on);
        float inv_h = float(grid) / r_cut;

        int e_beg = recv_ptr[i], e_end = recv_ptr[i + 1];
        for (int e = e_beg + sub_w; e < e_end; e += WPA) {
        int j = send_idx[e];
        float3 es_dummy;
        float3 sh = load_shift(shift_in, cell, e, g_id, flags, es_dummy);
        float dx = __ldg(pos + 3 * j) - pi.x + sh.x;
        float dy = __ldg(pos + 3 * j + 1) - pi.y + sh.y;
        float dz = __ldg(pos + 3 * j + 2) - pi.z + sh.z;
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < 1e-24f || r2 >= r_cut2) continue;

        float r_raw = sqrtf(r2);
        float inv_r = 1.0f / r_raw;
        float3 u = make_float3(dx * inv_r, dy * inv_r, dz * inv_r);
        float r_fl = r_raw;
        if (has_floor && r2 < max_doff_r2) {
            float r_min = __ldg(rmin_tab + zi_shift + clamp_z(z[j]));
            float d_off = r_min + 0.05f;
            float r_eps = sqrtf(r2 + 1e-8f);
            if (r_eps < d_off && r_min > 0.0f) {
                r_fl = d_off - 0.05f * tanhf((d_off - r_eps) * 20.0f);
            }
        }
        if (r_fl >= r_cut) continue;

        float chi = 1.0f;
        if (r_fl > r_on) {
            float te = (r_fl - r_on) * inv_w_env;
            float t2 = te * te;
            chi = 1.0f - (t2 * t2) * fmaf(te, fmaf(te, fmaf(-20.0f, te, 70.0f), -84.0f), 35.0f);
        }
        float x = r_fl;
        if (has_rho) {
            float sw = 1.52f / (rho_i + __ldg(rho + j));
            x = (sw * r_fl) / (1.0f + (sw - 1.0f) * (r_fl / r_cut));
        }
        float u_sp = fminf(fmaxf(x, 0.0f), r_cut) * inv_h;
        int idx = min(max(int(floorf(u_sp)), 0), grid - 1);
        float tsp = u_sp - float(idx);
        const float4* p4 = poly4 + (long)idx * TD;

        float d0, d1, d2, d3, w_l = 0.0f;
        qk_dots4(q0, q1, __ldg(kn + 64 * j + lane), __ldg(kn + 64 * j + 32 + lane), d0, d1, d2, d3);
        if (lane < 4) {
            float dl = (lane == 0) ? d0 : ((lane == 1) ? d1 : ((lane == 2) ? d2 : d3));
            w_l = chi * expf(dl + 2.0f * tanhf(0.5f * horner4_val(p4, lane, tsp)));
            acc_zc += w_l;
        }
        if (lane == 0) acc_dsum += chi;
        float ws0 = __shfl_sync(0xffffffffu, w_l, 0);
        float ws1 = __shfl_sync(0xffffffffu, w_l, 1);
        float ws2 = __shfl_sync(0xffffffffu, w_l, 2);
        float ws3 = __shfl_sync(0xffffffffu, w_l, 3);
        float ws[4] = {ws0, ws1, ws2, ws3};

        const float* gj = gates + (long)184 * j;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int col = 32 * m + lane;
            acc_s[m] = fmaf(ws[m] * horner4_val(p4, 4 + col, tsp), __ldg(gj + col), acc_s[m]);
        }

        float wv = ws[lane >> 3];
        float w_dir = horner4_val(p4, 132 + lane, tsp) * __ldg(gj + 128 + lane);
        if (LATER) {
            const float* bj = vec + (long)96 * j + 3 * lane;
            float b0 = __ldg(bj + 0), b1 = __ldg(bj + 1), b2 = __ldg(bj + 2);
            float st = wv * horner4_val(p4, 180 + lane, tsp);
            float w_lon = horner4_val(p4, 212 + lane, tsp) * fmaf(b0, u.x, fmaf(b1, u.y, b2 * u.z));
            float sc = wv * (w_dir + w_lon);
            acc_vx = fmaf(sc, u.x, fmaf(st, b0, acc_vx));
            acc_vy = fmaf(sc, u.y, fmaf(st, b1, acc_vy));
            acc_vz = fmaf(sc, u.z, fmaf(st, b2, acc_vz));
            acc_lon = fmaf(wv, w_lon, acc_lon);
        } else {
            float sc = wv * w_dir;
            acc_vx = fmaf(sc, u.x, acc_vx);
            acc_vy = fmaf(sc, u.y, acc_vy);
            acc_vz = fmaf(sc, u.z, acc_vz);
        }

        if (lane < 16) {
            float y2[5];
            eval_y2(u, y2);
            float wt = ws[lane >> 2];
            float w_ten = horner4_val(p4, 164 + lane, tsp) * __ldg(gj + 160 + lane);
            if (has_ten) {
                const float* tj = ten + (long)80 * j + 5 * lane;
                float t0 = __ldg(tj + 0), t1 = __ldg(tj + 1), t2 = __ldg(tj + 2), t3 = __ldg(tj + 3), t4 = __ldg(tj + 4);
                float dot_ty2 = fmaf(t0, y2[0], fmaf(t1, y2[1], fmaf(t2, y2[2], fmaf(t3, y2[3], t4 * y2[4]))));
                float st = wt * horner4_val(p4, 244 + lane, tsp);
                float w_lon2 = horner4_val(p4, 260 + lane, tsp) * dot_ty2;
                float sc = wt * (w_ten + w_lon2);
                acc_t[0] = fmaf(sc, y2[0], fmaf(st, t0, acc_t[0]));
                acc_t[1] = fmaf(sc, y2[1], fmaf(st, t1, acc_t[1]));
                acc_t[2] = fmaf(sc, y2[2], fmaf(st, t2, acc_t[2]));
                acc_t[3] = fmaf(sc, y2[3], fmaf(st, t3, acc_t[3]));
                acc_t[4] = fmaf(sc, y2[4], fmaf(st, t4, acc_t[4]));
                acc_lon2 = fmaf(wt, w_lon2, acc_lon2);
            } else {
                float sc = wt * w_ten;
                #pragma unroll
                for (int m = 0; m < 5; ++m) acc_t[m] = fmaf(sc, y2[m], acc_t[m]);
            }
        }
        if (lane < 8) {
            float y3[7];
            eval_y3(u, y3);
            float sc = ws[lane >> 1] * horner4_val(p4, F_O_OFF + lane, tsp) * __ldg(gj + 176 + lane);
            #pragma unroll
            for (int m = 0; m < 7; ++m) acc_o[m] = fmaf(sc, y3[m], acc_o[m]);
        }
        }
    }

    if constexpr (WPA > 1) {
        __shared__ float smem[4][416];
        float* sw = smem[warp_id];
        #pragma unroll
        for (int m = 0; m < 4; ++m) sw[32 * m + lane] = acc_s[m];
        sw[128 + 3 * lane + 0] = acc_vx;
        sw[128 + 3 * lane + 1] = acc_vy;
        sw[128 + 3 * lane + 2] = acc_vz;
        if (LATER) sw[224 + lane] = acc_lon;
        if (lane < 16) {
            #pragma unroll
            for (int m = 0; m < 5; ++m) sw[OUT_T_OFF + 5 * lane + m] = acc_t[m];
            if (LATER) sw[336 + lane] = acc_lon2;
        }
        if (lane < 8) {
            #pragma unroll
            for (int m = 0; m < 7; ++m) sw[OUT_O_OFF + 7 * lane + m] = acc_o[m];
        }
        if (lane < 4) sw[408 + lane] = acc_zc;
        if (lane == 0) sw[412] = acc_dsum;
        __syncthreads();
        if (sub_w != 0 || i >= n_atoms) return;
        for (int k = 1; k < WPA; ++k) {
            const float* sk = smem[warp_id + k];
            #pragma unroll
            for (int m = 0; m < 4; ++m) acc_s[m] += sk[32 * m + lane];
            acc_vx += sk[128 + 3 * lane + 0];
            acc_vy += sk[128 + 3 * lane + 1];
            acc_vz += sk[128 + 3 * lane + 2];
            if (LATER) acc_lon += sk[224 + lane];
            if (lane < 16) {
                #pragma unroll
                for (int m = 0; m < 5; ++m) acc_t[m] += sk[OUT_T_OFF + 5 * lane + m];
                if (LATER) acc_lon2 += sk[336 + lane];
            }
            if (lane < 8) {
                #pragma unroll
                for (int m = 0; m < 7; ++m) acc_o[m] += sk[OUT_O_OFF + 7 * lane + m];
            }
            if (lane < 4) acc_zc += sk[408 + lane];
            if (lane == 0) acc_dsum += sk[412];
        }
    }

    // Fused finalise: scale_h = sqrt(dsum + 1) / (zc[h] + 1)
    float dsum_b = __shfl_sync(0xffffffffu, acc_dsum, 0);
    float sq_d = sqrtf(dsum_b + 1.0f);
    float sc_l = (lane < 4) ? (sq_d / (acc_zc + 1.0f)) : 0.0f;
    float sc0 = __shfl_sync(0xffffffffu, sc_l, 0);
    float sc1 = __shfl_sync(0xffffffffu, sc_l, 1);
    float sc2 = __shfl_sync(0xffffffffu, sc_l, 2);
    float sc3 = __shfl_sync(0xffffffffu, sc_l, 3);
    float sc[4] = {sc0, sc1, sc2, sc3};

    float* oi = out_agg + (long)OD * i;
    #pragma unroll
    for (int m = 0; m < 4; ++m) oi[32 * m + lane] = acc_s[m] * sc[m];
    float scv = sc[lane >> 3];
    oi[128 + 3 * lane + 0] = acc_vx * scv;
    oi[128 + 3 * lane + 1] = acc_vy * scv;
    oi[128 + 3 * lane + 2] = acc_vz * scv;
    if (LATER) oi[224 + lane] = acc_lon * scv;
    if (lane < 16) {
        float sct = sc[lane >> 2];
        #pragma unroll
        for (int m = 0; m < 5; ++m) oi[OUT_T_OFF + 5 * lane + m] = acc_t[m] * sct;
        if (LATER) oi[336 + lane] = acc_lon2 * sct;
    }
    if (lane < 8) {
        float sco = sc[lane >> 1];
        #pragma unroll
        for (int m = 0; m < 7; ++m) oi[OUT_O_OFF + 7 * lane + m] = acc_o[m] * sco;
    }
    if (lane < 4) out_zc[4 * i + lane] = acc_zc;
    if (lane == 0) out_dsum[i] = acc_dsum;
}
"""
