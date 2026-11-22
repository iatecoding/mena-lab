"""CUDA C++ fused per-atom node kernels: `node_inv_fwd`, `node_vec_fwd/bwd`, `swiglu_fwd/bwd`, `env_swiglu_proj_fwd/bwd`."""
from __future__ import annotations

from lam4.core.angular import _y3_terms


def _y3_code_cuda() -> tuple[str, str]:
    fwd, bwd = [], []
    for j, terms in enumerate(_y3_terms()):
        fwd.append(f"    Y[{j}] = " + " + ".join(f"({c!r}f) * v[{a}] * m[{p}]" for a, p, c in terms) + ";")
        for a, p, c in terms:
            bwd.append(f"    gv[{a}] = fmaf(gy[{j}] * ({c!r}f), m[{p}], gv[{a}]); gm[{p}] = fmaf(gy[{j}] * ({c!r}f), v[{a}], gm[{p}]);")
    return "\n".join(fwd), "\n".join(bwd)


_Y3_FWD, _Y3_BWD = _y3_code_cuda()

CUDA_NODE_FWD_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr float R2 = 0.70710678118654752440f;
constexpr float R6 = 0.40824829046386301637f;
constexpr int F_LATER = 1;
constexpr int F_TEN   = 2;
constexpr int F_CUMT  = 4;

constexpr int WA_UVPQ = 0;
constexpr int WA_UVT  = 12288;
constexpr int WA_YT   = 15360;
constexpr int WA_YV   = 17152;
constexpr int WA_U3   = 20224;
constexpr int WA_UCV  = 20992;
constexpr int WA_VCV  = 22016;
constexpr int WA_UCT  = 23040;
constexpr int WA_VCT  = 23296;

constexpr int X_VN = 0, X_TN = 96, X_O = 176, X_VEC = 232, X_PV = 328, X_TEN = 424, X_TT = 504, X_N = 584;
constexpr int WB_UO = 0, WB_TB = 3072;
constexpr int V_VN = 0, V_TN = 96, V_N = 176;

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ void mat6(const float* t, float* m) {
    m[0] = t[3] * R2 - t[4] * R6; m[1] = -t[3] * R2 - t[4] * R6; m[2] = 2.0f * R6 * t[4];
    m[3] = t[0] * R2; m[4] = t[1] * R2; m[5] = t[2] * R2;
}

__device__ __forceinline__ void matvec(const float* m, const float* p, float* o) {
    o[0] = m[0] * p[0] + m[3] * p[1] + m[4] * p[2];
    o[1] = m[3] * p[0] + m[1] * p[1] + m[5] * p[2];
    o[2] = m[4] * p[0] + m[5] * p[1] + m[2] * p[2];
}

__device__ __forceinline__ void l2pair(const float* p, const float* q, float* o) {
    o[0] = (p[0] * q[1] + p[1] * q[0]) * R2;
    o[1] = (p[0] * q[2] + p[2] * q[0]) * R2;
    o[2] = (p[1] * q[2] + p[2] * q[1]) * R2;
    o[3] = (p[0] * q[0] - p[1] * q[1]) * R2;
    o[4] = (2.0f * p[2] * q[2] - p[0] * q[0] - p[1] * q[1]) * R6;
}

__device__ __forceinline__ void gm6_to_gt(const float* gm, float* gt) {
    gt[0] = R2 * gm[3]; gt[1] = R2 * gm[4]; gt[2] = R2 * gm[5];
    gt[3] = R2 * (gm[0] - gm[1]); gt[4] = R6 * (2.0f * gm[2] - gm[0] - gm[1]);
}

template <int IN, int M>
__device__ __forceinline__ void proj(const float* __restrict__ W, int rows, int r, const float* x, float s, float* o) {
    #pragma unroll
    for (int m = 0; m < M; ++m) o[m] = 0.0f;
    for (int c = 0; c < IN; ++c) {
        float w = __ldg(W + c * rows + r);
        #pragma unroll
        for (int m = 0; m < M; ++m) o[m] = fmaf(w, x[c * M + m], o[m]);
    }
    #pragma unroll
    for (int m = 0; m < M; ++m) o[m] *= s;
}

template <int M>
__device__ __forceinline__ float dotm(const float* a, const float* b) {
    float s = 0.0f;
    #pragma unroll
    for (int m = 0; m < M; ++m) s = fmaf(a[m], b[m], s);
    return s;
}

template <int M>
__device__ __forceinline__ void storeg(float* G, int r, float g, const float* d) {
    #pragma unroll
    for (int m = 0; m < M; ++m) G[r * M + m] = g * d[m];
}

__device__ __forceinline__ float sumsq(const float* x, int L, int lane) {
    float s = 0.0f;
    for (int i = lane; i < L; i += 32) s = fmaf(x[i], x[i], s);
    return warp_sum(s);
}

__device__ __forceinline__ float dotsg(const float* a, const float* b, int L, int lane) {
    float s = 0.0f;
    for (int i = lane; i < L; i += 32) s = fmaf(a[i], b[i], s);
    return warp_sum(s);
}

__device__ __forceinline__ float bound_scale(float ss, float k, float c2) {
    return 1.0f / sqrtf(1.0f + (ss / k) / c2);
}

__device__ __forceinline__ void y3vm(const float* v, const float* m, float* Y) {
@Y3_FWD@
}

__device__ __forceinline__ void y3vm_bwd(const float* v, const float* m, const float* gy, float* gv, float* gm) {
    for (int a = 0; a < 3; ++a) gv[a] = 0.0f;
    for (int p = 0; p < 6; ++p) gm[p] = 0.0f;
@Y3_BWD@
}

struct Lay { int av, at, ao, alon, alon2, out_dim, ang, inv_dim; };
__device__ __forceinline__ Lay lay(int flags) {
    Lay L;
    if (flags & F_LATER) {
        L.av = 128; L.alon = 224; L.at = 256; L.alon2 = 336; L.ao = 352; L.out_dim = 408; L.ang = 352; L.inv_dim = 464;
    } else {
        L.av = 128; L.alon = -1; L.at = 224; L.alon2 = -1; L.ao = 304; L.out_dim = 360; L.ang = 256; L.inv_dim = 368;
    }
    return L;
}

__global__ void cuda_node_inv_fwd_k(
    const float* __restrict__ vec, const float* __restrict__ agg, const float* __restrict__ ten,
    const float* __restrict__ W, float* __restrict__ out, int n, int flags, float c2
) {
    int i = blockIdx.x;
    if (i >= n) return;
    int tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
    Lay L = lay(flags);
    __shared__ float X[X_N];
    __shared__ float scl[6];
    const float* a = agg + (long)i * L.out_dim;
    bool has_ten = (flags & F_TEN) != 0;
    for (int e = tid; e < 96; e += 128) {
        float v0 = vec[(long)i * 96 + e], pv = a[L.av + e];
        X[X_VN + e] = v0 + pv; X[X_VEC + e] = v0; X[X_PV + e] = pv;
    }
    for (int e = tid; e < 80; e += 128) {
        float t = a[L.at + e], t0 = has_ten ? ten[(long)i * 80 + e] : 0.0f;
        X[X_TN + e] = (has_ten && (flags & F_CUMT)) ? t0 + t : t; X[X_TEN + e] = t0; X[X_TT + e] = t;
    }
    for (int e = tid; e < 56; e += 128) X[X_O + e] = a[L.ao + e];
    __syncthreads();
    if (sg == 0) { float s = sumsq(X + X_VN, 96, lane); if (lane == 0) scl[0] = bound_scale(s, 32.0f, c2); }
    if (sg == 1) { float s = sumsq(X + X_TN, 80, lane); if (lane == 0) scl[1] = (flags & F_CUMT) ? bound_scale(s, 16.0f, c2) : 1.0f; }
    if (sg == 2) { float s = sumsq(X + X_VEC, 96, lane), s2 = sumsq(X + X_PV, 96, lane); if (lane == 0) { scl[2] = bound_scale(s, 32.0f, c2); scl[3] = bound_scale(s2, 32.0f, c2); } }
    if (sg == 3) { float s = sumsq(X + X_TEN, 80, lane), s2 = sumsq(X + X_TT, 80, lane); if (lane == 0) { scl[4] = bound_scale(s, 16.0f, c2); scl[5] = bound_scale(s2, 16.0f, c2); } }
    __syncthreads();

    float su = scl[0], st = scl[1];
    float* y = out + (long)i * L.inv_dim;
    for (int o = tid; o < L.inv_dim; o += 128) {
        float r = 0.0f;
        float p[7], q[7], t[7], m[6], x[3], z[3];
        if (o < 128) {
            proj<32, 3>(W + WA_UVPQ, 384, o, X + X_VN, su, p); proj<32, 3>(W + WA_UVPQ, 384, 128 + o, X + X_VN, su, q); r = dotm<3>(p, q);
        } else if (o < 192) {
            int k = o - 128;
            proj<16, 5>(W + WA_UVT, 192, k, X + X_TN, st, p); proj<16, 5>(W + WA_UVT, 192, 64 + k, X + X_TN, st, q); r = dotm<5>(p, q);
        } else if (o < 256) {
            int k = o - 192;
            proj<32, 3>(W + WA_UVPQ, 384, 256 + k, X + X_VN, su, p); proj<32, 3>(W + WA_UVPQ, 384, 320 + k, X + X_VN, su, q);
            proj<16, 5>(W + WA_UVT, 192, 128 + k, X + X_TN, st, t); float l[5]; l2pair(p, q, l); r = dotm<5>(t, l);
        } else if (o < L.ang) {
            if (o < 272) r = a[L.alon2 + (o - 256)];
            else if (o < 304) { int k = o - 272; proj<32, 3>(W + WA_UCV, 32, k, X + X_VEC, scl[2], p); proj<32, 3>(W + WA_VCV, 32, k, X + X_PV, scl[3], q); r = dotm<3>(p, q); }
            else if (o < 320) { int k = o - 304; if (has_ten) { proj<16, 5>(W + WA_UCT, 16, k, X + X_TEN, scl[4], p); proj<16, 5>(W + WA_VCT, 16, k, X + X_TT, scl[5], q); r = dotm<5>(p, q); } }
            else r = a[L.alon + (o - 320)];
        } else {
            int g = o - L.ang;
            if (g < 16) {
                proj<16, 5>(W + WA_YT, 112, g, X + X_TN, st, t); mat6(t, m);
                r = 3.0f * (m[0] * (m[1] * m[2] - m[5] * m[5]) - m[3] * (m[3] * m[2] - m[5] * m[4]) + m[4] * (m[3] * m[5] - m[1] * m[4]));
            } else if (g < 48) { int k = g - 16;
                float ma[6], mb[6];
                proj<16, 5>(W + WA_YT, 112, 16 + k, X + X_TN, st, t); mat6(t, ma);
                proj<16, 5>(W + WA_YT, 112, 48 + k, X + X_TN, st, t); mat6(t, mb);
                proj<32, 3>(W + WA_YV, 96, k, X + X_VN, su, p); proj<32, 3>(W + WA_YV, 96, 32 + k, X + X_VN, su, q);
                matvec(ma, p, x); matvec(mb, q, z); r = dotm<3>(x, z);
            } else if (g < 80) { int k = g - 48;
                proj<8, 7>(W + WA_U3, 96, k, X + X_O, 1.0f, p); proj<8, 7>(W + WA_U3, 96, 32 + k, X + X_O, 1.0f, q); r = dotm<7>(p, q);
            } else { int k = g - 80;
                float Y[7];
                proj<8, 7>(W + WA_U3, 96, 64 + k, X + X_O, 1.0f, q); proj<32, 3>(W + WA_YV, 96, 64 + k, X + X_VN, su, p);
                proj<16, 5>(W + WA_YT, 112, 80 + k, X + X_TN, st, t); mat6(t, m); y3vm(p, m, Y); r = dotm<7>(q, Y);
            }
        }
        y[o] = r;
    }
}

__device__ __forceinline__ void load_vt(
    const float* __restrict__ vec, const float* __restrict__ agg, const float* __restrict__ ten,
    int i, int flags, float c2, float* X, float* scl, int tid, int lane, int sg
) {
    Lay L = lay(flags);
    const float* a = agg + (long)i * L.out_dim;
    bool has_ten = (flags & F_TEN) != 0;
    for (int e = tid; e < 96; e += 64) X[V_VN + e] = vec[(long)i * 96 + e] + a[L.av + e];
    for (int e = tid; e < 80; e += 64) X[V_TN + e] = has_ten ? (ten[(long)i * 80 + e] + a[L.at + e]) : a[L.at + e];
    __syncthreads();
    if (sg == 0) { float s = sumsq(X + V_VN, 96, lane); if (lane == 0) scl[0] = bound_scale(s, 32.0f, c2); }
    if (sg == 1) { float s = sumsq(X + V_TN, 80, lane); if (lane == 0) scl[1] = bound_scale(s, 16.0f, c2); }
    __syncthreads();
}

__global__ void cuda_node_vec_fwd_k(
    const float* __restrict__ vec, const float* __restrict__ agg, const float* __restrict__ ten,
    const float* __restrict__ gates, const float* __restrict__ W,
    float* __restrict__ v_out, float* __restrict__ t_out, int n, int flags, float c2
) {
    int i = blockIdx.x;
    if (i >= n) return;
    int tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
    __shared__ float X[V_N];
    __shared__ float scl[2];
    load_vt(vec, agg, ten, i, flags, c2, X, scl, tid, lane, sg);
    float su = scl[0], stb = scl[1];
    const float* gt = gates + (long)i * 96;
    int k = tid;
    float p[5], q[5], y[5], m[6], x[3];
    if (k < 32) {
        proj<32, 3>(W + WB_UO, 96, k, X + V_VN, su, p); proj<32, 3>(W + WB_UO, 96, 32 + k, X + V_VN, su, q);
        proj<16, 5>(W + WB_TB, 48, 16 + k, X + V_TN, stb, y); mat6(y, m); matvec(m, q, x);
        float g1 = gt[k], g2 = gt[32 + k];
        #pragma unroll
        for (int c = 0; c < 3; ++c) v_out[(long)i * 96 + k * 3 + c] = X[V_VN + k * 3 + c] + p[c] * g1 + x[c] * g2;
    } else if (k < 48) { int j = k - 32;
        float l[5];
        proj<16, 5>(W + WB_TB, 48, j, X + V_TN, stb, y);
        proj<32, 3>(W + WB_UO, 96, 64 + j, X + V_VN, su, p); proj<32, 3>(W + WB_UO, 96, 80 + j, X + V_VN, su, q); l2pair(p, q, l);
        float g3 = gt[64 + j], g4 = gt[80 + j];
        #pragma unroll
        for (int c = 0; c < 5; ++c) t_out[(long)i * 80 + j * 5 + c] = X[V_TN + j * 5 + c] + y[c] * g3 + l[c] * g4;
    }
}

__global__ void cuda_node_vec_bwd_k(
    const float* __restrict__ vec, const float* __restrict__ agg, const float* __restrict__ ten,
    const float* __restrict__ gates, const float* __restrict__ W,
    const float* __restrict__ gv_out, const float* __restrict__ gt_out,
    float* __restrict__ g_vec, float* __restrict__ g_agg, float* __restrict__ g_ten, float* __restrict__ g_gates,
    int n, int flags, float c2
) {
    int i = blockIdx.x;
    if (i >= n) return;
    int tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
    __shared__ float X[V_N], GB[V_N], GUO[96 * 3], GTB[48 * 5], scl[2], dts[2];
    load_vt(vec, agg, ten, i, flags, c2, X, scl, tid, lane, sg);
    float su = scl[0], stb = scl[1];
    const float* gt = gates + (long)i * 96;
    float* gg = g_gates + (long)i * 96;
    int k = tid;
    float p[5], q[5], y[5], m[6], x[3], d[5], gk[5];
    if (k < 32) {
        proj<32, 3>(W + WB_UO, 96, k, X + V_VN, su, p); proj<32, 3>(W + WB_UO, 96, 32 + k, X + V_VN, su, q);
        proj<16, 5>(W + WB_TB, 48, 16 + k, X + V_TN, stb, y); mat6(y, m); matvec(m, q, x);
        for (int c = 0; c < 3; ++c) gk[c] = gv_out[(long)i * 96 + k * 3 + c];
        float g1 = gt[k], g2 = gt[32 + k];
        gg[k] = dotm<3>(gk, p); gg[32 + k] = dotm<3>(gk, x);
        storeg<3>(GUO, k, g1, gk);
        matvec(m, gk, d); storeg<3>(GUO, 32 + k, g2, d);
        l2pair(gk, q, d); storeg<5>(GTB, 16 + k, g2, d);
    } else if (k < 48) { int j = k - 32;
        float l[5];
        proj<16, 5>(W + WB_TB, 48, j, X + V_TN, stb, y);
        proj<32, 3>(W + WB_UO, 96, 64 + j, X + V_VN, su, p); proj<32, 3>(W + WB_UO, 96, 80 + j, X + V_VN, su, q); l2pair(p, q, l);
        for (int c = 0; c < 5; ++c) gk[c] = gt_out[(long)i * 80 + j * 5 + c];
        float g3 = gt[64 + j], g4 = gt[80 + j];
        gg[64 + j] = dotm<5>(gk, y); gg[80 + j] = dotm<5>(gk, l);
        storeg<5>(GTB, j, g3, gk);
        mat6(gk, m); matvec(m, q, d); storeg<3>(GUO, 64 + j, g4, d);
        matvec(m, p, d); storeg<3>(GUO, 80 + j, g4, d);
    }
    __syncthreads();
    for (int e = tid; e < V_N; e += 64) {
        float acc = 0.0f;
        if (e < V_TN) { int c = e / 3, mm = e % 3;
            const float* w1 = W + WB_UO + c * 96; for (int r = 0; r < 96; ++r) acc = fmaf(__ldg(w1 + r), GUO[r * 3 + mm], acc);
        } else { int c = (e - V_TN) / 5, mm = (e - V_TN) % 5;
            const float* w1 = W + WB_TB + c * 48; for (int r = 0; r < 48; ++r) acc = fmaf(__ldg(w1 + r), GTB[r * 5 + mm], acc);
        }
        GB[e] = acc;
    }
    __syncthreads();
    if (sg == 0) { float dd = dotsg(GB + V_VN, X + V_VN, 96, lane); if (lane == 0) dts[0] = dd; }
    if (sg == 1) { float dd = dotsg(GB + V_TN, X + V_TN, 80, lane); if (lane == 0) dts[1] = dd; }
    __syncthreads();
    Lay L = lay(flags);
    float k0 = su * su * su / (c2 * 32.0f) * dts[0], k1 = stb * stb * stb / (c2 * 16.0f) * dts[1];
    float* ga = g_agg + (long)i * L.out_dim;
    for (int e = tid; e < L.out_dim; e += 64) {
        float r = 0.0f;
        if (e >= L.av && e < L.av + 96) { int j = e - L.av; r = gv_out[(long)i * 96 + j] + su * GB[V_VN + j] - k0 * X[V_VN + j]; }
        else if (e >= L.at && e < L.at + 80) { int j = e - L.at; r = gt_out[(long)i * 80 + j] + stb * GB[V_TN + j] - k1 * X[V_TN + j]; }
        ga[e] = r;
    }
    for (int j = tid; j < 96; j += 64) g_vec[(long)i * 96 + j] = gv_out[(long)i * 96 + j] + su * GB[V_VN + j] - k0 * X[V_VN + j];
    if (flags & F_TEN) {
        for (int j = tid; j < 80; j += 64) g_ten[(long)i * 80 + j] = gt_out[(long)i * 80 + j] + stb * GB[V_TN + j] - k1 * X[V_TN + j];
    }
}

__global__ void cuda_swiglu_fwd_v4_k(const float4* __restrict__ gv, float4* __restrict__ out, int N, int H4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * H4) return;
    int i = tid / H4, j = tid - i * H4;
    long row2 = (long)i * (2 * H4);
    float4 g = gv[row2 + j], v = gv[row2 + H4 + j];
    auto act = [](float gx, float vx) { float s = 1.0f / (1.0f + expf(-gx)); return (gx * s) * vx; };
    out[tid] = make_float4(act(g.x, v.x), act(g.y, v.y), act(g.z, v.z), act(g.w, v.w));
}

__global__ void cuda_swiglu_bwd_v4_k(
    const float4* __restrict__ gv, const float4* __restrict__ gy, float4* __restrict__ g_gv, int N, int H4
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * H4) return;
    int i = tid / H4, j = tid - i * H4;
    long row2 = (long)i * (2 * H4);
    float4 g = gv[row2 + j], v = gv[row2 + H4 + j], go = gy[tid];
    auto bwd_g = [](float gx, float vx, float gox) { float s = 1.0f / (1.0f + expf(-gx)); return (gox * vx) * (s * (1.0f + gx * (1.0f - s))); };
    auto bwd_v = [](float gx, float gox) { float s = 1.0f / (1.0f + expf(-gx)); return gox * (gx * s); };
    g_gv[row2 + j] = make_float4(bwd_g(g.x, v.x, go.x), bwd_g(g.y, v.y, go.y), bwd_g(g.z, v.z, go.z), bwd_g(g.w, v.w, go.w));
    g_gv[row2 + H4 + j] = make_float4(bwd_v(g.x, go.x), bwd_v(g.y, go.y), bwd_v(g.z, go.z), bwd_v(g.w, go.w));
}

__global__ void cuda_env_swiglu_act_fwd_v4_k(
    const float4* __restrict__ gv, const float4* __restrict__ dm, float4* __restrict__ out, int N, int H4
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * H4) return;
    int i = tid / H4, j = tid - i * H4;
    long row2 = (long)i * (2 * H4);
    float4 g = gv[row2 + j], v = gv[row2 + H4 + j];
    float4 dg = dm[row2 + j], dv = dm[row2 + H4 + j];
    auto act = [](float gx, float vx, float dgx, float dvx) {
        float u = gx * (1.0f + 0.5f * tanhf(dgx));
        float sig = 1.0f / (1.0f + expf(-u));
        return (u * sig) * (vx * (1.0f + 0.5f * tanhf(dvx)));
    };
    out[tid] = make_float4(act(g.x, v.x, dg.x, dv.x), act(g.y, v.y, dg.y, dv.y),
                           act(g.z, v.z, dg.z, dv.z), act(g.w, v.w, dg.w, dv.w));
}

__global__ void cuda_env_swiglu_act_bwd_v4_k(
    const float4* __restrict__ gv, const float4* __restrict__ dm, const float4* __restrict__ gy,
    float4* __restrict__ g_gv, float4* __restrict__ g_dm, int N, int H4
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * H4) return;
    int i = tid / H4, j = tid - i * H4;
    long row2 = (long)i * (2 * H4);
    float4 g = gv[row2 + j], v = gv[row2 + H4 + j];
    float4 dg = dm[row2 + j], dv = dm[row2 + H4 + j], go = gy[tid];
    auto step = [](float gx, float vx, float dgx, float dvx, float gox, float& ggx, float& gvx, float& gdgx, float& gdvx) {
        float tg = tanhf(dgx), tv = tanhf(dvx);
        float mg = 1.0f + 0.5f * tg, mv = 1.0f + 0.5f * tv;
        float u = gx * mg;
        float sig = 1.0f / (1.0f + expf(-u));
        float s = u * sig;
        float ds = gox * (vx * mv);
        float dv_eff = gox * s;
        float du = ds * (sig * (1.0f + u * (1.0f - sig)));
        ggx = du * mg;
        gvx = dv_eff * mv;
        gdgx = (du * gx) * (0.5f * (1.0f - tg * tg));
        gdvx = (dv_eff * vx) * (0.5f * (1.0f - tv * tv));
    };
    float4 gg, gv_o, gdg, gdv;
    step(g.x, v.x, dg.x, dv.x, go.x, gg.x, gv_o.x, gdg.x, gdv.x);
    step(g.y, v.y, dg.y, dv.y, go.y, gg.y, gv_o.y, gdg.y, gdv.y);
    step(g.z, v.z, dg.z, dv.z, go.z, gg.z, gv_o.z, gdg.z, gdv.z);
    step(g.w, v.w, dg.w, dv.w, go.w, gg.w, gv_o.w, gdg.w, gdv.w);
    g_gv[row2 + j] = gg;
    g_gv[row2 + H4 + j] = gv_o;
    g_dm[row2 + j] = gdg;
    g_dm[row2 + H4 + j] = gdv;
}
""".replace("@Y3_FWD@", _Y3_FWD).replace("@Y3_BWD@", _Y3_BWD)
