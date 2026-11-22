"""MSL source of the fused per-atom invariant block of a Stage (`lam4::node_inv_fwd` / `lam4::node_inv_bwd`).

One threadgroup (128 threads) per atom. Inputs are the stage's raw per-atom streams: vec [N, 32, 3], the edge-op output agg [N, 360 | 408]
and ten [N, 16, 5] (LATER stages). The kernel forms v_new = vec + v, t_new = ten + t, their bounded versions (vector_bound.bound_vectors),
and every invariant channel of Stage._post_impl + AngularBranch.features in the concat order of all_inv:
    nu2_v 128 | nu2_t 64 | nu3 64 | (LATER: lon2 16 | cross_v 32 | cross_t 16 | lon 32) | ttt 16 | vttv 32 | nu2_3 32 | nu3_vt3 32
Each channel is a contraction of a few channel projections y = W_r . x of one atom's small inputs, so one thread owns a channel and no
intermediate leaves the threadgroup. Projections are linear, so bounded inputs are handled as scale * (W . raw).
The backward (analytic) recomputes the projections, stores d(channel)/d(projection) per weight row in threadgroup memory (G), contracts
with W^T per input element and chains through bound_vectors: g_raw = s g_b - s^3 / (c^2 K) (g_b . raw) raw.
Weights: one packed buffer, every block transposed to [in][rows] (metal_node.pack_inv_weights); offsets below.
"""
from __future__ import annotations

from lam4.core.angular import _y3_terms


def _y3_code() -> tuple[str, str]:
    """MSL for Y[j] = sum c v[a] m[p] (forward) and its gradient (gv[a], gm[p] from gy[j]) from the float64 term table."""
    fwd, bwd = [], []
    for j, terms in enumerate(_y3_terms()):
        fwd.append(f"    Y[{j}] = " + " + ".join(f"({c!r}f) * v[{a}] * m[{p}]" for a, p, c in terms) + ";")
        for a, p, c in terms:
            bwd.append(f"    gv[{a}] = fma(gy[{j}] * ({c!r}f), m[{p}], gv[{a}]); gm[{p}] = fma(gy[{j}] * ({c!r}f), v[{a}], gm[{p}]);")
    return "\n".join(fwd), "\n".join(bwd)


_Y3_FWD, _Y3_BWD = _y3_code()

METAL_NODE_COMMON = r"""
#include <metal_stdlib>
using namespace metal;

constant float R2 = 0.70710678118654752440f;
constant float R6 = 0.40824829046386301637f;
constant int F_LATER = 1;
constant int F_TEN = 2;
constant int F_CUMT = 4;

inline void mat6(thread const float* t, thread float* m) {   // (xx, yy, zz, xy, xz, yz) of M(t), lam4.core.angular.l2_mat6
    m[0] = t[3] * R2 - t[4] * R6; m[1] = -t[3] * R2 - t[4] * R6; m[2] = 2.0f * R6 * t[4];
    m[3] = t[0] * R2; m[4] = t[1] * R2; m[5] = t[2] * R2;
}
inline void matvec(thread const float* m, thread const float* p, thread float* o) {
    o[0] = m[0] * p[0] + m[3] * p[1] + m[4] * p[2];
    o[1] = m[3] * p[0] + m[1] * p[1] + m[5] * p[2];
    o[2] = m[4] * p[0] + m[5] * p[1] + m[2] * p[2];
}
inline void l2pair(thread const float* p, thread const float* q, thread float* o) {   // angular.l2_pair
    o[0] = (p[0] * q[1] + p[1] * q[0]) * R2; o[1] = (p[0] * q[2] + p[2] * q[0]) * R2; o[2] = (p[1] * q[2] + p[2] * q[1]) * R2;
    o[3] = (p[0] * q[0] - p[1] * q[1]) * R2; o[4] = (2.0f * p[2] * q[2] - p[0] * q[0] - p[1] * q[1]) * R6;
}
inline void gm6_to_gt(thread const float* gm, thread float* gt) {   // gradient of mat6 inputs from gradients of its 6 entries
    gt[0] = R2 * gm[3]; gt[1] = R2 * gm[4]; gt[2] = R2 * gm[5];
    gt[3] = R2 * (gm[0] - gm[1]); gt[4] = R6 * (2.0f * gm[2] - gm[0] - gm[1]);
}
template <int IN, int M>
inline void proj(device const float* W, int rows, int r, threadgroup const float* x, float s, thread float* o) {
    for (int m = 0; m < M; ++m) o[m] = 0.0f;
    for (int c = 0; c < IN; ++c) {
        float w = W[c * rows + r];
        for (int m = 0; m < M; ++m) o[m] = fma(w, x[c * M + m], o[m]);
    }
    for (int m = 0; m < M; ++m) o[m] *= s;
}
template <int M>
inline float dotm(thread const float* a, thread const float* b) {
    float s = 0.0f;
    for (int m = 0; m < M; ++m) s = fma(a[m], b[m], s);
    return s;
}
template <int M>
inline void storeg(threadgroup float* G, int r, float g, thread const float* d) {
    for (int m = 0; m < M; ++m) G[r * M + m] = g * d[m];
}
inline float sumsq(threadgroup const float* x, int L, uint lane) {
    float s = 0.0f;
    for (int i = int(lane); i < L; i += 32) s = fma(x[i], x[i], s);
    return simd_sum(s);
}
inline float dotsg(threadgroup const float* a, threadgroup const float* b, int L, uint lane) {
    float s = 0.0f;
    for (int i = int(lane); i < L; i += 32) s = fma(a[i], b[i], s);
    return simd_sum(s);
}
inline float bound_scale(float ss, float k, float c2) { return precise::rsqrt(1.0f + (ss / k) / c2); }
"""

_INV_SRC = r"""
constant int WA_UVPQ = 0;       // [32][384]  u1 | v1 | p3 | q3
constant int WA_UVT = 12288;    // [16][192]  u2 | v2 | t3
constant int WA_YT = 15360;     // [16][112]  ttt | vttv_a | vttv_b | a3
constant int WA_YV = 17152;     // [32][96]   vttv_p | vttv_q | p3(ang)
constant int WA_U3 = 20224;     // [8][96]    u3 | v3 | w3
constant int WA_UCV = 20992;    // [32][32]
constant int WA_VCV = 22016;    // [32][32]
constant int WA_UCT = 23040;    // [16][16]
constant int WA_VCT = 23296;    // [16][16]

// raw per-atom inputs in threadgroup memory: vn 96 | tn 80 | o 56 | vec 96 | pv 96 | ten 80 | tt 80
constant int X_VN = 0, X_TN = 96, X_O = 176, X_VEC = 232, X_PV = 328, X_TEN = 424, X_TT = 504, X_N = 584;

inline void y3vm(thread const float* v, thread const float* m, thread float* Y) {
@Y3_FWD@
}
inline void y3vm_bwd(thread const float* v, thread const float* m, thread const float* gy, thread float* gv, thread float* gm) {
    for (int a = 0; a < 3; ++a) gv[a] = 0.0f;
    for (int p = 0; p < 6; ++p) gm[p] = 0.0f;
@Y3_BWD@
}

struct Lay { int av, at, ao, alon, alon2, out_dim, ang, inv_dim; };
inline Lay lay(int flags) {
    Lay L;
    if (flags & F_LATER) { L.av = 128; L.alon = 224; L.at = 256; L.alon2 = 336; L.ao = 352; L.out_dim = 408; L.ang = 352; L.inv_dim = 464; }
    else { L.av = 128; L.alon = -1; L.at = 224; L.alon2 = -1; L.ao = 304; L.out_dim = 360; L.ang = 256; L.inv_dim = 368; }
    return L;
}

// load raw streams of one atom; scl = bound scales of (vn, tn, vec, pv, ten, tt) (tn scale 1 without CUMT)
inline void load_atom(device const float* vec, device const float* agg, device const float* ten, int i, int flags, Lay L, float c2,
                      threadgroup float* X, threadgroup float* scl, uint tid, uint lane, uint sg) {
    device const float* a = agg + (long)i * L.out_dim;
    bool has_ten = (flags & F_TEN) != 0;
    for (int e = int(tid); e < 96; e += 128) {
        float v0 = vec[(long)i * 96 + e], pv = a[L.av + e];
        X[X_VN + e] = v0 + pv; X[X_VEC + e] = v0; X[X_PV + e] = pv;
    }
    for (int e = int(tid); e < 80; e += 128) {
        float t = a[L.at + e], t0 = has_ten ? ten[(long)i * 80 + e] : 0.0f;
        X[X_TN + e] = (has_ten && (flags & F_CUMT)) ? t0 + t : t; X[X_TEN + e] = t0; X[X_TT + e] = t;   // contraction input: bound(t_new) or t
    }
    for (int e = int(tid); e < 56; e += 128) X[X_O + e] = a[L.ao + e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float s = sumsq(X + X_VN, 96, lane); if (lane == 0) scl[0] = bound_scale(s, 32.0f, c2); }
    if (sg == 1) { float s = sumsq(X + X_TN, 80, lane); if (lane == 0) scl[1] = (flags & F_CUMT) ? bound_scale(s, 16.0f, c2) : 1.0f; }
    if (sg == 2) { float s = sumsq(X + X_VEC, 96, lane), s2 = sumsq(X + X_PV, 96, lane);
                   if (lane == 0) { scl[2] = bound_scale(s, 32.0f, c2); scl[3] = bound_scale(s2, 32.0f, c2); } }
    if (sg == 3) { float s = sumsq(X + X_TEN, 80, lane), s2 = sumsq(X + X_TT, 80, lane);
                   if (lane == 0) { scl[4] = bound_scale(s, 16.0f, c2); scl[5] = bound_scale(s2, 16.0f, c2); } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void node_inv_fwd(
    device const float* vec [[buffer(0)]], device const float* agg [[buffer(1)]], device const float* ten [[buffer(2)]],
    device const float* W [[buffer(3)]], device float* out [[buffer(4)]],
    constant int& n [[buffer(5)]], constant int& flags [[buffer(6)]], constant float& c2 [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    int i = int(gid);
    if (i >= n) return;
    Lay L = lay(flags);
    threadgroup float X[X_N];
    threadgroup float scl[6];
    load_atom(vec, agg, ten, i, flags, L, c2, X, scl, tid, lane, sg);
    float su = scl[0], st = scl[1];
    device float* y = out + (long)i * L.inv_dim;
    device const float* a = agg + (long)i * L.out_dim;
    for (int o = int(tid); o < L.inv_dim; o += 128) {
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
        } else if (o < L.ang) {   // LATER only: lon2 | cross_v | cross_t | lon
            if (o < 272) r = a[L.alon2 + (o - 256)];
            else if (o < 304) { int k = o - 272;
                proj<32, 3>(W + WA_UCV, 32, k, X + X_VEC, scl[2], p); proj<32, 3>(W + WA_VCV, 32, k, X + X_PV, scl[3], q); r = dotm<3>(p, q); }
            else if (o < 320) { int k = o - 304;
                if (flags & F_TEN) { proj<16, 5>(W + WA_UCT, 16, k, X + X_TEN, scl[4], p); proj<16, 5>(W + WA_VCT, 16, k, X + X_TT, scl[5], q); r = dotm<5>(p, q); } }
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
"""
