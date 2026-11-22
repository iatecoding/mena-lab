"""MSL source of the analytic backward of the fused invariant block (`lam4::node_inv_bwd`); see metal_node_src.py for the layout.

Memory-packed (< 16 KB threadgroup memory for 2x occupancy on Apple Silicon) and coalesced simdgroup-reduced W^T G:
- Shared memory: X[232] (vn 96 | tn 80 | o 56) + G[3632] (G_U 1440 | G_T 1520 | G_O 672) + scl[6] + dts[6] = 15,504 B (< 16 KB).
- Phase 1: thread per invariant channel recomputes projections and writes g * d(channel)/d(proj) into G_U / G_T / G_O.
- Phase 2a: 4 simdgroups cooperatively contract W^T G per input channel `c` with coalesced 32-lane loads of W[c * rows + r]
  (loading each weight once for all M spatial components). Computes GB_O first into G[G_O..G_O+55], then GB_TN directly into
  G[G_O+56..G_O+135], then GB_VN directly into G[G_O+136..G_O+231] with zero register-array spilling.
- Phase 2b (LATER only): reuses G[0..1055] for raw (vec, pv, ten, tt), cross_v/cross_t gradients, and their W^T G.
- Phase 3: bound_vectors backward per stream and deterministic outputs g_vec [N, 96], g_agg [N, out_dim], g_ten [N, 80].
"""
from __future__ import annotations

_INV_BWD_SRC = r"""
constant int G_U = 0, G_T = 1440, G_O = 2960, G_MAIN = 3632;
// GB_O, GB_TN, GB_VN live in G[2960..3191] after G_O is consumed:
constant int GB_O_OFF = 2960, GB_TN_OFF = 3016, GB_VN_OFF = 3096;
// Reuse layout inside G[0..1055] for LATER cross_v / cross_t after Phase 2a:
constant int R_X2 = 0, R_GCV = 352, R_GCT = 544, R_GB2 = 704;

kernel void node_inv_bwd(
    device const float* vec [[buffer(0)]], device const float* agg [[buffer(1)]], device const float* ten [[buffer(2)]],
    device const float* W [[buffer(3)]], device const float* gin [[buffer(4)]],
    device float* g_vec [[buffer(5)]], device float* g_agg [[buffer(6)]], device float* g_ten [[buffer(7)]],
    constant int& n [[buffer(8)]], constant int& flags [[buffer(9)]], constant float& c2 [[buffer(10)]],
    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    int i = int(gid);
    if (i >= n) return;
    Lay L = lay(flags);
    bool later = (flags & F_LATER) != 0, has_ten = (flags & F_TEN) != 0, cumt = (flags & F_CUMT) != 0;
    threadgroup float X[232];
    threadgroup float G[G_MAIN];
    threadgroup float scl[6];
    threadgroup float dts[6];

    device const float* a = agg + (long)i * L.out_dim;
    for (int e = int(tid); e < 96; e += 128) X[X_VN + e] = vec[(long)i * 96 + e] + a[L.av + e];
    for (int e = int(tid); e < 80; e += 128) {
        float t = a[L.at + e], t0 = has_ten ? ten[(long)i * 80 + e] : 0.0f;
        X[X_TN + e] = (has_ten && cumt) ? t0 + t : t;
    }
    for (int e = int(tid); e < 56; e += 128) X[X_O + e] = a[L.ao + e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float s = sumsq(X + X_VN, 96, lane); if (lane == 0) scl[0] = bound_scale(s, 32.0f, c2); }
    if (sg == 1) { float s = sumsq(X + X_TN, 80, lane); if (lane == 0) scl[1] = cumt ? bound_scale(s, 16.0f, c2) : 1.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float su = scl[0], st = scl[1];
    device const float* gi = gin + (long)i * L.inv_dim;

    // Phase 1: main invariant channels -> G_U, G_T, G_O
    for (int o = int(tid); o < L.inv_dim; o += 128) {
        float g = gi[o];
        if (o < 128) {
            float p[3], q[3];
            proj<32, 3>(W + WA_UVPQ, 384, o, X + X_VN, su, p); proj<32, 3>(W + WA_UVPQ, 384, 128 + o, X + X_VN, su, q);
            storeg<3>(G + G_U, o, g, q); storeg<3>(G + G_U, 128 + o, g, p);
        } else if (o < 192) { int k = o - 128;
            float p[5], q[5];
            proj<16, 5>(W + WA_UVT, 192, k, X + X_TN, st, p); proj<16, 5>(W + WA_UVT, 192, 64 + k, X + X_TN, st, q);
            storeg<5>(G + G_T, k, g, q); storeg<5>(G + G_T, 64 + k, g, p);
        } else if (o < 256) { int k = o - 192;
            float p[3], q[3], t[5], m[6], d1[5], d2[3];
            proj<32, 3>(W + WA_UVPQ, 384, 256 + k, X + X_VN, su, p); proj<32, 3>(W + WA_UVPQ, 384, 320 + k, X + X_VN, su, q);
            proj<16, 5>(W + WA_UVT, 192, 128 + k, X + X_TN, st, t);
            l2pair(p, q, d1); storeg<5>(G + G_T, 128 + k, g, d1);
            mat6(t, m); matvec(m, q, d1); matvec(m, p, d2);
            storeg<3>(G + G_U, 256 + k, g, d1); storeg<3>(G + G_U, 320 + k, g, d2);
        } else if (o >= L.ang) {
            int h = o - L.ang;
            if (h < 16) {
                float t[5], m[6], d1[5];
                proj<16, 5>(W + WA_YT, 112, h, X + X_TN, st, t); mat6(t, m);
                float cof[6];
                cof[0] = m[1] * m[2] - m[5] * m[5]; cof[1] = m[0] * m[2] - m[4] * m[4]; cof[2] = m[0] * m[1] - m[3] * m[3];
                cof[3] = m[4] * m[5] - m[3] * m[2]; cof[4] = m[3] * m[5] - m[1] * m[4]; cof[5] = m[3] * m[4] - m[0] * m[5];
                float gm[6] = {3.0f * cof[0], 3.0f * cof[1], 3.0f * cof[2], 6.0f * cof[3], 6.0f * cof[4], 6.0f * cof[5]};
                gm6_to_gt(gm, d1); storeg<5>(G + G_T, 192 + h, g, d1);
            } else if (h < 48) { int k = h - 16;
                float ta[5], tb[5], ma[6], mb[6], p[3], q[3], x[3], z[3], d1[5];
                proj<16, 5>(W + WA_YT, 112, 16 + k, X + X_TN, st, ta); mat6(ta, ma);
                proj<16, 5>(W + WA_YT, 112, 48 + k, X + X_TN, st, tb); mat6(tb, mb);
                proj<32, 3>(W + WA_YV, 96, k, X + X_VN, su, p); proj<32, 3>(W + WA_YV, 96, 32 + k, X + X_VN, su, q);
                matvec(ma, p, x); matvec(mb, q, z);
                l2pair(p, z, d1); storeg<5>(G + G_T, 192 + 16 + k, g, d1);
                l2pair(q, x, d1); storeg<5>(G + G_T, 192 + 48 + k, g, d1);
                matvec(ma, z, d1); storeg<3>(G + G_U, 384 + k, g, d1);
                matvec(mb, x, d1); storeg<3>(G + G_U, 384 + 32 + k, g, d1);
            } else if (h < 80) { int k = h - 48;
                float p[7], q[7];
                proj<8, 7>(W + WA_U3, 96, k, X + X_O, 1.0f, p); proj<8, 7>(W + WA_U3, 96, 32 + k, X + X_O, 1.0f, q);
                storeg<7>(G + G_O, k, g, q); storeg<7>(G + G_O, 32 + k, g, p);
            } else { int k = h - 80;
                float p[3], q[7], t[5], m[6], Y[7], gv[3], gm[6], d1[5];
                proj<8, 7>(W + WA_U3, 96, 64 + k, X + X_O, 1.0f, q); proj<32, 3>(W + WA_YV, 96, 64 + k, X + X_VN, su, p);
                proj<16, 5>(W + WA_YT, 112, 80 + k, X + X_TN, st, t); mat6(t, m); y3vm(p, m, Y);
                storeg<7>(G + G_O, 64 + k, g, Y);
                y3vm_bwd(p, m, q, gv, gm);
                storeg<3>(G + G_U, 384 + 64 + k, g, gv);
                gm6_to_gt(gm, d1); storeg<5>(G + G_T, 192 + 80 + k, g, d1);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2a: coalesced simdgroup-reduced W^T G (O -> TN -> VN, writing into G[2960..3191] once G_O is consumed)
    float gb_o[2][7];
    for (int ci = 0; ci < 2; ++ci) {
        int c = int(sg) + ci * 4;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f;
        device const float* w1 = W + WA_U3 + c * 96;
        for (int r = int(lane); r < 96; r += 32) {
            float w = w1[r];
            int idx = G_O + r * 7;
            a0 = fma(w, G[idx], a0); a1 = fma(w, G[idx + 1], a1); a2 = fma(w, G[idx + 2], a2); a3 = fma(w, G[idx + 3], a3);
            a4 = fma(w, G[idx + 4], a4); a5 = fma(w, G[idx + 5], a5); a6 = fma(w, G[idx + 6], a6);
        }
        gb_o[ci][0] = simd_sum(a0); gb_o[ci][1] = simd_sum(a1); gb_o[ci][2] = simd_sum(a2); gb_o[ci][3] = simd_sum(a3);
        gb_o[ci][4] = simd_sum(a4); gb_o[ci][5] = simd_sum(a5); gb_o[ci][6] = simd_sum(a6);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        for (int ci = 0; ci < 2; ++ci) { int c = int(sg) + ci * 4; for (int m = 0; m < 7; ++m) G[GB_O_OFF + c * 7 + m] = gb_o[ci][m]; }
    }
    // G[2960+56 .. 3631] is now free: write GB_TN and GB_VN directly with zero array-in-register storage!
    for (int c = int(sg); c < 16; c += 4) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f;
        device const float* w1 = W + WA_UVT + c * 192;
        for (int r = int(lane); r < 192; r += 32) {
            float w = w1[r];
            int idx = G_T + r * 5;
            a0 = fma(w, G[idx], a0); a1 = fma(w, G[idx + 1], a1); a2 = fma(w, G[idx + 2], a2); a3 = fma(w, G[idx + 3], a3); a4 = fma(w, G[idx + 4], a4);
        }
        device const float* w2 = W + WA_YT + c * 112;
        for (int r = int(lane); r < 112; r += 32) {
            float w = w2[r];
            int idx = G_T + (192 + r) * 5;
            a0 = fma(w, G[idx], a0); a1 = fma(w, G[idx + 1], a1); a2 = fma(w, G[idx + 2], a2); a3 = fma(w, G[idx + 3], a3); a4 = fma(w, G[idx + 4], a4);
        }
        a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3); a4 = simd_sum(a4);
        if (lane == 0) {
            G[GB_TN_OFF + c * 5] = a0; G[GB_TN_OFF + c * 5 + 1] = a1; G[GB_TN_OFF + c * 5 + 2] = a2;
            G[GB_TN_OFF + c * 5 + 3] = a3; G[GB_TN_OFF + c * 5 + 4] = a4;
        }
    }
    for (int c = int(sg); c < 32; c += 4) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        device const float* w1 = W + WA_UVPQ + c * 384;
        for (int r = int(lane); r < 384; r += 32) {
            float w = w1[r];
            int idx = G_U + r * 3;
            a0 = fma(w, G[idx], a0); a1 = fma(w, G[idx + 1], a1); a2 = fma(w, G[idx + 2], a2);
        }
        device const float* w2 = W + WA_YV + c * 96;
        for (int r = int(lane); r < 96; r += 32) {
            float w = w2[r];
            int idx = G_U + (384 + r) * 3;
            a0 = fma(w, G[idx], a0); a1 = fma(w, G[idx + 1], a1); a2 = fma(w, G[idx + 2], a2);
        }
        a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2);
        if (lane == 0) { G[GB_VN_OFF + c * 3] = a0; G[GB_VN_OFF + c * 3 + 1] = a1; G[GB_VN_OFF + c * 3 + 2] = a2; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2b (LATER only): cross_v & cross_t using G[0..1055]
    if (later) {
        for (int e = int(tid); e < 96; e += 128) { G[R_X2 + e] = vec[(long)i * 96 + e]; G[R_X2 + 96 + e] = a[L.av + e]; }
        for (int e = int(tid); e < 80; e += 128) { G[R_X2 + 192 + e] = has_ten ? ten[(long)i * 80 + e] : 0.0f; G[R_X2 + 272 + e] = a[L.at + e]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) { float d = dotsg(G + GB_VN_OFF, X + X_VN, 96, lane); if (lane == 0) dts[0] = d; }
        if (sg == 1) { float d = dotsg(G + GB_TN_OFF, X + X_TN, 80, lane); if (lane == 0) dts[1] = d; }
        if (sg == 2) { float s = sumsq(G + R_X2, 96, lane), s2 = sumsq(G + R_X2 + 96, 96, lane); if (lane == 0) { scl[2] = bound_scale(s, 32.0f, c2); scl[3] = bound_scale(s2, 32.0f, c2); } }
        if (sg == 3) { float s = sumsq(G + R_X2 + 192, 80, lane), s2 = sumsq(G + R_X2 + 272, 80, lane); if (lane == 0) { scl[4] = bound_scale(s, 16.0f, c2); scl[5] = bound_scale(s2, 16.0f, c2); } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(tid) < 32) {
            int k = int(tid); float g = gi[272 + k], p[3], q[3];
            proj<32, 3>(W + WA_UCV, 32, k, G + R_X2, scl[2], p);
            proj<32, 3>(W + WA_VCV, 32, k, G + R_X2 + 96, scl[3], q);
            storeg<3>(G + R_GCV, k, g, q); storeg<3>(G + R_GCV, 32 + k, g, p);
        } else if (int(tid) < 48) {
            int k = int(tid) - 32;
            if (has_ten) {
                float g = gi[304 + k], p[5], q[5];
                proj<16, 5>(W + WA_UCT, 16, k, G + R_X2 + 192, scl[4], p);
                proj<16, 5>(W + WA_VCT, 16, k, G + R_X2 + 272, scl[5], q);
                storeg<5>(G + R_GCT, k, g, q); storeg<5>(G + R_GCT, 16 + k, g, p);
            } else {
                for (int m = 0; m < 5; ++m) { G[R_GCT + k * 5 + m] = 0.0f; G[R_GCT + (16 + k) * 5 + m] = 0.0f; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = int(sg); c < 32; c += 4) {
            float wv = W[WA_UCV + c * 32 + int(lane)], wp = W[WA_VCV + c * 32 + int(lane)];
            int iv = R_GCV + int(lane) * 3, ip = R_GCV + (32 + int(lane)) * 3;
            float v0 = simd_sum(wv * G[iv]), v1 = simd_sum(wv * G[iv + 1]), v2 = simd_sum(wv * G[iv + 2]);
            float p0 = simd_sum(wp * G[ip]), p1 = simd_sum(wp * G[ip + 1]), p2 = simd_sum(wp * G[ip + 2]);
            if (lane == 0) {
                G[R_GB2 + c * 3] = v0; G[R_GB2 + c * 3 + 1] = v1; G[R_GB2 + c * 3 + 2] = v2;
                G[R_GB2 + 96 + c * 3] = p0; G[R_GB2 + 96 + c * 3 + 1] = p1; G[R_GB2 + 96 + c * 3 + 2] = p2;
            }
        }
        for (int c = int(sg); c < 16; c += 4) {
            float wt = (lane < 16) ? W[WA_UCT + c * 16 + int(lane)] : 0.0f;
            float wq = (lane < 16) ? W[WA_VCT + c * 16 + int(lane)] : 0.0f;
            int it = R_GCT + int(lane & 15) * 5, iq = R_GCT + (16 + int(lane & 15)) * 5;
            float t0 = simd_sum(wt * G[it]), t1 = simd_sum(wt * G[it + 1]), t2 = simd_sum(wt * G[it + 2]), t3 = simd_sum(wt * G[it + 3]), t4 = simd_sum(wt * G[it + 4]);
            float q0 = simd_sum(wq * G[iq]), q1 = simd_sum(wq * G[iq + 1]), q2 = simd_sum(wq * G[iq + 2]), q3 = simd_sum(wq * G[iq + 3]), q4 = simd_sum(wq * G[iq + 4]);
            if (lane == 0) {
                G[R_GB2 + 192 + c * 5] = t0; G[R_GB2 + 192 + c * 5 + 1] = t1; G[R_GB2 + 192 + c * 5 + 2] = t2; G[R_GB2 + 192 + c * 5 + 3] = t3; G[R_GB2 + 192 + c * 5 + 4] = t4;
                G[R_GB2 + 272 + c * 5] = q0; G[R_GB2 + 272 + c * 5 + 1] = q1; G[R_GB2 + 272 + c * 5 + 2] = q2; G[R_GB2 + 272 + c * 5 + 3] = q3; G[R_GB2 + 272 + c * 5 + 4] = q4;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 2) { float d = dotsg(G + R_GB2, G + R_X2, 96, lane), d2 = dotsg(G + R_GB2 + 96, G + R_X2 + 96, 96, lane); if (lane == 0) { dts[2] = d; dts[3] = d2; } }
        if (sg == 3) { float d = dotsg(G + R_GB2 + 192, G + R_X2 + 192, 80, lane), d2 = dotsg(G + R_GB2 + 272, G + R_X2 + 272, 80, lane); if (lane == 0) { dts[4] = d; dts[5] = d2; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        if (sg == 0) { float d = dotsg(G + GB_VN_OFF, X + X_VN, 96, lane); if (lane == 0) dts[0] = d; }
        if (sg == 1) { float d = dotsg(G + GB_TN_OFF, X + X_TN, 80, lane); if (lane == 0) dts[1] = d; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: bound_vectors backward, g_raw = s gb - s^3 / (c^2 K) (gb . raw) raw
    float k0 = scl[0] * scl[0] * scl[0] / (c2 * 32.0f) * dts[0];
    float k1 = scl[1] * scl[1] * scl[1] / (c2 * 16.0f) * dts[1];
    float k2 = later ? (scl[2] * scl[2] * scl[2] / (c2 * 32.0f) * dts[2]) : 0.0f;
    float k3 = later ? (scl[3] * scl[3] * scl[3] / (c2 * 32.0f) * dts[3]) : 0.0f;
    float k4 = later ? (scl[4] * scl[4] * scl[4] / (c2 * 16.0f) * dts[4]) : 0.0f;
    float k5 = later ? (scl[5] * scl[5] * scl[5] / (c2 * 16.0f) * dts[5]) : 0.0f;
    device float* ga = g_agg + (long)i * L.out_dim;
    for (int e = int(tid); e < L.out_dim; e += 128) {
        float r = 0.0f;
        if (e >= L.av && e < L.av + 96) { int j = e - L.av;
            r = scl[0] * G[GB_VN_OFF + j] - k0 * X[X_VN + j];
            if (later) r += scl[3] * G[R_GB2 + 96 + j] - k3 * G[R_X2 + 96 + j];
        } else if (e >= L.at && e < L.at + 80) { int j = e - L.at;
            r = cumt ? scl[1] * G[GB_TN_OFF + j] - k1 * X[X_TN + j] : G[GB_TN_OFF + j];
            if (later) r += scl[5] * G[R_GB2 + 272 + j] - k5 * G[R_X2 + 272 + j];
        } else if (e >= L.ao && e < L.ao + 56) { r = G[GB_O_OFF + (e - L.ao)];
        } else if (later && e >= L.alon && e < L.alon + 32) { r = gi[320 + (e - L.alon)];
        } else if (later && e >= L.alon2 && e < L.alon2 + 16) { r = gi[256 + (e - L.alon2)]; }
        ga[e] = r;
    }
    for (int j = int(tid); j < 96; j += 128) {
        float r = scl[0] * G[GB_VN_OFF + j] - k0 * X[X_VN + j];
        if (later) r += scl[2] * G[R_GB2 + j] - k2 * G[R_X2 + j];
        g_vec[(long)i * 96 + j] = r;
    }
    if (has_ten) {
        for (int j = int(tid); j < 80; j += 128) {
            float r = cumt ? scl[1] * G[GB_TN_OFF + j] - k1 * X[X_TN + j] : 0.0f;
            if (later) r += scl[4] * G[R_GB2 + 192 + j] - k4 * G[R_X2 + 192 + j];
            g_ten[(long)i * 80 + j] = r;
        }
    }
}
"""
