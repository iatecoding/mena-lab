"""MSL source of the fused vector / tensor stream update of a Stage (`lam4::node_vec_fwd` / `lam4::node_vec_bwd`).

inv_scale.post_vector_tensor_update (baked, inv_norm) per atom, with v_new = vec + v, t_new = ten + t, u = bound(v_new), tb = bound(t_new):
    v_out[k] = v_new[k] + (S_k u) gv[k] + M(TT_k tb) (TV_k u) gtv[k]                 k < 32
    t_out[j] = t_new[j] + (TS_j tb) gt[j] + l2_pair(VP_j u, VQ_j u) gvv[j]            j < 16
gates [N, 96] = tanh(...) in the order gv 32 | gtv 32 | gt 16 | gvv 16. One threadgroup (64 threads) per atom, thread per output channel.
Weights packed transposed: WB_UO [32][96] = (vec_self | tv_v | vv_p | vv_q)^T, WB_TB [16][48] = (ten_self | tv_t)^T.
"""
from __future__ import annotations

_VEC_SRC = r"""
constant int WB_UO = 0;      // [32][96]
constant int WB_TB = 3072;   // [16][48]
constant int V_VN = 0, V_TN = 96, V_N = 176;

inline void load_vt(device const float* vec, device const float* agg, device const float* ten, int i, int flags, float c2,
                    threadgroup float* X, threadgroup float* scl, uint tid, uint lane, uint sg) {
    Lay L = lay(flags);
    device const float* a = agg + (long)i * L.out_dim;
    bool has_ten = (flags & F_TEN) != 0;
    for (int e = int(tid); e < 96; e += 64) X[V_VN + e] = vec[(long)i * 96 + e] + a[L.av + e];
    for (int e = int(tid); e < 80; e += 64) X[V_TN + e] = has_ten ? ten[(long)i * 80 + e] + a[L.at + e] : a[L.at + e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float s = sumsq(X + V_VN, 96, lane); if (lane == 0) scl[0] = bound_scale(s, 32.0f, c2); }
    if (sg == 1) { float s = sumsq(X + V_TN, 80, lane); if (lane == 0) scl[1] = bound_scale(s, 16.0f, c2); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void node_vec_fwd(
    device const float* vec [[buffer(0)]], device const float* agg [[buffer(1)]], device const float* ten [[buffer(2)]],
    device const float* gates [[buffer(3)]], device const float* W [[buffer(4)]],
    device float* v_out [[buffer(5)]], device float* t_out [[buffer(6)]],
    constant int& n [[buffer(7)]], constant int& flags [[buffer(8)]], constant float& c2 [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    int i = int(gid);
    if (i >= n) return;
    threadgroup float X[V_N];
    threadgroup float scl[2];
    load_vt(vec, agg, ten, i, flags, c2, X, scl, tid, lane, sg);
    float su = scl[0], stb = scl[1];
    device const float* gt = gates + (long)i * 96;
    int k = int(tid);
    float p[5], q[5], y[5], m[6], x[3];
    if (k < 32) {
        proj<32, 3>(W + WB_UO, 96, k, X + V_VN, su, p); proj<32, 3>(W + WB_UO, 96, 32 + k, X + V_VN, su, q);
        proj<16, 5>(W + WB_TB, 48, 16 + k, X + V_TN, stb, y); mat6(y, m); matvec(m, q, x);
        float g1 = gt[k], g2 = gt[32 + k];
        for (int c = 0; c < 3; ++c) v_out[(long)i * 96 + k * 3 + c] = X[V_VN + k * 3 + c] + p[c] * g1 + x[c] * g2;
    } else if (k < 48) { int j = k - 32;
        float l[5];
        proj<16, 5>(W + WB_TB, 48, j, X + V_TN, stb, y);
        proj<32, 3>(W + WB_UO, 96, 64 + j, X + V_VN, su, p); proj<32, 3>(W + WB_UO, 96, 80 + j, X + V_VN, su, q); l2pair(p, q, l);
        float g3 = gt[64 + j], g4 = gt[80 + j];
        for (int c = 0; c < 5; ++c) t_out[(long)i * 80 + j * 5 + c] = X[V_TN + j * 5 + c] + y[c] * g3 + l[c] * g4;
    }
}

kernel void node_vec_bwd(
    device const float* vec [[buffer(0)]], device const float* agg [[buffer(1)]], device const float* ten [[buffer(2)]],
    device const float* gates [[buffer(3)]], device const float* W [[buffer(4)]],
    device const float* gv_out [[buffer(5)]], device const float* gt_out [[buffer(6)]],
    device float* g_vec [[buffer(7)]], device float* g_agg [[buffer(8)]], device float* g_ten [[buffer(9)]], device float* g_gates [[buffer(10)]],
    constant int& n [[buffer(11)]], constant int& flags [[buffer(12)]], constant float& c2 [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    int i = int(gid);
    if (i >= n) return;
    threadgroup float X[V_N];
    threadgroup float GB[V_N];
    threadgroup float GUO[96 * 3];
    threadgroup float GTB[48 * 5];
    threadgroup float scl[2];
    threadgroup float dts[2];
    load_vt(vec, agg, ten, i, flags, c2, X, scl, tid, lane, sg);
    float su = scl[0], stb = scl[1];
    device const float* gt = gates + (long)i * 96;
    device float* gg = g_gates + (long)i * 96;
    int k = int(tid);
    float p[5], q[5], y[5], m[6], x[3], d[5], gk[5];
    if (k < 32) {
        proj<32, 3>(W + WB_UO, 96, k, X + V_VN, su, p); proj<32, 3>(W + WB_UO, 96, 32 + k, X + V_VN, su, q);
        proj<16, 5>(W + WB_TB, 48, 16 + k, X + V_TN, stb, y); mat6(y, m); matvec(m, q, x);
        for (int c = 0; c < 3; ++c) gk[c] = gv_out[(long)i * 96 + k * 3 + c];
        float g1 = gt[k], g2 = gt[32 + k];
        gg[k] = dotm<3>(gk, p); gg[32 + k] = dotm<3>(gk, x);
        storeg<3>(GUO, k, g1, gk);
        matvec(m, gk, d); storeg<3>(GUO, 32 + k, g2, d);      // d/dw = M(y) gk
        l2pair(gk, q, d); storeg<5>(GTB, 16 + k, g2, d);      // d/dy = L(gk, w)
    } else if (k < 48) { int j = k - 32;
        float l[5];
        proj<16, 5>(W + WB_TB, 48, j, X + V_TN, stb, y);
        proj<32, 3>(W + WB_UO, 96, 64 + j, X + V_VN, su, p); proj<32, 3>(W + WB_UO, 96, 80 + j, X + V_VN, su, q); l2pair(p, q, l);
        for (int c = 0; c < 5; ++c) gk[c] = gt_out[(long)i * 80 + j * 5 + c];
        float g3 = gt[64 + j], g4 = gt[80 + j];
        gg[64 + j] = dotm<5>(gk, y); gg[80 + j] = dotm<5>(gk, l);
        storeg<5>(GTB, j, g3, gk);
        mat6(gk, m); matvec(m, q, d); storeg<3>(GUO, 64 + j, g4, d);   // d/dp = M(g) q
        matvec(m, p, d); storeg<3>(GUO, 80 + j, g4, d);                // d/dq = M(g) p
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < V_N; e += 64) {
        float acc = 0.0f;
        if (e < V_TN) { int c = e / 3, mm = e % 3;
            device const float* w1 = W + WB_UO + c * 96; for (int r = 0; r < 96; ++r) acc = fma(w1[r], GUO[r * 3 + mm], acc);
        } else { int c = (e - V_TN) / 5, mm = (e - V_TN) % 5;
            device const float* w1 = W + WB_TB + c * 48; for (int r = 0; r < 48; ++r) acc = fma(w1[r], GTB[r * 5 + mm], acc);
        }
        GB[e] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float dd = dotsg(GB + V_VN, X + V_VN, 96, lane); if (lane == 0) dts[0] = dd; }
    if (sg == 1) { float dd = dotsg(GB + V_TN, X + V_TN, 80, lane); if (lane == 0) dts[1] = dd; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Lay L = lay(flags);
    float k0 = su * su * su / (c2 * 32.0f) * dts[0], k1 = stb * stb * stb / (c2 * 16.0f) * dts[1];
    device float* ga = g_agg + (long)i * L.out_dim;
    for (int e = int(tid); e < L.out_dim; e += 64) {
        float r = 0.0f;
        if (e >= L.av && e < L.av + 96) { int j = e - L.av; r = gv_out[(long)i * 96 + j] + su * GB[V_VN + j] - k0 * X[V_VN + j]; }
        else if (e >= L.at && e < L.at + 80) { int j = e - L.at; r = gt_out[(long)i * 80 + j] + stb * GB[V_TN + j] - k1 * X[V_TN + j]; }
        ga[e] = r;
    }
    for (int j = int(tid); j < 96; j += 64) g_vec[(long)i * 96 + j] = gv_out[(long)i * 96 + j] + su * GB[V_VN + j] - k0 * X[V_VN + j];
    if (flags & F_TEN)
        for (int j = int(tid); j < 80; j += 64) g_ten[(long)i * 80 + j] = gt_out[(long)i * 80 + j] + stb * GB[V_TN + j] - k1 * X[V_TN + j];
}
"""
