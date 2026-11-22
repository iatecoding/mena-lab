"""Metal Shading Language source for O(N) open-boundary smeared Coulomb + Field + L=2 EFG forward & backward."""

METAL_COULOMB_EFG_MSL = r"""
constant constexpr float EFG_R2  = 0.70710678118654752440f;
constant constexpr float EFG_SR2 = 1.41421356237309504880f;
constant constexpr float EFG_R6  = 0.40824829046386301637f;

inline void smeared_k_f1_f2_f3(
    float r2, float inv_sig, float c0, float inv_sig2, int need_f2, int need_f3,
    thread float& k_out, thread float& f1_out, thread float& f2_out, thread float& f3_out
) {
    float x2 = r2 * inv_sig2;
    if (x2 < 2.5e-3f) {
        float x4 = x2 * x2;
        float c0_s2 = c0 * inv_sig2;
        k_out  = c0 * fma(x4, 0.1f, fma(x2, -0.333333343f, 1.0f));
        f1_out = c0_s2 * fma(x4, 0.142857149f, fma(x2, -0.4f, 0.666666687f));
        if (need_f2 != 0 || need_f3 != 0) {
            float c0_s4 = c0_s2 * inv_sig2;
            f2_out = c0_s4 * fma(x4, -0.222222224f, fma(x2, 0.571428597f, -0.8f));
            f3_out = (need_f3 != 0)
                ? (c0_s4 * inv_sig2) * fma(x4, 0.363636374f, fma(x2, -0.888888896f, 1.14285719f))
                : 0.0f;
        } else {
            f2_out = 0.0f;
            f3_out = 0.0f;
        }
    } else {
        float inv_r = rsqrt(r2);
        float inv_r2 = inv_r * inv_r;
        float r = r2 * inv_r;
        float ex = exp(-x2);
        k_out  = exact_erff(r * inv_sig) * inv_r;
        f1_out = (k_out - c0 * ex) * inv_r2;
        if (need_f2 != 0 || need_f3 != 0) {
            f2_out = (2.0f * c0 * inv_sig2 * ex - 3.0f * f1_out) * inv_r2;
            f3_out = (need_f3 != 0)
                ? (-4.0f * c0 * (inv_sig2 * inv_sig2) * ex - 5.0f * f2_out) * inv_r2
                : 0.0f;
        } else {
            f2_out = 0.0f;
            f3_out = 0.0f;
        }
    }
}

inline void decode_ge_sym(float g0, float g1, float g2, float g3, float g4, thread float4& g0_xyzw, thread float2& g1_xz_yz) {
    float t3 = EFG_R2 * g3;
    float t4 = EFG_R6 * g4;
    g0_xyzw = float4(t3 - t4, -t3 - t4, 2.0f * t4, EFG_R2 * g0);
    g1_xz_yz = float2(EFG_R2 * g1, EFG_R2 * g2);
}

inline void accum_efg(float3 d, float qf2, thread float& ae0, thread float& ae1, thread float& ae2, thread float& ae3, thread float& ae4) {
    ae0 = fma(qf2, EFG_SR2 * d.x * d.y, ae0);
    ae1 = fma(qf2, EFG_SR2 * d.x * d.z, ae1);
    ae2 = fma(qf2, EFG_SR2 * d.y * d.z, ae2);
    ae3 = fma(qf2, EFG_R2 * (d.x * d.x - d.y * d.y), ae3);
    ae4 = fma(qf2, EFG_R6 * (2.0f * d.z * d.z - d.x * d.x - d.y * d.y), ae4);
}

inline float3 sym_mat_vec(float4 g_0, float2 g_1, float3 d) {
    return float3(
        g_0.x * d.x + g_0.w * d.y + g_1.x * d.z,
        g_0.w * d.x + g_0.y * d.y + g_1.y * d.z,
        g_1.x * d.x + g_1.y * d.y + g_0.z * d.z
    );
}

kernel void coulomb_efg_fwd_grouped(
    device const float* pos [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* group_ptr [[buffer(4)]],
    device float* out_phi [[buffer(5)]],
    device float* out_fld [[buffer(6)]],
    device float* out_efg [[buffer(7)]],
    constant int& n_atoms [[buffer(8)]],
    constant float4& params [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms) return;
    if (mask[i] == 0) {
        out_phi[i] = 0.0f;
        out_fld[3 * i] = 0.0f; out_fld[3 * i + 1] = 0.0f; out_fld[3 * i + 2] = 0.0f;
        for (int m = 0; m < 5; ++m) out_efg[5 * i + m] = 0.0f;
        return;
    }
    int g = int(graph[i]);
    int j0 = group_ptr[g], j1 = group_ptr[g + 1];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;
    float ap = 0.0f, ae0 = 0.0f, ae1 = 0.0f, ae2 = 0.0f, ae3 = 0.0f, ae4 = 0.0f;
    float3 af = float3(0.0f);
    for (int j = j0; j < j1; ++j) {
        if (j == i || mask[j] == 0) continue;
        float qj = q[j];
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        float r2 = dot(d, d);
        float k, f1, f2;
        smeared_k_f1_f2(r2, inv_sig, c0, inv_sig2, k, f1, f2);
        ap = fma(qj, k, ap);
        af += (qj * f1) * d;
        accum_efg(d, qj * f2, ae0, ae1, ae2, ae3, ae4);
    }
    out_phi[i] = ap;
    out_fld[3 * i] = af.x; out_fld[3 * i + 1] = af.y; out_fld[3 * i + 2] = af.z;
    out_efg[5 * i] = ae0; out_efg[5 * i + 1] = ae1; out_efg[5 * i + 2] = ae2; out_efg[5 * i + 3] = ae3; out_efg[5 * i + 4] = ae4;
}

kernel void coulomb_efg_fwd_tiled(
    device const float* pos [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device float* out_phi [[buffer(3)]],
    device float* out_fld [[buffer(4)]],
    device float* out_efg [[buffer(5)]],
    constant int& n_atoms [[buffer(6)]],
    constant float4& params [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float4 tile_pq[256];
    threadgroup uchar tile_m[256];
    int i = int(tid);
    bool act_i = (i < n_atoms) && (mask[i] != 0);
    float3 pi = act_i ? float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) : float3(0.0f);
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;
    float ap = 0.0f, ae0 = 0.0f, ae1 = 0.0f, ae2 = 0.0f, ae3 = 0.0f, ae4 = 0.0f;
    float3 af = float3(0.0f);
    for (int base = 0; base < n_atoms; base += 256) {
        int jl = base + int(lid);
        if (jl < n_atoms && mask[jl] != 0) {
            tile_pq[lid] = float4(pos[3 * jl], pos[3 * jl + 1], pos[3 * jl + 2], q[jl]);
            tile_m[lid] = 1;
        } else {
            tile_pq[lid] = float4(0.0f);
            tile_m[lid] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act_i) {
            int lim = min(256, n_atoms - base);
            for (int t = 0; t < lim; ++t) {
                int j = base + t;
                if (j == i || tile_m[t] == 0) continue;
                float4 pq = tile_pq[t];
                float3 d = pi - pq.xyz;
                float k, f1, f2;
                smeared_k_f1_f2(dot(d, d), inv_sig, c0, inv_sig2, k, f1, f2);
                ap = fma(pq.w, k, ap);
                af += (pq.w * f1) * d;
                accum_efg(d, pq.w * f2, ae0, ae1, ae2, ae3, ae4);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < n_atoms) {
        out_phi[i] = act_i ? ap : 0.0f;
        out_fld[3 * i] = act_i ? af.x : 0.0f; out_fld[3 * i + 1] = act_i ? af.y : 0.0f; out_fld[3 * i + 2] = act_i ? af.z : 0.0f;
        out_efg[5 * i] = act_i ? ae0 : 0.0f; out_efg[5 * i + 1] = act_i ? ae1 : 0.0f; out_efg[5 * i + 2] = act_i ? ae2 : 0.0f;
        out_efg[5 * i + 3] = act_i ? ae3 : 0.0f; out_efg[5 * i + 4] = act_i ? ae4 : 0.0f;
    }
}

kernel void coulomb_efg_fwd_fast64(
    device const float4* pq_s [[buffer(0)]],
    device const uchar* mask_s [[buffer(1)]],
    device const float4* bsph [[buffer(2)]],
    device float* out_phi [[buffer(3)]],
    device float* out_fld [[buffer(4)]],
    device float* out_efg [[buffer(5)]],
    constant float4& params [[buffer(6)]],
    constant int& n_tiles [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float4 tile_pq[64];
    threadgroup uchar tile_m[64];
    int tg = int(tgid);
    int i = tg * 64 + int(lid);
    float3 pi = pq_s[i].xyz;
    uchar act_i = mask_s[i];
    float4 tgb = bsph[tg];
    float3 tc = tgb.xyz;
    float far_r = tgb.w + params.w;
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;
    float ap = 0.0f, ae0 = 0.0f, ae1 = 0.0f, ae2 = 0.0f, ae3 = 0.0f, ae4 = 0.0f;
    float3 af = float3(0.0f);
    for (int b = 0; b < n_tiles; ++b) {
        int jl = b * 64 + int(lid);
        tile_pq[lid] = pq_s[jl];
        tile_m[lid] = mask_s[jl];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act_i != 0) {
            float4 tb = bsph[b];
            float3 db = tc - tb.xyz;
            bool all_far = (dot(db, db) > (far_r + tb.w) * (far_r + tb.w)) && (tb.w >= 0.0f) && (tgb.w >= 0.0f);
            if (all_far) {
                for (int t = 0; t < 64; ++t) {
                    float4 pq = tile_pq[t];
                    if (pq.w == 0.0f) continue;
                    float3 d = pi - pq.xyz;
                    float ir = rsqrt(dot(d, d));
                    float ir2 = ir * ir;
                    float f1 = ir * ir2;
                    float f2 = -3.0f * f1 * ir2;
                    ap = fma(pq.w, ir, ap);
                    af += (pq.w * f1) * d;
                    accum_efg(d, pq.w * f2, ae0, ae1, ae2, ae3, ae4);
                }
            } else {
                bool diag = (b == tg);
                for (int t = 0; t < 64; ++t) {
                    if ((diag && t == int(lid)) || tile_m[t] == 0) continue;
                    float4 pq = tile_pq[t];
                    if (pq.w == 0.0f) continue;
                    float3 d = pi - pq.xyz;
                    float k, f1, f2;
                    smeared_k_f1_f2(dot(d, d), inv_sig, c0, inv_sig2, k, f1, f2);
                    ap = fma(pq.w, k, ap);
                    af += (pq.w * f1) * d;
                    accum_efg(d, pq.w * f2, ae0, ae1, ae2, ae3, ae4);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out_phi[i] = (act_i != 0) ? ap : 0.0f;
    out_fld[3 * i] = (act_i != 0) ? af.x : 0.0f; out_fld[3 * i + 1] = (act_i != 0) ? af.y : 0.0f; out_fld[3 * i + 2] = (act_i != 0) ? af.z : 0.0f;
    out_efg[5 * i] = (act_i != 0) ? ae0 : 0.0f; out_efg[5 * i + 1] = (act_i != 0) ? ae1 : 0.0f; out_efg[5 * i + 2] = (act_i != 0) ? ae2 : 0.0f;
    out_efg[5 * i + 3] = (act_i != 0) ? ae3 : 0.0f; out_efg[5 * i + 4] = (act_i != 0) ? ae4 : 0.0f;
}

kernel void coulomb_efg_bwd_grouped(
    device const float* pos [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* group_ptr [[buffer(4)]],
    device const float* g_phi [[buffer(5)]],
    device const float* g_fld [[buffer(6)]],
    device const float* g_efg [[buffer(7)]],
    device float* d_pos [[buffer(8)]],
    device float* d_q [[buffer(9)]],
    constant int& n_atoms [[buffer(10)]],
    constant int& has_gp [[buffer(11)]],
    constant int& has_gf [[buffer(12)]],
    constant int& has_ge [[buffer(13)]],
    constant int& need_dpos [[buffer(14)]],
    constant int& need_dq [[buffer(15)]],
    constant float4& params [[buffer(16)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms) return;
    if (mask[i] == 0) {
        if (need_dq != 0) d_q[i] = 0.0f;
        if (need_dpos != 0) { d_pos[3 * i] = 0.0f; d_pos[3 * i + 1] = 0.0f; d_pos[3 * i + 2] = 0.0f; }
        return;
    }
    int g = int(graph[i]);
    int j0 = group_ptr[g], j1 = group_ptr[g + 1];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float qi = q[i];
    float gpi = (has_gp != 0) ? g_phi[i] : 0.0f;
    float3 gfi = (has_gf != 0) ? float3(g_fld[3 * i], g_fld[3 * i + 1], g_fld[3 * i + 2]) : float3(0.0f);
    float4 gi0 = float4(0.0f); float2 gi1 = float2(0.0f);
    if (has_ge != 0) decode_ge_sym(g_efg[5 * i], g_efg[5 * i + 1], g_efg[5 * i + 2], g_efg[5 * i + 3], g_efg[5 * i + 4], gi0, gi1);
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;
    float dq = 0.0f;
    float3 dp = float3(0.0f);
    for (int j = j0; j < j1; ++j) {
        if (j == i || mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        float k, f1, f2, f3;
        smeared_k_f1_f2_f3(dot(d, d), inv_sig, c0, inv_sig2, (has_gf != 0 || has_ge != 0) ? 1 : 0, (has_ge != 0 && need_dpos != 0) ? 1 : 0, k, f1, f2, f3);
        float gpj = (has_gp != 0) ? g_phi[j] : 0.0f;
        float3 gfj = (has_gf != 0) ? float3(g_fld[3 * j], g_fld[3 * j + 1], g_fld[3 * j + 2]) : float3(0.0f);
        float4 gj0 = float4(0.0f); float2 gj1 = float2(0.0f);
        if (has_ge != 0) decode_ge_sym(g_efg[5 * j], g_efg[5 * j + 1], g_efg[5 * j + 2], g_efg[5 * j + 3], g_efg[5 * j + 4], gj0, gj1);
        float3 mdj = (has_ge != 0) ? sym_mat_vec(gj0, gj1, d) : float3(0.0f);
        float sj = (has_ge != 0) ? dot(mdj, d) : 0.0f;
        if (need_dq != 0) {
            if (has_gp != 0) dq = fma(gpj, k, dq);
            if (has_gf != 0) dq -= f1 * dot(gfj, d);
            if (has_ge != 0) dq = fma(f2, sj, dq);
        }
        if (need_dpos != 0) {
            float qj = q[j];
            float cd = (has_gp != 0) ? -(gpi * qj + gpj * qi) * f1 : 0.0f;
            float3 v = (has_gf != 0) ? (qj * gfi - qi * gfj) : float3(0.0f);
            if (has_gf != 0) cd = fma(f2, dot(v, d), cd);
            float3 g_vec = float3(0.0f);
            if (has_ge != 0) {
                float3 mdi = sym_mat_vec(gi0, gi1, d);
                float si = dot(mdi, d);
                cd = fma(f3, qj * si + qi * sj, cd);
                g_vec = (2.0f * f2) * (qj * mdi + qi * mdj);
            }
            dp += cd * d + f1 * v + g_vec;
        }
    }
    if (need_dq != 0) d_q[i] = dq;
    if (need_dpos != 0) { d_pos[3 * i] = dp.x; d_pos[3 * i + 1] = dp.y; d_pos[3 * i + 2] = dp.z; }
}

kernel void coulomb_efg_bwd_tiled(
    device const float* pos [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const float* g_phi [[buffer(3)]],
    device const float* g_fld [[buffer(4)]],
    device const float* g_efg [[buffer(5)]],
    device float* d_pos [[buffer(6)]],
    device float* d_q [[buffer(7)]],
    constant int& n_atoms [[buffer(8)]],
    constant int& has_gp [[buffer(9)]],
    constant int& has_gf [[buffer(10)]],
    constant int& has_ge [[buffer(11)]],
    constant int& need_dpos [[buffer(12)]],
    constant int& need_dq [[buffer(13)]],
    constant float4& params [[buffer(14)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float4 tile_pq[128];
    threadgroup float4 tile_gf[128];
    threadgroup float4 tile_g0[128];
    threadgroup float2 tile_g1[128];
    threadgroup uchar tile_m[128];
    int i = int(tid);
    bool act_i = (i < n_atoms) && (mask[i] != 0);
    float3 pi = act_i ? float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) : float3(0.0f);
    float qi = act_i ? q[i] : 0.0f;
    float gpi = (act_i && has_gp != 0) ? g_phi[i] : 0.0f;
    float3 gfi = (act_i && has_gf != 0) ? float3(g_fld[3 * i], g_fld[3 * i + 1], g_fld[3 * i + 2]) : float3(0.0f);
    float4 gi0 = float4(0.0f); float2 gi1 = float2(0.0f);
    if (act_i && has_ge != 0) decode_ge_sym(g_efg[5 * i], g_efg[5 * i + 1], g_efg[5 * i + 2], g_efg[5 * i + 3], g_efg[5 * i + 4], gi0, gi1);
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;
    float dq = 0.0f; float3 dp = float3(0.0f);
    for (int base = 0; base < n_atoms; base += 128) {
        int jl = base + int(lid);
        if (jl < n_atoms && mask[jl] != 0) {
            tile_pq[lid] = float4(pos[3 * jl], pos[3 * jl + 1], pos[3 * jl + 2], q[jl]);
            tile_gf[lid] = float4((has_gf != 0) ? float3(g_fld[3 * jl], g_fld[3 * jl + 1], g_fld[3 * jl + 2]) : float3(0.0f), (has_gp != 0) ? g_phi[jl] : 0.0f);
            float4 g0t = float4(0.0f); float2 g1t = float2(0.0f);
            if (has_ge != 0) decode_ge_sym(g_efg[5 * jl], g_efg[5 * jl + 1], g_efg[5 * jl + 2], g_efg[5 * jl + 3], g_efg[5 * jl + 4], g0t, g1t);
            tile_g0[lid] = g0t; tile_g1[lid] = g1t; tile_m[lid] = 1;
        } else {
            tile_pq[lid] = float4(0.0f); tile_gf[lid] = float4(0.0f); tile_g0[lid] = float4(0.0f); tile_g1[lid] = float2(0.0f); tile_m[lid] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act_i) {
            int lim = min(128, n_atoms - base);
            for (int t = 0; t < lim; ++t) {
                int j = base + t;
                if (j == i || tile_m[t] == 0) continue;
                float4 pq = tile_pq[t]; float4 gf = tile_gf[t];
                float3 d = pi - pq.xyz;
                float k, f1, f2, f3;
                smeared_k_f1_f2_f3(dot(d, d), inv_sig, c0, inv_sig2, (has_gf != 0 || has_ge != 0) ? 1 : 0, (has_ge != 0 && need_dpos != 0) ? 1 : 0, k, f1, f2, f3);
                float3 mdj = (has_ge != 0) ? sym_mat_vec(tile_g0[t], tile_g1[t], d) : float3(0.0f);
                float sj = (has_ge != 0) ? dot(mdj, d) : 0.0f;
                if (need_dq != 0) {
                    if (has_gp != 0) dq = fma(gf.w, k, dq);
                    if (has_gf != 0) dq -= f1 * dot(gf.xyz, d);
                    if (has_ge != 0) dq = fma(f2, sj, dq);
                }
                if (need_dpos != 0) {
                    float cd = (has_gp != 0) ? -(gpi * pq.w + gf.w * qi) * f1 : 0.0f;
                    float3 v = (has_gf != 0) ? (pq.w * gfi - qi * gf.xyz) : float3(0.0f);
                    if (has_gf != 0) cd = fma(f2, dot(v, d), cd);
                    float3 g_vec = float3(0.0f);
                    if (has_ge != 0) {
                        float3 mdi = sym_mat_vec(gi0, gi1, d);
                        float si = dot(mdi, d);
                        cd = fma(f3, pq.w * si + qi * sj, cd);
                        g_vec = (2.0f * f2) * (pq.w * mdi + qi * mdj);
                    }
                    dp += cd * d + f1 * v + g_vec;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < n_atoms) {
        if (need_dq != 0) d_q[i] = act_i ? dq : 0.0f;
        if (need_dpos != 0) {
            d_pos[3 * i] = act_i ? dp.x : 0.0f; d_pos[3 * i + 1] = act_i ? dp.y : 0.0f; d_pos[3 * i + 2] = act_i ? dp.z : 0.0f;
        }
    }
}

kernel void coulomb_efg_bwd_fast64(
    device const float4* pq_s [[buffer(0)]],
    device const uchar* mask_s [[buffer(1)]],
    device const float4* bsph [[buffer(2)]],
    device const float4* gf_s [[buffer(3)]],
    device const float4* g0_s [[buffer(4)]],
    device const float2* g1_s [[buffer(5)]],
    device float* d_pos [[buffer(6)]],
    device float* d_q [[buffer(7)]],
    constant float4& params [[buffer(8)]],
    constant int& n_tiles [[buffer(9)]],
    constant int& has_gp [[buffer(10)]],
    constant int& has_gf [[buffer(11)]],
    constant int& has_ge [[buffer(12)]],
    constant int& need_dpos [[buffer(13)]],
    constant int& need_dq [[buffer(14)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float4 tile_pq[64];
    threadgroup float4 tile_gf[64];
    threadgroup float4 tile_g0[64];
    threadgroup float2 tile_g1[64];
    threadgroup uchar tile_m[64];
    int tg = int(tgid);
    int i = tg * 64 + int(lid);
    float4 pqi = pq_s[i];
    float4 gfi_w = gf_s[i];
    float4 gi0 = g0_s[i];
    float2 gi1 = g1_s[i];
    uchar act_i = mask_s[i];
    float4 tgb = bsph[tg];
    float3 tc = tgb.xyz;
    float far_r = tgb.w + params.w;
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;
    float3 dp = float3(0.0f); float dq = 0.0f;
    for (int b = 0; b < n_tiles; ++b) {
        int j = b * 64 + int(lid);
        tile_pq[lid] = pq_s[j];
        tile_gf[lid] = gf_s[j];
        tile_g0[lid] = g0_s[j];
        tile_g1[lid] = g1_s[j];
        tile_m[lid] = mask_s[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act_i != 0) {
            float4 tb = bsph[b];
            float3 db = tc - tb.xyz;
            bool all_far = (dot(db, db) > (far_r + tb.w) * (far_r + tb.w)) && (tb.w >= 0.0f) && (tgb.w >= 0.0f);
            if (all_far) {
                for (int t = 0; t < 64; ++t) {
                    if (tile_m[t] == 0) continue;
                    float4 pq = tile_pq[t]; float4 gf = tile_gf[t];
                    float3 d = pqi.xyz - pq.xyz;
                    float ir = rsqrt(dot(d, d));
                    float ir2 = ir * ir;
                    float f1 = ir * ir2;
                    float f2 = -3.0f * f1 * ir2;
                    float f3 = -5.0f * f2 * ir2;
                    float3 mdj = (has_ge != 0) ? sym_mat_vec(tile_g0[t], tile_g1[t], d) : float3(0.0f);
                    float sj = (has_ge != 0) ? dot(mdj, d) : 0.0f;
                    if (need_dq != 0) {
                        if (has_gp != 0) dq = fma(gf.w, ir, dq);
                        if (has_gf != 0) dq -= f1 * dot(gf.xyz, d);
                        if (has_ge != 0) dq = fma(f2, sj, dq);
                    }
                    if (need_dpos != 0) {
                        float cd = (has_gp != 0) ? -(gfi_w.w * pq.w + gf.w * pqi.w) * f1 : 0.0f;
                        float3 v = (has_gf != 0) ? (pq.w * gfi_w.xyz - pqi.w * gf.xyz) : float3(0.0f);
                        if (has_gf != 0) cd = fma(f2, dot(v, d), cd);
                        float3 g_vec = float3(0.0f);
                        if (has_ge != 0) {
                            float3 mdi = sym_mat_vec(gi0, gi1, d);
                            float si = dot(mdi, d);
                            cd = fma(f3, pq.w * si + pqi.w * sj, cd);
                            g_vec = (2.0f * f2) * (pq.w * mdi + pqi.w * mdj);
                        }
                        dp += cd * d + f1 * v + g_vec;
                    }
                }
            } else {
                bool diag = (b == tg);
                for (int t = 0; t < 64; ++t) {
                    if ((diag && t == int(lid)) || tile_m[t] == 0) continue;
                    float4 pq = tile_pq[t]; float4 gf = tile_gf[t];
                    float3 d = pqi.xyz - pq.xyz;
                    float k, f1, f2, f3;
                    smeared_k_f1_f2_f3(dot(d, d), inv_sig, c0, inv_sig2, (has_gf != 0 || has_ge != 0) ? 1 : 0, (has_ge != 0 && need_dpos != 0) ? 1 : 0, k, f1, f2, f3);
                    float3 mdj = (has_ge != 0) ? sym_mat_vec(tile_g0[t], tile_g1[t], d) : float3(0.0f);
                    float sj = (has_ge != 0) ? dot(mdj, d) : 0.0f;
                    if (need_dq != 0) {
                        if (has_gp != 0) dq = fma(gf.w, k, dq);
                        if (has_gf != 0) dq -= f1 * dot(gf.xyz, d);
                        if (has_ge != 0) dq = fma(f2, sj, dq);
                    }
                    if (need_dpos != 0) {
                        float cd = (has_gp != 0) ? -(gfi_w.w * pq.w + gf.w * pqi.w) * f1 : 0.0f;
                        float3 v = (has_gf != 0) ? (pq.w * gfi_w.xyz - pqi.w * gf.xyz) : float3(0.0f);
                        if (has_gf != 0) cd = fma(f2, dot(v, d), cd);
                        float3 g_vec = float3(0.0f);
                        if (has_ge != 0) {
                            float3 mdi = sym_mat_vec(gi0, gi1, d);
                            float si = dot(mdi, d);
                            cd = fma(f3, pq.w * si + pqi.w * sj, cd);
                            g_vec = (2.0f * f2) * (pq.w * mdi + pqi.w * mdj);
                        }
                        dp += cd * d + f1 * v + g_vec;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (need_dq != 0) d_q[i] = (act_i != 0) ? dq : 0.0f;
    if (need_dpos != 0) {
        d_pos[3 * i] = (act_i != 0) ? dp.x : 0.0f;
        d_pos[3 * i + 1] = (act_i != 0) ? dp.y : 0.0f;
        d_pos[3 * i + 2] = (act_i != 0) ? dp.z : 0.0f;
    }
}
"""
