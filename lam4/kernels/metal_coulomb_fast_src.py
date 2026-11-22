"""Morton-sorted 64-atom tiled exact Coulomb forward/backward Metal kernels (concatenated onto METAL_COULOMB_MSL)."""

METAL_COULOMB_FAST_MSL = r"""
// Fast exact Morton-sorted 64-atom tiled forward/backward for large single open molecules (G == 1, N >= 1024):
// bounding-sphere separation > r_tg + 5.5*sigma + r_tile collapses erf(r/sigma)/r to pure rsqrt(r^2) (< 1e-14 error).
kernel void coulomb_fast64_fwd(
    device const float4* pq_s [[buffer(0)]],
    device const uchar* mask_s [[buffer(1)]],
    device const float4* bsph [[buffer(2)]],
    device float* out_phi [[buffer(3)]],
    device float* out_fld [[buffer(4)]],
    constant float4& params [[buffer(5)]],
    constant int& n_tiles [[buffer(6)]],
    constant int& need_fld [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float4 tile_pq[64];
    int tg = int(tgid);
    int i = tg * 64 + int(lid);
    float3 pi = pq_s[i].xyz;
    uchar act_i = mask_s[i];
    float4 tgb = bsph[tg];
    float3 tc = tgb.xyz;
    float far_r = tgb.w + params.w;
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;
    float ap = 0.0f;
    float3 af = 0.0f;
    for (int b = 0; b < n_tiles; ++b) {
        tile_pq[lid] = pq_s[b * 64 + int(lid)];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act_i != 0) {
            float4 tb = bsph[b];
            float3 db = tc - tb.xyz;
            bool all_far = (dot(db, db) > (far_r + tb.w) * (far_r + tb.w)) && (tb.w >= 0.0f) && (tgb.w >= 0.0f);
            if (all_far) {
                for (int t = 0; t < 64; ++t) {
                    float4 pq = tile_pq[t];
                    float3 d = pi - pq.xyz;
                    float ir = rsqrt(dot(d, d));
                    ap = fma(pq.w, ir, ap);
                    if (need_fld != 0) af += (pq.w * ir * ir * ir) * d;
                }
            } else {
                bool diag = (b == tg);
                for (int t = 0; t < 64; ++t) {
                    if (diag && t == int(lid)) continue;
                    float4 pq = tile_pq[t];
                    if (pq.w == 0.0f) continue;
                    float3 d = pi - pq.xyz;
                    float r2 = max(dot(d, d), 1e-16f);
                    float k, f1;
                    smeared_k_f1(r2, inv_sig, c0, inv_sig2, k, f1);
                    ap = fma(pq.w, k, ap);
                    if (need_fld != 0) af += (pq.w * f1) * d;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out_phi[i] = (act_i != 0) ? ap : 0.0f;
    if (need_fld != 0) {
        out_fld[3 * i]     = (act_i != 0) ? af.x : 0.0f;
        out_fld[3 * i + 1] = (act_i != 0) ? af.y : 0.0f;
        out_fld[3 * i + 2] = (act_i != 0) ? af.z : 0.0f;
    }
}

kernel void coulomb_fast64_bwd(
    device const float4* pq_s [[buffer(0)]],
    device const float4* g_s [[buffer(1)]],
    device const uchar* mask_s [[buffer(2)]],
    device const float4* bsph [[buffer(3)]],
    device float* d_pos [[buffer(4)]],
    device float* d_q [[buffer(5)]],
    constant float4& params [[buffer(6)]],
    constant int& n_tiles [[buffer(7)]],
    constant int& need_dpos [[buffer(8)]],
    constant int& need_dq [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float4 tile_pq[64];
    threadgroup float4 tile_g[64];
    int tg = int(tgid);
    int i = tg * 64 + int(lid);
    float4 pqi = pq_s[i];
    float4 gi = g_s[i];
    uchar act_i = mask_s[i];
    float4 tgb = bsph[tg];
    float3 tc = tgb.xyz;
    float far_r = tgb.w + params.w;
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;
    float3 dp = 0.0f;
    float dq = 0.0f;
    for (int b = 0; b < n_tiles; ++b) {
        int j = b * 64 + int(lid);
        tile_pq[lid] = pq_s[j];
        tile_g[lid]  = g_s[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act_i != 0) {
            float4 tb = bsph[b];
            float3 db = tc - tb.xyz;
            bool all_far = (dot(db, db) > (far_r + tb.w) * (far_r + tb.w)) && (tb.w >= 0.0f) && (tgb.w >= 0.0f);
            if (all_far) {
                for (int t = 0; t < 64; ++t) {
                    float4 pq = tile_pq[t];
                    float4 gg = tile_g[t];
                    float3 d = pqi.xyz - pq.xyz;
                    float ir = rsqrt(dot(d, d));
                    float ir2 = ir * ir;
                    float f1 = ir * ir2;
                    if (need_dq != 0) dq += gg.w * ir - f1 * dot(gg.xyz, d);
                    if (need_dpos != 0) {
                        float f2 = -3.0f * f1 * ir2;
                        float s_phi = gi.w * pq.w + gg.w * pqi.w;
                        float3 v = pq.w * gi.xyz - pqi.w * gg.xyz;
                        dp += (-s_phi * f1 + f2 * dot(v, d)) * d + f1 * v;
                    }
                }
            } else {
                bool diag = (b == tg);
                for (int t = 0; t < 64; ++t) {
                    if (diag && t == int(lid)) continue;
                    float4 pq = tile_pq[t];
                    float4 gg = tile_g[t];
                    float3 d = pqi.xyz - pq.xyz;
                    float r2 = max(dot(d, d), 1e-16f);
                    float k, f1, f2;
                    smeared_k_f1_f2(r2, inv_sig, c0, inv_sig2, k, f1, f2);
                    if (need_dq != 0) dq += gg.w * k - f1 * dot(gg.xyz, d);
                    if (need_dpos != 0) {
                        float s_phi = gi.w * pq.w + gg.w * pqi.w;
                        float3 v = pq.w * gi.xyz - pqi.w * gg.xyz;
                        dp += (-s_phi * f1 + f2 * dot(v, d)) * d + f1 * v;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (need_dq != 0) d_q[i] = (act_i != 0) ? dq : 0.0f;
    if (need_dpos != 0) {
        d_pos[3 * i]     = (act_i != 0) ? dp.x : 0.0f;
        d_pos[3 * i + 1] = (act_i != 0) ? dp.y : 0.0f;
        d_pos[3 * i + 2] = (act_i != 0) ? dp.z : 0.0f;
    }
}
"""
