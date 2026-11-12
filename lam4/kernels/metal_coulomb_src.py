"""Metal Shading Language source for Tier-2 open-boundary smeared ExactCoulomb."""

METAL_COULOMB_MSL = r"""
#include <metal_stdlib>
using namespace metal;

// 1-ULP IEEE-754 single-precision erf(a) (verified max error <= 1.2e-7 across [-6, 6])
inline float exact_erff(float a) {
    float x = fabs(a);
    float r;
    if (x < 1.25f) {
        float u = x * x;
        float p = 6.245970226e-05f;
        p = fma(p, u, -7.440516159e-04f);
        p = fma(p, u, 5.111238764e-03f);
        p = fma(p, u, -2.680452797e-02f);
        p = fma(p, u, 1.128210157e-01f);
        p = fma(p, u, -3.761244680e-01f);
        p = fma(p, u, 1.283791142e-01f);
        r = fma(p, x, x);
    } else if (x < 3.95f) {
        float t = 1.0f / fma(0.5f, x, 1.0f);
        float p = 3.475873057e-02f;
        p = fma(p, t, -1.453217770e-02f);
        p = fma(p, t, -2.472636555e-01f);
        p = fma(p, t, 3.508501074e-01f);
        p = fma(p, t, 2.135339677e-02f);
        p = fma(p, t, 3.011171738e-01f);
        p = fma(p, t, 2.705037481e-01f);
        p = fma(p, t, 2.834916809e-01f);
        p = fma(p, t, -7.315009595e-05f);
        r = 1.0f - p * exp(-x * x);
    } else {
        r = 1.0f;
    }
    return copysign(r, a);
}

inline void smeared_k_f1(float r2, float inv_sig, float c0, float inv_sig2, thread float& k_out, thread float& f1_out) {
    float x2 = r2 * (inv_sig * inv_sig);
    if (x2 < 2.5e-3f) {
        float x4 = x2 * x2;
        k_out  = c0 * fma(x4, 0.1f, fma(x2, -0.333333343f, 1.0f));
        f1_out = (c0 * inv_sig2) * fma(x4, 0.142857149f, fma(x2, -0.4f, 0.666666687f));
    } else {
        float inv_r = rsqrt(r2);
        float r = r2 * inv_r;
        float inv_r2 = inv_r * inv_r;
        float ex = exp(-x2);
        k_out  = exact_erff(r * inv_sig) * inv_r;
        f1_out = (k_out - c0 * ex) * inv_r2;
    }
}

inline void smeared_k_f1_f2(float r2, float inv_sig, float c0, float inv_sig2,
                            thread float& k_out, thread float& f1_out, thread float& f2_out) {
    float x2 = r2 * (inv_sig * inv_sig);
    if (x2 < 2.5e-3f) {
        float x4 = x2 * x2;
        float c0_s2 = c0 * inv_sig2;
        k_out  = c0 * fma(x4, 0.1f, fma(x2, -0.333333343f, 1.0f));
        f1_out = c0_s2 * fma(x4, 0.142857149f, fma(x2, -0.4f, 0.666666687f));
        f2_out = (c0_s2 * inv_sig2) * fma(x4, -0.222222224f, fma(x2, 0.571428597f, -0.8f));
    } else {
        float inv_r = rsqrt(r2);
        float r = r2 * inv_r;
        float inv_r2 = inv_r * inv_r;
        float ex = exp(-x2);
        k_out  = exact_erff(r * inv_sig) * inv_r;
        f1_out = (k_out - c0 * ex) * inv_r2;
        f2_out = (2.0f * c0 * inv_sig2 * ex - 3.0f * f1_out) * inv_r2;
    }
}

// General batched forward (G >= 1): phi [N] and optionally field [N, 3]
kernel void coulomb_fwd_grouped(
    device const float* pos [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* group_ptr [[buffer(4)]],
    device float* out_phi [[buffer(5)]],
    device float* out_fld [[buffer(6)]],
    constant int& n_atoms [[buffer(7)]],
    constant int& need_fld [[buffer(8)]],
    constant float4& params [[buffer(9)]], // (inv_sig, c0, inv_sig2, 0)
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms) return;
    if (mask[i] == 0) {
        out_phi[i] = 0.0f;
        if (need_fld != 0) {
            out_fld[3 * i] = 0.0f; out_fld[3 * i + 1] = 0.0f; out_fld[3 * i + 2] = 0.0f;
        }
        return;
    }
    int g = int(graph[i]);
    int j0 = group_ptr[g], j1 = group_ptr[g + 1];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;
    float acc_phi = 0.0f;
    float3 acc_fld = float3(0.0f);

    for (int j = j0; j < j1; ++j) {
        if (j == i || mask[j] == 0) continue;
        float qj = q[j];
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        float r2 = max(dot(d, d), 1e-16f);
        float k, f1;
        smeared_k_f1(r2, inv_sig, c0, inv_sig2, k, f1);
        acc_phi = fma(qj, k, acc_phi);
        if (need_fld != 0) {
            acc_fld += (qj * f1) * d;
        }
    }
    out_phi[i] = acc_phi;
    if (need_fld != 0) {
        out_fld[3 * i]     = acc_fld.x;
        out_fld[3 * i + 1] = acc_fld.y;
        out_fld[3 * i + 2] = acc_fld.z;
    }
}

// Shared-memory tiled forward for single large molecule (G == 1)
kernel void coulomb_fwd_tiled(
    device const float* pos [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device float* out_phi [[buffer(3)]],
    device float* out_fld [[buffer(4)]],
    constant int& n_atoms [[buffer(5)]],
    constant int& need_fld [[buffer(6)]],
    constant float4& params [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float4 tile_pq[256];
    threadgroup uchar  tile_m[256];

    int i = int(tid);
    bool active_i = (i < n_atoms) && (mask[i] != 0);
    float3 pi = active_i ? float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) : float3(0.0f);
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;
    float acc_phi = 0.0f;
    float3 acc_fld = float3(0.0f);

    for (int base = 0; base < n_atoms; base += 256) {
        int j_load = base + int(lid);
        if (j_load < n_atoms && mask[j_load] != 0) {
            tile_pq[lid] = float4(pos[3 * j_load], pos[3 * j_load + 1], pos[3 * j_load + 2], q[j_load]);
            tile_m[lid] = 1;
        } else {
            tile_pq[lid] = float4(0.0f);
            tile_m[lid] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_i) {
            int lim = min(256, n_atoms - base);
            for (int t = 0; t < lim; ++t) {
                int j = base + t;
                if (j == i || tile_m[t] == 0) continue;
                float4 pq = tile_pq[t];
                float3 d = pi - pq.xyz;
                float r2 = max(dot(d, d), 1e-16f);
                float k, f1;
                smeared_k_f1(r2, inv_sig, c0, inv_sig2, k, f1);
                acc_phi = fma(pq.w, k, acc_phi);
                if (need_fld != 0) {
                    acc_fld += (pq.w * f1) * d;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < n_atoms) {
        out_phi[i] = active_i ? acc_phi : 0.0f;
        if (need_fld != 0) {
            out_fld[3 * i]     = active_i ? acc_fld.x : 0.0f;
            out_fld[3 * i + 1] = active_i ? acc_fld.y : 0.0f;
            out_fld[3 * i + 2] = active_i ? acc_fld.z : 0.0f;
        }
    }
}

// General batched backward (G >= 1): d_pos [N, 3] and d_q [N]
kernel void coulomb_bwd_grouped(
    device const float* pos [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* group_ptr [[buffer(4)]],
    device const float* g_phi [[buffer(5)]],
    device const float* g_fld [[buffer(6)]],
    device float* d_pos [[buffer(7)]],
    device float* d_q [[buffer(8)]],
    constant int& n_atoms [[buffer(9)]],
    constant int& has_gfld [[buffer(10)]],
    constant int& need_dpos [[buffer(11)]],
    constant int& need_dq [[buffer(12)]],
    constant float4& params [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms) return;
    if (mask[i] == 0) {
        if (need_dpos != 0) {
            d_pos[3 * i] = 0.0f; d_pos[3 * i + 1] = 0.0f; d_pos[3 * i + 2] = 0.0f;
        }
        if (need_dq != 0) d_q[i] = 0.0f;
        return;
    }
    int g = int(graph[i]);
    int j0 = group_ptr[g], j1 = group_ptr[g + 1];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float qi = q[i];
    float gpi = g_phi[i];
    float3 gei = (has_gfld != 0) ? float3(g_fld[3 * i], g_fld[3 * i + 1], g_fld[3 * i + 2]) : float3(0.0f);
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;

    float3 acc_dp = float3(0.0f);
    float acc_dq = 0.0f;

    for (int j = j0; j < j1; ++j) {
        if (j == i || mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        float r2 = max(dot(d, d), 1e-16f);
        float k, f1, f2;
        smeared_k_f1_f2(r2, inv_sig, c0, inv_sig2, k, f1, f2);
        float qj = q[j];
        float gpj = g_phi[j];
        float3 gej = (has_gfld != 0) ? float3(g_fld[3 * j], g_fld[3 * j + 1], g_fld[3 * j + 2]) : float3(0.0f);

        if (need_dq != 0) {
            acc_dq += gpj * k - f1 * dot(gej, d);
        }
        if (need_dpos != 0) {
            float s_phi = gpi * qj + gpj * qi;
            if (has_gfld != 0) {
                float3 v = qj * gei - qi * gej;
                acc_dp += (-s_phi * f1 + f2 * dot(v, d)) * d + f1 * v;
            } else {
                acc_dp += (-s_phi * f1) * d;
            }
        }
    }
    if (need_dpos != 0) {
        d_pos[3 * i]     = acc_dp.x;
        d_pos[3 * i + 1] = acc_dp.y;
        d_pos[3 * i + 2] = acc_dp.z;
    }
    if (need_dq != 0) {
        d_q[i] = acc_dq;
    }
}

// Shared-memory tiled backward for single large molecule (G == 1)
kernel void coulomb_bwd_tiled(
    device const float* pos [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const float* g_phi [[buffer(3)]],
    device const float* g_fld [[buffer(4)]],
    device float* d_pos [[buffer(5)]],
    device float* d_q [[buffer(6)]],
    constant int& n_atoms [[buffer(7)]],
    constant int& has_gfld [[buffer(8)]],
    constant int& need_dpos [[buffer(9)]],
    constant int& need_dq [[buffer(10)]],
    constant float4& params [[buffer(11)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float4 tile_pq[256];
    threadgroup float4 tile_g[256];   // (ge.x, ge.y, ge.z, g_phi)
    threadgroup uchar  tile_m[256];

    int i = int(tid);
    bool active_i = (i < n_atoms) && (mask[i] != 0);
    float3 pi = active_i ? float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) : float3(0.0f);
    float qi = active_i ? q[i] : 0.0f;
    float gpi = active_i ? g_phi[i] : 0.0f;
    float3 gei = (active_i && has_gfld != 0) ? float3(g_fld[3 * i], g_fld[3 * i + 1], g_fld[3 * i + 2]) : float3(0.0f);
    float inv_sig = params.x, c0 = params.y, inv_sig2 = params.z;

    float3 acc_dp = float3(0.0f);
    float acc_dq = 0.0f;

    for (int base = 0; base < n_atoms; base += 256) {
        int j_load = base + int(lid);
        if (j_load < n_atoms && mask[j_load] != 0) {
            tile_pq[lid] = float4(pos[3 * j_load], pos[3 * j_load + 1], pos[3 * j_load + 2], q[j_load]);
            float3 ge_l = (has_gfld != 0) ? float3(g_fld[3 * j_load], g_fld[3 * j_load + 1], g_fld[3 * j_load + 2]) : float3(0.0f);
            tile_g[lid]  = float4(ge_l, g_phi[j_load]);
            tile_m[lid]  = 1;
        } else {
            tile_pq[lid] = float4(0.0f);
            tile_g[lid]  = float4(0.0f);
            tile_m[lid]  = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_i) {
            int lim = min(256, n_atoms - base);
            for (int t = 0; t < lim; ++t) {
                int j = base + t;
                if (j == i || tile_m[t] == 0) continue;
                float4 pq = tile_pq[t];
                float4 gg = tile_g[t];
                float3 d = pi - pq.xyz;
                float r2 = max(dot(d, d), 1e-16f);
                float k, f1, f2;
                smeared_k_f1_f2(r2, inv_sig, c0, inv_sig2, k, f1, f2);
                if (need_dq != 0) {
                    acc_dq += gg.w * k - f1 * dot(gg.xyz, d);
                }
                if (need_dpos != 0) {
                    float s_phi = gpi * pq.w + gg.w * qi;
                    if (has_gfld != 0) {
                        float3 v = pq.w * gei - qi * gg.xyz;
                        acc_dp += (-s_phi * f1 + f2 * dot(v, d)) * d + f1 * v;
                    } else {
                        acc_dp += (-s_phi * f1) * d;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < n_atoms) {
        if (need_dpos != 0) {
            d_pos[3 * i]     = active_i ? acc_dp.x : 0.0f;
            d_pos[3 * i + 1] = active_i ? acc_dp.y : 0.0f;
            d_pos[3 * i + 2] = active_i ? acc_dp.z : 0.0f;
        }
        if (need_dq != 0) {
            d_q[i] = active_i ? acc_dq : 0.0f;
        }
    }
}
"""
