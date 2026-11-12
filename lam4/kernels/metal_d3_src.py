"""Metal Shading Language source for Tier-2 Grimme D3(BJ) dispersion kernels."""

METAL_D3_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant float ANG_TO_BOHR     = 1.8897261245650618f;
constant float HARTREE_KCAL    = 627.509474f;
constant float D3_K1           = 16.0f;
constant float D3_K3           = -4.0f;

// Shared pair contribution helper for both grouped and CSR energy + direct-force kernels
inline void d3_accum_pair(
    float3 d, int zi, int zj, float cn_i, float cn_j, float r2r4_i,
    device const float* r2r4, device const float* c6ab_flat,
    float4 params0, float4 params1, int need_forces,
    thread float& acc_e, thread float& acc_dedcn, thread float3& acc_f
) {
    float r_cut = params1.x, taper_on = params1.y, inv_tspan = params1.w;
    float d2 = dot(d, d);
    if (d2 > r_cut * r_cut) return;
    float r2_eps = d2 + 1e-8f;
    float inv_r_ang = rsqrt(r2_eps);
    float r_ang = r2_eps * inv_r_ang;
    if (r_ang > r_cut) return;

    device const float* refs = c6ab_flat + 75 * (zi * 95 + zj);
    float s_max = -1.0e20f;
    for (int k = 0; k < 25; ++k) {
        float c0 = refs[3 * k];
        if (c0 > 0.0f) {
            float di = refs[3 * k + 1] - cn_i;
            float dj = refs[3 * k + 2] - cn_j;
            s_max = max(s_max, D3_K3 * (di * di + dj * dj));
        }
    }
    if (s_max <= -1.0e19f) return;

    float sum_w = 0.0f, sum_wc0 = 0.0f, sum_wc0_d = 0.0f, sum_wd = 0.0f;
    for (int k = 0; k < 25; ++k) {
        float c0 = refs[3 * k];
        if (c0 > 0.0f) {
            float di = refs[3 * k + 1] - cn_i;
            float dj = refs[3 * k + 2] - cn_j;
            float w = exp(D3_K3 * (di * di + dj * dj) - s_max);
            sum_w += w;
            sum_wc0 = fma(w, c0, sum_wc0);
            if (need_forces != 0) {
                float wd = w * di;
                sum_wd += wd;
                sum_wc0_d = fma(wd, c0, sum_wc0_d);
            }
        }
    }
    float inv_sw = 1.0f / max(sum_w, 1e-30f);
    float c6 = sum_wc0 * inv_sw;
    float q_ij = 3.0f * r2r4_i * r2r4[zj];
    float s6 = params0.x, s8 = params0.y, a1 = params0.z, a2 = params0.w;
    float tmp = a1 * sqrt(max(q_ij, 1e-12f)) + a2;
    float t2 = tmp * tmp, t4 = t2 * t2;
    float t6 = t4 * t2, t8 = t4 * t4;

    float r_b = r_ang * ANG_TO_BOHR;
    float rb2 = r_b * r_b, rb4 = rb2 * rb2;
    float rb6 = rb4 * rb2, rb8 = rb4 * rb4;
    float e6 = 1.0f / (rb6 + t6);
    float e8 = 1.0f / (rb8 + t8);
    float A = s6 * e6 + (s8 * q_ij) * e8;

    float t = clamp((r_ang - taper_on) * inv_tspan, 0.0f, 1.0f);
    float taper = 1.0f - t * t * t * (10.0f + t * (6.0f * t - 15.0f));
    acc_e += (-0.5f * HARTREE_KCAL) * c6 * A * taper;

    if (need_forces != 0) {
        float dc6_dcni = (-2.0f * D3_K3) * (sum_wc0_d * inv_sw - c6 * (sum_wd * inv_sw));
        acc_dedcn += (-HARTREE_KCAL) * dc6_dcni * A * taper;
        float dA_dr = -ANG_TO_BOHR * (6.0f * s6 * (rb4 * r_b) * (e6 * e6) + 8.0f * (s8 * q_ij) * (rb6 * r_b) * (e8 * e8));
        float dtaper_dr = (r_ang > taper_on && r_ang < r_cut) ? (-30.0f * (t * t) * ((1.0f - t) * (1.0f - t)) * inv_tspan) : 0.0f;
        float dU_dr = (-HARTREE_KCAL) * c6 * (dA_dr * taper + A * dtaper_dr);
        acc_f += (-dU_dr * inv_r_ang) * d;
    }
}

kernel void d3_cn_grouped(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* group_ptr [[buffer(4)]],
    device const float* rcov [[buffer(5)]],
    device float* out_cn [[buffer(6)]],
    constant int& n_atoms [[buffer(7)]],
    constant float4& params1 [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms) return;
    if (mask[i] == 0) { out_cn[i] = 0.0f; return; }
    int zi = int(clamp(z[i], 0L, 94L));
    float rc_i = rcov[zi], cn_cut_b = params1.z;
    int g = int(graph[i]);
    int j0 = group_ptr[g], j1 = group_ptr[g + 1];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float acc = 0.0f;
    for (int j = j0; j < j1; ++j) {
        if (j == i || mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        float r_b = sqrt(dot(d, d) + 1e-8f) * ANG_TO_BOHR;
        if (r_b <= cn_cut_b) {
            int zj = int(clamp(z[j], 0L, 94L));
            float rr = (rc_i + rcov[zj]) / max(r_b, 1e-6f);
            acc += 1.0f / (1.0f + exp(-D3_K1 * (rr - 1.0f)));
        }
    }
    out_cn[i] = acc;
}

kernel void d3_energy_fdir_grouped(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* group_ptr [[buffer(4)]],
    device const float* r2r4 [[buffer(5)]],
    device const float* c6ab_flat [[buffer(6)]],
    device const float* cn [[buffer(7)]],
    device float* out_e [[buffer(8)]],
    device float* out_dedcn [[buffer(9)]],
    device float* out_f [[buffer(10)]],
    constant int& n_atoms [[buffer(11)]],
    constant int& need_forces [[buffer(12)]],
    constant float4& params0 [[buffer(13)]],
    constant float4& params1 [[buffer(14)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms) return;
    if (mask[i] == 0) {
        out_e[i] = 0.0f;
        if (need_forces != 0) {
            out_dedcn[i] = 0.0f;
            out_f[3 * i] = 0.0f; out_f[3 * i + 1] = 0.0f; out_f[3 * i + 2] = 0.0f;
        }
        return;
    }
    int zi = int(clamp(z[i], 0L, 94L));
    float r2r4_i = r2r4[zi], cn_i = cn[i];
    int g = int(graph[i]);
    int j0 = group_ptr[g], j1 = group_ptr[g + 1];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float acc_e = 0.0f, acc_dedcn = 0.0f;
    float3 acc_f = float3(0.0f);
    for (int j = j0; j < j1; ++j) {
        if (j == i || mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        int zj = int(clamp(z[j], 0L, 94L));
        d3_accum_pair(d, zi, zj, cn_i, cn[j], r2r4_i, r2r4, c6ab_flat, params0, params1, need_forces, acc_e, acc_dedcn, acc_f);
    }
    out_e[i] = acc_e;
    if (need_forces != 0) {
        out_dedcn[i] = acc_dedcn;
        out_f[3 * i] = acc_f.x; out_f[3 * i + 1] = acc_f.y; out_f[3 * i + 2] = acc_f.z;
    }
}

kernel void d3_fcn_grouped(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* group_ptr [[buffer(4)]],
    device const float* rcov [[buffer(5)]],
    device const float* dedcn [[buffer(6)]],
    device float* inout_f [[buffer(7)]],
    constant int& n_atoms [[buffer(8)]],
    constant float4& params1 [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms || mask[i] == 0) return;
    int zi = int(clamp(z[i], 0L, 94L));
    float rc_i = rcov[zi], de_i = dedcn[i], cn_cut_b = params1.z;
    int g = int(graph[i]);
    int j0 = group_ptr[g], j1 = group_ptr[g + 1];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float3 acc_f = float3(inout_f[3 * i], inout_f[3 * i + 1], inout_f[3 * i + 2]);
    for (int j = j0; j < j1; ++j) {
        if (j == i || mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        float r2_eps = dot(d, d) + 1e-8f;
        float inv_r_ang = rsqrt(r2_eps);
        float r_b = (r2_eps * inv_r_ang) * ANG_TO_BOHR;
        if (r_b <= cn_cut_b && r_b > 1e-6f) {
            int zj = int(clamp(z[j], 0L, 94L));
            float rr = (rc_i + rcov[zj]) / r_b;
            float damp = 1.0f / (1.0f + exp(-D3_K1 * (rr - 1.0f)));
            acc_f += ((de_i + dedcn[j]) * D3_K1 * damp * (1.0f - damp) * rr * (inv_r_ang * inv_r_ang)) * d;
        }
    }
    inout_f[3 * i] = acc_f.x; inout_f[3 * i + 1] = acc_f.y; inout_f[3 * i + 2] = acc_f.z;
}

kernel void d3_cn_csr(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const int* recv_ptr [[buffer(2)]],
    device const int* send_idx [[buffer(3)]],
    device const float* rcov [[buffer(4)]],
    device float* out_cn [[buffer(5)]],
    constant int& n_atoms [[buffer(6)]],
    constant float4& params1 [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms) return;
    int zi = int(clamp(z[i], 0L, 94L));
    float rc_i = rcov[zi], cn_cut_b = params1.z;
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float acc = 0.0f;
    for (int e = recv_ptr[i], e_end = recv_ptr[i + 1]; e < e_end; ++e) {
        int j = send_idx[e];
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        float r_b = sqrt(dot(d, d) + 1e-8f) * ANG_TO_BOHR;
        if (r_b <= cn_cut_b) {
            int zj = int(clamp(z[j], 0L, 94L));
            float rr = (rc_i + rcov[zj]) / max(r_b, 1e-6f);
            acc += 1.0f / (1.0f + exp(-D3_K1 * (rr - 1.0f)));
        }
    }
    out_cn[i] = acc;
}

kernel void d3_energy_fdir_csr(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const int* recv_ptr [[buffer(2)]],
    device const int* send_idx [[buffer(3)]],
    device const float* r2r4 [[buffer(4)]],
    device const float* c6ab_flat [[buffer(5)]],
    device const float* cn [[buffer(6)]],
    device float* out_e [[buffer(7)]],
    device float* out_dedcn [[buffer(8)]],
    device float* out_f [[buffer(9)]],
    constant int& n_atoms [[buffer(10)]],
    constant int& need_forces [[buffer(11)]],
    constant float4& params0 [[buffer(12)]],
    constant float4& params1 [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms) return;
    int zi = int(clamp(z[i], 0L, 94L));
    float r2r4_i = r2r4[zi], cn_i = cn[i];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float acc_e = 0.0f, acc_dedcn = 0.0f;
    float3 acc_f = float3(0.0f);
    for (int e = recv_ptr[i], e_end = recv_ptr[i + 1]; e < e_end; ++e) {
        int j = send_idx[e];
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        int zj = int(clamp(z[j], 0L, 94L));
        d3_accum_pair(d, zi, zj, cn_i, cn[j], r2r4_i, r2r4, c6ab_flat, params0, params1, need_forces, acc_e, acc_dedcn, acc_f);
    }
    out_e[i] = acc_e;
    if (need_forces != 0) {
        out_dedcn[i] = acc_dedcn;
        out_f[3 * i] = acc_f.x; out_f[3 * i + 1] = acc_f.y; out_f[3 * i + 2] = acc_f.z;
    }
}

kernel void d3_fcn_csr(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const int* recv_ptr [[buffer(2)]],
    device const int* send_idx [[buffer(3)]],
    device const float* rcov [[buffer(4)]],
    device const float* dedcn [[buffer(5)]],
    device float* inout_f [[buffer(6)]],
    constant int& n_atoms [[buffer(7)]],
    constant float4& params1 [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid);
    if (i >= n_atoms) return;
    int zi = int(clamp(z[i], 0L, 94L));
    float rc_i = rcov[zi], de_i = dedcn[i], cn_cut_b = params1.z;
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float3 acc_f = float3(inout_f[3 * i], inout_f[3 * i + 1], inout_f[3 * i + 2]);
    for (int e = recv_ptr[i], e_end = recv_ptr[i + 1]; e < e_end; ++e) {
        int j = send_idx[e];
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        float r2_eps = dot(d, d) + 1e-8f;
        float inv_r_ang = rsqrt(r2_eps);
        float r_b = (r2_eps * inv_r_ang) * ANG_TO_BOHR;
        if (r_b <= cn_cut_b && r_b > 1e-6f) {
            int zj = int(clamp(z[j], 0L, 94L));
            float rr = (rc_i + rcov[zj]) / r_b;
            float damp = 1.0f / (1.0f + exp(-D3_K1 * (rr - 1.0f)));
            acc_f += ((de_i + dedcn[j]) * D3_K1 * damp * (1.0f - damp) * rr * (inv_r_ang * inv_r_ang)) * d;
        }
    }
    inout_f[3 * i] = acc_f.x; inout_f[3 * i + 1] = acc_f.y; inout_f[3 * i + 2] = acc_f.z;
}
"""
