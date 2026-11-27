"""Metal Shading Language source for rank-1 factorized Tier-2 Grimme D3(BJ)/D3(0) dispersion kernels.

Exact mathematical optimization: Grimme's `c6ab[95, 95, 5, 5, 3]` table is strictly rectangular with per-element
reference CNs (`0` mismatches and `0` holes across all 95x95 pairs). Therefore the 25-term Gaussian softmax
factorizes exactly as `w_ab = p_i[a] * p_j[b]`, computed ONCE per atom at the end of Pass 1 (CN), eliminating
all `25 * exp()` calls and the two-pass softmax from the inner pair loop and cutting C6 table traffic 3x.
All three passes use 8 lanes per atom (`simd_shuffle_down`) for high GPU occupancy even at N=64..1024.
"""

METAL_D3_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant float ANG_TO_BOHR  = 1.8897261245650618f;
constant float HARTREE_KCAL = 627.509474f;
constant float D3_K1        = 16.0f;
constant float D3_K3        = -4.0f;

inline float sub8_sum(float v) {
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    return v;
}

inline void write_cn_weights(
    int i, int zi, float cn_val,
    device const int* n_ref, device const float* cn_ref,
    device float* out_pw, device float* out_dpw
) {
    int nr = n_ref[zi];
    int base = 5 * i;
    if (nr <= 0) {
        for (int a = 0; a < 5; ++a) { out_pw[base + a] = 0.0f; out_dpw[base + a] = 0.0f; }
        return;
    }
    device const float* cr = cn_ref + 5 * zi;
    float d[5];
    float smax = -1.0e20f;
    for (int a = 0; a < nr; ++a) {
        d[a] = cr[a] - cn_val;
        smax = max(smax, D3_K3 * d[a] * d[a]);
    }
    float w[5];
    float sw = 0.0f;
    for (int a = 0; a < nr; ++a) {
        w[a] = exp(D3_K3 * d[a] * d[a] - smax);
        sw += w[a];
    }
    float inv_sw = 1.0f / max(sw, 1.0e-30f);
    float dbar = 0.0f;
    for (int a = 0; a < nr; ++a) {
        w[a] *= inv_sw;
        dbar = fma(w[a], d[a], dbar);
    }
    for (int a = 0; a < 5; ++a) {
        if (a < nr) {
            out_pw[base + a] = w[a];
            out_dpw[base + a] = (-2.0f * D3_K3) * w[a] * (d[a] - dbar);
        } else {
            out_pw[base + a] = 0.0f;
            out_dpw[base + a] = 0.0f;
        }
    }
}

inline void d3_accum_pair_r1(
    float3 d, int zi, int zj, int ni, int nj,
    const thread float pw_i[5], const thread float dpw_i[5],
    device const float* pw, float r2r4_i, device const float* r2r4,
    device const float* c6_mat, device const float* r0ab, float4 params0, float4 params1,
    int is_zero, int need_deriv, int need_forces, int need_stress, int j,
    thread float& acc_e, thread float& acc_dedcn, thread float3& acc_f,
    thread float3& acc_vd, thread float3& acc_vo
) {
    float r_cut = params1.x, taper_on = params1.y, inv_tspan = params1.w;
    float d2 = dot(d, d);
    if (d2 > r_cut * r_cut || ni <= 0 || nj <= 0) return;
    float r2_eps = d2 + 1.0e-8f;
    float r_ang = sqrt(r2_eps);
    if (r_ang > r_cut) return;
    float inv_r = 1.0f / r_ang;

    device const float* tab = c6_mat + 25 * (zi * 95 + zj);
    device const float* pj = pw + 5 * j;
    float c6 = 0.0f, dc6 = 0.0f;
    for (int a = 0; a < ni; ++a) {
        int row = a * 5;
        float s = 0.0f;
        for (int b = 0; b < nj; ++b) {
            s = fma(pj[b], tab[row + b], s);
        }
        c6 = fma(pw_i[a], s, c6);
        if (need_deriv != 0) dc6 = fma(dpw_i[a], s, dc6);
    }

    float q_ij = 3.0f * r2r4_i * r2r4[zj];
    float s6 = params0.x, s8 = params0.y, p_a1 = params0.z, p_a2 = params0.w;
    float r_b = r_ang * ANG_TO_BOHR;
    float rb2 = r_b * r_b, rb4 = rb2 * rb2;
    float rb6 = rb4 * rb2, rb8 = rb4 * rb4;
    float A, dA_dr = 0.0f;
    if (is_zero != 0) {
        float r0 = max(r0ab[zi * 95 + zj], 1.0e-6f);
        float rb_s = max(r_b, 1.0e-6f);
        float u6 = (p_a1 * r0) / rb_s, u8 = r0 / rb_s;
        float u6_2 = u6 * u6, u8_2 = u8 * u8;
        float u6_4 = u6_2 * u6_2, u8_4 = u8_2 * u8_2;
        float g6 = 6.0f * (u6_4 * u6_4 * u6_4 * u6_2);
        float g8 = 6.0f * (u8_4 * u8_4 * u8_4 * u8_4);
        float f6 = 1.0f / (1.0f + g6), f8 = 1.0f / (1.0f + g8);
        float t6_v = s6 * f6 / max(rb6, 1.0e-12f);
        float t8_v = (s8 * q_ij) * f8 / max(rb8, 1.0e-12f);
        A = t6_v + t8_v;
        if (need_deriv != 0) {
            float p6 = (rb6 > 1.0e-12f) ? 6.0f : 0.0f;
            float p8 = (rb8 > 1.0e-12f) ? 8.0f : 0.0f;
            dA_dr = ANG_TO_BOHR * ((t6_v * (14.0f * g6 * f6 - p6) + t8_v * (16.0f * g8 * f8 - p8)) / rb_s);
        }
    } else {
        float tmp = p_a1 * sqrt(max(q_ij, 1.0e-12f)) + p_a2;
        float t2 = tmp * tmp, t4 = t2 * t2;
        float e6 = 1.0f / (rb6 + t4 * t2);
        float e8 = 1.0f / (rb8 + t4 * t4);
        A = s6 * e6 + (s8 * q_ij) * e8;
        if (need_deriv != 0) {
            dA_dr = -ANG_TO_BOHR * (6.0f * s6 * (rb4 * r_b) * (e6 * e6) + 8.0f * (s8 * q_ij) * (rb6 * r_b) * (e8 * e8));
        }
    }

    float t = clamp((r_ang - taper_on) * inv_tspan, 0.0f, 1.0f);
    float taper = 1.0f - t * t * t * (10.0f + t * (6.0f * t - 15.0f));
    acc_e += (-0.5f * HARTREE_KCAL) * c6 * A * taper;

    if (need_deriv != 0) {
        acc_dedcn += (-HARTREE_KCAL) * dc6 * A * taper;
        float dtap = (r_ang > taper_on && r_ang < r_cut) ? (-30.0f * (t * t) * ((1.0f - t) * (1.0f - t)) * inv_tspan) : 0.0f;
        float fm = HARTREE_KCAL * c6 * (dA_dr * taper + A * dtap) * inv_r;
        if (need_forces != 0) acc_f += fm * d;
        if (need_stress != 0) {
            float vm = -0.5f * fm;
            acc_vd += vm * (d * d);
            acc_vo += vm * float3(d.x * d.y, d.x * d.z, d.y * d.z);
        }
    }
}

inline float3 apply_shift_cell(float3 d, device const float* shift, device const float* cell, int e, int g, int has_shift, int has_cell) {
    if (has_shift == 0) return d;
    float sx = shift[3 * e], sy = shift[3 * e + 1], sz = shift[3 * e + 2];
    if (has_cell != 0) {
        device const float* c = cell + 9 * g;
        d.x -= sx * c[0] + sy * c[3] + sz * c[6];
        d.y -= sx * c[1] + sy * c[4] + sz * c[7];
        d.z -= sx * c[2] + sy * c[5] + sz * c[8];
    } else {
        d -= float3(sx, sy, sz);
    }
    return d;
}

kernel void d3_cn_grouped(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* group_ptr [[buffer(4)]],
    device const float* rcov [[buffer(5)]],
    device const int* n_ref [[buffer(6)]],
    device const float* cn_ref [[buffer(7)]],
    device float* out_pw [[buffer(8)]],
    device float* out_dpw [[buffer(9)]],
    constant int& n_atoms [[buffer(10)]],
    constant float4& params1 [[buffer(11)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    int i = int(tid >> 3);
    int sub = int(lane & 7);
    if (i >= n_atoms) return;
    if (mask[i] == 0) {
        if (sub == 0) write_cn_weights(i, 0, 0.0f, n_ref, cn_ref, out_pw, out_dpw);
        return;
    }
    int zi = int(clamp(z[i], 0L, 94L));
    float rc_i = rcov[zi], cn_cut_b = params1.z;
    int g = int(graph[i]);
    int j0 = group_ptr[g], j1 = group_ptr[g + 1];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float acc = 0.0f;
    for (int j = j0 + sub; j < j1; j += 8) {
        if (j == i || mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        float r_b = sqrt(dot(d, d) + 1.0e-8f) * ANG_TO_BOHR;
        if (r_b <= cn_cut_b) {
            int zj = int(clamp(z[j], 0L, 94L));
            float rr = (rc_i + rcov[zj]) / max(r_b, 1.0e-6f);
            acc += 1.0f / (1.0f + exp(-D3_K1 * (rr - 1.0f)));
        }
    }
    acc = sub8_sum(acc);
    if (sub == 0) write_cn_weights(i, zi, acc, n_ref, cn_ref, out_pw, out_dpw);
}

kernel void d3_ef_grouped(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* group_ptr [[buffer(4)]],
    device const float* r2r4 [[buffer(5)]],
    device const int* n_ref [[buffer(6)]],
    device const float* c6_mat [[buffer(7)]],
    device const float* pw [[buffer(8)]],
    device const float* dpw [[buffer(9)]],
    device float* out_e [[buffer(10)]],
    device float* out_dedcn [[buffer(11)]],
    device float* out_f [[buffer(12)]],
    constant int& n_atoms [[buffer(13)]],
    constant int& is_zero [[buffer(14)]],
    constant int& need_forces [[buffer(15)]],
    constant float4& params0 [[buffer(16)]],
    constant float4& params1 [[buffer(17)]],
    device const float* r0ab [[buffer(18)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    int i = int(tid >> 3);
    int sub = int(lane & 7);
    if (i >= n_atoms) return;
    if (mask[i] == 0) {
        if (sub == 0) {
            out_e[i] = 0.0f;
            if (need_forces != 0) {
                out_dedcn[i] = 0.0f;
                out_f[3 * i] = 0.0f; out_f[3 * i + 1] = 0.0f; out_f[3 * i + 2] = 0.0f;
            }
        }
        return;
    }
    int zi = int(clamp(z[i], 0L, 94L));
    int ni = n_ref[zi];
    float r2r4_i = r2r4[zi];
    float pw_i[5], dpw_i[5];
    for (int a = 0; a < 5; ++a) { pw_i[a] = pw[5 * i + a]; dpw_i[a] = dpw[5 * i + a]; }
    int g = int(graph[i]);
    int j0 = group_ptr[g], j1 = group_ptr[g + 1];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float acc_e = 0.0f, acc_dedcn = 0.0f;
    float3 acc_f = float3(0.0f), vd = float3(0.0f), vo = float3(0.0f);
    for (int j = j0 + sub; j < j1; j += 8) {
        if (j == i || mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        int zj = int(clamp(z[j], 0L, 94L));
        d3_accum_pair_r1(d, zi, zj, ni, n_ref[zj], pw_i, dpw_i, pw, r2r4_i, r2r4, c6_mat, r0ab,
                         params0, params1, is_zero, need_forces, need_forces, 0, j,
                         acc_e, acc_dedcn, acc_f, vd, vo);
    }
    acc_e = sub8_sum(acc_e);
    if (need_forces != 0) {
        acc_dedcn = sub8_sum(acc_dedcn);
        acc_f = float3(sub8_sum(acc_f.x), sub8_sum(acc_f.y), sub8_sum(acc_f.z));
    }
    if (sub == 0) {
        out_e[i] = acc_e;
        if (need_forces != 0) {
            out_dedcn[i] = acc_dedcn;
            out_f[3 * i] = acc_f.x; out_f[3 * i + 1] = acc_f.y; out_f[3 * i + 2] = acc_f.z;
        }
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
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    int i = int(tid >> 3);
    int sub = int(lane & 7);
    if (i >= n_atoms || mask[i] == 0) return;
    int zi = int(clamp(z[i], 0L, 94L));
    float rc_i = rcov[zi], de_i = dedcn[i], cn_cut_b = params1.z;
    int g = int(graph[i]);
    int j0 = group_ptr[g], j1 = group_ptr[g + 1];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float3 acc_f = float3(0.0f);
    for (int j = j0 + sub; j < j1; j += 8) {
        if (j == i || mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        float r2_eps = dot(d, d) + 1.0e-8f;
        float r_ang = sqrt(r2_eps);
        float inv_r = 1.0f / r_ang;
        float r_b = r_ang * ANG_TO_BOHR;
        if (r_b <= cn_cut_b && r_b > 1.0e-6f) {
            int zj = int(clamp(z[j], 0L, 94L));
            float rr = (rc_i + rcov[zj]) / r_b;
            float damp = 1.0f / (1.0f + exp(-D3_K1 * (rr - 1.0f)));
            acc_f += ((de_i + dedcn[j]) * D3_K1 * damp * (1.0f - damp) * rr * (inv_r * inv_r)) * d;
        }
    }
    acc_f = float3(sub8_sum(acc_f.x), sub8_sum(acc_f.y), sub8_sum(acc_f.z));
    if (sub == 0) {
        inout_f[3 * i] += acc_f.x;
        inout_f[3 * i + 1] += acc_f.y;
        inout_f[3 * i + 2] += acc_f.z;
    }
}

kernel void d3_cn_csr(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* recv_ptr [[buffer(4)]],
    device const int* send_idx [[buffer(5)]],
    device const float* shift [[buffer(6)]],
    device const float* cell [[buffer(7)]],
    device const float* rcov [[buffer(8)]],
    device const int* n_ref [[buffer(9)]],
    device const float* cn_ref [[buffer(10)]],
    device float* out_pw [[buffer(11)]],
    device float* out_dpw [[buffer(12)]],
    constant int& n_atoms [[buffer(13)]],
    constant int& flags [[buffer(14)]],
    constant float4& params1 [[buffer(15)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    int i = int(tid >> 3);
    int sub = int(lane & 7);
    if (i >= n_atoms) return;
    int has_mask = flags & 1, has_shift = (flags >> 1) & 1, has_cell = (flags >> 2) & 1, multi_cell = (flags >> 3) & 1;
    if (has_mask != 0 && mask[i] == 0) {
        if (sub == 0) write_cn_weights(i, 0, 0.0f, n_ref, cn_ref, out_pw, out_dpw);
        return;
    }
    int zi = int(clamp(z[i], 0L, 94L));
    float rc_i = rcov[zi], cn_cut_b = params1.z;
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    int gi = (multi_cell != 0) ? int(graph[i]) : 0;
    int e0 = recv_ptr[i], e1 = recv_ptr[i + 1];
    float acc = 0.0f;
    for (int e = e0 + sub; e < e1; e += 8) {
        int j = send_idx[e];
        if (has_mask != 0 && mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        d = apply_shift_cell(d, shift, cell, e, gi, has_shift, has_cell);
        float d2 = dot(d, d);
        if (d2 <= 1.0e-12f) continue;
        float r_b = sqrt(d2 + 1.0e-8f) * ANG_TO_BOHR;
        if (r_b <= cn_cut_b) {
            int zj = int(clamp(z[j], 0L, 94L));
            float rr = (rc_i + rcov[zj]) / max(r_b, 1.0e-6f);
            acc += 1.0f / (1.0f + exp(-D3_K1 * (rr - 1.0f)));
        }
    }
    acc = sub8_sum(acc);
    if (sub == 0) write_cn_weights(i, zi, acc, n_ref, cn_ref, out_pw, out_dpw);
}

kernel void d3_ef_csr(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* recv_ptr [[buffer(4)]],
    device const int* send_idx [[buffer(5)]],
    device const float* shift [[buffer(6)]],
    device const float* cell [[buffer(7)]],
    device const float* r2r4 [[buffer(8)]],
    device const int* n_ref [[buffer(9)]],
    device const float* c6_mat [[buffer(10)]],
    device const float* pw [[buffer(11)]],
    device const float* dpw [[buffer(12)]],
    device float* out_e [[buffer(13)]],
    device float* out_dedcn [[buffer(14)]],
    device float* out_f [[buffer(15)]],
    device float* out_vir [[buffer(16)]],
    constant int& n_atoms [[buffer(17)]],
    constant int& flags [[buffer(18)]],
    constant float4& params0 [[buffer(19)]],
    constant float4& params1 [[buffer(20)]],
    device const float* r0ab [[buffer(21)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    int i = int(tid >> 3);
    int sub = int(lane & 7);
    if (i >= n_atoms) return;
    int has_mask = flags & 1, has_shift = (flags >> 1) & 1, has_cell = (flags >> 2) & 1, multi_cell = (flags >> 3) & 1;
    int is_zero = (flags >> 4) & 1, need_forces = (flags >> 5) & 1, need_stress = (flags >> 6) & 1;
    int need_deriv = (need_forces | need_stress);
    if (has_mask != 0 && mask[i] == 0) {
        if (sub == 0) {
            out_e[i] = 0.0f;
            if (need_deriv != 0) out_dedcn[i] = 0.0f;
            if (need_forces != 0) { out_f[3 * i] = 0.0f; out_f[3 * i + 1] = 0.0f; out_f[3 * i + 2] = 0.0f; }
            if (need_stress != 0) { for (int m = 0; m < 6; ++m) out_vir[6 * i + m] = 0.0f; }
        }
        return;
    }
    int zi = int(clamp(z[i], 0L, 94L));
    int ni = n_ref[zi];
    float r2r4_i = r2r4[zi];
    float pw_i[5], dpw_i[5];
    for (int a = 0; a < 5; ++a) { pw_i[a] = pw[5 * i + a]; dpw_i[a] = dpw[5 * i + a]; }
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    int gi = (multi_cell != 0) ? int(graph[i]) : 0;
    int e0 = recv_ptr[i], e1 = recv_ptr[i + 1];
    float acc_e = 0.0f, acc_dedcn = 0.0f;
    float3 acc_f = float3(0.0f), vd = float3(0.0f), vo = float3(0.0f);
    for (int e = e0 + sub; e < e1; e += 8) {
        int j = send_idx[e];
        if (has_mask != 0 && mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        d = apply_shift_cell(d, shift, cell, e, gi, has_shift, has_cell);
        if (dot(d, d) <= 1.0e-12f) continue;
        int zj = int(clamp(z[j], 0L, 94L));
        d3_accum_pair_r1(d, zi, zj, ni, n_ref[zj], pw_i, dpw_i, pw, r2r4_i, r2r4, c6_mat, r0ab,
                         params0, params1, is_zero, need_deriv, need_forces, need_stress, j,
                         acc_e, acc_dedcn, acc_f, vd, vo);
    }
    acc_e = sub8_sum(acc_e);
    if (need_deriv != 0) acc_dedcn = sub8_sum(acc_dedcn);
    if (need_forces != 0) acc_f = float3(sub8_sum(acc_f.x), sub8_sum(acc_f.y), sub8_sum(acc_f.z));
    if (need_stress != 0) {
        vd = float3(sub8_sum(vd.x), sub8_sum(vd.y), sub8_sum(vd.z));
        vo = float3(sub8_sum(vo.x), sub8_sum(vo.y), sub8_sum(vo.z));
    }
    if (sub == 0) {
        out_e[i] = acc_e;
        if (need_deriv != 0) out_dedcn[i] = acc_dedcn;
        if (need_forces != 0) { out_f[3 * i] = acc_f.x; out_f[3 * i + 1] = acc_f.y; out_f[3 * i + 2] = acc_f.z; }
        if (need_stress != 0) {
            out_vir[6 * i + 0] = vd.x; out_vir[6 * i + 1] = vd.y; out_vir[6 * i + 2] = vd.z;
            out_vir[6 * i + 3] = vo.x; out_vir[6 * i + 4] = vo.y; out_vir[6 * i + 5] = vo.z;
        }
    }
}

kernel void d3_fcn_csr(
    device const float* pos [[buffer(0)]],
    device const long* z [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const long* graph [[buffer(3)]],
    device const int* recv_ptr [[buffer(4)]],
    device const int* send_idx [[buffer(5)]],
    device const float* shift [[buffer(6)]],
    device const float* cell [[buffer(7)]],
    device const float* rcov [[buffer(8)]],
    device const float* dedcn [[buffer(9)]],
    device float* inout_f [[buffer(10)]],
    device float* inout_vir [[buffer(11)]],
    constant int& n_atoms [[buffer(12)]],
    constant int& flags [[buffer(13)]],
    constant float4& params1 [[buffer(14)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    int i = int(tid >> 3);
    int sub = int(lane & 7);
    if (i >= n_atoms) return;
    int has_mask = flags & 1, has_shift = (flags >> 1) & 1, has_cell = (flags >> 2) & 1, multi_cell = (flags >> 3) & 1;
    int need_forces = (flags >> 5) & 1, need_stress = (flags >> 6) & 1;
    if (has_mask != 0 && mask[i] == 0) return;
    int zi = int(clamp(z[i], 0L, 94L));
    float rc_i = rcov[zi], de_i = dedcn[i], cn_cut_b = params1.z;
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    int gi = (multi_cell != 0) ? int(graph[i]) : 0;
    int e0 = recv_ptr[i], e1 = recv_ptr[i + 1];
    float3 acc_f = float3(0.0f), vd = float3(0.0f), vo = float3(0.0f);
    for (int e = e0 + sub; e < e1; e += 8) {
        int j = send_idx[e];
        if (has_mask != 0 && mask[j] == 0) continue;
        float3 d = pi - float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
        d = apply_shift_cell(d, shift, cell, e, gi, has_shift, has_cell);
        float d2 = dot(d, d);
        if (d2 <= 1.0e-12f) continue;
        float r2_eps = d2 + 1.0e-8f;
        float r_ang = sqrt(r2_eps);
        float inv_r = 1.0f / r_ang;
        float r_b = r_ang * ANG_TO_BOHR;
        if (r_b <= cn_cut_b && r_b > 1.0e-6f) {
            int zj = int(clamp(z[j], 0L, 94L));
            float rr = (rc_i + rcov[zj]) / r_b;
            float damp = 1.0f / (1.0f + exp(-D3_K1 * (rr - 1.0f)));
            float fm = (de_i + dedcn[j]) * D3_K1 * damp * (1.0f - damp) * rr * (inv_r * inv_r);
            if (need_forces != 0) acc_f += fm * d;
            if (need_stress != 0) {
                float vm = -0.5f * fm;
                vd += vm * (d * d);
                vo += vm * float3(d.x * d.y, d.x * d.z, d.y * d.z);
            }
        }
    }
    if (need_forces != 0) {
        acc_f = float3(sub8_sum(acc_f.x), sub8_sum(acc_f.y), sub8_sum(acc_f.z));
        if (sub == 0) {
            inout_f[3 * i] += acc_f.x; inout_f[3 * i + 1] += acc_f.y; inout_f[3 * i + 2] += acc_f.z;
        }
    }
    if (need_stress != 0) {
        vd = float3(sub8_sum(vd.x), sub8_sum(vd.y), sub8_sum(vd.z));
        vo = float3(sub8_sum(vo.x), sub8_sum(vo.y), sub8_sum(vo.z));
        if (sub == 0) {
            inout_vir[6 * i + 0] += vd.x; inout_vir[6 * i + 1] += vd.y; inout_vir[6 * i + 2] += vd.z;
            inout_vir[6 * i + 3] += vo.x; inout_vir[6 * i + 4] += vo.y; inout_vir[6 * i + 5] += vo.z;
        }
    }
}

kernel void d3_csr_count_deg(
    device const int* recv [[buffer(0)]],
    device atomic_int* deg [[buffer(1)]],
    constant uint& n_edges [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_edges) return;
    atomic_fetch_add_explicit(&deg[recv[tid]], 1, memory_order_relaxed);
}

kernel void d3_csr_scatter(
    device const int* recv [[buffer(0)]],
    device const int* send [[buffer(1)]],
    device atomic_int* cursor [[buffer(2)]],
    device int* out_send [[buffer(3)]],
    constant uint& n_edges [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_edges) return;
    int pos = atomic_fetch_add_explicit(&cursor[recv[tid]], 1, memory_order_relaxed);
    out_send[pos] = send[tid];
}

kernel void d3_csr_scatter_shift(
    device const int* recv [[buffer(0)]],
    device const int* send [[buffer(1)]],
    device const float* shift [[buffer(2)]],
    device atomic_int* cursor [[buffer(3)]],
    device int* out_send [[buffer(4)]],
    device float* out_shift [[buffer(5)]],
    constant uint& n_edges [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_edges) return;
    int pos = atomic_fetch_add_explicit(&cursor[recv[tid]], 1, memory_order_relaxed);
    out_send[pos] = send[tid];
    out_shift[3 * pos + 0] = shift[3 * tid + 0];
    out_shift[3 * pos + 1] = shift[3 * tid + 1];
    out_shift[3 * pos + 2] = shift[3 * tid + 2];
}
"""
