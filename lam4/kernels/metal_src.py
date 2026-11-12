"""Metal Shading Language (MSL) common math + forward CSR kernel (`1 simdgroup = 1 atom`, zero atomics)."""
from __future__ import annotations

from lam4.kernels.metal_src_bwd import METAL_BWD_SOURCE

_METAL_FWD_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant int FLAG_FLOOR = 1;
constant int FLAG_RHO   = 2;
constant int FLAG_TEN   = 4;
constant int FLAG_CELL  = 8;
constant int FLAG_G1    = 16;
constant int FLAG_VIR   = 32;
constant int FLAG_DRHO  = 64;

constant float S2     = 1.4142135623730951f;
constant float INV_S2 = 0.7071067811865475f;
constant float INV_S6 = 0.4082482904638630f;
constant float C3_0   = 0.5f;
constant float C3_1   = 2.449489742783178f;
constant float C3_2   = 0.3872983346207417f;
constant float C3_3   = 0.31622776601683794f;
constant float C3_4   = 0.3872983346207417f;
constant float C3_5   = 1.224744871391589f;
constant float C3_6   = 0.5f;

inline void eval_y2(float3 u, thread float (&y2)[5]) {
    float x = u.x, y = u.y, z = u.z;
    y2[0] = S2 * x * y;
    y2[1] = S2 * x * z;
    y2[2] = S2 * y * z;
    y2[3] = (x * x - y * y) * INV_S2;
    y2[4] = (2.0f * z * z - x * x - y * y) * INV_S6;
}

inline void eval_y3(float3 u, thread float (&y3)[7]) {
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

inline float3 grad_y2_u(float3 u, thread const float (&g)[5]) {
    float x = u.x, y = u.y, z = u.z, k6 = 2.0f * INV_S6;
    float gx = S2 * (g[0] * y + g[1] * z + g[3] * x) - k6 * g[4] * x;
    float gy = S2 * (g[0] * x + g[2] * z - g[3] * y) - k6 * g[4] * y;
    float gz = S2 * (g[1] * x + g[2] * y) + 2.0f * k6 * g[4] * z;
    return float3(gx, gy, gz);
}

inline float3 grad_y3_u(float3 u, thread const float (&g)[7]) {
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
    return float3(gx, gy, gz);
}

inline float horner_val(device const float* p0, int td, int col, float t) {
    float a0 = p0[col], a1 = p0[td + col], a2 = p0[2 * td + col], a3 = p0[3 * td + col];
    return fma(fma(fma(a3, t, a2), t, a1), t, a0);
}

inline void horner_val_der(device const float* p0, int td, int col, float t, float inv_h, thread float& fv, thread float& df) {
    float a0 = p0[col], a1 = p0[td + col], a2 = p0[2 * td + col], a3 = p0[3 * td + col];
    fv = fma(fma(fma(a3, t, a2), t, a1), t, a0);
    df = fma(fma(3.0f * a3, t, 2.0f * a2), t, a1) * inv_h;
}

inline void qk_dots4(float q0, float q1, float k0, float k1, thread float& d0, thread float& d1, thread float& d2, thread float& d3) {
    float p0 = q0 * k0, p1 = q1 * k1;
    p0 += simd_shuffle_xor(p0, ushort(8)); p0 += simd_shuffle_xor(p0, ushort(4));
    p0 += simd_shuffle_xor(p0, ushort(2)); p0 += simd_shuffle_xor(p0, ushort(1));
    p1 += simd_shuffle_xor(p1, ushort(8)); p1 += simd_shuffle_xor(p1, ushort(4));
    p1 += simd_shuffle_xor(p1, ushort(2)); p1 += simd_shuffle_xor(p1, ushort(1));
    d0 = simd_broadcast(p0, ushort(0)); d1 = simd_broadcast(p0, ushort(16));
    d2 = simd_broadcast(p1, ushort(0)); d3 = simd_broadcast(p1, ushort(16));
}

inline float3 load_shift(device const float* shift_in, device const float* cell, int e, int g_id, int flags) {
    float3 es = float3(shift_in[3 * e], shift_in[3 * e + 1], shift_in[3 * e + 2]);
    if ((flags & FLAG_CELL) == 0) return es;
    device const float* c = cell + 9 * g_id;
    return float3(es.x * c[0] + es.y * c[3] + es.z * c[6],
                  es.x * c[1] + es.y * c[4] + es.z * c[7],
                  es.x * c[2] + es.y * c[5] + es.z * c[8]);
}

kernel void csr_count_deg(device const int* send_idx [[buffer(0)]],
                          device atomic_int* deg_send [[buffer(1)]],
                          constant int& n_edges [[buffer(2)]],
                          uint tid [[thread_position_in_grid]]) {
    int e = int(tid);
    if (e < n_edges) atomic_fetch_add_explicit(&deg_send[send_idx[e]], 1, memory_order_relaxed);
}

kernel void csr_scatter_send(device const int* send_idx [[buffer(0)]],
                             device atomic_int* cur_send [[buffer(1)]],
                             device int* send_perm [[buffer(2)]],
                             constant int& n_edges [[buffer(3)]],
                             uint tid [[thread_position_in_grid]]) {
    int e = int(tid);
    if (e < n_edges) {
        int dst = atomic_fetch_add_explicit(&cur_send[send_idx[e]], 1, memory_order_relaxed);
        send_perm[dst] = e;
    }
}

template <bool LATER>
inline void run_csr_fwd(
    device const float* pos, device const int* recv_ptr, device const int* send_idx,
    device const float* shift_in, device const float* cell, device const long* graph,
    device const long* z, device const float* rmin_tab, device const float* rho,
    device const float* poly, device const float* gates, device const float* qn,
    device const float* kn, device const float* vec, device const float* ten,
    device float* out_num, device float* out_zc, device float* out_dsum,
    int n_atoms, int grid, int flags, float r_cut, float r_on, float max_doff_r2,
    uint lane, int i
) {
    if (i >= n_atoms) return;
    constexpr int TD = LATER ? 284 : 188, OD = LATER ? 408 : 360;
    constexpr int F_O_OFF = LATER ? 276 : 180, OUT_T_OFF = LATER ? 256 : 224, OUT_O_OFF = LATER ? 352 : 304;
    bool has_floor = (flags & FLAG_FLOOR) != 0, has_rho = (flags & FLAG_RHO) != 0;
    bool has_ten = LATER && ((flags & FLAG_TEN) != 0);
    int g_id = ((flags & FLAG_CELL) != 0 && (flags & FLAG_G1) == 0) ? int(graph[i]) : 0;
    int zi_shift = has_floor ? (int(clamp(z[i], 0L, 127L)) << 7) : 0;
    float rho_i = has_rho ? rho[i] : 0.0f;
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float q0 = qn[64 * i + int(lane)], q1 = qn[64 * i + 32 + int(lane)];
    float r_cut2 = r_cut * r_cut, inv_w_env = 1.0f / (r_cut - r_on), inv_h = float(grid) / r_cut;

    float acc_s[4] = {0.0f, 0.0f, 0.0f, 0.0f}, acc_v[3] = {0.0f, 0.0f, 0.0f};
    float acc_lon = 0.0f, acc_lon2 = 0.0f, acc_zc = 0.0f, acc_dsum = 0.0f;
    float acc_t[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, acc_o[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int e = recv_ptr[i], e_end = recv_ptr[i + 1]; e < e_end; ++e) {
        int j = send_idx[e];
        float3 d = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]) - pi + load_shift(shift_in, cell, e, g_id, flags);
        float r2 = dot(d, d);
        if (r2 < 1e-24f || r2 >= r_cut2) continue;
        float r_raw = sqrt(r2), r_fl = r_raw;
        float3 u = d * (1.0f / r_raw);
        if (has_floor && r2 < max_doff_r2) {
            float r_min = rmin_tab[zi_shift + int(clamp(z[j], 0L, 127L))];
            float d_off = r_min + 0.05f, r_eps = sqrt(r2 + 1e-8f);
            if (r_eps < d_off && r_min > 0.0f) r_fl = d_off - 0.05f * tanh((d_off - r_eps) * 20.0f);
        }
        if (r_fl >= r_cut) continue;
        float chi = 1.0f;
        if (r_fl > r_on) {
            float te = (r_fl - r_on) * inv_w_env, t2 = te * te;
            chi = 1.0f - (t2 * t2) * fma(te, fma(te, fma(-20.0f, te, 70.0f), -84.0f), 35.0f);
        }
        float x = r_fl;
        if (has_rho) {
            float sw = 1.52f / (rho_i + rho[j]);
            x = (sw * r_fl) / (1.0f + (sw - 1.0f) * (r_fl / r_cut));
        }
        float u_sp = clamp(x, 0.0f, r_cut) * inv_h;
        int idx = min(max(int(floor(u_sp)), 0), grid - 1);
        float tsp = u_sp - float(idx);
        device const float* p0 = poly + (idx * 4) * TD;

        float d0, d1, d2, d3, w_l = 0.0f;
        qk_dots4(q0, q1, kn[64 * j + int(lane)], kn[64 * j + 32 + int(lane)], d0, d1, d2, d3);
        if (lane < 4) {
            float dl = (lane == 0) ? d0 : ((lane == 1) ? d1 : ((lane == 2) ? d2 : d3));
            w_l = chi * exp(dl + 2.0f * tanh(0.5f * horner_val(p0, TD, int(lane), tsp)));
            acc_zc += w_l;
        }
        if (lane == 0) acc_dsum += chi;
        float ws[4] = {simd_broadcast(w_l, ushort(0)), simd_broadcast(w_l, ushort(1)),
                       simd_broadcast(w_l, ushort(2)), simd_broadcast(w_l, ushort(3))};

        device const float* gj = gates + 184 * j;
        for (int m = 0; m < 4; ++m) {
            int col = 32 * m + int(lane);
            acc_s[m] += ws[m] * horner_val(p0, TD, 4 + col, tsp) * gj[col];
        }
        float wv = ws[lane >> 3], w_dir = horner_val(p0, TD, 132 + int(lane), tsp) * gj[128 + int(lane)];
        if (LATER) {
            device const float* bj = vec + 96 * j + 3 * int(lane);
            float b0 = bj[0], b1 = bj[1], b2 = bj[2];
            float st = wv * horner_val(p0, TD, 180 + int(lane), tsp);
            float w_lon = horner_val(p0, TD, 212 + int(lane), tsp) * (b0 * u.x + b1 * u.y + b2 * u.z);
            float sc = wv * (w_dir + w_lon);
            acc_v[0] += sc * u.x + st * b0; acc_v[1] += sc * u.y + st * b1; acc_v[2] += sc * u.z + st * b2;
            acc_lon += wv * w_lon;
        } else {
            float sc = wv * w_dir;
            acc_v[0] += sc * u.x; acc_v[1] += sc * u.y; acc_v[2] += sc * u.z;
        }
        float y2[5], y3[7];
        eval_y2(u, y2); eval_y3(u, y3);
        if (lane < 16) {
            float wt = ws[lane >> 2], w_ten = horner_val(p0, TD, 164 + int(lane), tsp) * gj[160 + int(lane)];
            if (has_ten) {
                device const float* tj = ten + 80 * j + 5 * int(lane);
                float dot_ty2 = tj[0]*y2[0] + tj[1]*y2[1] + tj[2]*y2[2] + tj[3]*y2[3] + tj[4]*y2[4];
                float st = wt * horner_val(p0, TD, 244 + int(lane), tsp);
                float w_lon2 = horner_val(p0, TD, 260 + int(lane), tsp) * dot_ty2;
                float sc = wt * (w_ten + w_lon2);
                for (int m = 0; m < 5; ++m) acc_t[m] += sc * y2[m] + st * tj[m];
                acc_lon2 += wt * w_lon2;
            } else {
                float sc = wt * w_ten;
                for (int m = 0; m < 5; ++m) acc_t[m] += sc * y2[m];
            }
        }
        if (lane < 8) {
            float sc = ws[lane >> 1] * horner_val(p0, TD, F_O_OFF + int(lane), tsp) * gj[176 + int(lane)];
            for (int m = 0; m < 7; ++m) acc_o[m] += sc * y3[m];
        }
    }

    device float* oi = out_num + OD * i;
    for (int m = 0; m < 4; ++m) oi[32 * m + int(lane)] = acc_s[m];
    oi[128 + 3 * int(lane) + 0] = acc_v[0]; oi[128 + 3 * int(lane) + 1] = acc_v[1]; oi[128 + 3 * int(lane) + 2] = acc_v[2];
    if (LATER) oi[224 + int(lane)] = acc_lon;
    if (lane < 16) {
        for (int m = 0; m < 5; ++m) oi[OUT_T_OFF + 5 * int(lane) + m] = acc_t[m];
        if (LATER) oi[336 + int(lane)] = acc_lon2;
    }
    if (lane < 8) { for (int m = 0; m < 7; ++m) oi[OUT_O_OFF + 7 * int(lane) + m] = acc_o[m]; }
    if (lane < 4) out_zc[4 * i + int(lane)] = acc_zc;
    if (lane == 0) out_dsum[i] = acc_dsum;
}

#define DECL_FWD(NAME, IS_LATER) \
kernel void NAME( \
    device const float* pos [[buffer(0)]], device const int* recv_ptr [[buffer(1)]], \
    device const int* send_idx [[buffer(2)]], device const float* shift_in [[buffer(3)]], \
    device const float* cell [[buffer(4)]], device const long* graph [[buffer(5)]], \
    device const long* z [[buffer(6)]], device const float* rmin_tab [[buffer(7)]], \
    device const float* rho [[buffer(8)]], device const float* poly [[buffer(9)]], \
    device const float* gates [[buffer(10)]], device const float* qn [[buffer(11)]], \
    device const float* kn [[buffer(12)]], device const float* vec [[buffer(13)]], \
    device const float* ten [[buffer(14)]], device float* out_num [[buffer(15)]], \
    device float* out_zc [[buffer(16)]], device float* out_dsum [[buffer(17)]], \
    constant int& n_atoms [[buffer(18)]], constant int& grid [[buffer(19)]], \
    constant int& flags [[buffer(20)]], constant float& r_cut [[buffer(21)]], \
    constant float& r_on [[buffer(22)]], constant float& max_doff_r2 [[buffer(23)]], \
    uint lane [[thread_index_in_simdgroup]], uint sg_id [[simdgroup_index_in_threadgroup]], \
    uint tg_id [[threadgroup_position_in_grid]]) { \
    run_csr_fwd<IS_LATER>(pos, recv_ptr, send_idx, shift_in, cell, graph, z, rmin_tab, rho, poly, \
        gates, qn, kn, vec, ten, out_num, out_zc, out_dsum, n_atoms, grid, flags, r_cut, r_on, \
        max_doff_r2, lane, int(tg_id * 4 + sg_id)); \
}

DECL_FWD(csr_fwd_first, false)
DECL_FWD(csr_fwd_later, true)
"""

METAL_CSR_SOURCE = _METAL_FWD_SOURCE + "\n" + METAL_BWD_SOURCE
