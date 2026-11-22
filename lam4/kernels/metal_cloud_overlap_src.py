"""Metal Shading Language (MSL) Dual-CSR kernels for CloudOverlapHead (zero [E, C] edge buffers)."""

METAL_CLOUD_OV_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant int OV_FLAG_SHIFT = 1;
constant int OV_FLAG_CELL  = 2;
constant int OV_FLAG_G1    = 4;
constant int OV_FLAG_VIR   = 8;
constant int OV_FLAG_DPOS  = 16;

constant int OV_C  = 44;
constant int OV_KE = 16;
constant int OV_KA = 8;
constant int OV_KT = 4;

constant float OV_S2     = 1.4142135623730951f;
constant float OV_INV_S2 = 0.7071067811865475f;
constant float OV_INV_S6 = 0.4082482904638630f;

inline void ov_eval_y2(float3 u, thread float (&y2)[5]) {
    float x = u.x, y = u.y, z = u.z;
    y2[0] = OV_S2 * x * y;
    y2[1] = OV_S2 * x * z;
    y2[2] = OV_S2 * y * z;
    y2[3] = (x * x - y * y) * OV_INV_S2;
    y2[4] = (2.0f * z * z - x * x - y * y) * OV_INV_S6;
}

inline float3 ov_grad_y2_u(float3 u, thread const float (&g)[5]) {
    float x = u.x, y = u.y, z = u.z, k6 = 2.0f * OV_INV_S6;
    float gx = OV_S2 * (g[0] * y + g[1] * z + g[3] * x) - k6 * g[4] * x;
    float gy = OV_S2 * (g[0] * x + g[2] * z - g[3] * y) - k6 * g[4] * y;
    float gz = OV_S2 * (g[1] * x + g[2] * y) + 2.0f * k6 * g[4] * z;
    return float3(gx, gy, gz);
}

inline void ov_c3_env(float r, float r_on, float inv_w, thread float& chi, thread float& dchi) {
    if (r <= r_on) {
        chi = 1.0f;
        dchi = 0.0f;
    } else {
        float t = clamp((r - r_on) * inv_w, 0.0f, 1.0f);
        float om = 1.0f - t;
        float t2 = t * t, t3 = t2 * t;
        chi = 1.0f - t3 * t * (35.0f - 84.0f * t + 70.0f * t2 - 20.0f * t3);
        dchi = -140.0f * t3 * (om * om * om) * inv_w;
    }
}

inline float ov_horner_val(device const float* p0, int col, float t) {
    float a0 = p0[col], a1 = p0[OV_C + col], a2 = p0[2 * OV_C + col], a3 = p0[3 * OV_C + col];
    return fma(fma(fma(a3, t, a2), t, a1), t, a0);
}

inline void ov_horner_val_der(device const float* p0, int col, float t, float inv_h, thread float& fv, thread float& df) {
    float a0 = p0[col], a1 = p0[OV_C + col], a2 = p0[2 * OV_C + col], a3 = p0[3 * OV_C + col];
    fv = fma(fma(fma(a3, t, a2), t, a1), t, a0);
    df = fma(fma(3.0f * a3, t, 2.0f * a2), t, a1) * inv_h;
}

inline float3 ov_load_shift(device const float* shift_in, device const float* cell, int e, int g_id, int flags) {
    if ((flags & OV_FLAG_SHIFT) == 0) return float3(0.0f);
    float3 es = float3(shift_in[3 * e], shift_in[3 * e + 1], shift_in[3 * e + 2]);
    if ((flags & OV_FLAG_CELL) == 0) return es;
    device const float* c = cell + 9 * g_id;
    return float3(
        es.x * c[0] + es.y * c[3] + es.z * c[6],
        es.x * c[1] + es.y * c[4] + es.z * c[7],
        es.x * c[2] + es.y * c[5] + es.z * c[8]
    );
}

kernel void cloud_ov_fwd_recv(
    device const float* pos [[buffer(0)]],
    device const int* recv_ptr [[buffer(1)]],
    device const int* send_idx [[buffer(2)]],
    device const uchar* atom_mask [[buffer(3)]],
    device const float* shift_in [[buffer(4)]],
    device const float* cell [[buffer(5)]],
    device const long* graph [[buffer(6)]],
    device const float* poly [[buffer(7)]],
    device const float* a_p [[buffer(8)]],
    device const float* c1 [[buffer(9)]],
    device const float* d1 [[buffer(10)]],
    device const float* c2 [[buffer(11)]],
    device const float* d2 [[buffer(12)]],
    device float* out_ap [[buffer(13)]],
    device float* out_ds [[buffer(14)]],
    device float* out_v [[buffer(15)]],
    device float* out_ys [[buffer(16)]],
    device float* out_yt [[buffer(17)]],
    constant int& n_atoms [[buffer(18)]],
    constant int& grid_flags [[buffer(19)]],
    constant float4& params [[buffer(20)]], // (r_cut, r_on, inv_w, inv_h)
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid >> 5);
    int lane = int(tid & 31);
    if (i >= n_atoms) return;
    if (atom_mask[i] == 0) {
        if (lane < OV_KE) out_ap[OV_KE * i + lane] = 0.0f;
        if (lane < OV_KA) out_ds[OV_KA * i + lane] = 0.0f;
        if (lane < 24)    out_v[24 * i + lane] = 0.0f;
        if (lane < OV_KT) out_ys[OV_KT * i + lane] = 0.0f;
        if (lane < 20)    out_yt[20 * i + lane] = 0.0f;
        return;
    }

    int grid = grid_flags >> 8;
    int flags = grid_flags & 255;
    int g_id = ((flags & OV_FLAG_CELL) != 0 && (flags & OV_FLAG_G1) == 0) ? int(graph[i]) : 0;
    float r_cut = params.x, r_on = params.y, inv_w = params.z, inv_h = params.w;
    float r_cut2 = r_cut * r_cut;
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);

    float acc_p[OV_KE];   for (int k = 0; k < OV_KE; ++k) acc_p[k] = 0.0f;
    float acc_ds[OV_KA];  for (int k = 0; k < OV_KA; ++k) acc_ds[k] = 0.0f;
    float3 acc_v[OV_KA];  for (int k = 0; k < OV_KA; ++k) acc_v[k] = float3(0.0f);
    float acc_ys[OV_KT];  for (int k = 0; k < OV_KT; ++k) acc_ys[k] = 0.0f;
    float acc_yt[OV_KT][5];
    for (int k = 0; k < OV_KT; ++k) for (int m = 0; m < 5; ++m) acc_yt[k][m] = 0.0f;

    int e0 = recv_ptr[i], e1 = recv_ptr[i + 1];
    for (int e = e0 + lane; e < e1; e += 32) {
        int j = send_idx[e];
        if (atom_mask[j] == 0) continue;
        float3 d = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]) - pi + ov_load_shift(shift_in, cell, e, g_id, flags);
        float r2 = dot(d, d);
        if (r2 <= 1e-12f || r2 >= r_cut2) continue;
        float r = sqrt(r2 + 1e-12f);
        float3 u = d * (1.0f / r);
        float chi, dchi;
        ov_c3_env(r, r_on, inv_w, chi, dchi);
        float xf = min(r, r_cut) * inv_h;
        int g = min(int(floor(xf)), grid - 1);
        float t = xf - float(g);
        device const float* p0 = poly + (4 * g) * OV_C;

        device const float* apj = a_p + OV_KE * j;
        for (int k = 0; k < OV_KE; ++k) {
            acc_p[k] = fma(chi * ov_horner_val(p0, k, t), apj[k], acc_p[k]);
        }
        device const float* c1j = c1 + 24 * j;
        device const float* d1j = d1 + OV_KA * j;
        for (int k = 0; k < OV_KA; ++k) {
            float3 c1k = float3(c1j[3 * k], c1j[3 * k + 1], c1j[3 * k + 2]);
            float lon_dot = dot(c1k, u);
            float fdir = chi * ov_horner_val(p0, OV_KE + k, t);
            float ftr  = chi * ov_horner_val(p0, OV_KE + OV_KA + k, t);
            float flon = chi * ov_horner_val(p0, OV_KE + 2 * OV_KA + k, t);
            acc_ds[k] = fma(fdir, lon_dot, acc_ds[k]);
            float coef_u = fdir * d1j[k] + flon * lon_dot;
            acc_v[k] += coef_u * u + ftr * c1k;
        }
        float y2[5];
        ov_eval_y2(u, y2);
        device const float* c2j = c2 + 20 * j;
        device const float* d2j = d2 + OV_KT * j;
        for (int k = 0; k < OV_KT; ++k) {
            float fy2 = chi * ov_horner_val(p0, OV_KE + 3 * OV_KA + k, t);
            float dot_y2 = 0.0f;
            float wy = fy2 * d2j[k];
            for (int m = 0; m < 5; ++m) {
                dot_y2 = fma(c2j[5 * k + m], y2[m], dot_y2);
                acc_yt[k][m] = fma(wy, y2[m], acc_yt[k][m]);
            }
            acc_ys[k] = fma(fy2, dot_y2, acc_ys[k]);
        }
    }

    for (int k = 0; k < OV_KE; ++k) {
        float s = simd_sum(acc_p[k]);
        if (lane == 0) out_ap[OV_KE * i + k] = s;
    }
    for (int k = 0; k < OV_KA; ++k) {
        float sds = simd_sum(acc_ds[k]);
        float svx = simd_sum(acc_v[k].x), svy = simd_sum(acc_v[k].y), svz = simd_sum(acc_v[k].z);
        if (lane == 0) {
            out_ds[OV_KA * i + k] = sds;
            out_v[24 * i + 3 * k] = svx; out_v[24 * i + 3 * k + 1] = svy; out_v[24 * i + 3 * k + 2] = svz;
        }
    }
    for (int k = 0; k < OV_KT; ++k) {
        float sys = simd_sum(acc_ys[k]);
        if (lane == 0) out_ys[OV_KT * i + k] = sys;
        for (int m = 0; m < 5; ++m) {
            float syt = simd_sum(acc_yt[k][m]);
            if (lane == 0) out_yt[20 * i + 5 * k + m] = syt;
        }
    }
}

inline float3 ov_edge_grad_d(
    float3 d, float r_cut, float r_on, float inv_w, float inv_h, int grid,
    device const float* poly,
    device const float* apj, device const float* c1j, device const float* d1j,
    device const float* c2j, device const float* d2j,
    device const float* gpi, device const float* gdsi, device const float* gvi,
    device const float* gysi, device const float* gyti
) {
    float r2 = dot(d, d);
    float r = sqrt(r2 + 1e-12f);
    float inv_r = 1.0f / r;
    float3 u = d * inv_r;
    float chi, dchi;
    ov_c3_env(r, r_on, inv_w, chi, dchi);
    float xf = min(r, r_cut) * inv_h;
    int g = min(int(floor(xf)), grid - 1);
    float t = xf - float(g);
    device const float* p0 = poly + (4 * g) * OV_C;

    float gr = 0.0f;
    float3 gu = float3(0.0f);
    for (int k = 0; k < OV_KE; ++k) {
        float s, ds;
        ov_horner_val_der(p0, k, t, inv_h, s, ds);
        float dfp = dchi * s + chi * ds;
        gr = fma(dfp, gpi[k] * apj[k], gr);
    }
    for (int k = 0; k < OV_KA; ++k) {
        float3 c1k = float3(c1j[3 * k], c1j[3 * k + 1], c1j[3 * k + 2]);
        float3 gvk = float3(gvi[3 * k], gvi[3 * k + 1], gvi[3 * k + 2]);
        float cu = dot(c1k, u), gvu = dot(gvk, u);
        float gdsk = gdsi[k], d1jk = d1j[k];
        float s_dir, ds_dir, s_tr, ds_tr, s_lon, ds_lon;
        ov_horner_val_der(p0, OV_KE + k, t, inv_h, s_dir, ds_dir);
        ov_horner_val_der(p0, OV_KE + OV_KA + k, t, inv_h, s_tr, ds_tr);
        ov_horner_val_der(p0, OV_KE + 2 * OV_KA + k, t, inv_h, s_lon, ds_lon);
        float fdir = chi * s_dir, dfdir = dchi * s_dir + chi * ds_dir;
        float dftr = dchi * s_tr + chi * ds_tr;
        float flon = chi * s_lon, dflon = dchi * s_lon + chi * ds_lon;
        gr = fma(dfdir, gdsk * cu + d1jk * gvu, gr);
        gr = fma(dftr, dot(gvk, c1k), gr);
        gr = fma(dflon, cu * gvu, gr);
        gu += (fdir * gdsk + flon * gvu) * c1k + (fdir * d1jk + flon * cu) * gvk;
    }
    float y2[5];
    ov_eval_y2(u, y2);
    float wy2[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < OV_KT; ++k) {
        float sy, dsy;
        ov_horner_val_der(p0, OV_KE + 3 * OV_KA + k, t, inv_h, sy, dsy);
        float fy2 = chi * sy, dfy2 = dchi * sy + chi * dsy;
        float gysk = gysi[k], d2jk = d2j[k];
        float c2y = 0.0f, gyty = 0.0f;
        for (int m = 0; m < 5; ++m) {
            float c2m = c2j[5 * k + m], gytm = gyti[5 * k + m];
            c2y = fma(c2m, y2[m], c2y);
            gyty = fma(gytm, y2[m], gyty);
            wy2[m] = fma(fy2, gysk * c2m + d2jk * gytm, wy2[m]);
        }
        gr = fma(dfy2, gysk * c2y + d2jk * gyty, gr);
    }
    gu += ov_grad_y2_u(u, wy2);
    return (gu - dot(gu, u) * u) * inv_r + gr * u;
}

kernel void cloud_ov_bwd_recv(
    device const float* pos [[buffer(0)]],
    device const int* recv_ptr [[buffer(1)]],
    device const int* send_idx [[buffer(2)]],
    device const uchar* atom_mask [[buffer(3)]],
    device const float* shift_in [[buffer(4)]],
    device const float* cell [[buffer(5)]],
    device const long* graph [[buffer(6)]],
    device const float* poly [[buffer(7)]],
    device const float* a_p [[buffer(8)]],
    device const float* c1 [[buffer(9)]],
    device const float* d1 [[buffer(10)]],
    device const float* c2 [[buffer(11)]],
    device const float* d2 [[buffer(12)]],
    device const float* gp [[buffer(13)]],
    device const float* gds [[buffer(14)]],
    device const float* gv [[buffer(15)]],
    device const float* gys [[buffer(16)]],
    device const float* gyt [[buffer(17)]],
    device float* d_pos_recv [[buffer(18)]],
    device float* vir_atom [[buffer(19)]],
    constant int& n_atoms [[buffer(20)]],
    constant int& grid_flags [[buffer(21)]],
    constant float4& params [[buffer(22)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid >> 5);
    int lane = int(tid & 31);
    if (i >= n_atoms) return;
    int grid = grid_flags >> 8;
    int flags = grid_flags & 255;
    bool need_vir = (flags & OV_FLAG_VIR) != 0;
    bool need_dpos = (flags & OV_FLAG_DPOS) != 0;
    if (atom_mask[i] == 0) {
        if (need_dpos && lane < 3) d_pos_recv[3 * i + lane] = 0.0f;
        if (need_vir && lane < 9) vir_atom[9 * i + lane] = 0.0f;
        return;
    }

    int g_id = ((flags & OV_FLAG_CELL) != 0 && (flags & OV_FLAG_G1) == 0) ? int(graph[i]) : 0;
    float r_cut = params.x, r_on = params.y, inv_w = params.z, inv_h = params.w;
    float r_cut2 = r_cut * r_cut;
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float3 dpos_i = float3(0.0f);
    float vir[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    device const float* gpi = gp + OV_KE * i;
    device const float* gdsi = gds + OV_KA * i;
    device const float* gvi = gv + 24 * i;
    device const float* gysi = gys + OV_KT * i;
    device const float* gyti = gyt + 20 * i;

    int e0 = recv_ptr[i], e1 = recv_ptr[i + 1];
    for (int e = e0 + lane; e < e1; e += 32) {
        int j = send_idx[e];
        if (atom_mask[j] == 0) continue;
        float3 d = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]) - pi + ov_load_shift(shift_in, cell, e, g_id, flags);
        float r2 = dot(d, d);
        if (r2 <= 1e-12f || r2 >= r_cut2) continue;
        float3 gd = ov_edge_grad_d(
            d, r_cut, r_on, inv_w, inv_h, grid, poly,
            a_p + OV_KE * j, c1 + 24 * j, d1 + OV_KA * j, c2 + 20 * j, d2 + OV_KT * j,
            gpi, gdsi, gvi, gysi, gyti
        );
        if (need_dpos) dpos_i -= gd;
        if (need_vir) {
            float sx = shift_in[3 * e], sy = shift_in[3 * e + 1], sz = shift_in[3 * e + 2];
            vir[0] += sx * gd.x; vir[1] += sx * gd.y; vir[2] += sx * gd.z;
            vir[3] += sy * gd.x; vir[4] += sy * gd.y; vir[5] += sy * gd.z;
            vir[6] += sz * gd.x; vir[7] += sz * gd.y; vir[8] += sz * gd.z;
        }
    }
    if (need_dpos) {
        float dx = simd_sum(dpos_i.x), dy = simd_sum(dpos_i.y), dz = simd_sum(dpos_i.z);
        if (lane == 0) { d_pos_recv[3 * i] = dx; d_pos_recv[3 * i + 1] = dy; d_pos_recv[3 * i + 2] = dz; }
    }
    if (need_vir) {
        for (int m = 0; m < 9; ++m) {
            float sv = simd_sum(vir[m]);
            if (lane == 0) vir_atom[9 * i + m] = sv;
        }
    }
}

kernel void cloud_ov_bwd_send(
    device const float* pos [[buffer(0)]],
    device const int* send_ptr [[buffer(1)]],
    device const int* send_perm [[buffer(2)]],
    device const int* recv_idx [[buffer(3)]],
    device const uchar* atom_mask [[buffer(4)]],
    device const float* shift_in [[buffer(5)]],
    device const float* cell [[buffer(6)]],
    device const long* graph [[buffer(7)]],
    device const float* poly [[buffer(8)]],
    device const float* a_p [[buffer(9)]],
    device const float* c1 [[buffer(10)]],
    device const float* d1 [[buffer(11)]],
    device const float* c2 [[buffer(12)]],
    device const float* d2 [[buffer(13)]],
    device const float* gp [[buffer(14)]],
    device const float* gds [[buffer(15)]],
    device const float* gv [[buffer(16)]],
    device const float* gys [[buffer(17)]],
    device const float* gyt [[buffer(18)]],
    device float* d_pos [[buffer(19)]],
    device float* d_ap [[buffer(20)]],
    device float* d_c1 [[buffer(21)]],
    device float* d_d1 [[buffer(22)]],
    device float* d_c2 [[buffer(23)]],
    device float* d_d2 [[buffer(24)]],
    constant int& n_atoms [[buffer(25)]],
    constant int& grid_flags [[buffer(26)]],
    constant float4& params [[buffer(27)]],
    uint tid [[thread_position_in_grid]]
) {
    int j = int(tid >> 5);
    int lane = int(tid & 31);
    if (j >= n_atoms) return;
    int grid = grid_flags >> 8;
    int flags = grid_flags & 255;
    bool need_dpos = (flags & OV_FLAG_DPOS) != 0;
    if (atom_mask[j] == 0) {
        if (lane < OV_KE) d_ap[OV_KE * j + lane] = 0.0f;
        if (lane < 24)    d_c1[24 * j + lane] = 0.0f;
        if (lane < OV_KA) d_d1[OV_KA * j + lane] = 0.0f;
        if (lane < 20)    d_c2[20 * j + lane] = 0.0f;
        if (lane < OV_KT) d_d2[OV_KT * j + lane] = 0.0f;
        return;
    }

    int g_id = ((flags & OV_FLAG_CELL) != 0 && (flags & OV_FLAG_G1) == 0) ? int(graph[j]) : 0;
    float r_cut = params.x, r_on = params.y, inv_w = params.z, inv_h = params.w;
    float r_cut2 = r_cut * r_cut;
    float3 pj = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);

    float3 dpos_j = float3(0.0f);
    float dap[OV_KE];   for (int k = 0; k < OV_KE; ++k) dap[k] = 0.0f;
    float3 dc1[OV_KA];  for (int k = 0; k < OV_KA; ++k) dc1[k] = float3(0.0f);
    float dd1[OV_KA];   for (int k = 0; k < OV_KA; ++k) dd1[k] = 0.0f;
    float dc2[OV_KT][5];
    for (int k = 0; k < OV_KT; ++k) for (int m = 0; m < 5; ++m) dc2[k][m] = 0.0f;
    float dd2[OV_KT];   for (int k = 0; k < OV_KT; ++k) dd2[k] = 0.0f;

    device const float* apj = a_p + OV_KE * j;
    device const float* c1j = c1 + 24 * j;
    device const float* d1j = d1 + OV_KA * j;
    device const float* c2j = c2 + 20 * j;
    device const float* d2j = d2 + OV_KT * j;

    int k0 = send_ptr[j], k1 = send_ptr[j + 1];
    for (int kp = k0 + lane; kp < k1; kp += 32) {
        int e = send_perm[kp];
        int i = recv_idx[e];
        if (atom_mask[i] == 0) continue;
        float3 d = pj - float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) + ov_load_shift(shift_in, cell, e, g_id, flags);
        float r2 = dot(d, d);
        if (r2 <= 1e-12f || r2 >= r_cut2) continue;
        float r = sqrt(r2 + 1e-12f);
        float3 u = d * (1.0f / r);
        float chi, dchi;
        ov_c3_env(r, r_on, inv_w, chi, dchi);
        float xf = min(r, r_cut) * inv_h;
        int g = min(int(floor(xf)), grid - 1);
        float t = xf - float(g);
        device const float* p0 = poly + (4 * g) * OV_C;

        device const float* gpi = gp + OV_KE * i;
        device const float* gdsi = gds + OV_KA * i;
        device const float* gvi = gv + 24 * i;
        device const float* gysi = gys + OV_KT * i;
        device const float* gyti = gyt + 20 * i;

        for (int k = 0; k < OV_KE; ++k) {
            dap[k] = fma(chi * ov_horner_val(p0, k, t), gpi[k], dap[k]);
        }
        for (int k = 0; k < OV_KA; ++k) {
            float3 gvk = float3(gvi[3 * k], gvi[3 * k + 1], gvi[3 * k + 2]);
            float fdir = chi * ov_horner_val(p0, OV_KE + k, t);
            float ftr  = chi * ov_horner_val(p0, OV_KE + OV_KA + k, t);
            float flon = chi * ov_horner_val(p0, OV_KE + 2 * OV_KA + k, t);
            float gvu = dot(gvk, u);
            dc1[k] += (fdir * gdsi[k] + flon * gvu) * u + ftr * gvk;
            dd1[k] = fma(fdir, gvu, dd1[k]);
        }
        float y2[5];
        ov_eval_y2(u, y2);
        for (int k = 0; k < OV_KT; ++k) {
            float fy2 = chi * ov_horner_val(p0, OV_KE + 3 * OV_KA + k, t);
            float w_c = fy2 * gysi[k];
            float gyty = 0.0f;
            for (int m = 0; m < 5; ++m) {
                dc2[k][m] = fma(w_c, y2[m], dc2[k][m]);
                gyty = fma(gyti[5 * k + m], y2[m], gyty);
            }
            dd2[k] = fma(fy2, gyty, dd2[k]);
        }
        if (need_dpos) {
            dpos_j += ov_edge_grad_d(
                d, r_cut, r_on, inv_w, inv_h, grid, poly,
                apj, c1j, d1j, c2j, d2j, gpi, gdsi, gvi, gysi, gyti
            );
        }
    }
    if (need_dpos) {
        float dx = simd_sum(dpos_j.x), dy = simd_sum(dpos_j.y), dz = simd_sum(dpos_j.z);
        if (lane == 0) { d_pos[3 * j] += dx; d_pos[3 * j + 1] += dy; d_pos[3 * j + 2] += dz; }
    }
    for (int k = 0; k < OV_KE; ++k) {
        float s = simd_sum(dap[k]);
        if (lane == 0) d_ap[OV_KE * j + k] = s;
    }
    for (int k = 0; k < OV_KA; ++k) {
        float sd = simd_sum(dd1[k]);
        float sx = simd_sum(dc1[k].x), sy = simd_sum(dc1[k].y), sz = simd_sum(dc1[k].z);
        if (lane == 0) {
            d_d1[OV_KA * j + k] = sd;
            d_c1[24 * j + 3 * k] = sx; d_c1[24 * j + 3 * k + 1] = sy; d_c1[24 * j + 3 * k + 2] = sz;
        }
    }
    for (int k = 0; k < OV_KT; ++k) {
        float sd = simd_sum(dd2[k]);
        if (lane == 0) d_d2[OV_KT * j + k] = sd;
        for (int m = 0; m < 5; ++m) {
            float sc = simd_sum(dc2[k][m]);
            if (lane == 0) d_c2[20 * j + 5 * k + m] = sc;
        }
    }
}
"""
