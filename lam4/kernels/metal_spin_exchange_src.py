"""Metal Shading Language (MSL) Dual-CSR kernels for SpinExchangeHead (zero [E, C] edge buffers)."""

METAL_SPIN_EXCH_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant int SE_FLAG_SHIFT = 1;
constant int SE_FLAG_CELL  = 2;
constant int SE_FLAG_G1    = 4;
constant int SE_FLAG_VIR   = 8;
constant int SE_FLAG_DPOS  = 16;

constant int SE_RANK = 16;
constant int SE_NRBF = 32;
constant float SE_PI = 3.14159265358979323846f;

inline float3 se_load_shift(device const float* shift_in, device const float* cell, int e, int g_id, int flags) {
    if ((flags & SE_FLAG_SHIFT) == 0) return float3(0.0f);
    float3 es = float3(shift_in[3 * e], shift_in[3 * e + 1], shift_in[3 * e + 2]);
    if ((flags & SE_FLAG_CELL) == 0) return es;
    device const float* c = cell + 9 * g_id;
    return float3(
        es.x * c[0] + es.y * c[3] + es.z * c[6],
        es.x * c[1] + es.y * c[4] + es.z * c[7],
        es.x * c[2] + es.y * c[5] + es.z * c[8]
    );
}

kernel void spin_exch_fwd_recv(
    device const float* pos [[buffer(0)]],
    device const int* recv_ptr [[buffer(1)]],
    device const int* send_idx [[buffer(2)]],
    device const uchar* atom_mask [[buffer(3)]],
    device const float* shift_in [[buffer(4)]],
    device const float* cell [[buffer(5)]],
    device const long* graph [[buffer(6)]],
    device const float* h_dir [[buffer(7)]],
    device const float* h_cov [[buffer(8)]],
    device const float* m_t [[buffer(9)]],
    device const float* u_rad [[buffer(10)]],
    device const float* w_dir [[buffer(11)]],
    device const float* w_cov [[buffer(12)]],
    device const float* centres [[buffer(13)]],
    device float* out_pair [[buffer(14)]],
    constant int& n_atoms [[buffer(15)]],
    constant int& flags [[buffer(16)]],
    constant float4& params [[buffer(17)]], // (r_cut, r_cut2, inv_sigma, 0)
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid >> 5);
    int lane = int(tid & 31);
    if (i >= n_atoms) return;
    if (atom_mask[i] == 0) {
        if (lane == 0) out_pair[i] = 0.0f;
        return;
    }

    int g_id = ((flags & SE_FLAG_CELL) != 0 && (flags & SE_FLAG_G1) == 0) ? int(graph[i]) : 0;
    float r_cut = params.x, r_cut2 = params.y, inv_sig = params.z;
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float mi = m_t[i], ui = u_rad[i];

    float hdi[SE_RANK], hci[SE_RANK];
    device const float* hdi_p = h_dir + SE_RANK * i;
    device const float* hci_p = h_cov + SE_RANK * i;
    for (int k = 0; k < SE_RANK; ++k) {
        hdi[k] = hdi_p[k];
        hci[k] = hci_p[k];
    }

    float acc = 0.0f;
    int e0 = recv_ptr[i], e1 = recv_ptr[i + 1];
    for (int e = e0 + lane; e < e1; e += 32) {
        int j = send_idx[e];
        if (atom_mask[j] == 0) continue;
        float3 d = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]) - pi + se_load_shift(shift_in, cell, e, g_id, flags);
        float r2 = dot(d, d);
        if (r2 <= 1e-12f || r2 >= r_cut2) continue;
        float r = sqrt(r2 + 1e-12f);
        float t = clamp(r / r_cut, 0.0f, 1.0f);
        float chi = 0.5f * (1.0f + cos(SE_PI * t));
        float rc = min(r, r_cut);

        float wd[SE_RANK], wc[SE_RANK];
        for (int k = 0; k < SE_RANK; ++k) { wd[k] = 0.0f; wc[k] = 0.0f; }
        for (int b = 0; b < SE_NRBF; ++b) {
            float zb = (rc - centres[b]) * inv_sig;
            if (fabs(zb) >= 9.0f) continue;
            float pb = exp(-0.5f * zb * zb);
            for (int k = 0; k < SE_RANK; ++k) {
                wd[k] = fma(pb, w_dir[k * SE_NRBF + b], wd[k]);
                wc[k] = fma(pb, w_cov[k * SE_NRBF + b], wc[k]);
            }
        }

        device const float* hdj_p = h_dir + SE_RANK * j;
        device const float* hcj_p = h_cov + SE_RANK * j;
        float j_dir = 0.0f, j_cov = 0.0f;
        for (int k = 0; k < SE_RANK; ++k) {
            j_dir = fma(wd[k], hdi[k] * hdj_p[k], j_dir);
            j_cov = fma(wc[k], hci[k] * hcj_p[k], j_cov);
        }
        float mp = m_t[j] * mi;
        float up = u_rad[j] * ui;
        acc += chi * (j_dir * mp + j_cov * (mp - up));
    }
    float s = simd_sum(acc);
    if (lane == 0) out_pair[i] = s;
}

kernel void spin_exch_bwd_recv(
    device const float* pos [[buffer(0)]],
    device const int* recv_ptr [[buffer(1)]],
    device const int* send_idx [[buffer(2)]],
    device const uchar* atom_mask [[buffer(3)]],
    device const float* shift_in [[buffer(4)]],
    device const float* cell [[buffer(5)]],
    device const long* graph [[buffer(6)]],
    device const float* h_dir [[buffer(7)]],
    device const float* h_cov [[buffer(8)]],
    device const float* m_t [[buffer(9)]],
    device const float* u_rad [[buffer(10)]],
    device const float* w_dir [[buffer(11)]],
    device const float* w_cov [[buffer(12)]],
    device const float* centres [[buffer(13)]],
    device const float* g_out [[buffer(14)]],
    device float* d_pos [[buffer(15)]],
    device float* d_hdir [[buffer(16)]],
    device float* d_hcov [[buffer(17)]],
    device float* d_m [[buffer(18)]],
    device float* d_u [[buffer(19)]],
    device float* vir_atom [[buffer(20)]],
    constant int& n_atoms [[buffer(21)]],
    constant int& flags [[buffer(22)]],
    constant float4& params [[buffer(23)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = int(tid >> 5);
    int lane = int(tid & 31);
    if (i >= n_atoms) return;
    bool need_vir = (flags & SE_FLAG_VIR) != 0;
    bool need_dpos = (flags & SE_FLAG_DPOS) != 0;
    if (atom_mask[i] == 0) {
        if (need_vir && lane < 9) vir_atom[9 * i + lane] = 0.0f;
        return;
    }

    int g_id = ((flags & SE_FLAG_CELL) != 0 && (flags & SE_FLAG_G1) == 0) ? int(graph[i]) : 0;
    float r_cut = params.x, r_cut2 = params.y, inv_sig = params.z;
    float gi = g_out[i];
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float mi = m_t[i], ui = u_rad[i];

    float hdi[SE_RANK], hci[SE_RANK];
    device const float* hdi_p = h_dir + SE_RANK * i;
    device const float* hci_p = h_cov + SE_RANK * i;
    for (int k = 0; k < SE_RANK; ++k) {
        hdi[k] = hdi_p[k];
        hci[k] = hci_p[k];
    }

    float3 dp_i = float3(0.0f);
    float dhd_i[SE_RANK], dhc_i[SE_RANK];
    for (int k = 0; k < SE_RANK; ++k) { dhd_i[k] = 0.0f; dhc_i[k] = 0.0f; }
    float dm_i = 0.0f, du_i = 0.0f;
    float vir[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    int e0 = recv_ptr[i], e1 = recv_ptr[i + 1];
    for (int e = e0 + lane; e < e1; e += 32) {
        int j = send_idx[e];
        if (atom_mask[j] == 0) continue;
        float3 d = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]) - pi + se_load_shift(shift_in, cell, e, g_id, flags);
        float r2 = dot(d, d);
        if (r2 <= 1e-12f || r2 >= r_cut2) continue;
        float r = sqrt(r2 + 1e-12f);
        float t = clamp(r / r_cut, 0.0f, 1.0f);
        float chi = 0.5f * (1.0f + cos(SE_PI * t));
        float dchi = (t < 1.0f) ? (-0.5f * SE_PI / r_cut * sin(SE_PI * t)) : 0.0f;
        float rc = min(r, r_cut);
        bool in_cut = (r < r_cut);

        float hdj[SE_RANK], hcj[SE_RANK];
        device const float* hdj_p = h_dir + SE_RANK * j;
        device const float* hcj_p = h_cov + SE_RANK * j;
        for (int k = 0; k < SE_RANK; ++k) { hdj[k] = hdj_p[k]; hcj[k] = hcj_p[k]; }

        float wd[SE_RANK], wc[SE_RANK];
        for (int k = 0; k < SE_RANK; ++k) { wd[k] = 0.0f; wc[k] = 0.0f; }
        float dj_dir_dr = 0.0f, dj_cov_dr = 0.0f;
        for (int b = 0; b < SE_NRBF; ++b) {
            float zb = (rc - centres[b]) * inv_sig;
            if (fabs(zb) >= 9.0f) continue;
            float pb = exp(-0.5f * zb * zb);
            float dpb = in_cut ? (-pb * zb * inv_sig) : 0.0f;
            float sdb = 0.0f, scb = 0.0f;
            for (int k = 0; k < SE_RANK; ++k) {
                float wdk = w_dir[k * SE_NRBF + b], wck = w_cov[k * SE_NRBF + b];
                wd[k] = fma(pb, wdk, wd[k]);
                wc[k] = fma(pb, wck, wc[k]);
                sdb = fma(wdk, hdi[k] * hdj[k], sdb);
                scb = fma(wck, hci[k] * hcj[k], scb);
            }
            dj_dir_dr = fma(dpb, sdb, dj_dir_dr);
            dj_cov_dr = fma(dpb, scb, dj_cov_dr);
        }

        float j_dir = 0.0f, j_cov = 0.0f;
        for (int k = 0; k < SE_RANK; ++k) {
            j_dir = fma(wd[k], hdi[k] * hdj[k], j_dir);
            j_cov = fma(wc[k], hci[k] * hcj[k], j_cov);
        }
        float mj = m_t[j], uj = u_rad[j];
        float mp = mj * mi, up = uj * ui;
        float c_dir = gi * chi * mp;
        float c_cov = gi * chi * (mp - up);

        for (int k = 0; k < SE_RANK; ++k) {
            dhd_i[k] = fma(c_dir * wd[k], hdj[k], dhd_i[k]);
            dhc_i[k] = fma(c_cov * wc[k], hcj[k], dhc_i[k]);
        }
        dm_i += gi * chi * (j_dir + j_cov) * mj;
        du_i += gi * chi * (-j_cov) * uj;

        if (need_dpos || need_vir) {
            float dmsg_dr = dchi * (j_dir * mp + j_cov * (mp - up)) + chi * (dj_dir_dr * mp + dj_cov_dr * (mp - up));
            float3 gd = (gi * dmsg_dr / r) * d;
            if (need_dpos) dp_i -= gd;
            if (need_vir) {
                float sx = shift_in[3 * e], sy = shift_in[3 * e + 1], sz = shift_in[3 * e + 2];
                vir[0] += sx * gd.x; vir[1] += sx * gd.y; vir[2] += sx * gd.z;
                vir[3] += sy * gd.x; vir[4] += sy * gd.y; vir[5] += sy * gd.z;
                vir[6] += sz * gd.x; vir[7] += sz * gd.y; vir[8] += sz * gd.z;
            }
        }
    }

    if (need_dpos) {
        float dx = simd_sum(dp_i.x), dy = simd_sum(dp_i.y), dz = simd_sum(dp_i.z);
        if (lane == 0) { d_pos[3 * i] = dx; d_pos[3 * i + 1] = dy; d_pos[3 * i + 2] = dz; }
    }
    for (int k = 0; k < SE_RANK; ++k) {
        float sd = simd_sum(dhd_i[k]), sc = simd_sum(dhc_i[k]);
        if (lane == 0) { d_hdir[SE_RANK * i + k] = sd; d_hcov[SE_RANK * i + k] = sc; }
    }
    float sm = simd_sum(dm_i), su = simd_sum(du_i);
    if (lane == 0) { d_m[i] = sm; d_u[i] = su; }
    if (need_vir) {
        for (int m = 0; m < 9; ++m) {
            float sv = simd_sum(vir[m]);
            if (lane == 0) vir_atom[9 * i + m] = sv;
        }
    }
}

kernel void spin_exch_bwd_send(
    device const float* pos [[buffer(0)]],
    device const int* send_ptr [[buffer(1)]],
    device const int* send_perm [[buffer(2)]],
    device const int* recv_idx [[buffer(3)]],
    device const uchar* atom_mask [[buffer(4)]],
    device const float* shift_in [[buffer(5)]],
    device const float* cell [[buffer(6)]],
    device const long* graph [[buffer(7)]],
    device const float* h_dir [[buffer(8)]],
    device const float* h_cov [[buffer(9)]],
    device const float* m_t [[buffer(10)]],
    device const float* u_rad [[buffer(11)]],
    device const float* w_dir [[buffer(12)]],
    device const float* w_cov [[buffer(13)]],
    device const float* centres [[buffer(14)]],
    device const float* g_out [[buffer(15)]],
    device float* d_pos [[buffer(16)]],
    device float* d_hdir [[buffer(17)]],
    device float* d_hcov [[buffer(18)]],
    device float* d_m [[buffer(19)]],
    device float* d_u [[buffer(20)]],
    constant int& n_atoms [[buffer(21)]],
    constant int& flags [[buffer(22)]],
    constant float4& params [[buffer(23)]],
    uint tid [[thread_position_in_grid]]
) {
    int j = int(tid >> 5);
    int lane = int(tid & 31);
    if (j >= n_atoms || atom_mask[j] == 0) return;

    float r_cut = params.x, r_cut2 = params.y, inv_sig = params.z;
    bool need_dpos = (flags & SE_FLAG_DPOS) != 0;
    float3 pj = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
    float mj = m_t[j], uj = u_rad[j];

    float hdj[SE_RANK], hcj[SE_RANK];
    device const float* hdj_p = h_dir + SE_RANK * j;
    device const float* hcj_p = h_cov + SE_RANK * j;
    for (int k = 0; k < SE_RANK; ++k) {
        hdj[k] = hdj_p[k];
        hcj[k] = hcj_p[k];
    }

    float3 dp_j = float3(0.0f);
    float dhd_j[SE_RANK], dhc_j[SE_RANK];
    for (int k = 0; k < SE_RANK; ++k) { dhd_j[k] = 0.0f; dhc_j[k] = 0.0f; }
    float dm_j = 0.0f, du_j = 0.0f;

    int k0 = send_ptr[j], k1 = send_ptr[j + 1];
    for (int kp = k0 + lane; kp < k1; kp += 32) {
        int e = send_perm[kp];
        int i = recv_idx[e];
        if (atom_mask[i] == 0) continue;
        int g_id = ((flags & SE_FLAG_CELL) != 0 && (flags & SE_FLAG_G1) == 0) ? int(graph[i]) : 0;
        float3 d = pj - float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) + se_load_shift(shift_in, cell, e, g_id, flags);
        float r2 = dot(d, d);
        if (r2 <= 1e-12f || r2 >= r_cut2) continue;
        float r = sqrt(r2 + 1e-12f);
        float t = clamp(r / r_cut, 0.0f, 1.0f);
        float chi = 0.5f * (1.0f + cos(SE_PI * t));
        float dchi = (t < 1.0f) ? (-0.5f * SE_PI / r_cut * sin(SE_PI * t)) : 0.0f;
        float rc = min(r, r_cut);
        bool in_cut = (r < r_cut);
        float gi = g_out[i];

        float hdi[SE_RANK], hci[SE_RANK];
        device const float* hdi_p = h_dir + SE_RANK * i;
        device const float* hci_p = h_cov + SE_RANK * i;
        for (int k = 0; k < SE_RANK; ++k) { hdi[k] = hdi_p[k]; hci[k] = hci_p[k]; }

        float wd[SE_RANK], wc[SE_RANK];
        for (int k = 0; k < SE_RANK; ++k) { wd[k] = 0.0f; wc[k] = 0.0f; }
        float dj_dir_dr = 0.0f, dj_cov_dr = 0.0f;
        for (int b = 0; b < SE_NRBF; ++b) {
            float zb = (rc - centres[b]) * inv_sig;
            if (fabs(zb) >= 9.0f) continue;
            float pb = exp(-0.5f * zb * zb);
            float dpb = in_cut ? (-pb * zb * inv_sig) : 0.0f;
            float sdb = 0.0f, scb = 0.0f;
            for (int k = 0; k < SE_RANK; ++k) {
                float wdk = w_dir[k * SE_NRBF + b], wck = w_cov[k * SE_NRBF + b];
                wd[k] = fma(pb, wdk, wd[k]);
                wc[k] = fma(pb, wck, wc[k]);
                sdb = fma(wdk, hdi[k] * hdj[k], sdb);
                scb = fma(wck, hci[k] * hcj[k], scb);
            }
            dj_dir_dr = fma(dpb, sdb, dj_dir_dr);
            dj_cov_dr = fma(dpb, scb, dj_cov_dr);
        }

        float j_dir = 0.0f, j_cov = 0.0f;
        for (int k = 0; k < SE_RANK; ++k) {
            j_dir = fma(wd[k], hdi[k] * hdj[k], j_dir);
            j_cov = fma(wc[k], hci[k] * hcj[k], j_cov);
        }
        float mi = m_t[i], ui = u_rad[i];
        float mp = mj * mi, up = uj * ui;
        float c_dir = gi * chi * mp;
        float c_cov = gi * chi * (mp - up);

        for (int k = 0; k < SE_RANK; ++k) {
            dhd_j[k] = fma(c_dir * wd[k], hdi[k], dhd_j[k]);
            dhc_j[k] = fma(c_cov * wc[k], hci[k], dhc_j[k]);
        }
        dm_j += gi * chi * (j_dir + j_cov) * mi;
        du_j += gi * chi * (-j_cov) * ui;

        if (need_dpos) {
            float dmsg_dr = dchi * (j_dir * mp + j_cov * (mp - up)) + chi * (dj_dir_dr * mp + dj_cov_dr * (mp - up));
            dp_j += (gi * dmsg_dr / r) * d;
        }
    }

    if (need_dpos) {
        float dx = simd_sum(dp_j.x), dy = simd_sum(dp_j.y), dz = simd_sum(dp_j.z);
        if (lane == 0) { d_pos[3 * j] += dx; d_pos[3 * j + 1] += dy; d_pos[3 * j + 2] += dz; }
    }
    for (int k = 0; k < SE_RANK; ++k) {
        float sd = simd_sum(dhd_j[k]), sc = simd_sum(dhc_j[k]);
        if (lane == 0) { d_hdir[SE_RANK * j + k] += sd; d_hcov[SE_RANK * j + k] += sc; }
    }
    float sm = simd_sum(dm_j), su = simd_sum(du_j);
    if (lane == 0) { d_m[j] += sm; d_u[j] += su; }
}
"""
