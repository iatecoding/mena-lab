"""Metal Shading Language (MSL) backward Pass 1 (Receiver-CSR) & Pass 2 (Sender-CSR) kernels."""
from __future__ import annotations

METAL_BWD_SOURCE = r"""
template <bool LATER>
inline void run_csr_bwd_recv(
    device const float* pos, device const int* recv_ptr, device const int* send_idx,
    device const float* shift_in, device const float* cell, device const long* graph,
    device const long* z, device const float* rmin_tab, device const float* rho,
    device const float* poly, device const float* gates, device const float* qn,
    device const float* kn, device const float* vec, device const float* ten,
    device const float* g_num, device const float* g_zc, device const float* g_dsum,
    device float* d_pos_recv, device float* d_qn, device float* edge_grad8,
    device float* vir_atom, device float* d_rho_recv,
    int n_atoms, int grid, int flags, float r_cut, float r_on, float max_doff_r2,
    uint lane, int i
) {
    if (i >= n_atoms) return;
    constexpr int TD = LATER ? 284 : 188;
    constexpr int OD = LATER ? 408 : 360;
    constexpr int F_O_OFF = LATER ? 276 : 180;
    constexpr int OUT_T_OFF = LATER ? 256 : 224;
    constexpr int OUT_O_OFF = LATER ? 352 : 304;
    bool has_floor = (flags & FLAG_FLOOR) != 0, has_rho = (flags & FLAG_RHO) != 0;
    bool has_ten = LATER && ((flags & FLAG_TEN) != 0);
    bool need_vir = (flags & FLAG_VIR) != 0, need_drho = (flags & FLAG_DRHO) != 0;
    int g_id = ((flags & FLAG_CELL) != 0 && (flags & FLAG_G1) == 0) ? int(graph[i]) : 0;
    int zi_shift = has_floor ? (int(clamp(z[i], 0L, 127L)) << 7) : 0;
    float rho_i = has_rho ? rho[i] : 0.0f;
    float3 pi = float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    float q0 = qn[64 * i + int(lane)], q1 = qn[64 * i + 32 + int(lane)];
    float dq0 = 0.0f, dq1 = 0.0f, drho_i = 0.0f, vir_lane = 0.0f;
    float3 dpos_i = float3(0.0f);

    device const float* gi = g_num + OD * i;
    float gs[4] = {gi[int(lane)], gi[32 + int(lane)], gi[64 + int(lane)], gi[96 + int(lane)]};
    float3 gv = float3(gi[128 + 3 * int(lane)], gi[128 + 3 * int(lane) + 1], gi[128 + 3 * int(lane) + 2]);
    float glon = LATER ? gi[224 + int(lane)] : 0.0f, glon2 = 0.0f;
    float gt[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < 16) {
        for (int m = 0; m < 5; ++m) gt[m] = gi[OUT_T_OFF + 5 * int(lane) + m];
        if (LATER) glon2 = gi[336 + int(lane)];
    }
    float go[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < 8) { for (int m = 0; m < 7; ++m) go[m] = gi[OUT_O_OFF + 7 * int(lane) + m]; }
    float gzc0 = g_zc[4 * i + 0], gzc1 = g_zc[4 * i + 1], gzc2 = g_zc[4 * i + 2], gzc3 = g_zc[4 * i + 3];
    float gdsum_i = g_dsum[i], r_cut2 = r_cut * r_cut, inv_w_env = 1.0f / (r_cut - r_on), inv_h = float(grid) / r_cut;

    for (int e = recv_ptr[i], e_end = recv_ptr[i + 1]; e < e_end; ++e) {
        int j = send_idx[e];
        float3 d = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]) - pi + load_shift(shift_in, cell, e, g_id, flags);
        float r2 = dot(d, d);
        if (r2 < 1e-24f || r2 >= r_cut2) {
            if (lane < 8) edge_grad8[8 * e + int(lane)] = 0.0f;
            continue;
        }
        float r_raw = sqrt(r2), inv_r_raw = 1.0f / r_raw;
        float3 u = d * inv_r_raw, drfl_dd = u;
        float r_fl = r_raw;
        if (has_floor && r2 < max_doff_r2) {
            float r_min = rmin_tab[zi_shift + int(clamp(z[j], 0L, 127L))];
            float d_off = r_min + 0.05f, r_eps = sqrt(r2 + 1e-8f);
            if (r_eps < d_off && r_min > 0.0f) {
                float th = tanh((d_off - r_eps) * 20.0f);
                r_fl = d_off - 0.05f * th;
                drfl_dd = d * ((1.0f - th * th) / r_eps);
            }
        }
        if (r_fl >= r_cut) {
            if (lane < 8) edge_grad8[8 * e + int(lane)] = 0.0f;
            continue;
        }
        float chi = 1.0f, dchi_drfl = 0.0f;
        if (r_fl > r_on) {
            float te = (r_fl - r_on) * inv_w_env, t2 = te * te, om = 1.0f - te;
            chi = 1.0f - (t2 * t2) * fma(te, fma(te, fma(-20.0f, te, 70.0f), -84.0f), 35.0f);
            dchi_drfl = -140.0f * (t2 * te) * (om * om * om) * inv_w_env;
        }
        float x = r_fl, dx_drfl = 1.0f, dx_drho = 0.0f;
        if (has_rho) {
            float rho_ij = rho_i + rho[j], sw = 1.52f / rho_ij;
            float inv_den = 1.0f / (1.0f + (sw - 1.0f) * (r_fl / r_cut));
            x = (sw * r_fl) * inv_den;
            dx_drfl = sw * (inv_den * inv_den);
            if (need_drho) dx_drho = -(1.52f / (rho_ij * rho_ij)) * r_fl * (1.0f - r_fl / r_cut) * (inv_den * inv_den);
        }
        float inv_h_eff = (x >= 0.0f && x <= r_cut) ? inv_h : 0.0f;
        float u_sp = clamp(x, 0.0f, r_cut) * inv_h;
        int idx = min(max(int(floor(u_sp)), 0), grid - 1);
        float tsp = u_sp - float(idx);
        device const float* p0 = poly + (idx * 4) * TD;

        float k0 = kn[64 * j + int(lane)], k1 = kn[64 * j + 32 + int(lane)];
        float d0, d1, d2, d3;
        qk_dots4(q0, q1, k0, k1, d0, d1, d2, d3);
        float ex_l = 0.0f, w_l = 0.0f, th_b = 0.0f, db_dx_l = 0.0f;
        if (lane < 4) {
            float dl = (lane == 0) ? d0 : ((lane == 1) ? d1 : ((lane == 2) ? d2 : d3)), braw;
            horner_val_der(p0, TD, int(lane), tsp, inv_h_eff, braw, db_dx_l);
            th_b = tanh(0.5f * braw);
            ex_l = exp(dl + 2.0f * th_b);
            w_l = chi * ex_l;
        }
        float w0 = simd_broadcast(w_l, ushort(0)), w1 = simd_broadcast(w_l, ushort(1));
        float w2 = simd_broadcast(w_l, ushort(2)), w3 = simd_broadcast(w_l, ushort(3));
        float ws[4] = {w0, w1, w2, w3}, dw_loc[4] = {0.0f, 0.0f, 0.0f, 0.0f}, dx_loc = 0.0f;
        float3 du_loc = float3(0.0f);

        device const float* gj = gates + 184 * j;
        for (int m = 0; m < 4; ++m) {
            int col = 32 * m + int(lane);
            float fs, dfs, gval = gj[col];
            horner_val_der(p0, TD, 4 + col, tsp, inv_h_eff, fs, dfs);
            dw_loc[m] += gs[m] * fs * gval;
            dx_loc += (gs[m] * ws[m] * gval) * dfs;
        }
        int hv = int(lane >> 3);
        float wv = ws[hv], f_dir, df_dir, gv_j = gj[128 + int(lane)];
        horner_val_der(p0, TD, 132 + int(lane), tsp, inv_h_eff, f_dir, df_dir);
        float w_dir = f_dir * gv_j, dot_gv_u = dot(gv, u);
        if (LATER) {
            device const float* bj = vec + 96 * j + 3 * int(lane);
            float3 b = float3(bj[0], bj[1], bj[2]);
            float f_tr, df_tr, f_lon, df_lon;
            horner_val_der(p0, TD, 180 + int(lane), tsp, inv_h_eff, f_tr, df_tr);
            horner_val_der(p0, TD, 212 + int(lane), tsp, inv_h_eff, f_lon, df_lon);
            float dot_bu = dot(b, u), dot_gv_b = dot(gv, b), w_lon = f_lon * dot_bu;
            dw_loc[hv] += (w_dir + w_lon) * dot_gv_u + f_tr * dot_gv_b + w_lon * glon;
            float dw_lon = wv * (dot_gv_u + glon);
            dx_loc += (wv * dot_gv_u * gv_j) * df_dir + (dw_lon * dot_bu) * df_lon + (wv * dot_gv_b) * df_tr;
            du_loc += (wv * (w_dir + w_lon)) * gv + (dw_lon * f_lon) * b;
        } else {
            dw_loc[hv] += w_dir * dot_gv_u;
            dx_loc += (wv * dot_gv_u * gv_j) * df_dir;
            du_loc += (wv * w_dir) * gv;
        }

        float y2[5], y3[7], dy2_loc[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        eval_y2(u, y2);
        eval_y3(u, y3);
        if (lane < 16) {
            int ht = int(lane >> 2);
            float wt = ws[ht], f_ten, df_ten, gt_j = gj[160 + int(lane)];
            horner_val_der(p0, TD, 164 + int(lane), tsp, inv_h_eff, f_ten, df_ten);
            float w_ten = f_ten * gt_j;
            float dot_gt_y2 = gt[0]*y2[0] + gt[1]*y2[1] + gt[2]*y2[2] + gt[3]*y2[3] + gt[4]*y2[4];
            if (has_ten) {
                device const float* tj = ten + 80 * j + 5 * int(lane);
                float f_tr2, df_tr2, f_lon2, df_lon2, dot_ty2 = 0.0f, dot_gt_tau = 0.0f;
                horner_val_der(p0, TD, 244 + int(lane), tsp, inv_h_eff, f_tr2, df_tr2);
                horner_val_der(p0, TD, 260 + int(lane), tsp, inv_h_eff, f_lon2, df_lon2);
                for (int m = 0; m < 5; ++m) { dot_ty2 += tj[m] * y2[m]; dot_gt_tau += gt[m] * tj[m]; }
                float w_lon2 = f_lon2 * dot_ty2, dw_lon2 = wt * (dot_gt_y2 + glon2);
                dw_loc[ht] += (w_ten + w_lon2) * dot_gt_y2 + f_tr2 * dot_gt_tau + w_lon2 * glon2;
                dx_loc += (wt * dot_gt_y2 * gt_j) * df_ten + (dw_lon2 * dot_ty2) * df_lon2 + (wt * dot_gt_tau) * df_tr2;
                float sc = wt * (w_ten + w_lon2), st = dw_lon2 * f_lon2;
                for (int m = 0; m < 5; ++m) dy2_loc[m] = sc * gt[m] + st * tj[m];
            } else {
                dw_loc[ht] += w_ten * dot_gt_y2;
                dx_loc += (wt * dot_gt_y2 * gt_j) * df_ten;
                float sc = wt * w_ten;
                for (int m = 0; m < 5; ++m) dy2_loc[m] = sc * gt[m];
            }
        }
        float dy3_loc[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (lane < 8) {
            int ho = int(lane >> 1);
            float wo = ws[ho], f_o, df_o, go_j = gj[176 + int(lane)], dot_go_y3 = 0.0f;
            horner_val_der(p0, TD, F_O_OFF + int(lane), tsp, inv_h_eff, f_o, df_o);
            for (int m = 0; m < 7; ++m) dot_go_y3 += go[m] * y3[m];
            float w_o = f_o * go_j, sc = wo * w_o;
            dw_loc[ho] += w_o * dot_go_y3;
            dx_loc += (wo * dot_go_y3 * go_j) * df_o;
            for (int m = 0; m < 7; ++m) dy3_loc[m] = sc * go[m];
        }

        float dw0 = simd_sum(dw_loc[0]) + gzc0, dw1 = simd_sum(dw_loc[1]) + gzc1;
        float dw2 = simd_sum(dw_loc[2]) + gzc2, dw3 = simd_sum(dw_loc[3]) + gzc3;
        float gl0 = dw0 * w0, gl1 = dw1 * w1, gl2 = dw2 * w2, gl3 = dw3 * w3, dchi_part = 0.0f;
        if (lane < 4) {
            float dwl = (lane == 0) ? dw0 : ((lane == 1) ? dw1 : ((lane == 2) ? dw2 : dw3));
            float gll = (lane == 0) ? gl0 : ((lane == 1) ? gl1 : ((lane == 2) ? gl2 : gl3));
            dchi_part = dwl * ex_l;
            dx_loc += gll * (1.0f - th_b * th_b) * db_dx_l;
        }
        float d_chi = gdsum_i + simd_sum(dchi_part), d_x = simd_sum(dx_loc);
        float d_rfl = d_chi * dchi_drfl + d_x * dx_drfl;
        float gy2[5], gy3[7];
        for (int m = 0; m < 5; ++m) gy2[m] = simd_sum(dy2_loc[m]);
        for (int m = 0; m < 7; ++m) gy3[m] = simd_sum(dy3_loc[m]);
        float3 gu = float3(simd_sum(du_loc.x), simd_sum(du_loc.y), simd_sum(du_loc.z)) + grad_y2_u(u, gy2) + grad_y3_u(u, gy3);
        float3 gd = (gu - dot(gu, u) * u) * inv_r_raw + d_rfl * drfl_dd;

        dpos_i -= gd;
        dq0 += ((lane < 16) ? gl0 : gl1) * k0;
        dq1 += ((lane < 16) ? gl2 : gl3) * k1;
        float drho_e = need_drho ? (d_x * dx_drho) : 0.0f;
        drho_i += drho_e;
        if (need_vir && lane < 9) {
            int r_ax = int(lane) / 3, c_ax = int(lane) % 3;
            float gd_comp = (c_ax == 0) ? gd.x : ((c_ax == 1) ? gd.y : gd.z);
            vir_lane += shift_in[3 * e + r_ax] * gd_comp;
        }
        if (lane < 8) {
            float o8[8] = {gd.x, gd.y, gd.z, gl0, gl1, gl2, gl3, drho_e};
            edge_grad8[8 * e + int(lane)] = o8[lane];
        }
    }
    if (lane == 0) {
        d_pos_recv[3 * i + 0] = dpos_i.x; d_pos_recv[3 * i + 1] = dpos_i.y; d_pos_recv[3 * i + 2] = dpos_i.z;
        if (need_drho) d_rho_recv[i] = drho_i;
    }
    d_qn[64 * i + int(lane)] = dq0;
    d_qn[64 * i + 32 + int(lane)] = dq1;
    if (need_vir && lane < 9) vir_atom[9 * i + int(lane)] = vir_lane;
}

template <bool LATER>
inline void run_csr_bwd_send(
    device const float* pos, device const int* send_ptr, device const int* send_perm,
    device const int* recv_idx, device const float* shift_in, device const float* cell,
    device const long* graph, device const long* z, device const float* rmin_tab,
    device const float* rho, device const float* poly, device const float* qn,
    device const float* kn, device const float* g_num, device const float* edge_grad8,
    device float* d_pos, device float* d_gates, device float* d_kn,
    device float* d_vec, device float* d_ten, device float* d_rho,
    int n_atoms, int grid, int flags, float r_cut, float r_on, float max_doff_r2,
    uint lane, int j
) {
    if (j >= n_atoms) return;
    constexpr int TD = LATER ? 284 : 188;
    constexpr int OD = LATER ? 408 : 360;
    constexpr int F_O_OFF = LATER ? 276 : 180;
    constexpr int OUT_T_OFF = LATER ? 256 : 224;
    constexpr int OUT_O_OFF = LATER ? 352 : 304;
    bool has_floor = (flags & FLAG_FLOOR) != 0, has_rho = (flags & FLAG_RHO) != 0;
    bool has_ten = LATER && ((flags & FLAG_TEN) != 0), need_drho = (flags & FLAG_DRHO) != 0;
    int zj = has_floor ? int(clamp(z[j], 0L, 127L)) : 0;
    float rho_j = has_rho ? rho[j] : 0.0f;
    float3 pj = float3(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]), db = float3(0.0f), dpos_j = float3(0.0f);
    float k0 = kn[64 * j + int(lane)], k1 = kn[64 * j + 32 + int(lane)], dk0 = 0.0f, dk1 = 0.0f, drho_j = 0.0f;
    float dgs[4] = {0.0f, 0.0f, 0.0f, 0.0f}, dgv = 0.0f, dgt = 0.0f, dgo = 0.0f, dtau[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float r_cut2 = r_cut * r_cut, inv_w_env = 1.0f / (r_cut - r_on), inv_h = float(grid) / r_cut;

    for (int kp = send_ptr[j], k_end = send_ptr[j + 1]; kp < k_end; ++kp) {
        int e = send_perm[kp], i = recv_idx[e];
        int g_id = ((flags & FLAG_CELL) != 0 && (flags & FLAG_G1) == 0) ? int(graph[i]) : 0;
        float3 d = pj - float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) + load_shift(shift_in, cell, e, g_id, flags);
        float r2 = dot(d, d);
        if (r2 < 1e-24f || r2 >= r_cut2) continue;
        float r_raw = sqrt(r2), r_fl = r_raw;
        float3 u = d * (1.0f / r_raw);
        if (has_floor && r2 < max_doff_r2) {
            float r_min = rmin_tab[(int(clamp(z[i], 0L, 127L)) << 7) + zj];
            float d_off = r_min + 0.05f, r_eps = sqrt(r2 + 1e-8f);
            if (r_eps < d_off && r_min > 0.0f) r_fl = d_off - 0.05f * tanh((d_off - r_eps) * 20.0f);
        }
        if (r_fl >= r_cut) continue;

        device const float* eg = edge_grad8 + 8 * e;
        dpos_j += float3(eg[0], eg[1], eg[2]);
        float gl0 = eg[3], gl1 = eg[4], gl2 = eg[5], gl3 = eg[6];
        if (need_drho) drho_j += eg[7];
        float q0 = qn[64 * i + int(lane)], q1 = qn[64 * i + 32 + int(lane)];
        dk0 += ((lane < 16) ? gl0 : gl1) * q0;
        dk1 += ((lane < 16) ? gl2 : gl3) * q1;

        float chi = 1.0f;
        if (r_fl > r_on) {
            float te = (r_fl - r_on) * inv_w_env, t2 = te * te;
            chi = 1.0f - (t2 * t2) * fma(te, fma(te, fma(-20.0f, te, 70.0f), -84.0f), 35.0f);
        }
        float x = r_fl;
        if (has_rho) {
            float sw = 1.52f / (rho[i] + rho_j);
            x = (sw * r_fl) / (1.0f + (sw - 1.0f) * (r_fl / r_cut));
        }
        float u_sp = clamp(x, 0.0f, r_cut) * inv_h;
        int idx = min(max(int(floor(u_sp)), 0), grid - 1);
        float tsp = u_sp - float(idx);
        device const float* p0 = poly + (idx * 4) * TD;

        float d0, d1, d2, d3, w_l = 0.0f;
        qk_dots4(q0, q1, k0, k1, d0, d1, d2, d3);
        if (lane < 4) {
            float dl = (lane == 0) ? d0 : ((lane == 1) ? d1 : ((lane == 2) ? d2 : d3));
            w_l = chi * exp(dl + 2.0f * tanh(0.5f * horner_val(p0, TD, int(lane), tsp)));
        }
        float ws[4] = {simd_broadcast(w_l, ushort(0)), simd_broadcast(w_l, ushort(1)),
                       simd_broadcast(w_l, ushort(2)), simd_broadcast(w_l, ushort(3))};

        device const float* gi = g_num + OD * i;
        for (int m = 0; m < 4; ++m) {
            int col = 32 * m + int(lane);
            dgs[m] += gi[col] * ws[m] * horner_val(p0, TD, 4 + col, tsp);
        }
        float wv = ws[lane >> 3];
        float3 gv = float3(gi[128 + 3 * int(lane)], gi[128 + 3 * int(lane) + 1], gi[128 + 3 * int(lane) + 2]);
        float dot_gv_u = dot(gv, u);
        dgv += dot_gv_u * wv * horner_val(p0, TD, 132 + int(lane), tsp);
        if (LATER) {
            float glon = gi[224 + int(lane)];
            db += wv * (horner_val(p0, TD, 180 + int(lane), tsp) * gv
                      + (horner_val(p0, TD, 212 + int(lane), tsp) * (dot_gv_u + glon)) * u);
        }
        float y2[5], y3[7];
        eval_y2(u, y2);
        eval_y3(u, y3);
        if (lane < 16) {
            float wt = ws[lane >> 2];
            device const float* gti = gi + OUT_T_OFF + 5 * int(lane);
            float dot_gt_y2 = gti[0]*y2[0] + gti[1]*y2[1] + gti[2]*y2[2] + gti[3]*y2[3] + gti[4]*y2[4];
            dgt += dot_gt_y2 * wt * horner_val(p0, TD, 164 + int(lane), tsp);
            if (has_ten) {
                float glon2 = gi[336 + int(lane)];
                float st2 = wt * horner_val(p0, TD, 244 + int(lane), tsp);
                float sl2 = wt * horner_val(p0, TD, 260 + int(lane), tsp) * (dot_gt_y2 + glon2);
                for (int m = 0; m < 5; ++m) dtau[m] += st2 * gti[m] + sl2 * y2[m];
            }
        }
        if (lane < 8) {
            float wo = ws[lane >> 1], dot_go_y3 = 0.0f;
            device const float* goi = gi + OUT_O_OFF + 7 * int(lane);
            for (int m = 0; m < 7; ++m) dot_go_y3 += goi[m] * y3[m];
            dgo += dot_go_y3 * wo * horner_val(p0, TD, F_O_OFF + int(lane), tsp);
        }
    }
    if (lane == 0) {
        d_pos[3 * j + 0] += dpos_j.x; d_pos[3 * j + 1] += dpos_j.y; d_pos[3 * j + 2] += dpos_j.z;
        if (need_drho) d_rho[j] += drho_j;
    }
    device float* dgj = d_gates + 184 * j;
    for (int m = 0; m < 4; ++m) dgj[32 * m + int(lane)] = dgs[m];
    dgj[128 + int(lane)] = dgv;
    if (lane < 16) dgj[160 + int(lane)] = dgt;
    if (lane < 8)  dgj[176 + int(lane)] = dgo;
    d_kn[64 * j + int(lane)] = dk0;
    d_kn[64 * j + 32 + int(lane)] = dk1;
    if (LATER) {
        d_vec[96 * j + 3 * int(lane) + 0] = db.x; d_vec[96 * j + 3 * int(lane) + 1] = db.y; d_vec[96 * j + 3 * int(lane) + 2] = db.z;
        if (has_ten && lane < 16) { for (int m = 0; m < 5; ++m) d_ten[80 * j + 5 * int(lane) + m] = dtau[m]; }
    }
}

#define DECL_BWD_RECV(NAME, IS_LATER) \
kernel void NAME( \
    device const float* pos [[buffer(0)]], device const int* recv_ptr [[buffer(1)]], device const int* send_idx [[buffer(2)]], \
    device const float* shift_in [[buffer(3)]], device const float* cell [[buffer(4)]], device const long* graph [[buffer(5)]], \
    device const long* z [[buffer(6)]], device const float* rmin_tab [[buffer(7)]], device const float* rho [[buffer(8)]], \
    device const float* poly [[buffer(9)]], device const float* gates [[buffer(10)]], device const float* qn [[buffer(11)]], \
    device const float* kn [[buffer(12)]], device const float* vec [[buffer(13)]], device const float* ten [[buffer(14)]], \
    device const float* g_num [[buffer(15)]], device const float* g_zc [[buffer(16)]], device const float* g_dsum [[buffer(17)]], \
    device float* d_pos_recv [[buffer(18)]], device float* d_qn [[buffer(19)]], device float* edge_grad8 [[buffer(20)]], \
    device float* vir_atom [[buffer(21)]], device float* d_rho_recv [[buffer(22)]], constant int& n_atoms [[buffer(23)]], \
    constant int& grid [[buffer(24)]], constant int& flags [[buffer(25)]], constant float& r_cut [[buffer(26)]], \
    constant float& r_on [[buffer(27)]], constant float& max_doff_r2 [[buffer(28)]], uint lane [[thread_index_in_simdgroup]], \
    uint sg_id [[simdgroup_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]) { \
    run_csr_bwd_recv<IS_LATER>(pos, recv_ptr, send_idx, shift_in, cell, graph, z, rmin_tab, rho, poly, gates, qn, kn, vec, ten, \
        g_num, g_zc, g_dsum, d_pos_recv, d_qn, edge_grad8, vir_atom, d_rho_recv, n_atoms, grid, flags, r_cut, r_on, max_doff_r2, \
        lane, int(tg_id * 4 + sg_id)); \
}

#define DECL_BWD_SEND(NAME, IS_LATER) \
kernel void NAME( \
    device const float* pos [[buffer(0)]], device const int* send_ptr [[buffer(1)]], device const int* send_perm [[buffer(2)]], \
    device const int* recv_idx [[buffer(3)]], device const float* shift_in [[buffer(4)]], device const float* cell [[buffer(5)]], \
    device const long* graph [[buffer(6)]], device const long* z [[buffer(7)]], device const float* rmin_tab [[buffer(8)]], \
    device const float* rho [[buffer(9)]], device const float* poly [[buffer(10)]], device const float* qn [[buffer(11)]], \
    device const float* kn [[buffer(12)]], device const float* g_num [[buffer(13)]], device const float* edge_grad8 [[buffer(14)]], \
    device float* d_pos [[buffer(15)]], device float* d_gates [[buffer(16)]], device float* d_kn [[buffer(17)]], \
    device float* d_vec [[buffer(18)]], device float* d_ten [[buffer(19)]], device float* d_rho [[buffer(20)]], \
    constant int& n_atoms [[buffer(21)]], constant int& grid [[buffer(22)]], constant int& flags [[buffer(23)]], \
    constant float& r_cut [[buffer(24)]], constant float& r_on [[buffer(25)]], constant float& max_doff_r2 [[buffer(26)]], \
    uint lane [[thread_index_in_simdgroup]], uint sg_id [[simdgroup_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]) { \
    run_csr_bwd_send<IS_LATER>(pos, send_ptr, send_perm, recv_idx, shift_in, cell, graph, z, rmin_tab, rho, poly, qn, kn, g_num, \
        edge_grad8, d_pos, d_gates, d_kn, d_vec, d_ten, d_rho, n_atoms, grid, flags, r_cut, r_on, max_doff_r2, lane, int(tg_id * 4 + sg_id)); \
}

DECL_BWD_RECV(csr_bwd_recv_first, false)
DECL_BWD_RECV(csr_bwd_recv_later, true)
DECL_BWD_SEND(csr_bwd_send_first, false)
DECL_BWD_SEND(csr_bwd_send_later, true)
"""
