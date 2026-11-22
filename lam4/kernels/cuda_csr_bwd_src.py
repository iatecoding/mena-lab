"""CUDA C++ backward Dual-CSR kernels: single-pass L2 RED.ADD (`1pass`) + 2-pass atomic-free (`2pass`, `edge_grad12`)."""
from __future__ import annotations

CUDA_CSR_BWD_SRC = r"""
template <bool LATER, bool ONE_PASS, int WPA = 1>
__global__ void cuda_csr_bwd_recv_k(
    const float* __restrict__ pos, const int* __restrict__ recv_ptr, const int* __restrict__ send_idx,
    const float* __restrict__ shift_in, const float* __restrict__ cell, const long* __restrict__ graph,
    const long* __restrict__ z, const float* __restrict__ rmin_tab, const float* __restrict__ rho,
    const float4* __restrict__ poly4, const float* __restrict__ gates, const float* __restrict__ qn,
    const float* __restrict__ kn, const float* __restrict__ vec, const float* __restrict__ ten,
    const float* __restrict__ agg, const float* __restrict__ zc, const float* __restrict__ dsum,
    const float* __restrict__ g_agg, const float* __restrict__ g_dsum_ext,
    float* __restrict__ d_pos, float* __restrict__ d_qn, float* __restrict__ d_gates,
    float* __restrict__ d_kn, float* __restrict__ d_vec, float* __restrict__ d_ten,
    float* __restrict__ edge_grad12, float* __restrict__ d_cell, float* __restrict__ d_rho,
    int n_atoms, int grid, int flags, float r_cut, float r_on, float max_doff_r2
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    constexpr int ATOMS_PER_CTA = 4 / WPA;
    int local_atom = warp_id / WPA;
    int sub_w = warp_id - local_atom * WPA;
    int i = blockIdx.x * ATOMS_PER_CTA + local_atom;
    if constexpr (WPA == 1) {
        if (i >= n_atoms) return;
    }

    constexpr int TD = LATER ? 284 : 188;
    constexpr int OD = LATER ? 408 : 360;
    constexpr int F_O_OFF = LATER ? 276 : 180;
    constexpr int OUT_T_OFF = LATER ? 256 : 224;
    constexpr int OUT_O_OFF = LATER ? 352 : 304;

    bool has_floor = (flags & F_FLOOR) != 0;
    bool has_rho = (flags & F_RHO) != 0;
    bool has_ten = LATER && ((flags & F_TEN) != 0);
    bool need_vir = (flags & F_VIR) != 0;
    bool need_drho = (flags & F_DRHO) != 0;
    bool need_send_feat = (flags & F_SEND_FEAT) != 0;
    bool need_dqn = (flags & F_DQN) != 0;

    float dq0 = 0.0f, dq1 = 0.0f, drho_i = 0.0f, vir_lane = 0.0f;
    float dpos_ix = 0.0f, dpos_iy = 0.0f, dpos_iz = 0.0f;
    int g_id = 0;

    if (i < n_atoms) {
    g_id = ((flags & F_CELL) != 0 && (flags & F_G1) == 0) ? int(graph[i]) : 0;
    int zi_shift = has_floor ? (clamp_z(z[i]) << 7) : 0;
    float rho_i = has_rho ? __ldg(rho + i) : 0.0f;
    float3 pi = make_float3(__ldg(pos + 3 * i), __ldg(pos + 3 * i + 1), __ldg(pos + 3 * i + 2));
    float q0 = __ldg(qn + 64 * i + lane);
    float q1 = __ldg(qn + 64 * i + 32 + lane);

    // Fused finalise backward prologue: load g_agg & agg, compute scale_h, g_num in registers, gzc & gdsum_i.
    float dsum_i = __ldg(dsum + i);
    float sq_d = sqrtf(dsum_i + 1.0f);
    float zc_l = (lane < 4) ? __ldg(zc + 4 * i + lane) : 0.0f;
    float sc_l = (lane < 4) ? (sq_d / (zc_l + 1.0f)) : 0.0f;
    float sc[4] = {
        __shfl_sync(0xffffffffu, sc_l, 0), __shfl_sync(0xffffffffu, sc_l, 1),
        __shfl_sync(0xffffffffu, sc_l, 2), __shfl_sync(0xffffffffu, sc_l, 3)
    };

    const float* ga = g_agg + (long)OD * i;
    const float* ag = agg + (long)OD * i;
    float dot_h[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float gs[4];
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int col = 32 * m + lane;
        float g_val = __ldg(ga + col);
        dot_h[m] += g_val * __ldg(ag + col);
        gs[m] = g_val * sc[m];
    }
    int hv = lane >> 3;
    float scv = sc[hv];
    float gvx = __ldg(ga + 128 + 3 * lane + 0);
    float gvy = __ldg(ga + 128 + 3 * lane + 1);
    float gvz = __ldg(ga + 128 + 3 * lane + 2);
    dot_h[hv] += gvx * __ldg(ag + 128 + 3 * lane + 0)
               + gvy * __ldg(ag + 128 + 3 * lane + 1)
               + gvz * __ldg(ag + 128 + 3 * lane + 2);
    float3 gv = make_float3(gvx * scv, gvy * scv, gvz * scv);

    float glon = 0.0f, glon2 = 0.0f;
    if (LATER) {
        float gl = __ldg(ga + 224 + lane);
        dot_h[hv] += gl * __ldg(ag + 224 + lane);
        glon = gl * scv;
    }
    float gt[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < 16) {
        int ht = lane >> 2;
        float sct = sc[ht];
        #pragma unroll
        for (int m = 0; m < 5; ++m) {
            float g_val = __ldg(ga + OUT_T_OFF + 5 * lane + m);
            dot_h[ht] += g_val * __ldg(ag + OUT_T_OFF + 5 * lane + m);
            gt[m] = g_val * sct;
        }
        if (LATER) {
            float gl2 = __ldg(ga + 336 + lane);
            dot_h[ht] += gl2 * __ldg(ag + 336 + lane);
            glon2 = gl2 * sct;
        }
    }
    float go[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < 8) {
        int ho = lane >> 1;
        float sco = sc[ho];
        #pragma unroll
        for (int m = 0; m < 7; ++m) {
            float g_val = __ldg(ga + OUT_O_OFF + 7 * lane + m);
            dot_h[ho] += g_val * __ldg(ag + OUT_O_OFF + 7 * lane + m);
            go[m] = g_val * sco;
        }
    }
    float dh0 = warp_sum(dot_h[0]), dh1 = warp_sum(dot_h[1]), dh2 = warp_sum(dot_h[2]), dh3 = warp_sum(dot_h[3]);
    float zc0 = __shfl_sync(0xffffffffu, zc_l, 0), zc1 = __shfl_sync(0xffffffffu, zc_l, 1);
    float zc2 = __shfl_sync(0xffffffffu, zc_l, 2), zc3 = __shfl_sync(0xffffffffu, zc_l, 3);
    float gzc0 = -dh0 / (zc0 + 1.0f);
    float gzc1 = -dh1 / (zc1 + 1.0f);
    float gzc2 = -dh2 / (zc2 + 1.0f);
    float gzc3 = -dh3 / (zc3 + 1.0f);
    float gdsum_ext_i = (flags & F_GDSUM) ? __ldg(g_dsum_ext + i) : 0.0f;
    float gdsum_i = gdsum_ext_i + 0.5f * (dh0 + dh1 + dh2 + dh3) / (dsum_i + 1.0f);

    float r_cut2 = r_cut * r_cut;
    float inv_w_env = 1.0f / (r_cut - r_on);
    float inv_h = float(grid) / r_cut;

    int e_beg = recv_ptr[i], e_end = recv_ptr[i + 1];
    for (int e = e_beg + sub_w; e < e_end; e += WPA) {
        int j = send_idx[e];
        float3 es;
        float3 sh = load_shift(shift_in, cell, e, g_id, flags, es);
        float dx = __ldg(pos + 3 * j) - pi.x + sh.x;
        float dy = __ldg(pos + 3 * j + 1) - pi.y + sh.y;
        float dz = __ldg(pos + 3 * j + 2) - pi.z + sh.z;
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < 1e-24f || r2 >= r_cut2) {
            if (!ONE_PASS && lane < 12) edge_grad12[(long)12 * e + lane] = 0.0f;
            continue;
        }
        float r_raw = sqrtf(r2);
        float inv_r = 1.0f / r_raw;
        float3 u = make_float3(dx * inv_r, dy * inv_r, dz * inv_r);
        float3 drfl_dd = u;
        float r_fl = r_raw;
        if (has_floor && r2 < max_doff_r2) {
            float r_min = __ldg(rmin_tab + zi_shift + clamp_z(z[j]));
            float d_off = r_min + 0.05f;
            float r_eps = sqrtf(r2 + 1e-8f);
            if (r_eps < d_off && r_min > 0.0f) {
                float th = tanhf((d_off - r_eps) * 20.0f);
                r_fl = d_off - 0.05f * th;
                float fsc = (1.0f - th * th) / r_eps;
                drfl_dd = make_float3(dx * fsc, dy * fsc, dz * fsc);
            }
        }
        if (r_fl >= r_cut) {
            if (!ONE_PASS && lane < 12) edge_grad12[(long)12 * e + lane] = 0.0f;
            continue;
        }

        float chi = 1.0f, dchi_drfl = 0.0f;
        if (r_fl > r_on) {
            float te = (r_fl - r_on) * inv_w_env, t2 = te * te, om = 1.0f - te;
            chi = 1.0f - (t2 * t2) * fmaf(te, fmaf(te, fmaf(-20.0f, te, 70.0f), -84.0f), 35.0f);
            dchi_drfl = -140.0f * (t2 * te) * (om * om * om) * inv_w_env;
        }
        float x = r_fl, dx_drfl = 1.0f, dx_drho = 0.0f;
        if (has_rho) {
            float rho_ij = rho_i + __ldg(rho + j);
            float sw = 1.52f / rho_ij;
            float inv_den = 1.0f / (1.0f + (sw - 1.0f) * (r_fl / r_cut));
            x = (sw * r_fl) * inv_den;
            dx_drfl = sw * (inv_den * inv_den);
            if (need_drho) dx_drho = -(1.52f / (rho_ij * rho_ij)) * r_fl * (1.0f - r_fl / r_cut) * (inv_den * inv_den);
        }
        float inv_h_eff = (x >= 0.0f && x <= r_cut) ? inv_h : 0.0f;
        float u_sp = fminf(fmaxf(x, 0.0f), r_cut) * inv_h;
        int idx = min(max(int(floorf(u_sp)), 0), grid - 1);
        float tsp = u_sp - float(idx);
        const float4* p4 = poly4 + (long)idx * TD;

        float k0 = __ldg(kn + 64 * j + lane);
        float k1 = __ldg(kn + 64 * j + 32 + lane);
        float d0, d1, d2, d3;
        qk_dots4(q0, q1, k0, k1, d0, d1, d2, d3);
        float ex_l = 0.0f, w_l = 0.0f, th_b = 0.0f, db_dx_l = 0.0f;
        if (lane < 4) {
            float dl = (lane == 0) ? d0 : ((lane == 1) ? d1 : ((lane == 2) ? d2 : d3));
            float braw;
            horner4_vd(p4, lane, tsp, inv_h_eff, braw, db_dx_l);
            th_b = tanhf(0.5f * braw);
            ex_l = expf(dl + 2.0f * th_b);
            w_l = chi * ex_l;
        }
        float w0 = __shfl_sync(0xffffffffu, w_l, 0);
        float w1 = __shfl_sync(0xffffffffu, w_l, 1);
        float w2 = __shfl_sync(0xffffffffu, w_l, 2);
        float w3 = __shfl_sync(0xffffffffu, w_l, 3);
        float ws[4] = {w0, w1, w2, w3};
        float dw_loc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float dx_loc = 0.0f;
        float dux = 0.0f, duy = 0.0f, duz = 0.0f;

        const float* gj = gates + (long)184 * j;
        float* dgj = (ONE_PASS && need_send_feat) ? (d_gates + (long)184 * j) : nullptr;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int col = 32 * m + lane;
            float fs, dfs, gval = __ldg(gj + col);
            horner4_vd(p4, 4 + col, tsp, inv_h_eff, fs, dfs);
            float gs_fs = gs[m] * fs;
            dw_loc[m] = fmaf(gs_fs, gval, dw_loc[m]);
            dx_loc = fmaf(gs[m] * ws[m] * gval, dfs, dx_loc);
            if (ONE_PASS && need_send_feat) {
                atomicAdd(dgj + col, gs_fs * ws[m]);
            }
        }

        float wv = ws[hv];
        float f_dir, df_dir, gv_j = __ldg(gj + 128 + lane);
        horner4_vd(p4, 132 + lane, tsp, inv_h_eff, f_dir, df_dir);
        float w_dir = f_dir * gv_j;
        float dot_gv_u = fmaf(gv.x, u.x, fmaf(gv.y, u.y, gv.z * u.z));
        if (ONE_PASS && need_send_feat) {
            atomicAdd(dgj + 128 + lane, dot_gv_u * wv * f_dir);
        }
        if (LATER) {
            const float* bj = vec + (long)96 * j + 3 * lane;
            float b0 = __ldg(bj + 0), b1 = __ldg(bj + 1), b2 = __ldg(bj + 2);
            float f_tr, df_tr, f_lon, df_lon;
            horner4_vd(p4, 180 + lane, tsp, inv_h_eff, f_tr, df_tr);
            horner4_vd(p4, 212 + lane, tsp, inv_h_eff, f_lon, df_lon);
            float dot_bu = fmaf(b0, u.x, fmaf(b1, u.y, b2 * u.z));
            float dot_gv_b = fmaf(gv.x, b0, fmaf(gv.y, b1, gv.z * b2));
            float w_lon = f_lon * dot_bu;
            float dw_lon = wv * (dot_gv_u + glon);
            dw_loc[hv] += (w_dir + w_lon) * dot_gv_u + f_tr * dot_gv_b + w_lon * glon;
            dx_loc += (wv * dot_gv_u * gv_j) * df_dir + (dw_lon * dot_bu) * df_lon + (wv * dot_gv_b) * df_tr;
            float scv_u = wv * (w_dir + w_lon), stv_u = dw_lon * f_lon;
            dux = fmaf(scv_u, gv.x, stv_u * b0);
            duy = fmaf(scv_u, gv.y, stv_u * b1);
            duz = fmaf(scv_u, gv.z, stv_u * b2);
            if (ONE_PASS && need_send_feat) {
                float* dbj = d_vec + (long)96 * j + 3 * lane;
                float sl1 = wv * f_lon * (dot_gv_u + glon);
                float str = wv * f_tr;
                atomicAdd(dbj + 0, fmaf(str, gv.x, sl1 * u.x));
                atomicAdd(dbj + 1, fmaf(str, gv.y, sl1 * u.y));
                atomicAdd(dbj + 2, fmaf(str, gv.z, sl1 * u.z));
            }
        } else {
            dw_loc[hv] += w_dir * dot_gv_u;
            dx_loc += (wv * dot_gv_u * gv_j) * df_dir;
            float scv_u = wv * w_dir;
            dux = scv_u * gv.x; duy = scv_u * gv.y; duz = scv_u * gv.z;
        }

        if (lane < 16) {
            float y2[5], dy2_loc[5];
            eval_y2(u, y2);
            int ht = lane >> 2;
            float wt = ws[ht], f_ten, df_ten, gt_j = __ldg(gj + 160 + lane);
            horner4_vd(p4, 164 + lane, tsp, inv_h_eff, f_ten, df_ten);
            float w_ten = f_ten * gt_j;
            float dot_gt_y2 = fmaf(gt[0], y2[0], fmaf(gt[1], y2[1], fmaf(gt[2], y2[2], fmaf(gt[3], y2[3], gt[4] * y2[4]))));
            if (ONE_PASS && need_send_feat) {
                atomicAdd(dgj + 160 + lane, dot_gt_y2 * wt * f_ten);
            }
            if (has_ten) {
                const float* tj = ten + (long)80 * j + 5 * lane;
                float t0 = __ldg(tj + 0), t1 = __ldg(tj + 1), t2 = __ldg(tj + 2), t3 = __ldg(tj + 3), t4 = __ldg(tj + 4);
                float f_tr2, df_tr2, f_lon2, df_lon2;
                horner4_vd(p4, 244 + lane, tsp, inv_h_eff, f_tr2, df_tr2);
                horner4_vd(p4, 260 + lane, tsp, inv_h_eff, f_lon2, df_lon2);
                float dot_ty2 = fmaf(t0, y2[0], fmaf(t1, y2[1], fmaf(t2, y2[2], fmaf(t3, y2[3], t4 * y2[4]))));
                float dot_gt_tau = fmaf(gt[0], t0, fmaf(gt[1], t1, fmaf(gt[2], t2, fmaf(gt[3], t3, gt[4] * t4))));
                float w_lon2 = f_lon2 * dot_ty2;
                float dw_lon2 = wt * (dot_gt_y2 + glon2);
                dw_loc[ht] += (w_ten + w_lon2) * dot_gt_y2 + f_tr2 * dot_gt_tau + w_lon2 * glon2;
                dx_loc += (wt * dot_gt_y2 * gt_j) * df_ten + (dw_lon2 * dot_ty2) * df_lon2 + (wt * dot_gt_tau) * df_tr2;
                float sct_u = wt * (w_ten + w_lon2), stt_u = dw_lon2 * f_lon2;
                dy2_loc[0] = fmaf(sct_u, gt[0], stt_u * t0);
                dy2_loc[1] = fmaf(sct_u, gt[1], stt_u * t1);
                dy2_loc[2] = fmaf(sct_u, gt[2], stt_u * t2);
                dy2_loc[3] = fmaf(sct_u, gt[3], stt_u * t3);
                dy2_loc[4] = fmaf(sct_u, gt[4], stt_u * t4);
                if (ONE_PASS && need_send_feat) {
                    float* dtj = d_ten + (long)80 * j + 5 * lane;
                    float st2 = wt * f_tr2;
                    float sl2 = dw_lon2 * f_lon2;
                    #pragma unroll
                    for (int m = 0; m < 5; ++m) atomicAdd(dtj + m, fmaf(st2, gt[m], sl2 * y2[m]));
                }
            } else {
                dw_loc[ht] += w_ten * dot_gt_y2;
                dx_loc += (wt * dot_gt_y2 * gt_j) * df_ten;
                float sct_u = wt * w_ten;
                #pragma unroll
                for (int m = 0; m < 5; ++m) dy2_loc[m] = sct_u * gt[m];
            }
            // Linear grad_y2_u directly into (dux, duy, duz) eliminates 5 warp reductions & 10 live registers.
            float3 g2u = grad_y2_u(u, dy2_loc);
            dux += g2u.x; duy += g2u.y; duz += g2u.z;
        }

        if (lane < 8) {
            float y3[7], dy3_loc[7];
            eval_y3(u, y3);
            int ho = lane >> 1;
            float wo = ws[ho], f_o, df_o, go_j = __ldg(gj + 176 + lane);
            horner4_vd(p4, F_O_OFF + lane, tsp, inv_h_eff, f_o, df_o);
            float dot_go_y3 = 0.0f;
            #pragma unroll
            for (int m = 0; m < 7; ++m) dot_go_y3 = fmaf(go[m], y3[m], dot_go_y3);
            float w_o = f_o * go_j, sco_u = wo * w_o;
            dw_loc[ho] += w_o * dot_go_y3;
            dx_loc += (wo * dot_go_y3 * go_j) * df_o;
            #pragma unroll
            for (int m = 0; m < 7; ++m) dy3_loc[m] = sco_u * go[m];
            // Linear grad_y3_u directly into (dux, duy, duz) eliminates 7 warp reductions & 14 live registers.
            float3 g3u = grad_y3_u(u, dy3_loc);
            dux += g3u.x; duy += g3u.y; duz += g3u.z;
            if (ONE_PASS && need_send_feat) {
                atomicAdd(dgj + 176 + lane, dot_go_y3 * wo * f_o);
            }
        }

        float dw0 = warp_sum(dw_loc[0]) + gzc0;
        float dw1 = warp_sum(dw_loc[1]) + gzc1;
        float dw2 = warp_sum(dw_loc[2]) + gzc2;
        float dw3 = warp_sum(dw_loc[3]) + gzc3;
        float gl0 = dw0 * w0, gl1 = dw1 * w1, gl2 = dw2 * w2, gl3 = dw3 * w3;
        float dchi_part = 0.0f;
        if (lane < 4) {
            float dwl = (lane == 0) ? dw0 : ((lane == 1) ? dw1 : ((lane == 2) ? dw2 : dw3));
            float gll = (lane == 0) ? gl0 : ((lane == 1) ? gl1 : ((lane == 2) ? gl2 : gl3));
            dchi_part = dwl * ex_l;
            dx_loc += gll * (1.0f - th_b * th_b) * db_dx_l;
        }
        float d_chi = gdsum_i + warp_sum4(dchi_part);
        float d_x = warp_sum(dx_loc);
        float d_rfl = d_chi * dchi_drfl + d_x * dx_drfl;

        float3 gu = make_float3(warp_sum(dux), warp_sum(duy), warp_sum(duz));
        float dgu = gu.x * u.x + gu.y * u.y + gu.z * u.z;
        float3 gd = make_float3(
            fmaf(gu.x - dgu * u.x, inv_r, d_rfl * drfl_dd.x),
            fmaf(gu.y - dgu * u.y, inv_r, d_rfl * drfl_dd.y),
            fmaf(gu.z - dgu * u.z, inv_r, d_rfl * drfl_dd.z)
        );

        dpos_ix -= gd.x; dpos_iy -= gd.y; dpos_iz -= gd.z;
        if (need_dqn) {
            dq0 = fmaf((lane < 16) ? gl0 : gl1, k0, dq0);
            dq1 = fmaf((lane < 16) ? gl2 : gl3, k1, dq1);
        }
        float drho_e = need_drho ? (d_x * dx_drho) : 0.0f;
        drho_i += drho_e;
        if (need_vir && lane < 9) {
            int r_ax = lane / 3, c_ax = lane % 3;
            float es_c = (r_ax == 0) ? es.x : ((r_ax == 1) ? es.y : es.z);
            float gd_c = (c_ax == 0) ? gd.x : ((c_ax == 1) ? gd.y : gd.z);
            vir_lane = fmaf(es_c, gd_c, vir_lane);
        }
        if (ONE_PASS) {
            if (lane == 0) {
                atomicAdd(d_pos + 3 * j + 0, gd.x);
                atomicAdd(d_pos + 3 * j + 1, gd.y);
                atomicAdd(d_pos + 3 * j + 2, gd.z);
                if (need_drho) atomicAdd(d_rho + j, drho_e);
            }
            if (need_send_feat) {
                atomicAdd(d_kn + 64 * j + lane, ((lane < 16) ? gl0 : gl1) * q0);
                atomicAdd(d_kn + 64 * j + 32 + lane, ((lane < 16) ? gl2 : gl3) * q1);
            }
        } else {
            if (lane < 12) {
                float o12[12] = {
                    gd.x, gd.y, gd.z, gl0, gl1, gl2, gl3, drho_e,
                    w0 * sc[0], w1 * sc[1], w2 * sc[2], w3 * sc[3]
                };
                edge_grad12[(long)12 * e + lane] = o12[lane];
            }
        }
    }
    } // end if (i < n_atoms)

    if constexpr (WPA > 1) {
        __shared__ float sb[4][80];
        float* sw = sb[warp_id];
        sw[lane] = dq0;
        sw[32 + lane] = dq1;
        if (lane < 9) sw[64 + lane] = vir_lane;
        if (lane == 0) {
            sw[73] = dpos_ix; sw[74] = dpos_iy; sw[75] = dpos_iz; sw[76] = drho_i;
        }
        __syncthreads();
        if (sub_w != 0 || i >= n_atoms) return;
        for (int k = 1; k < WPA; ++k) {
            const float* sk = sb[warp_id + k];
            dq0 += sk[lane];
            dq1 += sk[32 + lane];
            if (lane < 9) vir_lane += sk[64 + lane];
            if (lane == 0) {
                dpos_ix += sk[73]; dpos_iy += sk[74]; dpos_iz += sk[75]; drho_i += sk[76];
            }
        }
    }

    if (lane == 0) {
        if (ONE_PASS) {
            atomicAdd(d_pos + 3 * i + 0, dpos_ix);
            atomicAdd(d_pos + 3 * i + 1, dpos_iy);
            atomicAdd(d_pos + 3 * i + 2, dpos_iz);
            if (need_drho) atomicAdd(d_rho + i, drho_i);
        } else {
            d_pos[3 * i + 0] = dpos_ix;
            d_pos[3 * i + 1] = dpos_iy;
            d_pos[3 * i + 2] = dpos_iz;
            if (need_drho) d_rho[i] = drho_i;
        }
    }
    if (need_dqn) {
        d_qn[64 * i + lane] = dq0;
        d_qn[64 * i + 32 + lane] = dq1;
    }
    if (need_vir && lane < 9) {
        atomicAdd(d_cell + 9 * g_id + lane, vir_lane);
    }
}

template <bool LATER>
__global__ void cuda_csr_bwd_send_k(
    const float* __restrict__ pos, const int* __restrict__ send_ptr, const int* __restrict__ send_perm,
    const int* __restrict__ recv_idx, const float* __restrict__ shift_in, const float* __restrict__ cell,
    const long* __restrict__ graph, const long* __restrict__ z, const float* __restrict__ rmin_tab,
    const float* __restrict__ rho, const float4* __restrict__ poly4, const float* __restrict__ qn,
    const float* __restrict__ g_agg, const float* __restrict__ edge_grad12,
    float* __restrict__ d_pos, float* __restrict__ d_gates, float* __restrict__ d_kn,
    float* __restrict__ d_vec, float* __restrict__ d_ten, float* __restrict__ d_rho,
    int n_atoms, int grid, int flags, float r_cut, float r_on, float max_doff_r2
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * 4 + warp_id;
    if (j >= n_atoms) return;

    constexpr int TD = LATER ? 284 : 188;
    constexpr int OD = LATER ? 408 : 360;
    constexpr int F_O_OFF = LATER ? 276 : 180;
    constexpr int OUT_T_OFF = LATER ? 256 : 224;
    constexpr int OUT_O_OFF = LATER ? 352 : 304;

    bool has_floor = (flags & F_FLOOR) != 0;
    bool has_rho = (flags & F_RHO) != 0;
    bool has_ten = LATER && ((flags & F_TEN) != 0);
    bool need_drho = (flags & F_DRHO) != 0;
    bool need_send_feat = (flags & F_SEND_FEAT) != 0;

    int zj = has_floor ? clamp_z(z[j]) : 0;
    float rho_j = has_rho ? __ldg(rho + j) : 0.0f;
    float3 pj = make_float3(__ldg(pos + 3 * j), __ldg(pos + 3 * j + 1), __ldg(pos + 3 * j + 2));
    float dpos_jx = 0.0f, dpos_jy = 0.0f, dpos_jz = 0.0f, drho_j = 0.0f;
    float dk0 = 0.0f, dk1 = 0.0f;
    float dgs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float dgv = 0.0f, dgt = 0.0f, dgo = 0.0f;
    float dbx = 0.0f, dby = 0.0f, dbz = 0.0f;
    float dtau[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float r_cut2 = r_cut * r_cut, inv_h = float(grid) / r_cut;

    for (int kp = send_ptr[j], k_end = send_ptr[j + 1]; kp < k_end; ++kp) {
        int e = send_perm[kp];
        const float* eg = edge_grad12 + (long)12 * e;
        if (lane == 0) {
            dpos_jx += __ldg(eg + 0);
            dpos_jy += __ldg(eg + 1);
            dpos_jz += __ldg(eg + 2);
            if (need_drho) drho_j += __ldg(eg + 7);
        }
        if (!need_send_feat) continue;

        int i = recv_idx[e];
        int g_id = ((flags & F_CELL) != 0 && (flags & F_G1) == 0) ? int(graph[i]) : 0;
        float3 es_dummy;
        float3 sh = load_shift(shift_in, cell, e, g_id, flags, es_dummy);
        float dx = pj.x - __ldg(pos + 3 * i) + sh.x;
        float dy = pj.y - __ldg(pos + 3 * i + 1) + sh.y;
        float dz = pj.z - __ldg(pos + 3 * i + 2) + sh.z;
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < 1e-24f || r2 >= r_cut2) continue;
        float r_raw = sqrtf(r2), inv_r = 1.0f / r_raw;
        float3 u = make_float3(dx * inv_r, dy * inv_r, dz * inv_r);
        float r_fl = r_raw;
        if (has_floor && r2 < max_doff_r2) {
            float r_min = __ldg(rmin_tab + (clamp_z(z[i]) << 7) + zj);
            float d_off = r_min + 0.05f, r_eps = sqrtf(r2 + 1e-8f);
            if (r_eps < d_off && r_min > 0.0f) r_fl = d_off - 0.05f * tanhf((d_off - r_eps) * 20.0f);
        }
        if (r_fl >= r_cut) continue;

        float gl0 = __ldg(eg + 3), gl1 = __ldg(eg + 4), gl2 = __ldg(eg + 5), gl3 = __ldg(eg + 6);
        float q0 = __ldg(qn + 64 * i + lane), q1 = __ldg(qn + 64 * i + 32 + lane);
        dk0 = fmaf((lane < 16) ? gl0 : gl1, q0, dk0);
        dk1 = fmaf((lane < 16) ? gl2 : gl3, q1, dk1);

        float x = r_fl;
        if (has_rho) {
            float sw = 1.52f / (__ldg(rho + i) + rho_j);
            x = (sw * r_fl) / (1.0f + (sw - 1.0f) * (r_fl / r_cut));
        }
        float u_sp = fminf(fmaxf(x, 0.0f), r_cut) * inv_h;
        int idx = min(max(int(floorf(u_sp)), 0), grid - 1);
        float tsp = u_sp - float(idx);
        const float4* p4 = poly4 + (long)idx * TD;

        float ws[4] = {__ldg(eg + 8), __ldg(eg + 9), __ldg(eg + 10), __ldg(eg + 11)};
        const float* gi = g_agg + (long)OD * i;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int col = 32 * m + lane;
            dgs[m] = fmaf(__ldg(gi + col) * ws[m], horner4_val(p4, 4 + col, tsp), dgs[m]);
        }
        float wv = ws[lane >> 3];
        float gvx = __ldg(gi + 128 + 3 * lane + 0), gvy = __ldg(gi + 128 + 3 * lane + 1), gvz = __ldg(gi + 128 + 3 * lane + 2);
        float dot_gv_u = fmaf(gvx, u.x, fmaf(gvy, u.y, gvz * u.z));
        dgv = fmaf(dot_gv_u * wv, horner4_val(p4, 132 + lane, tsp), dgv);
        if (LATER) {
            float glon = __ldg(gi + 224 + lane);
            float str = wv * horner4_val(p4, 180 + lane, tsp);
            float sl1 = wv * horner4_val(p4, 212 + lane, tsp) * (dot_gv_u + glon);
            dbx = fmaf(str, gvx, fmaf(sl1, u.x, dbx));
            dby = fmaf(str, gvy, fmaf(sl1, u.y, dby));
            dbz = fmaf(str, gvz, fmaf(sl1, u.z, dbz));
        }
        if (lane < 16) {
            float y2[5];
            eval_y2(u, y2);
            float wt = ws[lane >> 2];
            float gt0 = __ldg(gi + OUT_T_OFF + 5 * lane + 0), gt1 = __ldg(gi + OUT_T_OFF + 5 * lane + 1);
            float gt2 = __ldg(gi + OUT_T_OFF + 5 * lane + 2), gt3 = __ldg(gi + OUT_T_OFF + 5 * lane + 3), gt4 = __ldg(gi + OUT_T_OFF + 5 * lane + 4);
            float dot_gt_y2 = fmaf(gt0, y2[0], fmaf(gt1, y2[1], fmaf(gt2, y2[2], fmaf(gt3, y2[3], gt4 * y2[4]))));
            dgt = fmaf(dot_gt_y2 * wt, horner4_val(p4, 164 + lane, tsp), dgt);
            if (has_ten) {
                float glon2 = __ldg(gi + 336 + lane);
                float st2 = wt * horner4_val(p4, 244 + lane, tsp);
                float sl2 = wt * horner4_val(p4, 260 + lane, tsp) * (dot_gt_y2 + glon2);
                dtau[0] = fmaf(st2, gt0, fmaf(sl2, y2[0], dtau[0]));
                dtau[1] = fmaf(st2, gt1, fmaf(sl2, y2[1], dtau[1]));
                dtau[2] = fmaf(st2, gt2, fmaf(sl2, y2[2], dtau[2]));
                dtau[3] = fmaf(st2, gt3, fmaf(sl2, y2[3], dtau[3]));
                dtau[4] = fmaf(st2, gt4, fmaf(sl2, y2[4], dtau[4]));
            }
        }
        if (lane < 8) {
            float y3[7];
            eval_y3(u, y3);
            float dot_go_y3 = 0.0f;
            #pragma unroll
            for (int m = 0; m < 7; ++m) dot_go_y3 = fmaf(__ldg(gi + OUT_O_OFF + 7 * lane + m), y3[m], dot_go_y3);
            dgo = fmaf(dot_go_y3 * ws[lane >> 1], horner4_val(p4, F_O_OFF + lane, tsp), dgo);
        }
    }

    if (lane == 0) {
        d_pos[3 * j + 0] += dpos_jx;
        d_pos[3 * j + 1] += dpos_jy;
        d_pos[3 * j + 2] += dpos_jz;
        if (need_drho) d_rho[j] += drho_j;
    }
    if (!need_send_feat) return;
    float* dgj = d_gates + (long)184 * j;
    #pragma unroll
    for (int m = 0; m < 4; ++m) dgj[32 * m + lane] = dgs[m];
    dgj[128 + lane] = dgv;
    if (lane < 16) dgj[160 + lane] = dgt;
    if (lane < 8) dgj[176 + lane] = dgo;
    d_kn[64 * j + lane] = dk0;
    d_kn[64 * j + 32 + lane] = dk1;
    if (LATER) {
        d_vec[(long)96 * j + 3 * lane + 0] = dbx;
        d_vec[(long)96 * j + 3 * lane + 1] = dby;
        d_vec[(long)96 * j + 3 * lane + 2] = dbz;
        if (has_ten && lane < 16) {
            #pragma unroll
            for (int m = 0; m < 5; ++m) d_ten[(long)80 * j + 5 * lane + m] = dtau[m];
        }
    }
}

} // anonymous namespace
"""
