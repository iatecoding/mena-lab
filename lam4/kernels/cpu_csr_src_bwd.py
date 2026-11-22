"""C++ (CPU) Dual-CSR backward: Pass 1 (receiver-owned: d_pos_i, d_qn, per-edge grads, virial, d_rho_i) and Pass 2 (sender-owned:
d_gates, d_kn, d_vec, d_ten, d_pos_j, d_rho_j). Port of `metal_src_bwd.py`; no atomics, every row written by one thread."""
from __future__ import annotations

CPU_CSR_BWD = r"""
namespace lam4cpu {

template <typename T> struct BwdOut {
    T* d_pos; T* d_qn; T* edge_grad8; T* vir_atom; T* d_rho;
    T* d_gates; T* d_kn; T* d_vec; T* d_ten;
};

template <typename T, bool LATER>
void csr_bwd_recv(const Args<T>& a, const T* __restrict g_num, const T* __restrict g_zc, const T* __restrict g_dsum, const BwdOut<T>& o) {
    using L = Lay<LATER>;
    constexpr int TD = L::TD, OD = L::OD, F_O = L::F_O, OT = L::OT, OO = L::OO;
    const bool has_ten = LATER && (a.flags & FLAG_TEN);
    const bool has_rho = (a.flags & FLAG_RHO) != 0;
    const bool need_vir = (a.flags & FLAG_VIR) != 0, need_drho = (a.flags & FLAG_DRHO) != 0;
    #pragma omp parallel for schedule(dynamic, 4) num_threads(n_threads())
    for (int i = 0; i < a.n_atoms; ++i) {
        alignas(64) T tab[TD];
        alignas(64) T dtab[TD];
        const int g_id = graph_of(a, i), zi_shift = zshift_of(a, i);
        const T rho_i = has_rho ? a.rho[i] : T(0);
        const T* qi = a.qn + 64 * i;
        const T* gi = g_num + (size_t)OD * i;
        T dq[64];
        for (int c = 0; c < 64; ++c) dq[c] = T(0);
        T dpos_i[3] = {T(0), T(0), T(0)}, drho_i = T(0), vir[9];
        for (int m = 0; m < 9; ++m) vir[m] = T(0);
        const T gzc[4] = {g_zc[4 * i], g_zc[4 * i + 1], g_zc[4 * i + 2], g_zc[4 * i + 3]};
        const T gdsum_i = g_dsum[i];
        Geo<T> g;
        for (int e = a.recv_ptr[i], e_end = a.recv_ptr[i + 1]; e < e_end; ++e) {
            const int j = a.send_idx[e];
            T* eg = o.edge_grad8 + 8 * (size_t)e;
            if (!edge_geo(a, i, j, e, g_id, rho_i, zi_shift, true, g)) {
                for (int m = 0; m < 8; ++m) eg[m] = T(0);
                continue;
            }
            horner_val_der<T, TD>(a.poly + (size_t)(g.idx * 4) * TD, g.tsp, g.inv_h_eff, tab, dtab);
            const T* kj = a.kn + 64 * j;
            T dots[4], ex[4], th[4], w[4];
            qk_dots(qi, kj, dots);
            for (int h = 0; h < 4; ++h) {
                th[h] = std::tanh(T(0.5) * tab[h]);
                ex[h] = std::exp(dots[h] + T(2) * th[h]);
                w[h] = g.chi * ex[h];
            }
            const T* gj = a.gates + 184 * j;
            const T* u = g.u;
            T dw[4], dx = T(0), du[3] = {T(0), T(0), T(0)};
            // scalar channels: dL/dw_h and dL/dx, 8 fixed partial sums per head
            for (int h = 0; h < 4; ++h) {
                T pw[8], px[8];
                for (int l = 0; l < 8; ++l) { pw[l] = T(0); px[l] = T(0); }
                for (int c0 = 32 * h; c0 < 32 * h + 32; c0 += 8)
                    for (int l = 0; l < 8; ++l) {
                        const int c = c0 + l;
                        const T gg = gi[c] * gj[c];
                        pw[l] += gg * tab[4 + c];
                        px[l] += gg * dtab[4 + c];
                    }
                const T sw = ((pw[0] + pw[1]) + (pw[2] + pw[3])) + ((pw[4] + pw[5]) + (pw[6] + pw[7]));
                const T sx = ((px[0] + px[1]) + (px[2] + px[3])) + ((px[4] + px[5]) + (px[6] + px[7]));
                dw[h] = sw;
                dx += w[h] * sx;
            }
            // l=1 channels
            for (int c = 0; c < 32; ++c) {
                const int hv = c >> 3;
                const T wv = w[hv];
                const T* gv = gi + 128 + 3 * c;
                const T dot_gv_u = gv[0] * u[0] + gv[1] * u[1] + gv[2] * u[2];
                const T gv_j = gj[128 + c], w_dir = tab[132 + c] * gv_j;
                if (LATER) {
                    const T* b = a.vec + 96 * j + 3 * c;
                    const T f_tr = tab[180 + c], f_lon = tab[212 + c], glon = gi[224 + c];
                    const T dot_bu = b[0] * u[0] + b[1] * u[1] + b[2] * u[2];
                    const T dot_gv_b = gv[0] * b[0] + gv[1] * b[1] + gv[2] * b[2];
                    const T w_lon = f_lon * dot_bu;
                    dw[hv] += (w_dir + w_lon) * dot_gv_u + f_tr * dot_gv_b + w_lon * glon;
                    const T dw_lon = wv * (dot_gv_u + glon);
                    dx += (wv * dot_gv_u * gv_j) * dtab[132 + c] + (dw_lon * dot_bu) * dtab[212 + c] + (wv * dot_gv_b) * dtab[180 + c];
                    const T s1 = wv * (w_dir + w_lon), s2 = dw_lon * f_lon;
                    for (int m = 0; m < 3; ++m) du[m] += s1 * gv[m] + s2 * b[m];
                } else {
                    dw[hv] += w_dir * dot_gv_u;
                    dx += (wv * dot_gv_u * gv_j) * dtab[132 + c];
                    const T s1 = wv * w_dir;
                    for (int m = 0; m < 3; ++m) du[m] += s1 * gv[m];
                }
            }
            T y2[5], y3[7], gy2[5] = {T(0), T(0), T(0), T(0), T(0)}, gy3[7] = {T(0), T(0), T(0), T(0), T(0), T(0), T(0)};
            eval_y2(u, y2);
            eval_y3(u, y3);
            // l=2 channels
            for (int c = 0; c < 16; ++c) {
                const int ht = c >> 2;
                const T wt = w[ht];
                const T* gt = gi + OT + 5 * c;
                const T gt_j = gj[160 + c], w_ten = tab[164 + c] * gt_j;
                const T dot_gt_y2 = gt[0] * y2[0] + gt[1] * y2[1] + gt[2] * y2[2] + gt[3] * y2[3] + gt[4] * y2[4];
                if (has_ten) {
                    const T* tj = a.ten + 80 * j + 5 * c;
                    const T f_tr2 = tab[244 + c], f_lon2 = tab[260 + c], glon2 = gi[336 + c];
                    const T dot_ty2 = tj[0] * y2[0] + tj[1] * y2[1] + tj[2] * y2[2] + tj[3] * y2[3] + tj[4] * y2[4];
                    const T dot_gt_tau = gt[0] * tj[0] + gt[1] * tj[1] + gt[2] * tj[2] + gt[3] * tj[3] + gt[4] * tj[4];
                    const T w_lon2 = f_lon2 * dot_ty2, dw_lon2 = wt * (dot_gt_y2 + glon2);
                    dw[ht] += (w_ten + w_lon2) * dot_gt_y2 + f_tr2 * dot_gt_tau + w_lon2 * glon2;
                    dx += (wt * dot_gt_y2 * gt_j) * dtab[164 + c] + (dw_lon2 * dot_ty2) * dtab[260 + c] + (wt * dot_gt_tau) * dtab[244 + c];
                    const T sc = wt * (w_ten + w_lon2), st = dw_lon2 * f_lon2;
                    for (int m = 0; m < 5; ++m) gy2[m] += sc * gt[m] + st * tj[m];
                } else {
                    dw[ht] += w_ten * dot_gt_y2;
                    dx += (wt * dot_gt_y2 * gt_j) * dtab[164 + c];
                    const T sc = wt * w_ten;
                    for (int m = 0; m < 5; ++m) gy2[m] += sc * gt[m];
                }
            }
            // l=3 channels
            for (int c = 0; c < 8; ++c) {
                const int ho = c >> 1;
                const T wo = w[ho];
                const T* go = gi + OO + 7 * c;
                T dot_go_y3 = T(0);
                for (int m = 0; m < 7; ++m) dot_go_y3 += go[m] * y3[m];
                const T go_j = gj[176 + c], w_o = tab[F_O + c] * go_j;
                dw[ho] += w_o * dot_go_y3;
                dx += (wo * dot_go_y3 * go_j) * dtab[F_O + c];
                const T sc = wo * w_o;
                for (int m = 0; m < 7; ++m) gy3[m] += sc * go[m];
            }
            T gl[4], d_chi = gdsum_i;
            for (int h = 0; h < 4; ++h) {
                const T dwh = dw[h] + gzc[h];
                gl[h] = dwh * w[h];
                d_chi += dwh * ex[h];
                dx += gl[h] * (T(1) - th[h] * th[h]) * dtab[h];
            }
            const T d_rfl = d_chi * g.dchi + dx * g.dx_drfl;
            T gu[3] = {du[0], du[1], du[2]};
            grad_y2_u(u, gy2, gu);
            grad_y3_u(u, gy3, gu);
            const T gu_u = gu[0] * u[0] + gu[1] * u[1] + gu[2] * u[2];
            T gd[3];
            for (int m = 0; m < 3; ++m) gd[m] = (gu[m] - gu_u * u[m]) / g.r_raw + d_rfl * g.drfl_dd[m];
            for (int m = 0; m < 3; ++m) dpos_i[m] -= gd[m];
            for (int h = 0; h < 4; ++h)
                for (int c = 0; c < 16; ++c) dq[16 * h + c] += gl[h] * kj[16 * h + c];
            const T drho_e = need_drho ? dx * g.dx_drho : T(0);
            drho_i += drho_e;
            if (need_vir) {
                const T* s = a.shift_in + 3 * (size_t)e;
                for (int r = 0; r < 3; ++r)
                    for (int m = 0; m < 3; ++m) vir[3 * r + m] += s[r] * gd[m];
            }
            eg[0] = gd[0]; eg[1] = gd[1]; eg[2] = gd[2];
            eg[3] = gl[0]; eg[4] = gl[1]; eg[5] = gl[2]; eg[6] = gl[3]; eg[7] = drho_e;
        }
        for (int m = 0; m < 3; ++m) o.d_pos[3 * i + m] = dpos_i[m];
        for (int c = 0; c < 64; ++c) o.d_qn[64 * (size_t)i + c] = dq[c];
        if (need_drho) o.d_rho[i] = drho_i;
        if (need_vir) for (int m = 0; m < 9; ++m) o.vir_atom[9 * (size_t)i + m] = vir[m];
    }
}

template <typename T, bool LATER>
void csr_bwd_send(const Args<T>& a, const T* __restrict g_num, const BwdOut<T>& o) {
    using L = Lay<LATER>;
    constexpr int TD = L::TD, OD = L::OD, F_O = L::F_O, OT = L::OT, OO = L::OO;
    const bool has_ten = LATER && (a.flags & FLAG_TEN);
    const bool has_rho = (a.flags & FLAG_RHO) != 0, need_drho = (a.flags & FLAG_DRHO) != 0;
    #pragma omp parallel for schedule(dynamic, 4) num_threads(n_threads())
    for (int j = 0; j < a.n_atoms; ++j) {
        alignas(64) T tab[TD];
        alignas(64) T dgate[184];
        alignas(64) T db[96];
        alignas(64) T dtau[80];
        T dk[64];
        for (int c = 0; c < 184; ++c) dgate[c] = T(0);
        for (int c = 0; c < 96; ++c) db[c] = T(0);
        for (int c = 0; c < 80; ++c) dtau[c] = T(0);
        for (int c = 0; c < 64; ++c) dk[c] = T(0);
        T dpos_j[3] = {T(0), T(0), T(0)}, drho_j = T(0);
        const T* kj = a.kn + 64 * j;
        Geo<T> g;
        for (int kp = a.send_ptr[j], k_end = a.send_ptr[j + 1]; kp < k_end; ++kp) {
            const int e = a.send_perm[kp], i = a.recv_idx[e];
            const T rho_i = has_rho ? a.rho[i] : T(0);
            if (!edge_geo(a, i, j, e, graph_of(a, i), rho_i, zshift_of(a, i), false, g)) continue;
            const T* eg = o.edge_grad8 + 8 * (size_t)e;
            for (int m = 0; m < 3; ++m) dpos_j[m] += eg[m];
            if (need_drho) drho_j += eg[7];
            const T* qi = a.qn + 64 * i;
            for (int h = 0; h < 4; ++h)
                for (int c = 0; c < 16; ++c) dk[16 * h + c] += eg[3 + h] * qi[16 * h + c];
            horner_val<T, TD>(a.poly + (size_t)(g.idx * 4) * TD, g.tsp, tab);
            T dots[4], w[4];
            qk_dots(qi, kj, dots);
            for (int h = 0; h < 4; ++h) w[h] = g.chi * std::exp(dots[h] + T(2) * std::tanh(T(0.5) * tab[h]));
            const T* gi = g_num + (size_t)OD * i;
            const T* u = g.u;
            for (int h = 0; h < 4; ++h) {
                const T wh = w[h];
                for (int c = 32 * h; c < 32 * h + 32; ++c) dgate[c] += (gi[c] * tab[4 + c]) * wh;
            }
            for (int c = 0; c < 32; ++c) {
                const T wv = w[c >> 3];
                const T* gv = gi + 128 + 3 * c;
                const T dot_gv_u = gv[0] * u[0] + gv[1] * u[1] + gv[2] * u[2];
                dgate[128 + c] += (dot_gv_u * tab[132 + c]) * wv;
                if (LATER) {
                    const T s_tr = wv * tab[180 + c], s_lon = wv * tab[212 + c] * (dot_gv_u + gi[224 + c]);
                    for (int m = 0; m < 3; ++m) db[3 * c + m] += s_tr * gv[m] + s_lon * u[m];
                }
            }
            T y2[5], y3[7];
            eval_y2(u, y2);
            eval_y3(u, y3);
            for (int c = 0; c < 16; ++c) {
                const T wt = w[c >> 2];
                const T* gt = gi + OT + 5 * c;
                const T dot_gt_y2 = gt[0] * y2[0] + gt[1] * y2[1] + gt[2] * y2[2] + gt[3] * y2[3] + gt[4] * y2[4];
                dgate[160 + c] += (dot_gt_y2 * tab[164 + c]) * wt;
                if (has_ten) {
                    const T st2 = wt * tab[244 + c], sl2 = wt * tab[260 + c] * (dot_gt_y2 + gi[336 + c]);
                    for (int m = 0; m < 5; ++m) dtau[5 * c + m] += st2 * gt[m] + sl2 * y2[m];
                }
            }
            for (int c = 0; c < 8; ++c) {
                const T* go = gi + OO + 7 * c;
                T dot_go_y3 = T(0);
                for (int m = 0; m < 7; ++m) dot_go_y3 += go[m] * y3[m];
                dgate[176 + c] += (dot_go_y3 * tab[F_O + c]) * w[c >> 1];
            }
        }
        for (int m = 0; m < 3; ++m) o.d_pos[3 * j + m] += dpos_j[m];
        if (need_drho) o.d_rho[j] += drho_j;
        T* dg = o.d_gates + 184 * (size_t)j;
        for (int c = 0; c < 184; ++c) dg[c] = dgate[c];
        for (int c = 0; c < 64; ++c) o.d_kn[64 * (size_t)j + c] = dk[c];
        if (LATER) {
            for (int c = 0; c < 96; ++c) o.d_vec[96 * (size_t)j + c] = db[c];
            if (has_ten) for (int c = 0; c < 80; ++c) o.d_ten[80 * (size_t)j + c] = dtau[c];
        }
    }
}

}  // namespace lam4cpu
"""
