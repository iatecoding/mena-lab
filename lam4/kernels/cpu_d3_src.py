"""C++ / OpenMP source for rank-1 factorized CPU Grimme D3(BJ) & D3(0) dispersion (FP32 and FP64)."""
from __future__ import annotations

CPU_D3_SOURCE = r"""
#include <torch/extension.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lam4cpud3 {

constexpr double ANG_TO_BOHR  = 1.8897261245650618;
constexpr double HARTREE_KCAL = 627.509474;
constexpr double D3_K1        = 16.0;
constexpr double D3_K3        = -4.0;

inline int n_threads() {
    int nt = at::get_num_threads();
    return nt > 0 ? nt : 1;
}

template <typename T>
inline void write_cn_weights(
    int i, int zi, T cn_val,
    const int* n_ref, const T* cn_ref,
    T* out_pw, T* out_dpw
) {
    int nr = n_ref[zi];
    T* pw_i = out_pw + 5 * i;
    T* dpw_i = out_dpw + 5 * i;
    if (nr <= 0) {
        for (int a = 0; a < 5; ++a) { pw_i[a] = T(0); dpw_i[a] = T(0); }
        return;
    }
    const T* cr = cn_ref + 5 * zi;
    T d[5], w[5];
    T smax = T(-1.0e20);
    for (int a = 0; a < nr; ++a) {
        d[a] = cr[a] - cn_val;
        T s = T(D3_K3) * d[a] * d[a];
        if (s > smax) smax = s;
    }
    T sw = T(0);
    for (int a = 0; a < nr; ++a) {
        w[a] = std::exp(T(D3_K3) * d[a] * d[a] - smax);
        sw += w[a];
    }
    T inv_sw = T(1) / (sw > T(1.0e-30) ? sw : T(1.0e-30));
    T dbar = T(0);
    for (int a = 0; a < nr; ++a) {
        w[a] *= inv_sw;
        dbar += w[a] * d[a];
    }
    for (int a = 0; a < 5; ++a) {
        if (a < nr) {
            pw_i[a] = w[a];
            dpw_i[a] = T(-2.0 * D3_K3) * w[a] * (d[a] - dbar);
        } else {
            pw_i[a] = T(0);
            dpw_i[a] = T(0);
        }
    }
}

template <typename T>
inline void accum_pair_r1(
    T dx, T dy, T dz, T d2, int zi, int zj, int ni, int nj,
    const T* pw_i, const T* dpw_i, const T* pw_j,
    T r2r4_i, const T* r2r4, const T* c6_mat, const T* r0ab,
    T s6, T s8, T p_a1, T p_a2, T r_cut, T taper_on, T inv_tspan,
    bool is_zero, bool need_deriv, bool need_forces, bool need_stress,
    double& acc_e, T& acc_dedcn, T& fx, T& fy, T& fz,
    double& vxx, double& vyy, double& vzz, double& vxy, double& vxz, double& vyz
) {
    if (d2 > r_cut * r_cut || ni <= 0 || nj <= 0) return;
    T r2_eps = d2 + T(1.0e-8);
    T r_ang = std::sqrt(r2_eps);
    if (r_ang > r_cut) return;
    T inv_r = T(1) / r_ang;

    const T* tab = c6_mat + 25 * (zi * 95 + zj);
    T c6 = T(0), dc6 = T(0);
    for (int a = 0; a < ni; ++a) {
        const T* row = tab + a * 5;
        T s = T(0);
        for (int b = 0; b < nj; ++b) s += pw_j[b] * row[b];
        c6 += pw_i[a] * s;
        if (need_deriv) dc6 += dpw_i[a] * s;
    }

    T q_ij = T(3) * r2r4_i * r2r4[zj];
    T r_b = r_ang * T(ANG_TO_BOHR);
    T rb2 = r_b * r_b, rb4 = rb2 * rb2;
    T rb6 = rb4 * rb2, rb8 = rb4 * rb4;
    T A, dA_dr = T(0);
    if (is_zero) {
        T r0_raw = r0ab[zi * 95 + zj];
        T r0 = r0_raw > T(1.0e-6) ? r0_raw : T(1.0e-6);
        T rb_s = r_b > T(1.0e-6) ? r_b : T(1.0e-6);
        T u6 = (p_a1 * r0) / rb_s, u8 = r0 / rb_s;
        T u6_2 = u6 * u6, u8_2 = u8 * u8;
        T u6_4 = u6_2 * u6_2, u8_4 = u8_2 * u8_2;
        T g6 = T(6) * (u6_4 * u6_4 * u6_4 * u6_2);
        T g8 = T(6) * (u8_4 * u8_4 * u8_4 * u8_4);
        T f6 = T(1) / (T(1) + g6), f8 = T(1) / (T(1) + g8);
        T rb6_s = rb6 > T(1.0e-12) ? rb6 : T(1.0e-12);
        T rb8_s = rb8 > T(1.0e-12) ? rb8 : T(1.0e-12);
        T t6_v = s6 * f6 / rb6_s;
        T t8_v = (s8 * q_ij) * f8 / rb8_s;
        A = t6_v + t8_v;
        if (need_deriv) {
            T p6 = rb6 > T(1.0e-12) ? T(6) : T(0);
            T p8 = rb8 > T(1.0e-12) ? T(8) : T(0);
            dA_dr = T(ANG_TO_BOHR) * ((t6_v * (T(14) * g6 * f6 - p6) + t8_v * (T(16) * g8 * f8 - p8)) / rb_s);
        }
    } else {
        T tmp = p_a1 * std::sqrt(q_ij > T(1.0e-12) ? q_ij : T(1.0e-12)) + p_a2;
        T t2 = tmp * tmp, t4 = t2 * t2;
        T e6 = T(1) / (rb6 + t4 * t2);
        T e8 = T(1) / (rb8 + t4 * t4);
        A = s6 * e6 + (s8 * q_ij) * e8;
        if (need_deriv) {
            dA_dr = -T(ANG_TO_BOHR) * (T(6) * s6 * (rb4 * r_b) * (e6 * e6) + T(8) * (s8 * q_ij) * (rb6 * r_b) * (e8 * e8));
        }
    }

    T t = (r_ang - taper_on) * inv_tspan;
    if (t < T(0)) t = T(0);
    else if (t > T(1)) t = T(1);
    T taper = T(1) - t * t * t * (T(10) + t * (T(6) * t - T(15)));
    acc_e += (-0.5 * HARTREE_KCAL) * double(c6) * double(A) * double(taper);

    if (need_deriv) {
        acc_dedcn += (-T(HARTREE_KCAL)) * dc6 * A * taper;
        T dtap = (r_ang > taper_on && r_ang < r_cut) ? (T(-30) * (t * t) * ((T(1) - t) * (T(1) - t)) * inv_tspan) : T(0);
        T fm = T(HARTREE_KCAL) * c6 * (dA_dr * taper + A * dtap) * inv_r;
        if (need_forces) { fx += fm * dx; fy += fm * dy; fz += fm * dz; }
        if (need_stress) {
            double vm = -0.5 * double(fm);
            double ddx = double(dx), ddy = double(dy), ddz = double(dz);
            vxx += vm * ddx * ddx; vyy += vm * ddy * ddy; vzz += vm * ddz * ddz;
            vxy += vm * ddx * ddy; vxz += vm * ddx * ddz; vyz += vm * ddy * ddz;
        }
    }
}

template <typename T>
void run_grouped(
    int n, const T* pos, const int64_t* z, const uint8_t* mask,
    const int64_t* graph, const int* group_ptr,
    const T* rcov, const T* r2r4, const int* n_ref, const T* cn_ref, const T* c6_mat, const T* r0ab,
    T s6, T s8, T p_a1, T p_a2, bool is_zero, bool need_forces,
    double* out_e, T* out_f
) {
    const T r_cut = is_zero ? T(20.0) : T(12.0);
    const T taper_on = is_zero ? T(20.0) : T(9.0);
    const T inv_tspan = is_zero ? T(0.0) : T(1.0 / 3.0);
    const T cn_cut_a = is_zero ? T(20.0) : T(11.2);
    const T cn_cut2 = cn_cut_a * cn_cut_a + T(1.0e-3);
    const T cn_cut_b = cn_cut_a * T(ANG_TO_BOHR);
    const bool par = (n >= 32);
    const int nt = n_threads();

    std::vector<T> pw(5 * n, T(0)), dpw(5 * n, T(0));
    std::vector<T> dedcn(need_forces ? n : 1, T(0));

    #pragma omp parallel for schedule(static) num_threads(nt) if(par)
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0) {
            write_cn_weights<T>(i, 0, T(0), n_ref, cn_ref, pw.data(), dpw.data());
            continue;
        }
        int zi = int(std::clamp<int64_t>(z[i], 0, 94));
        T rc_i = rcov[zi];
        T pix = pos[3 * i], piy = pos[3 * i + 1], piz = pos[3 * i + 2];
        int g = graph ? int(graph[i]) : 0;
        int j0 = group_ptr[g], j1 = group_ptr[g + 1];
        T acc = T(0);
        for (int j = j0; j < j1; ++j) {
            if (j == i || (mask && mask[j] == 0)) continue;
            T dx = pix - pos[3 * j], dy = piy - pos[3 * j + 1], dz = piz - pos[3 * j + 2];
            T d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > cn_cut2) continue;
            T r_b = std::sqrt(d2 + T(1.0e-8)) * T(ANG_TO_BOHR);
            if (r_b <= cn_cut_b) {
                int zj = int(std::clamp<int64_t>(z[j], 0, 94));
                T rr = (rc_i + rcov[zj]) / (r_b > T(1.0e-6) ? r_b : T(1.0e-6));
                acc += T(1) / (T(1) + std::exp(T(-D3_K1) * (rr - T(1))));
            }
        }
        write_cn_weights<T>(i, zi, acc, n_ref, cn_ref, pw.data(), dpw.data());
    }

    #pragma omp parallel for schedule(static) num_threads(nt) if(par)
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0) {
            out_e[i] = 0.0;
            if (need_forces) { dedcn[i] = T(0); out_f[3 * i] = out_f[3 * i + 1] = out_f[3 * i + 2] = T(0); }
            continue;
        }
        int zi = int(std::clamp<int64_t>(z[i], 0, 94));
        int ni = n_ref[zi];
        T r2r4_i = r2r4[zi];
        const T* pw_i = pw.data() + 5 * i;
        const T* dpw_i = dpw.data() + 5 * i;
        T pix = pos[3 * i], piy = pos[3 * i + 1], piz = pos[3 * i + 2];
        int g = graph ? int(graph[i]) : 0;
        int j0 = group_ptr[g], j1 = group_ptr[g + 1];
        double acc_e = 0.0, vxx = 0, vyy = 0, vzz = 0, vxy = 0, vxz = 0, vyz = 0;
        T acc_de = T(0), fx = T(0), fy = T(0), fz = T(0);
        for (int j = j0; j < j1; ++j) {
            if (j == i || (mask && mask[j] == 0)) continue;
            T dx = pix - pos[3 * j], dy = piy - pos[3 * j + 1], dz = piz - pos[3 * j + 2];
            T d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > r_cut * r_cut) continue;
            int zj = int(std::clamp<int64_t>(z[j], 0, 94));
            accum_pair_r1<T>(dx, dy, dz, d2, zi, zj, ni, n_ref[zj], pw_i, dpw_i, pw.data() + 5 * j,
                             r2r4_i, r2r4, c6_mat, r0ab, s6, s8, p_a1, p_a2, r_cut, taper_on, inv_tspan,
                             is_zero, need_forces, need_forces, false,
                             acc_e, acc_de, fx, fy, fz, vxx, vyy, vzz, vxy, vxz, vyz);
        }
        out_e[i] = acc_e;
        if (need_forces) {
            dedcn[i] = acc_de;
            out_f[3 * i] = fx; out_f[3 * i + 1] = fy; out_f[3 * i + 2] = fz;
        }
    }

    if (!need_forces) return;

    #pragma omp parallel for schedule(static) num_threads(nt) if(par)
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0) continue;
        int zi = int(std::clamp<int64_t>(z[i], 0, 94));
        T rc_i = rcov[zi], de_i = dedcn[i];
        T pix = pos[3 * i], piy = pos[3 * i + 1], piz = pos[3 * i + 2];
        int g = graph ? int(graph[i]) : 0;
        int j0 = group_ptr[g], j1 = group_ptr[g + 1];
        T fx = out_f[3 * i], fy = out_f[3 * i + 1], fz = out_f[3 * i + 2];
        for (int j = j0; j < j1; ++j) {
            if (j == i || (mask && mask[j] == 0)) continue;
            T dx = pix - pos[3 * j], dy = piy - pos[3 * j + 1], dz = piz - pos[3 * j + 2];
            T d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > cn_cut2) continue;
            T r2_eps = d2 + T(1.0e-8);
            T r_ang = std::sqrt(r2_eps);
            T r_b = r_ang * T(ANG_TO_BOHR);
            if (r_b <= cn_cut_b && r_b > T(1.0e-6)) {
                int zj = int(std::clamp<int64_t>(z[j], 0, 94));
                T rr = (rc_i + rcov[zj]) / r_b;
                T damp = T(1) / (T(1) + std::exp(T(-D3_K1) * (rr - T(1))));
                T fm = (de_i + dedcn[j]) * T(D3_K1) * damp * (T(1) - damp) * rr / r2_eps;
                fx += fm * dx; fy += fm * dy; fz += fm * dz;
            }
        }
        out_f[3 * i] = fx; out_f[3 * i + 1] = fy; out_f[3 * i + 2] = fz;
    }
}

template <typename T>
void run_csr(
    int n, const T* pos, const int64_t* z, const uint8_t* mask,
    const int64_t* graph, const int* recv_ptr, const int* send_idx,
    const T* shift, const T* cell, bool multi_cell,
    const T* rcov, const T* r2r4, const int* n_ref, const T* cn_ref, const T* c6_mat, const T* r0ab,
    T s6, T s8, T p_a1, T p_a2, bool is_zero, bool need_forces, bool need_stress,
    double* out_e, T* out_f, double* out_vir
) {
    const T r_cut = is_zero ? T(20.0) : T(12.0);
    const T taper_on = is_zero ? T(20.0) : T(9.0);
    const T inv_tspan = is_zero ? T(0.0) : T(1.0 / 3.0);
    const T cn_cut_a = is_zero ? T(20.0) : T(11.2);
    const T cn_cut2 = cn_cut_a * cn_cut_a + T(1.0e-3);
    const T cn_cut_b = cn_cut_a * T(ANG_TO_BOHR);
    const bool need_deriv = need_forces || need_stress;
    const bool has_shift = (shift != nullptr);
    const bool has_cell = (has_shift && cell != nullptr);
    const bool par = (n >= 32);
    const int nt = n_threads();

    std::vector<T> pw(5 * n, T(0)), dpw(5 * n, T(0));
    std::vector<T> dedcn(need_deriv ? n : 1, T(0));

    #pragma omp parallel for schedule(guided, 16) num_threads(nt) if(par)
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0) {
            write_cn_weights<T>(i, 0, T(0), n_ref, cn_ref, pw.data(), dpw.data());
            continue;
        }
        int zi = int(std::clamp<int64_t>(z[i], 0, 94));
        T rc_i = rcov[zi];
        T pix = pos[3 * i], piy = pos[3 * i + 1], piz = pos[3 * i + 2];
        const T* c = has_cell ? (cell + 9 * (multi_cell ? int(graph[i]) : 0)) : nullptr;
        T acc = T(0);
        for (int e = recv_ptr[i], e1 = recv_ptr[i + 1]; e < e1; ++e) {
            int j = send_idx[e];
            if (mask && mask[j] == 0) continue;
            T dx = pix - pos[3 * j], dy = piy - pos[3 * j + 1], dz = piz - pos[3 * j + 2];
            if (has_shift) {
                T sx = shift[3 * e], sy = shift[3 * e + 1], sz = shift[3 * e + 2];
                if (has_cell) {
                    dx -= sx * c[0] + sy * c[3] + sz * c[6];
                    dy -= sx * c[1] + sy * c[4] + sz * c[7];
                    dz -= sx * c[2] + sy * c[5] + sz * c[8];
                } else { dx -= sx; dy -= sy; dz -= sz; }
            }
            T d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= T(1.0e-12) || d2 > cn_cut2) continue;
            T r_b = std::sqrt(d2 + T(1.0e-8)) * T(ANG_TO_BOHR);
            if (r_b <= cn_cut_b) {
                int zj = int(std::clamp<int64_t>(z[j], 0, 94));
                T rr = (rc_i + rcov[zj]) / (r_b > T(1.0e-6) ? r_b : T(1.0e-6));
                acc += T(1) / (T(1) + std::exp(T(-D3_K1) * (rr - T(1))));
            }
        }
        write_cn_weights<T>(i, zi, acc, n_ref, cn_ref, pw.data(), dpw.data());
    }

    #pragma omp parallel for schedule(guided, 16) num_threads(nt) if(par)
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0) {
            out_e[i] = 0.0;
            if (need_deriv) dedcn[i] = T(0);
            if (need_forces) { out_f[3 * i] = out_f[3 * i + 1] = out_f[3 * i + 2] = T(0); }
            if (need_stress) { for (int m = 0; m < 6; ++m) out_vir[6 * i + m] = 0.0; }
            continue;
        }
        int zi = int(std::clamp<int64_t>(z[i], 0, 94));
        int ni = n_ref[zi];
        T r2r4_i = r2r4[zi];
        const T* pw_i = pw.data() + 5 * i;
        const T* dpw_i = dpw.data() + 5 * i;
        T pix = pos[3 * i], piy = pos[3 * i + 1], piz = pos[3 * i + 2];
        const T* c = has_cell ? (cell + 9 * (multi_cell ? int(graph[i]) : 0)) : nullptr;
        double acc_e = 0.0, vxx = 0, vyy = 0, vzz = 0, vxy = 0, vxz = 0, vyz = 0;
        T acc_de = T(0), fx = T(0), fy = T(0), fz = T(0);
        for (int e = recv_ptr[i], e1 = recv_ptr[i + 1]; e < e1; ++e) {
            int j = send_idx[e];
            if (mask && mask[j] == 0) continue;
            T dx = pix - pos[3 * j], dy = piy - pos[3 * j + 1], dz = piz - pos[3 * j + 2];
            if (has_shift) {
                T sx = shift[3 * e], sy = shift[3 * e + 1], sz = shift[3 * e + 2];
                if (has_cell) {
                    dx -= sx * c[0] + sy * c[3] + sz * c[6];
                    dy -= sx * c[1] + sy * c[4] + sz * c[7];
                    dz -= sx * c[2] + sy * c[5] + sz * c[8];
                } else { dx -= sx; dy -= sy; dz -= sz; }
            }
            T d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= T(1.0e-12) || d2 > r_cut * r_cut) continue;
            int zj = int(std::clamp<int64_t>(z[j], 0, 94));
            accum_pair_r1<T>(dx, dy, dz, d2, zi, zj, ni, n_ref[zj], pw_i, dpw_i, pw.data() + 5 * j,
                             r2r4_i, r2r4, c6_mat, r0ab, s6, s8, p_a1, p_a2, r_cut, taper_on, inv_tspan,
                             is_zero, need_deriv, need_forces, need_stress,
                             acc_e, acc_de, fx, fy, fz, vxx, vyy, vzz, vxy, vxz, vyz);
        }
        out_e[i] = acc_e;
        if (need_deriv) dedcn[i] = acc_de;
        if (need_forces) { out_f[3 * i] = fx; out_f[3 * i + 1] = fy; out_f[3 * i + 2] = fz; }
        if (need_stress) {
            out_vir[6 * i + 0] = vxx; out_vir[6 * i + 1] = vyy; out_vir[6 * i + 2] = vzz;
            out_vir[6 * i + 3] = vxy; out_vir[6 * i + 4] = vxz; out_vir[6 * i + 5] = vyz;
        }
    }

    if (!need_deriv) return;

    #pragma omp parallel for schedule(guided, 16) num_threads(nt) if(par)
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] == 0) continue;
        int zi = int(std::clamp<int64_t>(z[i], 0, 94));
        T rc_i = rcov[zi], de_i = dedcn[i];
        T pix = pos[3 * i], piy = pos[3 * i + 1], piz = pos[3 * i + 2];
        const T* c = has_cell ? (cell + 9 * (multi_cell ? int(graph[i]) : 0)) : nullptr;
        T fx = need_forces ? out_f[3 * i] : T(0);
        T fy = need_forces ? out_f[3 * i + 1] : T(0);
        T fz = need_forces ? out_f[3 * i + 2] : T(0);
        double vxx = 0, vyy = 0, vzz = 0, vxy = 0, vxz = 0, vyz = 0;
        for (int e = recv_ptr[i], e1 = recv_ptr[i + 1]; e < e1; ++e) {
            int j = send_idx[e];
            if (mask && mask[j] == 0) continue;
            T dx = pix - pos[3 * j], dy = piy - pos[3 * j + 1], dz = piz - pos[3 * j + 2];
            if (has_shift) {
                T sx = shift[3 * e], sy = shift[3 * e + 1], sz = shift[3 * e + 2];
                if (has_cell) {
                    dx -= sx * c[0] + sy * c[3] + sz * c[6];
                    dy -= sx * c[1] + sy * c[4] + sz * c[7];
                    dz -= sx * c[2] + sy * c[5] + sz * c[8];
                } else { dx -= sx; dy -= sy; dz -= sz; }
            }
            T d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= T(1.0e-12) || d2 > cn_cut2) continue;
            T r2_eps = d2 + T(1.0e-8);
            T r_ang = std::sqrt(r2_eps);
            T r_b = r_ang * T(ANG_TO_BOHR);
            if (r_b <= cn_cut_b && r_b > T(1.0e-6)) {
                int zj = int(std::clamp<int64_t>(z[j], 0, 94));
                T rr = (rc_i + rcov[zj]) / r_b;
                T damp = T(1) / (T(1) + std::exp(T(-D3_K1) * (rr - T(1))));
                T fm = (de_i + dedcn[j]) * T(D3_K1) * damp * (T(1) - damp) * rr / r2_eps;
                if (need_forces) { fx += fm * dx; fy += fm * dy; fz += fm * dz; }
                if (need_stress) {
                    double vm = -0.5 * double(fm);
                    double ddx = double(dx), ddy = double(dy), ddz = double(dz);
                    vxx += vm * ddx * ddx; vyy += vm * ddy * ddy; vzz += vm * ddz * ddz;
                    vxy += vm * ddx * ddy; vxz += vm * ddx * ddz; vyz += vm * ddy * ddz;
                }
            }
        }
        if (need_forces) { out_f[3 * i] = fx; out_f[3 * i + 1] = fy; out_f[3 * i + 2] = fz; }
        if (need_stress) {
            out_vir[6 * i + 0] += vxx; out_vir[6 * i + 1] += vyy; out_vir[6 * i + 2] += vzz;
            out_vir[6 * i + 3] += vxy; out_vir[6 * i + 4] += vxz; out_vir[6 * i + 5] += vyz;
        }
    }
}

std::vector<at::Tensor> d3_grouped(
    at::Tensor pos, at::Tensor z, at::Tensor mask, at::Tensor graph, at::Tensor group_ptr,
    at::Tensor rcov, at::Tensor r2r4, at::Tensor n_ref, at::Tensor cn_ref, at::Tensor c6_mat, at::Tensor r0ab,
    double s6, double s8, double p_a1, double p_a2, bool is_zero, bool need_forces
) {
    const int64_t n = pos.size(0);
    at::Tensor out_e = at::empty({n}, pos.options().dtype(at::kDouble));
    at::Tensor out_f = need_forces ? at::empty({n, 3}, pos.options()) : at::empty({0}, pos.options());
    const uint8_t* mptr = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
    const int64_t* gptr = graph.numel() ? graph.data_ptr<int64_t>() : nullptr;
    if (pos.scalar_type() == at::kDouble) {
        run_grouped<double>(
            (int)n, pos.data_ptr<double>(), z.data_ptr<int64_t>(), mptr, gptr, group_ptr.data_ptr<int>(),
            rcov.data_ptr<double>(), r2r4.data_ptr<double>(), n_ref.data_ptr<int>(),
            cn_ref.data_ptr<double>(), c6_mat.data_ptr<double>(), r0ab.data_ptr<double>(),
            s6, s8, p_a1, p_a2, is_zero, need_forces,
            out_e.data_ptr<double>(), need_forces ? out_f.data_ptr<double>() : nullptr);
    } else {
        TORCH_CHECK(pos.scalar_type() == at::kFloat, "lam4 cpu_d3: float32 or float64 only");
        run_grouped<float>(
            (int)n, pos.data_ptr<float>(), z.data_ptr<int64_t>(), mptr, gptr, group_ptr.data_ptr<int>(),
            rcov.data_ptr<float>(), r2r4.data_ptr<float>(), n_ref.data_ptr<int>(),
            cn_ref.data_ptr<float>(), c6_mat.data_ptr<float>(), r0ab.data_ptr<float>(),
            (float)s6, (float)s8, (float)p_a1, (float)p_a2, is_zero, need_forces,
            out_e.data_ptr<double>(), need_forces ? out_f.data_ptr<float>() : nullptr);
    }
    return {out_e, out_f};
}

std::vector<at::Tensor> d3_csr(
    at::Tensor pos, at::Tensor z, at::Tensor mask, at::Tensor graph,
    at::Tensor recv_ptr, at::Tensor send_idx, at::Tensor shift, at::Tensor cell, bool multi_cell,
    at::Tensor rcov, at::Tensor r2r4, at::Tensor n_ref, at::Tensor cn_ref, at::Tensor c6_mat, at::Tensor r0ab,
    double s6, double s8, double p_a1, double p_a2, bool is_zero, bool need_forces, bool need_stress
) {
    const int64_t n = pos.size(0);
    at::Tensor out_e = at::empty({n}, pos.options().dtype(at::kDouble));
    at::Tensor out_f = need_forces ? at::empty({n, 3}, pos.options()) : at::empty({0}, pos.options());
    at::Tensor out_vir = need_stress ? at::empty({n, 6}, pos.options().dtype(at::kDouble)) : at::empty({0}, pos.options().dtype(at::kDouble));
    const uint8_t* mptr = mask.numel() ? mask.data_ptr<uint8_t>() : nullptr;
    const int64_t* gptr = graph.numel() ? graph.data_ptr<int64_t>() : nullptr;
    if (pos.scalar_type() == at::kDouble) {
        run_csr<double>(
            (int)n, pos.data_ptr<double>(), z.data_ptr<int64_t>(), mptr, gptr,
            recv_ptr.data_ptr<int>(), send_idx.data_ptr<int>(),
            shift.numel() ? shift.data_ptr<double>() : nullptr,
            cell.numel() ? cell.data_ptr<double>() : nullptr, multi_cell,
            rcov.data_ptr<double>(), r2r4.data_ptr<double>(), n_ref.data_ptr<int>(),
            cn_ref.data_ptr<double>(), c6_mat.data_ptr<double>(), r0ab.data_ptr<double>(),
            s6, s8, p_a1, p_a2, is_zero, need_forces, need_stress,
            out_e.data_ptr<double>(), need_forces ? out_f.data_ptr<double>() : nullptr,
            need_stress ? out_vir.data_ptr<double>() : nullptr);
    } else {
        TORCH_CHECK(pos.scalar_type() == at::kFloat, "lam4 cpu_d3: float32 or float64 only");
        run_csr<float>(
            (int)n, pos.data_ptr<float>(), z.data_ptr<int64_t>(), mptr, gptr,
            recv_ptr.data_ptr<int>(), send_idx.data_ptr<int>(),
            shift.numel() ? shift.data_ptr<float>() : nullptr,
            cell.numel() ? cell.data_ptr<float>() : nullptr, multi_cell,
            rcov.data_ptr<float>(), r2r4.data_ptr<float>(), n_ref.data_ptr<int>(),
            cn_ref.data_ptr<float>(), c6_mat.data_ptr<float>(), r0ab.data_ptr<float>(),
            (float)s6, (float)s8, (float)p_a1, (float)p_a2, is_zero, need_forces, need_stress,
            out_e.data_ptr<double>(), need_forces ? out_f.data_ptr<float>() : nullptr,
            need_stress ? out_vir.data_ptr<double>() : nullptr);
    }
    return {out_e, out_f, out_vir};
}

}  // namespace lam4cpud3

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("d3_grouped", &lam4cpud3::d3_grouped, py::call_guard<py::gil_scoped_release>());
    m.def("d3_csr", &lam4cpud3::d3_csr, py::call_guard<py::gil_scoped_release>());
}
"""
