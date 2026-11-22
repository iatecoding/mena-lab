"""C++ (CPU) Dual-CSR edge kernel: common math + receiver-owned forward. A CPU port of `metal_src.py` (same flags, plan layout
and column offsets), templated on float / double, one OpenMP task per receiver block, channel loops written for clang -O3
auto-vectorisation (no -ffast-math: reductions use explicit partial sums, so the summation order is fixed by this source)."""
from __future__ import annotations

CPU_CSR_COMMON = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <omp.h>

namespace lam4cpu {

constexpr int FLAG_FLOOR = 1, FLAG_RHO = 2, FLAG_TEN = 4, FLAG_CELL = 8, FLAG_G1 = 16, FLAG_VIR = 32, FLAG_DRHO = 64;
constexpr double S2 = 1.4142135623730951, INV_S2 = 0.7071067811865475, INV_S6 = 0.4082482904638630;
constexpr double C3_0 = 0.5, C3_1 = 2.449489742783178, C3_2 = 0.3872983346207417, C3_3 = 0.31622776601683794;
constexpr double C3_4 = 0.3872983346207417, C3_5 = 1.224744871391589, C3_6 = 0.5;
constexpr double WARP_NUM = 1.52, FLOOR_W = 0.05, R_EPS2 = 1e-8;

template <bool LATER> struct Lay {
    static constexpr int TD = LATER ? 284 : 188;      // table columns (4 head biases + filters)
    static constexpr int OD = LATER ? 408 : 360;      // output columns
    static constexpr int F_O = LATER ? 276 : 180;     // l=3 filter column
    static constexpr int OT = LATER ? 256 : 224;      // l=2 output offset
    static constexpr int OO = LATER ? 352 : 304;      // l=3 output offset
};

template <typename T> struct Args {
    const T* pos; const int* recv_ptr; const int* send_idx; const int* send_ptr; const int* send_perm; const int* recv_idx;
    const T* shift_in; const T* cell; const int64_t* graph; const int64_t* z; const T* rmin; const T* rho;
    const T* poly; const T* gates; const T* qn; const T* kn; const T* vec; const T* ten;
    int n_atoms; int grid; int flags; T r_cut; T r_on; T max_doff_r2;
};

template <typename T> struct Geo {
    T d[3]; T r_raw; T u[3]; T drfl_dd[3]; T chi; T dchi; T dx_drfl; T dx_drho; T tsp; T inv_h_eff; int idx;
};

template <typename T> inline void eval_y2(const T* u, T* y2) {
    T x = u[0], y = u[1], z = u[2];
    y2[0] = T(S2) * x * y; y2[1] = T(S2) * x * z; y2[2] = T(S2) * y * z;
    y2[3] = (x * x - y * y) * T(INV_S2);
    y2[4] = (T(2) * z * z - x * x - y * y) * T(INV_S6);
}

template <typename T> inline void eval_y3(const T* u, T* y3) {
    T x = u[0], y = u[1], z = u[2], xx = x * x, yy = y * y, zz = z * z;
    y3[0] = T(C3_0) * y * (T(3) * xx - yy);
    y3[1] = T(C3_1) * x * y * z;
    y3[2] = T(C3_2) * y * (T(4) * zz - xx - yy);
    y3[3] = T(C3_3) * z * (T(2) * zz - T(3) * xx - T(3) * yy);
    y3[4] = T(C3_4) * x * (T(4) * zz - xx - yy);
    y3[5] = T(C3_5) * z * (xx - yy);
    y3[6] = T(C3_6) * x * (xx - T(3) * yy);
}

template <typename T> inline void grad_y2_u(const T* u, const T* g, T* out) {
    T x = u[0], y = u[1], z = u[2], k6 = T(2) * T(INV_S6);
    out[0] += T(S2) * (g[0] * y + g[1] * z + g[3] * x) - k6 * g[4] * x;
    out[1] += T(S2) * (g[0] * x + g[2] * z - g[3] * y) - k6 * g[4] * y;
    out[2] += T(S2) * (g[1] * x + g[2] * y) + T(2) * k6 * g[4] * z;
}

template <typename T> inline void grad_y3_u(const T* u, const T* g, T* out) {
    T x = u[0], y = u[1], z = u[2], xx = x * x, yy = y * y, zz = z * z;
    out[0] += g[0] * (T(C3_0) * T(6) * x * y) + g[1] * (T(C3_1) * y * z) + g[2] * (T(-2) * T(C3_2) * x * y)
            + g[3] * (T(-6) * T(C3_3) * x * z) + g[4] * (T(C3_4) * (T(4) * zz - T(3) * xx - yy))
            + g[5] * (T(2) * T(C3_5) * x * z) + g[6] * (T(C3_6) * (T(3) * xx - T(3) * yy));
    out[1] += g[0] * (T(C3_0) * (T(3) * xx - T(3) * yy)) + g[1] * (T(C3_1) * x * z)
            + g[2] * (T(C3_2) * (T(4) * zz - xx - T(3) * yy)) + g[3] * (T(-6) * T(C3_3) * y * z)
            + g[4] * (T(-2) * T(C3_4) * x * y) + g[5] * (T(-2) * T(C3_5) * y * z) + g[6] * (T(-6) * T(C3_6) * x * y);
    out[2] += g[1] * (T(C3_1) * x * y) + g[2] * (T(8) * T(C3_2) * y * z)
            + g[3] * (T(C3_3) * (T(6) * zz - T(3) * xx - T(3) * yy))
            + g[4] * (T(8) * T(C3_4) * x * z) + g[5] * (T(C3_5) * (xx - yy));
}

// Horner cubic on every table column: f[c] = ((a3 t + a2) t + a1) t + a0, rows a_k at p0 + k * TD.
template <typename T, int TD> inline void horner_val(const T* __restrict p0, T t, T* __restrict f) {
    const T* a0 = p0; const T* a1 = p0 + TD; const T* a2 = p0 + 2 * TD; const T* a3 = p0 + 3 * TD;
    for (int c = 0; c < TD; ++c) f[c] = ((a3[c] * t + a2[c]) * t + a1[c]) * t + a0[c];
}

template <typename T, int TD> inline void horner_val_der(const T* __restrict p0, T t, T inv_h, T* __restrict f, T* __restrict df) {
    const T* a0 = p0; const T* a1 = p0 + TD; const T* a2 = p0 + 2 * TD; const T* a3 = p0 + 3 * TD;
    for (int c = 0; c < TD; ++c) {
        f[c] = ((a3[c] * t + a2[c]) * t + a1[c]) * t + a0[c];
        df[c] = ((T(3) * a3[c] * t + T(2) * a2[c]) * t + a1[c]) * inv_h;
    }
}

// q_i . k_j per head (4 heads x 16), fixed order: 4 partial sums of 4 per head.
template <typename T> inline void qk_dots(const T* __restrict q, const T* __restrict k, T* dots) {
    for (int h = 0; h < 4; ++h) {
        T p[4] = {T(0), T(0), T(0), T(0)};
        for (int c = 0; c < 16; c += 4)
            for (int l = 0; l < 4; ++l) p[l] += q[16 * h + c + l] * k[16 * h + c + l];
        dots[h] = (p[0] + p[1]) + (p[2] + p[3]);
    }
}

// Edge geometry for receiver i, sender j, edge e (shift from graph g_id). Returns false for an inactive edge (zero message).
// Same formulas as lam4.core.edge_terms / radial / repulsion (the PyTorch oracle): floor on sqrt(|d|^2 + 1e-8), envelope on the
// floored length, pair warp, clamped B-spline index. With `der` it also fills the derivatives the backward needs.
template <typename T>
inline bool edge_geo(const Args<T>& a, int i, int j, int e, int g_id, T rho_i, int zi_shift, bool der, Geo<T>& g) {
    const T* pi = a.pos + 3 * i; const T* pj = a.pos + 3 * j;
    T sh[3] = {a.shift_in[3 * e], a.shift_in[3 * e + 1], a.shift_in[3 * e + 2]};
    if (a.flags & FLAG_CELL) {
        const T* c = a.cell + 9 * g_id;
        T s0 = sh[0], s1 = sh[1], s2 = sh[2];
        sh[0] = s0 * c[0] + s1 * c[3] + s2 * c[6];
        sh[1] = s0 * c[1] + s1 * c[4] + s2 * c[7];
        sh[2] = s0 * c[2] + s1 * c[5] + s2 * c[8];
    }
    for (int m = 0; m < 3; ++m) g.d[m] = (pj[m] - pi[m]) + sh[m];
    T r2 = g.d[0] * g.d[0] + g.d[1] * g.d[1] + g.d[2] * g.d[2];
    T rc = a.r_cut;
    if (!(r2 >= T(1e-24)) || r2 >= rc * rc) return false;
    T r_raw = std::sqrt(r2);
    g.r_raw = r_raw;
    for (int m = 0; m < 3; ++m) { g.u[m] = g.d[m] / r_raw; g.drfl_dd[m] = g.u[m]; }
    T r_fl = r_raw;
    if ((a.flags & FLAG_FLOOR) && r2 < a.max_doff_r2) {
        int zj = (int)std::min<int64_t>(std::max<int64_t>(a.z[j], 0), 127);
        T r_min = a.rmin[zi_shift + zj];
        T d_off = r_min + T(FLOOR_W), r_eps = std::sqrt(r2 + T(R_EPS2));
        if (r_eps < d_off && r_min > T(0)) {
            T th = std::tanh((d_off - r_eps) / T(FLOOR_W));
            r_fl = d_off - T(FLOOR_W) * th;
            if (der) { T s = (T(1) - th * th) / r_eps; for (int m = 0; m < 3; ++m) g.drfl_dd[m] = g.d[m] * s; }
        }
    }
    if (r_fl >= rc) return false;
    T w_env = rc - a.r_on;
    T te = (r_fl - a.r_on) / w_env;
    te = te < T(0) ? T(0) : (te > T(1) ? T(1) : te);
    T t2 = te * te;
    g.chi = T(1) - t2 * t2 * (T(35) + te * (T(-84) + te * (T(70) - T(20) * te)));
    g.dchi = T(0);
    if (der && r_fl > a.r_on) { T om = T(1) - te; g.dchi = T(-140) * (t2 * te) * (om * om * om) / w_env; }
    T x = r_fl;
    g.dx_drfl = T(1); g.dx_drho = T(0);
    if (a.flags & FLAG_RHO) {
        T rho_ij = rho_i + a.rho[j], s = T(WARP_NUM) / rho_ij;
        T den = T(1) + (s - T(1)) * (r_fl / rc);
        x = s * r_fl / den;
        if (der) {
            T inv_den = T(1) / den;
            g.dx_drfl = s * (inv_den * inv_den);
            g.dx_drho = -(T(WARP_NUM) / (rho_ij * rho_ij)) * r_fl * (T(1) - r_fl / rc) * (inv_den * inv_den);
        }
    }
    T h = rc / T(a.grid);
    g.inv_h_eff = (x >= T(0) && x <= rc) ? T(1) / h : T(0);
    T xc = x < T(0) ? T(0) : (x > rc ? rc : x);
    T u_sp = xc / h;
    int idx = (int)std::floor(u_sp);
    idx = idx < 0 ? 0 : (idx > a.grid - 1 ? a.grid - 1 : idx);
    g.idx = idx;
    g.tsp = u_sp - T(idx);
    return true;
}

template <typename T> inline int graph_of(const Args<T>& a, int i) {
    return ((a.flags & FLAG_CELL) && !(a.flags & FLAG_G1)) ? (int)a.graph[i] : 0;
}

template <typename T> inline int zshift_of(const Args<T>& a, int i) {
    return (a.flags & FLAG_FLOOR) ? ((int)std::min<int64_t>(std::max<int64_t>(a.z[i], 0), 127) << 7) : 0;
}

inline int n_threads() { return std::max(1, at::get_num_threads()); }

}  // namespace lam4cpu
"""

CPU_CSR_FWD = r"""
namespace lam4cpu {

// Receiver-owned forward: num [N, OD], zc [N, 4], dsum [N]; message math as lam4.core.edge_terms.edge_terms.
template <typename T, bool LATER>
void csr_fwd_kernel(const Args<T>& a, T* __restrict out_num, T* __restrict out_zc, T* __restrict out_dsum) {
    using L = Lay<LATER>;
    constexpr int TD = L::TD, OD = L::OD, F_O = L::F_O, OT = L::OT, OO = L::OO;
    const bool has_ten = LATER && (a.flags & FLAG_TEN);
    const bool has_rho = (a.flags & FLAG_RHO) != 0;
    #pragma omp parallel for schedule(dynamic, 4) num_threads(n_threads())
    for (int i = 0; i < a.n_atoms; ++i) {
        alignas(64) T acc[OD];
        alignas(64) T tab[TD];
        for (int c = 0; c < OD; ++c) acc[c] = T(0);
        T zc[4] = {T(0), T(0), T(0), T(0)}, dsum = T(0);
        const int g_id = graph_of(a, i), zi_shift = zshift_of(a, i);
        const T rho_i = has_rho ? a.rho[i] : T(0);
        const T* qi = a.qn + 64 * i;
        Geo<T> g;
        for (int e = a.recv_ptr[i], e_end = a.recv_ptr[i + 1]; e < e_end; ++e) {
            const int j = a.send_idx[e];
            if (!edge_geo(a, i, j, e, g_id, rho_i, zi_shift, false, g)) continue;
            horner_val<T, TD>(a.poly + (size_t)(g.idx * 4) * TD, g.tsp, tab);
            T dots[4], w[4];
            qk_dots(qi, a.kn + 64 * j, dots);
            for (int h = 0; h < 4; ++h) {
                w[h] = g.chi * std::exp(dots[h] + T(2) * std::tanh(T(0.5) * tab[h]));
                zc[h] += w[h];
            }
            dsum += g.chi;
            const T* gj = a.gates + 184 * j;
            const T* u = g.u;
            for (int h = 0; h < 4; ++h) {
                const T wh = w[h];
                for (int c = 32 * h; c < 32 * h + 32; ++c) acc[c] += (tab[4 + c] * gj[c]) * wh;
            }
            if (LATER) {
                const T* bj = a.vec + 96 * j;
                for (int c = 0; c < 32; ++c) {
                    const T wv = w[c >> 3];
                    const T b0 = bj[3 * c], b1 = bj[3 * c + 1], b2 = bj[3 * c + 2];
                    const T w_dir = tab[132 + c] * gj[128 + c];
                    const T w_lon = tab[212 + c] * (b0 * u[0] + b1 * u[1] + b2 * u[2]);
                    const T sc = (w_dir + w_lon) * wv, st = tab[180 + c] * wv;
                    acc[128 + 3 * c] += sc * u[0] + st * b0;
                    acc[129 + 3 * c] += sc * u[1] + st * b1;
                    acc[130 + 3 * c] += sc * u[2] + st * b2;
                    acc[224 + c] += w_lon * wv;
                }
            } else {
                for (int c = 0; c < 32; ++c) {
                    const T sc = (tab[132 + c] * gj[128 + c]) * w[c >> 3];
                    acc[128 + 3 * c] += sc * u[0];
                    acc[129 + 3 * c] += sc * u[1];
                    acc[130 + 3 * c] += sc * u[2];
                }
            }
            T y2[5], y3[7];
            eval_y2(u, y2);
            eval_y3(u, y3);
            if (has_ten) {
                const T* tj = a.ten + 80 * j;
                for (int c = 0; c < 16; ++c) {
                    const T wt = w[c >> 2];
                    const T* tc = tj + 5 * c;
                    const T dot_ty2 = tc[0] * y2[0] + tc[1] * y2[1] + tc[2] * y2[2] + tc[3] * y2[3] + tc[4] * y2[4];
                    const T w_ten = tab[164 + c] * gj[160 + c], w_lon2 = tab[260 + c] * dot_ty2;
                    const T sc = (w_ten + w_lon2) * wt, st = tab[244 + c] * wt;
                    for (int m = 0; m < 5; ++m) acc[OT + 5 * c + m] += sc * y2[m] + st * tc[m];
                    acc[336 + c] += w_lon2 * wt;
                }
            } else {
                for (int c = 0; c < 16; ++c) {
                    const T sc = (tab[164 + c] * gj[160 + c]) * w[c >> 2];
                    for (int m = 0; m < 5; ++m) acc[OT + 5 * c + m] += sc * y2[m];
                }
            }
            for (int c = 0; c < 8; ++c) {
                const T sc = (tab[F_O + c] * gj[176 + c]) * w[c >> 1];
                for (int m = 0; m < 7; ++m) acc[OO + 7 * c + m] += sc * y3[m];
            }
        }
        T* oi = out_num + (size_t)OD * i;
        for (int c = 0; c < OD; ++c) oi[c] = acc[c];
        for (int h = 0; h < 4; ++h) out_zc[4 * i + h] = zc[h];
        out_dsum[i] = dsum;
    }
}

}  // namespace lam4cpu
"""
