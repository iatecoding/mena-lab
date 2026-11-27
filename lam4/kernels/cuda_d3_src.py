"""CUDA C++ exact rank-1 factorized Grimme D3(BJ)/D3(0) dispersion kernels (grouped + 12 A CSR + stress)."""
from __future__ import annotations

CUDA_D3_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr float ANG_TO_BOHR  = 1.8897261245650618f;
constexpr float HARTREE_KCAL = 627.509474f;
constexpr float D3_K1        = 16.0f;
constexpr float D3_K3        = -4.0f;

__device__ __forceinline__ int clamp_z94(int64_t z) {
    return (z < 0) ? 0 : ((z > 94) ? 94 : int(z));
}

__device__ __forceinline__ float sub8_sum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 4, 8);
    v += __shfl_down_sync(0xffffffffu, v, 2, 8);
    v += __shfl_down_sync(0xffffffffu, v, 1, 8);
    return v;
}

__device__ __forceinline__ void write_cn_weights(
    int i, int zi, float cn_val,
    const int* __restrict__ n_ref, const float* __restrict__ cn_ref,
    float* __restrict__ out_pw, float* __restrict__ out_dpw
) {
    int nr = __ldg(n_ref + zi);
    int base = 5 * i;
    if (nr <= 0) {
        #pragma unroll
        for (int a = 0; a < 5; ++a) { out_pw[base + a] = 0.0f; out_dpw[base + a] = 0.0f; }
        return;
    }
    const float* cr = cn_ref + 5 * zi;
    float d[5], w[5];
    float smax = -1.0e20f;
    for (int a = 0; a < nr; ++a) {
        d[a] = __ldg(cr + a) - cn_val;
        smax = fmaxf(smax, D3_K3 * d[a] * d[a]);
    }
    float sw = 0.0f;
    for (int a = 0; a < nr; ++a) {
        w[a] = expf(D3_K3 * d[a] * d[a] - smax);
        sw += w[a];
    }
    float inv_sw = 1.0f / fmaxf(sw, 1.0e-30f);
    float dbar = 0.0f;
    for (int a = 0; a < nr; ++a) {
        w[a] *= inv_sw;
        dbar = fmaf(w[a], d[a], dbar);
    }
    #pragma unroll
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

__device__ __forceinline__ void d3_accum_pair_r1(
    float dx, float dy, float dz, int zi, int zj, int ni, int nj,
    const float pw_i[5], const float dpw_i[5],
    const float* __restrict__ pw, float r2r4_i, const float* __restrict__ r2r4,
    const float* __restrict__ c6_mat, const float* __restrict__ r0ab,
    float s6, float s8, float p_a1, float p_a2,
    float r_cut, float taper_on, float inv_tspan,
    int is_zero, int need_deriv, int need_forces, int need_stress, int j,
    float& acc_e, float& acc_dedcn, float& fx, float& fy, float& fz,
    float& vxx, float& vyy, float& vzz, float& vxy, float& vxz, float& vyz
) {
    float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > r_cut * r_cut || ni <= 0 || nj <= 0) return;
    float r2_eps = d2 + 1.0e-8f;
    float r_ang = sqrtf(r2_eps);
    if (r_ang > r_cut) return;
    float inv_r = 1.0f / r_ang;

    const float* tab = c6_mat + 25 * (zi * 95 + zj);
    const float* pj = pw + 5 * j;
    float c6 = 0.0f, dc6 = 0.0f;
    for (int a = 0; a < ni; ++a) {
        int row = a * 5;
        float s = 0.0f;
        for (int b = 0; b < nj; ++b) s = fmaf(__ldg(pj + b), __ldg(tab + row + b), s);
        c6 = fmaf(pw_i[a], s, c6);
        if (need_deriv) dc6 = fmaf(dpw_i[a], s, dc6);
    }

    float q_ij = 3.0f * r2r4_i * __ldg(r2r4 + zj);
    float r_b = r_ang * ANG_TO_BOHR;
    float rb2 = r_b * r_b, rb4 = rb2 * rb2, rb6 = rb4 * rb2, rb8 = rb4 * rb4;
    float A, dA_dr = 0.0f;
    if (is_zero) {
        float r0 = fmaxf(__ldg(r0ab + zi * 95 + zj), 1.0e-6f);
        float rb_s = fmaxf(r_b, 1.0e-6f);
        float u6 = (p_a1 * r0) / rb_s, u8 = r0 / rb_s;
        float u6_2 = u6 * u6, u8_2 = u8 * u8;
        float u6_4 = u6_2 * u6_2, u8_4 = u8_2 * u8_2;
        float g6 = 6.0f * (u6_4 * u6_4 * u6_4 * u6_2), g8 = 6.0f * (u8_4 * u8_4 * u8_4 * u8_4);
        float f6 = 1.0f / (1.0f + g6), f8 = 1.0f / (1.0f + g8);
        float t6_v = s6 * f6 / fmaxf(rb6, 1.0e-12f), t8_v = (s8 * q_ij) * f8 / fmaxf(rb8, 1.0e-12f);
        A = t6_v + t8_v;
        if (need_deriv) {
            float p6 = (rb6 > 1.0e-12f) ? 6.0f : 0.0f, p8 = (rb8 > 1.0e-12f) ? 8.0f : 0.0f;
            dA_dr = ANG_TO_BOHR * ((t6_v * (14.0f * g6 * f6 - p6) + t8_v * (16.0f * g8 * f8 - p8)) / rb_s);
        }
    } else {
        float tmp = p_a1 * sqrtf(fmaxf(q_ij, 1.0e-12f)) + p_a2;
        float t2 = tmp * tmp, t4 = t2 * t2;
        float e6 = 1.0f / (rb6 + t4 * t2), e8 = 1.0f / (rb8 + t4 * t4);
        A = s6 * e6 + (s8 * q_ij) * e8;
        if (need_deriv) {
            dA_dr = -ANG_TO_BOHR * (6.0f * s6 * (rb4 * r_b) * (e6 * e6) + 8.0f * (s8 * q_ij) * (rb6 * r_b) * (e8 * e8));
        }
    }

    float t = fminf(fmaxf((r_ang - taper_on) * inv_tspan, 0.0f), 1.0f);
    float taper = 1.0f - t * t * t * (10.0f + t * (6.0f * t - 15.0f));
    acc_e = fmaf((-0.5f * HARTREE_KCAL) * c6 * A, taper, acc_e);
    if (need_deriv) {
        acc_dedcn = fmaf((-HARTREE_KCAL) * dc6 * A, taper, acc_dedcn);
        float dtap = (r_ang > taper_on && r_ang < r_cut) ? (-30.0f * (t * t) * ((1.0f - t) * (1.0f - t)) * inv_tspan) : 0.0f;
        float fm = HARTREE_KCAL * c6 * (dA_dr * taper + A * dtap) * inv_r;
        if (need_forces) { fx = fmaf(fm, dx, fx); fy = fmaf(fm, dy, fy); fz = fmaf(fm, dz, fz); }
        if (need_stress) {
            float vm = -0.5f * fm;
            vxx = fmaf(vm, dx * dx, vxx); vyy = fmaf(vm, dy * dy, vyy); vzz = fmaf(vm, dz * dz, vzz);
            vxy = fmaf(vm, dx * dy, vxy); vxz = fmaf(vm, dx * dz, vxz); vyz = fmaf(vm, dy * dz, vyz);
        }
    }
}

__device__ __forceinline__ void apply_shift_cell(
    float& dx, float& dy, float& dz, const float* __restrict__ shift, const float* __restrict__ cell,
    int e, int g, int has_shift, int has_cell
) {
    if (!has_shift) return;
    float sx = __ldg(shift + 3 * e), sy = __ldg(shift + 3 * e + 1), sz = __ldg(shift + 3 * e + 2);
    if (has_cell) {
        const float* c = cell + 9 * g;
        dx -= sx * __ldg(c + 0) + sy * __ldg(c + 3) + sz * __ldg(c + 6);
        dy -= sx * __ldg(c + 1) + sy * __ldg(c + 4) + sz * __ldg(c + 7);
        dz -= sx * __ldg(c + 2) + sy * __ldg(c + 5) + sz * __ldg(c + 8);
    } else {
        dx -= sx; dy -= sy; dz -= sz;
    }
}

template <bool IS_CSR>
__global__ void cuda_d3_cn_k(
    const float* __restrict__ pos, const int64_t* __restrict__ z, const uint8_t* __restrict__ mask,
    const int64_t* __restrict__ graph, const int* __restrict__ ptr, const int* __restrict__ send_idx,
    const float* __restrict__ shift, const float* __restrict__ cell,
    const float* __restrict__ rcov, const int* __restrict__ n_ref, const float* __restrict__ cn_ref,
    float* __restrict__ out_pw, float* __restrict__ out_dpw,
    int n_atoms, int flags, float cn_cut_b
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid >> 3, sub = tid & 7;
    if (i >= n_atoms) return;
    int has_mask = flags & 1, has_shift = (flags >> 1) & 1, has_cell = (flags >> 2) & 1, multi_cell = (flags >> 3) & 1;
    if (has_mask && mask[i] == 0) {
        if (sub == 0) write_cn_weights(i, 0, 0.0f, n_ref, cn_ref, out_pw, out_dpw);
        return;
    }
    int zi = clamp_z94(z[i]);
    float rc_i = __ldg(rcov + zi);
    int g = (!IS_CSR || multi_cell) ? int(graph[i]) : 0;
    int row = IS_CSR ? i : g;
    int idx0 = ptr[row], idx1 = ptr[row + 1];
    float pix = __ldg(pos + 3 * i), piy = __ldg(pos + 3 * i + 1), piz = __ldg(pos + 3 * i + 2);
    float acc = 0.0f;
    for (int lane = idx0 + sub; lane < idx1; lane += 8) {
        int j = IS_CSR ? send_idx[lane] : lane;
        if (!IS_CSR && j == i) continue;
        if (has_mask && mask[j] == 0) continue;
        float dx = pix - __ldg(pos + 3 * j), dy = piy - __ldg(pos + 3 * j + 1), dz = piz - __ldg(pos + 3 * j + 2);
        if (IS_CSR) apply_shift_cell(dx, dy, dz, shift, cell, lane, g, has_shift, has_cell);
        float d2 = dx * dx + dy * dy + dz * dz;
        if (IS_CSR && d2 <= 1.0e-12f) continue;
        float r_b = sqrtf(d2 + 1.0e-8f) * ANG_TO_BOHR;
        if (r_b <= cn_cut_b) {
            int zj = clamp_z94(z[j]);
            float rr = (rc_i + __ldg(rcov + zj)) / fmaxf(r_b, 1.0e-6f);
            acc += 1.0f / (1.0f + expf(-D3_K1 * (rr - 1.0f)));
        }
    }
    acc = sub8_sum(acc);
    if (sub == 0) write_cn_weights(i, zi, acc, n_ref, cn_ref, out_pw, out_dpw);
}

template <bool IS_CSR>
__global__ void cuda_d3_ef_k(
    const float* __restrict__ pos, const int64_t* __restrict__ z, const uint8_t* __restrict__ mask,
    const int64_t* __restrict__ graph, const int* __restrict__ ptr, const int* __restrict__ send_idx,
    const float* __restrict__ shift, const float* __restrict__ cell,
    const float* __restrict__ r2r4, const int* __restrict__ n_ref, const float* __restrict__ c6_mat,
    const float* __restrict__ r0ab,
    const float* __restrict__ pw, const float* __restrict__ dpw,
    float* __restrict__ out_e, float* __restrict__ out_dedcn, float* __restrict__ out_f, float* __restrict__ out_vir,
    int n_atoms, int flags, float s6, float s8, float p_a1, float p_a2, float r_cut, float taper_on, float inv_tspan
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid >> 3, sub = tid & 7;
    if (i >= n_atoms) return;
    int has_mask = flags & 1, has_shift = (flags >> 1) & 1, has_cell = (flags >> 2) & 1, multi_cell = (flags >> 3) & 1;
    int is_zero = (flags >> 4) & 1, need_forces = (flags >> 5) & 1, need_stress = (flags >> 6) & 1;
    int need_deriv = (need_forces | need_stress) ? 1 : 0;
    if (has_mask && mask[i] == 0) {
        if (sub == 0) {
            out_e[i] = 0.0f;
            if (need_deriv) out_dedcn[i] = 0.0f;
            if (need_forces) { out_f[3 * i] = 0.0f; out_f[3 * i + 1] = 0.0f; out_f[3 * i + 2] = 0.0f; }
            if (need_stress) { for (int m = 0; m < 6; ++m) out_vir[6 * i + m] = 0.0f; }
        }
        return;
    }
    int zi = clamp_z94(z[i]), ni = __ldg(n_ref + zi);
    float r2r4_i = __ldg(r2r4 + zi);
    float pw_i[5], dpw_i[5];
    #pragma unroll
    for (int a = 0; a < 5; ++a) { pw_i[a] = pw[5 * i + a]; dpw_i[a] = dpw[5 * i + a]; }
    int g = (!IS_CSR || multi_cell) ? int(graph[i]) : 0;
    int row = IS_CSR ? i : g;
    int idx0 = ptr[row], idx1 = ptr[row + 1];
    float pix = __ldg(pos + 3 * i), piy = __ldg(pos + 3 * i + 1), piz = __ldg(pos + 3 * i + 2);
    float acc_e = 0.0f, acc_de = 0.0f, fx = 0.0f, fy = 0.0f, fz = 0.0f;
    float vxx = 0.0f, vyy = 0.0f, vzz = 0.0f, vxy = 0.0f, vxz = 0.0f, vyz = 0.0f;
    for (int lane = idx0 + sub; lane < idx1; lane += 8) {
        int j = IS_CSR ? send_idx[lane] : lane;
        if (!IS_CSR && j == i) continue;
        if (has_mask && mask[j] == 0) continue;
        float dx = pix - __ldg(pos + 3 * j), dy = piy - __ldg(pos + 3 * j + 1), dz = piz - __ldg(pos + 3 * j + 2);
        if (IS_CSR) apply_shift_cell(dx, dy, dz, shift, cell, lane, g, has_shift, has_cell);
        if (IS_CSR && (dx * dx + dy * dy + dz * dz) <= 1.0e-12f) continue;
        int zj = clamp_z94(z[j]);
        d3_accum_pair_r1(dx, dy, dz, zi, zj, ni, __ldg(n_ref + zj), pw_i, dpw_i, pw, r2r4_i, r2r4, c6_mat, r0ab,
                         s6, s8, p_a1, p_a2, r_cut, taper_on, inv_tspan, is_zero, need_deriv, need_forces, need_stress, j,
                         acc_e, acc_de, fx, fy, fz, vxx, vyy, vzz, vxy, vxz, vyz);
    }
    acc_e = sub8_sum(acc_e);
    if (need_deriv) acc_de = sub8_sum(acc_de);
    if (need_forces) { fx = sub8_sum(fx); fy = sub8_sum(fy); fz = sub8_sum(fz); }
    if (need_stress) {
        vxx = sub8_sum(vxx); vyy = sub8_sum(vyy); vzz = sub8_sum(vzz);
        vxy = sub8_sum(vxy); vxz = sub8_sum(vxz); vyz = sub8_sum(vyz);
    }
    if (sub == 0) {
        out_e[i] = acc_e;
        if (need_deriv) out_dedcn[i] = acc_de;
        if (need_forces) { out_f[3 * i] = fx; out_f[3 * i + 1] = fy; out_f[3 * i + 2] = fz; }
        if (need_stress) {
            out_vir[6 * i + 0] = vxx; out_vir[6 * i + 1] = vyy; out_vir[6 * i + 2] = vzz;
            out_vir[6 * i + 3] = vxy; out_vir[6 * i + 4] = vxz; out_vir[6 * i + 5] = vyz;
        }
    }
}

template <bool IS_CSR>
__global__ void cuda_d3_fcn_k(
    const float* __restrict__ pos, const int64_t* __restrict__ z, const uint8_t* __restrict__ mask,
    const int64_t* __restrict__ graph, const int* __restrict__ ptr, const int* __restrict__ send_idx,
    const float* __restrict__ shift, const float* __restrict__ cell,
    const float* __restrict__ rcov, const float* __restrict__ dedcn,
    float* __restrict__ inout_f, float* __restrict__ inout_vir,
    int n_atoms, int flags, float cn_cut_b
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid >> 3, sub = tid & 7;
    if (i >= n_atoms) return;
    int has_mask = flags & 1, has_shift = (flags >> 1) & 1, has_cell = (flags >> 2) & 1, multi_cell = (flags >> 3) & 1;
    int need_forces = (flags >> 5) & 1, need_stress = (flags >> 6) & 1;
    if (has_mask && mask[i] == 0) return;
    int zi = clamp_z94(z[i]);
    float rc_i = __ldg(rcov + zi), de_i = __ldg(dedcn + i);
    int g = (!IS_CSR || multi_cell) ? int(graph[i]) : 0;
    int row = IS_CSR ? i : g;
    int idx0 = ptr[row], idx1 = ptr[row + 1];
    float pix = __ldg(pos + 3 * i), piy = __ldg(pos + 3 * i + 1), piz = __ldg(pos + 3 * i + 2);
    float fx = 0.0f, fy = 0.0f, fz = 0.0f, vxx = 0.0f, vyy = 0.0f, vzz = 0.0f, vxy = 0.0f, vxz = 0.0f, vyz = 0.0f;
    for (int lane = idx0 + sub; lane < idx1; lane += 8) {
        int j = IS_CSR ? send_idx[lane] : lane;
        if (!IS_CSR && j == i) continue;
        if (has_mask && mask[j] == 0) continue;
        float dx = pix - __ldg(pos + 3 * j), dy = piy - __ldg(pos + 3 * j + 1), dz = piz - __ldg(pos + 3 * j + 2);
        if (IS_CSR) apply_shift_cell(dx, dy, dz, shift, cell, lane, g, has_shift, has_cell);
        float d2 = dx * dx + dy * dy + dz * dz;
        if (IS_CSR && d2 <= 1.0e-12f) continue;
        float r2_eps = d2 + 1.0e-8f, r_ang = sqrtf(r2_eps), inv_r = 1.0f / r_ang, r_b = r_ang * ANG_TO_BOHR;
        if (r_b <= cn_cut_b && r_b > 1.0e-6f) {
            int zj = clamp_z94(z[j]);
            float rr = (rc_i + __ldg(rcov + zj)) / r_b;
            float damp = 1.0f / (1.0f + expf(-D3_K1 * (rr - 1.0f)));
            float fm = (de_i + __ldg(dedcn + j)) * D3_K1 * damp * (1.0f - damp) * rr * (inv_r * inv_r);
            if (need_forces) { fx = fmaf(fm, dx, fx); fy = fmaf(fm, dy, fy); fz = fmaf(fm, dz, fz); }
            if (need_stress) {
                float vm = -0.5f * fm;
                vxx = fmaf(vm, dx * dx, vxx); vyy = fmaf(vm, dy * dy, vyy); vzz = fmaf(vm, dz * dz, vzz);
                vxy = fmaf(vm, dx * dy, vxy); vxz = fmaf(vm, dx * dz, vxz); vyz = fmaf(vm, dy * dz, vyz);
            }
        }
    }
    if (need_forces) { fx = sub8_sum(fx); fy = sub8_sum(fy); fz = sub8_sum(fz); }
    if (need_stress) {
        vxx = sub8_sum(vxx); vyy = sub8_sum(vyy); vzz = sub8_sum(vzz);
        vxy = sub8_sum(vxy); vxz = sub8_sum(vxz); vyz = sub8_sum(vyz);
    }
    if (sub == 0) {
        if (need_forces) { inout_f[3 * i] += fx; inout_f[3 * i + 1] += fy; inout_f[3 * i + 2] += fz; }
        if (need_stress) {
            inout_vir[6 * i + 0] += vxx; inout_vir[6 * i + 1] += vyy; inout_vir[6 * i + 2] += vzz;
            inout_vir[6 * i + 3] += vxy; inout_vir[6 * i + 4] += vxz; inout_vir[6 * i + 5] += vyz;
        }
    }
}

} // anonymous namespace

void cuda_d3_launch(
    torch::Tensor pos, torch::Tensor z, torch::Tensor mask, torch::Tensor graph,
    torch::Tensor ptr, torch::Tensor send_idx, torch::Tensor shift, torch::Tensor cell,
    torch::Tensor rcov, torch::Tensor r2r4, torch::Tensor n_ref, torch::Tensor cn_ref, torch::Tensor c6_mat, torch::Tensor r0ab,
    torch::Tensor pw, torch::Tensor dpw, torch::Tensor out_e, torch::Tensor out_dedcn,
    torch::Tensor out_f, torch::Tensor out_vir,
    int64_t n_atoms, int64_t flags, bool is_csr,
    double s6, double s8, double p_a1, double p_a2, double r_cut, double taper_on, double inv_tspan, double cn_cut_b
) {
    if (n_atoms <= 0) return;
    int blocks = int((n_atoms * 8 + 127) / 128);
    auto stream = at::cuda::getCurrentCUDAStream();
    bool need_deriv = ((flags >> 5) & 3) != 0;
    if (is_csr) {
        cuda_d3_cn_k<true><<<blocks, 128, 0, stream>>>(
            pos.data_ptr<float>(), z.data_ptr<int64_t>(), mask.data_ptr<uint8_t>(), graph.data_ptr<int64_t>(),
            ptr.data_ptr<int>(), send_idx.data_ptr<int>(), shift.data_ptr<float>(), cell.data_ptr<float>(),
            rcov.data_ptr<float>(), n_ref.data_ptr<int>(), cn_ref.data_ptr<float>(),
            pw.data_ptr<float>(), dpw.data_ptr<float>(), int(n_atoms), int(flags), float(cn_cut_b));
        cuda_d3_ef_k<true><<<blocks, 128, 0, stream>>>(
            pos.data_ptr<float>(), z.data_ptr<int64_t>(), mask.data_ptr<uint8_t>(), graph.data_ptr<int64_t>(),
            ptr.data_ptr<int>(), send_idx.data_ptr<int>(), shift.data_ptr<float>(), cell.data_ptr<float>(),
            r2r4.data_ptr<float>(), n_ref.data_ptr<int>(), c6_mat.data_ptr<float>(), r0ab.data_ptr<float>(),
            pw.data_ptr<float>(), dpw.data_ptr<float>(),
            out_e.data_ptr<float>(), out_dedcn.data_ptr<float>(), out_f.data_ptr<float>(), out_vir.data_ptr<float>(),
            int(n_atoms), int(flags), float(s6), float(s8), float(p_a1), float(p_a2),
            float(r_cut), float(taper_on), float(inv_tspan));
        if (need_deriv) {
            cuda_d3_fcn_k<true><<<blocks, 128, 0, stream>>>(
                pos.data_ptr<float>(), z.data_ptr<int64_t>(), mask.data_ptr<uint8_t>(), graph.data_ptr<int64_t>(),
                ptr.data_ptr<int>(), send_idx.data_ptr<int>(), shift.data_ptr<float>(), cell.data_ptr<float>(),
                rcov.data_ptr<float>(), out_dedcn.data_ptr<float>(), out_f.data_ptr<float>(), out_vir.data_ptr<float>(),
                int(n_atoms), int(flags), float(cn_cut_b));
        }
    } else {
        cuda_d3_cn_k<false><<<blocks, 128, 0, stream>>>(
            pos.data_ptr<float>(), z.data_ptr<int64_t>(), mask.data_ptr<uint8_t>(), graph.data_ptr<int64_t>(),
            ptr.data_ptr<int>(), send_idx.data_ptr<int>(), shift.data_ptr<float>(), cell.data_ptr<float>(),
            rcov.data_ptr<float>(), n_ref.data_ptr<int>(), cn_ref.data_ptr<float>(),
            pw.data_ptr<float>(), dpw.data_ptr<float>(), int(n_atoms), int(flags), float(cn_cut_b));
        cuda_d3_ef_k<false><<<blocks, 128, 0, stream>>>(
            pos.data_ptr<float>(), z.data_ptr<int64_t>(), mask.data_ptr<uint8_t>(), graph.data_ptr<int64_t>(),
            ptr.data_ptr<int>(), send_idx.data_ptr<int>(), shift.data_ptr<float>(), cell.data_ptr<float>(),
            r2r4.data_ptr<float>(), n_ref.data_ptr<int>(), c6_mat.data_ptr<float>(), r0ab.data_ptr<float>(),
            pw.data_ptr<float>(), dpw.data_ptr<float>(),
            out_e.data_ptr<float>(), out_dedcn.data_ptr<float>(), out_f.data_ptr<float>(), out_vir.data_ptr<float>(),
            int(n_atoms), int(flags), float(s6), float(s8), float(p_a1), float(p_a2),
            float(r_cut), float(taper_on), float(inv_tspan));
        if (need_deriv) {
            cuda_d3_fcn_k<false><<<blocks, 128, 0, stream>>>(
                pos.data_ptr<float>(), z.data_ptr<int64_t>(), mask.data_ptr<uint8_t>(), graph.data_ptr<int64_t>(),
                ptr.data_ptr<int>(), send_idx.data_ptr<int>(), shift.data_ptr<float>(), cell.data_ptr<float>(),
                rcov.data_ptr<float>(), out_dedcn.data_ptr<float>(), out_f.data_ptr<float>(), out_vir.data_ptr<float>(),
                int(n_atoms), int(flags), float(cn_cut_b));
        }
    }
}
"""
