"""CUDA C++ warp-synchronized 2-atoms/thread register-blocked (BLK=64, 32 threads) + 2D split-J ExactCoulomb kernels."""
from __future__ import annotations

CUDA_COULOMB_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int BLK = 64;
constexpr int NTH = 32; // 1 warp handles 64 i-atoms (2 atoms/thread -> 2x ILP, 2x less SMEM traffic, __syncwarp only)

__device__ __forceinline__ void smeared_k_f1_f2(
    float r2, float inv_sig, float c0, float inv_sig2, bool need_f2,
    float& k_out, float& f1_out, float& f2_out
) {
    float x2 = r2 * inv_sig2;
    if (x2 > 30.25f) {
        float inv_r = rsqrtf(r2);
        float inv_r2 = inv_r * inv_r;
        float f1g = inv_r * inv_r2;
        k_out = inv_r;
        f1_out = f1g;
        f2_out = need_f2 ? (-3.0f * f1g * inv_r2) : 0.0f;
        return;
    }
    if (x2 < 2.5e-3f) {
        float x4 = x2 * x2;
        float c0_s2 = c0 * inv_sig2;
        k_out = c0 * (1.0f - 0.333333343f * x2 + 0.1f * x4);
        f1_out = c0_s2 * (0.666666687f - 0.4f * x2 + 0.142857149f * x4);
        f2_out = need_f2 ? ((c0_s2 * inv_sig2) * (-0.8f + 0.571428597f * x2 - 0.222222224f * x4)) : 0.0f;
        return;
    }
    float inv_r = rsqrtf(r2);
    float inv_r2 = inv_r * inv_r;
    float ex = expf(-x2);
    float kg = erff((r2 * inv_r) * inv_sig) * inv_r;
    float f1g = (kg - c0 * ex) * inv_r2;
    k_out = kg;
    f1_out = f1g;
    f2_out = need_f2 ? ((2.0f * (c0 * inv_sig2) * ex - 3.0f * f1g) * inv_r2) : 0.0f;
}

template <bool IS_FAST64, bool IS_GROUPED, bool NEED_FLD, bool SPLIT_J>
__global__ void cuda_coulomb_fwd_k(
    const float* __restrict__ pos, const float* __restrict__ q, const uint8_t* __restrict__ mask,
    const int* __restrict__ g_arr, const int* __restrict__ gptr, const float* __restrict__ bsph,
    float* __restrict__ phi_out, float* __restrict__ fld_out,
    int n_atoms, float inv_sig, float c0, float inv_sig2, float far_pad
) {
    int tg = blockIdx.x;
    int tid = threadIdx.x;
    int oi0 = tg * BLK + tid;
    int oi1 = oi0 + 32;
    bool ib0 = oi0 < n_atoms, ib1 = oi1 < n_atoms;
    bool act0 = ib0 && (mask[oi0] != 0);
    bool act1 = ib1 && (mask[oi1] != 0);
    int gi0 = (IS_GROUPED && ib0) ? g_arr[oi0] : 0;
    int gi1 = (IS_GROUPED && ib1) ? g_arr[oi1] : 0;

    int j_start = 0, j_end = n_atoms;
    if (IS_GROUPED) {
        int i0 = tg * BLK;
        int i1 = min(i0 + BLK - 1, n_atoms - 1);
        j_start = gptr[g_arr[i0]];
        j_end = gptr[g_arr[i1] + 1];
    } else if (SPLIT_J) {
        int n_tiles = (n_atoms + BLK - 1) / BLK;
        int bj = blockIdx.y, sj = gridDim.y;
        j_start = ((bj * n_tiles) / sj) * BLK;
        j_end = min((((bj + 1) * n_tiles) / sj) * BLK, n_atoms);
    }
    float tcx = 0.0f, tcy = 0.0f, tcz = 0.0f, rtg = -1.0f;
    if (IS_FAST64) {
        tcx = bsph[tg * 4 + 0]; tcy = bsph[tg * 4 + 1]; tcz = bsph[tg * 4 + 2]; rtg = bsph[tg * 4 + 3];
    }
    float p0x = act0 ? pos[oi0 * 3 + 0] : 0.0f, p0y = act0 ? pos[oi0 * 3 + 1] : 0.0f, p0z = act0 ? pos[oi0 * 3 + 2] : 0.0f;
    float p1x = act1 ? pos[oi1 * 3 + 0] : 0.0f, p1y = act1 ? pos[oi1 * 3 + 1] : 0.0f, p1z = act1 ? pos[oi1 * 3 + 2] : 0.0f;
    float ap0 = 0.0f, fx0 = 0.0f, fy0 = 0.0f, fz0 = 0.0f;
    float ap1 = 0.0f, fx1 = 0.0f, fy1 = 0.0f, fz1 = 0.0f;

    __shared__ float spx[BLK], spy[BLK], spz[BLK], sq[BLK];
    __shared__ int sg_j[BLK];
    __shared__ uint8_t sact[BLK];

    for (int j_base = j_start; j_base < j_end; j_base += BLK) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            int loc = tid + h * 32;
            int oj = j_base + loc;
            bool jb = oj < j_end;
            uint8_t aj = (jb && mask[oj] != 0) ? 1 : 0;
            sact[loc] = aj;
            spx[loc] = aj ? pos[oj * 3 + 0] : 0.0f;
            spy[loc] = aj ? pos[oj * 3 + 1] : 0.0f;
            spz[loc] = aj ? pos[oj * 3 + 2] : 0.0f;
            sq[loc] = aj ? q[oj] : 0.0f;
            if (IS_GROUPED) sg_j[loc] = jb ? g_arr[oj] : -2;
        }
        __syncwarp();

        bool far = false;
        if (IS_FAST64) {
            int jb_idx = j_base >> 6;
            float tbx = bsph[jb_idx * 4 + 0], tby = bsph[jb_idx * 4 + 1], tbz = bsph[jb_idx * 4 + 2], rtb = bsph[jb_idx * 4 + 3];
            float thr = rtg + far_pad + rtb;
            float dcx = tcx - tbx, dcy = tcy - tby, dcz = tcz - tbz;
            far = (dcx * dcx + dcy * dcy + dcz * dcz > thr * thr) && (rtb >= 0.0f) && (rtg >= 0.0f);
        }

        if (far) {
            #pragma unroll 8
            for (int k = 0; k < BLK; ++k) {
                float sx = spx[k], sy = spy[k], sz = spz[k], qj = sq[k];
                float dx0 = p0x - sx, dy0 = p0y - sy, dz0 = p0z - sz;
                float dx1 = p1x - sx, dy1 = p1y - sy, dz1 = p1z - sz;
                float ir0 = rsqrtf(fmaxf(dx0 * dx0 + dy0 * dy0 + dz0 * dz0, 1.0e-16f));
                float ir1 = rsqrtf(fmaxf(dx1 * dx1 + dy1 * dy1 + dz1 * dz1, 1.0e-16f));
                ap0 = fmaf(qj, ir0, ap0);
                ap1 = fmaf(qj, ir1, ap1);
                if (NEED_FLD) {
                    float wf0 = qj * (ir0 * ir0 * ir0);
                    float wf1 = qj * (ir1 * ir1 * ir1);
                    fx0 = fmaf(wf0, dx0, fx0); fy0 = fmaf(wf0, dy0, fy0); fz0 = fmaf(wf0, dz0, fz0);
                    fx1 = fmaf(wf1, dx1, fx1); fy1 = fmaf(wf1, dy1, fy1); fz1 = fmaf(wf1, dz1, fz1);
                }
            }
        } else {
            #pragma unroll 4
            for (int k = 0; k < BLK; ++k) {
                if (!sact[k]) continue;
                int oj = j_base + k;
                float sx = spx[k], sy = spy[k], sz = spz[k], qj = sq[k];
                int gjk = IS_GROUPED ? sg_j[k] : 0;
                if (act0 && oj != oi0 && (!IS_GROUPED || gjk == gi0)) {
                    float dx = p0x - sx, dy = p0y - sy, dz = p0z - sz;
                    float kv, f1, f2_d;
                    smeared_k_f1_f2(fmaxf(dx * dx + dy * dy + dz * dz, 1.0e-16f), inv_sig, c0, inv_sig2, false, kv, f1, f2_d);
                    ap0 = fmaf(qj, kv, ap0);
                    if (NEED_FLD) { float wf = qj * f1; fx0 = fmaf(wf, dx, fx0); fy0 = fmaf(wf, dy, fy0); fz0 = fmaf(wf, dz, fz0); }
                }
                if (act1 && oj != oi1 && (!IS_GROUPED || gjk == gi1)) {
                    float dx = p1x - sx, dy = p1y - sy, dz = p1z - sz;
                    float kv, f1, f2_d;
                    smeared_k_f1_f2(fmaxf(dx * dx + dy * dy + dz * dz, 1.0e-16f), inv_sig, c0, inv_sig2, false, kv, f1, f2_d);
                    ap1 = fmaf(qj, kv, ap1);
                    if (NEED_FLD) { float wf = qj * f1; fx1 = fmaf(wf, dx, fx1); fy1 = fmaf(wf, dy, fy1); fz1 = fmaf(wf, dz, fz1); }
                }
            }
        }
        __syncwarp();
    }
    auto store = [&](int oi, bool ib, bool act, float ap, float fx, float fy, float fz) {
        if (SPLIT_J) {
            if (act) {
                atomicAdd(&phi_out[oi], ap);
                if (NEED_FLD) { atomicAdd(&fld_out[oi * 3 + 0], fx); atomicAdd(&fld_out[oi * 3 + 1], fy); atomicAdd(&fld_out[oi * 3 + 2], fz); }
            }
        } else if (ib) {
            phi_out[oi] = act ? ap : 0.0f;
            if (NEED_FLD) { fld_out[oi * 3 + 0] = act ? fx : 0.0f; fld_out[oi * 3 + 1] = act ? fy : 0.0f; fld_out[oi * 3 + 2] = act ? fz : 0.0f; }
        }
    };
    store(oi0, ib0, act0, ap0, fx0, fy0, fz0);
    store(oi1, ib1, act1, ap1, fx1, fy1, fz1);
}

template <bool IS_FAST64, bool IS_GROUPED, bool HAS_GFLD, bool NEED_DPOS, bool NEED_DQ, bool SPLIT_J>
__global__ void cuda_coulomb_bwd_k(
    const float* __restrict__ pos, const float* __restrict__ q, const uint8_t* __restrict__ mask,
    const int* __restrict__ g_arr, const int* __restrict__ gptr, const float* __restrict__ bsph,
    const float* __restrict__ gphi, const float* __restrict__ gfld,
    float* __restrict__ dpos_out, float* __restrict__ dq_out,
    int n_atoms, float inv_sig, float c0, float inv_sig2, float far_pad
) {
    int tg = blockIdx.x;
    int tid = threadIdx.x;
    int oi0 = tg * BLK + tid;
    int oi1 = oi0 + 32;
    bool ib0 = oi0 < n_atoms, ib1 = oi1 < n_atoms;
    bool act0 = ib0 && (mask[oi0] != 0);
    bool act1 = ib1 && (mask[oi1] != 0);
    int gi0 = (IS_GROUPED && ib0) ? g_arr[oi0] : 0;
    int gi1 = (IS_GROUPED && ib1) ? g_arr[oi1] : 0;

    int j_start = 0, j_end = n_atoms;
    if (IS_GROUPED) {
        int i0 = tg * BLK, i1 = min(i0 + BLK - 1, n_atoms - 1);
        j_start = gptr[g_arr[i0]];
        j_end = gptr[g_arr[i1] + 1];
    } else if (SPLIT_J) {
        int n_tiles = (n_atoms + BLK - 1) / BLK;
        int bj = blockIdx.y, sj = gridDim.y;
        j_start = ((bj * n_tiles) / sj) * BLK;
        j_end = min((((bj + 1) * n_tiles) / sj) * BLK, n_atoms);
    }
    float tcx = 0.0f, tcy = 0.0f, tcz = 0.0f, rtg = -1.0f;
    if (IS_FAST64) {
        tcx = bsph[tg * 4 + 0]; tcy = bsph[tg * 4 + 1]; tcz = bsph[tg * 4 + 2]; rtg = bsph[tg * 4 + 3];
    }
    float p0x = act0 ? pos[oi0 * 3 + 0] : 0.0f, p0y = act0 ? pos[oi0 * 3 + 1] : 0.0f, p0z = act0 ? pos[oi0 * 3 + 2] : 0.0f;
    float p1x = act1 ? pos[oi1 * 3 + 0] : 0.0f, p1y = act1 ? pos[oi1 * 3 + 1] : 0.0f, p1z = act1 ? pos[oi1 * 3 + 2] : 0.0f;
    float q0 = act0 ? q[oi0] : 0.0f, gp0 = act0 ? gphi[oi0] : 0.0f;
    float q1 = act1 ? q[oi1] : 0.0f, gp1 = act1 ? gphi[oi1] : 0.0f;
    float ge0x = (act0 && HAS_GFLD) ? gfld[oi0 * 3 + 0] : 0.0f, ge0y = (act0 && HAS_GFLD) ? gfld[oi0 * 3 + 1] : 0.0f, ge0z = (act0 && HAS_GFLD) ? gfld[oi0 * 3 + 2] : 0.0f;
    float ge1x = (act1 && HAS_GFLD) ? gfld[oi1 * 3 + 0] : 0.0f, ge1y = (act1 && HAS_GFLD) ? gfld[oi1 * 3 + 1] : 0.0f, ge1z = (act1 && HAS_GFLD) ? gfld[oi1 * 3 + 2] : 0.0f;
    float dp0x = 0.0f, dp0y = 0.0f, dp0z = 0.0f, dq0 = 0.0f;
    float dp1x = 0.0f, dp1y = 0.0f, dp1z = 0.0f, dq1 = 0.0f;

    __shared__ float spx[BLK], spy[BLK], spz[BLK], sq[BLK], sgp[BLK], sgex[BLK], sgey[BLK], sgez[BLK];
    __shared__ int sg_j[BLK];
    __shared__ uint8_t sact[BLK];

    for (int j_base = j_start; j_base < j_end; j_base += BLK) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            int loc = tid + h * 32;
            int oj = j_base + loc;
            bool jb = oj < j_end;
            uint8_t aj = (jb && mask[oj] != 0) ? 1 : 0;
            sact[loc] = aj;
            spx[loc] = aj ? pos[oj * 3 + 0] : 0.0f;
            spy[loc] = aj ? pos[oj * 3 + 1] : 0.0f;
            spz[loc] = aj ? pos[oj * 3 + 2] : 0.0f;
            sq[loc] = aj ? q[oj] : 0.0f;
            sgp[loc] = aj ? gphi[oj] : 0.0f;
            if (HAS_GFLD) {
                sgex[loc] = aj ? gfld[oj * 3 + 0] : 0.0f;
                sgey[loc] = aj ? gfld[oj * 3 + 1] : 0.0f;
                sgez[loc] = aj ? gfld[oj * 3 + 2] : 0.0f;
            }
            if (IS_GROUPED) sg_j[loc] = jb ? g_arr[oj] : -2;
        }
        __syncwarp();

        bool far = false;
        if (IS_FAST64) {
            int jb_idx = j_base >> 6;
            float tbx = bsph[jb_idx * 4 + 0], tby = bsph[jb_idx * 4 + 1], tbz = bsph[jb_idx * 4 + 2], rtb = bsph[jb_idx * 4 + 3];
            float thr = rtg + far_pad + rtb;
            float dcx = tcx - tbx, dcy = tcy - tby, dcz = tcz - tbz;
            far = (dcx * dcx + dcy * dcy + dcz * dcz > thr * thr) && (rtb >= 0.0f) && (rtg >= 0.0f);
        }

        auto acc_pair = [&](float dx, float dy, float dz, float kv, float f1, float f2,
                            float qi, float gpi, float geix, float geiy, float geiz,
                            float qj, float gpj, float gexj, float geyj, float gezj,
                            float& dpx, float& dpy, float& dpz, float& dq) {
            if (NEED_DQ) {
                if (HAS_GFLD) dq += gpj * kv - f1 * (gexj * dx + geyj * dy + gezj * dz);
                else dq = fmaf(gpj, kv, dq);
            }
            if (NEED_DPOS) {
                float s_phi = gpi * qj + gpj * qi;
                if (HAS_GFLD) {
                    float vx = qj * geix - qi * gexj, vy = qj * geiy - qi * geyj, vz = qj * geiz - qi * gezj;
                    float cd = -s_phi * f1 + f2 * (vx * dx + vy * dy + vz * dz);
                    dpx = fmaf(cd, dx, fmaf(f1, vx, dpx));
                    dpy = fmaf(cd, dy, fmaf(f1, vy, dpy));
                    dpz = fmaf(cd, dz, fmaf(f1, vz, dpz));
                } else {
                    float cd = -s_phi * f1;
                    dpx = fmaf(cd, dx, dpx); dpy = fmaf(cd, dy, dpy); dpz = fmaf(cd, dz, dpz);
                }
            }
        };

        if (far) {
            #pragma unroll 4
            for (int k = 0; k < BLK; ++k) {
                float sx = spx[k], sy = spy[k], sz = spz[k], qj = sq[k], gpj = sgp[k];
                float gexj = HAS_GFLD ? sgex[k] : 0.0f, geyj = HAS_GFLD ? sgey[k] : 0.0f, gezj = HAS_GFLD ? sgez[k] : 0.0f;
                float dx0 = p0x - sx, dy0 = p0y - sy, dz0 = p0z - sz;
                float dx1 = p1x - sx, dy1 = p1y - sy, dz1 = p1z - sz;
                float kv0 = rsqrtf(fmaxf(dx0 * dx0 + dy0 * dy0 + dz0 * dz0, 1.0e-16f));
                float kv1 = rsqrtf(fmaxf(dx1 * dx1 + dy1 * dy1 + dz1 * dz1, 1.0e-16f));
                float ir2_0 = kv0 * kv0, f1_0 = kv0 * ir2_0, f2_0 = HAS_GFLD ? (-3.0f * f1_0 * ir2_0) : 0.0f;
                float ir2_1 = kv1 * kv1, f1_1 = kv1 * ir2_1, f2_1 = HAS_GFLD ? (-3.0f * f1_1 * ir2_1) : 0.0f;
                acc_pair(dx0, dy0, dz0, kv0, f1_0, f2_0, q0, gp0, ge0x, ge0y, ge0z, qj, gpj, gexj, geyj, gezj, dp0x, dp0y, dp0z, dq0);
                acc_pair(dx1, dy1, dz1, kv1, f1_1, f2_1, q1, gp1, ge1x, ge1y, ge1z, qj, gpj, gexj, geyj, gezj, dp1x, dp1y, dp1z, dq1);
            }
        } else {
            #pragma unroll 4
            for (int k = 0; k < BLK; ++k) {
                if (!sact[k]) continue;
                int oj = j_base + k;
                float sx = spx[k], sy = spy[k], sz = spz[k], qj = sq[k], gpj = sgp[k];
                float gexj = HAS_GFLD ? sgex[k] : 0.0f, geyj = HAS_GFLD ? sgey[k] : 0.0f, gezj = HAS_GFLD ? sgez[k] : 0.0f;
                int gjk = IS_GROUPED ? sg_j[k] : 0;
                if (act0 && oj != oi0 && (!IS_GROUPED || gjk == gi0)) {
                    float dx = p0x - sx, dy = p0y - sy, dz = p0z - sz, kv, f1, f2;
                    smeared_k_f1_f2(fmaxf(dx * dx + dy * dy + dz * dz, 1.0e-16f), inv_sig, c0, inv_sig2, HAS_GFLD, kv, f1, f2);
                    acc_pair(dx, dy, dz, kv, f1, f2, q0, gp0, ge0x, ge0y, ge0z, qj, gpj, gexj, geyj, gezj, dp0x, dp0y, dp0z, dq0);
                }
                if (act1 && oj != oi1 && (!IS_GROUPED || gjk == gi1)) {
                    float dx = p1x - sx, dy = p1y - sy, dz = p1z - sz, kv, f1, f2;
                    smeared_k_f1_f2(fmaxf(dx * dx + dy * dy + dz * dz, 1.0e-16f), inv_sig, c0, inv_sig2, HAS_GFLD, kv, f1, f2);
                    acc_pair(dx, dy, dz, kv, f1, f2, q1, gp1, ge1x, ge1y, ge1z, qj, gpj, gexj, geyj, gezj, dp1x, dp1y, dp1z, dq1);
                }
            }
        }
        __syncwarp();
    }
    auto store = [&](int oi, bool ib, bool act, float dpx, float dpy, float dpz, float dq) {
        if (SPLIT_J) {
            if (act) {
                if (NEED_DQ) atomicAdd(&dq_out[oi], dq);
                if (NEED_DPOS) { atomicAdd(&dpos_out[oi * 3 + 0], dpx); atomicAdd(&dpos_out[oi * 3 + 1], dpy); atomicAdd(&dpos_out[oi * 3 + 2], dpz); }
            }
        } else if (ib) {
            if (NEED_DQ) dq_out[oi] = act ? dq : 0.0f;
            if (NEED_DPOS) { dpos_out[oi * 3 + 0] = act ? dpx : 0.0f; dpos_out[oi * 3 + 1] = act ? dpy : 0.0f; dpos_out[oi * 3 + 2] = act ? dpz : 0.0f; }
        }
    };
    store(oi0, ib0, act0, dp0x, dp0y, dp0z, dq0);
    store(oi1, ib1, act1, dp1x, dp1y, dp1z, dq1);
}

inline int pick_split_j(int blocks_i, bool is_grouped) {
    if (is_grouped || blocks_i <= 1 || blocks_i >= 768) return 1;
    int sj = (768 + blocks_i - 1) / blocks_i;
    if (sj > blocks_i) sj = blocks_i;
    if (sj > 16) sj = 16;
    return sj < 1 ? 1 : sj;
}

} // anonymous namespace

void cuda_coulomb_fwd_launch(
    torch::Tensor pos, torch::Tensor q, torch::Tensor mask, torch::Tensor g_arr, torch::Tensor gptr,
    torch::Tensor bsph, torch::Tensor phi, torch::Tensor fld, int64_t n_atoms,
    double inv_sig, double c0, double inv_sig2, double far_pad,
    bool is_fast64, bool is_grouped, bool need_fld
) {
    if (n_atoms <= 0) return;
    int blocks = int((n_atoms + BLK - 1) / BLK);
    int split_j = pick_split_j(blocks, is_grouped);
    if (split_j > 1) { phi.zero_(); if (need_fld) fld.zero_(); }
    dim3 grid(blocks, split_j);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto run_s = [&](auto f64, auto grp, auto nf, auto sj) {
        cuda_coulomb_fwd_k<decltype(f64)::value, decltype(grp)::value, decltype(nf)::value, decltype(sj)::value><<<grid, NTH, 0, stream>>>(
            pos.data_ptr<float>(), q.data_ptr<float>(), mask.data_ptr<uint8_t>(),
            g_arr.data_ptr<int>(), gptr.data_ptr<int>(), bsph.data_ptr<float>(),
            phi.data_ptr<float>(), fld.data_ptr<float>(), int(n_atoms),
            float(inv_sig), float(c0), float(inv_sig2), float(far_pad));
    };
    auto run = [&](auto f64, auto grp, auto nf) {
        if (split_j > 1) run_s(f64, grp, nf, std::true_type{});
        else run_s(f64, grp, nf, std::false_type{});
    };
    if (is_fast64) { if (need_fld) run(std::true_type{}, std::false_type{}, std::true_type{}); else run(std::true_type{}, std::false_type{}, std::false_type{}); }
    else if (is_grouped) { if (need_fld) run(std::false_type{}, std::true_type{}, std::true_type{}); else run(std::false_type{}, std::true_type{}, std::false_type{}); }
    else { if (need_fld) run(std::false_type{}, std::false_type{}, std::true_type{}); else run(std::false_type{}, std::false_type{}, std::false_type{}); }
}

void cuda_coulomb_bwd_launch(
    torch::Tensor pos, torch::Tensor q, torch::Tensor mask, torch::Tensor g_arr, torch::Tensor gptr,
    torch::Tensor bsph, torch::Tensor gphi, torch::Tensor gfld, torch::Tensor dpos, torch::Tensor dq,
    int64_t n_atoms, double inv_sig, double c0, double inv_sig2, double far_pad,
    bool is_fast64, bool is_grouped, bool has_gfld, bool need_dpos, bool need_dq
) {
    if (n_atoms <= 0) return;
    int blocks = int((n_atoms + BLK - 1) / BLK);
    int split_j = pick_split_j(blocks, is_grouped);
    if (split_j > 1) { if (need_dpos) dpos.zero_(); if (need_dq) dq.zero_(); }
    dim3 grid(blocks, split_j);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto run_s = [&](auto f64, auto grp, auto hg, auto ndp, auto ndq, auto sj) {
        cuda_coulomb_bwd_k<decltype(f64)::value, decltype(grp)::value, decltype(hg)::value, decltype(ndp)::value, decltype(ndq)::value, decltype(sj)::value><<<grid, NTH, 0, stream>>>(
            pos.data_ptr<float>(), q.data_ptr<float>(), mask.data_ptr<uint8_t>(), g_arr.data_ptr<int>(), gptr.data_ptr<int>(), bsph.data_ptr<float>(),
            gphi.data_ptr<float>(), gfld.data_ptr<float>(), dpos.data_ptr<float>(), dq.data_ptr<float>(),
            int(n_atoms), float(inv_sig), float(c0), float(inv_sig2), float(far_pad));
    };
    auto dispatch_nd = [&](auto f64, auto grp, auto hg) {
        if (split_j > 1) {
            if (need_dpos && need_dq) run_s(f64, grp, hg, std::true_type{}, std::true_type{}, std::true_type{});
            else if (need_dpos) run_s(f64, grp, hg, std::true_type{}, std::false_type{}, std::true_type{});
            else if (need_dq) run_s(f64, grp, hg, std::false_type{}, std::true_type{}, std::true_type{});
        } else {
            if (need_dpos && need_dq) run_s(f64, grp, hg, std::true_type{}, std::true_type{}, std::false_type{});
            else if (need_dpos) run_s(f64, grp, hg, std::true_type{}, std::false_type{}, std::false_type{});
            else if (need_dq) run_s(f64, grp, hg, std::false_type{}, std::true_type{}, std::false_type{});
        }
    };
    if (is_fast64) { if (has_gfld) dispatch_nd(std::true_type{}, std::false_type{}, std::true_type{}); else dispatch_nd(std::true_type{}, std::false_type{}, std::false_type{}); }
    else if (is_grouped) { if (has_gfld) dispatch_nd(std::false_type{}, std::true_type{}, std::true_type{}); else dispatch_nd(std::false_type{}, std::true_type{}, std::false_type{}); }
    else { if (has_gfld) dispatch_nd(std::false_type{}, std::false_type{}, std::true_type{}); else dispatch_nd(std::false_type{}, std::false_type{}, std::false_type{}); }
}
"""
