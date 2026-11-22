"""CUDA C++ analytic backward of fused invariant block (`cuda_node_inv_bwd_k`) + C++ launch wrappers."""
from __future__ import annotations

CUDA_NODE_BWD_SRC = r"""
constexpr int G_U = 0, G_T = 1440, G_O = 2960, G_MAIN = 3632;
constexpr int GB_O_OFF = 2960, GB_TN_OFF = 3016, GB_VN_OFF = 3096;
constexpr int R_X2 = 0, R_GCV = 352, R_GCT = 544, R_GB2 = 704;

__global__ void cuda_node_inv_bwd_k(
    const float* __restrict__ vec, const float* __restrict__ agg, const float* __restrict__ ten,
    const float* __restrict__ W, const float* __restrict__ gin,
    float* __restrict__ g_vec, float* __restrict__ g_agg, float* __restrict__ g_ten,
    int n, int flags, float c2
) {
    int i = blockIdx.x;
    if (i >= n) return;
    int tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
    Lay L = lay(flags);
    bool later = (flags & F_LATER) != 0, has_ten = (flags & F_TEN) != 0, cumt = (flags & F_CUMT) != 0;
    __shared__ float X[232], G[G_MAIN], scl[6], dts[6];

    const float* a = agg + (long)i * L.out_dim;
    for (int e = tid; e < 96; e += 128) X[X_VN + e] = vec[(long)i * 96 + e] + a[L.av + e];
    for (int e = tid; e < 80; e += 128) {
        float t = a[L.at + e], t0 = has_ten ? ten[(long)i * 80 + e] : 0.0f;
        X[X_TN + e] = (has_ten && cumt) ? (t0 + t) : t;
    }
    for (int e = tid; e < 56; e += 128) X[X_O + e] = a[L.ao + e];
    __syncthreads();
    if (sg == 0) { float s = sumsq(X + X_VN, 96, lane); if (lane == 0) scl[0] = bound_scale(s, 32.0f, c2); }
    if (sg == 1) { float s = sumsq(X + X_TN, 80, lane); if (lane == 0) scl[1] = cumt ? bound_scale(s, 16.0f, c2) : 1.0f; }
    __syncthreads();

    float su = scl[0], st = scl[1];
    const float* gi = gin + (long)i * L.inv_dim;
    for (int o = tid; o < L.inv_dim; o += 128) {
        float g = gi[o];
        if (o < 128) {
            float p[3], q[3];
            proj<32, 3>(W + WA_UVPQ, 384, o, X + X_VN, su, p); proj<32, 3>(W + WA_UVPQ, 384, 128 + o, X + X_VN, su, q);
            storeg<3>(G + G_U, o, g, q); storeg<3>(G + G_U, 128 + o, g, p);
        } else if (o < 192) { int k = o - 128;
            float p[5], q[5];
            proj<16, 5>(W + WA_UVT, 192, k, X + X_TN, st, p); proj<16, 5>(W + WA_UVT, 192, 64 + k, X + X_TN, st, q);
            storeg<5>(G + G_T, k, g, q); storeg<5>(G + G_T, 64 + k, g, p);
        } else if (o < 256) { int k = o - 192;
            float p[3], q[3], t[5], m[6], d1[5], d2[3];
            proj<32, 3>(W + WA_UVPQ, 384, 256 + k, X + X_VN, su, p); proj<32, 3>(W + WA_UVPQ, 384, 320 + k, X + X_VN, su, q);
            proj<16, 5>(W + WA_UVT, 192, 128 + k, X + X_TN, st, t);
            l2pair(p, q, d1); storeg<5>(G + G_T, 128 + k, g, d1);
            mat6(t, m); matvec(m, q, d1); matvec(m, p, d2);
            storeg<3>(G + G_U, 256 + k, g, d1); storeg<3>(G + G_U, 320 + k, g, d2);
        } else if (o >= L.ang) {
            int h = o - L.ang;
            if (h < 16) {
                float t[5], m[6], d1[5], cof[6];
                proj<16, 5>(W + WA_YT, 112, h, X + X_TN, st, t); mat6(t, m);
                cof[0] = m[1] * m[2] - m[5] * m[5]; cof[1] = m[0] * m[2] - m[4] * m[4]; cof[2] = m[0] * m[1] - m[3] * m[3];
                cof[3] = m[4] * m[5] - m[3] * m[2]; cof[4] = m[3] * m[5] - m[1] * m[4]; cof[5] = m[3] * m[4] - m[0] * m[5];
                float gm[6] = {3.0f * cof[0], 3.0f * cof[1], 3.0f * cof[2], 6.0f * cof[3], 6.0f * cof[4], 6.0f * cof[5]};
                gm6_to_gt(gm, d1); storeg<5>(G + G_T, 192 + h, g, d1);
            } else if (h < 48) { int k = h - 16;
                float ta[5], tb[5], ma[6], mb[6], p[3], q[3], x[3], z[3], d1[5];
                proj<16, 5>(W + WA_YT, 112, 16 + k, X + X_TN, st, ta); mat6(ta, ma);
                proj<16, 5>(W + WA_YT, 112, 48 + k, X + X_TN, st, tb); mat6(tb, mb);
                proj<32, 3>(W + WA_YV, 96, k, X + X_VN, su, p); proj<32, 3>(W + WA_YV, 96, 32 + k, X + X_VN, su, q);
                matvec(ma, p, x); matvec(mb, q, z);
                l2pair(p, z, d1); storeg<5>(G + G_T, 192 + 16 + k, g, d1);
                l2pair(q, x, d1); storeg<5>(G + G_T, 192 + 48 + k, g, d1);
                matvec(ma, z, d1); storeg<3>(G + G_U, 384 + k, g, d1);
                matvec(mb, x, d1); storeg<3>(G + G_U, 384 + 32 + k, g, d1);
            } else if (h < 80) { int k = h - 48;
                float p[7], q[7];
                proj<8, 7>(W + WA_U3, 96, k, X + X_O, 1.0f, p); proj<8, 7>(W + WA_U3, 96, 32 + k, X + X_O, 1.0f, q);
                storeg<7>(G + G_O, k, g, q); storeg<7>(G + G_O, 32 + k, g, p);
            } else { int k = h - 80;
                float p[3], q[7], t[5], m[6], Y[7], gv[3], gm[6], d1[5];
                proj<8, 7>(W + WA_U3, 96, 64 + k, X + X_O, 1.0f, q); proj<32, 3>(W + WA_YV, 96, 64 + k, X + X_VN, su, p);
                proj<16, 5>(W + WA_YT, 112, 80 + k, X + X_TN, st, t); mat6(t, m); y3vm(p, m, Y);
                storeg<7>(G + G_O, 64 + k, g, Y);
                y3vm_bwd(p, m, q, gv, gm);
                storeg<3>(G + G_U, 384 + 64 + k, g, gv);
                gm6_to_gt(gm, d1); storeg<5>(G + G_T, 192 + 80 + k, g, d1);
            }
        }
    }
    __syncthreads();

    float gb_o[2][7];
    for (int ci = 0; ci < 2; ++ci) {
        int c = sg + ci * 4;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f;
        const float* w1 = W + WA_U3 + c * 96;
        for (int r = lane; r < 96; r += 32) {
            float w = __ldg(w1 + r);
            int idx = G_O + r * 7;
            a0 = fmaf(w, G[idx], a0); a1 = fmaf(w, G[idx + 1], a1); a2 = fmaf(w, G[idx + 2], a2); a3 = fmaf(w, G[idx + 3], a3);
            a4 = fmaf(w, G[idx + 4], a4); a5 = fmaf(w, G[idx + 5], a5); a6 = fmaf(w, G[idx + 6], a6);
        }
        gb_o[ci][0] = warp_sum(a0); gb_o[ci][1] = warp_sum(a1); gb_o[ci][2] = warp_sum(a2); gb_o[ci][3] = warp_sum(a3);
        gb_o[ci][4] = warp_sum(a4); gb_o[ci][5] = warp_sum(a5); gb_o[ci][6] = warp_sum(a6);
    }
    __syncthreads();
    if (lane == 0) {
        for (int ci = 0; ci < 2; ++ci) { int c = sg + ci * 4; for (int m = 0; m < 7; ++m) G[GB_O_OFF + c * 7 + m] = gb_o[ci][m]; }
    }
    for (int c = sg; c < 16; c += 4) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f;
        const float* w1 = W + WA_UVT + c * 192;
        for (int r = lane; r < 192; r += 32) {
            float w = __ldg(w1 + r); int idx = G_T + r * 5;
            a0 = fmaf(w, G[idx], a0); a1 = fmaf(w, G[idx + 1], a1); a2 = fmaf(w, G[idx + 2], a2); a3 = fmaf(w, G[idx + 3], a3); a4 = fmaf(w, G[idx + 4], a4);
        }
        const float* w2 = W + WA_YT + c * 112;
        for (int r = lane; r < 112; r += 32) {
            float w = __ldg(w2 + r); int idx = G_T + (192 + r) * 5;
            a0 = fmaf(w, G[idx], a0); a1 = fmaf(w, G[idx + 1], a1); a2 = fmaf(w, G[idx + 2], a2); a3 = fmaf(w, G[idx + 3], a3); a4 = fmaf(w, G[idx + 4], a4);
        }
        a0 = warp_sum(a0); a1 = warp_sum(a1); a2 = warp_sum(a2); a3 = warp_sum(a3); a4 = warp_sum(a4);
        if (lane == 0) {
            G[GB_TN_OFF + c * 5] = a0; G[GB_TN_OFF + c * 5 + 1] = a1; G[GB_TN_OFF + c * 5 + 2] = a2;
            G[GB_TN_OFF + c * 5 + 3] = a3; G[GB_TN_OFF + c * 5 + 4] = a4;
        }
    }
    for (int c = sg; c < 32; c += 4) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        const float* w1 = W + WA_UVPQ + c * 384;
        for (int r = lane; r < 384; r += 32) {
            float w = __ldg(w1 + r); int idx = G_U + r * 3;
            a0 = fmaf(w, G[idx], a0); a1 = fmaf(w, G[idx + 1], a1); a2 = fmaf(w, G[idx + 2], a2);
        }
        const float* w2 = W + WA_YV + c * 96;
        for (int r = lane; r < 96; r += 32) {
            float w = __ldg(w2 + r); int idx = G_U + (384 + r) * 3;
            a0 = fmaf(w, G[idx], a0); a1 = fmaf(w, G[idx + 1], a1); a2 = fmaf(w, G[idx + 2], a2);
        }
        a0 = warp_sum(a0); a1 = warp_sum(a1); a2 = warp_sum(a2);
        if (lane == 0) { G[GB_VN_OFF + c * 3] = a0; G[GB_VN_OFF + c * 3 + 1] = a1; G[GB_VN_OFF + c * 3 + 2] = a2; }
    }
    __syncthreads();

    if (later) {
        for (int e = tid; e < 96; e += 128) { G[R_X2 + e] = vec[(long)i * 96 + e]; G[R_X2 + 96 + e] = a[L.av + e]; }
        for (int e = tid; e < 80; e += 128) { G[R_X2 + 192 + e] = has_ten ? ten[(long)i * 80 + e] : 0.0f; G[R_X2 + 272 + e] = a[L.at + e]; }
        __syncthreads();
        if (sg == 0) { float d = dotsg(G + GB_VN_OFF, X + X_VN, 96, lane); if (lane == 0) dts[0] = d; }
        if (sg == 1) { float d = dotsg(G + GB_TN_OFF, X + X_TN, 80, lane); if (lane == 0) dts[1] = d; }
        if (sg == 2) { float s = sumsq(G + R_X2, 96, lane), s2 = sumsq(G + R_X2 + 96, 96, lane); if (lane == 0) { scl[2] = bound_scale(s, 32.0f, c2); scl[3] = bound_scale(s2, 32.0f, c2); } }
        if (sg == 3) { float s = sumsq(G + R_X2 + 192, 80, lane), s2 = sumsq(G + R_X2 + 272, 80, lane); if (lane == 0) { scl[4] = bound_scale(s, 16.0f, c2); scl[5] = bound_scale(s2, 16.0f, c2); } }
        __syncthreads();
        if (tid < 32) {
            int k = tid; float g = gi[272 + k], p[3], q[3];
            proj<32, 3>(W + WA_UCV, 32, k, G + R_X2, scl[2], p); proj<32, 3>(W + WA_VCV, 32, k, G + R_X2 + 96, scl[3], q);
            storeg<3>(G + R_GCV, k, g, q); storeg<3>(G + R_GCV, 32 + k, g, p);
        } else if (tid < 48) {
            int k = tid - 32;
            if (has_ten) {
                float g = gi[304 + k], p[5], q[5];
                proj<16, 5>(W + WA_UCT, 16, k, G + R_X2 + 192, scl[4], p); proj<16, 5>(W + WA_VCT, 16, k, G + R_X2 + 272, scl[5], q);
                storeg<5>(G + R_GCT, k, g, q); storeg<5>(G + R_GCT, 16 + k, g, p);
            } else {
                for (int m = 0; m < 5; ++m) { G[R_GCT + k * 5 + m] = 0.0f; G[R_GCT + (16 + k) * 5 + m] = 0.0f; }
            }
        }
        __syncthreads();
        for (int c = sg; c < 32; c += 4) {
            float wv = __ldg(W + WA_UCV + c * 32 + lane), wp = __ldg(W + WA_VCV + c * 32 + lane);
            int iv = R_GCV + lane * 3, ip = R_GCV + (32 + lane) * 3;
            float v0 = warp_sum(wv * G[iv]), v1 = warp_sum(wv * G[iv + 1]), v2 = warp_sum(wv * G[iv + 2]);
            float p0 = warp_sum(wp * G[ip]), p1 = warp_sum(wp * G[ip + 1]), p2 = warp_sum(wp * G[ip + 2]);
            if (lane == 0) {
                G[R_GB2 + c * 3] = v0; G[R_GB2 + c * 3 + 1] = v1; G[R_GB2 + c * 3 + 2] = v2;
                G[R_GB2 + 96 + c * 3] = p0; G[R_GB2 + 96 + c * 3 + 1] = p1; G[R_GB2 + 96 + c * 3 + 2] = p2;
            }
        }
        for (int c = sg; c < 16; c += 4) {
            float wt = (lane < 16) ? __ldg(W + WA_UCT + c * 16 + lane) : 0.0f;
            float wq = (lane < 16) ? __ldg(W + WA_VCT + c * 16 + lane) : 0.0f;
            int it = R_GCT + (lane & 15) * 5, iq = R_GCT + (16 + (lane & 15)) * 5;
            float t0 = warp_sum(wt * G[it]), t1 = warp_sum(wt * G[it + 1]), t2 = warp_sum(wt * G[it + 2]), t3 = warp_sum(wt * G[it + 3]), t4 = warp_sum(wt * G[it + 4]);
            float q0 = warp_sum(wq * G[iq]), q1 = warp_sum(wq * G[iq + 1]), q2 = warp_sum(wq * G[iq + 2]), q3 = warp_sum(wq * G[iq + 3]), q4 = warp_sum(wq * G[iq + 4]);
            if (lane == 0) {
                G[R_GB2 + 192 + c * 5] = t0; G[R_GB2 + 192 + c * 5 + 1] = t1; G[R_GB2 + 192 + c * 5 + 2] = t2; G[R_GB2 + 192 + c * 5 + 3] = t3; G[R_GB2 + 192 + c * 5 + 4] = t4;
                G[R_GB2 + 272 + c * 5] = q0; G[R_GB2 + 272 + c * 5 + 1] = q1; G[R_GB2 + 272 + c * 5 + 2] = q2; G[R_GB2 + 272 + c * 5 + 3] = q3; G[R_GB2 + 272 + c * 5 + 4] = q4;
            }
        }
        __syncthreads();
        if (sg == 2) { float d = dotsg(G + R_GB2, G + R_X2, 96, lane), d2 = dotsg(G + R_GB2 + 96, G + R_X2 + 96, 96, lane); if (lane == 0) { dts[2] = d; dts[3] = d2; } }
        if (sg == 3) { float d = dotsg(G + R_GB2 + 192, G + R_X2 + 192, 80, lane), d2 = dotsg(G + R_GB2 + 272, G + R_X2 + 272, 80, lane); if (lane == 0) { dts[4] = d; dts[5] = d2; } }
        __syncthreads();
    } else {
        if (sg == 0) { float d = dotsg(G + GB_VN_OFF, X + X_VN, 96, lane); if (lane == 0) dts[0] = d; }
        if (sg == 1) { float d = dotsg(G + GB_TN_OFF, X + X_TN, 80, lane); if (lane == 0) dts[1] = d; }
        __syncthreads();
    }

    float k0 = scl[0] * scl[0] * scl[0] / (c2 * 32.0f) * dts[0];
    float k1 = scl[1] * scl[1] * scl[1] / (c2 * 16.0f) * dts[1];
    float k2 = later ? (scl[2] * scl[2] * scl[2] / (c2 * 32.0f) * dts[2]) : 0.0f;
    float k3 = later ? (scl[3] * scl[3] * scl[3] / (c2 * 32.0f) * dts[3]) : 0.0f;
    float k4 = later ? (scl[4] * scl[4] * scl[4] / (c2 * 16.0f) * dts[4]) : 0.0f;
    float k5 = later ? (scl[5] * scl[5] * scl[5] / (c2 * 16.0f) * dts[5]) : 0.0f;
    float* ga = g_agg + (long)i * L.out_dim;
    for (int e = tid; e < L.out_dim; e += 128) {
        float r = 0.0f;
        if (e >= L.av && e < L.av + 96) { int j = e - L.av;
            r = scl[0] * G[GB_VN_OFF + j] - k0 * X[X_VN + j];
            if (later) r += scl[3] * G[R_GB2 + 96 + j] - k3 * G[R_X2 + 96 + j];
        } else if (e >= L.at && e < L.at + 80) { int j = e - L.at;
            r = cumt ? (scl[1] * G[GB_TN_OFF + j] - k1 * X[X_TN + j]) : G[GB_TN_OFF + j];
            if (later) r += scl[5] * G[R_GB2 + 272 + j] - k5 * G[R_X2 + 272 + j];
        } else if (e >= L.ao && e < L.ao + 56) { r = G[GB_O_OFF + (e - L.ao)];
        } else if (later && e >= L.alon && e < L.alon + 32) { r = gi[320 + (e - L.alon)];
        } else if (later && e >= L.alon2 && e < L.alon2 + 16) { r = gi[256 + (e - L.alon2)]; }
        ga[e] = r;
    }
    for (int j = tid; j < 96; j += 128) {
        float r = scl[0] * G[GB_VN_OFF + j] - k0 * X[X_VN + j];
        if (later) r += scl[2] * G[R_GB2 + j] - k2 * G[R_X2 + j];
        g_vec[(long)i * 96 + j] = r;
    }
    if (has_ten) {
        for (int j = tid; j < 80; j += 128) {
            float r = cumt ? (scl[1] * G[GB_TN_OFF + j] - k1 * X[X_TN + j]) : 0.0f;
            if (later) r += scl[4] * G[R_GB2 + 192 + j] - k4 * G[R_X2 + 192 + j];
            g_ten[(long)i * 80 + j] = r;
        }
    }
}

} // anonymous namespace

torch::Tensor cuda_node_inv_fwd(torch::Tensor vec, torch::Tensor agg, torch::Tensor ten, torch::Tensor w, int64_t flags, double c2) {
    int64_t n = agg.size(0);
    int64_t inv_dim = (flags & F_LATER) ? 464 : 368;
    auto out = torch::empty({n, inv_dim}, agg.options());
    if (n > 0) {
        cuda_node_inv_fwd_k<<<int(n), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            vec.data_ptr<float>(), agg.data_ptr<float>(), ten.data_ptr<float>(),
            w.data_ptr<float>(), out.data_ptr<float>(), int(n), int(flags), float(c2));
    }
    return out;
}

std::vector<torch::Tensor> cuda_node_inv_bwd(torch::Tensor vec, torch::Tensor agg, torch::Tensor ten, torch::Tensor w, torch::Tensor g, int64_t flags, double c2) {
    int64_t n = agg.size(0);
    int64_t od = (flags & F_LATER) ? 408 : 360;
    auto g_vec = torch::empty({n, 32, 3}, vec.options());
    auto g_agg = torch::empty({n, od}, agg.options());
    auto g_ten = (flags & F_TEN) ? torch::empty({n, 16, 5}, agg.options()) : torch::empty({0}, agg.options());
    auto dummy = (flags & F_TEN) ? g_ten : torch::empty({1}, agg.options());
    if (n > 0) {
        cuda_node_inv_bwd_k<<<int(n), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            vec.data_ptr<float>(), agg.data_ptr<float>(), ten.data_ptr<float>(), w.data_ptr<float>(),
            g.data_ptr<float>(), g_vec.data_ptr<float>(), g_agg.data_ptr<float>(), dummy.data_ptr<float>(),
            int(n), int(flags), float(c2));
    }
    return {g_vec, g_agg, g_ten};
}

std::vector<torch::Tensor> cuda_node_vec_fwd(torch::Tensor vec, torch::Tensor agg, torch::Tensor ten, torch::Tensor gates, torch::Tensor w, int64_t flags, double c2) {
    int64_t n = agg.size(0);
    auto v_out = torch::empty({n, 32, 3}, vec.options());
    auto t_out = torch::empty({n, 16, 5}, agg.options());
    if (n > 0) {
        cuda_node_vec_fwd_k<<<int(n), 64, 0, at::cuda::getCurrentCUDAStream()>>>(
            vec.data_ptr<float>(), agg.data_ptr<float>(), ten.data_ptr<float>(), gates.data_ptr<float>(),
            w.data_ptr<float>(), v_out.data_ptr<float>(), t_out.data_ptr<float>(), int(n), int(flags), float(c2));
    }
    return {v_out, t_out};
}

std::vector<torch::Tensor> cuda_node_vec_bwd(torch::Tensor vec, torch::Tensor agg, torch::Tensor ten, torch::Tensor gates, torch::Tensor w, torch::Tensor g_v, torch::Tensor g_t, int64_t flags, double c2) {
    int64_t n = agg.size(0);
    int64_t od = (flags & F_LATER) ? 408 : 360;
    auto g_vec = torch::empty({n, 32, 3}, vec.options());
    auto g_agg = torch::empty({n, od}, agg.options());
    auto g_ten = (flags & F_TEN) ? torch::empty({n, 16, 5}, agg.options()) : torch::empty({0}, agg.options());
    auto g_gates = torch::empty({n, 96}, gates.options());
    auto dummy = (flags & F_TEN) ? g_ten : torch::empty({1}, agg.options());
    if (n > 0) {
        cuda_node_vec_bwd_k<<<int(n), 64, 0, at::cuda::getCurrentCUDAStream()>>>(
            vec.data_ptr<float>(), agg.data_ptr<float>(), ten.data_ptr<float>(), gates.data_ptr<float>(),
            w.data_ptr<float>(), g_v.data_ptr<float>(), g_t.data_ptr<float>(),
            g_vec.data_ptr<float>(), g_agg.data_ptr<float>(), dummy.data_ptr<float>(), g_gates.data_ptr<float>(),
            int(n), int(flags), float(c2));
    }
    return {g_vec, g_agg, g_ten, g_gates};
}

torch::Tensor cuda_swiglu_fwd(torch::Tensor gv) {
    int64_t n = gv.size(0), h = gv.size(1) / 2;
    auto out = torch::empty({n, h}, gv.options());
    if (n > 0) {
        int h4 = int(h / 4), tot = int(n) * h4;
        cuda_swiglu_fwd_v4_k<<<(tot + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(gv.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()), int(n), h4);
    }
    return out;
}

torch::Tensor cuda_swiglu_bwd(torch::Tensor gv, torch::Tensor gy) {
    int64_t n = gv.size(0), two_h = gv.size(1), h4 = two_h / 8;
    auto g_gv = torch::empty({n, two_h}, gv.options());
    if (n > 0) {
        int tot = int(n * h4);
        cuda_swiglu_bwd_v4_k<<<(tot + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(gv.data_ptr<float>()),
            reinterpret_cast<const float4*>(gy.data_ptr<float>()),
            reinterpret_cast<float4*>(g_gv.data_ptr<float>()), int(n), int(h4));
    }
    return g_gv;
}

torch::Tensor cuda_env_swiglu_proj_fwd(torch::Tensor gv, torch::Tensor h_env, torch::Tensor w_t, torch::Tensor bias) {
    int64_t n = gv.size(0), h = gv.size(1) / 2;
    auto out = torch::empty({n, h}, gv.options());
    if (n > 0) {
        auto dm = torch::addmm(bias, h_env, w_t).contiguous();
        int h4 = int(h / 4), tot = int(n) * h4;
        cuda_env_swiglu_act_fwd_v4_k<<<(tot + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(gv.data_ptr<float>()),
            reinterpret_cast<const float4*>(dm.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()), int(n), h4);
    }
    return out;
}

std::vector<torch::Tensor> cuda_env_swiglu_proj_bwd(torch::Tensor gv, torch::Tensor h_env, torch::Tensor w_t, torch::Tensor bias, torch::Tensor gy) {
    int64_t n = gv.size(0), two_h = gv.size(1), h = two_h / 2;
    auto g_gv = torch::empty({n, two_h}, gv.options());
    if (n == 0) return {g_gv, torch::empty({0, 32}, h_env.options())};
    auto dm = torch::addmm(bias, h_env, w_t).contiguous();
    auto g_dm = torch::empty({n, two_h}, gv.options());
    int h4 = int(h / 4), tot = int(n) * h4;
    cuda_env_swiglu_act_bwd_v4_k<<<(tot + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float4*>(gv.data_ptr<float>()),
        reinterpret_cast<const float4*>(dm.data_ptr<float>()),
        reinterpret_cast<const float4*>(gy.data_ptr<float>()),
        reinterpret_cast<float4*>(g_gv.data_ptr<float>()),
        reinterpret_cast<float4*>(g_dm.data_ptr<float>()), int(n), h4);
    auto g_h = torch::mm(g_dm, w_t.t());
    return {g_gv, g_h};
}
"""
