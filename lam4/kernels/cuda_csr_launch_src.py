"""CUDA C++ launch wrappers for Dual-CSR forward & backward kernels (auto multi-warp-per-atom for N <= 8192)."""
from __future__ import annotations

CUDA_CSR_LAUNCH_SRC = r"""
inline int pick_csr_wpa(int64_t n_atoms) {
    if (n_atoms <= 2048) return 4;
    if (n_atoms <= 8192) return 2;
    return 1;
}

std::vector<torch::Tensor> cuda_csr_fwd_launch(
    torch::Tensor pos, torch::Tensor recv_ptr, torch::Tensor send_idx,
    torch::Tensor shift_in, torch::Tensor cell, torch::Tensor graph,
    torch::Tensor z, torch::Tensor rmin_tab, torch::Tensor rho,
    torch::Tensor poly4, torch::Tensor gates, torch::Tensor qn,
    torch::Tensor kn, torch::Tensor vec, torch::Tensor ten,
    int64_t n_atoms, int64_t grid, int64_t flags, bool later,
    double r_cut, double r_on, double max_doff_r2
) {
    auto opts = pos.options().dtype(torch::kFloat32);
    int64_t od = later ? 408 : 360;
    auto out_agg = torch::empty({n_atoms, od}, opts);
    auto out_zc = torch::empty({n_atoms, 4}, opts);
    auto out_dsum = torch::empty({n_atoms}, opts);
    if (n_atoms == 0) return {out_agg, out_zc, out_dsum};
    int wpa = (n_atoms <= 1024) ? 4 : ((later && n_atoms <= 4096) ? 2 : 1);
    int atoms_per_cta = 4 / wpa;
    dim3 blocks(int((n_atoms + atoms_per_cta - 1) / atoms_per_cta));
    dim3 threads(128);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float4* p4 = reinterpret_cast<const float4*>(poly4.data_ptr<float>());
    auto run = [&](auto lat, auto w_ic) {
        cuda_csr_fwd_final_k<decltype(lat)::value, decltype(w_ic)::value><<<blocks, threads, 0, stream>>>(
            pos.data_ptr<float>(), recv_ptr.data_ptr<int>(), send_idx.data_ptr<int>(),
            shift_in.data_ptr<float>(), cell.data_ptr<float>(), graph.data_ptr<int64_t>(),
            z.data_ptr<int64_t>(), rmin_tab.data_ptr<float>(), rho.data_ptr<float>(),
            p4, gates.data_ptr<float>(), qn.data_ptr<float>(), kn.data_ptr<float>(),
            vec.data_ptr<float>(), ten.data_ptr<float>(),
            out_agg.data_ptr<float>(), out_zc.data_ptr<float>(), out_dsum.data_ptr<float>(),
            int(n_atoms), int(grid), int(flags), float(r_cut), float(r_on), float(max_doff_r2));
    };
    auto run_w = [&](auto lat) {
        if (wpa == 4) run(lat, std::integral_constant<int, 4>{});
        else if (wpa == 2) run(lat, std::integral_constant<int, 2>{});
        else run(lat, std::integral_constant<int, 1>{});
    };
    if (later) run_w(std::true_type{});
    else run_w(std::false_type{});
    return {out_agg, out_zc, out_dsum};
}

std::vector<torch::Tensor> cuda_csr_bwd_launch(
    torch::Tensor pos, torch::Tensor recv_ptr, torch::Tensor send_idx,
    torch::Tensor send_ptr, torch::Tensor send_perm, torch::Tensor recv_idx,
    torch::Tensor shift_in, torch::Tensor cell, torch::Tensor graph,
    torch::Tensor z, torch::Tensor rmin_tab, torch::Tensor rho,
    torch::Tensor poly4, torch::Tensor gates, torch::Tensor qn,
    torch::Tensor kn, torch::Tensor vec, torch::Tensor ten,
    torch::Tensor agg, torch::Tensor zc, torch::Tensor dsum,
    torch::Tensor g_agg, torch::Tensor g_dsum_ext,
    int64_t n_atoms, int64_t n_edges, int64_t grid, int64_t flags,
    bool later, bool one_pass, double r_cut, double r_on, double max_doff_r2
) {
    auto opts = pos.options().dtype(torch::kFloat32);
    bool has_ten = later && ((flags & F_TEN) != 0);
    bool need_vir = (flags & F_VIR) != 0;
    bool need_drho = (flags & F_DRHO) != 0;
    bool need_sf = (flags & F_SEND_FEAT) != 0;
    bool need_dqn = (flags & F_DQN) != 0;

    auto d_pos = one_pass ? torch::zeros({n_atoms, 3}, opts) : torch::empty({n_atoms, 3}, opts);
    auto d_qn = need_dqn ? torch::empty_like(qn) : torch::empty({0}, opts);
    auto d_gates = need_sf ? (one_pass ? torch::zeros({n_atoms, 184}, opts) : torch::empty({n_atoms, 184}, opts)) : torch::empty({0}, opts);
    auto d_kn = need_sf ? (one_pass ? torch::zeros_like(kn) : torch::empty_like(kn)) : torch::empty({0}, opts);
    auto d_vec = (later && need_sf) ? (one_pass ? torch::zeros({n_atoms, 32, 3}, opts) : torch::empty({n_atoms, 32, 3}, opts)) : torch::empty({0}, opts);
    auto d_ten = (has_ten && need_sf) ? (one_pass ? torch::zeros({n_atoms, 16, 5}, opts) : torch::empty({n_atoms, 16, 5}, opts)) : torch::empty({0}, opts);
    auto d_rho = need_drho ? (one_pass ? torch::zeros({n_atoms}, opts) : torch::empty({n_atoms}, opts)) : torch::empty({0}, opts);
    auto d_cell = need_vir ? torch::zeros_like(cell) : torch::empty({0}, opts);
    if (n_atoms == 0 || n_edges == 0) return {d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell};

    auto dummy = torch::empty({1}, opts);
    auto eg12 = one_pass ? dummy : torch::empty({n_edges, 12}, opts);
    int wpa = one_pass ? pick_csr_wpa(n_atoms) : 1;
    int atoms_per_cta = 4 / wpa;
    dim3 blocks_r(int((n_atoms + atoms_per_cta - 1) / atoms_per_cta));
    dim3 blocks_s(int((n_atoms + 3) / 4));
    dim3 threads(128);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float4* p4 = reinterpret_cast<const float4*>(poly4.data_ptr<float>());
    float* p_dqn = need_dqn ? d_qn.data_ptr<float>() : dummy.data_ptr<float>();
    float* p_dg = need_sf ? d_gates.data_ptr<float>() : dummy.data_ptr<float>();
    float* p_dk = need_sf ? d_kn.data_ptr<float>() : dummy.data_ptr<float>();
    float* p_dv = (later && need_sf) ? d_vec.data_ptr<float>() : dummy.data_ptr<float>();
    float* p_dt = (has_ten && need_sf) ? d_ten.data_ptr<float>() : dummy.data_ptr<float>();
    float* p_dr = need_drho ? d_rho.data_ptr<float>() : dummy.data_ptr<float>();
    float* p_dc = need_vir ? d_cell.data_ptr<float>() : dummy.data_ptr<float>();

    auto run_r = [&](auto lat, auto op, auto w_ic) {
        cuda_csr_bwd_recv_k<decltype(lat)::value, decltype(op)::value, decltype(w_ic)::value><<<blocks_r, threads, 0, stream>>>(
            pos.data_ptr<float>(), recv_ptr.data_ptr<int>(), send_idx.data_ptr<int>(),
            shift_in.data_ptr<float>(), cell.data_ptr<float>(), graph.data_ptr<int64_t>(),
            z.data_ptr<int64_t>(), rmin_tab.data_ptr<float>(), rho.data_ptr<float>(),
            p4, gates.data_ptr<float>(), qn.data_ptr<float>(), kn.data_ptr<float>(),
            vec.data_ptr<float>(), ten.data_ptr<float>(), agg.data_ptr<float>(),
            zc.data_ptr<float>(), dsum.data_ptr<float>(), g_agg.data_ptr<float>(), g_dsum_ext.data_ptr<float>(),
            d_pos.data_ptr<float>(), p_dqn, p_dg, p_dk, p_dv, p_dt, eg12.data_ptr<float>(), p_dc, p_dr,
            int(n_atoms), int(grid), int(flags), float(r_cut), float(r_on), float(max_doff_r2));
    };
    auto run_s = [&](auto lat) {
        cuda_csr_bwd_send_k<decltype(lat)::value><<<blocks_s, threads, 0, stream>>>(
            pos.data_ptr<float>(), send_ptr.data_ptr<int>(), send_perm.data_ptr<int>(),
            recv_idx.data_ptr<int>(), shift_in.data_ptr<float>(), cell.data_ptr<float>(),
            graph.data_ptr<int64_t>(), z.data_ptr<int64_t>(), rmin_tab.data_ptr<float>(),
            rho.data_ptr<float>(), p4, qn.data_ptr<float>(), g_agg.data_ptr<float>(), eg12.data_ptr<float>(),
            d_pos.data_ptr<float>(), p_dg, p_dk, p_dv, p_dt, p_dr,
            int(n_atoms), int(grid), int(flags), float(r_cut), float(r_on), float(max_doff_r2));
    };
    if (one_pass) {
        auto run_w = [&](auto lat) {
            if (wpa == 4) run_r(lat, std::true_type{}, std::integral_constant<int, 4>{});
            else if (wpa == 2) run_r(lat, std::true_type{}, std::integral_constant<int, 2>{});
            else run_r(lat, std::true_type{}, std::integral_constant<int, 1>{});
        };
        if (later) run_w(std::true_type{});
        else run_w(std::false_type{});
    } else {
        if (later) {
            run_r(std::true_type{}, std::false_type{}, std::integral_constant<int, 1>{});
            run_s(std::true_type{});
        } else {
            run_r(std::false_type{}, std::false_type{}, std::integral_constant<int, 1>{});
            run_s(std::false_type{});
        }
    }
    return {d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, d_cell};
}
"""
