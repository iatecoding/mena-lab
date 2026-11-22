"""C++ (CPU) Dual-CSR entry points: tensor checks, float/double dispatch and the pybind module (GIL released while running)."""
from __future__ import annotations

CPU_CSR_GLUE = r"""
namespace lam4cpu {

template <typename T> inline const T* fptr(const at::Tensor& t) { return t.numel() ? t.data_ptr<T>() : nullptr; }
inline const int* iptr(const at::Tensor& t) { return t.numel() ? t.data_ptr<int>() : nullptr; }
inline const int64_t* lptr(const at::Tensor& t) { return t.numel() ? t.data_ptr<int64_t>() : nullptr; }

inline void check_f(const at::Tensor& t, at::ScalarType st, const char* name) {
    TORCH_CHECK(t.numel() == 0 || (t.scalar_type() == st && t.is_contiguous() && t.device().is_cpu()), "lam4 cpu_csr: bad tensor ", name);
}
inline void check_i(const at::Tensor& t, at::ScalarType st, const char* name) {
    TORCH_CHECK(t.numel() == 0 || (t.scalar_type() == st && t.is_contiguous() && t.device().is_cpu()), "lam4 cpu_csr: bad index tensor ", name);
}

template <typename T>
Args<T> make_args(const at::Tensor& pos, const at::Tensor& recv_ptr, const at::Tensor& send_idx, const at::Tensor& send_ptr,
                  const at::Tensor& send_perm, const at::Tensor& recv_idx, const at::Tensor& shift_in, const at::Tensor& cell,
                  const at::Tensor& graph, const at::Tensor& z, const at::Tensor& rmin, const at::Tensor& rho, const at::Tensor& poly,
                  const at::Tensor& gates, const at::Tensor& qn, const at::Tensor& kn, const at::Tensor& vec, const at::Tensor& ten,
                  int64_t flags, double r_cut, double r_on, double max_doff_r2) {
    const auto st = pos.scalar_type();
    check_f(pos, st, "pos"); check_f(shift_in, st, "shift"); check_f(cell, st, "cell"); check_f(rmin, st, "rmin");
    check_f(rho, st, "rho"); check_f(poly, st, "poly"); check_f(gates, st, "gates"); check_f(qn, st, "qn"); check_f(kn, st, "kn");
    check_f(vec, st, "vec"); check_f(ten, st, "ten");
    check_i(recv_ptr, at::kInt, "recv_ptr"); check_i(send_idx, at::kInt, "send_idx"); check_i(send_ptr, at::kInt, "send_ptr");
    check_i(send_perm, at::kInt, "send_perm"); check_i(recv_idx, at::kInt, "recv_idx");
    check_i(graph, at::kLong, "graph"); check_i(z, at::kLong, "z");
    Args<T> a;
    a.pos = fptr<T>(pos); a.recv_ptr = iptr(recv_ptr); a.send_idx = iptr(send_idx); a.send_ptr = iptr(send_ptr);
    a.send_perm = iptr(send_perm); a.recv_idx = iptr(recv_idx); a.shift_in = fptr<T>(shift_in); a.cell = fptr<T>(cell);
    a.graph = lptr(graph); a.z = lptr(z); a.rmin = fptr<T>(rmin); a.rho = fptr<T>(rho); a.poly = fptr<T>(poly);
    a.gates = fptr<T>(gates); a.qn = fptr<T>(qn); a.kn = fptr<T>(kn); a.vec = fptr<T>(vec); a.ten = fptr<T>(ten);
    a.n_atoms = (int)pos.size(0); a.grid = (int)poly.size(0); a.flags = (int)flags;
    a.r_cut = T(r_cut); a.r_on = T(r_on); a.max_doff_r2 = T(max_doff_r2);
    return a;
}

std::vector<at::Tensor> csr_fwd(at::Tensor pos, at::Tensor recv_ptr, at::Tensor send_idx, at::Tensor shift_in, at::Tensor cell,
                                at::Tensor graph, at::Tensor z, at::Tensor rmin, at::Tensor rho, at::Tensor poly, at::Tensor gates,
                                at::Tensor qn, at::Tensor kn, at::Tensor vec, at::Tensor ten, int64_t flags, bool later,
                                double r_cut, double r_on, double max_doff_r2) {
    const int64_t n = pos.size(0), od = later ? 408 : 360;
    auto opt = pos.options();
    at::Tensor num = at::empty({n, od}, opt), zc = at::empty({n, 4}, opt), dsum = at::empty({n}, opt);
    at::Tensor none = at::empty({0}, recv_ptr.options());
    if (pos.scalar_type() == at::kDouble) {
        auto a = make_args<double>(pos, recv_ptr, send_idx, none, none, none, shift_in, cell, graph, z, rmin, rho, poly, gates, qn, kn, vec, ten, flags, r_cut, r_on, max_doff_r2);
        if (later) csr_fwd_kernel<double, true>(a, num.data_ptr<double>(), zc.data_ptr<double>(), dsum.data_ptr<double>());
        else csr_fwd_kernel<double, false>(a, num.data_ptr<double>(), zc.data_ptr<double>(), dsum.data_ptr<double>());
    } else {
        TORCH_CHECK(pos.scalar_type() == at::kFloat, "lam4 cpu_csr: float32 or float64 only");
        auto a = make_args<float>(pos, recv_ptr, send_idx, none, none, none, shift_in, cell, graph, z, rmin, rho, poly, gates, qn, kn, vec, ten, flags, r_cut, r_on, max_doff_r2);
        if (later) csr_fwd_kernel<float, true>(a, num.data_ptr<float>(), zc.data_ptr<float>(), dsum.data_ptr<float>());
        else csr_fwd_kernel<float, false>(a, num.data_ptr<float>(), zc.data_ptr<float>(), dsum.data_ptr<float>());
    }
    return {num, zc, dsum};
}

template <typename T>
void run_bwd(const Args<T>& a, bool later, const at::Tensor& g_num, const at::Tensor& g_zc, const at::Tensor& g_dsum, BwdOut<T>& o) {
    if (later) { csr_bwd_recv<T, true>(a, g_num.data_ptr<T>(), g_zc.data_ptr<T>(), g_dsum.data_ptr<T>(), o); csr_bwd_send<T, true>(a, g_num.data_ptr<T>(), o); }
    else { csr_bwd_recv<T, false>(a, g_num.data_ptr<T>(), g_zc.data_ptr<T>(), g_dsum.data_ptr<T>(), o); csr_bwd_send<T, false>(a, g_num.data_ptr<T>(), o); }
}

template <typename T> inline T* wptr(at::Tensor& t) { return t.numel() ? t.data_ptr<T>() : nullptr; }

// Returns d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, vir_atom [N, 9], edge_grad8 [E, 8] (absent ones are empty).
std::vector<at::Tensor> csr_bwd(at::Tensor pos, at::Tensor recv_ptr, at::Tensor send_idx, at::Tensor send_ptr, at::Tensor send_perm,
                                at::Tensor recv_idx, at::Tensor shift_in, at::Tensor cell, at::Tensor graph, at::Tensor z,
                                at::Tensor rmin, at::Tensor rho, at::Tensor poly, at::Tensor gates, at::Tensor qn, at::Tensor kn,
                                at::Tensor vec, at::Tensor ten, at::Tensor g_num, at::Tensor g_zc, at::Tensor g_dsum,
                                int64_t flags, bool later, double r_cut, double r_on, double max_doff_r2) {
    const int64_t n = pos.size(0), n_e = send_idx.size(0);
    const bool has_ten = later && (flags & FLAG_TEN), need_vir = (flags & FLAG_VIR) != 0, need_drho = (flags & FLAG_DRHO) != 0;
    auto opt = pos.options();
    const auto st = pos.scalar_type();
    check_f(g_num, st, "g_num"); check_f(g_zc, st, "g_zc"); check_f(g_dsum, st, "g_dsum");
    at::Tensor d_pos = at::empty({n, 3}, opt), d_gates = at::empty({n, 184}, opt), d_qn = at::empty({n, 64}, opt), d_kn = at::empty({n, 64}, opt);
    at::Tensor d_vec = later ? at::empty({n, 32, 3}, opt) : at::empty({0}, opt);
    at::Tensor d_ten = has_ten ? at::empty({n, 16, 5}, opt) : at::empty({0}, opt);
    at::Tensor d_rho = need_drho ? at::empty({n}, opt) : at::empty({0}, opt);
    at::Tensor vir = need_vir ? at::empty({n, 9}, opt) : at::empty({0}, opt);
    at::Tensor eg8 = at::empty({n_e, 8}, opt);
    if (st == at::kDouble) {
        auto a = make_args<double>(pos, recv_ptr, send_idx, send_ptr, send_perm, recv_idx, shift_in, cell, graph, z, rmin, rho, poly, gates, qn, kn, vec, ten, flags, r_cut, r_on, max_doff_r2);
        BwdOut<double> o{wptr<double>(d_pos), wptr<double>(d_qn), wptr<double>(eg8), wptr<double>(vir), wptr<double>(d_rho),
                         wptr<double>(d_gates), wptr<double>(d_kn), wptr<double>(d_vec), wptr<double>(d_ten)};
        run_bwd<double>(a, later, g_num, g_zc, g_dsum, o);
    } else {
        TORCH_CHECK(st == at::kFloat, "lam4 cpu_csr: float32 or float64 only");
        auto a = make_args<float>(pos, recv_ptr, send_idx, send_ptr, send_perm, recv_idx, shift_in, cell, graph, z, rmin, rho, poly, gates, qn, kn, vec, ten, flags, r_cut, r_on, max_doff_r2);
        BwdOut<float> o{wptr<float>(d_pos), wptr<float>(d_qn), wptr<float>(eg8), wptr<float>(vir), wptr<float>(d_rho),
                        wptr<float>(d_gates), wptr<float>(d_kn), wptr<float>(d_vec), wptr<float>(d_ten)};
        run_bwd<float>(a, later, g_num, g_zc, g_dsum, o);
    }
    return {d_pos, d_gates, d_vec, d_ten, d_qn, d_kn, d_rho, vir, eg8};
}

int omp_threads_probe() {
    int n = 0;
    #pragma omp parallel num_threads(n_threads())
    {
        #pragma omp single
        n = omp_get_num_threads();
    }
    return n;
}

}  // namespace lam4cpu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("csr_fwd", &lam4cpu::csr_fwd, py::call_guard<py::gil_scoped_release>());
    m.def("csr_bwd", &lam4cpu::csr_bwd, py::call_guard<py::gil_scoped_release>());
    m.def("omp_threads_probe", &lam4cpu::omp_threads_probe);
}
"""
