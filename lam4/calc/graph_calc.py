"""GraphPointCalculator: single-point E/F of fixed-size periodic structures as one CUDA-graph replay (diatomics sweeps, relaxations of one cell).

Built on MDEngine's static padded batch + static Dual-CSR plan: per call the host builds the neighbour list (skin 0) and copies positions, edges,
plan buffers and the composition tensors (z, u_flag, token, ref_id) into the static buffers, then replays ONE graph holding the fixed-shape CSR
forward + backward (5.6k kernel launches become one launch). Same atom count, cell and pbc required; the composition may change between calls
(a diatomics sweep over elements uses one graph). Energy = (E_net + E0) / CRYSTAL_EV exactly as `MenaCalculator.calculate`. No stress.
"""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from lam4.calc.md_engine import MDEngine, _dense_nlh, _has_triton_csr_bwd
from lam4.calc.physics import resolve_charge_and_spin, resolve_lot_and_u
from lam4.calc.precision_hygiene import center_positions_f64
from lam4.calc.verlet import warm_calc_caches


class GraphPointCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, calc, *, compute_stress: bool = True, **kwargs) -> None:
        super().__init__()
        self.calc, self.kwargs = calc, kwargs
        self.compute_stress = bool(compute_stress)
        self.eng: MDEngine | None = None
        self.graph = None
        self.graph_error: str | None = None
        self._key = None
        self._regrows = 0
        self._just_setup = False
        self._captured_sig: tuple[str, str | None] | None = None
        self._fast_dimer_diag: np.ndarray | None = None
        self._dimer_edge_state: tuple[int, tuple[int, int, int]] | None = None
        self._last_info_sig: tuple | None = None
        self._pos_cpu: torch.Tensor | None = None
        self._z_cpu: torch.Tensor | None = None
        self._out_buf: torch.Tensor | None = None
        self._out_cpu: torch.Tensor | None = None
        self._cell_arr: np.ndarray | None = None
        self._pbc_t: tuple[bool, bool, bool] | None = None

    def _setup(self, atoms) -> None:
        self.eng = self.calc.md_engine(
            atoms, temperature_K=300.0, maxwell=False, use_graph=False, skin=0.0, _skip_init_eval=True, **self.kwargs
        )
        n, sdt, dev = len(atoms), self.eng.state_dt, self.eng.dev
        self.e_buf = torch.zeros((), dtype=sdt, device=dev)
        self.f_buf = torch.zeros((n, 3), dtype=sdt, device=dev)
        self.s_buf = torch.zeros((3, 3), dtype=sdt, device=dev)
        want_s = bool(self.compute_stress and self.eng.can_stress)
        self._out_buf = torch.zeros(1 + 3 * n + (9 if want_s else 0), dtype=sdt, device=dev)
        pin = dev.type == "cuda" and torch.cuda.is_available()
        self._pos_cpu = torch.empty((n, 3), dtype=sdt, pin_memory=pin)
        self._z_cpu = torch.empty(n, dtype=torch.long, pin_memory=pin)
        self._out_cpu = torch.empty_like(self._out_buf, device="cpu", pin_memory=pin)
        self.graph, self._regrows, self._captured_sig = None, 0, None
        self._just_setup = True
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        pbc = tuple(bool(x) for x in atoms.get_pbc())
        self._cell_arr = cell.copy()
        self._pbc_t = pbc
        self._key = (n, cell.tobytes(), pbc)
        self._last_info_sig = self._info_sig(atoms)
        self._fast_dimer_diag = None
        self._dimer_edge_state = None
        if n == 2 and all(pbc) and self.eng.is_csr and self.eng.d3_mode is None:
            diag = np.diag(cell)
            if np.allclose(cell, np.diag(diag), atol=1e-9, rtol=0.0) and float(np.min(np.abs(diag))) > 2.0 * self.eng.r_cut + 1e-5:
                self._fast_dimer_diag = np.ascontiguousarray(diag, dtype=np.float64)
                self._d2_ptr = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32, device=dev)
                self._d0_ptr = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32, device=dev)
                self._d2_ei = torch.tensor([[1, 0], [0, 1]], dtype=torch.long, device=dev)
                self._d2_send = torch.tensor([1, 0], dtype=torch.int32, device=dev)
                self._d2_recv = torch.tensor([0, 1], dtype=torch.int32, device=dev)

    @staticmethod
    def _info_sig(atoms) -> tuple:
        info = atoms.info
        return (info.get("lot"), info.get("charge"), info.get("spin"), info.get("u_flag"))

    def _can_graph(self, eng: MDEngine) -> bool:
        if eng.dev.type != "cuda":
            return False
        csr_ok = (not eng.is_csr) or _has_triton_csr_bwd(eng.dev) or (eng.e_cap <= eng.STATIC_CHUNK_MAX_EDGES)
        d3_ok = (eng.d3_mode is None) or eng.can_graph_d3()
        return bool(csr_ok and d3_ok)

    def _body(self) -> None:
        assert self.eng is not None and self._out_buf is not None
        want_s = bool(self.compute_stress and self.eng.can_stress)
        e, f, s = self.eng._forces(want_s)
        self.e_buf.copy_(e)
        self.f_buf.copy_(f)
        self._out_buf[0].copy_(e)
        n3 = 3 * self.eng.n
        self._out_buf[1 : 1 + n3].copy_(f.reshape(-1))
        if want_s and s is not None:
            self.s_buf.copy_(s)
            self._out_buf[1 + n3 : 1 + n3 + 9].copy_(s.reshape(-1))

    def _fast_update_composition(self, eng: MDEngine, atoms) -> bool:
        """In-place update of z and E0 when only atomic numbers change under identical (lot, u=0, q, s, no-D3)."""
        z_np = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        info_sig = self._info_sig(atoms)
        if info_sig == self._last_info_sig and np.array_equal(z_np, eng.z_np):
            return True
        if eng.d3_mode is not None or eng.u != 0:
            return False
        src, u = self.calc._resolve_lot_cached(atoms, z_np) if hasattr(self.calc, "_resolve_lot_cached") else resolve_lot_and_u(atoms, self.calc.lot, self.calc.u_override)
        qv, sv = resolve_charge_and_spin(atoms, self.calc.charge_override, self.calc.spin_override, pbc_any=eng.pbc_any)
        if u != 0 or src.name != eng.src.name or qv != eng.qv or sv != eng.sv:
            return False
        eng.z_np = z_np.copy()
        self._last_info_sig = info_sig
        assert self._z_cpu is not None
        self._z_cpu.copy_(torch.from_numpy(z_np))
        eng.pb.z[: eng.n].copy_(self._z_cpu, non_blocking=(eng.dev.type == "cuda"))
        e0, eqs, _ = self.calc._ref_energies_kcal(src.name, src.domain, eng.pbc_any, z_np, qv, sv)
        eng.e_ref_kcal = float(e0 + eqs)
        return True

    def _fast_update_dimer_edges(self, eng: MDEngine, pos_c: np.ndarray) -> bool:
        """Exact O(1) neighbor/CSR update for 2-atom large orthorhombic boxes (zero vesin/argsort overhead)."""
        diag = self._fast_dimer_diag
        if diag is None or eng.pb is None or eng.plan is None or eng.e_cap < 2:
            return False
        dr = pos_c[1] - pos_c[0]
        s_vec = -np.round(dr / diag).astype(np.int64)
        d_mic = dr + s_vec * diag
        r2 = float(d_mic[0] * d_mic[0] + d_mic[1] * d_mic[1] + d_mic[2] * d_mic[2])
        if r2 < eng.r_cut * eng.r_cut:
            state = (2, (int(s_vec[0]), int(s_vec[1]), int(s_vec[2])))
            if self._dimer_edge_state == state:
                return True
            eng.pb.edge_index[:, :2].copy_(self._d2_ei)
            if state[1] == (0, 0, 0):
                eng.pb.edge_shift[:2].zero_()
            else:
                sh = np.stack([s_vec, -s_vec], axis=0).astype(np.float32 if eng.pb.edge_shift.dtype == torch.float32 else np.float64)
                eng.pb.edge_shift[:2].copy_(torch.from_numpy(sh).to(device=eng.dev))
            if self._dimer_edge_state is None or self._dimer_edge_state[0] != 2:
                eng.plan.recv_ptr.copy_(self._d2_ptr)
                eng.plan.send_ptr.copy_(self._d2_ptr)
                eng.plan.send_idx[:2].copy_(self._d2_send)
                eng.plan.recv_idx[:2].copy_(self._d2_recv)
                eng.plan.send_perm[:2].copy_(self._d2_send)
                eng._sync_chunks(fresh=False)
            self._dimer_edge_state = state
            return True
        state = (0, (0, 0, 0))
        if self._dimer_edge_state == state:
            return True
        eng.pb.edge_index[0, :2].fill_(2)
        eng.pb.edge_index[1, :2].fill_(3)
        eng.pb.edge_shift[:2].zero_()
        eng.plan.recv_ptr.copy_(self._d0_ptr)
        eng.plan.send_ptr.copy_(self._d0_ptr)
        eng.plan.send_idx[:2].fill_(2)
        eng.plan.recv_idx[:2].fill_(3)
        eng.plan.send_perm[:2].zero_()
        eng._sync_chunks(fresh=False)
        self._dimer_edge_state = state
        return True

    def _capture_once(self, eng: MDEngine, n_warm: int):
        with torch.cuda.device(eng.dev):
            side = torch.cuda.Stream(device=eng.dev)
            side.wait_stream(torch.cuda.current_stream(eng.dev))
            with torch.cuda.stream(side):
                for _ in range(n_warm):
                    self._body()
            g = torch.cuda.CUDAGraph()
            with _dense_nlh(), torch.cuda.graph(g, stream=side, capture_error_mode="thread_local"):
                self._body()
            torch.cuda.current_stream(eng.dev).wait_stream(side)
            return g

    def _capture(self) -> None:
        eng = self.eng
        assert eng is not None
        eng._q_cur = eng.st.q
        warm_calc_caches(self.calc, eng.dev, eng.n, eng.n + 2)
        eng._warm_coulomb_meta()
        if eng._compiled_runner is not None:
            try:
                self.graph = self._capture_once(eng, 2)
                self._captured_sig = (eng.src.domain, eng.d3_mode)
                return
            except Exception:  # noqa: BLE001
                torch.cuda.synchronize(eng.dev)
                eng._compiled_runner = None
        self.graph = self._capture_once(eng, 1)
        self._captured_sig = (eng.src.domain, eng.d3_mode)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        at = self.atoms
        assert at is not None
        pbc_now = tuple(bool(x) for x in at.pbc)
        if (
            self.eng is None
            or self._cell_arr is None
            or len(at) != self.eng.n
            or pbc_now != self._pbc_t
            or not np.array_equal(at.cell.array, self._cell_arr)
        ):
            self._setup(at)
        eng = self.eng
        assert eng is not None and self._out_buf is not None and self._pos_cpu is not None
        with torch.no_grad():
            if self._just_setup:
                self._just_setup = False
                if self._fast_dimer_diag is not None:
                    self._fast_update_dimer_edges(eng, center_positions_f64(np.ascontiguousarray(at.get_positions(), dtype=np.float64))[0])
            else:
                pos = np.ascontiguousarray(at.get_positions(), dtype=np.float64)
                pos_c, origin_f64 = center_positions_f64(pos)
                eng._origin_f64 = origin_f64
                self._pos_cpu.copy_(torch.from_numpy(pos_c))
                eng.st.q.copy_(self._pos_cpu, non_blocking=(eng.dev.type == "cuda"))
                if self._fast_update_composition(eng, at) and self._fast_update_dimer_edges(eng, pos_c):
                    pass
                else:
                    self._dimer_edge_state = None
                    eng.retarget(at)
                    self._last_info_sig = self._info_sig(at)
                    eng._rebuild(q_np=pos_c)
            can_graph = self._can_graph(eng)
            cur_sig = (eng.src.domain, eng.d3_mode)
            if eng.stats["regrows"] != self._regrows or not can_graph or cur_sig != self._captured_sig:
                self._regrows = eng.stats["regrows"]
                self.graph = None
                self._captured_sig = None
            if self.graph is None and self.graph_error is None and can_graph:
                try:
                    self._capture()
                except Exception as exc:  # eager fallback, same static buffers
                    self.graph = None
                    self.graph_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                    if eng.dev.type == "cuda":
                        torch.cuda.synchronize(eng.dev)
            if self.graph is not None:
                with torch.cuda.device(eng.dev):
                    self.graph.replay()
            else:
                eng._q_cur = eng.st.q
                self._body()
            if self._out_cpu is not None:
                self._out_cpu.copy_(self._out_buf, non_blocking=False)
                packed = np.asarray(self._out_cpu.numpy(), dtype=np.float64)
            else:
                packed = np.asarray(self._out_buf.cpu().numpy(), dtype=np.float64)
        n = len(at)
        e_ev = (float(packed[0]) + eng.e_ref_kcal) * eng.kcal_to_ev
        self.results = {
            "energy": e_ev,
            "free_energy": e_ev,
            "forces": packed[1 : 1 + 3 * n].reshape(n, 3).copy(),
            "lot_used": eng.src.name,
            "u_used": int(eng.u),
        }
        want_s = bool(self.compute_stress and eng.can_stress)
        if want_s:
            s33 = packed[1 + 3 * n : 1 + 3 * n + 9].reshape(3, 3)
            self.results["stress"] = s33[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].copy()
        elif "stress" in properties:
            self.results["stress"] = np.zeros(6, dtype=np.float64)
