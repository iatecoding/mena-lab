"""MDEngine: GPU-resident NVT molecular dynamics on top of MenaCalculator (`calc.md_engine(atoms, ...)`).

Built from the calculator's own parts: `load_lam4_checkpoint` model (edge_mode auto -> Triton Dual-CSR on CUDA, optional bucketed post/finish
compile), `verlet_neighbor_graph` + `build_or_reuse_batch` (receiver-sorted Verlet list, cached static GraphBatch), the trainer's `pad_batch`
to a fixed edge capacity `e_cap` (pad atoms/edges are a physical no-op) so every tensor of the force call, including the CSRPlan buffers, keeps a
static shape. Positions, momenta and the Nose-Hoover chain stay on the device (float64); per step there is no host work:

  * a window of `window` steps (default 10, one recorded frame) is one CUDA-graph replay of the fixed-shape CSR forward+backward and the
    integrator (`use_graph`), or, if capture is impossible, the same code run eagerly with the same device-resident state;
  * stress is computed on the last step of each window only;
  * the neighbour list is rebuilt on the host only when the largest displacement since the last build passes `rebuild_frac * skin / 2`;
    a window in which any atom moved more than skin / 2 is rolled back and redone with a fresh list every step (exactness, as in the
    reference engine); the one host sync per window is a packed read-back of (max displacement^2, energy, stress);
  * memory is the Tier-2 memory: no dense [E, C] tensors, the plan is int32 `[N+1]`/`[E]` buffers.

Pair energies beyond the ASE path are unchanged: energy = (E_net + E0) / CRYSTAL_EV with the same reference table as `calc.calculate`.
Periodic systems only (crystals, molecular crystals); the cell is fixed (NVT).
"""
from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from lam4.calc.md_nhc import NHCState, integrate_nhc
from lam4.calc.physics import (
    CRYSTAL_EV,
    MOL_KCAL_TO_EV,
    effective_periodic_cell,
    eval_dispersion_single,
    resolve_charge_and_spin,
    resolve_lot_and_u,
)
from lam4.calc.verlet import build_or_reuse_batch, verlet_neighbor_graph
from lam4.data.padding import pad_batch
from lam4.kernels.csr_plan import build_csr_plan


@contextmanager
def _dense_nlh():
    """Inside a graph capture `nlh_atom_energy` must not use its data-dependent `torch.nonzero` candidate filter: take the (identical-value) dense path."""
    import lam4.physics.repulsion as rep
    saved = rep.tracing
    rep.tracing = lambda: True
    try:
        yield
    finally:
        rep.tracing = saved


class MDEngine:
    def __init__(self, calc, atoms, *, temperature_K: float, timestep_fs: float = 0.25, tdamp_fs: float = 25.0, skin: float = 0.6,
                 window: int = 10, rebuild_frac: float = 0.5, use_graph: bool | str = "auto", seed: int = 0, maxwell: bool = True,
                 headroom: float = 1.12) -> None:
        from ase import units
        from ase.md.nose_hoover_chain import NoseHooverChainNVT
        from ase.md import velocitydistribution as _vd

        self.calc, self.model, self.dev = calc, calc.model, calc.device
        self.skin, self.window, self.rebuild_frac, self.headroom = float(skin), int(window), float(rebuild_frac), float(headroom)
        self.r_cut = float(self.model.cfg.r_cut)
        self.atoms = atoms.copy()
        if maxwell:
            if hasattr(_vd, "thermalize_momenta"):
                _vd.thermalize_momenta(self.atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
            else:
                _vd.MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
        dyn = NoseHooverChainNVT(self.atoms, timestep=timestep_fs * units.fs, temperature_K=temperature_K, tdamp=tdamp_fs * units.fs)
        self.st = NHCState.from_ase(dyn, self.dev)
        self.n = len(self.atoms)
        self.z_np = np.asarray(self.atoms.get_atomic_numbers(), dtype=np.int64)
        self.pbc_any = bool(np.any(self.atoms.get_pbc()))
        self.src, self.u = resolve_lot_and_u(self.atoms, calc.lot, calc.u_override)
        self.qv, self.sv = resolve_charge_and_spin(self.atoms, calc.charge_override, calc.spin_override, pbc_any=self.pbc_any)
        self.cell_np, self.pbc = effective_periodic_cell(self.atoms, self.r_cut) if self.pbc_any else (None, (False, False, False))
        self.can_stress = bool(self.pbc_any and all(self.pbc) and self.cell_np is not None and abs(float(np.linalg.det(self.cell_np))) > 1e-8)
        self.kcal_to_ev = (1.0 / CRYSTAL_EV) if self.src.domain == "crystal" else MOL_KCAL_TO_EV
        self.d3_mode = self.src.d3_addback if calc.add_dispersion else None
        self._d3_g0 = torch.zeros(self.n, dtype=torch.long, device=self.dev) if (self.d3_mode == "bj_wb97m" and not self.pbc_any) else None
        e0, eqs, _ = calc._ref_energies_kcal(self.src.name, self.src.domain, self.pbc_any, self.z_np, self.qv, self.sv)
        self.e_ref_kcal = float(e0 + eqs)
        calc.skin = self.skin
        # one standard calculator call first: runs the csr_check / compile_check parity oracles of run_model_checked (may fall back to
        # ref) and keeps the reference E/F(/S) at the initial positions for correctness checks
        a0 = self.atoms.copy()
        a0.set_momenta(np.zeros((self.n, 3)))
        init_props = ("energy", "forces", "stress") if self.can_stress else ("energy", "forces")
        calc.calculate(a0, properties=init_props, system_changes=all_changes)
        self.ref_results = {k: np.array(calc.results[k]) for k in init_props}
        if not self.can_stress:
            self.ref_results["stress"] = np.zeros(6, dtype=np.float64)
        calc.reset()
        self.is_csr = self.model.trunk.edge_mode == "csr"
        self.state_dt = torch.float32 if self.dev.type == "mps" else torch.float64
        fkw = dict(dtype=self.state_dt, device=self.dev)
        self.q_ref = self.st.q.clone()
        self.disp2 = torch.zeros((), **fkw)
        self.f = torch.zeros_like(self.st.q)
        self.e = torch.zeros((), **fkw)
        self.s = torch.zeros(3, 3, **fkw)
        self.pb = self.plan = None
        self.e_cap = 0
        self.graph = None
        self.graph_error: str | None = None
        self.want_graph = (
            (self.dev.type == "cuda" and (self.d3_mode is None))
            if use_graph == "auto"
            else bool(use_graph)
        )
        self.stats = dict(rebuilds=0, rollbacks=0, regrows=0, windows=0, steps=0)
        self.frames: list[dict[str, Any]] = []
        with torch.no_grad():
            self._rebuild(initial=True)
            self._forces_now()

    # ---- neighbour list / static padded batch -----------------------------------------------------------------------
    def _rebuild(self, initial: bool = False, q_np: np.ndarray | None = None) -> None:
        q_np = self.st.q.detach().cpu().numpy() if q_np is None else q_np
        self.calc._nl_cache = None
        ei, es, _ = verlet_neighbor_graph(self.calc, self.z_np, q_np, self.cell_np, self.pbc)
        if not self.is_csr and self.calc._nl_cache is not None:   # non-CSR modes return only the d0 <= r_cut edges: take the whole r_cut + skin list
            ei, es = self.calc._nl_cache["ei"], self.calc._nl_cache["es"]
        b, _, _ = build_or_reuse_batch(self.calc, self.z_np, q_np, self.cell_np, self.pbc, self.src, self.u, self.qv, self.sv, ei, es)
        n_e = int(ei.shape[1])
        self.e_cap_prev = self.e_cap
        if initial or n_e > self.e_cap:
            if not initial:
                self.stats["regrows"] += 1
                self.graph = None
            self.e_cap = int(-(-int(n_e * self.headroom) // 256) * 256)
            self.pb = pad_batch(b, self.n + 2, self.e_cap, 2, self.r_cut)
            self.plan = build_csr_plan(self.pb.edge_index[1], self.pb.edge_index[0], self.n + 2) if self.is_csr else None
        else:
            nb = pad_batch(b, self.n + 2, self.e_cap, 2, self.r_cut)
            self.pb.edge_index.copy_(nb.edge_index)
            self.pb.edge_shift.copy_(nb.edge_shift)
            for name in ("z", "u_flag", "token", "ref_id", "charge", "spin"):   # composition / Q / S may change between calls: in-place, graph stays valid
                getattr(self.pb, name).copy_(getattr(nb, name))
            if self.is_csr:
                np_ = build_csr_plan(nb.edge_index[1], nb.edge_index[0], self.n + 2)
                for name in ("recv_ptr", "send_idx", "recv_idx", "send_ptr", "send_perm"):
                    getattr(self.plan, name).copy_(getattr(np_, name))
        self._sync_chunks(fresh=initial or n_e > self.e_cap_prev)
        self.q_ref.copy_(self.st.q)
        self.disp2.zero_()
        self.stats["rebuilds"] += 1

    def retarget(self, atoms) -> None:
        """Same atom count and cell, new composition / lot / charge / spin: refresh host-side keys used by the next `_rebuild`."""
        self.z_np = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        self.src, self.u = resolve_lot_and_u(atoms, self.calc.lot, self.calc.u_override)
        self.qv, self.sv = resolve_charge_and_spin(atoms, self.calc.charge_override, self.calc.spin_override, pbc_any=self.pbc_any)
        self.kcal_to_ev = (1.0 / CRYSTAL_EV) if self.src.domain == "crystal" else MOL_KCAL_TO_EV
        self.d3_mode = self.src.d3_addback if self.calc.add_dispersion else None
        e0, eqs, _ = self.calc._ref_energies_kcal(self.src.name, self.src.domain, self.pbc_any, self.z_np, self.qv, self.sv)
        self.e_ref_kcal = float(e0 + eqs)

    STATIC_CHUNK_MAX_EDGES = 131_072   # single-chunk CUDA backward up to this many padded edges (~2 GB worst case); above: per-list chunks, no graph reuse

    def _sync_chunks(self, fresh: bool) -> None:
        """CUDA streaming VJP: one static chunk (all atoms, e_cap edges) updated in place, so the backward has no host sync and the graph stays valid."""
        if not self.is_csr or self.dev.type != "cuda":
            return
        if self.e_cap > self.STATIC_CHUNK_MAX_EDGES:
            self.plan.chunks = None   # per-list host chunks: a captured graph holds the old chunk tensors, so recapture after every rebuild
            self.graph = None
            return
        recv_l, send_l = self.plan.recv_idx.to(torch.long), self.plan.send_idx.to(torch.long)
        if fresh or not self.plan.chunks:
            from lam4.core.edge_chunks import Chunk
            n_pad = self.n + 2
            self.plan.chunks = [Chunk(0, n_pad, 0, self.e_cap, recv_l, send_l, torch.arange(n_pad, device=self.dev))]
        else:
            self.plan.chunks[0].recv.copy_(recv_l)
            self.plan.chunks[0].send.copy_(send_l)

    # ---- force call (static shapes; grad only inside) ----------------------------------------------------------------
    def _forces(self, need_stress: bool):
        eff_stress = bool(need_stress and self.can_stress)
        self.pb.pos[: self.n].copy_(self._q_cur)
        with torch.enable_grad():
            out = self.model(self.pb, create_graph=False, compute_forces=True, compute_stress=eff_stress, chunks=None,
                             add_ref=False, return_stage_energies=False, csr_plan=self.plan)
        k = self.kcal_to_ev
        sdt = self.state_dt
        e_kcal = out["energy"][0].detach().to(sdt)
        f_kcal = out["forces"][: self.n].detach().to(sdt)
        s_kcal = out["stress"][0].detach().to(sdt) if eff_stress else None
        if self.d3_mode is not None:
            if self.dev.type == "mps" and not self.pbc_any and self.d3_mode == "bj_wb97m" and self.n <= 1024:
                from lam4.kernels.metal_d3 import has_metal_d3, metal_d3_grouped

                if has_metal_d3():
                    ed3, fd3 = metal_d3_grouped(self.pb.z[: self.n], self.pb.pos[: self.n], self._d3_g0, 1, compute_forces=True)
                    e_kcal = e_kcal + ed3[0].to(sdt)
                    if fd3 is not None:
                        f_kcal = f_kcal + fd3.to(sdt)
                else:
                    q_np = self._q_cur.detach().cpu().to(torch.float64).numpy()
                    de, df, _ = eval_dispersion_single(self.z_np, q_np, self.cell_np, self.pbc, self.d3_mode, self.dev, self.calc.dtype, compute_forces=True)
                    e_kcal = e_kcal + de
                    if df is not None:
                        f_kcal = f_kcal + torch.from_numpy(df).to(self.dev, sdt)
            else:
                q_np = self._q_cur.detach().cpu().to(torch.float64).numpy()
                de, df, ds = eval_dispersion_single(
                    self.z_np, q_np, self.cell_np, self.pbc, self.d3_mode, self.dev, self.calc.dtype,
                    compute_forces=True, compute_stress=eff_stress,
                )
                e_kcal = e_kcal + de
                if df is not None:
                    f_kcal = f_kcal + torch.from_numpy(df).to(self.dev, sdt)
                if eff_stress and ds is not None and s_kcal is not None:
                    s_kcal = s_kcal + torch.from_numpy(ds).to(self.dev, sdt)
        f = f_kcal * k
        if need_stress:
            s = (s_kcal * k) if (eff_stress and s_kcal is not None) else torch.zeros_like(self.s)
        else:
            s = None
        return e_kcal, f, s

    def _forces_now(self) -> None:
        self._q_cur = self.st.q
        e, f, s = self._forces(need_stress=True)
        self.f.copy_(f)
        self.e.copy_(e)
        self.s.copy_(s)

    # ---- one window of integrator steps (ASE NoseHooverChainNVT.step) -------------------------------------------------
    def _window_body(self, nsteps: int) -> None:
        st = self.st
        dt, dt2 = st.dt, st.dt / 2
        q, p, eta, f = st.q, st.p, st.eta, self.f
        p_eta = list(st.p_eta.unbind(0))
        e = s = None
        with torch.no_grad():
            for i in range(nsteps):
                p, p_eta, eta = integrate_nhc(st, p, p_eta, eta, dt2)
                p = p + dt2 * f
                q = q + dt * p / st.masses
                self._q_cur = q
                e, f, s = self._forces(need_stress=(i == nsteps - 1))
                p = p + dt2 * f
                p, p_eta, eta = integrate_nhc(st, p, p_eta, eta, dt2)
                self.disp2.copy_(torch.maximum(self.disp2, ((q - self.q_ref) ** 2).sum(1).max()))
            st.q.copy_(q)
            st.p.copy_(p)
            st.eta.copy_(eta)
            st.p_eta.copy_(torch.stack(p_eta))
            self.f.copy_(f)
            self.e.copy_(e)
            self.s.copy_(s)

    def _state_tensors(self):
        return (self.st.q, self.st.p, self.st.eta, self.st.p_eta, self.f, self.e, self.s, self.q_ref, self.disp2)

    def _snapshot(self):
        return [t.clone() for t in self._state_tensors()]

    def _restore(self, snap) -> None:
        for dst, src in zip(self._state_tensors(), snap):
            dst.copy_(src)

    def _capture(self) -> None:
        snap = self._snapshot()
        try:
            self._capture_inner(snap)
        finally:
            self._restore(snap)

    def _capture_inner(self, snap) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with _dense_nlh():
            with torch.cuda.stream(side):
                for _ in range(2):   # warm-up on a side stream (Triton compile, allocator, cuBLAS handles), state restored after each
                    self._window_body(self.window)
                    self._restore(snap)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                self._window_body(self.window)
        self.graph = g

    def _run_window(self) -> None:
        if self.want_graph and self.graph_error is None:
            if self.graph is None:
                try:
                    self._capture()
                except Exception as exc:   # capture impossible: eager window, same device-resident state
                    import traceback
                    self.graph, self.graph_error = None, f"{type(exc).__name__}: {str(exc)[:200]} @ {traceback.format_exc()[-900:]}"
                    torch.cuda.synchronize()
                    return self._run_window()
            self.graph.replay()
        else:
            self._window_body(self.window)

    def window_step(self) -> None:
        """Advance `window` steps; roll back and redo with a rebuild per step if any atom passed skin / 2."""
        snap = self._snapshot()
        self._run_window()
        packed = torch.cat([self.disp2.view(1), self.e.view(1), self.s.reshape(-1)]).cpu().numpy()   # the window's one host sync
        d2 = float(packed[0])
        if d2 > (0.5 * self.skin) ** 2:
            self.stats["rollbacks"] += 1
            self._restore(snap)
            for _ in range(self.window):
                self._rebuild()
                self._window_body(1)
            packed = torch.cat([self.disp2.view(1), self.e.view(1), self.s.reshape(-1)]).cpu().numpy()
            d2 = float(packed[0])
        if d2 > (self.rebuild_frac * 0.5 * self.skin) ** 2:
            self._rebuild()
        self.stats["windows"] += 1
        self.stats["steps"] += self.window
        self._last = packed

    def run(self, n_steps: int, record: bool = False) -> dict[str, Any]:
        """Run `n_steps` (rounded up to whole windows). Returns timing; `record=True` keeps one host frame per window."""
        n_win = -(-int(n_steps) // self.window)
        sync = torch.cuda.synchronize if self.dev.type == "cuda" else (lambda: None)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_win):
            self.window_step()
            if record:
                self.frames.append(dict(step=self.stats["steps"], positions=self.st.q.cpu().numpy(), energy=self.energy_ev(),
                                        forces=self.f.cpu().numpy(), stress=self.s.cpu().numpy()))
        sync()
        dt = time.perf_counter() - t0
        steps = n_win * self.window
        return dict(steps=steps, seconds=dt, ms_per_step=1000.0 * dt / steps, atoms_per_s=self.n * steps / dt)

    # ---- results -------------------------------------------------------------------------------------------------------
    def energy_ev(self) -> float:
        return (float(self.e) + self.e_ref_kcal) * self.kcal_to_ev

    def positions(self) -> np.ndarray:
        return self.st.q.cpu().numpy()

    def forces(self) -> np.ndarray:
        return self.f.cpu().numpy()

    def stress_ev_a3(self) -> np.ndarray:
        return self.s.cpu().numpy()

    def info(self) -> dict[str, Any]:
        return dict(n_atoms=self.n, e_cap=self.e_cap, csr=self.is_csr, graph=self.graph is not None, graph_error=self.graph_error,
                    **self.stats)
