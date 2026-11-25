"""MDEngine: GPU-resident NVT/NVE molecular dynamics on top of MenaCalculator (`calc.md_engine(atoms, ...)`)."""
from __future__ import annotations

from contextlib import contextmanager
import math
import threading
import time
from typing import Any, Sequence
import warnings

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from lam4.calc.d3_nl_cache import add_md_dispersion, has_triton_d3_on, rebuild_md_buffers
from lam4.calc.md_nhc import get_fused_nhc_steps, integrate_nhc
from lam4.calc import md_props as _mdp
from lam4.calc.md_runner import init_md_engine_state, retarget_md_engine, sync_md_atoms
from lam4.calc.physics import (
    CRYSTAL_EV, MOL_KCAL_TO_EV, auto_covalent_fragments, center_positions_f64, effective_periodic_cell,
    is_periodic_mol_electro, resolve_charge_and_spin, resolve_charge_spin_fragments, resolve_lot_and_u, warn_if_periodic_mol_lot,
)
from lam4.calc.runtime import _max_mol_compile_atoms, _max_whole_compile_atoms, get_md_compiled_runner, precision_context
from lam4.calc.verlet import warm_calc_caches
from lam4.physics.coulomb_exact import resolve_coulomb_meta

_NLH_TLS = threading.local()
_NLH_LOCK = threading.Lock()


def _has_triton_csr_bwd(dev: torch.device) -> bool:
    if dev.type != "cuda":
        return False
    from lam4.kernels import triton_csr

    fn = getattr(triton_csr, "has_triton_csr_backward", None)
    if fn is None:
        return False
    try:
        return bool(fn(dev))
    except TypeError:
        return bool(fn())


def _consts_handles_capture() -> bool:
    from lam4.core import consts

    code = getattr(consts.tracing, "__code__", None)
    return code is not None and ("is_current_stream_capturing" in code.co_names or "capturing" in code.co_names)


@contextmanager
def _dense_nlh():
    """No-op when `consts.tracing()` handles CUDA stream capture; thread-local fallback otherwise."""
    if _consts_handles_capture():
        yield
        return
    import lam4.physics.repulsion as rep

    with _NLH_LOCK:
        if not getattr(rep.tracing, "_tls_wrapped", False):
            orig = rep.tracing
            wrapped = lambda: bool(getattr(_NLH_TLS, "active", False)) or orig()
            wrapped._tls_wrapped = True  # type: ignore[attr-defined]
            rep.tracing = wrapped
    prev = getattr(_NLH_TLS, "active", False)
    _NLH_TLS.active = True
    try:
        yield
    finally:
        _NLH_TLS.active = prev


def _voigt_to_3x3(v: np.ndarray) -> np.ndarray:
    return np.array([[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]], dtype=np.float64)


class MDEngine:
    STATIC_CHUNK_MAX_EDGES = 131_072
    AUTO_GRAPH_MAX_ATOMS = 131_072
    AUTO_GRAPH_MAX_EDGES = 8_500_000

    def __init__(
        self, calc, atoms, *, temperature_K: float | None = 300.0, timestep_fs: float = 0.25, tdamp_fs: float = 25.0, skin: float = 0.6,
        window: int = 10, rebuild_frac: float = 0.5, use_graph: bool | str = "auto", use_compile: bool | str = "auto",
        seed: int = 0, maxwell: bool = True, remove_com: bool = True, headroom: float = 1.12, _skip_init_eval: bool = False,
        record_properties: Sequence[str] = (), ensemble: str = "nvt",
    ) -> None:
        if getattr(atoms, "constraints", None):
            raise ValueError("MDEngine does not support ASE constraints; use calc.run_md(...) (ASE fallback).")
        rk = atoms.info.get("restraint_k", None)
        if rk is not None:
            try:
                if np.any(np.asarray(rk, dtype=np.float64) > 0.0):
                    raise ValueError("MDEngine does not support harmonic restraints (restraint_k); use calc.run_md(...) (ASE fallback).")
            except (TypeError, ValueError) as exc:
                if "restraint_k" in str(exc):
                    raise
                raise ValueError("MDEngine received invalid restraint_k; use calc.run_md(...) (ASE fallback).") from exc
        self.calc, self.model, self.dev = calc, calc.model, calc.device
        self.skin, self.window, self.rebuild_frac, self.headroom = float(skin), int(window), float(rebuild_frac), float(headroom)
        self.remove_com = bool(remove_com)
        self.r_cut = float(self.model.cfg.r_cut)
        self.ensemble, self.atoms, self.st, self._temperature_K, self._tdamp_fs = init_md_engine_state(
            atoms, self.dev, ensemble=ensemble, temperature_K=temperature_K, timestep_fs=timestep_fs,
            tdamp_fs=tdamp_fs, maxwell=maxwell, seed=seed, remove_com=self.remove_com,
        )
        self._origin_f64 = center_positions_f64(self.atoms.get_positions())[1]
        self.n = len(self.atoms)
        self.z_np = np.asarray(self.atoms.get_atomic_numbers(), dtype=np.int64)
        self.pbc_any = bool(np.any(self.atoms.get_pbc()))
        self.src, self.u = calc._resolve_lot_cached(self.atoms, self.z_np) if hasattr(calc, "_resolve_lot_cached") else resolve_lot_and_u(self.atoms, calc.lot, calc.u_override)
        if self.src.domain == "molecule" and (not self.pbc_any) and (self.atoms.arrays.get("frag_id") is not None or self.atoms.info.get("frag_id") is not None):
            raise ValueError("MDEngine does not support per-fragment charges/spins (frag_id); use calc.run_md(...) (ASE fallback).")
        pbc_t = tuple(bool(x) for x in self.atoms.get_pbc())
        raw_cell = np.asarray(self.atoms.cell.array, dtype=np.float64)
        self.periodic_electro = is_periodic_mol_electro(self.src, pbc_t, raw_cell)
        self.qv, self.sv, self.fid_np, self.fq_np, self.fs_np = resolve_charge_spin_fragments(
            self.atoms, self.z_np, calc.charge_override, calc.spin_override,
            pbc_any=self.pbc_any, domain=self.src.domain, allow_pbc_charge=self.periodic_electro,
        )
        frag_skip = None
        if self.periodic_electro and self.fid_np is None:
            self.fid_np, self.fq_np, self.fs_np, frag_skip = auto_covalent_fragments(self.atoms, self.qv, mode=getattr(calc, "auto_fragments", "auto"))
        warn_if_periodic_mol_lot(calc, self.src, self.pbc_any, pbc_flags=pbc_t, q_total=self.qv, frag_skip_reason=frag_skip, cell_valid=(abs(float(np.linalg.det(raw_cell))) > 1e-8))
        self.cell_np, self.pbc = effective_periodic_cell(self.atoms, self.r_cut) if self.pbc_any else (None, (False, False, False))
        self.can_stress = bool(self.pbc_any and all(self.pbc) and self.cell_np is not None and abs(float(np.linalg.det(self.cell_np))) > 1e-8)
        self.kcal_to_ev = (1.0 / CRYSTAL_EV) if self.src.domain == "crystal" else MOL_KCAL_TO_EV
        self.d3_mode = self.src.d3_addback if calc.add_dispersion else None
        self._d3_g0 = torch.zeros(self.n, dtype=torch.long, device=self.dev) if not self.pbc_any else None
        e0, eqs, _ = calc._ref_energies_kcal(self.src.name, self.src.domain, self.pbc_any, self.z_np, self.qv, self.sv, frag_charge=self.fq_np, frag_spin=self.fs_np)
        self.e_ref_kcal = float(e0 + eqs)

        need_csr_chk = bool(calc.model.trunk.edge_mode == "csr" and calc.csr_check and not calc._csr_checked)
        if not _skip_init_eval or need_csr_chk:
            saved_lot_cache = getattr(calc, "_lot_cache", None)
            a0 = self.atoms.copy()
            a0.set_momenta(np.zeros((self.n, 3)))
            init_props = ("energy", "forces", "stress") if self.can_stress else ("energy", "forces")
            calc.calculate(a0, properties=init_props, system_changes=all_changes)
            self.ref_results = {k: np.array(calc.results[k], dtype=np.float64) for k in init_props}
            s_voigt = np.asarray(self.ref_results.get("stress", np.zeros(6, dtype=np.float64)), dtype=np.float64).reshape(6)
            self.ref_results["stress"], self.ref_results["stress_3x3"] = s_voigt, _voigt_to_3x3(s_voigt)
            calc.reset()
            if saved_lot_cache is not None:
                calc._lot_cache = saved_lot_cache
        else:
            self.ref_results = {"stress": np.zeros(6, dtype=np.float64), "stress_3x3": np.zeros((3, 3), dtype=np.float64)}

        self._traced_mode: str | None = None
        if isinstance(use_compile, str) and use_compile.lower().startswith("traced"):
            parts = use_compile.split(":", 1)
            self._traced_mode = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "default"
        self.is_csr = (self.model.trunk.edge_mode == "csr") and (self._traced_mode is None)
        self._traced_ff = None
        self._compiled_runner = None if self._traced_mode is not None else get_md_compiled_runner(calc, use_compile)
        self._retargeted = False
        self._prev_d = 0.0
        self._sub_window = max(1, self.window)
        self.state_dt = torch.float32 if self.dev.type == "mps" else torch.float64
        fkw = dict(dtype=self.state_dt, device=self.dev)
        self.q_ref = self.st.q.clone()
        self.disp2 = torch.zeros((), **fkw)
        self.f = torch.zeros_like(self.st.q)
        self.e = torch.zeros((), **fkw)
        self.s = torch.zeros(3, 3, **fkw)
        self.pb = self.plan = None
        self.e_cap = self.d3_e_cap = 0
        self.d3_rp = self.d3_send = self.d3_shift = None
        self.graph = None
        self.graph_error: str | None = None
        self.compile_graph_error: str | None = None
        self._nhc_kT_t = torch.tensor(float(self.st.kT), **fkw)
        self._nhc_dt_t = torch.tensor(float(self.st.dt), **fkw)
        self._nhc_n3_t = torch.tensor(float(3 * self.st.n_global), **fkw)
        self._win_snap: list[torch.Tensor] | None = None
        self._pack_buf = torch.zeros(11, **fkw)
        self._auto_graph = use_graph == "auto"
        self.want_graph = (self.dev.type == "cuda" and (self.d3_mode is None or self.can_graph_d3())) if self._auto_graph else bool(use_graph)
        self.stats = dict(rebuilds=0, rollbacks=0, regrows=0, windows=0, steps=0)
        self.frames: list[dict[str, Any]] = []
        _mdp.init_md_record_properties(self, record_properties)
        with torch.no_grad():
            self._rebuild(initial=True)
            if not _skip_init_eval:
                self._forces_now()

    def _configure_record_properties(self, record_properties: Sequence[str]) -> None: _mdp.configure_md_record_properties(self, record_properties)

    def can_graph_d3(self) -> bool:
        return bool(self.dev.type == "cuda" and self.calc.dtype == torch.float32 and self.d3_mode in ("bj_wb97m", "zero_pbe") and has_triton_d3_on(self.dev))

    def can_metal_d3(self) -> bool:
        if self.dev.type != "mps" or self.calc.dtype != torch.float32 or self.d3_mode not in ("bj_wb97m", "zero_pbe"):
            return False
        from lam4.kernels.metal_d3 import has_metal_d3

        return bool(has_metal_d3())

    def can_capture_graph(self) -> bool:
        if self.dev.type != "cuda":
            return False
        # Monolithic multi-step CUDAGraph capture above ~131k atoms / ~8.5M edges doubles reserved VRAM and segfaults in replay().
        if self._auto_graph and (self.n > self.AUTO_GRAPH_MAX_ATOMS or self.e_cap > self.AUTO_GRAPH_MAX_EDGES):
            return False
        csr_ok = (not self.is_csr) or _has_triton_csr_bwd(self.dev) or (self.e_cap <= self.STATIC_CHUNK_MAX_EDGES)
        return bool(csr_ok and (self.d3_mode is None or self.can_graph_d3()))

    def _warm_coulomb_meta(self) -> None:
        if self.pb is not None and self.pb.domain == "molecule" and not self.pb.pbc:
            resolve_coulomb_meta(self.pb, self.model.cfg.coulomb_cap)

    def _finish_rebuild(self) -> None:
        self.q_ref.copy_(self.st.q)
        self.disp2.zero_()
        self._prev_d = 0.0
        self.stats["rebuilds"] += 1

    def _rebuild(self, initial: bool = False, q_np: np.ndarray | None = None) -> None:
        if self._traced_mode is not None:
            trunk = getattr(self.model, "trunk", None)
            prev = getattr(trunk, "edge_mode", None)
            try:
                if trunk is not None:
                    trunk.edge_mode = "ref"
                rebuild_md_buffers(self, initial=initial, q_np=q_np)
            finally:
                if trunk is not None and prev is not None:
                    trunk.edge_mode = prev
            if self._traced_ff is not None and not self._traced_ff.matches(self.pb):
                self._traced_ff = None
            return
        rebuild_md_buffers(self, initial=initial, q_np=q_np)

    def retarget(
        self, atoms=None, *, temperature_K: float | None = None, timestep_fs: float | None = None, tdamp_fs: float | None = None,
    ) -> None:
        """Same atom count and cell: refresh composition/lot/charge/spin keys and/or timestep/thermostat params."""
        if atoms is not None:
            prev_sig = (self.src.domain, self.d3_mode)
            self.z_np = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
            self.src, self.u = self.calc._resolve_lot_cached(atoms, self.z_np) if hasattr(self.calc, "_resolve_lot_cached") else resolve_lot_and_u(atoms, self.calc.lot, self.calc.u_override)
            self.periodic_electro = is_periodic_mol_electro(self.src, tuple(bool(x) for x in atoms.get_pbc()), np.asarray(atoms.cell.array, dtype=np.float64))
            self.qv, self.sv, self.fid_np, self.fq_np, self.fs_np = resolve_charge_spin_fragments(
                atoms, self.z_np, self.calc.charge_override, self.calc.spin_override,
                pbc_any=self.pbc_any, domain=self.src.domain, allow_pbc_charge=self.periodic_electro,
            )
            if self.periodic_electro and self.fid_np is None:
                self.fid_np, self.fq_np, self.fs_np, _ = auto_covalent_fragments(atoms, self.qv, mode=getattr(self.calc, "auto_fragments", "auto"))
            self.kcal_to_ev = (1.0 / CRYSTAL_EV) if self.src.domain == "crystal" else MOL_KCAL_TO_EV
            self.d3_mode = self.src.d3_addback if self.calc.add_dispersion else None
            if self._d3_g0 is None and not self.pbc_any:
                self._d3_g0 = torch.zeros(self.n, dtype=torch.long, device=self.dev)
            if (self.src.domain, self.d3_mode) != prev_sig or (self.d3_mode is not None and not self.can_graph_d3()):
                self.graph = None
            e0, eqs, _ = self.calc._ref_energies_kcal(self.src.name, self.src.domain, self.pbc_any, self.z_np, self.qv, self.sv, frag_charge=self.fq_np, frag_spin=self.fs_np)
            self.e_ref_kcal, self._retargeted = float(e0 + eqs), True
        if temperature_K is not None or timestep_fs is not None or tdamp_fs is not None:
            retarget_md_engine(self, temperature_K=temperature_K, timestep_fs=timestep_fs, tdamp_fs=tdamp_fs)
            self._nhc_kT_t.fill_(float(self.st.kT))
            self._nhc_dt_t.fill_(float(self.st.dt))
            self._nhc_n3_t.fill_(float(3 * self.st.n_global))
            self._sub_window = max(1, self.window)

    def _sync_chunks(self, fresh: bool) -> None:
        if not self.is_csr or self.dev.type != "cuda":
            return
        if _has_triton_csr_bwd(self.dev):
            self.plan.chunks = None
            return
        if self.e_cap > self.STATIC_CHUNK_MAX_EDGES:
            self.plan.chunks, self.graph = None, None
            return
        recv_l, send_l = self.plan.recv_idx.to(torch.long), self.plan.send_idx.to(torch.long)
        if fresh or not self.plan.chunks:
            from lam4.core.edge_chunks import Chunk

            n_pad = self.n + 2
            self.plan.chunks = [Chunk(0, n_pad, 0, self.e_cap, recv_l, send_l, torch.arange(n_pad, device=self.dev))]
        else:
            self.plan.chunks[0].recv.copy_(recv_l)
            self.plan.chunks[0].send.copy_(send_l)

    def _forces(self, need_stress: bool):
        from lam4.core.consts import capturing

        eff_stress = bool(need_stress and self.can_stress)
        self.pb.pos[: self.n].copy_(self._q_cur)
        k, sdt = self.kcal_to_ev, self.state_dt
        if self._traced_mode is not None and not self._want_props:
            try:
                if self._traced_ff is None or not self._traced_ff.matches(self.pb):
                    from lam4.calc.md_traced import TracedMDForceField

                    self._traced_ff = TracedMDForceField(
                        self.model, self.pb, compute_stress=self.can_stress, compile_mode=self._traced_mode,
                        precision=getattr(self.calc, "precision", "fp32_ieee"), dense_nlh_ctx=_dense_nlh,
                    )
                with precision_context(getattr(self.calc, "precision", "auto"), self.dev, self.model):
                    e_t, f_t, s_t = self._traced_ff(self.pb.pos, self.pb.edge_index, self.pb.edge_shift)
                e_kcal, f_kcal = e_t[0].detach().to(sdt), f_t[: self.n].detach().to(sdt)
                s_kcal = s_t[0].detach().to(sdt) if (eff_stress and s_t is not None) else None
                if self.d3_mode is not None:
                    e_kcal, f_kcal, s_kcal = add_md_dispersion(self, e_kcal, f_kcal, s_kcal, eff_stress)
                f = f_kcal * k
                if self.calc.zero_net_force and self.n > 0:
                    f = f - f.mean(dim=0, keepdim=True)
                s = ((s_kcal * k) if (eff_stress and s_kcal is not None) else torch.zeros_like(self.s)) if need_stress else None
                return e_kcal, f, s
            except Exception as exc:  # noqa: BLE001
                if capturing():
                    raise
                self.compile_graph_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                warnings.warn(f"MDEngine traced force field failed ({exc}); falling back to eager.", RuntimeWarning)
                self._traced_mode = self._traced_ff = None
        use_c = (
            self._compiled_runner is not None and (self.is_csr or self.dev.type == "cpu") and self.pb.n_atoms <= _max_whole_compile_atoms(self.calc, self.dev)
            and not (self.dev.type == "mps" and self.pb.frag_id is not None)
            and not (self.pb.domain == "molecule" and not self.pb.pbc and self.pb.n_atoms > _max_mol_compile_atoms(self.dev))
        )
        with precision_context(getattr(self.calc, "precision", "auto"), self.dev, self.model), torch.enable_grad():
            if use_c:
                try:
                    out = self._compiled_runner(self.pb, self.plan, need_forces=True, need_stress=eff_stress)
                except Exception as exc:  # noqa: BLE001
                    if capturing():
                        raise
                    self.compile_graph_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                    warnings.warn(f"MDEngine compiled forward failed ({exc}); falling back to eager.", RuntimeWarning)
                    self._compiled_runner = None
                    out = self.model(
                        self.pb, create_graph=False, compute_forces=True, compute_stress=eff_stress,
                        chunks=None, add_ref=False, return_stage_energies=False, csr_plan=self.plan,
                        return_phi=False, return_aux_heads=bool(self._want_props),
                    )
            else:
                out = self.model(
                    self.pb, create_graph=False, compute_forces=True, compute_stress=eff_stress,
                    chunks=None, add_ref=False, return_stage_energies=False, csr_plan=self.plan,
                    return_phi=False, return_aux_heads=bool(self._want_props),
                )
        e_kcal, f_kcal = out.get("energy_f64", out["energy"])[0].detach().to(sdt), out["forces"][: self.n].detach().to(sdt)
        s_kcal = out["stress"][0].detach().to(sdt) if eff_stress else None
        if self.d3_mode is not None:
            e_kcal, f_kcal, s_kcal = add_md_dispersion(self, e_kcal, f_kcal, s_kcal, eff_stress)
        f = f_kcal * k
        if self.calc.zero_net_force and self.n > 0:
            f = f - f.mean(dim=0, keepdim=True)
        s = ((s_kcal * k) if (eff_stress and s_kcal is not None) else torch.zeros_like(self.s)) if need_stress else None
        if self._want_props: _mdp.capture_step_properties(self, out)
        return e_kcal, f, s

    def _forces_now(self) -> None:
        self._q_cur = self.st.q
        e, f, s = self._forces(need_stress=True)
        self.f.copy_(f); self.e.copy_(e); self.s.copy_(s)
        if self._want_props: _mdp.flush_step_properties(self)

    def _window_body(self, nsteps: int, win_offset: int = 0, compute_last_stress: bool = True) -> None:
        st, dt, dt2, is_nvt = self.st, self.st.dt, self.st.dt / 2, self.ensemble == "nvt"
        use_fused_nhc = bool(is_nvt and st.tchain == 3 and st.tloop == 1)
        q, p, eta, f = st.q, st.p, st.eta, self.f
        e = s = None
        with torch.no_grad():
            if use_fused_nhc:
                pre_step, post_step = get_fused_nhc_steps(self.dev)
                pe, disp2 = st.p_eta, self.disp2
                for i in range(nsteps):
                    q, p, pe, eta = pre_step(q, p, f, pe, eta, st.masses, st.Q_t, self._nhc_kT_t, self._nhc_dt_t, self._nhc_n3_t)
                    self._q_cur = q
                    e, f, s = self._forces(need_stress=(compute_last_stress and i == nsteps - 1))
                    p, pe, eta, disp2 = post_step(p, f, pe, eta, st.masses, st.Q_t, self._nhc_kT_t, self._nhc_dt_t, self._nhc_n3_t, q, self.q_ref, disp2)
                    if self._want_props: _mdp.record_win_step_properties(self, win_offset + i)
                st.q.copy_(q); st.p.copy_(p); st.eta.copy_(eta); st.p_eta.copy_(pe); self.disp2.copy_(disp2)
            else:
                p_eta = list(st.p_eta.unbind(0)) if is_nvt else []
                for i in range(nsteps):
                    if is_nvt: p, p_eta, eta = integrate_nhc(st, p, p_eta, eta, dt2)
                    p = p + dt2 * f
                    q = q + dt * p / st.masses
                    self._q_cur = q
                    e, f, s = self._forces(need_stress=(compute_last_stress and i == nsteps - 1))
                    p = p + dt2 * f
                    if is_nvt: p, p_eta, eta = integrate_nhc(st, p, p_eta, eta, dt2)
                    self.disp2.copy_(torch.maximum(self.disp2, ((q - self.q_ref) ** 2).sum(1).max()))
                    if self._want_props: _mdp.record_win_step_properties(self, win_offset + i)
                st.q.copy_(q); st.p.copy_(p)
                if is_nvt: st.eta.copy_(eta); st.p_eta.copy_(torch.stack(p_eta))
            self.f.copy_(f); self.e.copy_(e)
            if s is not None:
                self.s.copy_(s)
            self._pack_buf[0].copy_(self.disp2); self._pack_buf[1].copy_(self.e); self._pack_buf[2:].copy_(self.s.reshape(-1))
            if self._want_props: _mdp.flush_step_properties(self)

    def _state_tensors(self):
        return (self.st.q, self.st.p, self.st.eta, self.st.p_eta, self.f, self.e, self.s, self.q_ref, self.disp2, *_mdp.extra_state_tensors(self))

    def _snapshot(self):
        return [t.clone() for t in self._state_tensors()]

    def _snapshot_win(self):
        st = self._state_tensors()
        if self._win_snap is None or len(self._win_snap) != len(st):
            self._win_snap = [t.clone() for t in st]
        else:
            for dst, src in zip(self._win_snap, st):
                dst.copy_(src)
        return self._win_snap

    def _restore(self, snap) -> None:
        for dst, src in zip(self._state_tensors(), snap):
            dst.copy_(src)

    def _capture_once(self, snap, n_warm: int):
        with torch.cuda.device(self.dev):
            side = torch.cuda.Stream(device=self.dev)
            side.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(side):
                for _ in range(n_warm):
                    self._window_body(min(self.window, 2))
                    self._restore(snap)
            g = torch.cuda.CUDAGraph()
            with _dense_nlh(), torch.cuda.graph(g, stream=side, capture_error_mode="thread_local"):
                self._window_body(self.window)
            torch.cuda.current_stream(self.dev).wait_stream(side)
            return g

    def _capture(self) -> None:
        snap = self._snapshot()
        warm_calc_caches(self.calc, self.dev, self.n, self.n + 2)
        self._warm_coulomb_meta()
        try:
            if self._compiled_runner is not None or self._traced_mode is not None:
                try:
                    self.graph = self._capture_once(snap, 2)
                    return
                except Exception as exc:  # noqa: BLE001
                    self.compile_graph_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                    torch.cuda.synchronize(self.dev)
                    self._restore(snap)
                    self._compiled_runner = None
                    if self._traced_mode is not None:
                        self._traced_mode, self._traced_ff = None, None
            self.graph = self._capture_once(snap, 1)
        finally:
            self._restore(snap)

    def _run_window(self) -> None:
        if self.want_graph and self.graph_error is None and self.can_capture_graph():
            if self.graph is None:
                try:
                    self._capture()
                except Exception as exc:
                    import traceback

                    self.graph, self.graph_error = None, f"{type(exc).__name__}: {str(exc)[:200]} @ {traceback.format_exc()[-900:]}"
                    if self.dev.type == "cuda":
                        torch.cuda.synchronize(self.dev)
                    return self._run_window()
            with torch.cuda.device(self.dev):
                self.graph.replay()
        else:
            self._window_body(self.window)

    def _pack_d2_e_s(self) -> np.ndarray:
        return self._pack_buf.cpu().numpy()

    def _step_chunk(self, nsteps: int, win_offset: int, *, compute_last_stress: bool, use_graph: bool) -> np.ndarray:
        d0, snap = self._prev_d, self._snapshot_win()
        if use_graph:
            self._run_window()
        else:
            self._window_body(nsteps, win_offset=win_offset, compute_last_stress=compute_last_stress)
        packed = self._pack_d2_e_s()
        d2, half_s = float(packed[0]), 0.5 * self.skin
        reb_d2 = (self.rebuild_frac * half_s) ** 2
        if d2 > half_s * half_s and (d0 > 0.0 or nsteps > 1):
            self.stats["rollbacks"] += 1
            self._restore(snap)
            if d0 > 0.0:
                self._rebuild()
                if use_graph:
                    self._run_window()
                else:
                    self._window_body(nsteps, win_offset=win_offset, compute_last_stress=compute_last_stress)
                packed = self._pack_d2_e_s()
                d2 = float(packed[0])
            if d2 > half_s * half_s and nsteps > 1:
                if d0 > 0.0:
                    self._restore(snap)
                    self._rebuild()
                if self.graph is None:
                    d_fail = math.sqrt(max(d2, 1e-12))
                    safe = max(1, min(nsteps - 1, int(0.75 * half_s / max(d_fail / nsteps, 1e-6))))
                    self._sub_window = min(self._sub_window, safe)
                for step_i in range(nsteps):
                    self._window_body(1, win_offset=win_offset + step_i, compute_last_stress=(compute_last_stress and step_i == nsteps - 1))
                    if step_i < nsteps - 1 and float(self.disp2) > reb_d2:
                        self._rebuild()
                packed = self._pack_d2_e_s()
                d2 = float(packed[0])
        d_cur = float(math.sqrt(max(d2, 0.0)))
        pred_k = 1.5 if self._sub_window < self.window else 1.25
        if d2 > reb_d2 or (d_cur + pred_k * max(d_cur - d0, 0.0) > half_s):
            self._rebuild()
        else:
            self._prev_d = d_cur
        return packed

    def window_step(self) -> None:
        if self._retargeted:
            self._rebuild()
            self._forces_now()
        if self._sub_window < self.window and self.graph is None:
            off = 0
            packed = self._pack_d2_e_s()
            while off < self.window:
                c = min(self._sub_window, self.window - off)
                packed = self._step_chunk(c, off, compute_last_stress=(off + c == self.window), use_graph=False)
                off += c
        else:
            packed = self._step_chunk(self.window, 0, compute_last_stress=True, use_graph=True)
        self.stats["windows"] += 1
        self.stats["steps"] += self.window
        self._last = packed

    def run(self, n_steps: int, record: bool = False, record_properties: Sequence[str] | None = None) -> dict[str, Any]:
        if record_properties is not None: self._configure_record_properties(record_properties)
        n_win = -(-int(n_steps) // self.window)
        if self.dev.type == "cuda":
            sync = lambda: torch.cuda.synchronize(self.dev)  # noqa: E731
        elif self.dev.type == "mps":
            sync = torch.mps.synchronize
        else:
            sync = lambda: None  # noqa: E731
        sync()
        t0 = time.perf_counter()
        for _ in range(n_win):
            self.window_step()
            prop_w = _mdp.collect_window_properties(self) if self._want_props else None
            if record: _mdp.append_recorded_frame(self, prop_w)
        sync()
        dt, steps = time.perf_counter() - t0, n_win * self.window
        return dict(steps=steps, seconds=dt, ms_per_step=1000.0 * dt / steps, atoms_per_s=self.n * steps / max(dt, 1e-12))

    def sync_atoms(self, atoms=None):
        """Sync device positions & momenta back to `atoms` (default `self.atoms`) and clear stale ASE calc cache."""
        return sync_md_atoms(self, atoms)

    def energy_ev(self) -> float: return (float(self.e) + self.e_ref_kcal) * self.kcal_to_ev
    def positions(self) -> np.ndarray: return np.asarray(self.st.q.detach().cpu().to(torch.float64).numpy(), dtype=np.float64) + self._origin_f64
    def forces(self) -> np.ndarray: return self.f.cpu().numpy()
    def stress_ev_a3(self) -> np.ndarray: return self.s.cpu().numpy()

    def stress_voigt(self) -> np.ndarray:
        s33 = np.asarray(self.s.cpu().numpy(), dtype=np.float64)
        return s33[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]

    def dipole(self) -> np.ndarray: return _mdp.current_dipole(self)
    def dipole_debye(self) -> np.ndarray: return _mdp.current_dipole_debye(self)
    def polarizability(self) -> np.ndarray: return _mdp.current_pol_tuple(self)[0]
    def polarizability_iso(self) -> float: return float(_mdp.current_pol_tuple(self)[1])
    def polarizability_aniso(self) -> float: return float(_mdp.current_pol_tuple(self)[2])
    def trajectory_properties(self) -> dict[str, np.ndarray]: return {k: np.concatenate(c, axis=0) for k, c in self._prop_history.items() if c}

    def info(self) -> dict[str, Any]:
        return dict(
            n_atoms=self.n, e_cap=self.e_cap, csr=self.is_csr, sub_window=self._sub_window,
            compiled=bool(self._compiled_runner is not None or self._traced_ff is not None),
            traced=self._traced_ff is not None,
            graph=self.graph is not None, graph_error=self.graph_error, compile_graph_error=self.compile_graph_error, **self.stats,
        )
