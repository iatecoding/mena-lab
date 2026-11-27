"""First-principles batched Thermostatted Ring-Polymer Molecular Dynamics (TRPMD).

Exact normal-mode OBABO integrator with PILE-L internal-mode thermostatting
(Ceriotti et al., JCP 133, 124104, 2010; Rossi, Ceriotti & Manolopoulos,
JCP 140, 234116, 2014). Evaluates all P beads in ONE batched GPU forward pass.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Sequence

import ase.units
from ase import Atoms
import numpy as np
import torch

from lam4.calc.open_mol_gpu import build_open_mol_edges_gpu, eval_batched_open_mol_d3
from lam4.calc.physics import (
    CRYSTAL_EV, MOL_KCAL_TO_EV, _get_d3, assemble_fragment_tensors,
    lot_u_atoms_for_calc, resolve_charge_spin_fragments, resolve_lot_and_u,
)
from lam4.calc.precision_hygiene import center_positions_f64, restore_charged_dipole_f64
from lam4.data.batch import GraphBatch
from lam4.data.buckets import source_ids
from lam4.kernels.csr_plan import build_csr_plan

KB_EV_K: float = float(ase.units.kB)
FS_TO_ASE: float = float(ase.units.fs)
HBAR_EV_FS: float = float(ase.units._hbar / ase.units._e * 1e15)
HBAR_ASE: float = HBAR_EV_FS * FS_TO_ASE
CM1_TO_OMEGA_ASE: float = float(2.0 * math.pi * 2.99792458e-5 / FS_TO_ASE)
THZ_TO_CM1: float = 33.3564095198152

ForceFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray | None]]


def ring_polymer_frequencies(n_beads: int, temperature_K: float) -> tuple[float, np.ndarray]:
    """Return `(omega_P, omega_k)` in ASE angular-frequency units (`1 / t_ASE`)."""
    if n_beads < 1 or temperature_K <= 0.0:
        raise ValueError(f"Invalid n_beads={n_beads} or temperature_K={temperature_K}")
    omega_p = (n_beads * KB_EV_K * temperature_K) / HBAR_ASE
    omega_k = 2.0 * omega_p * np.sin(np.arange(n_beads, dtype=np.float64) * math.pi / n_beads)
    omega_k[0] = 0.0
    return float(omega_p), omega_k


def normal_mode_matrix(n_beads: int) -> np.ndarray:
    """Exact orthogonal real DFT matrix `C[j, k]` (Ceriotti et al. 2010, Eq. 18)."""
    if n_beads < 1:
        raise ValueError(f"n_beads must be >= 1, got {n_beads}")
    c = np.zeros((n_beads, n_beads), dtype=np.float64)
    j = np.arange(n_beads, dtype=np.float64)
    c[:, 0] = math.sqrt(1.0 / n_beads)
    half, scale = n_beads // 2, math.sqrt(2.0 / n_beads)
    for k in range(1, (n_beads + 1) // 2):
        c[:, k] = scale * np.cos(2.0 * math.pi * j * k / n_beads)
    if n_beads % 2 == 0:
        c[:, half] = math.sqrt(1.0 / n_beads) * np.where((np.arange(n_beads) % 2) == 0, 1.0, -1.0)
    for k in range(half + 1, n_beads):
        c[:, k] = scale * np.sin(2.0 * math.pi * j * k / n_beads)
    return c


def to_normal_modes(x: np.ndarray, c_mat: np.ndarray) -> np.ndarray:
    """Transform bead array `[P, ...]` to normal modes `[P, ...]` via `C^T @ x`."""
    return np.tensordot(c_mat.T, x, axes=1)


def from_normal_modes(x_nm: np.ndarray, c_mat: np.ndarray) -> np.ndarray:
    """Transform normal-mode array `[P, ...]` to bead space `[P, ...]` via `C @ x_nm`."""
    return np.tensordot(c_mat, x_nm, axes=1)


def propagate_free_ring_polymer(
    q_nm: np.ndarray, p_nm: np.ndarray, masses: np.ndarray, omega_k: np.ndarray, dt_ase: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact analytic evolution of free ring-polymer normal modes for duration `dt_ase`."""
    m = np.asarray(masses, dtype=np.float64).reshape((1, -1) + (1,) * (q_nm.ndim - 2))
    q_new, p_new = q_nm.copy(), p_nm.copy()
    q_new[0] = q_nm[0] + (dt_ase / m[0]) * p_nm[0]
    if len(omega_k) > 1:
        wk = omega_k[1:].reshape((-1,) + (1,) * (q_nm.ndim - 1))
        cw, sw, mw = np.cos(wk * dt_ase), np.sin(wk * dt_ase), m * wk
        q_int, p_int = q_nm[1:], p_nm[1:]
        q_new[1:] = cw * q_int + (sw / mw) * p_int
        p_new[1:] = -mw * sw * q_int + cw * p_int
    return q_new, p_new


def pile_l_thermostat(
    p_nm: np.ndarray, masses: np.ndarray, omega_k: np.ndarray, temperature_K: float, dt_ase: float,
    rng: np.random.Generator, *, lam: float = 0.5, tau0_fs: float | None = None,
) -> np.ndarray:
    """Exact half-step (`dt_ase / 2`) PILE-L Ornstein-Uhlenbeck thermostat in normal-mode space."""
    n_beads = len(omega_k)
    gamma = np.zeros(n_beads, dtype=np.float64)
    if tau0_fs is not None and tau0_fs > 0.0:
        gamma[0] = 1.0 / (float(tau0_fs) * FS_TO_ASE)
    if n_beads > 1 and lam > 0.0:
        gamma[1:] = 2.0 * float(lam) * omega_k[1:]
    c1 = np.exp(-0.5 * dt_ase * gamma)
    c2 = np.sqrt(np.maximum(0.0, 1.0 - c1 * c1))
    sk = (-1,) + (1,) * (p_nm.ndim - 1)
    sm = (1, -1) + (1,) * (p_nm.ndim - 2)
    sigma = np.sqrt(np.asarray(masses, dtype=np.float64).reshape(sm) * (n_beads * KB_EV_K * temperature_K))
    return c1.reshape(sk) * p_nm + (c2.reshape(sk) * sigma) * rng.standard_normal(p_nm.shape)


def ring_polymer_spring_energy(q: np.ndarray, masses: np.ndarray, omega_p: float) -> float:
    """Exact ring-polymer spring potential energy (eV) across all beads."""
    if q.shape[0] <= 1:
        return 0.0
    dq = q - np.roll(q, -1, axis=0)
    m = np.asarray(masses, dtype=np.float64).reshape((1, -1) + (1,) * (q.ndim - 2))
    return float(0.5 * (omega_p ** 2) * np.sum(m * dq * dq))


def centroid_virial_kinetic_energy(q: np.ndarray, forces: np.ndarray, temperature_K: float) -> float:
    """Herman-Bruskin-Berne centroid-virial quantum kinetic energy estimator (eV)."""
    n_beads, n_atoms = q.shape[0], q.shape[1]
    ndim = int(np.prod(q.shape[2:])) if q.ndim > 2 else 1
    virial = -0.5 * float(np.sum((q - q.mean(axis=0, keepdims=True)) * forces)) / n_beads
    return 0.5 * n_atoms * ndim * KB_EV_K * temperature_K + virial


def compute_ir_spectrum(
    dipoles: np.ndarray, dt_fs: float = 0.5, *, max_lag_fs: float = 1000.0, n_fft: int = 16384, remove_drift: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Wiener-Khinchin dipole-derivative autocorrelation IR spectrum with Hann window.

    When `remove_drift=True` (default), subtracts the mean dipole velocity `<dmu/dt>` so constant
    center-of-mass drift (`Q * V_COM`) on charged ions does not inject a spurious DC spike.
    """
    dip = np.asarray(dipoles, dtype=np.float64)
    if dip.ndim != 2 or dip.shape[1] != 3 or dip.shape[0] < 4:
        raise ValueError(f"Expected dipoles shape [T>=4, 3], got {dip.shape}")
    dmu_dt = np.diff(dip, axis=0) / float(dt_fs)
    if remove_drift:
        dmu_dt = dmu_dt - np.mean(dmu_dt, axis=0, keepdims=True)
    n = len(dmu_dt)
    nfft_ac = 1 << (2 * n - 1).bit_length()
    acf = np.zeros(n, dtype=np.float64)
    norm = np.arange(n, 0, -1, dtype=np.float64)
    for c in range(3):
        fc = np.fft.rfft(dmu_dt[:, c], n=nfft_ac)
        acf += np.fft.irfft(fc * np.conj(fc), n=nfft_ac)[:n] / norm
    max_lag = max(2, min(int(round(max_lag_fs / dt_fs)), n))
    acf_w = acf[:max_lag] * np.hanning(2 * max_lag)[max_lag:]
    spec_n = max(int(n_fft), 2 * max_lag)
    spec = np.maximum(0.0, 2.0 * np.real(np.fft.rfft(acf_w, n=spec_n)) - acf_w[0])
    freqs_cm1 = np.fft.rfftfreq(spec_n, d=dt_fs * 1e-3) * THZ_TO_CM1
    return freqs_cm1, spec


class _BatchedBeadEvaluator:
    """Single-GPU-pass batched evaluator for P ring-polymer beads using MenaCalculator."""

    def __init__(self, calc: Any, template: Atoms, n_beads: int, *, skin: float = 0.6) -> None:
        self.calc, self.template = calc, template.copy()
        self.n_beads, self.n_atoms, self.skin = int(n_beads), len(template), float(skin)
        self.pbc_any = bool(np.any(template.get_pbc()))
        self.use_fast_gpu = (not self.pbc_any) and not (
            getattr(calc, "low_memory", False) and calc.model.trunk.edge_mode != "csr"
        )
        if not self.use_fast_gpu:
            return
        dev, dt = calc.device, calc.dtype
        z_np = np.asarray(template.get_atomic_numbers(), dtype=np.int64)
        src, u_val = resolve_lot_and_u(template, calc.lot, calc.u_override)
        q_val, s_val, fid_np, fq_np, fs_np = resolve_charge_spin_fragments(
            template, z_np, calc.charge_override, calc.spin_override,
            pbc_any=False, domain=src.domain, allow_pbc_charge=False,
        )
        tok_id, ref_id = source_ids(src.name)
        xc_v, paw_v, u_at_np = lot_u_atoms_for_calc(src.name, z_np, template.get_positions(), None, (False, False, False), u_val)
        e0_kcal, eqs_kcal, _ = calc._ref_energies_kcal(
            src.name, src.domain, False, z_np, q_val, s_val, frag_charge=fq_np, frag_spin=fs_np,
        )
        self.kcal_to_ev = (1.0 / CRYSTAL_EV) if src.domain == "crystal" else MOL_KCAL_TO_EV
        self.e_ref_kcal = float(e0_kcal + eqs_kcal)
        self.q_val = float(q_val)
        self.d3_mode = src.d3_addback if calc.add_dispersion else None
        self.r_cut, self.counts = float(calc.model.cfg.r_cut), [self.n_atoms] * self.n_beads
        p_n, n_tot = self.n_beads, self.n_beads * self.n_atoms
        z_one = torch.from_numpy(np.ascontiguousarray(z_np, dtype=np.int64)).to(dev)
        self.z_dev = z_one.repeat(p_n)
        self.z_pad = z_one.unsqueeze(0).expand(p_n, self.n_atoms).contiguous()
        self.m_pad = torch.ones((p_n, self.n_atoms), dtype=torch.bool, device=dev)
        self.graph_dev = torch.arange(p_n, dtype=torch.long, device=dev).repeat_interleave(self.n_atoms)
        fid_d, fq_d, fs_d = assemble_fragment_tensors([(self.n_atoms, q_val, s_val, fid_np, fq_np, fs_np)] * p_n, dev, dt)
        self.batch = GraphBatch(
            z=self.z_dev, pos=torch.zeros((n_tot, 3), dtype=dt, device=dev), graph=self.graph_dev,
            cell=torch.zeros((p_n, 3, 3), dtype=dt, device=dev),
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=dev),
            edge_shift=torch.zeros((0, 3), dtype=dt, device=dev),
            charge=torch.full((p_n,), float(q_val), dtype=dt, device=dev),
            spin=torch.full((p_n,), float(s_val), dtype=dt, device=dev),
            u_flag=torch.full((p_n,), int(u_val), dtype=torch.long, device=dev),
            token=torch.full((p_n,), int(tok_id), dtype=torch.long, device=dev),
            ref_id=torch.full((p_n,), int(ref_id), dtype=torch.long, device=dev),
            domain=src.domain, atom_mask=torch.ones(n_tot, dtype=torch.bool, device=dev),
            graph_mask=torch.ones(p_n, dtype=torch.bool, device=dev),
            energy=torch.zeros(p_n, dtype=dt, device=dev), forces=torch.zeros((n_tot, 3), dtype=dt, device=dev),
            energy_mask=torch.ones(p_n, dtype=torch.bool, device=dev),
            force_mask=torch.ones(n_tot, dtype=torch.bool, device=dev),
            coulomb_cap=max(int(calc.model.cfg.coulomb_cap), self.n_atoms) if src.domain == "molecule" else None,
            pbc=False, periodic_electro=False, frag_id=fid_d, frag_charge=fq_d, frag_spin=fs_d,
            xc_id=torch.full((p_n,), int(xc_v), dtype=torch.long, device=dev),
            paw_id=torch.full((p_n,), int(paw_v), dtype=torch.long, device=dev),
            u_atom=torch.from_numpy(np.ascontiguousarray(u_at_np, dtype=np.int64)).to(dev).repeat(p_n),
        )
        self._q_ref: np.ndarray | None = None
        self._csr_plan: Any = None
        self._is_csr = calc.model.trunk.edge_mode == "csr"

    def __call__(self, q_beads: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.use_fast_gpu:
            frames = [self.template.copy() for _ in range(self.n_beads)]
            for fr, qb in zip(frames, q_beads):
                fr.set_positions(qb)
            res = self.calc.eval_batch(frames, properties=("energy", "forces", "dipole"), max_atoms=10**6, max_batch=10**6)
            return (
                np.array([r["energy"] for r in res], dtype=np.float64),
                np.stack([r["forces"] for r in res], axis=0).astype(np.float64),
                np.stack([r["dipole"] for r in res], axis=0).astype(np.float64),
            )
        dev, dt = self.calc.device, self.calc.dtype
        np_dt = np.float32 if dt == torch.float32 else np.float64
        q_centered = np.empty((self.n_beads, self.n_atoms, 3), dtype=np.float64)
        origins = np.empty((self.n_beads, 3), dtype=np.float64)
        for j in range(self.n_beads):
            q_centered[j], origins[j] = center_positions_f64(q_beads[j])
        pos_dev = torch.from_numpy(np.ascontiguousarray(q_centered.reshape(-1, 3), dtype=np_dt)).to(dev)
        if self._q_ref is None or float(np.max(np.linalg.norm(q_centered - self._q_ref, axis=-1))) > 0.5 * self.skin:
            ei, es = build_open_mol_edges_gpu(pos_dev, self.graph_dev, self.counts, self.r_cut + self.skin, dt)
            self.batch.edge_index, self.batch.edge_shift = ei, es
            self._csr_plan = build_csr_plan(ei[1], ei[0], self.n_beads * self.n_atoms) if self._is_csr else None
            self._q_ref = q_centered.copy()
        self.batch.pos = pos_dev
        out = self.calc._run_model(self.batch, None, need_forces=True, need_stress=False, csr_plan=self._csr_plan)
        e_dev, f_dev = out["energy"].detach(), out["forces"].detach()
        has_dipole = out.get("dipole") is not None
        d_dev = out["dipole"].detach() if has_dipole else torch.zeros((self.n_beads, 3), dtype=dt, device=dev)
        if self.d3_mode is not None:
            ed3, fd3 = eval_batched_open_mol_d3(self.z_dev, pos_dev, self.graph_dev, self.counts, self.d3_mode, compute_forces=True)
            if ed3 is None and self.d3_mode == "bj_wb97m":
                d3 = _get_d3(dev, dt)
                p_pad = pos_dev.view(self.n_beads, self.n_atoms, 3).detach().requires_grad_(True)
                with torch.enable_grad():
                    ed3 = d3(self.z_pad, p_pad, self.m_pad)
                    fd3 = (-torch.autograd.grad(ed3.sum(), p_pad)[0]).reshape(-1, 3)
            if ed3 is not None:
                e_dev = e_dev + ed3.detach().to(dt)
            if fd3 is not None:
                f_dev = f_dev + fd3.detach().to(dt)
        hbuf = torch.cat([e_dev.reshape(-1), f_dev.reshape(-1), d_dev.reshape(-1)], dim=0).cpu().to(torch.float64).numpy()
        p_n, n_tot = self.n_beads, self.n_beads * self.n_atoms
        e_ev = (hbuf[:p_n] + self.e_ref_kcal) * self.kcal_to_ev
        f_ev = hbuf[p_n : p_n + 3 * n_tot].reshape(p_n, self.n_atoms, 3) * self.kcal_to_ev
        dip_ea = hbuf[p_n + 3 * n_tot :].reshape(p_n, 3).copy()
        if has_dipole and self.q_val != 0.0:
            for j in range(p_n):
                dip_ea[j] = restore_charged_dipole_f64(dip_ea[j], self.q_val, origins[j], has_dipole=True)
        if getattr(self.calc, "zero_net_force", False) and self.n_atoms > 0:
            f_ev = f_ev - f_ev.mean(axis=1, keepdims=True)
        return e_ev, f_ev, dip_ea


class TRPMDEngine:
    """First-principles batched Thermostatted Ring-Polymer MD engine (OBABO / PILE-L)."""

    def __init__(
        self, atoms_or_pos: Atoms | np.ndarray, *, calc: Any | None = None, force_fn: ForceFn | None = None,
        masses: Sequence[float] | np.ndarray | None = None, n_beads: int = 32, temperature_K: float = 300.0,
        timestep_fs: float = 0.5, lam: float = 0.5, tau0_fs: float | None = None, seed: int = 0,
        remove_com: bool = True, skin: float = 0.6,
    ) -> None:
        if isinstance(atoms_or_pos, Atoms):
            pos0 = np.asarray(atoms_or_pos.get_positions(), dtype=np.float64)
            m0 = np.asarray(atoms_or_pos.get_masses() if masses is None else masses, dtype=np.float64)
            if force_fn is None:
                if calc is None:
                    raise ValueError("Provide either `calc` (MenaCalculator) or `force_fn`.")
                force_fn = _BatchedBeadEvaluator(calc, atoms_or_pos, n_beads, skin=skin)
        else:
            pos0 = np.asarray(atoms_or_pos, dtype=np.float64)
            pos0 = pos0[:, None] if pos0.ndim == 1 else pos0
            if masses is None or force_fn is None:
                raise ValueError("Raw position array requires both `masses` and `force_fn`.")
            m0 = np.asarray(masses, dtype=np.float64)

        self.n_beads, self.n_atoms = int(n_beads), int(pos0.shape[0])
        self.temperature_K, self.timestep_fs = float(temperature_K), float(timestep_fs)
        self.dt_ase = self.timestep_fs * FS_TO_ASE
        self.lam, self.tau0_fs, self.remove_com = float(lam), tau0_fs, bool(remove_com)
        self.masses, self.force_fn = m0.copy(), force_fn
        self.rng = np.random.default_rng(int(seed))
        self.omega_p, self.omega_k = ring_polymer_frequencies(self.n_beads, self.temperature_K)
        self.c_mat = normal_mode_matrix(self.n_beads)
        self.q = np.broadcast_to(pos0[None, ...], (self.n_beads,) + pos0.shape).copy()
        m_shape = (1, -1) + (1,) * (self.q.ndim - 2)
        p_nm0 = self.rng.standard_normal(self.q.shape) * np.sqrt(
            self.masses.reshape(m_shape) * (self.n_beads * KB_EV_K * self.temperature_K)
        )
        if self.remove_com and self.n_atoms > 1:
            p_nm0[0] = p_nm0[0] - p_nm0[0].mean(axis=0, keepdims=True)
        self.p = from_normal_modes(p_nm0, self.c_mat)
        self.energies, self.forces, self.dipoles = self.force_fn(self.q)
        self.step_count = 0

    def step(self, *, tau0_fs: float | None = None, lam: float | None = None) -> None:
        """Advance one OBABO step: O(dt/2) -> B(dt/2) -> A(dt) -> eval -> B(dt/2) -> O(dt/2)."""
        use_tau0 = self.tau0_fs if tau0_fs is None else tau0_fs
        use_lam = self.lam if lam is None else float(lam)
        p_nm = pile_l_thermostat(
            to_normal_modes(self.p, self.c_mat), self.masses, self.omega_k,
            self.temperature_K, self.dt_ase, self.rng, lam=use_lam, tau0_fs=use_tau0,
        )
        if self.remove_com and self.n_atoms > 1 and use_tau0 is not None and use_tau0 > 0.0:
            p_nm[0] = p_nm[0] - p_nm[0].mean(axis=0, keepdims=True)
        p_nm = p_nm + 0.5 * self.dt_ase * to_normal_modes(self.forces, self.c_mat)
        q_nm, p_nm = propagate_free_ring_polymer(
            to_normal_modes(self.q, self.c_mat), p_nm, self.masses, self.omega_k, self.dt_ase,
        )
        self.q = from_normal_modes(q_nm, self.c_mat)
        self.energies, self.forces, self.dipoles = self.force_fn(self.q)
        p_nm = p_nm + 0.5 * self.dt_ase * to_normal_modes(self.forces, self.c_mat)
        p_nm = pile_l_thermostat(
            p_nm, self.masses, self.omega_k, self.temperature_K, self.dt_ase, self.rng, lam=use_lam, tau0_fs=use_tau0,
        )
        if self.remove_com and self.n_atoms > 1 and use_tau0 is not None and use_tau0 > 0.0:
            p_nm[0] = p_nm[0] - p_nm[0].mean(axis=0, keepdims=True)
        self.p = from_normal_modes(p_nm, self.c_mat)
        self.step_count += 1

    def run(
        self, n_steps: int, *, tau0_fs: float | None = None, lam: float | None = None,
        record: bool = True, record_positions: bool = False,
    ) -> dict[str, Any]:
        """Run `n_steps` TRPMD steps and return recorded centroid observables & timing."""
        n_steps = int(n_steps)
        cent_dip = np.empty((n_steps, 3), dtype=np.float64) if (record and self.dipoles is not None) else None
        epot_mean = np.empty(n_steps, dtype=np.float64) if record else None
        ekin_cv = np.empty(n_steps, dtype=np.float64) if record else None
        cent_pos = np.empty((n_steps,) + self.q.shape[1:], dtype=np.float64) if (record and record_positions) else None
        t0 = time.perf_counter()
        for i in range(n_steps):
            self.step(tau0_fs=tau0_fs, lam=lam)
            if record:
                epot_mean[i] = float(np.mean(self.energies))
                ekin_cv[i] = centroid_virial_kinetic_energy(self.q, self.forces, self.temperature_K)
                if cent_dip is not None and self.dipoles is not None:
                    cent_dip[i] = np.mean(self.dipoles, axis=0)
                if cent_pos is not None:
                    cent_pos[i] = np.mean(self.q, axis=0)
        elapsed = time.perf_counter() - t0
        return {
            "n_steps": n_steps, "seconds": elapsed, "ms_per_step": 1000.0 * elapsed / max(1, n_steps),
            "centroid_dipoles": cent_dip, "epot_mean_ev": epot_mean, "ekin_cv_ev": ekin_cv, "centroid_positions": cent_pos,
        }
