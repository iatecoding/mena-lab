"""GPU-resident Nose-Hoover chain (NVT) state and integrator for `MDEngine`.

`NoseHooverChainThermostat.integrate_nhc` / `NoseHooverChainNVT.step` of ASE transcribed op for op to float64 torch, so only the
arithmetic moves to the device. The initial state (momenta, masses, Q, kT) is read from an ASE `NoseHooverChainNVT` built exactly as
`matbench_discovery.md.run_nvt_md` builds it (dt 0.25 fs, tdamp 25 fs, Maxwell-Boltzmann velocities from a seeded rng).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

# ASE nose_hoover_chain.FOURTH_ORDER_COEFFS (Suzuki-Yoshida)
SY_COEFFS = (1 / (2 - 2 ** (1 / 3)), -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)), 1 / (2 - 2 ** (1 / 3)))


def compute_nhc_dof(atoms: Any, *, remove_com: bool = True) -> int:
    """Compute active kinetic degrees of freedom for NHC thermostatting.

    Matches `_thermalize_and_zero_com` / `_zero_rotation_safe`:
    - `remove_com=False`, `N==1`, or constrained systems: `3*N`
    - Periodic (`any(pbc)`): `max(1, 3*N - 3)` (3 translational COM DOF removed)
    - Open molecule: `max(1, 3*N - 3 - n_rot)` where `n_rot` is 3 (nonlinear) or 2 (linear)
    """
    n = len(atoms)
    if n <= 0:
        return 0
    if not remove_com or n == 1 or getattr(atoms, "constraints", None):
        return 3 * n
    if bool(np.any(atoms.get_pbc())):
        return max(1, 3 * n - 3)
    ip = np.asarray(atoms.get_moments_of_inertia(), dtype=np.float64)
    n_rot = int(np.sum(ip > 1e-9))
    removed_rot = n_rot if n_rot >= 2 else 0
    return max(1, 3 * n - 3 - removed_rot)


@dataclass
class NHCState:
    q: Tensor        # [N, 3] positions, A (float64)
    p: Tensor        # [N, 3] momenta
    eta: Tensor      # [tchain] thermostat positions
    p_eta: Tensor    # [tchain] thermostat momenta
    masses: Tensor   # [N, 1]
    Q: list          # thermostat masses (python floats)
    Q_t: Tensor
    kT: float
    n_global: int
    dt: float
    tchain: int
    tloop: int
    n_dof: int | None = None

    @property
    def dof(self) -> int:
        return int(self.n_dof) if self.n_dof is not None else 3 * int(self.n_global)

    @classmethod
    def from_ase(cls, dyn, device, *, n_dof: int | None = None) -> "NHCState":
        th = dyn._thermostat
        dev = torch.device(device)
        dt_state = torch.float32 if dev.type == "mps" else torch.float64
        kw = dict(dtype=dt_state, device=dev)
        n_glob_attr = getattr(th, "_num_atoms_global", None)
        n_glob = int(n_glob_attr) if n_glob_attr is not None else len(dyn.atoms)
        ase_dof_attr = getattr(th, "_ndof", None)
        ase_dof = int(ase_dof_attr) if ase_dof_attr is not None else 3 * n_glob
        dof_val = ase_dof if n_dof is None else max(1, int(n_dof))
        q_list = [float(x) for x in th._Q]
        if n_dof is not None and ase_dof > 0 and len(q_list) > 0:
            q_list[0] = q_list[0] * (float(dof_val) / float(ase_dof))
        return cls(q=torch.tensor(dyn._q, **kw), p=torch.tensor(dyn._p, **kw), eta=torch.tensor(th._eta, **kw),
                   p_eta=torch.tensor(th._p_eta, **kw), masses=torch.tensor(dyn.masses, **kw).reshape(-1, 1),
                   Q=q_list, Q_t=torch.tensor(q_list, **kw), kT=float(th._kT),
                   n_global=n_glob, dt=float(dyn.dt), tchain=int(th._tchain), tloop=int(th._tloop), n_dof=dof_val)

    @classmethod
    def for_nve(cls, atoms, timestep_fs: float, device, *, n_dof: int | None = None) -> "NHCState":
        """Construct a Velocity-Verlet `NHCState` on `device` from `atoms` without thermostat overhead."""
        from ase import units

        dev = torch.device(device)
        dt_state = torch.float32 if dev.type == "mps" else torch.float64
        kw = dict(dtype=dt_state, device=dev)
        tchain = 3
        dof_val = 3 * len(atoms) if n_dof is None else max(1, int(n_dof))
        return cls(
            q=torch.tensor(atoms.get_positions(), **kw), p=torch.tensor(atoms.get_momenta(), **kw),
            eta=torch.zeros(tchain, **kw), p_eta=torch.zeros(tchain, **kw),
            masses=torch.tensor(atoms.get_masses(), **kw).reshape(-1, 1),
            Q=[1.0] * tchain, Q_t=torch.ones(tchain, **kw), kT=0.0,
            n_global=len(atoms), dt=float(timestep_fs * units.fs), tchain=tchain, tloop=1, n_dof=dof_val,
        )

    def set_dof(self, n_dof: int) -> None:
        """Update active NHC kinetic degrees of freedom and scale chain mass `Q[0] = n_dof * q1`."""
        old_dof = max(1, self.dof)
        new_dof = max(1, int(n_dof))
        self.n_dof = new_dof
        if len(self.Q) > 0 and self.kT > 0.0:
            q1 = float(self.Q[1]) if len(self.Q) > 1 else float(self.Q[0]) / float(old_dof)
            self.Q[0] = float(new_dof) * q1
            self.Q_t.copy_(torch.tensor(self.Q, dtype=self.Q_t.dtype, device=self.Q_t.device))

    def update_params(self, *, timestep_fs: float | None = None, temperature_K: float | None = None, tdamp_fs: float | None = None) -> None:
        """Update integration `dt` and/or NHC thermostat `kT` and chain masses `Q` in place."""
        from ase import units

        if timestep_fs is not None:
            self.dt = float(timestep_fs * units.fs)
        if temperature_K is not None and tdamp_fs is not None:
            kT = float(units.kB * float(temperature_K))
            td = float(tdamp_fs * units.fs)
            q1 = kT * td * td
            self.kT = kT
            self.Q = [float(self.dof) * q1] + [q1] * (self.tchain - 1)
            self.Q_t.copy_(torch.tensor(self.Q, dtype=self.Q_t.dtype, device=self.Q_t.device))


def integrate_nhc(st: NHCState, p: Tensor, p_eta: list, eta: Tensor, delta: float):
    """`integrate_nhc`: p_eta is a python list of 0-d tensors (one per chain link); returns (p, p_eta, eta)."""
    M, Q, kT, n3 = st.tchain, st.Q, st.kT, st.dof

    def p_eta_j(p, j, d2, d4):
        if j < M - 1:
            p_eta[j] = p_eta[j] * torch.exp(-d4 * p_eta[j + 1] / Q[j + 1])
        g = (p * p / st.masses).sum() - n3 * kT if j == 0 else p_eta[j - 1] ** 2 / Q[j - 1] - kT
        p_eta[j] = p_eta[j] + d2 * g
        if j < M - 1:
            p_eta[j] = p_eta[j] * torch.exp(-d4 * p_eta[j + 1] / Q[j + 1])

    for _ in range(st.tloop):
        for c in SY_COEFFS:
            d = c * delta / st.tloop
            d2, d4 = d / 2, d / 4
            for j in range(M):
                p_eta_j(p, M - j - 1, d2, d4)
            eta = eta + d * torch.stack(p_eta) / st.Q_t
            p = p * torch.exp(-d * p_eta[0] / Q[0])
            for j in range(M):
                p_eta_j(p, j, d2, d4)
    return p, p_eta, eta


def _nhc_chain_scalar(
    ke2: Tensor, p_eta: Tensor, eta: Tensor, Q_t: Tensor, kT: Tensor, delta: Tensor, n3: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure-scalar 3-chain Suzuki-Yoshida NHC half-step returning `(tot_scale, p_eta_new, eta_new)`."""
    q0, q1, q2 = Q_t[0], Q_t[1], Q_t[2]
    pe0, pe1, pe2 = p_eta[0], p_eta[1], p_eta[2]
    eta0, eta1, eta2 = eta[0], eta[1], eta[2]
    tot_scale = ke2.new_ones(())
    for c in SY_COEFFS:
        d = c * delta
        d2, d4 = 0.5 * d, 0.25 * d
        pe2 = pe2 + d2 * (pe1 * pe1 / q1 - kT)
        e2 = torch.exp(-d4 * pe2 / q2)
        pe1 = (pe1 * e2 + d2 * (pe0 * pe0 / q0 - kT)) * e2
        e1 = torch.exp(-d4 * pe1 / q1)
        pe0 = (pe0 * e1 + d2 * (ke2 - n3 * kT)) * e1
        eta0 = eta0 + d * pe0 / q0
        eta1 = eta1 + d * pe1 / q1
        eta2 = eta2 + d * pe2 / q2
        s_step = torch.exp(-d * pe0 / q0)
        tot_scale = tot_scale * s_step
        ke2 = ke2 * (s_step * s_step)
        e1 = torch.exp(-d4 * pe1 / q1)
        pe0 = (pe0 * e1 + d2 * (ke2 - n3 * kT)) * e1
        e2 = torch.exp(-d4 * pe2 / q2)
        pe1 = (pe1 * e2 + d2 * (pe0 * pe0 / q0 - kT)) * e2
        pe2 = pe2 + d2 * (pe1 * pe1 / q1 - kT)
    return tot_scale, torch.stack((pe0, pe1, pe2)), torch.stack((eta0, eta1, eta2))


def _nhc_half_tensor(
    p: Tensor, p_eta: Tensor, eta: Tensor, masses: Tensor, Q_t: Tensor, kT: Tensor, delta: Tensor, n3: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact tensor-only 3-chain / 1-loop ASE Suzuki-Yoshida NHC half-step (single KE reduction + single velocity scale)."""
    ke2 = (p * p / masses).sum()
    tot_scale, p_eta_n, eta_n = _nhc_chain_scalar(ke2, p_eta, eta, Q_t, kT, delta, n3)
    return p * tot_scale, p_eta_n, eta_n


def _nhc_pre_apply(
    q: Tensor, p: Tensor, f: Tensor, p_eta: Tensor, eta: Tensor,
    masses: Tensor, Q_t: Tensor, kT: Tensor, dt: Tensor, n3: Tensor, ke2: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dt2 = 0.5 * dt
    tot_scale, p_eta, eta = _nhc_chain_scalar(ke2, p_eta, eta, Q_t, kT, dt2, n3)
    p = p * tot_scale + dt2 * f
    q = q + dt * p / masses
    return q, p, p_eta, eta


def _nhc_post_apply(
    p: Tensor, p_eta: Tensor, eta: Tensor, Q_t: Tensor, kT: Tensor, dt: Tensor, n3: Tensor, ke2: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    tot_scale, p_eta, eta = _nhc_chain_scalar(ke2, p_eta, eta, Q_t, kT, 0.5 * dt, n3)
    return p * tot_scale, p_eta, eta


def _nhc_pre_step_eager(
    q: Tensor, p: Tensor, f: Tensor, p_eta: Tensor, eta: Tensor,
    masses: Tensor, Q_t: Tensor, kT: Tensor, dt: Tensor, n3: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dt2 = 0.5 * dt
    p, p_eta, eta = _nhc_half_tensor(p, p_eta, eta, masses, Q_t, kT, dt2, n3)
    p = p + dt2 * f
    q = q + dt * p / masses
    return q, p, p_eta, eta


def _nhc_post_step_eager(
    p: Tensor, f: Tensor, p_eta: Tensor, eta: Tensor,
    masses: Tensor, Q_t: Tensor, kT: Tensor, dt: Tensor, n3: Tensor,
    q: Tensor, q_ref: Tensor, disp2: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dt2 = 0.5 * dt
    p = p + dt2 * f
    p, p_eta, eta = _nhc_half_tensor(p, p_eta, eta, masses, Q_t, kT, dt2, n3)
    d = q - q_ref
    disp2 = torch.maximum(disp2, (d * d).sum(1).max())
    return p, p_eta, eta, disp2


_COMPILED_NHC: tuple[Any, Any] | None = None
_NHC_COMPILE_FAILED: bool = False
_NHC_COMPILED_OK: bool = False


def get_fused_nhc_steps(device: torch.device):
    """Return fast `(pre_step, post_step)` on CUDA for `tchain==3, tloop==1`, with eager fallback when compile is unavailable."""
    global _COMPILED_NHC, _NHC_COMPILE_FAILED
    if device.type != "cuda" or _NHC_COMPILE_FAILED:
        return _nhc_pre_step_eager, _nhc_post_step_eager
    if _COMPILED_NHC is None:
        import warnings
        from lam4.core.consts import capturing

        try:
            # Keep reductions outside torch.compile so Inductor never collapses them into a 1-block (xnumel=1) kernel.
            pre_apply = torch.compile(_nhc_pre_apply, dynamic=True, mode="default")
            post_apply = torch.compile(_nhc_post_apply, dynamic=True, mode="default")
        except Exception as exc:  # noqa: BLE001
            _NHC_COMPILE_FAILED = True
            warnings.warn(f"Fused NHC torch.compile unavailable ({exc}); using eager NHC integrator.", RuntimeWarning)
            return _nhc_pre_step_eager, _nhc_post_step_eager

        def _pre_cuda(q, p, f, p_eta, eta, masses, Q_t, kT, dt, n3):
            global _COMPILED_NHC, _NHC_COMPILE_FAILED, _NHC_COMPILED_OK
            if _NHC_COMPILE_FAILED:
                return _nhc_pre_step_eager(q, p, f, p_eta, eta, masses, Q_t, kT, dt, n3)
            ke2 = (p * p / masses).sum()
            if _NHC_COMPILED_OK or capturing():
                return pre_apply(q, p, f, p_eta, eta, masses, Q_t, kT, dt, n3, ke2)
            try:
                res = pre_apply(q, p, f, p_eta, eta, masses, Q_t, kT, dt, n3, ke2)
            except Exception as exc:  # noqa: BLE001
                _NHC_COMPILE_FAILED = True
                _COMPILED_NHC = None
                warnings.warn(f"Fused NHC compiled step failed ({exc}); falling back to eager NHC integrator.", RuntimeWarning)
                return _nhc_pre_step_eager(q, p, f, p_eta, eta, masses, Q_t, kT, dt, n3)
            return res

        def _post_cuda(p, f, p_eta, eta, masses, Q_t, kT, dt, n3, q, q_ref, disp2):
            global _COMPILED_NHC, _NHC_COMPILE_FAILED, _NHC_COMPILED_OK
            if _NHC_COMPILE_FAILED:
                return _nhc_post_step_eager(p, f, p_eta, eta, masses, Q_t, kT, dt, n3, q, q_ref, disp2)
            if _NHC_COMPILED_OK or capturing():
                p_cur = p + (0.5 * dt) * f
                ke2 = (p_cur * p_cur / masses).sum()
                p_out, p_eta_out, eta_out = post_apply(p_cur, p_eta, eta, Q_t, kT, dt, n3, ke2)
                d = q - q_ref
                disp2 = torch.maximum(disp2, (d * d).sum(1).max())
                return p_out, p_eta_out, eta_out, disp2
            try:
                p_cur = p + (0.5 * dt) * f
                ke2 = (p_cur * p_cur / masses).sum()
                p_out, p_eta_out, eta_out = post_apply(p_cur, p_eta, eta, Q_t, kT, dt, n3, ke2)
            except Exception as exc:  # noqa: BLE001
                _NHC_COMPILE_FAILED = True
                _COMPILED_NHC = None
                warnings.warn(f"Fused NHC compiled step failed ({exc}); falling back to eager NHC integrator.", RuntimeWarning)
                return _nhc_post_step_eager(p, f, p_eta, eta, masses, Q_t, kT, dt, n3, q, q_ref, disp2)
            _NHC_COMPILED_OK = True
            d = q - q_ref
            disp2 = torch.maximum(disp2, (d * d).sum(1).max())
            return p_out, p_eta_out, eta_out, disp2

        _COMPILED_NHC = (_pre_cuda, _post_cuda)
    return _COMPILED_NHC
