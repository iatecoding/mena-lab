"""GPU-resident Nose-Hoover chain (NVT) state and integrator for `MDEngine`.

`NoseHooverChainThermostat.integrate_nhc` / `NoseHooverChainNVT.step` of ASE transcribed op for op to float64 torch, so only the
arithmetic moves to the device. The initial state (momenta, masses, Q, kT) is read from an ASE `NoseHooverChainNVT` built exactly as
`matbench_discovery.md.run_nvt_md` builds it (dt 0.25 fs, tdamp 25 fs, Maxwell-Boltzmann velocities from a seeded rng).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ASE nose_hoover_chain.FOURTH_ORDER_COEFFS (Suzuki-Yoshida)
SY_COEFFS = (1 / (2 - 2 ** (1 / 3)), -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)), 1 / (2 - 2 ** (1 / 3)))


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

    @classmethod
    def from_ase(cls, dyn, device) -> "NHCState":
        th = dyn._thermostat
        dev = torch.device(device)
        dt_state = torch.float32 if dev.type == "mps" else torch.float64
        kw = dict(dtype=dt_state, device=dev)
        return cls(q=torch.tensor(dyn._q, **kw), p=torch.tensor(dyn._p, **kw), eta=torch.tensor(th._eta, **kw),
                   p_eta=torch.tensor(th._p_eta, **kw), masses=torch.tensor(dyn.masses, **kw).reshape(-1, 1),
                   Q=[float(x) for x in th._Q], Q_t=torch.tensor(th._Q, **kw), kT=float(th._kT),
                   n_global=int(th._num_atoms_global), dt=float(dyn.dt), tchain=int(th._tchain), tloop=int(th._tloop))


def integrate_nhc(st: NHCState, p: Tensor, p_eta: list, eta: Tensor, delta: float):
    """`integrate_nhc`: p_eta is a python list of 0-d tensors (one per chain link); returns (p, p_eta, eta)."""
    M, Q, kT, n3 = st.tchain, st.Q, st.kT, 3 * st.n_global

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
