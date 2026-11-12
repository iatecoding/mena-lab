"""polarization.py: explicit polarisation energy (omol4 N2 + lam5 anisotropic tensor head).

E_i is the field of the final latent charges at atom i (ExactCoulomb.phi_field, e/A^2). Per atom, O(N) once E_i is known:

    E_pol,i = -k_C [ gamma p_i . E_i + p_ind,i . E_i / 2 ],     alpha_i = a softplus(w . RMSNorm(s_i) + b)   (A^3)
    p_ind,i = alpha_i E_i + M(alpha_aniso,i) E_i                 induced dipole (aniso term when aniso=True), added to the predicted dipole

gamma, a and aniso_head start at 0 (the module starts as the zero function). Pad atoms (atom_mask False) give exactly 0 energy,
alpha and induced dipole.

Signature
    Polarization(d_model, k2=16, vec_c=2.0, aniso=False)
        alpha(s [N, d], *, normed=False) -> [N]
        alpha_aniso(s [N, d], ten [N, k2, 5] | None, *, normed=False) -> [N, 5] | None
        forward(s [N, d], p [N, 3] atomic dipoles, field [N, 3], atom_mask [N], *, normed=False, return_alpha=False, ten=None)
            -> (e_atom [N], p_ind [N, 3])   or (e_atom, p_ind, alpha [N]) with return_alpha
        the graph energy is index_add of e_atom over `batch.graph` (done by the caller, as for every per-atom term).

Key map from an earlier-generation state dict: `polarization.{pol_gamma, pol_alpha, alpha_head.weight, alpha_head.bias}`
-> `Polarization.<same>` (drop the `polarization.` prefix).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam4.core.angular import l2_mat6, l2_matvec
from lam4.core.vector_bound import bound_vectors
from lam4.physics.coulomb_exact import COULOMB_K


class Polarization(nn.Module):
    def __init__(self, d_model: int, k2: int = 16, vec_c: float = 2.0, aniso: bool = False) -> None:
        super().__init__()
        self.vec_c, self.aniso = float(vec_c), bool(aniso)
        self.pol_gamma = nn.Parameter(torch.zeros(()))
        self.pol_alpha = nn.Parameter(torch.zeros(()))
        self.alpha_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, math.log(math.e - 1.0))  # softplus = 1: alpha_i = a at the start
        self.aniso_head = nn.Linear(d_model, k2, bias=False) if aniso else None
        if self.aniso_head is not None:
            nn.init.zeros_(self.aniso_head.weight)

    def alpha(self, s: Tensor, *, normed: bool = False) -> Tensor:
        h = s if normed else F.rms_norm(s, (s.shape[-1],), eps=1e-6)
        return self.pol_alpha * F.softplus(self.alpha_head(h).squeeze(-1))

    def alpha_aniso(self, s: Tensor, ten: Tensor | None, *, normed: bool = False) -> Tensor | None:
        if not self.aniso or self.aniso_head is None or ten is None:
            return None
        h = s if normed else F.rms_norm(s, (s.shape[-1],), eps=1e-6)
        tau = bound_vectors(ten, self.vec_c)
        return (self.aniso_head(h).unsqueeze(-1) * tau).sum(-2)

    def forward(self, s: Tensor, p: Tensor, field: Tensor, atom_mask: Tensor, *, normed: bool = False,
                return_alpha: bool = False, ten: Tensor | None = None, return_aniso: bool = False):
        mf = atom_mask.to(s.dtype)
        a = self.alpha(s, normed=normed) * mf
        e = -COULOMB_K * (self.pol_gamma * (p * field).sum(-1) + 0.5 * a * field.pow(2).sum(-1)) * mf
        p_ind = a.unsqueeze(-1) * field
        a_aniso = None
        if self.aniso and ten is not None:
            a_aniso = self.alpha_aniso(s, ten, normed=normed) * mf.unsqueeze(-1)
            p_aniso = l2_matvec(l2_mat6(a_aniso), field)
            p_ind = p_ind + p_aniso
            e = e - 0.5 * COULOMB_K * (p_aniso * field).sum(-1) * mf
        if return_aniso:
            return (e, p_ind, a, a_aniso) if return_alpha else (e, p_ind, a_aniso)
        return (e, p_ind, a) if return_alpha else (e, p_ind)
