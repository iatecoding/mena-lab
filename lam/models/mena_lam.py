"""mena_lam.py: Unified Mena-LAM Large Atom Model for Periodic Crystals (Stream A) and Molecules (Stream B).

Unifies `mat/mena/periodic.py` (MenaPeriodic) and `models/model_factory.py` (MenaACE + WithD3) around a single
shared `MenaACE` backbone with:
  - Stream A (`forward_crystal` / `forward_sparse(pbc=True)`):
      * `ElementPrior` periodic-table descriptor embedding (`e_Z = W phi(Z) + r_Z`)
      * Per-element, per-U reference energy table `mu` (`[95, 2]` in kcal/mol)
      * `pair_radius` + `pair_radius_warp` + septic C3 taper + smooth degree (`deg + 1.0`)
      * Angular branches (`ttt_mix=2, vttv_rank=8, l3_rank=8`)
      * Direct Edge-Vector Virial (`virial_mode="edge" | "strain"`) avoiding `[E, 3, 3]` strain bmm's
  - Stream B (`forward_molecule` / `forward(pbc=False)` / `forward_sparse(pbc=False)`):
      * `charge_mode="island"`, mid-trunk charge feedback (`fb_input_norm=True`), `IslandContext` at stages `[0, 1, 2]`
      * Full-range damped Coulomb (`coulomb_range="full"`), charge/feedback bounds (`2.0 e`), latent dipole prediction
      * Fixed analytical ωB97M-D3(BJ) dispersion (`D3BJ` / `WithD3`) and `omol_ref` atomic/charge/spin reference
  - Multi-Fidelity Level-of-Theory Conditioning (`lot_id`):
      * `lot_embedding = nn.Embedding(num_lots, d_lot)` (`num_lots=8, d_lot=64`)
      * Zero-init projections into initial atom state, each stage's SwiGLU pre-gate/value, and readout + bias
      * Exact rank-2 weight graft from `spin_embedding` rows 1 (`u=0`, `mp_pbe`) and 2 (`u=1`, `mp_pbeU`)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ase.calculators.calculator import Calculator, all_changes

from lam.data.registry import (
    ID_TO_LOT_NAME,
    LOT_ID_TO_NAME,
    LOT_NAMES,
    LOT_NAME_TO_ID,
    LOT_TO_ID,
    get_lot_entry,
    load_all_lot_e0_tables_f64,
)
from lam.models.dispersion import D3BJ
from lam.models.mena_ace import _CONST_FN, _const, MenaACE, baked_current, latent_dipole
from lam.units import EV, MEV, build_graph, cell_volume


def _element_descriptors_pymatgen(num_elements: int = 119) -> Tensor:
    import warnings
    from ase.data import atomic_masses, covalent_radii
    from pymatgen.core import Element

    warnings.filterwarnings("ignore", message="No Pauling electronegativity", category=UserWarning)
    rows, cont = [], []
    for z in range(num_elements):
        oh = np.zeros(18 + 7 + 4)
        c = [np.nan] * 7
        if 1 <= z <= 103:
            e = Element.from_Z(z)
            g, r = int(e.group), int(e.row)
            oh[g - 1] = 1.0
            oh[18 + min(r, 7) - 1] = 1.0
            oh[25 + "spdf".index(e.block)] = 1.0

            def f(x):
                try:
                    x = float(x)
                    return x if np.isfinite(x) and x != 0 else np.nan
                except Exception:
                    return np.nan

            c = [
                f(e.X),
                float(covalent_radii[z]),
                f(e.atomic_radius),
                f(e.ionization_energy),
                f(e.electron_affinity),
                f(e.mendeleev_no),
                float(np.log(atomic_masses[z])),
            ]
        rows.append(oh)
        cont.append(c)
    C = np.asarray(cont, np.float64)
    miss = np.isnan(C)
    mu, sd = np.nanmean(C[1:104], 0), np.nanstd(C[1:104], 0) + 1e-9
    C = np.where(miss, 0.0, (C - mu) / sd)
    X = np.concatenate([np.asarray(rows), C, miss[:, [0, 2, 3, 4]].astype(np.float64)], 1)
    X[0] = 0.0
    X[104:] = 0.0
    return torch.tensor(X, dtype=torch.float32)


def element_descriptors(num_elements: int = 119) -> Tensor:
    """[num_elements, D] standardised periodic-table descriptors: one-hot group (18), period (7), block (4);
    Pauling electronegativity, covalent radius (ase), atomic radius, first ionisation energy, electron affinity,
    Mendeleev number, log mass; a missing value is set to the column mean (0 after standardising) with a 'missing' flag.
    Row 0 (no element) is all zero."""
    n = int(num_elements)
    if n <= 119:
        npz_path = Path(__file__).resolve().parent / "params" / "element_descriptors_119x40.npz"
        if npz_path.is_file():
            with np.load(npz_path) as data:
                arr = data["phi"]
            return torch.from_numpy(arr[:n]).to(torch.float32).clone()
    return _element_descriptors_pymatgen(n)


class ElementPrior(nn.Module):
    """Element embedding e_Z = W phi(Z) + r_Z: phi = element_descriptors (fixed), W [d, D] shared,
    r_Z a free per-element residual, zero at init."""

    def __init__(self, num_elements: int, d_model: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_elements)
        self.embedding_dim = int(d_model)
        self.register_buffer("phi", element_descriptors(num_elements))
        self.proj = nn.Linear(self.phi.shape[1], d_model, bias=False)
        with torch.no_grad():
            out = self.proj(self.phi[1:90])
            self.proj.weight.mul_(0.02 / out.pow(2).mean().sqrt())
        self.resid = nn.Parameter(torch.zeros(num_elements, d_model))
        self._is_baked: bool = False
        self._baked_weight: Tensor | None = None
        self._baked_versions: tuple[int, int, int] = (-1, -1, -1)

    @torch.no_grad()
    def bake_for_inference(self) -> None:
        self._is_baked = True
        self._baked_weight = (self.proj(self.phi) + self.resid).detach().contiguous()
        self._baked_versions = (self.proj.weight._version, self.phi._version, self.resid._version)

    def unbake_for_inference(self) -> None:
        self._is_baked = False
        self._baked_weight = None
        self._baked_versions = (-1, -1, -1)

    def train(self, mode: bool = True) -> "ElementPrior":
        super().train(mode)
        if mode:
            self.unbake_for_inference()
        return self

    def _apply(self, fn, recurse: bool = True):
        was_baked = getattr(self, "_is_baked", False)
        self._baked_weight = None
        super()._apply(fn, recurse=recurse)
        if was_baked and not self.training:
            self.bake_for_inference()
        return self

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        was_baked = getattr(self, "_is_baked", False)
        self._baked_weight = None
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if was_baked and not self.training:
            self.bake_for_inference()

    @property
    def weight(self) -> Tensor:
        return self.proj(self.phi) + self.resid

    def forward(self, z: Tensor) -> Tensor:
        bw = self._baked_weight
        if not self.training and bw is not None:
            if bw.device != self.resid.device or bw.dtype != self.resid.dtype or not baked_current(
                self._baked_versions, (self.proj.weight._version, self.phi._version, self.resid._version)
            ):
                self.bake_for_inference()
                bw = self._baked_weight
            return bw.index_select(0, z) if z.ndim == 1 else bw[z]
        return self.proj(self.phi[z]) + self.resid[z]


class WithD3(nn.Module):
    """Adds the fixed ωB97M-D3(BJ) energy (and its forces) to a model trained on D3-subtracted labels."""

    def __init__(self, model: nn.Module, r_cut: float = 12.0, taper_on: float = 9.0) -> None:
        super().__init__()
        self.model = model
        self.d3 = D3BJ(r_cut=r_cut, taper_on=taper_on)

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        mask: Tensor,
        total_charge: Tensor | None = None,
        compute_gradient_forces: bool = True,
        compute_forces: bool | None = None,
        **kw,
    ) -> dict:
        if compute_forces is not None:
            compute_gradient_forces = compute_forces
        if compute_gradient_forces and not pos.requires_grad:
            pos = pos.detach().requires_grad_(True)
        out = dict(self.model(z, pos, mask, total_charge=total_charge, compute_gradient_forces=False, **kw))
        e_d3 = self.d3(z, pos, mask)
        e_tot = out["energy"] + e_d3
        out["energy"] = e_tot
        out["dispersion_energy"] = e_d3
        if compute_gradient_forces:
            f_tot = -torch.autograd.grad(e_tot.sum(), pos, create_graph=self.training)[0] * mask.unsqueeze(-1).float()
            out["forces"] = f_tot
        return out

    def forward_sparse(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        n_graphs: int,
        edge_index: Tensor,
        edge_shift: Tensor,
        total_charge: Tensor | None = None,
        lr_edge_index: Tensor | None = None,
        lr_edge_shift: Tensor | None = None,
        compute_gradient_forces: bool = True,
        lr_edge_diff: Tensor | None = None,
        **kw,
    ) -> dict:
        if compute_gradient_forces and not pos.requires_grad and lr_edge_diff is None:
            pos = pos.detach().requires_grad_(True)
        if lr_edge_index is None:
            raise ValueError(
                "WithD3.forward_sparse requires an explicit lr_edge_index (do not silently fall back to short-range edge_index)"
            )
        if lr_edge_shift is None:
            lr_edge_shift = pos.new_zeros((lr_edge_index.shape[1], 3))
        out = dict(
            self.model.forward_sparse(
                z,
                pos,
                batch,
                n_graphs,
                edge_index,
                edge_shift,
                total_charge=total_charge,
                lr_edge_index=lr_edge_index,
                lr_edge_shift=lr_edge_shift,
                compute_gradient_forces=False,
                **kw,
            )
        )
        e_d3 = self.d3.forward_sparse(z, pos, batch, n_graphs, lr_edge_index, lr_edge_shift, edge_diff=lr_edge_diff)
        e_tot = out["energy"] + e_d3
        out["energy"] = e_tot
        out["dispersion_energy"] = e_d3
        if compute_gradient_forces:
            f_tot = -torch.autograd.grad(e_tot.sum(), pos, create_graph=self.training)[0]
            out["forces"] = f_tot
        return out


def mena_lam_kwargs(
    d_model: int = 384,
    msg_dim: int = 128,
    vector_rank: int = 32,
    tensor_rank: int = 16,
    stages: int = 3,
    num_rbf: int = 32,
    r_cut: float = 6.0,
    r_on: float = 5.0,
    swiglu_ratio: float = 2.0,
    ttt_mix: int = 2,
    vttv_rank: int = 8,
    l3_rank: int = 8,
    c3_taper: bool = True,
    pair_radius: bool = True,
    pair_radius_warp: bool = True,
    use_charge_embedding: bool = True,
    use_electrostatics: bool = True,
    use_spin_embedding: bool = True,
    small_global_init: bool = True,
    charge_mode: str = "split",
    coulomb_range: str = "dsf",
    dsf_alpha: float = 0.0,
    pbc_dsf_alpha: float = 0.2,
    r_cut_elec: float = 18.0,
    r_on_elec: float = 16.5,
    charge_bound: float = 2.0,
    feedback_bound: float = 2.0,
    island_context: Sequence[int] | str | None = "all",
    feedback_points: int | None = None,
    ic_dim: int = 32,
    ic_heads: int = 4,
    ic_hidden: int = 64,
    ic_mode: str = "window",
    ic_r_cut: float = 18.0,
    ic_r_on: float = 6.0,
    ic_lr_dim: int = 16,
    elec_lot_ids: Sequence[int] | None = (6,),
    fb_input_norm: bool = True,
    num_lots: int = 8,
    d_lot: int = 64,
    swiglu_ratio_override: float | None = None,
    **extra,
) -> dict:
    if swiglu_ratio_override is not None:  # set by an arch preset; callers pass swiglu_ratio=2.0 explicitly
        swiglu_ratio = swiglu_ratio_override
    # Depth-aware defaults: IC after every stage and charge feedback after all but the last
    # (both reduce to the legacy (0, 1, 2) / 2 at stages=3).
    if isinstance(island_context, str) and island_context == "all":
        island_context = tuple(range(stages))
    if feedback_points is None:
        feedback_points = max(1, stages - 1)
    kw = dict(
        d_model=d_model,
        vector_rank=vector_rank,
        tensor_rank=tensor_rank,
        stages=stages,
        num_rbf=num_rbf,
        r_cut=r_cut,
        r_on=r_on,
        swiglu_ratio=swiglu_ratio,
        msg_dim=msg_dim,
        ttt_mix=ttt_mix,
        vttv_rank=vttv_rank,
        l3_rank=l3_rank,
        c3_taper=c3_taper,
        pair_radius=pair_radius,
        pair_radius_warp=pair_radius_warp,
        use_charge_embedding=use_charge_embedding,
        use_electrostatics=use_electrostatics,
        use_spin_embedding=use_spin_embedding,
        small_global_init=small_global_init,
        charge_mode=charge_mode,
        coulomb_range=coulomb_range,
        dsf_alpha=dsf_alpha,
        pbc_dsf_alpha=pbc_dsf_alpha,
        r_cut_elec=r_cut_elec,
        r_on_elec=r_on_elec,
        charge_bound=charge_bound,
        feedback_bound=feedback_bound,
        feedback_points=feedback_points,
        island_context=list(island_context) if island_context is not None else None,
        ic_dim=ic_dim,
        ic_heads=ic_heads,
        ic_hidden=ic_hidden,
        ic_mode=ic_mode,
        ic_r_cut=ic_r_cut,
        ic_r_on=ic_r_on,
        ic_lr_dim=ic_lr_dim,
        elec_lot_ids=tuple(elec_lot_ids) if elec_lot_ids is not None else None,
        fb_input_norm=fb_input_norm,
        num_lots=num_lots,
        d_lot=d_lot,
    )
    kw.update(extra)
    return kw


def arch_preset_kwargs(arch: str, stages: int = 4) -> dict:
    """MenaACE flags of a named architecture. "lam1" (the released Mena-LAM): the LoT token in every stage, per-stage
    readouts and a token-based physics gate; it needs no extra flags. Other names come from the optional research presets."""
    if arch == "lam1":
        return {}
    if _research is not None and arch in _research.ARCHS:
        return _research.preset_kwargs(arch, stages)
    raise ValueError(f"unknown arch {arch!r} (one of {ARCHS})")


try:
    from lam.models import arch_presets_research as _research
except ImportError:
    _research = None

ARCHS = ("lam1",) + (_research.ARCHS if _research is not None else ())


def _inv_softplus(y: Tensor, eps: float = 1e-12) -> Tensor:
    """Inverse of F.softplus(x, beta=1.0, threshold=20.0) for positive y."""
    y_pos = y.clamp_min(eps)
    return torch.where(y_pos > 20.0, y_pos, torch.log(torch.expm1(y_pos)))


class _SafeCopyParameter(nn.Parameter):
    """nn.Parameter subclass that wraps in-place `.copy_()` in `torch.no_grad()` so checkpoint loaders
    calling `getattr(model, name).copy_(...)` outside `torch.no_grad()` succeed on leaf parameters."""

    def copy_(self, src: Tensor, non_blocking: bool = False) -> Tensor:
        with torch.no_grad():
            return super().copy_(src, non_blocking=non_blocking)


class _EtaViewTensor(Tensor):
    """Read-through view of `F.softplus(omol_raw_eta)` whose `.copy_(src)` writes `_inv_softplus(src)`
    back into the underlying `omol_raw_eta` parameter."""

    _raw_eta: nn.Parameter | None = None

    def copy_(self, src: Tensor, non_blocking: bool = False) -> Tensor:
        raw = getattr(self, "_raw_eta", None)
        if raw is not None:
            with torch.no_grad():
                # float64 on CPU: MPS has no float64, so move to CPU before the cast (a fused .to() casts on MPS)
                raw.copy_(
                    _inv_softplus(src.detach().cpu().to(torch.float64)).to(
                        device=raw.device, dtype=raw.dtype
                    ),
                    non_blocking=non_blocking,
                )
        with torch.no_grad():
            return super().copy_(src, non_blocking=non_blocking)


def _func_grad_with_none_aux(fn, x: Tensor) -> dict[str, Tensor | None]:
    """`torch.func.grad_and_value(fn, has_aux=True)(x)` for `fn(x) -> (scalar, dict)` whose dict holds `None` values
    (torch.func aux must be tensors): returns the dict with the `None` keys restored and the gradient under `"_grad"`."""
    none_keys: list[str] = []

    def fn_tensors(xx: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        e, out = fn(xx)
        none_keys[:] = [k for k, v in out.items() if v is None]
        return e, {k: v for k, v in out.items() if v is not None}

    g, (_, aux) = torch.func.grad_and_value(fn_tensors, has_aux=True)(x)
    return {**aux, **{k: None for k in none_keys}, "_grad": g}


class MenaLAM(nn.Module):
    """Unified Mena-LAM model supporting Stream A (periodic crystals) and Stream B (molecules)."""

    arch_name = "mena_lam"
    _F64_BUFFERS: frozenset[str] = frozenset(
        {"mu", "crystal_e0", "omol_e0", "omol_e_q", "omol_e_s"}
    )

    def __init__(
        self,
        mu: Tensor | None = None,
        elem_prior: bool = True,
        with_d3: bool = True,
        virial_mode: str = "edge",
        crystal_r_cut: float = 6.0,
        crystal_r_on: float = 5.0,
        mol_r_cut: float = 6.0,
        mol_r_on: float = 5.0,
        use_charge_self_energy: bool = True,
        mu_in_ev: bool = True,
        **kw,
    ) -> None:
        super().__init__()
        if virial_mode not in ("edge", "strain"):
            raise ValueError(f"virial_mode must be 'edge' or 'strain', got {virial_mode!r}")
        full_kw = mena_lam_kwargs(**kw)
        self.kw = dict(full_kw, elem_prior=elem_prior, with_d3=with_d3, virial_mode=virial_mode)
        self.core = MenaACE(**full_kw)
        if elem_prior:
            self.core.embedding = ElementPrior(self.core.embedding.num_embeddings, self.core.embedding.embedding_dim)
        if mu is None:
            mu_buf = torch.zeros(101, 2, dtype=torch.float64)
        else:
            scale = EV if mu_in_ev else 1.0
            mu_buf = mu.to(torch.float64) * scale
        self.register_buffer("mu", mu_buf)

        num_lots = int(full_kw.get("num_lots", 8))
        n_e0_rows = max(1, num_lots)
        try:
            e0_t = load_all_lot_e0_tables_f64(num_lots=n_e0_rows, num_elements=119, in_kcal=True)
            crystal_e0 = torch.as_tensor(e0_t, dtype=torch.float64).clone()
            if crystal_e0.shape[0] < n_e0_rows:
                pad = torch.zeros(n_e0_rows - crystal_e0.shape[0], crystal_e0.shape[1], dtype=torch.float64)
                crystal_e0 = torch.cat([crystal_e0, pad], dim=0)
            elif crystal_e0.shape[0] > n_e0_rows:
                crystal_e0 = crystal_e0[:n_e0_rows]
        except Exception:
            crystal_e0 = torch.zeros(n_e0_rows, 119, dtype=torch.float64)
        if mu is not None:
            n_mu = min(mu_buf.shape[0], crystal_e0.shape[1])
            crystal_e0[0, :n_mu] = mu_buf[:n_mu, 0]
            if crystal_e0.shape[0] >= 2:
                crystal_e0[1, :n_mu] = mu_buf[:n_mu, 1]
        elif crystal_e0.shape[0] >= 2:
            n_mu = min(mu_buf.shape[0], crystal_e0.shape[1])
            self.mu[:n_mu, 0] = crystal_e0[0, :n_mu]
            self.mu[:n_mu, 1] = crystal_e0[1, :n_mu]
        self.register_buffer("crystal_e0", crystal_e0)

        self.d3 = D3BJ(r_cut=12.0, taper_on=9.0) if with_d3 else None
        self.register_buffer("omol_e0", torch.zeros(119, dtype=torch.float64))
        self.register_buffer("omol_e_q", torch.zeros(21, dtype=torch.float64))
        self.register_buffer("omol_e_s", torch.zeros(11, dtype=torch.float64))
        # B6: Learnable per-element electronegativity chi_Z and chemical hardness eta_Z = softplus(omol_raw_eta_Z) > 0
        self.omol_chi = _SafeCopyParameter(torch.zeros(119, dtype=torch.float32))
        self.omol_raw_eta = _SafeCopyParameter(torch.full((119,), -20.0, dtype=torch.float32))
        self.omol_q_corr = _SafeCopyParameter(torch.zeros(21, dtype=torch.float32))
        self._charge_self_energy_active: bool = False
        try:
            import json as _json

            ref_path = Path(__file__).resolve().parents[2] / "data" / "e0_reference_omol25.json"
            if ref_path.is_file():
                self.set_omol_reference(_json.loads(ref_path.read_text(encoding="utf-8")))
                self._charge_self_energy_active = True
        except Exception:
            pass
        self.use_charge_self_energy = bool(use_charge_self_energy)
        self.virial_mode = virial_mode
        self.crystal_r_cut = float(crystal_r_cut)
        self.crystal_r_on = float(crystal_r_on)
        self.mol_r_cut = float(mol_r_cut)
        self.mol_r_on = float(mol_r_on)
        self.mol_c3_taper: bool | None = bool(full_kw.get("c3_taper", True))
        self.mol_use_pair_radius: bool | None = None
        self._lot_from_u: bool = True
        self.default_mol_lot_id: int | None = LOT_NAME_TO_ID["omol_wb97mv"]

    @property
    def omol_eta(self) -> Tensor:
        """Per-element chemical hardness eta_Z = softplus(omol_raw_eta_Z) > 0 (strictly positive everywhere)."""
        t = F.softplus(self.omol_raw_eta).as_subclass(_EtaViewTensor)
        t._raw_eta = self.omol_raw_eta
        return t

    def _apply(self, fn, recurse: bool = True):
        """Preserve float64 precision on reference-energy tables across .float()/.to(dtype) and keep them on CPU when
        moving the module to MPS (since MPS does not support float64 tensors)."""
        was_baked = getattr(self, "_is_baked", False)
        saved: dict[str, Tensor] = {}
        for k in self._F64_BUFFERS:
            if k in self._buffers and self._buffers[k] is not None:
                saved[k] = self._buffers[k]
                self._buffers[k] = None
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            probe = next(self.parameters(), None)
            target_dev = probe.device if probe is not None else torch.device("cpu")
            buf_dev = torch.device("cpu") if target_dev.type == "mps" else target_dev
            for k, buf in saved.items():
                self._buffers[k] = buf.to(device=buf_dev, dtype=torch.float64)
            if was_baked and not self.training:
                self.bake_for_inference()
        return self

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        was_baked = getattr(self, "_is_baked", False)
        if was_baked:
            self.unbake_for_inference()
        res = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if was_baked and not self.training:
            self.bake_for_inference()
        return res

    @property
    def backbone(self) -> MenaACE:
        return self.core

    @property
    def r_cut(self) -> float:
        return float(self.core.r_cut)

    @torch.no_grad()
    def bake_for_inference(self) -> "MenaLAM":
        """Pre-compute and cache concatenated stage weights, ElementPrior lookup table, and exp(pair_log_radius)
        for zero-overhead inference. Automatically cleared when `.train(True)` is called."""
        self._is_baked = True
        if hasattr(self.core.embedding, "bake_for_inference"):
            self.core.embedding.bake_for_inference()
        self.core.bake_for_inference()
        self._baked_has_q_self: bool | None = bool(
            self.use_charge_self_energy
            and (
                self._charge_self_energy_active
                or float(self.omol_chi.detach().abs().max().item()) > 0.0
            )
        )
        self._baked_chi_version: int = self.omol_chi._version
        self._baked_use_q_self: bool = bool(self.use_charge_self_energy)
        self._baked_q_self_active: bool = bool(self._charge_self_energy_active)
        return self

    def unbake_for_inference(self) -> "MenaLAM":
        self._is_baked = False
        if hasattr(self.core.embedding, "unbake_for_inference"):
            self.core.embedding.unbake_for_inference()
        self.core.unbake_for_inference()
        self._baked_has_q_self = None
        self._baked_chi_version = -1
        return self

    def train(self, mode: bool = True) -> "MenaLAM":
        super().train(mode)
        if mode:
            self.unbake_for_inference()
        return self

    def _ref_mask(self, B: int, device: torch.device, lot_id, use_elec_context: bool | None = None) -> Tensor:
        """[B] bool: where the OMol-token references apply to an open system (molecular E0 + charge/spin lookup,
        wB97M-D3(BJ), charge self-energy). Legacy gate: same as the physics mask. System gate: the physics runs
        for every open system, but these references follow the token, so a molecule under `omat_pbe` gets PBE E0 and no
        wB97M-specific terms. Single-domain OMol (`num_lots == 0`): every open system uses the OMol references."""
        if self.core.num_lots == 0:
            return torch.ones(B, dtype=torch.bool, device=device)
        if self.core.elec_gate != "system":
            return self.core._elec_mask(B, device, lot_id=lot_id, pbc=False, use_elec_context=use_elec_context)
        return self.core._elec_mask(B, device, lot_id=lot_id, pbc=True)  # under PBC the system gate is the token rule

    def _token_reference(self, mol_ref: Tensor, ref_mask: Tensor, z: Tensor, batch: Tensor, B: int, lot_id) -> Tensor:
        """mol_ref where ref_mask, else the token's per-element E0 (crystal_e0 row), float64 [B]."""
        if bool(ref_mask.all()):
            return mol_ref
        tok = self.crystal_reference(z, batch, B=B, lot_id=lot_id).to(dtype=mol_ref.dtype, device=mol_ref.device)
        return torch.where(ref_mask.to(mol_ref.device), mol_ref, tok)

    def _token_reference_dense(self, mol_ref: Tensor, ref_mask: Tensor, z: Tensor, mask: Tensor, lot_id) -> Tensor:
        """Dense equivalent of `_token_reference` that avoids evaluating `_dense_flat(z, mask)` when `ref_mask` is all True."""
        if bool(ref_mask.all()):
            return mol_ref
        z_flat, b_flat = self._dense_flat(z, mask)
        tok = self.crystal_reference(z_flat, b_flat, B=z.shape[0], lot_id=lot_id).to(dtype=mol_ref.dtype, device=mol_ref.device)
        return torch.where(ref_mask.to(mol_ref.device), mol_ref, tok)

    @staticmethod
    def _dense_flat(z: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        b = torch.arange(z.shape[0], device=z.device).unsqueeze(1).expand_as(z)
        return z[mask], b[mask]

    def crystal_reference(
        self,
        z: Tensor,
        batch: Tensor,
        u: Tensor | int | None = None,
        B: int | None = None,
        *,
        lot_id: Tensor | int | None = None,
    ) -> Tensor:
        """[B] per-structure reference energy in kcal/mol computed in float64 from per-LoT `crystal_e0` (or `mu`)."""
        if B is None:
            B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        z_cpu = z.detach().to(device="cpu", dtype=torch.long)
        b_cpu = batch.detach().to(device="cpu", dtype=torch.long)
        eff = lot_id if lot_id is not None else u
        if eff is None:
            lot_per_struct = torch.zeros(B, dtype=torch.long)
        elif isinstance(eff, int):
            lot_per_struct = torch.full((B,), int(eff), dtype=torch.long)
        else:
            eff_cpu = eff.detach().to(device="cpu", dtype=torch.long)
            if eff_cpu.ndim == 0:
                lot_per_struct = eff_cpu.expand(B)
            elif eff_cpu.shape[0] == z_cpu.shape[0] and B != z_cpu.shape[0]:
                lot_atom = eff_cpu
                lot_per_struct = None
            else:
                lot_per_struct = eff_cpu

        if lot_id is None and u is not None and lot_per_struct is not None and int(lot_per_struct.max().item()) <= 1:
            mu_cpu = self.mu.detach().to(device="cpu", dtype=torch.float64)
            u_atom = lot_per_struct[b_cpu].clamp(0, mu_cpu.shape[1] - 1)
            m = mu_cpu[z_cpu.clamp(0, mu_cpu.shape[0] - 1), u_atom]
        else:
            e0_cpu = self.crystal_e0.detach().to(device="cpu", dtype=torch.float64)
            if lot_per_struct is not None:
                lot_atom = lot_per_struct[b_cpu]
            m = e0_cpu[lot_atom.clamp(0, e0_cpu.shape[0] - 1), z_cpu.clamp(0, e0_cpu.shape[1] - 1)]

        out_f64 = torch.zeros(B, dtype=torch.float64, device="cpu").index_add_(0, b_cpu, m)
        if z.device.type == "mps":
            return out_f64  # keep float64 on CPU; callers on MPS cast to target dtype/device when adding
        return out_f64.to(device=z.device, dtype=torch.float64)

    # Alias for drop-in compatibility with MenaPeriodic.reference
    reference = crystal_reference

    def charge_self_energy(self, z: Tensor, charges: Tensor, mask: Tensor | None = None) -> Tensor:
        """[B] per-atom additive charge self-energy sum_i (chi_{Z_i} q_i + 0.5 * eta_{Z_i} q_i^2) in kcal/mol,
        accumulated in float64 on CPU/CUDA while preserving autograd to omol_chi, omol_raw_eta, and charges."""
        z_idx = z.detach().to(device=self.omol_chi.device, dtype=torch.long).clamp(0, self.omol_chi.shape[0] - 1)
        work_dtype = charges.dtype if charges.device.type == "mps" else torch.float64
        chi = self.omol_chi[z_idx].to(device=charges.device, dtype=work_dtype)
        eta = F.softplus(self.omol_raw_eta)[z_idx].to(device=charges.device, dtype=work_dtype)
        q = charges.to(dtype=work_dtype)
        e_atom = chi * q + 0.5 * eta * q.square()
        if mask is not None:
            m = mask.to(device=charges.device, dtype=work_dtype)
            e_atom = e_atom * m
            q_tot = (q * m).sum(dim=-1)
        else:
            q_tot = q.sum(dim=-1)
        e_sum = e_atom.sum(dim=-1)
        q_int = q_tot.detach().round().to(dtype=torch.long)
        q_idx = (q_int + 10).clamp(0, 20)
        q_corr_mask = ((q_int.abs() >= 2) & (q_int.abs() <= 10)).to(device=charges.device, dtype=work_dtype)
        q_corr = self.omol_q_corr.to(device=charges.device, dtype=work_dtype)[q_idx] * q_corr_mask
        return e_sum + q_corr

    def charge_self_energy_sparse(self, z: Tensor, charges: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
        """[B] sparse per-atom additive charge self-energy sum_{i in g} (chi_{Z_i} q_i + 0.5 * eta_{Z_i} q_i^2) in kcal/mol."""
        z_idx = z.detach().to(device=self.omol_chi.device, dtype=torch.long).clamp(0, self.omol_chi.shape[0] - 1)
        work_dtype = charges.dtype if charges.device.type == "mps" else torch.float64
        chi = self.omol_chi[z_idx].to(device=charges.device, dtype=work_dtype)
        eta = F.softplus(self.omol_raw_eta)[z_idx].to(device=charges.device, dtype=work_dtype)
        q = charges.to(dtype=work_dtype)
        e_atom = chi * q + 0.5 * eta * q.square()
        e_sum = e_atom.new_zeros(n_graphs).index_add_(0, batch, e_atom)
        q_tot = q.new_zeros(n_graphs).index_add_(0, batch, q)
        q_int = q_tot.detach().round().to(dtype=torch.long)
        q_idx = (q_int + 10).clamp(0, 20)
        q_corr_mask = ((q_int.abs() >= 2) & (q_int.abs() <= 10)).to(device=charges.device, dtype=work_dtype)
        q_corr = self.omol_q_corr.to(device=charges.device, dtype=work_dtype)[q_idx] * q_corr_mask
        return e_sum + q_corr

    def molecule_reference(
        self,
        z: Tensor,
        mask: Tensor,
        total_charge: Tensor | None = None,
        spin: Tensor | None = None,
        charges: Tensor | None = None,
        include_charge_lookup: bool = True,
    ) -> Tensor:
        """[B] sum_i omol_e0[Z_i] + E_charge + omol_e_s[S] in kcal/mol, accumulated in float64.
        When `self.use_charge_self_energy` is True and `charges` are provided, replaces the non-additive
        `omol_e_q[Q_total]` lookup with the additive per-atom charge self-energy
        `sum_i (chi_{Z_i} q_i + 0.5 * eta_{Z_i} q_i^2)`."""
        ref_dev = self.omol_e0.device
        z_ref = z.detach().to(device=ref_dev, dtype=torch.long).clamp(0, self.omol_e0.shape[0] - 1)
        m_ref = mask.detach().to(device=ref_dev).to(dtype=torch.float64)
        e = (self.omol_e0[z_ref] * m_ref).sum(dim=-1)
        if spin is not None:
            s = spin.detach().to(device=ref_dev).round().long()
            ok_s = (s >= 1) & (s <= 11)
            e = e + torch.where(ok_s, self.omol_e_s[(s - 1).clamp(0, 10)], torch.zeros_like(e))

        if self.use_charge_self_energy and charges is not None:
            e_q = self.charge_self_energy(z, charges, mask)
            if (
                charges.requires_grad
                or self.omol_chi.requires_grad
                or self.omol_raw_eta.requires_grad
                or self.omol_q_corr.requires_grad
            ):
                return e.to(dtype=e_q.dtype).to(device=charges.device) + e_q
            e = e + e_q.detach().to(device=ref_dev).to(dtype=torch.float64)
        elif include_charge_lookup and total_charge is not None:
            q = total_charge.detach().to(device=ref_dev).round().long()
            ok_q = (q >= -10) & (q <= 10)
            e = e + torch.where(ok_q, self.omol_e_q[(q + 10).clamp(0, 20)], torch.zeros_like(e))

        if z.device.type == "mps":
            return e  # keep float64 on CPU; callers on MPS cast to target dtype/device when adding
        return e.to(device=z.device, dtype=torch.float64)

    def molecule_reference_sparse(
        self,
        z: Tensor,
        batch: Tensor,
        n_graphs: int,
        total_charge: Tensor | None = None,
        spin: Tensor | None = None,
        charges: Tensor | None = None,
        include_charge_lookup: bool = True,
    ) -> Tensor:
        """[B] sparse equivalent of `molecule_reference` in kcal/mol."""
        ref_dev = self.omol_e0.device
        z_ref = z.detach().to(device=ref_dev, dtype=torch.long).clamp(0, self.omol_e0.shape[0] - 1)
        b_ref = batch.detach().to(device=ref_dev, dtype=torch.long)
        e = torch.zeros(n_graphs, dtype=torch.float64, device=ref_dev).index_add_(0, b_ref, self.omol_e0[z_ref])
        if spin is not None:
            s = spin.detach().to(device=ref_dev).round().long()
            if s.ndim == 0:
                s = s.expand(n_graphs)
            ok_s = (s >= 1) & (s <= 11)
            e = e + torch.where(ok_s, self.omol_e_s[(s - 1).clamp(0, 10)], torch.zeros_like(e))

        if self.use_charge_self_energy and charges is not None:
            e_q = self.charge_self_energy_sparse(z, charges, batch, n_graphs)
            if (
                charges.requires_grad
                or self.omol_chi.requires_grad
                or self.omol_raw_eta.requires_grad
                or self.omol_q_corr.requires_grad
            ):
                return e.to(dtype=e_q.dtype).to(device=charges.device) + e_q
            e = e + e_q.detach().to(device=ref_dev).to(dtype=torch.float64)
        elif include_charge_lookup and total_charge is not None:
            q = total_charge.detach().to(device=ref_dev).round().long()
            if q.ndim == 0:
                q = q.expand(n_graphs)
            ok_q = (q >= -10) & (q <= 10)
            e = e + torch.where(ok_q, self.omol_e_q[(q + 10).clamp(0, 20)], torch.zeros_like(e))

        if z.device.type == "mps":
            return e
        return e.to(device=z.device, dtype=torch.float64)

    @torch.no_grad()
    def set_omol_reference(self, ref: dict) -> None:
        """Load omol_ref dict (either {'e0', 'e_q', 'e_s'} arrays or {'e0_kcal', 'charge_kcal', 'spin_kcal'} JSON dict)
        in float64, and initialize per-element charge self-energy parameters (omol_chi, omol_raw_eta, omol_q_corr) from omol_e_q."""
        if "e0_kcal" in ref:
            self.omol_e0.zero_()
            self.omol_e_q.zero_()
            self.omol_e_s.zero_()
            for k, v in ref["e0_kcal"].items():
                self.omol_e0[int(k)] = float(v)
            for k, v in ref["charge_kcal"].items():
                self.omol_e_q[int(k) + 10] = float(v)
            for k, v in ref["spin_kcal"].items():
                self.omol_e_s[int(k) - 1] = float(v)
        else:
            self.omol_e0.copy_(torch.as_tensor(ref["e0"], dtype=torch.float64))
            self.omol_e_q.copy_(torch.as_tensor(ref["e_q"], dtype=torch.float64))
            self.omol_e_s.copy_(torch.as_tensor(ref["e_s"], dtype=torch.float64))

        chi_t = torch.zeros(self.omol_chi.shape[0], dtype=torch.float64)
        eta_t = torch.full((self.omol_raw_eta.shape[0],), 1e-6, dtype=torch.float64)
        q_corr_t = torch.zeros(self.omol_q_corr.shape[0], dtype=torch.float64)
        if "chi" in ref and "eta" in ref:
            chi_t.copy_(torch.as_tensor(ref["chi"], dtype=torch.float64))
            eta_t.copy_(torch.as_tensor(ref["eta"], dtype=torch.float64).clamp_min(1e-6))
        elif "chi_kcal" in ref and "eta_kcal" in ref:
            for k, v in ref["chi_kcal"].items():
                chi_t[int(k)] = float(v)
            for k, v in ref["eta_kcal"].items():
                eta_t[int(k)] = max(float(v), 1e-6)
        else:
            # Initialize from E_Q[+1], E_Q[0], E_Q[-1]:
            # chi_0 * (+1) + 0.5 * eta_0 * (+1)^2 = E_Q[+1] - E_Q[0]
            # chi_0 * (-1) + 0.5 * eta_0 * (-1)^2 = E_Q[-1] - E_Q[0]
            chi_0 = float(0.5 * (self.omol_e_q[11] - self.omol_e_q[9]))
            eta_0 = max(float(self.omol_e_q[11] + self.omol_e_q[9] - 2.0 * self.omol_e_q[10]), 1e-6)
            chi_t[1:] = chi_0
            eta_t[1:] = eta_0
            if float(self.omol_e_q.abs().max().item()) > 0.0:
                for q_val in range(-10, 11):
                    if abs(q_val) >= 2:
                        q_corr_t[q_val + 10] = float(self.omol_e_q[q_val + 10]) - (
                            chi_0 * q_val + 0.5 * eta_0 * (q_val * q_val)
                        )

        if "q_corr" in ref:
            q_corr_t.copy_(torch.as_tensor(ref["q_corr"], dtype=torch.float64))

        self.omol_chi.copy_(chi_t.to(device=self.omol_chi.device, dtype=self.omol_chi.dtype))
        self.omol_raw_eta.copy_(_inv_softplus(eta_t).to(device=self.omol_raw_eta.device, dtype=self.omol_raw_eta.dtype))
        self.omol_q_corr.copy_(q_corr_t.to(device=self.omol_q_corr.device, dtype=self.omol_q_corr.dtype))
        self._charge_self_energy_active = True

    def _has_active_charge_self_energy(self) -> bool:
        if (
            not self.training
            and getattr(self, "_baked_has_q_self", None) is not None
            and getattr(self, "_baked_chi_version", -1) == self.omol_chi._version
            and getattr(self, "_baked_use_q_self", None) == bool(self.use_charge_self_energy)
            and getattr(self, "_baked_q_self_active", None) == bool(self._charge_self_energy_active)
        ):
            return bool(self._baked_has_q_self)
        val = bool(
            self.use_charge_self_energy
            and (
                self._charge_self_energy_active
                or self.training
                or float(self.omol_chi.detach().abs().max().item()) > 0.0
            )
        )
        if not self.training and getattr(self, "_is_baked", False):
            self._baked_has_q_self = val
            self._baked_chi_version = self.omol_chi._version
            self._baked_use_q_self = bool(self.use_charge_self_energy)
            self._baked_q_self_active = bool(self._charge_self_energy_active)
        return val

    @staticmethod
    def _add_ref_energy(e: Tensor, e_ref: Tensor) -> tuple[Tensor, Tensor]:
        """Return `(e_tot_device, e_tot_cpu_f64)` so float32 / MPS callers keep a device tensor in `energy`
        while `energy_f64` preserves exact IEEE-754 float64 precision (eliminating 0.5 eV ULP steps at >5k atoms)."""
        e_f64 = e.detach().cpu().double() + e_ref.detach().cpu().double()
        e_dev = (
            (e.double() + e_ref.detach()).to(e.dtype)
            if e.device.type != "mps"
            else e + e_ref.detach().to(dtype=e.dtype).to(device=e.device)
        )
        return e_dev, e_f64

    @staticmethod
    def _scale_out_units(out: dict[str, Tensor | None], out_units: str) -> dict[str, Tensor | None]:
        if out_units == "kcal":
            return out
        if out_units != "ev":
            raise ValueError(f"out_units must be 'kcal' or 'ev', got {out_units!r}")
        inv_ev = 1.0 / EV
        for k in ("energy", "energy_f64", "energy_residual", "forces", "stress", "dispersion_energy", "electrostatic_energy"):
            if out.get(k) is not None:
                out[k] = out[k] * inv_ev
        return out

    def enable_open_long_range(self, coulomb: bool = True, window: bool = False, r_inf: float = 1000.0) -> "MenaLAM":
        """Inference-time helper for open-boundary molecules/clusters (`pbc=False`):
        - `coulomb=True`: lifts `taper_elec` (`16.5..18.0 A` -> `r_inf..r_inf+1 A`), eliminating the 16.5-18.0 A
          vacuum Coulomb/charge-neutralization cutoff spike (and improving OMol25 `val_sub5k` force MAE by 1.3-1.5%
          on trained checkpoints while remaining bit-identical for all pairs <= 16.5 A).
        - `window=True`: also lifts `ic_windows` (`6.0..18.0 A` -> `r_inf..r_inf+1 A`) so gas-phase ion pairs
          retain constant asymptotic charges and exact -1/r^2 attraction out to arbitrary separation."""
        if coulomb and getattr(self.core, "taper_elec", None) is not None:
            self.core.taper_elec.r_on = float(r_inf)
            self.core.taper_elec.r_cut = float(r_inf + 1.0)
        if window:
            self.core.ic_r_on = float(r_inf)
            self.core.ic_r_cut = float(r_inf + 1.0)
            self.core.ic_windows = [(float(r_inf), float(r_inf + 1.0))]
        return self

    def forward_crystal(
        self,
        z: Tensor,
        pos: Tensor,
        cell: Tensor,
        batch: Tensor,
        edge_index: Tensor,
        edge_shift_frac: Tensor | None = None,
        u: Tensor | None = None,
        lot_id: Tensor | int | None = None,
        compute_stress: bool = True,
        compute_forces: bool = True,
        create_graph: bool | None = None,
        virial_mode: str | None = None,
        edge_shift_cart: Tensor | None = None,
        total_charge: Tensor | None = None,
        spin: Tensor | None = None,
        atom_charge: Tensor | None = None,
        lr_edge_index: Tensor | None = None,
        lr_edge_shift_frac: Tensor | None = None,
        lr_edge_shift_cart: Tensor | None = None,
        add_d3: bool = False,
        add_ref: bool = False,
        use_elec_context: bool | None = None,
        unwrap_dipole: bool = False,
        out_units: str = "kcal",
    ) -> dict[str, Tensor | None]:
        """Periodic forward pass (`pbc=True`).
        Supports both `virial_mode="edge"` (Direct Edge-Vector Virial for crystal LoTs 0..5) and `virial_mode="strain"`
        (3x3 strain tensor, also used automatically when `lot_id == omol_wb97mv` activates long-range electrostatics/D3).
        """
        if self.training and lot_id is None:
            raise ValueError(
                "forward_crystal requires an explicit lot_id when self.training is True "
                "so multi-dataset training never silently falls back to MP tokens."
            )
        B = cell.shape[0]
        N = pos.shape[0]
        vmode = self.virial_mode if virial_mode is None else virial_mode
        if vmode not in ("edge", "strain"):
            raise ValueError(f"virial_mode must be 'edge' or 'strain', got {vmode!r}")
        create_graph = self.training if create_graph is None else create_graph

        eff_lot_id = lot_id
        if eff_lot_id is None and u is not None and self.core.lot_embedding is not None and self._lot_from_u:
            eff_lot_id = u.long().clamp(0, 1)  # 0 -> mp_pbe (0), 1 -> mp_pbeU (1)
        run_elec = self.core._should_run_elec(True, eff_lot_id, use_elec_context)
        if spin is not None:
            spin_arg = spin
        else:
            spin_arg = (u.to(pos.dtype) + 1.0) if (u is not None and eff_lot_id is None) else None

        # When electrostatics/IC are active under PBC (e.g. omol_wb97mv in a periodic box), use strain/pos grad
        if run_elec:
            vmode = "strain"

        if vmode == "strain":
            if edge_shift_frac is None and edge_shift_cart is None:
                raise ValueError("virial_mode='strain' requires either edge_shift_frac or edge_shift_cart")
            pos = pos.detach().requires_grad_(compute_forces or compute_stress)
            eps = torch.zeros(B, 3, 3, dtype=pos.dtype, device=pos.device, requires_grad=compute_stress)
            pos_s = pos + torch.bmm(pos.unsqueeze(1), eps[batch]).squeeze(1)
            eb = batch[edge_index[0]]
            cell_s = cell + torch.bmm(cell, eps) if (edge_shift_frac is not None or lr_edge_shift_frac is not None) else None
            if edge_shift_frac is not None:
                shift = torch.bmm(edge_shift_frac.to(pos.dtype).unsqueeze(1), cell_s[eb]).squeeze(1)
            else:
                shift = edge_shift_cart + torch.bmm(edge_shift_cart.unsqueeze(1), eps[eb]).squeeze(1)

            lr_idx = lr_edge_index if lr_edge_index is not None else edge_index
            if lr_edge_index is not None:
                eb_lr = batch[lr_edge_index[0]]
                if lr_edge_shift_frac is not None:
                    lr_shift = torch.bmm(lr_edge_shift_frac.to(pos.dtype).unsqueeze(1), cell_s[eb_lr]).squeeze(1)
                elif lr_edge_shift_cart is not None:
                    lr_shift = lr_edge_shift_cart + torch.bmm(lr_edge_shift_cart.unsqueeze(1), eps[eb_lr]).squeeze(1)
                elif lr_edge_index.shape == edge_index.shape:
                    lr_shift = shift
                else:
                    raise ValueError(
                        "forward_crystal requires lr_edge_shift_frac or lr_edge_shift_cart when lr_edge_index is provided"
                    )
            else:
                lr_shift = shift

            rc = self.mol_r_cut if run_elec else self.crystal_r_cut
            ro = self.mol_r_on if run_elec else self.crystal_r_on
            c3 = self.mol_c3_taper if run_elec else None
            upr = self.mol_use_pair_radius if run_elec else None
            o = dict(
                self.core.forward_sparse(
                    z,
                    pos_s,
                    batch,
                    B,
                    edge_index,
                    shift,
                    total_charge=total_charge,
                    lr_edge_index=lr_idx,
                    lr_edge_shift=lr_shift,
                    compute_gradient_forces=False,
                    pbc=True,
                    spin=spin_arg,
                    atom_charge=atom_charge,
                    lot_id=eff_lot_id,
                    r_cut=rc,
                    r_on=ro,
                    c3_taper=c3,
                    use_pair_radius=upr,
                    use_elec_context=(use_elec_context if use_elec_context is not None else (None if eff_lot_id is not None else run_elec)),
                    unwrap_dipole=unwrap_dipole,
                    cell=cell + torch.bmm(cell, eps),  # strained cell: the mesh (long_range="mesh") gives the virial
                )
            )
            e_core = o["energy"]
            e = e_core
            elec_f = self.core._elec_mask(B, z.device, lot_id=eff_lot_id, pbc=True, use_elec_context=use_elec_context).to(
                dtype=e.dtype
            )
            if add_d3 and self.d3 is not None:
                e_d3 = self.d3.forward_sparse(z, pos_s, batch, B, lr_idx, lr_shift) * elec_f
                e = e + e_d3
                o["dispersion_energy"] = e_d3
            has_q_self = run_elec and self._has_active_charge_self_energy() and (o.get("charges") is not None)
            if has_q_self:
                e_q = self.charge_self_energy_sparse(z, o["charges"], batch, B).to(dtype=e.dtype, device=e.device) * elec_f
                e = e + e_q
            o["energy_residual"] = e
            if not compute_forces and not compute_stress:
                if add_ref:
                    e_ref = (
                        self.molecule_reference_sparse(
                            z,
                            batch,
                            B,
                            total_charge=total_charge,
                            spin=spin_arg,
                            charges=None,
                            include_charge_lookup=not has_q_self,
                        )
                        if run_elec
                        else self.crystal_reference(z, batch, u=u, B=B, lot_id=eff_lot_id)
                    )
                    e, o["energy_f64"] = self._add_ref_energy(e, e_ref)
                return self._scale_out_units({**o, "energy": e, "forces": None, "stress": None}, out_units)
            inputs = [pos, eps] if compute_stress else [pos]
            grads = torch.autograd.grad(e.sum(), inputs, create_graph=create_graph)
            forces = -grads[0] if compute_forces else None
            stress = None
            if compute_stress:
                vol = cell_volume(cell).clamp_min(1e-6)
                d = grads[1]
                stress = 0.5 * (d + d.transpose(-1, -2)) / vol.view(-1, 1, 1)
            if add_ref:
                e_ref = (
                    self.molecule_reference_sparse(
                        z,
                        batch,
                        B,
                        total_charge=total_charge,
                        spin=spin_arg,
                        charges=None,
                        include_charge_lookup=not has_q_self,
                    )
                    if run_elec
                    else self.crystal_reference(z, batch, u=u, B=B, lot_id=eff_lot_id)
                )
                e, o["energy_f64"] = self._add_ref_energy(e, e_ref)
            return self._scale_out_units({**o, "energy": e, "forces": forces, "stress": stress}, out_units)

        # virial_mode == "edge": Direct Edge-Vector Virial
        recv, send = edge_index[0], edge_index[1]
        if edge_shift_cart is not None:
            shift = edge_shift_cart
        else:
            if edge_shift_frac is None:
                raise ValueError("forward_crystal requires either edge_shift_frac or edge_shift_cart")
            if B == 1:
                shift = edge_shift_frac.to(pos.dtype) @ cell[0]
            else:
                shift = torch.bmm(edge_shift_frac.to(pos.dtype).unsqueeze(1), cell[batch[recv]]).squeeze(1)
        need_grad = compute_forces or compute_stress
        if "gather" in self.core.fast_ops:
            edge_diff = (pos.index_select(0, recv) - pos.index_select(0, send) - shift).detach()
        else:
            edge_diff = (pos[recv] - pos[send] - shift).detach()

        def core_energy(ed: Tensor) -> tuple[Tensor, Tensor]:
            o = self.core.forward_sparse(
                z,
                pos,
                batch,
                B,
                edge_index,
                shift,
                compute_gradient_forces=False,
                pbc=True,
                spin=spin_arg,
                lot_id=eff_lot_id,
                edge_diff=ed,
                r_cut=self.crystal_r_cut,
                r_on=self.crystal_r_on,
                use_elec_context=False,
            )
            return o["energy"].sum(), o["energy"]

        # force_backend "func": dE/d(edge_diff) from torch.func, the parameter backward is then first order through the
        # (compiled) stages; "autograd" needs create_graph double backward, which per-stage torch.compile cannot do
        if need_grad and getattr(self, "force_backend", "autograd") == "func":
            g_e, (_, e) = torch.func.grad_and_value(core_energy, has_aux=True)(edge_diff)
        else:
            if need_grad:
                edge_diff.requires_grad_(True)
            _, e = core_energy(edge_diff)
            g_e = None
        e_res = e
        if not need_grad:
            e_f64 = None
            if add_ref:
                e_ref = self.crystal_reference(z, batch, u=u, B=B, lot_id=eff_lot_id)
                e, e_f64 = self._add_ref_energy(e, e_ref)
            return self._scale_out_units(
                {"energy": e, "energy_f64": e_f64, "energy_residual": e_res, "forces": None, "stress": None, "dipole": None},
                out_units,
            )
        if g_e is None:
            g_e = torch.autograd.grad(e.sum(), edge_diff, create_graph=create_graph)[0]  # [E, 3]
        grad_pos = None
        forces = None
        if compute_forces or (create_graph and compute_stress and edge_shift_frac is not None):
            grad_pos = pos.new_zeros(N, 3).index_add_(0, recv, g_e) - pos.new_zeros(N, 3).index_add_(0, send, g_e)
            if compute_forces:
                forces = -grad_pos
        stress = None
        if compute_stress:
            vol = cell_volume(cell).clamp_min(1e-6)
            if create_graph and edge_shift_frac is not None and grad_pos is not None:
                eb = batch[recv]
                es_f = edge_shift_frac.to(pos.dtype)
                grad_defo_pos = pos.new_zeros(B, 3, 3).index_add_(
                    0, batch, pos.unsqueeze(-1) * grad_pos.unsqueeze(-2)
                )
                grad_cell_s = pos.new_zeros(B, 3, 3).index_add_(
                    0, eb, es_f.unsqueeze(-1) * (-g_e).unsqueeze(-2)
                )
                vir = grad_defo_pos + torch.bmm(cell.transpose(-1, -2), grad_cell_s)
            elif B == 1:
                vir = (edge_diff.transpose(0, 1) @ g_e).unsqueeze(0)
            else:
                eb = batch[recv]
                vir_e = edge_diff.unsqueeze(-1) * g_e.unsqueeze(-2)  # [E, 3, 3]
                vir = pos.new_zeros(B, 3, 3).index_add_(0, eb, vir_e)
            stress = 0.5 * (vir + vir.transpose(-1, -2)) / vol.view(-1, 1, 1)
        if add_ref:
            e_ref = self.crystal_reference(z, batch, u=u, B=B, lot_id=eff_lot_id)
            e, e_f64 = self._add_ref_energy(e, e_ref)
            return self._scale_out_units(
                {"energy": e, "energy_f64": e_f64, "energy_residual": e_res, "forces": forces, "stress": stress, "dipole": None},
                out_units,
            )
        return self._scale_out_units({"energy": e, "forces": forces, "stress": stress, "dipole": None}, out_units)

    def forward_molecule(
        self,
        z: Tensor,
        pos: Tensor,
        mask: Tensor,
        total_charge: Tensor | None = None,
        spin: Tensor | None = None,
        lot_id: Tensor | int | None = None,
        compute_gradient_forces: bool = True,
        compute_forces: bool | None = None,
        single_island: bool = False,
        add_d3: bool = False,
        add_ref: bool = False,
        out_units: str = "kcal",
        sparse: bool = False,
        **kw,
    ) -> dict[str, Tensor | None]:
        """Stream B 17-bucket forward pass on molecules (`pbc=False`), dense by default or sparse (`sparse=True`)."""
        if compute_forces is not None:
            compute_gradient_forces = compute_forces
        eff_lot_id = self.default_mol_lot_id if lot_id is None else lot_id
        rc = kw.pop("r_cut", self.mol_r_cut)
        ro = kw.pop("r_on", self.mol_r_on)
        c3 = kw.pop("c3_taper", self.mol_c3_taper)
        upr = kw.pop("use_pair_radius", self.mol_use_pair_radius)
        edge_index = kw.pop("edge_index", None)
        edge_shift = kw.pop("edge_shift", None)
        lr_edge_index = kw.pop("lr_edge_index", None)
        lr_edge_shift = kw.pop("lr_edge_shift", None)
        e_mult = int(kw.pop("e_mult", 0))
        e_min = int(kw.pop("e_min", 0))
        use_sparse = bool(sparse or edge_index is not None)

        if use_sparse:
            B, N = int(z.shape[0]), int(z.shape[1])
            dev = z.device
            ghost_atom = B * N
            if edge_index is None:
                from lam.models.traced_force import crystal_e_pad

                with torch.no_grad():
                    eye = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
                    pm = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye
                    d = torch.cdist(pos.detach(), pos.detach())
                    b_sr, i_sr, j_sr = (pm & (d < rc + 1e-3)).nonzero(as_tuple=True)
                    r_sr = b_sr * N + i_sr
                    s_sr = b_sr * N + j_sr
                    e_sr = int(r_sr.numel())
                    e_pad = crystal_e_pad(e_sr, e_mult, e_min) if e_mult > 0 else max(e_sr, 1)
                    pad_sr = e_pad - e_sr
                    edge_index = torch.stack(
                        [torch.cat([r_sr, torch.full((pad_sr,), ghost_atom, device=dev, dtype=torch.long)]),
                         torch.cat([s_sr, torch.full((pad_sr,), ghost_atom, device=dev, dtype=torch.long)])],
                        dim=0,
                    )
                    edge_shift = pos.new_zeros((e_pad, 3))
                    if pad_sr > 0:
                        edge_shift[e_sr:, 0] = 100.0
            elif edge_shift is None:
                edge_shift = pos.new_zeros((edge_index.shape[1], 3))

            if lr_edge_index is None:
                # B4 hybrid sparse-SR + dense-block-LR: short-range stages 0..3 run on sparse edge_index (E_pad),
                # while long-range (18 A IC window bmm, split-charge w_lr, Coulomb) runs on dense [B, N, N].
                def core_out(p: Tensor, grad_forces: bool) -> dict[str, Tensor | None]:
                    return dict(
                        self.core(
                            z,
                            p,
                            mask,
                            total_charge=total_charge,
                            spin=spin,
                            compute_gradient_forces=grad_forces,
                            single_island=single_island,
                            lot_id=eff_lot_id,
                            r_cut=rc,
                            r_on=ro,
                            c3_taper=c3,
                            use_pair_radius=upr,
                            edge_index=edge_index,
                            edge_shift=edge_shift,
                            **kw,
                        )
                    )
            else:
                ghost_b = B
                n_graphs = B + 1
                m_flat = mask.reshape(B * N)
                b_idx = torch.arange(B, device=dev, dtype=torch.long).unsqueeze(1).expand(B, N).reshape(B * N)
                batch = torch.cat(
                    [torch.where(m_flat, b_idx, torch.full_like(b_idx, ghost_b)),
                     torch.tensor([ghost_b], device=dev, dtype=torch.long)],
                    dim=0,
                )
                z_flat = torch.cat(
                    [torch.where(m_flat, z.reshape(B * N), torch.zeros_like(z.reshape(B * N))),
                     torch.zeros(1, device=dev, dtype=z.dtype)],
                    dim=0,
                )
                if lr_edge_shift is None:
                    lr_edge_shift = pos.new_zeros((lr_edge_index.shape[1], 3))

                q_pad = torch.cat([total_charge.reshape(B), total_charge.new_zeros(1)]) if total_charge is not None else None
                sp_pad = torch.cat([spin.reshape(B), spin.new_ones(1)]) if spin is not None else None
                eff_lot_pad = (
                    torch.cat([eff_lot_id.reshape(B), eff_lot_id.reshape(B)[:1]])
                    if (torch.is_tensor(eff_lot_id) and eff_lot_id.ndim > 0)
                    else eff_lot_id
                )
                ac = kw.pop("atom_charge", None)
                ac_pad = (
                    torch.cat([(ac * mask.to(ac.dtype)).reshape(B * N), ac.new_zeros(1)])
                    if ac is not None
                    else None
                )

                def core_out(p: Tensor, grad_forces: bool) -> dict[str, Tensor | None]:
                    pf = torch.cat([p.reshape(B * N, 3) * m_flat.unsqueeze(-1).to(p.dtype), p.new_zeros((1, 3))], dim=0)
                    raw = dict(
                        self.core.forward_sparse(
                            z_flat,
                            pf,
                            batch,
                            n_graphs,
                            edge_index,
                            edge_shift,
                            total_charge=q_pad,
                            spin=sp_pad,
                            atom_charge=ac_pad,
                            lr_edge_index=lr_edge_index,
                            lr_edge_shift=lr_edge_shift,
                            compute_gradient_forces=grad_forces,
                            single_island=single_island,
                            pbc=False,
                            lot_id=eff_lot_pad,
                            r_cut=rc,
                            r_on=ro,
                            c3_taper=c3,
                            use_pair_radius=upr,
                            **kw,
                        )
                    )
                    for k_g in ("energy", "electrostatic_energy", "dispersion_energy", "dipole"):
                        if raw.get(k_g) is not None:
                            raw[k_g] = raw[k_g][:B]
                    if raw.get("charges") is not None:
                        raw["charges"] = raw["charges"][:-1].reshape(B, N) * mask.to(raw["charges"].dtype)
                    if raw.get("forces") is not None:
                        raw["forces"] = raw["forces"][:-1].reshape(B, N, 3) * mask.unsqueeze(-1).to(raw["forces"].dtype)
                    return raw
        else:
            def core_out(p: Tensor, grad_forces: bool) -> dict[str, Tensor | None]:
                return dict(
                    self.core(
                        z,
                        p,
                        mask,
                        total_charge=total_charge,
                        spin=spin,
                        compute_gradient_forces=grad_forces,
                        single_island=single_island,
                        lot_id=eff_lot_id,
                        r_cut=rc,
                        r_on=ro,
                        c3_taper=c3,
                        use_pair_radius=upr,
                        **kw,
                    )
                )

        any_elec = self.core._should_run_elec(False, eff_lot_id, kw.get("use_elec_context"))
        has_q_self = any_elec and self._has_active_charge_self_energy()
        outer_grad = (any_elec and add_d3 and self.d3 is not None) or has_q_self
        if outer_grad:

            def total_energy(p: Tensor) -> tuple[Tensor, dict[str, Tensor | None]]:
                out = core_out(p, False)
                e_tot = out["energy"]
                elec_f = self._ref_mask(z.shape[0], z.device, eff_lot_id, kw.get("use_elec_context")).to(dtype=e_tot.dtype)
                if add_d3 and self.d3 is not None:
                    e_d3 = self.d3(z, p, mask) * elec_f
                    e_tot = e_tot + e_d3
                    out["dispersion_energy"] = e_d3
                if has_q_self and (out.get("charges") is not None):
                    e_q = self.charge_self_energy(z, out["charges"], mask).to(dtype=e_tot.dtype, device=e_tot.device) * elec_f
                    e_tot = e_tot + e_q
                out["energy_residual"] = e_tot
                return e_tot.sum(), out

            if compute_gradient_forces and getattr(self, "force_backend", "autograd") == "func":
                out = _func_grad_with_none_aux(total_energy, pos.detach())
                out["forces"] = -out.pop("_grad") * mask.unsqueeze(-1).float()
            else:
                if compute_gradient_forces and not pos.requires_grad:
                    pos = pos.detach().requires_grad_(True)
                _, out = total_energy(pos)
                if compute_gradient_forces:
                    f_tot = -torch.autograd.grad(
                        out["energy_residual"].sum(), pos, create_graph=self.training, retain_graph=self.training
                    )[0] * mask.unsqueeze(-1).float()
                    out["forces"] = f_tot
            e_tot = out["energy_residual"]
            q_self_added = has_q_self and (out.get("charges") is not None)
            if add_ref:
                e_ref = self.molecule_reference(
                    z, mask, total_charge, spin, charges=None, include_charge_lookup=not q_self_added
                )
                e_ref = self._token_reference_dense(
                    e_ref, self._ref_mask(z.shape[0], z.device, eff_lot_id, kw.get("use_elec_context")),
                    z, mask, eff_lot_id,
                )
                e_tot, out["energy_f64"] = self._add_ref_energy(e_tot, e_ref)
            out["energy"] = e_tot
            return self._scale_out_units(out, out_units)

        if compute_gradient_forces and getattr(self, "force_backend", "autograd") == "func":
            # torch.func forces (see forward_crystal): first-order parameter backward through the (compiled) stages

            def core_energy(p: Tensor) -> tuple[Tensor, dict[str, Tensor | None]]:
                o = core_out(p, False)
                return o["energy"].sum(), o

            out = _func_grad_with_none_aux(core_energy, pos.detach())
            g = out.pop("_grad")
            out["forces"] = -g * mask.unsqueeze(-1).to(g.dtype)
        else:
            out = core_out(pos, compute_gradient_forces)
        out["energy_residual"] = out["energy"]
        if add_ref:
            e_ref = self.molecule_reference(z, mask, total_charge, spin, charges=None, include_charge_lookup=True)
            e_ref = self._token_reference_dense(
                e_ref, self._ref_mask(z.shape[0], z.device, eff_lot_id, kw.get("use_elec_context")),
                z, mask, eff_lot_id,
            )
            e_pred = out["energy"]
            out["energy"], out["energy_f64"] = self._add_ref_energy(e_pred, e_ref)
        return self._scale_out_units(out, out_units)

    def forward_sparse(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        n_graphs: int,
        edge_index: Tensor,
        edge_shift: Tensor,
        *,
        pbc: bool = False,
        cell: Tensor | None = None,
        edge_shift_frac: Tensor | None = None,
        u: Tensor | None = None,
        lot_id: Tensor | int | None = None,
        total_charge: Tensor | None = None,
        spin: Tensor | None = None,
        atom_charge: Tensor | None = None,
        lr_edge_index: Tensor | None = None,
        lr_edge_shift: Tensor | None = None,
        compute_gradient_forces: bool = True,
        compute_stress: bool = False,
        single_island: bool = False,
        add_d3: bool = False,
        add_ref: bool = False,
        virial_mode: str | None = None,
        unwrap_dipole: bool = False,
        out_units: str = "kcal",
        **kw,
    ) -> dict[str, Tensor | None]:
        """Unified sparse forward supporting both `pbc=True` and `pbc=False` across all LoTs."""
        if pbc:
            if self.training and lot_id is None:
                raise ValueError(
                    "forward_sparse(pbc=True) requires an explicit lot_id when self.training is True "
                    "so multi-dataset training never silently falls back to MP tokens."
                )
            if cell is not None and (compute_stress or (virial_mode or self.virial_mode) in ("edge", "strain")):
                return self.forward_crystal(
                    z,
                    pos,
                    cell,
                    batch,
                    edge_index,
                    edge_shift_frac=edge_shift_frac,
                    edge_shift_cart=edge_shift,
                    u=u,
                    lot_id=lot_id,
                    compute_stress=compute_stress,
                    compute_forces=compute_gradient_forces,
                    virial_mode=virial_mode,
                    total_charge=total_charge,
                    spin=spin,
                    atom_charge=atom_charge,
                    lr_edge_index=lr_edge_index,
                    lr_edge_shift_cart=lr_edge_shift,
                    add_d3=add_d3,
                    add_ref=add_ref,
                    unwrap_dipole=unwrap_dipole,
                    out_units=out_units,
                )
            eff_lot_id = lot_id
            if eff_lot_id is None and u is not None and self.core.lot_embedding is not None and self._lot_from_u:
                eff_lot_id = u.long().clamp(0, 1)
            if not self.core._should_run_elec(True, eff_lot_id, kw.get("use_elec_context")):
                rc = kw.pop("r_cut", self.crystal_r_cut)
                ro = kw.pop("r_on", self.crystal_r_on)
                out_c = dict(
                    self.core.forward_sparse(
                        z,
                        pos,
                        batch,
                        n_graphs,
                        edge_index,
                        edge_shift,
                        compute_gradient_forces=compute_gradient_forces,
                        pbc=True,
                        spin=(u.to(pos.dtype) + 1.0) if (u is not None and eff_lot_id is None) else spin,
                        lot_id=eff_lot_id,
                        r_cut=rc,
                        r_on=ro,
                        use_elec_context=False,
                        unwrap_dipole=unwrap_dipole,
                        **kw,
                    )
                )
                out_c["energy_residual"] = out_c["energy"]
                if add_ref:
                    e_ref = self.crystal_reference(z, batch, u=u, B=n_graphs, lot_id=eff_lot_id)
                    e_pred = out_c["energy"]
                    out_c["energy"], out_c["energy_f64"] = self._add_ref_energy(e_pred, e_ref)
                return self._scale_out_units(out_c, out_units)

        eff_lot_id = self.default_mol_lot_id if lot_id is None else lot_id
        rc = kw.pop("r_cut", self.mol_r_cut)
        ro = kw.pop("r_on", self.mol_r_on)
        c3 = kw.pop("c3_taper", self.mol_c3_taper)
        upr = kw.pop("use_pair_radius", self.mol_use_pair_radius)
        any_elec = self.core._should_run_elec(pbc, eff_lot_id, kw.get("use_elec_context"))
        mesh = self.core.mesh_def  # long range on the mesh: lr_edge_index serves D3 only, the core never sees it
        if not pbc and ((any_elec and not mesh) or (add_d3 and self.d3 is not None)) and lr_edge_index is None:
            raise ValueError(
                "forward_sparse(pbc=False) with active electrostatics/IC or D3 requires an explicit lr_edge_index "
                "(up to r_cut_elec=18.0 A); refusing to silently substitute 6.0 A short-range edge_index."
            )
        has_q_self = any_elec and self._has_active_charge_self_energy()
        outer_grad = (any_elec and add_d3 and self.d3 is not None) or has_q_self
        if outer_grad:
            if compute_gradient_forces and not pos.requires_grad:
                pos = pos.detach().requires_grad_(True)
            if lr_edge_index is None:
                lr_edge_index, lr_edge_shift = edge_index, edge_shift
            elif lr_edge_shift is None:
                lr_edge_shift = pos.new_zeros((lr_edge_index.shape[1], 3))
            out = dict(
                self.core.forward_sparse(
                    z,
                    pos,
                    batch,
                    n_graphs,
                    edge_index,
                    edge_shift,
                    total_charge=total_charge,
                    spin=spin,
                    atom_charge=atom_charge,
                    lr_edge_index=None if mesh else lr_edge_index,
                    lr_edge_shift=None if mesh else lr_edge_shift,
                    compute_gradient_forces=False,
                    single_island=single_island,
                    cell=cell,
                    pbc=pbc,
                    lot_id=eff_lot_id,
                    r_cut=rc,
                    r_on=ro,
                    c3_taper=c3,
                    use_pair_radius=upr,
                    unwrap_dipole=unwrap_dipole,
                    **kw,
                )
            )
            e_tot = out["energy"]
            elec_f = (
                self.core._elec_mask(n_graphs, z.device, lot_id=eff_lot_id, pbc=True, use_elec_context=kw.get("use_elec_context"))
                if pbc else self._ref_mask(n_graphs, z.device, eff_lot_id, kw.get("use_elec_context"))
            ).to(dtype=e_tot.dtype)
            if add_d3 and self.d3 is not None:
                e_d3 = self.d3.forward_sparse(z, pos, batch, n_graphs, lr_edge_index, lr_edge_shift) * elec_f
                e_tot = e_tot + e_d3
                out["dispersion_energy"] = e_d3
            q_self_added = has_q_self and (out.get("charges") is not None)
            if q_self_added:
                e_q = self.charge_self_energy_sparse(z, out["charges"], batch, n_graphs).to(
                    dtype=e_tot.dtype, device=e_tot.device
                ) * elec_f
                e_tot = e_tot + e_q
            if compute_gradient_forces:
                out["forces"] = -torch.autograd.grad(
                    e_tot.sum(), pos, create_graph=self.training, retain_graph=self.training
                )[0]
            out["energy_residual"] = e_tot
            if add_ref:
                e_ref = self.molecule_reference_sparse(
                    z,
                    batch,
                    n_graphs,
                    total_charge=total_charge,
                    spin=spin,
                    charges=None,
                    include_charge_lookup=not q_self_added,
                )
                e_ref = self._token_reference(e_ref, elec_f.bool(), z, batch, n_graphs, eff_lot_id)
                e_tot, out["energy_f64"] = self._add_ref_energy(e_tot, e_ref)
            out["energy"] = e_tot
            return self._scale_out_units(out, out_units)

        out = dict(
            self.core.forward_sparse(
                z,
                pos,
                batch,
                n_graphs,
                edge_index,
                edge_shift,
                total_charge=total_charge,
                spin=spin,
                atom_charge=atom_charge,
                lr_edge_index=None if mesh else lr_edge_index,
                lr_edge_shift=None if mesh else lr_edge_shift,
                compute_gradient_forces=compute_gradient_forces,
                single_island=single_island,
                cell=cell,
                pbc=pbc,
                lot_id=eff_lot_id,
                r_cut=rc,
                r_on=ro,
                c3_taper=c3,
                use_pair_radius=upr,
                unwrap_dipole=unwrap_dipole,
                **kw,
            )
        )
        out["energy_residual"] = out["energy"]
        if add_ref:
            e_ref = self.molecule_reference_sparse(
                z, batch, n_graphs, total_charge=total_charge, spin=spin, charges=None, include_charge_lookup=True
            )
            if not pbc:
                e_ref = self._token_reference(
                    e_ref, self._ref_mask(n_graphs, z.device, eff_lot_id, kw.get("use_elec_context")), z, batch,
                    n_graphs, eff_lot_id,
                )
            e_pred = out["energy"]
            out["energy"], out["energy_f64"] = self._add_ref_energy(e_pred, e_ref)
        return self._scale_out_units(out, out_units)

    def forward(self, z: Tensor, pos: Tensor, *args, **kwargs) -> dict[str, Tensor | None]:
        """Auto-dispatching forward:
        - `model(z, pos, cell, batch, edge_index, edge_shift_frac, u=...)` -> `forward_crystal`
        - `model(z, pos, mask, ...)` -> `forward_molecule`
        """
        if "cell" in kwargs or (len(args) >= 4 and args[0].ndim == 3 and args[0].shape[-2:] == (3, 3)):
            return self.forward_crystal(z, pos, *args, **kwargs)
        return self.forward_molecule(z, pos, *args, **kwargs)


@torch.no_grad()
def _copy_swiglu_wider(dst_ffn: nn.Module, src_in: Tensor, src_out: Tensor) -> None:
    """Function-preserving width copy from src SwiGLU [2*H_old, D] / [D, H_old] into dst SwiGLU [2*H_new, D] / [D, H_new].
    When H_new == H_old, performs an exact copy. When H_new > H_old, places src gate/val rows in the first H_old slots
    of each half and sets dst_ffn.ffn_out.weight[:, H_old:] = 0 so the output is bit-for-bit identical at step 0 while
    extra hidden channels have non-zero activations and receive gradients immediately."""
    h_old = src_out.shape[1]
    h_new = dst_ffn.ffn_out.weight.shape[1]
    if h_old == h_new:
        dst_ffn.ffn_in.weight.copy_(src_in)
        dst_ffn.ffn_out.weight.copy_(src_out)
        return
    if h_old > h_new:
        raise ValueError(f"Cannot shrink SwiGLU hidden_dim from {h_old} to {h_new}")
    gate_old, val_old = src_in.chunk(2, dim=0)
    dst_ffn.ffn_in.weight[:h_old].copy_(gate_old)
    dst_ffn.ffn_in.weight[h_new : h_new + h_old].copy_(val_old)
    dst_ffn.ffn_out.weight.zero_()
    dst_ffn.ffn_out.weight[:, :h_old].copy_(src_out)


@torch.no_grad()
def graft_materials_checkpoint(
    ckpt_path: str | Path,
    device: torch.device | str = "cpu",
    swiglu_ratio: float = 2.0,
    virial_mode: str = "edge",
) -> tuple[MenaLAM, dict]:
    """Load a MenaPeriodic checkpoint (e.g. `a100_v2_efs_v1/final.pt`) into a full unified `MenaLAM` model
    (including Stream B electrostatics/IC/D3 modules and LoT conditioning), grafting `spin_embedding` rows 1 & 2
    into `mp_pbe` (id 0) and `mp_pbeU` (id 1) and widening SwiGLU if `swiglu_ratio > ckpt_swiglu_ratio`."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kw = dict(ck.get("kw", {}))
    sd = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()}
    elem_prior = bool(kw.pop("elem_prior", True))
    ckpt_ratio = float(kw.get("swiglu_ratio", 1.0))
    target_ratio = max(float(swiglu_ratio), ckpt_ratio)

    lam_kw = mena_lam_kwargs(
        d_model=kw.get("d_model", 384),
        msg_dim=kw.get("msg_dim", 128),
        vector_rank=kw.get("vector_rank", 32),
        tensor_rank=kw.get("tensor_rank", 16),
        stages=kw.get("stages", 3),
        num_rbf=kw.get("num_rbf", 32),
        r_cut=kw.get("r_cut", 6.0),
        r_on=kw.get("r_on", 5.0),
        swiglu_ratio=target_ratio,
        ttt_mix=kw.get("ttt_mix", 2),
        vttv_rank=kw.get("vttv_rank", 8),
        l3_rank=kw.get("l3_rank", 8),
        c3_taper=kw.get("c3_taper", True),
        pair_radius=kw.get("pair_radius", True),
        pair_radius_warp=kw.get("pair_radius_warp", True),
    )
    model = MenaLAM(
        mu=ck.get("mu"),
        elem_prior=elem_prior,
        with_d3=True,
        virial_mode=virial_mode,
        crystal_r_cut=kw.get("r_cut", 6.0),
        crystal_r_on=kw.get("r_on", 5.0),
        mu_in_ev=True,
        **lam_kw,
    )
    if ck.get("mu") is None and "mu" in sd:
        model.mu.copy_(sd["mu"].to(torch.float64))
        n_mu = min(model.mu.shape[0], model.crystal_e0.shape[1])
        model.crystal_e0[0, :n_mu] = model.mu[:n_mu, 0]
        model.crystal_e0[1, :n_mu] = model.mu[:n_mu, 1]

    model_sd = dict(model.core.named_parameters())
    model_sd.update(model.core.named_buffers())
    copied, widened = [], []
    for k, v in sd.items():
        if not k.startswith("core."):
            continue
        sub = k[len("core.") :]
        if sub.endswith(".swiglu.ffn_in.weight"):
            stage_idx = int(sub.split(".")[1])
            out_key = k.replace(".ffn_in.weight", ".ffn_out.weight")
            _copy_swiglu_wider(model.core.stages_list[stage_idx].swiglu, v, sd[out_key])
            widened.append(sub)
            continue
        if sub.endswith(".swiglu.ffn_out.weight"):
            continue
        if sub in model_sd and model_sd[sub].shape == v.shape:
            model_sd[sub].copy_(v)
            copied.append(sub)

    if "core.spin_embedding.weight" in sd and model.core.lot_embedding is not None:
        model.core.graft_spin_into_lot(
            sd["core.spin_embedding.weight"],
            mp_pbe_id=LOT_NAME_TO_ID["mp_pbe"],
            mp_pbeu_id=LOT_NAME_TO_ID["mp_pbeU"],
        )
        model._lot_from_u = True

    model = model.to(device).eval()
    return model, {"copied_keys": len(copied), "widened_swiglu": len(widened), "ckpt_kw": kw}


@torch.no_grad()
def graft_molecules_checkpoint(
    ckpt_path: str | Path,
    device: torch.device | str = "cpu",
    swiglu_ratio: float = 2.0,
    weights: str = "model_state",
    continuous: bool = False,
    unify_cutoff: bool = False,
) -> tuple[MenaLAM, dict]:
    """Load an OMol25 MenaACE checkpoint into a unified `MenaLAM` model.
    - When `continuous=False` (default), preserves the legacy checkpoint's exact cutoff (`5.5 A`) and island settings
      so `verify_l0_graft_identity.py` reproduces `< 1e-12` (f64) / `< 1e-5` (f32) output identity on E, F, and dipole.
    - When `continuous=True`, upgrades the grafted model to the continuous L0 design (`charge_mode='split'`,
      `coulomb_range='dsf'`, `ic_mode='window'`, `use_charge_self_energy=True`), keeping the checkpoint's trained
      message-passing cutoff (`5.5 A` for older molecular checkpoints, `6.0 A` for newer ones) unless `unify_cutoff=True`."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    mkw = dict(cfg.get("model_kwargs", {}))
    raw_sd = ck.get(weights) or ck["model_state"]
    sd = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v for k, v in raw_sd.items()}

    ckpt_ratio = float(mkw.get("swiglu_ratio", 2.0 / 3.0))
    target_ratio = max(float(swiglu_ratio), ckpt_ratio)
    with_d3 = cfg.get("d3_mode", "subtract") == "subtract"
    ckpt_rc = float(mkw.get("r_cut", 6.0))
    ckpt_ro = float(mkw.get("r_on", 5.0))

    if continuous:
        rc = 6.0 if unify_cutoff else ckpt_rc
        ro = 5.0 if unify_cutoff else ckpt_ro
        charge_mode = "split"
        coulomb_range = "dsf"
        ic_mode = "window"
        ic_r_cut = 18.0
        ic_lr_dim = 16
        use_charge_self_energy = True
    else:
        rc = ckpt_rc
        ro = ckpt_ro
        charge_mode = mkw.get("charge_mode", "island")
        coulomb_range = mkw.get("coulomb_range", "full")
        ic_mode = mkw.get("ic_mode", "island")
        ic_r_cut = float(mkw.get("ic_r_cut", 12.0))
        ic_lr_dim = int(mkw.get("ic_lr_dim", 0))
        use_charge_self_energy = False

    lam_kw = mena_lam_kwargs(
        d_model=mkw.get("d_model", 384),
        msg_dim=mkw.get("msg_dim", 0),
        vector_rank=mkw.get("vector_rank", 32),
        tensor_rank=mkw.get("tensor_rank", 16),
        stages=mkw.get("stages", 3),
        num_rbf=mkw.get("num_rbf", 32),
        r_cut=rc,
        r_on=ro,
        swiglu_ratio=target_ratio,
        ttt_mix=mkw.get("ttt_mix", 2),
        vttv_rank=mkw.get("vttv_rank", 8),
        l3_rank=mkw.get("l3_rank", 8),
        c3_taper=True,
        pair_radius=True,
        pair_radius_warp=True,
        use_charge_embedding=mkw.get("use_charge_embedding", True),
        use_electrostatics=mkw.get("use_electrostatics", True),
        use_spin_embedding=mkw.get("use_spin_embedding", True),
        small_global_init=mkw.get("small_global_init", True),
        charge_mode=charge_mode,
        coulomb_range=coulomb_range,
        charge_bound=mkw.get("charge_bound", 2.0),
        feedback_bound=mkw.get("feedback_bound", 2.0),
        island_context=mkw.get("island_context", [0, 1, 2]),
        ic_dim=mkw.get("ic_dim", 32),
        ic_heads=mkw.get("ic_heads", 4),
        ic_hidden=mkw.get("ic_hidden", 64),
        ic_mode=ic_mode,
        ic_r_cut=ic_r_cut,
        ic_r_on=6.0,
        ic_lr_dim=ic_lr_dim,
        fb_input_norm=mkw.get("fb_input_norm", True),
    )
    model = MenaLAM(
        elem_prior=True,
        with_d3=with_d3,
        mol_r_cut=rc,
        mol_r_on=ro,
        use_charge_self_energy=use_charge_self_energy,
        **lam_kw,
    )
    if continuous and unify_cutoff:
        model.mol_c3_taper = True
    else:
        model.mol_c3_taper = bool(mkw.get("c3_taper", False))
    model.mol_use_pair_radius = bool(mkw.get("pair_radius", False))
    if "omol_ref" in cfg and cfg["omol_ref"] is not None:
        model.set_omol_reference(cfg["omol_ref"])

    model_sd = dict(model.core.named_parameters())
    model_sd.update(model.core.named_buffers())
    copied, widened = [], []
    for k, v in sd.items():
        if k == "embedding.weight" and isinstance(model.core.embedding, ElementPrior):
            # Graft standard Embedding into ElementPrior: proj=0, resid=embedding.weight -> bit-identical at step 0
            model.core.embedding.proj.weight.zero_()
            model.core.embedding.resid.copy_(v)
            copied.append(k)
            continue
        if k.endswith(".swiglu.ffn_in.weight"):
            stage_idx = int(k.split(".")[1])
            out_key = k.replace(".ffn_in.weight", ".ffn_out.weight")
            _copy_swiglu_wider(model.core.stages_list[stage_idx].swiglu, v, sd[out_key])
            widened.append(k)
            continue
        if k.endswith(".swiglu.ffn_out.weight"):
            continue
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k].copy_(v)
            copied.append(k)

    model = model.to(device).eval()
    return model, {"copied_keys": len(copied), "widened_swiglu": len(widened), "config": cfg}


def resolve_inference_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve `'auto'` (or `None`) to `'cuda'` -> `'mps'` -> `'cpu'`."""
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


_MP_U_ELEMENTS: frozenset[int] = frozenset({23, 24, 25, 26, 27, 28, 42, 74})
_MP_O_OR_F: frozenset[int] = frozenset({8, 9})


def _compute_mp_u_flag(atomic_numbers: Sequence[int] | np.ndarray | Tensor) -> int:
    """MP2020 Hubbard +U rule: u = 1 iff structure contains (O or F) and any of {V, Cr, Mn, Fe, Co, Ni, Mo, W}."""
    if isinstance(atomic_numbers, Tensor):
        nums = set(atomic_numbers.detach().cpu().numpy().tolist())
    elif isinstance(atomic_numbers, np.ndarray):
        nums = set(atomic_numbers.tolist())
    else:
        nums = set(int(x) for x in atomic_numbers)
    return 1 if (nums & _MP_O_OR_F) and (nums & _MP_U_ELEMENTS) else 0


def _resolve_torch_dtype(
    dtype: str | torch.dtype | None,
    device: torch.device,
    current_dtype: torch.dtype,
    *,
    stacklevel: int = 3,
) -> torch.dtype:
    explicit = dtype is not None
    if dtype is None:
        target = current_dtype
    elif isinstance(dtype, torch.dtype):
        target = dtype
    else:
        s = str(dtype).lower().strip()
        if s in ("float32", "fp32", "f32", "float"):
            target = torch.float32
        elif s in ("float64", "fp64", "f64", "double"):
            target = torch.float64
        elif s in ("bfloat16", "bf16"):
            target = torch.bfloat16
        elif s in ("float16", "fp16", "f16", "half"):
            target = torch.float16
        else:
            raise ValueError(f"Unsupported dtype {dtype!r}")
    if device.type == "mps" and target == torch.float64:
        if explicit:
            import warnings

            warnings.warn("MPS does not support float64; falling back to float32.", UserWarning, stacklevel=stacklevel)
        return torch.float32
    return target


@torch.no_grad()
def load_mena_lam_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device | None = "auto",
    weights: str = "ema",
    dtype: str | torch.dtype | None = "float32",
) -> MenaLAM:
    """Load a trained `MenaLAM` checkpoint (any arch in `ARCHS`)
    for inference on CUDA, MPS, or CPU, selecting EMA shadow weights (`weights='ema'`) or live weights (`weights='live'`)."""
    dev = resolve_inference_device(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ck, MenaLAM):
        model = ck
    else:
        kw = dict(ck.get("kw", {}))
        args = ck.get("args", {}) or {}
        if not kw:
            stages = int(args.get("stages", 4))
            arch = str(args.get("arch") or "lam1")
            kw = mena_lam_kwargs(stages=stages, **arch_preset_kwargs(arch, stages=stages))
        elem_prior = bool(kw.pop("elem_prior", True))
        with_d3 = bool(kw.pop("with_d3", True) if args.get("use_d3") is None else args.get("use_d3"))
        virial_mode = str(kw.pop("virial_mode", args.get("virial_mode") or "edge"))
        use_q_self = bool(True if args.get("use_charge_self_energy") is None else args.get("use_charge_self_energy"))
        model = MenaLAM(
            mu=ck.get("mu"),
            elem_prior=elem_prior,
            with_d3=with_d3,
            virial_mode=virial_mode,
            crystal_r_cut=float(ck.get("crystal_r_cut", kw.get("r_cut", 6.0))),
            crystal_r_on=float(ck.get("crystal_r_on", kw.get("r_on", 5.0))),
            mol_r_cut=float(ck.get("mol_r_cut", kw.get("r_cut", 6.0))),
            mol_r_on=float(ck.get("mol_r_on", kw.get("r_on", 5.0))),
            use_charge_self_energy=use_q_self,
            mu_in_ev=False,
            **kw,
        )
        if args.get("mol_c3_taper") is not None:
            model.mol_c3_taper = bool(args["mol_c3_taper"])

        raw_ms = ck.get("model_state", ck)
        sd = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v for k, v in raw_ms.items() if torch.is_tensor(v)}
        w_mode = str(weights).lower().strip()
        if w_mode == "ema" and isinstance(ck.get("ema"), dict) and isinstance(ck["ema"].get("shadow"), dict):
            for k, v in ck["ema"]["shadow"].items():
                sd[k.replace("._orig_mod.", ".").replace("_orig_mod.", "")] = v
        elif w_mode not in ("ema", "live", "model", "model_state"):
            raise ValueError(f"weights must be 'ema' or 'live', got {weights!r}")

        if model.d3 is not None and (bool(args.get("d3_learnable", False)) or "d3.s8" in sd):
            if not isinstance(getattr(model.d3, "s8", None), nn.Parameter) and "d3.s8" in sd:
                model.d3 = D3BJ(r_cut=12.0, taper_on=9.0, learnable=True)
        if "crystal_e0" in sd and sd["crystal_e0"].shape != model.crystal_e0.shape:
            model.register_buffer("crystal_e0", torch.zeros_like(sd["crystal_e0"], dtype=torch.float64))

        model.load_state_dict(sd, strict=False)
        for buf_name in MenaLAM._F64_BUFFERS:
            if buf_name in sd:
                getattr(model, buf_name).copy_(sd[buf_name].to(torch.float64))
            elif buf_name in ck and torch.is_tensor(ck[buf_name]):
                getattr(model, buf_name).copy_(ck[buf_name].to(torch.float64))
        if (
            float(model.omol_e0.detach().abs().max().item()) > 0.0
            or float(model.omol_chi.detach().abs().max().item()) > 0.0
        ):
            model._charge_self_energy_active = True

    cur_dtype = next(model.parameters()).dtype
    target_dtype = _resolve_torch_dtype(dtype, dev, cur_dtype)
    if target_dtype != cur_dtype:
        model = model.to(dtype=target_dtype)
    model = model.to(dev).eval()
    return model


def compile_for_inference(model: MenaLAM, fast: str = "gather,proj,mix", mode: str | None = None) -> None:
    """Inference only (frozen weights, create_graph=False): fast ops + torch.compile(dynamic=True) of each stage."""
    par = next(model.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    if fast:
        model.core.set_fast_ops(fast)
    for name in _CONST_FN:
        _const(name, torch.zeros(1, device=par.device, dtype=par.dtype))
    for st in model.core.stages_list:
        eager = st.__dict__.get("_eager_forward_sparse", st.forward_sparse)
        st._eager_forward_sparse = eager
        st._compiled_forward_sparse = torch.compile(eager, dynamic=True, mode=mode)
        st.forward_sparse = st._compiled_forward_sparse


def use_compiled_stages(model: MenaLAM, on: bool) -> None:
    """Switch the stages of a compile_for_inference model between their compiled and eager forwards (no recompilation)."""
    for st in model.core.stages_list:
        fn = st.__dict__.get("_compiled_forward_sparse" if on else "_eager_forward_sparse")
        if fn is not None:
            st.forward_sparse = fn


def enable_low_memory(model: MenaLAM) -> None:
    """Inference only: recompute each stage's activations in the backward pass (torch.utils.checkpoint)."""
    from torch.utils.checkpoint import checkpoint

    for stage in model.core.stages_list:
        fn = stage.forward_sparse
        stage.forward_sparse = (lambda f: (lambda s, v, g, lot_bias=None: checkpoint(f, s, v, g, lot_bias, use_reentrant=False)))(fn)


class MenaLAMCalculator(Calculator):
    """Flexible, hardware-optimized ASE Calculator for MenaLAM across CUDA, MPS, and CPU.

    Supports both periodic crystals and open molecules/clusters, direct checkpoint loading (`model='path/to/ckpt.pt'`
    or `MenaLAMCalculator.from_checkpoint(...)`), automatic device selection (`device='auto'`), automatic domain and
    MP2020 Hubbard `+U` Level-of-Theory routing (`lot=None` or `lot='auto'`), per-structure `atoms.info['lot']` overrides,
    optional Verlet neighbor-list skin caching (`skin > 0`), and batched evaluation (`eval_batch`).
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "dipole", "charges"]

    def __init__(
        self,
        model: MenaLAM | str | Path,
        device: str | torch.device | None = None,
        lot: str | int | None = None,
        u_flag: int | None = None,
        add_d3: bool = True,
        unwrap_dipole: bool = False,
        mesh_sparse_min_atoms: int = 512,
        open_coulomb: bool = True,
        open_window: bool = False,
        *,
        weights: str = "ema",
        dtype: str | torch.dtype | None = None,
        freeze: bool | None = None,
        fast_ops: str | Sequence[str] | None = "auto",
        compile: bool | str = False,
        compile_mode: str | None = None,
        low_memory: bool = False,
        skin: float = 0.0,
        compute_stress: bool = True,
        compute_forces: bool = True,
        zero_net_force: bool = False,
        compile_check: bool = True,
        repulsion: str = "nlh_c2_shift",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        loaded_from_path = isinstance(model, (str, Path))
        if loaded_from_path:
            self.device = resolve_inference_device(device)
            self.model = load_mena_lam_checkpoint(
                model, device=self.device, weights=weights, dtype="float32" if dtype is None else dtype
            )
        else:
            if device is None:
                par = next(model.parameters(), None)
                self.device = par.device if par is not None else torch.device("cpu")
            else:
                self.device = resolve_inference_device(device)
            cur_dtype = next(model.parameters()).dtype
            target_dtype = _resolve_torch_dtype(dtype, self.device, cur_dtype, stacklevel=3)
            if target_dtype != cur_dtype:
                model = model.to(dtype=target_dtype)
            self.model = model.to(self.device).eval()

        rep_norm = str(repulsion).lower().strip()
        if rep_norm != "model":
            self.model.core.set_repulsion_mode(rep_norm)
        self.repulsion = getattr(self.model.core, "repulsion_mode", "zbl" if self.model.core.zbl_core else "none")

        # "auto": compile on GPUs (MPS measured 2-3.5x faster per call with a skin; ~5-20 s once per process), not on the
        # CPU (1.3-1.7x for ~30 s of compilation).
        do_compile = (
            (self.device.type in ("cuda", "mps") and not low_memory)
            if (isinstance(compile, str) and compile.lower() == "auto")
            else bool(compile and not low_memory)
        )
        do_freeze = (loaded_from_path or do_compile) if freeze is None else bool(freeze)
        par = next(self.model.parameters())
        if do_freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            if hasattr(self.model, "bake_for_inference"):
                self.model.bake_for_inference()
        if loaded_from_path or do_compile or self.device.type in ("cuda", "mps"):
            for name in _CONST_FN:
                _const(name, torch.zeros(1, device=par.device, dtype=par.dtype))
        if fast_ops == "auto":
            if not self.model.core.fast_ops:
                if self.device.type == "cuda" or do_compile or (self.device.type == "cpu" and do_freeze):
                    self.model.core.set_fast_ops("gather,proj,radial,mix")
                elif self.device.type == "mps":
                    self.model.core.set_fast_ops("gather,proj,radial,mix" if do_freeze else "gather")
        elif fast_ops is not None:
            self.model.core.set_fast_ops(fast_ops)

        self._compiled = False
        self.compile_check = bool(compile_check)
        self.compile_status = "off"
        if do_compile:
            active_fast = ",".join(sorted(self.model.core.fast_ops))
            compile_for_inference(self.model, fast=active_fast, mode=compile_mode)
            self._compiled = True
            self.compile_status = "unchecked" if self.compile_check else "on"
        if low_memory:
            enable_low_memory(self.model)

        self._u_flag_override = int(u_flag) if u_flag is not None else None
        self.auto_lot = False
        self.lot_mode = "explicit"
        if lot is None or (isinstance(lot, str) and lot.lower() == "auto"):
            self.auto_lot = True
            self.lot_mode = "auto"
            default_lot = "omol_wb97mv" if self.model.core.num_lots == 0 else "mp_pbe"
            self.lot_name = default_lot
            self.lot_id = LOT_NAME_TO_ID[default_lot]
        elif isinstance(lot, str) and lot.lower() in ("mp_auto", "mp"):
            self.auto_lot = True
            self.lot_mode = "mp_auto"
            self.lot_name = "mp_pbe"
            self.lot_id = LOT_NAME_TO_ID["mp_pbe"]
        elif isinstance(lot, str) and lot.lower() in ("omat_auto", "omat24"):
            self.auto_lot = True
            self.lot_mode = "omat_auto"
            self.lot_name = "omat_pbe"
            self.lot_id = LOT_NAME_TO_ID["omat_pbe"]
        elif isinstance(lot, str):
            self.lot_name = lot
            self.lot_id = LOT_NAME_TO_ID[lot]
        else:
            self.lot_id = int(lot)
            self.lot_name = LOT_ID_TO_NAME[self.lot_id]

        self._ev_to_kcal_by_lot: dict[int, float] = {}
        for lid in LOT_ID_TO_NAME:
            try:
                self._ev_to_kcal_by_lot[lid] = float(get_lot_entry(lid).get("ev_to_kcal", EV))
            except (KeyError, FileNotFoundError, ValueError, ImportError, Exception):
                self._ev_to_kcal_by_lot[lid] = float(EV)
        self.ev_to_kcal = self._ev_to_kcal_by_lot.get(self.lot_id, float(EV))

        if self._u_flag_override is not None:
            self.u_flag = self._u_flag_override
        else:
            self.u_flag = 1 if self.lot_id in (LOT_NAME_TO_ID["mp_pbeU"], LOT_NAME_TO_ID["omat_pbeU"]) else 0

        self.add_d3 = bool(add_d3)
        self.unwrap_dipole = bool(unwrap_dipole)
        self.mesh_sparse_min_atoms = int(mesh_sparse_min_atoms)
        self.open_coulomb = bool(open_coulomb)
        self.open_window = bool(open_window)
        self.skin = max(0.0, float(skin))
        self.compute_stress = bool(compute_stress)
        self.compute_forces = bool(compute_forces)
        self.zero_net_force = bool(zero_net_force)

        self._nl_cache: dict[float, object] = {}
        self._verlet_cache: dict[str, object] = {}
        self._static_cache: dict[str, object] = {}
        self._last_cell_t: Tensor | None = None
        self._last_edge_shift_cart: Tensor | None = None
        self._last_info_state: tuple | None = None
        self.n_neighbor_builds = 0
        self.n_neighbor_hits = 0

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        device: str | torch.device | None = "auto",
        weights: str = "ema",
        dtype: str | torch.dtype | None = "float32",
        **kwargs,
    ) -> "MenaLAMCalculator":
        """Construct a `MenaLAMCalculator` directly from a checkpoint file path."""
        return cls(model=ckpt_path, device=device, weights=weights, dtype=dtype, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._verlet_cache.clear()
        self._static_cache.clear()
        self._last_cell_t = None
        self._last_edge_shift_cart = None
        self._last_info_state = None

    def reset_cache(self) -> None:
        """Clear any cached Verlet neighbor lists and static composition tensors."""
        self._verlet_cache.clear()
        self._static_cache.clear()
        self._last_cell_t = None
        self._last_edge_shift_cart = None
        self._last_info_state = None

    @staticmethod
    def _extract_info_state(atoms) -> tuple:
        if atoms is None:
            return ()
        info = getattr(atoms, "info", None)
        arrays = getattr(atoms, "arrays", None)
        ac_arr = arrays.get("atom_charge") if arrays else None
        if not info and ac_arr is None:
            return ()
        info = info or {}
        ac = info.get("atom_charge", ac_arr)
        ac_bytes = np.asarray(ac, dtype=np.float64).tobytes() if ac is not None else None
        return (
            info.get("lot"),
            info.get("lot_id"),
            info.get("u_flag"),
            info.get("charge"),
            info.get("spin"),
            ac_bytes,
        )

    def check_state(self, atoms, tol: float = 1e-15) -> list[str]:
        system_changes = list(super().check_state(atoms, tol=tol))
        if "info" not in system_changes and self._last_info_state != self._extract_info_state(atoms):
            system_changes.append("info")
        return system_changes

    def summary(self) -> dict[str, object]:
        """Return a serializable configuration summary for diagnostics and frontend interfaces."""
        par = next(self.model.parameters())
        return {
            "device": str(self.device),
            "dtype": str(par.dtype).replace("torch.", ""),
            "lot_mode": self.lot_mode,
            "lot_name": self.lot_name,
            "lot_id": self.lot_id,
            "u_flag": self.u_flag,
            "num_lots": int(self.model.core.num_lots),
            "stages": len(self.model.core.stages_list),
            "crystal_r_cut": float(self.model.crystal_r_cut),
            "mol_r_cut": float(self.model.mol_r_cut),
            "add_d3": self.add_d3,
            "skin": self.skin,
            "zero_net_force": self.zero_net_force,
            "repulsion": self.repulsion,
            "fast_ops": sorted(self.model.core.fast_ops),
            "compiled": self._compiled,
            "compile_status": self.compile_status,
        }

    def _resolve_lot_and_u(
        self, z_np: np.ndarray, pbc_any: bool, atoms_info: dict | None = None
    ) -> tuple[int, int, float]:
        """Resolve `(lot_id, u_flag, ev_to_kcal)` for a structure, honoring per-structure `atoms.info` overrides
        and automatic domain / MP2020 `+U` routing when `self.auto_lot` is enabled."""
        info = atoms_info or {}
        info_lot = info.get("lot", info.get("lot_id"))
        info_u = info.get("u_flag")

        if info_lot is not None:
            if isinstance(info_lot, str):
                lid = LOT_NAME_TO_ID[info_lot]
            else:
                lid = int(info_lot)
            if info_u is not None:
                u_val = int(info_u)
            elif self._u_flag_override is not None:
                u_val = self._u_flag_override
            else:
                u_val = 1 if lid in (LOT_NAME_TO_ID["mp_pbeU"], LOT_NAME_TO_ID["omat_pbeU"]) else 0
            return lid, u_val, self._ev_to_kcal_by_lot.get(lid, float(EV))

        if self.auto_lot:
            if self.model.core.num_lots == 0:
                lid = LOT_NAME_TO_ID["omol_wb97mv"]
                u_val = 0 if info_u is None else int(info_u)
            elif self.lot_mode == "auto" and not pbc_any:
                lid = LOT_NAME_TO_ID["omol_wb97mv"]
                u_val = 0 if info_u is None else int(info_u)
            else:
                if info_u is not None:
                    u_val = int(info_u)
                elif self._u_flag_override is not None:
                    u_val = self._u_flag_override
                else:
                    u_val = _compute_mp_u_flag(z_np)
                if self.lot_mode == "omat_auto":
                    lid = LOT_NAME_TO_ID["omat_pbeU"] if u_val == 1 else LOT_NAME_TO_ID["omat_pbe"]
                else:
                    lid = LOT_NAME_TO_ID["mp_pbeU"] if u_val == 1 else LOT_NAME_TO_ID["mp_pbe"]
            return lid, u_val, self._ev_to_kcal_by_lot.get(lid, float(EV))

        lid = self.lot_id
        u_val = int(info_u) if info_u is not None else self.u_flag
        return lid, u_val, self.ev_to_kcal

    @property
    def _compile_enabled(self) -> bool:
        return self._compiled

    @_compile_enabled.setter
    def _compile_enabled(self, val: bool) -> None:
        self._compiled = bool(val)

    @staticmethod
    def _effective_periodic_cell(
        pos_np: np.ndarray,
        cell_np: np.ndarray,
        pbc: np.ndarray,
        max_cut: float,
    ) -> np.ndarray:
        """Return a non-singular 3x3 cell for mixed PBC (2D slabs / 1D wires) by replacing `pbc == False` rows
        with orthogonal normal vectors padded beyond `span + 2 * max_cut + 20.0` A so `vesin` and `MeshGrid.for_cell`
        never fail on zero/singular rows or wrap across small vacuum boxes."""
        p = np.asarray(pbc, dtype=bool).reshape(-1)
        if p.size != 3:
            p = np.broadcast_to(p, (3,)).copy()
        if not p.any():
            return np.zeros((3, 3), dtype=np.float64)
        c = np.array(cell_np, dtype=np.float64, copy=True).reshape(3, 3)
        if p.all():
            return c
        act = np.flatnonzero(p)
        inact = np.flatnonzero(~p)
        pts = np.asarray(pos_np, dtype=np.float64).reshape(-1, 3)
        pad_extra = 2.0 * max(0.0, float(max_cut)) + 20.0
        if len(act) == 2:
            ax = int(inact[0])
            n_vec = np.cross(c[act[0]], c[act[1]])
            n_norm = float(np.linalg.norm(n_vec))
            if n_norm > 1e-8:
                u_hat = n_vec / n_norm
            else:
                u_hat = np.zeros(3, dtype=np.float64)
                u_hat[ax] = 1.0
            span = float((pts @ u_hat).max() - (pts @ u_hat).min()) if len(pts) > 1 else 0.0
            c[ax] = u_hat * (span + pad_extra)
            if np.linalg.det(c) < 0.0:
                c[ax] = -c[ax]
        else:
            a0 = int(act[0])
            v0 = c[a0]
            v_norm = float(np.linalg.norm(v0))
            if v_norm > 1e-8:
                t0 = v0 / v_norm
                k = int(np.argmin(np.abs(t0)))
                ek = np.zeros(3, dtype=np.float64)
                ek[k] = 1.0
                u1 = np.cross(t0, ek)
                u1 = u1 / np.linalg.norm(u1)
                u2 = np.cross(t0, u1)
                u2 = u2 / np.linalg.norm(u2)
            else:
                u1 = np.zeros(3, dtype=np.float64)
                u1[int(inact[0])] = 1.0
                u2 = np.zeros(3, dtype=np.float64)
                u2[int(inact[1])] = 1.0
            for ax_i, u_i in zip(inact, (u1, u2)):
                span_i = float((pts @ u_i).max() - (pts @ u_i).min()) if len(pts) > 1 else 0.0
                c[int(ax_i)] = u_i * (span_i + pad_extra)
            if np.linalg.det(c) < 0.0:
                c[int(inact[-1])] = -c[int(inact[-1])]
        return c

    def _compute_neighbors(
        self,
        pos_np: np.ndarray,
        cell_np: np.ndarray,
        pbc: np.ndarray,
        cutoff: float,
        return_dists: bool = False,
    ) -> tuple[np.ndarray, ...]:
        import vesin

        rc = float(cutoff)
        nl = self._nl_cache.get(rc)
        if nl is None:
            nl = vesin.NeighborList(cutoff=rc, full_list=True)
            self._nl_cache[rc] = nl
        p_arr = np.asarray(pbc, dtype=bool).reshape(-1)
        if p_arr.size != 3:
            p_arr = np.broadcast_to(p_arr, (3,)).copy()
        p = (bool(p_arr[0]), bool(p_arr[1]), bool(p_arr[2]))
        box = self._effective_periodic_cell(pos_np, cell_np, p_arr, rc) if any(p) else np.zeros((3, 3))
        if return_dists:
            i, j, d, S = nl.compute(points=np.asarray(pos_np, np.float64), box=box, periodic=p, quantities="ijdS")
            i_a = np.asarray(i, np.int64)
            j_a = np.asarray(j, np.int64)
            S_a = np.asarray(S, np.int64)
            d_a = np.asarray(d, np.float64)
            if any(p) and not all(p) and len(i_a) > 0:
                keep = np.all(S_a[:, ~p_arr] == 0, axis=1)
                if not np.all(keep):
                    i_a, j_a, S_a, d_a = i_a[keep], j_a[keep], S_a[keep], d_a[keep]
            return i_a, j_a, S_a, d_a
        i, j, S = nl.compute(points=np.asarray(pos_np, np.float64), box=box, periodic=p, quantities="ijS")
        i_a = np.asarray(i, np.int64)
        j_a = np.asarray(j, np.int64)
        S_a = np.asarray(S, np.int64)
        if any(p) and not all(p) and len(i_a) > 0:
            keep = np.all(S_a[:, ~p_arr] == 0, axis=1)
            if not np.all(keep):
                i_a, j_a, S_a = i_a[keep], j_a[keep], S_a[keep]
        return i_a, j_a, S_a

    @staticmethod
    def _partition_verlet_edges(
        i: np.ndarray,
        j: np.ndarray,
        S: np.ndarray,
        d: np.ndarray,
        rc: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        """Partition a Verlet master list into base edges (`d <= rc`, in original `vesin` order) followed by
        skin edges (`d > rc`) sorted ascending by initial distance `d` so that at any displacement `dr_max`,
        only the contiguous prefix `[:n_base + searchsorted(skin_d0, rc + 2*dr_max)]` can have `d_cur <= rc`."""
        base_mask = d <= rc
        n_base = int(base_mask.sum())
        if n_base == len(d):
            return i, j, S, n_base, np.empty(0, dtype=np.float64)
        skin_idx = np.flatnonzero(~base_mask)
        skin_order = skin_idx[np.argsort(d[skin_idx], kind="mergesort")]
        order = np.concatenate([np.flatnonzero(base_mask), skin_order])
        return i[order], j[order], S[order], n_base, d[skin_order]

    def _empty_device_cache(self) -> None:
        if self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _bucket_active(n_base: int, n_needed: int, n_skin: int) -> int:
        """Active edge count for a Verlet hit: the base edges plus the needed skin prefix, rounded up to steps of ~3% of
        the base list. The extra skin edges lie beyond the cutoff and weigh exactly zero, so results are unchanged;
        the edge count then changes a few times between rebuilds instead of on every call. On MPS a new edge count
        per call stops the caching allocator from reusing blocks: Cu 4000 atoms grew 6.3 -> 15.3 GB in five calls."""
        if n_needed <= 0:
            return n_base
        step = max(1024, n_base // 32)
        return n_base + min(n_skin, -(-n_needed // step) * step)

    def _get_static_tensors(
        self,
        z_np: np.ndarray,
        lot_id: int,
        u_val: int,
        model_dtype: torch.dtype,
        run_elec: bool = False,
        pbc_any: bool = True,
        q_val: float = 0.0,
        spin_val: int = 1,
    ) -> dict[str, object]:
        """Build or reuse cached device composition tensors and float64 reference energy (kcal/mol)."""
        key = (
            z_np.tobytes(),
            int(lot_id),
            int(u_val),
            model_dtype,
            self.device,
            bool(run_elec),
            bool(pbc_any),
            float(q_val),
            int(spin_val),
            self.model.crystal_e0._version,
            self.model.mu._version,
            self.model.omol_e0._version,
            self.model.omol_e_q._version,
            self.model.omol_e_s._version,
            bool(self.model.use_charge_self_energy),
            bool(self.model._charge_self_energy_active),
        )
        sc = self._static_cache
        if sc.get("key") == key:
            return sc
        N = len(z_np)
        z_cpu = torch.from_numpy(z_np)
        batch_cpu = torch.zeros(N, dtype=torch.long)
        u_cpu = torch.tensor([u_val], dtype=torch.long)
        z_t = z_cpu.to(self.device)
        batch_t = batch_cpu.to(self.device)
        u_t = u_cpu.to(self.device)
        q_t = torch.tensor([q_val], dtype=model_dtype, device=self.device)
        spin_t = torch.tensor([spin_val], dtype=torch.long, device=self.device)
        z_2d = z_t.unsqueeze(0)
        mask_2d = torch.ones((1, N), dtype=torch.bool, device=self.device)
        if pbc_any and not run_elec:
            e_ref_t = self.model.crystal_reference(z_cpu, batch_cpu, u_cpu, 1, lot_id=lot_id)
        else:
            q_self_added = bool(
                run_elec and self.model._has_active_charge_self_energy() and self.model.core.charge_head is not None
            )
            e_ref_t = self.model.molecule_reference_sparse(
                z_cpu,
                batch_cpu,
                1,
                total_charge=torch.tensor([q_val], dtype=torch.float64),
                spin=torch.tensor([spin_val], dtype=torch.long),
                charges=None,
                include_charge_lookup=not q_self_added,
            )
            if not pbc_any:
                ref_mask = self.model._ref_mask(1, torch.device("cpu"), lot_id, None)
                e_ref_t = self.model._token_reference(e_ref_t, ref_mask, z_cpu, batch_cpu, 1, lot_id)
        e_ref_kcal = float(e_ref_t.detach().cpu().double()[0].item())
        sc.clear()
        sc.update(
            {
                "key": key,
                "z": z_t,
                "z_2d": z_2d,
                "mask_2d": mask_2d,
                "batch": batch_t,
                "u": u_t,
                "q_tot": q_t,
                "spin": spin_t,
                "e_ref_kcal": e_ref_kcal,
            }
        )
        return sc

    def _get_graph_tensors(
        self,
        z_np: np.ndarray,
        pos_np: np.ndarray,
        cell_np: np.ndarray,
        pbc: np.ndarray,
        rc: float,
        lr_cut: float | None = None,
        model_dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Build or reuse `(edge_index, edge_shift_frac, lr_edge_index, lr_edge_shift_frac)` on `self.device`."""
        if model_dtype is None:
            model_dtype = self.model.core.readout[0].weight.dtype
        if self.skin > 0.0:
            vc = self._verlet_cache
            if (
                vc.get("rc") == rc
                and vc.get("lr_cut") == lr_cut
                and vc.get("n") == len(z_np)
                and vc.get("dtype") == model_dtype
                and np.array_equal(vc.get("z_np"), z_np)
                and np.array_equal(vc.get("pbc"), pbc)
                and np.array_equal(vc.get("cell_np"), cell_np)
            ):
                dr_max = float(np.linalg.norm(pos_np - vc["ref_pos"], axis=1).max()) if len(pos_np) > 0 else 0.0
                if dr_max <= 0.5 * self.skin:
                    self.n_neighbor_hits += 1
                    self._last_cell_t = vc.get("cell_t")
                    skin_d0 = vc["skin_d0"]
                    n_act = self._bucket_active(
                        vc["n_base"],
                        int(np.searchsorted(skin_d0, rc + 2.0 * dr_max + 1e-9, side="right")) if len(skin_d0) else 0,
                        len(skin_d0),
                    )
                    if n_act != vc.get("n_act_last", vc["n_base"]):
                        vc["n_act_last"] = n_act
                        if len(z_np) >= 1024:  # blocks cached for the old edge count would otherwise stay reserved
                            self._empty_device_cache()
                    esc = vc.get("edge_shift_cart")
                    self._last_edge_shift_cart = esc[:n_act] if esc is not None else None
                    lr_ei = vc.get("lr_edge_index")
                    lr_esf = vc.get("lr_edge_shift_frac")
                    if lr_ei is not None and lr_cut is not None:
                        lr_skin_d0 = vc["lr_skin_d0"]
                        lr_n_act = self._bucket_active(
                            vc["lr_n_base"],
                            int(np.searchsorted(lr_skin_d0, lr_cut + 2.0 * dr_max + 1e-9, side="right"))
                            if len(lr_skin_d0)
                            else 0,
                            len(lr_skin_d0),
                        )
                        lr_ei = lr_ei[:, :lr_n_act]
                        lr_esf = lr_esf[:lr_n_act]
                    return vc["edge_index"][:, :n_act], vc["edge_shift_frac"][:n_act], lr_ei, lr_esf
            if vc:
                vc.clear()
                if len(z_np) >= 1024:
                    self._empty_device_cache()

        self.n_neighbor_builds += 1
        eff_rc = rc + self.skin
        if self.skin > 0.0:
            i, j, S, d = self._compute_neighbors(pos_np, cell_np, pbc, eff_rc, return_dists=True)
            i, j, S, n_base, skin_d0 = self._partition_verlet_edges(i, j, S, d, rc)
        else:
            i, j, S = self._compute_neighbors(pos_np, cell_np, pbc, eff_rc, return_dists=False)
            n_base = len(i)
            skin_d0 = np.empty(0, dtype=np.float64)
        edge_index = torch.stack([torch.from_numpy(i), torch.from_numpy(j)]).to(self.device)
        edge_shift_frac = torch.from_numpy(S).to(self.device)
        pbc_any = bool(np.any(pbc))
        if pbc_any:
            eff_cell_np = (
                cell_np
                if bool(np.all(pbc))
                else self._effective_periodic_cell(pos_np, cell_np, pbc, max(rc, lr_cut or 0.0) + self.skin)
            )
            cell_t = torch.as_tensor(eff_cell_np, dtype=model_dtype, device=self.device).unsqueeze(0)
        else:
            cell_t = None
        edge_shift_cart = (
            (edge_shift_frac.to(model_dtype) @ cell_t[0]) if (cell_t is not None and lr_cut is None) else None
        )
        self._last_cell_t = cell_t
        self._last_edge_shift_cart = edge_shift_cart[:n_base] if edge_shift_cart is not None else None
        lr_edge_index = None
        lr_edge_shift_frac = None
        lr_n_base = 0
        lr_skin_d0 = np.empty(0, dtype=np.float64)
        if lr_cut is not None:
            if self.skin > 0.0:
                li, lj, lS, ld = self._compute_neighbors(pos_np, cell_np, pbc, lr_cut + self.skin, return_dists=True)
                li, lj, lS, lr_n_base, lr_skin_d0 = self._partition_verlet_edges(li, lj, lS, ld, lr_cut)
            else:
                li, lj, lS = self._compute_neighbors(pos_np, cell_np, pbc, lr_cut + self.skin, return_dists=False)
                lr_n_base = len(li)
            lr_edge_index = torch.stack([torch.from_numpy(li), torch.from_numpy(lj)]).to(self.device)
            lr_edge_shift_frac = torch.from_numpy(lS).to(self.device)

        if self.skin > 0.0:
            self._verlet_cache = {
                "rc": rc,
                "lr_cut": lr_cut,
                "n": len(z_np),
                "dtype": model_dtype,
                "z_np": z_np.copy(),
                "pbc": pbc.copy(),
                "cell_np": cell_np.copy(),
                "ref_pos": pos_np.copy(),
                "edge_index": edge_index,
                "edge_shift_frac": edge_shift_frac,
                "cell_t": cell_t,
                "edge_shift_cart": edge_shift_cart,
                "n_base": n_base,
                "skin_d0": skin_d0,
                "lr_edge_index": lr_edge_index,
                "lr_edge_shift_frac": lr_edge_shift_frac,
                "lr_n_base": lr_n_base,
                "lr_skin_d0": lr_skin_d0,
            }
            return (
                edge_index[:, :n_base],
                edge_shift_frac[:n_base],
                lr_edge_index[:, :lr_n_base] if lr_edge_index is not None else None,
                lr_edge_shift_frac[:lr_n_base] if lr_edge_shift_frac is not None else None,
            )
        return edge_index, edge_shift_frac, lr_edge_index, lr_edge_shift_frac

    # first compiled call vs eager: a miscompile is orders of magnitude above these (torch 2.13 MPS gave 3.6 eV/atom
    # when the whole core was compiled); float32 compiled vs eager agree to ~1e-4 meV/atom and ~1e-5 eV/A
    COMPILE_CHECK_E_EV_PER_ATOM = 1e-3
    COMPILE_CHECK_F_EV_PER_A = 1e-2

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ) -> dict:
        if self.compile_status != "unchecked":
            if not self._compile_enabled:
                return self._calculate(atoms, properties, system_changes)
            try:
                return self._calculate(atoms, properties, system_changes)
            except Exception as exc:
                import warnings

                self._fallback_to_eager(
                    f"runtime compilation error ({type(exc).__name__}: {str(exc).splitlines()[0][:200]})"
                )
                warnings.warn(
                    f"MenaLAMCalculator: {self.compile_status}; running without compilation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._calculate(atoms, properties, system_changes)
        import warnings

        try:
            res = dict(self._calculate(atoms, properties, system_changes))
        except Exception as exc:  # compilation failed on this machine / torch build: run eager instead
            self._fallback_to_eager(f"compilation failed ({type(exc).__name__}: {str(exc).splitlines()[0][:200]})")
            warnings.warn(f"MenaLAMCalculator: {self.compile_status}; running without compilation.", RuntimeWarning)
            return self._calculate(atoms, properties, system_changes)
        use_compiled_stages(self.model, False)
        try:
            ref = self._calculate(atoms, properties, system_changes)
        finally:
            use_compiled_stages(self.model, True)
        n = max(len(self.atoms), 1)
        de = abs(res["energy"] - ref["energy"]) / n
        df = float(np.abs(np.asarray(res["forces"]) - np.asarray(ref["forces"])).max()) if n else 0.0
        if de > self.COMPILE_CHECK_E_EV_PER_ATOM or df > self.COMPILE_CHECK_F_EV_PER_A:
            self._fallback_to_eager(f"compiled result differs from eager (dE {de:.2e} eV/atom, max dF {df:.2e} eV/A)")
            warnings.warn(f"MenaLAMCalculator: {self.compile_status}; running without compilation.", RuntimeWarning)
            self.results = ref
            return ref
        self.compile_status = "on"
        self.results = res
        return res

    def _fallback_to_eager(self, why: str) -> None:
        use_compiled_stages(self.model, False)
        self._compile_enabled = False
        self.compile_status = f"fell back to eager: {why}"
        self._verlet_cache.clear()

    def _calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ) -> dict:
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        self._last_info_state = self._extract_info_state(atoms)
        model_dtype = self.model.core.readout[0].weight.dtype
        z_np = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        pos_np = np.asarray(atoms.get_positions(), dtype=np.float64)
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        pbc_any = bool(pbc.any())

        lot_id, u_val, ev_scale = self._resolve_lot_and_u(z_np, pbc_any, atoms.info)
        run_elec = self.model.core._should_run_elec(pbc_any, lot_id)
        q_val = float(atoms.info.get("charge", 0.0)) if (run_elec or not pbc_any) else 0.0
        spin_val = int(atoms.info.get("spin", 1)) if (run_elec or not pbc_any) else 1
        st = self._get_static_tensors(
            z_np,
            lot_id,
            u_val,
            model_dtype,
            run_elec=run_elec,
            pbc_any=pbc_any,
            q_val=q_val,
            spin_val=spin_val,
        )
        z = st["z"]
        pos = torch.as_tensor(pos_np, dtype=model_dtype, device=self.device)
        raw_ac = atoms.info.get("atom_charge", atoms.arrays.get("atom_charge"))
        ac_t = (
            torch.as_tensor(raw_ac, dtype=model_dtype, device=self.device)
            if raw_ac is not None
            else None
        )
        if pbc_any:
            cell_np = np.asarray(atoms.get_cell(), dtype=np.float64)
            can_stress = bool(pbc.all()) and abs(float(np.linalg.det(cell_np))) > 1e-6
            do_stress = self.compute_stress and can_stress
            batch = st["batch"]
            rc = self.model.mol_r_cut if run_elec else self.model.crystal_r_cut
            if run_elec:
                if self.model.core.mesh_def:  # long range on the mesh: the only long list left is D3's (12 A)
                    lr_cut = float(self.model.d3.r_cut) if (self.add_d3 and self.model.d3 is not None) else rc
                else:
                    lr_cut = max(
                        float(self.model.core.ic_r_cut),
                        float(getattr(self.model.core.taper_elec, "r_cut", 18.0)),
                        18.0,
                    )
                edge_index, edge_shift_frac, lr_edge_index, lr_edge_shift_frac = self._get_graph_tensors(
                    z_np, pos_np, cell_np, pbc, rc, lr_cut=lr_cut, model_dtype=model_dtype
                )
                cell = self._last_cell_t
                out = self.model.forward_crystal(
                    z,
                    pos,
                    cell,
                    batch,
                    edge_index,
                    edge_shift_frac,
                    lot_id=lot_id,
                    total_charge=st["q_tot"],
                    spin=st["spin"],
                    atom_charge=ac_t,
                    lr_edge_index=lr_edge_index,
                    lr_edge_shift_frac=lr_edge_shift_frac,
                    add_d3=self.add_d3,
                    add_ref=False,
                    compute_forces=self.compute_forces,
                    compute_stress=do_stress,
                    create_graph=False,
                    unwrap_dipole=self.unwrap_dipole,
                )
            else:
                edge_index, edge_shift_frac, _, _ = self._get_graph_tensors(
                    z_np, pos_np, cell_np, pbc, rc, lr_cut=None, model_dtype=model_dtype
                )
                cell = self._last_cell_t
                out = self.model.forward_crystal(
                    z,
                    pos,
                    cell,
                    batch,
                    edge_index,
                    edge_shift_frac,
                    edge_shift_cart=self._last_edge_shift_cart,
                    u=st["u"],
                    lot_id=lot_id,
                    compute_forces=self.compute_forces,
                    compute_stress=do_stress,
                    create_graph=False,
                    virial_mode="edge",
                )
            e_ev = float((out["energy"][0].detach().cpu().double().item() + st["e_ref_kcal"]) / ev_scale)
            f_ev = (
                (out["forces"].detach().cpu().numpy() / ev_scale).astype(np.float64)
                if out.get("forces") is not None
                else np.zeros((len(z_np), 3), dtype=np.float64)
            )
            if self.zero_net_force and len(f_ev) > 0:
                f_ev = f_ev - f_ev.mean(axis=0, keepdims=True)
            self.results = {
                "energy": e_ev,
                "free_energy": e_ev,
                "forces": f_ev,
                "charges": out["charges"].detach().cpu().numpy() if out.get("charges") is not None else None,
            }
            if can_stress:
                if out.get("stress") is not None:
                    s_3x3 = (out["stress"][0].detach().cpu().numpy() / ev_scale).astype(np.float64)
                    s_voigt = np.array(
                        [s_3x3[0, 0], s_3x3[1, 1], s_3x3[2, 2], s_3x3[1, 2], s_3x3[0, 2], s_3x3[0, 1]],
                        dtype=np.float64,
                    )
                else:
                    s_voigt = np.zeros(6, dtype=np.float64)
                self.results["stress"] = s_voigt
            if self.unwrap_dipole and out.get("dipole") is not None:
                self.results["dipole"] = out["dipole"][0].detach().cpu().numpy()
            return self.results

        q_tot = st["q_tot"]
        spin = st["spin"]
        if self.model.core.mesh_def and len(z_np) >= self.mesh_sparse_min_atoms:
            # large open systems: 6 A edges + mesh long range, O(N log N); the only long list is D3's (12 A)
            no_pbc = np.zeros(3, dtype=bool)
            zero_cell = np.zeros((3, 3), dtype=np.float64)
            rc = self.model.mol_r_cut
            lr_cut = float(self.model.d3.r_cut) if (self.add_d3 and self.model.d3 is not None) else None
            edge_index, _, lr_edge_index, _ = self._get_graph_tensors(
                z_np, pos_np, zero_cell, no_pbc, rc, lr_cut=lr_cut, model_dtype=model_dtype
            )
            out = self.model.forward_sparse(
                z,
                pos,
                st["batch"],
                1,
                edge_index,
                pos.new_zeros((edge_index.shape[1], 3)),
                pbc=False,
                lot_id=lot_id,
                total_charge=q_tot,
                spin=spin,
                atom_charge=ac_t,
                lr_edge_index=lr_edge_index,
                add_d3=self.add_d3,
                add_ref=False,
                compute_gradient_forces=self.compute_forces,
                unwrap_dipole=self.unwrap_dipole,
            )
            f_ev = (
                (out["forces"].detach().cpu().numpy() / ev_scale).astype(np.float64)
                if out.get("forces") is not None
                else np.zeros((len(z_np), 3), dtype=np.float64)
            )
            if self.zero_net_force and len(f_ev) > 0:
                f_ev = f_ev - f_ev.mean(axis=0, keepdims=True)
            e_ev = float((out["energy"][0].detach().cpu().double().item() + st["e_ref_kcal"]) / ev_scale)
            self.results = {
                "energy": e_ev,
                "free_energy": e_ev,
                "forces": f_ev,
                "charges": out["charges"].detach().cpu().numpy() if out.get("charges") is not None else None,
                "dipole": out["dipole"][0].detach().cpu().numpy() if out.get("dipole") is not None else None,
            }
            return self.results

        te = getattr(self.model.core, "taper_elec", None)
        saved_elec = (te.r_on, te.r_cut) if te is not None else None
        saved_ic = (self.model.core.ic_r_on, self.model.core.ic_r_cut, list(self.model.core.ic_windows))
        try:
            if run_elec and (self.open_coulomb or self.open_window):
                self.model.enable_open_long_range(coulomb=self.open_coulomb, window=self.open_window)
            out = self.model.forward_molecule(
                st["z_2d"],
                pos.unsqueeze(0),
                st["mask_2d"],
                total_charge=q_tot,
                spin=spin,
                atom_charge=ac_t.unsqueeze(0) if ac_t is not None else None,
                lot_id=lot_id,
                compute_forces=self.compute_forces,
                add_d3=self.add_d3,
                add_ref=False,
            )
        finally:
            if te is not None and saved_elec is not None:
                te.r_on, te.r_cut = saved_elec
            self.model.core.ic_r_on, self.model.core.ic_r_cut, self.model.core.ic_windows = saved_ic
        e_ev = float((out["energy"][0].detach().cpu().double().item() + st["e_ref_kcal"]) / ev_scale)
        f_ev = (
            (out["forces"][0].detach().cpu().numpy() / ev_scale).astype(np.float64)
            if out.get("forces") is not None
            else np.zeros((len(z_np), 3), dtype=np.float64)
        )
        if self.zero_net_force and len(f_ev) > 0:
            f_ev = f_ev - f_ev.mean(axis=0, keepdims=True)
        self.results = {
            "energy": e_ev,
            "free_energy": e_ev,
            "forces": f_ev,
            "charges": out["charges"][0].detach().cpu().numpy() if out.get("charges") is not None else None,
            "dipole": out["dipole"][0].detach().cpu().numpy() if out.get("dipole") is not None else None,
        }
        return self.results

    def eval_graph(
        self,
        atoms,
        edge_index: np.ndarray | Tensor,
        edge_shift_frac: np.ndarray | Tensor | None = None,
        edge_shift_cart: np.ndarray | Tensor | None = None,
        properties: Sequence[str] = ("energy", "forces", "stress"),
    ) -> dict[str, object]:
        """Evaluate energy, forces, and stress on `atoms` using a caller-supplied sparse neighbor graph
        (`edge_index` `[2, E]` and `edge_shift_frac` `[E, 3]` or `edge_shift_cart` `[E, 3]`), bypassing `vesin`."""
        model_dtype = self.model.core.readout[0].weight.dtype
        z_np = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        pos_np = np.asarray(atoms.get_positions(), dtype=np.float64)
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        pbc_any = bool(pbc.any())

        lot_id, u_val, ev_scale = self._resolve_lot_and_u(z_np, pbc_any, atoms.info)
        run_elec = self.model.core._should_run_elec(pbc_any, lot_id)
        q_val = float(atoms.info.get("charge", 0.0)) if (run_elec or not pbc_any) else 0.0
        spin_val = int(atoms.info.get("spin", 1)) if (run_elec or not pbc_any) else 1
        st = self._get_static_tensors(
            z_np,
            lot_id,
            u_val,
            model_dtype,
            run_elec=run_elec,
            pbc_any=pbc_any,
            q_val=q_val,
            spin_val=spin_val,
        )
        pos = torch.as_tensor(pos_np, dtype=model_dtype, device=self.device)
        ei_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        esf_t = (
            torch.as_tensor(edge_shift_frac, dtype=torch.long, device=self.device)
            if edge_shift_frac is not None
            else None
        )
        esc_t = (
            torch.as_tensor(edge_shift_cart, dtype=model_dtype, device=self.device)
            if edge_shift_cart is not None
            else None
        )
        need_forces = ("forces" in properties) and self.compute_forces
        need_stress = ("stress" in properties) and self.compute_stress
        raw_ac = atoms.info.get("atom_charge", atoms.arrays.get("atom_charge"))
        ac_t = (
            torch.as_tensor(raw_ac, dtype=model_dtype, device=self.device)
            if raw_ac is not None
            else None
        )

        if pbc_any:
            cell_np = np.asarray(atoms.get_cell(), dtype=np.float64)
            can_stress = bool(pbc.all()) and abs(float(np.linalg.det(cell_np))) > 1e-6
            eff_cell_np = (
                cell_np
                if bool(pbc.all())
                else self._effective_periodic_cell(pos_np, cell_np, pbc, self.model.crystal_r_cut)
            )
            cell = torch.as_tensor(eff_cell_np, dtype=model_dtype, device=self.device).unsqueeze(0)
            if esc_t is None and esf_t is None:
                esf_t = torch.zeros((ei_t.shape[1], 3), dtype=torch.long, device=self.device)
            out = self.model.forward_crystal(
                st["z"],
                pos,
                cell,
                st["batch"],
                ei_t,
                edge_shift_frac=esf_t,
                edge_shift_cart=esc_t,
                u=st["u"],
                lot_id=lot_id,
                total_charge=st["q_tot"] if run_elec else None,
                spin=st["spin"] if run_elec else None,
                atom_charge=ac_t,
                add_d3=self.add_d3 and run_elec,
                add_ref=False,
                compute_forces=need_forces,
                compute_stress=need_stress and can_stress,
                create_graph=False,
                virial_mode="strain" if run_elec else "edge",
                unwrap_dipole=self.unwrap_dipole,
            )
            e_ev = float((out["energy"][0].detach().cpu().double().item() + st["e_ref_kcal"]) / ev_scale)
            f_ev = (
                (out["forces"].detach().cpu().numpy() / ev_scale).astype(np.float64)
                if out.get("forces") is not None
                else np.zeros((len(z_np), 3), dtype=np.float64)
            )
            if self.zero_net_force and len(f_ev) > 0:
                f_ev = f_ev - f_ev.mean(axis=0, keepdims=True)
            self.results = {
                "energy": e_ev,
                "free_energy": e_ev,
                "forces": f_ev,
                "charges": out["charges"].detach().cpu().numpy() if out.get("charges") is not None else None,
            }
            if can_stress:
                if out.get("stress") is not None:
                    s_3x3 = (out["stress"][0].detach().cpu().numpy() / ev_scale).astype(np.float64)
                    s_voigt = np.array(
                        [s_3x3[0, 0], s_3x3[1, 1], s_3x3[2, 2], s_3x3[1, 2], s_3x3[0, 2], s_3x3[0, 1]],
                        dtype=np.float64,
                    )
                else:
                    s_voigt = np.zeros(6, dtype=np.float64)
                self.results["stress"] = s_voigt
            if self.unwrap_dipole and out.get("dipole") is not None:
                self.results["dipole"] = out["dipole"][0].detach().cpu().numpy()
            return self.results

        if esc_t is None:
            esc_t = pos.new_zeros((ei_t.shape[1], 3))
        out = self.model.forward_sparse(
            st["z"],
            pos,
            st["batch"],
            1,
            ei_t,
            esc_t,
            pbc=False,
            lot_id=lot_id,
            total_charge=st["q_tot"],
            spin=st["spin"],
            atom_charge=ac_t,
            lr_edge_index=ei_t,
            lr_edge_shift=esc_t,
            add_d3=self.add_d3,
            add_ref=False,
            compute_gradient_forces=need_forces,
            unwrap_dipole=self.unwrap_dipole,
        )
        e_ev = float((out["energy"][0].detach().cpu().double().item() + st["e_ref_kcal"]) / ev_scale)
        f_ev = (
            (out["forces"].detach().cpu().numpy() / ev_scale).astype(np.float64)
            if out.get("forces") is not None
            else np.zeros((len(z_np), 3), dtype=np.float64)
        )
        if self.zero_net_force and len(f_ev) > 0:
            f_ev = f_ev - f_ev.mean(axis=0, keepdims=True)
        self.results = {
            "energy": e_ev,
            "free_energy": e_ev,
            "forces": f_ev,
            "charges": out["charges"].detach().cpu().numpy() if out.get("charges") is not None else None,
            "dipole": out["dipole"][0].detach().cpu().numpy() if out.get("dipole") is not None else None,
        }
        return self.results

    def eval_batch(
        self,
        atoms_list: Sequence,
        properties: Sequence[str] = ("energy", "forces", "stress"),
    ) -> list[dict[str, object]]:
        """Evaluate a batch of ASE `Atoms` structures in a single fused sparse forward pass on CUDA, MPS, or CPU
        whenever all structures are non-electrostatic periodic crystals, falling back to per-structure `calculate`
        for mixed/electrostatic batches."""
        if not atoms_list:
            return []
        if self.compile_status == "unchecked":  # the compiled stages are checked against eager once, on a single structure
            self.calculate(atoms=atoms_list[0], properties=properties, system_changes=all_changes)
        need_forces = ("forces" in properties) and self.compute_forces
        need_stress = ("stress" in properties) and self.compute_stress

        meta = []
        can_batch_crystal = True
        for at in atoms_list:
            z_np = np.asarray(at.get_atomic_numbers(), dtype=np.int64)
            pbc = np.asarray(at.get_pbc(), dtype=bool)
            pbc_any = bool(pbc.any())
            lid, u_val, ev_scale = self._resolve_lot_and_u(z_np, pbc_any, at.info)
            run_elec = self.model.core._should_run_elec(pbc_any, lid)
            if not pbc_any or run_elec:
                can_batch_crystal = False
                break
            meta.append((at, z_np, pbc, lid, u_val, ev_scale))

        if not can_batch_crystal:
            out_list: list[dict[str, object]] = []
            for at in atoms_list:
                self.calculate(atoms=at, properties=properties, system_changes=all_changes)
                out_list.append(dict(self.results))
            return out_list

        model_dtype = self.model.core.readout[0].weight.dtype
        rc = self.model.crystal_r_cut
        B = len(meta)
        z_list, pos_list, cell_list, batch_list = [], [], [], []
        ei_list, esf_list, u_list, lot_list, scales, counts, can_stress_list = [], [], [], [], [], [], []
        n_off = 0
        for b, (at, z_np, pbc, lid, u_val, ev_scale) in enumerate(meta):
            pos_np = np.asarray(at.get_positions(), dtype=np.float64)
            cell_np = np.asarray(at.get_cell(), dtype=np.float64)
            can_stress_b = bool(pbc.all()) and abs(float(np.linalg.det(cell_np))) > 1e-6
            eff_cell_np = cell_np if bool(pbc.all()) else self._effective_periodic_cell(pos_np, cell_np, pbc, rc)
            i, j, S = self._compute_neighbors(pos_np, cell_np, pbc, rc)
            self.n_neighbor_builds += 1
            n_at = len(z_np)
            counts.append(n_at)
            can_stress_list.append(can_stress_b)
            z_list.append(torch.from_numpy(z_np))
            pos_list.append(torch.as_tensor(pos_np, dtype=model_dtype))
            cell_list.append(torch.as_tensor(eff_cell_np, dtype=model_dtype))
            batch_list.append(torch.full((n_at,), b, dtype=torch.long))
            ei_list.append(torch.stack([torch.from_numpy(i + n_off), torch.from_numpy(j + n_off)]))
            esf_list.append(torch.from_numpy(S))
            u_list.append(u_val)
            lot_list.append(lid)
            scales.append(ev_scale)
            n_off += n_at

        z_cpu = torch.cat(z_list)
        batch_cpu = torch.cat(batch_list)
        u_cpu = torch.tensor(u_list, dtype=torch.long)
        lot_cpu = torch.tensor(lot_list, dtype=torch.long)

        z_t = z_cpu.to(self.device)
        pos_t = torch.cat(pos_list).to(self.device)
        cell_t = torch.stack(cell_list).to(self.device)
        batch_t = batch_cpu.to(self.device)
        ei_t = torch.cat(ei_list, dim=1).to(self.device)
        esf_t = torch.cat(esf_list, dim=0).to(self.device)
        u_t = u_cpu.to(self.device)
        lot_t = int(lot_list[0]) if len(set(lot_list)) == 1 else lot_cpu.to(self.device)
        do_batch_stress = need_stress and any(can_stress_list)

        def _run_batch():
            return self.model.forward_crystal(
                z_t,
                pos_t,
                cell_t,
                batch_t,
                ei_t,
                esf_t,
                u=u_t,
                lot_id=lot_t,
                compute_forces=need_forces,
                compute_stress=do_batch_stress,
                create_graph=False,
                virial_mode="edge",
            )

        if self._compile_enabled:
            try:
                out = _run_batch()
            except Exception as exc:
                import warnings

                self._fallback_to_eager(
                    f"runtime compilation error ({type(exc).__name__}: {str(exc).splitlines()[0][:200]})"
                )
                warnings.warn(
                    f"MenaLAMCalculator: {self.compile_status}; running without compilation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                out = _run_batch()
        else:
            out = _run_batch()
        e_ref = self.model.crystal_reference(z_cpu, batch_cpu, u_cpu, B, lot_id=lot_cpu)
        e_kcal = out["energy"].detach().cpu().double().numpy() + e_ref.double().numpy()
        f_kcal = out["forces"].detach().cpu().numpy().astype(np.float64) if out.get("forces") is not None else None
        s_kcal = out["stress"].detach().cpu().numpy().astype(np.float64) if out.get("stress") is not None else None

        results_list: list[dict[str, object]] = []
        off = 0
        for b, n_at in enumerate(counts):
            sc = scales[b]
            e_ev = float(e_kcal[b] / sc)
            f_ev = (f_kcal[off : off + n_at] / sc) if f_kcal is not None else np.zeros((n_at, 3), dtype=np.float64)
            if self.zero_net_force and len(f_ev) > 0:
                f_ev = f_ev - f_ev.mean(axis=0, keepdims=True)
            item_res: dict[str, object] = {
                "energy": e_ev,
                "free_energy": e_ev,
                "forces": f_ev,
                "charges": None,
            }
            if can_stress_list[b]:
                if s_kcal is not None:
                    sb = s_kcal[b] / sc
                    s_voigt = np.array(
                        [sb[0, 0], sb[1, 1], sb[2, 2], sb[1, 2], sb[0, 2], sb[0, 1]],
                        dtype=np.float64,
                    )
                else:
                    s_voigt = np.zeros(6, dtype=np.float64)
                item_res["stress"] = s_voigt
            results_list.append(item_res)
            off += n_at
        return results_list


