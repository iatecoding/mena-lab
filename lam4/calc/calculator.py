"""MenaCalculator: ASE Calculator & Batched Evaluation Engine for `lam4` / `lam5` checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import torch
from torch import Tensor

from lam4.calc.batched import eval_batch_frames, format_calc_summary
from lam4.calc.loader import load_lam4_checkpoint
from lam4.calc.physics import (
    CRYSTAL_EV,
    E_ANG_TO_DEBYE,
    MOL_KCAL_TO_EV,
    _get_d3,
    build_energy_decomposition,
    effective_periodic_cell,
    eval_dispersion_single,
    polarizability_from_sums,
    resolve_charge_and_spin,
    resolve_lot_and_u,
)
from lam4.calc.runtime import run_model_checked, setup_calc_compile
from lam4.calc.verlet import build_or_reuse_batch, ref_energies_kcal, verlet_neighbor_graph
from lam4.data.batch import GraphBatch
from lam4.model import Lam4Model

_TERM_KEYS = ("stage0", "stage1", "final", "iso")


class MenaCalculator(Calculator):
    """Unified periodic + molecular ASE Calculator for `lam4` / `lam5` models with full interpretability."""

    implemented_properties = ["energy", "free_energy", "forces", "stress", "dipole", "charges", "magmoms"]

    def __init__(
        self,
        checkpoint: str | Path | Lam4Model,
        *,
        lot: str = "auto",
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "float32",
        weights: str = "ema",
        skin: float = 0.0,
        low_memory: bool = False,
        edge_mode: str = "auto",
        csr_check: bool = True,
        compile: bool | str = False,
        compile_check: bool = True,
        charge: float | None = None,
        spin: int | None = None,
        u_flag: int | None = None,
        add_dispersion: bool = True,
        add_ref: bool = True,
        compute_forces: bool = True,
        zero_net_force: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.device, self.dtype, self.meta = load_lam4_checkpoint(
            checkpoint, device=device, dtype=dtype, weights=weights, low_memory=low_memory, edge_mode=edge_mode, bake=True
        )
        self.lot = str(lot)
        self.skin = max(0.0, float(skin))
        self.low_memory = bool(low_memory)
        self.charge_override = charge
        self.spin_override = spin
        self.u_override = u_flag
        self.add_dispersion = bool(add_dispersion)
        if self.add_dispersion and not self.low_memory and self.device.type == "mps":
            _get_d3(self.device, self.dtype)
        self.add_ref = bool(add_ref)
        self.compute_forces = bool(compute_forces)
        self.zero_net_force = bool(zero_net_force)
        self.csr_check = bool(csr_check)
        self._csr_checked = False
        self.csr_status = ("unchecked" if self.csr_check else "on") if self.model.trunk.edge_mode == "csr" else "off"
        self.compile_check = bool(compile_check)
        self._compiled = False
        self._compile_checked = False
        self.compile_status = "off"
        self._nl_cache: dict[str, Any] | None = None
        self._active_csr_plan: Any = None
        self._static_cache: dict[str, Any] | None = None
        self._last_info_sig: tuple[Any, ...] | None = None
        self._set_compiled_active(False)
        self._want_compile = compile is True or (
            isinstance(compile, str)
            and compile.lower() not in ("false", "off", "none", "0", "eager")
            and (compile.lower() != "auto" or self.device.type in ("mps", "cuda"))
        )
        self._compile_mode = (
            None if (compile is True or str(compile).lower() in ("on", "true", "1", "auto", "default")) else str(compile)
        )
        if self._want_compile:
            self._setup_compile(mode=self._compile_mode)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path | Lam4Model, **kwargs: Any) -> "MenaCalculator":
        return cls(checkpoint, **kwargs)

    def _setup_compile(self, mode: str | None = None) -> None:
        setup_calc_compile(self, mode=mode)

    def _set_compiled_active(self, active: bool) -> None:
        for st in self.model.trunk.stages:
            if active:
                if "_compiled_core" in st.__dict__:
                    st.core_plain = st._compiled_core
                if "_compiled_finish" in st.__dict__:
                    st.finish_plain = st._compiled_finish
                if "_compiled_agg" in st.__dict__:
                    st.aggregate_plain = st._compiled_agg
                if "_compiled_post" in st.__dict__:
                    st.post_plain = st._compiled_post
            else:
                if "_eager_core" in st.__dict__:
                    st.core_plain = st._eager_core
                if "_eager_finish" in st.__dict__:
                    st.finish_plain = st._eager_finish
                if "_eager_agg" in st.__dict__:
                    st.aggregate_plain = st._eager_agg
                if "_eager_post" in st.__dict__:
                    st.post_plain = st._eager_post
        self._compiled = bool(active)

    def _disable_compile(self, reason: str = "off") -> None:
        self._set_compiled_active(False)
        self.compile_status = reason

    def _resolve_lot_and_u(self, atoms: Atoms) -> tuple[str, int]:
        src, u = resolve_lot_and_u(atoms, self.lot, self.u_override)
        return src.name, u

    def _info_sig(self, atoms: Atoms) -> tuple[Any, ...]:
        return (
            atoms.info.get("charge"),
            atoms.info.get("spin"),
            atoms.info.get("multiplicity"),
            atoms.info.get("lot"),
            self.lot,
            self.charge_override,
            self.spin_override,
            self.u_override,
        )

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list[str]:
        changes = super().check_state(atoms, tol=tol)
        if not changes and self._last_info_sig != self._info_sig(atoms):
            changes.append("info")
        return changes

    def reset(self) -> None:
        super().reset()
        self._nl_cache = None
        self._active_csr_plan = None
        self._static_cache = None
        self._last_info_sig = None

    def _neighbor_graph(
        self, z_np: np.ndarray, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool]
    ) -> tuple[Tensor, Tensor, list | None]:
        return verlet_neighbor_graph(self, z_np, pos_np, cell_np, pbc_flags)

    def _single_batch(
        self, atoms: Atoms
    ) -> tuple[GraphBatch, list | None, Any, int, float, int, np.ndarray, np.ndarray, np.ndarray | None, tuple[bool, bool, bool]]:
        z_np = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        pos_np = np.asarray(atoms.get_positions(), dtype=np.float64)
        src, u_val = resolve_lot_and_u(atoms, self.lot, self.u_override)
        pbc_any = bool(np.any(atoms.get_pbc()))
        q_val, s_val = resolve_charge_and_spin(atoms, self.charge_override, self.spin_override, pbc_any=pbc_any)
        cell_np, pbc_flags = effective_periodic_cell(atoms, self.model.cfg.r_cut) if pbc_any else (None, (False, False, False))
        ei, es, chunks = verlet_neighbor_graph(self, z_np, pos_np, cell_np, pbc_flags)
        b, _, _ = build_or_reuse_batch(self, z_np, pos_np, cell_np, pbc_flags, src, u_val, q_val, s_val, ei, es)
        return b, chunks, src, u_val, q_val, s_val, z_np, pos_np, cell_np, pbc_flags

    def _ref_energies_kcal(self, src_name: str, domain: str, pbc_any: bool, z_np: np.ndarray, q_val: float, s_val: int) -> tuple[float, float, np.ndarray]:
        return ref_energies_kcal(self.add_ref, src_name, domain, pbc_any, z_np, q_val, s_val)

    def _run_model(
        self, b: GraphBatch, chunks: list | None, *, need_forces: bool, need_stress: bool, csr_plan: Any = None
    ) -> dict[str, Any]:
        return run_model_checked(self, b, chunks, csr_plan, need_forces=need_forces, need_stress=need_stress)

    def calculate(self, atoms: Atoms | None = None, properties: Sequence[str] = ("energy", "forces"), system_changes: Sequence[str] = all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None
        self._last_info_sig = self._info_sig(self.atoms)
        z_np = np.asarray(self.atoms.get_atomic_numbers(), dtype=np.int64)
        pos_np = np.asarray(self.atoms.get_positions(), dtype=np.float64)
        src, u_val = resolve_lot_and_u(self.atoms, self.lot, self.u_override)
        pbc_any = bool(np.any(self.atoms.get_pbc()))
        q_val, s_val = resolve_charge_and_spin(self.atoms, self.charge_override, self.spin_override, pbc_any=pbc_any)
        cell_np, pbc_flags = effective_periodic_cell(self.atoms, self.model.cfg.r_cut) if pbc_any else (None, (False, False, False))
        ei, es, chunks = verlet_neighbor_graph(self, z_np, pos_np, cell_np, pbc_flags)
        b, (e0_kcal, eqs_kcal, atom_e0_kcal), sc = build_or_reuse_batch(self, z_np, pos_np, cell_np, pbc_flags, src, u_val, q_val, s_val, ei, es)

        can_stress = bool(all(pbc_flags)) and cell_np is not None and abs(float(np.linalg.det(cell_np))) > 1e-8
        need_forces = self.compute_forces or ("forces" in properties)
        need_stress = ("stress" in properties) and can_stress
        kcal_to_ev = (1.0 / CRYSTAL_EV) if src.domain == "crystal" else MOL_KCAL_TO_EV

        # Launch model forward/backward first; on MPS small open molecules, overlap CPU D3BJ with async Metal execution!
        out = self._run_model(b, chunks, need_forces=need_forces, need_stress=need_stress, csr_plan=self._active_csr_plan)
        d3_mode = src.d3_addback if self.add_dispersion else None
        d3_dev = torch.device("cpu") if (self.device.type == "mps" and not pbc_any and d3_mode == "bj_wb97m" and len(z_np) <= 48) else self.device
        d3_e, d3_f, d3_s = eval_dispersion_single(
            z_np, pos_np, cell_np, pbc_flags, d3_mode, d3_dev, self.dtype, compute_forces=need_forces, compute_stress=need_stress
        )

        # Pack all device scalar outputs + readout term sums into ONE host sync!
        terms_d = out.get("terms") or {}
        z0 = out["energy"][0] * 0.0
        t_list = [terms_d[k].sum() if k in terms_d else z0 for k in _TERM_KEYS]
        s_list = [
            out["energy"][0],
            out["network_energy"][0],
            out["electrostatic_energy"][0],
            out["polarization_energy"][0],
            out["repulsion_energy"][0],
            *t_list,
        ]
        has_mol_pol = out.get("alpha") is not None
        if has_mol_pol:
            s_list.append(out["alpha"].sum())
            if out.get("alpha_aniso") is not None:
                s_list.extend(out["alpha_aniso"].sum(dim=0).unbind(0))
        if out.get("dipole") is not None:
            s_list.extend(out["dipole"][0].unbind(0))
        scalars = torch.stack(s_list).detach().cpu().to(torch.float64).numpy()

        e_res_kcal = float(scalars[0])
        e_tot_ev = float((e_res_kcal + d3_e + e0_kcal + eqs_kcal) * kcal_to_ev)
        term_sums = {k: float(scalars[5 + idx]) for idx, k in enumerate(_TERM_KEYS) if k in terms_d}
        idx_cur = 9
        if has_mol_pol:
            iso_sum = float(scalars[idx_cur])
            idx_cur += 1
            an_sum = None
            if out.get("alpha_aniso") is not None:
                an_sum = scalars[idx_cur : idx_cur + 5]
                idx_cur += 5
            pol_mat, pol_iso, pol_aniso = polarizability_from_sums(iso_sum, an_sum)
        else:
            pol_mat, pol_iso, pol_aniso = sc["zeros_33"].copy(), 0.0, 0.0
        if out.get("dipole") is not None:
            dip_ea = scalars[idx_cur : idx_cur + 3].copy()
        else:
            dip_ea = sc["zeros_3"].copy()
        dip_deb = dip_ea * E_ANG_TO_DEBYE

        n = len(z_np)
        forces_ev = np.zeros((n, 3), dtype=np.float64)
        if need_forces and out.get("forces") is not None:
            f_kcal = out["forces"].detach().cpu().to(torch.float64).numpy()
            if d3_f is not None:
                f_kcal = f_kcal + d3_f
            forces_ev = f_kcal * kcal_to_ev
            if self.zero_net_force and n > 0:
                forces_ev = forces_ev - forces_ev.mean(axis=0, keepdims=True)

        stress_voigt = np.zeros(6, dtype=np.float64)
        if "stress" in properties and need_stress and out.get("stress") is not None:
            s33 = out["stress"][0].detach().cpu().to(torch.float64).numpy()
            if d3_s is not None:
                s33 = s33 + d3_s
            stress_voigt = s33[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] * kcal_to_ev

        if out.get("charges") is not None:
            per_atom_cols = [out["energy_atom"], out["charges"]]
            has_sd = out.get("spin_density") is not None
            if has_sd:
                per_atom_cols.append(out["spin_density"])
            if has_mol_pol:
                per_atom_cols.append(out["alpha"])
            pa_np = torch.stack(per_atom_cols, dim=0).detach().cpu().to(torch.float64).numpy()
            at_ener_ev = (pa_np[0] + atom_e0_kcal) * kcal_to_ev
            charges = pa_np[1]
            c_idx = 2
            spin_dens = pa_np[c_idx] if has_sd else sc["zeros_n"].copy()
            if has_sd:
                c_idx += 1
            at_alpha = pa_np[c_idx] if has_mol_pol else sc["zeros_n"].copy()
            stage_charges = [q.detach().cpu().to(torch.float64).numpy() for q in (out.get("stage_charges") or [])]
        else:
            at_ener_ev = (out["energy_atom"].detach().cpu().to(torch.float64).numpy() + atom_e0_kcal) * kcal_to_ev
            charges = sc["zeros_n"].copy()
            spin_dens = sc["zeros_n"].copy()
            at_alpha = sc["zeros_n"].copy()
            stage_charges = []

        decomp = build_energy_decomposition(
            kcal_to_ev=kcal_to_ev,
            net_kcal=float(scalars[1]),
            coul_kcal=float(scalars[2]),
            pol_kcal=float(scalars[3]),
            rep_kcal=float(scalars[4]),
            d3_kcal=d3_e,
            e0_kcal=e0_kcal,
            eqs_kcal=eqs_kcal,
            term_sums_kcal=term_sums,
        )

        self.results = {
            "energy": e_tot_ev,
            "free_energy": e_tot_ev,
            "dipole": dip_ea,
            "dipole_debye": dip_deb,
            "dipole_norm_debye": float(np.linalg.norm(dip_deb)),
            "charges": charges,
            "magmoms": spin_dens,
            "spin_density": spin_dens,
            "stage_charges": stage_charges,
            "polarizability": pol_mat,
            "polarizability_iso": pol_iso,
            "polarizability_aniso": pol_aniso,
            "atomic_polarizabilities": at_alpha,
            "atomic_energies": at_ener_ev,
            "energy_decomposition": decomp,
            "lot_used": src.name,
            "u_used": int(u_val),
        }
        if need_forces:
            self.results["forces"] = forces_ev
        if "stress" in properties:
            self.results["stress"] = stress_voigt

    def eval_batch(
        self,
        atoms_list: Sequence[Atoms],
        properties: Sequence[str] = ("energy", "forces", "dipole", "polarizability"),
        *,
        max_atoms: int = 4096,
        max_batch: int = 128,
    ) -> list[dict[str, Any]]:
        return eval_batch_frames(self, atoms_list, properties=properties, max_atoms=max_atoms, max_batch=max_batch)

    def relax_batch(
        self,
        atoms_list: Sequence[Atoms],
        *,
        fmax: float = 0.01,
        steps: int = 200,
        maxstep: float = 0.2,
        skin: float = 0.6,
        history_size: int = 8,
        max_atoms: int = 3072,
        max_batch: int = 64,
    ) -> list[dict[str, Any]]:
        """Tier-2 GPU-resident sparse-batched L-BFGS geometry relaxation."""
        from lam4.calc.relax import relax_batch_frames

        return relax_batch_frames(
            self,
            atoms_list,
            fmax=fmax,
            steps=steps,
            maxstep=maxstep,
            skin=skin,
            history_size=history_size,
            max_atoms=max_atoms,
            max_batch=max_batch,
        )

    def md_engine(self, atoms: Atoms, **kwargs: Any) -> "Any":
        """GPU-resident NVT MD (positions, momenta and thermostat on the device, one CUDA-graph replay per window); see lam4.calc.md_engine."""
        from lam4.calc.md_engine import MDEngine

        return MDEngine(self, atoms, **kwargs)

    def explain(self) -> str:
        return format_calc_summary(self.results, self.device)


Lam4Calculator = MenaCalculator
