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
    CRYSTAL_EV, E_ANG_TO_DEBYE, MOL_KCAL_TO_EV, _get_d3,
    build_energy_decomposition, effective_periodic_cell, eval_dispersion_device, eval_dispersion_single,
    polarizability_from_sums, resolve_charge_and_spin, resolve_charge_spin_fragments,
    resolve_lot_and_u,
)
from lam4.calc.multipole_pack import MULTIPOLE_PROPERTIES, append_multipole_flat_parts, unpack_multipoles_from_host, wants_multipoles
from lam4.calc.precision_hygiene import center_positions_f64, restore_charged_dipole_f64, run_recentered_relax_batch
from lam4.calc.runtime import run_model_checked, setup_calc_compile
from lam4.calc.verlet import _try_pin_empty, build_or_reuse_batch, ref_energies_kcal, verlet_neighbor_graph
from lam4.data.batch import GraphBatch
from lam4.model import Lam4Model

_TERM_KEYS = ("stage0", "stage1", "final", "iso")


class MenaCalculator(Calculator):
    """Unified periodic + molecular ASE Calculator for `lam4` / `lam5` models with full interpretability."""

    implemented_properties = [
        "energy", "free_energy", "forces", "stress", "dipole", "charges", "magmoms",
        *MULTIPOLE_PROPERTIES,
    ]

    def __init__(
        self, checkpoint: str | Path | Lam4Model, *, lot: str = "auto",
        device: str | torch.device = "auto", dtype: str | torch.dtype = "float32",
        precision: str = "auto", weights: str = "ema", skin: float = 0.0,
        low_memory: bool = False, edge_mode: str = "auto", csr_check: bool = True,
        compile: bool | str = False, compile_check: bool = True,
        charge: float | None = None, spin: int | None = None, u_flag: int | None = None,
        lock_lot: bool = True, add_dispersion: bool = True, add_ref: bool = True,
        compute_forces: bool = True, compute_stress: bool | str = "auto", zero_net_force: bool = False,
        interpret: bool = False, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.interpret = bool(interpret)  # opt-in: one extra eager pass per calculate() fills results['interpretation']
        self.model, self.device, self.dtype, self.meta = load_lam4_checkpoint(
            checkpoint, device=device, dtype=dtype, precision=precision, weights=weights,
            low_memory=low_memory, edge_mode=edge_mode, bake=(str(precision).strip().lower() != "bf16_nodes"),
        )
        self.precision = str(self.meta.get("precision", precision))
        self.lot = str(lot)
        self.lock_lot = bool(lock_lot)
        self.skin = max(0.0, float(skin))
        self.low_memory = bool(low_memory)
        self.charge_override, self.spin_override, self.u_override = charge, spin, u_flag
        self.add_dispersion = bool(add_dispersion)
        if self.add_dispersion and not self.low_memory and self.device.type in ("mps", "cuda"):
            _get_d3(self.device, self.dtype)
        self.add_ref = bool(add_ref)
        self.compute_forces = bool(compute_forces)
        self.compute_stress = compute_stress
        self.zero_net_force = bool(zero_net_force)
        self.csr_check = bool(csr_check)
        self._csr_checked = False
        self.csr_status = ("unchecked" if self.csr_check else "on") if self.model.trunk.edge_mode == "csr" else "off"
        self.compile_check = bool(compile_check)
        self._compiled = False
        self._compile_checked = False
        self._compiled_whole_model: Any = None
        self.compile_status = "off"
        self._nl_cache: dict[str, Any] | None = None
        self._active_csr_plan: Any = None
        self._static_cache: dict[str, Any] | None = None
        self._lot_cache: tuple[tuple[Any, ...], Any, int] | None = None
        self._last_info_sig: tuple[Any, ...] | None = None
        self._set_compiled_active(False)
        self._want_compile = compile is True or (
            isinstance(compile, str)
            and compile.lower() not in ("false", "off", "none", "0", "eager")
            and (compile.lower() != "auto" or self.device.type in ("mps", "cuda"))
        )
        self._compile_mode = None if (compile is True or str(compile).lower() in ("on", "true", "1", "auto", "default")) else str(compile)
        if self._want_compile:
            self._setup_compile(mode=self._compile_mode)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path | Lam4Model, **kwargs: Any) -> "MenaCalculator":
        return cls(checkpoint, **kwargs)

    def _setup_compile(self, mode: str | None = None) -> None:
        setup_calc_compile(self, mode=mode)

    def _set_compiled_active(self, active: bool) -> None:
        pairs = (("core_plain", "_compiled_core", "_eager_core"), ("finish_plain", "_compiled_finish", "_eager_finish"),
                 ("aggregate_plain", "_compiled_agg", "_eager_agg"), ("post_plain", "_compiled_post", "_eager_post"))
        for st in self.model.trunk.stages:
            for attr, c_key, e_key in pairs:
                key = c_key if active else e_key
                if key in st.__dict__:
                    setattr(st, attr, st.__dict__[key])
        self._compiled = bool(active)

    def _disable_compile(self, reason: str = "off") -> None:
        self._set_compiled_active(False)
        self.compile_status = reason

    def _resolve_lot_cached(self, atoms: Atoms, z_np: np.ndarray | None = None) -> tuple[Any, int]:
        if not self.lock_lot:
            return resolve_lot_and_u(atoms, self.lot, self.u_override)
        z_arr = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64) if z_np is None else z_np
        pbc_t = tuple(bool(x) for x in atoms.get_pbc())
        sig = (z_arr.tobytes(), pbc_t, atoms.info.get("lot"), atoms.info.get("u_flag"), self.lot, self.u_override)
        cached = self._lot_cache
        if cached is not None and cached[0] == sig:
            return cached[1], cached[2]
        src, u_val = resolve_lot_and_u(atoms, self.lot, self.u_override)
        self._lot_cache = (sig, src, u_val)
        return src, u_val

    def _resolve_lot_and_u(self, atoms: Atoms) -> tuple[str, int]:
        src, u = self._resolve_lot_cached(atoms)
        return src.name, u

    def retarget(self, atoms: Atoms | None = None) -> tuple[str, int] | None:
        """Invalidate locked `(lot, u_flag)` cache (and re-resolve immediately when `atoms` is passed)."""
        self._lot_cache = None
        self._last_info_sig = None
        self.results = {}
        if atoms is None:
            return None
        src, u_val = self._resolve_lot_cached(atoms)
        return src.name, u_val

    @staticmethod
    def _sig_val(val: Any) -> Any:
        if val is None or isinstance(val, (bool, int, float, str, bytes)):
            return val
        if isinstance(val, np.ndarray):
            arr = np.ascontiguousarray(val)
            return ("ndarray", str(arr.dtype), arr.shape, arr.tobytes())
        if isinstance(val, torch.Tensor):
            arr = np.ascontiguousarray(val.detach().cpu().numpy())
            return ("tensor", str(arr.dtype), arr.shape, arr.tobytes())
        if isinstance(val, dict):
            return ("dict", tuple(
                (MenaCalculator._sig_val(k), MenaCalculator._sig_val(v))
                for k, v in sorted(val.items(), key=lambda kv: (type(kv[0]).__name__, str(kv[0])))
            ))
        if isinstance(val, (list, tuple)):
            return tuple(MenaCalculator._sig_val(x) for x in val)
        return val

    def _info_sig(self, atoms: Atoms) -> tuple[Any, ...]:
        return (
            atoms.info.get("charge"), atoms.info.get("spin"), atoms.info.get("multiplicity"),
            atoms.info.get("lot"), atoms.info.get("u_flag"),
            self._sig_val(atoms.arrays.get("frag_id")),
            self._sig_val(atoms.info.get("frag_id")),
            self._sig_val(atoms.info.get("frag_charge")),
            self._sig_val(atoms.info.get("frag_spin")),
            self.lot,
            self.charge_override, self.spin_override, self.u_override,
            self.add_dispersion, self.add_ref, self.zero_net_force,
            self.compute_forces, self.compute_stress, self.precision,
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
        self._lot_cache = None
        self._last_info_sig = None
        from lam4.calc.d3_nl_cache import clear_d3_caches
        clear_d3_caches()

    def _neighbor_graph(self, z_np: np.ndarray, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool]) -> tuple[Tensor, Tensor, list | None]:
        return verlet_neighbor_graph(self, z_np, pos_np, cell_np, pbc_flags)

    def _prepare_single_batch(
        self, atoms: Atoms,
    ) -> tuple[GraphBatch, list | None, tuple[float, float, np.ndarray], dict[str, Any], Any, int, float, int, np.ndarray, np.ndarray, np.ndarray | None, tuple[bool, bool, bool]]:
        z_np = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        pos_np = np.asarray(atoms.get_positions(), dtype=np.float64)
        pos_c, origin_f64 = center_positions_f64(pos_np)
        src, u_val = self._resolve_lot_cached(atoms, z_np)
        pbc_any = bool(np.any(atoms.get_pbc()))
        q_val, s_val, fid_np, fq_np, fs_np = resolve_charge_spin_fragments(
            atoms, z_np, self.charge_override, self.spin_override, pbc_any=pbc_any, domain=src.domain,
        )
        cell_np, pbc_flags = effective_periodic_cell(atoms, self.model.cfg.r_cut) if pbc_any else (None, (False, False, False))
        ei, es, chunks = verlet_neighbor_graph(self, z_np, pos_np, cell_np, pbc_flags)
        b, ref_tup, sc = build_or_reuse_batch(
            self, z_np, pos_c, cell_np, pbc_flags, src, u_val, q_val, s_val, ei, es,
            frag_id_np=fid_np, frag_q_np=fq_np, frag_s_np=fs_np, recenter=False,
        )
        sc["origin_f64"] = origin_f64
        return b, chunks, ref_tup, sc, src, u_val, q_val, s_val, z_np, pos_c, cell_np, pbc_flags

    def _single_batch(self, atoms: Atoms) -> tuple[GraphBatch, list | None, Any, int, float, int, np.ndarray, np.ndarray, np.ndarray | None, tuple[bool, bool, bool]]:
        b, chunks, _, _, src, u_val, q_val, s_val, z_np, pos_np, cell_np, pbc_flags = self._prepare_single_batch(atoms)
        return b, chunks, src, u_val, q_val, s_val, z_np, pos_np, cell_np, pbc_flags

    def _ref_energies_kcal(
        self, src_name: str, domain: str, pbc_any: bool, z_np: np.ndarray, q_val: float, s_val: int,
        frag_charge: np.ndarray | Sequence[float] | None = None,
        frag_spin: np.ndarray | Sequence[int | float] | None = None,
    ) -> tuple[float, float, np.ndarray]:
        return ref_energies_kcal(
            self.add_ref, src_name, domain, pbc_any, z_np, q_val, s_val,
            frag_charge=frag_charge, frag_spin=frag_spin,
        )

    def _run_model(self, b: GraphBatch, chunks: list | None, *, need_forces: bool, need_stress: bool, csr_plan: Any = None, return_multipoles: bool = False) -> dict[str, Any]:
        return run_model_checked(self, b, chunks, csr_plan, need_forces=need_forces, need_stress=need_stress, return_multipoles=return_multipoles)

    def calculate(self, atoms: Atoms | None = None, properties: Sequence[str] = ("energy", "forces"), system_changes: Sequence[str] = all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None
        self._last_info_sig = self._info_sig(self.atoms)
        b, chunks, (e0_kcal, eqs_kcal, atom_e0_kcal), sc, src, u_val, q_val, s_val, z_np, pos_np, cell_np, pbc_flags = self._prepare_single_batch(self.atoms)
        pbc_any = bool(any(pbc_flags))

        can_stress = bool(all(pbc_flags)) and cell_np is not None and abs(float(np.linalg.det(cell_np))) > 1e-8
        need_forces = self.compute_forces or ("forces" in properties)
        cs_mode = self.compute_stress
        if cs_mode is True:
            need_stress = can_stress
        elif cs_mode is False or str(cs_mode).strip().lower() in ("requested", "off", "false", "0"):
            need_stress = can_stress and ("stress" in properties)
        else:
            need_stress = can_stress and (("stress" in properties) or (need_forces and not self.low_memory))
        need_multipoles = wants_multipoles(properties)
        kcal_to_ev = (1.0 / CRYSTAL_EV) if src.domain == "crystal" else MOL_KCAL_TO_EV

        out = self._run_model(
            b, chunks, need_forces=need_forces, need_stress=need_stress,
            csr_plan=self._active_csr_plan, return_multipoles=need_multipoles,
        )
        d3_mode = src.d3_addback if self.add_dispersion else None
        # Small open molecules on MPS (N <= 256): run fast C++ CPU D3 (0.02-0.32 ms) in parallel while the MPS GPU computes the model!
        d3_dev = torch.device("cpu") if (self.device.type == "mps" and not pbc_any and d3_mode in ("bj_wb97m", "zero_pbe") and len(z_np) <= 256) else self.device
        d3_on_dev, d3_e_v, d3_f_v, d3_s_v = eval_dispersion_device(
            z_np, pos_np, cell_np, pbc_flags, d3_mode, d3_dev, self.dtype,
            compute_forces=need_forces, compute_stress=need_stress,
            z_t=(b.z if d3_dev == self.device else None),
            pos_t=(b.pos if d3_dev == self.device else None),
            cell_t=(b.cell if (pbc_any and d3_dev == self.device) else None),
        )

        # Pack scalars + forces + stress + per-atom arrays + stage charges into ONE single D2H host transfer!
        terms_d = out.get("terms") or {}
        z0 = out["energy"][0] * 0.0
        t_list = [terms_d[k].sum() if k in terms_d else z0 for k in _TERM_KEYS]
        ov_t = out["overlap_energy"][0] if out.get("overlap_energy") is not None else z0
        ex_t = out["spin_exchange_energy"][0] if out.get("spin_exchange_energy") is not None else z0
        d3_e_t = d3_e_v.to(dtype=z0.dtype) if d3_on_dev else z0
        s_list = [out["energy"][0], out["network_energy"][0], out["electrostatic_energy"][0],
                  out["polarization_energy"][0], out["repulsion_energy"][0], *t_list, ov_t, ex_t, d3_e_t]
        has_mol_pol = out.get("alpha") is not None
        if has_mol_pol:
            s_list.append(out["alpha"].sum())
            if out.get("alpha_aniso") is not None:
                s_list.extend(out["alpha_aniso"].sum(dim=0).unbind(0))
        if out.get("dipole") is not None:
            s_list.extend(out["dipole"][0].unbind(0))

        n, n_s = len(z_np), len(s_list)
        has_f = bool(need_forces and out.get("forces") is not None)
        has_s = bool(need_stress and out.get("stress") is not None)
        has_q = out.get("charges") is not None
        has_sd = bool(has_q and out.get("spin_density") is not None)
        has_al = bool(has_q and has_mol_pol)
        st_q_list = list(out.get("stage_charges") or []) if has_q else []

        flat_parts: list[Tensor] = [torch.stack(s_list)]
        if has_f:
            f_dev_out = out["forces"] + d3_f_v.to(dtype=out["forces"].dtype) if (d3_on_dev and d3_f_v is not None) else out["forces"]
            flat_parts.append(f_dev_out.reshape(-1))
        if has_s:
            s_dev_out = out["stress"][0] + d3_s_v.to(dtype=out["stress"].dtype) if (d3_on_dev and d3_s_v is not None) else out["stress"][0]
            flat_parts.append(s_dev_out.reshape(-1))
        flat_parts.append(out["energy_atom"].reshape(-1))
        if has_q:
            flat_parts.append(out["charges"].reshape(-1))
            if has_sd:
                flat_parts.append(out["spin_density"].reshape(-1))
            if has_al:
                flat_parts.append(out["alpha"].reshape(-1))
            for q_st in st_q_list:
                flat_parts.append(q_st.reshape(-1))
        has_mp = bool(need_multipoles and append_multipole_flat_parts(flat_parts, out))

        cat_gpu = torch.cat(flat_parts, dim=0).detach()
        tot_len = int(cat_gpu.numel())
        d2h_pin = sc.get("d2h_pin") if self.device.type == "cuda" else None
        if self.device.type == "cuda" and (d2h_pin is None or d2h_pin.numel() < tot_len or d2h_pin.dtype != cat_gpu.dtype):
            d2h_pin = _try_pin_empty((max(tot_len, 256),), cat_gpu.dtype, self.device)
            sc["d2h_pin"] = d2h_pin
        if d2h_pin is not None:
            pin_view = d2h_pin[:tot_len]
            pin_view.copy_(cat_gpu, non_blocking=False)
            host_buf = np.asarray(pin_view.numpy(), dtype=np.float64)
        else:
            host_buf = cat_gpu.cpu().to(torch.float64).numpy()
        scalars, cur = host_buf[:n_s], n_s
        term_sums = {k: float(scalars[5 + idx]) for idx, k in enumerate(_TERM_KEYS) if k in terms_d}
        ov_k, exch_k = float(scalars[5 + len(_TERM_KEYS)]), float(scalars[6 + len(_TERM_KEYS)])
        d3_e = float(scalars[7 + len(_TERM_KEYS)]) if d3_on_dev else float(d3_e_v)
        d3_f = None if d3_on_dev else d3_f_v
        d3_s = None if d3_on_dev else d3_s_v
        idx_cur = 8 + len(_TERM_KEYS)
        if has_mol_pol:
            iso_sum = float(scalars[idx_cur])
            idx_cur += 1
            an_sum = scalars[idx_cur : idx_cur + 5] if out.get("alpha_aniso") is not None else None
            if an_sum is not None:
                idx_cur += 5
            pol_mat, pol_iso, pol_aniso = polarizability_from_sums(iso_sum, an_sum)
        else:
            pol_mat, pol_iso, pol_aniso = sc["zeros_33"].copy(), 0.0, 0.0
        dip_ea = restore_charged_dipole_f64(scalars[idx_cur : idx_cur + 3], q_val, sc["origin_f64"], True) if out.get("dipole") is not None else sc["zeros_3"].copy()
        dip_deb = dip_ea * E_ANG_TO_DEBYE

        forces_ev = np.zeros((n, 3), dtype=np.float64)
        if has_f:
            f_kcal = host_buf[cur : cur + 3 * n].reshape(n, 3)
            cur += 3 * n
            if d3_f is not None:
                f_kcal = f_kcal + d3_f
            forces_ev = f_kcal * kcal_to_ev
            if self.zero_net_force and n > 0:
                forces_ev = forces_ev - forces_ev.mean(axis=0, keepdims=True)

        stress_voigt = np.zeros(6, dtype=np.float64)
        if has_s:
            s33 = host_buf[cur : cur + 9].reshape(3, 3)
            cur += 9
            if d3_s is not None:
                s33 = s33 + d3_s
            stress_voigt = s33[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] * kcal_to_ev

        at_ener_raw = host_buf[cur : cur + n]
        e_model_kcal = float(at_ener_raw.sum())
        e_tot_ev = float((e_model_kcal + d3_e + e0_kcal + eqs_kcal) * kcal_to_ev)
        at_ener_ev = (at_ener_raw + atom_e0_kcal) * kcal_to_ev
        cur += n
        if has_q:
            charges = host_buf[cur : cur + n].copy(); cur += n
            spin_dens = host_buf[cur : cur + n].copy() if has_sd else sc["zeros_n"].copy(); cur += n if has_sd else 0
            at_alpha = host_buf[cur : cur + n].copy() if has_al else sc["zeros_n"].copy(); cur += n if has_al else 0
            stage_charges = [host_buf[cur + k * n : cur + (k + 1) * n].copy() for k in range(len(st_q_list))]
            cur += len(st_q_list) * n
        else:
            charges, spin_dens, at_alpha, stage_charges = sc["zeros_n"].copy(), sc["zeros_n"].copy(), sc["zeros_n"].copy(), []
        mp_dict, cur = unpack_multipoles_from_host(host_buf, cur, n, has_mp=has_mp, kcal_to_ev=kcal_to_ev) if need_multipoles else (None, cur)

        coul_k, pol_k, rep_k = float(scalars[2]), float(scalars[3]), float(scalars[4])
        decomp = build_energy_decomposition(
            kcal_to_ev=kcal_to_ev, net_kcal=float(e_model_kcal - coul_k - pol_k - rep_k - ov_k - exch_k), coul_kcal=coul_k,
            pol_kcal=pol_k, rep_kcal=rep_k, ov_kcal=ov_k, exch_kcal=exch_k, d3_kcal=d3_e, e0_kcal=e0_kcal, eqs_kcal=eqs_kcal, term_sums_kcal=term_sums,
        )
        self.results = {
            "energy": e_tot_ev, "free_energy": e_tot_ev, "dipole": dip_ea, "dipole_debye": dip_deb,
            "dipole_norm_debye": float(np.linalg.norm(dip_deb)), "charges": charges,
            "magmoms": spin_dens, "spin_density": spin_dens, "stage_charges": stage_charges,
            "polarizability": pol_mat, "polarizability_iso": pol_iso, "polarizability_aniso": pol_aniso,
            "atomic_polarizabilities": at_alpha, "atomic_energies": at_ener_ev,
            "energy_decomposition": decomp, "lot_used": src.name, "u_used": int(u_val),
        }
        if need_forces:
            self.results["forces"] = forces_ev
        if has_s or "stress" in properties:
            self.results["stress"] = stress_voigt
        if mp_dict is not None:
            self.results.update(mp_dict)
            self.results["multipoles"] = mp_dict
        if self.interpret:
            self.results["interpretation"] = self.interpretation(self.atoms)

    def eval_batch(
        self, atoms_list: Sequence[Atoms], properties: Sequence[str] = ("energy", "forces", "dipole", "polarizability"),
        *, max_atoms: int = 4096, max_batch: int = 128,
    ) -> list[dict[str, Any]]:
        return eval_batch_frames(self, atoms_list, properties=properties, max_atoms=max_atoms, max_batch=max_batch)

    def relax_batch(
        self, atoms_list: Sequence[Atoms], *, fmax: float = 0.01, steps: int = 200, maxstep: float = 0.2,
        skin: float = 0.6, history_size: int = 8, max_atoms: int = 3072, max_batch: int = 64,
    ) -> list[dict[str, Any]]:
        """Tier-2 GPU-resident sparse-batched L-BFGS geometry relaxation."""
        return run_recentered_relax_batch(self, atoms_list, fmax=fmax, steps=steps, maxstep=maxstep, skin=skin, history_size=history_size, max_atoms=max_atoms, max_batch=max_batch)

    def relax(self, atoms: Atoms, **kwargs: Any) -> dict[str, Any]:
        """Single-structure geometry/cell relaxation across molecules, crystals, slabs & restraints; see `lam4.calc.md_runner`."""
        from lam4.calc.md_runner import relax_atoms
        return relax_atoms(self, atoms, **kwargs)

    def run_md(self, atoms: Atoms, **kwargs: Any) -> dict[str, Any]:
        """Plug-and-play NVT/NVE/NPT molecular dynamics with automatic `MDEngine`/ASE routing; see `lam4.calc.md_runner`."""
        from lam4.calc.md_runner import run_md
        return run_md(self, atoms, **kwargs)

    def md_engine(self, atoms: Atoms, **kwargs: Any) -> "Any":
        """Low-level persistent GPU-resident NVT/NVE engine (prefer `calc.run_md(...)` for casual MD); see `lam4.calc.md_engine`."""
        from lam4.calc.md_engine import MDEngine
        return MDEngine(self, atoms, **kwargs)

    def interpretation(self, atoms: Atoms | None = None) -> dict[str, Any]:
        """Physics-head read-out (charges, spins, Pauli overlap, exchange, gap); see `lam4.calc.interpret_heads`."""
        from lam4.calc.interpret_heads import interpret_heads
        target = atoms if atoms is not None else self.atoms
        if target is None:
            raise ValueError("interpretation() needs atoms (or a previous calculate())")
        return interpret_heads(self, target)

    def explain(self) -> str:
        return format_calc_summary(self.results, self.device)


Lam4Calculator = MenaCalculator
