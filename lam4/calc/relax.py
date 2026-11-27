"""Tier-2 GPU-resident sparse-batched L-BFGS geometry optimizer for MenaCalculator."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
import numpy as np
import torch
from torch import Tensor

from lam4.calc.physics import MOL_KCAL_TO_EV, lot_u_atoms_for_calc, resolve_charge_spin_fragments, resolve_lot_and_u
from lam4.calc.relax_batch_io import (
    build_mol_subbatch as _build_subbatch,
    build_mol_subbatch_constraints as _build_subbatch_constraints,
    eval_mol_subbatch_ef as _eval_mol_subbatch_ef,
)
from lam4.data.buckets import source_ids


def _parse_move_mask(atoms: Atoms) -> tuple[bool, np.ndarray | None]:
    """Return (only_fix_atoms, move_mask_bool_or_None)."""
    cons = list(getattr(atoms, "constraints", ()) or ())
    if not cons:
        return True, None
    for c in cons:
        if not isinstance(c, FixAtoms):
            return False, None
    mask = np.ones(len(atoms), dtype=bool)
    for c in cons:
        idx = np.asarray(c.get_indices(), dtype=np.int64)
        if len(idx) > 0:
            mask[idx] = False
    if bool(mask.all()):
        return True, None
    return True, mask


def _parse_restraints(atoms: Atoms) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Parse optional harmonic restraints (`restraint_k`, `restraint_ref`) from `Atoms`."""
    rk = atoms.info.get("restraint_k", None)
    if rk is None:
        return None, None
    n = len(atoms)
    rk_arr = np.asarray(rk, dtype=np.float64)
    if rk_arr.ndim == 0:
        rk_np = np.full(n, float(rk_arr), dtype=np.float64)
    elif rk_arr.ndim == 1 and len(rk_arr) == n:
        rk_np = np.ascontiguousarray(rk_arr, dtype=np.float64)
    else:
        raise ValueError(f"restraint_k must be scalar or 1D of length {n}, got shape {rk_arr.shape}")
    if not np.all(np.isfinite(rk_np)) or np.any(rk_np < 0.0):
        raise ValueError("restraint_k values must be finite and >= 0")
    if not np.any(rk_np > 0.0):
        return None, None
    rref = atoms.arrays.get("restraint_ref", None)
    if rref is None:
        rref = atoms.info.get("restraint_ref", None)
    if rref is None:
        ref_np = np.asarray(atoms.get_positions(), dtype=np.float64).copy()
    else:
        ref_np = np.asarray(rref, dtype=np.float64).copy()
        if ref_np.shape != (n, 3):
            raise ValueError(f"restraint_ref must have shape ({n}, 3), got {ref_np.shape}")
        if not np.all(np.isfinite(ref_np)):
            raise ValueError("restraint_ref coordinates must be finite")
    return rk_np, ref_np


class _RestrainedCalc(Calculator):
    """Lightweight ASE wrapper adding harmonic positional restraints around a base calculator."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc: "MenaCalculator", rest_k: np.ndarray, rest_ref: np.ndarray) -> None:
        super().__init__()
        self.base_calc = base_calc
        self.rest_k = np.asarray(rest_k, dtype=np.float64)
        self.rest_ref = np.asarray(rest_ref, dtype=np.float64)
        self._shadow: Atoms | None = None

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | tuple[str, ...] = ("energy", "forces"),
        system_changes: list[str] | tuple[str, ...] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None
        if self._shadow is None or len(self._shadow) != len(self.atoms):
            sh = self.atoms.copy()
            sh.constraints = []
            sh.calc = None
            self._shadow = sh
        else:
            self._shadow.set_positions(self.atoms.get_positions())
            self._shadow.set_cell(self.atoms.get_cell())
            self._shadow.set_pbc(self.atoms.get_pbc())
        e_base = float(self.base_calc.get_potential_energy(self._shadow))
        f_base = np.asarray(self.base_calc.get_forces(self._shadow), dtype=np.float64)
        pos = np.asarray(self.atoms.get_positions(), dtype=np.float64)
        dr = pos - self.rest_ref
        e_rest = 0.5 * float(np.sum(self.rest_k * np.sum(dr * dr, axis=-1)))
        f_rest = -self.rest_k[:, None] * dr
        self.results["energy"] = e_base + e_rest
        self.results["forces"] = f_base + f_rest

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator


def _relax_open_mol_microbatch(
    calc: "MenaCalculator",
    atoms_batch: list[Atoms],
    *,
    fmax_thresh: float,
    max_steps: int,
    maxstep: float,
    skin: float,
    history_size: int,
) -> list[dict[str, Any]]:
    B0 = len(atoms_batch)
    if B0 == 0:
        return []
    dev, dt = calc.device, calc.dtype
    acc_dt = torch.float32 if dev.type == "mps" else torch.float64
    if calc.model.trunk.edge_mode == "csr" and calc.csr_check and not calc._csr_checked:
        calc.calculate(atoms_batch[0].copy(), properties=("energy", "forces"), system_changes=all_changes)

    r_cut = float(calc.model.cfg.r_cut)
    eff_skin = max(0.0, float(skin))
    r_list = r_cut + eff_skin

    all_entries: list[dict[str, Any]] = []
    pos_per_mol: list[Tensor] = []
    for idx, at in enumerate(atoms_batch):
        z_np = np.asarray(at.get_atomic_numbers(), dtype=np.int64)
        p_np = np.asarray(at.get_positions(), dtype=np.float32 if dt == torch.float32 else np.float64)
        src, u_val = calc._resolve_lot_cached(at, z_np) if hasattr(calc, "_resolve_lot_cached") else resolve_lot_and_u(at, calc.lot, calc.u_override)
        q_val, s_val, fid_np, fq_np, fs_np = resolve_charge_spin_fragments(
            at, z_np, calc.charge_override, calc.spin_override, pbc_any=False, domain=src.domain,
        )
        tok_id, ref_id = source_ids(src.name)
        xc_v, paw_v, u_at_np = lot_u_atoms_for_calc(src.name, z_np, p_np, None, (False, False, False), u_val)
        e0_k, eqs_k, _ = calc._ref_energies_kcal(
            src.name, src.domain, False, z_np, q_val, s_val,
            frag_charge=fq_np, frag_spin=fs_np,
        )
        ok_mask, move_mask_np = _parse_move_mask(at)
        if not ok_mask:
            raise ValueError("Unsupported non-FixAtoms constraint inside batched molecular relaxation")
        rest_k_np, rest_ref_np = _parse_restraints(at)
        all_entries.append({
            "orig_idx": idx,
            "atoms0": at,
            "n": len(z_np),
            "z_np": z_np,
            "z_t": torch.from_numpy(np.ascontiguousarray(z_np, dtype=np.int64)).to(dev),
            "q": q_val,
            "s": s_val,
            "frag_id_np": fid_np,
            "frag_q_np": fq_np,
            "frag_s_np": fs_np,
            "move_mask_np": move_mask_np,
            "rest_k_np": rest_k_np,
            "rest_ref_np": rest_ref_np,
            "u": u_val,
            "tok": tok_id,
            "ref": ref_id,
            "xc": xc_v,
            "paw": paw_v,
            "u_atom_t": torch.from_numpy(u_at_np).to(dev),
            "e_ref_kcal": float(e0_k + eqs_k),
            "d3_mode": src.d3_addback if calc.add_dispersion else None,
        })
        pos_per_mol.append(torch.from_numpy(np.ascontiguousarray(p_np)).to(device=dev, dtype=dt))

    init_e = np.zeros(B0, dtype=np.float64)
    final_e = np.zeros(B0, dtype=np.float64)
    final_fmax = np.zeros(B0, dtype=np.float64)
    step_counts = np.zeros(B0, dtype=np.int64)
    converged = np.zeros(B0, dtype=bool)

    sub_ids = list(range(B0))
    entries = [all_entries[i] for i in sub_ids]
    e_ref_np = np.array([e["e_ref_kcal"] for e in entries], dtype=np.float64)
    b, csr_plan, group_ptr, pos_ref = _build_subbatch(calc, entries, [pos_per_mol[i] for i in sub_ids], r_list)
    move_mask_t, rest_k_t, rest_ref_t = _build_subbatch_constraints(entries, dev, dt)
    pos = b.pos.clone()
    active = torch.ones(len(sub_ids), dtype=torch.bool, device=dev)
    act_np = np.ones(len(sub_ids), dtype=bool)

    s_hist: list[Tensor] = []
    y_hist: list[Tensor] = []
    rho_hist: list[Tensor] = []
    pos_prev: Tensor | None = None
    grad_prev: Tensor | None = None

    def bdot(x: Tensor, y: Tensor, graph_idx: Tensor, n_g: int) -> Tensor:
        xy = (x.to(acc_dt) * y.to(acc_dt)).sum(dim=-1)
        return torch.zeros(n_g, dtype=acc_dt, device=dev).index_add_(0, graph_idx, xy)

    def writeback_positions(cur_sub_ids: list[int], cur_pos: Tensor) -> None:
        off = 0
        for mol_i in cur_sub_ids:
            n = all_entries[mol_i]["n"]
            pos_per_mol[mol_i] = cur_pos[off : off + n].detach().clone()
            off += n

    def _eval_ef(cur_pos: Tensor) -> tuple[Tensor, Tensor]:
        nonlocal b, csr_plan, group_ptr, pos_ref
        need_rebuild = True if eff_skin <= 0.0 else (float(((cur_pos - pos_ref) ** 2).sum(dim=-1).max().item()) > (0.5 * eff_skin) ** 2)
        if need_rebuild:
            writeback_positions(sub_ids, cur_pos)
            b, csr_plan, group_ptr, pos_ref = _build_subbatch(calc, entries, [pos_per_mol[i] for i in sub_ids], r_list)
        e_k, f_e = _eval_mol_subbatch_ef(calc, b, cur_pos, csr_plan, group_ptr, entries, need_forces=True)
        assert f_e is not None
        if rest_k_t is not None and rest_ref_t is not None:
            dr_rest = cur_pos - rest_ref_t
            e_rest_atom_kcal = (0.5 * rest_k_t * (dr_rest * dr_rest).sum(dim=-1)) / MOL_KCAL_TO_EV
            e_k = e_k.index_add(0, b.graph, e_rest_atom_kcal.to(dtype=e_k.dtype))
            f_e = f_e - rest_k_t.unsqueeze(-1) * dr_rest
        if move_mask_t is not None:
            f_e = f_e.masked_fill(~move_mask_t.unsqueeze(-1), 0.0)
        return e_k, f_e

    c1_armijo = 1e-4
    eps_e = 1e-5 if dt == torch.float32 else 1e-10
    max_ls = 5
    e_kcal_t, f_ev_t = _eval_ef(pos)

    for step_idx in range(max_steps + 1):
        B = len(sub_ids)
        if step_idx > 0:
            n_inactive = int((~act_np).sum())
            want_compact = (n_inactive >= max(8, B // 3)) and (n_inactive < B)
            if want_compact:
                writeback_positions(sub_ids, pos)
                keep_local = [k for k, ok in enumerate(act_np.tolist()) if ok]
                keep_mask_atom = active.index_select(0, b.graph)
                sub_ids = [sub_ids[k] for k in keep_local]
                entries = [all_entries[i] for i in sub_ids]
                e_ref_np = np.array([e["e_ref_kcal"] for e in entries], dtype=np.float64)
                b, csr_plan, group_ptr, pos_ref = _build_subbatch(calc, entries, [pos_per_mol[i] for i in sub_ids], r_list)
                move_mask_t, rest_k_t, rest_ref_t = _build_subbatch_constraints(entries, dev, dt)
                pos = b.pos.clone()
                B = len(sub_ids)
                active = torch.ones(B, dtype=torch.bool, device=dev)
                act_np = np.ones(B, dtype=bool)
                keep_t = torch.tensor(keep_local, dtype=torch.long, device=dev)
                e_kcal_t = e_kcal_t.index_select(0, keep_t).contiguous()
                f_ev_t = f_ev_t[keep_mask_atom].contiguous()
                if pos_prev is not None and grad_prev is not None:
                    pos_prev = pos_prev[keep_mask_atom].contiguous()
                    grad_prev = grad_prev[keep_mask_atom].contiguous()
                s_hist = [s[keep_mask_atom].contiguous() for s in s_hist]
                y_hist = [y[keep_mask_atom].contiguous() for y in y_hist]
                rho_hist = [r.index_select(0, keep_t).contiguous() for r in rho_hist]

        f_atom_norm = f_ev_t.norm(dim=-1)
        fmax_t = torch.zeros(B, dtype=f_atom_norm.dtype, device=dev).scatter_reduce_(
            0, b.graph, f_atom_norm, reduce="amax", include_self=False
        )
        newly_conv = active & (fmax_t <= fmax_thresh)

        # Single packed host sync per step for energies + fmax + newly-converged flags!
        efc_np = torch.cat([e_kcal_t, fmax_t.to(e_kcal_t.dtype), newly_conv.to(e_kcal_t.dtype)]).detach().cpu().to(torch.float64).numpy()
        e_np = (efc_np[:B] + e_ref_np) * MOL_KCAL_TO_EV
        fm_np = efc_np[B : 2 * B]
        nc_np = efc_np[2 * B :] > 0.5

        if step_idx == 0:
            for k, mol_i in enumerate(sub_ids):
                init_e[mol_i] = float(e_np[k])
                final_e[mol_i] = float(e_np[k])
                final_fmax[mol_i] = float(fm_np[k])

        for k, mol_i in enumerate(sub_ids):
            if act_np[k]:
                final_e[mol_i] = float(e_np[k])
                final_fmax[mol_i] = float(fm_np[k])
                if nc_np[k]:
                    converged[mol_i] = True

        active = active & (~newly_conv)
        prev_act_np = act_np
        act_np = act_np & (~nc_np)
        if not bool(act_np.any()) or step_idx == max_steps:
            break

        for k, mol_i in enumerate(sub_ids):
            if prev_act_np[k] and not nc_np[k]:
                step_counts[mol_i] += 1

        grad = -f_ev_t
        if pos_prev is not None and grad_prev is not None:
            sk = pos - pos_prev
            yk = grad - grad_prev
            sy = bdot(sk, yk, b.graph, B)
            rho_k = torch.where(sy > 1e-7, 1.0 / sy.clamp_min(1e-12), torch.zeros_like(sy))
            if len(s_hist) >= history_size:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(sk)
            y_hist.append(yk)
            rho_hist.append(rho_k)

        pos_prev = pos.clone()
        grad_prev = grad.clone()

        q_vec = grad.to(acc_dt)
        alphas: list[Tensor] = []
        kh = len(s_hist)
        for i_h in reversed(range(kh)):
            alpha_i = rho_hist[i_h] * bdot(s_hist[i_h], q_vec, b.graph, B)
            alphas.append(alpha_i)
            q_vec = q_vec - alpha_i.index_select(0, b.graph).unsqueeze(-1) * y_hist[i_h].to(acc_dt)
        alphas.reverse()

        if kh > 0:
            sy_last = bdot(s_hist[-1], y_hist[-1], b.graph, B)
            yy_last = bdot(y_hist[-1], y_hist[-1], b.graph, B).clamp_min(1e-12)
            gamma = torch.clamp(sy_last / yy_last, min=1e-4, max=1.0)
            r_vec = gamma.index_select(0, b.graph).unsqueeze(-1) * q_vec
        else:
            r_vec = 0.01 * q_vec

        for i_h in range(kh):
            beta_i = rho_hist[i_h] * bdot(y_hist[i_h], r_vec, b.graph, B)
            r_vec = r_vec + (alphas[i_h] - beta_i).index_select(0, b.graph).unsqueeze(-1) * s_hist[i_h].to(acc_dt)

        dpos = (-r_vec).to(dt)
        # Descent-direction guard: per-molecule fallback to damped steepest descent if dpos . grad >= 0 or non-finite
        g_dot_d = bdot(grad, dpos, b.graph, B)
        bad_dir = active & ((~torch.isfinite(g_dot_d)) | (g_dot_d >= 0.0))
        if bool(bad_dir.any().item()):
            dpos_sd = (-0.01 * grad).to(dt)
            bad_atom = bad_dir.index_select(0, b.graph).unsqueeze(-1)
            dpos = torch.where(bad_atom, dpos_sd, dpos)
            for i_h in range(len(rho_hist)):
                rho_hist[i_h] = torch.where(bad_dir, torch.zeros_like(rho_hist[i_h]), rho_hist[i_h])
            g_dot_d = bdot(grad, dpos, b.graph, B)

        d_norm = dpos.norm(dim=-1)
        steplen = torch.zeros(B, dtype=d_norm.dtype, device=dev).scatter_reduce_(
            0, b.graph, d_norm, reduce="amax", include_self=False
        )
        scale = torch.clamp(maxstep / steplen.clamp_min(1e-8), max=1.0) if maxstep > 0.0 else torch.zeros_like(steplen)
        dpos = dpos * scale.index_select(0, b.graph).unsqueeze(-1)
        g_dot_d = g_dot_d * scale.to(acc_dt)

        # Per-molecule independent Armijo backtracking line search (zero extra evals when alpha=1 accepts)
        e_cur_ev = e_kcal_t.to(acc_dt) * MOL_KCAL_TO_EV
        alpha_t = torch.ones(B, dtype=acc_dt, device=dev)
        pending = active.clone()
        pos_acc = pos.clone()
        e_acc = e_kcal_t.clone()
        f_acc = f_ev_t.clone()

        for _ls in range(max_ls + 1):
            pend_atom = pending.index_select(0, b.graph).unsqueeze(-1)
            if move_mask_t is not None:
                pend_atom = pend_atom & move_mask_t.unsqueeze(-1)
            a_atom = alpha_t.to(dt).index_select(0, b.graph).unsqueeze(-1)
            pos_try = torch.where(pend_atom, pos + a_atom * dpos, pos_acc)
            e_try_kcal, f_try_ev = _eval_ef(pos_try)
            e_try_ev = e_try_kcal.to(acc_dt) * MOL_KCAL_TO_EV
            armijo_ok = torch.isfinite(e_try_ev) & (e_try_ev <= e_cur_ev + c1_armijo * alpha_t * g_dot_d + eps_e)
            newly_acc = pending & armijo_ok
            if bool(newly_acc.any().item()):
                na_atom = newly_acc.index_select(0, b.graph).unsqueeze(-1)
                pos_acc = torch.where(na_atom, pos_try, pos_acc)
                e_acc = torch.where(newly_acc, e_try_kcal, e_acc)
                f_acc = torch.where(na_atom, f_try_ev, f_acc)
                pending = pending & (~newly_acc)
            if not bool(pending.any().item()):
                break
            alpha_t = torch.where(pending, 0.5 * alpha_t, alpha_t)

        # Fallback for any molecule still failing Armijo after backtracking: reset curvature + tiny damped SD step
        if bool(pending.any().item()):
            for i_h in range(len(rho_hist)):
                rho_hist[i_h] = torch.where(pending, torch.zeros_like(rho_hist[i_h]), rho_hist[i_h])
            sd_step = (-0.002 * grad).to(dt)
            sd_norm = sd_step.norm(dim=-1)
            sd_len = torch.zeros(B, dtype=sd_norm.dtype, device=dev).scatter_reduce_(
                0, b.graph, sd_norm, reduce="amax", include_self=False
            )
            sd_cap = min(maxstep, 0.02) if maxstep > 0.0 else 0.0
            sd_scale = torch.clamp(sd_cap / sd_len.clamp_min(1e-8), max=1.0) if sd_cap > 0.0 else torch.zeros_like(sd_len)
            sd_step = sd_step * sd_scale.index_select(0, b.graph).unsqueeze(-1)
            pend_atom = pending.index_select(0, b.graph).unsqueeze(-1)
            if move_mask_t is not None:
                pend_atom = pend_atom & move_mask_t.unsqueeze(-1)
            pos_try = torch.where(pend_atom, pos + sd_step, pos_acc)
            e_try_kcal, f_try_ev = _eval_ef(pos_try)
            e_try_ev = e_try_kcal.to(acc_dt) * MOL_KCAL_TO_EV
            sd_ok = pending & torch.isfinite(e_try_ev) & (e_try_ev <= e_cur_ev + eps_e)
            if bool(sd_ok.any().item()):
                ok_atom = sd_ok.index_select(0, b.graph).unsqueeze(-1)
                pos_acc = torch.where(ok_atom, pos_try, pos_acc)
                e_acc = torch.where(sd_ok, e_try_kcal, e_acc)
                f_acc = torch.where(ok_atom, f_try_ev, f_acc)

        pos, e_kcal_t, f_ev_t = pos_acc, e_acc, f_acc

    writeback_positions(sub_ids, pos)
    out_list: list[dict[str, Any]] = []
    for i in range(B0):
        p_out = pos_per_mol[i].detach().cpu().to(torch.float64).numpy()
        mm = all_entries[i]["move_mask_np"]
        if mm is not None:
            orig_p64 = np.asarray(all_entries[i]["atoms0"].get_positions(), dtype=np.float64)
            p_out[~mm] = orig_p64[~mm]
        at_out = all_entries[i]["atoms0"].copy()
        at_out.set_positions(p_out)
        out_list.append({
            "atoms": at_out,
            "positions": p_out,
            "initial_energy": float(init_e[i]),
            "final_energy": float(final_e[i]),
            "final_fmax": float(final_fmax[i]),
            "steps": int(step_counts[i]),
            "converged": bool(converged[i]),
        })
    return out_list


def _relax_single_ase(
    calc: "MenaCalculator",
    atoms: Atoms,
    *,
    fmax_thresh: float,
    max_steps: int,
    maxstep: float,
) -> dict[str, Any]:
    at = atoms.copy()
    rest_k_np, rest_ref_np = _parse_restraints(at)
    at.calc = _RestrainedCalc(calc, rest_k_np, rest_ref_np) if rest_k_np is not None else calc
    e0 = float(at.get_potential_energy())
    opt = LBFGS(at, maxstep=maxstep, logfile=None)
    conv = bool(opt.run(fmax=fmax_thresh, steps=max_steps))
    e1 = float(at.get_potential_energy())
    f1 = np.asarray(at.get_forces(), dtype=np.float64)
    fm1 = float(np.linalg.norm(f1, axis=-1).max()) if len(f1) else 0.0
    p_out = np.asarray(at.get_positions(), dtype=np.float64).copy()
    at.calc = None
    return {
        "atoms": at,
        "positions": p_out,
        "initial_energy": e0,
        "final_energy": e1,
        "final_fmax": fm1,
        "steps": int(opt.nsteps),
        "converged": conv,
    }


def relax_batch_frames(
    calc: "MenaCalculator",
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
    """Relax a list of structures using Tier-2 GPU-resident sparse-batched L-BFGS."""
    n_tot = len(atoms_list)
    if n_tot == 0:
        return []

    mol_groups: dict[tuple[str, bool, str | None], list[int]] = {}
    other_indices: list[int] = []
    for idx, at in enumerate(atoms_list):
        src, _ = resolve_lot_and_u(at, calc.lot, calc.u_override)
        pbc_any = bool(np.any(at.get_pbc()))
        only_fix_atoms, _ = _parse_move_mask(at)
        if src.domain == "molecule" and not pbc_any and len(at) > 0 and only_fix_atoms:
            d3_mode = src.d3_addback if calc.add_dispersion else None
            mol_groups.setdefault((src.domain, False, d3_mode), []).append(idx)
        else:
            other_indices.append(idx)

    results: list[dict[str, Any] | None] = [None] * n_tot

    for _, mol_indices in mol_groups.items():
        mol_sorted = sorted(mol_indices, key=lambda idx: len(atoms_list[idx]))
        cur_idx: list[int] = []
        cur_atoms = 0
        for idx in mol_sorted:
            na = len(atoms_list[idx])
            if cur_idx and (len(cur_idx) >= max_batch or cur_atoms + na > max_atoms):
                sub_out = _relax_open_mol_microbatch(
                    calc, [atoms_list[j] for j in cur_idx],
                    fmax_thresh=fmax, max_steps=steps, maxstep=maxstep, skin=skin, history_size=history_size,
                )
                for j, r in zip(cur_idx, sub_out):
                    results[j] = r
                cur_idx, cur_atoms = [], 0
            cur_idx.append(idx)
            cur_atoms += na
        if cur_idx:
            sub_out = _relax_open_mol_microbatch(
                calc, [atoms_list[j] for j in cur_idx],
                fmax_thresh=fmax, max_steps=steps, maxstep=maxstep, skin=skin, history_size=history_size,
            )
            for j, r in zip(cur_idx, sub_out):
                results[j] = r

    for idx in other_indices:
        results[idx] = _relax_single_ase(calc, atoms_list[idx], fmax_thresh=fmax, max_steps=steps, maxstep=maxstep)

    return [r for r in results if r is not None]
