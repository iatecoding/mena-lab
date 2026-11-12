"""Tier-2 GPU-resident sparse-batched L-BFGS geometry optimizer for MenaCalculator.

Combines:
  - Size-sorted microbatching across arbitrary lists of molecules,
  - Verlet skin (`r_cut + skin`) neighbor-list & `CSRPlan` reuse on GPU (rebuilt only when max displacement > skin / 2),
  - Tier-2 Dual-CSR + `MetalExactCoulomb` + `metal_d3_grouped` (`O(N + E)` memory, zero host sync inside force eval),
  - Two-loop L-BFGS (`history_size=8`, exact CPU float64 reference energies, per-graph step-length cap `maxstep`) on device,
  - Dynamic batch compaction when >= 1/3 of active structures converge so stragglers never waste GPU FLOPs.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence

from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.optimize import LBFGS
import numpy as np
import torch
from torch import Tensor

from lam4.calc.physics import (
    MOL_KCAL_TO_EV,
    resolve_charge_and_spin,
    resolve_lot_and_u,
)
from lam4.calc.relax_batch_io import (
    build_mol_subbatch as _build_subbatch,
    eval_mol_subbatch_ef as _eval_mol_subbatch_ef,
)
from lam4.data.buckets import source_ids

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
        p_np = np.asarray(at.get_positions(), dtype=np.float32)
        src, u_val = resolve_lot_and_u(at, calc.lot, calc.u_override)
        q_val, s_val = resolve_charge_and_spin(at, calc.charge_override, calc.spin_override, pbc_any=False)
        tok_id, ref_id = source_ids(src.name)
        e0_k, eqs_k, _ = calc._ref_energies_kcal(src.name, src.domain, False, z_np, q_val, s_val)
        all_entries.append({
            "orig_idx": idx,
            "atoms0": at,
            "n": len(z_np),
            "z_np": z_np,
            "z_t": torch.from_numpy(np.ascontiguousarray(z_np, dtype=np.int64)).to(dev),
            "q": q_val,
            "s": s_val,
            "u": u_val,
            "tok": tok_id,
            "ref": ref_id,
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
    pos = b.pos.clone()
    active = torch.ones(len(sub_ids), dtype=torch.bool, device=dev)

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

    for step_idx in range(max_steps + 1):
        B = len(sub_ids)
        if step_idx > 0:
            if eff_skin <= 0.0:
                need_rebuild = True
            else:
                disp2_max = float(((pos - pos_ref) ** 2).sum(dim=-1).max().item())
                need_rebuild = disp2_max > (0.5 * eff_skin) ** 2
            n_inactive = int((~active).sum().item())
            want_compact = (n_inactive >= max(8, B // 3)) and (n_inactive < B)
            if want_compact:
                writeback_positions(sub_ids, pos)
                act_cpu = active.detach().cpu().tolist()
                keep_local = [k for k, ok in enumerate(act_cpu) if ok]
                keep_mask_atom = active.index_select(0, b.graph)
                sub_ids = [sub_ids[k] for k in keep_local]
                entries = [all_entries[i] for i in sub_ids]
                e_ref_np = np.array([e["e_ref_kcal"] for e in entries], dtype=np.float64)
                b, csr_plan, group_ptr, pos_ref = _build_subbatch(calc, entries, [pos_per_mol[i] for i in sub_ids], r_list)
                pos = b.pos.clone()
                B = len(sub_ids)
                active = torch.ones(B, dtype=torch.bool, device=dev)
                if pos_prev is not None and grad_prev is not None:
                    pos_prev = pos_prev[keep_mask_atom].contiguous()
                    grad_prev = grad_prev[keep_mask_atom].contiguous()
                keep_t = torch.tensor(keep_local, dtype=torch.long, device=dev)
                s_hist = [s[keep_mask_atom].contiguous() for s in s_hist]
                y_hist = [y[keep_mask_atom].contiguous() for y in y_hist]
                rho_hist = [r.index_select(0, keep_t).contiguous() for r in rho_hist]
            elif need_rebuild:
                writeback_positions(sub_ids, pos)
                b, csr_plan, group_ptr, pos_ref = _build_subbatch(calc, entries, [pos_per_mol[i] for i in sub_ids], r_list)
                pos = b.pos.clone()

        e_kcal_t, f_ev_t = _eval_mol_subbatch_ef(calc, b, pos, csr_plan, group_ptr, entries, need_forces=True)
        assert f_ev_t is not None
        f_atom_norm = f_ev_t.norm(dim=-1)
        fmax_t = torch.zeros(B, dtype=f_atom_norm.dtype, device=dev).scatter_reduce_(
            0, b.graph, f_atom_norm, reduce="amax", include_self=False
        )

        # Single packed host sync per step for energies + fmax
        ef_np = torch.cat([e_kcal_t, fmax_t]).detach().cpu().to(torch.float64).numpy()
        e_np = (ef_np[:B] + e_ref_np) * MOL_KCAL_TO_EV
        fm_np = ef_np[B:]
        act_np = active.detach().cpu().numpy()

        if step_idx == 0:
            for k, mol_i in enumerate(sub_ids):
                init_e[mol_i] = float(e_np[k])
                final_e[mol_i] = float(e_np[k])
                final_fmax[mol_i] = float(fm_np[k])

        newly_conv = active & (fmax_t <= fmax_thresh)
        nc_np = newly_conv.detach().cpu().numpy()
        for k, mol_i in enumerate(sub_ids):
            if act_np[k]:
                final_e[mol_i] = float(e_np[k])
                final_fmax[mol_i] = float(fm_np[k])
                if nc_np[k]:
                    converged[mol_i] = True

        active = active & (~newly_conv)
        if not bool(active.any().item()) or step_idx == max_steps:
            break

        for k, mol_i in enumerate(sub_ids):
            if act_np[k] and not nc_np[k]:
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
        d_norm = dpos.norm(dim=-1)
        steplen = torch.zeros(B, dtype=d_norm.dtype, device=dev).scatter_reduce_(
            0, b.graph, d_norm, reduce="amax", include_self=False
        )
        scale = torch.clamp(maxstep / steplen.clamp_min(1e-8), max=1.0)
        dpos = dpos * scale.index_select(0, b.graph).unsqueeze(-1)
        act_atom = active.index_select(0, b.graph).unsqueeze(-1)
        pos = torch.where(act_atom, pos + dpos, pos)

    writeback_positions(sub_ids, pos)
    out_list: list[dict[str, Any]] = []
    for i in range(B0):
        p_out = pos_per_mol[i].detach().cpu().to(torch.float64).numpy()
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
    at.calc = calc
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

    mol_indices: list[int] = []
    other_indices: list[int] = []
    for idx, at in enumerate(atoms_list):
        src, _ = resolve_lot_and_u(at, calc.lot, calc.u_override)
        if src.domain == "molecule" and not bool(np.any(at.get_pbc())) and len(at) > 0:
            mol_indices.append(idx)
        else:
            other_indices.append(idx)

    results: list[dict[str, Any] | None] = [None] * n_tot

    if mol_indices:
        mol_sorted = sorted(mol_indices, key=lambda idx: len(atoms_list[idx]))
        cur_idx: list[int] = []
        cur_atoms = 0
        for idx in mol_sorted:
            na = len(atoms_list[idx])
            if cur_idx and (len(cur_idx) >= max_batch or cur_atoms + na > max_atoms):
                sub_out = _relax_open_mol_microbatch(
                    calc,
                    [atoms_list[j] for j in cur_idx],
                    fmax_thresh=fmax,
                    max_steps=steps,
                    maxstep=maxstep,
                    skin=skin,
                    history_size=history_size,
                )
                for j, r in zip(cur_idx, sub_out):
                    results[j] = r
                cur_idx = []
                cur_atoms = 0
            cur_idx.append(idx)
            cur_atoms += na
        if cur_idx:
            sub_out = _relax_open_mol_microbatch(
                calc,
                [atoms_list[j] for j in cur_idx],
                fmax_thresh=fmax,
                max_steps=steps,
                maxstep=maxstep,
                skin=skin,
                history_size=history_size,
            )
            for j, r in zip(cur_idx, sub_out):
                results[j] = r

    for idx in other_indices:
        results[idx] = _relax_single_ase(calc, atoms_list[idx], fmax_thresh=fmax, max_steps=steps, maxstep=maxstep)

    return [r for r in results if r is not None]
