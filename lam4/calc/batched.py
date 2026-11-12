"""Batched evaluation & summary formatting helpers for MenaCalculator."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from ase import Atoms
from ase.calculators.calculator import all_changes
import numpy as np
import torch

from lam4.calc.physics import (
    CRYSTAL_EV,
    E_ANG_TO_DEBYE,
    MOL_KCAL_TO_EV,
    _get_d3,
    build_edge_tensors,
    build_open_mol_edges_gpu,
    effective_periodic_cell,
    eval_dispersion_single,
    polarizability_from_sums,
    resolve_charge_and_spin,
    resolve_lot_and_u,
)
from lam4.data.batch import GraphBatch
from lam4.data.buckets import source_ids
from lam4.kernels.csr_plan import build_csr_plan

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator


def _eval_homogeneous_microbatch(
    calc: "MenaCalculator",
    atoms_list: Sequence[Atoms],
    properties: Sequence[str],
) -> list[dict[str, Any]]:
    """Evaluate a single homogeneous `(domain, pbc)` microbatch in ONE GPU/CPU pass."""
    if len(atoms_list) == 1 or (calc.low_memory and calc.model.trunk.edge_mode != "csr"):
        outs = []
        for a in atoms_list:
            cp = a.copy()
            calc.calculate(cp, properties=properties, system_changes=all_changes)
            outs.append(dict(calc.results))
        return outs

    first_src, _ = resolve_lot_and_u(atoms_list[0], calc.lot, calc.u_override)
    first_pbc = bool(np.any(atoms_list[0].get_pbc()))
    r_cut = float(calc.model.cfg.r_cut)
    np_dt = np.float32 if calc.dtype == torch.float32 else np.float64

    # Check if all frames are small displacements of frame 0 (e.g. finite-difference Vibrations/IR/Raman)
    z0 = np.asarray(atoms_list[0].get_atomic_numbers(), dtype=np.int64)
    p0 = np.asarray(atoms_list[0].get_positions(), dtype=np.float64)
    c0, pbc0 = effective_periodic_cell(atoms_list[0], r_cut) if first_pbc else (None, (False, False, False))
    shared_ei_c = shared_es_c = None
    pos_all = [p0]
    same_topo = len(z0) > 0
    max_disp = 0.0
    for at in atoms_list[1:]:
        zi = np.asarray(at.get_atomic_numbers(), dtype=np.int64)
        pi = np.asarray(at.get_positions(), dtype=np.float64)
        pos_all.append(pi)
        if not same_topo or len(zi) != len(z0) or not np.array_equal(zi, z0) or bool(np.any(at.get_pbc())) != first_pbc:
            same_topo = False
            continue
        if first_pbc:
            ci, pbci = effective_periodic_cell(at, r_cut)
            if pbci != pbc0 or not np.allclose(ci, c0, atol=1e-12, rtol=0.0):
                same_topo = False
                continue
        dr = pi - p0
        max_disp = max(max_disp, float(np.sqrt((dr * dr).sum(axis=-1).max())))
        if max_disp > 0.25:
            same_topo = False
    if same_topo:
        shared_ei_c, shared_es_c = build_edge_tensors(p0, c0, pbc0, r_cut + 2.0 * max_disp + 1e-3, calc.dtype)

    dev, dt = calc.device, calc.dtype
    zs, poss, graphs, cells, eis, ess, qs, ss, us, toks, refs = [], [], [], [], [], [], [], [], [], [], []
    counts: list[int] = []
    meta_per_frame = []
    offset = 0
    max_n = max((len(p) for p in pos_all), default=0)
    use_gpu_nl = (not first_pbc) and (dev.type in ("mps", "cuda")) and (max_n <= 256) and (shared_ei_c is None)
    for g_idx, at in enumerate(atoms_list):
        src, u_val = resolve_lot_and_u(at, calc.lot, calc.u_override)
        pbc_any = bool(np.any(at.get_pbc()))
        z_np = z0 if same_topo else np.asarray(at.get_atomic_numbers(), dtype=np.int64)
        pos_np = pos_all[g_idx]
        q_val, s_val = resolve_charge_and_spin(at, calc.charge_override, calc.spin_override, pbc_any=pbc_any)
        cell_np, pbc_flags = (c0, pbc0) if same_topo else (effective_periodic_cell(at, r_cut) if pbc_any else (None, (False, False, False)))
        tok_id, ref_id = source_ids(src.name)
        n = len(z_np)
        counts.append(n)
        zs.append(torch.from_numpy(np.ascontiguousarray(z_np, dtype=np.int64)))
        poss.append(torch.from_numpy(np.ascontiguousarray(pos_np, dtype=np_dt)))
        graphs.append(torch.full((n,), g_idx, dtype=torch.long))
        if first_pbc:
            cells.append(torch.from_numpy(np.ascontiguousarray(cell_np, dtype=np_dt)) if cell_np is not None else torch.zeros((3, 3), dtype=calc.dtype))
        if not use_gpu_nl:
            ei_c, es_c = (shared_ei_c, shared_es_c) if shared_ei_c is not None else build_edge_tensors(pos_np, cell_np, pbc_flags, r_cut, calc.dtype)
            eis.append(ei_c + offset)
            ess.append(es_c)
        qs.append(q_val)
        ss.append(float(s_val))
        us.append(u_val)
        toks.append(tok_id)
        refs.append(ref_id)
        meta_per_frame.append((src, u_val, q_val, s_val, z_np, pos_np, cell_np, pbc_flags, offset, offset + n))
        offset += n

    G = len(atoms_list)
    pos_dev = torch.cat(poss).to(dev)
    graph_dev = torch.cat(graphs).to(dev)
    cell_dev = torch.stack(cells).to(dev) if first_pbc else torch.zeros((G, 3, 3), dtype=dt, device=dev)
    if use_gpu_nl:
        ei_dev, es_dev = build_open_mol_edges_gpu(pos_dev, graph_dev, counts, r_cut, dt)
    else:
        ei_dev = torch.cat(eis, dim=1).to(dev)
        es_dev = torch.cat(ess, dim=0).to(dev)
    b = GraphBatch(
        z=torch.cat(zs).to(dev),
        pos=pos_dev,
        graph=graph_dev,
        cell=cell_dev,
        edge_index=ei_dev,
        edge_shift=es_dev,
        charge=torch.tensor(qs, dtype=dt, device=dev),
        spin=torch.tensor(ss, dtype=dt, device=dev),
        u_flag=torch.tensor(us, dtype=torch.long, device=dev),
        token=torch.tensor(toks, dtype=torch.long, device=dev),
        ref_id=torch.tensor(refs, dtype=torch.long, device=dev),
        domain=first_src.domain,
        atom_mask=torch.ones(offset, dtype=torch.bool, device=dev),
        graph_mask=torch.ones(G, dtype=torch.bool, device=dev),
        energy=torch.zeros(G, dtype=dt, device=dev),
        forces=torch.zeros((offset, 3), dtype=dt, device=dev),
        energy_mask=torch.ones(G, dtype=torch.bool, device=dev),
        force_mask=torch.ones(offset, dtype=torch.bool, device=dev),
        coulomb_cap=max(64, max_n) if (first_src.domain == "molecule" and not first_pbc) else None,
        pbc=first_pbc,
    )
    need_forces = "forces" in properties
    need_stress = "stress" in properties and first_pbc and all(abs(float(np.linalg.det(m[6]))) > 1e-8 for m in meta_per_frame)
    csr_plan = build_csr_plan(b.edge_index[1], b.edge_index[0], offset) if calc.model.trunk.edge_mode == "csr" else None
    out = calc._run_model(b, None, need_forces=need_forces, need_stress=need_stress, csr_plan=csr_plan)
    stress_all = out["stress"].detach().cpu().to(torch.float64).numpy() if (need_stress and out.get("stress") is not None) else None

    batched_d3_e = batched_d3_f_flat = batched_d3_f_pad = None
    all_bj = calc.add_dispersion and not first_pbc and all(m[0].d3_addback == "bj_wb97m" for m in meta_per_frame)
    if all_bj and dev.type == "mps" and dt == torch.float32 and max_n <= 1024:
        from lam4.kernels.metal_d3 import has_metal_d3, metal_d3_grouped

        if has_metal_d3():
            ed3_t, fd3_t = metal_d3_grouped(b.z, b.pos, b.graph, G, compute_forces=need_forces)
            batched_d3_e = ed3_t.detach().cpu().to(torch.float64).numpy()
            if need_forces and fd3_t is not None:
                batched_d3_f_flat = fd3_t.detach().cpu().to(torch.float64).numpy()
    elif all_bj and max_n <= 256:
        d3 = _get_d3(dev, dt)
        z_pad_cpu = torch.zeros((G, max_n), dtype=torch.long)
        p_pad_cpu = torch.zeros((G, max_n, 3), dtype=dt)
        m_pad_cpu = torch.zeros((G, max_n), dtype=torch.bool)
        for g_i, m in enumerate(meta_per_frame):
            ni = len(m[4])
            z_pad_cpu[g_i, :ni] = zs[g_i]
            p_pad_cpu[g_i, :ni] = poss[g_i]
            m_pad_cpu[g_i, :ni] = True
        z_pad, p_pad, m_pad = z_pad_cpu.to(dev), p_pad_cpu.to(dev), m_pad_cpu.to(dev)
        if need_forces:
            p_pad = p_pad.detach().requires_grad_(True)
        with torch.set_grad_enabled(need_forces):
            ed3 = d3(z_pad, p_pad, m_pad)
            if need_forces:
                batched_d3_f_pad = (-torch.autograd.grad(ed3.sum(), p_pad)[0]).detach().cpu().to(torch.float64).numpy()
        batched_d3_e = ed3.detach().cpu().to(torch.float64).numpy()

    e_res_all = out["energy"].detach().cpu().to(torch.float64).numpy()
    f_res_all = out["forces"].detach().cpu().to(torch.float64).numpy() if (need_forces and out.get("forces") is not None) else None
    dip_all = out["dipole"].detach().cpu().to(torch.float64).numpy() if out.get("dipole") is not None else None
    q_all = out["charges"].detach().cpu().to(torch.float64).numpy() if out.get("charges") is not None else None
    alpha_iso_np = out["alpha"].detach().cpu().to(torch.float64).numpy() if out.get("alpha") is not None else None
    alpha_an_np = out["alpha_aniso"].detach().cpu().to(torch.float64).numpy() if out.get("alpha_aniso") is not None else None

    ref_cache: dict[tuple[Any, ...], tuple[float, float]] = {}
    results_list: list[dict[str, Any]] = []
    for g_i, (src, u_val, q_val, s_val, z_np, pos_np, cell_np, pbc_flags, s0, s1) in enumerate(meta_per_frame):
        kcal_to_ev = (1.0 / CRYSTAL_EV) if src.domain == "crystal" else MOL_KCAL_TO_EV
        if batched_d3_e is not None:
            d3_e = float(batched_d3_e[g_i])
            if batched_d3_f_flat is not None:
                d3_f = batched_d3_f_flat[s0:s1]
            elif batched_d3_f_pad is not None:
                d3_f = batched_d3_f_pad[g_i, : s1 - s0]
            else:
                d3_f = None
        else:
            d3_mode = src.d3_addback if calc.add_dispersion else None
            d3_e, d3_f, _ = eval_dispersion_single(z_np, pos_np, cell_np, pbc_flags, d3_mode, dev, dt, compute_forces=need_forces)
        rkey = (src.name, z_np.tobytes(), float(q_val), int(s_val))
        if rkey not in ref_cache:
            e0_k, eqs_k, _ = calc._ref_energies_kcal(src.name, src.domain, first_pbc, z_np, q_val, s_val)
            ref_cache[rkey] = (e0_k, eqs_k)
        e0_kcal, eqs_kcal = ref_cache[rkey]
        e_ev = float((e_res_all[g_i] + d3_e + e0_kcal + eqs_kcal) * kcal_to_ev)
        f_ev = np.zeros((s1 - s0, 3), dtype=np.float64)
        if f_res_all is not None:
            fk = f_res_all[s0:s1] + (d3_f if d3_f is not None else 0.0)
            f_ev = fk * kcal_to_ev
        dip_ea = dip_all[g_i] if dip_all is not None else np.zeros(3, dtype=np.float64)
        if alpha_iso_np is not None:
            pol_mat, pol_i, pol_a = polarizability_from_sums(
                float(alpha_iso_np[s0:s1].sum()),
                alpha_an_np[s0:s1].sum(axis=0) if alpha_an_np is not None else None,
            )
        else:
            pol_mat, pol_i, pol_a = np.zeros((3, 3), dtype=np.float64), 0.0, 0.0
        stress_voigt = (stress_all[g_i][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] * kcal_to_ev) if stress_all is not None else None
        results_list.append({
            "energy": e_ev,
            "free_energy": e_ev,
            "forces": f_ev,
            **({"stress": stress_voigt} if stress_voigt is not None else {}),
            "dipole": dip_ea,
            "dipole_debye": dip_ea * E_ANG_TO_DEBYE,
            "polarizability": pol_mat,
            "polarizability_iso": pol_i,
            "polarizability_aniso": pol_a,
            "charges": q_all[s0:s1] if q_all is not None else np.zeros(s1 - s0, dtype=np.float64),
            "atomic_polarizabilities": alpha_iso_np[s0:s1] if alpha_iso_np is not None else np.zeros(s1 - s0, dtype=np.float64),
            "lot_used": src.name,
            "u_used": int(u_val),
        })
    return results_list


def eval_batch_frames(
    calc: "MenaCalculator",
    atoms_list: Sequence[Atoms],
    properties: Sequence[str] = ("energy", "forces", "dipole", "polarizability"),
    *,
    max_atoms: int = 4096,
    max_batch: int = 128,
) -> list[dict[str, Any]]:
    """Evaluate an arbitrary list of structures with size-sorted Tier-2 microbatching."""
    n_tot_frames = len(atoms_list)
    if n_tot_frames == 0:
        return []
    if n_tot_frames == 1 or (calc.low_memory and calc.model.trunk.edge_mode != "csr"):
        return _eval_homogeneous_microbatch(calc, atoms_list, properties)

    # Group by (domain, pbc) and check if everything already fits in a single microbatch
    groups: dict[tuple[str, bool], list[int]] = {}
    tot_atoms = 0
    for idx, at in enumerate(atoms_list):
        src, _ = resolve_lot_and_u(at, calc.lot, calc.u_override)
        pbc_any = bool(np.any(at.get_pbc()))
        groups.setdefault((src.domain, pbc_any), []).append(idx)
        tot_atoms += len(at)

    if len(groups) == 1 and n_tot_frames <= max_batch and tot_atoms <= max_atoms:
        return _eval_homogeneous_microbatch(calc, atoms_list, properties)

    out_all: list[dict[str, Any] | None] = [None] * n_tot_frames
    for _, indices in groups.items():
        # Sort by atom count so microbatches have tight sizes and minimal cache churn
        indices_sorted = sorted(indices, key=lambda idx: len(atoms_list[idx]))
        cur_idx: list[int] = []
        cur_atoms = 0
        for idx in indices_sorted:
            na = len(atoms_list[idx])
            if cur_idx and (len(cur_idx) >= max_batch or cur_atoms + na > max_atoms):
                sub_res = _eval_homogeneous_microbatch(calc, [atoms_list[j] for j in cur_idx], properties)
                for j, r in zip(cur_idx, sub_res):
                    out_all[j] = r
                cur_idx = []
                cur_atoms = 0
            cur_idx.append(idx)
            cur_atoms += na
        if cur_idx:
            sub_res = _eval_homogeneous_microbatch(calc, [atoms_list[j] for j in cur_idx], properties)
            for j, r in zip(cur_idx, sub_res):
                out_all[j] = r
    return [r for r in out_all if r is not None]


def format_calc_summary(results: dict[str, Any], device: torch.device) -> str:
    if not results:
        return "MenaCalculator: no calculation run yet."
    r = results
    dec = r.get("energy_decomposition", {}).get("kcal_mol", {})
    lines = [
        f"MenaCalculator ({r.get('lot_used', 'auto')}, +U={r.get('u_used', 0)}, device={device})",
        f"  Total Energy:       {r['energy']:14.6f} eV  ({dec.get('total', 0.0):12.4f} kcal/mol)",
        f"  Interaction Energy: {r.get('energy_decomposition', {}).get('eV', {}).get('interaction', 0.0):14.6f} eV  ({dec.get('interaction', 0.0):12.4f} kcal/mol)",
        f"    |- Network:       {dec.get('network', 0.0):12.4f} kcal/mol  (stage0={dec.get('stage0', 0.0):.2f}, stage1={dec.get('stage1', 0.0):.2f}, final={dec.get('final', 0.0):.2f})",
        f"    |- Electrostatic: {dec.get('electrostatic', 0.0):12.4f} kcal/mol",
        f"    |- Polarization:  {dec.get('polarization', 0.0):12.4f} kcal/mol",
        f"    |- Dispersion D3: {dec.get('dispersion_d3', 0.0):12.4f} kcal/mol",
        f"    `- NLH Repulsion: {dec.get('repulsion', 0.0):12.4f} kcal/mol",
    ]
    if r.get("dipole_norm_debye", 0.0) > 0.0 or r.get("polarizability_iso", 0.0) > 0.0:
        d = r["dipole_debye"]
        lines.append(f"  Dipole Moment:      |mu| = {r['dipole_norm_debye']:.4f} D  ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f} D)")
        lines.append(f"  Polarizability:     iso = {r['polarizability_iso']:.3f} A^3,  aniso = {r['polarizability_aniso']:.3f} A^3")
    return "\n".join(lines)
