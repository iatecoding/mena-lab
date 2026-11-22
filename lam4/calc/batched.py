"""Batched evaluation & summary formatting helpers for MenaCalculator."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from ase import Atoms
from ase.calculators.calculator import all_changes
import numpy as np
import torch
from torch import Tensor

from lam4.calc.multipole_pack import (
    append_multipole_flat_parts, slice_multipoles_dict, unpack_multipoles_from_host, wants_multipoles,
)
from lam4.calc.physics import (
    CRYSTAL_EV, E_ANG_TO_DEBYE, MOL_KCAL_TO_EV, OPEN_MOL_GPU_NL_MAX_ATOMS, _get_d3, assemble_fragment_tensors,
    build_edge_tensors, build_energy_decomposition, build_open_mol_edges_gpu,
    effective_periodic_cell, eval_batched_open_mol_d3, eval_dispersion_single, lot_u_atoms_for_calc, polarizability_from_sums,
    resolve_charge_spin_fragments, resolve_lot_and_u,
)
from lam4.calc.precision_hygiene import center_positions_f64, restore_charged_dipole_f64
from lam4.data.batch import GraphBatch
from lam4.data.buckets import source_ids
from lam4.kernels.csr_plan import build_csr_plan

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator

_TERM_KEYS = ("stage0", "stage1", "final", "iso")


def _resolve_frame_lot(calc: "MenaCalculator", at: Atoms, cache: dict[tuple[Any, ...], tuple[Any, int]]) -> tuple[Any, int]:
    if not getattr(calc, "lock_lot", True):
        return resolve_lot_and_u(at, calc.lot, calc.u_override)
    z_arr = np.asarray(at.get_atomic_numbers(), dtype=np.int64)
    pbc_t = tuple(bool(x) for x in at.get_pbc())
    u_exp = at.info.get("u_flag") if calc.u_override is None else calc.u_override
    if u_exp is None and any(pbc_t):
        from lam4.registry.hubbard import MP_HUBBARD_ANIONS, MP_HUBBARD_METALS
        z_set = set(z_arr.tolist())
        if (z_set & MP_HUBBARD_ANIONS) and (z_set & MP_HUBBARD_METALS):
            return resolve_lot_and_u(at, calc.lot, calc.u_override)
    sig = (z_arr.tobytes(), pbc_t, at.info.get("lot"), u_exp, calc.lot, calc.u_override)
    hit = cache.get(sig)
    if hit is not None:
        return hit
    c_lot = getattr(calc, "_lot_cache", None)
    if c_lot is not None and c_lot[0] == sig:
        res = (c_lot[1], c_lot[2])
    else:
        res = resolve_lot_and_u(at, calc.lot, calc.u_override)
    cache[sig] = res
    return res


def _eval_homogeneous_microbatch(
    calc: "MenaCalculator",
    atoms_list: Sequence[Atoms],
    properties: Sequence[str],
    lot_cache: dict[tuple[Any, ...], tuple[Any, int]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a single homogeneous `(domain, pbc)` microbatch in ONE GPU/CPU pass."""
    if len(atoms_list) == 1 or (calc.low_memory and calc.model.trunk.edge_mode != "csr"):
        outs = []
        for a in atoms_list:
            cp = a.copy()
            calc.calculate(cp, properties=properties, system_changes=all_changes)
            outs.append(dict(calc.results))
        return outs

    lcache = {} if lot_cache is None else lot_cache
    first_src, _ = _resolve_frame_lot(calc, atoms_list[0], lcache)
    first_pbc = bool(np.any(atoms_list[0].get_pbc()))
    r_cut = float(calc.model.cfg.r_cut)
    dev, dt = calc.device, calc.dtype
    np_dt = np.float32 if dt == torch.float32 else np.float64

    z0 = np.asarray(atoms_list[0].get_atomic_numbers(), dtype=np.int64)
    p0_c, orig0 = center_positions_f64(atoms_list[0].get_positions())
    c0, pbc0 = effective_periodic_cell(atoms_list[0], r_cut) if first_pbc else (None, (False, False, False))
    shared_ei_c = shared_es_c = None
    pos_all = [(p0_c, orig0)]
    same_topo = len(z0) > 0
    max_disp = 0.0
    for at in atoms_list[1:]:
        zi = np.asarray(at.get_atomic_numbers(), dtype=np.int64)
        pi_c, orig_i = center_positions_f64(at.get_positions())
        pos_all.append((pi_c, orig_i))
        if not same_topo or len(zi) != len(z0) or not np.array_equal(zi, z0) or bool(np.any(at.get_pbc())) != first_pbc:
            same_topo = False
            continue
        if first_pbc:
            ci, pbci = effective_periodic_cell(at, r_cut)
            if pbci != pbc0 or not np.allclose(ci, c0, atol=1e-12, rtol=0.0):
                same_topo = False
                continue
        dr = pi_c - p0_c
        max_disp = max(max_disp, float(np.sqrt((dr * dr).sum(axis=-1).max())))
        if max_disp > 0.25:
            same_topo = False
    if same_topo:
        shared_ei_c, shared_es_c = build_edge_tensors(p0_c, c0, pbc0, r_cut + 2.0 * max_disp + 1e-3, dt, device=None)

    zs, poss, graphs, cells, eis, ess, qs, ss, us, toks, refs, xcs, paws, u_atoms = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    counts: list[int] = []
    meta_per_frame = []
    frag_items: list[tuple[int, float, int, np.ndarray | None, np.ndarray | None, np.ndarray | None]] = []
    offset = 0
    max_n = max((len(p[0]) for p in pos_all), default=0)
    use_gpu_nl = (not first_pbc) and (dev.type in ("mps", "cuda")) and (max_n <= OPEN_MOL_GPU_NL_MAX_ATOMS) and (shared_ei_c is None)
    shared_u_cache: tuple[str, int, int, int, np.ndarray] | None = None
    for g_idx, at in enumerate(atoms_list):
        src, u_val = _resolve_frame_lot(calc, at, lcache)
        pbc_any = bool(np.any(at.get_pbc()))
        z_np = z0 if same_topo else np.asarray(at.get_atomic_numbers(), dtype=np.int64)
        pos_np, origin_f64 = pos_all[g_idx]
        q_val, s_val, fid_np, fq_np, fs_np = resolve_charge_spin_fragments(
            at, z_np, calc.charge_override, calc.spin_override, pbc_any=pbc_any, domain=src.domain,
        )
        cell_np, pbc_flags = (c0, pbc0) if same_topo else (effective_periodic_cell(at, r_cut) if pbc_any else (None, (False, False, False)))
        tok_id, ref_id = source_ids(src.name)
        if same_topo and shared_u_cache is not None and shared_u_cache[0] == src.name and shared_u_cache[1] == u_val:
            xc_v, paw_v, u_at_np = shared_u_cache[2], shared_u_cache[3], shared_u_cache[4]
        else:
            xc_v, paw_v, u_at_np = lot_u_atoms_for_calc(src.name, z_np, pos_np, cell_np, pbc_flags, u_val)
            if same_topo and shared_u_cache is None:
                shared_u_cache = (src.name, u_val, xc_v, paw_v, u_at_np)
        n = len(z_np)
        counts.append(n)
        zs.append(torch.from_numpy(np.ascontiguousarray(z_np, dtype=np.int64)))
        poss.append(torch.from_numpy(np.ascontiguousarray(pos_np, dtype=np_dt)))
        graphs.append(torch.full((n,), g_idx, dtype=torch.long))
        if first_pbc:
            cells.append(torch.from_numpy(np.ascontiguousarray(cell_np, dtype=np_dt)) if cell_np is not None else torch.zeros((3, 3), dtype=dt))
        if not use_gpu_nl:
            ei_c, es_c = (shared_ei_c, shared_es_c) if shared_ei_c is not None else build_edge_tensors(pos_np, cell_np, pbc_flags, r_cut, dt, device=None)
            eis.append(ei_c + offset)
            ess.append(es_c)
        qs.append(q_val); ss.append(float(s_val)); us.append(u_val); toks.append(tok_id); refs.append(ref_id); xcs.append(xc_v); paws.append(paw_v); u_atoms.append(torch.from_numpy(u_at_np))
        frag_items.append((n, q_val, s_val, fid_np, fq_np, fs_np))
        meta_per_frame.append((src, u_val, q_val, s_val, z_np, pos_np, cell_np, pbc_flags, offset, offset + n, origin_f64))
        offset += n

    G = len(atoms_list)
    pos_dev = torch.cat(poss).to(dev)
    graph_dev = torch.cat(graphs).to(dev)
    cell_dev = torch.stack(cells).to(dev) if first_pbc else torch.zeros((G, 3, 3), dtype=dt, device=dev)
    ei_dev, es_dev = (
        build_open_mol_edges_gpu(pos_dev, graph_dev, counts, r_cut, dt)
        if use_gpu_nl
        else (torch.cat(eis, dim=1).to(dev), torch.cat(ess, dim=0).to(dev))
    )
    frag_id_dev, frag_q_dev, frag_s_dev = assemble_fragment_tensors(frag_items, dev, dt)
    qs_t = torch.tensor([qs, ss], dtype=dt, device=dev)
    utr_t = torch.tensor([us, toks, refs, xcs, paws], dtype=torch.long, device=dev)
    b = GraphBatch(
        z=torch.cat(zs).to(dev), pos=pos_dev, graph=graph_dev, cell=cell_dev, edge_index=ei_dev, edge_shift=es_dev,
        charge=qs_t[0], spin=qs_t[1], u_flag=utr_t[0], token=utr_t[1], ref_id=utr_t[2], domain=first_src.domain,
        atom_mask=torch.ones(offset, dtype=torch.bool, device=dev), graph_mask=torch.ones(G, dtype=torch.bool, device=dev),
        energy=torch.zeros(G, dtype=dt, device=dev), forces=torch.zeros((offset, 3), dtype=dt, device=dev),
        energy_mask=torch.ones(G, dtype=torch.bool, device=dev), force_mask=torch.ones(offset, dtype=torch.bool, device=dev),
        coulomb_cap=max(int(calc.model.cfg.coulomb_cap), max_n) if (first_src.domain == "molecule" and not first_pbc) else None,
        pbc=first_pbc,
        frag_id=frag_id_dev, frag_charge=frag_q_dev, frag_spin=frag_s_dev,
        xc_id=utr_t[3], paw_id=utr_t[4], u_atom=torch.cat(u_atoms).to(dev),
    )
    need_forces = bool(calc.compute_forces or ("forces" in properties))
    need_stress = "stress" in properties and first_pbc and all(all(m[7]) and m[6] is not None and abs(float(np.linalg.det(m[6]))) > 1e-8 for m in meta_per_frame)
    need_multipoles = wants_multipoles(properties)
    csr_plan = build_csr_plan(b.edge_index[1], b.edge_index[0], offset) if calc.model.trunk.edge_mode == "csr" else None
    out = calc._run_model(b, None, need_forces=need_forces, need_stress=need_stress, csr_plan=csr_plan, return_multipoles=need_multipoles)

    batched_d3_e = batched_d3_f_flat = batched_d3_f_pad = None
    ed3_dev = fd3_dev = None
    d3_modes = {m[0].d3_addback for m in meta_per_frame} if (calc.add_dispersion and not first_pbc and meta_per_frame) else set()
    uniform_mode = next(iter(d3_modes)) if len(d3_modes) == 1 else None
    if uniform_mode is not None:
        ed3_dev, fd3_dev = eval_batched_open_mol_d3(
            b.z, b.pos, b.graph, counts, uniform_mode, compute_forces=need_forces,
        )
        if ed3_dev is None and uniform_mode == "bj_wb97m" and max_n <= 256:
            d3 = _get_d3(dev, dt)
            z_pad_cpu, p_pad_cpu, m_pad_cpu = torch.zeros((G, max_n), dtype=torch.long), torch.zeros((G, max_n, 3), dtype=dt), torch.zeros((G, max_n), dtype=torch.bool)
            for g_i, m in enumerate(meta_per_frame):
                ni = len(m[4])
                z_pad_cpu[g_i, :ni], p_pad_cpu[g_i, :ni], m_pad_cpu[g_i, :ni] = zs[g_i], poss[g_i], True
            z_pad, p_pad, m_pad = z_pad_cpu.to(dev), p_pad_cpu.to(dev), m_pad_cpu.to(dev)
            if need_forces:
                p_pad = p_pad.detach().requires_grad_(True)
            with torch.set_grad_enabled(need_forces):
                ed3 = d3(z_pad, p_pad, m_pad)
                if need_forces:
                    batched_d3_f_pad = (-torch.autograd.grad(ed3.sum(), p_pad)[0]).detach().cpu().to(torch.float64).numpy()
            batched_d3_e = ed3.detach().cpu().to(torch.float64).numpy()

    # Pack all device output tensors into ONE single D2H host transfer!
    terms_d = out.get("terms") or {}
    z_g = out["energy"] * 0.0
    ov_t = out["overlap_energy"] if out.get("overlap_energy") is not None else z_g
    ex_t = out["spin_exchange_energy"] if out.get("spin_exchange_energy") is not None else z_g
    flat: list[Tensor] = [
        out["energy"], out["network_energy"], out["electrostatic_energy"],
        out["polarization_energy"], out["repulsion_energy"], ov_t, ex_t, out["energy_atom"],
    ]
    has_f = bool(need_forces and out.get("forces") is not None)
    has_s = bool(need_stress and out.get("stress") is not None)
    has_dip = out.get("dipole") is not None
    has_q = out.get("charges") is not None
    has_sd = out.get("spin_density") is not None
    has_ai = out.get("alpha") is not None
    has_aa = out.get("alpha_aniso") is not None
    st_qs = list(out.get("stage_charges") or [])
    present_terms = [k for k in _TERM_KEYS if k in terms_d]
    for k in present_terms:
        flat.append(terms_d[k].reshape(-1))
    for _k, _flag in (("forces", has_f), ("stress", has_s), ("dipole", has_dip), ("charges", has_q), ("spin_density", has_sd), ("alpha", has_ai), ("alpha_aniso", has_aa)):
        if _flag:
            flat.append(out[_k].reshape(-1))
    for sq in st_qs:
        flat.append(sq.reshape(-1))
    if ed3_dev is not None:
        flat.append(ed3_dev.detach().reshape(-1).to(dt))
    if fd3_dev is not None:
        flat.append(fd3_dev.detach().reshape(-1).to(dt))
    has_mp = bool(need_multipoles and append_multipole_flat_parts(flat, out))

    hbuf = torch.cat(flat, dim=0).detach().cpu().to(torch.float64).numpy()
    e_res_all, net_all, coul_all, pol_all, rep_all, ov_all, ex_all = (
        hbuf[i * G : (i + 1) * G] for i in range(7)
    )
    cur = 7 * G
    e_at_all = hbuf[cur : cur + offset]
    cur += offset
    term_at = {}
    for k in present_terms:
        term_at[k] = hbuf[cur : cur + offset]
        cur += offset
    f_res_all = hbuf[cur : cur + 3 * offset].reshape(offset, 3) if has_f else None
    cur += 3 * offset if has_f else 0
    stress_all = hbuf[cur : cur + 9 * G].reshape(G, 3, 3) if has_s else None
    cur += 9 * G if has_s else 0
    dip_all = hbuf[cur : cur + 3 * G].reshape(G, 3) if has_dip else None
    cur += 3 * G if has_dip else 0
    q_all = hbuf[cur : cur + offset] if has_q else None
    cur += offset if has_q else 0
    sd_all = hbuf[cur : cur + offset] if has_sd else None
    cur += offset if has_sd else 0
    alpha_iso_np = hbuf[cur : cur + offset] if has_ai else None
    cur += offset if has_ai else 0
    alpha_an_np = hbuf[cur : cur + 5 * offset].reshape(offset, 5) if has_aa else None
    cur += 5 * offset if has_aa else 0
    st_q_np = [hbuf[cur + i * offset : cur + (i + 1) * offset] for i in range(len(st_qs))]
    cur += len(st_qs) * offset
    if ed3_dev is not None:
        batched_d3_e = hbuf[cur : cur + G]
        cur += G
    if fd3_dev is not None:
        batched_d3_f_flat = hbuf[cur : cur + 3 * offset].reshape(offset, 3)
        cur += 3 * offset
    mp_all, cur = unpack_multipoles_from_host(hbuf, cur, offset, has_mp=has_mp, kcal_to_ev=1.0) if need_multipoles else (None, cur)

    ref_cache: dict[tuple[Any, ...], tuple[float, float, np.ndarray]] = {}
    results_list: list[dict[str, Any]] = []
    for g_i, (src, u_val, q_val, s_val, z_np, pos_np, cell_np, pbc_flags, s0, s1, origin_f64) in enumerate(meta_per_frame):
        kcal_to_ev = (1.0 / CRYSTAL_EV) if src.domain == "crystal" else MOL_KCAL_TO_EV
        d3_s = None
        if batched_d3_e is not None:
            d3_e = float(batched_d3_e[g_i])
            d3_f = batched_d3_f_flat[s0:s1] if batched_d3_f_flat is not None else (batched_d3_f_pad[g_i, : s1 - s0] if batched_d3_f_pad is not None else None)
        else:
            d3_mode = src.d3_addback if calc.add_dispersion else None
            d3_e, d3_f, d3_s = eval_dispersion_single(
                z_np, pos_np, cell_np, pbc_flags, d3_mode, dev, dt,
                compute_forces=need_forces, compute_stress=need_stress,
            )
        _, _, _, _, fq_np, fs_np = frag_items[g_i]
        rkey = (
            src.name, z_np.tobytes(), float(q_val), int(s_val),
            fq_np.tobytes() if fq_np is not None else None,
            fs_np.tobytes() if fs_np is not None else None,
        )
        if rkey not in ref_cache:
            ref_cache[rkey] = calc._ref_energies_kcal(
                src.name, src.domain, first_pbc, z_np, q_val, s_val,
                frag_charge=fq_np, frag_spin=fs_np,
            )
        e0_kcal, eqs_kcal, atom_e0_kcal = ref_cache[rkey]
        e_model_kcal = float(e_at_all[s0:s1].sum())
        e_ev = float((e_model_kcal + d3_e + e0_kcal + eqs_kcal) * kcal_to_ev)
        ni = s1 - s0
        f_ev = np.zeros((ni, 3), dtype=np.float64)
        if f_res_all is not None:
            f_ev = (f_res_all[s0:s1] + (d3_f if d3_f is not None else 0.0)) * kcal_to_ev
            if calc.zero_net_force and ni > 0:
                f_ev = f_ev - f_ev.mean(axis=0, keepdims=True)
        dip_ea = restore_charged_dipole_f64(dip_all[g_i], q_val, origin_f64, True) if dip_all is not None else np.zeros(3, dtype=np.float64)
        dip_deb = dip_ea * E_ANG_TO_DEBYE
        if alpha_iso_np is not None:
            pol_mat, pol_i, pol_a = polarizability_from_sums(
                float(alpha_iso_np[s0:s1].sum()),
                alpha_an_np[s0:s1].sum(axis=0) if alpha_an_np is not None else None,
            )
        else:
            pol_mat, pol_i, pol_a = np.zeros((3, 3), dtype=np.float64), 0.0, 0.0
        stress_voigt = None
        if "stress" in properties:
            if stress_all is not None:
                s33 = stress_all[g_i] + d3_s if d3_s is not None else stress_all[g_i]
                stress_voigt = s33[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] * kcal_to_ev
            else:
                stress_voigt = np.zeros(6, dtype=np.float64)
        spin_d = sd_all[s0:s1].copy() if sd_all is not None else np.zeros(ni, dtype=np.float64)
        coul_k, pol_k, rep_k, ov_k, ex_k = float(coul_all[g_i]), float(pol_all[g_i]), float(rep_all[g_i]), float(ov_all[g_i]), float(ex_all[g_i])
        decomp = build_energy_decomposition(
            kcal_to_ev=kcal_to_ev, net_kcal=float(e_model_kcal - coul_k - pol_k - rep_k - ov_k - ex_k), coul_kcal=coul_k,
            pol_kcal=pol_k, rep_kcal=rep_k, ov_kcal=ov_k, exch_kcal=ex_k, d3_kcal=d3_e,
            e0_kcal=e0_kcal, eqs_kcal=eqs_kcal,
            term_sums_kcal={k: float(term_at[k][s0:s1].sum()) for k in present_terms},
        )
        res_i: dict[str, Any] = {
            "energy": e_ev, "free_energy": e_ev,
            "dipole": dip_ea, "dipole_debye": dip_deb, "dipole_norm_debye": float(np.linalg.norm(dip_deb)),
            "charges": q_all[s0:s1].copy() if q_all is not None else np.zeros(ni, dtype=np.float64),
            "magmoms": spin_d, "spin_density": spin_d,
            "stage_charges": [sq[s0:s1].copy() for sq in st_q_np],
            "polarizability": pol_mat, "polarizability_iso": pol_i, "polarizability_aniso": pol_a,
            "atomic_polarizabilities": alpha_iso_np[s0:s1].copy() if alpha_iso_np is not None else np.zeros(ni, dtype=np.float64),
            "atomic_energies": (e_at_all[s0:s1] + atom_e0_kcal) * kcal_to_ev,
            "energy_decomposition": decomp, "lot_used": src.name, "u_used": int(u_val),
        }
        if need_forces:
            res_i["forces"] = f_ev
        if stress_voigt is not None:
            res_i["stress"] = stress_voigt
        if mp_all is not None:
            mp_i = slice_multipoles_dict(mp_all, s0, s1, kcal_to_ev=kcal_to_ev)
            res_i.update(mp_i)
            res_i["multipoles"] = mp_i
        results_list.append(res_i)
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

    lot_cache: dict[tuple[Any, ...], tuple[Any, int]] = {}
    groups: dict[tuple[str, bool, bool], list[int]] = {}
    tot_atoms = 0
    for idx, at in enumerate(atoms_list):
        src, _ = _resolve_frame_lot(calc, at, lot_cache)
        pbc_any = bool(np.any(at.get_pbc()))
        has_frag = (src.domain == "molecule") and (
            at.arrays.get("frag_id") is not None or at.info.get("frag_id") is not None
        )
        groups.setdefault((src.domain, pbc_any, bool(pbc_any and has_frag)), []).append(idx)
        tot_atoms += len(at)

    if len(groups) == 1 and n_tot_frames <= max_batch and tot_atoms <= max_atoms:
        return _eval_homogeneous_microbatch(calc, atoms_list, properties, lot_cache=lot_cache)

    out_all: list[dict[str, Any] | None] = [None] * n_tot_frames
    for _, indices in groups.items():
        indices_sorted = sorted(indices, key=lambda idx: len(atoms_list[idx]))
        cur_idx: list[int] = []
        cur_atoms = 0
        for idx in indices_sorted:
            na = len(atoms_list[idx])
            if cur_idx and (len(cur_idx) >= max_batch or cur_atoms + na > max_atoms):
                sub_res = _eval_homogeneous_microbatch(calc, [atoms_list[j] for j in cur_idx], properties, lot_cache=lot_cache)
                for j, r in zip(cur_idx, sub_res):
                    out_all[j] = r
                cur_idx, cur_atoms = [], 0
            cur_idx.append(idx)
            cur_atoms += na
        if cur_idx:
            sub_res = _eval_homogeneous_microbatch(calc, [atoms_list[j] for j in cur_idx], properties, lot_cache=lot_cache)
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
    ]
    if abs(float(dec.get("cloud_overlap", 0.0))) > 1e-12:
        lines.append(f"    |- Cloud Overlap: {dec.get('cloud_overlap', 0.0):12.4f} kcal/mol")
    if abs(float(dec.get("spin_exchange", 0.0))) > 1e-12:
        lines.append(f"    |- Spin Exchange: {dec.get('spin_exchange', 0.0):12.4f} kcal/mol")
    lines.extend([
        f"    |- Dispersion D3: {dec.get('dispersion_d3', 0.0):12.4f} kcal/mol",
        f"    `- NLH Repulsion: {dec.get('repulsion', 0.0):12.4f} kcal/mol",
    ])
    if r.get("dipole_norm_debye", 0.0) > 0.0 or r.get("polarizability_iso", 0.0) > 0.0:
        d = r["dipole_debye"]
        lines.append(f"  Dipole Moment:      |mu| = {r['dipole_norm_debye']:.4f} D  ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f} D)")
        lines.append(f"  Polarizability:     iso = {r['polarizability_iso']:.3f} A^3,  aniso = {r['polarizability_aniso']:.3f} A^3")
    return "\n".join(lines)
