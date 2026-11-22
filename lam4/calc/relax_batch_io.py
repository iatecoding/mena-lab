"""Subbatch construction and evaluation helpers for Tier-2 sparse-batched molecular L-BFGS."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from lam4.calc.d3_nl_cache import _get_triton_d3, has_triton_d3_on
from lam4.calc.physics import (
    MOL_KCAL_TO_EV,
    OPEN_MOL_GPU_NL_MAX_ATOMS,
    assemble_fragment_tensors,
    build_edge_tensors,
    build_open_mol_edges_gpu,
    eval_batched_open_mol_d3,
    eval_dispersion_single,
)
from lam4.data.batch import GraphBatch
from lam4.kernels.csr_plan import build_csr_plan

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator


def build_mol_subbatch(
    calc: "MenaCalculator",
    entries: list[dict[str, Any]],
    pos_list_f32: list[Tensor],
    r_list: float,
) -> tuple[GraphBatch, Any, Any, Tensor]:
    dev, dt = calc.device, calc.dtype
    G = len(entries)
    counts = [int(ent["n"]) for ent in entries]
    max_n = max(counts) if G > 0 else 0
    pos_cat = torch.cat(pos_list_f32, dim=0)
    zs, graphs, qs, ss, us, toks, refs, xcs, paws, u_atoms = [], [], [], [], [], [], [], [], [], []
    for g_i, ent in enumerate(entries):
        n = ent["n"]
        zs.append(ent["z_t"])
        graphs.append(torch.full((n,), g_i, dtype=torch.long, device=dev))
        qs.append(ent["q"])
        ss.append(float(ent["s"]))
        us.append(ent["u"])
        toks.append(ent["tok"])
        refs.append(ent["ref"])
        xcs.append(int(ent.get("xc", 0)))
        paws.append(int(ent.get("paw", 0)))
        u_atoms.append(ent["u_atom_t"] if "u_atom_t" in ent else torch.zeros(n, dtype=torch.long, device=dev))

    graph_cat = torch.cat(graphs, dim=0)
    offset = int(pos_cat.shape[0])
    if dev.type in ("mps", "cuda") and max_n <= OPEN_MOL_GPU_NL_MAX_ATOMS:
        ei_dev, es_dev = build_open_mol_edges_gpu(pos_cat, graph_cat, counts, r_list, dt)
    else:
        pos_all_np = pos_cat.detach().cpu().to(torch.float64).numpy()
        eis, ess = [], []
        off = 0
        for n in counts:
            ei_c, es_c = build_edge_tensors(pos_all_np[off : off + n], None, (False, False, False), r_list, dt, device=dev)
            eis.append(ei_c.to(dev) + off)
            ess.append(es_c.to(dev))
            off += n
        ei_dev = torch.cat(eis, dim=1)
        es_dev = torch.cat(ess, dim=0)

    frag_id_dev, frag_q_dev, frag_s_dev = assemble_fragment_tensors(
        [
            (
                int(e["n"]),
                float(e["q"]),
                int(e["s"]),
                e.get("frag_id_np"),
                e.get("frag_q_np"),
                e.get("frag_s_np"),
            )
            for e in entries
        ],
        dev,
        dt,
    )
    b = GraphBatch(
        z=torch.cat(zs, dim=0),
        pos=pos_cat,
        graph=graph_cat,
        cell=torch.zeros((G, 3, 3), dtype=dt, device=dev),
        edge_index=ei_dev,
        edge_shift=es_dev,
        charge=torch.tensor(qs, dtype=dt, device=dev),
        spin=torch.tensor(ss, dtype=dt, device=dev),
        u_flag=torch.tensor(us, dtype=torch.long, device=dev),
        token=torch.tensor(toks, dtype=torch.long, device=dev),
        ref_id=torch.tensor(refs, dtype=torch.long, device=dev),
        domain="molecule",
        atom_mask=torch.ones(offset, dtype=torch.bool, device=dev),
        graph_mask=torch.ones(G, dtype=torch.bool, device=dev),
        energy=torch.zeros(G, dtype=dt, device=dev),
        forces=torch.zeros((offset, 3), dtype=dt, device=dev),
        energy_mask=torch.ones(G, dtype=torch.bool, device=dev),
        force_mask=torch.ones(offset, dtype=torch.bool, device=dev),
        coulomb_cap=max(int(calc.model.cfg.coulomb_cap), max_n),
        pbc=False,
        frag_id=frag_id_dev,
        frag_charge=frag_q_dev,
        frag_spin=frag_s_dev,
        xc_id=torch.tensor(xcs, dtype=torch.long, device=dev),
        paw_id=torch.tensor(paws, dtype=torch.long, device=dev),
        u_atom=torch.cat(u_atoms, dim=0),
    )
    csr_plan = build_csr_plan(b.edge_index[1], b.edge_index[0], offset) if calc.model.trunk.edge_mode == "csr" else None
    group_ptr = None
    if dev.type == "mps":
        from lam4.kernels.metal_coulomb import build_group_ptr

        group_ptr = build_group_ptr(b.graph, G)
    elif dev.type == "cuda" and has_triton_d3_on(dev):
        td3 = _get_triton_d3()
        if td3 is not None and hasattr(td3, "build_group_ptr"):
            group_ptr = td3.build_group_ptr(b.graph, G)
    return b, csr_plan, group_ptr, pos_cat.clone()


def build_mol_subbatch_constraints(
    entries: list[dict[str, Any]],
    dev: torch.device,
    dt: torch.dtype,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """Concatenate per-entry FixAtoms masks and harmonic restraint arrays for a subbatch."""
    if not entries:
        return None, None, None
    move_mask_t: Tensor | None = None
    if any(e.get("move_mask_np") is not None for e in entries):
        masks = [
            np.ones(int(e["n"]), dtype=bool)
            if e.get("move_mask_np") is None
            else np.asarray(e["move_mask_np"], dtype=bool)
            for e in entries
        ]
        move_mask_t = torch.from_numpy(np.ascontiguousarray(np.concatenate(masks))).to(dev)

    rest_k_t: Tensor | None = None
    rest_ref_t: Tensor | None = None
    if any(e.get("rest_k_np") is not None for e in entries):
        ks = [
            np.zeros(int(e["n"]), dtype=np.float64)
            if e.get("rest_k_np") is None
            else np.asarray(e["rest_k_np"], dtype=np.float64)
            for e in entries
        ]
        refs = [
            np.zeros((int(e["n"]), 3), dtype=np.float64)
            if e.get("rest_ref_np") is None
            else np.asarray(e["rest_ref_np"], dtype=np.float64)
            for e in entries
        ]
        rest_k_t = torch.from_numpy(np.ascontiguousarray(np.concatenate(ks))).to(device=dev, dtype=dt)
        rest_ref_t = torch.from_numpy(np.ascontiguousarray(np.concatenate(refs))).to(device=dev, dtype=dt)
    return move_mask_t, rest_k_t, rest_ref_t


def eval_mol_subbatch_ef(
    calc: "MenaCalculator",
    b: GraphBatch,
    pos: Tensor,
    csr_plan: Any,
    group_ptr: Any,
    entries: list[dict[str, Any]],
    *,
    need_forces: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Return `(e_res_kcal [G] device tensor, f_ev [N, 3] device tensor | None)`."""
    dev, dt = calc.device, calc.dtype
    G = len(entries)
    b_cur = replace(b, pos=pos)
    out = calc._run_model(b_cur, None, need_forces=need_forces, need_stress=False, csr_plan=csr_plan)
    e_kcal = out.get("energy_f64", out["energy"]).detach()
    f_kcal = out["forces"].detach() if need_forces else None

    if calc.add_dispersion and G > 0:
        d3_modes = {ent["d3_mode"] for ent in entries}
        if d3_modes != {None}:
            uniform_mode = entries[0]["d3_mode"] if len(d3_modes) == 1 else None
            ed3, fd3 = eval_batched_open_mol_d3(
                b.z, pos, b.graph, [int(ent["n"]) for ent in entries], uniform_mode,
                group_ptr=group_ptr, compute_forces=need_forces,
            )
            if ed3 is not None:
                e_kcal = e_kcal + ed3.to(dtype=e_kcal.dtype)
                if need_forces and fd3 is not None and f_kcal is not None:
                    f_kcal = f_kcal + fd3
            else:
                e_kcal, f_kcal = _fallback_d3_subbatch(entries, pos, e_kcal, f_kcal, dev, dt, need_forces)

    f_ev = (f_kcal * float(MOL_KCAL_TO_EV)) if (need_forces and f_kcal is not None) else None
    if f_ev is not None and calc.zero_net_force and G > 0:
        counts_t = torch.tensor([max(1, int(ent["n"])) for ent in entries], dtype=f_ev.dtype, device=dev).unsqueeze(-1)
        f_sum = torch.zeros((G, 3), dtype=f_ev.dtype, device=dev).index_add_(0, b.graph, f_ev)
        f_ev = f_ev - (f_sum / counts_t).index_select(0, b.graph)
    return e_kcal, f_ev


def _fallback_d3_subbatch(
    entries: list[dict[str, Any]],
    pos: Tensor,
    e_kcal: Tensor,
    f_kcal: Tensor | None,
    dev: torch.device,
    dt: torch.dtype,
    need_forces: bool,
) -> tuple[Tensor, Tensor | None]:
    pos_np = pos.detach().cpu().to(torch.float64).numpy()
    np_dt = np.float32 if dt == torch.float32 else np.float64
    off = 0
    ed3_list = []
    fd3_list = []
    for ent in entries:
        n = ent["n"]
        de, df, _ = eval_dispersion_single(
            ent["z_np"], pos_np[off : off + n], None, (False, False, False),
            ent["d3_mode"], dev, dt, compute_forces=need_forces,
        )
        ed3_list.append(de)
        if need_forces:
            if df is not None:
                fd3_list.append(torch.from_numpy(df.astype(np_dt)).to(dev))
            else:
                fd3_list.append(torch.zeros((n, 3), dtype=dt, device=dev))
        off += n
    e_kcal = e_kcal + torch.tensor(ed3_list, dtype=e_kcal.dtype, device=dev)
    if need_forces and f_kcal is not None and fd3_list:
        f_kcal = f_kcal + torch.cat(fd3_list, dim=0)
    return e_kcal, f_kcal
