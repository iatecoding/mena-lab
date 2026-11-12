"""Subbatch construction and evaluation helpers for Tier-2 sparse-batched molecular L-BFGS."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from lam4.calc.physics import (
    MOL_KCAL_TO_EV,
    build_edge_tensors,
    build_open_mol_edges_gpu,
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
    zs, graphs, qs, ss, us, toks, refs = [], [], [], [], [], [], []
    for g_i, ent in enumerate(entries):
        n = ent["n"]
        zs.append(ent["z_t"])
        graphs.append(torch.full((n,), g_i, dtype=torch.long, device=dev))
        qs.append(ent["q"])
        ss.append(float(ent["s"]))
        us.append(ent["u"])
        toks.append(ent["tok"])
        refs.append(ent["ref"])

    graph_cat = torch.cat(graphs, dim=0)
    offset = int(pos_cat.shape[0])
    if dev.type in ("mps", "cuda") and max_n <= 256:
        ei_dev, es_dev = build_open_mol_edges_gpu(pos_cat, graph_cat, counts, r_list, dt)
    else:
        pos_all_np = pos_cat.detach().cpu().to(torch.float64).numpy()
        eis, ess = [], []
        off = 0
        for n in counts:
            ei_c, es_c = build_edge_tensors(pos_all_np[off : off + n], None, (False, False, False), r_list, dt)
            eis.append(ei_c.to(dev) + off)
            ess.append(es_c.to(dev))
            off += n
        ei_dev = torch.cat(eis, dim=1)
        es_dev = torch.cat(ess, dim=0)

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
        coulomb_cap=max(64, max_n),
        pbc=False,
    )
    csr_plan = build_csr_plan(b.edge_index[1], b.edge_index[0], offset) if calc.model.trunk.edge_mode == "csr" else None
    group_ptr = None
    if dev.type == "mps":
        from lam4.kernels.metal_coulomb import build_group_ptr

        group_ptr = build_group_ptr(b.graph, G)
    return b, csr_plan, group_ptr, pos_cat.clone()


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
    e_kcal = out["energy"].detach()
    f_kcal = out["forces"].detach() if need_forces else None

    if calc.add_dispersion and entries[0]["d3_mode"] == "bj_wb97m":
        max_n = int(b.coulomb_cap or 0)
        if dev.type == "mps" and dt == torch.float32 and max_n <= 1024:
            from lam4.kernels.metal_d3 import has_metal_d3, metal_d3_grouped

            if has_metal_d3():
                ed3, fd3 = metal_d3_grouped(b.z, pos, b.graph, G, group_ptr=group_ptr, compute_forces=need_forces)
                e_kcal = e_kcal + ed3
                if need_forces and fd3 is not None and f_kcal is not None:
                    f_kcal = f_kcal + fd3
            else:
                e_kcal, f_kcal = _fallback_d3_subbatch(entries, pos, e_kcal, f_kcal, dev, dt, need_forces)
        else:
            e_kcal, f_kcal = _fallback_d3_subbatch(entries, pos, e_kcal, f_kcal, dev, dt, need_forces)

    f_ev = (f_kcal * float(MOL_KCAL_TO_EV)) if (need_forces and f_kcal is not None) else None
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
        if need_forces and df is not None:
            fd3_list.append(torch.from_numpy(df.astype(np.float32 if dt == torch.float32 else np.float64)).to(dev))
        off += n
    e_kcal = e_kcal + torch.tensor(ed3_list, dtype=dt, device=dev)
    if need_forces and f_kcal is not None and fd3_list:
        f_kcal = f_kcal + torch.cat(fd3_list, dim=0)
    return e_kcal, f_kcal
