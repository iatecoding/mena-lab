"""Verlet active-edge pruning, ~3% bucket padding, and static GraphBatch template caching for MenaCalculator."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from lam4.calc.loader import crystal_e0_table_f64, molecule_ref_tables_f64
from lam4.calc.physics import build_edge_tensors, cached_vesin_nl
from lam4.core.edge_chunks import chunks_to, make_chunks, stream_chunk_edges
from lam4.data.batch import GraphBatch
from lam4.data.buckets import source_ids
from lam4.kernels.csr_plan import build_csr_plan

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator


def empty_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def bucket_active_edges(n_base: int, n_skin_need: int, n_skin_total: int) -> int:
    """Round active Verlet edge count up to ~3% buckets so compiled/cached shapes stay stable across steps."""
    if n_skin_need <= 0 or n_skin_total <= 0:
        return n_base
    step = max(256, n_base // 32)
    rounded_skin = ((n_skin_need + step - 1) // step) * step
    return min(n_base + n_skin_total, n_base + rounded_skin)


def _partition_verlet_cpu(
    pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool], r_cut: float, r_list: float, dtype: torch.dtype
) -> tuple[Tensor, Tensor, int, np.ndarray]:
    box = np.zeros((3, 3), dtype=np.float64) if cell_np is None else np.asarray(cell_np, dtype=np.float64)
    p = (bool(pbc_flags[0]), bool(pbc_flags[1]), bool(pbc_flags[2]))
    i, j, d, s = cached_vesin_nl(r_list).compute(
        points=np.asarray(pos_np, dtype=np.float64), box=box if any(p) else np.zeros((3, 3)), periodic=p, quantities="ijdS"
    )
    i_a, j_a, s_a, d_a = np.asarray(i, np.int64), np.asarray(j, np.int64), np.asarray(s, np.int64), np.asarray(d, np.float64)
    o = np.argsort(i_a, kind="stable")
    j_a, i_a, s_a, d_a = j_a[o], i_a[o], s_a[o], d_a[o]
    base_mask = d_a <= r_cut
    n_base = int(base_mask.sum())
    if n_base == len(d_a):
        ei = torch.from_numpy(np.stack([j_a, i_a], axis=0))
        return ei, torch.from_numpy(s_a).to(dtype=dtype), n_base, np.empty(0, dtype=np.float64)
    skin_idx = np.flatnonzero(~base_mask)
    skin_ord = skin_idx[np.argsort(d_a[skin_idx], kind="mergesort")]
    order = np.concatenate([np.flatnonzero(base_mask), skin_ord])
    ei = torch.from_numpy(np.ascontiguousarray(np.stack([j_a[order], i_a[order]], axis=0)))
    es = torch.from_numpy(np.ascontiguousarray(s_a[order])).to(dtype=dtype)
    return ei, es, n_base, np.ascontiguousarray(d_a[skin_ord], dtype=np.float64)


def verlet_neighbor_graph(
    calc: "MenaCalculator", z_np: np.ndarray, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool]
) -> tuple[Tensor, Tensor, list | None]:
    r_cut = float(calc.model.cfg.r_cut)
    r_list = r_cut + calc.skin
    is_csr = calc.model.trunk.edge_mode == "csr"
    c = calc._nl_cache
    if calc.skin > 0.0 and c is not None:
        same_sys = (
            c["pbc"] == pbc_flags
            and len(c["z"]) == len(z_np)
            and (c["z_bytes"] == z_np.tobytes())
            and ((cell_np is None and c["cell"] is None) or (cell_np is not None and c["cell"] is not None and np.allclose(cell_np, c["cell"], atol=1e-12, rtol=0.0)))
        )
        if same_sys:
            dr = pos_np - c["pos"]
            max_disp = float(np.sqrt((dr * dr).sum(axis=-1).max())) if len(pos_np) else 0.0
            if max_disp <= 0.5 * calc.skin:
                if is_csr and c.get("csr_plan") is not None:
                    calc._active_csr_plan = c["csr_plan"]
                    return c["ei"], c["es"], None
                if calc.low_memory:
                    calc._active_csr_plan = None
                    return c["ei"], c["es"], c["chunks"]
                skin_d0 = c["skin_d0"]
                need = int(np.searchsorted(skin_d0, r_cut + 2.0 * max_disp + 1e-9, side="right")) if len(skin_d0) else 0
                n_act = bucket_active_edges(c["n_base"], need, len(skin_d0))
                if n_act != c["last_bucket"]:
                    c["last_bucket"] = n_act
                    if len(z_np) >= 1024:
                        empty_device_cache(calc.device)
                calc._active_csr_plan = None
                return c["ei"][:, :n_act], c["es"][:n_act], None
        if len(z_np) >= 1024:
            calc._nl_cache = None
            empty_device_cache(calc.device)

    csr_plan = None
    if is_csr:
        ei_cpu, es_cpu = build_edge_tensors(pos_np, cell_np, pbc_flags, r_list, calc.dtype)
        n_base, skin_d0, chunks = int(ei_cpu.shape[1]), np.empty(0, dtype=np.float64), None
        ei, es = ei_cpu.to(calc.device), es_cpu.to(calc.device)
        csr_plan = build_csr_plan(ei[1], ei[0], len(z_np))
    elif calc.skin > 0.0 and not calc.low_memory:
        ei_cpu, es_cpu, n_base, skin_d0 = _partition_verlet_cpu(pos_np, cell_np, pbc_flags, r_cut, r_list, calc.dtype)
        chunks = None
        ei, es = ei_cpu.to(calc.device), es_cpu.to(calc.device)
    else:
        ei_cpu, es_cpu = build_edge_tensors(pos_np, cell_np, pbc_flags, r_list, calc.dtype)
        n_base, skin_d0 = int(ei_cpu.shape[1]), np.empty(0, dtype=np.float64)
        chunks = chunks_to(make_chunks(ei_cpu[1], ei_cpu[0], len(z_np), stream_chunk_edges(calc.device)), calc.device) if calc.low_memory else None
        ei, es = ei_cpu.to(calc.device), es_cpu.to(calc.device)
    calc._active_csr_plan = csr_plan
    if calc.skin > 0.0:
        calc._nl_cache = {
            "z": z_np.copy(),
            "z_bytes": z_np.tobytes(),
            "pbc": pbc_flags,
            "cell": None if cell_np is None else cell_np.copy(),
            "pos": pos_np.copy(),
            "ei": ei,
            "es": es,
            "n_base": n_base,
            "skin_d0": skin_d0,
            "last_bucket": n_base,
            "chunks": chunks,
            "csr_plan": csr_plan,
        }
    return (ei[:, :n_base], es[:n_base], None) if (calc.skin > 0.0 and not calc.low_memory and not is_csr) else (ei, es, chunks)


def ref_energies_kcal(add_ref: bool, src_name: str, domain: str, pbc_any: bool, z_np: np.ndarray, q_val: float, s_val: int) -> tuple[float, float, np.ndarray]:
    if not add_ref or len(z_np) == 0:
        return 0.0, 0.0, np.zeros(len(z_np), dtype=np.float64)
    if domain == "crystal":
        tab = crystal_e0_table_f64(src_name).numpy()
        atom_e0 = tab[np.clip(z_np, 0, len(tab) - 1)]
        return float(atom_e0.sum()), 0.0, atom_e0
    mt = molecule_ref_tables_f64(src_name)
    atom_e0 = mt["e0"][np.clip(z_np, 0, len(mt["e0"]) - 1)]
    eqs = 0.0
    if not pbc_any:
        qi, si = int(round(q_val)), int(round(s_val))
        if -10 <= qi <= 10:
            eqs += float(mt["e_q"][qi + 10])
        if 1 <= si <= 11:
            eqs += float(mt["e_s"][si - 1])
    return float(atom_e0.sum()), eqs, atom_e0


def build_or_reuse_batch(
    calc: "MenaCalculator", z_np: np.ndarray, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool],
    src: Any, u_val: int, q_val: float, s_val: int, ei: Tensor, es: Tensor
) -> tuple[GraphBatch, tuple[float, float, np.ndarray], dict[str, Any]]:
    dev, dt = calc.device, calc.dtype
    np_dt = np.float32 if dt == torch.float32 else np.float64
    pbc_any = bool(any(pbc_flags))
    key = (z_np.tobytes(), src.name, int(u_val), float(q_val), int(s_val), pbc_flags, calc.add_ref)
    sc = calc._static_cache
    pos_t = torch.from_numpy(np.ascontiguousarray(pos_np, dtype=np_dt)).to(dev)
    if sc is not None and sc.get("key") == key:
        if cell_np is None:
            cell_t = sc["zero_cell_t"]
        elif sc["cell_np"] is not None and np.array_equal(cell_np, sc["cell_np"]):
            cell_t = sc["cell_t"]
        else:
            cell_t = torch.from_numpy(np.ascontiguousarray(cell_np, dtype=np_dt)).to(dev).unsqueeze(0)
            sc["cell_np"] = cell_np.copy()
            sc["cell_t"] = cell_t
        return replace(sc["batch"], pos=pos_t, cell=cell_t, edge_index=ei, edge_shift=es), sc["ref_e"], sc

    tok_id, ref_id = source_ids(src.name)
    n = len(z_np)
    zero_cell_t = torch.zeros((1, 3, 3), dtype=dt, device=dev)
    cell_t = torch.from_numpy(np.ascontiguousarray(cell_np, dtype=np_dt)).to(dev).unsqueeze(0) if cell_np is not None else zero_cell_t
    b = GraphBatch(
        z=torch.from_numpy(np.ascontiguousarray(z_np, dtype=np.int64)).to(dev),
        pos=pos_t,
        graph=torch.zeros(n, dtype=torch.long, device=dev),
        cell=cell_t,
        edge_index=ei,
        edge_shift=es,
        charge=torch.tensor([q_val], dtype=dt, device=dev),
        spin=torch.tensor([float(s_val)], dtype=dt, device=dev),
        u_flag=torch.tensor([u_val], dtype=torch.long, device=dev),
        token=torch.tensor([tok_id], dtype=torch.long, device=dev),
        ref_id=torch.tensor([ref_id], dtype=torch.long, device=dev),
        domain=src.domain,
        atom_mask=torch.ones(n, dtype=torch.bool, device=dev),
        graph_mask=torch.ones(1, dtype=torch.bool, device=dev),
        energy=torch.zeros(1, dtype=dt, device=dev),
        forces=torch.zeros((n, 3), dtype=dt, device=dev),
        energy_mask=torch.ones(1, dtype=torch.bool, device=dev),
        force_mask=torch.ones(n, dtype=torch.bool, device=dev),
        coulomb_cap=max(calc.model.cfg.coulomb_cap, n) if (src.domain == "molecule" and not pbc_any) else None,
        pbc=pbc_any,
    )
    ref_e = ref_energies_kcal(calc.add_ref, src.name, src.domain, pbc_any, z_np, q_val, s_val)
    sc = {
        "key": key,
        "batch": b,
        "zero_cell_t": zero_cell_t,
        "cell_np": None if cell_np is None else cell_np.copy(),
        "cell_t": cell_t,
        "ref_e": ref_e,
        "zeros_n": np.zeros(n, dtype=np.float64),
        "zeros_3": np.zeros(3, dtype=np.float64),
        "zeros_33": np.zeros((3, 3), dtype=np.float64),
    }
    calc._static_cache = sc
    return b, ref_e, sc
