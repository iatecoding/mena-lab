"""Verlet active-edge pruning, ~3% bucket padding, and static GraphBatch template caching for MenaCalculator."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import Tensor

from lam4.calc.loader import crystal_e0_table_f64, molecule_ref_tables_f64
from lam4.calc.physics import build_edge_tensors, cached_vesin_nl, center_positions_f64, is_periodic_mol_electro, lot_u_atoms_for_calc
from lam4.core.edge_chunks import chunks_to, make_chunks, stream_chunk_edges
from lam4.data.batch import GraphBatch
from lam4.data.buckets import source_ids
from lam4.kernels.csr_plan import CSRPlan, build_csr_plan

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator


def empty_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        with torch.cuda.device(device):
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
        points=center_positions_f64(pos_np)[0], box=box if any(p) else np.zeros((3, 3)), periodic=p, quantities="ijdS"
    )
    i_a, j_a, s_a, d_a = np.asarray(i, np.int64), np.asarray(j, np.int64), np.asarray(s, np.int64), np.asarray(d, np.float64)
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


def _is_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_cls) or "out of memory" in str(exc).lower()


def verlet_neighbor_graph(
    calc: "MenaCalculator",
    z_np: np.ndarray,
    pos_np: np.ndarray,
    cell_np: np.ndarray | None,
    pbc_flags: tuple[bool, bool, bool],
    *,
    skin: float | None = None,
) -> tuple[Tensor, Tensor, list | None]:
    eff_skin = calc.skin if skin is None else max(0.0, float(skin))
    r_cut = float(calc.model.cfg.r_cut)
    r_list = r_cut + eff_skin
    is_csr = calc.model.trunk.edge_mode == "csr"
    c = calc._nl_cache
    prev_n = int(len(c["z"])) if c is not None else 0
    prev_e = int(c["ei"].shape[1]) if (c is not None and "ei" in c) else 0

    if eff_skin > 0.0 and c is not None and c.get("skin", calc.skin) == eff_skin:
        same_sys = (
            c["pbc"] == pbc_flags
            and prev_n == len(z_np)
            and (c["z_bytes"] == z_np.tobytes())
            and (
                (cell_np is None and c["cell"] is None)
                or (cell_np is not None and c["cell"] is not None and np.allclose(cell_np, c["cell"], atol=1e-12, rtol=0.0))
            )
        )
        if same_sys:
            dr = pos_np - c["pos"]
            max_disp = float(np.sqrt((dr * dr).sum(axis=-1).max())) if len(pos_np) else 0.0
            if max_disp <= 0.5 * eff_skin:
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
                calc._active_csr_plan = None
                return c["ei"][:, :n_act], c["es"][:n_act], None

    # Drop reference to old cache tensors before allocating new ones; only flush allocator on >2x shrink or OOM!
    calc._nl_cache = None
    if prev_n >= 1024 and len(z_np) * 2 < prev_n:
        empty_device_cache(calc.device)

    def _alloc_edges() -> tuple[Tensor, Tensor, int, np.ndarray, list | None, Any]:
        csr_p = None
        if is_csr:
            ei_raw, es_raw = build_edge_tensors(pos_np, cell_np, pbc_flags, r_list, calc.dtype, device=calc.device)
            nb, sd0, ch = int(ei_raw.shape[1]), np.empty(0, dtype=np.float64), None
            ei_d, es_d = ei_raw.to(calc.device), es_raw.to(calc.device)
            csr_p = build_csr_plan(ei_d[1], ei_d[0], len(z_np))
            return ei_d, es_d, nb, sd0, ch, csr_p
        if eff_skin > 0.0 and not calc.low_memory:
            ei_cpu, es_cpu, nb, sd0 = _partition_verlet_cpu(pos_np, cell_np, pbc_flags, r_cut, r_list, calc.dtype)
            return ei_cpu.to(calc.device), es_cpu.to(calc.device), nb, sd0, None, None
        if calc.low_memory:
            ei_cpu, es_cpu = build_edge_tensors(pos_np, cell_np, pbc_flags, r_list, calc.dtype)
            nb, sd0 = int(ei_cpu.shape[1]), np.empty(0, dtype=np.float64)
            ch = chunks_to(make_chunks(ei_cpu[1], ei_cpu[0], len(z_np), stream_chunk_edges(calc.device)), calc.device)
            return ei_cpu.to(calc.device), es_cpu.to(calc.device), nb, sd0, ch, None
        ei_raw, es_raw = build_edge_tensors(pos_np, cell_np, pbc_flags, r_list, calc.dtype, device=calc.device)
        nb, sd0 = int(ei_raw.shape[1]), np.empty(0, dtype=np.float64)
        return ei_raw.to(calc.device), es_raw.to(calc.device), nb, sd0, None, None

    try:
        ei, es, n_base, skin_d0, chunks, csr_plan = _alloc_edges()
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        empty_device_cache(calc.device)
        ei, es, n_base, skin_d0, chunks, csr_plan = _alloc_edges()

    if prev_e >= 16_384 and int(ei.shape[1]) * 2 < prev_e:
        empty_device_cache(calc.device)

    calc._active_csr_plan = csr_plan
    if eff_skin > 0.0:
        calc._nl_cache = {
            "skin": eff_skin,
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
    return (ei[:, :n_base], es[:n_base], None) if (eff_skin > 0.0 and not calc.low_memory and not is_csr) else (ei, es, chunks)


def ref_energies_kcal(
    add_ref: bool,
    src_name: str,
    domain: str,
    pbc_any: bool,
    z_np: np.ndarray,
    q_val: float,
    s_val: int,
    frag_charge: np.ndarray | Sequence[float] | None = None,
    frag_spin: np.ndarray | Sequence[int | float] | None = None,
) -> tuple[float, float, np.ndarray]:
    if not add_ref or len(z_np) == 0:
        return 0.0, 0.0, np.zeros(len(z_np), dtype=np.float64)
    from lam4.registry.sources import _OVERLAY, reference_key_of
    if domain == "crystal" or (src_name in _OVERLAY and reference_key_of(src_name) in ("mp_pbe", "mp_pbeU", "matpes_r2scan", "oc20_rpbe")):
        tab = crystal_e0_table_f64(src_name).numpy()
        atom_e0 = tab[np.clip(z_np, 0, len(tab) - 1)]
        return float(atom_e0.sum()), 0.0, atom_e0
    mt = molecule_ref_tables_f64(src_name)
    atom_e0 = mt["e0"][np.clip(z_np, 0, len(mt["e0"]) - 1)]
    eqs = 0.0
    if frag_charge is not None:
        fs_seq = [1] * len(frag_charge) if frag_spin is None else frag_spin
        for q_f, s_f in zip(frag_charge, fs_seq):
            qi, si = int(round(float(q_f))), int(round(float(s_f)))
            if -10 <= qi <= 10:
                eqs += float(mt["e_q"][qi + 10])
            if 1 <= si <= 11:
                eqs += float(mt["e_s"][si - 1])
    elif not pbc_any:
        qi, si = int(round(q_val)), int(round(s_val))
        if -10 <= qi <= 10:
            eqs += float(mt["e_q"][qi + 10])
        if 1 <= si <= 11:
            eqs += float(mt["e_s"][si - 1])
    return float(atom_e0.sum()), eqs, atom_e0


def _try_pin_empty(shape: tuple[int, ...], dtype: torch.dtype, dev: torch.device) -> Tensor | None:
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    except RuntimeError:
        return None


def sync_padded_csr_plan(
    real_plan: CSRPlan | None, pb_ei: Tensor, n: int, e0: int, e_cap: int, cur_plan: CSRPlan | None,
) -> CSRPlan:
    """Build or update a static `[n+2, e_cap]` `CSRPlan` where pad atoms `[n, n+1]` have 0 edges (`recv_ptr[n:] = send_ptr[n:] = e0`)."""
    dev, n_pad = pb_ei.device, n + 2
    if cur_plan is None or cur_plan.n_atoms != n_pad or cur_plan.n_edges != e_cap or cur_plan.recv_ptr.device != dev:
        cur_plan = CSRPlan(
            n_pad, e_cap,
            torch.empty(n_pad + 1, dtype=torch.int32, device=dev), torch.empty(e_cap, dtype=torch.int32, device=dev),
            torch.empty(e_cap, dtype=torch.int32, device=dev), torch.empty(n_pad + 1, dtype=torch.int32, device=dev),
            torch.empty(e_cap, dtype=torch.int32, device=dev),
        )
    if real_plan is None or real_plan.n_atoms != n or real_plan.n_edges != e0 or real_plan.recv_ptr.device != dev:
        try:
            full_p = build_csr_plan(pb_ei[1], pb_ei[0], n_pad, active_edges=e0)  # type: ignore[call-arg]
            for attr in ("recv_ptr", "send_idx", "recv_idx", "send_ptr", "send_perm"):
                getattr(cur_plan, attr).copy_(getattr(full_p, attr))
            return cur_plan
        except TypeError:
            real_plan = build_csr_plan(pb_ei[1, :e0], pb_ei[0, :e0], n)
    cur_plan.recv_ptr[: n + 1].copy_(real_plan.recv_ptr)
    cur_plan.recv_ptr[n + 1 :].fill_(e0)
    cur_plan.send_ptr[: n + 1].copy_(real_plan.send_ptr)
    cur_plan.send_ptr[n + 1 :].fill_(e0)
    if e0 > 0:
        cur_plan.send_idx[:e0].copy_(real_plan.send_idx)
        cur_plan.recv_idx[:e0].copy_(real_plan.recv_idx)
        cur_plan.send_perm[:e0].copy_(real_plan.send_perm)
    if e_cap > e0:
        cur_plan.send_idx[e0:].copy_(pb_ei[0, e0:e_cap].to(torch.int32))
        cur_plan.recv_idx[e0:].copy_(pb_ei[1, e0:e_cap].to(torch.int32))
        cur_plan.send_perm[e0:].zero_()
    return cur_plan


def warm_calc_caches(calc: "MenaCalculator", dev: torch.device, n_real: int, n_pad: int) -> None:
    """Pre-populate read-only CSR, NLH, D3, Coulomb, and model device caches outside CUDA stream capture."""
    if dev.type != "cuda" or not torch.cuda.is_available():
        return
    with torch.cuda.device(dev):
        try:
            if hasattr(calc.model, "warm_caches"):
                for m in ({n_real, n_pad} - {0}):
                    calc.model.warm_caches(dev, calc.dtype, m)
        except Exception:  # noqa: BLE001
            pass
        try:
            from lam4.kernels import csr_plan, triton_csr, triton_nlh

            if hasattr(csr_plan, "warm_csr_nlh_caches"):
                csr_plan.warm_csr_nlh_caches(dev)
            else:
                csr_plan.rmin_table_f32(dev)
                if hasattr(csr_plan, "_default_p_horner"):
                    csr_plan._default_p_horner(dev)
                if hasattr(triton_csr, "dummy_f32"):
                    triton_csr.dummy_f32(dev); triton_csr.dummy_i64(dev)
                if hasattr(triton_nlh, "_nlh_cb_jet_f32"):
                    triton_nlh._nlh_cb_jet_f32(dev)
            for st in getattr(getattr(calc.model, "trunk", None), "stages", ()):
                csr_plan.coef_to_horner_poly(st.tables.weight, dev)
        except Exception:  # noqa: BLE001
            pass
        try:
            from lam4.kernels import triton_coulomb
            from lam4.physics import repulsion

            if hasattr(repulsion, "_tables"):
                repulsion._tables(dev, calc.dtype)
            if hasattr(triton_coulomb, "has_triton_coulomb") and triton_coulomb.has_triton_coulomb(dev):
                for m in ({n_real, n_pad} - {0}):
                    if hasattr(triton_coulomb, "warm_triton_coulomb"):
                        triton_coulomb.warm_triton_coulomb(dev, m)
                    else:
                        triton_coulomb.build_group_ptr(torch.zeros(m, dtype=torch.long, device=dev), 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            from lam4.calc.d3_nl_cache import warm_d3_caches

            warm_d3_caches(dev)
        except Exception:  # noqa: BLE001
            pass


def build_or_reuse_batch(
    calc: "MenaCalculator", z_np: np.ndarray, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool],
    src: Any, u_val: int, q_val: float, s_val: int, ei: Tensor, es: Tensor,
    *,
    frag_id_np: np.ndarray | None = None,
    frag_q_np: np.ndarray | None = None,
    frag_s_np: np.ndarray | None = None,
    recenter: bool = True,
    periodic_electro: bool | None = None,
) -> tuple[GraphBatch, tuple[float, float, np.ndarray], dict[str, Any]]:
    dev, dt = calc.device, calc.dtype
    np_dt = np.float32 if dt == torch.float32 else np.float64
    pbc_any = bool(any(pbc_flags))
    pbc_elec = is_periodic_mol_electro(src, pbc_flags, cell_np) if periodic_electro is None else bool(periodic_electro)
    n = len(z_np)
    frag_sig = None if frag_id_np is None else (
        frag_id_np.tobytes(), None if frag_q_np is None else frag_q_np.tobytes(), None if frag_s_np is None else frag_s_np.tobytes(),
    )
    key = (z_np.tobytes(), src.name, int(u_val), float(q_val), int(s_val), pbc_flags, pbc_elec, calc.add_ref, frag_sig)
    sc = calc._static_cache
    pos_use = center_positions_f64(pos_np)[0] if recenter else pos_np
    pos_cpu = torch.from_numpy(np.ascontiguousarray(pos_use, dtype=np_dt))
    if sc is not None and sc.get("key") == key:
        pos_pin = sc.get("pos_pin")
        if pos_pin is not None and n > 0:
            pos_pin.copy_(pos_cpu)
            pos_t = torch.empty((n, 3), dtype=dt, device=dev)
            pos_t.copy_(pos_pin, non_blocking=True)
        else:
            pos_t = pos_cpu.to(dev)
        cell_changed = False
        if cell_np is None:
            cell_t = sc["zero_cell_t"]
        elif sc["cell_np"] is not None and np.array_equal(cell_np, sc["cell_np"]):
            cell_t = sc["cell_t"]
        else:
            cell_t = torch.from_numpy(np.ascontiguousarray(cell_np, dtype=np_dt)).to(dev).unsqueeze(0)
            sc["cell_np"] = cell_np.copy()
            sc["cell_t"] = cell_t
            cell_changed = True
        b_out = replace(sc["batch"], pos=pos_t, cell=cell_t, edge_index=ei, edge_shift=es)
        if int(u_val) != 0 and (cell_changed or not getattr(calc, "lock_lot", True)):
            _, _, u_at_np = lot_u_atoms_for_calc(src.name, z_np, pos_use, cell_np, pbc_flags, u_val)
            b_out = replace(b_out, u_atom=torch.from_numpy(np.ascontiguousarray(u_at_np, dtype=np.int64)).to(dev))
        return b_out, sc["ref_e"], sc

    pos_pin = _try_pin_empty((n, 3), dt, dev) if n > 0 else None
    if pos_pin is not None:
        pos_pin.copy_(pos_cpu)
        pos_t = torch.empty((n, 3), dtype=dt, device=dev)
        pos_t.copy_(pos_pin, non_blocking=True)
    else:
        pos_t = pos_cpu.to(dev)
    tok_id, ref_id = source_ids(src.name)
    xc_v, paw_v, u_at_np = lot_u_atoms_for_calc(src.name, z_np, pos_use, cell_np, pbc_flags, u_val)
    zero_cell_t = torch.zeros((1, 3, 3), dtype=dt, device=dev)
    cell_t = torch.from_numpy(np.ascontiguousarray(cell_np, dtype=np_dt)).to(dev).unsqueeze(0) if cell_np is not None else zero_cell_t
    qs_t = torch.tensor([q_val, float(s_val)], dtype=dt, device=dev)
    utr_t = torch.tensor([u_val, tok_id, ref_id, xc_v, paw_v], dtype=torch.long, device=dev)
    frag_id_t = None if frag_id_np is None else torch.from_numpy(np.ascontiguousarray(frag_id_np, dtype=np.int64)).to(dev)
    frag_q_t = None if frag_q_np is None else torch.from_numpy(np.ascontiguousarray(frag_q_np, dtype=np_dt)).to(dev)
    frag_s_t = None if frag_s_np is None else torch.from_numpy(np.ascontiguousarray(frag_s_np, dtype=np_dt)).to(dev)
    b = GraphBatch(
        z=torch.from_numpy(np.ascontiguousarray(z_np, dtype=np.int64)).to(dev),
        pos=pos_t,
        graph=torch.zeros(n, dtype=torch.long, device=dev),
        cell=cell_t,
        edge_index=ei,
        edge_shift=es,
        charge=qs_t[0:1],
        spin=qs_t[1:2],
        u_flag=utr_t[0:1],
        token=utr_t[1:2],
        ref_id=utr_t[2:3],
        xc_id=utr_t[3:4],
        paw_id=utr_t[4:5],
        u_atom=torch.from_numpy(np.ascontiguousarray(u_at_np, dtype=np.int64)).to(dev),
        domain=src.domain,
        atom_mask=torch.ones(n, dtype=torch.bool, device=dev),
        graph_mask=torch.ones(1, dtype=torch.bool, device=dev),
        energy=torch.zeros(1, dtype=dt, device=dev),
        forces=torch.zeros((n, 3), dtype=dt, device=dev),
        energy_mask=torch.ones(1, dtype=torch.bool, device=dev),
        force_mask=torch.ones(n, dtype=torch.bool, device=dev),
        coulomb_cap=max(calc.model.cfg.coulomb_cap, n) if (src.domain == "molecule" and not pbc_any) else None,
        frag_id=frag_id_t,
        frag_charge=frag_q_t,
        frag_spin=frag_s_t,
        pbc=pbc_any,
        periodic_electro=pbc_elec,
    )
    ref_e = ref_energies_kcal(
        calc.add_ref, src.name, src.domain, pbc_any, z_np, q_val, s_val,
        frag_charge=frag_q_np, frag_spin=frag_s_np,
    )
    sc = {
        "key": key,
        "batch": b,
        "pos_pin": pos_pin,
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
