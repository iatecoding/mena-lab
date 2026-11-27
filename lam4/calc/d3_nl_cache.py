"""Verlet-cached 12 A CSR neighbour list & MDEngine dispersion helpers for Metal & Triton D3 kernels."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from lam4.calc.precision_hygiene import center_positions_f64

SKIN = 1.0  # A

_CACHE: dict[str, Any] = {}
_PERIODIC_CACHE: dict[str, Any] = {}


def clear_d3_caches() -> None:
    """Release cached D3 neighbor-list GPU tensors."""
    _CACHE.clear()
    _PERIODIC_CACHE.clear()


def _get_triton_d3() -> Any | None:
    try:
        from lam4.kernels import triton_d3

        return triton_d3
    except ImportError:
        return None


def has_triton_d3_on(dev: torch.device) -> bool:
    if dev.type != "cuda":
        return False
    td3 = _get_triton_d3()
    return bool(td3 is not None and getattr(td3, "has_triton_d3", lambda *a: False)(dev))


def warm_d3_caches(dev: torch.device) -> None:
    """Pre-populate read-only Triton D3 device tables before CUDA stream capture."""
    if not has_triton_d3_on(dev):
        return
    td3 = _get_triton_d3()
    if td3 is None:
        return
    if hasattr(td3, "warm_d3_tables"):
        td3.warm_d3_tables(dev)
    elif hasattr(td3, "_get_d3_tables"):
        td3._get_d3_tables(dev)


def _pack_ij_i32(i_raw: np.ndarray, j_raw: np.ndarray, device: torch.device) -> tuple[Tensor, Tensor]:
    ij = np.stack([np.asarray(i_raw, dtype=np.int32), np.asarray(j_raw, dtype=np.int32)], axis=0)
    ij_t = torch.from_numpy(ij).to(device)
    return ij_t[0], ij_t[1]


def d3_csr_list(
    z_np: np.ndarray, pos_np: np.ndarray, r_cut: float, device: torch.device, build=None,
) -> tuple[Tensor, Tensor]:
    """`(recv_ptr int32 [N+1], send int32 [E])` on `device` for open-boundary positions."""
    n = int(len(z_np))
    c = _CACHE
    if (
        c.get("n") == n and c.get("device") == device and c.get("r_cut") == r_cut
        and np.array_equal(c["z"], z_np)
        and float(np.max(np.sum((pos_np - c["pos0"]) ** 2, axis=1), initial=0.0)) < (0.5 * SKIN) ** 2
    ):
        return c["rp"], c["send"]
    if build is None and device.type == "mps":
        from lam4.calc.physics import cached_vesin_nl
        from lam4.kernels.metal_d3 import has_metal_d3, metal_build_csr_i32

        if has_metal_d3():
            try:
                i_raw, j_raw = cached_vesin_nl(float(r_cut) + SKIN).compute(
                    points=center_positions_f64(pos_np)[0], box=np.zeros((3, 3)), periodic=(False, False, False), quantities="ij"
                )
                rp, send_i32 = metal_build_csr_i32(*_pack_ij_i32(i_raw, j_raw, device), n)
                _CACHE.clear()
                _CACHE.update(n=n, device=device, r_cut=r_cut, z=z_np.copy(), pos0=pos_np.copy(), rp=rp, send=send_i32)
                return rp, send_i32
            except Exception:  # noqa: BLE001
                pass
    if build is None and device.type == "cuda":
        td3 = _get_triton_d3()
        if td3 is not None and getattr(td3, "has_triton_d3", lambda *a: False)(device):
            from lam4.calc.physics import cached_vesin_nl

            i_raw, j_raw = cached_vesin_nl(float(r_cut) + SKIN).compute(
                points=center_positions_f64(pos_np)[0], box=np.zeros((3, 3)), periodic=(False, False, False), quantities="ij"
            )
            rp, send_i32 = td3.triton_build_csr_i32(*_pack_ij_i32(i_raw, j_raw, device), n)
            _CACHE.clear()
            _CACHE.update(n=n, device=device, r_cut=r_cut, z=z_np.copy(), pos0=pos_np.copy(), rp=rp, send=send_i32)
            return rp, send_i32
    if build is None and device.type == "cpu":
        rp, send_i32, _ = _build_raw_d3_csr(n, pos_np, None, (False, False, False), float(r_cut) + SKIN, device)
        _CACHE.clear()
        _CACHE.update(n=n, device=device, r_cut=r_cut, z=z_np.copy(), pos0=pos_np.copy(), rp=rp, send=send_i32)
        return rp, send_i32
    if build is None:
        from lam4.calc.physics import build_edge_tensors

        build = lambda p, rc: build_edge_tensors(
            p, np.eye(3, dtype=np.float64) * 100.0, (False, False, False), rc, torch.float32, device=device
        )[0]
    ei = build(pos_np, float(r_cut) + SKIN).to(device)
    recv_i32, send_i32 = ei[1].to(torch.int32).contiguous(), ei[0].to(torch.int32).contiguous()
    rp = torch.zeros(n + 1, dtype=torch.int32, device=device)
    rp[1:] = torch.searchsorted(recv_i32, torch.arange(n, dtype=torch.int32, device=device), right=True).to(torch.int32)
    _CACHE.clear()
    _CACHE.update(n=n, device=device, r_cut=r_cut, z=z_np.copy(), pos0=pos_np.copy(), rp=rp, send=send_i32)
    return rp, send_i32


def _build_raw_d3_csr(
    n: int, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool],
    r_list: float, device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None]:
    p_t = (bool(pbc_flags[0]), bool(pbc_flags[1]), bool(pbc_flags[2]))
    pbc_any = any(p_t)
    c_np = np.zeros((3, 3), dtype=np.float64) if cell_np is None else np.asarray(cell_np, dtype=np.float64)
    if device.type == "cpu":
        from lam4.calc.physics import cached_vesin_nl

        pos_c = center_positions_f64(pos_np)[0]
        if pbc_any:
            i_raw, j_raw, s_raw = cached_vesin_nl(r_list).compute(points=pos_c, box=c_np, periodic=p_t, quantities="ijS")
            sh_t = torch.from_numpy(np.ascontiguousarray(s_raw, dtype=np.float32))
        else:
            i_raw, j_raw = cached_vesin_nl(r_list).compute(points=pos_c, box=np.zeros((3, 3)), periodic=p_t, quantities="ij")
            sh_t = None
        i_a = np.asarray(i_raw, dtype=np.int32)
        j_a = np.asarray(j_raw, dtype=np.int32)
        if len(i_a) > 1 and not bool(np.all(i_a[:-1] <= i_a[1:])):
            ord_idx = np.argsort(i_a, kind="mergesort")
            i_a, j_a = i_a[ord_idx], j_a[ord_idx]
            if sh_t is not None:
                sh_t = sh_t.index_select(0, torch.from_numpy(ord_idx.astype(np.int64)))
        rp_np = np.zeros(n + 1, dtype=np.int32)
        if len(i_a) > 0:
            np.cumsum(np.bincount(i_a, minlength=n), out=rp_np[1:])
        return torch.from_numpy(rp_np), torch.from_numpy(np.ascontiguousarray(j_a)), sh_t
    if device.type == "mps":
        from lam4.calc.physics import cached_vesin_nl
        from lam4.kernels.metal_d3 import has_metal_d3, metal_build_csr_i32

        if has_metal_d3():
            try:
                if pbc_any:
                    i_raw, j_raw, s_raw = cached_vesin_nl(r_list).compute(points=center_positions_f64(pos_np)[0], box=c_np, periodic=p_t, quantities="ijS")
                    r_t, s_t = _pack_ij_i32(i_raw, j_raw, device)
                    sh_t = torch.from_numpy(np.ascontiguousarray(s_raw, dtype=np.float32)).to(device)
                    return metal_build_csr_i32(r_t, s_t, n, edge_shift=sh_t)
                i_raw, j_raw = cached_vesin_nl(r_list).compute(points=center_positions_f64(pos_np)[0], box=np.zeros((3, 3)), periodic=p_t, quantities="ij")
                rp, send_i32 = metal_build_csr_i32(*_pack_ij_i32(i_raw, j_raw, device), n)
                return rp, send_i32, None
            except Exception:  # noqa: BLE001
                pass
    td3 = _get_triton_d3() if device.type == "cuda" else None
    if td3 is not None and getattr(td3, "has_triton_d3", lambda *a: False)(device) and hasattr(td3, "triton_build_csr_i32"):
        from lam4.calc.physics import cached_vesin_nl

        if pbc_any:
            i_raw, j_raw, s_raw = cached_vesin_nl(r_list).compute(points=center_positions_f64(pos_np)[0], box=c_np, periodic=p_t, quantities="ijS")
            r_t, s_t = _pack_ij_i32(i_raw, j_raw, device)
            sh_t = torch.from_numpy(np.ascontiguousarray(s_raw, dtype=np.float32)).to(device)
            return td3.triton_build_csr_i32(r_t, s_t, n, edge_shift=sh_t)
        i_raw, j_raw = cached_vesin_nl(r_list).compute(points=center_positions_f64(pos_np)[0], box=np.zeros((3, 3)), periodic=p_t, quantities="ij")
        rp, send_i32 = td3.triton_build_csr_i32(*_pack_ij_i32(i_raw, j_raw, device), n)
        return rp, send_i32, None
    from lam4.calc.physics import build_edge_tensors

    ei, es = build_edge_tensors(pos_np, c_np if pbc_any else None, p_t, r_list, torch.float32, device=device)
    ei, shift_f32 = ei.to(device), (es.to(device=device, dtype=torch.float32).contiguous() if pbc_any else None)
    recv_i32, send_i32 = ei[1].to(torch.int32).contiguous(), ei[0].to(torch.int32).contiguous()
    rp = torch.zeros(n + 1, dtype=torch.int32, device=device)
    rp[1:] = torch.searchsorted(recv_i32, torch.arange(n, dtype=torch.int32, device=device), right=True).to(torch.int32)
    return rp, send_i32, shift_f32


def _periodic_cache_valid(
    c: dict[str, Any], pos_np: np.ndarray, c_arr: np.ndarray,
    pbc_flags: tuple[bool, bool, bool], r_cut: float, eff_skin: float,
) -> bool:
    cell0 = c["cell0"]
    pos0 = c["pos0"]
    if np.allclose(c_arr, cell0, atol=1e-12, rtol=0.0):
        return float(np.max(np.sum((pos_np - pos0) ** 2, axis=1), initial=0.0)) < (0.5 * eff_skin) ** 2
    if not all(pbc_flags):
        return False
    if abs(float(np.linalg.det(cell0))) <= 1e-8 or abs(float(np.linalg.det(c_arr))) <= 1e-8:
        return False
    def_grad = np.linalg.solve(cell0, c_arr)
    s_min = float(np.linalg.svd(def_grad, compute_uv=False)[-1])
    margin = s_min * (float(r_cut) + eff_skin) - float(r_cut) - 1.0e-6
    if margin <= 0.0:
        return False
    u = pos_np - pos0 @ def_grad
    if len(u) > 0:
        u = u - u.mean(axis=0, keepdims=True)
    max_u2 = float(np.max(np.sum(u * u, axis=1), initial=0.0))
    return (4.0 * max_u2) < (margin * margin)


def d3_csr_periodic_list(
    z_np: np.ndarray, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool],
    r_cut: float, device: torch.device, *, skin: float = SKIN,
) -> tuple[Tensor, Tensor, Tensor]:
    """`(recv_ptr int32 [N+1], send_idx int32 [E], edge_shift float32 [E, 3])` on `device` for periodic or open systems."""
    n = int(len(z_np))
    eff_skin = max(0.0, float(skin))
    c_arr = np.zeros((3, 3), dtype=np.float64) if cell_np is None else np.asarray(cell_np, dtype=np.float64)
    c = _PERIODIC_CACHE
    if (
        eff_skin > 0.0 and c.get("n") == n and c.get("device") == device
        and c.get("r_cut") == r_cut and c.get("skin") == eff_skin and c.get("pbc") == pbc_flags
        and np.array_equal(c["z"], z_np)
        and _periodic_cache_valid(c, pos_np, c_arr, pbc_flags, r_cut, eff_skin)
    ):
        return c["rp"], c["send"], c["shift"]
    rp, send_i32, shift_f32 = _build_raw_d3_csr(n, pos_np, c_arr, pbc_flags, float(r_cut) + eff_skin, device)
    if shift_f32 is None:
        shift_f32 = torch.zeros((int(send_i32.numel()), 3), dtype=torch.float32, device=device)
    if eff_skin > 0.0:
        _PERIODIC_CACHE.clear()
        _PERIODIC_CACHE.update(
            n=n, device=device, r_cut=r_cut, skin=eff_skin, pbc=pbc_flags,
            z=z_np.copy(), cell0=c_arr.copy(), pos0=pos_np.copy(), rp=rp, send=send_i32, shift=shift_f32,
        )
    return rp, send_i32, shift_f32


def sync_padded_d3_csr(
    n: int, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool],
    skin: float, headroom: float, device: torch.device,
    cur_cap: int, d3_rp: Tensor | None, d3_send: Tensor | None, d3_shift: Tensor | None,
    *, r_cut: float = 12.0,
) -> tuple[int, Tensor, Tensor, Tensor | None, bool]:
    """Build or update static-padded `(d3_rp [n+3], d3_send [cap], d3_shift [cap,3]|None)` with 0 pad-atom edges (`d3_rp[n:] = e0`)."""
    pbc_any = any(pbc_flags)
    rp_r, send_r, shift_r = _build_raw_d3_csr(n, pos_np, cell_np, pbc_flags, float(r_cut) + float(skin), device)
    e0 = int(send_r.numel())
    regrew = d3_rp is None or d3_send is None or e0 > cur_cap
    if regrew:
        cur_cap = max(256, int(-(-int(max(e0, 1) * headroom) // 256) * 256))
        d3_rp = torch.zeros(n + 3, dtype=torch.int32, device=device)
        d3_send = torch.zeros(cur_cap, dtype=torch.int32, device=device)
        d3_shift = torch.zeros((cur_cap, 3), dtype=torch.float32, device=device) if pbc_any else None
    d3_rp[: n + 1].copy_(rp_r)
    d3_rp[n + 1 :].fill_(e0)
    if e0 > 0:
        d3_send[:e0].copy_(send_r)
    if cur_cap > e0:
        d3_send[e0:].fill_(n)
    if pbc_any and d3_shift is not None:
        if e0 > 0 and shift_r is not None:
            d3_shift[:e0].copy_(shift_r)
        if cur_cap > e0:
            d3_shift[e0:].zero_()
    return cur_cap, d3_rp, d3_send, d3_shift, regrew


def add_md_dispersion(eng: Any, e_kcal: Tensor, f_kcal: Tensor, s_kcal: Tensor | None, eff_stress: bool) -> tuple[Tensor, Tensor, Tensor | None]:
    """Accumulate D3 dispersion `(e_kcal, f_kcal, s_kcal)` on device for `MDEngine._forces`."""
    sdt = eng.state_dt
    if eng.can_graph_d3():
        td3 = _get_triton_d3()
        if td3 is not None:
            try:
                if not eng.pbc_any and eng.n <= 2048 and not eff_stress:
                    ed3, fd3, _ = td3.triton_d3_grouped(
                        eng.pb.z[: eng.n], eng.pb.pos[: eng.n].to(torch.float32), eng._d3_g0, 1,
                        d3_mode=eng.d3_mode, compute_forces=True, compute_stress=False,
                    )
                    return e_kcal + ed3[0].to(sdt), (f_kcal + fd3.to(sdt) if fd3 is not None else f_kcal), s_kcal
                ed3, fd3, sd3 = td3.triton_d3_csr(
                    eng.pb.z, eng.pb.pos.to(torch.float32), eng.d3_rp, eng.d3_send,
                    edge_shift=eng.d3_shift if eng.pbc_any else None,
                    cell=eng.pb.cell.to(torch.float32) if eng.pbc_any else None,
                    d3_mode=eng.d3_mode, atom_mask=eng.pb.atom_mask, graph=eng.pb.graph,
                    n_graphs=2, compute_forces=True, compute_stress=eff_stress,
                )
                e_out = e_kcal + ed3[0].to(sdt)
                f_out = f_kcal + fd3[: eng.n].to(sdt) if fd3 is not None else f_kcal
                s_out = s_kcal + sd3[0].to(sdt) if (eff_stress and sd3 is not None and s_kcal is not None) else s_kcal
                return e_out, f_out, s_out
            except Exception:  # noqa: BLE001
                pass
    if eng.dev.type == "mps" and eng.calc.dtype == torch.float32 and eng.d3_mode in ("bj_wb97m", "zero_pbe"):
        from lam4.kernels.metal_d3 import has_metal_d3, metal_d3_csr_full, metal_d3_grouped

        if has_metal_d3():
            try:
                if not eng.pbc_any and eng.n <= 1024 and not eff_stress:
                    ed3, fd3 = metal_d3_grouped(eng.pb.z[: eng.n], eng.pb.pos[: eng.n], eng._d3_g0, 1, d3_mode=eng.d3_mode, compute_forces=True)
                    return e_kcal + ed3[0].to(sdt), (f_kcal + fd3.to(sdt) if fd3 is not None else f_kcal), s_kcal
                if eng.d3_rp is not None and eng.d3_send is not None:
                    ed3, fd3, sd3 = metal_d3_csr_full(
                        eng.pb.z, eng.pb.pos.to(torch.float32), eng.d3_rp, eng.d3_send,
                        edge_shift=eng.d3_shift if eng.pbc_any else None,
                        cell=eng.pb.cell.to(torch.float32) if eng.pbc_any else None,
                        d3_mode=eng.d3_mode, atom_mask=eng.pb.atom_mask, graph=eng.pb.graph,
                        n_graphs=2, compute_forces=True, compute_stress=eff_stress,
                    )
                    e_out = e_kcal + ed3[0].to(sdt)
                    f_out = f_kcal + fd3[: eng.n].to(sdt) if fd3 is not None else f_kcal
                    s_out = s_kcal + sd3[0].to(sdt) if (eff_stress and sd3 is not None and s_kcal is not None) else s_kcal
                    return e_out, f_out, s_out
            except Exception:  # noqa: BLE001
                pass
    from lam4.calc.physics import eval_dispersion_single

    q_np = eng._q_cur.detach().cpu().to(torch.float64).numpy()
    de, df, ds = eval_dispersion_single(
        eng.z_np, q_np, eng.cell_np, eng.pbc, eng.d3_mode, eng.dev, eng.calc.dtype,
        compute_forces=True, compute_stress=eff_stress,
    )
    e_out = e_kcal + de
    f_out = f_kcal + torch.from_numpy(df).to(eng.dev, sdt) if df is not None else f_kcal
    s_out = s_kcal + torch.from_numpy(ds).to(eng.dev, sdt) if (eff_stress and ds is not None and s_kcal is not None) else s_kcal
    return e_out, f_out, s_out


def _fill_md_edges(pb: Any, ei: Tensor, es: Tensor, n: int, n_e: int, e_cap: int) -> None:
    if n_e > 0:
        pb.edge_index[:, :n_e].copy_(ei)
        pb.edge_shift[:n_e].copy_(es)
    if e_cap > n_e:
        pb.edge_index[0, n_e:].fill_(n)
        pb.edge_index[1, n_e:].fill_(n + 1)
        pb.edge_shift[n_e:].zero_()


def _sync_md_d3_csr(eng: Any, initial: bool, q_np: np.ndarray | None) -> None:
    need_csr_d3 = (
        eng.d3_mode is not None
        and (
            (eng.can_graph_d3() and (eng.pbc_any or eng.n > 2048))
            or (getattr(eng, "can_metal_d3", lambda: False)() and (eng.pbc_any or eng.n > 1024))
        )
    )
    if not need_csr_d3:
        return
    from lam4.kernels.metal_d3 import d3_cutoff_for_mode

    pos_host = eng.st.q.detach().cpu().numpy() if q_np is None else q_np
    prev_d3_cap = eng.d3_e_cap
    eng.d3_e_cap, eng.d3_rp, eng.d3_send, eng.d3_shift, regrew = sync_padded_d3_csr(
        eng.n, pos_host, eng.cell_np, eng.pbc, eng.skin, eng.headroom, eng.dev,
        eng.d3_e_cap, eng.d3_rp, eng.d3_send, eng.d3_shift,
        r_cut=d3_cutoff_for_mode(eng.d3_mode),
    )
    if regrew and not initial and prev_d3_cap > 0:
        eng.stats["regrows"] += 1
        eng.graph = None


def rebuild_md_buffers(eng: Any, initial: bool = False, q_np: np.ndarray | None = None) -> None:
    """Rebuild Verlet edges, static-padded `Batch`, `CSRPlan`, and D3 CSR buffers for `MDEngine`."""
    from lam4.calc.physics import OPEN_MOL_GPU_NL_MAX_ATOMS, build_open_mol_edges_gpu
    from lam4.calc.verlet import build_or_reuse_batch, sync_padded_csr_plan, verlet_neighbor_graph
    from lam4.data.padding import pad_batch

    if (
        not initial and not eng._retargeted and q_np is None and not eng.pbc_any
        and eng.n <= OPEN_MOL_GPU_NL_MAX_ATOMS and eng.is_csr and eng.dev.type in ("cuda", "mps") and eng.pb is not None
    ):
        ei, es = build_open_mol_edges_gpu(
            eng.st.q.to(eng.calc.dtype), eng.pb.graph[: eng.n], [eng.n], eng.r_cut + eng.skin, eng.calc.dtype
        )
        n_e = int(ei.shape[1])
        if n_e <= eng.e_cap:
            _fill_md_edges(eng.pb, ei, es, eng.n, n_e, eng.e_cap)
            eng.plan = sync_padded_csr_plan(None, eng.pb.edge_index, eng.n, n_e, eng.e_cap, eng.plan)
            eng._sync_chunks(fresh=False)
            _sync_md_d3_csr(eng, initial=False, q_np=None)
            return eng._finish_rebuild()

    q_np = eng.st.q.detach().cpu().numpy() if q_np is None else q_np
    eng.calc._nl_cache = None
    ei, es, _ = verlet_neighbor_graph(eng.calc, eng.z_np, q_np, eng.cell_np, eng.pbc, skin=eng.skin)
    if not eng.is_csr and eng.calc._nl_cache is not None:
        ei, es = eng.calc._nl_cache["ei"], eng.calc._nl_cache["es"]
    real_plan = eng.calc._active_csr_plan if eng.is_csr else None
    n_e = int(ei.shape[1])
    eng.e_cap_prev = eng.e_cap
    if initial or eng.pb is None or n_e > eng.e_cap:
        if not initial and eng.pb is not None:
            eng.stats["regrows"] += 1
            eng.graph = None
        b, _, _ = build_or_reuse_batch(
            eng.calc, eng.z_np, q_np, eng.cell_np, eng.pbc, eng.src, eng.u, eng.qv, eng.sv, ei, es,
            frag_id_np=getattr(eng, "fid_np", None), frag_q_np=getattr(eng, "fq_np", None), frag_s_np=getattr(eng, "fs_np", None),
            recenter=False, periodic_electro=getattr(eng, "periodic_electro", None),
        )
        eng.e_cap = max(256, int(-(-int(max(n_e, 1) * eng.headroom) // 256) * 256))
        eng.pb = pad_batch(b, eng.n + 2, eng.e_cap, 2, eng.r_cut)
        eng.pb.coulomb_cap = None
        eng.plan = sync_padded_csr_plan(real_plan, eng.pb.edge_index, eng.n, n_e, eng.e_cap, None) if eng.is_csr else None
    else:
        if eng._retargeted:
            b, _, _ = build_or_reuse_batch(
                eng.calc, eng.z_np, q_np, eng.cell_np, eng.pbc, eng.src, eng.u, eng.qv, eng.sv, ei, es,
                frag_id_np=getattr(eng, "fid_np", None), frag_q_np=getattr(eng, "fq_np", None), frag_s_np=getattr(eng, "fs_np", None),
                recenter=False, periodic_electro=getattr(eng, "periodic_electro", None),
            )
            nb = pad_batch(b, eng.n + 2, eng.e_cap, 2, eng.r_cut)
            if eng.pb.domain != nb.domain or eng.pb.periodic_electro != nb.periodic_electro or (eng.pb.frag_id is None) != (nb.frag_id is None) or (eng.pb.frag_charge is not None and nb.frag_charge is not None and eng.pb.frag_charge.shape != nb.frag_charge.shape):
                eng.pb, eng.graph = nb, None
                eng.pb.coulomb_cap = None
            else:
                eng.pb.coulomb_cap = None
                for name in ("z", "u_flag", "token", "ref_id", "charge", "spin", "xc_id", "paw_id", "u_atom", "frag_id", "frag_charge", "frag_spin"):
                    dst, src_t = getattr(eng.pb, name), getattr(nb, name)
                    if dst is not None and src_t is not None and dst.shape == src_t.shape:
                        dst.copy_(src_t)
        _fill_md_edges(eng.pb, ei, es, eng.n, n_e, eng.e_cap)
        if eng.is_csr:
            eng.plan = sync_padded_csr_plan(real_plan, eng.pb.edge_index, eng.n, n_e, eng.e_cap, eng.plan)
    eng._retargeted = False
    eng._warm_coulomb_meta()
    eng._sync_chunks(fresh=initial or n_e > eng.e_cap_prev)
    _sync_md_d3_csr(eng, initial=initial, q_np=q_np)
    eng._finish_rebuild()


def clear() -> None:
    _CACHE.clear()
    _PERIODIC_CACHE.clear()
