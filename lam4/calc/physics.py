"""Physical unit conversions, LoT routing, dispersion add-back, and interpretability helpers for MenaCalculator."""
from __future__ import annotations

import threading
from typing import Any, Sequence

from ase import Atoms
import ase.units
import numpy as np
import torch
from torch import Tensor

from lam.models.dispersion import ANGSTROM_TO_BOHR, D3_K1, D3_K3, D3BJ, HARTREE_TO_KCAL
from lam.units import EV as CRYSTAL_EV
from lam4.core.angular import l2_mat6
from lam4.data.crystal import mp_hubbard_u_atoms
from lam4.registry import CRYSTAL, LAM6_LOT_SPECS, XC_ID, build_sources
from lam4.registry.sources import Source, lot_ids

from lam4.calc.fragments import (  # noqa: F401
    _parse_frag_dict_keys, assemble_fragment_tensors, resolve_charge_and_spin, resolve_charge_spin_fragments,
)
from lam4.calc.open_mol_gpu import (  # noqa: F401
    OPEN_MOL_GPU_D3_CSR_MAX_ATOMS, OPEN_MOL_GPU_NL_MAX_ATOMS, build_open_mol_edges_gpu, eval_batched_open_mol_d3,
)
from lam4.calc.precision_hygiene import center_positions_f64
from lam4.calc.spectroscopy import (
    E_ANG_TO_DEBYE, IR_KM_MOL_FACTOR, build_energy_decomposition,
    harmonic_ir_raman, polarizability_from_sums, polarizability_tensor,
)
from lam4.registry.hubbard import MP_HUBBARD_ANIONS, MP_HUBBARD_METALS, _sanitized_box, mp_hubbard_u_flag

MOL_KCAL_TO_EV = float(ase.units.kcal / ase.units.mol)

LOT_ALIASES: dict[str, str] = {
    "omol": "omol_wb97mv", "omol25": "omol_wb97mv", "wb97mv": "omol_wb97mv", "wb97m-v": "omol_wb97mv",
    "opoly": "opoly26_wb97mv", "opoly26": "opoly26_wb97mv", "opoly_wb97mv": "opoly26_wb97mv",
    "spice": "spice_wb97md3", "wb97md3": "spice_wb97md3",
    "openrxn": "openrxn_wb97xd", "wb97xd": "openrxn_wb97xd",
    "t1x": "t1x_wb97x", "qcml": "qcml_pbe0", "rgd1": "rgd1_b3lyp", "tmqm": "tmqm_tpssh",
    "splinter": "splinter_sapt0", "splinter_jun": "splinter_sapt0",
    "r2scan": "matpes_r2scan", "matpes": "matpes_r2scan", "pbe_no_u": "matpes_pbe", "matpes_pbe64": "matpes_pbe",
    "mp_pbe52": "mp_pbe", "mp_pbe52u": "mp_pbeU", "omat_pbe54": "omat_pbe", "omat_pbe54u": "omat_pbeU",
    "oc20": "oc20_rpbe", "rpbe": "oc20_rpbe", "oc25": "oc25_rpbe",
    "mad": "mad16_r2scan", "mad16": "mad16_r2scan", "mad_r2scan": "mad16_r2scan",
    "omc": "omc25_pbe", "omc25": "omc25_pbe",
}

_SOURCES_CACHE: dict[str, Source] | None = None
_D3_CACHE: dict[tuple[str, torch.dtype], D3BJ] = {}
_VESIN_TLS = threading.local()


def _get_triton_d3() -> Any | None:
    try:
        from lam4.kernels import triton_d3

        return triton_d3
    except ImportError:
        return None


def cached_vesin_nl(r_cut: float) -> Any:
    import vesin

    cache = getattr(_VESIN_TLS, "cache", None)
    if cache is None:
        cache = {}
        _VESIN_TLS.cache = cache
    key = round(float(r_cut), 6)
    nl = cache.get(key)
    if nl is None:
        try:
            nl = vesin.NeighborList(cutoff=float(r_cut), full_list=True, sorted=True)
        except TypeError:
            nl = vesin.NeighborList(cutoff=float(r_cut), full_list=True)
        cache[key] = nl
    return nl


def sources_registry() -> dict[str, Source]:
    global _SOURCES_CACHE
    if _SOURCES_CACHE is None:
        srcs = build_sources()
        for name, spec in LAM6_LOT_SPECS.items():
            if name not in srcs:
                u_flag = None if spec.u_policy == "mp2020_auto" else (1 if spec.u_policy == "fixed_one" else 0)
                srcs[name] = Source(
                    name=name, domain=spec.domain_legacy, pbc=spec.pbc, token=spec.xc,
                    token_id=int(XC_ID.get(spec.xc, 0)), u_flag=u_flag, d3_addback=spec.d3_addback,
                    e0_file=None, e0_column=0, energy_unit=spec.energy_unit,
                    ev_to_kcal=(CRYSTAL_EV if spec.domain_legacy == CRYSTAL else 1.0), share=None,
                    xc=spec.xc, xc_id=spec.xc_id, paw_basis=spec.paw_basis, paw_id=spec.paw_id,
                )
        _SOURCES_CACHE = srcs
    return _SOURCES_CACHE


def resolve_lot_and_u(atoms: Atoms, lot: str = "auto", u_flag: int | None = None) -> tuple[Source, int]:
    """Resolve `(Source, u_flag)` from structure PBC, local coordination (MP +U rule), and user `lot`."""
    srcs = sources_registry()
    z_arr = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    pos_arr = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell_arr = np.asarray(atoms.cell.array, dtype=np.float64)
    pbc_arr = np.asarray(atoms.get_pbc(), dtype=bool)
    pbc_any = bool(np.any(pbc_arr))
    raw = str(atoms.info.get("lot", lot) if lot == "auto" else lot).strip()
    key = LOT_ALIASES.get(raw.lower(), raw)
    u_eff = atoms.info.get("u_flag") if u_flag is None else u_flag

    def _u() -> int:
        return mp_hubbard_u_flag(z_arr, pos_arr, cell_arr, pbc_arr) if u_eff is None else int(u_eff)

    if key.lower() in ("auto", "default"):
        if pbc_any:
            u = _u()
            name = "mp_pbeU" if u else "mp_pbe"
        else:
            u = 0 if u_eff is None else int(u_eff)
            name = "omol_wb97mv"
    elif key.lower() in ("mp", "pbe", "mptrj", "salex"):
        u = _u()
        name = "mp_pbeU" if u else "mp_pbe"
    elif key.lower() in ("omat", "omat24"):
        u = _u()
        name = "omat_pbeU" if u else "omat_pbe"
    elif key.lower() == "oc22":
        u = _u()
        name = "oc22_pbeU" if u else "mp_pbe"
    elif key in srcs:
        name = key
        u = (_u() if srcs[name].u_flag is None else int(srcs[name].u_flag)) if u_eff is None else int(u_eff)
    else:
        raise ValueError(f"Unknown level of theory {lot!r}; valid keys: {sorted(srcs)}")
    return srcs[name], u


def lot_u_atoms_for_calc(src_name: str, z_np: np.ndarray, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags, u_val: int) -> tuple[int, int, np.ndarray]:
    """Return `(xc_id, paw_id, u_atom [N])` aligned with `CrystalSource` / `OmolSource` training batches."""
    xc_v, paw_v = lot_ids(src_name)
    if int(u_val):
        u_at = mp_hubbard_u_atoms(z_np, pos_np, cell_np, pbc_flags, int(u_val))
    else:
        u_at = np.zeros(len(z_np), dtype=np.int64)
    return xc_v, paw_v, np.asarray(u_at, dtype=np.int64)


def effective_periodic_cell(atoms: Atoms, r_cut: float) -> tuple[np.ndarray, tuple[bool, bool, bool]]:
    """Non-singular 3x3 cell for 3D crystals, 2D slabs, and 1D wires (including tilted axes) for `vesin`."""
    pbc = tuple(bool(x) for x in atoms.get_pbc())
    cell = np.asarray(atoms.get_cell().array, dtype=np.float64)
    if all(pbc) and abs(float(np.linalg.det(cell))) > 1e-8:
        return cell.copy(), pbc
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    return _sanitized_box(pos, cell, pbc, cutoff=float(r_cut)), pbc


def build_edge_tensors(
    pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool],
    r_cut: float, dtype: torch.dtype, *, device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    box = np.zeros((3, 3), dtype=np.float64) if cell_np is None else np.asarray(cell_np, dtype=np.float64)
    p = (bool(pbc_flags[0]), bool(pbc_flags[1]), bool(pbc_flags[2]))
    pos_c, _ = center_positions_f64(pos_np)
    np_dt = np.float32 if dtype == torch.float32 else np.float64
    if any(p):
        i, j, s = cached_vesin_nl(r_cut).compute(
            points=pos_c, box=box, periodic=p, quantities="ijS"
        )
        i_a, j_a, s_a = np.asarray(i, np.int64), np.asarray(j, np.int64), np.asarray(s, np_dt)
        ei_t = torch.from_numpy(np.ascontiguousarray(np.stack([j_a, i_a], axis=0)))
        es_t = torch.from_numpy(np.ascontiguousarray(s_a)).to(dtype=dtype)
        return (ei_t.to(device), es_t.to(device)) if (device is not None and device.type == "cuda") else (ei_t, es_t)
    i, j = cached_vesin_nl(r_cut).compute(
        points=pos_c, box=np.zeros((3, 3)), periodic=p, quantities="ij"
    )
    i_a, j_a = np.asarray(i, np.int64), np.asarray(j, np.int64)
    ei_t = torch.from_numpy(np.ascontiguousarray(np.stack([j_a, i_a], axis=0)))
    if device is not None and device.type == "cuda":
        return ei_t.to(device), torch.zeros((len(i_a), 3), dtype=dtype, device=device)
    return ei_t, torch.zeros((len(i_a), 3), dtype=dtype)


def _get_d3(device: torch.device, dtype: torch.dtype) -> D3BJ:
    key = (str(device), dtype)
    mod = _D3_CACHE.get(key)
    if mod is None:
        mod = D3BJ().to(device=device, dtype=dtype).eval()
        _D3_CACHE[key] = mod
    return mod


def _d3_zero_pbe_sparse(d3: D3BJ, z: Tensor, edge_index: Tensor, edge_diff: Tensor) -> Tensor:
    """PBE-D3(0) zero-damping dispersion energy (kcal/mol) for periodic OMC25 molecular crystals."""
    s6, s8, rs6, alp = 1.0, 0.722, 1.217, 14.0
    n = z.shape[0]
    zc = z.clamp(max=d3.rcov.shape[0] - 1)
    ei, ej = edge_index[0], edge_index[1]
    r_ang = (edge_diff.pow(2).sum(-1) + 1e-8).sqrt()
    r_bohr = r_ang * ANGSTROM_TO_BOHR
    zi, zj = zc[ei], zc[ej]
    rr = (d3.rcov[zi] + d3.rcov[zj]) / r_bohr.clamp_min(1e-6)
    damp = (1.0 / (1.0 + torch.exp(-D3_K1 * (rr - 1.0)))) * (r_bohr <= d3.cn_cut * ANGSTROM_TO_BOHR)
    cn = damp.new_zeros(n).index_add_(0, ei, damp)
    refs = d3.c6ab_flat.index_select(0, zi * d3.c6ab.shape[1] + zj)
    cn0 = refs[..., 0]
    r2 = (refs[..., 1] - cn[ei].unsqueeze(-1)) ** 2 + (refs[..., 2] - cn[ej].unsqueeze(-1)) ** 2
    c6 = (torch.softmax(torch.where(cn0 > 0.0, D3_K3 * r2, torch.full_like(r2, -1e20)), dim=-1) * cn0).sum(-1)
    r0_ij = torch.sqrt(3.0 * d3.r2r4[zi] * d3.r2r4[zj]).clamp_min(1e-6)
    c8 = 3.0 * c6 * d3.r2r4[zi] * d3.r2r4[zj]
    f6 = 1.0 / (1.0 + 6.0 * (r_bohr / (rs6 * r0_ij)).pow(-alp))
    f8 = 1.0 / (1.0 + 6.0 * (r_bohr / r0_ij).pow(-(alp + 2.0)))
    r2_b = r_bohr.pow(2)
    r6 = r2_b.pow(3)
    t = ((r_ang - d3.taper_on) / max(1e-6, d3.r_cut - d3.taper_on)).clamp(0.0, 1.0)
    taper = 1.0 - t * t * t * (10.0 + t * (6.0 * t - 15.0))
    ep = -0.5 * (s6 * c6 * f6 / r6.clamp_min(1e-12) + s8 * c8 * f8 / (r6 * r2_b).clamp_min(1e-12)) * taper * (r_ang <= d3.r_cut)
    return (ep.sum() * HARTREE_TO_KCAL).unsqueeze(0)


def _unpack_d3_host(e_t: Tensor, f_t: Tensor | None, s_t: Tensor | None, n: int) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    parts = [e_t.detach().reshape(1)]
    if f_t is not None:
        parts.append(f_t.detach().reshape(-1))
    if s_t is not None:
        parts.append(s_t.detach().reshape(-1))
    buf = torch.cat(parts, dim=0).cpu().to(torch.float64).numpy()
    off = 1
    f_np = buf[off : off + 3 * n].reshape(n, 3).copy() if f_t is not None else None
    if f_t is not None:
        off += 3 * n
    s_np = buf[off : off + 9].reshape(3, 3).copy() if s_t is not None else None
    return float(buf[0]), f_np, s_np


def eval_dispersion_device(
    z_np: np.ndarray, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool],
    d3_mode: str | None, device: torch.device, dtype: torch.dtype, *, compute_forces: bool = True, compute_stress: bool = False,
    z_t: Tensor | None = None, pos_t: Tensor | None = None, cell_t: Tensor | None = None,
) -> tuple[bool, Any, Any, Any]:
    """Evaluate dispersion, returning `(on_device, E, F, S)`.

    When `on_device=True`, `(E scalar Tensor, F [N,3] Tensor|None, S [3,3] Tensor|None)` stay on `device` (no D2H sync).
    When `on_device=False`, returns host `(float, np.ndarray|None, np.ndarray|None)`.
    """
    n = len(z_np)
    if d3_mode is None or n == 0:
        return False, 0.0, None, None
    pos_raw = pos_np
    pos_np, _ = center_positions_f64(pos_np)
    pbc_any = bool(any(pbc_flags))
    eff_stress = bool(compute_stress and pbc_any)
    if device.type == "cuda" and dtype == torch.float32 and d3_mode in ("bj_wb97m", "zero_pbe"):
        td3 = _get_triton_d3()
        if td3 is not None and getattr(td3, "has_triton_d3", lambda *a: False)(device):
            try:
                zd = z_t if (z_t is not None and z_t.device == device and z_t.shape[0] == n) else torch.as_tensor(z_np, dtype=torch.long, device=device)
                pd = pos_t.detach() if (pos_t is not None and pos_t.device == device and pos_t.shape[0] == n and pos_t.dtype == torch.float32) else torch.as_tensor(pos_np, dtype=torch.float32, device=device)
                if not pbc_any:
                    if n <= 2048:
                        e_t, f_t, s_t = td3.triton_d3_grouped(
                            zd, pd, torch.zeros(n, dtype=torch.long, device=device), 1,
                            d3_mode=d3_mode, compute_forces=compute_forces, compute_stress=False,
                        )
                    else:
                        from lam4.calc.d3_nl_cache import d3_csr_list

                        rp, send_i32 = d3_csr_list(z_np, pos_raw, 12.0, device)
                        e_t, f_t, s_t = td3.triton_d3_csr(
                            zd, pd, rp, send_i32, d3_mode=d3_mode, compute_forces=compute_forces, compute_stress=False,
                        )
                else:
                    from lam4.calc.d3_nl_cache import d3_csr_periodic_list

                    c_np = np.eye(3, dtype=np.float64) * 100.0 if cell_np is None else np.asarray(cell_np, dtype=np.float64)
                    rp, send_i32, shift_f32 = d3_csr_periodic_list(z_np, pos_raw, c_np, pbc_flags, 12.0, device)
                    cd = cell_t.detach() if (cell_t is not None and cell_t.device == device and cell_t.dtype == torch.float32) else torch.as_tensor(c_np, dtype=torch.float32, device=device).unsqueeze(0)
                    e_t, f_t, s_t = td3.triton_d3_csr(
                        zd, pd, rp, send_i32, edge_shift=shift_f32, cell=cd,
                        d3_mode=d3_mode, compute_forces=compute_forces, compute_stress=eff_stress,
                    )
                return True, e_t[0], (f_t if compute_forces else None), (s_t[0] if (eff_stress and s_t is not None) else None)
            except Exception:
                pass
    if device.type == "mps" and dtype == torch.float32 and d3_mode in ("bj_wb97m", "zero_pbe"):
        from lam4.kernels.metal_d3 import has_metal_d3, metal_d3_csr_full, metal_d3_grouped

        if has_metal_d3():
            try:
                zd = z_t if (z_t is not None and z_t.device == device and z_t.shape[0] == n) else torch.as_tensor(z_np, dtype=torch.long, device=device)
                pd = pos_t.detach() if (pos_t is not None and pos_t.device == device and pos_t.shape[0] == n and pos_t.dtype == torch.float32) else torch.as_tensor(pos_np, dtype=torch.float32, device=device)
                if not pbc_any:
                    if n <= 1024:
                        e_t, f_t = metal_d3_grouped(zd, pd, torch.zeros(n, dtype=torch.long, device=device), 1, d3_mode=d3_mode, compute_forces=compute_forces)
                        s_t = None
                    else:
                        from lam4.calc.d3_nl_cache import d3_csr_list

                        rp, send_i32 = d3_csr_list(z_np, pos_raw, 12.0, device)
                        e_t, f_t, s_t = metal_d3_csr_full(zd, pd, rp, send_i32, d3_mode=d3_mode, compute_forces=compute_forces, compute_stress=False)
                else:
                    from lam4.calc.d3_nl_cache import d3_csr_periodic_list

                    c_np = np.eye(3, dtype=np.float64) * 100.0 if cell_np is None else np.asarray(cell_np, dtype=np.float64)
                    rp, send_i32, shift_f32 = d3_csr_periodic_list(z_np, pos_raw, c_np, pbc_flags, 12.0, device)
                    cd = cell_t.detach() if (cell_t is not None and cell_t.device == device and cell_t.dtype == torch.float32) else torch.as_tensor(c_np, dtype=torch.float32, device=device).unsqueeze(0)
                    e_t, f_t, s_t = metal_d3_csr_full(
                        zd, pd, rp, send_i32, edge_shift=shift_f32, cell=cd,
                        d3_mode=d3_mode, compute_forces=compute_forces, compute_stress=eff_stress,
                    )
                return True, e_t[0], (f_t if compute_forces else None), (s_t[0] if (eff_stress and s_t is not None) else None)
            except Exception:
                pass
    if device.type == "cpu" and dtype in (torch.float32, torch.float64) and d3_mode in ("bj_wb97m", "zero_pbe"):
        from lam4.kernels.cpu_d3 import cpu_d3_csr, cpu_d3_grouped, has_cpu_d3

        if has_cpu_d3():
            try:
                zd = torch.as_tensor(z_np, dtype=torch.long)
                pd = torch.as_tensor(pos_np, dtype=dtype)
                if not pbc_any:
                    if n <= 512:
                        e_t, f_t = cpu_d3_grouped(zd, pd, torch.zeros(n, dtype=torch.long), 1, d3_mode=d3_mode, compute_forces=compute_forces)
                        s_t = None
                    else:
                        from lam4.calc.d3_nl_cache import d3_csr_list

                        rp, send_i32 = d3_csr_list(z_np, pos_raw, 12.0, device)
                        e_t, f_t, s_t = cpu_d3_csr(zd, pd, rp, send_i32, d3_mode=d3_mode, compute_forces=compute_forces, compute_stress=False)
                else:
                    from lam4.calc.d3_nl_cache import d3_csr_periodic_list

                    c_np = np.eye(3, dtype=np.float64) * 100.0 if cell_np is None else np.asarray(cell_np, dtype=np.float64)
                    rp, send_i32, shift_f32 = d3_csr_periodic_list(z_np, pos_raw, c_np, pbc_flags, 12.0, device)
                    cd = torch.as_tensor(c_np, dtype=dtype).unsqueeze(0)
                    e_t, f_t, s_t = cpu_d3_csr(
                        zd, pd, rp, send_i32, edge_shift=shift_f32, cell=cd,
                        d3_mode=d3_mode, compute_forces=compute_forces, compute_stress=eff_stress,
                    )
                f_np = f_t.to(torch.float64).numpy().copy() if (compute_forces and f_t is not None) else None
                s_np = s_t[0].to(torch.float64).numpy().copy() if (eff_stress and s_t is not None) else None
                return False, float(e_t[0].item()), f_np, s_np
            except Exception:
                pass
    d3 = _get_d3(device, dtype)
    zd = torch.as_tensor(z_np, dtype=torch.long, device=device)
    if not pbc_any and d3_mode == "bj_wb97m" and n > 64:
        from lam4.calc.d3_chunked import eval_d3_bj_open_chunked

        ei, _ = build_edge_tensors(pos_np, np.eye(3, dtype=np.float64) * 100.0, (False, False, False), d3.r_cut, dtype, device=device)
        pd = torch.as_tensor(pos_np, dtype=dtype, device=device)
        e_kcal, f_np = eval_d3_bj_open_chunked(d3, zd, pd, ei.to(device), compute_forces=compute_forces)
        return False, e_kcal, f_np, None
    leaf = torch.as_tensor(pos_np, dtype=dtype, device=device)
    need_grad = compute_forces or eff_stress
    if need_grad:
        leaf = leaf.detach().requires_grad_(True)
    with torch.set_grad_enabled(need_grad):
        if not pbc_any and d3_mode == "bj_wb97m" and n <= 64:
            e_t = d3(zd.unsqueeze(0), leaf.unsqueeze(0), torch.ones((1, n), dtype=torch.bool, device=device))
            eps = None
        else:
            c_np = np.eye(3, dtype=np.float64) * 100.0 if cell_np is None else cell_np
            ei, es = build_edge_tensors(pos_np, c_np, pbc_flags, d3.r_cut, dtype, device=device)
            ei, es = ei.to(device), es.to(device)
            send, recv = ei[0], ei[1]
            pos_cur, eps = leaf, None
            cd = torch.as_tensor(c_np, dtype=dtype, device=device)
            if eff_stress:
                eps = leaf.new_zeros(3, 3).requires_grad_(True)
                sym = 0.5 * (eps + eps.T)
                pos_cur = leaf + leaf @ sym
                cd = cd + cd @ sym
            diff = pos_cur.index_select(0, send) - pos_cur.index_select(0, recv) + (es @ cd if pbc_any else 0.0)
            e_idx = torch.stack([recv, send], dim=0)
            e_t = _d3_zero_pbe_sparse(d3, zd, e_idx, diff) if d3_mode == "zero_pbe" else d3.forward_sparse(zd, pos_cur, torch.zeros_like(zd), 1, e_idx, edge_diff=-diff)
        f_dev = s_dev = None
        if need_grad:
            wrt = ([leaf] if compute_forces else []) + ([eps] if (eff_stress and eps is not None) else [])
            grads = torch.autograd.grad(e_t.sum(), wrt) if wrt else ()
            if compute_forces:
                f_dev = -grads[0]
            if eff_stress:
                vol = abs(float(np.linalg.det(cell_np))) if cell_np is not None else 0.0
                s_dev = (grads[-1] / vol) if (eps is not None and vol > 1e-8) else leaf.new_zeros((3, 3))
    e_h, f_h, s_h = _unpack_d3_host(e_t[0], f_dev, s_dev, n)
    return False, e_h, f_h, s_h


def eval_dispersion_single(
    z_np: np.ndarray, pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool],
    d3_mode: str | None, device: torch.device, dtype: torch.dtype, *, compute_forces: bool = True, compute_stress: bool = False,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Evaluate dispersion `(E_kcal, F_kcal_A [N,3] | None, S_kcal_A3 [3,3] | None)`."""
    on_dev, e_v, f_v, s_v = eval_dispersion_device(
        z_np, pos_np, cell_np, pbc_flags, d3_mode, device, dtype, compute_forces=compute_forces, compute_stress=compute_stress,
    )
    if on_dev:
        return _unpack_d3_host(e_v, f_v, s_v, len(z_np))
    return float(e_v), f_v, s_v
