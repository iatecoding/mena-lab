"""Physical unit conversions, LoT routing, dispersion add-back, and interpretability helpers for MenaCalculator."""
from __future__ import annotations

from typing import Any

from ase import Atoms
import ase.units
import numpy as np
import torch
from torch import Tensor

from lam.models.dispersion import ANGSTROM_TO_BOHR, D3_K1, D3_K3, D3BJ, HARTREE_TO_KCAL
from lam.units import EV as CRYSTAL_EV
from lam4.core.angular import l2_mat6
from lam4.data.neighbors import vesin_edges
from lam4.registry import build_sources
from lam4.registry.sources import Source

from lam4.calc.spectroscopy import (
    E_ANG_TO_DEBYE,
    IR_KM_MOL_FACTOR,
    build_energy_decomposition,
    harmonic_ir_raman,
    polarizability_from_sums,
    polarizability_tensor,
)

MOL_KCAL_TO_EV = float(ase.units.kcal / ase.units.mol)

MP_HUBBARD_METALS = frozenset({23, 24, 25, 26, 27, 28, 42, 74})  # V, Cr, Mn, Fe, Co, Ni, Mo, W
MP_HUBBARD_ANIONS = frozenset({8, 9})                             # O, F

LOT_ALIASES: dict[str, str] = {
    "omol": "omol_wb97mv",
    "omol25": "omol_wb97mv",
    "wb97mv": "omol_wb97mv",
    "wb97m-v": "omol_wb97mv",
    "spice": "spice_wb97md3",
    "wb97md3": "spice_wb97md3",
    "openrxn": "openrxn_wb97xd",
    "wb97xd": "openrxn_wb97xd",
    "r2scan": "matpes_r2scan",
    "matpes": "matpes_r2scan",
    "pbe_no_u": "matpes_pbe",
    "oc20": "oc20_rpbe",
    "rpbe": "oc20_rpbe",
    "oc22": "oc22_pbeU",
    "omc": "omc25_pbe",
    "omc25": "omc25_pbe",
}

_SOURCES_CACHE: dict[str, Source] | None = None
_D3_CACHE: dict[tuple[str, torch.dtype], D3BJ] = {}
_VESIN_CACHE: dict[float, Any] = {}


def cached_vesin_nl(r_cut: float) -> Any:
    import vesin

    key = round(float(r_cut), 6)
    nl = _VESIN_CACHE.get(key)
    if nl is None:
        nl = vesin.NeighborList(cutoff=float(r_cut), full_list=True)
        _VESIN_CACHE[key] = nl
    return nl


def sources_registry() -> dict[str, Source]:
    global _SOURCES_CACHE
    if _SOURCES_CACHE is None:
        _SOURCES_CACHE = build_sources()
    return _SOURCES_CACHE


def mp_hubbard_u_flag(z_arr: np.ndarray) -> int:
    zs = {int(z) for z in z_arr.tolist() if z > 0}
    return 1 if ((zs & MP_HUBBARD_ANIONS) and (zs & MP_HUBBARD_METALS)) else 0


def resolve_lot_and_u(
    atoms: Atoms,
    lot: str = "auto",
    u_flag: int | None = None,
) -> tuple[Source, int]:
    """Resolve `(Source, u_flag)` from structure PBC, composition (MP2020 +U rule), and user `lot`."""
    srcs = sources_registry()
    z_arr = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    pbc_any = bool(np.any(atoms.get_pbc()))
    raw = str(atoms.info.get("lot", lot) if lot == "auto" else lot).strip()
    key = LOT_ALIASES.get(raw.lower(), raw)

    if key.lower() in ("auto", "default"):
        if pbc_any:
            u = mp_hubbard_u_flag(z_arr) if u_flag is None else int(u_flag)
            name = "mp_pbeU" if u else "mp_pbe"
        else:
            u = 0 if u_flag is None else int(u_flag)
            name = "omol_wb97mv"
    elif key.lower() in ("mp", "pbe", "mptrj", "salex"):
        u = mp_hubbard_u_flag(z_arr) if u_flag is None else int(u_flag)
        name = "mp_pbeU" if u else "mp_pbe"
    elif key.lower() in ("omat", "omat24"):
        u = mp_hubbard_u_flag(z_arr) if u_flag is None else int(u_flag)
        name = "omat_pbeU" if u else "omat_pbe"
    elif key in srcs:
        name = key
        u = int(srcs[name].u_flag or 0) if u_flag is None else int(u_flag)
    else:
        raise ValueError(f"Unknown level of theory {lot!r}; valid keys: {sorted(srcs)}")
    return srcs[name], u


def resolve_charge_and_spin(atoms: Atoms, charge: float | None = None, spin: int | None = None, *, pbc_any: bool = False) -> tuple[float, int]:
    if pbc_any:
        return 0.0, 1
    q = charge if charge is not None else atoms.info.get("charge", 0.0)
    q_val = float(q)
    s = spin if spin is not None else atoms.info.get("spin", atoms.info.get("multiplicity", None))
    if s is None:
        n_elec = int(np.sum(atoms.get_atomic_numbers())) - int(round(q_val))
        s_val = 1 if (n_elec % 2 == 0) else 2
    else:
        s_val = max(1, int(round(float(s))))
    return q_val, s_val


def effective_periodic_cell(atoms: Atoms, r_cut: float) -> tuple[np.ndarray, tuple[bool, bool, bool]]:
    """Non-singular 3x3 cell for 3D crystals, 2D slabs, and 1D wires so `vesin` handles mixed PBC cleanly."""
    pbc = tuple(bool(x) for x in atoms.get_pbc())
    cell = np.asarray(atoms.get_cell().array, dtype=np.float64).copy()
    if all(pbc) and abs(float(np.linalg.det(cell))) > 1e-8:
        return cell, pbc
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    span = (pos.max(axis=0) - pos.min(axis=0)) if len(pos) > 0 else np.zeros(3)
    pad = np.maximum(span + 2.0 * r_cut + 4.0, 20.0)
    for ax in range(3):
        if not pbc[ax] or np.linalg.norm(cell[ax]) < 1e-6:
            vec = np.zeros(3, dtype=np.float64)
            vec[ax] = float(pad[ax])
            cell[ax] = vec
    return cell, pbc


def build_edge_tensors(
    pos_np: np.ndarray, cell_np: np.ndarray | None, pbc_flags: tuple[bool, bool, bool], r_cut: float, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    box = np.zeros((3, 3), dtype=np.float64) if cell_np is None else np.asarray(cell_np, dtype=np.float64)
    p = (bool(pbc_flags[0]), bool(pbc_flags[1]), bool(pbc_flags[2]))
    if any(p):
        i, j, s = cached_vesin_nl(r_cut).compute(
            points=np.asarray(pos_np, dtype=np.float64), box=box, periodic=p, quantities="ijS"
        )
        i_a, j_a, s_a = np.asarray(i, np.int64), np.asarray(j, np.int64), np.asarray(s, np.int64)
        o = np.argsort(i_a, kind="stable")
        return torch.from_numpy(np.stack([j_a[o], i_a[o]], axis=0)), torch.from_numpy(s_a[o]).to(dtype=dtype)
    i, j = cached_vesin_nl(r_cut).compute(
        points=np.asarray(pos_np, dtype=np.float64), box=np.zeros((3, 3)), periodic=p, quantities="ij"
    )
    i_a, j_a = np.asarray(i, np.int64), np.asarray(j, np.int64)
    o = np.argsort(i_a, kind="stable")
    return torch.from_numpy(np.stack([j_a[o], i_a[o]], axis=0)), torch.zeros((len(i_a), 3), dtype=dtype)


def build_open_mol_edges_gpu(
    pos_cat: Tensor,
    graph: Tensor,
    counts: list[int] | np.ndarray,
    r_cut: float,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Build receiver-sorted `(edge_index [2, E], edge_shift [E, 3])` on GPU for open molecules (`max_n <= 256`)."""
    dev = pos_cat.device
    G = len(counts)
    N = int(pos_cat.shape[0])
    max_n = int(max(counts)) if G > 0 else 0
    if N == 0 or max_n <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=dev), torch.zeros((0, 3), dtype=dtype, device=dev)
    with torch.no_grad():
        counts_t = torch.as_tensor(counts, dtype=torch.long, device=dev)
        starts_t = torch.cumsum(counts_t, dim=0) - counts_t
        loc = torch.arange(N, dtype=torch.long, device=dev) - starts_t.index_select(0, graph)
        p_pad = pos_cat.detach().new_zeros((G, max_n, 3))
        p_pad[graph, loc] = pos_cat.detach()
        ar = torch.arange(max_n, dtype=torch.long, device=dev)
        mask = ar.unsqueeze(0) < counts_t.unsqueeze(1)
        d2 = (p_pad.unsqueeze(2) - p_pad.unsqueeze(1)).pow(2).sum(dim=-1)
        eye = torch.eye(max_n, dtype=torch.bool, device=dev).unsqueeze(0)
        valid = mask.unsqueeze(2) & mask.unsqueeze(1) & (~eye) & (d2 < float(r_cut * r_cut))
        g_idx, i_loc, j_loc = torch.nonzero(valid, as_tuple=True)
        if g_idx.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=dev), torch.zeros((0, 3), dtype=dtype, device=dev)
        st_g = starts_t.index_select(0, g_idx)
        recv = st_g + i_loc
        send = st_g + j_loc
        if recv.numel() > 1 and bool((recv[1:] < recv[:-1]).any().item()):
            order = torch.argsort(recv, stable=True)
            recv = recv.index_select(0, order)
            send = send.index_select(0, order)
        return torch.stack([send, recv], dim=0), torch.zeros((recv.numel(), 3), dtype=dtype, device=dev)


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


def eval_dispersion_single(
    z_np: np.ndarray,
    pos_np: np.ndarray,
    cell_np: np.ndarray | None,
    pbc_flags: tuple[bool, bool, bool],
    d3_mode: str | None,
    device: torch.device,
    dtype: torch.dtype,
    *,
    compute_forces: bool = True,
    compute_stress: bool = False,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Evaluate dispersion `(E_kcal, F_kcal_A [N,3] | None, S_kcal_A3 [3,3] | None)`."""
    if d3_mode is None or len(z_np) == 0:
        return 0.0, None, None
    if device.type == "mps" and dtype == torch.float32 and not any(pbc_flags) and d3_mode == "bj_wb97m":
        from lam4.kernels.metal_d3 import has_metal_d3, metal_d3_csr, metal_d3_grouped

        if has_metal_d3():
            try:
                n = len(z_np)
                z_t = torch.as_tensor(z_np, dtype=torch.long, device=device)
                pos_t = torch.as_tensor(pos_np, dtype=torch.float32, device=device)
                if n <= 1024:
                    e_t, f_t = metal_d3_grouped(z_t, pos_t, torch.zeros(n, dtype=torch.long, device=device), 1, compute_forces=compute_forces)
                else:
                    ei, _ = build_edge_tensors(pos_np, np.eye(3, dtype=np.float64) * 100.0, (False, False, False), 12.0, torch.float32)
                    ei = ei.to(device)
                    recv_i32, send_i32 = ei[1].to(torch.int32).contiguous(), ei[0].to(torch.int32).contiguous()
                    rp = torch.zeros(n + 1, dtype=torch.int32, device=device)
                    rp[1:] = torch.searchsorted(recv_i32, torch.arange(n, dtype=torch.int32, device=device), right=True).to(torch.int32)
                    e_t, f_t = metal_d3_csr(z_t, pos_t, rp, send_i32, compute_forces=compute_forces)
                f_out = f_t.detach().cpu().to(torch.float64).numpy() if (compute_forces and f_t is not None) else None
                return float(e_t[0].detach().cpu().double().item()), f_out, None
            except Exception:
                pass
    d3 = _get_d3(device, dtype)
    z_t = torch.as_tensor(z_np, dtype=torch.long, device=device)
    if not any(pbc_flags) and d3_mode == "bj_wb97m" and len(z_np) > 64 and not compute_stress:
        from lam4.calc.d3_chunked import eval_d3_bj_open_chunked

        ei, _ = build_edge_tensors(pos_np, np.eye(3, dtype=np.float64) * 100.0, (False, False, False), d3.r_cut, dtype)
        pos_t = torch.as_tensor(pos_np, dtype=dtype, device=device)
        e_kcal, f_np = eval_d3_bj_open_chunked(d3, z_t, pos_t, ei.to(device), compute_forces=compute_forces)
        return e_kcal, f_np, None
    leaf = torch.as_tensor(pos_np, dtype=dtype, device=device)
    need_grad = compute_forces or compute_stress
    if need_grad:
        leaf = leaf.detach().requires_grad_(True)
    with torch.set_grad_enabled(need_grad):
        if not any(pbc_flags) and d3_mode == "bj_wb97m" and len(z_np) <= 64:
            mask = torch.ones((1, len(z_np)), dtype=torch.bool, device=device)
            e_t = d3(z_t.unsqueeze(0), leaf.unsqueeze(0), mask)
            eps = None
        else:
            c_np = np.eye(3, dtype=np.float64) * 100.0 if cell_np is None else cell_np
            ei, es = build_edge_tensors(pos_np, c_np, pbc_flags, d3.r_cut, dtype)
            ei, es = ei.to(device), es.to(device)
            send, recv = ei[0], ei[1]
            pos_cur = leaf
            eps = None
            cell_t = torch.as_tensor(c_np, dtype=dtype, device=device)
            if compute_stress:
                eps = leaf.new_zeros(3, 3).requires_grad_(True)
                sym = 0.5 * (eps + eps.T)
                pos_cur = leaf + leaf @ sym
                cell_t = cell_t + cell_t @ sym
            diff = pos_cur.index_select(0, send) - pos_cur.index_select(0, recv) + (es @ cell_t if any(pbc_flags) else 0.0)
            e_idx = torch.stack([recv, send], dim=0)
            if d3_mode == "zero_pbe":
                e_t = _d3_zero_pbe_sparse(d3, z_t, e_idx, diff)
            else:
                e_t = d3.forward_sparse(z_t, pos_cur, torch.zeros_like(z_t), 1, e_idx, edge_diff=-diff)
        f_np = s_np = None
        if need_grad:
            wrt = ([leaf] if compute_forces else []) + ([eps] if (compute_stress and eps is not None) else [])
            grads = torch.autograd.grad(e_t.sum(), wrt) if wrt else ()
            if compute_forces:
                f_np = (-grads[0]).detach().cpu().to(torch.float64).numpy()
            if compute_stress:
                vol = abs(float(np.linalg.det(cell_np))) if cell_np is not None else 0.0
                s_np = (grads[-1].detach().cpu().to(torch.float64).numpy() / vol) if vol > 1e-8 else np.zeros((3, 3), dtype=np.float64)
    return float(e_t[0].detach().cpu().double().item()), f_np, s_np
