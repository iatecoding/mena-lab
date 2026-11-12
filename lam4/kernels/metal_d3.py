"""Atomic-free Tier-2 Metal kernels for Grimme D3(BJ) dispersion (`bj_wb97m`).

Replaces the O(B * N^2 * 25 * 3) (`~700 bytes/pair`) PyTorch autograd `D3BJ` path on MPS:
  - Memory: O(N) (`cn [N]`, `dE_dcn [N]`, `e_atom [N]`, `forces [N, 3]`, `< 1 MB` at N = 32,768).
  - Lookup: 25-reference `c6ab` Gaussian interpolation stays 100% in GPU registers (read ONCE per pair).
  - Modes:
      1. `metal_d3_grouped`: zero neighbor-list overhead over `group_ptr [G + 1]` for single/batched molecules.
      2. `metal_d3_csr`:     over a 12.0 A receiver-sorted CSR neighbor list for large N > 1024 molecules.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lam.models.dispersion import ANGSTROM_TO_BOHR, D3BJ, WB97M_D3BJ
from lam4.kernels.metal_coulomb import build_group_ptr, has_metal_csr
from lam4.kernels.metal_d3_src import METAL_D3_MSL

_D3_LIB: Any = None
_D3_FAILED: bool = False
_D3_TABLES: dict[tuple[str, int | None], tuple[Tensor, Tensor, Tensor]] = {}


def has_metal_d3() -> bool:
    return bool(not _D3_FAILED and has_metal_csr())


def get_metal_d3_lib() -> Any:
    global _D3_LIB, _D3_FAILED
    if _D3_LIB is None:
        try:
            _D3_LIB = torch.mps.compile_shader(METAL_D3_MSL)
        except Exception:
            _D3_FAILED = True
            raise
    return _D3_LIB


def _get_d3_tables(dev: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    key = (dev.type, dev.index)
    tabs = _D3_TABLES.get(key)
    if tabs is None:
        ref = D3BJ()
        tabs = (
            ref.rcov.to(device=dev, dtype=torch.float32).contiguous(),
            ref.r2r4.to(device=dev, dtype=torch.float32).contiguous(),
            ref.c6ab_flat.to(device=dev, dtype=torch.float32).contiguous(),
        )
        _D3_TABLES[key] = tabs
    return tabs


_PARAMS0 = (float(WB97M_D3BJ["s6"]), float(WB97M_D3BJ["s8"]), float(WB97M_D3BJ["a1"]), float(WB97M_D3BJ["a2"]))
_PARAMS1 = (12.0, 9.0, float(11.2 * ANGSTROM_TO_BOHR), float(1.0 / (12.0 - 9.0)))


@torch.no_grad()
def metal_d3_grouped(
    z: Tensor,
    pos: Tensor,
    graph: Tensor,
    n_graphs: int,
    *,
    atom_mask: Tensor | None = None,
    group_ptr: Tensor | None = None,
    compute_forces: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Tier-2 Metal D3(BJ) over grouped open molecules (`e_graph [G]` kcal/mol, `forces [N, 3]` kcal/mol/A | None)."""
    dev = pos.device
    n_atoms = int(pos.shape[0])
    G = int(n_graphs)
    if n_atoms == 0 or G == 0:
        e0 = torch.zeros(G, dtype=torch.float32, device=dev)
        f0 = torch.zeros((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else None
        return e0, f0

    lib = get_metal_d3_lib()
    rcov, r2r4, c6ab_flat = _get_d3_tables(dev)
    pos_c = pos.to(torch.float32).contiguous()
    z_c = z.to(torch.int64).contiguous()
    g_c = graph.to(torch.int64).contiguous()
    ptr = build_group_ptr(g_c, G) if group_ptr is None else group_ptr
    mask_u8 = (
        torch.ones(n_atoms, dtype=torch.uint8, device=dev)
        if atom_mask is None
        else atom_mask.to(torch.uint8).contiguous()
    )

    cn = torch.empty(n_atoms, dtype=torch.float32, device=dev)
    e_atom = torch.empty(n_atoms, dtype=torch.float32, device=dev)
    dedcn = torch.empty(n_atoms, dtype=torch.float32, device=dev) if compute_forces else cn[:1]
    forces = torch.empty((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else pos_c[:1]

    tg = 256
    tot = ((n_atoms + tg - 1) // tg) * tg
    lib.d3_cn_grouped(pos_c, z_c, mask_u8, g_c, ptr, rcov, cn, n_atoms, _PARAMS1, threads=(tot, 1, 1), group_size=(tg, 1, 1))
    lib.d3_energy_fdir_grouped(
        pos_c, z_c, mask_u8, g_c, ptr, r2r4, c6ab_flat, cn, e_atom, dedcn, forces,
        n_atoms, int(compute_forces), _PARAMS0, _PARAMS1,
        threads=(tot, 1, 1), group_size=(tg, 1, 1),
    )
    if compute_forces:
        lib.d3_fcn_grouped(pos_c, z_c, mask_u8, g_c, ptr, rcov, dedcn, forces, n_atoms, _PARAMS1, threads=(tot, 1, 1), group_size=(tg, 1, 1))

    e_graph = e_atom.sum(0, keepdim=True) if G == 1 else torch.zeros(G, dtype=torch.float32, device=dev).index_add_(0, g_c, e_atom)
    return e_graph, (forces if compute_forces else None)


@torch.no_grad()
def metal_d3_csr(
    z: Tensor,
    pos: Tensor,
    recv_ptr: Tensor,
    send_idx: Tensor,
    *,
    graph: Tensor | None = None,
    n_graphs: int = 1,
    compute_forces: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Tier-2 Metal D3(BJ) over a receiver-sorted 12.0 A CSR neighbor list for large open molecules (`N > 1024`)."""
    dev = pos.device
    n_atoms = int(pos.shape[0])
    G = int(n_graphs)
    if n_atoms == 0:
        return torch.zeros(G, dtype=torch.float32, device=dev), (torch.zeros((0, 3), dtype=torch.float32, device=dev) if compute_forces else None)

    lib = get_metal_d3_lib()
    rcov, r2r4, c6ab_flat = _get_d3_tables(dev)
    pos_c = pos.to(torch.float32).contiguous()
    z_c = z.to(torch.int64).contiguous()
    rp_c = recv_ptr.to(torch.int32).contiguous()
    si_c = send_idx.to(torch.int32).contiguous()

    cn = torch.empty(n_atoms, dtype=torch.float32, device=dev)
    e_atom = torch.empty(n_atoms, dtype=torch.float32, device=dev)
    dedcn = torch.empty(n_atoms, dtype=torch.float32, device=dev) if compute_forces else cn[:1]
    forces = torch.empty((n_atoms, 3), dtype=torch.float32, device=dev) if compute_forces else pos_c[:1]

    tg = 256
    tot = ((n_atoms + tg - 1) // tg) * tg
    lib.d3_cn_csr(pos_c, z_c, rp_c, si_c, rcov, cn, n_atoms, _PARAMS1, threads=(tot, 1, 1), group_size=(tg, 1, 1))
    lib.d3_energy_fdir_csr(
        pos_c, z_c, rp_c, si_c, r2r4, c6ab_flat, cn, e_atom, dedcn, forces,
        n_atoms, int(compute_forces), _PARAMS0, _PARAMS1,
        threads=(tot, 1, 1), group_size=(tg, 1, 1),
    )
    if compute_forces:
        lib.d3_fcn_csr(pos_c, z_c, rp_c, si_c, rcov, dedcn, forces, n_atoms, _PARAMS1, threads=(tot, 1, 1), group_size=(tg, 1, 1))

    if G == 1 or graph is None:
        e_graph = e_atom.sum(0, keepdim=True)
    else:
        e_graph = torch.zeros(G, dtype=torch.float32, device=dev).index_add_(0, graph.to(torch.int64), e_atom)
    return e_graph, (forces if compute_forces else None)
