"""repulsion.py: NLH short-range repulsion (`c2_shift_rmin`) and the 5 eV tanh descriptor floor on a GraphBatch.

Port of lam/models/nlh_core.py (`nlh_energy_sparse(mode="c2_shift_rmin")`, `nlh_clamp_sparse(mode="tanh")`) as plain autograd: no custom Function, so
double backward (force training) works, no host sync, no data-dependent shape, no clamp that zeroes a gradient. Tables: lam4/physics/nlh_tables.py.
Units: energies kcal/mol (the earlier Mena trunk's unit, k_C = 332.06371), distances A.

Per directed edge (receiver i, sender j), pair index (Z_i << 7) | Z_j, r = sqrt(|d|^2 + 1e-8) (the earlier Mena trunk's `dist_raw`; the 1e-4 floor on r replaces the old
`clamp_min(1e-4)` + "zero gradient below 1e-4" pair: r >= 1e-4 always, smooth, so the gradient exists everywhere):
    d_off = r_min + 0.05 A,  r_min: SO3LR switched NLH(r_min) = 5 eV               (heavy pairs ~1.25 A, H-H 0.40 A; bonds sit above d_off)
    energy  e_edge = 1/2 [ E_NLH(r) - (V0 + dr V1 + dr^2 V2 / 2) ]  for r < d_off, dr = r - d_off, else 0   (C2 at d_off: value, slope, curvature 0)
    floor   r~ = d_off - w tanh((d_off - r) / w)  for r < d_off, w = 0.05 A, else r   (C2 handover at d_off, slope 1 above; same d_off as the energy)
The 1/2 is per directed edge, so each unordered pair counts once. Energy and floor use the SAME d_off: that is why bonds feel exactly 0 meV/A (a mismatched
onset leaves a dead shelf, report section 4). Pad contract: pad atoms have Z = 0, whose table row is all zero (prefactor 0, d_off 0), so `active` is False
and a pad edge (r = r_cut, receiver-sorted at the end) adds exactly 0.0 to the energy and exactly 0.0 to every gradient, whatever its r.
Replicated from the old code: Z clamped to 0..127; the energy sums on the RECEIVER; the floor needs r_min > 0 (Z = 0 pairs never floor).

API (all pure functions of tensors; `Repulsion` is a parameter-free module around them)
    edge_distance(d, eps=1e-4)                         [E, 3] -> [E]
    descriptor_floor(r, z_recv, z_send)                [E] -> [E]     the 5 eV floor on a distance
    floor_vectors(d, z_recv, z_send)                   [E, 3] -> [E, 3]  d scaled to the floored length (same direction)
    floored_length(d, z_recv, z_send, r=None)          [E, 3] -> [E]     |d~| with the exact |d| where inactive: what lam4.core.edge_terms.edge_geometry reads
    nlh_edge_energy(r, z_recv, z_send)                 [E] -> [E]
    nlh_atom_energy(b, pos=None) -> [N]  (on receivers);  nlh_graph_energy(b, pos=None) -> [G]   (b: GraphBatch; pos overrides b.pos for autograd leaves)
    Repulsion(mode).forward(b, pos=None) -> [G]        mode "nlh_c2_shift_rmin" | "none" (ModelConfig.repulsion)
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.nn as nn
from torch import Tensor

from lam4.core.consts import storable, tracing
from lam4.data.batch import GraphBatch
from lam4.physics.nlh_tables import FLOOR_WIDTH, nlh_tables_f64

R_EPS = 1e-4  # sqrt(|d|^2 + R_EPS^2): the earlier Mena trunk's 1e-8 under the root
_CACHE: dict[tuple[str, int | None, torch.dtype], tuple[Tensor, Tensor, Tensor]] = {}
_CPU_TABS: dict[torch.dtype, tuple[Tensor, Tensor, Tensor]] = {}
_MAX_DOFF_R2: float = 0.0
_TRACED_RMIN: ContextVar[Tensor | None] = ContextVar("_TRACED_RMIN", default=None)


@contextmanager
def with_rmin_table(rmin: Tensor | None):
    tok = _TRACED_RMIN.set(rmin)
    try:
        yield
    finally:
        _TRACED_RMIN.reset(tok)


def warm_tables() -> None:
    global _MAX_DOFF_R2
    if torch.float64 not in _CPU_TABS:
        raw = nlh_tables_f64()
        _MAX_DOFF_R2 = float((raw[2][:, 0].max() + 1e-3) ** 2)
        f64 = tuple(torch.from_numpy(a) for a in raw)
        _CPU_TABS[torch.float64] = f64
        _CPU_TABS[torch.float32] = tuple(b.to(dtype=torch.float32).contiguous() for b in f64)


def _cpu_tabs(dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    t = _CPU_TABS.get(dtype)
    if t is None:
        warm_tables()
        t = _CPU_TABS.get(dtype)
        if t is None:
            t = tuple(b.to(dtype=dtype).contiguous() for b in _CPU_TABS[torch.float64])
            if all(storable(x) for x in t):
                _CPU_TABS[dtype] = t
    return t


def _tables(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    idx = device.index
    if idx is None and device.type in ("cuda", "mps"):
        idx = int(torch.cuda.current_device()) if (device.type == "cuda" and torch.cuda.is_available()) else 0
    key = (device.type, idx, dtype)
    t = _CACHE.get(key)
    if t is None:
        cpu_t = _cpu_tabs(dtype)
        t = cpu_t if device.type == "cpu" else tuple(b.to(device=device) for b in cpu_t)
        if all(storable(x) for x in t):   # never cache a traced / fake cast (lam4.core.consts)
            _CACHE[key] = t
    return t


def warm_device_tables(device: torch.device | str, dtype: torch.dtype = torch.float32) -> None:
    dev = torch.device(device)
    if dev.type == "meta" or tracing():
        return
    _tables(dev, torch.float64 if dtype == torch.float64 else torch.float32)
    if dtype != torch.float32:
        _tables(dev, torch.float32)


def _pair_index(z_recv: Tensor, z_send: Tensor) -> Tensor:
    return (z_recv.clamp(0, 127) << 7) + z_send.clamp(0, 127)


def edge_distance(d: Tensor, eps: float = R_EPS) -> Tensor:
    return (d.pow(2).sum(-1) + eps * eps).sqrt()


def _floor(r: Tensor, z_recv: Tensor, z_send: Tensor, width: float) -> tuple[Tensor, Tensor]:
    tr_rmin = _TRACED_RMIN.get()
    rmin = tr_rmin if (tr_rmin is not None and tr_rmin.numel() > 0) else _tables(r.device, r.dtype)[1]
    r_min = rmin.index_select(0, _pair_index(z_recv, z_send))
    d_off = r_min + width
    active = (r < d_off) & (r_min > 0.0)
    th = torch.tanh(torch.where(active, (d_off - r) / width, torch.zeros_like(r)))
    return torch.where(active, d_off - width * th, r), active


def descriptor_floor(r: Tensor, z_recv: Tensor, z_send: Tensor, width: float = FLOOR_WIDTH) -> Tensor:
    """5 eV tanh floor: r for r >= d_off, else d_off - w tanh((d_off - r) / w). The masked branch sees u = 0, so no NaN and no gradient leak."""
    return _floor(r, z_recv, z_send, width)[0]


def floor_vectors(d: Tensor, z_recv: Tensor, z_send: Tensor) -> Tensor:
    """d scaled so that |d~| = descriptor_floor(|d|) (same direction). Inactive edges are returned unchanged (scale exactly 1)."""
    r = edge_distance(d)
    rf, active = _floor(r, z_recv, z_send, FLOOR_WIDTH)
    return d * torch.where(active, rf / r, torch.ones_like(r)).unsqueeze(-1)


def floored_length(d: Tensor, z_recv: Tensor, z_send: Tensor, r: Tensor | None = None) -> Tensor:
    """|d~| [E]: the floored edge length that the edge geometry (chi, warp) reads, WITHOUT the 1e-4 regulariser where the floor is inactive.

    Active edges (r < d_off, real pair) get d_off - w tanh((d_off - r) / w) with r = sqrt(|d|^2 + 1e-8) (the NLH energy's distance, so floor and repulsion
    share one onset); every other edge keeps its exact |d| (`r`, computed here if not given), so geometries above the floor are bit-identical to the
    unfloored ones. Pad edges (Z = 0 pairs, r_min = 0) are inactive. Same value as |floor_vectors(d)| on active edges."""
    r_raw = d.pow(2).sum(-1).sqrt() if r is None else r
    rf, active = _floor(edge_distance(d), z_recv, z_send, FLOOR_WIDTH)
    return torch.where(active, rf, r_raw)


def nlh_edge_energy(r: Tensor, z_recv: Tensor, z_send: Tensor) -> Tensor:
    """Half the C2-shifted NLH pair energy of every directed edge [E] (kcal/mol); exactly 0 for r >= d_off and for pad pairs (Z = 0)."""
    cb, _, jet = _tables(r.device, r.dtype)
    p = _pair_index(z_recv, z_send)
    c0, c1, c2, b0, b1, b2 = cb.index_select(0, p).unbind(-1)
    d_off, v0, v1, v2 = jet.index_select(0, p).unbind(-1)
    e_nlh = (c0 * torch.exp(-b0 * r) + c1 * torch.exp(-b1 * r) + c2 * torch.exp(-b2 * r)) / r
    dr = r - d_off
    return 0.5 * torch.where((r < d_off) & (d_off > 0.0), e_nlh - (v0 + dr * (v1 + 0.5 * dr * v2)), torch.zeros_like(e_nlh))


def _edge_energy(b: GraphBatch, pos: Tensor | None) -> Tensor:
    j, i = b.edge_index
    p = b.pos if pos is None else pos
    r = edge_distance(p.index_select(0, j) - p.index_select(0, i) + b.cart_shift())
    return nlh_edge_energy(r, b.z.index_select(0, i), b.z.index_select(0, j))


def _cand_shift(b: GraphBatch, cand: Tensor) -> Tensor:
    es = b.edge_shift.index_select(0, cand).to(b.cell.dtype)
    if b.n_graphs == 1:
        return es @ b.cell[0]
    g = b.graph.index_select(0, b.edge_index[1].index_select(0, cand))
    return torch.bmm(es.unsqueeze(1), b.cell.index_select(0, g)).squeeze(1)


def _raw_proxy_tracing() -> bool:
    """True inside standalone `make_fx` outside `torch.compile` (where custom ops should decompose to PyTorch)."""
    if torch.compiler.is_compiling():
        return False
    try:
        from torch.fx.experimental.proxy_tensor import get_proxy_mode
    except ImportError:
        return False
    return get_proxy_mode() is not None


def nlh_atom_energy(
    b: GraphBatch, pos: Tensor | None = None, chunk_edges: int = 65_536, *, csr_plan=None, create_graph: bool = False
) -> Tensor:
    """[N] kcal/mol: the edge energies summed on the receiver atom. Pad atoms get exactly 0."""
    j, i = b.edge_index
    p = b.pos if pos is None else pos
    if csr_plan is not None and p.dtype == torch.float32 and not create_graph:
        if p.device.type == "mps" and not _raw_proxy_tracing():
            from lam4.kernels.metal_nlh import has_metal_nlh, metal_nlh_atom_energy

            if has_metal_nlh():
                return metal_nlh_atom_energy(csr_plan, p, b.edge_shift, b.cell, b.graph, b.z)
        if p.device.type == "cuda" and not _raw_proxy_tracing():
            from lam4.kernels.triton_nlh import has_triton_nlh, triton_nlh_atom_energy

            if has_triton_nlh(p.device):
                return triton_nlh_atom_energy(csr_plan, p, b.edge_shift, b.cell, b.graph, b.z)
    n_e = j.shape[0]
    if n_e == 0:
        out = p.new_zeros(b.n_atoms)
        if torch.is_grad_enabled():
            if p.requires_grad and p.numel() > 0:
                out = out + p[:1, :1].sum() * 0.0
            if b.cell.requires_grad and b.cell.numel() > 0:
                out = out + b.cell[:1, :1, :1].sum() * 0.0
        return out
    if not tracing() and (p.device.type == "cpu" or n_e >= 8192):
        warm_tables()
        r2_cut = _MAX_DOFF_R2
        cands = []
        with torch.no_grad():
            pd, cd = p.detach(), b.cell.detach()
            for s in range(0, n_e, chunk_edges):
                e1 = min(s + chunk_edges, n_e)
                jj, ii = j[s:e1], i[s:e1]
                es = b.edge_shift[s:e1].to(cd.dtype)
                sh = (es @ cd[0]) if b.n_graphs == 1 else torch.bmm(es.unsqueeze(1), cd.index_select(0, b.graph.index_select(0, ii))).squeeze(1)
                d2 = (pd.index_select(0, jj) - pd.index_select(0, ii) + sh).pow(2).sum(-1)
                loc = torch.nonzero(d2 < r2_cut, as_tuple=True)[0]
                if loc.numel() > 0:
                    cands.append(loc + s)
        out = p.new_zeros(b.n_atoms)
        if not cands:
            if torch.is_grad_enabled():
                if p.requires_grad and p.numel() > 0:
                    out = out + p[:1, :1].sum() * 0.0
                if b.cell.requires_grad and b.cell.numel() > 0:
                    out = out + b.cell[:1, :1, :1].sum() * 0.0
            return out
        cand = cands[0] if len(cands) == 1 else torch.cat(cands, dim=0)
        jj, ii = j.index_select(0, cand), i.index_select(0, cand)
        r = edge_distance(p.index_select(0, jj) - p.index_select(0, ii) + _cand_shift(b, cand))
        return out.index_add(0, ii, nlh_edge_energy(r, b.z.index_select(0, ii), b.z.index_select(0, jj)))
    e = _edge_energy(b, pos)
    return e.new_zeros(b.n_atoms).index_add(0, i, e)


def nlh_graph_energy(b: GraphBatch, pos: Tensor | None = None) -> Tensor:
    """[G] kcal/mol. Extensive in the supercell: every term is a function of one edge, edges are counted once per receiver."""
    if b.edge_index.shape[1] == 0:
        p = b.pos if pos is None else pos
        out = p.new_zeros(b.n_graphs)
        if torch.is_grad_enabled() and p.requires_grad and p.numel() > 0:
            out = out + p[:1, :1].sum() * 0.0
        return out
    e = _edge_energy(b, pos)
    return e.new_zeros(b.n_graphs).index_add(0, b.graph.index_select(0, b.edge_index[1]), e)


class Repulsion(nn.Module):
    """No parameters, no state_dict keys. forward(b, pos=None) -> [G] kcal/mol (zeros for mode "none")."""

    def __init__(self, mode: str = "nlh_c2_shift_rmin") -> None:
        super().__init__()
        if mode not in ("nlh_c2_shift_rmin", "none"):
            raise ValueError("mode in {nlh_c2_shift_rmin, none}")
        self.mode = mode

    def forward(self, b: GraphBatch, pos: Tensor | None = None) -> Tensor:
        if self.mode == "none":
            return b.pos.new_zeros(b.n_graphs)
        return nlh_graph_energy(b, pos)

    def floor(self, d: Tensor, z_recv: Tensor, z_send: Tensor) -> Tensor:
        """Edge vectors with the 5 eV descriptor floor applied (identity for mode "none")."""
        return d if self.mode == "none" else floor_vectors(d, z_recv, z_send)


warm_tables()
