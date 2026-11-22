"""GPU-resident batched FP64 Crystal FIRE + FrechetCellFilter window engine."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

from ase import Atoms
import numpy as np
import torch
from torch import Tensor

from lam4.calc.physics import CRYSTAL_EV
from lam4.calc.relax_crystal_fire_prep import (
    CrystalPrefetcher,
    PreparedCrystal,
    RawCrystal,
    batched_frechet_expm_3x3,
    batched_pade_expm_3x3,
)
from lam4.data.batch import GraphBatch
from lam4.kernels.csr_plan import build_csr_plan

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator


@dataclass(slots=True)
class _SlotMeta:
    key: Any
    n: int
    z_np: np.ndarray
    t0: float
    n_edges: int
    ei_gpu: Tensor
    es_gpu: Tensor
    u_atom_gpu: Tensor
    u_val: int
    tok_id: int
    ref_id: int
    xc_id: int
    paw_id: int
    e0_kcal: float


class _GpuFireWindow:
    """Device-resident FP64 batched FIRE + FrechetCellFilter + Verlet skin state."""

    _GRAPH_FIELDS = (
        "orig_cell", "inv_orig_cell", "cell", "cell_ref", "U", "F", "v_cell", "g_cell",
        "ecf", "dt", "a", "e0_kcal", "n_pos_steps", "steps", "counts", "vel_init",
    )
    _ATOM_F64_FIELDS = ("x_at", "pos", "v_at", "g_at", "frac_ref")

    def __init__(self, calc: "MenaCalculator", state_dev: torch.device, fmax: float, max_steps: int, scalar_pressure: float) -> None:
        self.calc = calc
        self.model_dev = calc.device
        self.dev = state_dev
        self.dt_model = calc.dtype
        self.np_dt = np.float32 if self.dt_model == torch.float32 else np.float64
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        self.scalar_pressure = float(scalar_pressure)
        self.is_csr = calc.model.trunk.edge_mode == "csr"
        self.eye3 = torch.eye(3, dtype=torch.float64, device=self.dev)
        self.p_diag = torch.diag(torch.full((3,), self.scalar_pressure, dtype=torch.float64, device=self.dev))
        self.slots: list[_SlotMeta] = []
        self.cur_atoms = 0
        self._init_empty()

    def _init_empty(self) -> None:
        d, md = self.dev, self.model_dev
        f64, i64, b_t = torch.float64, torch.int64, torch.bool
        for name in ("orig_cell", "inv_orig_cell", "cell", "cell_ref", "U", "F", "v_cell", "g_cell"):
            setattr(self, name, torch.empty((0, 3, 3), dtype=f64, device=d))
        for name in ("ecf", "dt", "a", "e0_kcal"):
            setattr(self, name, torch.empty((0,), dtype=f64, device=d))
        for name in ("n_pos_steps", "steps", "counts"):
            setattr(self, name, torch.empty((0,), dtype=i64, device=d))
        self.vel_init = torch.empty((0,), dtype=b_t, device=d)
        self.utr = torch.empty((5, 0), dtype=i64, device=md)
        for name in self._ATOM_F64_FIELDS:
            setattr(self, name, torch.empty((0, 3), dtype=f64, device=d))
        self.z = torch.empty((0,), dtype=i64, device=md)
        self.u_atom = torch.empty((0,), dtype=i64, device=md)
        self.graph = torch.empty((0,), dtype=i64, device=md)

    def _split_edges_to_device(
        self, edges: list[tuple[np.ndarray, np.ndarray]], lens: list[int],
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        md = self.model_dev
        if sum(lens) > 0:
            ei_cat = torch.from_numpy(np.concatenate([e[0] for e in edges if e[0].shape[1] > 0], axis=1)).to(md)
            es_cat = torch.from_numpy(np.concatenate([e[1] for e in edges if e[0].shape[1] > 0], axis=0)).to(md)
            return torch.split(ei_cat, lens, dim=1), torch.split(es_cat, lens, dim=0)
        empty_ei = torch.zeros((2, 0), dtype=torch.int64, device=md)
        empty_es = torch.zeros((0, 3), dtype=self.dt_model, device=md)
        return (empty_ei,) * len(lens), (empty_es,) * len(lens)

    def admit(self, preps: list[PreparedCrystal]) -> None:
        if not preps:
            return
        d, md = self.dev, self.model_dev
        K = len(preps)
        g_old = len(self.slots)
        counts_np = np.array([len(p.z) for p in preps], dtype=np.int64)
        tot_new = int(counts_np.sum())
        now = time.time()

        ei_lens = [int(p.ei.shape[1]) for p in preps]
        ei_splits, es_splits = self._split_edges_to_device([(p.ei, p.es) for p in preps], ei_lens)
        uat_dev = torch.from_numpy(np.concatenate([p.u_atom for p in preps], axis=0)).to(md)
        uat_splits = torch.split(uat_dev, counts_np.tolist(), dim=0)

        for idx, p in enumerate(preps):
            self.slots.append(_SlotMeta(
                key=p.key, n=len(p.z), z_np=p.z, t0=now, n_edges=ei_lens[idx],
                ei_gpu=ei_splits[idx], es_gpu=es_splits[idx], u_atom_gpu=uat_splits[idx],
                u_val=p.u_val, tok_id=p.tok_id, ref_id=p.ref_id, xc_id=p.xc_id, paw_id=p.paw_id,
                e0_kcal=p.e0_kcal,
            ))

        orig_t = torch.from_numpy(np.stack([p.cell0 for p in preps], axis=0)).to(d)
        inv_t = torch.from_numpy(np.stack([p.inv_orig_cell for p in preps], axis=0)).to(d)
        counts_t = torch.from_numpy(counts_np).to(d)
        eye_k = self.eye3.unsqueeze(0).expand(K, 3, 3)
        zeros_k33 = torch.zeros((K, 3, 3), dtype=torch.float64, device=d)
        zeros_a3 = torch.zeros((tot_new, 3), dtype=torch.float64, device=d)
        pos_t = torch.from_numpy(np.concatenate([p.pos0 for p in preps], axis=0)).to(d)
        utr_np = np.array(
            [[p.u_val for p in preps], [p.tok_id for p in preps], [p.ref_id for p in preps],
             [p.xc_id for p in preps], [p.paw_id for p in preps]],
            dtype=np.int64,
        )
        graph_np = np.repeat(np.arange(g_old, g_old + K, dtype=np.int64), counts_np)

        for name, val in (
            ("orig_cell", orig_t), ("inv_orig_cell", inv_t), ("cell", orig_t), ("cell_ref", orig_t),
            ("U", zeros_k33), ("F", eye_k), ("v_cell", zeros_k33), ("g_cell", zeros_k33),
            ("ecf", counts_t.to(torch.float64)),
            ("dt", torch.full((K,), 0.1, dtype=torch.float64, device=d)),
            ("a", torch.full((K,), 0.1, dtype=torch.float64, device=d)),
            ("e0_kcal", torch.from_numpy(np.array([p.e0_kcal for p in preps], dtype=np.float64)).to(d)),
            ("n_pos_steps", torch.zeros(K, dtype=torch.int64, device=d)),
            ("steps", torch.zeros(K, dtype=torch.int64, device=d)),
            ("counts", counts_t),
            ("vel_init", torch.zeros(K, dtype=torch.bool, device=d)),
            ("x_at", pos_t), ("pos", pos_t), ("v_at", zeros_a3), ("g_at", zeros_a3),
            ("frac_ref", torch.from_numpy(np.concatenate([p.frac_ref for p in preps], axis=0)).to(d)),
        ):
            setattr(self, name, torch.cat((getattr(self, name), val), dim=0))

        self.utr = torch.cat((self.utr, torch.from_numpy(utr_np).to(md)), dim=1)
        self.z = torch.cat((self.z, torch.from_numpy(np.concatenate([p.z for p in preps], axis=0)).to(md)), dim=0)
        self.u_atom = torch.cat((self.u_atom, uat_dev), dim=0)
        self.graph = torch.cat((self.graph, torch.from_numpy(graph_np).to(md)), dim=0)
        self.cur_atoms += tot_new

    def step_fire(
        self, *, maxstep: float = 0.2, dtmax: float = 1.0, nmin: int = 5,
        finc: float = 1.1, fdec: float = 0.5, astart: float = 0.1, fa: float = 0.99,
    ) -> None:
        G = len(self.slots)
        if G == 0:
            return
        g_idx = self.graph if self.dev == self.model_dev else self.graph.to(self.dev)
        ga, gc, va, vc = self.g_at, self.g_cell, self.v_at, self.v_cell

        vf = torch.zeros(G, dtype=torch.float64, device=self.dev).index_add_(0, g_idx, (ga * va).sum(dim=-1)) + (gc * vc).sum(dim=(-2, -1))
        pos_vf = self.vel_init & (vf > 0.0)
        neg_vf = self.vel_init & (~(vf > 0.0))

        if bool(pos_vf.any()):
            g2 = torch.zeros(G, dtype=torch.float64, device=self.dev).index_add_(0, g_idx, (ga * ga).sum(dim=-1)) + (gc * gc).sum(dim=(-2, -1))
            v2 = torch.zeros(G, dtype=torch.float64, device=self.dev).index_add_(0, g_idx, (va * va).sum(dim=-1)) + (vc * vc).sum(dim=(-2, -1))
            has_g2 = pos_vf & (g2 > 0.0)
            scale = torch.where(has_g2, self.a * torch.sqrt(v2 / g2.clamp_min(1e-300)), torch.zeros_like(self.a))
            om = torch.where(has_g2, 1.0 - self.a, torch.ones_like(self.a))
            va = om[g_idx].unsqueeze(-1) * va + scale[g_idx].unsqueeze(-1) * ga
            vc = om.view(G, 1, 1) * vc + scale.view(G, 1, 1) * gc
            inc = pos_vf & (self.n_pos_steps > nmin)
            self.dt = torch.where(inc, (self.dt * finc).clamp_max(dtmax), self.dt)
            self.a = torch.where(inc, self.a * fa, self.a)
            self.n_pos_steps = torch.where(pos_vf, self.n_pos_steps + 1, self.n_pos_steps)

        if bool(neg_vf.any()):
            va = torch.where(neg_vf[g_idx].unsqueeze(-1), torch.zeros_like(va), va)
            vc = torch.where(neg_vf.view(G, 1, 1), torch.zeros_like(vc), vc)
            self.a = torch.where(neg_vf, torch.full_like(self.a, astart), self.a)
            self.dt = torch.where(neg_vf, self.dt * fdec, self.dt)
            self.n_pos_steps = torch.where(neg_vf, torch.zeros_like(self.n_pos_steps), self.n_pos_steps)

        va = va + self.dt[g_idx].unsqueeze(-1) * ga
        vc = vc + self.dt.view(G, 1, 1) * gc
        dr_at = self.dt[g_idx].unsqueeze(-1) * va
        dr_cell = self.dt.view(G, 1, 1) * vc
        normdr2 = torch.zeros(G, dtype=torch.float64, device=self.dev).index_add_(0, g_idx, (dr_at * dr_at).sum(dim=-1)) + (dr_cell * dr_cell).sum(dim=(-2, -1))
        normdr = torch.sqrt(normdr2)
        fac = torch.where(normdr > maxstep, maxstep / normdr.clamp_min(1e-300), torch.ones_like(normdr))

        self.v_at, self.v_cell = va, vc
        self.vel_init.fill_(True)
        self.x_at = self.x_at + dr_at * fac[g_idx].unsqueeze(-1)
        self.U = self.U + (dr_cell * fac.view(G, 1, 1)) / self.ecf.view(G, 1, 1)
        bad_u = ~torch.isfinite(self.U).all(dim=(-2, -1))
        self.F = batched_pade_expm_3x3(torch.where(bad_u.view(G, 1, 1), torch.zeros_like(self.U), self.U))
        ft = self.F.transpose(1, 2)
        self.cell = torch.bmm(self.orig_cell, ft)
        self.pos = torch.bmm(self.x_at.unsqueeze(1), ft[g_idx]).squeeze(1)
        self.steps += 1

    def rebuild_dirty(
        self, r_cut: float, eff_skin: float, r_list: float,
        pool: ThreadPoolExecutor | None,
        build_edges_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        G = len(self.slots)
        if G == 0:
            return
        g_idx = self.graph if self.dev == self.model_dev else self.graph.to(self.dev)
        if eff_skin <= 0.0:
            dirty_idx = list(range(G))
        else:
            det = torch.linalg.det(self.cell)
            bad_c = (~torch.isfinite(det)) | (det.abs() <= 1e-12)
            cell_safe = torch.where(bad_c.view(G, 1, 1), self.eye3.unsqueeze(0).expand(G, 3, 3), self.cell)
            B = torch.bmm(torch.linalg.inv(cell_safe), self.cell_ref) - self.eye3.unsqueeze(0)
            strain_bound = r_cut * torch.sqrt((B * B).sum(dim=(-2, -1)))
            frac_cur = torch.bmm(self.x_at.unsqueeze(1), self.inv_orig_cell[g_idx]).squeeze(1)
            ds = torch.bmm((frac_cur - self.frac_ref).unsqueeze(1), self.cell_ref[g_idx]).squeeze(1)
            disp2_max = torch.zeros(G, dtype=torch.float64, device=self.dev).scatter_reduce_(
                0, g_idx, (ds * ds).sum(dim=-1), reduce="amax", include_self=True,
            )
            disp_bound = 2.0 * torch.sqrt(disp2_max)
            dirty_mask = bad_c | ((strain_bound + disp_bound) >= (eff_skin - 1e-6)) | (~torch.isfinite(strain_bound)) | (~torch.isfinite(disp_bound))
            if not bool(dirty_mask.any()):
                return
            dirty_idx = torch.nonzero(dirty_mask, as_tuple=False).reshape(-1).cpu().tolist()

        d_t = torch.tensor(dirty_idx, dtype=torch.long, device=self.dev)
        cell_dirty_np = self.cell.index_select(0, d_t).cpu().numpy()
        dirty_mask_g = torch.zeros(G, dtype=torch.bool, device=self.dev)
        dirty_mask_g[d_t] = True
        atom_mask_dirty = dirty_mask_g[g_idx]
        pos_dirty_np = self.pos[atom_mask_dirty].cpu().numpy()

        dirty_inputs = []
        off = 0
        for idx, g_i in enumerate(dirty_idx):
            s = self.slots[g_i]
            dirty_inputs.append((s.z_np, pos_dirty_np[off : off + s.n], cell_dirty_np[idx], r_list, s.u_val, self.np_dt))
            off += s.n

        if pool is not None and len(dirty_inputs) >= 2:
            rebuilt = [f.result() for f in [pool.submit(build_edges_fn, *args) for args in dirty_inputs]]
        else:
            rebuilt = [build_edges_fn(*args) for args in dirty_inputs]

        ei_lens = [int(r[0].shape[1]) for r in rebuilt]
        ei_splits, es_splits = self._split_edges_to_device([(r[0], r[1]) for r in rebuilt], ei_lens)
        uat_dirty_dev = torch.from_numpy(np.concatenate([r[2] for r in rebuilt], axis=0)).to(self.model_dev)
        uat_splits = torch.split(uat_dirty_dev, [self.slots[g_i].n for g_i in dirty_idx], dim=0)

        for idx, g_i in enumerate(dirty_idx):
            s = self.slots[g_i]
            s.n_edges, s.ei_gpu, s.es_gpu, s.u_atom_gpu = ei_lens[idx], ei_splits[idx], es_splits[idx], uat_splits[idx]

        self.u_atom[atom_mask_dirty.to(self.model_dev) if self.dev != self.model_dev else atom_mask_dirty] = uat_dirty_dev
        self.cell_ref.index_copy_(0, d_t, self.cell.index_select(0, d_t))
        d_atom_idx = torch.nonzero(atom_mask_dirty, as_tuple=False).reshape(-1)
        g_dirty_atom = g_idx.index_select(0, d_atom_idx)
        frac_dirty = torch.bmm(self.x_at.index_select(0, d_atom_idx).unsqueeze(1), self.inv_orig_cell.index_select(0, g_dirty_atom)).squeeze(1)
        self.frac_ref.index_copy_(0, d_atom_idx, frac_dirty)

    def build_model_batch(self) -> tuple[GraphBatch, Any]:
        G, tot_n, md = len(self.slots), self.cur_atoms, self.model_dev
        g_idx_s = self.graph if self.dev == md else self.graph.to(self.dev)
        sums = torch.zeros((G, 3), dtype=torch.float64, device=self.dev).index_add_(0, g_idx_s, self.pos)
        pos_c = self.pos - (sums / self.counts.to(torch.float64).unsqueeze(-1))[g_idx_s]
        pos_model = pos_c.to(device=md, dtype=self.dt_model)
        cell_model = self.cell.to(device=md, dtype=self.dt_model)

        edge_counts = torch.tensor([s.n_edges for s in self.slots], dtype=torch.long, device=md)
        if int(edge_counts.sum().item()) > 0:
            atom_offsets = torch.cumsum(self.counts.to(md), dim=0) - self.counts.to(md)
            edge_atom_off = torch.repeat_interleave(atom_offsets, edge_counts).unsqueeze(0)
            ei_all = torch.cat([s.ei_gpu for s in self.slots if s.n_edges > 0], dim=1) + edge_atom_off
            es_all = torch.cat([s.es_gpu for s in self.slots if s.n_edges > 0], dim=0)
        else:
            ei_all = torch.zeros((2, 0), dtype=torch.int64, device=md)
            es_all = torch.zeros((0, 3), dtype=self.dt_model, device=md)

        qs_t = torch.zeros((2, G), dtype=self.dt_model, device=md)
        qs_t[1].fill_(1.0)
        b = GraphBatch(
            z=self.z, pos=pos_model, graph=self.graph, cell=cell_model,
            edge_index=ei_all, edge_shift=es_all, charge=qs_t[0], spin=qs_t[1],
            u_flag=self.utr[0], token=self.utr[1], ref_id=self.utr[2], domain="crystal",
            atom_mask=torch.ones(tot_n, dtype=torch.bool, device=md),
            graph_mask=torch.ones(G, dtype=torch.bool, device=md),
            energy=torch.zeros(G, dtype=self.dt_model, device=md),
            forces=torch.zeros((tot_n, 3), dtype=self.dt_model, device=md),
            energy_mask=torch.ones(G, dtype=torch.bool, device=md),
            force_mask=torch.ones(tot_n, dtype=torch.bool, device=md),
            coulomb_cap=None, pbc=True, xc_id=self.utr[3], paw_id=self.utr[4], u_atom=self.u_atom,
        )
        csr_plan = build_csr_plan(ei_all[1], ei_all[0], tot_n) if self.is_csr else None
        return b, csr_plan

    def _compact_keep(self, keep_mask_g: Tensor) -> None:
        if not bool(keep_mask_g.any()):
            self.slots = []
            self.cur_atoms = 0
            self._init_empty()
            return
        d, md = self.dev, self.model_dev
        g_idx = self.graph if d == md else self.graph.to(d)
        keep_idx_list = torch.nonzero(keep_mask_g, as_tuple=False).reshape(-1).cpu().tolist()
        self.slots = [self.slots[i] for i in keep_idx_list]
        keep_g_idx = torch.nonzero(keep_mask_g, as_tuple=False).reshape(-1)
        keep_a_idx = torch.nonzero(keep_mask_g[g_idx], as_tuple=False).reshape(-1)
        for name in self._GRAPH_FIELDS:
            setattr(self, name, getattr(self, name).index_select(0, keep_g_idx))
        for name in self._ATOM_F64_FIELDS:
            setattr(self, name, getattr(self, name).index_select(0, keep_a_idx))
        kg_m = keep_g_idx if d == md else keep_g_idx.to(md)
        ka_m = keep_a_idx if d == md else keep_a_idx.to(md)
        self.utr = self.utr.index_select(1, kg_m)
        self.z = self.z.index_select(0, ka_m)
        self.u_atom = self.u_atom.index_select(0, ka_m)
        self.graph = torch.repeat_interleave(torch.arange(len(self.slots), dtype=torch.int64, device=md), self.counts.to(md))
        self.cur_atoms = int(self.z.shape[0])

    def update_and_pop_finished(self, out: dict[str, Any]) -> list[dict[str, Any]]:
        G, d = len(self.slots), self.dev
        g_idx = self.graph if d == self.model_dev else self.graph.to(d)
        inv_ev = 1.0 / CRYSTAL_EV

        e_at_f64 = out["energy_atom"].detach().to(device=d, dtype=torch.float64)
        e_ev = (torch.zeros(G, dtype=torch.float64, device=d).index_add_(0, g_idx, e_at_f64) + self.e0_kcal) * inv_ev
        f_ev = out["forces"].detach().to(device=d, dtype=torch.float64) * inv_ev
        s33_ev = out["stress"].detach().to(device=d, dtype=torch.float64) * inv_ev

        vol = torch.linalg.det(self.cell).abs()
        bad_vol = (~torch.isfinite(vol)) | (vol <= 1e-8)
        virial = -vol.view(G, 1, 1) * (s33_ev + self.p_diag.unsqueeze(0))
        bad_F = ~torch.isfinite(self.F).all(dim=(-2, -1))
        F_safe = torch.where(bad_F.view(G, 1, 1), self.eye3.unsqueeze(0).expand(G, 3, 3), self.F)
        ucf_cell_grad = torch.bmm(virial, torch.linalg.inv(F_safe.transpose(1, 2)))
        bad_u = ~torch.isfinite(self.U).all(dim=(-2, -1))
        U_safe = torch.where(bad_u.view(G, 1, 1), torch.zeros_like(self.U), self.U)
        g_cell = batched_frechet_expm_3x3(U_safe.transpose(1, 2), ucf_cell_grad) / self.ecf.view(G, 1, 1)
        g_at = torch.bmm(f_ev.unsqueeze(1), self.F[g_idx]).squeeze(1)
        self.g_at, self.g_cell = g_at, g_cell

        gmax_at = torch.sqrt(torch.zeros(G, dtype=torch.float64, device=d).scatter_reduce_(0, g_idx, (g_at * g_at).sum(dim=-1), reduce="amax", include_self=True))
        gmax_cell = torch.sqrt((g_cell * g_cell).sum(dim=-1).amax(dim=-1))
        gmax = torch.maximum(gmax_at, gmax_cell)
        fmax_comp = torch.zeros(G, dtype=torch.float64, device=d).scatter_reduce_(0, g_idx, f_ev.abs().amax(dim=-1), reduce="amax", include_self=True)

        is_conv = gmax < self.fmax
        is_bad = bad_vol | bad_F | bad_u | (~torch.isfinite(e_ev)) | (~torch.isfinite(gmax))
        is_done = is_conv | is_bad | (self.steps >= self.max_steps)

        flags_cpu = torch.stack((is_done, is_conv, is_bad), dim=1).cpu().numpy()
        if not bool(flags_cpu[:, 0].any()):
            return []

        done_idx = np.flatnonzero(flags_cpu[:, 0])
        done_t = torch.tensor(done_idx, dtype=torch.long, device=d)
        e_done = e_ev.index_select(0, done_t).cpu().numpy()
        st_done = self.steps.index_select(0, done_t).cpu().numpy()
        fm_done = fmax_comp.index_select(0, done_t).cpu().numpy()
        gm_done = gmax.index_select(0, done_t).cpu().numpy()
        cell_done = self.cell.index_select(0, done_t).cpu().numpy()
        pos_done_all = self.pos[is_done[g_idx]].cpu().numpy()

        yielded: list[dict[str, Any]] = []
        off, t_now = 0, time.time()
        for k, g_i in enumerate(done_idx.tolist()):
            s = self.slots[g_i]
            pos_s = pos_done_all[off : off + s.n]
            off += s.n
            if bool(flags_cpu[g_i, 2]):
                yielded.append({"key": s.key, "n_atoms": s.n, "error": "RuntimeError('Degenerate cell or non-finite state')"})
            else:
                yielded.append({
                    "key": s.key, "e_uncorr_eV": float(e_done[k]), "n_atoms": s.n,
                    "steps": int(st_done[k]), "fmax_final": float(fm_done[k]),
                    "gmax_final": float(gm_done[k]), "converged": bool(flags_cpu[g_i, 1]),
                    "sec": round(t_now - s.t0, 2),
                    "cell": cell_done[k].tolist(), "pos": pos_s.tolist(), "z": s.z_np.tolist(), "error": "",
                })
        self._compact_keep(~is_done)
        return yielded

    def isolate_slot_failure(self, eval_one_fn: Callable[[Any], tuple[np.ndarray, list[np.ndarray], np.ndarray]]) -> list[dict[str, Any]]:
        """Evaluate slots individually on batch exception, drop failed/finished, keep healthy slots batched."""
        from lam4.calc.relax_crystal_fire import _CrystalSlot

        G = len(self.slots)
        orig_np, inv_np = self.orig_cell.cpu().numpy(), self.inv_orig_cell.cpu().numpy()
        cell_np, U_np, F_np = self.cell.cpu().numpy(), self.U.cpu().numpy(), self.F.cpu().numpy()
        x_np, pos_np = self.x_at.cpu().numpy(), self.pos.cpu().numpy()
        st_np = self.steps.cpu().numpy()

        yielded: list[dict[str, Any]] = []
        keep_mask = [False] * G
        ga_parts, gc_parts = [], []
        off = 0
        for g_i, m in enumerate(self.slots):
            n = m.n
            s = _CrystalSlot(
                key=m.key, z=m.z_np, n=n, orig_cell=orig_np[g_i], inv_orig_cell=inv_np[g_i],
                ecf=float(n), x_at=x_np[off : off + n], U=U_np[g_i], F=F_np[g_i],
                pos=pos_np[off : off + n], cell=cell_np[g_i],
                v_at=np.zeros((n, 3)), v_cell=np.zeros((3, 3)), vel_init=False, dt=0.1, a=0.1, n_pos_steps=0,
                steps=int(st_np[g_i]), max_steps=self.max_steps, fmax=self.fmax, scalar_pressure=self.scalar_pressure,
                src_name="mp_pbeU" if m.u_val else "mp_pbe", u_val=m.u_val, tok_id=m.tok_id, ref_id=m.ref_id,
                xc_id=m.xc_id, paw_id=m.paw_id, e0_kcal=m.e0_kcal, t0=m.t0,
                ei=m.ei_gpu.cpu().numpy(), es=m.es_gpu.cpu().numpy(), u_atom=m.u_atom_gpu.cpu().numpy(),
            )
            off += n
            try:
                ei_s, fl_s, st_s = eval_one_fn(s)
                is_conv, gmax, fmax_c = s.update_forces_and_check(fl_s[0], st_s[0])
                if not (np.isfinite(ei_s[0]) and np.isfinite(gmax)):
                    raise RuntimeError("Non-finite energy or force")
                if is_conv or s.steps >= s.max_steps:
                    yielded.append({
                        "key": s.key, "e_uncorr_eV": float(ei_s[0]), "n_atoms": s.n,
                        "steps": int(s.steps), "fmax_final": float(fmax_c),
                        "gmax_final": float(gmax), "converged": bool(is_conv),
                        "sec": round(time.time() - s.t0, 2),
                        "cell": s.cell.tolist(), "pos": s.pos.tolist(), "z": s.z.tolist(), "error": "",
                    })
                else:
                    keep_mask[g_i] = True
                    ga_parts.append(s.g_at)
                    gc_parts.append(s.g_cell)
            except Exception as exc_s:
                yielded.append({"key": s.key, "n_atoms": s.n, "error": repr(exc_s)[:200]})

        self._compact_keep(torch.tensor(keep_mask, dtype=torch.bool, device=self.dev))
        if self.slots:
            self.g_at = torch.from_numpy(np.concatenate(ga_parts, axis=0)).to(self.dev)
            self.g_cell = torch.from_numpy(np.stack(gc_parts, axis=0)).to(self.dev)
        return yielded


def relax_crystals_fire_gpu_stream(
    calc: "MenaCalculator",
    items: Iterable[tuple[Any, Atoms | RawCrystal]],
    *,
    fmax: float = 0.02,
    steps: int = 500,
    maxstep: float = 0.2,
    skin: float = 0.5,
    max_batch: int = 1024,
    max_atoms: int = 8192,
    scalar_pressure: float = 0.0,
    nl_threads: int = 8,
    u_flag_fn: Callable[[np.ndarray], int] | None = None,
    state_device: torch.device | None = None,
) -> Iterator[dict[str, Any]]:
    """GPU-resident continuous-window Crystal FIRE + FrechetCellFilter with overlapped host prep."""
    from lam4.calc.relax_crystal_fire import _build_slot_edges, _eval_crystal_slots_efs

    r_cut = float(calc.model.cfg.r_cut)
    eff_skin = max(0.0, float(skin))
    r_list = r_cut + eff_skin
    np_dt = np.float32 if calc.dtype == torch.float32 else np.float64
    s_dev = calc.device if state_device is None else state_device
    pool = ThreadPoolExecutor(max_workers=max(1, int(nl_threads))) if int(nl_threads) > 1 else None
    prefetch = CrystalPrefetcher(
        items, lot=calc.lot, u_override=calc.u_override, add_ref=calc.add_ref,
        r_list=r_list, np_dt=np_dt, u_flag_fn=u_flag_fn, build_edges_fn=_build_slot_edges, pool=pool,
    )
    win = _GpuFireWindow(calc, s_dev, fmax=fmax, max_steps=steps, scalar_pressure=scalar_pressure)
    pending: PreparedCrystal | None = None
    exhausted = False

    try:
        while True:
            if win.slots:
                win.step_fire(maxstep=maxstep)
                win.rebuild_dirty(r_cut, eff_skin, r_list, pool, _build_slot_edges)

            new_preps: list[PreparedCrystal] = []
            tot_n, tot_g = win.cur_atoms, len(win.slots)
            while tot_g < max_batch and not exhausted:
                if pending is not None:
                    p, pending = pending, None
                else:
                    p = prefetch.next_prepared()
                    if p is None:
                        exhausted = True
                        break
                if p.error:
                    yield {"key": p.key, "n_atoms": len(p.z), "error": p.error}
                    continue
                na = len(p.z)
                if (tot_g > 0 or new_preps) and (tot_n + na > max_atoms):
                    pending = p
                    break
                new_preps.append(p)
                tot_n += na
                tot_g += 1

            if new_preps:
                win.admit(new_preps)
            if not win.slots:
                break

            try:
                b, csr_plan = win.build_model_batch()
                out = calc._run_model(b, None, need_forces=True, need_stress=True, csr_plan=csr_plan)
                for row in win.update_and_pop_finished(out):
                    yield row
            except Exception:
                if calc.device.type == "cuda":
                    torch.cuda.empty_cache()
                for row in win.isolate_slot_failure(lambda s: _eval_crystal_slots_efs(calc, [s])):
                    yield row
    finally:
        prefetch.close()
        if pool is not None:
            pool.shutdown(wait=False)
