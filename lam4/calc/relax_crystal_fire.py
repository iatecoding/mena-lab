"""Tier-2 continuous-window batched Crystal FIRE + FrechetCellFilter relaxation engine for MenaCalculator.

Mathematically matches ASE `FIRE(FrechetCellFilter(atoms))` to machine precision (~1e-14) while eliminating:
  - Per-structure Python ASE/FCF overhead (vectorized FP64 state updates),
  - Chunk tail-straggler drain (single continuous global sliding window refilling finished slots immediately),
  - Per-step `vesin` neighbor rebuilds (exact cell-strain + fractional-displacement Verlet skin with threaded rebuilds),
  - Per-structure CUDA H2D/argsort storms and unused dict/head/decomposition packing.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Sequence

from ase import Atoms
import numpy as np
import torch

from lam4.calc.physics import CRYSTAL_EV, cached_vesin_nl, resolve_lot_and_u
from lam4.calc.precision_hygiene import center_positions_f64
from lam4.calc.relax_crystal_fire_prep import RawCrystal
from lam4.calc.verlet import ref_energies_kcal
from lam4.data.batch import GraphBatch
from lam4.data.buckets import source_ids
from lam4.kernels.csr_plan import build_csr_plan
from lam4.registry.hubbard import ANION_BOND_TOL_A, MP_HUBBARD_ANIONS, MP_HUBBARD_METALS, NEIGHBOR_CUTOFF_A, _COV_RADII
from lam4.registry.sources import lot_ids

if TYPE_CHECKING:
    from lam4.calc.calculator import MenaCalculator

_I3 = np.eye(3, dtype=np.float64)
_IS_TM_LUT = np.zeros(128, dtype=bool)
_IS_TM_LUT[sorted(MP_HUBBARD_METALS)] = True
_IS_AN_LUT = np.zeros(128, dtype=bool)
_IS_AN_LUT[sorted(MP_HUBBARD_ANIONS)] = True


def pade_expm_3x3(A: np.ndarray) -> np.ndarray:
    """Exact scaling-and-squaring matrix exponential for 3x3 FP64 matrices."""
    norm = float(np.max(np.sum(np.abs(A), axis=0)))
    if norm == 0.0:
        return _I3.copy()
    s = max(0, int(math.ceil(math.log2(norm / 0.125)))) if norm > 0.125 else 0
    As = A * (0.5 ** s) if s else A
    A2 = As @ As
    A3 = A2 @ As
    A4 = A2 @ A2
    P = (
        _I3 + As + 0.5 * A2 + (1.0 / 6.0) * A3
        + A4 @ ((1.0 / 24.0) * _I3 + (1.0 / 120.0) * As + (1.0 / 720.0) * A2 + (1.0 / 5040.0) * A3)
        + (A4 @ A4) @ ((1.0 / 40320.0) * _I3 + (1.0 / 362880.0) * As + (1.0 / 3628800.0) * A2 + (1.0 / 39916800.0) * A3)
    )
    for _ in range(s):
        P = P @ P
    return P


def frechet_expm_3x3(A: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Exact Fréchet derivative D exp(A)[E] for 3x3 FP64 matrices."""
    norm = float(np.max(np.sum(np.abs(A), axis=0)))
    s = max(0, int(math.ceil(math.log2(norm / 0.0625)))) if norm > 0.0625 else 0
    scale = 0.5 ** s
    As, Es = A * scale, E * scale
    X1, D1 = As, Es
    X2, D2 = 0.5 * (X1 @ As), 0.5 * (D1 @ As + X1 @ Es)
    X3, D3 = (1.0 / 3.0) * (X2 @ As), (1.0 / 3.0) * (D2 @ As + X2 @ Es)
    X4, D4 = 0.25 * (X3 @ As), 0.25 * (D3 @ As + X3 @ Es)
    X5, D5 = 0.2 * (X4 @ As), 0.2 * (D4 @ As + X4 @ Es)
    X6, D6 = (1.0 / 6.0) * (X5 @ As), (1.0 / 6.0) * (D5 @ As + X5 @ Es)
    X7, D7 = (1.0 / 7.0) * (X6 @ As), (1.0 / 7.0) * (D6 @ As + X6 @ Es)
    X8, D8 = 0.125 * (X7 @ As), 0.125 * (D7 @ As + X7 @ Es)
    P = _I3 + X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8
    L = D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8
    for _ in range(s):
        L = P @ L + L @ P
        P = P @ P
    return L


def _u_atoms_from_edges(z: np.ndarray, ii: np.ndarray, jj: np.ndarray, dd: np.ndarray, u_flag: int) -> np.ndarray:
    """Derive per-atom Hubbard +U mask from already-built neighbor distances (zero second `vesin` call)."""
    n = len(z)
    out = np.zeros(n, dtype=np.int64)
    if int(u_flag) == 0 or n == 0 or len(ii) == 0:
        return out
    zc = np.clip(z, 0, len(_IS_TM_LUT) - 1)
    is_tm = _IS_TM_LUT[zc]
    is_an = _IS_AN_LUT[zc]
    if not (is_tm.any() and is_an.any()):
        return out
    m_edge = is_tm[ii] & is_an[jj]
    if not m_edge.any():
        return out
    i_m, j_m, d_m = ii[m_edge], jj[m_edge], dd[m_edge]
    rc_i = _COV_RADII[np.clip(z[i_m], 0, len(_COV_RADII) - 1)]
    rc_j = _COV_RADII[np.clip(z[j_m], 0, len(_COV_RADII) - 1)]
    max_r = np.minimum(rc_i + rc_j + ANION_BOND_TOL_A, NEIGHBOR_CUTOFF_A)
    hit = d_m <= max_r
    if hit.any():
        out[i_m[hit]] = 1
    return out


def _build_slot_edges(
    z: np.ndarray, pos: np.ndarray, cell: np.ndarray, r_list: float, u_val: int, np_dt: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_c, _ = center_positions_f64(pos)
    nl = cached_vesin_nl(r_list)
    if u_val:
        i, j, d, s = nl.compute(points=pos_c, box=cell, periodic=(True, True, True), quantities="ijdS")
        i_a, j_a, d_a = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64), np.asarray(d, dtype=np.float64)
        u_at = _u_atoms_from_edges(z, i_a, j_a, d_a, u_val)
    else:
        i, j, s = nl.compute(points=pos_c, box=cell, periodic=(True, True, True), quantities="ijS")
        i_a, j_a = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
        u_at = np.zeros(len(z), dtype=np.int64)
    ei = np.ascontiguousarray(np.stack([j_a, i_a], axis=0))
    es = np.ascontiguousarray(s, dtype=np_dt)
    return ei, es, u_at


@dataclass
class _CrystalSlot:
    key: Any
    z: np.ndarray
    n: int
    orig_cell: np.ndarray
    inv_orig_cell: np.ndarray
    ecf: float
    x_at: np.ndarray
    U: np.ndarray
    F: np.ndarray
    pos: np.ndarray
    cell: np.ndarray
    v_at: np.ndarray
    v_cell: np.ndarray
    vel_init: bool
    dt: float
    a: float
    n_pos_steps: int
    steps: int
    max_steps: int
    fmax: float
    scalar_pressure: float
    src_name: str
    u_val: int
    tok_id: int
    ref_id: int
    xc_id: int
    paw_id: int
    e0_kcal: float
    t0: float
    # Verlet cache
    cell_ref: np.ndarray | None = None
    frac_ref: np.ndarray | None = None
    ei: np.ndarray | None = None
    es: np.ndarray | None = None
    u_atom: np.ndarray | None = None
    # Latest generalized forces
    g_at: np.ndarray | None = None
    g_cell: np.ndarray | None = None

    def needs_nl_rebuild(self, r_cut: float, skin: float) -> bool:
        if self.ei is None or self.cell_ref is None or self.frac_ref is None or skin <= 0.0:
            return True
        try:
            inv_c = np.linalg.inv(self.cell)
        except np.linalg.LinAlgError:
            return True
        B = inv_c @ self.cell_ref - _I3
        strain_bound = r_cut * float(np.sqrt(np.sum(B * B)))
        if strain_bound >= skin - 1e-6:
            return True
        frac_cur = self.x_at @ self.inv_orig_cell
        ds = (frac_cur - self.frac_ref) @ self.cell_ref
        disp_bound = 2.0 * float(np.sqrt(np.max(np.sum(ds * ds, axis=-1)))) if self.n > 0 else 0.0
        return (strain_bound + disp_bound) >= (skin - 1e-6)

    def fire_step(
        self, *, maxstep: float = 0.2, dtmax: float = 1.0, nmin: int = 5,
        finc: float = 1.1, fdec: float = 0.5, astart: float = 0.1, fa: float = 0.99,
    ) -> None:
        ga, gc = self.g_at, self.g_cell
        assert ga is not None and gc is not None
        if not self.vel_init:
            self.v_at.fill(0.0)
            self.v_cell.fill(0.0)
            self.vel_init = True
        else:
            vf = float(np.sum(ga * self.v_at) + np.sum(gc * self.v_cell))
            if vf > 0.0:
                g2 = float(np.sum(ga * ga) + np.sum(gc * gc))
                v2 = float(np.sum(self.v_at * self.v_at) + np.sum(self.v_cell * self.v_cell))
                if g2 > 0.0:
                    scale = self.a * math.sqrt(v2 / g2)
                    om = 1.0 - self.a
                    self.v_at = om * self.v_at + scale * ga
                    self.v_cell = om * self.v_cell + scale * gc
                if self.n_pos_steps > nmin:
                    self.dt = min(self.dt * finc, dtmax)
                    self.a *= fa
                self.n_pos_steps += 1
            else:
                self.v_at.fill(0.0)
                self.v_cell.fill(0.0)
                self.a = astart
                self.dt *= fdec
                self.n_pos_steps = 0

        self.v_at += self.dt * ga
        self.v_cell += self.dt * gc
        dr_at = self.dt * self.v_at
        dr_cell = self.dt * self.v_cell
        normdr = math.sqrt(float(np.sum(dr_at * dr_at) + np.sum(dr_cell * dr_cell)))
        if normdr > maxstep:
            fac = maxstep / normdr
            dr_at = dr_at * fac
            dr_cell = dr_cell * fac

        self.x_at = self.x_at + dr_at
        self.U = self.U + dr_cell / self.ecf
        self.F = pade_expm_3x3(self.U)
        ft = self.F.T
        self.cell = self.orig_cell @ ft
        self.pos = self.x_at @ ft
        self.steps += 1

    def update_forces_and_check(self, f_ev: np.ndarray, s33_ev: np.ndarray) -> tuple[bool, float, float]:
        vol = float(abs(np.linalg.det(self.cell)))
        if not np.isfinite(vol) or vol <= 1e-8:
            raise RuntimeError(f"Degenerate cell volume ({vol:.3e}) during relaxation")
        virial = -vol * (s33_ev + np.diag([self.scalar_pressure] * 3))
        ucf_cell_grad = virial @ np.linalg.inv(self.F.T)
        g_cell = frechet_expm_3x3(self.U.T, ucf_cell_grad) / self.ecf
        g_at = f_ev @ self.F
        self.g_at, self.g_cell = g_at, g_cell
        gmax_at = float(np.sqrt(np.max(np.sum(g_at * g_at, axis=-1)))) if self.n > 0 else 0.0
        gmax_cell = float(np.sqrt(np.max(np.sum(g_cell * g_cell, axis=-1))))
        gmax = max(gmax_at, gmax_cell)
        fmax_comp = float(np.max(np.abs(f_ev))) if self.n > 0 else 0.0
        return (gmax < self.fmax), gmax, fmax_comp


def _eval_crystal_slots_efs(
    calc: "MenaCalculator",
    slots: list[_CrystalSlot],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Lean E+F+S forward+backward on a homogeneous crystal batch (single H2D & single D2H)."""
    dev, dt = calc.device, calc.dtype
    np_dt = np.float32 if dt == torch.float32 else np.float64
    G = len(slots)
    counts = [s.n for s in slots]
    tot_n = sum(counts)

    z_all = np.concatenate([s.z for s in slots])
    pos_list = []
    for s in slots:
        pc, _ = center_positions_f64(s.pos)
        pos_list.append(pc.astype(np_dt, copy=False))
    pos_all = np.concatenate(pos_list, axis=0)
    cell_all = np.stack([s.cell.astype(np_dt, copy=False) for s in slots], axis=0)
    graph_all = np.repeat(np.arange(G, dtype=np.int64), counts)
    u_atom_all = np.concatenate([s.u_atom for s in slots])  # type: ignore[arg-type]

    ei_parts, es_parts = [], []
    off = 0
    for s in slots:
        assert s.ei is not None and s.es is not None
        if s.ei.shape[1] > 0:
            ei_parts.append(s.ei + off)
            es_parts.append(s.es)
        off += s.n
    if ei_parts:
        ei_all = np.concatenate(ei_parts, axis=1)
        es_all = np.concatenate(es_parts, axis=0)
    else:
        ei_all = np.zeros((2, 0), dtype=np.int64)
        es_all = np.zeros((0, 3), dtype=np_dt)

    us = [s.u_val for s in slots]
    toks = [s.tok_id for s in slots]
    refs = [s.ref_id for s in slots]
    xcs = [s.xc_id for s in slots]
    paws = [s.paw_id for s in slots]
    utr_t = torch.tensor([us, toks, refs, xcs, paws], dtype=torch.long, device=dev)
    qs_t = torch.zeros((2, G), dtype=dt, device=dev)
    qs_t[1].fill_(1.0)

    ei_dev = torch.from_numpy(ei_all).to(dev)
    b = GraphBatch(
        z=torch.from_numpy(z_all).to(dev),
        pos=torch.from_numpy(pos_all).to(dev),
        graph=torch.from_numpy(graph_all).to(dev),
        cell=torch.from_numpy(cell_all).to(dev),
        edge_index=ei_dev,
        edge_shift=torch.from_numpy(es_all).to(dev),
        charge=qs_t[0], spin=qs_t[1],
        u_flag=utr_t[0], token=utr_t[1], ref_id=utr_t[2],
        domain="crystal",
        atom_mask=torch.ones(tot_n, dtype=torch.bool, device=dev),
        graph_mask=torch.ones(G, dtype=torch.bool, device=dev),
        energy=torch.zeros(G, dtype=dt, device=dev),
        forces=torch.zeros((tot_n, 3), dtype=dt, device=dev),
        energy_mask=torch.ones(G, dtype=torch.bool, device=dev),
        force_mask=torch.ones(tot_n, dtype=torch.bool, device=dev),
        coulomb_cap=None,
        pbc=True,
        xc_id=utr_t[3], paw_id=utr_t[4],
        u_atom=torch.from_numpy(u_atom_all).to(dev),
    )
    csr_plan = build_csr_plan(ei_dev[1], ei_dev[0], tot_n) if calc.model.trunk.edge_mode == "csr" else None
    out = calc._run_model(b, None, need_forces=True, need_stress=True, csr_plan=csr_plan)

    flat = torch.cat([
        out["energy_atom"].reshape(-1),
        out["forces"].reshape(-1),
        out["stress"].reshape(-1),
    ], dim=0).detach().cpu().to(torch.float64).numpy()

    e_at = flat[:tot_n]
    f_all = flat[tot_n : tot_n + 3 * tot_n].reshape(tot_n, 3)
    s_all = flat[tot_n + 3 * tot_n : tot_n + 3 * tot_n + 9 * G].reshape(G, 3, 3)

    inv_ev = 1.0 / CRYSTAL_EV
    e_ev = np.empty(G, dtype=np.float64)
    f_list: list[np.ndarray] = []
    off = 0
    for g_i, s in enumerate(slots):
        e_model_k = float(e_at[off : off + s.n].sum())
        e_ev[g_i] = (e_model_k + s.e0_kcal) * inv_ev
        f_list.append((f_all[off : off + s.n] * inv_ev).copy())
        off += s.n
    return e_ev, f_list, s_all * inv_ev


def _make_slot(
    calc: "MenaCalculator", key: Any, atoms: Atoms | RawCrystal, *, fmax: float, max_steps: int,
    scalar_pressure: float, u_flag_fn: Callable[[np.ndarray], int] | None,
) -> _CrystalSlot:
    if isinstance(atoms, RawCrystal):
        z = np.ascontiguousarray(atoms.numbers, dtype=np.int64)
        pos0 = np.ascontiguousarray(atoms.positions, dtype=np.float64)
        cell0 = np.ascontiguousarray(atoms.cell, dtype=np.float64)
        u_in = atoms.u_flag
    else:
        z = np.ascontiguousarray(atoms.get_atomic_numbers(), dtype=np.int64)
        pos0 = np.ascontiguousarray(atoms.get_positions(), dtype=np.float64)
        cell0 = np.ascontiguousarray(atoms.get_cell().array, dtype=np.float64)
        u_in = atoms.info.get("u_flag")
    n = len(z)
    u_override = u_flag_fn(z) if u_flag_fn is not None else (calc.u_override if calc.u_override is not None else u_in)
    if u_override is not None and calc.lot == "mp":
        u_val = int(u_override)
        src_name = "mp_pbeU" if u_val else "mp_pbe"
    else:
        at_tmp = Atoms(numbers=z, positions=pos0, cell=cell0, pbc=True) if isinstance(atoms, RawCrystal) else atoms
        src, u_val = resolve_lot_and_u(at_tmp, calc.lot, u_override)
        src_name = src.name
    tok_id, ref_id = source_ids(src_name)
    xc_id, paw_id = lot_ids(src_name)
    e0_kcal, _, _ = ref_energies_kcal(calc.add_ref, src_name, "crystal", True, z, 0.0, 1)
    return _CrystalSlot(
        key=key, z=z, n=n,
        orig_cell=cell0.copy(), inv_orig_cell=np.linalg.inv(cell0),
        ecf=float(n), x_at=pos0.copy(), U=np.zeros((3, 3), dtype=np.float64), F=_I3.copy(),
        pos=pos0.copy(), cell=cell0.copy(),
        v_at=np.zeros((n, 3), dtype=np.float64), v_cell=np.zeros((3, 3), dtype=np.float64),
        vel_init=False, dt=0.1, a=0.1, n_pos_steps=0, steps=0,
        max_steps=int(max_steps), fmax=float(fmax), scalar_pressure=float(scalar_pressure),
        src_name=src_name, u_val=int(u_val), tok_id=tok_id, ref_id=ref_id, xc_id=xc_id, paw_id=paw_id,
        e0_kcal=float(e0_kcal), t0=time.time(),
    )


def relax_crystals_fire_stream(
    calc: "MenaCalculator",
    items: Iterable[tuple[Any, Atoms | RawCrystal]],
    *,
    fmax: float = 0.02,
    steps: int = 500,
    maxstep: float = 0.2,
    skin: float = 0.5,
    max_batch: int = 128,
    max_atoms: int = 4096,
    scalar_pressure: float = 0.0,
    nl_threads: int = 4,
    u_flag_fn: Callable[[np.ndarray], int] | None = None,
    batched_fire: bool | str = "auto",
) -> Iterator[dict[str, Any]]:
    """Stream-relax periodic 3D crystals with continuous sliding-window batched FIRE + FrechetCellFilter."""
    use_gpu_fire = (batched_fire is True) or (batched_fire == "auto" and calc.device.type == "cuda")
    if use_gpu_fire:
        from lam4.calc.relax_crystal_fire_gpu import relax_crystals_fire_gpu_stream

        yield from relax_crystals_fire_gpu_stream(
            calc, items, fmax=fmax, steps=steps, maxstep=maxstep, skin=skin,
            max_batch=max_batch, max_atoms=max_atoms, scalar_pressure=scalar_pressure,
            nl_threads=nl_threads, u_flag_fn=u_flag_fn, state_device=calc.device,
        )
        return
    r_cut = float(calc.model.cfg.r_cut)
    eff_skin = max(0.0, float(skin))
    r_list = r_cut + eff_skin
    np_dt = np.float32 if calc.dtype == torch.float32 else np.float64
    it = iter(items)
    exhausted = False
    pending: tuple[Any, Atoms | RawCrystal] | None = None
    active: list[_CrystalSlot] = []
    cur_atoms = 0
    pool = ThreadPoolExecutor(max_workers=max(1, int(nl_threads))) if int(nl_threads) > 1 else None

    def _refill() -> None:
        nonlocal exhausted, pending, cur_atoms
        while len(active) < max_batch and not exhausted:
            if pending is not None:
                key, at = pending
                pending = None
            else:
                try:
                    key, at = next(it)
                except StopIteration:
                    exhausted = True
                    break
            na = len(at.numbers) if isinstance(at, RawCrystal) else len(at)
            if active and cur_atoms + na > max_atoms:
                pending = (key, at)
                break
            try:
                slot = _make_slot(
                    calc, key, at, fmax=fmax, max_steps=steps,
                    scalar_pressure=scalar_pressure, u_flag_fn=u_flag_fn,
                )
            except Exception as exc:
                yield_err.append({"key": key, "n_atoms": na, "error": repr(exc)[:200]})
                continue
            active.append(slot)
            cur_atoms += na

    def _rebuild_dirty(slots: list[_CrystalSlot]) -> None:
        dirty = [s for s in slots if s.needs_nl_rebuild(r_cut, eff_skin)]
        if not dirty:
            return
        if pool is not None and len(dirty) >= 2:
            futs = [pool.submit(_build_slot_edges, s.z, s.pos, s.cell, r_list, s.u_val, np_dt) for s in dirty]
            for s, fut in zip(dirty, futs):
                ei, es, u_at = fut.result()
                s.ei, s.es, s.u_atom = ei, es, u_at
                s.cell_ref = s.cell.copy()
                s.frac_ref = s.x_at @ s.inv_orig_cell
        else:
            for s in dirty:
                ei, es, u_at = _build_slot_edges(s.z, s.pos, s.cell, r_list, s.u_val, np_dt)
                s.ei, s.es, s.u_atom = ei, es, u_at
                s.cell_ref = s.cell.copy()
                s.frac_ref = s.x_at @ s.inv_orig_cell

    yield_err: list[dict[str, Any]] = []
    try:
        while True:
            _refill()
            while yield_err:
                yield yield_err.pop(0)
            if not active:
                break

            # Advance slots that already have a force evaluation (all except newly admitted slots)
            for s in active:
                if s.g_at is not None:
                    s.fire_step(maxstep=maxstep)

            _rebuild_dirty(active)
            try:
                e_ev, f_list, s_all = _eval_crystal_slots_efs(calc, active)
            except Exception:
                if calc.device.type == "cuda":
                    torch.cuda.empty_cache()
                # Isolate any bad structure in the current batch cleanly
                keep_fallback: list[_CrystalSlot] = []
                for s in active:
                    try:
                        ei_s, fl_s, st_s = _eval_crystal_slots_efs(calc, [s])
                        is_conv, gmax, fmax_c = s.update_forces_and_check(fl_s[0], st_s[0])
                        if is_conv or s.steps >= s.max_steps:
                            yield {
                                "key": s.key, "e_uncorr_eV": float(ei_s[0]), "n_atoms": s.n,
                                "steps": int(s.steps), "fmax_final": float(fmax_c),
                                "gmax_final": float(gmax), "converged": bool(is_conv),
                                "sec": round(time.time() - s.t0, 2),
                                "cell": s.cell.tolist(), "pos": s.pos.tolist(), "z": s.z.tolist(), "error": "",
                            }
                        else:
                            keep_fallback.append(s)
                    except Exception as exc_s:
                        yield {"key": s.key, "n_atoms": s.n, "error": repr(exc_s)[:200]}
                active = keep_fallback
                cur_atoms = sum(s.n for s in active)
                continue

            keep: list[_CrystalSlot] = []
            for g_i, s in enumerate(active):
                try:
                    is_conv, gmax, fmax_c = s.update_forces_and_check(f_list[g_i], s_all[g_i])
                    if not (np.isfinite(e_ev[g_i]) and np.isfinite(gmax)):
                        raise RuntimeError("Non-finite energy or force")
                    if is_conv or s.steps >= s.max_steps:
                        yield {
                            "key": s.key, "e_uncorr_eV": float(e_ev[g_i]), "n_atoms": s.n,
                            "steps": int(s.steps), "fmax_final": float(fmax_c),
                            "gmax_final": float(gmax), "converged": bool(is_conv),
                            "sec": round(time.time() - s.t0, 2),
                            "cell": s.cell.tolist(), "pos": s.pos.tolist(), "z": s.z.tolist(), "error": "",
                        }
                    else:
                        keep.append(s)
                except Exception as exc_i:
                    yield {"key": s.key, "n_atoms": s.n, "error": repr(exc_i)[:200]}
            active = keep
            cur_atoms = sum(s.n for s in active)
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
