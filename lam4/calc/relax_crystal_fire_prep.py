"""Batched FP64 Frechet math and background zip/extxyz/vesin prefetcher for Crystal FIRE."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import queue
import threading
from typing import Any, Callable, Iterable

from ase import Atoms
import numpy as np
import torch
from torch import Tensor

from lam4.calc.physics import resolve_lot_and_u
from lam4.calc.verlet import ref_energies_kcal
from lam4.data.buckets import source_ids
from lam4.registry.sources import lot_ids


@dataclass(slots=True)
class RawCrystal:
    """Lightweight pre-parsed periodic crystal input bypassing ASE Atoms construction."""

    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    u_flag: int | None = None


@dataclass(slots=True)
class PreparedCrystal:
    """Fully initialized crystal slot payload with pre-built initial neighbor graph."""

    key: Any
    z: np.ndarray
    pos0: np.ndarray
    cell0: np.ndarray
    inv_orig_cell: np.ndarray
    u_val: int
    tok_id: int
    ref_id: int
    xc_id: int
    paw_id: int
    e0_kcal: float
    ei: np.ndarray
    es: np.ndarray
    u_atom: np.ndarray
    frac_ref: np.ndarray
    error: str = ""


def batched_pade_expm_3x3(A: Tensor) -> Tensor:
    """Exact batched scaling-and-squaring matrix exponential for [G, 3, 3] FP64 matrices."""
    G = int(A.shape[0])
    eye = torch.eye(3, dtype=A.dtype, device=A.device).unsqueeze(0).expand(G, 3, 3)
    if G == 0:
        return eye.clone()
    norm = A.abs().sum(dim=1).amax(dim=-1)
    s = torch.where(
        norm > 0.125,
        torch.ceil(torch.log2(norm.clamp_min(1e-300) / 0.125)).clamp_min(0.0),
        torch.zeros_like(norm),
    ).to(torch.int64)
    scale = torch.pow(0.5, s.to(A.dtype)).view(G, 1, 1)
    As = A * scale
    A2 = torch.bmm(As, As)
    A3 = torch.bmm(A2, As)
    A4 = torch.bmm(A2, A2)
    P = (
        eye + As + 0.5 * A2 + (1.0 / 6.0) * A3
        + torch.bmm(A4, (1.0 / 24.0) * eye + (1.0 / 120.0) * As + (1.0 / 720.0) * A2 + (1.0 / 5040.0) * A3)
        + torch.bmm(
            torch.bmm(A4, A4),
            (1.0 / 40320.0) * eye + (1.0 / 362880.0) * As + (1.0 / 3628800.0) * A2 + (1.0 / 39916800.0) * A3,
        )
    )
    if bool((norm > 0.125).any()):
        s_max = int(s.max().item())
        for k in range(s_max):
            mask = (k < s).view(G, 1, 1)
            P = torch.where(mask, torch.bmm(P, P), P)
    return torch.where((norm == 0.0).view(G, 1, 1), eye, P)


def batched_frechet_expm_3x3(A: Tensor, E: Tensor) -> Tensor:
    """Exact batched Fréchet derivative D exp(A)[E] for [G, 3, 3] FP64 matrices."""
    G = int(A.shape[0])
    if G == 0:
        return torch.empty_like(A)
    eye = torch.eye(3, dtype=A.dtype, device=A.device).unsqueeze(0).expand(G, 3, 3)
    norm = A.abs().sum(dim=1).amax(dim=-1)
    s = torch.where(
        norm > 0.0625,
        torch.ceil(torch.log2(norm.clamp_min(1e-300) / 0.0625)).clamp_min(0.0),
        torch.zeros_like(norm),
    ).to(torch.int64)
    scale = torch.pow(0.5, s.to(A.dtype)).view(G, 1, 1)
    As, Es = A * scale, E * scale
    X1, D1 = As, Es
    X2, D2 = 0.5 * torch.bmm(X1, As), 0.5 * (torch.bmm(D1, As) + torch.bmm(X1, Es))
    X3, D3 = (1.0 / 3.0) * torch.bmm(X2, As), (1.0 / 3.0) * (torch.bmm(D2, As) + torch.bmm(X2, Es))
    X4, D4 = 0.25 * torch.bmm(X3, As), 0.25 * (torch.bmm(D3, As) + torch.bmm(X3, Es))
    X5, D5 = 0.2 * torch.bmm(X4, As), 0.2 * (torch.bmm(D4, As) + torch.bmm(X4, Es))
    X6, D6 = (1.0 / 6.0) * torch.bmm(X5, As), (1.0 / 6.0) * (torch.bmm(D5, As) + torch.bmm(X5, Es))
    X7, D7 = (1.0 / 7.0) * torch.bmm(X6, As), (1.0 / 7.0) * (torch.bmm(D6, As) + torch.bmm(X6, Es))
    X8, D8 = 0.125 * torch.bmm(X7, As), 0.125 * (torch.bmm(D7, As) + torch.bmm(X7, Es))
    P = eye + X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8
    L = D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8
    if bool((norm > 0.0625).any()):
        s_max = int(s.max().item())
        for k in range(s_max):
            mask = (k < s).view(G, 1, 1)
            L = torch.where(mask, torch.bmm(P, L) + torch.bmm(L, P), L)
            P = torch.where(mask, torch.bmm(P, P), P)
    return L


def prepare_one_crystal(
    key: Any,
    item: Atoms | RawCrystal,
    *,
    lot: str,
    u_override: int | None,
    add_ref: bool,
    r_list: float,
    np_dt: Any,
    u_flag_fn: Callable[[np.ndarray], int] | None,
    build_edges_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> PreparedCrystal:
    """Resolve LoT/E0/inv_cell and build initial `vesin` edges for one crystal."""
    if isinstance(item, RawCrystal):
        z = np.ascontiguousarray(item.numbers, dtype=np.int64)
        pos0 = np.ascontiguousarray(item.positions, dtype=np.float64)
        cell0 = np.ascontiguousarray(item.cell, dtype=np.float64)
        u_in = item.u_flag
    else:
        z = np.ascontiguousarray(item.get_atomic_numbers(), dtype=np.int64)
        pos0 = np.ascontiguousarray(item.get_positions(), dtype=np.float64)
        cell0 = np.ascontiguousarray(item.get_cell().array, dtype=np.float64)
        u_in = item.info.get("u_flag")
    n = len(z)
    try:
        u_ov = u_flag_fn(z) if u_flag_fn is not None else (u_override if u_override is not None else u_in)
        if u_ov is not None and lot == "mp":
            u_val = int(u_ov)
            src_name = "mp_pbeU" if u_val else "mp_pbe"
        else:
            at_tmp = Atoms(numbers=z, positions=pos0, cell=cell0, pbc=True) if isinstance(item, RawCrystal) else item
            src, u_val = resolve_lot_and_u(at_tmp, lot, u_ov)
            src_name = src.name
        tok_id, ref_id = source_ids(src_name)
        xc_id, paw_id = lot_ids(src_name)
        e0_kcal, _, _ = ref_energies_kcal(add_ref, src_name, "crystal", True, z, 0.0, 1)
        inv_orig = np.ascontiguousarray(np.linalg.inv(cell0), dtype=np.float64)
        ei, es, u_atom = build_edges_fn(z, pos0, cell0, r_list, int(u_val), np_dt)
        frac_ref = np.ascontiguousarray(pos0 @ inv_orig, dtype=np.float64)
        return PreparedCrystal(
            key=key, z=z, pos0=pos0, cell0=cell0, inv_orig_cell=inv_orig,
            u_val=int(u_val), tok_id=int(tok_id), ref_id=int(ref_id),
            xc_id=int(xc_id), paw_id=int(paw_id), e0_kcal=float(e0_kcal),
            ei=ei, es=es, u_atom=u_atom, frac_ref=frac_ref,
        )
    except Exception as exc:
        return PreparedCrystal(
            key=key, z=z, pos0=pos0, cell0=cell0, inv_orig_cell=np.eye(3, dtype=np.float64),
            u_val=0, tok_id=0, ref_id=0, xc_id=0, paw_id=0, e0_kcal=0.0,
            ei=np.zeros((2, 0), dtype=np.int64), es=np.zeros((0, 3), dtype=np_dt),
            u_atom=np.zeros(n, dtype=np.int64), frac_ref=np.zeros((n, 3), dtype=np.float64),
            error=repr(exc)[:200],
        )


class CrystalPrefetcher:
    """Overlaps incoming crystal parsing and chunked initial `vesin` builds behind GPU execution."""

    def __init__(
        self,
        items: Iterable[tuple[Any, Atoms | RawCrystal]],
        *,
        lot: str,
        u_override: int | None,
        add_ref: bool,
        r_list: float,
        np_dt: Any,
        u_flag_fn: Callable[[np.ndarray], int] | None,
        build_edges_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
        pool: ThreadPoolExecutor | None,
        max_inflight_batches: int = 32,
        chunk_size: int = 32,
    ) -> None:
        self._chunk_size = max(1, int(chunk_size))
        self._fut_q: queue.Queue[Any] = queue.Queue(maxsize=max(4, int(max_inflight_batches)))
        self._stop = threading.Event()
        self._ready: deque[PreparedCrystal] = deque()
        self._exhausted = False
        self._prep_kw = dict(
            lot=lot, u_override=u_override, add_ref=add_ref,
            r_list=r_list, np_dt=np_dt, u_flag_fn=u_flag_fn, build_edges_fn=build_edges_fn,
        )
        self._pool = pool
        self._thread = threading.Thread(target=self._run, args=(items,), daemon=True)
        self._thread.start()

    def _submit_chunk(self, chunk: list[tuple[Any, Atoms | RawCrystal]]) -> None:
        kw = self._prep_kw
        if self._pool is not None:
            self._put(self._pool.submit(lambda ch: [prepare_one_crystal(k, at, **kw) for k, at in ch], chunk))
        else:
            self._put([prepare_one_crystal(k, at, **kw) for k, at in chunk])

    def _put(self, obj: Any) -> None:
        while not self._stop.is_set():
            try:
                self._fut_q.put(obj, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run(self, items: Iterable[tuple[Any, Atoms | RawCrystal]]) -> None:
        try:
            chunk: list[tuple[Any, Atoms | RawCrystal]] = []
            flushed_first = False
            for pair in items:
                if self._stop.is_set():
                    break
                chunk.append(pair)
                if len(chunk) >= (self._chunk_size if flushed_first else 8):
                    self._submit_chunk(chunk)
                    chunk = []
                    flushed_first = True
            if chunk and not self._stop.is_set():
                self._submit_chunk(chunk)
        except Exception as exc:
            self._put(exc)
        finally:
            self._put(None)

    def next_prepared(self) -> PreparedCrystal | None:
        while not self._ready:
            if self._exhausted:
                return None
            item = self._fut_q.get()
            if item is None:
                self._exhausted = True
                return None
            if isinstance(item, Exception):
                raise item
            self._ready.extend(item.result() if isinstance(item, Future) else item)
        return self._ready.popleft()

    def close(self) -> None:
        self._stop.set()
        while not self._fut_q.empty():
            try:
                self._fut_q.get_nowait()
            except queue.Empty:
                break
