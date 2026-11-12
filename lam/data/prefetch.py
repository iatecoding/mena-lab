"""prefetch.py: Atom-budget batch packing and multi-threaded background collation for crystal shards."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import time

import numpy as np
import torch

from lam.data.shards import MatFlat, collate
from lam.units import EV


CRYSTAL_N_BUCKETS = (256, 512, 1024, 2048, 4096, 7168, 14336, 21504, 32768)
CRYSTAL_E_BUCKETS = (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152)
CRYSTAL_B_BUCKETS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def pack(rng, idx_pool: np.ndarray, n_atoms: np.ndarray, budget: int):
    """Endless generator of index arrays whose atom counts sum to <= budget (random order, greedy packing)."""
    while True:
        perm = rng.permutation(idx_pool)
        cur, tot = [], 0
        for s in perm:
            n = int(n_atoms[s])
            if n > budget:
                continue
            if tot + n > budget:
                yield np.array(cur, dtype=np.int64)
                cur, tot = [], 0
            cur.append(s)
            tot += n
        if cur:
            yield np.array(cur, dtype=np.int64)


def pack_with_pairs(rng, idx_pool: np.ndarray, n_atoms: np.ndarray, budget: int, pairs: np.ndarray, frac: float):
    """pack() with about `frac` of the atom budget filled by nearby frame pairs."""
    rest = pack(rng, idx_pool, n_atoms, int(budget * (1.0 - frac)))
    order, k = rng.permutation(len(pairs)), 0
    while True:
        base = next(rest)
        room = budget - int(n_atoms[base].sum())
        A, Bf, na = [], [], 0
        for _ in range(64):
            if k >= len(order):
                order, k = rng.permutation(len(pairs)), 0
            i, j = pairs[order[k]]
            k += 1
            n = int(n_atoms[i])
            if 2 * n > room:
                if room < 2 * 8:
                    break
                continue
            A.append(i)
            Bf.append(j)
            na += n
            room -= 2 * n
        yield np.concatenate([np.asarray(A, np.int64), np.asarray(Bf, np.int64), base]), len(A), na


def _round_to_bucket(val: int, buckets: tuple[int, ...]) -> int:
    for b in buckets:
        if val <= b:
            return b
    return int(((val + 1023) // 1024) * 1024)


def pad_crystal_batch(
    bt: dict,
    n_pad: int | None = None,
    e_pad: int | None = None,
    b_pad: int | None = None,
    n_buckets: tuple[int, ...] = CRYSTAL_N_BUCKETS,
    e_buckets: tuple[int, ...] = CRYSTAL_E_BUCKETS,
    b_buckets: tuple[int, ...] = CRYSTAL_B_BUCKETS,
) -> dict:
    """Pad a collated crystal batch `(N, E, B)` to static bucket dimensions `(N_pad, E_pad, B_pad)` for `torch.compile`.
    Adds at least 1 dummy ghost graph at index `B_pad - 1` with `NaN` energy/stress and dummy atoms `N..N_pad-1` with
    `z = 0`, `forces = NaN`, and dummy edges `E..E_pad-1` with distance 100.0 A > r_cut (zero cutoff weight).
    Because `crystal_loss_fn` masks non-finite `e_res`, `forces`, and `stress`, the padded batch has identical loss
    and gradients to the unpadded batch while presenting fixed static shapes to `torch._dynamo`."""
    n_real = int(bt["z"].shape[0])
    e_real = int(bt["edge_index"].shape[1])
    b_real = int(bt["cell"].shape[0])

    target_n = int(n_pad) if n_pad is not None else _round_to_bucket(n_real + 1, n_buckets)
    target_e = int(e_pad) if e_pad is not None else _round_to_bucket(e_real + 1, e_buckets)
    target_b = int(b_pad) if b_pad is not None else _round_to_bucket(b_real + 1, b_buckets)

    target_n = max(target_n, n_real + 1)
    target_e = max(target_e, e_real + 1)
    target_b = max(target_b, b_real + 1)

    dn = target_n - n_real
    de = target_e - e_real
    db = target_b - b_real
    ghost_b = target_b - 1
    ghost_atom = target_n - 1

    dev = bt["z"].device
    pos_dtype = bt["pos"].dtype

    z_pad = torch.cat([bt["z"], torch.zeros(dn, dtype=bt["z"].dtype, device=dev)], dim=0)
    pos_pad = torch.cat([bt["pos"], torch.zeros((dn, 3), dtype=pos_dtype, device=dev)], dim=0)
    f_pad = torch.cat([bt["forces"], torch.full((dn, 3), float("nan"), dtype=pos_dtype, device=dev)], dim=0)
    batch_pad = torch.cat([bt["batch"], torch.full((dn,), ghost_b, dtype=bt["batch"].dtype, device=dev)], dim=0)

    dummy_ei = torch.full((2, de), ghost_atom, dtype=bt["edge_index"].dtype, device=dev)
    ei_pad = torch.cat([bt["edge_index"], dummy_ei], dim=1)

    dummy_es = torch.zeros((de, 3), dtype=bt["edge_shift"].dtype, device=dev)
    dummy_es[:, 0] = 1
    es_pad = torch.cat([bt["edge_shift"], dummy_es], dim=0)

    dummy_esc = torch.zeros((de, 3), dtype=pos_dtype, device=dev)
    dummy_esc[:, 0] = 100.0
    esc_pad = (
        torch.cat([bt["edge_shift_cart"], dummy_esc], dim=0)
        if bt.get("edge_shift_cart") is not None
        else None
    )

    dummy_cell = torch.eye(3, dtype=pos_dtype, device=dev).unsqueeze(0).repeat(db, 1, 1) * 100.0
    cell_pad = torch.cat([bt["cell"], dummy_cell], dim=0)

    e_res_pad = torch.cat([bt["e_res"], torch.full((db,), float("nan"), dtype=bt["e_res"].dtype, device=dev)], dim=0)
    stress_pad = torch.cat(
        [bt["stress"], torch.full((db, 3, 3), float("nan"), dtype=bt["stress"].dtype, device=dev)], dim=0
    )
    u_pad = torch.cat([bt["u"], torch.zeros(db, dtype=bt["u"].dtype, device=dev)], dim=0)
    lot_pad = torch.cat([bt["lot_id"], torch.zeros(db, dtype=bt["lot_id"].dtype, device=dev)], dim=0)

    na_tail = torch.ones(db, dtype=bt["n_atoms"].dtype, device=dev)
    na_tail[-1] = max(1, dn)
    n_atoms_pad = torch.cat([bt["n_atoms"], na_tail], dim=0)

    out = dict(bt)
    out.update(
        {
            "z": z_pad,
            "pos": pos_pad,
            "forces": f_pad,
            "batch": batch_pad,
            "edge_index": ei_pad,
            "edge_shift": es_pad,
            "edge_shift_cart": esc_pad,
            "cell": cell_pad,
            "e_res": e_res_pad,
            "stress": stress_pad,
            "u": u_pad,
            "lot_id": lot_pad,
            "n_atoms": n_atoms_pad,
            "n_real_atoms": n_real,
            "n_real_graphs": b_real,
            "bucket_shape": (target_n, target_e, target_b),
        }
    )
    return out


class Prefetch:
    """Background threads that collate crystal batches (vesin neighbour lists + pre-allocated buffers)."""

    def __init__(
        self,
        flat,
        gen,
        cutoff: float,
        mu_kcal: np.ndarray | torch.Tensor,
        workers: int = 3,
        depth: int = 12,
        pin_memory: bool = False,
        lot_id: int | tuple[int, int] | dict[int, int] | str | None = None,
        lot_id_by_u: tuple[int, int] | dict[int, int] | str | int | None = None,
        ev: float = EV,
        units: str | None = None,
        mu_in_kcal: bool | None = None,
    ) -> None:
        self.q: queue.Queue = queue.Queue(maxsize=max(depth, workers + 2))
        self.lock = threading.Lock()
        self.gen = gen
        self.flat = flat
        self.cutoff = cutoff
        self.mu = mu_kcal
        self.pin_memory = pin_memory
        self.lot_id = lot_id
        self.lot_id_by_u = lot_id_by_u
        self.ev = ev
        self.units = units
        self.mu_in_kcal = mu_in_kcal
        self.stop = False
        self.threads = [threading.Thread(target=self._run, daemon=True) for _ in range(workers)]
        for t in self.threads:
            t.start()

    def _run(self) -> None:
        while not self.stop:
            try:
                with self.lock:
                    if self.stop:
                        break
                    idx = next(self.gen)
            except StopIteration:
                break
            meta = None
            if isinstance(idx, tuple):
                idx, *meta = idx
            bt = collate(
                self.flat,
                idx,
                self.cutoff,
                self.mu,
                ev=self.ev,
                pin_memory=self.pin_memory,
                lot_id=self.lot_id,
                lot_id_by_u=self.lot_id_by_u,
                units=self.units,
                mu_in_kcal=self.mu_in_kcal,
            )
            if meta is not None:
                bt["n_pairs"], bt["pair_atoms"] = int(meta[0]), int(meta[1])
            while not self.stop:
                try:
                    self.q.put(bt, timeout=0.05)
                    break
                except queue.Full:
                    continue

    def next(self) -> dict:
        return self.q.get()

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        return self.next()

    @property
    def t(self) -> threading.Thread:
        return self.threads[0]

    def close(self) -> None:
        self.stop = True
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=1.0)


class CrystalPartStream:
    """Streams multiple crystal shard directories (e.g. `omat24_sub_p0..p3` or local subsets),
    prefetching the next shard's memory-mapped `MatFlat` and initial batches in a background thread."""

    def __init__(
        self,
        dirs: list[str | Path],
        cutoff: float,
        mu_kcal: np.ndarray | torch.Tensor,
        budget: int = 7168,
        batches_per_shard: int = 500,
        seed: int = 42,
        workers: int = 1,
        lot_id_by_u: tuple[int, int] | dict[int, int] | str | int | None = None,
        units: str = "ev",
        mu_in_kcal: bool | None = None,
        state: dict | None = None,
    ) -> None:
        if not dirs:
            raise ValueError("CrystalPartStream requires at least one shard directory")
        self.dirs = [str(Path(d)) for d in dirs]
        self.cutoff = float(cutoff)
        self.mu = mu_kcal
        self.budget = int(budget)
        self.batches_per_shard = int(batches_per_shard)
        self.seed = int(seed)
        self.workers = int(workers)
        self.lot_id_by_u = lot_id_by_u
        self.units = units
        self.mu_in_kcal = mu_in_kcal

        st = state or {"cycle": 0, "k": 0, "pos": 0}
        self.cycle = int(st.get("cycle", 0))
        self.k = int(st.get("k", 0)) % len(self.dirs)
        self.pos = int(st.get("pos", 0))
        self.flat: MatFlat | None = None
        self.pf: Prefetch | None = None
        self._next_box: dict | None = None
        self.switch_log: list[dict] = []
        self._open_current(advance_to_pos=self.pos)

    def _shard_seed(self, cycle: int, k: int) -> int:
        return self.seed + 7919 * k + 104729 * cycle

    def _build_pf(self, flat: MatFlat, cycle: int, k: int, advance_to_pos: int = 0) -> Prefetch:
        rng = np.random.default_rng(self._shard_seed(cycle, k))
        pool = np.arange(len(flat), dtype=np.int64)
        gen = pack(rng, pool, flat.n_atoms, self.budget)
        for _ in range(max(0, int(advance_to_pos))):
            next(gen)
        return Prefetch(
            flat,
            gen,
            self.cutoff,
            self.mu,
            workers=self.workers,
            lot_id_by_u=self.lot_id_by_u,
            units=self.units,
            mu_in_kcal=self.mu_in_kcal,
        )

    def _open_current(self, advance_to_pos: int = 0) -> None:
        if self.pf is not None:
            self.pf.close()
        if self.flat is not None:
            self.flat.close()
        self.flat = MatFlat(self.dirs[self.k])
        self.pf = self._build_pf(self.flat, self.cycle, self.k, advance_to_pos=advance_to_pos)
        self.pos = int(advance_to_pos)

    def prefetch(self) -> None:
        if len(self.dirs) < 2 or self._next_box is not None:
            return
        next_k = (self.k + 1) % len(self.dirs)
        next_c = self.cycle + (1 if next_k == 0 else 0)
        next_dir = self.dirs[next_k]
        box: dict = {"cycle": next_c, "k": next_k, "dir": next_dir}

        def _bg():
            try:
                f = MatFlat(next_dir)
                pf = self._build_pf(f, next_c, next_k, advance_to_pos=0)
                box["flat"] = f
                box["pf"] = pf
            except Exception as exc:
                box["error"] = repr(exc)

        th = threading.Thread(target=_bg, daemon=True)
        th.start()
        box["thread"] = th
        self._next_box = box

    def _switch(self) -> None:
        t0 = time.perf_counter()
        if self.pf is not None:
            self.pf.close()
        if self.flat is not None:
            self.flat.close()
        next_k = (self.k + 1) % len(self.dirs)
        next_c = self.cycle + (1 if next_k == 0 else 0)
        how = "direct"
        if self._next_box is not None and self._next_box.get("k") == next_k and self._next_box.get("cycle") == next_c:
            self._next_box["thread"].join(timeout=5.0)
            if "pf" in self._next_box and "flat" in self._next_box:
                self.flat = self._next_box["flat"]
                self.pf = self._next_box["pf"]
                self.cycle = next_c
                self.k = next_k
                self.pos = 0
                how = "prefetched"
        self._next_box = None
        if how == "direct":
            self.cycle = next_c
            self.k = next_k
            self._open_current(advance_to_pos=0)
        self.switch_log.append(
            {
                "cycle": self.cycle,
                "k": self.k,
                "dir": self.dirs[self.k],
                "how": how,
                "sec": round(time.perf_counter() - t0, 4),
            }
        )

    def next(self) -> dict:
        if self.pos >= self.batches_per_shard and len(self.dirs) > 1:
            self._switch()
        if self.pos == 1 and len(self.dirs) > 1:
            self.prefetch()
        assert self.pf is not None
        bt = self.pf.next()
        self.pos += 1
        return bt

    next_batch = next
    __next__ = next

    def __iter__(self):
        return self

    def set_budget(self, new_budget: int) -> None:
        if int(new_budget) != self.budget:
            self.budget = int(new_budget)
            self._open_current(advance_to_pos=0)

    def state(self) -> dict:
        return {
            "cycle": self.cycle,
            "k": self.k,
            "pos": self.pos,
            "dirs": list(self.dirs),
            "budget": self.budget,
            "seed": self.seed,
            "batches_per_shard": self.batches_per_shard,
        }

    state_dict = state

    def load_state_dict(self, st: dict) -> None:
        if st.get("dirs"):
            self.dirs = [str(Path(d)) for d in st["dirs"]]
        if "budget" in st:
            self.budget = int(st["budget"])
        if "seed" in st:
            self.seed = int(st["seed"])
        if "batches_per_shard" in st:
            self.batches_per_shard = int(st["batches_per_shard"])
        self.cycle = int(st.get("cycle", 0))
        self.k = int(st.get("k", 0)) % len(self.dirs)
        self.pos = int(st.get("pos", 0))
        self._open_current(advance_to_pos=self.pos)

    def close(self) -> None:
        if self._next_box is not None:
            th = self._next_box.get("thread")
            if th is not None and th.is_alive():
                th.join(timeout=1.0)
            if "pf" in self._next_box:
                self._next_box["pf"].close()
            if "flat" in self._next_box:
                self._next_box["flat"].close()
            self._next_box = None
        if self.pf is not None:
            self.pf.close()
            self.pf = None
        if self.flat is not None:
            self.flat.close()
            self.flat = None


class ResidentCrystalStream:
    """Stateful batch stream over a resident `MatFlat` dataset restricted to a held-out-disjoint `train_pool`.
    Supports exact checkpoint round-trip via `state_dict()` and `load_state_dict()`."""

    def __init__(
        self,
        flat: MatFlat,
        train_pool: np.ndarray,
        cutoff: float,
        mu_kcal: np.ndarray | torch.Tensor,
        budget: int = 7168,
        seed: int = 42,
        workers: int = 1,
        depth: int = 8,
        lot_id_by_u: tuple[int, int] | dict[int, int] | str | int | None = None,
        units: str = "ev",
        mu_in_kcal: bool | None = None,
        pairs: np.ndarray | None = None,
        pair_frac: float = 0.0,
        state: dict | None = None,
        owns_flat: bool = False,
    ) -> None:
        self.flat = flat
        self.train_pool = np.asarray(train_pool, dtype=np.int64)
        if self.train_pool.size == 0:
            raise ValueError("ResidentCrystalStream requires a non-empty train_pool")
        self.cutoff = float(cutoff)
        self.mu = mu_kcal
        self.budget = int(budget)
        self.seed = int(seed)
        self.workers = int(workers)
        self.depth = int(depth)
        self.lot_id_by_u = lot_id_by_u
        self.units = units
        self.mu_in_kcal = mu_in_kcal
        self.pairs = pairs if (pairs is not None and len(pairs) > 0 and pair_frac > 0.0) else None
        self.pair_frac = float(pair_frac) if self.pairs is not None else 0.0
        self.owns_flat = bool(owns_flat)

        st = state or {"pos": 0}
        self.pos = int(st.get("pos", 0))
        self.pf: Prefetch | None = None
        self._open_pf(advance_to_pos=self.pos)

    def _open_pf(self, advance_to_pos: int = 0) -> None:
        if self.pf is not None:
            self.pf.close()
        rng = np.random.default_rng(self.seed)
        if self.pairs is not None and self.pair_frac > 0.0:
            gen = pack_with_pairs(rng, self.train_pool, self.flat.n_atoms, self.budget, self.pairs, self.pair_frac)
        else:
            gen = pack(rng, self.train_pool, self.flat.n_atoms, self.budget)
        for _ in range(max(0, int(advance_to_pos))):
            next(gen)
        self.pf = Prefetch(
            self.flat,
            gen,
            self.cutoff,
            self.mu,
            workers=self.workers,
            depth=self.depth,
            lot_id_by_u=self.lot_id_by_u,
            units=self.units,
            mu_in_kcal=self.mu_in_kcal,
        )
        self.pos = int(advance_to_pos)

    def next(self) -> dict:
        assert self.pf is not None
        bt = self.pf.next()
        self.pos += 1
        return bt

    next_batch = next
    __next__ = next

    def __iter__(self):
        return self

    def state_dict(self) -> dict:
        return {
            "pos": int(self.pos),
            "seed": int(self.seed),
            "budget": int(self.budget),
            "pool_size": int(len(self.train_pool)),
        }

    state = state_dict

    def load_state_dict(self, st: dict) -> None:
        if "budget" in st:
            self.budget = int(st["budget"])
        if "seed" in st:
            self.seed = int(st["seed"])
        self.pos = int(st.get("pos", 0))
        self._open_pf(advance_to_pos=self.pos)

    def close(self) -> None:
        if self.pf is not None:
            self.pf.close()
            self.pf = None
        if self.owns_flat and self.flat is not None:
            self.flat.close()


class MultiCrystalStream:
    """Manages multiple crystal streams (both `ResidentCrystalStream` and `CrystalPartStream`) keyed by
    dataset name, providing unified batch retrieval and checkpoint `state_dict` / `load_state_dict`."""

    def __init__(self, streams: dict[str, ResidentCrystalStream | CrystalPartStream]) -> None:
        if not streams:
            raise ValueError("MultiCrystalStream requires at least one crystal stream")
        self.streams = dict(streams)

    def next_for(self, ds_name: str) -> dict:
        if ds_name not in self.streams:
            raise KeyError(f"Unknown crystal dataset stream {ds_name!r}; available: {sorted(self.streams)}")
        bt = self.streams[ds_name].next()
        bt["dataset_name"] = ds_name
        return bt

    def sample_next(self, sampler, progress: float = 0.0) -> tuple[object, dict]:
        spec = sampler.sample_crystal(progress)
        return spec, self.next_for(spec.name)

    def state_dict(self) -> dict:
        return {
            "streams": {name: s.state_dict() for name, s in self.streams.items()},
        }

    state = state_dict

    def load_state_dict(self, st: dict) -> None:
        if "streams" in st and isinstance(st["streams"], dict):
            for name, sub_st in st["streams"].items():
                if name in self.streams and sub_st is not None:
                    self.streams[name].load_state_dict(sub_st)
        elif len(self.streams) == 1:
            only_stream = next(iter(self.streams.values()))
            only_stream.load_state_dict(st)

    def close(self) -> None:
        for s in self.streams.values():
            s.close()


def to_dev(bt: dict, dev: torch.device | str) -> dict:
    """Move collated crystal batch tensors to `dev` (keeping float64 raw `energy` on CPU)."""
    return {
        k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in bt.items()
        if k != "energy"
    }

