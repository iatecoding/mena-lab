"""sampler.py: training streams of `Packed` batches. `PackedStream` (molecules: shuffled passes over every size bucket, buckets visited in proportion to the atoms they
hold, optional composition chunks for the group-relative energy term), `Prefetcher` (worker thread, side CUDA stream), `DomainSampler` (single stream over domains: one
domain-homogeneous batch per step, chosen by atom-visit share, default molecule 0.92 / crystal 0.08).

Adapted from the earlier Mena training pipeline (PackedStream, chunk_order, Prefetcher); DomainSampler is new (one stream, shares by atom-visits).
The crystal stream (lam4/data/crystal.py `CrystalStream`) only needs `next() -> Packed`, `state()`, `load(state)`.
"""
from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np
import torch

from lam4.data.buckets import BucketPlan, OversizeStructures, Packed, pack_molecules
from lam4.data.omol import OmolSource


def chunk_order(groups: list[np.ndarray], chunk: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Composition chunks of the parent: each group shuffled and cut into pieces of <= chunk rows (a trailing single row joins the piece before it), pieces in random order.
    Returns (rows in epoch order, piece id of each position)."""
    pieces = []
    for g in groups:
        gp = rng.permutation(g)
        parts = [gp[s:s + chunk] for s in range(0, len(gp), chunk)]
        if len(parts) > 1 and len(parts[-1]) == 1:
            last = parts.pop()
            parts[-1] = np.concatenate([parts[-1], last])
        pieces.extend(parts)
    order = rng.permutation(len(pieces))
    perm = np.concatenate([pieces[i] for i in order]).astype(np.int64)
    pid = np.repeat(np.arange(len(pieces), dtype=np.int64), [len(pieces[i]) for i in order])
    return perm, pid


class PackedStream:
    """Infinite stream of padded molecule batches. chunk > 0: composition-chunk epoch order and `extra["seg"]`."""

    domain = "molecule"

    def __init__(self, src: OmolSource, plans: dict[int, BucketPlan], n_pad: int, seed: int = 0, chunk: int = 0, device="cpu", r_cut: float = 6.0,
                 pair_frac: float = 0.0) -> None:
        self.src, self.plans, self.n_pad, self.chunk, self.device, self.r_cut = src, plans, n_pad, int(chunk), device, r_cut
        self.pair_frac = float(pair_frac)
        self.rng = np.random.default_rng(seed)
        self._pair_rng = np.random.default_rng(seed + 99991)
        self.epoch = 0
        self.skipped = 0   # structures dropped for having more edges than the bucket's e_pad alone
        self.perm: dict[int, np.ndarray] = {}
        self.pid: dict[int, np.ndarray] = {}
        self.ptr: dict[int, int] = {}
        self._rem: dict[int, int] = {}
        self._pairs: dict[int, np.ndarray] = {}
        self.pair_stats: dict = {}
        self._pair_perm: dict[int, np.ndarray] = {}
        self._pair_ptr: dict[int, int] = {}
        if self.pair_frac > 0.0:
            self._pairs, self.pair_stats = self.src.pair_index()
            for c, arr in self._pairs.items():
                if len(arr) > 0:
                    self._pair_perm[c] = self._pair_rng.permutation(len(arr))
                    self._pair_ptr[c] = 0
        self._new_epoch()

    def _recompute_rem(self) -> None:
        self._rem = {c: int(self.src.n_at[self.perm[c][self.ptr[c]:]].sum()) for c in self.plans if c in self.perm}

    def _advance(self, cap: int, count: int) -> None:
        p0 = self.ptr[cap]
        p1 = p0 + count
        self._rem[cap] -= int(self.src.n_at[self.perm[cap][p0:p1]].sum())
        self.ptr[cap] = p1

    def _new_epoch(self) -> None:
        self.epoch += 1
        gid = self.src.group_ids() if self.chunk > 0 else None
        for cap in self.plans:
            rows = self.src.buckets[cap]
            if self.chunk > 0:
                g = gid[rows]
                order = np.argsort(g, kind="stable")
                groups = [rows[o] for o in np.split(order, np.where(np.diff(g[order]) != 0)[0] + 1)]
                self.perm[cap], self.pid[cap] = chunk_order(groups, self.chunk, self.rng)
            else:
                self.perm[cap] = self.rng.permutation(rows)
            self.ptr[cap] = 0
        self._recompute_rem()

    def remaining_atoms(self) -> dict[int, float]:
        return {c: float(self._rem.get(c, 0)) for c in self.plans}

    def _draw_pair_cands(self, cap: int, max_k: int) -> tuple[np.ndarray, np.ndarray]:
        pairs_cap = self._pairs.get(cap)
        if self.pair_frac <= 0.0 or pairs_cap is None or len(pairs_cap) == 0 or max_k < 2:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        cap_n = self.n_pad - 2
        target_n = min(cap_n, int(round(self.pair_frac * cap_n)))
        if target_n <= 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        pos_l: list[int] = []
        pid_l: list[int] = []
        n_est, tok = 0, 0
        perm = self._pair_perm[cap]
        ptr = self._pair_ptr[cap]
        tries, max_tries = 0, min(len(pairs_cap), max(32, max_k))
        while n_est < target_n and (len(pos_l) + 2) <= max_k and tries < max_tries:
            if ptr >= len(perm):
                perm = self._pair_rng.permutation(len(pairs_cap))
                self._pair_perm[cap] = perm
                ptr = 0
            pa, pb = int(pairs_cap[perm[ptr], 0]), int(pairs_cap[perm[ptr], 1])
            ptr += 1
            tries += 1
            na, nb = int(self.src.n_at[pa]), int(self.src.n_at[pb])
            if na != nb or na <= 0 or (na + nb) > cap_n:
                continue
            if n_est + na + nb > target_n and pos_l:
                break
            if n_est + na + nb > cap_n:
                continue
            tok += 1
            pos_l.extend((pa, pb))
            pid_l.extend((tok, tok))
            n_est += na + nb
        self._pair_ptr[cap] = ptr
        return np.asarray(pos_l, dtype=np.int64), np.asarray(pid_l, dtype=np.int64)

    def next(self) -> Packed:
        rem = self.remaining_atoms()
        tot = sum(rem.values())
        if tot < 0.5 * self.n_pad:
            self._new_epoch()
            rem = self.remaining_atoms()
            tot = sum(rem.values())
        caps = list(rem)
        cap = int(self.rng.choice(caps, p=np.array([rem[c] for c in caps]) / tot))
        pl, p0 = self.plans[cap], self.ptr[cap]
        pair_pos, pair_ids_p = self._draw_pair_cands(cap, pl.k)
        n_pair_c = len(pair_pos)
        if n_pair_c > 0:
            rem_k = max(0, pl.k - n_pair_c)
            sing_pos = self.perm[cap][p0:p0 + rem_k]
            cand_pos = np.concatenate([pair_pos, sing_pos]) if len(sing_pos) else pair_pos
            cand_pids = np.concatenate([pair_ids_p, np.zeros(len(sing_pos), dtype=np.int64)])
            if self.chunk > 0:
                sing_pieces = self.pid[cap][p0:p0 + rem_k]
                # Unique negative chunk ids for pair slots so lam_rel stays strictly on singletons.
                pieces = np.concatenate([-(np.arange(1, n_pair_c + 1, dtype=np.int64)), sing_pieces])
            else:
                pieces = None
        else:
            cand_pos = self.perm[cap][p0:p0 + pl.k]
            cand_pids = None
            pieces = self.pid[cap][p0:p0 + pl.k] if self.chunk > 0 else None
        try:
            pk = pack_molecules(self.src, cap, cand_pos, n_pad=self.n_pad, e_pad=pl.e_pad, k=pl.k, r_cut=self.r_cut,
                                device=self.device, pieces=pieces, source=getattr(self.src, "source", None), pair_ids=cand_pids)
        except OversizeStructures as e:   # nothing in the prefix fits e_pad on its own: skip it and draw again
            self.skipped += e.n
            self._advance(cap, max(0, e.n - n_pair_c))
            return self.next()
        self.skipped += pk.skipped
        self._advance(cap, max(0, pk.used + pk.skipped - n_pair_c))
        return pk

    def state(self) -> dict:
        st = {"epoch": self.epoch, "rng": self.rng.bit_generator.state, "ptr": dict(self.ptr), "perm": {c: p.copy() for c, p in self.perm.items()},
              "pid": {c: p.copy() for c, p in self.pid.items()}}
        if self.pair_frac > 0.0 and self._pair_ptr:
            st["pair_rng"] = self._pair_rng.bit_generator.state
            st["pair_ptr"] = dict(self._pair_ptr)
        return st

    def load(self, st: dict) -> None:
        self.epoch, self.ptr = st["epoch"], dict(st["ptr"])
        self.rng.bit_generator.state = st["rng"]
        for c in self.plans:
            if c in st["perm"]:
                self.perm[c] = np.asarray(st["perm"][c])
                if self.chunk > 0:
                    self.pid[c] = np.asarray(st["pid"][c])
        if self.pair_frac > 0.0 and "pair_rng" in st:
            self._pair_rng.bit_generator.state = st["pair_rng"]
            for c, ptr in (st.get("pair_ptr") or {}).items():
                ci = int(c)
                if ci in self._pairs and len(self._pairs[ci]) > 0:
                    self._pair_ptr[ci] = int(ptr) % len(self._pairs[ci])
        self._recompute_rem()


class DomainSampler:
    """One domain-homogeneous batch per step. Each step takes the domain with the lowest visited_atoms / share (deterministic deficit scheduling, so the long-run atom
    shares equal `shares`); a domain without a stream is dropped and the shares renormalise."""

    def __init__(self, streams: dict[str, object], shares: dict[str, float] | None = None) -> None:
        shares = shares or {"molecule": 0.92, "crystal": 0.08}
        live = {d: shares[d] for d in streams if shares.get(d, 0.0) > 0}
        tot = sum(live.values())
        self.streams, self.shares = streams, {d: s / tot for d, s in live.items()}
        self.device = next((s.device for s in streams.values() if hasattr(s, "device")), "cpu")   # the Prefetcher side stream
        self.visited = {d: 0.0 for d in self.shares}

    def next(self) -> Packed:
        d = min(self.shares, key=lambda k: self.visited[k] / self.shares[k])
        return self.next_domain(d)

    def next_domain(self, domain: str) -> Packed:
        pk = self.streams[domain].next()
        self.visited[domain] += pk.n_real
        return pk

    def next_pair(self) -> tuple[Packed, Packed]:
        mol_dom = "molecule" if "molecule" in self.streams else next(iter(self.streams))
        per_dom = "crystal" if "crystal" in self.streams else mol_dom
        return self.next_domain(mol_dom), self.next_domain(per_dom)

    def state(self) -> dict:
        return {"visited": dict(self.visited), "shares": dict(self.shares), "streams": {d: s.state() for d, s in self.streams.items()}}

    def load(self, st: dict) -> None:
        old = st.get("shares") or {}
        same = set(old) == set(self.shares) and all(abs(old[d] - self.shares[d]) < 1e-9 for d in self.shares)
        if same:
            self.visited.update(st["visited"])
        else:   # shares changed on resume: the restored deficit would serve one domain only until it caught up, so restart the counts AT the new shares
            tot = sum(st["visited"].values())
            self.visited = {d: tot * s for d, s in self.shares.items()}
        for d, s in self.streams.items():
            if d in st["streams"]:
                s.load(st["streams"][d])


def _pin_to_device(pk: Packed, dev: torch.device, pin_memory: bool) -> Packed:
    if dev.type != "cuda":
        return pk
    from dataclasses import fields, replace
    b_updates = {}
    for f in fields(pk.batch):
        v = getattr(pk.batch, f.name)
        if torch.is_tensor(v) and v.device != dev:
            t = v.pin_memory() if (pin_memory and v.device.type == "cpu") else v
            b_updates[f.name] = t.to(dev, non_blocking=True)
    ex_updates = {
        k: ((v.pin_memory() if (pin_memory and v.device.type == "cpu") else v).to(dev, non_blocking=True) if (torch.is_tensor(v) and v.device != dev) else v)
        for k, v in pk.extra.items()
    }
    return replace(pk, batch=(replace(pk.batch, **b_updates) if b_updates else pk.batch), extra=ex_updates)


class Prefetcher:
    """Builds batches in a worker thread (on a side CUDA stream with pinned H2D transfer); `depth` (default 4) batches are kept ready.
    Supports single (`next()`) or paired (`paired=True`, `next_pair()`) prefetch plus optional background `csr_workers` plan prep."""

    def __init__(self, stream, depth: int = 4, *, paired: bool = False, pin_memory: bool = True, csr_workers: int = 0) -> None:
        self.stream = stream
        self.wait_s = 0.0
        self.depth = int(os.environ.get("LAM4_PREFETCH_DEPTH", depth))
        self.paired, self.pin_memory, self.csr_workers = bool(paired), bool(pin_memory), max(0, int(csr_workers))
        self.q: queue.Queue = queue.Queue(maxsize=max(1, self.depth))
        dev = getattr(stream, "device", "cpu")
        self.target_device = torch.device(dev)
        self.cuda = torch.cuda.is_available() and self.target_device.type == "cuda"
        if self.cuda and self.target_device.index is None:
            self.target_device = torch.device("cuda", torch.cuda.current_device())
        self.side = torch.cuda.Stream(self.target_device) if self.cuda else None
        self.err: BaseException | None = None
        self.lock = threading.Lock()   # stream.next() in the worker vs state() / load() in the training thread
        self._csr_pool = None
        if self.csr_workers > 0:
            from concurrent.futures import ThreadPoolExecutor
            self._csr_pool = ThreadPoolExecutor(max_workers=self.csr_workers)
        self._start()

    def _start(self) -> None:
        self.stop = False
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()

    def _prep_csr(self, pk: Packed) -> None:
        if self._csr_pool is None:
            return
        from lam4.data.padding import active_edge_count
        from lam4.kernels.csr_plan import build_csr_plan_cpu
        b = pk.batch
        if b.pos.device.type == "cpu":
            ei = b.edge_index
            pk._csr_plan = self._csr_pool.submit(build_csr_plan_cpu, ei[1], ei[0], b.n_atoms, active_edges=active_edge_count(b)).result()  # type: ignore[attr-defined]

    def _fetch_one(self):
        if self.paired and hasattr(self.stream, "next_pair"):
            if self.cuda:
                with torch.cuda.stream(self.side):
                    pm, pp = self.stream.next_pair()
                    self._prep_csr(pm)
                    self._prep_csr(pp)
                    pm = _pin_to_device(pm, self.target_device, self.pin_memory)
                    pp = _pin_to_device(pp, self.target_device, self.pin_memory)
                    ev = torch.cuda.Event()
                    ev.record(self.side)
                return (pm, pp), ev
            pm, pp = self.stream.next_pair()
            self._prep_csr(pm)
            self._prep_csr(pp)
            return (pm, pp), None
        if self.cuda:
            with torch.cuda.stream(self.side):
                pk = self.stream.next()
                self._prep_csr(pk)
                pk = _pin_to_device(pk, self.target_device, self.pin_memory)
                ev = torch.cuda.Event()
                ev.record(self.side)
            return pk, ev
        pk = self.stream.next()
        self._prep_csr(pk)
        return pk, None

    def _work(self) -> None:
        try:
            while not self.stop:
                with self.lock:
                    item, ev = self._fetch_one()
                while not self.stop:
                    try:
                        self.q.put((item, ev), timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:   # surfaced in next()
            self.err = e
            self.q.put((None, None))

    def _sync_item(self, item, ev):
        if ev is not None:
            cur = torch.cuda.current_stream()
            cur.wait_event(ev)
            pks = item if isinstance(item, tuple) else (item,)
            for pk in pks:
                for t in [*vars(pk.batch).values(), *pk.extra.values()]:
                    if torch.is_tensor(t):
                        t.record_stream(cur)
        return item

    def next(self) -> Packed:
        t0 = time.perf_counter()
        item, ev = self.q.get()
        self.wait_s += time.perf_counter() - t0   # time the training thread was blocked on the packer (logged per window as feed_wait_ms)
        if item is None:
            raise RuntimeError(f"batch worker died: {self.err!r}")
        item = self._sync_item(item, ev)
        return item[0] if isinstance(item, tuple) else item

    def next_pair(self) -> tuple[Packed, Packed]:
        if self.paired:
            t0 = time.perf_counter()
            item, ev = self.q.get()
            self.wait_s += time.perf_counter() - t0
            if item is None:
                raise RuntimeError(f"batch worker died: {self.err!r}")
            return self._sync_item(item, ev)
        with self.lock:
            pm, pp = self.stream.next_pair()
        if self.cuda:
            pm = _pin_to_device(pm, self.target_device, self.pin_memory)
            pp = _pin_to_device(pp, self.target_device, self.pin_memory)
        return pm, pp

    def state(self) -> dict:
        with self.lock:
            return self.stream.state()

    def load(self, st: dict) -> None:
        """Resume: stop the worker (full join: it may be inside a part switch), drop the batches it built from the pre-resume position, load the state, restart."""
        self.stop = True
        self.t.join()
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        with self.lock:
            self.stream.load(st)
        self._start()

    def close(self) -> None:
        self.stop = True
        self.t.join(timeout=5.0)
        if self._csr_pool is not None:
            self._csr_pool.shutdown(wait=False)
