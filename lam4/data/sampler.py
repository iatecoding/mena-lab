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

    def __init__(self, src: OmolSource, plans: dict[int, BucketPlan], n_pad: int, seed: int = 0, chunk: int = 0, device="cpu", r_cut: float = 6.0) -> None:
        self.src, self.plans, self.n_pad, self.chunk, self.device, self.r_cut = src, plans, n_pad, int(chunk), device, r_cut
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.skipped = 0   # structures dropped for having more edges than the bucket's e_pad alone
        self.perm: dict[int, np.ndarray] = {}
        self.pid: dict[int, np.ndarray] = {}
        self.ptr: dict[int, int] = {}
        self._new_epoch()

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

    def remaining_atoms(self) -> dict[int, float]:
        return {c: float(self.src.n_at[self.perm[c][self.ptr[c]:]].sum()) for c in self.plans}

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
        pieces = self.pid[cap][p0:p0 + pl.k] if self.chunk > 0 else None
        try:
            pk = pack_molecules(self.src, cap, self.perm[cap][p0:p0 + pl.k], n_pad=self.n_pad, e_pad=pl.e_pad, k=pl.k, r_cut=self.r_cut, device=self.device, pieces=pieces)
        except OversizeStructures as e:   # nothing in the prefix fits e_pad on its own: skip it and draw again
            self.skipped += e.n
            self.ptr[cap] = p0 + e.n
            return self.next()
        self.skipped += pk.skipped
        self.ptr[cap] = p0 + pk.used + pk.skipped
        return pk

    def state(self) -> dict:
        return {"epoch": self.epoch, "rng": self.rng.bit_generator.state, "ptr": dict(self.ptr), "perm": {c: p.copy() for c, p in self.perm.items()},
                "pid": {c: p.copy() for c, p in self.pid.items()}}

    def load(self, st: dict) -> None:
        self.epoch, self.ptr = st["epoch"], dict(st["ptr"])
        self.rng.bit_generator.state = st["rng"]
        for c in self.plans:
            if c in st["perm"]:
                self.perm[c] = np.asarray(st["perm"][c])
                if self.chunk > 0:
                    self.pid[c] = np.asarray(st["pid"][c])


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
        pk = self.streams[d].next()
        self.visited[d] += pk.n_real
        return pk

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


class Prefetcher:
    """Builds batches in a worker thread (on a side CUDA stream when the batches live on a GPU); `depth` batches are kept ready. Stateful streams: `state()` of the
    stream runs ahead of training by up to depth + 1 batches, so a resume skips them (harmless for an epoch sampler). state() and the worker's next() share a lock
    (no torn state in a checkpoint); load() restarts the worker so no pre-resume batch is served."""

    def __init__(self, stream, depth: int = 3) -> None:
        self.stream = stream
        self.wait_s = 0.0
        self.q: queue.Queue = queue.Queue(maxsize=int(os.environ.get("LAM4_PREFETCH_DEPTH", depth)))
        dev = getattr(stream, "device", "cpu")
        self.cuda = torch.cuda.is_available() and torch.device(dev).type == "cuda"
        self.side = torch.cuda.Stream(torch.device(dev)) if self.cuda else None
        self.err: BaseException | None = None
        self.lock = threading.Lock()   # stream.next() in the worker vs state() / load() in the training thread
        self._start()

    def _start(self) -> None:
        self.stop = False
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()

    def _work(self) -> None:
        try:
            while not self.stop:
                with self.lock:
                    if self.cuda:
                        with torch.cuda.stream(self.side):
                            pk = self.stream.next()
                            ev = torch.cuda.Event()
                            ev.record(self.side)
                    else:
                        pk, ev = self.stream.next(), None
                while not self.stop:
                    try:
                        self.q.put((pk, ev), timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:   # surfaced in next()
            self.err = e
            self.q.put((None, None))

    def next(self) -> Packed:
        t0 = time.perf_counter()
        pk, ev = self.q.get()
        self.wait_s += time.perf_counter() - t0   # time the training thread was blocked on the packer (logged per window as feed_wait_ms)
        if pk is None:
            raise RuntimeError(f"batch worker died: {self.err!r}")
        if ev is not None:
            cur = torch.cuda.current_stream()
            cur.wait_event(ev)
            for t in [*vars(pk.batch).values(), *pk.extra.values()]:
                if torch.is_tensor(t):
                    t.record_stream(cur)
        return pk

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
