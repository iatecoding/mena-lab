"""parts.py: training stream over several OMol flat parts (one in host memory at a time, the next prepared in a thread).

Each part is one full shuffled pass (a `PackedStream` epoch), then the stream switches to the next part (cyclic). The bucket plans (graph slots, E_PAD) are calibrated on one part
and reused, so the compiled graphs never change; `pack_molecules` keeps the prefix of a batch that fits N_PAD / E_PAD, so a part with denser structures stays valid.
Adapted from the earlier Mena training pipeline (PartStream): same switch / retry / resume logic on the lam4 `PackedStream` (which carries its epoch counter instead of
a field in the batch). The part loader runs `OmolSource` (D3 labels from `<part>/d3_<sig>.pt`, `$OMOL_D3_CACHE_DIR/<part>_d3_<sig>.pt` or `.../d3cache_v2`: never recomputed
when a cache exists) and the composition groups in the thread, so a switch only waits for the thread. Stream interface: `next() -> Packed`, `state()`, `load(state)`.
"""
from __future__ import annotations

import gc
import threading
import time
from typing import Callable

from lam4.data.buckets import BucketPlan, Packed
from lam4.data.omol import OmolSource
from lam4.data.sampler import PackedStream


class PartStream:
    domain = "molecule"

    def __init__(self, dirs: list[str], load: Callable[[str], OmolSource], plans: dict[int, BucketPlan], n_pad: int, seed: int = 0, first: OmolSource | None = None,
                 log: Callable[[str], None] = print, k0: int = 0, chunk: int = 0, device="cpu") -> None:
        """k0 > 0 (resume): start at part k0, `first` must then be dirs[k0 % len(dirs)] (the parts before it are never read)."""
        self.dirs, self.loader, self.plans, self.n_pad, self.seed, self.log, self.chunk, self.device = dirs, load, plans, n_pad, seed, log, chunk, device
        self.k = k0   # parts started so far - 1 (the current part is dirs[k % len(dirs)])
        self.src = first if first is not None else self._load(self.part)
        self.stream = self._stream()
        self._next: OmolSource | None = None
        self._err: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._prepare(self.k + 1)

    @property
    def part(self) -> str:
        return self.dirs[self.k % len(self.dirs)]

    @property
    def epoch(self) -> float:
        """Passes over all parts (fractional within a cycle)."""
        return self.k / len(self.dirs)

    def _load(self, d: str) -> OmolSource:
        src = self.loader(d)
        if self.chunk > 0:
            src.group_ids()
        return src

    def _stream(self) -> PackedStream:
        plans = {c: p for c, p in self.plans.items() if c in self.src.buckets}
        return PackedStream(self.src, plans, self.n_pad, self.seed + self.k, self.chunk, device=self.device)

    def _prepare(self, k: int) -> None:
        if len(self.dirs) == 1:
            return
        self._next, self._err = None, None

        def work() -> None:
            d = self.dirs[k % len(self.dirs)]
            for attempt in range(1, 21):
                try:
                    t0 = time.time()
                    self._next, self._err = self._load(d), None
                    self.log(f"part {k} ({d}) prepared in {time.time() - t0:.0f}s")
                    return
                except BaseException as e:  # noqa: BLE001 - a partial FUSE copy: retry, surface after the retries
                    self._err = e
                    self.log(f"[prepare retry {attempt}/20] part {k} ({d}) load error: {e!r}; retrying in 30s")
                    time.sleep(30)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def _switch(self) -> None:
        t0 = time.time()
        self._thread.join()
        if self._err is not None or self._next is None:
            d = self.dirs[(self.k + 1) % len(self.dirs)]
            self.log(f"[switch fallback] synchronous load of part {self.k + 1} ({d}) after err={self._err!r}")
            self._next, self._err = self._load(d), None
        self.k += 1
        self.src, self._next = self._next, None
        self.stream = self._stream()
        self.log(f"part switch -> {self.k} ({self.part}), waited {time.time() - t0:.0f}s")
        self._prepare(self.k + 1)

    def next(self) -> Packed:
        pk = self.stream.next()
        if len(self.dirs) > 1 and self.stream.epoch > 1:   # the current part is used up: the batch just drawn opened its 2nd pass
            self._switch()
            pk = self.stream.next()
        return pk

    def state(self) -> dict:
        return {"k": self.k, "stream": self.stream.state()}

    def load(self, st: dict) -> None:
        """Resume: jump to part st['k'] (loaded synchronously when it is not the current one) and restore its stream position."""
        if st["k"] != self.k:
            if self._thread is not None:
                self._thread.join()
            self.k = st["k"]
            self.src = self._next = self.stream = None   # free host memory before loading part k
            gc.collect()
            self.src = self._load(self.part)
            self.stream = self._stream()
            self._prepare(self.k + 1)
        self.stream.load(st["stream"])
