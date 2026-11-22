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
from pathlib import Path
from typing import Callable

import numpy as np

from lam4.data.buckets import BucketPlan, Packed, calibrate_plans, exact_batches
from lam4.data.omol import OmolSource, resolve_omol_dir_source
from lam4.data.sampler import PackedStream


class PartStream:
    domain = "molecule"

    def __init__(self, dirs: list[str], load: Callable[[str], OmolSource], plans: dict[int, BucketPlan], n_pad: int, seed: int = 0, first: OmolSource | None = None,
                 log: Callable[[str], None] = print, k0: int = 0, chunk: int = 0, device="cpu", pair_frac: float = 0.0) -> None:
        """k0 > 0 (resume): start at part k0, `first` must then be dirs[k0 % len(dirs)] (the parts before it are never read)."""
        self.dirs, self.loader, self.plans, self.n_pad, self.seed, self.log, self.chunk, self.device = dirs, load, plans, n_pad, seed, log, chunk, device
        self.pair_frac = float(pair_frac)
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
        if self.pair_frac > 0.0:
            src.pair_index()
        return src

    def _stream(self) -> PackedStream:
        plans = {c: p for c, p in self.plans.items() if c in self.src.buckets}
        st = PackedStream(self.src, plans, self.n_pad, self.seed + self.k, self.chunk, device=self.device, pair_frac=self.pair_frac)
        if st.pair_stats:
            self.log(f"mol part {self.part}: close_pairs={st.pair_stats['n_pairs']:,} by_cap={st.pair_stats.get('by_cap', {})} ({'cached' if st.pair_stats.get('cached') else str(st.pair_stats.get('build_s', 0)) + 's'})")
        return st

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
        if hasattr(self.src, "reload_sidecars") and self.src.reload_sidecars():
            self.log(f"reloaded sidecars for part {self.k} ({self.part})")
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
        if "stream" not in st and isinstance(st.get("streams"), dict):
            om = st["streams"].get("omol_wb97mv")
            if not (isinstance(om, dict) and "stream" in om):
                om = next((v for v in st["streams"].values() if isinstance(v, dict) and "stream" in v), st)
            st = om
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


_CANONICAL_MOL_SOURCE_SHARES = {"omol_wb97mv": 0.90, "spice_wb97md3": 0.10}


class MultiSourceMolStream:
    """Simultaneous deficit-balanced stream across multiple molecular `source` LoTs (same atom-visit scheduler as DomainSampler)."""

    domain = "molecule"

    def __init__(self, streams: dict[str, PartStream | PackedStream], shares: dict[str, float] | None = None, device="cpu") -> None:
        if not streams:
            raise ValueError("MultiSourceMolStream requires at least one stream")
        self.streams = dict(streams)
        self.device = device
        if shares is None:
            w = 1.0 / len(self.streams)
            self.shares = {k: w for k in self.streams}
        else:
            tot = sum(float(shares[k]) for k in self.streams) or 1.0
            self.shares = {k: float(shares[k]) / tot for k in self.streams}
        self.visited = {k: 0.0 for k in self.streams}

    def next(self) -> Packed:
        src = min(self.shares, key=lambda k: self.visited[k] / self.shares[k])
        pk = self.streams[src].next()
        self.visited[src] += max(1, int(pk.n_real))
        return pk

    def state(self) -> dict:
        st_streams = {k: s.state() for k, s in self.streams.items()}
        om_st = st_streams.get("omol_wb97mv")
        part_k = int(om_st["k"]) if (isinstance(om_st, dict) and "k" in om_st) else next(
            (int(v["k"]) for v in st_streams.values() if isinstance(v, dict) and "k" in v), 0
        )
        return {
            "k": part_k,
            "shares": dict(self.shares),
            "visited": dict(self.visited),
            "streams": st_streams,
        }

    def load(self, st: dict) -> None:
        if "streams" not in st and "stream" in st:
            target = "omol_wb97mv" if "omol_wb97mv" in self.streams else next(iter(self.streams))
            self.streams[target].load(st)
            return
        old = st.get("shares") or {}
        same = set(old) == set(self.shares) and all(abs(old[k] - self.shares[k]) < 1e-9 for k in self.shares)
        if same:
            self.visited.update(st["visited"])
        else:
            tot = sum(st["visited"].values())
            self.visited = {k: tot * s for k, s in self.shares.items()}
        for k, s in self.streams.items():
            if k in st.get("streams", {}):
                s.load(st["streams"][k])


def build_mol_stream(
    dirs: list[str],
    dir_sources: dict[str, str],
    load_fn: Callable[[str, str], OmolSource],
    plans: dict[int, BucketPlan],
    n_pad: int,
    seed: int = 0,
    *,
    first: OmolSource | None = None,
    k0: int = 0,
    chunk: int = 0,
    device="cpu",
    shares: dict[str, float] | None = None,
    log: Callable[[str], None] = print,
    pair_frac: float = 0.0,
):
    """Build single PackedStream/PartStream (1 LoT) or simultaneous MultiSourceMolStream (2+ LoTs)."""
    by_source: dict[str, list[str]] = {}
    for d in dirs:
        by_source.setdefault(dir_sources[d], []).append(d)
    primary_src = "omol_wb97mv" if "omol_wb97mv" in by_source else next(iter(by_source))
    sub: dict[str, PartStream | PackedStream] = {}
    for idx_s, (src_name, s_dirs) in enumerate(by_source.items()):
        src_k0 = k0 if (src_name == primary_src and len(s_dirs) > 1) else 0
        start_dir = s_dirs[src_k0 % len(s_dirs)]
        first_ok = (first is not None and Path(first.flat.path) == Path(start_dir))
        s_first = first if first_ok else load_fn(start_dir, src_name)
        if any(c not in plans for c in s_first.buckets):
            for c, p in calibrate_plans(s_first, n_pad, seed).items():
                plans.setdefault(c, p)
        if len(s_dirs) > 1:
            sub[src_name] = PartStream(
                s_dirs, lambda d, s=src_name: load_fn(d, s), plans, n_pad, seed + idx_s,
                first=s_first, log=log, k0=src_k0, chunk=chunk, device=device, pair_frac=pair_frac,
            )
        else:
            s_plans = {c: p for c, p in plans.items() if c in s_first.buckets}
            ps = PackedStream(s_first, s_plans, n_pad, seed + idx_s, chunk, device=device, pair_frac=pair_frac)
            if ps.pair_stats:
                log(f"mol source {src_name} ({start_dir}): close_pairs={ps.pair_stats['n_pairs']:,} by_cap={ps.pair_stats.get('by_cap', {})} ({'cached' if ps.pair_stats.get('cached') else str(ps.pair_stats.get('build_s', 0)) + 's'})")
            sub[src_name] = ps
    if len(sub) == 1:
        return next(iter(sub.values()))
    if shares is None and "omol_wb97mv" in sub and "spice_wb97md3" in sub:
        shares = {k: _CANONICAL_MOL_SOURCE_SHARES.get(k, 0.10) for k in sub}
    return MultiSourceMolStream(sub, shares=shares, device=device)


def build_omol_val_batches(
    val_data: str,
    *,
    val_split: str | Path | None = None,
    tier1_split: str | Path | None = None,
    val_max: int = 0,
    seed: int = 0,
    max_atoms: int = 6000,
    d3_compute: bool = True,
    d3_cache: str | Path | None = None,
    device="cpu",
) -> list[Packed]:
    """Load exact validation batches across one or more `path[@source]` specs using each pool's true LoT."""
    if not val_data:
        return []
    split_p = Path(val_split) if val_split else None
    split_arr = np.load(split_p) if (split_p is not None and split_p.exists()) else None
    is_tier1 = bool(split_p is not None and tier1_split is not None and split_p.resolve() == Path(tier1_split).resolve())
    batches: list[Packed] = []
    for spec in [s.strip() for s in str(val_data).split(",") if s.strip()]:
        vdir, vsrc = resolve_omol_dir_source(spec)
        if not Path(vdir).exists():
            continue
        idx = split_arr if (split_arr is not None and (not is_tier1 or Path(vdir).name == "val_sub5k")) else None
        if val_max and idx is not None:
            idx = idx[:val_max]
        src = OmolSource(
            vdir, source=vsrc, indices=idx, max_structs=val_max if idx is None else 0,
            seed=seed, d3_compute=d3_compute, d3_cache=d3_cache or None,
        )
        batches.extend(pk for pk, _ in exact_batches(src, max_atoms=max_atoms, device=device))
    return batches
