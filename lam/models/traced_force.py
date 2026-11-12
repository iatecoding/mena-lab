"""traced_force.py: DPA4-style compiled conservative crystal training (DPA4 paper §4.4; `papers/DPA4_2606.02419.txt`).

The crystal step's double backward (force/stress loss -> parameters) costs 2.9x a first-order step in eager mode.
Instead of compiling the model (dynamo cannot follow `autograd.grad(create_graph=True)`), trace the whole function
params -> (energy, forces, stress) with `make_fx`: the coordinate derivative runs DURING tracing, so one flat aten
graph holds the energy and its first derivative, with every parameter as an explicit graph input. Inductor compiles
that graph and AOTAutograd derives the outer backward (loss -> parameters) as an ordinary first-order backward.

Shapes are static per padded batch (`pad_crystal_batch` with fixed N/B and E rounded up to `e_mult`): one graph per
distinct E, ~40 s to compile each on an L4 GPU. Measured there: 378 -> 146 ms per 14.3k-atom / 885k-edge crystal step; the
gradient check must use trained weights (the detach-pair bug fixed in `strip_autograd_detach_chains` is invisible at
fresh init).

Molecules (`TracedMolForces`): the dense 17-bucket path, B padded per bucket to the stream's full batch size with
empty, loss-neutral molecules (`pad_mol_batch`), so one graph per bucket. Measured: ~2x per molecular step, gradient
rel 4-8e-5 on trained weights.

Start-up cost: a run compiles ~7-10 crystal graphs + 17 molecular graphs;
the compile is host-CPU bound. `trace_mode="fake"` traces with FakeTensors (no real forward+create_graph pass on the
GPU, so no eager-sized memory spike at trace time; a data-dependent Python branch raises instead of being silently
baked in). `warmup_shapes` style pre-compilation lives in the trainer (`--traced-warmup`).
"""
from __future__ import annotations

import time

import torch
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils.stateless import _reparametrize_module

CRYSTAL_KEYS = ("z", "pos", "cell", "batch", "edge_index", "edge_shift", "u", "lot_id", "edge_shift_cart")
MOL_KEYS = ("z", "pos", "mask", "charge", "spin")
MOL_SPARSE_KEYS = ("z", "pos", "mask", "charge", "spin", "edge_index", "edge_shift")
MOL_E_MULT = 8192
# per-molecule fields of a PartStream batch and the fill of a padded (empty) molecule: NaN labels drop out of every
# molecular loss term (isfinite masks), mask False / n_atoms 0 make the model see no atoms
_MOL_PAD_FILL = {"z": 0, "mask": False, "pos": 0.0, "forces": float("nan"), "y_f_resid": float("nan"),
                 "energy": float("nan"), "y_e_resid": float("nan"), "total_charge": 0, "charge": 0, "spin": 1,
                 "domain": 0, "n_atoms": 0, "dipole": float("nan"), "subdomain": 0, "source": 0, "q_ref": 0.0,
                 "seg": -1, "batch_seg": -1}


def pad_mol_batch(
    bm: dict,
    b_pad: int,
    *,
    sparse: bool = False,
    r_cut: float = 6.0,
    e_mult: int = MOL_E_MULT,
    e_min: int = 0,
) -> dict:
    """Append `b_pad - B` empty molecules to a dense molecular batch (every per-molecule tensor field).
    When `sparse=True`, also builds `edge_index` and `edge_shift` padded to `crystal_e_pad(e_real, e_mult, e_min)`."""
    B = int(bm["z"].shape[0])
    if b_pad > B:
        out = dict(bm)
        for k, fill in _MOL_PAD_FILL.items():
            v = bm.get(k)
            if not torch.is_tensor(v) or v.dim() == 0 or v.shape[0] != B:
                continue
            out[k] = torch.cat([v, torch.full((b_pad - B, *v.shape[1:]), fill, dtype=v.dtype, device=v.device)], 0)
        if "seg" in out and "batch_seg" in bm and bm["batch_seg"] is bm.get("seg"):
            out["batch_seg"] = out["seg"]
    else:
        out = dict(bm) if sparse else bm
    if sparse:
        out = pad_mol_sparse_batch(out, r_cut=r_cut, e_mult=e_mult, e_min=e_min)
    return out


def pad_mol_sparse_batch(
    bm: dict,
    b_pad: int = 0,
    *,
    r_cut: float = 6.0,
    e_mult: int = MOL_E_MULT,
    e_min: int = 0,
) -> dict:
    """Build static-padded short-range `edge_index` (`[2, e_pad]`) and `edge_shift` (`[e_pad, 3]`) for a molecular
    bucket batch using `crystal_e_pad(e_real, e_mult, e_min)` with ghost self-loops on `ghost_atom = B * cap`."""
    if b_pad > int(bm["z"].shape[0]):
        bm = pad_mol_batch(bm, b_pad, sparse=False)
    out = dict(bm)
    z = out["z"]
    pos = out["pos"]
    mask = out["mask"] if "mask" in out else (z > 0)
    out["mask"] = mask
    if "charge" not in out and "total_charge" in out:
        out["charge"] = out["total_charge"]
    B, N = int(z.shape[0]), int(z.shape[1])
    dev = z.device
    ghost_atom = B * N
    with torch.no_grad():
        eye = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
        pm = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye
        d = torch.cdist(pos.detach(), pos.detach())
        b_sr, i_sr, j_sr = (pm & (d < float(r_cut) + 1e-3)).nonzero(as_tuple=True)
        r_sr = b_sr * N + i_sr
        s_sr = b_sr * N + j_sr
        e_sr = int(r_sr.numel())
        e_pad = crystal_e_pad(e_sr, e_mult, e_min) if e_mult > 0 else max(e_sr, 1)
        pad_sr = e_pad - e_sr
        edge_index = torch.stack(
            [torch.cat([r_sr, torch.full((pad_sr,), ghost_atom, device=dev, dtype=torch.long)]),
             torch.cat([s_sr, torch.full((pad_sr,), ghost_atom, device=dev, dtype=torch.long)])],
            dim=0,
        )
        edge_shift = pos.new_zeros((e_pad, 3))
        if pad_sr > 0:
            edge_shift[e_sr:, 0] = 100.0
    out["edge_index"] = edge_index
    out["edge_shift"] = edge_shift
    return out


def crystal_e_pad(e_real: int, e_mult: int, e_min: int = 0) -> int:
    """Padded edge count of a crystal batch: the next multiple of `e_mult` strictly above `e_real` (room for the
    ghost edges), floored at `e_min` (rare small batches then reuse the smallest pre-compiled graph)."""
    return max(-(-(int(e_real) + 1) // int(e_mult)) * int(e_mult), int(e_min))


def crystal_e_ladder(e_lo: int, e_hi: int, e_mult: int) -> list[int]:
    """Every padded edge count in [e_lo, e_hi] (multiples of e_mult): the crystal graphs a warm-up compiles."""
    lo = -(-int(e_lo) // int(e_mult)) * int(e_mult)
    return list(range(lo, int(e_hi) + 1, int(e_mult)))


_CRYSTAL_ATOM_KEYS = ("z", "pos", "forces", "batch")
_CRYSTAL_EDGE_KEYS = ("edge_shift", "edge_shift_cart")


def slice_crystal_batch(bt: dict, n_graphs: int) -> dict:
    """The first `n_graphs` structures of a collated (unpadded) crystal batch. Atoms of a graph are contiguous and
    every edge stays inside its graph, so the sub-batch is a valid batch (used to build warm-up batches below an
    edge-count target). Per-graph tensors are those whose first dim is B; other non-tensor entries are kept."""
    B = int(bt["cell"].shape[0])
    N = int(bt["z"].shape[0])
    n_graphs = max(1, min(int(n_graphs), B))
    keep_atom = bt["batch"] < n_graphs
    n_keep = int(keep_atom.sum())
    keep_edge = bt["edge_index"][0] < n_keep
    out = {}
    for k, v in bt.items():
        if not torch.is_tensor(v):
            out[k] = v
        elif k == "edge_index":
            out[k] = v[:, keep_edge]
        elif k in _CRYSTAL_EDGE_KEYS:
            out[k] = v[keep_edge]
        elif k in _CRYSTAL_ATOM_KEYS or (v.dim() > 0 and v.shape[0] == N and N != B):
            out[k] = v[:n_keep]
        elif v.dim() > 0 and v.shape[0] == B:
            out[k] = v[:n_graphs]
        else:
            out[k] = v
    return out


def strip_autograd_detach_chains(gm: torch.fx.GraphModule) -> int:
    """DPA4 §4.4: collapse every autograd-inserted `detach(detach(x))` to `x`, then drop detaches left without users.
    An op whose backward reads its own saved output (rsqrt in RMSNorm, exp, sqrt, tanh, ...) is recorded under
    `autograd.grad(create_graph=True)` as detach(out) at save time and detach(detach(out)) when the backward reads it;
    replayed, that pair cuts the double-backward path the eager graph keeps (force-loss parameter gradients off by up
    to 16% with trained weights). Removing only
    one detach of the pair (the first version of this function) leaves the cut in place. The model's own single
    `.detach()` calls are kept."""
    det = torch.ops.aten.detach.default
    removed = 0
    for n in list(gm.graph.nodes):
        if n.op == "call_function" and n.target is det:
            src = n.args[0]
            if isinstance(src, torch.fx.Node) and src.op == "call_function" and src.target is det:
                n.replace_all_uses_with(src.args[0])
                gm.graph.erase_node(n)
                removed += 1
    for n in list(gm.graph.nodes):
        if n.op == "call_function" and n.target is det and not n.users:
            gm.graph.erase_node(n)
            removed += 1
    gm.graph.lint()
    gm.recompile()
    return removed


class TracedCrystalForces:
    """Callable b -> {"energy", "forces", "stress"} (internal kcal units, same as `MenaLAM.forward_crystal`), one
    traced + compiled graph per batch shape. Crystal LoTs never run electrostatics (`use_elec_context=False`)."""

    def __init__(self, model: torch.nn.Module, compile_mode: str = "default", log=print, trace_mode: str = "real"):
        if trace_mode not in ("real", "fake"):
            raise ValueError(f"trace_mode must be 'real' or 'fake', got {trace_mode!r}")
        self.model = model
        self.compile_mode = compile_mode
        self.log = log
        self.trace_mode = trace_mode
        self.graphs: dict[tuple, object] = {}
        self.first_call_s: dict[tuple, float] = {}  # trace + Inductor compile + first run, per graph key

    def _trace(self, fn, pv: list[Tensor], bv: list[Tensor], xs: list[Tensor]) -> torch.fx.GraphModule:
        if self.trace_mode == "fake":  # module-level constant tensors (lam/models caches) stay real constants
            return make_fx(fn, tracing_mode="fake", _allow_non_fake_inputs=True)(pv, bv, *xs)
        return make_fx(fn, tracing_mode="real")(pv, bv, *xs)

    def _run(self, key: tuple, b: dict, keys: tuple[str, ...]):
        fn = self.graphs.get(key)
        new = fn is None
        if new:
            t0 = time.perf_counter()
            fn = self.graphs[key] = self._build(b, key)
        _, pv, _, bv = self._state()
        out = fn(pv, bv, *[b[k] for k in keys])
        if new:
            dev = out[0].device
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            self.first_call_s[key] = time.perf_counter() - t0
            self.log(f"[traced-force] graph {key} ready: trace + forward compile + first run "
                     f"{self.first_call_s[key]:.1f}s (backward compiles at its first call)")
        return out

    def _state(self) -> tuple[list[str], list[Tensor], list[str], list[Tensor]]:
        pn, pv = zip(*self.model.named_parameters())
        bn, bv = zip(*self.model.named_buffers())
        return list(pn), list(pv), list(bn), list(bv)

    def _build(self, b: dict, key: tuple):
        pn, pv, bn, bv = self._state()
        model = self.model

        def energy_forces_stress(params: list[Tensor], buffers: list[Tensor], *xs: Tensor):
            d = dict(zip(CRYSTAL_KEYS, xs))
            with _reparametrize_module(model, {**dict(zip(pn, params)), **dict(zip(bn, buffers))}):
                out = model.forward_crystal(
                    d["z"], d["pos"], d["cell"], d["batch"], d["edge_index"], d["edge_shift"], u=d["u"],
                    lot_id=d["lot_id"], virial_mode="edge", edge_shift_cart=d["edge_shift_cart"], create_graph=True,
                    use_elec_context=False,
                )
            return out["energy"], out["forces"], out["stress"]

        t0 = time.perf_counter()
        gm = self._trace(energy_forces_stress, pv, bv, [b[k] for k in CRYSTAL_KEYS])
        n_nodes = len(gm.graph.nodes)
        n_det = strip_autograd_detach_chains(gm)
        fn = torch.compile(gm, dynamic=False, mode=self.compile_mode) if self.compile_mode != "none" else gm
        self.log(f"[traced-force] new crystal graph {key}: {n_nodes} nodes, {n_det} detaches removed, "
                 f"traced ({self.trace_mode}) in {time.perf_counter() - t0:.1f}s (compiles on first call); graphs={len(self.graphs) + 1}")
        return fn

    def __call__(self, b: dict) -> dict[str, Tensor]:
        key = (tuple(b["z"].shape), tuple(b["edge_index"].shape), tuple(b["cell"].shape), self.model.training)
        e, f, s = self._run(key, b, CRYSTAL_KEYS)
        return {"energy": e, "forces": f, "stress": s}


class TracedMolForces(TracedCrystalForces):
    """Callable b -> {"energy", "forces", "dipole"} (internal kcal units, as `MenaLAM.forward_molecule` with
    add_d3=False, add_ref=False), one traced + compiled graph per dense bucket shape or sparse (cap, E_pad) key."""

    def __init__(self, model: torch.nn.Module, lot_id: int, compile_mode: str = "default", log=print,
                 trace_mode: str = "real", sparse: bool = False, e_mult: int = MOL_E_MULT, e_min: int = 0):
        super().__init__(model, compile_mode=compile_mode, log=log, trace_mode=trace_mode)
        self.lot_id = int(lot_id)
        self.sparse = bool(sparse)
        self.e_mult = int(e_mult)
        self.e_min = int(e_min)
        self._key_b: dict[tuple, int] = {}

    def _build(self, b: dict, key: tuple):
        pn, pv, bn, bv = self._state()
        model, lot_id = self.model, self.lot_id
        is_sparse = self.sparse or ("edge_index" in b)
        keys = MOL_SPARSE_KEYS if is_sparse else MOL_KEYS

        def energy_forces_dipole(params: list[Tensor], buffers: list[Tensor], *xs: Tensor):
            d = dict(zip(keys, xs))
            with _reparametrize_module(model, {**dict(zip(pn, params)), **dict(zip(bn, buffers))}):
                if is_sparse:
                    out = model.forward_molecule(
                        d["z"].long(), d["pos"], d["mask"], total_charge=d["charge"],
                        spin=d["spin"], lot_id=lot_id, compute_gradient_forces=True,
                        add_d3=False, add_ref=False, sparse=True,
                        edge_index=d["edge_index"], edge_shift=d["edge_shift"],
                    )
                else:
                    out = model.forward_molecule(
                        d["z"].long(), d["pos"], d["mask"], total_charge=d["charge"],
                        spin=d["spin"], lot_id=lot_id, compute_gradient_forces=True,
                        add_d3=False, add_ref=False,
                    )
            return out["energy"], out["forces"], out["dipole"]

        t0 = time.perf_counter()
        gm = self._trace(energy_forces_dipole, pv, bv, [b[k] for k in keys])
        n_nodes = len(gm.graph.nodes)
        n_det = strip_autograd_detach_chains(gm)
        use_inductor = (self.compile_mode != "none") and (is_sparse or int(b["z"].shape[1]) <= 256)
        fn = torch.compile(gm, dynamic=False, mode=self.compile_mode) if use_inductor else gm
        self.log(f"[traced-force] new molecular graph {key}: {n_nodes} nodes, {n_det} detaches removed, "
                 f"traced ({self.trace_mode}) in {time.perf_counter() - t0:.1f}s "
                 f"({'compiles on first call' if use_inductor else 'eager fx gm (cap>256)'}); graphs={len(self.graphs) + 1}")
        return fn

    def __call__(self, b: dict) -> dict[str, Tensor]:
        is_sparse = self.sparse or ("edge_index" in b)
        if is_sparse:
            if "edge_index" not in b:
                b = pad_mol_sparse_batch(
                    b,
                    r_cut=float(getattr(self.model, "mol_r_cut", 6.0)),
                    e_mult=self.e_mult,
                    e_min=self.e_min,
                )
            elif "edge_shift" not in b:
                b = dict(b, edge_shift=b["pos"].new_zeros((b["edge_index"].shape[1], 3)))
            if "charge" not in b and "total_charge" in b:
                b = dict(b, charge=b["total_charge"])
            key = ((int(b["z"].shape[1]), int(b["edge_index"].shape[1])), self.model.training)
            if self._key_b.get(key) not in (None, int(b["z"].shape[0])):
                self.graphs.pop(key, None)
            self._key_b[key] = int(b["z"].shape[0])
            e, f, dip = self._run(key, b, MOL_SPARSE_KEYS)
        else:
            if "charge" not in b and "total_charge" in b:
                b = dict(b, charge=b["total_charge"])
            key = (tuple(b["z"].shape), self.model.training)
            e, f, dip = self._run(key, b, MOL_KEYS)
        return {"energy": e, "forces": f, "dipole": dip}
