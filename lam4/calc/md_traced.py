"""Static `make_fx` + `torch.compile(dynamic=False)` force field for `MDEngine` (`edge_mode='ref'`).

Eliminates Dual-CSR sequential atom loops and per-step Python autograd dispatch by tracing forward + coordinate/strain
autograd backward into one pure-ATen FX graph for fixed padded `(pos, edge_index, edge_shift)` buffers, compiled with
static-shape Inductor (`dynamic=False`) so all 10 NHC steps capture cleanly into one CUDA Graph.
"""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any
import warnings

import torch
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx

import lam4.core.radial  # noqa: F401 - registers lam4::stop_grad custom op
from lam4.calc.runtime import precision_context
from lam4.core.consts import capturing
from lam4.physics.coulomb_exact import resolve_coulomb_meta


def alias_to_detach(gm: torch.fx.GraphModule) -> int:
    """Lower explicit `lam4::stop_grad` markers to `aten.detach.default` without touching single `aten.alias.default` views."""
    sg = torch.ops.lam4.stop_grad.default
    det = torch.ops.aten.detach.default
    n = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is sg:
            node.target = det
            n += 1
    return n


def strip_autograd_detach_chains(gm: torch.fx.GraphModule) -> int:
    """Collapse autograd-inserted `detach(detach(x))` and `alias(alias(x))` pairs to `x`, and lower `lam4::stop_grad`."""
    pairs = (torch.ops.aten.detach.default, torch.ops.aten.alias.default)
    inners: list[torch.fx.Node] = []
    removed = 0
    for n in list(gm.graph.nodes):
        if n.op == "call_function" and n.target in pairs:
            src = n.args[0]
            if isinstance(src, torch.fx.Node) and src.op == "call_function" and src.target is n.target:
                n.replace_all_uses_with(src.args[0])
                gm.graph.erase_node(n)
                inners.append(src)
                removed += 1
    for src in reversed(inners):
        if src.op == "call_function" and not src.users:
            gm.graph.erase_node(src)
            removed += 1
    alias_to_detach(gm)
    gm.graph.lint()
    gm.recompile()
    return removed


def _warm_traced_caches(model: torch.nn.Module, pb: Any) -> Any:
    """Resolve static SPME grid bucket, basis solve matrices, and open-molecule Coulomb metadata before `make_fx`."""
    dev = pb.pos.device
    dt = pb.pos.dtype
    for st in getattr(getattr(model, "trunk", None), "stages", ()):
        if hasattr(st, "tables") and hasattr(st.tables, "solve_matrix"):
            st.tables.solve_matrix(dt, dev)
    if pb.domain == "molecule" and not pb.pbc:
        resolve_coulomb_meta(pb, model.cfg.coulomb_cap)
    cfg = getattr(model, "cfg", None)
    if pb.pbc and cfg is not None and (getattr(cfg, "spme_batched", False) or getattr(pb, "periodic_electro", False)) and getattr(pb, "spme_grid", None) is None:
        from lam4.physics.spme_batched import select_grid_bucket, select_spme_grid

        spacing = float(getattr(cfg, "mesh_spacing", 0.8))
        grid = (
            select_spme_grid(pb.cell, spacing=spacing, min_size=max(8, int(getattr(cfg, "mesh_order", 6))))
            if getattr(pb, "periodic_electro", False)
            else select_grid_bucket(pb.cell, spacing)
        )
        return replace(pb, spme_grid=grid)
    return pb


class TracedMDForceField:
    """Fixed-shape static compiled `(pos, edge_index, edge_shift) -> (energy_f64, forces, stress|None)` for `MDEngine`."""

    def __init__(
        self,
        model: torch.nn.Module,
        pb: Any,
        *,
        compute_stress: bool,
        compile_mode: str = "default",
        precision: str = "fp32_ieee",
        dense_nlh_ctx: Any = None,
    ) -> None:
        self.model = model
        self.compute_stress = bool(compute_stress)
        self.compile_mode = str(compile_mode)
        self.precision = str(precision)
        self.n_atoms = int(pb.n_atoms)
        self.n_edges = int(pb.n_edges)
        self.domain = str(pb.domain)
        self.pbc = bool(pb.pbc)

        pb_base = _warm_traced_caches(model, pb)
        dev = pb_base.pos.device
        want_stress = self.compute_stress

        def _fn(pos: Tensor, edge_index: Tensor, edge_shift: Tensor):
            bb = replace(pb_base, pos=pos, edge_index=edge_index, edge_shift=edge_shift)
            out = model(
                bb,
                create_graph=False,
                compute_forces=True,
                compute_stress=want_stress,
                chunks=None,
                add_ref=False,
                return_stage_energies=False,
                csr_plan=None,
                return_phi=False,
                return_aux_heads=False,
            )
            e = out.get("energy_f64", out["energy"])
            f = out["forces"]
            if want_stress:
                return e, f, out["stress"]
            return e, f

        trunk = getattr(model, "trunk", None)
        prev_mode = getattr(trunk, "edge_mode", None)
        t0 = time.perf_counter()
        try:
            if trunk is not None:
                trunk.edge_mode = "ref"
            if dense_nlh_ctx is not None:
                with dense_nlh_ctx(), precision_context(self.precision, dev, model), torch.enable_grad():
                    gm = make_fx(_fn, tracing_mode="real")(pb_base.pos, pb_base.edge_index, pb_base.edge_shift)
            else:
                with precision_context(self.precision, dev, model), torch.enable_grad():
                    gm = make_fx(_fn, tracing_mode="real")(pb_base.pos, pb_base.edge_index, pb_base.edge_shift)
            self.n_nodes = len(gm.graph.nodes)
            self.n_detaches_removed = strip_autograd_detach_chains(gm)
            self.trace_sec = time.perf_counter() - t0
            self._gm = gm
            self._compiled = False
            if self.compile_mode != "none":
                try:
                    self.fn = torch.compile(gm, dynamic=False, mode=self.compile_mode)
                    self._compiled = True
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"TracedMDForceField torch.compile failed ({exc}); using traced graph directly.", RuntimeWarning)
                    self.fn = gm
            else:
                self.fn = gm
        finally:
            if trunk is not None and prev_mode is not None:
                trunk.edge_mode = prev_mode

    def matches(self, pb: Any) -> bool:
        return (
            int(pb.n_atoms) == self.n_atoms
            and int(pb.n_edges) == self.n_edges
            and str(pb.domain) == self.domain
            and bool(pb.pbc) == self.pbc
        )

    def __call__(self, pos: Tensor, edge_index: Tensor, edge_shift: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        try:
            res = self.fn(pos, edge_index, edge_shift)
        except Exception as exc:  # noqa: BLE001
            if not self._compiled or capturing():
                raise
            warnings.warn(f"TracedMDForceField compiled execution failed ({exc}); falling back to traced graph.", RuntimeWarning)
            self.fn = self._gm
            self._compiled = False
            res = self.fn(pos, edge_index, edge_shift)
        if self.compute_stress:
            return res[0], res[1], res[2]
        return res[0], res[1], None
