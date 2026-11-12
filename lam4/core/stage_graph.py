"""Plain containers the stage reads: `EdgeGraph` (what the edge op needs) and `NodeContext` (per-atom / per-graph data of the node side).

EdgeGraph: pos [N, 3] (the differentiable leaf), recv, send [E] receiver-sorted, shift [E, 3] Cartesian (edge vector = pos[send] - pos[recv] + shift),
rho [N] PairWarp.node_rho or None, chunks (edge_chunks.make_chunks plan, built in the data pipeline; only for mode == "stream"),
edge_scale [E] or None (COMPAT mode: earlier Mena ChemicalEdgeWeights, heads off; equivalence test only), mode "ref" (oracle) | "stream" (chunked),
z [N] long atomic numbers (0 for pad atoms) for the descriptor floor; None = no floor even when ModelConfig.descriptor_floor is on (`from_batch` always sets it).
NodeContext: graph [N] atom -> graph id, atom_mask [N] bool, charge / spin [G] (group context), cond [G, cond_dim] (Conditioning output, or None
= unconditioned). `from_batch` builds both from a GraphBatch (edge_index row 0 = sender, row 1 = receiver).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lam4.data.batch import GraphBatch


@dataclass
class EdgeGraph:
    pos: Tensor
    recv: Tensor
    send: Tensor
    shift: Tensor
    rho: Tensor | None = None
    chunks: list | None = None
    edge_scale: Tensor | None = None
    mode: str = "ref"
    z: Tensor | None = None
    geom: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None
    phi: Tensor | None = None
    csr_plan: Any = None
    edge_shift: Tensor | None = None
    cell: Tensor | None = None
    graph: Tensor | None = None
    create_graph: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("ref", "stream", "csr"):
            raise ValueError("mode in {ref, stream, csr}")

    @classmethod
    def from_batch(
        cls,
        b: GraphBatch,
        rho: Tensor | None = None,
        chunks: list | None = None,
        mode: str = "ref",
        pos: Tensor | None = None,
        *,
        csr_plan: Any = None,
        create_graph: bool = False,
    ) -> "EdgeGraph":
        p = b.pos if pos is None else pos
        if mode == "csr" and not create_graph:
            return cls(
                p, b.edge_index[1], b.edge_index[0], p.new_zeros(0), rho, chunks, None, mode, b.z,
                csr_plan=csr_plan, edge_shift=b.edge_shift, cell=b.cell, graph=b.graph, create_graph=False,
            )
        return cls(p, b.edge_index[1], b.edge_index[0], b.cart_shift(), rho, chunks, None, mode, b.z, create_graph=create_graph)


@dataclass
class NodeContext:
    graph: Tensor | None = None
    atom_mask: Tensor | None = None
    charge: Tensor | None = None
    spin: Tensor | None = None
    cond: Tensor | None = None

    @classmethod
    def from_batch(cls, b: GraphBatch, cond: Tensor | None = None) -> "NodeContext":
        if cond is not None and b.frag_id is not None and cond.shape[0] != b.frag_q.shape[0]:
            fg = torch.arange(b.frag_q.shape[0], device=b.graph.device, dtype=torch.long).clamp_max(b.n_graphs - 1)
            cond = cond.index_select(0, fg.index_put_((b.frag_group,), b.graph))
        return cls(b.frag_group, b.atom_mask, b.frag_q, b.frag_s, cond)
