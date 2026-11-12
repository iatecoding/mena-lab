"""Host-side chunk plan for the streaming edge op: contiguous receiver ranges holding <= chunk_edges edges.

`make_chunks` is called ONCE in the data pipeline (it reads `recv` / `send` on the host: a device sync if they live on a GPU), and the
resulting list is passed to `edge_aggregate_stream(chunks=...)` every step, so the training step has no host sync. Edges must be sorted
by receiver. Chunk k covers receivers [a, b) = edges [e0, e1); a receiver with more than chunk_edges edges gets a chunk of its own, so
the bound is max(chunk_edges, max degree). Receivers without edges produce no chunk (their outputs stay zero).

Besides the ranges, every chunk carries the index tensors that let the op touch only the rows it needs:
    recv  [e]  receiver of each edge, local to the chunk (recv - a), rows of the receiver range a:b
    uniq  [u]  the distinct sender atoms of the chunk (sorted), global indices
    send  [e]  sender of each edge as an index into `uniq`
so gradients and double-backward cotangents are allocated for [b - a] receiver rows and [u] sender rows, never for all N atoms.
A plan is tied to one edge list (and its receiver order) and is reusable as long as the neighbour list is; `chunks_to` moves it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

DEFAULT_CHUNK_EDGES = 8192  # ~15 kB/edge in backward, ~40 kB/edge in double backward on CPU: 8192 keeps the chunk under ~0.35 GB
GPU_INFER_CHUNK_EDGES = 24_576  # first-order inference on MPS/CUDA: 3x fewer chunk dispatches while staying <40 MB/chunk


def stream_chunk_edges(device: torch.device | str) -> int:
    dev = device if isinstance(device, str) else device.type
    return GPU_INFER_CHUNK_EDGES if dev in ("mps", "cuda") else DEFAULT_CHUNK_EDGES


@dataclass
class Chunk:
    a: int
    b: int
    e0: int
    e1: int
    recv: Tensor
    send: Tensor
    uniq: Tensor

    def to(self, device) -> "Chunk":
        return Chunk(self.a, self.b, self.e0, self.e1, self.recv.to(device), self.send.to(device), self.uniq.to(device))


def chunks_to(chunks: list[Chunk], device) -> list[Chunk]:
    return [c.to(device) for c in chunks]


def make_chunks(recv: Tensor, send: Tensor, n_atoms: int, chunk_edges: int = DEFAULT_CHUNK_EDGES, check_sorted: bool = True) -> list[Chunk]:
    """Chunk plan for receiver-sorted edges (recv, send) [E]; tensors of the plan are placed on recv.device. Host work, call once."""
    dev = recv.device
    rc, sc = recv.cpu(), send.cpu()
    if check_sorted and rc.numel() > 1 and not bool((rc[1:] >= rc[:-1]).all()):
        raise ValueError("edges must be sorted by receiver for the streaming op")
    ptr = np.concatenate(([0], np.cumsum(torch.bincount(rc, minlength=n_atoms).numpy())))
    out, a = [], 0
    while a < n_atoms:
        b = min(max(int(np.searchsorted(ptr, ptr[a] + chunk_edges, side="right")) - 1, a + 1), n_atoms)
        e0, e1 = int(ptr[a]), int(ptr[b])
        if e1 > e0:
            uniq, inv = torch.unique(sc[e0:e1], return_inverse=True)
            out.append(Chunk(a, b, e0, e1, rc[e0:e1] - a, inv, uniq))
        a = b
    return chunks_to(out, dev)
