"""Static-shape sparse batch for lam4. One structure for every domain; the domain is a per-batch field.

Edges are stored as (sender j -> receiver i) index pairs plus an integer lattice shift, and the edge vector
is recomputed from positions inside the model, so forces are the exact gradient of the energy and the same
batch layout serves training, MD and relaxation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class GraphBatch:
    z: torch.Tensor            # [N] long, atomic number (0 for pad atoms)
    pos: torch.Tensor          # [N, 3]
    graph: torch.Tensor        # [N] long, graph id of each atom
    cell: torch.Tensor         # [G, 3, 3] (zeros for non-periodic and pad graphs)
    edge_index: torch.Tensor   # [2, E] long, row 0 sender j, row 1 receiver i
    edge_shift: torch.Tensor   # [E, 3] integer lattice shifts, float dtype
    charge: torch.Tensor       # [G] total charge Q
    spin: torch.Tensor         # [G] spin multiplicity 2S+1
    u_flag: torch.Tensor       # [G] long, 0/1 (+U), derived from composition by the data pipeline
    token: torch.Tensor        # [G] long, trunk token id (per functional)
    ref_id: torch.Tensor       # [G] long, reference-convention id (per dataset)
    domain: str                # "crystal" | "molecule": one value per batch picks the node-MLP expert
    atom_mask: torch.Tensor    # [N] bool, False for pad atoms
    graph_mask: torch.Tensor   # [G] bool, False for pad graphs
    energy: torch.Tensor       # [G] label (0 where masked)
    forces: torch.Tensor       # [N, 3] label (0 where masked)
    energy_mask: torch.Tensor  # [G] bool
    force_mask: torch.Tensor   # [N] bool
    coulomb_cap: int | None = None  # optional per-batch ExactCoulomb slot capacity (e.g. bucketed 64 | 128 | 256 | 350)
    frag_id: torch.Tensor | None = None      # [N] long, covalent fragment/island group id per atom; defaults to self.graph when None
    frag_charge: torch.Tensor | None = None  # [F] float, per-fragment formal charge; defaults to self.charge when None
    frag_spin: torch.Tensor | None = None    # [F] float, per-fragment spin multiplicity; defaults to self.spin when None
    pbc: bool = False          # one value per batch (sources fix it): periodic cells. Open-boundary physics (ExactCoulomb, whole-graph group
                               # context) is never applied to a periodic batch; group physics on pbc needs finite fragments (frag_id)
    spme_grid: tuple[int, int, int] | None = None  # optional static FFT grid bucket for traced periodic BatchedSPME
    xc_id: torch.Tensor | None = None              # [G] long, factorized XC functional id (defaults to token when None)
    paw_id: torch.Tensor | None = None             # [G] long, factorized PAW/basis id (0 = vasp_pbe52 when None)
    u_atom: torch.Tensor | None = None             # [N] long, per-atom Hubbard +U bit (0/1)

    @property
    def frag_group(self) -> torch.Tensor:
        return self.graph if self.frag_id is None else self.frag_id

    @property
    def frag_q(self) -> torch.Tensor:
        return self.charge if self.frag_charge is None else self.frag_charge

    @property
    def frag_s(self) -> torch.Tensor:
        return self.spin if self.frag_spin is None else self.frag_spin

    @property
    def n_atoms(self) -> int:
        return int(self.z.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def n_graphs(self) -> int:
        return int(self.cell.shape[0])

    def cart_shift(self) -> torch.Tensor:
        """Cartesian shift [E, 3] of each edge (integer lattice shift times the receiver's cell): the form the edge ops take."""
        if self.cell.shape[0] == 1:
            return self.edge_shift @ self.cell[0]
        i = self.edge_index[1]
        return torch.einsum("ek,ekl->el", self.edge_shift, self.cell[self.graph[i]])

    def edge_vectors(self) -> torch.Tensor:
        j, i = self.edge_index
        return self.pos[j] - self.pos[i] + self.cart_shift()

    def to(self, device) -> "GraphBatch":
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        return GraphBatch(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()})
