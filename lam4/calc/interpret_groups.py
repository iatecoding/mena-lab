"""Functional-group annotation and unit-conversion helpers for Dipole/Field X-Ray interpretability."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ase import Atoms
from ase.data import covalent_radii
import numpy as np

from lam4.calc.physics import E_ANG_TO_DEBYE, MOL_KCAL_TO_EV
from lam4.physics.coulomb_exact import COULOMB_K

# Exact conversion from lam4 internal electric-field units (e / A^2) to V/A and MV/cm:
#   phi [e/A] -> energy [kcal/mol] = K_COUL * q * phi, so fld [e/A^2] = -grad phi.
#   1 (e/A^2) * COULOMB_K [kcal*A/(mol*e^2)] = kcal/(mol*e*A).
#   Multiplying by MOL_KCAL_TO_EV converts kcal/(mol*e*A) -> eV/(e*A) = V/A.
VA_PER_EA2: float = COULOMB_K * MOL_KCAL_TO_EV
MV_CM_PER_EA2: float = 100.0 * VA_PER_EA2
EA2_PER_VA: float = 1.0 / VA_PER_EA2
EA2_PER_MV_CM: float = 1.0 / MV_CM_PER_EA2

SMARTS_RULES: tuple[tuple[str, str], ...] = (
    ("nitrile", "[CX2]#[NX1]"),
    ("sulfonamide", "[SX4](=[OX1])(=[OX1])[NX3]"),
    ("sulfone_sulfoxide", "[SX4,SX3](=[OX1])"),
    ("guanidine_amidine", "[CX3](=[NX2])([NX3])[NX3,CX4]"),
    ("urea_carbamate", "[NX3][CX3](=[OX1])[NX3,OX2]"),
    ("carboxylate_acid", "[CX3](=[OX1])[OX1-,OX2H1]"),
    ("ester", "[CX3](=[OX1])[OX2][#6]"),
    ("amide", "[CX3](=[OX1])[NX3]"),
    ("carbonyl", "[CX3]=[OX1]"),
    ("phosphate", "[PX4](=[OX1])"),
    ("hydroxyl", "[OX2H1]"),
    ("ether", "[OX2]([#6])[#6]"),
    ("amine", "[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!a]"),
    ("heteroaryl_N", "[n]"),
    ("halogen", "[F,Cl,Br,I]"),
    ("aryl", "[a]"),
    ("aliphatic", "[CX4]"),
)

_HALOGEN_Z = frozenset({9, 17, 35, 53})
_HALOGEN_NAMES = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}


def field_to_va(fld_ea2: np.ndarray | float) -> np.ndarray | float:
    """Convert `lam4` electric field from `e/A^2` to `V/A`."""
    return fld_ea2 * VA_PER_EA2


def field_to_mv_cm(fld_ea2: np.ndarray | float) -> np.ndarray | float:
    """Convert `lam4` electric field from `e/A^2` to `MV/cm`."""
    return fld_ea2 * MV_CM_PER_EA2


def dipole_to_debye(dip_ea: np.ndarray | float) -> np.ndarray | float:
    """Convert dipole from `e*A` to `Debye`."""
    return dip_ea * E_ANG_TO_DEBYE


def covalent_adj(z: np.ndarray, pos: np.ndarray, *, scale: float = 1.25) -> list[list[int]]:
    """Build covalent adjacency list from scaled ASE covalent radii."""
    n = len(z)
    adj: list[list[int]] = [[] for _ in range(n)]
    if n < 2:
        return adj
    rad = np.array([covalent_radii[int(zi)] if 0 <= int(zi) < len(covalent_radii) else 1.5 for zi in z], dtype=np.float64)
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    cutoff = scale * (rad[:, None] + rad[None, :])
    ii, jj = np.where((dist > 0.35) & (dist <= cutoff))
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i < j:
            adj[i].append(j)
            adj[j].append(i)
    return adj


def assign_groups_rdkit(lig_atoms: Atoms, rdkit_mol: Any | None, lig_smiles: str | None) -> list[str] | None:
    """Classify atoms by RDKit SMARTS hierarchy when RDKit is installed."""
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(lig_smiles) if (rdkit_mol is None and lig_smiles is not None) else rdkit_mol
    if mol is None:
        return None
    n_lig = len(lig_atoms)
    z_lig = lig_atoms.get_atomic_numbers()
    if mol.GetNumAtoms() != n_lig:
        heavy_idx = [i for i, zi in enumerate(z_lig) if zi != 1]
        if mol.GetNumAtoms() != len(heavy_idx):
            return None
        mol_to_ase = heavy_idx
    else:
        mol_to_ase = list(range(n_lig))

    labels = [""] * n_lig
    for name, smarts in SMARTS_RULES:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        for match in mol.GetSubstructMatches(pat):
            for m_idx in match:
                a_idx = mol_to_ase[m_idx]
                if not labels[a_idx]:
                    labels[a_idx] = name
    adj = covalent_adj(z_lig, lig_atoms.get_positions())
    for i in range(n_lig):
        if not labels[i] and int(z_lig[i]) == 1 and adj[i]:
            labels[i] = labels[adj[i][0]] or "aliphatic"
        elif not labels[i]:
            labels[i] = _HALOGEN_NAMES.get(int(z_lig[i]), f"elem_{lig_atoms[i].symbol}")
    return labels


def assign_groups_topology(lig_atoms: Atoms) -> list[str]:
    """Deterministic covalent-graph functional group detector (used when RDKit is absent)."""
    n = len(lig_atoms)
    z = lig_atoms.get_atomic_numbers()
    pos = lig_atoms.get_positions()
    adj = covalent_adj(z, pos)
    labels = [""] * n

    def _dist(a: int, b: int) -> float:
        return float(np.linalg.norm(pos[a] - pos[b]))

    for i in range(n):
        zi, nbrs = int(z[i]), adj[i]
        if zi == 16:
            ox_nbrs = [j for j in nbrs if int(z[j]) == 8 and len(adj[j]) == 1]
            n_nbrs = [j for j in nbrs if int(z[j]) == 7]
            if len(ox_nbrs) >= 2 and n_nbrs:
                for k in [i, *ox_nbrs, *n_nbrs]:
                    labels[k] = labels[k] or "sulfonamide"
            elif ox_nbrs:
                for k in [i, *ox_nbrs]:
                    labels[k] = labels[k] or "sulfone_sulfoxide"
        elif zi == 6:
            n_term = [j for j in nbrs if int(z[j]) == 7 and len(adj[j]) == 1 and _dist(i, j) < 1.24]
            if n_term:
                for k in [i, *n_term]:
                    labels[k] = labels[k] or "nitrile"
                continue
            o_term = [j for j in nbrs if int(z[j]) == 8 and len(adj[j]) == 1 and _dist(i, j) < 1.32]
            o_single = [j for j in nbrs if int(z[j]) == 8 and len(adj[j]) == 2]
            n_all = [j for j in nbrs if int(z[j]) == 7]
            if len(o_term) >= 2:
                for k in [i, *o_term]:
                    labels[k] = labels[k] or "carboxylate_acid"
            elif len(o_term) == 1:
                oh_nbrs = [j for j in nbrs if int(z[j]) == 8 and len(adj[j]) == 2 and any(int(z[h]) == 1 for h in adj[j])]
                tag = (
                    "carboxylate_acid" if oh_nbrs else
                    "urea_carbamate" if len(n_all) >= 2 else
                    "amide" if len(n_all) == 1 else
                    "ester" if o_single else "carbonyl"
                )
                extra = oh_nbrs or (n_all if len(n_all) >= 1 else o_single)
                for k in [i, *o_term, *extra]:
                    labels[k] = labels[k] or tag
            elif len(n_all) >= 3:
                for k in [i, *n_all]:
                    labels[k] = labels[k] or "guanidine_amidine"

    for i in range(n):
        if labels[i]:
            continue
        zi, nbrs = int(z[i]), adj[i]
        if zi == 8:
            has_h = any(int(z[j]) == 1 for j in nbrs)
            labels[i] = "hydroxyl" if (has_h or len(nbrs) <= 1) else "ether"
        elif zi == 7:
            heavy_deg = sum(1 for j in nbrs if int(z[j]) != 1)
            short_ring = heavy_deg >= 2 and all(_dist(i, j) < 1.41 for j in nbrs if int(z[j]) != 1)
            labels[i] = "heteroaryl_N" if short_ring else "amine"
        elif zi in _HALOGEN_Z:
            labels[i] = _HALOGEN_NAMES[zi]
        elif zi == 6:
            heavy_deg = sum(1 for j in nbrs if int(z[j]) != 1)
            labels[i] = "aryl_or_alkene" if len(nbrs) <= 3 or (heavy_deg >= 2 and all(_dist(i, j) < 1.44 for j in nbrs if int(z[j]) != 1)) else "aliphatic"
        elif zi != 1:
            labels[i] = f"elem_{lig_atoms[i].symbol}"

    for i in range(n):
        if not labels[i] and int(z[i]) == 1:
            labels[i] = labels[adj[i][0]] if adj[i] else "hydrogen"
    return labels


def cluster_functional_groups(
    lig_atoms: Atoms,
    atom_labels: list[str],
    adj: list[list[int]],
) -> list[tuple[str, list[int]]]:
    """Group connected atoms sharing the same functional group label into distinct group instances."""
    n = len(lig_atoms)
    visited = [False] * n
    raw_instances: list[tuple[str, list[int]]] = []
    for i in range(n):
        if visited[i]:
            continue
        lbl = atom_labels[i]
        stack, comp = [i], []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v] and atom_labels[v] == lbl:
                    visited[v] = True
                    stack.append(v)
        comp.sort()
        raw_instances.append((lbl, comp))

    totals: dict[str, int] = defaultdict(int)
    for lbl, _ in raw_instances:
        totals[lbl] += 1
    counts: dict[str, int] = defaultdict(int)
    return [
        (lbl if totals[lbl] == 1 else f"{lbl}#{(counts.__setitem__(lbl, counts[lbl] + 1) or counts[lbl])}", comp)
        for lbl, comp in raw_instances
    ]
