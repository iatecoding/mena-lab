"""Heteroaromatic Kekulizer, implicit-hydrogen valence assignment, and radical-electron count for SMILES."""

from __future__ import annotations

from collections import deque
from typing import Any

_ORGANIC_VALENCES: dict[int, tuple[int, ...]] = {
    5: (3,), 6: (4,), 7: (3, 5), 8: (2,), 9: (1,),
    15: (3, 5), 16: (2, 4, 6), 17: (1,), 35: (1,), 53: (1,),
}
_OUTER_VE = {
    1: 1, 2: 2, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
    19: 1, 20: 2, 31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8,
    37: 1, 38: 2, 49: 3, 50: 4, 51: 5, 52: 6, 53: 7, 54: 8,
    55: 1, 56: 2, 81: 3, 82: 4, 83: 5, 84: 6, 85: 7, 86: 8,
}


def _err(msg: str):
    from mena_app.structures import StructureError
    raise StructureError(msg)


def _is_ring_edge(n: int, edges: list[tuple[int, int]], skip_idx: int) -> bool:
    u, target = edges[skip_idx]
    adj: list[list[int]] = [[] for _ in range(n)]
    for k, (a, b) in enumerate(edges):
        if k != skip_idx:
            adj[a].append(b)
            adj[b].append(a)
    seen = {u}
    q = deque([u])
    while q:
        cur = q.popleft()
        if cur == target:
            return True
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def _perfect_matching(nodes: set[int], adj: dict[int, list[int]]) -> list[tuple[int, int]] | None:
    """Exact perfect matching on small bounded-degree graphs via degree-1 unit propagation + backtracking."""
    rem = set(nodes)
    matched: list[tuple[int, int]] = []
    while True:
        forced = False
        for u in sorted(rem):
            if u not in rem:
                continue
            nbrs = [v for v in adj[u] if v in rem]
            if len(nbrs) == 0:
                return None
            if len(nbrs) == 1:
                v = nbrs[0]
                matched.append((u, v))
                rem.remove(u)
                rem.remove(v)
                forced = True
        if not forced:
            break
    if not rem:
        return matched
    u0 = min(sorted(rem), key=lambda u: sum(1 for v in adj[u] if v in rem))
    for v0 in adj[u0]:
        if v0 in rem:
            sub = _perfect_matching(rem - {u0, v0}, adj)
            if sub is not None:
                return matched + [(u0, v0)] + sub
    return None


def kekulize_and_assign_h(atoms_meta: list[dict[str, Any]], bonds: list[list[Any]]) -> list[int]:
    """Resolve `1.5` aromatic bonds to `1.0`/`2.0` and return implicit H counts for every heavy atom."""
    n = len(atoms_meta)
    edge_pairs = [(b[0], b[1]) for b in bonds]

    for idx, b in enumerate(bonds):
        if b[2] == 1.5 and not _is_ring_edge(n, edge_pairs, idx):
            if b[4]:
                _err("aromatic bond ':' outside a ring is invalid SMILES")
            b[2] = 1.0
            b[3] = False

    arom_edges = [idx for idx, b in enumerate(bonds) if b[2] == 1.5]
    imp_h = [0] * n
    arom_incident: list[list[int]] = [[] for _ in range(n)]
    for idx in arom_edges:
        u, v = bonds[idx][0], bonds[idx][1]
        arom_incident[u].append(idx)
        arom_incident[v].append(idx)

    if arom_edges:
        need_dbl: dict[int, int] = {}
        for i, m in enumerate(atoms_meta):
            a_deg = len(arom_incident[i])
            if a_deg == 0:
                if m["arom"] and not m["bracket"]:
                    _err(f"aromatic atom {i + 1} is not in an aromatic ring")
                continue
            exo_bo = int(round(sum(bonds[k][2] for k in range(len(bonds)) if i in (bonds[k][0], bonds[k][1]) and bonds[k][2] != 1.5)))
            base = exo_bo + m["exp_h"] + a_deg
            z, q = m["z"], m["q"]
            if m["bracket"]:
                target_map = {5: 3 - q, 6: 4 - abs(q), 7: 3 + q, 15: 3 + q, 33: 3 + q, 8: 2 + q, 16: 2 + q, 34: 2 + q}
                if z not in target_map:
                    _err(f"unsupported bracket aromatic atom Z={z}")
                nd = target_map[z] - base
                if nd not in (0, 1):
                    _err("cannot Kekulize aromatic ring: invalid bracket valence")
                need_dbl[i] = nd
            elif z == 6:
                if base > 4:
                    _err("aromatic carbon exceeds valence 4")
                imp_h[i] = 1 if base == 2 else 0
                need_dbl[i] = 4 - (base + imp_h[i])
            elif z in (7, 15):
                if base not in (2, 3, 4):
                    _err("cannot Kekulize aromatic nitrogen/phosphorus")
                need_dbl[i] = 0 if base == 3 else 1
            elif z == 8:
                if base != 2:
                    _err("aromatic oxygen exceeds valence 2")
                need_dbl[i] = 0
            elif z == 16:
                if base not in (2, 4, 6):
                    _err("invalid aromatic sulfur valence")
                need_dbl[i] = 0
            elif z == 5:
                if base not in (2, 3):
                    _err("invalid aromatic boron valence")
                imp_h[i] = 1 if base == 2 else 0
                need_dbl[i] = 0
            else:
                _err(f"unsupported unbracketed aromatic element Z={z}")

        v1 = {i for i, nd in need_dbl.items() if nd == 1}
        if len(v1) % 2 != 0:
            _err("cannot Kekulize aromatic ring system (odd pi-electron count)")
        adj_v1: dict[int, list[int]] = {i: [] for i in v1}
        edge_lookup: dict[tuple[int, int], int] = {}
        for idx in arom_edges:
            u, v = bonds[idx][0], bonds[idx][1]
            bonds[idx][2] = 1.0
            if u in v1 and v in v1:
                adj_v1[u].append(v)
                adj_v1[v].append(u)
                edge_lookup[(min(u, v), max(u, v))] = idx
        matching = _perfect_matching(v1, adj_v1) if v1 else []
        if matching is None:
            _err("cannot Kekulize aromatic ring system")
        for u, v in matching:
            bonds[edge_lookup[(min(u, v), max(u, v))]][2] = 2.0
    else:
        for m in atoms_meta:
            if m["arom"] and not m["bracket"]:
                _err("aromatic atom is not part of a ring")

    bo_sums = [m["exp_h"] for m in atoms_meta]
    for u, v, bo, *_ in bonds:
        bo_sums[u] += int(round(bo))
        bo_sums[v] += int(round(bo))

    for i, m in enumerate(atoms_meta):
        z, q, bo = m["z"], m["q"], bo_sums[i]
        if not m["bracket"] and len(arom_incident[i]) == 0:
            vals = _ORGANIC_VALENCES[z]
            valid = [v for v in vals if v >= bo]
            if not valid:
                _err(f"atom {i + 1} (Z={z}) has bond order sum {bo}, exceeding max valence {max(vals)}")
            imp_h[i] = valid[0] - bo
        elif m["bracket"] and z in (6, 7, 8, 9):
            max_b = 4 if z in (6, 7) else (3 if z == 8 else 2)
            if bo > max_b or (_OUTER_VE[z] - q + bo) > 8:
                _err(f"bracket atom {i + 1} (Z={z}) exceeds octet valence")
    return imp_h


def atom_radicals(z: int, q: int, bo_sum: int, is_bracket: bool) -> int:
    """Count unpaired valence electrons on a bracket atom (`0` for closed-shell ions and organic atoms)."""
    if not is_bracket:
        return 0
    if z in (1, 2):
        ve = z - q
        if ve <= 0:
            return 0
        u = max(0, ve - bo_sum)
        s = ve + bo_sum
        return min(u, 2 - s) if s <= 2 else u % 2
    if z in _OUTER_VE:
        ve = _OUTER_VE[z] - q
        if ve <= 0:
            return 0
        u = ve - bo_sum
        if u <= 0:
            return abs(u) % 2
        if _OUTER_VE[z] == 1:
            return u
        if _OUTER_VE[z] == 2:
            return u % 2
        s = ve + bo_sum
        return min(u, 8 - s) if s <= 8 else u % 2
    return abs(z - q - bo_sum) % 2
