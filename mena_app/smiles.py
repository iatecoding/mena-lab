"""Pure-Python + NumPy OpenSMILES parser and 3D `ase.Atoms` builder."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from mena_app.smiles_embed import build_3d_coordinates
from mena_app.smiles_kekule import _err, atom_radicals, kekulize_and_assign_h

_AROM_LOWER = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S", "se": "Se", "as": "As", "te": "Te"}
_BRACKET_RE = re.compile(
    r"^(\d+)?([A-Z][a-z]?|se|as|te|[bcnops])(@@|@)?(H\d*)?([+-]\d+|\d+[+-]|[+]+|[-]+)?(?::\d+)?$"
)
_BOND_ORDERS = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5, "/": 1.0, "\\": 1.0}


def _parse_charge(tok: str | None) -> int:
    if not tok:
        return 0
    if tok[0] in "+-" and len(set(tok)) == 1:
        return len(tok) if tok[0] == "+" else -len(tok)
    if tok[0] in "+-":
        return int(tok)
    return int(tok[:-1]) if tok[-1] == "+" else -int(tok[:-1])


def _parse_bracket(body: str) -> tuple[int, bool, str | None, int, int]:
    m = _BRACKET_RE.match(body)
    if not m:
        _err(f"invalid bracket atom [{body}] in SMILES")
    _, raw_sym, chiral, h_tok, q_tok = m.groups()
    is_arom = raw_sym[0].islower()
    sym = _AROM_LOWER.get(raw_sym, raw_sym)
    z = atomic_numbers.get(sym, 0)
    if z <= 0:
        _err(f"unknown chemical element {raw_sym!r} in SMILES")
    h_cnt = 0 if not h_tok else (1 if h_tok == "H" else int(h_tok[1:]))
    return int(z), is_arom, chiral, h_cnt, _parse_charge(q_tok)


def smiles_to_atoms(smiles: str) -> Atoms:
    """Parse an OpenSMILES string and return a clash-free 3D `ase.Atoms` with formal charges and spin hints."""
    if not isinstance(smiles, str):
        _err("SMILES input must be a string")
    s = smiles.strip()
    if not s or any(ch.isspace() for ch in s):
        _err("empty or whitespace-containing SMILES string")

    atoms_meta: list[dict[str, Any]] = []
    bonds: list[list[Any]] = []  # [u, v, bo, is_arom_ring, explicit_colon]
    bonded_pairs: set[tuple[int, int]] = set()
    dir_map: dict[tuple[int, int], int] = {}
    lex_nbrs: list[list[Any]] = []

    stack: list[int] = []
    open_rings: dict[int, tuple[int, str | None, int]] = {}
    prev: int | None = None
    pend_bond: str | None = None
    just_opened_branch = False

    def _add_bond(u: int, v: int, b_tok: str | None) -> None:
        if u == v:
            _err("self-loop ring closure in SMILES")
        pair = (min(u, v), max(u, v))
        if pair in bonded_pairs:
            _err("duplicate bond between the same two atoms in SMILES")
        bonded_pairs.add(pair)
        if b_tok is None:
            ia = atoms_meta[u]["arom"] and atoms_meta[v]["arom"]
            bo = 1.5 if ia else 1.0
        else:
            bo = _BOND_ORDERS[b_tok]
            ia = bo == 1.5
            if b_tok == "/":
                dir_map[(u, v)], dir_map[(v, u)] = 1, -1
            elif b_tok == "\\":
                dir_map[(u, v)], dir_map[(v, u)] = -1, 1
        bonds.append([u, v, bo, ia, b_tok == ":"])

    i, n_chr = 0, len(s)
    while i < n_chr:
        ch = s[i]
        if ch == "(":
            if prev is None or pend_bond is not None:
                _err("invalid branch '(' placement in SMILES")
            stack.append(prev)
            just_opened_branch = True
            i += 1
        elif ch == ")":
            if not stack or pend_bond is not None or just_opened_branch:
                _err("unmatched or empty ')' in SMILES")
            prev = stack.pop()
            i += 1
        elif ch == ".":
            if prev is None or pend_bond is not None or just_opened_branch or i == n_chr - 1:
                _err("invalid disconnected '.' separator in SMILES")
            prev = None
            i += 1
        elif ch in _BOND_ORDERS:
            if prev is None or pend_bond is not None:
                _err(f"unexpected bond symbol {ch!r} in SMILES")
            pend_bond = ch
            i += 1
        elif ch.isdigit() or ch == "%":
            if prev is None:
                _err("ring closure digit without preceding atom in SMILES")
            if ch == "%":
                if i + 2 >= n_chr or not s[i + 1 : i + 3].isdigit():
                    _err("invalid '%nn' ring closure in SMILES")
                r_id = int(s[i + 1 : i + 3])
                i += 3
            else:
                r_id = int(ch)
                i += 1
            just_opened_branch = False
            if r_id not in open_rings:
                slot = len(lex_nbrs[prev])
                lex_nbrs[prev].append(("R", r_id))
                open_rings[r_id] = (prev, pend_bond, slot)
            else:
                u, open_tok, u_slot = open_rings.pop(r_id)
                close_tok = pend_bond
                if open_tok and close_tok and _BOND_ORDERS[open_tok] != _BOND_ORDERS[close_tok]:
                    _err(f"conflicting bond orders on ring closure {r_id}")
                lex_nbrs[u][u_slot] = prev
                lex_nbrs[prev].append(u)
                if open_tok in ("/", "\\"):
                    _add_bond(u, prev, open_tok)
                elif close_tok in ("/", "\\"):
                    _add_bond(prev, u, close_tok)
                else:
                    _add_bond(u, prev, open_tok or close_tok)
            pend_bond = None
        elif ch == "[":
            end = s.find("]", i + 1)
            if end < 0:
                _err("unclosed '[' in SMILES")
            z, is_arom, chiral, exp_h, q = _parse_bracket(s[i + 1 : end])
            idx = len(atoms_meta)
            atoms_meta.append({"z": z, "arom": is_arom, "chiral": chiral, "exp_h": exp_h, "q": q, "bracket": True})
            nbr_slots: list[Any] = []
            if prev is not None:
                nbr_slots.append(prev)
                lex_nbrs[prev].append(idx)
                _add_bond(prev, idx, pend_bond)
            for h_k in range(exp_h):
                nbr_slots.append(("H", h_k))
            lex_nbrs.append(nbr_slots)
            prev, pend_bond, just_opened_branch = idx, None, False
            i = end + 1
        else:
            two = s[i : i + 2]
            if two in ("Cl", "Br"):
                sym, is_arom, step = two, False, 2
            elif ch in ("B", "C", "N", "O", "P", "S", "F", "I"):
                sym, is_arom, step = ch, False, 1
            elif ch in ("b", "c", "n", "o", "p", "s"):
                sym, is_arom, step = ch.upper(), True, 1
            else:
                _err(f"unexpected character {ch!r} in SMILES {smiles!r}")
            idx = len(atoms_meta)
            atoms_meta.append({"z": int(atomic_numbers[sym]), "arom": is_arom, "chiral": None, "exp_h": 0, "q": 0, "bracket": False})
            nbr_slots = []
            if prev is not None:
                nbr_slots.append(prev)
                lex_nbrs[prev].append(idx)
                _add_bond(prev, idx, pend_bond)
            lex_nbrs.append(nbr_slots)
            prev, pend_bond, just_opened_branch = idx, None, False
            i += step

    if pend_bond is not None:
        _err("trailing bond symbol at end of SMILES")
    if stack:
        _err("unmatched '(' in SMILES")
    if open_rings:
        _err(f"unclosed ring {sorted(open_rings)[0]} in SMILES")
    if not atoms_meta:
        _err("SMILES holds no atoms")

    imp_h = kekulize_and_assign_h(atoms_meta, bonds)

    n_heavy = len(atoms_meta)
    all_z = [m["z"] for m in atoms_meta]
    all_q = [float(m["q"]) for m in atoms_meta]
    all_arom = [bool(m["arom"]) for m in atoms_meta]
    full_bonds: list[tuple[int, int, float, bool]] = [(b[0], b[1], float(b[2]), bool(b[3])) for b in bonds]

    for i_atom, m in enumerate(atoms_meta):
        for h_k in range(m["exp_h"] + imp_h[i_atom]):
            h_idx = len(all_z)
            all_z.append(1)
            all_q.append(0.0)
            all_arom.append(False)
            full_bonds.append((i_atom, h_idx, 1.0, False))
            if h_k < m["exp_h"]:
                lex_nbrs[i_atom][lex_nbrs[i_atom].index(("H", h_k))] = h_idx
            else:
                lex_nbrs[i_atom].append(h_idx)

    adj_all: list[list[int]] = [[] for _ in range(len(all_z))]
    for u, v, *_ in full_bonds:
        adj_all[u].append(v)
        adj_all[v].append(u)

    ez_specs: list[tuple[int, int, int, int, float]] = []
    for u, v, bo, ia in full_bonds:
        if bo == 2.0 and not ia and 2 <= len(adj_all[u]) <= 3 and 2 <= len(adj_all[v]) <= 3:
            nu = [x for x in adj_all[u] if x != v]
            nv = [y for y in adj_all[v] if y != u]
            du = {x: dir_map[(u, x)] for x in nu if (u, x) in dir_map}
            dv = {y: dir_map[(v, y)] for y in nv if (v, y) in dir_map}
            if du and dv:
                if len(du) == 2 and sum(du.values()) != 0:
                    _err("conflicting directional '/' '\\' bonds on alkene atom")
                if len(dv) == 2 and sum(dv.values()) != 0:
                    _err("conflicting directional '/' '\\' bonds on alkene atom")
                x0, sx0 = next(iter(du.items()))
                y0, sy0 = next(iter(dv.items()))
                side_u = {x0: sx0, **({x: -sx0 for x in nu if x != x0} if len(nu) == 2 else {})}
                side_v = {y0: sy0, **({y: -sy0 for y in nv if y != y0} if len(nv) == 2 else {})}
                for a, sa in side_u.items():
                    for d, sd in side_v.items():
                        ez_specs.append((a, u, v, d, 1.0 if sa == sd else -1.0))

    chiral_specs: list[tuple[int, int, int, int, float]] = []
    for i_atom in range(n_heavy):
        ch_tag = atoms_meta[i_atom]["chiral"]
        if ch_tag in ("@", "@@") and len(lex_nbrs[i_atom]) == 4 and all(isinstance(x, int) for x in lex_nbrs[i_atom]):
            n0, n1, n2, n3 = lex_nbrs[i_atom]
            chiral_specs.append((n0, n1, n2, n3, 1.0 if ch_tag == "@" else -1.0))

    bo_heavy = [0] * n_heavy
    for u, v, bo, _ in full_bonds:
        if u < n_heavy:
            bo_heavy[u] += int(round(bo))
        if v < n_heavy:
            bo_heavy[v] += int(round(bo))
    rad = sum(atom_radicals(atoms_meta[k]["z"], atoms_meta[k]["q"], bo_heavy[k], atoms_meta[k]["bracket"]) for k in range(n_heavy))

    z_arr = np.array(all_z, dtype=np.int64)
    q_arr = np.array(all_q, dtype=np.float64)
    pos = build_3d_coordinates(z_arr, q_arr, full_bonds, ez_specs, chiral_specs, np.array(all_arom, dtype=bool))

    atoms = Atoms(numbers=z_arr.tolist(), positions=pos)
    if any(q != 0.0 for q in all_q):
        atoms.set_initial_charges(all_q)
    total_q = int(round(sum(all_q)))
    atoms.info["charge"] = total_q
    total_e = int(z_arr.sum())
    if rad > 0:
        spin = int(rad + 1)
        if (total_e - total_q) % 2 != (spin - 1) % 2:
            spin = 2 if (total_e - total_q) % 2 else 1
        atoms.info["spin"] = spin
    elif (total_e - total_q) % 2 == 1:
        atoms.info["spin"] = 2
    return atoms
