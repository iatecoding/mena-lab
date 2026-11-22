"""Distance-Geometry 3D molecular coordinate embedding with VSEPR hydrogen placement."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from ase.data import covalent_radii

from mena_app.smiles_ff import lbfgs_minimize, make_forcefield, place_terminal_h

_XH_LEN = {6: 1.09, 7: 1.01, 8: 0.97}


def _bond_length(z1: int, z2: int, bo: float, is_arom: bool) -> float:
    if z1 == 1 and z2 in _XH_LEN:
        return _XH_LEN[z2]
    if z2 == 1 and z1 in _XH_LEN:
        return _XH_LEN[z1]
    r_sum = float(covalent_radii[z1] + covalent_radii[z2])
    if is_arom:
        return 0.93 * r_sum
    if bo >= 2.5:
        return 0.79 * r_sum
    if bo >= 1.8:
        return 0.87 * r_sum
    return r_sum


def _shortest_cycle_len(n: int, adj: list[list[int]], u: int, v: int) -> int:
    dist = [-1] * n
    dist[u] = 0
    q = deque([u])
    while q:
        cur = q.popleft()
        if dist[cur] >= 8:
            continue
        for nxt in adj[cur]:
            if (cur == u and nxt == v) or (cur == v and nxt == u):
                continue
            if nxt == v:
                return dist[cur] + 2
            if dist[nxt] < 0:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return 0


def _shared_ring_size(n: int, adj: list[list[int]], i: int, j: int, k: int) -> int:
    dist = [-1] * n
    dist[i] = 0
    q = deque([i])
    while q:
        cur = q.popleft()
        if dist[cur] >= 6:
            continue
        for nxt in adj[cur]:
            if cur == j or nxt == j:
                continue
            if nxt == k:
                return dist[cur] + 3
            if dist[nxt] < 0:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return 0


def _hybridizations(
    numbers: np.ndarray,
    charges: np.ndarray,
    adj: list[list[int]],
    bo_map: dict[tuple[int, int], float],
    arom_atoms: np.ndarray,
) -> list[str]:
    n = len(numbers)
    hyb = ["sp3"] * n
    for i in range(n):
        nbrs = adj[i]
        if len(nbrs) <= 1:
            hyb[i] = "term"
            continue
        bos = [bo_map[(min(i, j), max(i, j))] for j in nbrs]
        n_dbl = sum(1 for b in bos if 1.8 <= b < 2.5)
        n_trp = sum(1 for b in bos if b >= 2.5)
        if n_trp >= 1 or (len(nbrs) == 2 and n_dbl == 2):
            hyb[i] = "sp"
        elif arom_atoms[i] or n_dbl >= 1 or (numbers[i] == 5 and len(nbrs) == 3 and charges[i] == 0):
            hyb[i] = "sp2"
    for i in range(n):
        if hyb[i] == "sp3" and numbers[i] == 7 and len(adj[i]) == 3 and charges[i] == 0:
            if any(hyb[j] == "sp2" for j in adj[i]):
                hyb[i] = "sp2"
        # three-coordinate carbon with only single bonds: a radical or a cation, both planar (allyl, benzyl, methyl);
        # a carbanion is planar only when it can conjugate. Left sp3, a radical CH2 next to a pi system came out twisted
        # by 90 degrees: the rotational transition state, with the unpaired electron cut off from the pi system.
        elif hyb[i] == "sp3" and numbers[i] == 6 and len(adj[i]) == 3 and (charges[i] >= 0 or any(hyb[j] == "sp2" for j in adj[i])):
            hyb[i] = "sp2"
    return hyb


def _ideal_angle_rad(z_j: int, hyb_j: str, shared_ring: int, j_ring_min: int) -> float:
    if hyb_j == "sp":
        return math.pi
    if shared_ring == 3:
        return math.radians(60.0)
    if shared_ring == 4:
        return math.radians(90.0 if hyb_j == "sp2" else 88.5)
    if hyb_j == "sp2":
        if shared_ring == 5:
            return math.radians(108.0)
        if shared_ring == 0 and j_ring_min in (3, 4, 5):
            return math.radians(0.5 * (360.0 - {3: 60.0, 4: 90.0, 5: 108.0}[j_ring_min]))
        return math.radians(120.0)
    if shared_ring == 5:
        return math.radians(104.5)
    if z_j in (8, 16, 7):
        return math.radians({8: 104.5, 16: 96.0, 7: 107.5}[z_j])
    return math.acos(-1.0 / 3.0)


def _torsion_dist(r_ij: float, r_jk: float, r_kl: float, th_ijk: float, th_jkl: float, phi: float) -> float:
    a, b = math.pi - th_ijk, math.pi - th_jkl
    dx = r_ij * math.cos(a) + r_jk + r_kl * math.cos(b)
    dy = r_ij * math.sin(a) - r_kl * math.sin(b) * math.cos(phi)
    dz = r_kl * math.sin(b) * math.sin(phi)
    return math.sqrt(max(dx * dx + dy * dy + dz * dz, 1e-8))


def _build_terms(
    numbers: np.ndarray,
    hyb: list[str],
    adj: list[list[int]],
    bo_map: dict[tuple[int, int], float],
    arom_bond: dict[tuple[int, int], bool],
    r0_map: dict[tuple[int, int], float],
    ez_specs: list[tuple[int, int, int, int, float]],
    chiral_specs: list[tuple[int, int, int, int, float]],
):
    n = len(numbers)
    atom_ring_min = [0] * n
    bond_ring: dict[tuple[int, int], int] = {}
    for a, b in bo_map:
        rc = _shortest_cycle_len(n, adj, a, b)
        bond_ring[(a, b)] = rc
        if rc:
            atom_ring_min[a] = rc if atom_ring_min[a] == 0 else min(atom_ring_min[a], rc)
            atom_ring_min[b] = rc if atom_ring_min[b] == 0 else min(atom_ring_min[b], rc)

    b_idx = np.array([[a, b] for a, b in bo_map], dtype=np.int64)
    b_r0 = np.array([r0_map[(int(a), int(b))] for a, b in b_idx], dtype=np.float64)
    ang_list, ang_map, excl_12_13 = [], {}, set(bo_map.keys())
    for j in range(n):
        nbrs = adj[j]
        for p in range(len(nbrs)):
            for q in range(p + 1, len(nbrs)):
                i, k = nbrs[p], nbrs[q]
                th0 = _ideal_angle_rad(int(numbers[j]), hyb[j], _shared_ring_size(n, adj, i, j, k), atom_ring_min[j])
                ang_list.append((i, j, k, th0))
                ang_map[(i, j, k)] = ang_map[(k, j, i)] = th0
                excl_12_13.add((min(i, k), max(i, k)))

    a_idx = np.array([[i, j, k] for i, j, k, _ in ang_list], dtype=np.int64) if ang_list else np.zeros((0, 3), dtype=np.int64)
    a_th0 = np.array([th for *_, th in ang_list], dtype=np.float64)
    imp_list = [[j, adj[j][0], adj[j][1], adj[j][2]] for j in range(n) if hyb[j] == "sp2" and len(adj[j]) == 3]
    imp_idx = np.array(imp_list, dtype=np.int64) if imp_list else np.zeros((0, 4), dtype=np.int64)

    ez_map = {}
    for i, u, v, l, ct in ez_specs:
        ez_map[(i, u, v, l)] = ez_map[(l, v, u, i)] = ct

    t_ez, t_sp2, t_sp3, dg_14_ex, dg_14_sp3, pairs_14 = [], [], [], [], [], set()
    for u, v in b_idx.tolist():
        key = (min(u, v), max(u, v))
        bo, ia, rc, r_uv = bo_map[key], arom_bond[key], bond_ring[key], r0_map[key]
        is_conj = ia or bo >= 1.8 or (hyb[u] == "sp2" and hyb[v] == "sp2")
        for i in adj[u]:
            if i == v or hyb[u] == "sp":
                continue
            for l in adj[v]:
                if l in (u, i) or hyb[v] == "sp":
                    continue
                pairs_14.add((min(i, l), max(i, l)))
                r_iu, r_vl = r0_map[(min(i, u), max(i, u))], r0_map[(min(v, l), max(v, l))]
                th_u, th_v = ang_map[(i, u, v)], ang_map[(u, v, l)]
                if (i, u, v, l) in ez_map:
                    ct = ez_map[(i, u, v, l)]
                    t_ez.append((i, u, v, l, ct))
                    t_sp2.append((i, u, v, l))
                    dg_14_ex.append((i, l, _torsion_dist(r_iu, r_uv, r_vl, th_u, th_v, 0.0 if ct > 0 else math.pi)))
                elif is_conj:
                    t_sp2.append((i, u, v, l))
                    if rc in (5, 6):
                        su, sv = _shared_ring_size(n, adj, i, u, v) == rc, _shared_ring_size(n, adj, u, v, l) == rc
                        if su and sv:
                            t_ez.append((i, u, v, l, 1.0))
                            dg_14_ex.append((i, l, _torsion_dist(r_iu, r_uv, r_vl, th_u, th_v, 0.0)))
                        elif su ^ sv:
                            t_ez.append((i, u, v, l, -1.0))
                            dg_14_ex.append((i, l, _torsion_dist(r_iu, r_uv, r_vl, th_u, th_v, math.pi)))
                elif hyb[u] == "sp3" and hyb[v] == "sp3":
                    t_sp3.append((i, u, v, l))
                    dg_14_sp3.append((i, l, _torsion_dist(r_iu, r_uv, r_vl, th_u, th_v, math.radians(58.0)),
                                      _torsion_dist(r_iu, r_uv, r_vl, th_u, th_v, math.pi)))

    rcov = covalent_radii[numbers]
    nb_list, nb_rmin, nb_rcore = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in excl_12_13:
                rs = float(rcov[i] + rcov[j])
                sc = 0.88 if (i, j) in pairs_14 else 1.0
                nb_list.append((i, j))
                nb_rmin.append(sc * max(0.85 * rs + 0.55, 1.15 * rs))
                nb_rcore.append(0.82 * rs)
    nb_idx = np.array(nb_list, dtype=np.int64) if nb_list else np.zeros((0, 2), dtype=np.int64)

    ff = make_forcefield(
        n, b_idx, b_r0, a_idx, a_th0, a_th0 > math.radians(175.0), imp_idx,
        np.array([[a, b, c, d] for a, b, c, d, _ in chiral_specs], dtype=np.int64) if chiral_specs else np.zeros((0, 4), dtype=np.int64),
        np.array([s for *_, s in chiral_specs], dtype=np.float64),
        np.array([[a, b, c, d] for a, b, c, d, _ in t_ez], dtype=np.int64) if t_ez else np.zeros((0, 4), dtype=np.int64),
        np.array([s for *_, s in t_ez], dtype=np.float64),
        np.array(t_sp2, dtype=np.int64) if t_sp2 else np.zeros((0, 4), dtype=np.int64),
        np.array(t_sp3, dtype=np.int64) if t_sp3 else np.zeros((0, 4), dtype=np.int64),
        nb_idx, np.array(nb_rmin, dtype=np.float64), np.array(nb_rcore, dtype=np.float64),
    )
    return ff, b_idx, b_r0, ang_list, excl_12_13, dg_14_sp3, dg_14_ex, nb_idx, rcov


def _dg_embed_graph(
    numbers: np.ndarray, hyb: list[str], adj: list[list[int]],
    bo_map: dict[tuple[int, int], float], arom_bond: dict[tuple[int, int], bool],
    r0_map: dict[tuple[int, int], float], ez_specs: list[tuple[int, int, int, int, float]],
    chiral_specs: list[tuple[int, int, int, int, float]], seed: int, max_iter: int = 85,
) -> np.ndarray:
    n = len(numbers)
    ff, b_idx, b_r0, ang_list, excl_12_13, dg_14_sp3, dg_14_ex, nb_idx, rcov = _build_terms(
        numbers, hyb, adj, bo_map, arom_bond, r0_map, ez_specs, chiral_specs
    )
    L = 0.80 * (rcov[:, None] + rcov[None, :]) + 0.35
    U = np.full((n, n), 60.0, dtype=np.float64)
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(U, 0.0)
    for (u, v), r0 in zip(b_idx.tolist(), b_r0.tolist()):
        L[u, v] = L[v, u] = U[u, v] = U[v, u] = r0
    for i, j, k, th0 in ang_list:
        rij, rjk = r0_map[(min(i, j), max(i, j))], r0_map[(min(j, k), max(j, k))]
        dik = math.sqrt(max(rij * rij + rjk * rjk - 2.0 * rij * rjk * math.cos(th0), 1e-6))
        if (min(i, k), max(i, k)) not in bo_map:
            L[i, k] = L[k, i] = min(U[i, k], dik) if U[i, k] < 59.0 else dik
            U[i, k] = U[k, i] = max(L[i, k], dik) if U[i, k] < 59.0 else dik
    for i, l, d_lo, d_hi in dg_14_sp3:
        if (min(i, l), max(i, l)) not in excl_12_13:
            L[i, l] = L[l, i] = max(L[i, l], min(d_lo, d_hi) - 0.05)
            U[i, l] = U[l, i] = min(U[i, l], max(d_lo, d_hi) + 0.05)
    for i, l, d_ex in dg_14_ex:
        if (min(i, l), max(i, l)) not in excl_12_13:
            L[i, l] = L[l, i] = d_ex - 0.04
            U[i, l] = U[l, i] = d_ex + 0.04
    for k in range(n):
        U = np.minimum(U, U[:, k : k + 1] + U[k : k + 1, :])
    for k in range(n):
        L = np.maximum(L, np.maximum(L[:, k : k + 1] - U[k : k + 1, :], L[k : k + 1, :] - U[:, k : k + 1]))
    L = np.minimum(L, U)
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(U, 0.0)

    best_pos, best_e = None, math.inf
    for trial in range(4):
        rng = np.random.default_rng(seed + 13 * trial)
        w = rng.uniform(0.35, 0.65, size=(n, n))
        D = L + 0.5 * (w + w.T) * (U - L)
        np.fill_diagonal(D, 0.0)
        D2 = D * D
        rm = D2.mean(axis=1)
        ev, ec = np.linalg.eigh(0.5 * (rm[:, None] + rm[None, :] - float(D2.mean()) - D2))
        init = ec[:, -3:][:, ::-1] * np.sqrt(np.maximum(ev[-3:][::-1], 1e-2))[None, :] + rng.normal(0.0, 0.08, size=(n, 3))
        opt = lbfgs_minimize(init.reshape(-1), ff, max_iter=max_iter, gtol=0.05).reshape(n, 3)
        e_val, _ = ff(opt.reshape(-1))
        ch_ok = all(sgn * float(np.dot(opt[a] - opt[d], np.cross(opt[b] - opt[d], opt[c] - opt[d]))) > 0.5 for a, b, c, d, sgn in chiral_specs)
        nb_ok = not len(nb_idx) or bool(np.all(np.linalg.norm(opt[nb_idx[:, 0]] - opt[nb_idx[:, 1]], axis=1) >= 0.78 * (rcov[nb_idx[:, 0]] + rcov[nb_idx[:, 1]])))
        if ch_ok and nb_ok and e_val < 1.5 * n:
            return opt
        if e_val < best_e:
            best_e, best_pos = e_val, opt
    assert best_pos is not None
    return best_pos


def _embed_component(
    numbers: np.ndarray, charges: np.ndarray, bonds: list[tuple[int, int, float, bool]],
    ez_specs: list[tuple[int, int, int, int, float]], chiral_specs: list[tuple[int, int, int, int, float]],
    arom_atoms: np.ndarray, seed: int = 7,
) -> np.ndarray:
    n = len(numbers)
    if n == 1:
        return np.zeros((1, 3), dtype=np.float64)
    if n == 2:
        u, v, bo, ia = bonds[0]
        return np.array([[0.0, 0.0, 0.0], [_bond_length(int(numbers[u]), int(numbers[v]), bo, ia), 0.0, 0.0]])

    adj: list[list[int]] = [[] for _ in range(n)]
    bo_map, arom_bond, r0_map = {}, {}, {}
    for u, v, bo, ia in bonds:
        key = (u, v) if u < v else (v, u)
        adj[u].append(v)
        adj[v].append(u)
        bo_map[key], arom_bond[key] = bo, ia
        r0_map[key] = _bond_length(int(numbers[u]), int(numbers[v]), bo, ia)
    hyb = _hybridizations(numbers, charges, adj, bo_map, arom_atoms)

    stereo_h = {x for a, b, c, d, _ in chiral_specs + ez_specs for x in (a, b, c, d) if numbers[x] == 1}
    core = [i for i in range(n) if not (numbers[i] == 1 and len(adj[i]) == 1 and i not in stereo_h)]
    if 3 <= len(core) < n:
        cmap = {g: k for k, g in enumerate(core)}
        c_adj: list[list[int]] = [[] for _ in range(len(core))]
        c_bo, c_ar, c_r0 = {}, {}, {}
        for (u, v), bo in bo_map.items():
            if u in cmap and v in cmap:
                cu, cv = cmap[u], cmap[v]
                ck = (min(cu, cv), max(cu, cv))
                c_adj[cu].append(cv)
                c_adj[cv].append(cu)
                c_bo[ck], c_ar[ck], c_r0[ck] = bo, arom_bond[(u, v)], r0_map[(u, v)]
        c_ez = [(cmap[i], cmap[u], cmap[v], cmap[l], s) for i, u, v, l, s in ez_specs if i in cmap and l in cmap]
        c_ch = [(cmap[a], cmap[b], cmap[c], cmap[d], s) for a, b, c, d, s in chiral_specs if all(x in cmap for x in (a, b, c, d))]
        c_pos = _dg_embed_graph(numbers[core], [hyb[g] for g in core], c_adj, c_bo, c_ar, c_r0, c_ez, c_ch, seed, max_iter=80)
        pos = np.zeros((n, 3), dtype=np.float64)
        placed = [False] * n
        for k, g in enumerate(core):
            pos[g] = c_pos[k]
            placed[g] = True
        place_terminal_h(pos, placed, numbers, hyb, adj, r0_map, _ideal_angle_rad)
        full_ff = _build_terms(numbers, hyb, adj, bo_map, arom_bond, r0_map, ez_specs, chiral_specs)[0]
        opt = lbfgs_minimize(pos.reshape(-1), full_ff, max_iter=35, gtol=0.08).reshape(n, 3)
    else:
        opt = _dg_embed_graph(numbers, hyb, adj, bo_map, arom_bond, r0_map, ez_specs, chiral_specs, seed, max_iter=90)
    return opt - opt.mean(axis=0, keepdims=True)


def build_3d_coordinates(
    numbers: np.ndarray, charges: np.ndarray, bonds: list[tuple[int, int, float, bool]],
    ez_specs: list[tuple[int, int, int, int, float]], chiral_specs: list[tuple[int, int, int, int, float]],
    arom_atoms: np.ndarray, seed: int = 7,
) -> np.ndarray:
    """Embed one or more disconnected molecular fragments into clash-free 3D Cartesian coordinates (Å)."""
    n = len(numbers)
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v, _, _ in bonds:
        adj[u].append(v)
        adj[v].append(u)
    comp_id, comps = [-1] * n, []
    for i in range(n):
        if comp_id[i] >= 0:
            continue
        cid = len(comps)
        q, comp_id[i], nodes = deque([i]), cid, []
        while q:
            u = q.popleft()
            nodes.append(u)
            for v in adj[u]:
                if comp_id[v] < 0:
                    comp_id[v] = cid
                    q.append(v)
        comps.append(nodes)

    pos = np.zeros((n, 3), dtype=np.float64)
    x_cursor = 0.0
    for cid, nodes in enumerate(comps):
        loc = {g: k for k, g in enumerate(nodes)}
        sub_b = [(loc[u], loc[v], bo, ia) for u, v, bo, ia in bonds if comp_id[u] == cid]
        sub_ez = [(loc[i], loc[u], loc[v], loc[l], s) for i, u, v, l, s in ez_specs if comp_id[u] == cid]
        sub_ch = [(loc[a], loc[b], loc[c], loc[d], s) for a, b, c, d, s in chiral_specs if comp_id[a] == cid]
        sub_pos = _embed_component(numbers[nodes], charges[nodes], sub_b, sub_ez, sub_ch, arom_atoms[nodes], seed=seed + cid)
        if cid > 0:
            sub_pos[:, 0] += x_cursor + 3.6 - float(sub_pos[:, 0].min())
        x_cursor = float(sub_pos[:, 0].max())
        pos[nodes] = sub_pos
    return pos
