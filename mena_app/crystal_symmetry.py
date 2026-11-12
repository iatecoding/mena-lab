"""Pure-NumPy 3D crystal space-group operations, equivalent-atom orbits, and point-group classification."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

# Exact (det, trace) multiset signatures for all 32 crystallographic point groups (Hermann-Mauguin, crystal system).
_PG_TABLE: list[tuple[str, str, dict[tuple[int, int], int]]] = [
    ("1", "triclinic", {(1, 3): 1}),
    ("-1", "triclinic", {(1, 3): 1, (-1, -3): 1}),
    ("2", "monoclinic", {(1, 3): 1, (1, -1): 1}),
    ("m", "monoclinic", {(1, 3): 1, (-1, 1): 1}),
    ("2/m", "monoclinic", {(1, 3): 1, (1, -1): 1, (-1, -3): 1, (-1, 1): 1}),
    ("222", "orthorhombic", {(1, 3): 1, (1, -1): 3}),
    ("mm2", "orthorhombic", {(1, 3): 1, (1, -1): 1, (-1, 1): 2}),
    ("mmm", "orthorhombic", {(1, 3): 1, (1, -1): 3, (-1, -3): 1, (-1, 1): 3}),
    ("4", "tetragonal", {(1, 3): 1, (1, 1): 2, (1, -1): 1}),
    ("-4", "tetragonal", {(1, 3): 1, (1, -1): 1, (-1, -1): 2}),
    ("4/m", "tetragonal", {(1, 3): 1, (1, 1): 2, (1, -1): 1, (-1, -3): 1, (-1, -1): 2, (-1, 1): 1}),
    ("422", "tetragonal", {(1, 3): 1, (1, 1): 2, (1, -1): 5}),
    ("4mm", "tetragonal", {(1, 3): 1, (1, 1): 2, (1, -1): 1, (-1, 1): 4}),
    ("-42m", "tetragonal", {(1, 3): 1, (1, -1): 3, (-1, -1): 2, (-1, 1): 2}),
    ("4/mmm", "tetragonal", {(1, 3): 1, (1, 1): 2, (1, -1): 5, (-1, -3): 1, (-1, -1): 2, (-1, 1): 5}),
    ("3", "trigonal", {(1, 3): 1, (1, 0): 2}),
    ("-3", "trigonal", {(1, 3): 1, (1, 0): 2, (-1, -3): 1, (-1, 0): 2}),
    ("32", "trigonal", {(1, 3): 1, (1, 0): 2, (1, -1): 3}),
    ("3m", "trigonal", {(1, 3): 1, (1, 0): 2, (-1, 1): 3}),
    ("-3m", "trigonal", {(1, 3): 1, (1, 0): 2, (1, -1): 3, (-1, -3): 1, (-1, 0): 2, (-1, 1): 3}),
    ("6", "hexagonal", {(1, 3): 1, (1, 2): 2, (1, 0): 2, (1, -1): 1}),
    ("-6", "hexagonal", {(1, 3): 1, (1, 0): 2, (-1, -2): 2, (-1, 1): 1}),
    ("6/m", "hexagonal", {(1, 3): 1, (1, 2): 2, (1, 0): 2, (1, -1): 1, (-1, -3): 1, (-1, -2): 2, (-1, 0): 2, (-1, 1): 1}),
    ("622", "hexagonal", {(1, 3): 1, (1, 2): 2, (1, 0): 2, (1, -1): 7}),
    ("6mm", "hexagonal", {(1, 3): 1, (1, 2): 2, (1, 0): 2, (1, -1): 1, (-1, 1): 6}),
    ("-6m2", "hexagonal", {(1, 3): 1, (1, 0): 2, (1, -1): 3, (-1, -2): 2, (-1, 1): 4}),
    ("6/mmm", "hexagonal", {(1, 3): 1, (1, 2): 2, (1, 0): 2, (1, -1): 7, (-1, -3): 1, (-1, -2): 2, (-1, 0): 2, (-1, 1): 7}),
    ("23", "cubic", {(1, 3): 1, (1, 0): 8, (1, -1): 3}),
    ("m-3", "cubic", {(1, 3): 1, (1, 0): 8, (1, -1): 3, (-1, -3): 1, (-1, 0): 8, (-1, 1): 3}),
    ("432", "cubic", {(1, 3): 1, (1, 1): 6, (1, 0): 8, (1, -1): 9}),
    ("-43m", "cubic", {(1, 3): 1, (1, 0): 8, (1, -1): 3, (-1, -1): 6, (-1, 1): 6}),
    ("m-3m", "cubic", {(1, 3): 1, (1, 1): 6, (1, 0): 8, (1, -1): 9, (-1, -3): 1, (-1, -1): 6, (-1, 0): 8, (-1, 1): 9}),
]
_POINT_GROUPS = {tuple(sorted(sig.items())): (hm, cs) for hm, cs, sig in _PG_TABLE}

_V26 = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i or j or k)],
    dtype=np.int64,
)


def _det3_int(m: np.ndarray) -> np.ndarray:
    return (
        m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
        - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
        + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0])
    )


def _inv3_unimodular(u: np.ndarray) -> np.ndarray:
    det = int(_det3_int(u))
    adj = np.array(
        [
            [
                u[1, 1] * u[2, 2] - u[1, 2] * u[2, 1],
                u[0, 2] * u[2, 1] - u[0, 1] * u[2, 2],
                u[0, 1] * u[1, 2] - u[0, 2] * u[1, 1],
            ],
            [
                u[1, 2] * u[2, 0] - u[1, 0] * u[2, 2],
                u[0, 0] * u[2, 2] - u[0, 2] * u[2, 0],
                u[0, 2] * u[1, 0] - u[0, 0] * u[1, 2],
            ],
            [
                u[1, 0] * u[2, 1] - u[1, 1] * u[2, 0],
                u[0, 1] * u[2, 0] - u[0, 0] * u[2, 1],
                u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0],
            ],
        ],
        dtype=np.int64,
    )
    return det * adj


def _minkowski_reduce_3d(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return `(B, U)` with `B = U @ cell`, `|det U| = 1`, and `B` Minkowski-reduced in 3D."""
    U = np.eye(3, dtype=np.int64)
    B = np.asarray(cell, dtype=np.float64).copy()
    shifts2 = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)], dtype=np.int64)
    for _ in range(64):
        n2 = np.einsum("ij,ij->i", B, B)
        order = np.argsort(n2, kind="mergesort")
        if not np.array_equal(order, [0, 1, 2]):
            B = B[order]
            U = U[order]
            n2 = n2[order]
        q01 = int(round(float(np.dot(B[1], B[0])) / max(float(n2[0]), 1e-30)))
        if q01 != 0:
            new_b1 = B[1] - q01 * B[0]
            if float(np.dot(new_b1, new_b1)) < float(n2[1]) - 1e-12 * float(n2[1]):
                B[1] = new_b1
                U[1] -= q01 * U[0]
                continue
        g00, g01, g11 = float(n2[0]), float(np.dot(B[0], B[1])), float(n2[1])
        denom = g00 * g11 - g01 * g01
        if denom <= 1e-24:
            break
        p0, p1 = float(np.dot(B[2], B[0])), float(np.dot(B[2], B[1]))
        c0 = int(round((g11 * p0 - g01 * p1) / denom))
        c1 = int(round((g00 * p1 - g01 * p0) / denom))
        cands = shifts2 + np.array([c0, c1], dtype=np.int64)
        trials = B[2][None, :] - cands[:, 0:1] * B[0][None, :] - cands[:, 1:2] * B[1][None, :]
        t_n2 = np.einsum("ij,ij->i", trials, trials)
        best = int(np.argmin(t_n2))
        q0, q1 = int(cands[best, 0]), int(cands[best, 1])
        if (q0 != 0 or q1 != 0) and float(t_n2[best]) < float(n2[2]) - 1e-12 * float(n2[2]):
            B[2] = trials[best]
            U[2] -= q0 * U[0] + q1 * U[1]
            continue
        break
    return B, U


def _lattice_rotations(cell: np.ndarray, symprec: float) -> np.ndarray:
    """All integer `3x3` matrices `R` in the original cell basis whose Cartesian action `C^T R C^-T` is orthogonal."""
    B, U = _minkowski_reduce_3d(cell)
    U_inv = _inv3_unimodular(U)
    vecs = _V26 @ B
    vlens = np.linalg.norm(vecs, axis=1)
    blens = np.linalg.norm(B, axis=1)
    row_cands = [
        _V26[np.abs(vlens - blens[i]) <= max(3.0 * symprec, 1e-4 * blens[i])]
        for i in range(3)
    ]
    if any(len(rc) == 0 for rc in row_cands):
        return np.eye(3, dtype=np.int64)[None, :, :]
    i0, i1, i2 = np.meshgrid(
        np.arange(len(row_cands[0])),
        np.arange(len(row_cands[1])),
        np.arange(len(row_cands[2])),
        indexing="ij",
    )
    W_red = np.stack(
        [row_cands[0][i0.ravel()], row_cands[1][i1.ravel()], row_cands[2][i2.ravel()]],
        axis=1,
    )
    W_red = W_red[np.abs(_det3_int(W_red)) == 1]
    if len(W_red) == 0:
        return np.eye(3, dtype=np.int64)[None, :, :]
    W_orig = np.einsum("ij,mjk,kl->mil", U_inv, W_red, U)
    R_orig = np.swapaxes(W_orig, -1, -2)
    cT = cell.T
    cT_inv = np.linalg.inv(cT)
    Rc = np.einsum("ij,mjk,kl->mil", cT, R_orig, cT_inv)
    orth_err = np.max(np.abs(np.einsum("mji,mjk->mik", Rc, Rc) - np.eye(3)), axis=(1, 2))
    l_min = max(float(blens.min()), 1e-6)
    keep = orth_err <= max(6.0 * symprec / l_min, 1e-4)
    W_red, R_orig, Rc = W_red[keep], R_orig[keep], Rc[keep]
    if len(R_orig) == 0:
        return np.eye(3, dtype=np.int64)[None, :, :]
    u_s, _, vt_s = np.linalg.svd(Rc)
    Q = u_s @ vt_s
    lat_err = np.linalg.norm((W_red @ B) - np.einsum("ij,mkj->mik", B, Q), axis=-1).max(axis=1)
    R_orig = R_orig[lat_err <= max(2.0 * symprec, 1e-5)]
    if len(R_orig) == 0:
        return np.eye(3, dtype=np.int64)[None, :, :]
    return R_orig.astype(np.int64)


def classify_pointgroup(rotations: np.ndarray) -> tuple[str, str, int]:
    """Identify `(pointgroup_hm, crystal_system, n_pointgroup)` from `[G, 3, 3]` integer rotation matrices."""
    uniq = np.unique(rotations.reshape(len(rotations), 9), axis=0).reshape(-1, 3, 3)
    dets = _det3_int(uniq).tolist()
    trs = np.trace(uniq, axis1=1, axis2=2).tolist()
    sig = tuple(sorted(Counter(zip(dets, trs)).items()))
    hm, cs = _POINT_GROUPS.get(sig, (f"{len(uniq)}", "triclinic"))
    return hm, cs, len(uniq)


def find_crystal_symmetry(
    cell: np.ndarray,
    frac: np.ndarray,
    numbers: np.ndarray,
    symprec: float = 1e-3,
) -> dict[str, Any] | None:
    """Find space-group operations `(rotations, translations)`, `equivalent_atoms`, and point-group info in pure NumPy."""
    C = np.asarray(cell, dtype=np.float64)
    F = np.asarray(frac, dtype=np.float64)
    Z = np.asarray(numbers, dtype=np.int64)
    n = len(Z)
    if C.shape != (3, 3) or n == 0 or F.shape != (n, 3):
        return None
    if not (np.isfinite(C).all() and np.isfinite(F).all()) or abs(float(np.linalg.det(C))) <= 1e-8:
        return None

    frac_w = F % 1.0
    lat_rots = _lattice_rotations(C, symprec)
    vals, cnts = np.unique(Z, return_counts=True)
    order_z = vals[np.argsort(cnts, kind="mergesort")]
    by_z = [np.flatnonzero(Z == z) for z in order_z]
    rare_idx = by_z[0]
    p0 = int(rare_idx[0])

    loose2 = (2.5 * symprec + 1e-9) ** 2
    tight = symprec + 1e-9
    eye3 = np.eye(3, dtype=np.int64)

    ops_r: list[np.ndarray] = []
    ops_t: list[np.ndarray] = []
    ops_p: list[np.ndarray] = []

    for R in lat_rots:
        p0_rot = frac_w[p0] @ R.T
        cand_ts = (frac_w[rare_idx] - p0_rot) % 1.0
        accepted_ts: list[np.ndarray] = []
        for t0 in cand_ts:
            perm = np.empty(n, dtype=np.int64)
            ok = True
            for idx_z in by_z:
                m = len(idx_z)
                img = frac_w[idx_z] @ R.T + t0
                df = img[:, None, :] - frac_w[None, idx_z, :]
                df -= np.round(df)
                d2 = np.einsum("ijk,ijk->ij", df @ C, df @ C)
                best = np.argmin(d2, axis=1)
                if float(d2[np.arange(m), best].max()) > loose2 or np.bincount(best, minlength=m).min() == 0:
                    ok = False
                    break
                perm[idx_z] = idx_z[best]
            if not ok:
                continue
            res = frac_w[perm] - (frac_w @ R.T + t0)
            res -= np.round(res)
            t = (t0 + res.mean(axis=0)) % 1.0
            t[np.isclose(t, 1.0, atol=1e-12) | np.isclose(t, 0.0, atol=1e-12)] = 0.0
            df_ref = (frac_w @ R.T + t) - frac_w[perm]
            df_ref -= np.round(df_ref)
            if float(np.linalg.norm(df_ref @ C, axis=-1).max()) > tight:
                continue
            if any(float(np.linalg.norm(((t - ta) - np.round(t - ta)) @ C)) <= tight for ta in accepted_ts):
                continue
            accepted_ts.append(t)
            ops_r.append(R)
            ops_t.append(t)
            ops_p.append(perm)

    if not ops_r:
        ops_r = [eye3]
        ops_t = [np.zeros(3, dtype=np.float64)]
        ops_p = [np.arange(n, dtype=np.int64)]

    # Place the identity operation (I, 0) at index 0.
    id_pos = next(
        (k for k, (r, t) in enumerate(zip(ops_r, ops_t)) if np.array_equal(r, eye3) and float(np.abs(t).max()) < 1e-7),
        0,
    )
    if id_pos != 0:
        ops_r[0], ops_r[id_pos] = ops_r[id_pos], ops_r[0]
        ops_t[0], ops_t[id_pos] = ops_t[id_pos], ops_t[0]
        ops_p[0], ops_p[id_pos] = ops_p[id_pos], ops_p[0]

    rotations = np.stack(ops_r, axis=0)
    translations = np.stack(ops_t, axis=0)
    perms = np.stack(ops_p, axis=0)

    eq = np.full(n, -1, dtype=np.int64)
    for b in range(n):
        if eq[b] < 0:
            reach = np.unique(perms[:, b])
            eq[reach[eq[reach] < 0]] = b

    pg, cs, n_pg = classify_pointgroup(rotations)
    g_tot = len(rotations)
    label = f"{pg} · {cs} ({g_tot} {'op' if g_tot == 1 else 'ops'})"
    return {
        "rotations": rotations,
        "translations": translations,
        "equivalent_atoms": eq,
        "pointgroup": pg,
        "crystal_system": cs,
        "n_pointgroup": n_pg,
        "n_operations": g_tot,
        "label": label,
    }
