"""Finite-displacement force constants from symmetry-irreducible displacements only. Pure numpy, no torch.

ASE's `Phonons` displaces every atom of the cell along ±x, ±y, ±z: 6N + 1 supercell evaluations. Equivalent atoms give
the same force response up to a rotation and a relabelling of the supercell atoms, and an atom's site symmetry relates
its responses along different directions. Here only one atom per orbit is displaced, along as few directions as its site
symmetry needs (1 for a cubic site, 3 for a general one), and every response the full method would have measured is
rebuilt from those: SrTiO3 takes 9 evaluations instead of 31.

Linear algebra, for a unit displacement of supercell atom b along d, the force change on atom j is M_b[j] d, with M_b[j]
a 3x3 block of minus the force constants. A space-group operation g = (R, t), with Cartesian rotation Rc, that maps b to
a and j to g(j), gives M_a[g(j)] = Rc M_b[j] Rc^T. For the site group of b (operations with g(b) = b) the measured
responses r_k = M_b d_k give M_b[g(j)] (Rc d_k) = Rc r_k[j]; collecting all site operations and measured directions,
M_b[j] follows by least squares (exact when the structure is symmetric; the symmetry average otherwise).

Equality with the full method: when the rotations map Cartesian axes onto axes (cubic, tetragonal, orthorhombic) the same
supercells are evaluated and the force constants agree to rounding. In hexagonal cells a 120° rotation infers y from x,
and the two central differences carry different O(delta^2) anharmonic errors: both approximate the same harmonic
constants, and they agree to within the finite-difference error of the full method (tests/mena_app/test_phonon_symmetry).

Supercells are the diagonal (k, k, k) repeats ASE builds: supercell atom ((m0 * k + m1) * k + m2) * n + p is primitive
atom p in cell m. An operation maps (m, p) to (R m + L_p mod k, perm_p), with L_p the lattice vector found on the
primitive cell, so no supercell-wide atom matching is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SYMPREC = 1e-3   # Å, Cartesian symmetry tolerance; a relaxed structure keeps its symmetry to well within this
CANDIDATES = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])


@dataclass
class Plan:
    """What to evaluate: for each orbit representative (supercell index), the directions to displace along."""
    k: int
    n: int
    offset: int                      # reference cell index (ASE's `offset`): atoms offset * n .. offset * n + n - 1
    rot_frac: np.ndarray             # [G, 3, 3] integer rotations in fractional coordinates of the primitive cell
    rot_cart: np.ndarray             # [G, 3, 3]
    perm_p: np.ndarray               # [G, n] image of each primitive atom
    lat_p: np.ndarray                # [G, n, 3] integer lattice shift of each image
    reps: list[int]                  # primitive index of each orbit representative
    orbit_of: np.ndarray             # [n] representative of each primitive atom
    directions: dict[int, np.ndarray]  # rep -> [m, 3] unit directions to measure

    @property
    def n_evaluations(self) -> int:
        return 1 + 2 * sum(len(d) for d in self.directions.values())


def symmetry_plan(atoms, k: int, offset: int = 0, symprec: float = SYMPREC, *, dataset: dict | None = None) -> Plan | None:
    """None when the crystal has only the identity operation (then the full 6N + 1 method is as cheap)."""
    from mena_app.crystal_symmetry import find_crystal_symmetry

    cell = np.asarray(atoms.cell.array, float)
    frac = atoms.get_scaled_positions(wrap=False)  # unwrapped: ASE's supercell atom (m, p) sits at frac[p] + m exactly
    if dataset is not None:
        ds = dataset
    else:
        try:
            ds = find_crystal_symmetry(cell, frac, atoms.numbers, symprec=symprec)
        except Exception:  # noqa: BLE001
            return None
    if ds is None:
        return None
    rots, trans = np.asarray(ds["rotations"]), np.asarray(ds["translations"])
    if len(rots) <= 1:
        return None
    n = len(atoms)
    G = len(rots)
    tol_frac = 10 * symprec / max(np.linalg.norm(cell, axis=1).min(), 1e-9) + 1e-6
    perms_in = ds.get("perms")
    if perms_in is not None and len(perms_in) == G and np.asarray(perms_in).shape == (G, n):
        perm = np.asarray(perms_in, dtype=int)
        img = np.einsum("nj,gij->gni", frac, rots) + trans[:, None, :]
        res = img - frac[perm]
        lat_arr = np.round(res)
        if float(np.abs(res - lat_arr).max()) > tol_frac:
            return None
        lat = lat_arr.astype(int)
    else:
        perm = np.empty((G, n), int)
        lat = np.empty((G, n, 3), int)
        for g in range(G):
            img = frac @ rots[g].T + trans[g]                       # [n, 3]
            diff = img[:, None, :] - frac[None, :, :]               # [n, n, 3]
            dist = np.abs(diff - np.round(diff)) @ np.abs(cell)     # rough Cartesian size of the residual
            q = dist.max(-1).argmin(1) if n > 1 else np.zeros(1, int)
            res = diff[np.arange(n), q]
            if np.abs(res - np.round(res)).max() > tol_frac:
                return None
            perm[g], lat[g] = q, np.round(res).astype(int)
    cT = cell.T
    rot_cart = np.einsum("ij,gjk,kl->gil", cT, rots, np.linalg.inv(cT))
    orbit_of = np.asarray(ds["equivalent_atoms"])
    reps = sorted(set(orbit_of.tolist()))
    directions = {b: _directions(rot_cart[perm[:, b] == b]) for b in reps}
    return Plan(k, n, offset, rots, rot_cart, perm, lat, reps, orbit_of, directions)


def _directions(site_rots: np.ndarray) -> np.ndarray:
    """Fewest Cartesian axes whose images under the site rotations span 3D."""
    def rank(dirs: list[np.ndarray]) -> int:
        return int(np.linalg.matrix_rank(np.concatenate([site_rots @ d for d in dirs]), tol=1e-6)) if dirs else 0

    chosen: list[np.ndarray] = []
    for d in CANDIDATES:
        if rank(chosen + [d]) > rank(chosen):
            chosen.append(d)
        if rank(chosen) == 3:
            break
    return np.array(chosen)


def supercell_perm(plan: Plan, g: int, shift: np.ndarray) -> np.ndarray:
    """[k^3 n] image of every supercell atom under operation g followed by the lattice translation `shift` (cell units)."""
    k, n = plan.k, plan.n
    m = np.stack(np.meshgrid(np.arange(k), np.arange(k), np.arange(k), indexing="ij"), -1).reshape(-1, 3)  # ASE order
    new_m = (m[:, None, :] @ plan.rot_frac[g].T)[:, 0, :][:, None, :] + plan.lat_p[g][None, :, :] + shift  # [k^3, n, 3]
    new_m %= k
    cell_idx = (new_m[..., 0] * k + new_m[..., 1]) * k + new_m[..., 2]
    return (cell_idx * n + plan.perm_p[g][None, :]).reshape(-1)


def _cell_of(plan: Plan, idx: int) -> np.ndarray:
    c = idx // plan.n
    return np.array([c // (plan.k * plan.k), (c // plan.k) % plan.k, c % plan.k])


def _op_to(plan: Plan, g: int, src: int, dst: int) -> np.ndarray:
    """Lattice shift so that operation g carries supercell atom src onto dst (same primitive atom by construction)."""
    m_src, m_dst = _cell_of(plan, src), _cell_of(plan, dst)
    p = src % plan.n
    m_img = plan.rot_frac[g] @ m_src + plan.lat_p[g][p]
    return m_dst - m_img


def rep_response(plan: Plan, b: int, measured: np.ndarray) -> np.ndarray:
    """M_b [N_sc, 3, 3] (column c = force change per Å of displacement along Cartesian c) from measured responses
    `measured` [m, N_sc, 3] along plan.directions[b], using b's site group."""
    b_sc = plan.offset * plan.n + b
    D, RHS = [], []
    for g in np.flatnonzero(plan.perm_p[:, b] == b):
        P = supercell_perm(plan, g, _op_to(plan, g, b_sc, b_sc))
        Rc = plan.rot_cart[g]
        for d, r in zip(plan.directions[b], measured):
            D.append(Rc @ d)
            img = np.empty_like(r)
            img[P] = r @ Rc.T                                    # force change on g(j) is Rc r[j]
            RHS.append(img)
    D = np.array(D).T                                            # [3, q]
    RHS = np.stack(RHS, -1)                                      # [N_sc, 3, q]
    return RHS @ np.linalg.pinv(D)                               # [N_sc, 3, 3]


def all_responses(plan: Plan, rep_M: dict[int, np.ndarray]) -> np.ndarray:
    """[n, N_sc, 3, 3]: M_a for every atom a of the reference cell, rotated from its orbit representative."""
    out = []
    for a in range(plan.n):
        b = int(plan.orbit_of[a])
        b_sc, a_sc = plan.offset * plan.n + b, plan.offset * plan.n + a
        g = int(np.flatnonzero(plan.perm_p[:, b] == a)[0])
        P = supercell_perm(plan, g, _op_to(plan, g, b_sc, a_sc))
        Rc = plan.rot_cart[g]
        Ma = np.empty_like(rep_M[b])
        Ma[P] = Rc @ rep_M[b] @ Rc.T
        out.append(Ma)
    return np.array(out)
