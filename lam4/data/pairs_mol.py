"""pairs_mol.py: Vectorized work-consistent close aligned molecular pair indexing for `OmolSource`."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from lam4.data.pairs import DEFAULT_PAIR_CACHE_DIR, _cache_file, _cap_per_group


def _hash_z_sequences(flat, rows: np.ndarray, n_at: np.ndarray, chunk_structs: int = 100_000) -> np.ndarray:
    """Vectorized order-sensitive 64-bit hash of each structure's exact Z sequence."""
    n = len(rows)
    out = np.zeros(n, dtype=np.uint64)
    c1 = np.uint64(0x9E3779B97F4A7C15)
    c2 = np.uint64(0xBF58476D1CE4E5B9)
    c3 = np.uint64(0x94D049BB133111EB)
    for s in range(0, n, chunk_structs):
        e = min(n, s + chunk_structs)
        rs, ns = rows[s:e], n_at[s:e]
        m = int(ns.sum())
        if m == 0:
            continue
        starts = np.zeros(len(rs), dtype=np.int64)
        if len(rs) > 1:
            np.cumsum(ns[:-1], out=starts[1:])
        arange_m = np.arange(m, dtype=np.int64)
        atoms = np.repeat(np.asarray(flat.offsets[rs], dtype=np.int64) - starts, ns) + arange_m
        within = (arange_m - np.repeat(starts, ns)).astype(np.uint64)
        z_u = np.asarray(flat.z[atoms], dtype=np.uint64)
        with np.errstate(over="ignore"):
            mix = (within * c1 + c2) * (z_u + np.uint64(1))
            mix = (mix ^ (mix >> np.uint64(30))) * c3
            out[s:e] = np.add.reduceat(mix, starts)
    return out


def _centered_rg(flat, rows: np.ndarray, n_at: np.ndarray, chunk_structs: int = 100_000) -> np.ndarray:
    """Vectorized radius of gyration Rg = sqrt(mean_i ||pos_i - mean(pos)||^2)."""
    n = len(rows)
    out = np.zeros(n, dtype=np.float64)
    for s in range(0, n, chunk_structs):
        e = min(n, s + chunk_structs)
        rs, ns = rows[s:e], n_at[s:e]
        m = int(ns.sum())
        if m == 0:
            continue
        starts = np.zeros(len(rs), dtype=np.int64)
        if len(rs) > 1:
            np.cumsum(ns[:-1], out=starts[1:])
        arange_m = np.arange(m, dtype=np.int64)
        atoms = np.repeat(np.asarray(flat.offsets[rs], dtype=np.int64) - starts, ns) + arange_m
        pos = np.asarray(flat.pos[atoms], dtype=np.float64)
        com = np.add.reduceat(pos, starts, axis=0) / np.maximum(ns[:, None], 1)
        pos -= np.repeat(com, ns, axis=0)
        msd = np.add.reduceat(np.sum(pos * pos, axis=-1), starts) / np.maximum(ns, 1)
        out[s:e] = np.sqrt(np.maximum(msd, 0.0))
    return out


def _mol_pair_rmsd_chunk(flat, row_a: np.ndarray, row_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized exact-Z check, centered raw (unrotated) RMSD, and trapezoidal work residual (kcal/mol/atom)."""
    c = len(row_a)
    z_ok = np.zeros(c, dtype=bool)
    rmsd = np.full(c, np.inf, dtype=np.float64)
    work_kcal = np.full(c, np.inf, dtype=np.float64)
    if c == 0:
        return z_ok, rmsd, work_kcal
    na = np.asarray(flat.n_atoms[row_a], dtype=np.int64)
    nb = np.asarray(flat.n_atoms[row_b], dtype=np.int64)
    sub = np.flatnonzero((na == nb) & (na > 0))
    if len(sub) == 0:
        return z_ok, rmsd, work_kcal
    ra, rb, ns = row_a[sub], row_b[sub], na[sub]
    m = int(ns.sum())
    starts = np.zeros(len(sub), dtype=np.int64)
    if len(sub) > 1:
        np.cumsum(ns[:-1], out=starts[1:])
    arange_m = np.arange(m, dtype=np.int64)
    idx_a = np.repeat(np.asarray(flat.offsets[ra], dtype=np.int64) - starts, ns) + arange_m
    idx_b = np.repeat(np.asarray(flat.offsets[rb], dtype=np.int64) - starts, ns) + arange_m
    z_diff = (np.asarray(flat.z[idx_a]) != np.asarray(flat.z[idx_b])).astype(np.int64)
    same_z = np.add.reduceat(z_diff, starts) == 0
    if not same_z.any():
        return z_ok, rmsd, work_kcal
    if not same_z.all():
        sub = sub[same_z]
        ra, rb, ns = ra[same_z], rb[same_z], ns[same_z]
        m = int(ns.sum())
        starts = np.zeros(len(sub), dtype=np.int64)
        if len(sub) > 1:
            np.cumsum(ns[:-1], out=starts[1:])
        arange_m = np.arange(m, dtype=np.int64)
        idx_a = np.repeat(np.asarray(flat.offsets[ra], dtype=np.int64) - starts, ns) + arange_m
        idx_b = np.repeat(np.asarray(flat.offsets[rb], dtype=np.int64) - starts, ns) + arange_m
    pa = np.asarray(flat.pos[idx_a], dtype=np.float64)
    pb = np.asarray(flat.pos[idx_b], dtype=np.float64)
    dpos = pb - pa
    mean_d = np.add.reduceat(dpos, starts, axis=0) / ns[:, None]
    dpos -= np.repeat(mean_d, ns, axis=0)
    msd = np.add.reduceat(np.sum(dpos * dpos, axis=-1), starts) / ns
    fa = np.asarray(flat.forces[idx_a], dtype=np.float64)
    fb = np.asarray(flat.forces[idx_b], dtype=np.float64)
    de = np.asarray(flat.energy[rb], dtype=np.float64) - np.asarray(flat.energy[ra], dtype=np.float64)
    wf = -0.5 * np.add.reduceat(np.sum((fa + fb) * dpos, axis=-1), starts)
    res = np.abs(de - wf) / ns
    z_ok[sub] = np.isfinite(res)
    rmsd[sub] = np.sqrt(np.maximum(msd, 0.0))
    work_kcal[sub] = res
    return z_ok, rmsd, work_kcal


def build_omol_pairs(
    src,
    *,
    rmsd_min: float = 0.01,
    rmsd_max: float = 0.25,
    work_tol_kcal: float = 0.05,
    max_pairs_per_group: int = 8,
    chunk_pairs: int = 32768,
    cache_dir: str | Path | None = DEFAULT_PAIR_CACHE_DIR,
) -> tuple[dict[int, np.ndarray], dict]:
    """Build (or load cached) work-consistent close aligned molecular pairs per size bucket `cap -> pairs [M_cap, 2]`."""
    t0 = time.perf_counter()
    n_src = len(src)
    key = (
        "omol_v2_work",
        str(src.flat.path.resolve()),
        len(src.flat),
        n_src,
        int(src.idx[0]) if n_src else 0,
        int(src.idx[-1]) if n_src else 0,
        int(src.idx.sum()) if n_src else 0,
        tuple(int(e) for e in src.edges),
        round(float(rmsd_min), 6),
        round(float(rmsd_max), 6),
        round(float(work_tol_kcal), 6),
        int(max_pairs_per_group),
    )
    cf = _cache_file(cache_dir, "mol_pairs", key)
    if cf is not None and cf.exists():
        try:
            d = np.load(cf)
            out = {int(c): np.asarray(d[f"cap_{c}"], dtype=np.int64).reshape(-1, 2) for c in src.buckets if f"cap_{c}" in d}
            stats = {
                "n_structs": n_src,
                "n_pairs": int(sum(len(v) for v in out.values())),
                "n_candidates": int(d["n_candidates"]),
                "by_cap": {int(c): int(len(v)) for c, v in out.items()},
                "cached": True,
                "build_s": round(time.perf_counter() - t0, 3),
            }
            return out, stats
        except Exception:
            pass

    out: dict[int, np.ndarray] = {}
    tot_cand = 0
    cand_cap = max(max_pairs_per_group * 8, 64)
    flat = src.flat
    for cap, bpos in src.buckets.items():
        if len(bpos) < 2:
            out[int(cap)] = np.zeros((0, 2), dtype=np.int64)
            continue
        rows = src.idx[bpos]
        na = src.n_at[bpos]
        q = np.round(src.charge[bpos]).astype(np.int64)
        sp = np.round(src.spin[bpos]).astype(np.int64)
        zh = _hash_z_sequences(flat, rows, na)

        order = np.lexsort((zh, sp, q, na))
        n_s, q_s, sp_s, zh_s = na[order], q[order], sp[order], zh[order]
        same = (n_s[1:] == n_s[:-1]) & (q_s[1:] == q_s[:-1]) & (sp_s[1:] == sp_s[:-1]) & (zh_s[1:] == zh_s[:-1])
        if not same.any():
            out[int(cap)] = np.zeros((0, 2), dtype=np.int64)
            continue

        run_id = np.zeros(len(order), dtype=np.int64)
        run_id[1:] = np.cumsum(~same)
        in_multi = np.zeros(len(order), dtype=bool)
        in_multi[:-1] |= same
        in_multi[1:] |= same
        m_ord = order[in_multi]
        m_rid = run_id[in_multi]

        rg = _centered_rg(flat, rows[m_ord], na[m_ord])
        ca_l, cb_l, cg_l, prio_l = [], [], [], []
        for stride in (1, 2):
            if len(m_ord) <= stride:
                break
            ok_st = (m_rid[stride:] == m_rid[:-stride]) & (np.abs(rg[stride:] - rg[:-stride]) <= rmsd_max)
            if ok_st.any():
                ii = np.flatnonzero(ok_st)
                ca_l.append(m_ord[ii])
                cb_l.append(m_ord[ii + stride])
                cg_l.append(m_rid[ii])
                prio_l.append(np.abs(rg[ii + stride] - rg[ii]))

        rg_ord = np.lexsort((rg, m_rid))
        r_idx = m_ord[rg_ord]
        r_rid = m_rid[rg_ord]
        r_rg = rg[rg_ord]
        for stride in (1, 2, 3, 4):
            if len(r_idx) <= stride:
                break
            drg = np.abs(r_rg[stride:] - r_rg[:-stride])
            ok_st = (r_rid[stride:] == r_rid[:-stride]) & (drg <= rmsd_max)
            if ok_st.any():
                ii = np.flatnonzero(ok_st)
                ca_l.append(r_idx[ii])
                cb_l.append(r_idx[ii + stride])
                cg_l.append(r_rid[ii])
                prio_l.append(drg[ii] + 0.5)

        if not ca_l:
            out[int(cap)] = np.zeros((0, 2), dtype=np.int64)
            continue
        ca = np.concatenate(ca_l)
        cb = np.concatenate(cb_l)
        cg = np.concatenate(cg_l)
        prio = np.concatenate(prio_l)
        lo = np.minimum(ca, cb)
        hi = np.maximum(ca, cb)
        p_ord = np.argsort(prio, kind="stable")
        lo, hi, cg = lo[p_ord], hi[p_ord], cg[p_ord]
        pair_key = lo.astype(np.uint64) * np.uint64(len(bpos)) + hi.astype(np.uint64)
        _, uniq_idx = np.unique(pair_key, return_index=True)
        uniq_idx.sort()
        lo, hi, cg = lo[uniq_idx], hi[uniq_idx], cg[uniq_idx]
        keep_c = _cap_per_group(cg, cand_cap, evenly_spaced=False)
        lo, hi, cg = lo[keep_c], hi[keep_c], cg[keep_c]
        tot_cand += len(lo)

        acc_a, acc_b, acc_g = [], [], []
        for s in range(0, len(lo), chunk_pairs):
            e = min(len(lo), s + chunk_pairs)
            ra, rb = rows[lo[s:e]], rows[hi[s:e]]
            z_ok, rmsd, w_kcal = _mol_pair_rmsd_chunk(flat, ra, rb)
            ok = z_ok & (rmsd >= rmsd_min) & (rmsd <= rmsd_max)
            if work_tol_kcal > 0.0:
                ok &= (w_kcal <= work_tol_kcal)
            if ok.any():
                acc_a.append(bpos[lo[s:e][ok]])
                acc_b.append(bpos[hi[s:e][ok]])
                acc_g.append(cg[s:e][ok])

        if acc_a:
            pa = np.concatenate(acc_a)
            pb = np.concatenate(acc_b)
            pg = np.concatenate(acc_g)
            keep_g = _cap_per_group(pg, max_pairs_per_group, evenly_spaced=True)
            out[int(cap)] = np.stack([pa[keep_g], pb[keep_g]], axis=1).astype(np.int64)
        else:
            out[int(cap)] = np.zeros((0, 2), dtype=np.int64)

    stats = {
        "n_structs": n_src,
        "n_pairs": int(sum(len(v) for v in out.values())),
        "n_candidates": int(tot_cand),
        "by_cap": {int(c): int(len(v)) for c, v in out.items()},
        "cached": False,
        "build_s": round(time.perf_counter() - t0, 3),
    }
    if cf is not None:
        try:
            cf.parent.mkdir(parents=True, exist_ok=True)
            tmp = cf.with_name(cf.name + f".{os.getpid()}.tmp.npz")
            np.savez(tmp, n_candidates=tot_cand, **{f"cap_{c}": v for c, v in out.items()})
            os.replace(tmp, cf)
        except Exception:
            pass
    return out, stats
