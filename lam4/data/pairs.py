"""pairs.py: Fast vectorized secant trajectory-difference pair indexing & static-shape partner tensors for crystals and molecules.

Crystals:
  Groups structures by `group_id.npy` / `mp_id.npy` (within each flat), `n_atoms`, and `u_flag`.
  Strict physics-based filters (eliminates MPtrj cross-calc basis-reset jumps without reading external parquets):
    - exact Z sequence match, identical pbc & u_flag
    - `frame_max_force <= max_force_ev` (default 1.0 eV/A) on both frames
    - `|ln(V_b / V_a)| <= max_dlnv` (default 0.10) and `||cell_b - cell_a||_F / ||cell_a||_F <= max_strain_rel` (default 0.10)
    - wrapped COM-subtracted RMSD <= rmsd_max (0.20 A) and (RMSD >= rmsd_min (0.01 A) or strain_rel >= min_strain_rel)
    - trapezoidal energy-force-stress work consistency:
        `|(E_b - E_a) + 0.5*(F_a + F_b)·dr_int - 0.5*(V_a*S_a + V_b*S_b):deps| / N <= work_tol_mev` (default 1.0 meV/atom)
      where `dr_int = wrap(frac_b - frac_a) @ cell_mid` (COM-subtracted) and `deps = inv(cell_mid) @ (cell_b - cell_a)`.
Molecules:
  Groups structures within each size bucket by exact `(Z_sequence, Q, S)` using a vectorized order-sensitive 64-bit hash + exact Z verification.
  Discovers close pairs via original-order neighbors and Rg-sorted neighbors, filtering:
    - centered raw (unrotated) RMSD in `[rmsd_min, rmsd_max]` (default `[0.01, 0.25]` A)
    - trapezoidal work consistency `|(E_b - E_a) + 0.5*(F_a + F_b)·dr| / N <= work_tol_kcal` (default 0.05 kcal/mol/atom)
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_PAIR_CACHE_DIR = Path(os.environ.get("LAM4_PAIR_CACHE_DIR") or (Path(tempfile.gettempdir()) / "lam4_pair_cache"))


def _cache_file(cache_dir: str | Path | None, prefix: str, key_parts: tuple) -> Path | None:
    if cache_dir is None or str(cache_dir) == "":
        return None
    d = Path(cache_dir)
    h = hashlib.sha256(repr(key_parts).encode("utf-8")).hexdigest()[:16]
    return d / f"{prefix}_{h}.npz"


def _cap_per_group(group_ids: np.ndarray, cap: int, *, evenly_spaced: bool = False) -> np.ndarray:
    """Boolean mask keeping at most `cap` items per group in `group_ids` (first `cap` or evenly spaced across each group)."""
    n = len(group_ids)
    if n == 0 or cap <= 0:
        return np.zeros(n, dtype=bool)
    order = np.argsort(group_ids, kind="stable")
    g = group_ids[order]
    starts = np.concatenate(([0], np.flatnonzero(g[1:] != g[:-1]) + 1))
    counts = np.diff(np.concatenate((starts, [n])))
    rank = np.arange(n, dtype=np.int64) - np.repeat(starts, counts)
    if evenly_spaced:
        cnt_rep = np.repeat(counts, counts)
        keep_ord = (cnt_rep <= cap) | (rank == 0) | (((rank * cap) // cnt_rep) != (((rank - 1) * cap) // cnt_rep))
    else:
        keep_ord = rank < cap
    keep = np.zeros(n, dtype=bool)
    keep[order[keep_ord]] = True
    return keep


def _crystal_pair_metrics_chunk(
    flat,
    row_a: np.ndarray,
    row_b: np.ndarray,
    *,
    max_strain_rel: float,
    max_dlnv: float,
    max_force_ev: float,
    fmax_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized exact-Z, cell-strain, max-force, wrapped RMSD, and trapezoidal work residual (meV/atom) for crystal pairs.
    Returns `(valid [C], rmsd_cart [C], strain_rel [C], work_res_mev [C])`."""
    c = len(row_a)
    valid = np.zeros(c, dtype=bool)
    rmsd_out = np.full(c, np.inf, dtype=np.float64)
    strain_rel = np.full(c, np.inf, dtype=np.float64)
    work_mev = np.full(c, np.inf, dtype=np.float64)
    if c == 0:
        return valid, rmsd_out, strain_rel, work_mev

    na = np.asarray(flat.n_atoms[row_a], dtype=np.int64)
    nb = np.asarray(flat.n_atoms[row_b], dtype=np.int64)
    cell_a = np.asarray(flat.cell[row_a], dtype=np.float64)
    cell_b = np.asarray(flat.cell[row_b], dtype=np.float64)
    pbc_a = np.asarray(flat.pbc[row_a], dtype=bool)
    pbc_b = np.asarray(flat.pbc[row_b], dtype=bool)
    cell_mid = 0.5 * (cell_a + cell_b)

    norm_a = np.linalg.norm(cell_a.reshape(c, 9), axis=-1)
    diff_c = np.linalg.norm((cell_b - cell_a).reshape(c, 9), axis=-1)
    has_pbc = pbc_a.any(axis=-1)
    same_pbc = np.all(pbc_a == pbc_b, axis=-1)
    va = np.abs(np.linalg.det(cell_a))
    vb = np.abs(np.linalg.det(cell_b))
    det_mid = np.abs(np.linalg.det(cell_mid))
    dlnv = np.where(has_pbc & (va > 1e-8) & (vb > 1e-8), np.abs(np.log(np.maximum(vb, 1e-12) / np.maximum(va, 1e-12))), 0.0)
    s_rel = np.where(has_pbc, diff_c / np.maximum(norm_a, 1e-8), 0.0)
    strain_rel[:] = s_rel

    pre_ok = (na == nb) & (na > 0) & same_pbc & (s_rel <= max_strain_rel) & (dlnv <= max_dlnv) & (~has_pbc | ((va > 1e-6) & (vb > 1e-6) & (det_mid > 1e-6)))
    if fmax_arr is not None and max_force_ev > 0.0:
        pre_ok &= (np.asarray(fmax_arr[row_a], dtype=np.float64) <= max_force_ev) & (np.asarray(fmax_arr[row_b], dtype=np.float64) <= max_force_ev)

    sub = np.flatnonzero(pre_ok)
    if len(sub) == 0:
        return valid, rmsd_out, strain_rel, work_mev

    ra, rb, n_sub = row_a[sub], row_b[sub], na[sub]
    m_tot = int(n_sub.sum())
    starts = np.zeros(len(sub), dtype=np.int64)
    if len(sub) > 1:
        np.cumsum(n_sub[:-1], out=starts[1:])
    arange_m = np.arange(m_tot, dtype=np.int64)
    off_a = np.asarray(flat.offsets[ra], dtype=np.int64)
    off_b = np.asarray(flat.offsets[rb], dtype=np.int64)
    idx_a = np.repeat(off_a - starts, n_sub) + arange_m
    idx_b = np.repeat(off_b - starts, n_sub) + arange_m

    z_diff = (np.asarray(flat.z[idx_a]) != np.asarray(flat.z[idx_b])).astype(np.int64)
    z_ok = np.add.reduceat(z_diff, starts) == 0
    if not z_ok.any():
        return valid, rmsd_out, strain_rel, work_mev

    if not z_ok.all():
        sub = sub[z_ok]
        ra, rb, n_sub = ra[z_ok], rb[z_ok], n_sub[z_ok]
        m_tot = int(n_sub.sum())
        starts = np.zeros(len(sub), dtype=np.int64)
        if len(sub) > 1:
            np.cumsum(n_sub[:-1], out=starts[1:])
        arange_m = np.arange(m_tot, dtype=np.int64)
        off_a = np.asarray(flat.offsets[ra], dtype=np.int64)
        off_b = np.asarray(flat.offsets[rb], dtype=np.int64)
        idx_a = np.repeat(off_a - starts, n_sub) + arange_m
        idx_b = np.repeat(off_b - starts, n_sub) + arange_m

    pa = np.asarray(flat.pos[idx_a], dtype=np.float64)
    pb = np.asarray(flat.pos[idx_b], dtype=np.float64)
    fa = np.asarray(flat.forces[idx_a], dtype=np.float64)
    fb = np.asarray(flat.forces[idx_b], dtype=np.float64)
    if max_force_ev > 0.0 and fmax_arr is None:
        fn_a = np.maximum.reduceat(np.linalg.norm(fa, axis=-1), starts)
        fn_b = np.maximum.reduceat(np.linalg.norm(fb, axis=-1), starts)
        f_ok = (fn_a <= max_force_ev) & (fn_b <= max_force_ev)
    else:
        f_ok = np.ones(len(sub), dtype=bool)

    ca_s, cb_s, cm_s = cell_a[sub], cell_b[sub], cell_mid[sub]
    pb_s, hp_s = pbc_a[sub], has_pbc[sub]
    pid_atom = np.repeat(np.arange(len(sub), dtype=np.int64), n_sub)

    # 1. Cartesian wrapped COM-subtracted displacement (for RMSD check)
    dpos = pb - pa
    dpos -= np.repeat(dpos[starts], n_sub, axis=0)
    # 2. Internal fractional displacement (for exact continuum-mechanics work integral without double-counting strain)
    dr_int = dpos.copy()

    if hp_s.any():
        inv_ca = np.zeros_like(ca_s)
        inv_cb = np.zeros_like(cb_s)
        inv_cm = np.zeros_like(cm_s)
        inv_ca[hp_s] = np.linalg.inv(ca_s[hp_s])
        inv_cb[hp_s] = np.linalg.inv(cb_s[hp_s])
        inv_cm[hp_s] = np.linalg.inv(cm_s[hp_s])
        hp_atom = hp_s[pid_atom]
        if hp_atom.any():
            p_idx = pid_atom[hp_atom]
            df_c = np.einsum("ni,nij->nj", dpos[hp_atom], inv_cm[p_idx])
            df_c = np.where(pb_s[p_idx], df_c - np.round(df_c), df_c)
            dpos[hp_atom] = np.einsum("ni,nij->nj", df_c, cm_s[p_idx])

            frac_a = np.einsum("ni,nij->nj", pa[hp_atom], inv_ca[p_idx])
            frac_b = np.einsum("ni,nij->nj", pb[hp_atom], inv_cb[p_idx])
            df_i = frac_b - frac_a
            # Anchor fractional diff per periodic structure
            df_full = np.zeros_like(dr_int)
            df_full[hp_atom] = df_i
            df_full -= np.repeat(df_full[starts], n_sub, axis=0)
            df_i = df_full[hp_atom]
            df_i = np.where(pb_s[p_idx], df_i - np.round(df_i), df_i)
            dr_int[hp_atom] = np.einsum("ni,nij->nj", df_i, cm_s[p_idx])

    dpos -= np.repeat(np.add.reduceat(dpos, starts, axis=0) / n_sub[:, None], n_sub, axis=0)
    dr_int -= np.repeat(np.add.reduceat(dr_int, starts, axis=0) / n_sub[:, None], n_sub, axis=0)
    msd = np.add.reduceat(np.sum(dpos * dpos, axis=-1), starts) / n_sub
    rmsd_sub = np.sqrt(np.maximum(msd, 0.0))

    # Trapezoidal energy-force-stress work consistency: dE - (-0.5*(Fa+Fb)·dr_int + 0.5*(Va*Sa+Vb*Sb):deps)
    de = np.asarray(flat.energy[rb], dtype=np.float64) - np.asarray(flat.energy[ra], dtype=np.float64)
    w_f = -0.5 * np.add.reduceat(np.sum((fa + fb) * dr_int, axis=-1), starts)
    w_s = np.zeros(len(sub), dtype=np.float64)
    st_ok = np.ones(len(sub), dtype=bool)
    if hp_s.any():
        sa_s = np.asarray(flat.stress[ra], dtype=np.float64)
        sb_s = np.asarray(flat.stress[rb], dtype=np.float64)
        fin_s = np.isfinite(sa_s).all(axis=(1, 2)) & np.isfinite(sb_s).all(axis=(1, 2))
        # If cell changed and stress is missing/NaN, cannot verify strain work; allow non-stress slabs only when cell is fixed.
        st_ok = ~hp_s | fin_s | (s_rel[sub] < 1e-6)
        use_s = hp_s & fin_s
        if use_s.any():
            deps = np.matmul(inv_cm[use_s], cb_s[use_s] - ca_s[use_s])
            vs_sum = va[sub][use_s, None, None] * sa_s[use_s] + vb[sub][use_s, None, None] * sb_s[use_s]
            w_s[use_s] = 0.5 * np.sum(vs_sum * deps, axis=(1, 2))

    res_mev = np.abs(de - (w_f + w_s)) / n_sub * 1000.0
    ok_sub = f_ok & st_ok & np.isfinite(res_mev)
    valid[sub] = ok_sub
    rmsd_out[sub] = rmsd_sub
    work_mev[sub] = res_mev
    return valid, rmsd_out, strain_rel, work_mev


def build_crystal_pairs(
    src,
    *,
    rmsd_min: float = 0.01,
    rmsd_max: float = 0.20,
    max_strain_rel: float = 0.10,
    max_dlnv: float = 0.10,
    min_strain_rel: float = 0.002,
    max_force_ev: float = 1.0,
    work_tol_mev: float = 1.0,
    max_pairs_per_group: int = 4,
    chunk_pairs: int = 32768,
    cache_dir: str | Path | None = DEFAULT_PAIR_CACHE_DIR,
) -> tuple[np.ndarray, dict]:
    """Build (or load cached) clean physics-filtered secant crystal pairs `pairs [M, 2]` (positions in `src`) + stats."""
    t0 = time.perf_counter()
    n_src = len(src)
    key = (
        "cry_v2_work",
        tuple((str(f.path.resolve()), len(f)) for f in src.flats),
        n_src,
        int(src.idx[0]) if n_src else 0,
        int(src.idx[-1]) if n_src else 0,
        int(src.idx.sum()) if n_src else 0,
        round(float(rmsd_min), 6),
        round(float(rmsd_max), 6),
        round(float(max_strain_rel), 6),
        round(float(max_dlnv), 6),
        round(float(min_strain_rel), 6),
        round(float(max_force_ev), 6),
        round(float(work_tol_mev), 6),
        int(max_pairs_per_group),
    )
    cf = _cache_file(cache_dir, "cry_pairs", key)
    if cf is not None and cf.exists():
        try:
            d = np.load(cf)
            pairs = np.asarray(d["pairs"], dtype=np.int64).reshape(-1, 2)
            stats = {
                "n_structs": n_src,
                "n_pairs": int(len(pairs)),
                "n_candidates": int(d["n_candidates"]),
                "n_groups_multi": int(d["n_groups_multi"]),
                "cached": True,
                "build_s": round(time.perf_counter() - t0, 3),
            }
            return pairs, stats
        except Exception:
            pass

    all_pairs: list[np.ndarray] = []
    tot_cand, tot_groups = 0, 0
    cand_cap = max(max_pairs_per_group * 4, 12)
    for k, flat in enumerate(src.flats):
        gid_path = flat.path / "group_id.npy"
        grp_arr = np.load(gid_path, mmap_mode="r") if gid_path.exists() else getattr(flat, "mp_id", None)
        if grp_arr is None:
            continue
        fmax_p = flat.path / "frame_max_force.npy"
        fmax_arr = np.load(fmax_p, mmap_mode="r") if fmax_p.exists() else None
        pos_k = np.flatnonzero(src.fid == k)
        if len(pos_k) < 2:
            continue
        rows_k = src.idx[pos_k]
        # Merged mptrj_salex_flat stores MPtrj in rows 0..1,580,394 and unique-sid sAlex after that.
        if "mptrj_salex" in flat.path.name and len(flat) > 1_580_395:
            m_mp = rows_k < 1_580_395
            pos_k, rows_k = pos_k[m_mp], rows_k[m_mp]
            if len(pos_k) < 2:
                continue
        # Fast probe: skip massive unique-per-frame flats (e.g. OMat24, OC20) in <1ms without 10M-row lexsort.
        if len(rows_k) > 50_000:
            probe = np.asarray(grp_arr[rows_k[:4096]])
            if np.unique(probe).size == len(probe):
                probe_tail = np.asarray(grp_arr[rows_k[-4096:]])
                if np.unique(probe_tail).size == len(probe_tail):
                    continue
        g_raw = np.asarray(grp_arr[rows_k])
        n_k = src.n_at[pos_k]
        if src.fixed_u is not None:
            u_k = np.zeros(len(pos_k), dtype=np.int8)
        else:
            u_k = np.clip(np.asarray(flat.u_flag[rows_k]), 0, 1).astype(np.int8)

        order = np.lexsort((u_k, n_k, g_raw))
        g_s, n_s, u_s = g_raw[order], n_k[order], u_k[order]
        same = (g_s[1:] == g_s[:-1]) & (n_s[1:] == n_s[:-1]) & (u_s[1:] == u_s[:-1])
        if not same.any():
            continue

        run_id = np.zeros(len(order), dtype=np.int64)
        run_id[1:] = np.cumsum(~same)
        idx1 = np.flatnonzero(same)
        ca_list = [order[idx1]]
        cb_list = [order[idx1 + 1]]
        cg_list = [run_id[idx1]]

        # Include stride-2 trajectory pairs to rescue converged adjacent tails (rmsd < 0.01).
        if len(same) >= 2:
            same2 = same[:-1] & same[1:]
            if same2.any():
                idx2 = np.flatnonzero(same2)
                ca_list.append(order[idx2])
                cb_list.append(order[idx2 + 2])
                cg_list.append(run_id[idx2])

        ca = np.concatenate(ca_list)
        cb = np.concatenate(cb_list)
        cg = np.concatenate(cg_list)
        tot_groups += int(np.unique(cg).size)

        keep_c = _cap_per_group(cg, cand_cap, evenly_spaced=True)
        ca, cb, cg = ca[keep_c], cb[keep_c], cg[keep_c]
        tot_cand += len(ca)

        acc_a, acc_b, acc_g = [], [], []
        for s in range(0, len(ca), chunk_pairs):
            e = min(len(ca), s + chunk_pairs)
            ra, rb = rows_k[ca[s:e]], rows_k[cb[s:e]]
            ok_zc, rmsd, s_rel, w_mev = _crystal_pair_metrics_chunk(
                flat, ra, rb, max_strain_rel=max_strain_rel, max_dlnv=max_dlnv, max_force_ev=max_force_ev, fmax_arr=fmax_arr,
            )
            ok = ok_zc & (((rmsd >= rmsd_min) | (s_rel >= min_strain_rel)) & (rmsd <= rmsd_max) & (s_rel <= max_strain_rel))
            if work_tol_mev > 0.0:
                ok &= (w_mev <= work_tol_mev)
            if ok.any():
                acc_a.append(pos_k[ca[s:e][ok]])
                acc_b.append(pos_k[cb[s:e][ok]])
                acc_g.append(cg[s:e][ok])

        if acc_a:
            pa = np.concatenate(acc_a)
            pb = np.concatenate(acc_b)
            pg = np.concatenate(acc_g)
            keep_g = _cap_per_group(pg, max_pairs_per_group, evenly_spaced=True)
            all_pairs.append(np.stack([pa[keep_g], pb[keep_g]], axis=1))

    pairs = np.concatenate(all_pairs, axis=0).astype(np.int64) if all_pairs else np.zeros((0, 2), dtype=np.int64)
    stats = {
        "n_structs": n_src,
        "n_pairs": int(len(pairs)),
        "n_candidates": int(tot_cand),
        "n_groups_multi": int(tot_groups),
        "cached": False,
        "build_s": round(time.perf_counter() - t0, 3),
    }
    if cf is not None:
        try:
            cf.parent.mkdir(parents=True, exist_ok=True)
            tmp = cf.with_name(cf.name + f".{os.getpid()}.tmp.npz")
            np.savez(tmp, pairs=pairs, n_candidates=tot_cand, n_groups_multi=tot_groups)
            os.replace(tmp, cf)
        except Exception:
            pass
    return pairs, stats


def build_omol_pairs(*args, **kwargs):
    from lam4.data.pairs_mol import build_omol_pairs as _build_omol_pairs
    return _build_omol_pairs(*args, **kwargs)


def build_pair_tensors_crystal(
    structs: list[dict],
    start: np.ndarray,
    n_tot: int,
    g_tot: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Build static-shape partner tensors for a packed crystal batch; only complete verified pairs are active."""
    ap = np.arange(n_tot, dtype=np.int64)
    am = np.zeros(n_tot, dtype=bool)
    gp = np.arange(g_tot, dtype=np.int64)
    gm = np.zeros(g_tot, dtype=bool)
    by_pid: dict[int, list[int]] = {}
    for gi, s in enumerate(structs):
        pid = s.get("_pair_id")
        if pid is not None and int(pid) > 0:
            by_pid.setdefault(int(pid), []).append(gi)
    for idxs in by_pid.values():
        if len(idxs) != 2:
            continue
        gi, gj = idxs
        si, sj = structs[gi], structs[gj]
        zi, zj = si["z"], sj["z"]
        if len(zi) != len(zj) or len(zi) == 0:
            continue
        if int(si["u"]) != int(sj["u"]) or bool(si.get("pbc", True)) != bool(sj.get("pbc", True)):
            continue
        if not np.array_equal(zi, zj):
            continue
        na = len(zi)
        ai, aj = int(start[gi]), int(start[gj])
        gp[gi], gp[gj] = gj, gi
        gm[gi] = gm[gj] = True
        ap[ai : ai + na] = np.arange(aj, aj + na, dtype=np.int64)
        ap[aj : aj + na] = np.arange(ai, ai + na, dtype=np.int64)
        am[ai : ai + na] = True
        am[aj : aj + na] = True
    return {
        "pair_atom_partner": torch.from_numpy(ap).to(device),
        "pair_atom_mask": torch.from_numpy(am).to(device),
        "pair_graph_partner": torch.from_numpy(gp).to(device),
        "pair_graph_mask": torch.from_numpy(gm).to(device),
    }


def build_pair_tensors_mol(
    pair_ids: np.ndarray | None,
    z_kept: np.ndarray,
    n_kept: np.ndarray,
    charge_kept: np.ndarray,
    spin_kept: np.ndarray,
    n_tot: int,
    g_tot: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Build static-shape partner tensors for a packed molecule batch; only complete verified pairs are active."""
    ap = np.arange(n_tot, dtype=np.int64)
    am = np.zeros(n_tot, dtype=bool)
    gp = np.arange(g_tot, dtype=np.int64)
    gm = np.zeros(g_tot, dtype=bool)
    if pair_ids is not None and len(pair_ids) > 0:
        starts = np.zeros(len(n_kept), dtype=np.int64)
        if len(n_kept) > 1:
            np.cumsum(n_kept[:-1], out=starts[1:])
        by_pid: dict[int, list[int]] = {}
        for gi, pid in enumerate(pair_ids):
            ip = int(pid)
            if ip > 0:
                by_pid.setdefault(ip, []).append(gi)
        for idxs in by_pid.values():
            if len(idxs) != 2:
                continue
            gi, gj = idxs
            na, nb = int(n_kept[gi]), int(n_kept[gj])
            if na != nb or na == 0:
                continue
            if round(float(charge_kept[gi])) != round(float(charge_kept[gj])) or round(float(spin_kept[gi])) != round(float(spin_kept[gj])):
                continue
            if not np.array_equal(z_kept[gi, :na], z_kept[gj, :na]):
                continue
            ai, aj = int(starts[gi]), int(starts[gj])
            gp[gi], gp[gj] = gj, gi
            gm[gi] = gm[gj] = True
            ap[ai : ai + na] = np.arange(aj, aj + na, dtype=np.int64)
            ap[aj : aj + na] = np.arange(ai, ai + na, dtype=np.int64)
            am[ai : ai + na] = True
            am[aj : aj + na] = True
    return {
        "pair_atom_partner": torch.from_numpy(ap).to(device),
        "pair_atom_mask": torch.from_numpy(am).to(device),
        "pair_graph_partner": torch.from_numpy(gp).to(device),
        "pair_graph_mask": torch.from_numpy(gm).to(device),
    }
