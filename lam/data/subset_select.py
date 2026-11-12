"""subset_select.py: training subsets as random base + latent-token coverage.

Recipe (after the "atoms that matter" and MAD-core subset-selection literature):
  1. Per-atom features from a trained model (stage-3 invariants), projected by PCA (fit on a sample).
  2. Latent token per atom = (Z, k-means centroid id), with a per-element codebook sized by sqrt(atom count).
  3. A random base takes `random_frac` of the atom budget. Its token counts seed the coverage.
  4. Greedy multi-cover fills the rest: each token needs `need` structures; a structure's gain is the number of its
     distinct tokens still short of `need`, divided by n_atoms**cost_power. Lazy greedy (gains only fall).
  5. If every token reaches `need` first, need grows by 1.5x and the cover continues (adaptive); a random fill takes
     what is left only when the cover cannot place anything more.
The model pass lives with the training scripts; everything here is model-free and works on numpy arrays.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
import torch


def fit_pca(x: np.ndarray, k: int, max_rows: int = 1_000_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(mean [d], components [d, k]) from a random row sample of x [n, d]."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(x), size=min(len(x), max_rows), replace=False) if len(x) > max_rows else slice(None)
    s = torch.as_tensor(np.asarray(x[rows], dtype=np.float32), dtype=torch.float64)
    mu = s.mean(0)
    _, _, v = torch.linalg.svd(s - mu, full_matrices=False)
    return mu.numpy(), v[:k].T.contiguous().numpy()


def project(x: np.ndarray, mean: np.ndarray, comps: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    out = np.empty((len(x), comps.shape[1]), np.float32)
    for s in range(0, len(x), chunk):
        out[s : s + chunk] = (np.asarray(x[s : s + chunk], np.float64) - mean) @ comps
    return out


def kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0, max_fit: int = 200_000) -> np.ndarray:
    """Centroids [k, d] by k-means++ seeding and Lloyd iterations on at most max_fit rows."""
    rng = np.random.default_rng(seed)
    if len(x) > max_fit:
        x = x[rng.choice(len(x), size=max_fit, replace=False)]
    t = torch.as_tensor(np.asarray(x, np.float32))
    k = min(k, len(t))
    c = t[[int(rng.integers(len(t)))]]
    d2 = torch.cdist(t, c).squeeze(1).pow(2)
    for _ in range(1, k):  # k-means++
        w = d2.double().numpy()
        p = w / w.sum() if w.sum() > 0 else None
        nxt = int(rng.choice(len(t), p=p)) if p is not None else int(rng.integers(len(t)))
        c = torch.cat([c, t[nxt : nxt + 1]])
        d2 = torch.minimum(d2, (t - t[nxt]).pow(2).sum(-1))
    for _ in range(iters):
        a = torch.cdist(t, c).argmin(1)
        s = torch.zeros_like(c).index_add_(0, a, t)
        n = torch.bincount(a, minlength=k).unsqueeze(1)
        c = torch.where(n > 0, s / n.clamp_min(1), c)
    return c.numpy()


def assign(x: np.ndarray, c: np.ndarray, chunk: int = 1 << 18) -> np.ndarray:
    ct = torch.as_tensor(c)
    out = np.empty(len(x), np.int32)
    for s in range(0, len(x), chunk):
        out[s : s + chunk] = torch.cdist(torch.as_tensor(np.asarray(x[s : s + chunk], np.float32)), ct).argmin(1).numpy()
    return out


def latent_tokens(
    z: np.ndarray, feat: np.ndarray, atoms_per_centroid: int = 2000, k_max: int = 4096, seed: int = 0
) -> tuple[np.ndarray, dict]:
    """Global token id per atom: per-element k-means on feat [n_atoms, d] with k = clip(n_Z / atoms_per_centroid,
    1, k_max). Returns (token [n_atoms] int64, info {Z: (offset, k)})."""
    tok = np.empty(len(z), np.int64)
    info, off = {}, 0
    for el in np.unique(z):
        rows = np.where(z == el)[0]
        k = int(np.clip(len(rows) // atoms_per_centroid, 1, k_max))
        c = kmeans(feat[rows], k, seed=seed + int(el)) if k > 1 else feat[rows[:1]].astype(np.float32)
        tok[rows] = off + assign(feat[rows], c)
        info[int(el)] = (off, len(c))
        off += len(c)
    return tok, info


def structure_tokens(tok: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CSR of the distinct tokens of each structure: (indptr [S+1], ids)."""
    sid = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    pair = np.unique(sid.astype(np.int64) * (int(tok.max()) + 1) + tok)
    s_of, ids = pair // (int(tok.max()) + 1), pair % (int(tok.max()) + 1)
    indptr = np.zeros(len(offsets), np.int64)
    np.add.at(indptr, s_of + 1, 1)
    return np.cumsum(indptr), ids


def select(
    indptr: np.ndarray,
    ids: np.ndarray,
    n_atoms: np.ndarray,
    atom_budget: int,
    random_frac: float = 0.6,
    need: int = 3,
    seed: int = 0,
    eligible: np.ndarray | None = None,
    cost_power: float = 1.0,
    adaptive: bool = True,
    need_max: int = 1 << 20,
) -> tuple[np.ndarray, dict]:
    """Structure ids: random base (random_frac of atom_budget) + greedy multi-cover + random fill.
    `eligible` (bool [S]) restricts the pool (e.g. after a force-outlier filter). The cover ranks structures by new
    tokens / n_atoms**cost_power: 1 minimises atoms (EnvTok), which favours small cells; 0 ranks per structure.
    adaptive: when every token reaches `need` before the budget is spent, need grows by 1.5x and the cover continues, so
    the coverage share flattens the token distribution instead of falling back to random."""
    rng = np.random.default_rng(seed)
    S = len(n_atoms)
    pool = np.arange(S) if eligible is None else np.where(eligible)[0]
    perm = rng.permutation(pool)
    csum = np.cumsum(n_atoms[perm])
    n_base = int(np.searchsorted(csum, random_frac * atom_budget, side="right"))
    base = perm[:n_base]
    taken = np.zeros(S, bool)
    taken[base] = True
    cnt = np.zeros(int(ids.max()) + 1 if len(ids) else 0, np.int64)
    for s in base:
        cnt[ids[indptr[s] : indptr[s + 1]]] += 1
    used = int(n_atoms[base].sum())

    def gain(s: int) -> float:
        t = ids[indptr[s] : indptr[s + 1]]
        return float((cnt[t] < need).sum()) / float(n_atoms[s]) ** cost_power

    sid = np.repeat(np.arange(S), np.diff(indptr))
    denom = n_atoms.astype(np.float64) ** cost_power
    cover = []
    rounds = []
    while random_frac < 1.0 and used < atom_budget:
        g_all = np.bincount(sid, weights=(cnt[ids] < need).astype(np.float64), minlength=S) / denom  # vectorised init
        cand = pool[(~taken[pool]) & (g_all[pool] > 0)]
        heap = list(zip((-g_all[cand]).tolist(), cand.tolist()))
        heapq.heapify(heap)
        n0 = len(cover)
        while heap and used < atom_budget:
            g, s = heapq.heappop(heap)
            g_now = gain(s)
            if g_now <= 0:
                continue
            if heap and -g_now > heap[0][0]:  # stale: push back with the true gain
                heapq.heappush(heap, (-g_now, s))
                continue
            if used + int(n_atoms[s]) > atom_budget:
                continue
            taken[s] = True
            cover.append(s)
            cnt[ids[indptr[s] : indptr[s + 1]]] += 1
            used += int(n_atoms[s])
        rounds.append((need, len(cover) - n0))
        if not adaptive or need >= need_max or len(cover) == n0 or used >= atom_budget:
            break
        need = int(math.ceil(need * 1.5))  # every token reached `need`: deepen the coverage of the rare ones
    fill = []
    for s in perm[n_base:]:
        if used >= atom_budget:
            break
        if not taken[s] and used + int(n_atoms[s]) <= atom_budget:
            taken[s] = True
            fill.append(int(s))
            used += int(n_atoms[s])
    sel = np.concatenate([base, np.asarray(cover, np.int64), np.asarray(fill, np.int64)]).astype(np.int64)
    n_tok = len(cnt)
    stats = dict(
        n_base=len(base), n_cover=len(cover), n_fill=len(fill), atoms=used, atom_budget=int(atom_budget),
        tokens=n_tok, tokens_seen=int((cnt > 0).sum()), tokens_at_need=int((cnt >= need).sum()), final_need=need,
        cover_rounds=rounds,
    )
    return sel, stats


def token_coverage(indptr: np.ndarray, ids: np.ndarray, sel: np.ndarray, n_tokens: int, need: int = 1) -> float:
    """Fraction of the pool's tokens seen in >= need selected structures."""
    cnt = np.zeros(n_tokens, np.int64)
    for s in sel:
        cnt[ids[indptr[s] : indptr[s + 1]]] += 1
    present = np.zeros(n_tokens, bool)
    present[ids] = True
    return float((cnt[present] >= need).mean())
