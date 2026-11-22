"""omol_buckets.py: OMol25 structures as padded size buckets on the device for Stream B."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from lam.models.dispersion import D3BJ, WB97M_D3BJ

DEFAULT_EDGES = (12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256, 350)
CHARGES = tuple(range(-10, 11))
D3_PAIR_BUDGET = 2_000_000
SPINS = tuple(range(1, 12))
REF_FIT_DIRECT_MAX = 6_000_000


class OmolFlat:
    """Memory-mapped arrays of one converted OMol25 split."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ld = lambda k: np.load(self.path / f"{k}.npy", mmap_mode="r")  # noqa: E731
        self.z, self.pos, self.forces = ld("z"), ld("pos"), ld("forces")
        self.offsets = np.load(self.path / "offsets.npy")
        self.energy, self.charge, self.spin = ld("energy"), ld("charge"), ld("spin")
        self.domain = np.asarray(ld("domain"))
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.domains = self.manifest["domains"]
        self.n_atoms = np.diff(self.offsets)
        opt = lambda k: ld(k) if (self.path / f"{k}.npy").exists() else None  # noqa: E731
        self.dipole, self.subdomain, self.source_set = opt("dipole"), opt("subdomain"), opt("source_set")
        self.subdomains = self.manifest.get("subdomains")
        n = len(self.n_atoms)
        for name, arr, shape in (
            ("dipole", self.dipole, (n, 3)),
            ("subdomain", self.subdomain, (n,)),
            ("source_set", self.source_set, (n,)),
        ):
            if arr is not None and tuple(arr.shape) != shape:
                raise ValueError(f"{self.path}/{name}.npy has shape {tuple(arr.shape)}, expected {shape}")

    def __len__(self) -> int:
        return len(self.offsets) - 1


def gather_atoms(flat: OmolFlat, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat atom indices of the structures idx (concatenated in idx order) and each structure's start in that list."""
    n = flat.n_atoms[idx]
    starts = np.zeros(len(idx), np.int64)
    np.cumsum(n[:-1], out=starts[1:])
    atoms = np.repeat(flat.offsets[idx] - starts, n) + np.arange(int(n.sum()))
    return atoms, starts


def design_row(z: np.ndarray, q: int, s: int) -> np.ndarray:
    x = np.zeros(119 + len(CHARGES) + len(SPINS))
    np.add.at(x, z.astype(np.int64), 1.0)
    if -10 <= q <= 10:
        x[119 + q + 10] = 1.0
    if 1 <= s <= 11:
        x[119 + len(CHARGES) + s - 1] = 1.0
    return x


def design_matrix(flat: OmolFlat, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, np.int64)
    n = len(rows)
    k = 119 + len(CHARGES) + len(SPINS)
    atoms, _ = gather_atoms(flat, rows)
    sid = np.repeat(np.arange(n, dtype=np.int64), flat.n_atoms[rows])
    X = np.bincount(sid * k + np.asarray(flat.z[atoms]).astype(np.int64), minlength=n * k).reshape(n, k).astype(np.float64)
    q = np.asarray(flat.charge[rows]).astype(np.int64)
    sp = np.asarray(flat.spin[rows]).astype(np.int64)
    ok = (q >= -10) & (q <= 10)
    X[np.where(ok)[0], 119 + q[ok] + 10] = 1.0
    ok = (sp >= 1) & (sp <= 11)
    X[np.where(ok)[0], 119 + len(CHARGES) + sp[ok] - 1] = 1.0
    X[:, 119 + 10] = 0.0
    X[:, 119 + len(CHARGES)] = 0.0
    return X


def chi_eta_from_e_q(e_q: np.ndarray, num_elements: int = 119) -> tuple[np.ndarray, np.ndarray]:
    """Initialize per-element electronegativity chi_Z and hardness eta_Z from E_Q[+1], E_Q[0], E_Q[-1]."""
    e_q_arr = np.asarray(e_q, dtype=np.float64)
    chi_0 = 0.5 * (e_q_arr[11] - e_q_arr[9])
    eta_0 = e_q_arr[11] + e_q_arr[9] - 2.0 * e_q_arr[10]
    chi = np.zeros(num_elements, dtype=np.float64)
    eta = np.zeros(num_elements, dtype=np.float64)
    chi[1:] = chi_0
    eta[1:] = eta_0
    return chi, eta


def _fit_reference_normal(flat: OmolFlat, idx: np.ndarray, e_d3: np.ndarray, chunk: int = 500_000) -> dict:
    k = 119 + len(CHARGES) + len(SPINS)
    xtx, xty = np.zeros((k, k)), np.zeros(k)
    for s in range(0, len(idx), chunk):
        X = design_matrix(flat, idx[s : s + chunk])
        y = flat.energy[idx[s : s + chunk]] - e_d3[s : s + chunk]
        xtx += X.T @ X
        xty += X.T @ y
    used = np.where(np.diag(xtx) > 0)[0]
    w = np.zeros(k)
    w[used] = np.linalg.lstsq(xtx[np.ix_(used, used)], xty[used], rcond=None)[0]
    n, s1, s2, sa = 0, 0.0, 0.0, 0.0
    for s in range(0, len(idx), chunk):
        r = flat.energy[idx[s : s + chunk]] - e_d3[s : s + chunk] - design_matrix(flat, idx[s : s + chunk]) @ w
        n, sa, s1, s2 = n + len(r), sa + np.abs(r).sum(), s1 + r.sum(), s2 + (r ** 2).sum()
    e_q = w[119 : 119 + len(CHARGES)]
    chi, eta = chi_eta_from_e_q(e_q)
    return {
        "e0": w[:119],
        "e_q": e_q,
        "e_s": w[119 + len(CHARGES) :],
        "chi": chi,
        "eta": eta,
        "n_fit": int(n),
        "resid_mae": float(sa / n),
        "resid_std": float(np.sqrt(max(s2 / n - (s1 / n) ** 2, 0.0))),
    }


def fit_reference(flat: OmolFlat, idx: np.ndarray, e_d3: np.ndarray, max_n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sel = np.arange(len(idx)) if len(idx) <= max_n else np.sort(rng.choice(len(idx), max_n, replace=False))
    if len(sel) > REF_FIT_DIRECT_MAX:
        return _fit_reference_normal(flat, idx[sel], e_d3[sel])
    X = np.stack([design_row(flat.z[flat.offsets[i] : flat.offsets[i + 1]], int(flat.charge[i]), int(flat.spin[i])) for i in idx[sel]])
    y = flat.energy[idx[sel]] - e_d3[sel]
    X[:, 119 + 10] = 0.0
    X[:, 119 + len(CHARGES)] = 0.0
    used = np.where(np.abs(X).sum(0) > 0)[0]
    coef, *_ = np.linalg.lstsq(X[:, used], y, rcond=None)
    w = np.zeros(X.shape[1])
    w[used] = coef
    resid = y - X @ w
    e_q = w[119 : 119 + len(CHARGES)]
    chi, eta = chi_eta_from_e_q(e_q)
    return {
        "e0": w[:119],
        "e_q": e_q,
        "e_s": w[119 + len(CHARGES) :],
        "chi": chi,
        "eta": eta,
        "n_fit": int(len(sel)),
        "resid_mae": float(np.abs(resid).mean()),
        "resid_std": float(resid.std()),
    }


def reference_energy(
    flat: OmolFlat,
    idx: np.ndarray,
    ref: dict,
    *,
    q_atom: np.ndarray | None = None,
    use_charge_self_energy: bool = False,
    include_charge_lookup: bool = True,
) -> np.ndarray:
    e0 = np.asarray(ref["e0"])
    atoms, starts = gather_atoms(flat, idx)
    z_at = np.asarray(flat.z[atoms] if len(atoms) else np.zeros(0, np.uint8)).astype(np.int64)
    per_atom = e0[z_at]
    e = np.add.reduceat(per_atom, starts) if len(atoms) else np.zeros(len(idx))
    e[flat.n_atoms[idx] == 0] = 0.0
    q = np.asarray(flat.charge[idx]).astype(np.int64)
    sp = np.asarray(flat.spin[idx]).astype(np.int64)
    if q_atom is not None or use_charge_self_energy:
        if "chi" in ref and "eta" in ref:
            chi, eta = np.asarray(ref["chi"], dtype=np.float64), np.asarray(ref["eta"], dtype=np.float64)
        else:
            chi, eta = chi_eta_from_e_q(ref["e_q"])
        if q_atom is None:
            n_at = np.maximum(flat.n_atoms[idx], 1)
            q_i = np.repeat(q.astype(np.float64) / n_at, flat.n_atoms[idx])
        else:
            q_i = np.asarray(q_atom, dtype=np.float64)
        e_self_at = chi[z_at] * q_i + 0.5 * eta[z_at] * (q_i ** 2)
        e_q = np.add.reduceat(e_self_at, starts) if len(atoms) else np.zeros(len(idx))
        e_q[flat.n_atoms[idx] == 0] = 0.0
    elif include_charge_lookup:
        e_q = np.where((q >= -10) & (q <= 10), np.asarray(ref["e_q"])[np.clip(q + 10, 0, 20)], 0.0)
    else:
        e_q = np.zeros(len(idx), dtype=np.float64)
    e_s = np.where((sp >= 1) & (sp <= 11), np.asarray(ref["e_s"])[np.clip(sp - 1, 0, 10)], 0.0)
    return e + e_q + e_s


def ref_to_json(ref: dict) -> dict:
    chi, eta = (ref["chi"], ref["eta"]) if ("chi" in ref and "eta" in ref) else chi_eta_from_e_q(ref["e_q"])
    return {
        "e0_kcal": {int(z): float(v) for z, v in enumerate(ref["e0"]) if v != 0.0},
        "charge_kcal": {int(q): float(v) for q, v in zip(CHARGES, ref["e_q"]) if v != 0.0},
        "spin_kcal": {int(s): float(v) for s, v in zip(SPINS, ref["e_s"]) if v != 0.0},
        "chi_kcal": {int(z): float(v) for z, v in enumerate(chi) if v != 0.0},
        "eta_kcal": {int(z): float(v) for z, v in enumerate(eta) if v != 0.0},
        "n_fit": ref["n_fit"],
        "resid_mae": ref["resid_mae"],
        "resid_std": ref["resid_std"],
        "note": "E_DFT = model energy + D3(BJ, wB97M) + sum_i (e0[Z_i] + chi[Z_i]*q_i + 0.5*eta[Z_i]*q_i^2) + spin[S]",
    }


def ref_from_json(d: dict) -> dict:
    e0, e_q, e_s = np.zeros(119), np.zeros(len(CHARGES)), np.zeros(len(SPINS))
    for z, v in d["e0_kcal"].items():
        e0[int(z)] = v
    for q, v in d["charge_kcal"].items():
        e_q[int(q) + 10] = v
    for s, v in d["spin_kcal"].items():
        e_s[int(s) - 1] = v
    chi, eta = chi_eta_from_e_q(e_q)
    if "chi_kcal" in d:
        for z, v in d["chi_kcal"].items():
            chi[int(z)] = v
    if "eta_kcal" in d:
        for z, v in d["eta_kcal"].items():
            eta[int(z)] = v
    return {
        "e0": e0,
        "e_q": e_q,
        "e_s": e_s,
        "chi": chi,
        "eta": eta,
        "n_fit": d.get("n_fit", 0),
        "resid_mae": d.get("resid_mae", 0.0),
        "resid_std": d.get("resid_std", 0.0),
    }


def default_subset(flat: OmolFlat, max_structs: int = 0, seed: int = 0, max_atoms: int = DEFAULT_EDGES[-1]) -> np.ndarray:
    idx = np.arange(len(flat))
    idx = idx[np.isfinite(np.asarray(flat.energy)) & (flat.n_atoms <= max_atoms)]
    if max_structs and len(idx) > max_structs:
        idx = np.sort(np.random.default_rng(seed + 777).choice(idx, max_structs, replace=False))
    return idx


def d3_sig(flat: OmolFlat, idx: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(idx, dtype=np.int64).tobytes()
        + json.dumps(WB97M_D3BJ, sort_keys=True).encode()
        + str(flat.manifest["n_atoms_total"]).encode()
    ).hexdigest()[:16]


def compute_d3(flat: OmolFlat, idx: np.ndarray, dev: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = torch.float32
    d3 = D3BJ(r_cut=12.0, taper_on=9.0).to(device=dev, dtype=dtype)
    n_at = flat.n_atoms[idx]
    off = np.zeros(len(idx) + 1, np.int64)
    np.cumsum(n_at, out=off[1:])
    e_all, f_all = np.zeros(len(idx)), np.zeros((int(off[-1]), 3), np.float32)
    if not len(idx):
        return e_all, f_all, off
    Z, P = np.asarray(flat.z), np.asarray(flat.pos)
    order = np.argsort(n_at, kind="stable")
    budget = D3_PAIR_BUDGET * (16 if dev.type == "cuda" else 1)
    if dev.type == "cuda":
        budget = min(budget, int(0.4 * torch.cuda.mem_get_info(dev)[0] / 500))
    if os.environ.get("MENA_D3_MAX_PAIRS"):
        budget = min(budget, int(os.environ["MENA_D3_MAX_PAIRS"]))
    t0, s, n_chunks = time.time(), 0, 0
    while s < len(order):
        lo, hi = 1, len(order) - s
        while lo < hi:
            mid = (lo + hi + 1) // 2
            c = int(n_at[order[s + mid - 1]])
            lo, hi = (mid, hi) if mid * c * c <= budget else (lo, mid - 1)
        step = lo
        cap = int(n_at[order[s + step - 1]])
        chunk = order[s : s + step]
        valid = np.arange(cap)[None, :] < n_at[chunk][:, None]
        src = np.where(valid, flat.offsets[idx[chunk]][:, None] + np.arange(cap)[None, :], 0)
        z = torch.from_numpy(np.where(valid, Z[src], 0).astype(np.int64)).to(dev)
        m = torch.from_numpy(valid).to(dev)
        p = torch.from_numpy(np.where(valid[..., None], P[src], 0.0).astype(np.float32)).to(device=dev, dtype=dtype)
        p.requires_grad_(True)
        with torch.enable_grad():
            e = d3(z, p, m)
            f = -torch.autograd.grad(e.sum(), p)[0] * m.unsqueeze(-1)
        e, f = e.detach().cpu().double().numpy(), f.detach().cpu().float().numpy()
        e_all[chunk] = e
        tgt = off[chunk][:, None] + np.arange(cap)[None, :]
        f_all[tgt[valid]] = f[valid]
        s += len(chunk)
        n_chunks += 1
    print(f"  D3 labels for {len(idx):,} structures computed in {n_chunks:,} chunks, {time.time() - t0:.1f}s", flush=True)
    return e_all, f_all, off


def d3_labels(
    flat: OmolFlat,
    idx: np.ndarray,
    dev: torch.device,
    cache_dir: str | Path | None = None,
    reuse: list[tuple[np.ndarray, Path]] | None = None,
    compute: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig = d3_sig(flat, idx)
    cache = Path(cache_dir or flat.path) / f"d3_{sig}.pt"
    candidates = [cache]
    env_dir = os.environ.get("OMOL_D3_CACHE_DIR", "").strip()
    if env_dir:
        ed = Path(env_dir)
        candidates.append(ed / f"{flat.path.name}_d3_{sig}.pt")
        candidates.append(ed / f"d3_{sig}.pt")
    try:
        resolved = flat.path.resolve()
        candidates.append(resolved.parent.parent / "d3cache_v2" / f"{resolved.name}_d3_{sig}.pt")
    except OSError:
        pass
    for c in candidates:
        if c.exists():
            try:
                ck = torch.load(c, map_location="cpu", weights_only=False)
                print(f"  D3 labels from cache {c}", flush=True)
                return ck["e"], ck["f"], ck["off"]
            except Exception as e:
                if not compute:
                    raise RuntimeError(f"D3 cache {c} unreadable ({e!r:.120})") from e
                if c == cache:
                    c.unlink(missing_ok=True)
    n_at = flat.n_atoms[idx]
    off = np.zeros(len(idx) + 1, np.int64)
    np.cumsum(n_at, out=off[1:])
    e_all, f_all = np.zeros(len(idx)), np.zeros((int(off[-1]), 3), np.float32)
    todo = np.ones(len(idx), bool)
    for r_idx, r_path in reuse or []:
        r_path = Path(r_path)
        if not r_path.exists() or r_path.stem != f"d3_{d3_sig(flat, r_idx)}":
            continue
        pos = np.clip(np.searchsorted(r_idx, idx), 0, len(r_idx) - 1)
        hit = todo & (r_idx[pos] == idx)
        if not hit.any():
            continue
        rk = torch.load(r_path, map_location="cpu", weights_only=False)
        e_all[hit] = rk["e"][pos[hit]]
        h = np.where(hit)[0]
        dst = np.repeat(off[h], n_at[h]) + (np.arange(int(n_at[h].sum())) - np.repeat(np.cumsum(n_at[h]) - n_at[h], n_at[h]))
        src = np.repeat(rk["off"][pos[h]], n_at[h]) + (dst - np.repeat(off[h], n_at[h]))
        f_all[dst] = rk["f"][src]
        todo &= ~hit
        del rk
    if todo.any():
        if not compute:   # after the superset caches (reuse): a subset of a cached flat is sliced, only rows no cache holds are refused
            raise RuntimeError(f"no D3 cache {cache} and {int(todo.sum())} rows outside the reuse caches (compute=False)")
        h = np.where(todo)[0]
        e_new, f_new, off_new = compute_d3(flat, idx[h], dev)
        e_all[h] = e_new
        dst = np.repeat(off[h], n_at[h]) + (np.arange(int(off_new[-1])) - np.repeat(off_new[:-1], n_at[h]))
        f_all[dst] = f_new
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + ".tmp")
        torch.save({"e": e_all, "f": f_all, "off": off}, tmp, pickle_protocol=4)
        tmp.replace(cache)
    except (OSError, RuntimeError):
        pass
    return e_all, f_all, off


_KEY_RNG = np.random.default_rng(20261001)
ELEMENT_KEY = _KEY_RNG.integers(1, 2 ** 63, size=256, dtype=np.uint64)
CHARGE_KEY = _KEY_RNG.integers(1, 2 ** 63, size=64, dtype=np.uint64)
SPIN_KEY = _KEY_RNG.integers(1, 2 ** 63, size=32, dtype=np.uint64)


def composition_key(flat: OmolFlat, rows: np.ndarray | None = None, chunk_atoms: int = 40_000_000) -> np.ndarray:
    """[n] uint64 composition key of the structures `rows` (element counts, total charge, multiplicity)."""
    rows = np.arange(len(flat)) if rows is None else np.asarray(rows, np.int64)
    out = np.zeros(len(rows), np.uint64)
    s = 0
    with np.errstate(over="ignore"):
        while s < len(rows):
            n_at = flat.n_atoms[rows[s:]]
            e = s + max(1, int(np.searchsorted(np.cumsum(n_at), chunk_atoms)))
            e = min(e, len(rows))
            atoms, starts = gather_atoms(flat, rows[s:e])
            vals = ELEMENT_KEY[np.asarray(flat.z[atoms]).astype(np.int64)] if len(atoms) else np.zeros(0, np.uint64)
            sums = np.add.reduceat(vals, starts) if len(atoms) else np.zeros(e - s, np.uint64)
            sums[flat.n_atoms[rows[s:e]] == 0] = 0
            out[s:e] = sums
            s = e
        q = np.asarray(flat.charge)[rows].astype(np.int64) + 32
        sp = np.asarray(flat.spin)[rows].astype(np.int64)
        out = out + CHARGE_KEY[np.clip(q, 0, 63)] + SPIN_KEY[np.clip(sp, 0, 31)]
    return out


def compute_batch_seg(
    z: torch.Tensor,
    mask: torch.Tensor,
    charge: torch.Tensor,
    spin: torch.Tensor,
) -> torch.Tensor:
    """Compute segment IDs `[B]` in `{0..B-1}` grouping molecules with identical (element counts, charge, spin).
    Padding structures (`~mask.any(-1)`) each receive a unique singleton segment ID."""
    B = int(z.shape[0])
    z_np = (z.detach().cpu() * mask.detach().cpu().to(z.dtype)).numpy().astype(np.int64)
    m_np = mask.detach().cpu().numpy()
    q_np = np.clip(charge.detach().cpu().numpy().round().astype(np.int64) + 32, 0, 63)
    s_np = np.clip(spin.detach().cpu().numpy().round().astype(np.int64), 0, 31)
    with np.errstate(over="ignore"):
        elem_vals = np.where(m_np, ELEMENT_KEY[np.clip(z_np, 0, 255)], np.uint64(0))
        keys = elem_vals.sum(axis=1, dtype=np.uint64) + CHARGE_KEY[q_np] + SPIN_KEY[s_np]
    real = m_np.any(axis=1)
    seg = np.arange(B, dtype=np.int64)
    seen: dict[int, int] = {}
    next_id = 0
    for i in range(B):
        if not real[i]:
            continue
        k = int(keys[i])
        if k not in seen:
            seen[k] = next_id
            next_id += 1
        seg[i] = seen[k]
    for i in range(B):
        if not real[i]:
            seg[i] = min(B - 1, next_id)
            next_id += 1
    return torch.from_numpy(seg).to(device=z.device, dtype=torch.long)


class OmolBuckets:
    """Padded 17-size buckets on `device` for Stream B molecules."""

    def __init__(
        self,
        path: str | Path,
        device: torch.device,
        *,
        max_structs: int = 0,
        seed: int = 0,
        edges: tuple[int, ...] = DEFAULT_EDGES,
        d3_subtract: bool = True,
        ref: dict | None = None,
        ref_fit_max: int = 300_000,
        max_force: float = 0.0,
        cache_dir: str | Path | None = None,
        indices: np.ndarray | None = None,
        d3_reuse: list[tuple[np.ndarray, Path]] | None = None,
        atom_charges: str | None = None,
        d3_compute: bool = True,
        include_charge_lookup: bool = False,
        units: str = "kcal",
    ) -> None:
        from lam.units import EV

        t0 = time.time()
        u_norm = units.lower()
        if u_norm not in ("kcal", "ev"):
            raise ValueError(f"Unsupported units={units!r} for OmolBuckets; expected 'kcal' or 'ev'")
        self.units = u_norm
        unit_scale = (1.0 / float(EV)) if u_norm == "ev" else 1.0

        self.flat = flat = OmolFlat(path)
        self.device = device
        self.d3_compute = d3_compute
        self.include_charge_lookup = include_charge_lookup
        finite = np.isfinite(np.asarray(flat.energy))
        too_big = flat.n_atoms > edges[-1]
        if indices is not None:
            idx = np.asarray(indices, dtype=np.int64)
            n_drop_big = int(too_big[idx].sum())
            idx = idx[finite[idx] & ~too_big[idx]]
        else:
            n_drop_big = int((finite & too_big).sum())
            idx = default_subset(flat, max_structs, seed, edges[-1])
        self.idx = idx
        self.d3 = d3_labels(flat, idx, device, cache_dir, d3_reuse, compute=d3_compute) if d3_subtract else None
        e_d3 = self.d3[0] if self.d3 is not None else np.zeros(len(idx))
        if ref is None:
            ref = fit_reference(flat, idx, e_d3, ref_fit_max, seed)
        self.ref = ref
        e_res_kcal = (
            np.asarray(flat.energy[idx], dtype=np.float64)
            - np.asarray(e_d3, dtype=np.float64)
            - reference_energy(flat, idx, ref, include_charge_lookup=include_charge_lookup)
        )
        e_res = e_res_kcal * unit_scale
        comp_keys = composition_key(flat, idx) if len(idx) else np.zeros(0, dtype=np.uint64)
        gid_all = np.unique(comp_keys, return_inverse=True)[1].ravel() if len(idx) else np.zeros(0, dtype=np.int64)

        n_at = flat.n_atoms[idx]
        edges_arr = np.asarray(edges)
        which = np.searchsorted(edges_arr, n_at)
        self.buckets: dict[int, dict] = {}
        n_drop_force = 0
        q_all = np.load(flat.path / f"charges_{atom_charges}.npy", mmap_mode="r") if atom_charges else None
        for bi in np.unique(which):
            cap = int(edges_arr[bi])
            loc = np.where(which == bi)[0]
            n = len(loc)
            n_k = n_at[loc]
            slot = np.arange(cap)[None, :]
            valid = slot < n_k[:, None]
            src = np.where(valid, flat.offsets[idx[loc]][:, None] + slot, 0)
            z = np.where(valid, np.asarray(flat.z[src.ravel()]).reshape(n, cap), 0).astype(np.uint8)
            pos = np.where(valid[..., None], np.asarray(flat.pos[src.ravel()]).reshape(n, cap, 3), 0.0).astype(np.float32)
            frc = np.where(valid[..., None], np.asarray(flat.forces[src.ravel()]).reshape(n, cap, 3), 0.0).astype(np.float32)
            if self.d3 is not None:
                d3_src = np.where(valid, self.d3[2][loc][:, None] + slot, 0)
                frc -= np.where(valid[..., None], self.d3[1][d3_src.ravel()].reshape(n, cap, 3), 0.0)
            keep = np.ones(n, bool)
            if max_force > 0:
                keep = np.abs(frc).reshape(n, -1).max(1) <= max_force
                n_drop_force += int((~keep).sum())
            sl = np.where(keep)[0]
            frc_scaled = (frc[sl] * float(unit_scale)).astype(np.float32)
            bd = {
                "z": torch.from_numpy(z[sl]),
                "pos": torch.from_numpy(pos[sl]),
                "forces": torch.from_numpy(frc_scaled),
                "n_atoms": torch.from_numpy(n_at[loc][sl].astype(np.int64)),
                "energy": torch.from_numpy(e_res[loc][sl].astype(np.float32)),
                "total_charge": torch.from_numpy(flat.charge[idx[loc][sl]].astype(np.float32)),
                "spin": torch.from_numpy(flat.spin[idx[loc][sl]].astype(np.int64)),
                "domain": torch.from_numpy(flat.domain[idx[loc][sl]].astype(np.int64)),
                "row": torch.from_numpy(idx[loc][sl].astype(np.int64)),
            }
            if q_all is not None:
                qr = np.where(valid, np.asarray(q_all[src.ravel()]).reshape(n, cap), np.nan).astype(np.float32)
                bd["q_ref"] = torch.from_numpy(qr[sl])
            rows_k = idx[loc][sl]
            if flat.dipole is not None:
                bd["dipole"] = torch.from_numpy(np.asarray(flat.dipole[rows_k], np.float32))
            else:
                bd["dipole"] = torch.full((len(sl), 3), float("nan"), dtype=torch.float32)
            if flat.subdomain is not None:
                bd["subdomain"] = torch.from_numpy(np.asarray(flat.subdomain[rows_k]).astype(np.int16))
            if flat.source_set is not None:
                bd["source"] = torch.from_numpy(np.asarray(flat.source_set[rows_k]).astype(np.int8))
            bd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in bd.items()}
            bd["units"] = self.units
            bd["energy_unit"] = "eV" if self.units == "ev" else "kcal/mol"
            bd["bucket_cap"] = int(cap)
            bd["count"] = len(sl)
            if len(sl):
                g_sub = gid_all[loc[sl]]
                order = np.argsort(g_sub, kind="stable")
                cuts = np.where(np.diff(g_sub[order]) != 0)[0] + 1
                bd["_groups"] = np.split(order, cuts)
                self.buckets[cap] = bd
        self.total = sum(bd["count"] for bd in self.buckets.values())
        pad = sum(cap * bd["count"] for cap, bd in self.buckets.items()) / max(1, int(n_at.sum()))
        print(
            f"Loaded {flat.path.name}: {self.total:,} structures ({n_drop_big:,} > {edges[-1]} atoms dropped), "
            f"padding x{pad:.2f}, buckets { {c: bd['count'] for c, bd in self.buckets.items()} } ({time.time() - t0:.1f}s)",
            flush=True,
        )

    def to_device(self, device: torch.device) -> OmolBuckets:
        self.device = device
        for bd in self.buckets.values():
            for k, v in list(bd.items()):
                if torch.is_tensor(v):
                    bd[k] = v.to(device, non_blocking=True)
        return self

    def epoch_plan(self, batch_sizes: dict[int, int], seed: int, chunk: int = 8) -> list:
        rng = np.random.default_rng(seed)
        plan = []
        for cap, bd in self.buckets.items():
            bs = int(batch_sizes.get(cap, 16))
            if chunk > 0 and "_groups" in bd:
                c = max(1, min(int(chunk), bs))
                pieces = []
                for g in bd["_groups"]:
                    g_perm = rng.permutation(g)
                    parts = [g_perm[s : s + c] for s in range(0, len(g_perm), c)]
                    if len(parts) > 1 and len(parts[-1]) == 1 and (len(parts[-2]) + len(parts[-1]) <= bs):
                        last = parts.pop()
                        parts[-1] = np.concatenate([parts[-1], last])
                    pieces.extend(parts)
                cur, size = [], 0
                for i in rng.permutation(len(pieces)):
                    p = pieces[i]
                    if size + len(p) > bs and cur:
                        b_idx = np.concatenate(cur)
                        b_seg = np.repeat(np.arange(len(cur), dtype=np.int64), [len(x) for x in cur])
                        plan.append((cap, b_idx, b_seg))
                        cur, size = [], 0
                    cur.append(p)
                    size += len(p)
                if cur:
                    b_idx = np.concatenate(cur)
                    b_seg = np.repeat(np.arange(len(cur), dtype=np.int64), [len(x) for x in cur])
                    plan.append((cap, b_idx, b_seg))
            else:
                perm = rng.permutation(bd["count"])
                plan += [(cap, perm[s : s + bs]) for s in range(0, len(perm), bs)]
        return [plan[i] for i in rng.permutation(len(plan))]

    def batch(self, cap: int, idx: np.ndarray, seg: np.ndarray | None = None) -> dict:
        bd = self.buckets[cap]
        it = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(bd["z"].device, non_blocking=True)
        z = bd["z"][it].long()
        mask = z > 0
        charge = bd["total_charge"][it]
        spin = bd["spin"][it]
        out = {
            "z": z,
            "mask": mask,
            "pos": bd["pos"][it],
            "forces": bd["forces"][it],
            "y_f_resid": bd["forces"][it],
            "energy": bd["energy"][it],
            "y_e_resid": bd["energy"][it],
            "total_charge": charge,
            "charge": charge,
            "spin": spin,
            "domain": bd["domain"][it],
            "n_atoms": bd["n_atoms"][it],
            "n_real_atoms": int(bd["n_atoms"][it].sum().item()),
            "bucket_cap": int(cap),
            "units": self.units,
            "energy_unit": "eV" if self.units == "ev" else "kcal/mol",
            **({"q_ref": bd["q_ref"][it]} if "q_ref" in bd else {}),
        }
        if "dipole" in bd:
            out["dipole"] = bd["dipole"][it]
        if "subdomain" in bd:
            out["subdomain"] = bd["subdomain"][it].long()
        if "source" in bd:
            out["source"] = bd["source"][it].long()
        if seg is not None:
            seg_t = torch.from_numpy(np.asarray(seg, dtype=np.int64)).to(z.device, non_blocking=True)
        else:
            seg_t = compute_batch_seg(z, mask, charge, spin)
        out["seg"] = seg_t
        out["batch_seg"] = seg_t
        return out


class OmolStream:
    """Cycles through composition-grouped epoch plans of one `OmolBuckets` instance."""

    def __init__(
        self,
        data: OmolBuckets,
        batch_sizes: dict[int, int] | None = None,
        chunk: int = 8,
        seed: int = 0,
        *,
        budget: int | None = None,
        bgt: int | None = None,
    ) -> None:
        self.data = data
        if batch_sizes is None:
            atom_bgt = int(budget if budget is not None else (bgt if bgt is not None else 8192))
            batch_sizes = batch_sizes_for(tuple(data.buckets.keys()) or DEFAULT_EDGES, atom_bgt, atom_bgt * 64)
        self.bs = batch_sizes
        self.chunk = chunk
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self.plan: list = []
        self._new_epoch()

    def _new_epoch(self) -> None:
        self.epoch += 1
        self.plan = self.data.epoch_plan(self.bs, seed=self.seed * 1000 + self.epoch, chunk=self.chunk)
        self.pos = 0

    def next(self) -> dict:
        if self.pos >= len(self.plan):
            self._new_epoch()
        item = self.plan[self.pos]
        self.pos += 1
        if len(item) == 3:
            cap, idx, seg = item
            return self.data.batch(cap, idx, seg=seg)
        cap, idx = item
        return self.data.batch(cap, idx)

    next_batch = next
    __next__ = next

    def __iter__(self):
        return self

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed, "chunk": self.chunk}

    def load_state_dict(self, st: dict) -> None:
        target_ep = int(st.get("epoch", 1))
        self.epoch = 0
        while self.epoch < target_ep:
            self._new_epoch()
        self.pos = int(st.get("pos", 0))


class PartStream:
    """Streams batches across one or more OMol part directories (`dirs`), keeping one part resident on `device`
    at a time and prefetching the next part (and its D3 cache) on the host in a background thread."""

    def __init__(
        self,
        dirs: list[str | Path],
        device: torch.device = torch.device("cpu"),
        batch_sizes: dict[int, int] | None = None,
        *,
        budget: int | None = None,
        bgt: int | None = None,
        ref: dict | None = None,
        cache_dir: str | Path | None = None,
        d3_reuse: list[tuple[np.ndarray, Path]] | None = None,
        chunk: int = 8,
        seed: int = 42,
        max_structs: int = 0,
        edges: tuple[int, ...] = DEFAULT_EDGES,
        d3_subtract: bool = True,
        units: str = "ev",
        state: dict | None = None,
        load_fn=None,
    ) -> None:
        import threading

        if not dirs:
            raise ValueError("PartStream requires at least one part directory")
        self.dirs = [str(Path(d)) for d in dirs]
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if batch_sizes is None:
            atom_bgt = int(budget if budget is not None else (bgt if bgt is not None else 8192))
            batch_sizes = batch_sizes_for(edges, atom_bgt, atom_bgt * 64)
        self.bs = batch_sizes
        self.ref = ref
        self.cache_dir = cache_dir
        self.d3_reuse = d3_reuse
        self.chunk = chunk
        self.seed = seed
        self.max_structs = max_structs
        self.edges = edges
        self.d3_subtract = d3_subtract
        self.units = units
        self._threading = threading
        self.load = load_fn or self._default_load

        st = state or {"cycle": 0, "k": 0, "pos": 0, "epoch": 1}
        if st.get("dirs"):
            self.dirs = [str(Path(d)) for d in st["dirs"]]
        self.cycle = int(st.get("cycle", 0))
        self.k = int(st.get("k", 0)) % len(self.dirs)
        self.data = self.load(self.dirs[self.k], self.k, host=False)
        self.stream = self._make_stream(self.data, self.cycle, self.k, epoch=int(st.get("epoch", 1)))
        self.stream.pos = int(st.get("pos", 0))
        self._next = None
        self.switch_log: list[dict] = []

    def _default_load(self, part_dir: str, k: int, host: bool = False) -> OmolBuckets:
        target_dev = torch.device("cpu") if host else self.device
        return OmolBuckets(
            part_dir,
            target_dev,
            max_structs=self.max_structs,
            seed=self.seed + k,
            edges=self.edges,
            d3_subtract=self.d3_subtract,
            ref=self.ref,
            cache_dir=self.cache_dir,
            d3_reuse=self.d3_reuse,
            units=self.units,
        )

    def _make_stream(self, data: OmolBuckets, cycle: int, k: int, epoch: int = 1) -> OmolStream:
        s = OmolStream(data, self.bs, chunk=self.chunk, seed=self.seed + 7919 * k + 104729 * cycle)
        while s.epoch < epoch:
            s._new_epoch()
        return s

    def _following(self) -> tuple[int, int, list[str]]:
        if self.k + 1 < len(self.dirs):
            return self.cycle, self.k + 1, self.dirs
        return self.cycle + 1, 0, self.dirs

    def prefetch(self) -> None:
        if len(self.dirs) < 2 or self._next is not None:
            return
        c, k, dirs = self._following()
        box: dict = {}

        def _run():
            try:
                d = self.load(dirs[k], k, True)
                box["stream"] = self._make_stream(d, c, k)
                box["data"] = d
            except Exception as exc:
                box["error"] = repr(exc)

        th = self._threading.Thread(target=_run, daemon=True)
        th.start()
        self._next = (c, k, dirs[k], dirs, th, box)

    def _switch(self) -> None:
        t0 = time.time()
        c, k, dirs = self._following()
        self.data = None
        self.stream = None
        data, stream, how = None, None, "direct"
        if self._next is not None and self._next[:3] == (c, k, dirs[k]):
            th, box = self._next[4], self._next[5]
            th.join(timeout=30.0)
            if box.get("data") is not None:
                data, stream, how = box["data"], box["stream"], "prefetched"
                data.to_device(self.device)
        self._next = None
        if data is None:
            data = self.load(dirs[k], k, False)
            stream = self._make_stream(data, c, k)
        self.cycle, self.k, self.dirs, self.data, self.stream = c, k, dirs, data, stream
        self.switch_log.append(
            {
                "cycle": c,
                "k": k,
                "dir": dirs[k],
                "how": how,
                "sec": round(time.time() - t0, 3),
                "structures": data.total,
                "batches": len(stream.plan),
            }
        )

    def next(self) -> dict:
        if len(self.dirs) > 1 and self.stream.pos >= len(self.stream.plan):
            self._switch()
        if len(self.dirs) > 1 and (self.stream.pos == 0 or (self._next is None and self.stream.pos > 10)):
            self.prefetch()
        return self.stream.next()

    next_batch = next
    __next__ = next

    def __iter__(self):
        return self

    def state(self) -> dict:
        return {
            "cycle": self.cycle,
            "k": self.k,
            "pos": self.stream.pos,
            "epoch": self.stream.epoch,
            "dirs": list(self.dirs),
        }

    state_dict = state

    def load_state_dict(self, st: dict) -> None:
        target_c = int(st.get("cycle", 0))
        target_k = int(st.get("k", 0)) % len(self.dirs)
        target_ep = int(st.get("epoch", 1))
        target_pos = int(st.get("pos", 0))
        if target_c != self.cycle or target_k != self.k:
            self.cycle = target_c
            self.k = target_k
            self.data = self.load(self.dirs[self.k], self.k, False)
        self.stream = self._make_stream(self.data, self.cycle, self.k, epoch=target_ep)
        self.stream.pos = target_pos

    def close(self) -> None:
        if self._next is not None:
            th = self._next[4]
            if th.is_alive():
                th.join(timeout=1.0)
            self._next = None


def batch_sizes_for(caps, atom_budget: int, pair_budget: int, max_bs: int = 1024) -> dict[int, int]:
    return {int(c): int(max(1, min(max_bs, atom_budget // c, pair_budget // (c * c)))) for c in caps}


