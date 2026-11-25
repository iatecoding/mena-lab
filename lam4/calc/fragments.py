"""Fragment charge/spin parsing and batch tensor assembly for MenaCalculator."""
from __future__ import annotations

from typing import Any

from ase import Atoms
import numpy as np
import torch
from torch import Tensor


def resolve_charge_and_spin(
    atoms: Atoms, charge: float | None = None, spin: int | None = None, *,
    pbc_any: bool = False, allow_pbc_charge: bool = False,
) -> tuple[float, int]:
    """Resolve scalar total `(charge, spin_multiplicity)` from `Atoms` and optional overrides."""
    if pbc_any and not allow_pbc_charge:
        return 0.0, 1
    q_val = float(charge if charge is not None else atoms.info.get("charge", 0.0))
    s = spin if spin is not None else atoms.info.get("spin", atoms.info.get("multiplicity", None))
    if s is None:
        if pbc_any:
            s_val = 1
        else:
            n_elec = int(np.sum(atoms.get_atomic_numbers())) - int(round(q_val))
            s_val = 1 if (n_elec % 2 == 0) else 2
    else:
        s_val = max(1, int(round(float(s))))
    return q_val, s_val


def _parse_frag_dict_keys(raw: dict[Any, Any], label: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for k, v in raw.items():
        if isinstance(k, bool):
            raise ValueError(f"{label} dict keys must be non-negative integers, got {k!r}")
        try:
            ki = int(k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} dict keys must be non-negative integers, got {k!r}") from exc
        if isinstance(k, (float, np.floating)) and float(k) != float(ki):
            raise ValueError(f"{label} dict keys must be non-negative integers, got {k!r}")
        if ki < 0:
            raise ValueError(f"{label} dict keys must be >= 0, got {ki}")
        fv = float(v)
        if not np.isfinite(fv):
            raise ValueError(f"{label}[{ki}] must be finite, got {v!r}")
        out[ki] = fv
    return out


def resolve_charge_spin_fragments(
    atoms: Atoms,
    z_np: np.ndarray,
    charge: float | None = None,
    spin: int | None = None,
    *,
    pbc_any: bool = False,
    domain: str = "molecule",
    allow_pbc_charge: bool = False,
) -> tuple[float, int, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Resolve total `(charge, spin)` and optional `(frag_id [N], frag_charge [F], frag_spin [F])` from `Atoms`."""
    raw_fid = atoms.arrays.get("frag_id")
    if raw_fid is None:
        raw_fid = atoms.info.get("frag_id")
    raw_fq = atoms.info.get("frag_charge")
    raw_fs = atoms.info.get("frag_spin")

    if raw_fid is None:
        if domain == "molecule" and (raw_fq is not None or raw_fs is not None):
            raise ValueError("frag_charge / frag_spin provided on Atoms without frag_id")
        q_val, s_val = resolve_charge_and_spin(
            atoms, charge, spin, pbc_any=pbc_any, allow_pbc_charge=(allow_pbc_charge and domain == "molecule"),
        )
        return q_val, s_val, None, None, None

    if domain == "crystal":
        q_val, s_val = resolve_charge_and_spin(atoms, charge, spin, pbc_any=pbc_any)
        return q_val, s_val, None, None, None

    n_atoms = len(z_np)
    fid_arr = raw_fid.detach().cpu().numpy() if torch.is_tensor(raw_fid) else np.asarray(raw_fid)
    if fid_arr.ndim != 1 or len(fid_arr) != n_atoms:
        raise ValueError(f"frag_id must be a 1D integer array of length {n_atoms}, got shape {fid_arr.shape}")
    if np.issubdtype(fid_arr.dtype, np.bool_):
        raise ValueError("frag_id must be a 1D integer array, got bool dtype")
    if np.issubdtype(fid_arr.dtype, np.integer):
        fid_np = np.ascontiguousarray(fid_arr, dtype=np.int64)
    elif np.issubdtype(fid_arr.dtype, np.floating) and np.all(np.isfinite(fid_arr)) and np.all(fid_arr == np.floor(fid_arr)):
        fid_np = np.ascontiguousarray(fid_arr, dtype=np.int64)
    else:
        raise ValueError(f"frag_id must be a 1D integer array, got dtype {fid_arr.dtype}")

    if n_atoms > 0 and int(fid_np.min()) < 0:
        raise ValueError(f"frag_id must be non-negative, got min={int(fid_np.min())}")
    if raw_fq is None:
        raise ValueError("frag_id is set on Atoms, but atoms.info['frag_charge'] is omitted")

    u_ids = np.unique(fid_np)
    if isinstance(raw_fq, dict):
        fq_map = _parse_frag_dict_keys(raw_fq, "frag_charge")
        u_set = set(u_ids.tolist())
        missing = sorted(u_set - fq_map.keys())
        extra = sorted(fq_map.keys() - u_set)
        if missing or extra:
            raise ValueError(f"frag_charge dict keys mismatch frag_id unique IDs (missing={missing}, extra={extra})")
        fid_contig = np.ascontiguousarray(np.searchsorted(u_ids, fid_np), dtype=np.int64)
        n_frags = len(u_ids)
        frag_q = np.array([fq_map[int(u)] for u in u_ids], dtype=np.float64)
    else:
        fq_arr = raw_fq.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(raw_fq) else np.asarray(raw_fq, dtype=np.float64)
        if fq_arr.ndim != 1 or not np.all(np.isfinite(fq_arr)):
            raise ValueError(f"frag_charge must be a finite 1D sequence, got shape {fq_arr.shape}")
        n_frags = int(len(fq_arr))
        if n_atoms > 0:
            if n_frags == 0 or int(fid_np.max()) >= n_frags:
                raise ValueError(f"frag_id max={int(fid_np.max())} is out of range for frag_charge of length {n_frags}")
            if not np.array_equal(u_ids, np.arange(n_frags, dtype=np.int64)):
                raise ValueError(f"frag_id must cover contiguous 0..{n_frags - 1}, got {u_ids.tolist()}")
        fid_contig = fid_np
        frag_q = np.ascontiguousarray(fq_arr, dtype=np.float64)

    q_sum = float(frag_q.sum())
    if charge is not None:
        q_val = float(charge)
        if abs(q_sum - q_val) > 1e-4:
            raise ValueError(f"sum(frag_charge)={q_sum} disagrees with charge override={q_val}")
    else:
        q_info = atoms.info.get("charge", None)
        if q_info is not None and abs(q_sum - float(q_info)) > 1e-4:
            raise ValueError(f"sum(frag_charge)={q_sum} disagrees with atoms.info['charge']={float(q_info)}")
        q_val = q_sum

    if raw_fs is not None:
        if isinstance(raw_fs, dict):
            fs_map = _parse_frag_dict_keys(raw_fs, "frag_spin")
            u_set = set(u_ids.tolist())
            missing = sorted(u_set - fs_map.keys())
            extra = sorted(fs_map.keys() - u_set)
            if missing or extra:
                raise ValueError(f"frag_spin dict keys mismatch frag_id unique IDs (missing={missing}, extra={extra})")
            fs_arr = np.array([fs_map[int(u)] for u in u_ids], dtype=np.float64)
        else:
            fs_arr = raw_fs.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(raw_fs) else np.asarray(raw_fs, dtype=np.float64)
            if fs_arr.ndim != 1 or len(fs_arr) != n_frags:
                raise ValueError(f"frag_spin must be a 1D sequence of length {n_frags}, got shape {fs_arr.shape}")
        if not np.all(np.isfinite(fs_arr)) or (len(fs_arr) > 0 and float(fs_arr.min()) < 1.0 - 1e-6):
            raise ValueError(f"frag_spin multiplicities must be finite and >= 1, got {fs_arr.tolist()}")
        frag_s = np.ascontiguousarray(np.rint(fs_arr), dtype=np.int64)
    else:
        frag_s = np.array(
            [1 + (int(z_np[fid_contig == f].sum()) - int(round(frag_q[f]))) % 2 for f in range(n_frags)],
            dtype=np.int64,
        )

    s_raw = spin if spin is not None else atoms.info.get("spin", atoms.info.get("multiplicity", None))
    if s_raw is not None:
        s_val = max(1, int(round(float(s_raw))))
    elif n_frags == 1:
        s_val = int(frag_s[0])
    else:
        s_val = 1 + (int(z_np.sum()) - int(round(q_val))) % 2

    return q_val, s_val, fid_contig, frag_q, frag_s


def assemble_fragment_tensors(
    items: list[tuple[int, float, int, np.ndarray | None, np.ndarray | None, np.ndarray | None]],
    dev: torch.device,
    dt: torch.dtype,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """Concatenate per-graph fragment IDs, charges, and spins, promoting unfragmented graphs if mixed."""
    if not items or all(it[3] is None for it in items):
        return None, None, None
    fids: list[np.ndarray] = []
    fqs: list[np.ndarray] = []
    fss: list[np.ndarray] = []
    f_off = 0
    for n, q_val, s_val, fid_np, fq_np, fs_np in items:
        if fid_np is None:
            fids.append(np.full(n, f_off, dtype=np.int64))
            fqs.append(np.array([float(q_val)], dtype=np.float64))
            fss.append(np.array([float(s_val)], dtype=np.float64))
            f_off += 1
        else:
            assert fq_np is not None and fs_np is not None
            fids.append(np.asarray(fid_np, dtype=np.int64) + f_off)
            fqs.append(np.asarray(fq_np, dtype=np.float64))
            fss.append(np.asarray(fs_np, dtype=np.float64))
            f_off += len(fq_np)
    frag_id_t = torch.from_numpy(np.ascontiguousarray(np.concatenate(fids), dtype=np.int64)).to(dev)
    frag_q_t = torch.from_numpy(np.ascontiguousarray(np.concatenate(fqs))).to(device=dev, dtype=dt)
    frag_s_t = torch.from_numpy(np.ascontiguousarray(np.concatenate(fss))).to(device=dev, dtype=dt)
    return frag_id_t, frag_q_t, frag_s_t


def detect_covalent_fragments(
    atoms: Atoms,
    *,
    bond_scale: float = 1.25,
    bond_tol: float = 0.0,
    max_cutoff: float = 4.0,
    default_charge: float = 0.0,
    return_meta: bool = False,
):
    """Detect PBC-aware covalent molecular fragments via Cordero covalent radii + BFS with shift-cycle detection.

    Returns `(frag_id [N] int64, frag_charge [F] float64, frag_spin [F] int64)` with contiguous IDs `0..F-1`
    (plus `meta` dict with `has_periodic_cycle` and `all_even_electrons` when `return_meta=True`).
    Separates covalent bonds (e.g. O-H ~0.96 A <= 1.21 A) from H-bonds (~1.8 A) cleanly on water, X23, and ice.
    """
    from collections import deque
    from ase.data import covalent_radii
    from lam4.calc.physics import cached_vesin_nl, effective_periodic_cell

    z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    n = len(z)
    if n == 0:
        empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64))
        return (*empty, {"has_periodic_cycle": False, "all_even_electrons": True}) if return_meta else empty
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    pbc_t = tuple(bool(x) for x in atoms.get_pbc())
    pbc_any = any(pbc_t)
    box = effective_periodic_cell(atoms, float(max_cutoff))[0] if pbc_any else np.zeros((3, 3), dtype=np.float64)
    rcov = np.pad(np.asarray(covalent_radii, dtype=np.float64), (0, 20), constant_values=1.5)
    z_clip = np.clip(z, 0, len(rcov) - 1)
    if pbc_any:
        ii, jj, dd, ss = cached_vesin_nl(float(max_cutoff)).compute(points=pos, box=box, periodic=pbc_t, quantities="ijdS")
        ss_a = np.asarray(ss, dtype=np.int64)
    else:
        ii, jj, dd = cached_vesin_nl(float(max_cutoff)).compute(points=pos, box=box, periodic=(False, False, False), quantities="ijd")
        ss_a = None
    ii_a, jj_a, dd_a = np.asarray(ii, dtype=np.int64), np.asarray(jj, dtype=np.int64), np.asarray(dd, dtype=np.float64)
    thresh = float(bond_scale) * (rcov[z_clip[ii_a]] + rcov[z_clip[jj_a]]) + float(bond_tol)
    bonded = dd_a <= thresh

    adj: list[list[tuple[int, tuple[int, int, int]]]] = [[] for _ in range(n)]
    b_i = ii_a[bonded].tolist()
    b_j = jj_a[bonded].tolist()
    if ss_a is not None:
        for u, v, sh in zip(b_i, b_j, ss_a[bonded].tolist()):
            adj[u].append((v, (int(sh[0]), int(sh[1]), int(sh[2]))))
    else:
        for u, v in zip(b_i, b_j):
            adj[u].append((v, (0, 0, 0)))

    fid = np.full(n, -1, dtype=np.int64)
    disp: list[tuple[int, int, int] | None] = [None] * n
    has_periodic_cycle = False
    n_f = 0
    for start in range(n):
        if fid[start] != -1:
            continue
        fid[start] = n_f
        disp[start] = (0, 0, 0)
        q_bfs: deque[int] = deque([start])
        while q_bfs:
            u = q_bfs.popleft()
            du = disp[u]
            assert du is not None
            for v, (sx, sy, sz) in adj[u]:
                cand = (du[0] + sx, du[1] + sy, du[2] + sz)
                if fid[v] == -1:
                    fid[v] = n_f
                    disp[v] = cand
                    q_bfs.append(v)
                elif disp[v] != cand:
                    has_periodic_cycle = True
        n_f += 1

    fq = np.full(n_f, float(default_charge), dtype=np.float64)
    z_sum = np.bincount(fid, weights=z, minlength=n_f)
    z_int = np.rint(z_sum).astype(np.int64)
    fs = np.array([1 + (int(z_int[f]) - int(round(fq[f]))) % 2 for f in range(n_f)], dtype=np.int64)
    if return_meta:
        return fid, fq, fs, {"has_periodic_cycle": has_periodic_cycle, "all_even_electrons": bool(np.all((z_int % 2) == 0))}
    return fid, fq, fs


def auto_covalent_fragments(
    atoms: Atoms,
    q_total: float,
    mode: str | bool = "auto",
    *,
    bond_scale: float = 1.25,
    bond_tol: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str | None]:
    """Auto-detect neutral finite molecular fragments for 3D periodic molecular SPME.

    Returns `(fid, fq, fs, skip_reason)` where `skip_reason` is `None` when fragments were assigned, else one of:
    `'disabled'`, `'net_charge'`, `'periodic_network'`, `'odd_electron_fragments'`.
    """
    m = mode.strip().lower() if isinstance(mode, str) else mode
    if m in (False, "false", "off", "none", "0"):
        return None, None, None, "disabled"
    if abs(float(q_total)) > 1e-6:
        return None, None, None, "net_charge"
    fid, fq, fs, meta = detect_covalent_fragments(atoms, bond_scale=bond_scale, bond_tol=bond_tol, return_meta=True)
    if meta["has_periodic_cycle"]:
        return None, None, None, "periodic_network"
    if m == "auto" and not meta["all_even_electrons"]:
        return None, None, None, "odd_electron_fragments"
    return fid, fq, fs, None


def assign_covalent_fragments(
    atoms: Atoms,
    *,
    copy: bool = False,
    bond_scale: float = 1.25,
    bond_tol: float = 0.0,
    max_cutoff: float = 4.0,
) -> Atoms:
    """Attach PBC-aware covalent `frag_id`, `frag_charge`, and `frag_spin` onto `atoms` (or a copy)."""
    out = atoms.copy() if copy else atoms
    fid, fq, fs = detect_covalent_fragments(out, bond_scale=bond_scale, bond_tol=bond_tol, max_cutoff=max_cutoff)
    out.arrays["frag_id"] = fid
    out.info["frag_charge"] = fq
    out.info["frag_spin"] = fs
    return out
