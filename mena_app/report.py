"""Builders for the report format every test returns, so the browser can show any result with one generic renderer.

report = {
  "headline": str,                       one sentence
  "tiles":    [{"key", "label", "value", "unit", "digits", "better": "lower"|"higher"|None, "status": "pass"|"fail"|"warn"|None, "hint"}],
  "tables":   [{"title", "columns": [...], "rows": [[...]], "note"}],
  "charts":   [{"id", "title", "kind": "line"|"bar"|"sticks", "x_label", "y_label", "logx", "logy", "series": [{"name", "x", "y"}], "labels"?}],
  "viewer":   {"frames": [{"symbols", "positions", "forces"?, "label"?}], "cell": 3x3 | null, "modes"?: [...]},
  "files":    [{"name", "label"}],          files written into the job folder
  "notes":    [str],                        caveats in plain English
  "cancelled": bool,
  "raw":      {...}                         the numbers behind the tables, for the JSON download
}
"""

from __future__ import annotations

import math

E_ANG_TO_DEBYE = 1.0 / 0.20819434


def tile(key: str, label: str, value, unit: str = "", digits: int | None = None, better: str | None = None,
         status: str | None = None, hint: str = "") -> dict:
    return {"key": key, "label": label, "value": value, "unit": unit, "digits": digits, "better": better, "status": status, "hint": hint}


def table(title: str, columns: list[str], rows: list[list], note: str = "") -> dict:
    return {"title": title, "columns": columns, "rows": rows, "note": note}


def chart(cid: str, title: str, series: list[dict], kind: str = "line", x_label: str = "", y_label: str = "", **kw) -> dict:
    return {"id": cid, "title": title, "kind": kind, "x_label": x_label, "y_label": y_label, "series": series, **kw}


def _non_zero_vec(arr, n: int, tol: float = 1e-4) -> list[float] | None:
    if arr is None:
        return None
    vals = [float(v) for v in arr]
    if len(vals) != n or max((abs(v) for v in vals), default=0.0) <= tol:
        return None
    return [round(v, 4) for v in vals]


def snapshot_physics(atoms, calc_results: dict | None = None) -> dict:
    """Lightweight per-frame physics snapshot from `calc.results` (zero extra model evaluations)."""
    res = calc_results if calc_results is not None else getattr(getattr(atoms, "calc", None), "results", None)
    if not isinstance(res, dict) or not res:
        return {}
    n = len(atoms)
    out: dict = {}
    q = _non_zero_vec(res.get("charges"), n, 1e-4)
    if q is not None:
        out["charges"] = q
    s_raw = res.get("spin_density")
    if s_raw is None:
        s_raw = res.get("spins")
    s = _non_zero_vec(s_raw, n, 1e-4)
    if s is not None:
        out["spins"] = s
    a_raw = res.get("atomic_polarizabilities")
    if a_raw is None:
        a_raw = res.get("alpha")
    al = _non_zero_vec(a_raw, n, 1e-4)
    if al is not None:
        out["alpha"] = al
    if not any(atoms.pbc):
        dip = res.get("dipole_debye")
        if dip is None and res.get("dipole") is not None:
            dip = [float(v) * E_ANG_TO_DEBYE for v in res["dipole"]]
        if dip is not None and len(dip) == 3:
            dv = [float(v) for v in dip]
            if math.hypot(dv[0], dv[1], dv[2]) > 0.01:
                out["dipole"] = [round(v, 4) for v in dv]
    if res.get("fmag") is not None and len(res["fmag"]) == n:
        out["fmag"] = [round(float(v), 4) for v in res["fmag"]]
    elif res.get("forces") is not None and len(res["forces"]) == n:
        out["fmag"] = [round(math.hypot(float(p[0]), float(p[1]), float(p[2])), 4) for p in res["forces"]]
    return out


def frame(atoms, forces=None, label: str = "", dipole=None, *, charges=None, spins=None, alpha=None, fmag=None, decomp=None) -> dict:
    n = len(atoms)
    f = {"symbols": atoms.get_chemical_symbols(), "positions": [[round(float(v), 4) for v in p] for p in atoms.get_positions()], "label": label}
    if any(atoms.pbc):
        f["cell"] = atoms.cell.tolist()
    if forces is not None:
        f["forces"] = [[round(float(v), 4) for v in p] for p in forces]
    elif fmag is not None and len(fmag) == n:
        f["fmag"] = [round(float(v), 4) for v in fmag]
    if dipole is not None:
        f["dipole"] = [round(float(v), 4) for v in dipole]
    q = _non_zero_vec(charges, n, 1e-4)
    if q is not None:
        f["charges"] = q
    s = _non_zero_vec(spins, n, 1e-4)
    if s is not None:
        f["spins"] = s
    al = _non_zero_vec(alpha, n, 1e-4)
    if al is not None:
        f["alpha"] = al
    if decomp is not None:
        f["decomp"] = decomp
    return f


def frame_with_physics(atoms, calc_results: dict | None = None, *, include_forces: bool = False, label: str = "") -> dict:
    res = calc_results if calc_results is not None else getattr(getattr(atoms, "calc", None), "results", None)
    snap = snapshot_physics(atoms, res)
    forces = res.get("forces") if (include_forces and isinstance(res, dict) and res.get("forces") is not None) else None
    return frame(
        atoms,
        forces=forces,
        label=label,
        dipole=snap.get("dipole"),
        charges=snap.get("charges"),
        spins=snap.get("spins"),
        alpha=snap.get("alpha"),
        fmag=None if forces is not None else snap.get("fmag"),
    )


def viewer(frames: list[dict], atoms=None, **kw) -> dict:
    cell = None
    if atoms is not None and atoms.pbc.any():
        cell = [[float(v) for v in row] for row in atoms.cell.array]
    return {"frames": frames, "cell": cell, **kw}


def subsample(n: int, limit: int) -> list[int]:
    if n <= limit:
        return list(range(n))
    step = (n - 1) / (limit - 1)
    return sorted({round(i * step) for i in range(limit)})
