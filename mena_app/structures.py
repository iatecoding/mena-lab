"""Structure input and analysis, shared by the server, the CLI and the worker. Uses ase + numpy, never torch."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.data import covalent_radii
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list
from ase.spacegroup import crystal as sg_crystal

from mena_app.paths import EXAMPLES_DIR

FILE_FORMATS = {".xyz": "extxyz", ".extxyz": "extxyz", ".cif": "cif", ".vasp": "vasp", ".poscar": "vasp"}
CUTOFF_A = 6.0
MP_HUBBARD_METALS = frozenset({"V", "Cr", "Mn", "Fe", "Co", "Ni", "Mo", "W"})


class StructureError(ValueError):
    """The input cannot be used; the message is shown to the user as it is."""


def _nh3(sign: float) -> Atoms:
    a = molecule("NH3")
    a.positions[:, 2] = sign * (a.positions[:, 2] - a.positions[0, 2])
    return a


def _malonaldehyde(right: bool) -> Atoms:
    symbols = ["C", "C", "C", "O", "O", "H", "H", "H", "H"]
    pos = np.array([
        [0.0069, 0.7737, 0.0],
        [-1.1528, 0.0706, 0.0],
        [1.2559, 0.0643, 0.0],
        [-1.2202, -1.2428, 0.0],
        [1.3386, -1.1649, 0.0],
        [-0.2768, -1.5672, 0.0],
        [-0.0090, 1.8520, 0.0],
        [-2.1236, 0.5538, 0.0],
        [2.1811, 0.6605, 0.0],
    ], dtype=float)
    if right:
        pos = pos[[0, 2, 1, 4, 3, 5, 6, 8, 7]].copy()
        pos[:, 0] *= -1.0
    return Atoms(symbols, positions=pos)


def _pt111_co() -> Atoms:
    slab = fcc111("Pt", size=(3, 3, 3), a=3.98, vacuum=8.0)
    top = slab.positions[slab.positions[:, 2] > slab.positions[:, 2].max() - 0.1]
    ctr = 0.5 * (slab.cell[0, :2] + slab.cell[1, :2])
    site = top[int(np.argmin(np.linalg.norm(top[:, :2] - ctr, axis=1)))]
    slab.extend(Atoms("CO", positions=[[site[0], site[1], site[2] + 1.88], [site[0], site[1], site[2] + 1.88 + 1.15]]))
    slab.center(vacuum=8.0, axis=2)
    slab.pbc = [True, True, False]
    return slab


def _xyz(name: str):
    return lambda: ase_read(EXAMPLES_DIR / f"{name}.xyz", format="extxyz")


# id -> (title, group, builder). Only safe, closed builders: a request can name one but cannot pass arguments.
BUILTINS = {
    "water": ("Water", "Molecules", lambda: molecule("H2O")),
    "ethanol": ("Ethanol", "Molecules", _xyz("ethanol")),
    "benzene": ("Benzene", "Molecules", lambda: molecule("C6H6")),
    "caffeine": ("Caffeine", "Molecules", _xyz("caffeine")),
    "aspirin": ("Aspirin", "Molecules", _xyz("aspirin")),
    "ibuprofen": ("Ibuprofen", "Molecules", _xyz("ibuprofen")),
    "nh3_up": ("Ammonia, pyramid up", "Reaction ends", lambda: _nh3(1.0)),
    "nh3_down": ("Ammonia, pyramid down", "Reaction ends", lambda: _nh3(-1.0)),
    "malonaldehyde_l": ("Malonaldehyde, enol left", "Reaction ends", lambda: _malonaldehyde(False)),
    "malonaldehyde_r": ("Malonaldehyde, enol right", "Reaction ends", lambda: _malonaldehyde(True)),
    "si": ("Silicon", "Crystals", lambda: bulk("Si", cubic=False)),
    "mgo": ("Magnesium oxide", "Crystals", lambda: bulk("MgO", "rocksalt", a=4.21)),
    "nacl": ("Rock salt", "Crystals", lambda: bulk("NaCl", "rocksalt", a=5.64)),
    "cu": ("Copper", "Crystals", lambda: bulk("Cu")),
    "tio2": ("Rutile TiO₂", "Crystals", lambda: sg_crystal(["Ti", "O"], [(0, 0, 0), (0.305, 0.305, 0)], spacegroup=136, cellpar=[4.59, 4.59, 2.96, 90, 90, 90])),
    "srtio3": ("Strontium titanate", "Crystals", lambda: sg_crystal(["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221, cellpar=[3.905] * 3 + [90] * 3)),
    "cssni3": ("CsSnI₃ perovskite", "Crystals", lambda: sg_crystal(["Cs", "Sn", "I"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221, cellpar=[6.219] * 3 + [90] * 3)),
    "diamond": ("Diamond", "Crystals", lambda: bulk("C", "diamond", a=3.567, cubic=True)),
    "cu111": ("Copper (111) surface", "Surfaces", lambda: fcc111("Cu", size=(3, 3, 4), vacuum=8.0)),
    "pt111_co": ("CO on Pt(111) catalyst", "Surfaces", _pt111_co),
}


def builtin_list() -> list[dict]:
    return [{"id": k, "title": v[0], "group": v[1]} for k, v in BUILTINS.items()]


def _from_smiles(smiles: str) -> Atoms:
    from mena_app.smiles import smiles_to_atoms

    return smiles_to_atoms(smiles)


def load(s: dict | str) -> Atoms:
    """A structure request {"format", "text"} (or CLI path/builtin string) -> ASE Atoms."""
    if isinstance(s, str):
        s = from_path(s)
    elif not isinstance(s, dict):
        raise StructureError("invalid structure specification: expected a dictionary or string")
    fmt, text = s.get("format"), s.get("text")
    if not isinstance(fmt, str) or not isinstance(text, str):
        raise StructureError("invalid structure specification: missing format or text")
    if fmt == "builtin":
        if text not in BUILTINS:
            raise StructureError(f"unknown example {text!r}")
        a = BUILTINS[text][2]()
    elif fmt == "smiles":
        a = _from_smiles(text)
    elif fmt in ("xyz", "extxyz", "cif", "vasp"):
        ase_fmt = {"xyz": "extxyz", "extxyz": "extxyz", "cif": "cif", "vasp": "vasp"}[fmt]
        try:
            a = ase_read(io.StringIO(text), format=ase_fmt, index=0)
        except Exception as exc:  # noqa: BLE001 - parsers raise anything on bad input
            raise StructureError(f"could not read this {fmt} file: {str(exc)[:160]}") from None
    else:
        raise StructureError(f"unsupported structure format {fmt!r}")
    if len(a) == 0:
        raise StructureError("the file holds no atoms")
    if not math_ok(a):
        raise StructureError("invalid geometry: check for NaN/Inf coordinates or zero-length periodic cell vectors")
    _check_species(a)
    lot = a.info.pop("lot", None)  # a level of theory named in the file is a request the plan checks, never read silently
    if isinstance(lot, str) and lot.strip():
        a.info["file_lot"] = lot.strip()
    _drop_vacuum_periodicity(a)
    return a


VACUUM_A = 6.0  # an empty gap this wide along a periodic axis means vacuum (no bulk material has one)


def _check_species(a: Atoms) -> None:
    occ = a.info.get("occupancy")
    if isinstance(occ, dict) and any(isinstance(v, dict) and (len(v) > 1 or any(float(x) < 0.999 for x in v.values()))
                                     for v in occ.values()):
        raise StructureError("this CIF has partially occupied (disordered) sites. Choose one species per site first, "
                             "for example an ordered supercell")
    if (a.numbers <= 0).any():
        raise StructureError("the structure contains dummy atoms (X, atomic number 0); remove them")


def _vacuum_axes(a: Atoms) -> list[int]:
    """Periodic axes along which the atoms leave an empty slab at least VACUUM_A wide (perpendicular distance)."""
    if len(a) == 0 or not a.pbc.any() or abs(float(np.linalg.det(a.cell.array))) < 1e-8:
        return []
    frac = a.get_scaled_positions(wrap=True)
    vol = abs(float(np.linalg.det(a.cell.array)))
    out = []
    for ax in range(3):
        if not a.pbc[ax]:
            continue
        width = vol / float(np.linalg.norm(np.cross(a.cell[(ax + 1) % 3], a.cell[(ax + 2) % 3])))
        s = np.sort(frac[:, ax])
        gaps = np.diff(np.concatenate([s, [s[0] + 1.0]]))
        if float(gaps.max()) * width >= VACUUM_A:
            out.append(ax)
    return out


def _drop_vacuum_periodicity(a: Atoms) -> None:
    """A slab or a molecule written as a fully periodic cell with vacuum (common for POSCAR and CIF) is treated as what
    it is: the vacuum directions stop being periodic. Bulk cells, liquids and porous frameworks have no such gap."""
    if not a.pbc.all():
        return
    vac = _vacuum_axes(a)
    if len(vac) == 3:
        a.pbc = False
        a.info["vacuum_note"] = "Vacuum on every side: treated as an isolated molecule."
    elif len(vac) == 1:
        pbc = [True, True, True]
        pbc[vac[0]] = False
        a.pbc = pbc
        a.info["vacuum_note"] = f"Vacuum along cell axis {'abc'[vac[0]]}: treated as a surface slab."


def from_path(path: str) -> dict:
    """Command-line argument -> structure request: a file, `smiles:CCO`, or an example name."""
    if path.startswith("smiles:"):
        return {"format": "smiles", "text": path[7:], "name": path[7:], "kind": "smiles", "source": path}
    if path in BUILTINS:
        return {"format": "builtin", "text": path, "name": BUILTINS[path][0], "kind": "builtin", "source": path}
    p = Path(path)
    if not p.is_file():
        raise StructureError(f"{path}: no such file, and not an example ({', '.join(BUILTINS)})")
    suffix = p.suffix.lower()
    fmt = {".xyz": "xyz", ".extxyz": "extxyz", ".cif": "cif", ".vasp": "vasp", ".poscar": "vasp"}.get(suffix)
    if fmt is None and p.name.upper().startswith(("POSCAR", "CONTCAR")):
        fmt = "vasp"
    if fmt is None:
        raise StructureError(f"{path}: use .xyz, .extxyz, .cif, .vasp or a POSCAR file")
    return {"format": fmt, "text": read_text_any(p), "name": p.stem, "kind": "path", "source": str(p)}


def read_text_any(p: Path) -> str:
    """Text of a file written by any editor: UTF-8 with or without a BOM (Windows Notepad), else Latin-1."""
    raw = p.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def domain_of(a: Atoms) -> str:
    n = int(np.sum(a.pbc))
    return "molecule" if n == 0 else ("crystal" if n == 3 else "slab")


def _effective_box(points: np.ndarray, cell: np.ndarray, pbc: np.ndarray, cutoff: float) -> np.ndarray:
    """Invertible 3x3 box where non-periodic axes are padded orthogonal vectors so mixed-PBC slabs work with zero c-vectors."""
    box = np.asarray(cell, dtype=np.float64).copy()
    if pbc.all() or not pbc.any():
        return box
    pad_len = (float(np.ptp(points)) if len(points) else 0.0) + 2.0 * cutoff + 20.0
    basis = [box[ax] / max(float(np.linalg.norm(box[ax])), 1e-12) for ax in range(3) if pbc[ax]]
    for ax in range(3):
        if not pbc[ax]:
            if len(basis) == 2:
                v = np.cross(basis[0], basis[1])
            else:
                e = np.eye(3)[int(np.argmin(np.abs(basis[0])))]
                v = e - float(np.dot(e, basis[0])) * basis[0]
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            basis.append(v)
            box[ax] = v * pad_len
    return box


def _count_edges_numpy(points: np.ndarray, box: np.ndarray, pbc: np.ndarray, cutoff: float) -> int:
    n = len(points)
    r2 = cutoff * cutoff
    if not pbc.any():
        total = 0
        for s in range(0, n, 512):
            d2 = np.sum((points[s : s + 512, None, :] - points[None, :, :]) ** 2, axis=-1)
            idx = np.arange(s, min(s + 512, n))
            d2[idx - s, idx] = np.inf
            total += int((d2 <= r2).sum())
        return total
    frac = points @ np.linalg.inv(box)
    vol = abs(float(np.linalg.det(box)))
    ranges = []
    for ax in range(3):
        if not pbc[ax]:
            ranges.append(np.array([0]))
        else:
            area = float(np.linalg.norm(np.cross(box[(ax + 1) % 3], box[(ax + 2) % 3])))
            h = vol / max(area, 1e-12)
            m = int(np.ceil(cutoff / h + (float(np.ptp(frac[:, ax])) if n else 0.0))) + 1
            ranges.append(np.arange(-m, m + 1))
    shifts = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    total = 0
    for sv, sh in zip(shifts @ box, shifts):
        zero_shift = bool((sh == 0).all())
        shifted = points + sv
        for s in range(0, n, 512):
            d2 = np.sum((points[s : s + 512, None, :] - shifted[None, :, :]) ** 2, axis=-1)
            if zero_shift:
                idx = np.arange(s, min(s + 512, n))
                d2[idx - s, idx] = np.inf
            total += int((d2 <= r2).sum())
    return total


def count_edges(a: Atoms, cutoff: float = CUTOFF_A) -> int:
    """Real sparse directed neighbour pairs within `cutoff` Å across all system sizes and periodicities."""
    points = np.asarray(a.positions, dtype=np.float64)
    pbc = np.asarray(a.pbc, dtype=bool)
    if len(points) == 0 or (len(points) == 1 and not pbc.any()):
        return 0
    box = _effective_box(points, np.asarray(a.cell.array, dtype=np.float64), pbc, cutoff) if pbc.any() else np.zeros((3, 3))
    try:
        import vesin
        nl = vesin.NeighborList(cutoff=cutoff, full_list=True)
        if not pbc.any():
            res = nl.compute(points=points, box=np.zeros((3, 3)), periodic=False, quantities="i")
            return int(len(res[0] if isinstance(res, (tuple, list)) else res))
        if pbc.all():
            res = nl.compute(points=points, box=box, periodic=True, quantities="i")
            return int(len(res[0] if isinstance(res, (tuple, list)) else res))
        _, S = nl.compute(points=points, box=box, periodic=True, quantities="iS")
        return int(np.all(S[:, ~pbc] == 0, axis=1).sum())
    except ImportError:
        return _count_edges_numpy(points, box, pbc, cutoff)


def _spacegroup(a: Atoms) -> str | None:
    from mena_app.crystal_symmetry import find_crystal_symmetry

    try:
        ds = find_crystal_symmetry(a.cell.array, a.get_scaled_positions(wrap=False), a.numbers, symprec=1e-3)
        return ds["label"] if ds is not None else None
    except Exception:  # noqa: BLE001 - optional nicety
        return None


def _int_hint(info: dict, keys: tuple[str, ...]) -> int | None:
    for k in keys:
        v = info.get(k)
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) and math.isfinite(float(v)):
            if float(v) == round(float(v)):
                return int(round(float(v)))
    return None


def _charge_hint(a: Atoms) -> int | None:
    """Charge hint for non-periodic structures from atoms.info or non-zero initial formal charges."""
    if domain_of(a) != "molecule":
        return None
    q = _int_hint(a.info, ("charge", "total_charge"))
    if q is not None:
        return q
    if "initial_charges" in a.arrays:
        s = float(a.get_initial_charges().sum())
        if math.isfinite(s) and abs(s) > 1e-6:
            return int(round(s))
    return None


def _spin_hint(a: Atoms) -> int | None:
    """Spin multiplicity hint (2S+1) for non-periodic structures from atoms.info."""
    if domain_of(a) != "molecule":
        return None
    s = _int_hint(a.info, ("spin", "multiplicity", "spin_multiplicity"))
    return s if s is not None and s >= 1 else None


def analyze(a: Atoms) -> dict:
    """What the page and the plan need to know about a structure, and what might be wrong with it."""
    dom = domain_of(a)
    electrons = int(a.numbers.sum())
    elems = sorted({s for s in a.get_chemical_symbols()})
    elem_set = set(elems)
    # +U for oxides and fluorides of the Materials Project U metals; a metal with a few adsorbed O atoms (CO on Fe) is not one
    syms = a.get_chemical_symbols()
    n_u = sum(s in MP_HUBBARD_METALS for s in syms)
    n_of = sum(s in ("O", "F") for s in syms)
    hubbard_u = dom != "molecule" and n_u > 0 and n_of >= 0.5 * n_u
    out = {"formula": a.get_chemical_formula("hill" if dom == "molecule" else "metal", empirical=False), "n_atoms": len(a), "domain": dom,
           "elements": elems, "electrons": electrons, "edges": count_edges(a), "hubbard_u": hubbard_u,
           "charge_hint": _charge_hint(a), "spin_hint": _spin_hint(a), "file_lot": a.info.get("file_lot"),
           "min_distance": None, "warnings": [a.info["vacuum_note"]] if a.info.get("vacuum_note") else []}
    if dom != "molecule":
        c = a.cell.cellpar()
        det = abs(float(np.linalg.det(a.cell.array)))
        vol = float(a.get_volume()) if bool(np.all(a.pbc)) and det > 1e-8 else None
        out["cell"] = [round(float(x), 4) for x in c] if all(math.isfinite(float(x)) for x in c) else None
        out["volume"] = round(vol, 3) if vol is not None else None
        if vol is not None:
            widths = [vol / float(np.linalg.norm(np.cross(a.cell[i - 2], a.cell[i - 1]))) for i in range(3)]
            out.update(widths=[round(w, 3) for w in widths],
                       density=round(float(a.get_masses().sum()) * 1.66054 / vol, 3), spacegroup=_spacegroup(a),
                       phonon_evaluations=_phonon_evaluations(a))
        else:
            out.update(widths=None, density=None, spacegroup=None)
    if 1 < len(a) <= 3000:
        try:
            i, j, d = neighbor_list("ijd", a, 3.0)
        except Exception:  # noqa: BLE001
            i, j, d = (), (), ()
        if len(d):
            out["min_distance"] = round(float(np.min(d)), 4)
            ratio = d / (covalent_radii[a.numbers[i]] + covalent_radii[a.numbers[j]])
            k = int(np.argmin(ratio))
            if ratio[k] < 0.6:
                sym = a.get_chemical_symbols()
                out["warnings"].append(f"Atoms {sym[i[k]]}{i[k] + 1} and {sym[j[k]]}{j[k] + 1} are only {d[k]:.2f} Å apart, which looks too close. Check the input.")
    if dom == "crystal" and out.get("density") is not None and out["density"] < 0.25 and "H" in out["elements"] and len(a) < 300:
        out["warnings"].append("This cell is mostly empty. If it is a molecule in a box, remove the cell so it is treated as a molecule.")
    return out


def _phonon_evaluations(a: Atoms) -> int:
    """Supercell evaluations for phonons: the symmetry-irreducible displacements, or 6N + 1 when only the identity applies."""
    from mena_app import phonon_symmetry

    plan = phonon_symmetry.symmetry_plan(a, 1) if len(a) <= 500 else None
    return plan.n_evaluations if plan is not None else 6 * len(a) + 1


def to_viewer(a: Atoms) -> dict:
    return {"symbols": a.get_chemical_symbols(), "positions": np.round(a.positions, 4).tolist(),
            "cell": np.round(a.cell.array, 4).tolist() if a.pbc.any() else None}


def tile_system(unit: Atoms, repeat: tuple[int, int, int] | list[int] | int) -> Atoms:
    """Repeat a periodic crystal/slab via `unit.repeat`, or tile a non-periodic molecule onto a clash-free 3D cubic grid."""
    r = (int(repeat), int(repeat), int(repeat)) if isinstance(repeat, int) else tuple(int(x) for x in repeat)
    if any(unit.pbc):
        return unit.repeat(r)
    nx, ny, nz = r
    n_copies = nx * ny * nz
    p0 = np.asarray(unit.positions, dtype=np.float64)
    p0 = p0 - p0.mean(axis=0, keepdims=True)
    max_diam = float(np.linalg.norm(p0[:, None, :] - p0[None, :, :], axis=-1).max()) if len(unit) > 1 else 0.0
    box_step = max_diam + 3.2
    shifts = [
        p0 + box_step * np.array([i - 0.5 * (nx - 1), j - 0.5 * (ny - 1), k - 0.5 * (nz - 1)], dtype=np.float64)
        for i in range(nx)
        for j in range(ny)
        for k in range(nz)
    ]
    out = Atoms(numbers=list(unit.numbers) * n_copies, positions=np.vstack(shifts), pbc=False, cell=[0.0, 0.0, 0.0])
    q_unit = _charge_hint(unit) or 0
    s_unit = _spin_hint(unit)
    if s_unit is None:
        s_unit = 2 if (int(unit.numbers.sum()) - q_unit) % 2 == 1 else 1
    out.info["charge"] = q_unit * n_copies
    out.info["spin"] = 1 if s_unit == 1 else (n_copies * (s_unit - 1) + 1)
    return out


def math_ok(a: Atoms) -> bool:
    if not (np.isfinite(a.positions).all() and np.isfinite(a.cell.array).all()):
        return False
    pbc = [bool(x) for x in a.pbc]
    cell = np.asarray(a.cell.array, dtype=np.float64)
    for ax in range(3):
        if pbc[ax] and float(np.linalg.norm(cell[ax])) <= 1e-6:
            return False
    if sum(pbc) == 2:
        ax0, ax1 = [ax for ax in range(3) if pbc[ax]]
        if float(np.linalg.norm(np.cross(cell[ax0], cell[ax1]))) <= 1e-6:
            return False
    if all(pbc) and abs(float(np.linalg.det(cell))) <= 1e-6:
        return False
    return True
