"""Model-agnostic pocket preparation, PDB/mmCIF fallback parser, and bio-dependency guards."""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import importlib
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any, Literal, Sequence

from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import atomic_numbers
import numpy as np

OPTIONAL_BIO_DEPS: tuple[str, ...] = ("gemmi", "pdbfixer", "openmm", "rdkit", "posebusters")
CHAIN_CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BACKBONE_ATOM_NAMES: frozenset[str] = frozenset(
    {"N", "CA", "C", "O", "OXT", "H", "HN", "H1", "H2", "H3", "HA", "HA2", "HA3"}
)
WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "TIP4", "SOL"})
_MODIFIED_AA_RESNAMES: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "ACE", "NME", "HIP", "HID", "HIE",
    "CYX", "CYM", "ASH", "GLH", "LYN", "ARN", "SEP", "TPO", "PTR", "SEC", "PYL",
})
ION_SPINS: dict[str, int] = {
    "[Mn+2]": 6, "[Fe+3]": 6, "[Fe+2]": 5, "[Cu+2]": 2, "[Cu+]": 1, "[Co+2]": 4, "[Ni+2]": 3,
    "[Zn+2]": 1, "[Mg+2]": 1, "[Ca+2]": 1, "[Na+]": 1, "[K+]": 1, "[Li+]": 1,
    "[Cl-]": 1, "[Br-]": 1, "[F-]": 1, "[I-]": 1,
}
# Residue/element name -> (formal_charge, spin_multiplicity, canonical_smiles)
MONATOMIC_ION_DEFAULTS: dict[str, tuple[int, int, str]] = {
    "ZN": (2, 1, "[Zn+2]"), "MG": (2, 1, "[Mg+2]"), "CA": (2, 1, "[Ca+2]"), "NA": (1, 1, "[Na+]"),
    "K": (1, 1, "[K+]"), "LI": (1, 1, "[Li+]"), "CL": (-1, 1, "[Cl-]"), "BR": (-1, 1, "[Br-]"),
    "F": (-1, 1, "[F-]"), "I": (-1, 1, "[I-]"), "IOD": (-1, 1, "[I-]"), "MN": (2, 6, "[Mn+2]"),
    "FE2": (2, 5, "[Fe+2]"), "FE": (3, 6, "[Fe+3]"), "FE3": (3, 6, "[Fe+3]"),
    "CU": (2, 2, "[Cu+2]"), "CU1": (1, 1, "[Cu+]"), "CO": (2, 4, "[Co+2]"), "NI": (2, 3, "[Ni+2]"),
}
_BRACKET_ION_RE = re.compile(r"^\[(?:\d*)([A-Za-z]{1,2})H?\d*([+-]\d*)?\]$")
_CIF_TOK_RE = re.compile(r"""(?<!\S)(?:'([^']*)'|"([^"]*)")(?=\s|$)|(\S+)""")


@dataclass
class _AtomRec:
    name: str
    elem: str
    z: int
    pos: np.ndarray


@dataclass
class _ResRec:
    chain: str
    resseq: str
    resname: str
    atoms: list[_AtomRec]
    atom_map: dict[str, _AtomRec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atom_map = {a.name: a for a in self.atoms}

    def add_or_replace(self, atom: _AtomRec) -> None:
        if atom.name in self.atom_map:
            self.atoms[self.atoms.index(self.atom_map[atom.name])] = atom
        else:
            self.atoms.append(atom)
        self.atom_map[atom.name] = atom


def _has_optional_module(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def available_bio_dependencies() -> dict[str, bool]:
    """Return availability flags for optional biomolecular packages."""
    return {name: _has_optional_module(name) for name in OPTIONAL_BIO_DEPS}


def require_bio_dependency(name: str, *, purpose: str = "") -> Any:
    """Import an optional bio dependency or raise a clear `ImportError`."""
    try:
        if name in sys.modules and sys.modules[name] is None:
            raise ModuleNotFoundError(f"Module {name!r} is disabled/unavailable")
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        why = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"Optional bio dependency '{name}' is required{why}, but is not installed in "
            f"{sys.executable} (available={available_bio_dependencies()})."
        ) from exc


def closed_mult(z: int | Sequence[int] | np.ndarray, charge: int | float = 0) -> int:
    """Lowest spin multiplicity (`1` if even electron count, `2` if odd) given atomic numbers and charge."""
    z_sum = int(z) if isinstance(z, (int, np.integer)) else int(np.asarray(z, dtype=np.int64).sum())
    return 1 if ((z_sum - int(round(float(charge)))) % 2 == 0) else 2


closed_shell_multiplicity = closed_mult


def residue_formal_charge(
    resname: str, atom_names: set[str] | frozenset[str] | Sequence[str], *, has_ss_bond: bool = False
) -> tuple[int, str | None]:
    """Infer residue formal charge and detect unprotonated non-disulfide CYS."""
    names, rn, q, warn = set(atom_names), resname.strip().upper(), 0, None
    if rn in ("ARG", "ARN"):
        q += 0 if rn == "ARN" else 1
    elif rn == "LYS" and "HZ3" in names:
        q += 1
    elif rn == "ASP" and "HD2" not in names and "HD1" not in names:
        q -= 1
    elif rn == "GLU" and "HE2" not in names and "HE1" not in names:
        q -= 1
    elif rn in ("HIS", "HIP") and (("HD1" in names and "HE2" in names) or rn == "HIP"):
        q += 1
    elif rn == "CYM":
        q -= 1
    elif rn == "CYS" and "HG" not in names and not has_ss_bond:
        warn = "CYS_NO_HG"
    if "H3" in names and "H2" in names:
        q += 1
    if "OXT" in names and "HXT" not in names:
        q -= 1
    return q, warn


def _parse_smiles_formal_charge(smiles: str) -> int | None:
    s = smiles.strip()
    m = _BRACKET_ION_RE.match(s)
    if m:
        chg = m.group(2) or ""
        return 0 if not chg else ((1 if chg[0] == "+" else -1) * (int(chg[1:]) if len(chg) > 1 else 1))
    if _has_optional_module("rdkit"):
        from rdkit import Chem  # type: ignore[import-not-found]

        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            return int(Chem.GetFormalCharge(mol))
    return None


def resolve_cofactor_charge_spin(
    z: Sequence[int] | np.ndarray, *, smiles: str | None = None, resname: str | None = None,
    formal_charge: int | float | None = None, spin: int | None = None,
) -> tuple[int, int]:
    """Resolve `(formal_charge, spin_multiplicity)` for a cofactor or metal ion."""
    z_arr, rn = np.asarray(z, dtype=np.int64), (resname or "").strip().upper()
    smi = smiles.strip() if smiles else None
    q: int | None = None if formal_charge is None else int(round(float(formal_charge)))
    if q is None and smi is not None:
        q = _parse_smiles_formal_charge(smi)
    if q is None and len(z_arr) == 1 and rn in MONATOMIC_ION_DEFAULTS:
        q = MONATOMIC_ION_DEFAULTS[rn][0]
    q = 0 if q is None else q

    if spin is not None:
        s = max(1, int(round(float(spin))))
    elif smi is not None and smi in ION_SPINS:
        s = ION_SPINS[smi]
    elif len(z_arr) == 1 and rn in MONATOMIC_ION_DEFAULTS:
        s = MONATOMIC_ION_DEFAULTS[rn][1]
    else:
        s = closed_mult(z_arr, q)
    return q, s


def build_complex_atoms(
    z: Sequence[int] | np.ndarray, pos: Sequence[Sequence[float]] | np.ndarray,
    frag_id: Sequence[int] | np.ndarray, frag_charge: Sequence[float],
    frag_spin: Sequence[int] | None = None, *,
    mode: Literal["pocket_fix", "pocket_sc_restr", "vacuum"] = "pocket_fix",
    ligand_frag_id: int = 1, pkt_is_bb: Sequence[bool] | Sequence[int] | np.ndarray | None = None,
    restraint_k: float = 0.5, lot: str = "omol_wb97mv", fix_cofactors: bool = True, fix_pocket_h: bool = False,
) -> Atoms:
    """Construct ASE `Atoms` compatible with `MenaCalculator.relax_batch`."""
    z_arr, pos_arr, fid_arr = np.asarray(z, dtype=np.int64), np.asarray(pos, dtype=np.float64), np.asarray(frag_id, dtype=np.int64)
    fq_list = [float(x) for x in frag_charge]
    fs_list = [max(1, int(x)) for x in frag_spin] if frag_spin is not None else [closed_mult(z_arr[fid_arr == f], fq_list[f]) for f in range(len(fq_list))]

    if mode == "vacuum":
        lig_mask = fid_arr == int(ligand_frag_id)
        if not np.any(lig_mask):
            raise ValueError(f"No atoms found for ligand_frag_id={ligand_frag_id}")
        atoms = Atoms(numbers=z_arr[lig_mask], positions=pos_arr[lig_mask], pbc=False)
        atoms.info.update({"charge": float(fq_list[int(ligand_frag_id)]), "spin": int(fs_list[int(ligand_frag_id)]), "lot": lot})
        return atoms
    if mode not in ("pocket_fix", "pocket_sc_restr"):
        raise ValueError(f"Unsupported build_complex_atoms mode={mode!r}")

    atoms = Atoms(numbers=z_arr, positions=pos_arr, pbc=False)
    atoms.arrays["frag_id"] = fid_arr.copy()
    atoms.info.update({
        "frag_charge": fq_list, "frag_spin": fs_list, "charge": float(sum(fq_list)),
        "spin": max(1, 1 + sum(max(0, s - 1) for s in fs_list)), "lot": lot,
    })
    is_lig, is_pkt = fid_arr == int(ligand_frag_id), fid_arr == 0
    is_cof = (fid_arr != 0) & (~is_lig)
    if mode == "pocket_fix":
        fix_idx = np.where((is_pkt | is_cof) if fix_cofactors else is_pkt)[0]
        if len(fix_idx) > 0:
            atoms.set_constraint(FixAtoms(indices=fix_idx))
        return atoms

    bb_arr = np.asarray(pkt_is_bb, dtype=bool) if pkt_is_bb is not None else np.zeros(len(z_arr), dtype=bool)
    fix_mask = (is_pkt & bb_arr) | (is_cof if fix_cofactors else False) | ((is_pkt & (z_arr == 1)) if fix_pocket_h else False)
    fix_idx = np.where(fix_mask)[0]
    if len(fix_idx) > 0:
        atoms.set_constraint(FixAtoms(indices=fix_idx))
    sc_free = is_pkt & (~fix_mask)
    if float(restraint_k) > 0.0 and np.any(sc_free):
        rk_vec = np.zeros(len(z_arr), dtype=np.float64)
        rk_vec[sc_free] = float(restraint_k)
        atoms.info["restraint_k"] = rk_vec
        atoms.arrays["restraint_ref"] = pos_arr.copy()
    return atoms


@dataclass
class PreparedPocketComplex:
    """Extracted pocket + ligand (+ cofactors/ions) container for MLIP relaxation and GBSA evaluation."""

    z: np.ndarray
    pos: np.ndarray
    frag_id: np.ndarray
    frag_charge: list[float]
    frag_spin: list[int]
    pkt_is_bb: np.ndarray
    n_pkt: int
    n_lig: int
    cofactor_sizes: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_atoms(
        self, mode: Literal["pocket_fix", "pocket_sc_restr", "vacuum"] = "pocket_fix", *,
        restraint_k: float = 0.5, lot: str = "omol_wb97mv", fix_cofactors: bool = True, fix_pocket_h: bool = False,
    ) -> Atoms:
        """Build ASE `Atoms` for `MenaCalculator` in `pocket_fix`, `pocket_sc_restr`, or `vacuum`."""
        return build_complex_atoms(
            self.z, self.pos, self.frag_id, self.frag_charge, self.frag_spin, mode=mode, ligand_frag_id=1,
            pkt_is_bb=self.pkt_is_bb, restraint_k=restraint_k, lot=lot, fix_cofactors=fix_cofactors, fix_pocket_h=fix_pocket_h,
        )


def _elem_to_z(sym: str) -> int:
    clean = re.sub(r"[^A-Za-z]", "", sym or "") or "C"
    for cand in (clean[0].upper() + clean[1:].lower(), clean[0].upper()):
        if cand in atomic_numbers:
            return int(atomic_numbers[cand])
    raise ValueError(f"Unrecognized chemical element symbol: {sym!r}")


def _append_res_atom(
    target: dict[tuple[str, str, str], _ResRec], chain: str, rseq: str, rname: str,
    aname: str, elem: str, pos: np.ndarray,
) -> None:
    rec = target.setdefault((chain, rseq, rname), _ResRec(chain=chain, resseq=rseq, resname=rname, atoms=[]))
    rec.add_or_replace(_AtomRec(name=aname, elem=elem, z=_elem_to_z(elem), pos=pos))


def _tokenize_cif_line(line: str) -> list[str]:
    return [g1 or g2 or g3 for g1, g2, g3 in _CIF_TOK_RE.findall(line.split("#", 1)[0] if line.lstrip().startswith("#") else line)]


def _parse_cif_text(text: str, poly_map: dict[Any, _ResRec], nonpoly_map: dict[Any, _ResRec]) -> None:
    lines, cols, rows, i = text.splitlines(), [], [], 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j, lcols = i + 1, []
            while j < len(lines) and lines[j].strip().startswith("_"):
                lcols.append(lines[j].strip()); j += 1
            if lcols and all(c.startswith("_atom_site.") for c in lcols):
                cols = [c.split(".", 1)[1] for c in lcols]
                while j < len(lines):
                    s = lines[j].strip()
                    if not s or s.startswith("#"):
                        j += 1; continue
                    if s == "loop_" or s.startswith(("_", "data_")):
                        break
                    toks = _tokenize_cif_line(lines[j])
                    while len(toks) < len(cols) and j + 1 < len(lines):
                        j += 1; toks.extend(_tokenize_cif_line(lines[j]))
                    rows.append(toks[: len(cols)]); j += 1
                break
        i += 1
    if not cols or not rows:
        raise ValueError("No _atom_site loop found in mmCIF structure")
    idx, first_model, seen = {c: k for k, c in enumerate(cols)}, None, {}
    def col(row: list[str], *names: str, default: str = "") -> str:
        for nm in names:
            k = idx.get(nm)
            if k is not None and k < len(row) and row[k] not in (".", "?"):
                return row[k]
        return default
    for row in rows:
        mnum = col(row, "pdbx_PDB_model_num", default="1")
        first_model = first_model or mnum
        if mnum != first_model:
            break
        rname = col(row, "auth_comp_id", "label_comp_id", default="UNK").upper()
        if rname in WATER_RESNAMES:
            continue
        aname, chain, rseq = col(row, "auth_atom_id", "label_atom_id", default="X"), col(row, "auth_asym_id", "label_asym_id", default="A"), col(row, "auth_seq_id", "label_seq_id", default="1")
        alt, occ = col(row, "label_alt_id", default=""), float(col(row, "occupancy", default="1.0"))
        akey = (chain, rseq, rname, aname)
        if akey in seen and (seen[akey][0] == "" or (alt != "" and occ <= seen[akey][1])):
            continue
        seen[akey] = (alt, occ)
        pos = np.array([float(col(row, "Cartn_x")), float(col(row, "Cartn_y")), float(col(row, "Cartn_z"))], dtype=np.float64)
        elem, grp = col(row, "type_symbol", default=aname[:1]), col(row, "group_PDB", default="ATOM")
        _append_res_atom(poly_map if (grp == "ATOM" or rname in _MODIFIED_AA_RESNAMES) else nonpoly_map, chain, rseq, rname, aname, elem, pos)


def _load_structure(path: Path, *, use_gemmi: bool = True) -> tuple[list[_ResRec], list[_ResRec]]:
    try:
        gemmi = require_bio_dependency("gemmi", purpose="structure loading") if use_gemmi else None
    except ImportError:
        gemmi = None
    if gemmi is not None:
        st = gemmi.read_structure(str(path))
        poly = [_ResRec(ch.name, str(r.seqid.num), r.name.upper(), [_AtomRec(a.name, a.element.name, int(a.element.atomic_number), np.array([a.pos.x, a.pos.y, a.pos.z])) for a in r]) for ch in st[0] for r in ch.get_polymer()]
        nonpoly = [_ResRec(ch.name, str(r.seqid.num), r.name.upper(), [_AtomRec(a.name, a.element.name, int(a.element.atomic_number), np.array([a.pos.x, a.pos.y, a.pos.z])) for a in r]) for ch in st[0] for r in ch.get_ligands() if r.name.upper() not in WATER_RESNAMES]
        return poly, nonpoly
    text = gzip.open(path, "rt", encoding="utf-8", errors="replace").read() if path.suffix.lower() == ".gz" else path.read_text(encoding="utf-8", errors="replace")
    poly_map: dict[Any, _ResRec] = {}
    nonpoly_map: dict[Any, _ResRec] = {}
    if path.name.lower().endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")) or "_atom_site." in text:
        _parse_cif_text(text, poly_map, nonpoly_map)
        return list(poly_map.values()), list(nonpoly_map.values())
    seen: dict[tuple[str, str, str, str], tuple[str, float]] = {}
    for ln in text.splitlines():
        if ln.startswith("ENDMDL"):
            break
        rec_type = ln[:6]
        if rec_type not in ("ATOM  ", "HETATM") or len(ln) < 54:
            continue
        aname, alt, rname = ln[12:16].strip(), ln[16:17].strip(), ln[17:20].strip().upper()
        if rname in WATER_RESNAMES:
            continue
        chain, rseq = ln[21:22].strip() or "A", ln[22:27].strip() or "1"
        occ = float(ln[54:60].strip()) if len(ln) >= 60 and ln[54:60].strip() else 1.0
        akey = (chain, rseq, rname, aname)
        if akey in seen and (seen[akey][0] == "" or (alt != "" and occ <= seen[akey][1])):
            continue
        seen[akey] = (alt, occ)
        pos = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])], dtype=np.float64)
        raw_el = ln[76:78].strip() if len(ln) >= 78 else ""
        if not raw_el:
            lead = re.sub(r"[^A-Za-z]", "", aname)
            raw_el = lead[:2] if lead[:2].upper() in ("CL", "BR", "FE", "ZN", "MG", "CA", "NA", "MN", "CU", "CO", "NI", "SE") else (lead[:1] or "C")
        _append_res_atom(poly_map if (rec_type == "ATOM  " or rname in _MODIFIED_AA_RESNAMES) else nonpoly_map, chain, rseq, rname, aname, raw_el, pos)
    return list(poly_map.values()), list(nonpoly_map.values())


def _residues_to_pdb(poly: Sequence[_ResRec]) -> tuple[str, dict[str, str]]:
    lines, chain_map, serial, prev_chain = [], {}, 1, None
    for res in poly:
        if res.chain not in chain_map:
            if prev_chain is not None:
                lines.append("TER")
            chain_map[res.chain] = CHAIN_CHARS[len(chain_map) % len(CHAIN_CHARS)]
        prev_chain, cid = res.chain, chain_map[res.chain]
        num_m = re.match(r"^(-?\d+)(.*)$", res.resseq.strip())
        resnum, icode = (int(num_m.group(1)) if num_m else 1), ((num_m.group(2)[:1] if num_m else "") or " ")
        for a in res.atoms:
            aname = f" {a.name:<3}" if len(a.name) < 4 else a.name[:4]
            lines.append(f"ATOM  {serial % 100000:5d} {aname:4s} {res.resname[:3]:>3s} {cid:1s}{resnum:4d}{icode:1s}   {a.pos[0]:8.3f}{a.pos[1]:8.3f}{a.pos[2]:8.3f}  1.00  0.00          {a.elem.upper():>2s}")
            serial += 1
    return "\n".join([*lines, "TER", "END"]), chain_map


from lam4.prep.pocket_complex import (  # noqa: E402
    LigandValidationResult, extract_pocket_complex, reconstruct_ligand_mol,
    run_posebusters_check, validate_ligand_integrity,
)

__all__ = [
    "BACKBONE_ATOM_NAMES", "CHAIN_CHARS", "ION_SPINS", "LigandValidationResult", "MONATOMIC_ION_DEFAULTS",
    "OPTIONAL_BIO_DEPS", "PreparedPocketComplex", "WATER_RESNAMES", "available_bio_dependencies",
    "build_complex_atoms", "closed_mult", "closed_shell_multiplicity", "extract_pocket_complex",
    "reconstruct_ligand_mol", "require_bio_dependency", "residue_formal_charge",
    "resolve_cofactor_charge_spin", "run_posebusters_check", "validate_ligand_integrity",
]
