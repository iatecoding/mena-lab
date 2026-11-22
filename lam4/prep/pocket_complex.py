"""Pocket cropping, terminal ACE/NME capping, disulfide/inter-chain audit, and ligand integrity checks."""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from lam4.prep import (
    BACKBONE_ATOM_NAMES, MONATOMIC_ION_DEFAULTS, PreparedPocketComplex,
    _AtomRec, _ResRec, _load_structure, _residues_to_pdb, closed_mult,
    require_bio_dependency, residue_formal_charge, resolve_cofactor_charge_spin,
)


@dataclass
class LigandValidationResult:
    """Integrity checks preventing silent chirality inversion, bond cleavage, or valence errors."""

    valid: bool
    sanitized: bool
    connected: bool
    no_radicals: bool
    inchi_match: bool
    stereo_match: bool
    max_bond_delta_A: float
    broken_bonds: list[tuple[int, int, float]]
    inverted_centers: list[int]
    planarized_centers: list[int]
    inchi_key_before: str | None
    inchi_key_after: str | None
    isomeric_smiles_before: str
    isomeric_smiles_after: str
    message: str = ""


def _protonate_with_pdbfixer(
    pdb_str: str, lig_heavy: np.ndarray, pocket_radius: float, ph: float
) -> tuple[list[_ResRec], set[int], dict[int, int], list[tuple[tuple[int, _AtomRec], tuple[int, _AtomRec]]], float]:
    require_bio_dependency("pdbfixer", purpose="protein protonation & missing atom repair")
    require_bio_dependency("openmm", purpose="PDBFixer topology & coordinates")
    from pdbfixer import PDBFixer  # type: ignore[import-not-found]

    best: Any = None
    for _ in range(4):
        fx = PDBFixer(pdbfile=io.StringIO(pdb_str))
        fx.findMissingResidues(); fx.missingResidues = {}
        fx.findNonstandardResidues(); fx.replaceNonstandardResidues()
        fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(float(ph))
        pos_A = np.array([[v.x, v.y, v.z] for v in fx.positions.value_in_unit(fx.positions.unit)], dtype=np.float64) * 10.0
        recs, atom_map = [], {}
        for r in fx.topology.residues():
            ratoms = [_AtomRec(a.name, a.element.symbol, int(a.element.atomic_number), pos_A[a.index]) for a in r.atoms()]
            for a, ar in zip(r.atoms(), ratoms):
                atom_map[a.index] = (r.index, ar)
            recs.append(_ResRec(r.chain.id, str(r.id), r.name.upper(), ratoms))
        sel = {ri for ri, rec in enumerate(recs) if any(a.z > 1 and np.min(np.linalg.norm(lig_heavy - a.pos[None, :], axis=-1)) <= pocket_radius for a in rec.atoms)}
        ss_partners, inter_bonds = {}, []
        for a1, a2 in fx.topology.bonds():
            if a1.residue.index != a2.residue.index:
                r1, ar1 = atom_map[a1.index]; r2, ar2 = atom_map[a2.index]
                inter_bonds.append(((r1, ar1), (r2, ar2)))
                if a1.name == "SG" and a2.name == "SG":
                    ss_partners[r1], ss_partners[r2] = r2, r1
        h_pts = [a.pos for ri in sel for a in recs[ri].atoms if a.z == 1]
        h_min = 99.0
        if len(h_pts) > 1:
            hp = np.asarray(h_pts, dtype=np.float64)
            dh = np.linalg.norm(hp[:, None, :] - hp[None, :, :], axis=-1); np.fill_diagonal(dh, 99.0)
            h_min = float(dh.min())
        best = (recs, sel, ss_partners, inter_bonds, h_min)
        if h_min >= 0.85:
            break
    return best


def _geometric_bonds(recs: Sequence[_ResRec]) -> tuple[dict[int, int], list[tuple[tuple[int, _AtomRec], tuple[int, _AtomRec]]]]:
    ss_partners: dict[int, int] = {}
    inter_bonds = []
    cys_sg = [(i, r.atom_map["SG"]) for i, r in enumerate(recs) if "SG" in r.atom_map]
    for k, (i1, sg1) in enumerate(cys_sg):
        for i2, sg2 in cys_sg[k + 1 :]:
            if float(np.linalg.norm(sg1.pos - sg2.pos)) < 2.5:
                ss_partners[i1], ss_partners[i2] = i2, i1; inter_bonds.append(((i1, sg1), (i2, sg2)))
    for i in range(len(recs) - 1):
        r1, r2 = recs[i], recs[i + 1]
        if r1.chain == r2.chain and "C" in r1.atom_map and "N" in r2.atom_map and float(np.linalg.norm(r1.atom_map["C"].pos - r2.atom_map["N"].pos)) < 1.9:
            inter_bonds.append(((i, r1.atom_map["C"]), (i + 1, r2.atom_map["N"])))
    return ss_partners, inter_bonds


def _staggered_methyl_hs(p_c: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    u = (p_a - p_c) / max(float(np.linalg.norm(p_a - p_c)), 1e-8)
    v_raw = (p_b - p_a) - float(np.dot(p_b - p_a, u)) * u
    if float(np.linalg.norm(v_raw)) < 1e-6:
        t = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        v_raw = t - float(np.dot(t, u)) * u
    v, sin_t, cos_t = v_raw / float(np.linalg.norm(v_raw)), math.sqrt(8.0) / 3.0, -1.0 / 3.0
    w = np.cross(u, v)
    return np.array([p_c + 1.09 * (cos_t * u + sin_t * (math.cos(math.pi + k * 2.0 * math.pi / 3.0) * v + math.sin(math.pi + k * 2.0 * math.pi / 3.0) * w)) for k in range(3)], dtype=np.float64)


def _set_conformer(mol: Any, pos_np: np.ndarray) -> Any:
    from rdkit import Chem  # type: ignore[import-not-found]

    conf = Chem.Conformer(len(pos_np))
    for k, p in enumerate(pos_np):
        conf.SetAtomPosition(k, (float(p[0]), float(p[1]), float(p[2])))
    mol.RemoveAllConformers(); mol.AddConformer(conf, assignId=True)
    return mol


def _mol_arrays(mol: Any, q_ov: int | float | None = None, s_ov: int | None = None) -> tuple[np.ndarray, np.ndarray, int, int]:
    from rdkit import Chem  # type: ignore[import-not-found]

    conf = mol.GetConformer()
    z_arr = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    p_arr = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z] for k in range(len(z_arr))], dtype=np.float64)
    q = int(round(float(q_ov))) if q_ov is not None else int(Chem.GetFormalCharge(mol))
    return z_arr, p_arr, q, (int(s_ov) if s_ov is not None else closed_mult(z_arr, q))


def _rm_hs_params() -> Any:
    from rdkit import Chem  # type: ignore[import-not-found]

    ps = Chem.RemoveHsParameters()
    ps.removeDefiningBondStereo = ps.removeDegreeZero = ps.removeWithWedgedBond = True
    return ps


def reconstruct_ligand_mol(
    template_smiles_or_mol: Any, pos: Sequence[Sequence[float]] | np.ndarray | None = None, *,
    z: Sequence[int] | np.ndarray | None = None, protonate: bool = True,
) -> Any:
    """Reconstruct sanitized RDKit Mol (bond orders, 3D chiral tags, explicit Hs) from SMILES/Mol and coords."""
    require_bio_dependency("rdkit", purpose="ligand reconstruction")
    from rdkit import Chem  # type: ignore[import-not-found]
    from rdkit.Chem import AllChem, rdDetermineBonds  # type: ignore[import-not-found]

    tmpl_raw = Chem.MolFromSmiles(template_smiles_or_mol) if isinstance(template_smiles_or_mol, str) else Chem.Mol(template_smiles_or_mol)
    if tmpl_raw is None:
        raise ValueError(f"Failed to parse ligand template: {template_smiles_or_mol!r}")
    if pos is None:
        Chem.SanitizeMol(tmpl_raw)
        return Chem.AddHs(tmpl_raw, addCoords=True) if protonate else tmpl_raw

    pos_np, tmpl = np.asarray(pos, dtype=np.float64), Chem.RemoveHs(tmpl_raw, _rm_hs_params())
    z_np = np.asarray(z, dtype=np.int64) if z is not None else np.array([a.GetAtomicNum() for a in (tmpl if len(pos_np) == tmpl.GetNumAtoms() else tmpl_raw).GetAtoms()], dtype=np.int64)
    heavy_idx = np.where(z_np > 1)[0]
    if len(heavy_idx) < len(z_np):
        mol_full = Chem.AddHs(tmpl)
        if [a.GetAtomicNum() for a in mol_full.GetAtoms()] == [int(x) for x in z_np]:
            _set_conformer(mol_full, pos_np); Chem.SanitizeMol(mol_full); Chem.AssignStereochemistryFrom3D(mol_full)
            return mol_full

    pos_nh, z_nh = pos_np[heavy_idx], z_np[heavy_idx]
    if [int(x) for x in z_nh] == [a.GetAtomicNum() for a in tmpl.GetAtoms()]:
        mol = _set_conformer(Chem.Mol(tmpl), pos_nh)
    else:
        rw = Chem.RWMol()
        for zi in z_nh:
            rw.AddAtom(Chem.Atom(int(zi)))
        mol = _set_conformer(rw.GetMol(), pos_nh)
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False)
        mol = AllChem.AssignBondOrdersFromTemplate(tmpl, mol)
        for a in mol.GetAtoms():
            a.SetNoImplicit(False); a.SetNumRadicalElectrons(0)
        Chem.SanitizeMol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return Chem.AddHs(mol, addCoords=True) if protonate else mol


def _tet_heights(mol: Any) -> dict[int, float]:
    from rdkit import Chem  # type: ignore[import-not-found]

    if mol.GetNumConformers() == 0:
        return {}
    m = Chem.Mol(mol); Chem.AssignStereochemistryFrom3D(m); conf = m.GetConformer()
    pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(m.GetNumAtoms())])
    out: dict[int, float] = {}
    for ch_idx, _ in Chem.FindMolChiralCenters(m, includeUnassigned=True):
        nbrs = [n.GetIdx() for n in m.GetAtomWithIdx(ch_idx).GetNeighbors()]
        if len(nbrs) >= 3:
            v1, v2, v3 = pts[nbrs[0]] - pts[ch_idx], pts[nbrs[1]] - pts[ch_idx], pts[nbrs[2]] - pts[ch_idx]
            nrm = np.cross(v1, v2)
            out[int(ch_idx)] = float(np.dot(v3, nrm) / max(float(np.linalg.norm(nrm)), 1e-8))
    return out


def validate_ligand_integrity(
    mol_before: Any, mol_after_or_pos: Any, *, max_bond_stretch: float = 0.45, planar_thresh_A: float = 0.12
) -> LigandValidationResult:
    """Verify ligand valence, connectivity, covalent bond lengths, InChI, and 3D chirality."""
    require_bio_dependency("rdkit", purpose="ligand stereo/InChI validation")
    from rdkit import Chem  # type: ignore[import-not-found]

    m0 = Chem.Mol(mol_before)
    if isinstance(mol_after_or_pos, (np.ndarray, list, tuple)):
        pos_new = np.asarray(mol_after_or_pos, dtype=np.float64)
        if pos_new.shape != (m0.GetNumAtoms(), 3):
            raise ValueError(f"Expected positions shape ({m0.GetNumAtoms()}, 3), got {pos_new.shape}")
        m1 = _set_conformer(Chem.Mol(m0), pos_new)
    else:
        m1 = Chem.Mol(mol_after_or_pos)
    try:
        Chem.SanitizeMol(m0); Chem.SanitizeMol(m1)
    except Exception as exc:
        return LigandValidationResult(False, False, False, False, False, False, math.inf, [], [], [], None, None, "", "", f"Sanitize failed: {exc}")

    connected = len(Chem.GetMolFrags(m0)) == len(Chem.GetMolFrags(m1))
    no_radicals = sum(a.GetNumRadicalElectrons() for a in m1.GetAtoms()) == 0
    ps = _rm_hs_params()
    for m in (m0, m1):
        if m.GetNumConformers() > 0:
            Chem.AssignStereochemistryFrom3D(m)
    m0_nh, m1_nh = Chem.RemoveHs(Chem.Mol(m0), ps), Chem.RemoveHs(Chem.Mol(m1), ps)
    smi0, smi1 = Chem.MolToSmiles(m0_nh, isomericSmiles=True), Chem.MolToSmiles(m1_nh, isomericSmiles=True)
    has_inchi = getattr(Chem, "INCHI_AVAILABLE", True) and hasattr(Chem, "MolToInchiKey")
    ik0, ik1 = (Chem.MolToInchiKey(m0_nh), Chem.MolToInchiKey(m1_nh)) if has_inchi else (None, None)
    inchi_match = ((ik0 == ik1) and (Chem.MolToInchi(m0_nh) == Chem.MolToInchi(m1_nh))) if has_inchi else (Chem.MolToSmiles(m0_nh, isomericSmiles=False) == Chem.MolToSmiles(m1_nh, isomericSmiles=False))

    broken, max_delta = [], 0.0
    if m0.GetNumConformers() > 0 and m1.GetNumConformers() > 0 and m0.GetNumAtoms() == m1.GetNumAtoms():
        c0, c1 = m0.GetConformer(), m1.GetConformer()
        for b in m0.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            delta = abs(float(c1.GetAtomPosition(i).Distance(c1.GetAtomPosition(j))) - float(c0.GetAtomPosition(i).Distance(c0.GetAtomPosition(j))))
            max_delta = max(max_delta, delta)
            if delta > max_bond_stretch:
                broken.append((i, j, delta))
    h0, h1 = _tet_heights(m0), _tet_heights(m1)
    planarized = [idx for idx in sorted(set(h0) & set(h1)) if min(abs(h0[idx]), abs(h1[idx])) < planar_thresh_A]
    inverted = [idx for idx in sorted(set(h0) & set(h1)) if idx not in planarized and h0[idx] * h1[idx] < 0.0]
    stereo_match = (len(inverted) == 0) and (bool(planarized) or smi0 == smi1)
    valid = bool(connected and no_radicals and inchi_match and stereo_match and not broken)
    return LigandValidationResult(valid, True, connected, no_radicals, inchi_match, stereo_match, max_delta, broken, inverted, planarized, ik0, ik1, smi0, smi1, "")


def _load_ligand_inputs(
    nonpoly: Sequence[_ResRec], *, ligand_sdf: str | Path | None, ligand_smiles: str | None, ligand_mol: Any,
    ligand_z: Sequence[int] | np.ndarray | None, ligand_pos: Sequence[Sequence[float]] | np.ndarray | None,
    ligand_chain: str | None, ligand_resname: str | None, ligand_charge: int | float | None, ligand_spin: int | None,
    protonate_lig: bool,
) -> tuple[np.ndarray, np.ndarray, int, int, Any | None]:
    if ligand_z is not None and ligand_pos is not None:
        z_arr, p_arr = np.asarray(ligand_z, dtype=np.int64), np.asarray(ligand_pos, dtype=np.float64)
        return z_arr, p_arr, *resolve_cofactor_charge_spin(z_arr, smiles=ligand_smiles, formal_charge=ligand_charge, spin=ligand_spin), None
    mol = ligand_mol
    if mol is None and ligand_sdf is not None:
        require_bio_dependency("rdkit", purpose="reading ligand .sdf")
        from rdkit import Chem  # type: ignore[import-not-found]

        mol = next((m for m in Chem.SDMolSupplier(str(ligand_sdf), removeHs=False) if m is not None), None)
        if mol is None:
            raise ValueError(f"Could not read molecule from {ligand_sdf}")
        if protonate_lig and not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=True)
    if mol is not None:
        require_bio_dependency("rdkit", purpose="extracting ligand coordinates")
        if ligand_smiles is not None and protonate_lig and not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            mz, mp, _, _ = _mol_arrays(mol)
            mol = reconstruct_ligand_mol(ligand_smiles, mp, z=mz, protonate=True)
        return (*_mol_arrays(mol, ligand_charge, ligand_spin), mol)

    matches = [r for r in nonpoly if (ligand_chain and r.chain == ligand_chain) or (ligand_resname and r.resname == ligand_resname.upper())]
    if not matches:
        raise ValueError("Specify ligand_sdf, ligand_mol, (ligand_z & ligand_pos), ligand_chain, or ligand_resname")
    r_lig = matches[0]
    z_raw, p_raw = np.array([a.z for a in r_lig.atoms], dtype=np.int64), np.array([a.pos for a in r_lig.atoms], dtype=np.float64)
    if ligand_smiles is not None:
        mol = reconstruct_ligand_mol(ligand_smiles, p_raw, z=z_raw, protonate=protonate_lig)
        return (*_mol_arrays(mol, ligand_charge, ligand_spin), mol)
    return z_raw, p_raw, *resolve_cofactor_charge_spin(z_raw, formal_charge=ligand_charge, spin=ligand_spin), None


def extract_pocket_complex(
    receptor_path: str | Path, *, ligand_sdf: str | Path | None = None, ligand_smiles: str | None = None,
    ligand_mol: Any | None = None, ligand_z: Sequence[int] | np.ndarray | None = None,
    ligand_pos: Sequence[Sequence[float]] | np.ndarray | None = None, ligand_chain: str | None = None,
    ligand_resname: str | None = None, ligand_charge: int | float | None = None, ligand_spin: int | None = None,
    pocket_radius: float = 6.0, cofactor_radius: float = 7.0, cap_mode: Literal["ace_nme", "hydrogen"] = "ace_nme",
    close_single_residue_gaps: bool = True, include_disulfide_partners: bool = True, protonate: bool = True, ph: float = 7.0,
    cofactor_smiles: Mapping[str, str] | None = None, cofactor_charges: Mapping[str, int | float] | None = None, cofactor_spins: Mapping[str, int] | None = None, use_gemmi: bool = True,
) -> PreparedPocketComplex:
    """Extract a capped pocket + ligand + cofactors/ions `PreparedPocketComplex` from PDB/mmCIF + SDF/SMILES."""
    poly_raw, nonpoly_raw = _load_structure(Path(receptor_path), use_gemmi=use_gemmi)
    lig_z, lig_pos, lig_q, lig_s, lig_mol_obj = _load_ligand_inputs(
        nonpoly_raw, ligand_sdf=ligand_sdf, ligand_smiles=ligand_smiles, ligand_mol=ligand_mol, ligand_z=ligand_z, ligand_pos=ligand_pos,
        ligand_chain=ligand_chain, ligand_resname=ligand_resname, ligand_charge=ligand_charge, ligand_spin=ligand_spin, protonate_lig=protonate,
    )
    lig_heavy = lig_pos[lig_z > 1] if np.any(lig_z > 1) else lig_pos
    chain_map: dict[str, str] = {}
    if protonate:
        pdb_str, chain_map = _residues_to_pdb(poly_raw)
        recs, sel_res, ss_partners, inter_bonds, h_min = _protonate_with_pdbfixer(pdb_str, lig_heavy, pocket_radius, ph)
    else:
        recs, h_min = poly_raw, 99.0
        sel_res = {ri for ri, rec in enumerate(recs) if any(a.z > 1 and np.min(np.linalg.norm(lig_heavy - a.pos[None, :], axis=-1)) <= pocket_radius for a in rec.atoms)}
        ss_partners, inter_bonds = _geometric_bonds(recs)
    if include_disulfide_partners:
        sel_res.update(ss_partners[ri] for ri in list(sel_res) if ri in ss_partners)
    if close_single_residue_gaps:
        sel_res.update(ri for ri in range(1, len(recs) - 1) if ri not in sel_res and (ri - 1) in sel_res and (ri + 1) in sel_res and recs[ri - 1].chain == recs[ri].chain == recs[ri + 1].chain)
    adj_bonds: dict[int, list[tuple[int, _AtomRec, _AtomRec]]] = {}
    for (r1, a1), (r2, a2) in inter_bonds:
        if (r1 in sel_res) != (r2 in sel_res):
            in_ri, in_a, out_ri, out_a = (r1, a1, r2, a2) if r1 in sel_res else (r2, a2, r1, a1)
            adj_bonds.setdefault(in_ri, []).append((out_ri, in_a, out_a))
    pkt_z, pkt_pos, pkt_bb, q_prot, cys_warnings, used_cap_res = [], [], [], 0, [], set()
    caps_added, has_h = {"ace": 0, "nme": 0, "hydrogen": 0}, (protonate or any(a.z == 1 for r in recs for a in r.atoms))
    def add_pkt(az: int, ap: np.ndarray, bb: bool) -> None:
        pkt_z.append(az); pkt_pos.append(ap.copy()); pkt_bb.append(bb)

    for ri in sorted(sel_res):
        rec = recs[ri]
        qr, warn = residue_formal_charge(rec.resname, set(rec.atom_map), has_ss_bond=(ri in ss_partners))
        q_prot += qr
        if warn:
            cys_warnings.append(f"{rec.chain}:{rec.resname}{rec.resseq}")
        for a in rec.atoms:
            add_pkt(a.z, a.pos, a.name in BACKBONE_ATOM_NAMES)
        for out_ri, in_a, out_a in adj_bonds.get(ri, []):
            out_rec = recs[out_ri]
            is_ace = cap_mode == "ace_nme" and in_a.name == "N" and out_a.name == "C" and all(k in out_rec.atom_map for k in ("CA", "C", "O"))
            is_nme = cap_mode == "ace_nme" and in_a.name == "C" and out_a.name == "N" and all(k in out_rec.atom_map for k in ("N", "CA"))
            if is_ace and out_ri not in used_cap_res:
                used_cap_res.add(out_ri)
                ca, c, o = out_rec.atom_map["CA"].pos, out_rec.atom_map["C"].pos, out_rec.atom_map["O"].pos
                for az, ap in ((6, ca), (6, c), (8, o), *([(1, hp) for hp in _staggered_methyl_hs(ca, c, o)] if has_h else [])):
                    add_pkt(az, ap, True)
                caps_added["ace"] += 1
            elif is_nme and out_ri not in used_cap_res:
                used_cap_res.add(out_ri)
                n_a, ca_a = out_rec.atom_map["N"].pos, out_rec.atom_map["CA"].pos
                add_pkt(7, n_a, True); add_pkt(6, ca_a, True)
                if has_h:
                    hn = out_rec.atom_map.get("H") or out_rec.atom_map.get("HN")
                    if hn is not None:
                        add_pkt(1, hn.pos, True)
                    else:
                        b1, b2 = in_a.pos - n_a, ca_a - n_a
                        bis = b1 / max(float(np.linalg.norm(b1)), 1e-8) + b2 / max(float(np.linalg.norm(b2)), 1e-8)
                        add_pkt(1, n_a - 1.01 * bis / max(float(np.linalg.norm(bis)), 1e-8), True)
                    for hp in _staggered_methyl_hs(ca_a, n_a, in_a.pos):
                        add_pkt(1, hp, True)
                caps_added["nme"] += 1
            else:
                v = out_a.pos - in_a.pos
                add_pkt(1, in_a.pos + (1.01 if in_a.z == 7 else 1.09) * v / max(float(np.linalg.norm(v)), 1e-6), True)
                caps_added["hydrogen"] += 1

    cof_z_list, cof_pos_list, cof_q_list, cof_s_list, cof_sizes = [], [], [], [], []
    for res in nonpoly_raw:
        if (ligand_chain and res.chain == ligand_chain) or (ligand_resname and res.resname == ligand_resname.upper()):
            continue
        r_pts = np.array([a.pos for a in res.atoms], dtype=np.float64)
        if np.mean(np.min(np.linalg.norm(r_pts[:, None, :] - lig_heavy[None, :, :], axis=-1), axis=1) < 0.5) > 0.5:
            continue
        r_nh = r_pts[[a.z > 1 for a in res.atoms]] if any(a.z > 1 for a in res.atoms) else r_pts
        if float(np.min(np.linalg.norm(r_nh[:, None, :] - lig_heavy[None, :, :], axis=-1))) > cofactor_radius:
            continue
        k_full = f"{res.chain}:{res.resname}"
        smi = (cofactor_smiles or {}).get(k_full) or (cofactor_smiles or {}).get(res.resname)
        q_ov = (cofactor_charges or {}).get(k_full) or (cofactor_charges or {}).get(res.resname)
        s_ov = (cofactor_spins or {}).get(k_full) or (cofactor_spins or {}).get(res.resname)
        rz = np.array([a.z for a in res.atoms], dtype=np.int64)
        if smi is not None and res.resname not in MONATOMIC_ION_DEFAULTS:
            try:
                rz, r_pts, _, _ = _mol_arrays(reconstruct_ligand_mol(smi, r_pts, z=rz, protonate=protonate))
            except ImportError:
                pass
        cq, cs = resolve_cofactor_charge_spin(rz, smiles=smi, resname=res.resname, formal_charge=q_ov, spin=s_ov)
        cof_z_list.append(rz); cof_pos_list.append(r_pts); cof_q_list.append(float(cq)); cof_s_list.append(int(cs)); cof_sizes.append(len(rz))

    pkt_z_np, pkt_pos_np = np.asarray(pkt_z, dtype=np.int64), (np.asarray(pkt_pos, dtype=np.float64) if pkt_pos else np.zeros((0, 3), dtype=np.float64))
    all_z = np.concatenate([pkt_z_np, lig_z, *cof_z_list])
    frag_id = np.concatenate([np.zeros(len(pkt_z_np), dtype=np.int64), np.ones(len(lig_z), dtype=np.int64), *[np.full(sz, 2 + k, dtype=np.int64) for k, sz in enumerate(cof_sizes)]])
    inter_ss = [f"{recs[r1].chain}:{recs[r1].resseq}-{recs[r2].chain}:{recs[r2].resseq}" for r1, r2 in ss_partners.items() if r1 < r2 and recs[r1].chain != recs[r2].chain]
    return PreparedPocketComplex(
        z=all_z, pos=np.concatenate([pkt_pos_np, lig_pos, *cof_pos_list], axis=0), frag_id=frag_id,
        frag_charge=[float(q_prot), float(lig_q), *cof_q_list], frag_spin=[closed_mult(pkt_z_np, q_prot), int(lig_s), *cof_s_list],
        pkt_is_bb=np.concatenate([np.asarray(pkt_bb, dtype=bool), np.zeros(len(all_z) - len(pkt_z_np), dtype=bool)]),
        n_pkt=len(pkt_z_np), n_lig=len(lig_z), cofactor_sizes=cof_sizes,
        metadata={"n_pocket_residues": len(sel_res), "caps_added": caps_added, "cys_warnings": cys_warnings, "inter_chain_disulfides": inter_ss, "min_hh_dist_A": h_min, "chain_map": chain_map, "ligand_mol": lig_mol_obj},
    )


def run_posebusters_check(mol_pred: Any, *, receptor_pdb: str | Path | None = None, mol_true: Any | None = None, config: str = "redock") -> Any:
    """Run guarded PoseBusters geometric/chemical audit (`posebusters` + `rdkit` required)."""
    require_bio_dependency("rdkit", purpose="PoseBusters input handling")
    pb_mod = require_bio_dependency("posebusters", purpose="pose quality audit")
    mode = "redock" if receptor_pdb is not None and config == "redock" else ("mol" if receptor_pdb is None else config)
    return pb_mod.PoseBusters(config=mode).bust([mol_pred], mol_true, str(receptor_pdb) if receptor_pdb is not None else None)
