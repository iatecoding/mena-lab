"""raw_readers.py: decoders for the LAM-2 extra crystal pools (formats checked on real samples).

All readers yield the same frame dict (eV, Angstrom, eV/A, ASE tensile-positive eV/A^3):
  {"z": int[n], "pos": [n,3], "forces": [n,3], "cell": [3,3], "pbc": bool[3], "energy": float, "stress": [6] | None,
   "frame_id": str, "group": str, "meta": dict}
`group` is the trajectory/system id (used for the val split and trajectory thinning).

OC20 S2EF (s2ef_train_2M.tar: <k>.extxyz.xz + <k>.txt.xz, 5000 frames per file):
  extxyz `energy=` is the TOTAL RPBE energy (E/atom ~ -4.5 eV; the txt `reference_energy` is the slab+gas reference used
  to make adsorption energies, so it is only kept in meta). Fixed atoms (move_mask F) carry real non-zero DFT forces, so
  every atom is trained. Columns: species pos(3) move_mask tags forces(3). No stress.
OC22 S2EF-Total (LMDB of pickled PyG Data, keys b"0".."N-1"): y = total PBE(+U) energy, force, pos, cell [1,3,3],
  atomic_numbers (float), fixed, tags, sid, fid. Unpickled without torch_geometric via a stub class. No stress.
OMC25 (ASE LMDB `dataNNNN.aselmdb`): value = zlib(JSON) of an ASE db row; ndarrays as {"__ndarray__": [shape, dtype,
  flat]}. energy/forces/stress are PBE-D3(0) (VASP IVDW=11), stress ASE Voigt eV/A^3. data.sid = <refcode>-<Z>-<step>-
  <xtal.id>-<frame index>; group = sid without the frame index.
"""
from __future__ import annotations

import io
import json
import lzma
import pickle
import re
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import atomic_numbers as _SYM2Z

Frame = dict[str, Any]


# ----------------------------------------------------------------------------------------------------------- OC20
def parse_oc20_extxyz(text: str, ref_txt: str | None = None, file_tag: str = "") -> Iterator[Frame]:
    lines = text.splitlines()
    refs = ref_txt.splitlines() if ref_txt else []
    i = k = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i])
        hdr = lines[i + 1]
        body = lines[i + 2:i + 2 + n]
        i += 2 + n
        if "Properties=species:S:1:pos:R:3:move_mask:L:1:tags:I:1:forces:R:3" not in hdr:
            raise ValueError(f"unexpected OC20 extxyz properties: {hdr[:200]}")
        lat = np.array(re.search(r'Lattice="([^"]+)"', hdr).group(1).split(), dtype=np.float64).reshape(3, 3)
        energy = float(re.search(r"(?<![a-z_])energy=(\S+)", hdr).group(1))
        pbc = [t == "T" for t in re.search(r'pbc="([^"]+)"', hdr).group(1).split()]
        cols = [ln.split() for ln in body]
        z = np.array([_SYM2Z[c[0]] for c in cols], dtype=np.int64)
        pos = np.array([c[1:4] for c in cols], dtype=np.float64)
        fixed = np.array([c[4] == "F" for c in cols])
        tags = np.array([int(c[5]) for c in cols], dtype=np.int64)
        forces = np.array([c[6:9] for c in cols], dtype=np.float64)
        sid = fid = ""
        ref_e = None
        if k < len(refs):
            sid, fid, ref = refs[k].split(",")
            ref_e = float(ref)
        yield {"z": z, "pos": pos, "forces": forces, "cell": lat, "pbc": np.array(pbc), "energy": energy,
               "stress": None, "frame_id": f"{sid}:{fid}" if sid else f"{file_tag}:{k}", "group": sid or file_tag,
               "meta": {"tags": tags, "fixed": fixed, "reference_energy": ref_e}}
        k += 1


def iter_oc20_xz_pair(xyz_xz: bytes, txt_xz: bytes | None, file_tag: str = "") -> Iterator[Frame]:
    text = lzma.decompress(xyz_xz).decode()
    txt = lzma.decompress(txt_xz).decode() if txt_xz else None
    yield from parse_oc20_extxyz(text, txt, file_tag)


# ----------------------------------------------------------------------------------------------------------- OC22
class _PygStub:
    def __init__(self, *a, **k) -> None:
        pass

    def __setstate__(self, state) -> None:
        self.__dict__["_state"] = state


class _PygUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("torch_geometric"):
            return _PygStub
        return super().find_class(module, name)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def decode_oc22_record(raw: bytes, key: str = "") -> Frame:
    obj = _PygUnpickler(io.BytesIO(raw)).load()
    st = obj.__dict__.get("_state", obj.__dict__) if isinstance(obj, _PygStub) else obj
    if isinstance(st, dict) and "_store" in st:  # PyG >= 2 layout
        st = st["_store"]
    z = _np(st["atomic_numbers"]).astype(np.int64)
    cell = _np(st["cell"]).reshape(3, 3).astype(np.float64)
    sid, fid = st.get("sid"), st.get("fid")
    sid = int(_np(sid)) if sid is not None else -1
    fid = int(_np(fid)) if fid is not None else -1
    return {"z": z, "pos": _np(st["pos"]).astype(np.float64), "forces": _np(st["force"]).astype(np.float64),
            "cell": cell, "pbc": np.array([True, True, True]), "energy": float(_np(st["y"])), "stress": None,
            "frame_id": f"{sid}:{fid}", "group": str(sid),
            "meta": {"tags": _np(st["tags"]).astype(np.int64) if st.get("tags") is not None else None,
                     "fixed": _np(st["fixed"]).astype(bool) if st.get("fixed") is not None else None,
                     "oc22_id": str(st.get("id", key))}}


def iter_lmdb(path: Path, decode, keys_filter=None) -> Iterator[Frame]:
    import lmdb

    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=True, meminit=False)
    with env.begin() as txn:
        for k, v in txn.cursor():
            ks = k.decode("ascii", "replace")
            if keys_filter is not None and not keys_filter(ks):
                continue
            fr = decode(v, ks)
            if fr is not None:
                yield fr
    env.close()


# ----------------------------------------------------------------------------------------------------------- OMC25
def _nd(v):
    if isinstance(v, dict) and "__ndarray__" in v:
        shape, dtype, flat = v["__ndarray__"]
        return np.asarray(flat, dtype=dtype).reshape(shape)
    return v


def decode_aselmdb_record(raw: bytes, key: str = "") -> Frame | None:
    """Returns None for the metadata keys (nextid, metadata, deleted_ids)."""
    if not key.isdigit():
        return None
    d = json.loads(zlib.decompress(raw).decode())
    if "numbers" not in d:
        return None
    data = d.get("data") or {}
    sid = str(data.get("sid", key))
    head, _, tail = sid.rpartition("-")
    group = head if tail.isdigit() else sid
    stress = d.get("stress")
    return {"z": _nd(d["numbers"]).astype(np.int64), "pos": _nd(d["positions"]).astype(np.float64),
            "forces": _nd(d["forces"]).astype(np.float64), "cell": _nd(d["cell"]).astype(np.float64).reshape(3, 3),
            "pbc": _nd(d["pbc"]).astype(bool), "energy": float(d["energy"]),
            "stress": None if stress is None else _nd(stress).astype(np.float64),
            "frame_id": sid, "group": group,
            "meta": {"frame_index": int(tail) if tail.isdigit() else -1, "csd_refcode": data.get("csd_refcode"),
                     "z_value": data.get("z_value"), "genarris_step": data.get("genarris_step")}}
