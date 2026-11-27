"""buckets.py: static-shape molecule batches (`GraphBatch` + loss extras) from an `OmolSource`; bucket plans (N_PAD, E_PAD, graph slots).

A training batch takes structures of ONE size bucket (cap in 24 .. 350), packs their real atoms contiguously, builds the open-molecule radius graph (r_cut 6.0,
`dense_open_mask`) and pads with `lam4.data.padding.pad_batch` to the bucket's fixed (n_pad, e_pad, k + 1 graphs): one traced / CUDA graph per bucket. A batch that would exceed
n_pad - 2 atoms or e_pad edges keeps the longest prefix of structures that fits (the sampler advances by what was used). `coulomb_cap` = the bucket cap (ExactCoulomb is [G, cap, cap]).
Extras (not in GraphBatch): dipole [G, 3] label, dipole_mask [G], seg [G] group-relative chunk ids (pad / unused slots get ids of their own).
Adapted from the earlier Mena training pipeline (pack_batch, calibrate_plans, candidates_per_batch).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from lam4.data.batch import GraphBatch
from lam4.data.neighbors import dense_open_mask, edges_from_mask
from lam4.data.omol import OMOL_SOURCE_CONFIGS, PAW_ID, OmolSource, XC_ID
from lam4.data.padding import pad_batch
from lam4.registry.sources import _OVERLAY
from lam4.registry.tokens import TOKEN_ID

R_CUT = 6.0
_FALLBACK_TOKEN_ID = {t: i for i, t in enumerate(("pbe", "r2scan", "rpbe", "wb97mv", "wb97xd", "wb97m_d3bj", "wb97x", "b3lyp_d3", "pbe0", "tpssh_d3bj", "sapt0"))}
_MOL_PAW_BY_OVERLAY = {"omol_wb97mv": "def2_tzvpd", "spice_wb97md3": "def2_tzvppd", "openrxn_wb97xd": "pop_631gd"}


@dataclass
class BucketPlan:
    cap: int
    k: int       # graph slots (the most structures a batch can hold)
    e_pad: int   # static edge count


@dataclass
class Packed:
    batch: GraphBatch
    extra: dict[str, torch.Tensor] = field(default_factory=dict)
    cap: int = 0
    used: int = 0       # real structures in the batch
    n_real: int = 0     # real atoms
    domain: str = "molecule"
    skipped: int = 0    # structures consumed from the stream but left out: more edges than the bucket's e_pad on their own (the stream advances by used + skipped)
    n_real_edges: int | None = None


class OversizeStructures(Exception):
    """Every structure of the candidate prefix has more edges than e_pad by itself: the stream skips `n` positions and draws again."""

    def __init__(self, n: int) -> None:
        super().__init__(f"{n} structures above e_pad")
        self.n = n


def source_ids(name: str) -> tuple[int, int]:
    """(trunk token id, reference id) of a registry source (supports LAM-6 extended sources cleanly)."""
    if name in _OVERLAY:
        return TOKEN_ID[_OVERLAY[name]["token"]], list(_OVERLAY).index(name)
    if name in OMOL_SOURCE_CONFIGS:
        xc = OMOL_SOURCE_CONFIGS[name]["xc"]
        return int(XC_ID.get(xc, _FALLBACK_TOKEN_ID.get(xc, 3))), len(_OVERLAY) + sorted(OMOL_SOURCE_CONFIGS).index(name)
    from lam4.registry.sources import LAM6_LOT_SPECS, _LOT_ALIASES

    key = _LOT_ALIASES.get(name, name)
    if key in LAM6_LOT_SPECS:
        xc = LAM6_LOT_SPECS[key].xc
        extra = sorted(set(LAM6_LOT_SPECS) - set(_OVERLAY) - set(OMOL_SOURCE_CONFIGS))
        return int(XC_ID.get(xc, _FALLBACK_TOKEN_ID.get(xc, 0))), len(_OVERLAY) + len(OMOL_SOURCE_CONFIGS) + extra.index(key)
    raise KeyError(f"unknown source {name!r}")


def _mol_lot_ids(name: str, src: OmolSource) -> tuple[int, int]:
    if name in OMOL_SOURCE_CONFIGS and getattr(src, "source", None) == name:
        return int(src.xc_id), int(src.paw_id)
    if name in OMOL_SOURCE_CONFIGS:
        c = OMOL_SOURCE_CONFIGS[name]
        return int(XC_ID[c["xc"]]), int(PAW_ID[c["paw"]])
    tok = _OVERLAY[name]["token"]
    paw = _MOL_PAW_BY_OVERLAY.get(name, "def2_tzvpd")
    return int(XC_ID.get(tok, 3)), int(PAW_ID.get(paw, 4))


def _masked_sidecar(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.from_numpy(arr).to(device=device, dtype=dtype)
    mask = torch.isfinite(t) if t.ndim == 1 else torch.isfinite(t).all(-1)
    safe = torch.where(mask if t.ndim == 1 else mask.unsqueeze(-1), t, torch.zeros_like(t))
    return safe, mask


@torch.no_grad()
def pack_molecules(src: OmolSource, cap: int, pos: np.ndarray, *, n_pad: int | None = None, e_pad: int | None = None, k: int | None = None, r_cut: float = R_CUT,
                   device: torch.device | str = "cpu", pieces: np.ndarray | None = None, source: str | None = None, dtype=torch.float32,
                   pair_ids: np.ndarray | None = None) -> Packed:
    """Pack the first structures of `pos` (positions in `src`) of bucket `cap` that fit. Defaults `source` to `src.source` so non-OMol LoTs never collide."""
    device = torch.device(device)
    source_name = getattr(src, "source", "omol_wb97mv") if source is None else source
    pos = np.asarray(pos, np.int64)
    if n_pad is not None:
        keep = int(np.searchsorted(np.cumsum(src.n_at[pos]), n_pad - 2, side="right"))
        if keep < 1:
            raise ValueError(f"structure of {src.n_at[pos[0]]} atoms does not fit n_pad {n_pad} (needs 2 pad atoms)")
        lim = min(keep, k or keep)
        pos = pos[:lim]
        if pieces is not None:
            pieces = pieces[:lim]
        if pair_ids is not None:
            pair_ids = pair_ids[:lim]
    d = src.gather(cap, pos)
    z2 = torch.from_numpy(d["z"]).to(device)
    p2 = torch.from_numpy(d["pos"]).to(device=device, dtype=dtype)
    nk = torch.from_numpy(d["n"]).to(device)
    mask2 = z2 > 0
    m = dense_open_mask(p2, mask2, r_cut)
    used, skipped, sl = len(pos), 0, slice(None)
    if e_pad is not None:   # the longest prefix whose edges fit e_pad; a structure that is above e_pad on its own is dropped (counted in `skipped`), not a crash 10 h into a run
        ec = m.sum((1, 2))
        big = ec > e_pad
        consumed = max(1, int((torch.cumsum(torch.where(big, torch.zeros_like(ec), ec), 0) <= e_pad).sum()))
        sl = (~big[:consumed]).nonzero().flatten()
        used, skipped = len(sl), consumed - len(sl)
        if used == 0:
            raise OversizeStructures(consumed)
    z2, p2, nk, m, mask2 = z2[sl], p2[sl], nk[sl], m[sl], mask2[sl]
    tok, ref = source_ids(source_name)
    xc_i, paw_i = _mol_lot_ids(source_name, src)
    fl = dict(device=device, dtype=dtype)
    n = int(nk.sum())
    sn = sl.cpu().numpy() if torch.is_tensor(sl) else sl
    f2 = torch.from_numpy(d["forces"][sn]).to(**fl)
    f_real = f2[mask2]
    if not getattr(src, "has_forces", True):
        f_mask = torch.zeros(n, dtype=torch.bool, device=device)
        f_real = torch.zeros_like(f_real)
    elif "force_mask" in d:
        fm2 = torch.from_numpy(d["force_mask"][sn]).to(device=device, dtype=torch.bool)
        f_mask = fm2[mask2] & torch.isfinite(f_real).all(dim=-1)
        f_real = torch.where(f_mask.unsqueeze(-1), f_real, torch.zeros_like(f_real))
    else:
        f_mask = torch.isfinite(f_real).all(dim=-1)
        f_real = torch.where(f_mask.unsqueeze(-1), f_real, torch.zeros_like(f_real))
    b = GraphBatch(
        z=z2[mask2], pos=p2[mask2], graph=torch.repeat_interleave(torch.arange(used, device=device), nk), cell=torch.zeros(used, 3, 3, **fl),
        edge_index=edges_from_mask(m, nk), edge_shift=torch.zeros(int(m.sum()), 3, **fl),
        charge=torch.from_numpy(d["charge"][sn]).to(**fl), spin=torch.from_numpy(d["spin"][sn]).to(**fl),
        u_flag=torch.zeros(used, dtype=torch.long, device=device), token=torch.full((used,), tok, dtype=torch.long, device=device),
        ref_id=torch.full((used,), ref, dtype=torch.long, device=device), domain="molecule", atom_mask=torch.ones(n, dtype=torch.bool, device=device),
        graph_mask=torch.ones(used, dtype=torch.bool, device=device), energy=torch.from_numpy(d["energy"][sn]).to(**fl), forces=f_real,
        energy_mask=torch.ones(used, dtype=torch.bool, device=device), force_mask=f_mask, coulomb_cap=cap, pbc=False,
        xc_id=torch.full((used,), xc_i, dtype=torch.long, device=device),
        paw_id=torch.full((used,), paw_i, dtype=torch.long, device=device),
        u_atom=torch.zeros(n, dtype=torch.long, device=device))
    dip, dmask = _masked_sidecar(d["dipole"][sn], device, dtype)
    sp_low_2 = torch.from_numpy(d["spins_lowdin"][sn]).to(**fl)[mask2]
    sp_mask = torch.isfinite(sp_low_2)
    sp_low = torch.where(sp_mask, sp_low_2, torch.zeros_like(sp_low_2))
    fid_545 = torch.from_numpy(d["frag_id_545"][sn]).to(device=device, dtype=torch.long)[mask2]
    unr = torch.from_numpy(d["unrestricted"][sn]).to(device=device, dtype=torch.uint8)
    s2_dev, s2_mask = _masked_sidecar(d["s_squared_dev"][sn], device, dtype)
    hgap, hgap_mask = _masked_sidecar(d["homo_gap"][sn], device, dtype)
    quad, quad_mask = _masked_sidecar(d["quadrupole"][sn], device, dtype)
    n_isl = torch.from_numpy(d["n_islands"][sn]).to(device=device, dtype=torch.long)
    fq_raw = torch.from_numpy(d["frag_q_lowdin"][sn]).to(**fl)
    isl_rng = torch.arange(8, device=device)[None, :] < n_isl[:, None]
    fq_mask = torch.isfinite(fq_raw) & isl_rng
    fq_low = torch.where(fq_mask, fq_raw, torch.zeros_like(fq_raw))
    g = used
    if n_pad is not None and e_pad is not None:
        g = k + 1
        pa, pg = n_pad - n, g - used
        b = pad_batch(b, n_pad, e_pad, g, r_cut)
        sp_low = torch.cat([sp_low, sp_low.new_zeros(pa)])
        sp_mask = torch.cat([sp_mask, sp_mask.new_zeros(pa)])
        fid_545 = torch.cat([fid_545, fid_545.new_zeros(pa)])
        dip = torch.cat([dip, dip.new_zeros(pg, 3)])
        dmask = torch.cat([dmask, dmask.new_zeros(pg)])
        unr = torch.cat([unr, unr.new_full((pg,), 255)])
        s2_dev = torch.cat([s2_dev, s2_dev.new_zeros(pg)])
        s2_mask = torch.cat([s2_mask, s2_mask.new_zeros(pg)])
        hgap = torch.cat([hgap, hgap.new_zeros(pg, 2)])
        hgap_mask = torch.cat([hgap_mask, hgap_mask.new_zeros(pg)])
        quad = torch.cat([quad, quad.new_zeros(pg, 6)])
        quad_mask = torch.cat([quad_mask, quad_mask.new_zeros(pg)])
        n_isl = torch.cat([n_isl, n_isl.new_zeros(pg)])
        fq_low = torch.cat([fq_low, fq_low.new_zeros(pg, 8)])
        fq_mask = torch.cat([fq_mask, fq_mask.new_zeros(pg, 8)])
    if pieces is None:
        seg = torch.arange(g, device=device)
    else:   # chunk ids of the used structures, renumbered; unused slots and the pad graphs get ids of their own
        real = np.unique(pieces[sn] if torch.is_tensor(sl) else pieces[:used], return_inverse=True)[1].ravel()
        seg = torch.from_numpy(np.concatenate([real, real.max() + 1 + np.arange(g - used)]).astype(np.int64)).to(device)
    from lam4.data.pairs import build_pair_tensors_mol
    kept_pids = (pair_ids[sn] if torch.is_tensor(sl) else pair_ids[:used]) if pair_ids is not None else None
    pair_t = build_pair_tensors_mol(kept_pids, d["z"][sn], d["n"][sn], d["charge"][sn], d["spin"][sn], len(b.z), g, device)
    extra = {
        "dipole": dip, "dipole_mask": dmask, "seg": seg,
        "xc_id": torch.full((g,), xc_i, dtype=torch.long, device=device),
        "paw_id": torch.full((g,), paw_i, dtype=torch.long, device=device),
        "u_atom": torch.zeros(len(b.z), dtype=torch.long, device=device),
        "spins_lowdin": sp_low, "spins_lowdin_mask": sp_mask, "frag_id_545": fid_545,
        "unrestricted": unr, "s_squared_dev": s2_dev, "s_squared_dev_mask": s2_mask,
        "homo_gap": hgap, "homo_gap_mask": hgap_mask, "quadrupole": quad, "quadrupole_mask": quad_mask,
        "n_islands": n_isl, "frag_q_lowdin": fq_low, "frag_q_mask": fq_mask,
        **pair_t,
    }
    return Packed(b, extra, cap=cap, used=used, n_real=n, skipped=skipped)


def candidates_per_batch(src: OmolSource, cap: int, n_pad: int) -> int:
    n = np.sort(src.n_at[src.buckets[cap]])
    lo = float(n[int(0.02 * (len(n) - 1))])
    return int(min(len(n), max(8, np.ceil((n_pad - 2) / max(lo, 1.0)) + 1)))


def calibrate_plans(src: OmolSource, n_pad: int, seed: int = 0, n_probe: int = 40, margin: float = 1.06, mult: int = 2048, r_cut: float = R_CUT) -> dict[int, BucketPlan]:
    """e_pad per bucket = the largest edge count of n_probe random packs (atoms-limited), times margin, rounded up to a multiple of `mult`."""
    rng = np.random.default_rng(seed + 11)
    plans = {}
    for cap, rows in src.buckets.items():
        k = candidates_per_batch(src, cap, n_pad)
        top = 0
        for _ in range(n_probe):
            pk = pack_molecules(src, cap, rng.choice(rows, size=min(k, len(rows)), replace=False), n_pad=n_pad, e_pad=None, k=k, r_cut=r_cut,
                                source=getattr(src, "source", None))
            top = max(top, pk.batch.n_edges)
        plans[cap] = BucketPlan(cap, k, int(np.ceil(top * margin / mult) * mult))
    return plans


def exact_batches(src: OmolSource, max_atoms: int = 6000, r_cut: float = R_CUT, device: torch.device | str = "cpu", source: str | None = None) -> list[tuple[Packed, np.ndarray]]:
    """Evaluation batches of exact size (no padding), per bucket, up to ~max_atoms atoms each. Returns [(Packed, positions)]."""
    out = []
    src_name = getattr(src, "source", "omol_wb97mv") if source is None else source
    for cap, rows in src.buckets.items():
        per = max(1, max_atoms // cap)
        for s in range(0, len(rows), per):
            pos = rows[s:s + per]
            out.append((pack_molecules(src, cap, pos, r_cut=r_cut, device=device, source=src_name), pos))
    return out
