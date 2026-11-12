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
from lam4.data.omol import OmolSource
from lam4.data.padding import pad_batch
from lam4.registry.sources import _OVERLAY
from lam4.registry.tokens import TOKEN_ID

R_CUT = 6.0


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


class OversizeStructures(Exception):
    """Every structure of the candidate prefix has more edges than e_pad by itself: the stream skips `n` positions and draws again."""

    def __init__(self, n: int) -> None:
        super().__init__(f"{n} structures above e_pad")
        self.n = n


def source_ids(name: str) -> tuple[int, int]:
    """(trunk token id, reference id) of a registry source: token per functional, ref_id = position in the registry overlay (one per dataset)."""
    return TOKEN_ID[_OVERLAY[name]["token"]], list(_OVERLAY).index(name)


@torch.no_grad()
def pack_molecules(src: OmolSource, cap: int, pos: np.ndarray, *, n_pad: int | None = None, e_pad: int | None = None, k: int | None = None, r_cut: float = R_CUT,
                   device: torch.device | str = "cpu", pieces: np.ndarray | None = None, source: str = "omol_wb97mv", dtype=torch.float32) -> Packed:
    """Pack the first structures of `pos` (positions in `src`) of bucket `cap` that fit. n_pad None: exact sizes, every structure, no padding (evaluation)."""
    device = torch.device(device)
    pos = np.asarray(pos, np.int64)
    if n_pad is not None:
        keep = int(np.searchsorted(np.cumsum(src.n_at[pos]), n_pad - 2, side="right"))
        if keep < 1:
            raise ValueError(f"structure of {src.n_at[pos[0]]} atoms does not fit n_pad {n_pad} (needs 2 pad atoms)")
        pos = pos[: min(keep, k or keep)]
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
    tok, ref = source_ids(source)
    fl = dict(device=device, dtype=dtype)
    n = int(nk.sum())
    sn = sl.cpu().numpy() if torch.is_tensor(sl) else sl
    f2 = torch.from_numpy(d["forces"][sn]).to(**fl)
    b = GraphBatch(
        z=z2[mask2], pos=p2[mask2], graph=torch.repeat_interleave(torch.arange(used, device=device), nk), cell=torch.zeros(used, 3, 3, **fl),
        edge_index=edges_from_mask(m, nk), edge_shift=torch.zeros(int(m.sum()), 3, **fl),
        charge=torch.from_numpy(d["charge"][sn]).to(**fl), spin=torch.from_numpy(d["spin"][sn]).to(**fl),
        u_flag=torch.zeros(used, dtype=torch.long, device=device), token=torch.full((used,), tok, dtype=torch.long, device=device),
        ref_id=torch.full((used,), ref, dtype=torch.long, device=device), domain="molecule", atom_mask=torch.ones(n, dtype=torch.bool, device=device),
        graph_mask=torch.ones(used, dtype=torch.bool, device=device), energy=torch.from_numpy(d["energy"][sn]).to(**fl), forces=f2[mask2],
        energy_mask=torch.ones(used, dtype=torch.bool, device=device), force_mask=torch.ones(n, dtype=torch.bool, device=device), coulomb_cap=cap, pbc=False)
    dip = torch.from_numpy(d["dipole"][sn]).to(**fl)
    dmask = torch.isfinite(dip).all(-1)
    dip = torch.where(dmask.unsqueeze(-1), dip, torch.zeros_like(dip))
    g = used
    if n_pad is not None and e_pad is not None:
        g = k + 1
        b = pad_batch(b, n_pad, e_pad, g, r_cut)
        dip = torch.cat([dip, dip.new_zeros(g - used, 3)])
        dmask = torch.cat([dmask, dmask.new_zeros(g - used)])
    if pieces is None:
        seg = torch.arange(g, device=device)
    else:   # chunk ids of the used structures, renumbered; unused slots and the pad graphs get ids of their own
        real = np.unique(pieces[sn] if torch.is_tensor(sl) else pieces[:used], return_inverse=True)[1].ravel()
        seg = torch.from_numpy(np.concatenate([real, real.max() + 1 + np.arange(g - used)]).astype(np.int64)).to(device)
    return Packed(b, {"dipole": dip, "dipole_mask": dmask, "seg": seg}, cap=cap, used=used, n_real=n, skipped=skipped)


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
            pk = pack_molecules(src, cap, rng.choice(rows, size=min(k, len(rows)), replace=False), n_pad=n_pad, e_pad=None, k=k, r_cut=r_cut)
            top = max(top, pk.batch.n_edges)
        plans[cap] = BucketPlan(cap, k, int(np.ceil(top * margin / mult) * mult))
    return plans


def exact_batches(src: OmolSource, max_atoms: int = 6000, r_cut: float = R_CUT, device: torch.device | str = "cpu", source: str = "omol_wb97mv") -> list[tuple[Packed, np.ndarray]]:
    """Evaluation batches of exact size (no padding), per bucket, up to ~max_atoms atoms each. Returns [(Packed, positions)]."""
    out = []
    for cap, rows in src.buckets.items():
        per = max(1, max_atoms // cap)
        for s in range(0, len(rows), per):
            pos = rows[s:s + per]
            out.append((pack_molecules(src, cap, pos, r_cut=r_cut, device=device, source=source), pos))
    return out
