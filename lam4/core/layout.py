"""Stage layouts of the product-only edge op: group sizes, offsets and column counts of the two stage kinds.

FIRST (stage 0): the sender carries no vector or quadrupole.
    table columns   4 head biases | filters: scalar 128 | f_dir 32 | f_ten 16 | f3 8                       (4 + 184 = 188)
    gates [N, 184]  scalar 128 | gate_v 32 | gate_t 16 | gate_o 8                                          (sender node projections)
    outputs [N, 360]  s 128 | v 32x3 | t 16x5 | o 8x7
LATER (stages >= 1): the sender also carries transported l=1 b_j = W_tr * bound(vec_j) [N, 32, 3] and l=2 tau_j = W_tr2 * bound(ten_j) [N, 16, 5].
    table columns   4 head biases | filters: s 128 | dir 32 | ten 16 | tr 32 | lon 32 | tr2 16 | lon2 16 | o 8   (4 + 280 = 284)
    gates [N, 184]  as FIRST (tr, lon, tr2, lon2 have no gate: they act on b_j and tau_j)
    outputs [N, 408]  s 128 | v 32x3 | lon 32 | t 16x5 | lon2 16 | o 8x7

SIGN CONVENTION (differs from earlier Mena trunk): the edge vector is d = x_send - x_recv + shift and u = d / |d| (GraphBatch.edge_vectors), whereas the earlier Mena trunk's
geometry.py uses unit = x_recv - x_send, i.e. u_lam4 = -u_prev. Y2 is even and unchanged (so tr2 and lon2 = <tau, Y2(u)> are even too); Y1 and Y3 flip,
lon = <b, u> flips. Grafting earlier Mena weights into a lam4 stage therefore needs SIGN FLIPS on the f_dir and f3 table columns (W[:, f_dir cols] and W[:, f3 cols]
negated) and zeros on the new f_tr2 / f_lon2 columns. f_tr and f_lon stay (f_lon lon u is invariant); the lon OUTPUT scalar changes sign, so the first layer
that consumes it must absorb that sign (a column flip).

Head h (of 4) scales a contiguous quarter of the channels of every output group (and, in the filter order, the matching channels).
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

HEADS = 4


@dataclass(frozen=True)
class StageLayout:
    name: str
    k_s: int = 128    # scalar message channels
    k_v: int = 32     # l=1 channels (f_dir, gate_v)
    k_t: int = 16     # l=2 channels (f_ten, gate_t)
    k_o: int = 8      # l=3 channels (f3, gate_o)
    k_tr: int = 0     # l=1 transport channels (LATER: 32; filter f_tr, acts on b_j)
    k_lon: int = 0    # l=1 longitudinal channels (LATER: 32; filter f_lon, acts on <b_j, u_hat>)
    k_tr2: int = 0    # l=2 transport channels (LATER: 16; filter f_tr2, acts on tau_j)
    k_lon2: int = 0   # l=2 longitudinal channels (LATER: 16; filter f_lon2, acts on <tau_j, Y2(u_hat)>)

    @property
    def later(self) -> bool:
        return self.k_tr > 0

    @property
    def gate_seg(self) -> tuple[int, ...]:
        """Sender gate groups, columns of `gates` [N, gate_dim]: s, v, t, o."""
        return (self.k_s, self.k_v, self.k_t, self.k_o)

    @property
    def gate_dim(self) -> int:
        return sum(self.gate_seg)

    @property
    def filt_names(self) -> tuple[str, ...]:
        return ("s", "dir", "ten") + (("tr", "lon", "tr2", "lon2") if self.later else ()) + ("o",)

    @property
    def filt_seg(self) -> tuple[int, ...]:
        k = {"s": self.k_s, "dir": self.k_v, "ten": self.k_t, "tr": self.k_tr, "lon": self.k_lon,
             "tr2": self.k_tr2, "lon2": self.k_lon2, "o": self.k_o}
        return tuple(k[n] for n in self.filt_names)

    @property
    def filt_dim(self) -> int:
        return sum(self.filt_seg)

    @property
    def table_dim(self) -> int:
        """Columns of the radial table W [32, table_dim]: HEADS bias columns first, then the filters in `filt_names` order."""
        return HEADS + self.filt_dim

    @property
    def out_names(self) -> tuple[str, ...]:
        return ("s", "v") + (("lon",) if self.later else ()) + ("t",) + (("lon2",) if self.later else ()) + ("o",)

    @property
    def out_seg(self) -> tuple[int, ...]:
        """Widths of the output groups in `out_names` order (components included: v = 3 k_v, t = 5 k_t, o = 7 k_o)."""
        k = {"s": self.k_s, "v": 3 * self.k_v, "lon": self.k_lon, "t": 5 * self.k_t, "lon2": self.k_lon2, "o": 7 * self.k_o}
        return tuple(k[n] for n in self.out_names)

    @property
    def out_dim(self) -> int:
        return sum(self.out_seg)

    def filt_slices(self) -> dict[str, slice]:
        """Column ranges of each filter group INSIDE the filter block (add HEADS for table columns)."""
        out, o = {}, 0
        for n, k in zip(self.filt_names, self.filt_seg):
            out[n] = slice(o, o + k)
            o += k
        return out

    def out_slices(self) -> dict[str, slice]:
        out, o = {}, 0
        for n, k in zip(self.out_names, self.out_seg):
            out[n] = slice(o, o + k)
            o += k
        return out

    def gate_slices(self) -> dict[str, slice]:
        out, o = {}, 0
        for n, k in zip(("s", "v", "t", "o"), self.gate_seg):
            out[n] = slice(o, o + k)
            o += k
        return out

    @property
    def head_ok(self) -> bool:
        return all(k % HEADS == 0 for k in (self.k_s, self.k_v, self.k_t, self.k_o, self.k_tr, self.k_lon, self.k_tr2, self.k_lon2))


FIRST = StageLayout("first")
LATER = StageLayout("later", k_tr=32, k_lon=32, k_tr2=16, k_lon2=16)


def split_output(out: Tensor, layout: StageLayout) -> dict[str, Tensor]:
    """[N, out_dim] -> {"s": [N, 128], "v": [N, 32, 3], ("lon": [N, 32],) "t": [N, 16, 5], ("lon2": [N, 16],) "o": [N, 8, 7]}."""
    parts = dict(zip(layout.out_names, out.split(layout.out_seg, dim=-1)))
    n = out.shape[0]
    parts["v"] = parts["v"].reshape(n, layout.k_v, 3)
    parts["t"] = parts["t"].reshape(n, layout.k_t, 5)
    parts["o"] = parts["o"].reshape(n, layout.k_o, 7)
    return parts
