"""MenaACE: dense- and sparse-graph potential with a Cartesian l <= 4 many-body angular basis.

Successor of MenaMinimal (models/mena_minimal.py) for from-scratch SPICE 2 training. Design record and the evidence
behind each change, in short:

- MenaMinimal's only angular invariants are the per-channel norms ||v_k||^2 (one l=1 channel family, body order 3);
  its equivariant path is 3.4% of the parameters and it underfits in distribution (train = val error). Here each
  stage builds, per atom, an l=1 density (vectors, K channels) and an l=2 density (traceless symmetric tensors in an
  orthonormal 5-component basis, K2 channels), and feeds three invariant families to the scalar update:
    nu2_v = <U v, V v>          cross-channel l=1 products (angles; PaiNN/ViSNet)
    nu2_t = <U T, V T>_F        l=2 products (P2(cos theta) angular resolution)
    nu3   = (P v)^T (W T) (Q v) vector-tensor-vector contractions (4-body, ACE correlation order 3)
- Dihedrals: MenaMinimal's <rej(W v_i), rej(W v_j)> was quadratic in a zero-initialised W, so its gradient was
  identically zero and it never trained. Here the identity
      sum_j c_ij <rej_ij(a_i), rej_ij(b_j)> = a_i . sum_j c_ij (I - r_ij r_ij^T) b_j
  turns it into per-atom quantities: sender vectors b_j = W v_j are transported with two learned radial filters, a
  full one and a longitudinal one ((r.b) r). Any mix of the two gives the transverse (dihedral) and bond components,
  and nu2_v then contracts the receiver's own vectors with them. No [B, N, N, D] tensor is formed.
- Scalar continuous-filter messages sum_j R(r_ij) * s'_j * w_ij are computed as a batched GEMM through the radial
  basis (factorised) when the bucket has more atoms than radial functions, instead of materialising [B, N, N, D].
- No learned "dispersion": ωB97M-D3(BJ) is exact in the SPICE labels, so it is subtracted from the training labels
  and added back at inference by models/model_factory.WithD3.

Every new bilinear branch has one randomly initialised factor and a zero-initialised output, so every parameter has
a non-zero gradient at initialisation (tests/test_mena_ace.py checks this for all parameters).

Mena-LAM unification additions:
- Supports both Stream A (`forward_sparse(pbc=True)` with `edge_diff` Direct Edge-Vector Virial and angular branches
  `ttt_mix`, `vttv_rank`, `l3_rank`, `l4_rank`) and Stream B (`forward` dense 17-bucket compiled path with
  `charge_mode="island"`, `IslandContext`, and full-range damped Coulomb) from the exact same parameter tensors.
- Multi-fidelity Level-of-Theory (`lot_id`) conditioning via zero-initialised projections into the initial atom
  embedding, each stage's SwiGLU pre-gate/value, and the final readout + per-LoT scalar bias.

Units: Angstrom, kcal/mol. Shapes: B molecules, N padded atoms, D scalar width, K vector channels, K2 tensor channels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lam.models import edge_ops as eo
from lam.models.edge_heads import EdgeHeadFactor
from lam.models.latent_mesh import MeshContext, MeshWindowPool, erf_coulomb_kernel, window_kernel
from lam.models.island_context import MultiWindowPool
from lam.models.island_context import IslandContext, IslandPool
from lam.models.mena_minimal_parts import ChemicalEdgeWeights, GaussianRBF, MinimalSwiGLU, TaperCutoff, taper_envelope
from lam.models.nlh_core import nlh_clamp_dense, nlh_clamp_sparse, nlh_energy_dense, nlh_energy_sparse
from lam.models.vector_swiglu import VectorSwiGLU
from lam.models.zbl_core import zbl_clamp_dense, zbl_clamp_sparse, zbl_energy_dense, zbl_energy_sparse

_S2 = 1.0 / math.sqrt(2.0)
_S6 = 1.0 / math.sqrt(6.0)

# Under torch.compile, graphs with N <= SMALL_N atoms (dense forward) write the sums over senders j as broadcast
# multiply + sum, which inductor fuses into one kernel without materialising the product; for tiny N the library
# bmm/einsum calls (inner dim N, 3-5 output columns) are slow gemv kernels plus transpose/reshape copies (A100 profile,
# measured on OMol25). Eager, multiply + sum materialises [B,N,N,K,3] and is slower
# (MPS: x1.35 at N = 12, x1.85 at N = 64), so ACEStage.small_n defaults to 0 (bmm) and the trainer sets SMALL_N when
# compiling. Same math; float summation order differs.
SMALL_N = 64


def l2_pair(p: Tensor, q: Tensor) -> Tensor:
    """Components p^T E_m q of sym(p q^T) in an orthonormal (Frobenius) basis E_m of 3x3 traceless symmetric matrices.

    p, q: [..., 3] -> [..., 5]. l2_pair(r, r) is the l=2 harmonic of a unit vector r (norm^2 = 2/3), and for a tensor
    T = sum_m t_m E_m, p^T T q = sum_m t_m * l2_pair(p, q)_m.
    """
    px, py, pz = p.unbind(-1)
    qx, qy, qz = q.unbind(-1)
    return torch.stack((
        (px * qy + py * qx) * _S2,
        (px * qz + pz * qx) * _S2,
        (py * qz + pz * qy) * _S2,
        (px * qx - py * qy) * _S2,
        (2.0 * pz * qz - px * qx - py * qy) * _S6,
    ), dim=-1)


_E9: dict[tuple[torch.dtype, torch.device], Tensor] = {}


def l2_pair_mm(p: Tensor, q: Tensor) -> Tensor:
    """l2_pair as one outer product and one matmul: p^T E_m q = sum_ab (p_a q_b) E_m[a, b], E_m from _l2_matrices.
    The same linear map as l2_pair (equal up to float rounding; tests/test_mena_ace_fast_ops.py); 3 ops instead of ~24,
    and its autograd graph (and double backward) is correspondingly smaller. Used under fast_ops "l2"."""
    key = (p.dtype, p.device)
    e = _E9.get(key)
    if e is None:
        e = _l2_matrices().reshape(5, 9).t().contiguous().to(device=p.device, dtype=p.dtype)
        if not torch.compiler.is_compiling() and not isinstance(e, torch._subclasses.FakeTensor):
            _E9[key] = e
    outer = (p.unsqueeze(-1) * q.unsqueeze(-2)).reshape(*p.shape[:-1], 9)
    return outer @ e


FAST_OPS = ("l2", "mix", "proj", "split", "gather", "edge", "radial")  # MenaACE.set_fast_ops names (execution-only options; weights
# and maths unchanged); "gather", "edge", "radial" (and "proj") act on forward_sparse (see ACEStage.forward_sparse)


def _mix(x: Tensor, lin: nn.Linear) -> Tensor:
    """Channel mixing of an equivariant feature [..., C, M] (M = 3 or 5 components) with a bias-free Linear.
    The input is made contiguous: a linear on the strided transposed view runs as N tiny matmuls (20-80x slower on MPS / CPU)."""
    return lin(x.transpose(-1, -2).contiguous()).transpose(-1, -2)


def _pair_inv(x: Tensor, u: nn.Linear | None, v: nn.Linear | None) -> Tensor:
    """<U x, V x> over the component axis; per-channel norms |x_k|^2 when u is None (cross_channel=False ablation)."""
    if u is None:
        return (x * x).sum(-1)
    return (_mix(x, u) * _mix(x, v)).sum(-1)


def latent_dipole(q: Tensor, pos: Tensor, mask: Tensor) -> Tensor:
    """Dipole mu = sum_i q_i r_i [B, 3] (e*A) of the latent Coulomb charges q [B, N] about the coordinate origin of pos
    (for a charged system the dipole depends on the origin; the OMol-1 dipole.npy targets use the pos.npy origin).
    Padding atoms (mask False) are excluded here as well as by their zero charge. 1 e*A = 4.803 D."""
    return (q.unsqueeze(-1) * pos * mask.unsqueeze(-1).to(pos.dtype)).sum(-2)


def _l2_matrices() -> Tensor:
    """E [5, 3, 3] with p^T E_m q = l2_pair(p, q)_m (symmetric traceless, orthonormal basis of l2_pair), so that
    M(t) = sum_m t_m E_m acts as t . l2_pair(p, q) = p^T M(t) q."""
    eye = torch.eye(3)
    return l2_pair(eye.unsqueeze(1).expand(3, 3, 3), eye.unsqueeze(0).expand(3, 3, 3)).permute(2, 0, 1).contiguous()


class GrowthBranch(nn.Module):
    """Additive modules inserted into a trained ACEStage (MenaACE grow_tensor_rank / edge_quadratic /
    tensor_vector_update). Everything here is new parameters; the stage's own weights are untouched. The outputs enter
    through zero-initialised maps (`inv_extra` into the pre-activation of inv_mlp[0]; `tv_gate` into the vector update),
    so a stage with the branch computes exactly what it computed before until those maps train."""

    def __init__(self, d_model: int, k: int, k2: int, num_rbf: int, first: bool, edge_quadratic: bool,
                 tensor_vector_update: bool) -> None:
        super().__init__()
        self.k, self.k2 = k, k2
        if k2:  # additive l=2 branch: same maths as the tensor_rank code of ACEStage, own parameters
            self.ten_gate = nn.Linear(d_model, k2)
            self.ten_radial = nn.Linear(num_rbf, k2, bias=False)
            self.u2, self.v2, self.t3 = (nn.Linear(k2, k2, bias=False) for _ in range(3))
            self.p3, self.q3 = nn.Linear(k, k2, bias=False), nn.Linear(k, k2, bias=False)
        self.edge_quad = nn.Linear(num_rbf, k, bias=False) if edge_quadratic and not first else None
        n_extra = 2 * k2 + (k if self.edge_quad is not None else 0)
        self.inv_extra = nn.Linear(n_extra, d_model, bias=False) if n_extra else None
        if self.inv_extra is not None:
            nn.init.zeros_(self.inv_extra.weight)
        self.tv_w = self.tv_gate = None
        if tensor_vector_update:  # l=2 x l=1 -> l=1: v_k += gamma_k(s) (sum_c W_kc M(T_c)) v_k  (needs k2)
            self.tv_w = nn.Linear(k2, k, bias=False)
            self.tv_gate = nn.Linear(d_model, k)
            nn.init.zeros_(self.tv_gate.weight)
            nn.init.zeros_(self.tv_gate.bias)
            self.register_buffer("l2_mats", _l2_matrices(), persistent=False)

    def tensor_features(self, T: Tensor, v_new: Tensor) -> list[Tensor]:
        nu2_t = (_mix(T, self.u2) * _mix(T, self.v2)).sum(-1)
        nu3 = (_mix(T, self.t3) * l2_pair(_mix(v_new, self.p3), _mix(v_new, self.q3))).sum(-1)
        return [nu2_t, nu3]

    def vector_update(self, T: Tensor, v_new: Tensor, s: Tensor) -> Tensor:
        """gamma_k(s) (A_k v_new_k), A_k = sum_c W_kc M(T_c); T [.., K2, 5], v_new [.., K, 3], s [.., D] -> [.., K, 3]."""
        a = _mix(T, self.tv_w)  # [.., K, 5]
        ev = torch.einsum("mab,...kb->...kma", self.l2_mats.to(v_new.dtype), v_new)  # [.., K, 5, 3]
        return self.tv_gate(s).unsqueeze(-1) * (a.unsqueeze(-1) * ev).sum(-2)


def l2_mat6(t: Tensor) -> tuple[Tensor, ...]:
    """The six independent entries (xx, yy, zz, xy, xz, yz) of M(t) = sum_m t_m E_m (E_m of _l2_matrices) as elementwise
    expressions of t [..., 5]: no 3x3 matmul, no reduction over a small axis (MPS-friendly, any order of autograd)."""
    t0, t1, t2, t3, t4 = t.unbind(-1)
    return t3 * _S2 - t4 * _S6, -t3 * _S2 - t4 * _S6, 2.0 * _S6 * t4, t0 * _S2, t1 * _S2, t2 * _S2


def l2_matvec(m6: tuple[Tensor, ...], p: Tensor) -> Tensor:
    """M p for M given by its l2_mat6 entries [...] and p [..., 3] -> [..., 3]."""
    xx, yy, zz, xy, xz, yz = m6
    px, py, pz = p.unbind(-1)
    return torch.stack((xx * px + xy * py + xz * pz, xy * px + yy * py + yz * pz, xz * px + yz * py + zz * pz), dim=-1)


def l2_det(m6: tuple[Tensor, ...]) -> Tensor:
    """det M of the symmetric 3x3 matrix given by its l2_mat6 entries. For traceless M (Cayley-Hamilton) tr(M^3) = 3 det M."""
    xx, yy, zz, xy, xz, yz = m6
    return xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)


# l = 3: rank-3 symmetric traceless (STF) tensors, Cartesian. The real l = 3 harmonic polynomials as
# {(i, j, k): coefficient of x^i y^j z^k}; each is B . (r r r) for a unique STF tensor B (harmonic <=> traceless).
_L3_POLYS = (
    {(2, 1, 0): 3.0, (0, 3, 0): -1.0},                                  # y (3x^2 - y^2)
    {(1, 1, 1): 1.0},                                                   # xyz
    {(0, 1, 2): 4.0, (2, 1, 0): -1.0, (0, 3, 0): -1.0},                 # y (4z^2 - x^2 - y^2)
    {(0, 0, 3): 2.0, (2, 0, 1): -3.0, (0, 2, 1): -3.0},                 # z (2z^2 - 3x^2 - 3y^2)
    {(1, 0, 2): 4.0, (3, 0, 0): -1.0, (1, 2, 0): -1.0},                 # x (4z^2 - x^2 - y^2)
    {(2, 0, 1): 1.0, (0, 2, 1): -1.0},                                  # z (x^2 - y^2)
    {(3, 0, 0): 1.0, (1, 2, 0): -3.0},                                  # x (x^2 - 3y^2)
)
# the 10 cubic monomials (exponents of x, y, z), in the order l3_monomials stacks them
_L3_MONOS = ((3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1))


def _multinomial3(e: tuple[int, int, int]) -> int:
    return math.factorial(3) // (math.factorial(e[0]) * math.factorial(e[1]) * math.factorial(e[2]))


def _l3_tensors() -> Tensor:
    """B [7, 3, 3, 3] (float64): an orthonormal basis (full contraction) of the 7-dimensional space of rank-3 symmetric traceless
    tensors, from the real l = 3 harmonics (tests/test_angular_branches.py checks symmetry, tracelessness, orthonormality).
    y3(r)_m = B_m . (r r r) are then the coordinates of the STF part of r r r (|y3(r_hat)|^2 = 2/5)."""
    import itertools
    out = torch.zeros(7, 3, 3, 3, dtype=torch.float64)
    for m, poly in enumerate(_L3_POLYS):
        for idx in itertools.product(range(3), repeat=3):
            e = tuple(sum(int(a == ax) for a in idx) for ax in range(3))
            if e in poly:  # a monomial's coefficient is shared equally by its 3!/(i!j!k!) index orders
                out[(m,) + idx] = poly[e] / _multinomial3(e)
        out[m] /= out[m].norm()
    return out


def _l3_monomial_matrix() -> Tensor:
    """C [10, 7] with y3(r)_m = sum_n l3_monomials(r)_n C[n, m] = B_m . (r r r)."""
    B = _l3_tensors()
    C = torch.zeros(10, 7, dtype=B.dtype)
    for n, e in enumerate(_L3_MONOS):
        C[n] = B[(slice(None),) + (0,) * e[0] + (1,) * e[1] + (2,) * e[2]] * _multinomial3(e)
    return C


def l3_monomials(u: Tensor) -> Tensor:
    """The 10 cubic monomials of u [..., 3] -> [..., 10] (elementwise products and one stack, no small-axis reduction)."""
    x, y, z = u.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    return torch.stack((xx * x, yy * y, zz * z, xx * y, xx * z, yy * x, yy * z, zz * x, zz * y, x * y * z), dim=-1)


# l = 4: rank-4 symmetric traceless (STF) tensors, Cartesian (9 components; first non-zero multipole at O_h and T_d cubic sites).
_L4_POLYS = (
    {(3, 1, 0): 1.0, (1, 3, 0): -1.0},                                                  # xy(x^2 - y^2)
    {(2, 1, 1): 3.0, (0, 3, 1): -1.0},                                                  # yz(3x^2 - y^2)
    {(1, 1, 2): 6.0, (3, 1, 0): -1.0, (1, 3, 0): -1.0},                                 # xy(6z^2 - x^2 - y^2)
    {(0, 1, 3): 4.0, (2, 1, 1): -3.0, (0, 3, 1): -3.0},                                 # yz(4z^2 - 3x^2 - 3y^2)
    {(0, 0, 4): 8.0, (2, 0, 2): -24.0, (0, 2, 2): -24.0, (4, 0, 0): 3.0, (2, 2, 0): 6.0, (0, 4, 0): 3.0},
    {(1, 0, 3): 4.0, (3, 0, 1): -3.0, (1, 2, 1): -3.0},                                 # xz(4z^2 - 3x^2 - 3y^2)
    {(2, 0, 2): 6.0, (0, 2, 2): -6.0, (4, 0, 0): -1.0, (0, 4, 0): 1.0},                 # (x^2 - y^2)(6z^2 - x^2 - y^2)
    {(3, 0, 1): 1.0, (1, 2, 1): -3.0},                                                  # xz(x^2 - 3y^2)
    {(4, 0, 0): 1.0, (2, 2, 0): -6.0, (0, 4, 0): 1.0},                                  # x^4 - 6x^2y^2 + y^4
)
_L4_MONOS = (
    (4, 0, 0), (0, 4, 0), (0, 0, 4),
    (3, 1, 0), (3, 0, 1), (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3),
    (2, 2, 0), (2, 0, 2), (0, 2, 2),
    (2, 1, 1), (1, 2, 1), (1, 1, 2),
)


def _multinomial4(e: tuple[int, int, int]) -> int:
    return 24 // (math.factorial(e[0]) * math.factorial(e[1]) * math.factorial(e[2]))


def _l4_tensors() -> Tensor:
    """B4 [9, 3, 3, 3, 3] (float64): orthonormal basis of rank-4 symmetric traceless tensors (|y4(r_hat)|^2 = 8/35)."""
    import itertools
    out = torch.zeros(9, 3, 3, 3, 3, dtype=torch.float64)
    for m, poly in enumerate(_L4_POLYS):
        for idx in itertools.product(range(3), repeat=4):
            e = tuple(sum(int(a == ax) for a in idx) for ax in range(3))
            if e in poly:
                out[(m,) + idx] = poly[e] / _multinomial4(e)
        out[m] /= out[m].norm()
    return out


def _l4_monomial_matrix() -> Tensor:
    """C4 [15, 9] with y4(r)_m = sum_n l4_monomials(r)_n C4[n, m] = B4_m . (r r r r)."""
    B = _l4_tensors()
    C = torch.zeros(15, 9, dtype=B.dtype)
    for n, e in enumerate(_L4_MONOS):
        C[n] = B[(slice(None),) + (0,) * e[0] + (1,) * e[1] + (2,) * e[2]] * _multinomial4(e)
    return C


def l4_monomials(u: Tensor) -> Tensor:
    """The 15 quartic monomials of u [..., 3] -> [..., 15]."""
    x, y, z = u.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack((
        xx * xx, yy * yy, zz * zz,
        xx * xy, xx * xz, yy * xy, yy * yz, zz * xz, zz * yz,
        xx * yy, xx * zz, yy * zz,
        xx * yz, yy * xz, zz * xy,
    ), dim=-1)


_CONST: dict[tuple[str, torch.dtype, torch.device], Tensor] = {}
_CONST_FN = {
    "l3_mono": _l3_monomial_matrix,  # [10, 7]
    "l3_b27": lambda: _l3_tensors().reshape(7, 27).t().contiguous(),  # [27, 7]
    "l4_mono": _l4_monomial_matrix,  # [15, 9]
    "l4_b81": lambda: _l4_tensors().reshape(9, 81).t().contiguous(),  # [81, 9]
    # [5, 9]: E_m row-major, built in float64 (l2_pair with float64 unit vectors)
    "l2_e9": lambda: l2_pair(torch.eye(3, dtype=torch.float64).unsqueeze(1).expand(3, 3, 3),
                             torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(3, 3, 3)).permute(2, 0, 1).reshape(5, 9),
}


def _const(name: str, like: Tensor) -> Tensor:
    """Constant basis matrices built in float64 and cast once per (dtype, device), so a float64 model sees them exact
    (a float32 buffer upcast by .double() would carry float32 rounding, e.g. 1.5e-8 traces of the l=3 basis)."""
    key = (name, like.dtype, like.device)
    c = _CONST.get(key)
    if c is None:
        c = _CONST_FN[name]().to(device=like.device, dtype=like.dtype)
        if not torch.compiler.is_compiling() and not isinstance(c, torch._subclasses.FakeTensor):
            _CONST[key] = c
    return c


def baked_current(baked_versions, current_versions) -> bool:
    """True when baked inference weights still match their source tensors. Under torch.compile the `_version`
    counters are data-dependent ints that split the graph; compiled inference freezes the weights first
    (compile_for_inference / MenaLAMCalculator), so the bake is trusted while tracing."""
    return torch.compiler.is_compiling() or baked_versions == current_versions


def graph_sum(like: Tensor, B: int, idx: Tensor, val: Tensor) -> Tensor:
    """Per-graph sum `like.new_zeros(B).index_add_(0, idx, val)`. A single graph is a plain sum: inductor (torch 2.13)
    miscompiles a scatter into a size-1 buffer (codegen assertion on CPU) when the whole core is compiled. Only while
    compiling: eager keeps index_add_ so its output stays bit-identical to stored fingerprints (summation order)."""
    if B == 1 and torch.compiler.is_compiling():
        return val.sum(0, keepdim=True)
    return like.new_zeros(B).index_add_(0, idx, val)


def l3_harmonic(u: Tensor) -> Tensor:
    """y3 [..., 7] = B_m . (u u u) for u [..., 3] (the l = 3 analogue of l2_pair(u, u)): 10 monomials and one matmul."""
    return l3_monomials(u) @ _const("l3_mono", u)


def l4_harmonic(u: Tensor) -> Tensor:
    """y4 [..., 9] = B4_m . (u u u u) for u [..., 3]: 15 quartic monomials and one matmul."""
    return l4_monomials(u) @ _const("l4_mono", u)


class L4Branch(nn.Module):
    """Cartesian l=4 hexadecapole branch (Breakthrough 2.2: non-zero at O_h rocksalt/perovskite and T_d diamond/zincblende
    sites where l=1,2 and O_h l=3 vanish identically). Separate module (`stages_list.*.l4.*`) with zero-init `out`, so it
    grafts onto checkpoints that already have `ang` (B1/B2) trained."""

    def __init__(self, d_model: int, k2: int, num_rbf: int, l4_rank: int) -> None:
        super().__init__()
        self.k4 = int(l4_rank)
        if self.k4 and not k2:
            raise ValueError("l4_rank needs tensor_rank > 0 (for the l=2 x l=2 -> l=4 coupling)")
        lin = lambda i, o: nn.Linear(i, o, bias=False)  # noqa: E731
        self.radial4 = lin(num_rbf, self.k4)
        self.gate4 = nn.Linear(d_model, self.k4)
        self.u4, self.v4, self.w4 = lin(self.k4, self.k4), lin(self.k4, self.k4), lin(self.k4, self.k4)
        self.a4, self.b4 = lin(k2, self.k4), lin(k2, self.k4)
        self.n_out = 2 * self.k4
        self.out = lin(self.n_out, d_model)
        nn.init.zeros_(self.out.weight)
        self.record: dict | None = None

    def families(self) -> list[tuple[str, int, int]]:
        return [("nu2_4", 0, self.k4), ("tt4", self.k4, 2 * self.k4)]

    @torch.no_grad()
    def rescale(self, family: str, ratio: float) -> None:
        groups = {"nu2_4": ([self.radial4], 2), "tt4": ([self.w4, self.a4, self.b4], 3)}
        mods, deg = groups[family]
        for m in mods:
            m.weight.mul_(ratio ** (1.0 / deg))

    def _features_from_T4(self, T4: Tensor, T: Tensor) -> Tensor:
        nu2_4 = (_mix(T4, self.u4) * _mix(T4, self.v4)).sum(-1)
        yt = F.linear(T.transpose(-1, -2).contiguous(), torch.cat([self.a4.weight, self.b4.weight], 0)).transpose(-1, -2)  # [..., 2*k4, 5]
        m9 = yt @ _const("l2_e9", T)  # [..., 2*k4, 9]
        ma, mb = m9[..., :self.k4, :], m9[..., self.k4:, :]
        mm81 = (ma.unsqueeze(-1) * mb.unsqueeze(-2)).flatten(-2)  # [..., k4, 81]
        tt4 = (_mix(T4, self.w4) * (mm81 @ _const("l4_b81", mm81))).sum(-1)
        out = torch.cat([nu2_4, tt4], dim=-1)
        if self.record is not None:
            self.record.update({n: float(out[..., a:b].detach().pow(2).mean().sqrt()) for n, a, b in self.families()})
        return out

    def forward(self, edge: Tensor, hs: Tensor, T: Tensor, y4: Tensor, norm: Tensor, small: bool) -> Tensor:
        B, N, _ = hs.shape
        w4 = self.radial4(edge) * self.gate4(hs).unsqueeze(1)
        if small:
            T4 = (w4.unsqueeze(-1) * y4.unsqueeze(3)).sum(dim=2) * norm.unsqueeze(-1)
        else:
            T4 = torch.bmm(w4.reshape(B * N, N, self.k4).transpose(1, 2), y4.reshape(B * N, N, 9)).view(B, N, self.k4, 9) * norm.unsqueeze(-1)
        return self._features_from_T4(T4, T)

    def forward_sparse(self, edge: Tensor, hs: Tensor, T: Tensor, y4: Tensor, recv: Tensor, gat, N: int, norm: Tensor, fe: bool) -> Tensor:
        w4 = self.radial4(edge) * gat(self.gate4(hs))
        prod = eo.outer(w4, y4) if fe else w4.unsqueeze(-1) * y4.unsqueeze(1)
        T4 = w4.new_zeros(N, self.k4, 9).index_add_(0, recv, prod) * norm.unsqueeze(-1)
        return self._features_from_T4(T4, T)


class AngularBranch(nn.Module):
    """Angular-resolution branches of one ACEStage .

    All parameters are new (`stages_list.*.ang.*`); the invariants enter through ONE zero-initialised bias-free Linear
    `out` added to the pre-activation of inv_mlp[0] (the GrowthBranch pattern), so a model with the branch, loaded with the
    weights of the same model without it, computes exactly the same energies, forces and stresses. Every other new matrix
    is randomly initialised (one random factor per product), so no invariant is a product of zero factors: `out` gets
    gradient at init, and every upstream parameter as soon as `out` is non-zero (one optimiser step).

    B1 (ttt_mix = k, vttv_rank = r; no new per-edge work), from the stage's l=2 density T [..., K2, 5] and v = v_new [..., K, 3]:
        nu3_ttt_c  = tr(M(C_c T)^3) = 3 det M(C_c T)                c = 1..k   (pure l=2 cubic: correlation order 3, 4-body)
        nu4_vttv_c = (P_c v)^T M(A_c T) M(B_c T) (Q_c v)            c = 1..r   (correlation order 4)
    M(t) the 3x3 symmetric traceless matrix of the l=2 components t; C, A, B: K2 -> k or r and P, Q: K -> r channel mixes.

    B2 (l3_rank = k3): an l=3 density per atom T3_ic = norm_i sum_j f3_c(r_ij) g3_c(s_j) y3(r_hat_ij), y3 the 7 coordinates
    of r_hat r_hat r_hat in an orthonormal basis of rank-3 STF tensors (Cartesian, _l3_tensors; no Clebsch-Gordan), f3 one
    extra radial filter block (num_rbf -> k3, fused into the stage's radial GEMM under fast "radial"), g3 a sender gate
    (d_model -> k3). Invariants
        nu2_3_c   = <U3 T3, V3 T3>_c                                 (P3(cos theta) resolution)
        nu3_vt3_c = sum_abc (W3 T3)_c,abc (P3 v)_c,a M(A3 T)_c,bc    (l1 x l2 -> l3, contracted with T3)
    Per edge: +k3 radial columns, a gated [E, k3] weight and the [E, k3, 7] outer product (edge_ops.outer under "edge").
    """

    def __init__(self, d_model: int, k: int, k2: int, num_rbf: int, ttt_mix: int, vttv_rank: int, l3_rank: int) -> None:
        super().__init__()
        self.kt, self.kr, self.k3 = int(ttt_mix), int(vttv_rank), int(l3_rank)
        if (self.kt or self.kr) and not k2:
            raise ValueError("ttt_mix / vttv_rank need tensor_rank > 0 (they contract the stage's l=2 density)")
        if self.kr and not k:
            raise ValueError("vttv_rank needs vector_rank > 0")
        if self.k3 and (not k2 or not k):
            raise ValueError("l3_rank needs vector_rank > 0 and tensor_rank > 0 (the l1 x l2 -> l3 coupling)")
        lin = lambda i, o: nn.Linear(i, o, bias=False)  # noqa: E731
        if self.kt:
            self.ttt_c = lin(k2, self.kt)
        if self.kr:
            self.vttv_a, self.vttv_b = lin(k2, self.kr), lin(k2, self.kr)
            self.vttv_p, self.vttv_q = lin(k, self.kr), lin(k, self.kr)
        if self.k3:
            self.radial3 = lin(num_rbf, self.k3)
            self.gate3 = nn.Linear(d_model, self.k3)
            self.u3, self.v3, self.w3 = lin(self.k3, self.k3), lin(self.k3, self.k3), lin(self.k3, self.k3)
            self.p3, self.a3 = lin(k, self.k3), lin(k2, self.k3)
        self.n_out = self.kt + self.kr + 2 * self.k3
        self.out = lin(self.n_out, d_model)
        nn.init.zeros_(self.out.weight)
        self.record: dict | None = None  # calibration only (MenaACE.calibrate_angular): family RMS of the last forward
        self._is_baked: bool = False
        self._baked_wt: Tensor | None = None
        self._baked_wv: Tensor | None = None
        self._baked_versions: tuple[int, ...] | None = None

    def _weight_versions(self) -> tuple[int, ...]:
        return tuple(
            w._version
            for w in (
                ([self.ttt_c.weight] if self.kt else [])
                + ([self.vttv_a.weight, self.vttv_b.weight, self.vttv_p.weight, self.vttv_q.weight] if self.kr else [])
                + ([self.a3.weight, self.p3.weight] if self.k3 else [])
            )
        )

    @torch.no_grad()
    def bake_for_inference(self) -> None:
        self._is_baked = True
        wt = ([self.ttt_c.weight] if self.kt else []) + ([self.vttv_a.weight, self.vttv_b.weight] if self.kr else []) \
            + ([self.a3.weight] if self.k3 else [])
        wv = ([self.vttv_p.weight, self.vttv_q.weight] if self.kr else []) + ([self.p3.weight] if self.k3 else [])
        self._baked_wt = torch.cat(wt, 0).detach().contiguous() if wt else None
        self._baked_wv = torch.cat(wv, 0).detach().contiguous() if wv else None
        self._baked_versions = self._weight_versions()

    def unbake_for_inference(self) -> None:
        self._is_baked = False
        self._baked_wt = None
        self._baked_wv = None
        self._baked_versions = None

    def _apply(self, fn, recurse: bool = True):
        was_baked = self._is_baked
        self.unbake_for_inference()
        super()._apply(fn, recurse=recurse)
        if was_baked and not self.training:
            self.bake_for_inference()
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        was_baked = self._is_baked
        self.unbake_for_inference()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if was_baked and not self.training:
            self.bake_for_inference()

    def train(self, mode: bool = True) -> "AngularBranch":
        super().train(mode)
        if mode:
            self.unbake_for_inference()
        return self

    def families(self) -> list[tuple[str, int, int]]:
        """(name, first column, last column) of each invariant family in the features() output."""
        fam, o = [], 0
        for name, n in (("ttt", self.kt), ("vttv", self.kr), ("nu2_3", self.k3), ("vt3", self.k3)):
            if n:
                fam.append((name, o, o + n))
                o += n
        return fam

    @torch.no_grad()
    def rescale(self, family: str, ratio: float) -> None:
        """Multiply the family's invariants by `ratio` by scaling the factors it is homogeneous in (calibration):
        ttt cubic in C; vttv degree 4 in (A, B, P, Q); nu2_3 quadratic in radial3 (T3 is linear in it; this also
        scales vt3 by sqrt(ratio)); vt3 cubic in (W3, P3, A3)."""
        groups = {"ttt": ([self.ttt_c] if self.kt else [], 3), "vttv": ([self.vttv_a, self.vttv_b, self.vttv_p,
                  self.vttv_q] if self.kr else [], 4), "nu2_3": ([self.radial3] if self.k3 else [], 2),
                  "vt3": ([self.w3, self.p3, self.a3] if self.k3 else [], 3)}
        mods, deg = groups[family]
        for m in mods:
            m.weight.mul_(ratio ** (1.0 / deg))

    def density(self, filt3: Tensor, gate_j: Tensor, y3: Tensor, recv: Tensor, N: int, norm: Tensor, fe: bool) -> Tensor:
        """T3 [N, k3, 7] from the per-edge filter filt3 [E, k3], the gathered sender gate gate_j [E, k3] and y3 [E, 7]."""
        w3 = filt3 * gate_j
        prod = eo.outer(w3, y3) if fe else w3.unsqueeze(-1) * y3.unsqueeze(1)
        return filt3.new_zeros(N, self.k3, 7).index_add_(0, recv, prod) * norm.unsqueeze(-1)

    def density_dense(self, filt3: Tensor, gate_j: Tensor, y3: Tensor, norm: Tensor, small: bool) -> Tensor:
        """Dense version of density(): filt3 [B, N, N, k3], gate_j [B, N, k3], y3 [B, N, N, 7] -> T3 [B, N, k3, 7]."""
        B, N, _ = gate_j.shape
        w3 = filt3 * gate_j.unsqueeze(1)
        if small:
            T3 = (w3.unsqueeze(-1) * y3.unsqueeze(3)).sum(dim=2)
        else:
            T3 = torch.bmm(w3.reshape(B * N, N, self.k3).transpose(1, 2), y3.reshape(B * N, N, 7)).view(B, N, self.k3, 7)
        return T3 * norm.unsqueeze(-1)

    def features(self, T: Tensor, v: Tensor, T3: Tensor | None) -> Tensor:
        """Invariants [..., n_out] from T [..., K2, 5] (scaled by norm, as for the stage's own invariants), v [..., K, 3] and
        T3 [..., k3, 7] (or None). Supports both [N, ...] (sparse) and [B, N, ...] (dense)."""
        if (
            not self.training
            and self._baked_wt is not None
            and self._baked_wt.device == T.device
            and self._baked_wt.dtype == T.dtype
            and baked_current(self._baked_versions, self._weight_versions())
        ):
            wt_cat = self._baked_wt
            wv_cat = self._baked_wv
        else:
            wt = ([self.ttt_c.weight] if self.kt else []) + ([self.vttv_a.weight, self.vttv_b.weight] if self.kr else []) \
                + ([self.a3.weight] if self.k3 else [])
            wv = ([self.vttv_p.weight, self.vttv_q.weight] if self.kr else []) + ([self.p3.weight] if self.k3 else [])
            wt_cat = torch.cat(wt, 0)
            wv_cat = torch.cat(wv, 0) if wv else None
        yt = F.linear(T.transpose(-1, -2).contiguous(), wt_cat).transpose(-1, -2)  # [..., kt + 2 kr + k3, 5]
        yv = F.linear(v.transpose(-1, -2).contiguous(), wv_cat).transpose(-1, -2) if wv_cat is not None else None  # [..., 2 kr + k3, 3]
        feats, o = [], 0
        if self.kt:
            feats.append(3.0 * l2_det(l2_mat6(yt[..., :self.kt, :])))  # tr(M^3)
            o = self.kt
        if self.kr:
            ma, mb = l2_mat6(yt[..., o:o + self.kr, :]), l2_mat6(yt[..., o + self.kr:o + 2 * self.kr, :])
            feats.append((l2_matvec(ma, yv[..., :self.kr, :]) * l2_matvec(mb, yv[..., self.kr:2 * self.kr, :])).sum(-1))
            o += 2 * self.kr
        if self.k3:
            feats.append((_mix(T3, self.u3) * _mix(T3, self.v3)).sum(-1))
            m9 = yt[..., o:o + self.k3, :] @ _const("l2_e9", T)  # [..., k3, 9]: M(A3 T) row-major
            pv = yv[..., yv.shape[-2] - self.k3:, :]  # [..., k3, 3]
            vm = (pv.unsqueeze(-1) * m9.unsqueeze(-2)).flatten(-2)  # [..., k3, 27]: v_a M_bc
            feats.append((_mix(T3, self.w3) * (vm @ _const("l3_b27", vm))).sum(-1))
        out = torch.cat(feats, dim=-1)
        if self.record is not None:
            self.record.update({n: float(out[..., a:b].detach().pow(2).mean().sqrt()) for n, a, b in self.families()})
        return out


class ACEStage(nn.Module):
    """One message-passing stage: scalar convolution + l=1/l=2 densities + transported vectors -> invariants."""

    def __init__(self, d_model: int, k: int, k2: int, num_rbf: int, swiglu_hidden: int, first: bool,
                 cross_channel: bool = True, msg_dim: int = 0) -> None:
        super().__init__()
        self.k, self.k2, self.first = k, k2, first
        self.stage_idx: int = -1
        m = int(msg_dim) if msg_dim else d_model  # scalar message width (see MenaACE msg_dim)
        self.small_n = 0  # set to SMALL_N under torch.compile (see SMALL_N)
        self.fast: frozenset = frozenset()  # execution-only op fusions, set by MenaACE.set_fast_ops (not in the config)
        self.norm_in = nn.RMSNorm(d_model, eps=1e-6)

        # Scalar continuous-filter convolution: sum_j (W_r rbf_ij) * w_ij * s'_j
        self.scalar_pre = nn.Linear(d_model, m)
        self.scalar_radial = nn.Linear(num_rbf, m, bias=False)

        # Sender gates for the l=1 and l=2 densities
        self.vec_gate = nn.Linear(d_model, k)
        self.ten_gate = nn.Linear(d_model, k2)
        # One radial filter bank per stage: directional l=1, l=2 density, and (stages > 0) full + longitudinal transport
        self.n_filt = k + k2 + (0 if first else 2 * k)
        self.radial = nn.Linear(num_rbf, self.n_filt, bias=False)
        self.transport_mix = None if first else nn.Linear(k, k, bias=False)

        # Invariants
        # cross_channel=False (ablation): per-channel norms |v_k|^2 instead of <U v, V v>, as in MenaMinimal
        self.u1 = nn.Linear(k, k, bias=False) if cross_channel else None
        self.v1 = nn.Linear(k, k, bias=False) if cross_channel else None
        self.u2 = nn.Linear(k2, k2, bias=False)
        self.v2 = nn.Linear(k2, k2, bias=False)
        self.p3 = nn.Linear(k, k2, bias=False)
        self.q3 = nn.Linear(k, k2, bias=False)
        self.t3 = nn.Linear(k2, k2, bias=False)
        inv_dim = k + k2 + k2 + (0 if first else k)
        self.inv_mlp = nn.Sequential(nn.Linear(inv_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        nn.init.zeros_(self.inv_mlp[-1].weight)
        nn.init.zeros_(self.inv_mlp[-1].bias)

        self.out_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.swiglu = MinimalSwiGLU(d_model=d_model, hidden_dim=swiglu_hidden)

        # Scalar-gated vector update (PaiNN-style): v += (W v) * g(s), g zero-initialised
        self.vec_self = nn.Linear(k, k, bias=False)
        self.vec_out_gate = nn.Linear(d_model, k)
        nn.init.zeros_(self.vec_out_gate.weight)
        nn.init.zeros_(self.vec_out_gate.bias)
        # msg_dim < d_model: the summed scalar message (width m) is mapped back to the atom width (built last)
        self.msg_up = nn.Linear(m, d_model, bias=False) if m != d_model else None
        self.grow: GrowthBranch | None = None  # see add_growth
        self.ang: AngularBranch | None = None  # see add_angular
        self.l4: L4Branch | None = None  # see add_l4
        self.norm_scale: nn.Linear | None = None  # see MenaACE norm_scale
        self.vswiglu: VectorSwiGLU | None = None  # see add_vswiglu
        self.ehead: EdgeHeadFactor | None = None  # see add_edge_heads
        self._is_baked: bool = False
        self._baked_proj_w: Tensor | None = None
        self._baked_proj_b: Tensor | None = None
        self._baked_radial_w: Tensor | None = None
        self._baked_uvpq_w: Tensor | None = None
        self._baked_uvt_w: Tensor | None = None

    @torch.no_grad()
    def bake_for_inference(self) -> None:
        self._is_baked = True
        K, K2 = self.k, self.k2
        ang = self.ang
        k3 = ang.k3 if ang is not None else 0
        if K and K2:
            ws = [self.scalar_pre.weight, self.vec_gate.weight, self.ten_gate.weight] + ([ang.gate3.weight] if k3 else [])
            bs = [self.scalar_pre.bias, self.vec_gate.bias, self.ten_gate.bias] + ([ang.gate3.bias] if k3 else [])
            self._baked_proj_w = torch.cat(ws, 0).detach().contiguous()
            self._baked_proj_b = torch.cat(bs, 0).detach().contiguous()
        else:
            self._baked_proj_w = None
            self._baked_proj_b = None
        if K or K2:
            rws = [self.scalar_radial.weight, self.radial.weight] + ([ang.radial3.weight] if k3 else [])
            self._baked_radial_w = torch.cat(rws, 0).detach().contiguous()
        else:
            self._baked_radial_w = None
        if K2 and self.u1 is not None:
            self._baked_uvpq_w = torch.cat([self.u1.weight, self.v1.weight, self.p3.weight, self.q3.weight], 0).detach().contiguous()
            self._baked_uvt_w = torch.cat([self.u2.weight, self.v2.weight, self.t3.weight], 0).detach().contiguous()
        else:
            self._baked_uvpq_w = None
            self._baked_uvt_w = None
        if ang is not None:
            ang.bake_for_inference()

    def unbake_for_inference(self) -> None:
        self._is_baked = False
        self._baked_proj_w = None
        self._baked_proj_b = None
        self._baked_radial_w = None
        self._baked_uvpq_w = None
        self._baked_uvt_w = None
        if self.ang is not None:
            self.ang.unbake_for_inference()

    def _apply(self, fn, recurse: bool = True):
        was_baked = self._is_baked
        self.unbake_for_inference()
        super()._apply(fn, recurse=recurse)
        if was_baked and not self.training:
            self.bake_for_inference()
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        was_baked = self._is_baked
        self.unbake_for_inference()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if was_baked and not self.training:
            self.bake_for_inference()

    def train(self, mode: bool = True) -> "ACEStage":
        super().train(mode)
        if mode:
            self.unbake_for_inference()
        return self

    def add_vswiglu(self, d_model: int, hidden: int) -> None:
        """Attach the VectorSwiGLU branch (called by MenaACE after every other parameter is built)."""
        self.vswiglu = VectorSwiGLU(self.k, d_model, hidden)

    def add_edge_heads(self, d_model: int, num_rbf: int, heads: int) -> None:
        """Attach the multi-head edge factor (called by MenaACE after every other parameter is built)."""
        m = self.scalar_radial.out_features
        k3 = self.ang.k3 if self.ang is not None else 0
        for name, c in (("msg_dim", m), ("vector_rank", self.k), ("tensor_rank", self.k2), ("l3_rank", k3)):
            if c and c % heads != 0:
                raise ValueError(f"{name}={c} must be divisible by edge_heads={heads}")
        self.ehead = EdgeHeadFactor(d_model, num_rbf, heads)

    def add_angular(self, d_model: int, num_rbf: int, ttt_mix: int, vttv_rank: int, l3_rank: int) -> None:
        """Attach the B1/B2 angular branches (called by MenaACE after every other parameter is built)."""
        self.ang = AngularBranch(d_model, self.k, self.k2, num_rbf, ttt_mix, vttv_rank, l3_rank)

    def add_l4(self, d_model: int, num_rbf: int, l4_rank: int) -> None:
        """Attach the B3 l=4 hexadecapole branch (called by MenaACE after every other parameter is built)."""
        self.l4 = L4Branch(d_model, self.k2, num_rbf, l4_rank)

    def add_growth(self, d_model: int, num_rbf: int, grow_tensor_rank: int, edge_quadratic: bool,
                   tensor_vector_update: bool) -> None:
        """Attach the additive growth modules (called by MenaACE after every other parameter is built)."""
        self.grow = GrowthBranch(d_model, self.k, grow_tensor_rank, num_rbf, self.first, edge_quadratic,
                                 tensor_vector_update)

    def _post_norm(self, x: Tensor) -> Tensor:
        """out_norm(x), plus (norm_scale) a zero-initialised linear map of the per-atom log RMS of x, the one number the
        RMSNorm removes. Without it, a perturbation that only rescales an atom's residual stream (uniform compression at
        stage 0: 40% of the strain response of an MPtrj-trained model) cannot reach the next stage."""
        y = self.out_norm(x)
        if self.norm_scale is not None:
            y = y + self.norm_scale(0.5 * torch.log(x.pow(2).mean(-1, keepdim=True) + 1e-6))
        return y

    def _inv_update(self, inv: Tensor, extra: list[Tensor], ang: Tensor | None = None,
                    l4: Tensor | None = None) -> Tensor:
        """inv_mlp(inv), plus the growth invariants through inv_extra and the angular branch invariants
        through their zero-init .out layers, added to the pre-activation of inv_mlp[0]."""
        if not extra and ang is None and l4 is None:
            return self.inv_mlp(inv)
        h = self.inv_mlp[0](inv)
        if extra:
            h = h + self.grow.inv_extra(torch.cat(extra, dim=-1))
        if ang is not None:
            h = h + self.ang.out(ang)
        if l4 is not None:
            h = h + self.l4.out(l4)
        return self.inv_mlp[2](self.inv_mlp[1](h))

    def _invariants_fused(self, v_new: Tensor, T: Tensor) -> tuple[Tensor, list[Tensor]]:
        """nu2_v, [nu2_t, nu3] with the bias-free channel mixes that read the same tensor merged into one matmul each
        (u1, v1, p3, q3 on v_new; u2, v2, t3 on T), as the maps are linear in the stacked weight. Equal to the
        unfused code up to float rounding (the summation order inside a matmul)."""
        K, K2 = self.k, self.k2
        if not self.training and self._baked_uvpq_w is not None and self._baked_uvpq_w.device == v_new.device and self._baked_uvpq_w.dtype == v_new.dtype:
            uvpq_w = self._baked_uvpq_w
            uvt_w = self._baked_uvt_w
        else:
            uvpq_w = torch.cat([self.u1.weight, self.v1.weight, self.p3.weight, self.q3.weight], 0)
            uvt_w = torch.cat([self.u2.weight, self.v2.weight, self.t3.weight], 0)
        yv = F.linear(v_new.transpose(-1, -2).contiguous(), uvpq_w)  # [...,3,2K+2K2]
        uv, vv, pv, qv = yv.split([K, K, K2, K2], dim=-1)
        nu2_v = (uv * vv).sum(-2)
        yt = F.linear(T.transpose(-1, -2).contiguous(), uvt_w)  # [...,5,3K2]
        ut, vt, tt = yt.split([K2, K2, K2], dim=-1)
        nu2_t = (ut * vt).sum(-2)
        l2 = l2_pair_mm if "l2" in self.fast else l2_pair
        nu3 = (tt.transpose(-1, -2) * l2(pv.transpose(-1, -2), qv.transpose(-1, -2))).sum(-1)
        return nu2_v, [nu2_t, nu3]

    def _ffn(self, s: Tensor, lot_bias: Tensor | None, lot_bias_energy: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """s + SwiGLU(s): conditioned by lot_bias (lot_stages), or (lot_bias_energy, lot_energy_stages) token-free plus
        the token's correction returned separately for the energy readout only (None when lot_bias_energy is None)."""
        if lot_bias_energy is None:
            return s + self.swiglu(s, lot_bias=lot_bias), None
        if lot_bias is not None:
            raise ValueError("a stage is either in lot_stages or in lot_energy_stages, not both")
        free, delta = self.swiglu.forward_pair(s, lot_bias_energy)
        return s + free, delta

    def forward(self, s: Tensor, v: Tensor, g: dict, lot_bias: Tensor | None = None,
                lot_bias_energy: Tensor | None = None) -> tuple[Tensor, ...]:
        """(s, v), or (s, v, delta) when lot_bias_energy is given: s, v token-free, delta = the token's correction to s
        (read by the energy readout only, MenaACE lot_energy_stages)."""
        B, N, D = s.shape
        K, K2 = self.k, self.k2
        mask_f, norm = g["mask_f"], g["norm"]  # [B, N, 1], [B, N, 1] = 1/sqrt(degree + 1)
        edge, unit, y2 = g["edge"], g["unit"], g["y2"]  # [B,N,N,R], [B,N,N,3], [B,N,N,5]
        hs = self.norm_in(s)
        if lot_bias is None and self.stage_idx >= 0:
            lot_bias = g.get(f"lot_bias_{self.stage_idx}")

        # 1. Scalar convolution. Direct form materialises [B,N,N,D]; factorised form contracts j first
        #    ([B, N*R, N] @ [B, N, D]) and is cheaper whenever N > R.
        gate_v = gate_t = gate3 = None
        ang = self.ang
        k3 = ang.k3 if ang is not None else 0
        if "proj" in self.fast:  # scalar_pre | vec_gate | ten_gate (| ang.gate3) read the same hs: one addmm
            m_ = self.scalar_pre.out_features
            if not self.training and self._baked_proj_w is not None and self._baked_proj_w.device == hs.device and self._baked_proj_w.dtype == hs.dtype:
                proj_w, proj_b = self._baked_proj_w, self._baked_proj_b
            else:
                ws = [self.scalar_pre.weight, self.vec_gate.weight, self.ten_gate.weight] + ([ang.gate3.weight] if k3 else [])
                bs = [self.scalar_pre.bias, self.vec_gate.bias, self.ten_gate.bias] + ([ang.gate3.bias] if k3 else [])
                proj_w, proj_b = torch.cat(ws, 0), torch.cat(bs, 0)
            proj = F.linear(hs, proj_w, proj_b)
            sp = proj[..., :m_] * mask_f
            gate_v, gate_t = proj[..., m_:m_ + K], proj[..., m_ + K:m_ + K + K2]
            if k3:
                gate3 = proj[..., m_ + K + K2:]
        else:
            sp = self.scalar_pre(hs) * mask_f
        R = edge.shape[-1]
        small = N <= self.small_n  # multiply + sum instead of bmm (see SMALL_N)
        phi = self.ehead(hs, g["rbf"], g["chi"], g.get("deg1")) if self.ehead is not None else None
        if phi is not None:
            conv = self.ehead.scalar_conv_dense(edge, g.get("edge_t"), sp, self.scalar_radial.weight, phi, small=small)
        elif N <= R:
            if small:
                conv = (self.scalar_radial(edge) * sp.unsqueeze(1)).sum(dim=2)
            else:
                conv = torch.einsum("bijd,bjd->bid", self.scalar_radial(edge), sp)
        else:
            y = torch.bmm(g["edge_t"], sp).view(B, N, R, sp.shape[-1])  # [B, N, R, m]
            conv = (y * self.scalar_radial.weight.t()).sum(dim=2)
        conv = conv * norm
        if self.msg_up is not None:
            conv = self.msg_up(conv)
        if K == 0 and K2 == 0 and self.grow is None:  # radial-only (ladder G0): no angular invariants, no vectors
            s, d_e = self._ffn(self._post_norm(s + conv), lot_bias, lot_bias_energy)
            if d_e is not None:
                return s * mask_f, v, d_e * mask_f
            return s * mask_f, v

        # 2. Densities and transport
        f_tr = f_lon = None
        if "split" in self.fast and phi is None:  # one matmul per filter group: the backward has no full-width zero-padded slice gradients
            wr = self.radial.weight  # [n_filt, R]
            filt, f_dir, f_ten = None, F.linear(edge, wr[:K]), F.linear(edge, wr[K:K + K2])
        else:
            filt = self.radial(edge)  # [B, N, N, n_filt]
            if phi is not None:
                f_slices = tuple(c for c in ((K, K2) if self.first else (K, K2, K, K)) if c)
                filt = self.ehead.scale_channels(filt, phi, slices=f_slices)
            if self.first:
                f_dir, f_ten = filt.split([K, K2], dim=-1)
            else:
                f_dir, f_ten, f_tr, f_lon = filt.split([K, K2, K, K], dim=-1)
        w_dir = f_dir * (gate_v if gate_v is not None else self.vec_gate(hs)).unsqueeze(1)  # sender j on dim 2
        if small:
            dv = (w_dir.unsqueeze(-1) * unit.unsqueeze(3)).sum(dim=2)  # [B,N,N,K,1] * [B,N,N,1,3] -> [B,N,K,3]
        else:
            dv = torch.bmm(w_dir.reshape(B * N, N, K).transpose(1, 2), unit.reshape(B * N, N, 3)).view(B, N, K, 3)
        if K2:  # tensor_rank = 0 removes the l=2 channel (ablation)
            w_ten = f_ten * (gate_t if gate_t is not None else self.ten_gate(hs)).unsqueeze(1)
            if small:
                T = (w_ten.unsqueeze(-1) * y2.unsqueeze(3)).sum(dim=2)
            else:
                T = torch.bmm(w_ten.reshape(B * N, N, K2).transpose(1, 2), y2.reshape(B * N, N, 5)).view(B, N, K2, 5)

        feats = []
        if not self.first:
            if filt is None:
                f_tr, f_lon = F.linear(edge, wr[K + K2:2 * K + K2]), F.linear(edge, wr[2 * K + K2:])
            b = _mix(v, self.transport_mix)  # [B, N, K, 3]
            # full transport sum_j f_tr_ij b_j (batched over B*K)
            if small:
                dv_tr = (f_tr.unsqueeze(-1) * b.unsqueeze(1)).sum(dim=2)  # [B,N,N,K,1] * [B,1,N,K,3]
            else:
                wt = f_tr.permute(0, 3, 1, 2).reshape(B * K, N, N)
                dv_tr = torch.bmm(wt, b.transpose(1, 2).reshape(B * K, N, 3)).view(B, K, N, 3).transpose(1, 2)
            # longitudinal part: u_ij = f_lon_ij (r_ij . b_j); scalar bond projection sum_j u_ij, vector sum_j u_ij r_ij
            if small:
                lon = (unit.unsqueeze(3) * b.unsqueeze(1)).sum(dim=-1)  # [B,N,N,1,3] * [B,1,N,K,3]
            else:
                lon = torch.einsum("bijc,bjkc->bijk", unit, b)
            u = f_lon * lon
            bond = u.sum(dim=2) * norm
            if small:
                dv_lon = (u.unsqueeze(-1) * unit.unsqueeze(3)).sum(dim=2)
            else:
                dv_lon = torch.bmm(u.reshape(B * N, N, K).transpose(1, 2), unit.reshape(B * N, N, 3)).view(B, N, K, 3)
            dv = dv + dv_tr + dv_lon
            feats.append(bond)
        filt3 = self.ehead.scale_channels(ang.radial3(edge), phi) if (phi is not None and k3) else None

        v_new = (v + dv * norm.unsqueeze(-1)) * mask_f.unsqueeze(-1)

        # Growth branch (absent in every trained model before it was inserted): l=2 density and edge-quadratic features
        grow, extra, T_g = self.grow, [], None
        if grow is not None:
            if grow.k2:
                w_g = grow.ten_radial(edge) * grow.ten_gate(hs).unsqueeze(1)
                if small:
                    T_g = (w_g.unsqueeze(-1) * y2.unsqueeze(3)).sum(dim=2)
                else:
                    T_g = torch.bmm(w_g.reshape(B * N, N, grow.k2).transpose(1, 2), y2.reshape(B * N, N, 5)).view(B, N, grow.k2, 5)
                T_g = T_g * norm.unsqueeze(-1)
                extra += grow.tensor_features(T_g, v_new)
            if grow.edge_quad is not None:
                extra.append((grow.edge_quad(edge) * lon.pow(2)).sum(dim=2) * norm)

        # 3. Invariants
        if "mix" in self.fast and K2 and self.u1 is not None and grow is None:
            T = T * norm.unsqueeze(-1)
            nu2_v, ten_feats = self._invariants_fused(v_new, T)
        else:
            nu2_v = _pair_inv(v_new, self.u1, self.v1)
            ten_feats = []
            if K2:
                T = T * norm.unsqueeze(-1)
                nu2_t = (_mix(T, self.u2) * _mix(T, self.v2)).sum(-1)
                l2 = l2_pair_mm if "l2" in self.fast else l2_pair
                nu3 = (_mix(T, self.t3) * l2(_mix(v_new, self.p3), _mix(v_new, self.q3))).sum(-1)
                ten_feats = [nu2_t, nu3]
        inv = torch.cat([nu2_v] + ten_feats + feats, dim=-1)

        # 3b. Angular branches B1/B2/B3
        ang_feats = l4_feats = None
        if ang is not None:
            T3 = None
            if k3:
                if filt3 is None:
                    filt3 = ang.radial3(edge)
                T3 = ang.density_dense(filt3, gate3 if gate3 is not None else ang.gate3(hs), g["y3"], norm, small)
            ang_feats = ang.features(T, v_new, T3)
            if ang.record is not None:
                ang.record["inv"] = float(inv.detach().pow(2).mean().sqrt())
        if self.l4 is not None:
            l4_feats = self.l4.forward(edge, hs, T, g["y4"], norm, small)
            if self.l4.record is not None:
                self.l4.record["inv"] = float(inv.detach().pow(2).mean().sqrt())

        # 4. Scalar update
        s = self._post_norm(s + conv + self._inv_update(inv, extra, ang_feats, l4_feats))
        s, d_e = self._ffn(s, lot_bias, lot_bias_energy)
        s = s * mask_f

        # 5. Gated vector update
        v_out = v_new + _mix(v_new, self.vec_self) * self.vec_out_gate(s).unsqueeze(-1)
        if grow is not None and grow.tv_w is not None:
            v_out = v_out + grow.vector_update(T_g, v_new, s)
        if self.vswiglu is not None:
            v_out = self.vswiglu(v_out, s)
        if d_e is not None:
            return s, v_out * mask_f.unsqueeze(-1), d_e * mask_f
        return s, v_out * mask_f.unsqueeze(-1)

    def forward_sparse(self, s: Tensor, v: Tensor, g: dict, lot_bias: Tensor | None = None,
                       lot_bias_energy: Tensor | None = None) -> tuple[Tensor, ...]:
        """Edge-list version of `forward`: s [N, D], v [N, K, 3]; g holds per-edge tensors and receiver/sender index.

        Every sum over senders j becomes an index_add_ over edges into receivers i; per-atom math is identical.
        """
        N = s.shape[0]
        K, K2 = self.k, self.k2
        recv, send, norm = g["recv"], g["send"], g["norm"]  # [E], [E], [N, 1]
        edge, unit, y2 = g["edge"], g["unit"], g["y2"]  # [E, R], [E, 3], [E, 5]
        hs = self.norm_in(s)
        if lot_bias is None and self.stage_idx >= 0:
            lot_bias = g.get(f"lot_bias_{self.stage_idx}")
        # execution-only options (MenaACE.set_fast_ops; same maths): "gather" index_select instead of advanced indexing
        # (MPS: 6x faster forward, index_add backward); "edge" the per-edge products of lam/models/edge_ops.py (no
        # reductions over the 3/5-component axis in backward or double backward, the MPS bottleneck)
        gat = (lambda t: t.index_select(0, send)) if "gather" in self.fast else (lambda t: t[send])
        fe = "edge" in self.fast

        # 1. Scalar convolution sum_j (W_r rbf_ij w_ij) * s'_j
        gate_v = gate_t = gate3 = filt = None
        m_ = self.scalar_radial.out_features
        ang = self.ang
        k3 = ang.k3 if ang is not None else 0
        if "proj" in self.fast and K and K2:  # scalar_pre | vec_gate | ten_gate (| ang.gate3) read the same hs: one addmm
            if not self.training and self._baked_proj_w is not None and self._baked_proj_w.device == hs.device and self._baked_proj_w.dtype == hs.dtype:
                proj_w, proj_b = self._baked_proj_w, self._baked_proj_b
            else:
                ws = [self.scalar_pre.weight, self.vec_gate.weight, self.ten_gate.weight] + ([ang.gate3.weight] if k3 else [])
                bs = [self.scalar_pre.bias, self.vec_gate.bias, self.ten_gate.bias] + ([ang.gate3.bias] if k3 else [])
                proj_w, proj_b = torch.cat(ws, 0), torch.cat(bs, 0)
            proj = F.linear(hs, proj_w, proj_b)
            sp, gate_v, gate_t = proj[:, :m_], proj[:, m_:m_ + K], proj[:, m_ + K:m_ + K + K2]
            if k3:
                gate3 = proj[:, m_ + K + K2:]
        else:
            sp = self.scalar_pre(hs)
        filt3 = None
        phi = self.ehead.forward_sparse(hs, g["rbf"], g["chi"], recv, send, g.get("deg1")) if self.ehead is not None else None
        if ("radial" in self.fast or phi is not None) and (K or K2):  # scalar_radial | radial (| ang.radial3) read the same edge basis: one GEMM
            if not self.training and self._baked_radial_w is not None and self._baked_radial_w.device == edge.device and self._baked_radial_w.dtype == edge.dtype:
                rad_w = self._baked_radial_w
            else:
                rad_w = torch.cat([self.scalar_radial.weight, self.radial.weight] + ([ang.radial3.weight] if k3 else []), 0)
            fr = F.linear(edge, rad_w)
            if phi is not None:
                fr_slices = tuple(c for c in ((m_, K, K2) + (() if self.first else (K, K)) + ((k3,) if k3 else ())) if c)
                fr = self.ehead.scale_channels(fr, phi, slices=fr_slices)
            if k3:
                sr, filt, filt3 = fr.split([m_, self.n_filt, k3], dim=-1)
            else:
                sr, filt = fr.split([m_, self.n_filt], dim=-1)
        else:
            sr = self.scalar_radial(edge)
            if phi is not None:
                sr = self.ehead.scale_channels(sr, phi)
        conv = s.new_zeros(N, sp.shape[-1]).index_add_(0, recv, sr * gat(sp)) * norm
        if self.msg_up is not None:
            conv = self.msg_up(conv)
        if K == 0 and K2 == 0 and self.grow is None:  # radial-only (ladder G0), as in forward
            s, d_e = self._ffn(self._post_norm(s + conv), lot_bias, lot_bias_energy)
            return (s, v) if d_e is None else (s, v, d_e)

        # 2. Densities and transport
        f_tr = f_lon = None
        if filt is None:
            filt = self.radial(edge)  # [E, n_filt]
            if phi is not None:
                f_slices = tuple(c for c in ((K, K2) if self.first else (K, K2, K, K)) if c)
                filt = self.ehead.scale_channels(filt, phi, slices=f_slices)
        if self.first:
            f_dir, f_ten = filt.split([K, K2], dim=-1)
        else:
            f_dir, f_ten, f_tr, f_lon = filt.split([K, K2, K, K], dim=-1)
        w_dir = f_dir * gat(gate_v if gate_v is not None else self.vec_gate(hs))
        if K2:  # tensor_rank = 0 removes the l=2 channel (ablation)
            w_ten = f_ten * gat(gate_t if gate_t is not None else self.ten_gate(hs))
            T = s.new_zeros(N, K2, 5).index_add_(0, recv, eo.outer(w_ten, y2) if fe else w_ten.unsqueeze(-1) * y2.unsqueeze(1))

        feats = []
        if self.first:
            dv = s.new_zeros(N, K, 3).index_add_(0, recv, eo.outer(w_dir, unit) if fe else w_dir.unsqueeze(-1) * unit.unsqueeze(1))
        else:
            b = _mix(v, self.transport_mix)  # [N, K, 3]
            b_j = gat(b)  # [E, K, 3]
            lon = eo.contract_c(b_j, unit) if fe else (unit.unsqueeze(1) * b_j).sum(-1)  # [E, K]
            u = f_lon * lon
            bond = s.new_zeros(N, K).index_add_(0, recv, u) * norm
            w_vec = w_dir + u
            vec_edge = (eo.outer(w_vec, unit) if fe else w_vec.unsqueeze(-1) * unit.unsqueeze(1)) \
                + (eo.scale(b_j, f_tr) if fe else f_tr.unsqueeze(-1) * b_j)
            dv = s.new_zeros(N, K, 3).index_add_(0, recv, vec_edge)
            feats.append(bond)
        if phi is not None and k3 and filt3 is None:
            filt3 = self.ehead.scale_channels(ang.radial3(edge), phi)

        v_new = v + dv * norm.unsqueeze(-1)

        grow, extra, T_g = self.grow, [], None
        if grow is not None:
            if grow.k2:
                w_g = grow.ten_radial(edge) * grow.ten_gate(hs)[send]
                T_g = s.new_zeros(N, grow.k2, 5).index_add_(0, recv, w_g.unsqueeze(-1) * y2.unsqueeze(1)) * norm.unsqueeze(-1)
                extra += grow.tensor_features(T_g, v_new)
            if grow.edge_quad is not None:
                extra.append(s.new_zeros(N, self.k).index_add_(0, recv, grow.edge_quad(edge) * lon.pow(2)) * norm)

        # 3. Invariants
        if "mix" in self.fast and K2 and self.u1 is not None and grow is None:
            T = T * norm.unsqueeze(-1)
            nu2_v, ten_feats = self._invariants_fused(v_new, T)
        else:
            nu2_v = _pair_inv(v_new, self.u1, self.v1)
            ten_feats = []
            if K2:
                T = T * norm.unsqueeze(-1)
                nu2_t = (_mix(T, self.u2) * _mix(T, self.v2)).sum(-1)
                nu3 = (_mix(T, self.t3) * l2_pair(_mix(v_new, self.p3), _mix(v_new, self.q3))).sum(-1)
                ten_feats = [nu2_t, nu3]
        inv = torch.cat([nu2_v] + ten_feats + feats, dim=-1)

        # 3b. Angular branches B1/B2/B3
        ang_feats = l4_feats = None
        if ang is not None:
            T3 = None
            if k3:
                if filt3 is None:
                    filt3 = ang.radial3(edge)
                T3 = ang.density(filt3, gat(gate3 if gate3 is not None else ang.gate3(hs)), g["y3"], recv, N, norm, fe)
            ang_feats = ang.features(T, v_new, T3)
            if ang.record is not None:
                ang.record["inv"] = float(inv.detach().pow(2).mean().sqrt())
        if self.l4 is not None:
            l4_feats = self.l4.forward_sparse(edge, hs, T, g["y4"], recv, gat, N, norm, fe)
            if self.l4.record is not None:
                self.l4.record["inv"] = float(inv.detach().pow(2).mean().sqrt())

        # 4. Scalar update
        s = self._post_norm(s + conv + self._inv_update(inv, extra, ang_feats, l4_feats))
        s, d_e = self._ffn(s, lot_bias, lot_bias_energy)

        # 5. Gated vector update
        v_out = v_new + _mix(v_new, self.vec_self) * self.vec_out_gate(s).unsqueeze(-1)
        if grow is not None and grow.tv_w is not None:
            v_out = v_out + grow.vector_update(T_g, v_new, s)
        if self.vswiglu is not None:
            v_out = self.vswiglu(v_out, s)
        return (s, v_out) if d_e is None else (s, v_out, d_e)


class MenaACE(nn.Module):
    """Dense-graph E(3)-invariant potential; energy = sum of atomic energies + latent-charge screened Coulomb.

    The ωB97M-D3(BJ) term is not part of this module (see module docstring and models/model_factory.WithD3).
    """

    arch_name = "mena_ace"

    def __init__(
        self,
        d_model: int = 256,
        vector_rank: int = 48,
        tensor_rank: int = 32,
        stages: int = 3,
        num_rbf: int = 32,
        r_cut: float = 5.5,
        r_on: float = 4.0,
        num_elements: int = 119,
        swiglu_ratio: float = 2.0 / 3.0,
        charge_hidden: int = 64,
        use_charge_embedding: bool = True,
        use_electrostatics: bool = True,
        r_cut_elec: float = 18.0,
        r_on_elec: float = 16.5,
        cross_channel: bool = True,
        use_spin_embedding: bool = False,
        max_spin: int = 11,
        small_global_init: bool = False,
        use_nqe: bool = False,
        charge_feedback: bool = False,
        feedback_points: int = 2,
        feedback_hidden: int = 64,
        charge_mode: str = "global",
        direct_force_head: bool = False,
        output_scale: float = 1.0,
        coulomb_range: str = "taper",
        qeq: bool = False,
        qeq_j0: float = 230.0,
        charge_bound: float | None = None,
        feedback_bound: float | None = None,
        gated_stages: int = 0,
        grow_tensor_rank: int = 0,
        tensor_vector_update: bool = False,
        edge_quadratic: bool = False,
        null_probe_hidden: int = 0,
        band_energy: bool = False,
        band_version: int = 1,
        attention_mixer: str = "none",
        attention_heads: int = 4,
        island_context: str | Sequence[int] | None = None,
        ic_dim: int = 32,
        ic_heads: int = 4,
        ic_hidden: int = 64,
        ic_mode: str = "island",
        ic_r_cut: float = 18.0,
        ic_r_on: float = 6.0,
        ic_lr_dim: int = 0,
        dsf_alpha: float = 0.0,
        pbc_dsf_alpha: float = 0.2,
        elec_lot_ids: Sequence[int] = (6,),
        fb_input_norm: bool = False,
        pair_radius: bool = False,
        pair_basis_hidden: int = 0,
        pair_spline: bool = False,
        msg_dim: int = 0,
        ttt_mix: int = 0,
        vttv_rank: int = 0,
        l3_rank: int = 0,
        norm_scale: bool = False,
        env_pair_rank: int = 0,
        l4_rank: int = 0,
        c3_taper: bool = False,
        pair_radius_warp: bool = False,
        num_lots: int = 0,
        d_lot: int = 64,
        lot_stages: Sequence[int] | None = None,
        lot_atom_input: bool = True,
        stage_readout: bool = True,
        elec_gate: str = "lot",
        long_range: str = "edges",
        ic_windows: Sequence[Sequence[float]] | None = None,
        coulomb_sigma: float = 2.0,
        mesh_spacing: float = 0.8,
        mesh_order: int = 6,
        lot_energy_stages: Sequence[int] = (),
        lot_embed_init_std: float = 0.02,
        vector_swiglu_hidden: int = 0,
        edge_heads: int = 0,
        zbl_core: bool = False,
        zbl_r_out_scale: float = 0.6,
        domain_readout: bool = False,
    ) -> None:
        """use_nqe (borrowed idea: AIMNet2's neutral charge equilibration, Anstine, Zubatyuk and Isayev 2025, adapted
        here to MenaACE's latent Coulomb charges): the charge correction that makes the charges sum to Q is shared out
        with learned per-atom weights f_i = softplus(nqe_head(state_i)) instead of uniformly,
            q_i = q_raw_i + f_i (Q - sum_j q_raw_j) / sum_j f_j.
        The head starts at weight 0, bias log(e - 1), so f = 1 and the expression is exactly the uniform one. Absent from
        the model config when False, so old configs, checkpoints and their hashes are unchanged.

        charge_feedback (borrowed idea: AIMNet2's charge feedback into message passing, Anstine, Zubatyuk and Isayev
        2025; the constrained-charge form and the zero-initialised embedding are this repo's adaptation). Until now the
        many-body stages knew the net charge only through Q/N added to every atom and never where it sits. With the flag
        on, after each of the first `feedback_points` stages a small head predicts per-atom charges from the current
        state, shared out to sum to the charge (of each charge group, or of the graph) with learned positive weights,
            q_i = q_raw_i + f_i (Q - sum_j q_raw_j) / sum_j f_j,   f_i = softplus(w(state_i)),
        and an MLP(1 -> feedback_hidden -> d_model) of q_i is ADDED to atom i's scalar state before the next stage. The
        last layer of that MLP starts at zero (and f = 1 at start), so a model built with the flag and loaded with
        weights of the same model without it (missing keys: fb_* only) computes exactly the same energies and forces.
        The Q/N embedding, the final charge head and the Coulomb term are unchanged. Absent from the model config when
        False, so old configs, checkpoints and their hashes are unchanged.

        charge_mode="island" (size-consistent charge input): an
        island is a connected component of the r < r_cut graph, i.e. a set of atoms that exchange messages. In "global"
        mode every atom gets Q/N of the whole system, so a distant fragment changes every atom's input (ion + water at
        60 A was off by 35-71 kcal/mol). In "island" mode:
          - no Q/N broadcast: the total charge reaches the network only through per-atom charges;
          - each island carries its own charge Q_I (the sum over the island of the per-atom seed `atom_charge`, default
            Q/N spread uniformly, so a single island gets exactly Q); the charge-feedback charges after the first
            stages and the final latent charges are equilibrated per island with learned weights (AIMNet2's NQE,
            borrowed, see use_nqe / charge_feedback), so use_nqe and charge_feedback are switched on;
          - the spin input is per island: the total multiplicity for a single island, otherwise the lowest one allowed
            by the island's electron count (singlet or doublet).
        Atoms of different islands share no edge, so E(A + B far apart) = E(A) + E(B) exactly for any weights, once
        each island has its integer charge (from the caller, e.g. the inference wrapper's charge assignment). Known
        limit: the partition is discrete, so a fragment crossing r_cut changes its seed charge by its share of the
        island charge (small when the learned weights localise the charge). Absent from the config in "global" mode.

        coulomb_range="full": the latent-charge Coulomb sums every pair, with no 8-12 A taper (default "taper" keeps
        r_on_elec/r_cut_elec). Why: on OMol25 val-Comp (C1L2) the energy residual of charged structures correlates with
        the Coulomb the taper removes (|Q| >= 3: r = 0.63).

        qeq=True (charge equilibration with Coulomb coupling; dense forward only; implies coulomb_range="full"):
        the charges minimise
            E_qeq(q) = sum_i chi_i q_i + 1/2 sum_i J_i q_i^2 + 1/2 sum_{i != j} q_i q_j k erf(r_ij / s_ij) / r_ij
        subject to sum_{i in I} q_i = Q_I for every island I (one constraint per graph outside island mode), and
        E_qeq(q*) is the electrostatic energy. chi_i = qeq_j0 * charge_head(state_i) (kcal/mol/e), J_i = k / (sigma_i
        sqrt(pi)) + qeq_j0 * (softplus(qeq_head(state_i)) + 0.1) (kcal/mol/e^2), s_ij = sqrt(2 (sigma_i^2 + sigma_j^2)) with a
        learned width sigma_Z per element (Gaussian charge densities; J_i >= the Gaussian self-energy keeps the matrix
        positive definite, so the solve is a Cholesky). The NQE sharing used so far (use_nqe) is the special case
        without the Coulomb coupling (q = q_raw + f (Q - sum q_raw) / sum f with q_raw = -chi / J, f = 1 / J); with it
        the charges respond to the field of every other atom: polarisation and charge transfer inside an island,
        polarisation only between islands (their charges stay integer, so size consistency is kept).
        Borrowed and labelled: QEq (Rappe and Goddard 1991); environment-dependent chi from a network and Gaussian
        charges (4G-HDNNP, Ko et al. 2021); training chi and J from energies only through the variational QEq energy
        (Shaidu et al. 2024, npj Comput. Mater.); charge conserved per molecule with polarisation between molecules
        (fluctuating-charge force fields, Rick, Stuart and Berne 1994). This repo's part: the molecules are the islands
        of the message-passing graph, which gives exact size consistency with the island charge input. Both flags are
        absent from the config at their defaults.

        charge_bound / feedback_bound (q_max in e; None = off, the default): the raw output x of the final charge head
        (charge_bound) or of every charge-feedback head (feedback_bound) is replaced by q_max tanh(x / q_max) BEFORE the
        equilibration, so the island charge sums stay exact (the equilibration only adds f_i (Q - sum q_raw) / sum f).
        Why: the raw charge-head output diverges at compressed geometries (LiF at 0.7 A: Li -40 e, F -143 e from the
        head itself, not from the equilibration weights; Coulomb energies of 1e5 kcal/mol), and the feedback charges,
        which enter the network state, reach 3,142 e there. tanh is the identity to first order, so a trained model
        loaded with a bound at q_max = 2 e changes by +0.34% E / +0.05% F on val-Comp.
        Training with the bound lets the network adapt. Absent from
        the config when None, so old configs, checkpoints and hashes are unchanged. charge_bound is incompatible with qeq
        (there the charge head is an electronegativity, not a charge); feedback_bound needs charge_feedback.

        gated_stages (int, default 0): number of extra ACEStage blocks appended AFTER the `stages` base stages and
        before the readouts, each wrapped in a zero-initialised per-channel gated residual (ReZero-style; borrowed:
        Bachlechner et al. 2020, "ReZero is all you need"; the per-channel form and its use to deepen a trained model
        are this repo's adaptation):
            s <- s + alpha_s * (s_stage - s),   v <- v + alpha_v[:, None] * (v_stage - v),
        alpha_s [d_model] and alpha_v [vector_rank] start at 0. An ACEStage is NOT the identity even with its output
        layers at zero (its out_norm RMSNorm rescales s), hence the gate: at alpha = 0 the state passes through
        unchanged, so a model with gated_stages=g loaded with the weights of the same model without them (missing keys:
        `gated_stages_list.*`, `gate_s.*`, `gate_v.*` only) computes exactly the same energies and forces. Gate gradients
        are nonzero at start (they see the stage's output); the new stage's own weights get zero gradient until a gate
        opens. The new blocks have no stage readout and no charge feedback (feedback points and stage readouts of the
        base stages are unchanged); the final readout, charge head and NQE head read the gated state. Built last, so
        every other parameter keeps its seeded initial value. Absent from the config when 0, so old configs,
        checkpoints and hashes are unchanged.

        Growth options (all default off and absent from the config then; all new parameters are built after every
        other one, so a trained model loads with strict=False, missing keys = only the new ones, and computes exactly
        the same energies and forces; function-preserving growth: borrowed idea Net2Net, Chen et al. 2016; zero-gated
        residuals, ReZero):
        - grow_tensor_rank=K2 (A1): an additive l=2 branch per stage in NEW parameters (`stages_list.*.grow.*`): the
          same density T_ic = sum_j f_c(r_ij) g_c(s_j) Y2(r_ij), nu2_t = <U T, V T> and nu3 = (Pv)^T (W T) (Qv) as the
          tensor_rank code, fed through a zero-initialised Linear `inv_extra` into the PRE-activation of inv_mlp[0].
        - tensor_vector_update (A3, needs grow_tensor_rank): v_k += gamma_k(s) (A_k v_k), A_k = sum_c W_kc M(T_c),
          M(t) the 3x3 symmetric traceless matrix of the l=2 components (p^T M(t) q = t . l2_pair(p, q)), gamma zero
          (l=2 x l=1 -> l=1 in Cartesian form; borrowed idea: TensorNet's Cartesian matrix products, Simeon and De
          Fabritiis 2023).
        - edge_quadratic (A2, stages after the first): q_ik = norm_i sum_j h_k(r_ij) (r_ij . b_jk)^2 with the transport
          projection the stage already computes, through the same zero-initialised inv_extra.
        - null_probe_hidden=H: a per-atom MLP on cat[final state, v_inv] (the readout's inputs) with a zero last layer,
          added to the atomic energies: re-reads existing features only (the null baseline for the three above).

        island_context (IC; list of stage indices, "0,1,2" or None = off; needs charge_mode="island" or "split"): the whole-structure
        channel. After the listed stages (after readout and charge feedback) a per-island/window context vector, built by mean
        pooling and 4-head attention pooling of the atom states over the atom's island or smooth window together with the
        charge Q_I / 4 and the bounded size term n_I^(-1/3), is broadcast back to every atom through a gated MLP, and added to the
        scalar state (ic_dim, ic_heads, ic_hidden size it; models/island_context.py). Borrowed ideas, labelled: attention
        pooling (Li et al. 2016; Lee et al. 2019, Set Transformer), a non-local context channel for charge and spin
        (Unke et al. 2021, SpookyNet), zero-initialised residual output (ReZero). This repo's part: the pooling stays
        inside islands or smooth distance windows, so E(A + B far apart) = E(A) + E(B) is kept exactly. The output layer and the attention logits start
        at zero, so a model built with IC and loaded with the weights of the same model without it (missing keys:
        `island_ctx.*` only) computes bit-identical energies and forces. Built last (after the band), so every other
        parameter keeps its seeded initial value. Absent from the config when off, so old configs, checkpoints and hashes
        are unchanged.

        fb_input_norm (needs charge feedback): the inputs of the charge-feedback heads (fb_charge, fb_weight) are
        RMS-normalised first (`fb_norm.*`), so the SiLU units of the heads cannot all die together. NOT function-
        preserving, so for models trained from scratch only. Absent from the config when False.

        pair_radius (borrowed idea: covalent-radius-scaled distances, as in Grimme's D3 coordination number and GFN-FF;
        the learnable radii and their use as the only input of the Gaussian basis are this repo's adaptation): a learnable
        radius per element rho_Z (parameter `pair_log_radius` [num_elements], stored as log rho, initialised to
        log(ase.data.covalent_radii), 0 or NaN replaced by 0.76 A), and the distance fed ONLY to the Gaussian RBF becomes
            r_tilde_ij = r_ij * 1.52 / (rho_i + rho_j)
        (a "C-C-equivalent distance": 1.52 = 2 x rho_C, so C-C pairs are unchanged and a C-Cl bond looks like a C-C bond
        of the same relative length). The taper envelope, unit vectors, Coulomb term and island detection keep the TRUE
        r, so smoothness at the cutoff is preserved. Caveat: pairs of light atoms (rho_i + rho_j < 1.52) get r_tilde > r,
        so their features leave the basis range [0, r_cut] earlier (H-C beyond about 3.9 A, H-H beyond about 2.2 A give
        a near-zero RBF row). Built last of all (after the island context), so every other parameter keeps its seeded
        initial value. Absent from the config when False, so old configs, checkpoints and hashes are unchanged.

        pair_basis_hidden=H (default 0 = off): a learned, element-pair-conditioned
        nonlinear correction to the Gaussian basis that every radial filter reads. Per element a 16-d embedding e_Z
        (`pair_basis_emb`); per pair
            rbf'_ij = rbf_ij + MLP([rbf_ij, e_i * e_j, e_i + e_j])     (R + 32 -> H -> R, SiLU, last layer zero-init)
        and edge_ij = rbf'_ij * w_ij as before (the taper envelope inside w_ij keeps the cutoff smooth). The filters stay
        linear maps of this basis, but the basis now depends nonlinearly on r and on both elements (MACE/NequIP-style radial
        MLPs, borrowed, conditioned on the element pair here). Zero-init output: the model starts as the one without it.
        Shared by all stages (computed once per forward). Absent from the config when 0.

        pair_spline (borrowed idea: per-element-pair two-body terms, e.g. UF3, Xie et al. 2023;
        here coefficients on the model's own Gaussian basis): an additive two-body energy
            E_pair = 1/2 sum_{i != j} taper(r_ij) * c(Z_i, Z_j) . rbf(r_ij),   c = W (e_i * e_j)
        with a 32-d element embedding e_Z (`pair_spline_emb`) and W (R x 32, `pair_spline_out`) zero-initialised, so the
        model starts unchanged. Uses the TRUE distance (not pair_radius's scaled one). Not multiplied by output_scale.
        Absent from the config when False.

        msg_dim=m (default 0 = off = d_model): width of the scalar continuous-filter message. scalar_pre maps
        the atom state to m channels, the radial filters have m outputs, and the summed message is mapped back to d_model
        by `msg_up` (bias-free). The scalar message is ~3/4 of the per-pair work (R x d multiply-adds per pair), so a
        narrow message with a wider atom state moves capacity from pairs to per-atom dense layers (the transformer
        split: cheap mixing, wide MLP). Absent from the config when off, so old configs, checkpoints and hashes are
        unchanged.

        ttt_mix=k / vttv_rank=r / l3_rank=k3 (angular branches B1 / B2;
        maths in AngularBranch; forward_sparse only): per base stage an `ang` module with
          - B1 (k, r; no new per-edge work): tr(M(C_c T)^3) for k channel mixes of the l=2 density, and
            (P_c v)^T M(A_c T) M(B_c T) (Q_c v) at rank r (vttv_rank 0 with ttt_mix > 0 means r = k);
          - B2 (k3): an l=3 density (7 Cartesian STF components per channel, one extra radial filter block), with
            <U3 T3, V3 T3> and the l1 x l2 -> l3 coupling sum_abc T3_abc v_a M(T)_bc.
        Fed through one zero-initialised Linear `ang.out` into the pre-activation of inv_mlp[0], so a model built with
        them and loaded with the weights of the same model without them (strict=False; missing keys `stages_list.*.ang.*`
        only) computes the same energies, forces and stresses. Built last of all. Absent from the config when 0."""
        super().__init__()
        if charge_mode not in ("global", "island", "split"):
            raise ValueError(f"charge_mode must be 'global', 'island', or 'split', got {charge_mode!r}")
        if ic_mode not in ("island", "window"):
            raise ValueError(f"ic_mode must be 'island' or 'window', got {ic_mode!r}")
        self.charge_mode = charge_mode
        self.split_mode = charge_mode == "split"
        self.island_mode = charge_mode in ("island", "split")
        self.ic_mode = ic_mode
        self.ic_r_cut = float(ic_r_cut)
        self.ic_r_on = float(ic_r_on)
        self.ic_lr_dim = int(ic_lr_dim) if ic_lr_dim else (16 if self.ic_mode == "window" else 0)
        self.elec_lot_ids = tuple(int(x) for x in elec_lot_ids) if elec_lot_ids is not None else (6,)
        if self.island_mode:
            if not use_electrostatics:
                raise ValueError(f"charge_mode={charge_mode!r} needs use_electrostatics (the per-atom latent charges)")
            use_nqe, charge_feedback, use_charge_embedding = True, True, False
        self.config = {k: v for k, v in locals().items()
                       if k not in ("self", "__class__", "use_nqe", "charge_feedback", "feedback_points",
                                    "feedback_hidden", "charge_mode", "direct_force_head", "output_scale",
                                    "coulomb_range", "qeq", "qeq_j0", "charge_bound", "feedback_bound",
                                    "gated_stages", "grow_tensor_rank", "tensor_vector_update", "edge_quadratic",
                                    "null_probe_hidden", "band_energy", "band_version", "island_context", "ic_dim",
                                    "ic_heads", "ic_hidden", "ic_mode", "ic_r_cut", "ic_r_on", "ic_lr_dim",
                                    "dsf_alpha", "pbc_dsf_alpha", "elec_lot_ids", "fb_input_norm",
                                    "attention_mixer", "attention_heads",
                                    "pair_radius", "pair_basis_hidden", "pair_spline", "msg_dim", "ttt_mix",
                                    "vttv_rank", "l3_rank", "norm_scale", "env_pair_rank", "l4_rank",
                                    "c3_taper", "pair_radius_warp", "num_lots", "d_lot", "lot_stages",
                                    "lot_atom_input", "stage_readout", "elec_gate", "long_range", "ic_windows",
                                    "coulomb_sigma", "mesh_spacing", "mesh_order", "lot_energy_stages",
                                    "lot_embed_init_std", "vector_swiglu_hidden", "edge_heads",
                                    "zbl_core", "zbl_r_out_scale", "domain_readout")}
        self.config["elec_lot_ids"] = list(self.elec_lot_ids)
        if min(int(ttt_mix), int(vttv_rank), int(l3_rank), int(l4_rank)) < 0:
            raise ValueError("ttt_mix, vttv_rank, l3_rank and l4_rank must be >= 0")
        ttt_mix, l3_rank, l4_rank = int(ttt_mix), int(l3_rank), int(l4_rank)
        vttv_rank = int(vttv_rank) if vttv_rank else ttt_mix
        if ttt_mix:
            self.config["ttt_mix"] = ttt_mix
        if vttv_rank:
            self.config["vttv_rank"] = vttv_rank
        if l3_rank:
            self.config["l3_rank"] = l3_rank
        if norm_scale:
            self.config["norm_scale"] = True
        if env_pair_rank:
            self.config["env_pair_rank"] = int(env_pair_rank)
        if l4_rank:
            self.config["l4_rank"] = l4_rank
        self.c3_taper = bool(c3_taper)
        if self.c3_taper:
            self.config["c3_taper"] = True
        self.pair_radius_warp = bool(pair_radius_warp)
        if self.pair_radius_warp:
            if not pair_radius:
                raise ValueError("pair_radius_warp needs pair_radius=True")
            self.config["pair_radius_warp"] = True
        self.num_lots = int(num_lots)
        self.d_lot = int(d_lot)
        if self.num_lots > 0:
            self.config.update(num_lots=self.num_lots, d_lot=self.d_lot)
        # Late LoT conditioning: the token reaches only the SwiGLUs of
        # `lot_stages` and the readout, so the earlier stages are one LoT-agnostic trunk shared by every dataset.
        # None = every stage (legacy). lot_atom_input=False drops the token from the initial atom state.
        self.lot_stages = tuple(range(stages)) if lot_stages is None else tuple(sorted(int(x) for x in lot_stages))
        if any(not 0 <= x < stages for x in self.lot_stages):
            raise ValueError(f"lot_stages {self.lot_stages} outside 0..{stages - 1}")
        self.lot_atom_input = bool(lot_atom_input)
        self._lot_slot = {st: i for i, st in enumerate(self.lot_stages)}
        if lot_stages is not None:
            self.config["lot_stages"] = list(self.lot_stages)
        if not self.lot_atom_input:
            self.config["lot_atom_input"] = False
        # lot_energy_stages: the token conditions these stages' SwiGLUs on an energy-only branch. The stage
        # state that flows on (charge feedback, IC, the next stages, the charge / NQE heads, so charges, dipole and
        # Coulomb) is token-free; the token's correction delta = SwiGLU(s; b_L) - SwiGLU(s) is summed over these stages
        # and added to the scalar input of the energy readout only. Absent from the config when empty.
        self.lot_energy_stages = tuple(sorted(int(x) for x in lot_energy_stages))
        if any(not 0 <= x < stages for x in self.lot_energy_stages):
            raise ValueError(f"lot_energy_stages {self.lot_energy_stages} outside 0..{stages - 1}")
        if set(self.lot_energy_stages) & set(self.lot_stages):
            raise ValueError(f"lot_energy_stages {self.lot_energy_stages} overlap lot_stages {self.lot_stages}")
        if self.lot_energy_stages and num_lots <= 0:
            raise ValueError("lot_energy_stages needs num_lots > 0")
        self._lot_e_slot = {st: i for i, st in enumerate(self.lot_energy_stages)}
        if self.lot_energy_stages:
            self.config["lot_energy_stages"] = list(self.lot_energy_stages)
        self.lot_embed_init_std = float(lot_embed_init_std)
        if self.lot_embed_init_std != 0.02:
            self.config["lot_embed_init_std"] = self.lot_embed_init_std
        self.zbl_core = bool(zbl_core)
        self._repulsion_variant = "zbl"  # which core is used when zbl_core is on; see the repulsion_mode property
        self.zbl_r_out_scale = float(zbl_r_out_scale)
        if self.zbl_r_out_scale <= 0.0:
            raise ValueError(f"zbl_r_out_scale must be > 0, got {zbl_r_out_scale}")
        if self.zbl_core:
            self.config["zbl_core"] = True
            if self.zbl_r_out_scale != 0.6:
                self.config["zbl_r_out_scale"] = self.zbl_r_out_scale
        # stage_readout=False: one readout on the final state. The per-stage linear readouts carry large energies that
        # cancel, and the residue drifts with the trunk at peak LR.
        self.stage_readout = bool(stage_readout)
        if not self.stage_readout:
            self.config["stage_readout"] = False
        # elec_gate: "lot" switches charges/spin/IC/Coulomb on by token (lot_id in elec_lot_ids); "system" switches them
        # on for every non-periodic structure, whatever the token (plus PBC structures of an elec LoT or with an explicit
        # use_elec_context), so one molecule runs the same physics under every token.
        if elec_gate not in ("lot", "system"):
            raise ValueError(f"elec_gate must be 'lot' or 'system', got {elec_gate!r}")
        self.elec_gate = elec_gate
        if elec_gate != "lot":
            self.config["elec_gate"] = elec_gate
        # long_range="mesh": mesh-native definitions of every long-range
        # operator, so a particle mesh evaluates them in O(N log N) with no long-range neighbour list:
        #   - IC pooling over ic_windows (C3 windows, the lr_dim channels split evenly over them; multi-scale context);
        #   - the charge-equilibration window = the widest IC window (was a 17 -> 18 A step, too sharp for a mesh);
        #   - Coulomb K(r) = erf(r / sigma) / r + K_sr(r), K_sr = (1 / r_s - erf(r / sigma) / r) * taper(r_on, r_cut) on the
        #     short-range edges: exactly 1 / r_s below r_on, a true 1 / r tail (vacuum) and Ewald under PBC.
        # Backends: exact sums over explicit lr edges (training molecules, tests) or the mesh (no lr edges given, and
        # always under PBC). "edges" keeps the legacy definitions.
        if long_range not in ("edges", "mesh"):
            raise ValueError(f"long_range must be 'edges' or 'mesh', got {long_range!r}")
        self.long_range = long_range
        self.mesh_def = long_range == "mesh"
        wins = [tuple(float(v) for v in w) for w in ic_windows] if ic_windows else [(float(ic_r_on), float(ic_r_cut))]
        self.ic_windows = wins
        self.coulomb_sigma, self.mesh_spacing, self.mesh_order = float(coulomb_sigma), float(mesh_spacing), int(mesh_order)
        if self.mesh_def:
            if ic_mode != "window":
                raise ValueError("long_range='mesh' needs ic_mode='window'")
            self.config.update(long_range="mesh", ic_windows=[list(w) for w in wins], coulomb_sigma=self.coulomb_sigma,
                               mesh_spacing=self.mesh_spacing, mesh_order=self.mesh_order)
            self._ic_kernels = [window_kernel(a, b) for a, b in wins]
            self._widest = max(range(len(wins)), key=lambda i: wins[i][1])
            self._coul_lr = erf_coulomb_kernel(self.coulomb_sigma)
            self._coul_open: dict[float, object] = {}
        if msg_dim and int(msg_dim) != d_model:
            self.config["msg_dim"] = int(msg_dim)
        msg_dim = self.config.get("msg_dim", 0)
        if pair_radius:
            self.config["pair_radius"] = True
        if pair_basis_hidden < 0:
            raise ValueError("pair_basis_hidden must be >= 0")
        if pair_basis_hidden:
            self.config["pair_basis_hidden"] = int(pair_basis_hidden)
        if pair_spline:
            self.config["pair_spline"] = True
        if attention_mixer != "none":
            self.config.update(attention_mixer=attention_mixer, attention_heads=int(attention_heads))
        ic_points = self._parse_island_context(island_context, stages)
        if ic_points:
            if not self.island_mode:
                raise ValueError("island_context needs charge_mode='island' or 'split'")
            self.config.update(island_context=ic_points, ic_dim=int(ic_dim), ic_heads=int(ic_heads),
                               ic_hidden=int(ic_hidden))
            if self.ic_mode != "island":
                self.config.update(
                    ic_mode=self.ic_mode,
                    ic_r_cut=self.ic_r_cut,
                    ic_r_on=self.ic_r_on,
                    ic_lr_dim=self.ic_lr_dim,
                )
        if fb_input_norm:
            if not charge_feedback:
                raise ValueError("fb_input_norm needs charge_feedback (charge_mode='island' or 'split' switches it on)")
            self.config["fb_input_norm"] = True
        if gated_stages < 0:
            raise ValueError(f"gated_stages must be >= 0, got {gated_stages}")
        if gated_stages:
            self.config["gated_stages"] = int(gated_stages)
        if grow_tensor_rank < 0 or null_probe_hidden < 0:
            raise ValueError("grow_tensor_rank and null_probe_hidden must be >= 0")
        if band_energy:
            self.config["band_energy"] = True
            if band_version != 1:
                self.config["band_version"] = int(band_version)  # absent at 1: v1 configs/checkpoints unchanged
        if tensor_vector_update and not grow_tensor_rank:
            raise ValueError("tensor_vector_update needs grow_tensor_rank > 0 (it couples the grown l=2 density)")
        if grow_tensor_rank:
            self.config["grow_tensor_rank"] = int(grow_tensor_rank)
        if tensor_vector_update:
            self.config["tensor_vector_update"] = True
        if edge_quadratic:
            self.config["edge_quadratic"] = True
        if null_probe_hidden:
            self.config["null_probe_hidden"] = int(null_probe_hidden)
        if self.island_mode:
            self.config["charge_mode"] = charge_mode
        if direct_force_head:
            self.config["direct_force_head"] = True
        # output_scale: the network's atomic energies (and direct forces) are multiplied by this constant, so the
        # network works in units of ~1 when it is set to the training force RMS (fairchem's normalizer_rmsd, MACE's
        # scale). The Coulomb term is not scaled. Absent from the config at 1.0.
        self.output_scale = float(output_scale)
        if self.output_scale != 1.0:
            self.config["output_scale"] = self.output_scale
        if coulomb_range not in ("taper", "full", "dsf"):
            raise ValueError(f"coulomb_range must be 'taper', 'full', or 'dsf', got {coulomb_range!r}")
        if qeq:
            if not use_electrostatics:
                raise ValueError("qeq needs use_electrostatics (it reuses the charge head as the electronegativity)")
            coulomb_range = "full"
            self.config.update(qeq=True, qeq_j0=float(qeq_j0))
        self.coulomb_range = coulomb_range
        self.full_coulomb = coulomb_range == "full"
        self.dsf_coulomb = coulomb_range == "dsf"
        self.dsf_alpha = float(dsf_alpha)
        self.pbc_dsf_alpha = float(pbc_dsf_alpha)
        if coulomb_range != "taper":
            self.config["coulomb_range"] = coulomb_range
            if self.dsf_coulomb:
                self.config["dsf_alpha"] = self.dsf_alpha
                self.config["pbc_dsf_alpha"] = self.pbc_dsf_alpha
        if use_nqe:
            self.config["use_nqe"] = True
        self.charge_bound = None if charge_bound is None else float(charge_bound)
        self.feedback_bound = None if feedback_bound is None else float(feedback_bound)
        if self.charge_bound is not None:
            if not use_electrostatics:
                raise ValueError("charge_bound needs use_electrostatics (it bounds the latent charge head)")
            if qeq:
                raise ValueError("charge_bound is incompatible with qeq (the charge head is an electronegativity there)")
            self.config["charge_bound"] = self.charge_bound
        if self.feedback_bound is not None:
            if not charge_feedback:
                raise ValueError("feedback_bound needs charge_feedback (charge_mode='island' switches it on)")
            self.config["feedback_bound"] = self.feedback_bound
        if self.charge_bound is not None and self.charge_bound <= 0 or self.feedback_bound is not None and self.feedback_bound <= 0:
            raise ValueError("charge_bound / feedback_bound must be positive (e)")
        if charge_feedback:
            if not 1 <= feedback_points <= stages - 1:
                raise ValueError(f"feedback_points must be in [1, stages - 1] = [1, {stages - 1}], got {feedback_points}")
            self.config.update(charge_feedback=True, feedback_points=feedback_points, feedback_hidden=feedback_hidden)
        self.d_model, self.vector_rank, self.tensor_rank = d_model, vector_rank, tensor_rank
        self.r_cut, self.r_on = float(r_cut), float(r_on)
        swiglu_hidden = max(16, int(round(d_model * swiglu_ratio)))

        self.embedding = nn.Embedding(num_elements, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.charge_embedding = nn.Sequential(
            nn.Linear(1, charge_hidden), nn.SiLU(), nn.Linear(charge_hidden, d_model)
        ) if use_charge_embedding else None
        # Spin multiplicity (2S+1, OMol25 convention) as a table broadcast to every atom, as the charge embedding is
        # (and as eSEN/UMA do). Index 0 = unknown/absent spin; multiplicities above max_spin share the last row.
        self.max_spin = int(max_spin)
        self.spin_embedding = nn.Embedding(self.max_spin + 1, d_model) if use_spin_embedding else None
        if self.spin_embedding is not None:
            nn.init.normal_(self.spin_embedding.weight, std=0.02)
        if small_global_init and self.charge_embedding is not None:
            # Default init makes the charge-embedding output (identical for every atom of a molecule) 8-9x the element
            # embedding (RMS ~0.18 vs 0.02), so the initial state is nearly element-blind
            # (found in a mechanism audit). Scale it to the element embedding's size.
            with torch.no_grad():
                self.charge_embedding[-1].weight.mul_(0.1)
                self.charge_embedding[-1].bias.zero_()

        self.rbf = GaussianRBF(num_rbf=num_rbf, r_cut=self.r_cut, envelope=False)
        self.envelope = TaperCutoff(self.r_on, self.r_cut, c3=self.c3_taper)
        self.edge_weights = ChemicalEdgeWeights(d_model, smooth_degree=self.c3_taper)

        self.stages_list = nn.ModuleList([
            ACEStage(d_model, vector_rank, tensor_rank, num_rbf, swiglu_hidden, first=(l == 0),
                     cross_channel=cross_channel, msg_dim=msg_dim) for l in range(stages)
        ])

        if attention_mixer == "linear":
            from models.linear_attention import LinearGlobalAttention
            self.linear_mixers = nn.ModuleList([
                LinearGlobalAttention(d_model=d_model, heads=attention_heads, attn_width=d_model, use_qk_norm=True)
                for _ in range(stages)
            ])
        else:
            self.linear_mixers = None

        # Readouts: linear per intermediate stage (deep supervision path), MLP on the last stage + readout invariants
        self.stage_readouts = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(stages - 1)] if stage_readout else [])
        for lin in self.stage_readouts:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.read_u = nn.Linear(vector_rank, vector_rank, bias=False) if cross_channel else None
        self.read_v = nn.Linear(vector_rank, vector_rank, bias=False) if cross_channel else None
        self.readout = nn.Sequential(
            nn.Linear(d_model + vector_rank, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.normal_(self.readout[-1].weight, std=0.02)
        nn.init.zeros_(self.readout[-1].bias)
        # direct_force_head (borrowed idea: eSEN/UMA's direct-force pretraining, Fu et al. 2025, here a minimal head):
        # F_i = sum_k a_k(s_i) v_ik, scalars of the last stage gating its vector channels, so equivariant but not a
        # gradient. Used only when forward(direct_forces=True); absent from the config when False.
        self.force_head = nn.Linear(d_model, vector_rank) if direct_force_head else None

        self.use_electrostatics = use_electrostatics
        if use_electrostatics:
            self.coulomb_k = 332.06371  # kcal A / (mol e^2)
            self.coulomb_gamma = 1.2  # A, screening
            self.taper_elec = TaperCutoff(r_on=r_on_elec, r_cut=r_cut_elec, c3=True)
            self.charge_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1))
            # Small random, not zero: the Coulomb energy is quadratic in q, so zero-initialised charges get no gradient
            # from neutral molecules (MenaMinimal's charge head trained only through charged ones).
            nn.init.normal_(self.charge_head[-1].weight, std=0.01)
            nn.init.zeros_(self.charge_head[-1].bias)
        else:
            self.charge_head = None
        if use_nqe and not use_electrostatics:
            raise ValueError("use_nqe needs use_electrostatics (it reshapes the latent charges)")
        self.nqe_head = None
        if use_nqe:
            self.nqe_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.nqe_head.weight)
            nn.init.constant_(self.nqe_head.bias, math.log(math.e - 1.0))  # softplus(bias) = 1
        # Charge feedback (built last, so every other parameter keeps its seeded initial value)
        self.feedback_points = feedback_points if charge_feedback else 0
        self.fb_charge = nn.ModuleList()
        self.fb_weight = nn.ModuleList()
        self.fb_embed = nn.ModuleList()
        for _ in range(self.feedback_points):
            head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1))
            nn.init.normal_(head[-1].weight, std=0.01)  # small random, as charge_head: q^2-type gradients need q != 0
            nn.init.zeros_(head[-1].bias)
            self.fb_charge.append(head)
            wt = nn.Linear(d_model, 1)
            nn.init.zeros_(wt.weight)
            nn.init.constant_(wt.bias, math.log(math.e - 1.0))  # softplus(bias) = 1: uniform sharing at start
            self.fb_weight.append(wt)
            emb = nn.Sequential(nn.Linear(1, feedback_hidden), nn.SiLU(), nn.Linear(feedback_hidden, d_model))
            nn.init.zeros_(emb[-1].weight)  # zero output layer: the model starts as the one without feedback
            nn.init.zeros_(emb[-1].bias)
            self.fb_embed.append(emb)
        # QEq heads (built after everything else, so every other parameter keeps its seeded initial value)
        self.qeq_head, self.qeq_log_sigma, self.qeq_j0 = None, None, float(qeq_j0)
        if qeq:
            self.qeq_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.qeq_head.weight)
            nn.init.constant_(self.qeq_head.bias, math.log(math.e - 1.0))  # softplus = 1: J = self-energy + j0
            self.qeq_log_sigma = nn.Parameter(torch.zeros(num_elements))  # sigma_Z = 1 A at start
        # Gated extra stages (built after everything else, so every other parameter keeps its seeded initial value)
        self.gated_stages_list = nn.ModuleList([
            ACEStage(d_model, vector_rank, tensor_rank, num_rbf, swiglu_hidden, first=False, cross_channel=cross_channel,
                     msg_dim=msg_dim)
            for _ in range(gated_stages)])
        self.gate_s = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(gated_stages)])
        self.gate_v = nn.ParameterList([nn.Parameter(torch.zeros(vector_rank)) for _ in range(gated_stages)])
        # Growth modules and the null probe: built last of all (existing parameters keep their seeded values)
        if grow_tensor_rank or edge_quadratic:
            for stage in self.stages_list:
                stage.add_growth(d_model, num_rbf, int(grow_tensor_rank), bool(edge_quadratic), bool(tensor_vector_update))
        self.null_probe = None
        if null_probe_hidden:
            self.null_probe = nn.Sequential(nn.Linear(d_model + vector_rank, null_probe_hidden), nn.SiLU(),
                                            nn.Linear(null_probe_hidden, 1))
            nn.init.zeros_(self.null_probe[-1].weight)
            nn.init.zeros_(self.null_probe[-1].bias)
        self.band = None  # electron-counting band free energy (models/band_energy.py): lam = 0 at init, built last of all
        if band_energy:
            from models.band_energy import BandEnergy
            self.band = BandEnergy(d_model, num_rbf, version=band_version)
        # Island context and the feedback input norm: built after everything else, so every other parameter keeps its
        # seeded initial value. An empty ModuleList adds no state_dict keys.
        self.ic_index = {p: k for k, p in enumerate(ic_points)}  # plain dict: stage index -> module index
        self.island_ctx = nn.ModuleList([
            IslandContext(d_model, ic_dim, ic_heads, ic_hidden, lr_dim=self.ic_lr_dim, mode=self.ic_mode)
            for _ in ic_points
        ])
        self.fb_norm = nn.ModuleList([nn.RMSNorm(d_model, eps=1e-6) for _ in range(self.feedback_points)]) \
            if fb_input_norm else None
        # Per-element covalent radii for the Gaussian basis (pair_radius): built last of all
        self.pair_log_radius = None
        if pair_radius:
            from ase.data import covalent_radii
            rad = torch.tensor(covalent_radii[:num_elements], dtype=torch.get_default_dtype())
            if rad.numel() < num_elements:
                rad = torch.cat([rad, rad.new_full((num_elements - rad.numel(),), 0.76)])
            rad = torch.where(torch.isfinite(rad) & (rad > 0), rad, torch.full_like(rad, 0.76))
            self.pair_log_radius = nn.Parameter(rad.log())
        # Element-pair-conditioned basis (pair_basis_hidden) and two-body term (pair_spline): built last of all
        self.pair_basis_emb = self.pair_basis_mlp = None
        if pair_basis_hidden:
            self.pair_basis_emb = nn.Parameter(torch.randn(num_elements, 16))
            self.pair_basis_mlp = nn.Sequential(nn.Linear(num_rbf + 32, pair_basis_hidden), nn.SiLU(),
                                                nn.Linear(pair_basis_hidden, num_rbf))
            nn.init.zeros_(self.pair_basis_mlp[-1].weight)
            nn.init.zeros_(self.pair_basis_mlp[-1].bias)
        self.pair_spline_emb = self.pair_spline_out = None
        if pair_spline:
            self.pair_spline_emb = nn.Parameter(torch.randn(num_elements, 32))
            self.pair_spline_out = nn.Linear(32, num_rbf, bias=False)
            nn.init.zeros_(self.pair_spline_out.weight)
        # Angular branches B1/B2 (ttt_mix, vttv_rank, l3_rank): built last of all
        self.use_l3 = bool(l3_rank)
        if ttt_mix or vttv_rank or l3_rank:
            for stage in self.stages_list:
                stage.add_angular(d_model, num_rbf, ttt_mix, vttv_rank, l3_rank)
        if norm_scale:  # log-RMS pass-through around each stage's out_norm (ACEStage._post_norm), zero-initialised
            for stage in self.stages_list:
                stage.norm_scale = nn.Linear(1, d_model)
                nn.init.zeros_(stage.norm_scale.weight)
                nn.init.zeros_(stage.norm_scale.bias)
        # Environment-dependent direct pair energy (env_pair_rank > 0, forward_sparse only): after the last stage,
        #   E_pair = 1/2 sum_{i != j} chi(r_ij) sum_k c_k(h_i, h_j) phi_k(r_ij),
        #   c(h_i, h_j) = W [p_i * p_j, p_i + p_j],  p = P rms_norm(h)   (symmetric in i, j; W zero-initialised).
        # Each neighbour pair gets its own energy-vs-distance curve on the 32-Gaussian basis, with coefficients set by
        # both atoms' learned environments, added straight to the energy (no RMSNorm, no deep readout in between), so the
        # pair's stiffness (curvature) and anharmonicity are first-order outputs. MPtrj models fit the stiffness last
        # (0.75-0.80 of DFT on an MPtrj-trained model's own training materials).
        self.env_pair_proj = self.env_pair_out = None
        if env_pair_rank:
            self.env_pair_proj = nn.Linear(d_model, env_pair_rank, bias=False)
            self.env_pair_out = nn.Linear(2 * env_pair_rank, num_rbf)
            nn.init.zeros_(self.env_pair_out.weight)
            nn.init.zeros_(self.env_pair_out.bias)
        # B3 l=4 hexadecapole branch (l4_rank): built last of all so every earlier parameter keeps its seeded value.
        self.use_l4 = bool(l4_rank)
        if l4_rank:
            for stage in self.stages_list:
                stage.add_l4(d_model, num_rbf, l4_rank)
        # Level-of-Theory (LoT) conditioning (num_lots > 0): built last of all with zero-init projections
        self.lot_embedding = None
        self.lot_atom_proj = None
        self.lot_swiglu_projs = None
        self.lot_readout_proj = None
        self.lot_readout_bias = None
        if self.num_lots > 0:
            self.lot_embedding = nn.Embedding(self.num_lots, self.d_lot)
            nn.init.normal_(self.lot_embedding.weight, std=self.lot_embed_init_std)
            if self.lot_atom_input:
                self.lot_atom_proj = nn.Linear(self.d_lot, d_model, bias=False)
                nn.init.zeros_(self.lot_atom_proj.weight)
            # one projection per conditioned stage, in lot_stages order (legacy: one per stage, index = stage)
            self.lot_swiglu_projs = nn.ModuleList([
                nn.Linear(self.d_lot, 2 * swiglu_hidden, bias=False) for _ in self.lot_stages
            ])
            for proj in self.lot_swiglu_projs:
                nn.init.zeros_(proj.weight)
            self.lot_readout_proj = nn.Linear(self.d_lot, d_model + vector_rank, bias=False)
            nn.init.zeros_(self.lot_readout_proj.weight)
            self.lot_readout_bias = nn.Embedding(self.num_lots, 1)
            nn.init.zeros_(self.lot_readout_bias.weight)
        # B1 pairwise bond transfer heads and B5 spin weight head (built last so earlier seeded RNGs are untouched;
        # named with charge_head_* / fb_charge_* / nqe_head_* so build_optimizers routes them to AdamW, never Muon)
        self.charge_head_transfer: nn.Linear | None = None
        self.fb_charge_transfer: nn.ModuleList | None = None
        self.nqe_head_spin: nn.Linear | None = None
        self.use_spin_radical_gate: bool = True
        if self.split_mode:
            if use_electrostatics:
                self.charge_head_transfer = nn.Linear(d_model, 1, bias=False)
                nn.init.normal_(self.charge_head_transfer.weight, std=0.01)
            if self.feedback_points > 0:
                self.fb_charge_transfer = nn.ModuleList([
                    nn.Linear(d_model, 1, bias=False) for _ in range(self.feedback_points)
                ])
                for th in self.fb_charge_transfer:
                    nn.init.normal_(th.weight, std=0.01)
            if self.spin_embedding is not None:
                self.nqe_head_spin = nn.Linear(d_model, 1)
                nn.init.zeros_(self.nqe_head_spin.weight)
                nn.init.constant_(self.nqe_head_spin.bias, math.log(math.e - 1.0))
        # Energy-only LoT projections (lot_energy_stages): built last of all, zero-init, so every other parameter keeps
        # its seeded value and the model starts equal to its token-free-stage counterpart
        self.lot_energy_projs: nn.ModuleList | None = None
        if self.lot_energy_stages:
            self.lot_energy_projs = nn.ModuleList([
                nn.Linear(self.d_lot, 2 * swiglu_hidden, bias=False) for _ in self.lot_energy_stages
            ])
            for proj in self.lot_energy_projs:
                nn.init.zeros_(proj.weight)
        if int(vector_swiglu_hidden) < 0:
            raise ValueError("vector_swiglu_hidden must be >= 0")
        self.vector_swiglu_hidden = int(vector_swiglu_hidden)
        if self.vector_swiglu_hidden:
            if not vector_rank:
                raise ValueError("vector_swiglu_hidden needs vector_rank > 0")
            self.config["vector_swiglu_hidden"] = self.vector_swiglu_hidden
            for stage in self.stages_list:
                stage.add_vswiglu(d_model, self.vector_swiglu_hidden)
        # Multi-head edge factor (edge_heads > 0): built after earlier seeded RNGs
        self.edge_heads = int(edge_heads)
        if self.edge_heads < 0:
            raise ValueError(f"edge_heads must be >= 0, got {edge_heads}")
        if self.edge_heads:
            self.config["edge_heads"] = self.edge_heads
            for stage in self.stages_list:
                stage.add_edge_heads(d_model, num_rbf, self.edge_heads)
        # Domain readout heads (domain_readout=True): molecular copies of stage_readouts and readout, built last
        # so earlier parameters keep their seeded values. Periodic forward uses stage_readouts / readout; molecular
        # forward (non-PBC or molecular LoT) uses stage_readouts_mol / readout_mol.
        self.domain_readout = bool(domain_readout)
        self.stage_readouts_mol: nn.ModuleList | None = None
        self.readout_mol: nn.Sequential | None = None
        if self.domain_readout:
            self.config["domain_readout"] = True
            self.stage_readouts_mol = nn.ModuleList(
                [nn.Linear(d_model, 1) for _ in range(stages - 1)] if stage_readout else []
            )
            for lin in self.stage_readouts_mol:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            self.readout_mol = nn.Sequential(
                nn.Linear(d_model + vector_rank, d_model), nn.SiLU(),
                nn.Linear(d_model, d_model // 2), nn.SiLU(),
                nn.Linear(d_model // 2, 1),
            )
            nn.init.normal_(self.readout_mol[-1].weight, std=0.02)
            nn.init.zeros_(self.readout_mol[-1].bias)
        self._diag: dict[tuple[int, torch.device], Tensor] = {}
        self.fast_ops: frozenset = frozenset()
        self._is_baked: bool = False
        self._baked_pair_radius: Tensor | None = None
        self._baked_pr_version: int | None = None

    @torch.no_grad()
    def bake_for_inference(self) -> "MenaACE":
        self._is_baked = True
        if hasattr(self.embedding, "bake_for_inference"):
            self.embedding.bake_for_inference()
        if self.pair_log_radius is not None:
            self._baked_pair_radius = self.pair_log_radius.exp().detach().contiguous()
            self._baked_pr_version = self.pair_log_radius._version
        else:
            self._baked_pair_radius = None
            self._baked_pr_version = None
        for st in [*self.stages_list, *self.gated_stages_list]:
            st.bake_for_inference()
        return self

    def unbake_for_inference(self) -> "MenaACE":
        self._is_baked = False
        if hasattr(self.embedding, "unbake_for_inference"):
            self.embedding.unbake_for_inference()
        self._baked_pair_radius = None
        self._baked_pr_version = None
        for st in [*self.stages_list, *self.gated_stages_list]:
            st.unbake_for_inference()
        return self

    def _apply(self, fn, recurse: bool = True):
        was_baked = self._is_baked
        self.unbake_for_inference()
        super()._apply(fn, recurse=recurse)
        if was_baked and not self.training:
            self.bake_for_inference()
        return self

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        was_baked = self._is_baked
        self.unbake_for_inference()
        res = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if was_baked and not self.training:
            self.bake_for_inference()
        return res

    def train(self, mode: bool = True) -> "MenaACE":
        super().train(mode)
        if mode:
            self.unbake_for_inference()
        return self

    @property
    def repulsion_mode(self) -> str:
        """Effective core: "none" when zbl_core is off, else the selected variant ("zbl" by default).

        Derived, never stored, so code that flips `zbl_core` directly (tests, grafts) cannot desynchronise it."""
        return getattr(self, "_repulsion_variant", "zbl") if self.zbl_core else "none"

    def set_repulsion_mode(self, mode: str) -> None:
        """Configure short-range nuclear repulsion and descriptor distance clamping mode."""
        m = str(mode).lower().strip()
        if m in ("none", "off", "false"):
            self.zbl_core = False
        elif m == "zbl":
            self._repulsion_variant = "zbl"
            self.zbl_core = True
        elif m in ("nlh_c2_shift", "c2_shift", "c2_shift_rmin"):
            self._repulsion_variant = "nlh_c2_shift"
            self.zbl_core = True
        elif m == "nlh_c3":
            self._repulsion_variant = "nlh_c3"
            self.zbl_core = True
        elif m in ("nlh_so3lr", "so3lr"):
            self._repulsion_variant = "nlh_so3lr"
            self.zbl_core = True
        else:
            raise ValueError(
                f"Unknown repulsion mode {mode!r}; expected 'nlh_c2_shift', 'zbl', 'nlh_c3', 'nlh_so3lr', or 'none'"
            )

    def _clamp_dense(self, z: Tensor, dist: Tensor, pmf: Tensor) -> Tensor:
        rm = getattr(self, "repulsion_mode", "zbl" if self.zbl_core else "none")
        if rm == "zbl":
            return zbl_clamp_dense(z, dist, pmf, scale=self.zbl_r_out_scale)
        if rm == "nlh_c3":
            return nlh_clamp_dense(z, dist, pmf, scale=self.zbl_r_out_scale, mode="c3")
        return nlh_clamp_dense(z, dist, pmf, scale=self.zbl_r_out_scale, mode="tanh")

    def _clamp_sparse(self, z: Tensor, dist: Tensor, recv: Tensor, send: Tensor) -> Tensor:
        rm = getattr(self, "repulsion_mode", "zbl" if self.zbl_core else "none")
        if rm == "zbl":
            return zbl_clamp_sparse(z, dist, recv, send, scale=self.zbl_r_out_scale)
        if rm == "nlh_c3":
            return nlh_clamp_sparse(z, dist, recv, send, scale=self.zbl_r_out_scale, mode="c3")
        return nlh_clamp_sparse(z, dist, recv, send, scale=self.zbl_r_out_scale, mode="tanh")

    def _repulsion_energy_dense(self, z: Tensor, dist_raw: Tensor, pmf: Tensor) -> Tensor:
        rm = getattr(self, "repulsion_mode", "zbl" if self.zbl_core else "none")
        if rm == "zbl":
            return zbl_energy_dense(z, dist_raw, pmf, scale=self.zbl_r_out_scale)
        if rm == "nlh_c3":
            return nlh_energy_dense(z, dist_raw, pmf, scale=self.zbl_r_out_scale, mode="c3")
        if rm == "nlh_so3lr":
            return nlh_energy_dense(z, dist_raw, pmf, scale=self.zbl_r_out_scale, mode="so3lr")
        return nlh_energy_dense(z, dist_raw, pmf, scale=self.zbl_r_out_scale, mode="c2_shift_rmin")

    def _repulsion_energy_sparse(
        self, z: Tensor, dist_raw: Tensor, recv: Tensor, send: Tensor, batch: Tensor, n_graphs: int
    ) -> Tensor:
        rm = getattr(self, "repulsion_mode", "zbl" if self.zbl_core else "none")
        if rm == "zbl":
            return zbl_energy_sparse(z, dist_raw, recv, send, batch, n_graphs, scale=self.zbl_r_out_scale)
        if rm == "nlh_c3":
            return nlh_energy_sparse(z, dist_raw, recv, send, batch, n_graphs, scale=self.zbl_r_out_scale, mode="c3")
        if rm == "nlh_so3lr":
            return nlh_energy_sparse(z, dist_raw, recv, send, batch, n_graphs, scale=self.zbl_r_out_scale, mode="so3lr")
        return nlh_energy_sparse(
            z, dist_raw, recv, send, batch, n_graphs, scale=self.zbl_r_out_scale, mode="c2_shift_rmin"
        )

    @torch.no_grad()
    def copy_domain_readout_from_base(self) -> None:
        """Copy-initialise readout_mol and stage_readouts_mol from readout and stage_readouts."""
        if not self.domain_readout:
            return
        if self.readout_mol is not None:
            self.readout_mol.load_state_dict(self.readout.state_dict())
        if self.stage_readouts_mol is not None:
            self.stage_readouts_mol.load_state_dict(self.stage_readouts.state_dict())

    def _is_mol_domain(
        self,
        pbc: bool = False,
        lot_id: Tensor | int | None = None,
        use_elec_context: bool | None = None,
    ) -> bool:
        """True when domain_readout=True and this call routes to readout_mol / stage_readouts_mol:
        any non-PBC call (forward, forward_dense_as_sparse, forward_molecule, forward_sparse(pbc=False)),
        or a PBC call whose LoT is molecular (lot_id in elec_lot_ids / {6, 10, 11}, or use_elec_context=True)."""
        if not self.domain_readout:
            return False
        if not pbc:
            return True
        mol_lots = frozenset(self.elec_lot_ids) | {6, 10, 11}
        if isinstance(lot_id, int) and not isinstance(lot_id, bool):
            return int(lot_id) in mol_lots
        if use_elec_context is not None:
            return bool(use_elec_context)
        if lot_id is None:
            return False
        if torch.is_tensor(lot_id):
            if torch._subclasses.fake_tensor.is_fake(lot_id) or torch.compiler.is_compiling():
                return False
            if lot_id.numel() > 0:
                return any(int(x) in mol_lots for x in lot_id.detach().reshape(-1).tolist())
        return False

    @property
    def transfer_head(self) -> nn.Linear | None:
        return self.charge_head_transfer

    @property
    def fb_transfer(self) -> nn.ModuleList | None:
        return self.fb_charge_transfer

    @property
    def spin_weight_head(self) -> nn.Linear | None:
        return self.nqe_head_spin

    @torch.no_grad()
    def graft_spin_into_lot(
        self,
        spin_weight: Tensor,
        mp_pbe_id: int = 0,
        mp_pbeu_id: int = 1,
        zero_spin_rows: bool = True,
        copy_init_other_lots: bool = True,
        pbeu_like_ids: Sequence[int] = (3,),
    ) -> None:
        """Map v2/v4's spin_embedding rows 1 (u=0, PBE) and 2 (u=1, PBE+U) into lot_embedding + lot_atom_proj
        with exact IEEE-754 float32 identity for mp_pbe and mp_pbeU.
        When `copy_init_other_lots=True`, LoT tokens k >= 2 are copy-initialised from mp_pbe ([1, 0, ...]) or
        mp_pbeU ([0, 1, ...] for k in `pbeu_like_ids`, e.g. omat_pbeU=3) while preserving their small random
        symmetry-breaking coordinates in columns 2:, preventing the 5.8x uninitialised force spike on OMat24/MatPES/OMol25.
        When `zero_spin_rows=True`, rows 1 and 2 of `self.spin_embedding` are zeroed out afterwards so singlet/doublet
        molecules do not inherit the PBE Hubbard-U vectors."""
        if self.lot_embedding is None or self.lot_atom_proj is None:
            raise RuntimeError("graft_spin_into_lot requires num_lots > 0")
        if self.d_lot < 2:
            raise RuntimeError("graft_spin_into_lot requires d_lot >= 2")
        r1 = spin_weight[1].detach().clone()
        r2 = spin_weight[2].detach().clone()
        self.lot_embedding.weight[:, :2].zero_()
        self.lot_embedding.weight[mp_pbe_id].zero_()
        self.lot_embedding.weight[mp_pbe_id, 0] = 1.0
        self.lot_embedding.weight[mp_pbeu_id].zero_()
        self.lot_embedding.weight[mp_pbeu_id, 1] = 1.0
        if copy_init_other_lots:
            pbeu_set = set(int(x) for x in pbeu_like_ids)
            for k in range(self.num_lots):
                if k in (mp_pbe_id, mp_pbeu_id):
                    continue
                if k in pbeu_set:
                    self.lot_embedding.weight[k, 0] = 0.0
                    self.lot_embedding.weight[k, 1] = 1.0
                else:
                    self.lot_embedding.weight[k, 0] = 1.0
                    self.lot_embedding.weight[k, 1] = 0.0
        self.lot_atom_proj.weight.zero_()
        self.lot_atom_proj.weight[:, 0].copy_(r1)
        self.lot_atom_proj.weight[:, 1].copy_(r2)
        if zero_spin_rows and self.spin_embedding is not None:
            self.spin_embedding.weight[1:3].zero_()

    def _resolve_lot_ids(self, lot_id: Tensor | int | None, B: int, device: torch.device) -> Tensor | None:
        """Normalize lot_id (int, 0-D tensor, [1], or [B]) to [B] long on device, or None when inactive."""
        if lot_id is None or self.lot_embedding is None:
            return None
        if isinstance(lot_id, int):
            return torch.full((B,), lot_id, dtype=torch.long, device=device)
        t = lot_id.to(device=device, dtype=torch.long)
        if t.ndim == 0 or (t.ndim == 1 and t.shape[0] == 1 and B > 1):
            return t.reshape(1).expand(B)
        return t

    def calibrate_angular(self, run, target: str = "inv") -> list[dict]:
        """Data-dependent scale initialisation of newly grafted branches (st.ang, st.l4) whose .out.weight
        is still zero (borrowed idea: LSUV, Mishkin and Matas 2016; here by homogeneity instead of iterations).
        Branches loaded from a checkpoint (non-zero .out.weight) are left untouched so function-preserving grafts onto
        models that already trained earlier branches (e.g. a100_pairloss_v1 with st.ang) remain exact."""
        active: list[tuple[int, str, nn.Module]] = []
        for idx, st in enumerate(self.stages_list):
            for attr in ("ang", "l4"):
                mod = getattr(st, attr, None)
                if mod is not None and float(mod.out.weight.detach().abs().max()) == 0.0:
                    active.append((idx, attr, mod))
        if not active:
            return []

        def measure() -> list[dict]:
            for _, _, mod in active:
                mod.record = {}
            run()
            rec = [dict(mod.record) for _, _, mod in active]
            for _, _, mod in active:
                mod.record = None
            return rec

        before = measure()
        for (_, _, mod), r in zip(active, before):  # vt3 / tt4 depend on radial3 / radial4 too: rescaled in a second pass
            for name, _, _ in mod.families():
                if name not in ("vt3", "tt4") and r.get(name, 0.0) > 0:
                    mod.rescale(name, r[target] / r[name])
        mid = measure()
        for (_, _, mod), r in zip(active, mid):
            for dep in ("vt3", "tt4"):
                if dep in r and r[dep] > 0:
                    mod.rescale(dep, r[target] / r[dep])
        after = measure()
        return [{"stage": idx, "branch": attr, "before": b, "after": a}
                for (idx, attr, _), b, a in zip(active, before, after)]

    def set_fast_ops(self, names: str | Sequence[str] | None = "all") -> frozenset:
        """Execution-only op fusions of the dense forward (default off; nothing is stored in the config or the state
        dict): "l2" l2_pair as an outer product + matmul (geometry and the nu3 invariant), "mix" the channel mixes that
        read the same tensor as one matmul each, "proj" scalar_pre | vec_gate | ten_gate as one linear. Each is the same
        function up to float rounding (tests/test_mena_ace_fast_ops.py); the point is fewer, larger kernels (launch-bound
        training steps; every forward op appears ~8 times in force training: forward, force, backward, double backward).
        names: comma list or sequence of FAST_OPS, "all" or ""/None for none. Returns the active set."""
        if names in (None, "", "none", ()):
            fast = frozenset()
        elif names == "all":
            fast = frozenset(FAST_OPS)
        else:
            lst = [x.strip() for x in names.split(",")] if isinstance(names, str) else list(names)
            bad = [x for x in lst if x and x not in FAST_OPS]
            if bad:
                raise ValueError(f"unknown fast op(s) {bad}; known: {FAST_OPS}")
            fast = frozenset(x for x in lst if x)
        self.fast_ops = fast
        for st in [*self.stages_list, *self.gated_stages_list]:
            st.fast = fast
        return fast

    def forward_dense_as_sparse(self, z: Tensor, pos: Tensor, mask: Tensor, total_charge: Tensor | None = None,
                                spin: Tensor | None = None, atom_charge: Tensor | None = None,
                                compute_gradient_forces: bool = True, single_island: bool = False,
                                **kw) -> dict[str, Tensor | None]:
        """A dense padded batch (z [B, N], pos [B, N, 3], mask [B, N]) evaluated on neighbour lists: same inputs and
        dense-shaped outputs (energy [B], forces / charges [B, N, ...]) as `forward`, but the pair work runs only on
        the real pairs inside r_cut (about a quarter of the dense slots at 128 atoms, 10% at 350) through
        `forward_sparse`. Lists are built on the device (one host sync for the edge count). Differentiable in pos.
        Needs the dense batch's structures to be single islands when single_island=True (see forward_sparse). Not
        supported: outputs only forward_sparse has (the charge-feedback list `feedback_charges` is absent)."""
        B, N = z.shape
        m = mask.bool()
        n = m.sum(1)
        flat = (torch.cumsum(m.long(), 1) - 1) + (torch.cumsum(n, 0) - n)[:, None]
        rc = float(kw.get("r_cut") or self.r_cut)
        r_elec = float(self.taper_elec.r_cut) if getattr(self, "taper_elec", None) is not None else rc
        r_ic = self.ic_r_cut if (self.ic_mode == "window" and self.ic_index) else 0.0
        r_lr = max(r_elec, r_ic, rc)
        with torch.no_grad():
            p = pos.detach()
            d = torch.cdist(p, p, compute_mode="donot_use_mm_for_euclid_dist")
            pm = m[:, :, None] & m[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=z.device)[None]
            ei = tuple(torch.stack([flat[b, i], flat[b, j]])
                       for b, i, j in (torch.nonzero(pm & (d < r + 1e-3), as_tuple=True) for r in (rc, r_lr)))
            batch = torch.arange(B, device=z.device)[:, None].expand(B, N)[m]
        pos_f = pos[m]
        if compute_gradient_forces and not pos_f.requires_grad:
            pos_f = pos_f.detach().requires_grad_(True)
        ac_f = atom_charge[m] if atom_charge is not None else None
        zero3 = lambda e: pos_f.new_zeros(e.shape[1], 3)  # noqa: E731
        out = self.forward_sparse(z[m], pos_f, batch, B, ei[0], zero3(ei[0]), total_charge=total_charge,
                                  lr_edge_index=ei[1], lr_edge_shift=zero3(ei[1]), spin=spin, atom_charge=ac_f,
                                  compute_gradient_forces=compute_gradient_forces, single_island=single_island, **kw)
        res = dict(out)
        for k in ("forces", "charges"):
            v = out.get(k)
            if torch.is_tensor(v):
                dense = v.new_zeros((B, N) + tuple(v.shape[1:]))
                dense[m] = v
                res[k] = dense
        return res

    def _pair_basis(self, rbf: Tensor, zi: Tensor, zj: Tensor) -> Tensor:
        """rbf [..., R] + MLP([rbf, e_i * e_j, e_i + e_j]) (pair_basis_hidden); zi, zj broadcast to rbf[..., 0]."""
        ei, ej = self.pair_basis_emb[zi], self.pair_basis_emb[zj]
        return rbf + self.pair_basis_mlp(torch.cat([rbf, ei * ej, ei + ej], dim=-1))

    def _pair_spline_coef(self, zi: Tensor, zj: Tensor) -> Tensor:
        """c(Z_i, Z_j) [..., R] = W (e_i * e_j) (pair_spline)."""
        return self.pair_spline_out(self.pair_spline_emb[zi] * self.pair_spline_emb[zj])

    @staticmethod
    def _parse_island_context(spec: str | Sequence[int] | None, stages: int) -> list[int]:
        """None / "" / "none" -> []; "0,1,2" or a sequence -> sorted unique stage indices in [0, stages - 1]."""
        if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "none")):
            return []
        vals = [int(x) for x in spec.split(",") if x.strip()] if isinstance(spec, str) else [int(x) for x in spec]
        pts = sorted(set(vals))
        if any(not 0 <= p <= stages - 1 for p in pts):
            raise ValueError(f"island_context points must be in [0, {stages - 1}], got {pts}")
        return pts

    def _fb_in(self, k: int, state: Tensor) -> Tensor:
        """Input of the charge-feedback heads k: the state, RMS-normalised when fb_input_norm is on."""
        return state if self.fb_norm is None else self.fb_norm[k](state)

    def _spin_index(self, spin: Tensor | None, B: int, device: torch.device) -> Tensor:
        """[B] long rows of the spin table: multiplicity clamped to [0, max_spin]; None -> 0 (unknown)."""
        if spin is None:
            return torch.zeros(B, dtype=torch.long, device=device)
        return spin.view(B).round().long().clamp(0, self.max_spin)

    def _eye(self, n: int, device: torch.device) -> Tensor:
        if torch.compiler.is_compiling():  # a cache write inside the traced torch.func step leaks a functorch wrapper
            return torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0)  # (AOTAutograd: "Cannot access storage of TensorWrapper")
        key = (n, device)
        if key not in self._diag:
            eye_t = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0)
            if isinstance(eye_t, torch._subclasses.FakeTensor):
                return eye_t
            self._diag[key] = eye_t
        return self._diag[key]

    @torch.no_grad()
    def _islands_dense(self, dist: Tensor, pair_mask: Tensor, mask: Tensor, r_cut: float | None = None) -> Tensor:
        """[B, N, N] float, 1 where atoms i and j are in the same island (connected through pairs with r < r_cut);
        padding rows are 0. Transitive closure by repeated squaring: ceil(log2 N) batched matmuls of 0/1 matrices."""
        B, N = mask.shape
        rc = self.r_cut if r_cut is None else float(r_cut)
        R = ((dist < rc) & pair_mask).to(dist.dtype) + torch.diag_embed(mask.to(dist.dtype))
        for _ in range(max(1, math.ceil(math.log2(max(N, 2))))):
            R = (torch.bmm(R, R) > 0).to(dist.dtype)
        return R

    def _island_inputs(self, R: Tensor | None, z: Tensor, mask_f: Tensor, total_charge: Tensor, atom_charge: Tensor | None,
                       spin: Tensor | None, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        """Per-atom island charge Q_I [B, N] and per-atom spin-table rows [B, N] (None without a spin embedding).
        R None: every graph is one island (forward single_island=True), so the island sum is the graph sum."""
        m = mask_f.squeeze(-1).to(dtype)
        B = m.shape[0]
        if atom_charge is None:
            atom_charge = total_charge.view(B, 1) / m.sum(-1, keepdim=True).clamp_min(1.0) * m
        if R is None:
            isum = lambda x: (x * m).sum(-1, keepdim=True) * m  # noqa: E731  same as R = m m^T, without the bmm
        else:
            isum = lambda x: torch.bmm(R, x.unsqueeze(-1)).squeeze(-1)  # noqa: E731  sum over the atom's island
        q_isl = isum(atom_charge.to(m.dtype)) * m
        if self.spin_embedding is None:
            return q_isl, None
        n_isl = (m / isum(m).clamp_min(1.0)).sum(-1, keepdim=True)  # sum over atoms of 1 / island size = #islands
        electrons = (isum(z.to(m.dtype) * m) - q_isl).round().long()
        lowest = electrons.remainder(2) + 1
        total = self._spin_index(spin, B, z.device).view(B, 1).expand_as(lowest)
        rows = torch.where(n_isl < 1.5, total, lowest.clamp(0, self.max_spin))
        if spin is None:
            rows = torch.zeros_like(rows)
        return q_isl, rows * mask_f.squeeze(-1).long()

    @staticmethod
    def _cov_radii(device: torch.device, dtype: torch.dtype) -> Tensor:
        """Covalent radii table [128] in Angstrom (Cordero et al. 2008 / ASE), cached without adding state_dict keys."""
        key = ("cov_radii", dtype, device)
        t = _E9.get(key)
        if t is None:
            from ase.data import covalent_radii

            rad = torch.tensor(covalent_radii[:128], dtype=dtype, device=device)
            if rad.numel() < 128:
                rad = torch.cat([rad, rad.new_full((128 - rad.numel(),), 0.76)])
            t = torch.where(torch.isfinite(rad) & (rad > 0), rad, torch.full_like(rad, 0.76))
            if not torch.compiler.is_compiling() and not isinstance(t, torch._subclasses.FakeTensor):
                _E9[key] = t
        return t

    def _bond_weights_dense(self, z: Tensor, dist: Tensor, pmf: Tensor, r_cut: float | None = None) -> Tensor:
        """Smooth C3 covalent bond taper w_bond(r_ij) [B, N, N] in [0, 1] based on element covalent radii,
        clamped so 1.45 * r0 <= r_cut (strictly zero at min(1.45*(r_cov_i + r_cov_j), r_cut))."""
        rc = self.r_cut if r_cut is None else float(r_cut)
        rad = self._cov_radii(z.device, dist.dtype)[z.clamp(0, 127)]  # [B, N]
        r0 = (rad.unsqueeze(2) + rad.unsqueeze(1)).clamp(min=0.4, max=rc / 1.45)
        return taper_envelope(dist / r0, 1.25, 1.45, True) * pmf

    def _bond_weights_sparse(self, z: Tensor, dist: Tensor, recv: Tensor, send: Tensor, r_cut: float | None = None) -> Tensor:
        """Smooth C3 covalent bond taper w_bond(r_ij) [E] in [0, 1] based on element covalent radii,
        clamped so 1.45 * r0 <= r_cut (strictly zero at min(1.45*(r_cov_i + r_cov_j), r_cut))."""
        rc = self.r_cut if r_cut is None else float(r_cut)
        rad = self._cov_radii(z.device, dist.dtype)[z.clamp(0, 127)]  # [N]
        r0 = (rad[recv] + rad[send]).clamp(min=0.4, max=rc / 1.45)
        return taper_envelope(dist / r0, 1.25, 1.45, True)

    def _split_seed_dense(
        self,
        z: Tensor,
        dist: Tensor,
        pmf: Tensor,
        mask_f: Tensor,
        total_charge: Tensor,
        atom_charge: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        """Per-atom formal charge seed q_seed [B, N] for B1/B3 split-charge allocation (no hardcoded solvent rules).
        When `atom_charge` is supplied, it enters q_tilde_i directly; when None, q_seed is 0 and Q_total is allocated
        by the learned NQE weights f_i."""
        m = mask_f.squeeze(-1).to(dtype)
        if atom_charge is not None:
            return atom_charge.to(dtype) * m
        return torch.zeros_like(m)

    def _split_seed_sparse(
        self,
        z: Tensor,
        dist: Tensor,
        recv: Tensor,
        send: Tensor,
        batch: Tensor,
        B: int,
        total_charge: Tensor | None,
        group: Tensor | None,
        group_charge: Tensor | None,
        atom_charge: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        """Sparse equivalent of _split_seed_dense: returns q_seed [N] without any hardcoded water-solvent rules."""
        N = z.shape[0]
        if atom_charge is not None:
            return atom_charge.to(dtype)
        if group is not None and group_charge is not None:
            G = int(group_charge.shape[0])
            ones = torch.ones(N, dtype=dtype, device=z.device)
            n_group = torch.zeros(G, dtype=dtype, device=z.device).index_add_(0, group, ones).clamp_min(1.0)
            return (group_charge.to(dtype) / n_group)[group]
        return torch.zeros(N, dtype=dtype, device=z.device)

    @staticmethod
    def _pairwise_transfer_dense(phi: Tensor, w_bond: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
        """Exact antisymmetric bond charge transfer dq_ij = w_bond_ij * tanh(phi_i - phi_j) = -dq_ji (dense).
        Returns (dq_net [B, N], dq_mat [B, N, N]) where sum_{i,j} dq_ij == 0 identically."""
        dq_mat = w_bond * torch.tanh(phi.unsqueeze(2) - phi.unsqueeze(1))
        return dq_mat.sum(dim=-1) * m, dq_mat

    @staticmethod
    def _pairwise_transfer_sparse(phi: Tensor, w_bond: Tensor, recv: Tensor, send: Tensor) -> tuple[Tensor, Tensor]:
        """Exact antisymmetric bond charge transfer dq_e = w_bond_e * tanh(phi_recv - phi_send) (sparse).
        Returns (dq_net [N], dq_edge [E])."""
        N = phi.shape[0]
        dq_e = w_bond * torch.tanh(phi[recv] - phi[send])
        dq_net = phi.new_zeros(N).index_add_(0, recv, dq_e)
        return dq_net, dq_e

    @staticmethod
    def _equilibrate_split_dense(
        q_local: Tensor,
        f: Tensor,
        q_seed: Tensor,
        w_bond: Tensor,
        m: Tensor,
        total_charge: Tensor | None = None,
        phi: Tensor | None = None,
        w_lr: Tensor | None = None,
        seeded: bool = False,
    ) -> Tensor:
        """B1 split-charge allocation (dense):
        1. Pairwise antisymmetric short-range bond transfer dq_ij = w_bond_ij * tanh(phi_i - phi_j) = -dq_ji,
           so sum_{i in M, j in M} dq_ij == 0 on every covalent cluster M.
        2. Per-atom base charges q_tilde_i = q_seed_i + dq_local_i + sum_j dq_ij (or q_seed_i + sum_j dq_ij when
           formal `atom_charge` seeds are explicitly supplied via `seeded=True`, preserving each fragment's formal
           charge while redistributing charge internally via dq_ij).
        3. SpookyNet/NQE total charge conservation:
           q_i = q_tilde_i + f_i * (Q_total - sum_k q_tilde_k) / sum_k f_k."""
        B = m.shape[0]
        f = f * m
        q_local = q_local * m
        q_seed = q_seed * m
        if seeded:
            dq_local = torch.zeros_like(q_local)
        elif w_lr is not None:
            # Floor (W f)_i at f_i so f_i / (W f)_i <= 1 and |dq_local_i| <= |q_i| + |(W q)_i|. The window kernel rings
            # slightly negative (~-1e-5) and f_i -> 0 on isolated atoms: (W f)_i < 0 was clamped to 1e-12 and gave
            # +587 e on an isolated He (seen during training). Identical to before wherever (W f)_i >= f_i.
            W = w_lr + torch.diag_embed(m)
            loc_num = torch.bmm(W, q_local.unsqueeze(-1)).squeeze(-1)
            loc_den = torch.maximum(torch.bmm(W, f.unsqueeze(-1)).squeeze(-1), f).clamp_min(1e-12)
            dq_local = (q_local - f * (loc_num / loc_den)) * m
        else:
            dq_local = q_local
        if phi is not None:
            dq_bond = (w_bond * torch.tanh(phi.unsqueeze(2) - phi.unsqueeze(1))).sum(dim=-1) * m
        else:
            dq_bond = torch.zeros_like(q_local)
        q_tilde = (q_seed + dq_local + dq_bond) * m
        q_tot = (
            total_charge.to(q_local.dtype).view(B, 1)
            if total_charge is not None
            else q_seed.sum(dim=-1, keepdim=True)
        )
        f_sum = f.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return (q_tilde + f * (q_tot - q_tilde.sum(dim=-1, keepdim=True)) / f_sum) * m

    @staticmethod
    def _equilibrate_split_sparse(
        q_local: Tensor,
        f: Tensor,
        q_seed: Tensor,
        w_bond: Tensor,
        recv: Tensor,
        send: Tensor,
        batch: Tensor,
        B: int,
        total_charge: Tensor | None = None,
        phi: Tensor | None = None,
        lr_recv: Tensor | None = None,
        lr_send: Tensor | None = None,
        w_lr: Tensor | None = None,
        seeded: bool = False,
        win_sum=None,
    ) -> Tensor:
        """Sparse equivalent of _equilibrate_split_dense: O(E_sr + E_lr) index_add_ with exact edge antisymmetry.
        win_sum (mesh backend): x -> sum_j w(r_ij) x_j over the window, self included, in place of the lr edges."""
        N = q_local.shape[0]
        if seeded:
            dq_local = torch.zeros_like(q_local)
        elif win_sum is not None:
            dq_local = q_local - f * (win_sum(q_local) / torch.maximum(win_sum(f), f).clamp_min(1e-12))
        elif w_lr is not None and lr_recv is not None and lr_send is not None:
            loc_num = q_local + q_local.new_zeros(N).index_add_(0, lr_recv, w_lr * q_local[lr_send])
            # floor at f: see _equilibrate_split_dense
            loc_den = torch.maximum(f + f.new_zeros(N).index_add_(0, lr_recv, w_lr * f[lr_send]), f).clamp_min(1e-12)
            dq_local = q_local - f * (loc_num / loc_den)
        else:
            dq_local = q_local
        if phi is not None:
            dq_e = w_bond * torch.tanh(phi[recv] - phi[send])
            dq_bond = q_local.new_zeros(N).index_add_(0, recv, dq_e)
        else:
            dq_bond = torch.zeros_like(q_local)
        q_tilde = q_seed + dq_local + dq_bond
        if total_charge is not None:
            q_tot = total_charge.to(q_local.dtype).view(B)[batch]
        else:
            q_tot = q_local.new_zeros(B).index_add_(0, batch, q_seed)[batch]
        f_sum = q_local.new_zeros(B).index_add_(0, batch, f).clamp_min(1e-12)[batch]
        q_tilde_sum = q_local.new_zeros(B).index_add_(0, batch, q_tilde)[batch]
        return q_tilde + f * (q_tot - q_tilde_sum) / f_sum

    def _spin_embed_split_dense(
        self,
        state: Tensor,
        z: Tensor,
        q_seed: Tensor,
        w_bond: Tensor,
        mask_f: Tensor,
        spin: Tensor | None,
    ) -> Tensor:
        """B5 learned per-atom spin allocation (dense):
        Allocates unpaired spin over atoms with learned positive weights w_raw,i = softplus(nqe_head_spin(h_i^(0)))
        gated by the smooth 2-hop covalent cluster odd-electron indicator p_rad,i = sin^2(pi/2 * e_2hop,i) (with
        fallback to w_raw,i when all clusters have even electron count). Closed-shell molecules (such as H2O at 20 A
        from CH3.) receive exact singlet embedding, preserving strict size consistency."""
        B = state.shape[0]
        m = mask_f.squeeze(-1)
        s_idx = self._spin_index(spin, B, state.device)
        if self.nqe_head_spin is None:
            return self.spin_embedding(s_idx).unsqueeze(1) * mask_f
        base_idx = s_idx.clamp_max(1)
        e_base = self.spin_embedding(base_idx).unsqueeze(1)
        e_targ = self.spin_embedding(s_idx).unsqueeze(1)
        w_raw = F.softplus(self.nqe_head_spin(state).squeeze(-1)) * m
        if self.use_spin_radical_gate:
            wb = w_bond.detach().to(state.dtype)
            e0 = (z.to(state.dtype) - q_seed.detach().to(state.dtype)) * m
            e1 = e0 + torch.bmm(wb, e0.unsqueeze(-1)).squeeze(-1)
            wb2 = wb.pow(2).sum(dim=-1)
            e2 = (e1 + torch.bmm(wb, (e1 - e0).unsqueeze(-1)).squeeze(-1) - wb2 * e0) * m
            p_rad = (torch.sin(0.5 * math.pi * e2).pow(2) * m).detach()
            w_rad = w_raw * p_rad
            rad_sum = w_rad.sum(dim=-1, keepdim=True)
            gate = rad_sum / (rad_sum + 1e-2)
            w_s = gate * w_rad + (1.0 - gate) * w_raw
        else:
            w_s = w_raw
        w_sum = w_s.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        w_sq_sum = w_s.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        n_eff = w_sum.pow(2) / w_sq_sum
        alpha = (n_eff * w_s / w_sum).unsqueeze(-1)
        return (e_base + alpha * (e_targ - e_base)) * mask_f

    def _spin_embed_split_sparse(
        self,
        state: Tensor,
        z: Tensor,
        q_seed: Tensor,
        w_bond: Tensor,
        recv: Tensor,
        send: Tensor,
        batch: Tensor,
        B: int,
        spin: Tensor | None,
    ) -> Tensor:
        """Sparse equivalent of _spin_embed_split_dense."""
        N = state.shape[0]
        s_idx = self._spin_index(spin, B, state.device)
        if self.nqe_head_spin is None:
            return self.spin_embedding(s_idx)[batch]
        base_idx = s_idx.clamp_max(1)
        e_base = self.spin_embedding(base_idx)[batch]
        e_targ = self.spin_embedding(s_idx)[batch]
        w_raw = F.softplus(self.nqe_head_spin(state).squeeze(-1))
        if self.use_spin_radical_gate:
            wb = w_bond.detach().to(state.dtype)
            e0 = z.to(state.dtype) - q_seed.detach().to(state.dtype)
            e1 = e0 + state.new_zeros(N).index_add_(0, recv, wb * e0[send])
            wb2 = state.new_zeros(N).index_add_(0, recv, wb.pow(2))
            e2 = e1 + state.new_zeros(N).index_add_(0, recv, wb * (e1[send] - e0[send])) - wb2 * e0
            p_rad = torch.sin(0.5 * math.pi * e2).pow(2).detach()
            w_rad = w_raw * p_rad
            rad_sum = state.new_zeros(B).index_add_(0, batch, w_rad)[batch]
            gate = rad_sum / (rad_sum + 1e-2)
            w_s = gate * w_rad + (1.0 - gate) * w_raw
        else:
            w_s = w_raw
        w_sum = state.new_zeros(B).index_add_(0, batch, w_s).clamp_min(1e-6)[batch]
        w_sq_sum = state.new_zeros(B).index_add_(0, batch, w_s.pow(2)).clamp_min(1e-6)[batch]
        n_eff = w_sum.pow(2) / w_sq_sum
        alpha = (n_eff * w_s / w_sum).unsqueeze(-1)
        return e_base + alpha * (e_targ - e_base)

    def _dsf_kernel(self, r: Tensor, alpha: float | None = None) -> Tensor:
        """Damped shifted-force (DSF) Coulomb kernel V_DSF(r) with short-range gamma damping:
        phi(r) = erfc(alpha * r_s) / r_s, r_s = (r^3 + gamma^3)^(1/3),
        V_DSF(r) = [phi(r) - phi(R_c) - phi'(R_c) * (r - R_c)] * 1_{r < R_c}."""
        rc = float(self.taper_elec.r_cut)
        gam = float(self.coulomb_gamma)
        alp = float(self.dsf_alpha if alpha is None else alpha)
        r_s = (r.pow(3) + gam ** 3).pow(1.0 / 3.0)
        rs_c = (rc ** 3 + gam ** 3) ** (1.0 / 3.0)
        erfc_c = math.erfc(alp * rs_c)
        phi_c = erfc_c / rs_c
        dphi_drs_c = (-2.0 * alp / math.sqrt(math.pi) * math.exp(-((alp * rs_c) ** 2)) * rs_c - erfc_c) / (rs_c ** 2)
        drs_dr_c = (rc ** 2) / (rs_c ** 2)
        dphi_c = dphi_drs_c * drs_dr_c
        phi = torch.erfc(alp * r_s) / r_s if alp > 0.0 else (1.0 / r_s)
        return (phi - phi_c - dphi_c * (r - rc)) * (r < rc).to(r.dtype)

    def _coulomb_sr(self, r: Tensor) -> Tensor:
        """Mesh definition, short part: (1 / r_s - erf(r / sigma) / r) * C3 taper(r_on, r_cut) (short-range edges only)."""
        r_s = (r.pow(3) + self.coulomb_gamma ** 3).pow(1.0 / 3.0)
        return (1.0 / r_s - self._coul_lr.real(r)) * taper_envelope(r, self.r_on, self.r_cut, True)

    def _coulomb_full(self, r: Tensor) -> Tensor:
        """Mesh definition, whole kernel erf(r / sigma) / r + K_sr(r), for exact pair sums (no long-range taper)."""
        return self._coul_lr.real(r) + self._coulomb_sr(r)

    def _coulomb_open_kernel(self, extent: float):
        """erf(r / sigma) / r truncated beyond the structure's diameter (open boundary), cached per 10 A bucket."""
        r_t = 10.0 * math.ceil((extent + 6.0) / 10.0)
        if r_t not in self._coul_open:
            self._coul_open[r_t] = erf_coulomb_kernel(self.coulomb_sigma, r_trunc=r_t, trunc_width=4.0)
        return self._coul_open[r_t]

    def _vacuum_coulomb_kernel(self, r: Tensor) -> Tensor:
        """B2 vacuum Coulomb kernel (pbc=False): smooth C3-tapered 1/r_s when dsf_coulomb is True (matching exact
        1/r_s for r <= r_on_elec = 15 A and vanishing C3-smoothly at r_cut_elec = 18 A), or the legacy OMol1 kernel
        when coulomb_range is 'full' or 'local'."""
        r_s = (r.pow(3) + self.coulomb_gamma ** 3).pow(1.0 / 3.0)
        if self.dsf_coulomb:
            alp = float(self.dsf_alpha)
            base = torch.erfc(alp * r_s) / r_s if alp > 0.0 else (1.0 / r_s)
            return base * self.taper_elec(r)
        damp = 1.0 if self.full_coulomb else self.taper_elec(r)
        return damp / r_s

    def _elec_mask(
        self,
        B: int,
        device: torch.device,
        lot_id: Tensor | int | None = None,
        pbc: bool = False,
        use_elec_context: bool | None = None,
    ) -> Tensor:
        """Per-structure boolean mask [B] on `device`: True where electrostatics/IC/feedback are active.
        Pure tensor ops without any .item() GPU->CPU syncs."""
        if use_elec_context is not None:
            return torch.full((B,), bool(use_elec_context), dtype=torch.bool, device=device)
        if not self.use_electrostatics and not self.ic_index and not self.feedback_points:
            return torch.zeros(B, dtype=torch.bool, device=device)
        if self.num_lots == 0:
            return torch.ones(B, dtype=torch.bool, device=device)
        if self.elec_gate == "system" and not pbc:
            return torch.ones(B, dtype=torch.bool, device=device)
        if lot_id is None:
            return torch.full((B,), not pbc, dtype=torch.bool, device=device)
        if isinstance(lot_id, int):
            return torch.full((B,), int(lot_id) in self.elec_lot_ids, dtype=torch.bool, device=device)
        t = lot_id.to(device=device, dtype=torch.long)
        if t.ndim == 0 or (t.ndim == 1 and t.shape[0] == 1 and B > 1):
            t = t.reshape(1).expand(B)
        mask = torch.zeros(B, dtype=torch.bool, device=device)
        for lid in self.elec_lot_ids:
            mask = mask | (t == lid)
        return mask

    def _static_no_elec(self, lot_id: Tensor | int | None, pbc: bool, use_elec_context: bool | None) -> bool:
        """True when the electrostatic/IC path is off for the whole batch, decidable without looking at tensors."""
        if self.num_lots == 0:
            return use_elec_context is False
        if self.elec_gate == "lot":  # legacy rule, kept verbatim (an int non-elec lot_id wins over use_elec_context=True)
            return ((use_elec_context is False)
                    or (isinstance(lot_id, int) and (int(lot_id) not in self.elec_lot_ids))
                    or (lot_id is None and pbc and use_elec_context is None))
        if use_elec_context is not None:
            return use_elec_context is False
        if self.elec_gate == "system" and not pbc:
            return False
        if isinstance(lot_id, int):
            return int(lot_id) not in self.elec_lot_ids
        return lot_id is None and pbc

    def _should_run_elec(self, lot_id_or_pbc: Tensor | int | bool | None = None,
                         lot_g: Tensor | int | None = None, use_elec_context: bool | None = None) -> bool:
        """LoT-based gate: ON when any structure in the batch has lot_id in self.elec_lot_ids;
        OFF when all structures have crystal LoTs (lot_id in 0..5). Falls back to `not pbc` when lot_id is None."""
        if use_elec_context is not None:
            return bool(use_elec_context)
        if not self.use_electrostatics and not self.ic_index and not self.feedback_points:
            return False
        if self.num_lots == 0:
            return True
        if isinstance(lot_id_or_pbc, bool):
            pbc = lot_id_or_pbc
            lot_val = lot_g
        else:
            pbc = False
            lot_val = lot_id_or_pbc if lot_g is None else lot_g
        if self.elec_gate == "system" and not pbc:
            return True
        if isinstance(lot_val, int):
            return lot_val in self.elec_lot_ids
        if torch.is_tensor(lot_val) and lot_val.numel() > 0:
            mask = self._elec_mask(int(lot_val.numel()), lot_val.device, lot_id=lot_val.reshape(-1), pbc=pbc)
            return bool(mask.any())
        return not pbc

    @staticmethod
    def _bounded(x: Tensor, q_max: float | None) -> Tensor:
        """q_max tanh(x / q_max): the raw charge output kept inside (-q_max, q_max) e (identity when q_max is None)."""
        return x if q_max is None else q_max * torch.tanh(x / q_max)

    @staticmethod
    def _equilibrate(q_raw: Tensor, f: Tensor, q_target: Tensor, R: Tensor | None, m: Tensor) -> Tensor:
        """q_i = q_raw_i + f_i (Q - sum q_raw) / sum f, the sums over the island (R) or, with R None, the graph."""
        if R is None:
            s_q, s_f = q_raw.sum(-1, keepdim=True), f.sum(-1, keepdim=True)
        else:
            s_q = torch.bmm(R, q_raw.unsqueeze(-1)).squeeze(-1)
            s_f = torch.bmm(R, f.unsqueeze(-1)).squeeze(-1)
        return (q_raw + f * (q_target - s_q) / s_f.clamp_min(1e-12)) * m

    def _qeq_matrix(self, state: Tensor, z: Tensor, dist: Tensor, pmf: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
        """Electronegativities chi [B, N] and the positive-definite QEq matrix A [B, N, N] (padding rows: identity)."""
        k = self.coulomb_k
        m, pmf = m.to(state.dtype), pmf.to(state.dtype)
        chi = self.qeq_j0 * self.charge_head(state).squeeze(-1) * m
        sig = self.qeq_log_sigma.exp()[z]  # [B, N]
        s2 = sig.pow(2)
        s_ij = (2.0 * (s2.unsqueeze(2) + s2.unsqueeze(1))).sqrt()
        d = dist + (1.0 - pmf)  # 1 on the diagonal and padding (masked below), keeps erf(d / s) / d finite
        A = k * torch.erf(d / s_ij) / d * pmf
        # floor 0.1 j0 (~1 eV) above the Gaussian self-energy: smallest eigenvalue >= 23 kcal/mol/e^2, safe in float32
        J = k / (sig * math.sqrt(math.pi)) + self.qeq_j0 * (F.softplus(self.qeq_head(state).squeeze(-1)) + 0.1)
        return chi, A + torch.diag_embed(J * m + (1.0 - m))  # padding: identity, chi = 0, no constraint -> q = 0

    def _qeq_dense(self, state: Tensor, z: Tensor, dist: Tensor, pmf: Tensor, m: Tensor, q_target: Tensor,
                   R: Tensor | None) -> tuple[Tensor, Tensor]:
        """Charges [B, N] and energy [B] of the island-constrained QEq (see qeq in __init__). q_target [B, N] is the
        charge of each atom's island (or the graph's total charge); R [B, N, N] the same-island matrix (None: one
        constraint per graph)."""
        B, N = m.shape
        m, q_target = m.to(state.dtype), q_target.to(state.dtype)
        chi, A = self._qeq_matrix(state, z, dist, pmf, m)
        # constraint matrix C [B, N, I]: one column per island (islands numbered by their lowest atom index)
        if R is None:
            C = m.unsqueeze(-1)
        else:
            Rb = R > 0.5
            label = torch.argmax(Rb.to(torch.uint8), dim=-1)  # first atom of the island
            root = (label == torch.arange(N, device=m.device)) & (m > 0.5)
            cid = torch.cumsum(root.long(), -1) - 1  # compact island id of every root
            n_isl = int(root.sum(-1).max().item())
            atom_cid = torch.gather(cid, 1, label).clamp_min(0)
            C = F.one_hot(atom_cid, max(1, n_isl)).to(m.dtype) * m.unsqueeze(-1)
        cnt = C.sum(1)  # [B, I] atoms per island (0 = no such island in this graph)
        Qc = (C * q_target.unsqueeze(-1)).sum(1) / cnt.clamp_min(1.0)
        L = torch.linalg.cholesky(A)
        X = torch.cholesky_solve(torch.cat([-chi.unsqueeze(-1), C], -1), L)
        x0, Y = X[..., 0], X[..., 1:]
        S = C.transpose(1, 2) @ Y + torch.diag_embed((cnt < 0.5).to(m.dtype))
        lam = torch.linalg.solve(S, (Qc - (C * x0.unsqueeze(-1)).sum(1)).unsqueeze(-1)).squeeze(-1)
        q = (x0 + (Y @ lam.unsqueeze(-1)).squeeze(-1)) * m
        e = (chi * q).sum(-1) + 0.5 * (q * (A @ q.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return q, e

    def _feedback_dense(self, k: int, state: Tensor, total_charge: Tensor, mask_f: Tensor) -> tuple[Tensor, Tensor]:
        """Charge feedback k (dense): state [B, N, D] -> (state + embed(q), q) with q summing to total_charge per graph."""
        B = state.shape[0]
        m = mask_f.squeeze(-1)
        x = self._fb_in(k, state)
        q_raw = self._bounded(self.fb_charge[k](x).squeeze(-1), self.feedback_bound) * m
        f = F.softplus(self.fb_weight[k](x).squeeze(-1)) * m
        q = (q_raw + f * (total_charge.view(B, 1) - q_raw.sum(-1, keepdim=True)) / f.sum(-1, keepdim=True).clamp_min(1e-12)) * m
        return (state + self.fb_embed[k](q.unsqueeze(-1))) * mask_f, q

    def _feedback_island(self, k: int, state: Tensor, q_isl: Tensor, R: Tensor | None, mask_f: Tensor) -> tuple[Tensor, Tensor]:
        """Charge feedback k, island mode: the charges sum to each island's charge (R None: one island per graph)."""
        m = mask_f.squeeze(-1)
        x = self._fb_in(k, state)
        q_raw = self._bounded(self.fb_charge[k](x).squeeze(-1), self.feedback_bound) * m
        f = F.softplus(self.fb_weight[k](x).squeeze(-1)) * m
        q = self._equilibrate(q_raw, f, q_isl, R, m)
        return (state + self.fb_embed[k](q.unsqueeze(-1))) * mask_f, q

    def _feedback_split_dense(
        self,
        k: int,
        state: Tensor,
        q_seed: Tensor,
        w_bond: Tensor,
        mask_f: Tensor,
        total_charge: Tensor,
        w_lr: Tensor | None = None,
        elec_f: Tensor | None = None,
        seeded: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Charge feedback k, split mode (dense): B1 pairwise bond transfer + SpookyNet/NQE charge allocation."""
        m = mask_f.squeeze(-1)
        x = self._fb_in(k, state)
        q_local = self._bounded(self.fb_charge[k](x).squeeze(-1), self.feedback_bound) * m
        f = F.softplus(self.fb_weight[k](x).squeeze(-1)) * m
        phi = self.fb_charge_transfer[k](x).squeeze(-1) * m if self.fb_charge_transfer is not None else None
        q = self._equilibrate_split_dense(
            q_local, f, q_seed, w_bond, m, total_charge=total_charge, phi=phi, w_lr=w_lr, seeded=seeded
        )
        if elec_f is not None:
            q = q * elec_f.view(-1, 1)
            delta_s = elec_f.view(-1, 1, 1) * self.fb_embed[k](q.unsqueeze(-1))
        else:
            delta_s = self.fb_embed[k](q.unsqueeze(-1))
        return (state + delta_s) * mask_f, q

    @torch.no_grad()
    def _islands_sparse(self, N: int, recv: Tensor, send: Tensor, dist: Tensor, r_cut: float | None = None) -> Tensor:
        """[N] island index (0..I-1) of every atom: connected components of the edges with r < r_cut, by min-label
        propagation (iterations = the longest shortest path in hops; inference only)."""
        lab = torch.arange(N, device=recv.device)
        rc = self.r_cut if r_cut is None else float(r_cut)
        keep = dist < rc
        r, s = recv[keep], send[keep]
        while True:
            new = lab.scatter_reduce(0, r, lab[s], reduce="amin", include_self=True)
            if torch.equal(new, lab):
                break
            lab = new
        return torch.unique(lab, return_inverse=True)[1]

    def _feedback_charges_sparse(self, k: int, state: Tensor, group: Tensor, group_charge: Tensor, q_per_atom: Tensor,
                                 batch: Tensor, n_graphs: int, group_neutralize: bool) -> Tensor:
        """Charge feedback k (sparse): the per-atom charges [N], constrained exactly as the final latent charges are
        (per charge group; or, with group_neutralize=False, one constraint per graph with the groups' Q_g/N_g baseline)."""
        N, G, B = state.shape[0], int(group_charge.shape[0]), int(n_graphs)
        x = self._fb_in(k, state)
        q_raw = self._bounded(self.fb_charge[k](x).squeeze(-1), self.feedback_bound)
        f = F.softplus(self.fb_weight[k](x).squeeze(-1))
        if group_neutralize:
            s_q = q_raw.new_zeros(G).index_add_(0, group, q_raw)[group]
            s_f = q_raw.new_zeros(G).index_add_(0, group, f)[group]
            return q_raw + f * (group_charge.to(q_raw.dtype)[group] - s_q) / s_f
        s_q = q_raw.new_zeros(B).index_add_(0, batch, q_raw)[batch]
        s_f = q_raw.new_zeros(B).index_add_(0, batch, f)[batch]
        return q_raw - f * s_q / s_f + q_per_atom

    def _feedback_split_sparse(
        self,
        k: int,
        state: Tensor,
        q_seed: Tensor,
        w_bond: Tensor,
        recv: Tensor,
        send: Tensor,
        batch: Tensor,
        B: int,
        total_charge: Tensor | None = None,
        lr_recv: Tensor | None = None,
        lr_send: Tensor | None = None,
        w_lr: Tensor | None = None,
        seeded: bool = False,
        win_sum=None,
    ) -> Tensor:
        """Charge feedback k, split mode (sparse): B1 pairwise bond transfer + SpookyNet/NQE charge allocation."""
        x = self._fb_in(k, state)
        q_local = self._bounded(self.fb_charge[k](x).squeeze(-1), self.feedback_bound)
        f = F.softplus(self.fb_weight[k](x).squeeze(-1))
        phi = self.fb_charge_transfer[k](x).squeeze(-1) if self.fb_charge_transfer is not None else None
        return self._equilibrate_split_sparse(
            q_local,
            f,
            q_seed,
            w_bond,
            recv,
            send,
            batch,
            B,
            total_charge=total_charge,
            phi=phi,
            lr_recv=lr_recv,
            lr_send=lr_send,
            w_lr=w_lr,
            seeded=seeded,
            win_sum=win_sum,
        )

    @staticmethod
    def _unwrap_pbc_dipole(
        q: Tensor,
        pos: Tensor,
        diff: Tensor,
        w_bond: Tensor | None,
        dist: Tensor,
        recv: Tensor,
        send: Tensor,
        batch: Tensor,
        B: int,
    ) -> Tensor:
        """B8 unwrapped molecular dipole under PBC: reconstructs internal coordinates of each covalently bonded molecule
        using the minimum-image bond vectors `diff = pos[recv] - pos[send] - edge_shift` relative to each molecule's
        unwrapped center of geometry, so wrapping any atom across periodic cell boundaries leaves the dipole unchanged."""
        N = pos.shape[0]
        if N == 0:
            return pos.new_zeros(B, 3)
        bond_mask = (w_bond > 0.05) if w_bond is not None else (dist < 2.1)
        b_recv = recv[bond_mask]
        b_send = send[bond_mask]
        b_diff = diff[bond_mask]
        lab = torch.arange(N, device=pos.device)
        rel_pos = pos.new_zeros(N, 3)
        if b_recv.numel() > 0:
            for _ in range(12):
                cand_lab = lab[b_send]
                better = cand_lab < lab[b_recv]
                if not bool(better.any()):
                    break
                r_upd = b_recv[better]
                s_upd = b_send[better]
                d_upd = b_diff[better]
                new_lab = lab.clone()
                new_lab[r_upd] = cand_lab[better]
                new_rel = rel_pos.clone()
                new_rel[r_upd] = rel_pos[s_upd] + d_upd
                lab = new_lab
                rel_pos = new_rel
        _, mol_id = torch.unique(lab, return_inverse=True)
        n_mols = int(mol_id.max().item()) + 1
        cnt_m = pos.new_zeros(n_mols, 1).index_add_(0, mol_id, pos.new_ones(N, 1)).clamp_min(1.0)
        cog_m = pos.new_zeros(n_mols, 3).index_add_(0, mol_id, rel_pos) / cnt_m
        r_internal = rel_pos - cog_m[mol_id]
        return pos.new_zeros(B, 3).index_add_(0, batch, q.unsqueeze(-1) * r_internal)

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        mask: Tensor,
        total_charge: Tensor | None = None,
        compute_gradient_forces: bool = True,
        compute_forces: bool | None = None,
        spin: Tensor | None = None,
        atom_charge: Tensor | None = None,
        direct_forces: bool = False,
        single_island: bool = False,
        lot_id: Tensor | int | None = None,
        r_cut: float | None = None,
        r_on: float | None = None,
        c3_taper: bool | None = None,
        use_pair_radius: bool | None = None,
        use_elec_context: bool | None = None,
        edge_index: Tensor | None = None,
        edge_shift: Tensor | None = None,
        **_: object,
    ) -> dict[str, Tensor | None]:
        """z [B, N] long, pos [B, N, 3] Angstrom, mask [B, N] bool, total_charge [B].
        direct_forces: forces from the direct head instead of -dE/dpos (needs direct_force_head; pretraining only). Extra kwargs (dataset_id,
        compute_orbitals) are accepted for drop-in compatibility with the MenaMinimal evaluation code.
        atom_charge [B, N] (island/split mode, optional): per-atom formal charge seed that enters q_tilde_i directly.
        single_island (island mode only, dense path): the caller guarantees that every graph is ONE island.
        edge_index [2, E] / edge_shift [E, 3] (optional, B4 hybrid sparse-SR + dense-LR): runs the 4 short-range
        stages on the sparse edge list (with ghost atom at index B * N) while keeping long-range IC window, split-charge
        and Coulomb in dense [B, N, N] block form."""
        if compute_forces is not None:
            compute_gradient_forces = compute_forces
        if direct_forces:
            if self.force_head is None:
                raise ValueError("direct_forces=True needs a model built with direct_force_head=True")
            compute_gradient_forces = False
        if compute_gradient_forces and not pos.requires_grad:
            pos = pos.detach().requires_grad_(True)
        if self.env_pair_out is not None:
            raise NotImplementedError("env_pair_rank: forward_sparse only")
        B, N = z.shape
        use_sparse_sr = edge_index is not None
        mask_f = mask.float().unsqueeze(-1)  # [B, N, 1]
        n_atoms = mask_f.sum(dim=1).clamp_min(1.0)  # [B, 1]
        if total_charge is None:
            total_charge = pos.new_zeros(B)
        rc = self.r_cut if r_cut is None else float(r_cut)
        ro = self.r_on if r_on is None else float(r_on)
        c3 = self.c3_taper if c3_taper is None else bool(c3_taper)
        use_pr = (self.pair_log_radius is not None) if use_pair_radius is None else (
            bool(use_pair_radius) and self.pair_log_radius is not None
        )

        # Geometry, shared by all stages (a function of pos only; computed first so island/split mode can use it)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # r_i - r_j
        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)) & ~self._eye(N, pos.device)
        pmf = pair_mask.float()
        dist = (diff.pow(2).sum(-1) + 1e-8).sqrt() * pmf
        unit = None if use_sparse_sr else (diff / (dist.unsqueeze(-1) + 1e-7) * pmf.unsqueeze(-1))
        if self.zbl_core:
            dist_raw = dist
            dist = self._clamp_dense(z, dist, pmf)
        if not use_sparse_sr or self.band is not None or self.pair_spline_out is not None:
            if rc == self.r_cut and ro == self.r_on and c3 == self.c3_taper:
                chi = self.envelope(dist) * pmf
            else:
                chi = taper_envelope(dist, ro, rc, c3) * pmf
        else:
            chi = None

        lot_g = self._resolve_lot_ids(lot_id, B, pos.device)
        e_lot = self.lot_embedding(lot_g).unsqueeze(1) if lot_g is not None else None  # [B, 1, d_lot]
        elec_mask_b = self._elec_mask(
            B, pos.device, lot_id=(lot_id if isinstance(lot_id, int) else lot_g), pbc=False, use_elec_context=use_elec_context
        )
        elec_f = elec_mask_b.to(pos.dtype)  # [B]
        static_no_elec = self._static_no_elec(lot_id, False, use_elec_context)

        R, q_isl, q_seed, w_bond, w_lr = None, None, None, None, None
        state = self.embedding(z)
        if not static_no_elec:
            if self.split_mode:
                w_bond = self._bond_weights_dense(z, dist, pmf, r_cut=rc)
                q_seed = self._split_seed_dense(z, dist, pmf, mask_f, total_charge, atom_charge, dist.dtype)
                rc_elec = float(self.taper_elec.r_cut) if getattr(self, "taper_elec", None) is not None else 18.0
                if self.mesh_def:
                    w_lr = self._ic_kernels[self._widest].real(dist) * pmf
                else:
                    w_lr = taper_envelope(dist, max(rc, rc_elec - 1.0), rc_elec, True) * pmf
                q_isl = q_seed
                if self.spin_embedding is not None:
                    e_sp = self._spin_embed_split_dense(state, z, q_seed, w_bond, mask_f, spin)
                    state = state + elec_f.view(B, 1, 1) * e_sp
                if self.ic_index and self.ic_mode == "island" and not single_island:
                    R = self._islands_dense(dist.detach(), pair_mask, mask, r_cut=rc)
                    q_isl = torch.bmm(R, q_seed.unsqueeze(-1)).squeeze(-1) * mask_f.squeeze(-1)
            elif self.island_mode:
                # single_island: R = m m^T (what _islands_dense returns for one island per graph) is never built; None = graph sums
                R = None if single_island else self._islands_dense(dist.detach(), pair_mask, mask, r_cut=rc)
                q_isl, spin_rows = self._island_inputs(R, z, mask_f, total_charge, atom_charge, spin, dist.dtype)
                if spin_rows is not None:
                    state = state + elec_f.view(B, 1, 1) * self.spin_embedding(spin_rows)
            else:
                if self.charge_embedding is not None:
                    state = state + elec_f.view(B, 1, 1) * self.charge_embedding((total_charge.view(B, 1) / n_atoms).unsqueeze(-1))
                if self.spin_embedding is not None:
                    state = state + elec_f.view(B, 1, 1) * self.spin_embedding(self._spin_index(spin, B, pos.device)).unsqueeze(1)
        elif e_lot is None and self.spin_embedding is not None:
            state = state + self.spin_embedding(self._spin_index(spin, B, pos.device)).unsqueeze(1)
        if e_lot is not None:
            if self.lot_atom_proj is not None:
                state = state + self.lot_atom_proj(e_lot)
        state = state * mask_f

        if use_sparse_sr:
            N_flat = B * N + 1
            recv, send = edge_index[0], edge_index[1]
            if "gather" in self.fast_ops:
                at_r, at_s = (lambda t: t.index_select(0, recv)), (lambda t: t.index_select(0, send))
            else:
                at_r, at_s = (lambda t: t[recv]), (lambda t: t[send])
            pos_flat = torch.cat([(pos * mask_f).reshape(B * N, 3), pos.new_zeros((1, 3))], dim=0)
            z_flat = torch.cat([(z * mask.long()).reshape(B * N), z.new_zeros(1)], dim=0)
            es = edge_shift if edge_shift is not None else pos.new_zeros((recv.shape[0], 3))
            diff_e = at_r(pos_flat) - at_s(pos_flat) - es
            dist_e = (diff_e.pow(2).sum(-1) + 1e-8).sqrt()
            unit_e = diff_e / (dist_e.unsqueeze(-1) + 1e-7)
            if self.zbl_core:
                dist_e = self._clamp_sparse(z_flat, dist_e, recv, send)
            if rc == self.r_cut and ro == self.r_on and c3 == self.c3_taper:
                chi_e = self.envelope(dist_e)
            else:
                chi_e = taper_envelope(dist_e, ro, rc, c3)
            state_flat0 = torch.cat([state.reshape(B * N, -1), state.new_zeros((1, state.shape[-1]))], dim=0)
            ew = self.edge_weights
            norm_state = F.rms_norm(state_flat0, (state_flat0.shape[-1],), eps=1e-6)
            key_e, query_e = ew.qk(norm_state).chunk(2, -1)
            modulation = torch.tanh(ew.scale * (at_r(query_e) * at_s(key_e)).sum(-1) * 0.25)
            degree = dist_e.new_zeros(N_flat).index_add_(0, recv, chi_e)
            deg_denom = (degree + 1.0) if c3 else degree.clamp_min(1.0)
            mean_mod = dist_e.new_zeros(N_flat).index_add_(0, recv, chi_e * modulation) / deg_denom
            w_e = chi_e * (1.0 + 0.49 * (modulation - at_r(mean_mod)))
            rbf_dist_e = dist_e
            if use_pr:
                pr = (
                    self._baked_pair_radius
                    if (
                        not self.training
                        and self._baked_pair_radius is not None
                        and self._baked_pair_radius.device == z_flat.device
                        and self._baked_pair_radius.dtype == dist_e.dtype
                        and baked_current(self._baked_pr_version, self.pair_log_radius._version)
                    )
                    else self.pair_log_radius.exp()
                )
                rho_e = pr.index_select(0, z_flat) if "gather" in self.fast_ops else pr[z_flat]
                s_ij_e = 1.52 / (at_r(rho_e) + at_s(rho_e))
                if self.pair_radius_warp:
                    rbf_dist_e = (s_ij_e * dist_e) / (1.0 + (s_ij_e - 1.0) * (dist_e / rc)).clamp_min(1e-6)
                else:
                    rbf_dist_e = dist_e * s_ij_e
            rbf_e = self.rbf(rbf_dist_e)
            if self.pair_basis_mlp is not None:
                rbf_e = self._pair_basis(rbf_e, z_flat[recv], z_flat[send])
            edge_e = rbf_e * w_e.unsqueeze(-1)
            g = {
                "recv": recv,
                "send": send,
                "norm": torch.rsqrt(degree + 1.0).unsqueeze(-1),
                "edge": edge_e,
                "unit": unit_e,
                "y2": l2_pair(unit_e, unit_e),
            }
            if self.use_l3:
                g["y3"] = l3_harmonic(unit_e)
            if self.use_l4:
                g["y4"] = l4_harmonic(unit_e)
            if self.edge_heads:
                g["rbf"] = rbf_e
                g["chi"] = chi_e
                g["deg1"] = (degree + 1.0).unsqueeze(-1)
            edge = None
        else:
            w = self.edge_weights(state, chi, smooth_degree=c3)  # [B, N, N]
            rbf_dist = dist
            if use_pr:  # C-C-equivalent distance for the basis only (see pair_radius)
                pr = (
                    self._baked_pair_radius
                    if (
                        not self.training
                        and self._baked_pair_radius is not None
                        and self._baked_pair_radius.device == z.device
                        and self._baked_pair_radius.dtype == dist.dtype
                        and baked_current(self._baked_pr_version, self.pair_log_radius._version)
                    )
                    else self.pair_log_radius.exp()
                )
                rho = pr[z]  # [B, N]
                s_ij = 1.52 / (rho.unsqueeze(2) + rho.unsqueeze(1))
                if self.pair_radius_warp:
                    rbf_dist = (s_ij * dist) / (1.0 + (s_ij - 1.0) * (dist / rc)).clamp_min(1e-6)
                else:
                    rbf_dist = dist * s_ij
            rbf = self.rbf(rbf_dist)
            if self.pair_basis_mlp is not None:
                rbf = self._pair_basis(rbf, z.unsqueeze(2).expand(B, N, N), z.unsqueeze(1).expand(B, N, N))
            edge = rbf * w.unsqueeze(-1)  # [B, N, N, R]
            g = {
                "mask_f": mask_f,
                "norm": torch.rsqrt(chi.sum(-1, keepdim=True) + 1.0),
                "edge": edge,
                "edge_t": edge.permute(0, 1, 3, 2).reshape(B, N * edge.shape[-1], N) if N > edge.shape[-1] else None,
                "unit": unit,
                "y2": (l2_pair_mm if "l2" in self.fast_ops else l2_pair)(unit, unit),
            }
            if self.use_l3:
                g["y3"] = l3_harmonic(unit)
            if self.use_l4:
                g["y4"] = l4_harmonic(unit)
            if self.edge_heads:
                g["rbf"] = rbf
                g["chi"] = chi
                g["deg1"] = chi.sum(-1, keepdim=True) + 1.0

        vectors = state.new_zeros(B, N, self.vector_rank, 3)
        atomic = state.new_zeros(B, N)
        fb_q: list[Tensor] = []  # the charge-feedback charges, one [B, N] per feedback point
        lot_delta = None  # the token's energy-only correction to the scalar state (lot_energy_stages)
        pool = None
        q_ic = q_isl
        if not static_no_elec and self.ic_index:
            if self.ic_mode == "window":
                eye_m = torch.diag_embed(mask.to(dist.dtype))
                if self.mesh_def:
                    pool = MultiWindowPool([IslandPool.dense_matrix(k.real(dist) * pmf + eye_m, mask)
                                            for k in self._ic_kernels], self._widest, sparse=False)
                else:
                    w_ic = taper_envelope(dist, self.ic_r_on, self.ic_r_cut, True) * pmf
                    pool = IslandPool.dense_matrix(w_ic + eye_m, mask)
                seed_for_ic = (
                    atom_charge.to(dist.dtype) * mask_f.squeeze(-1)
                    if atom_charge is not None
                    else total_charge.to(dist.dtype).view(B, 1) / n_atoms * mask_f.squeeze(-1)
                )
                q_ic = pool.sum(seed_for_ic) * mask_f.squeeze(-1)
            else:
                pool = IslandPool.dense_single(mask) if R is None else IslandPool.dense_matrix(R, mask)
        use_mol_ro = self._is_mol_domain(pbc=False, lot_id=lot_id, use_elec_context=use_elec_context)
        stage_ro = self.stage_readouts_mol if use_mol_ro else self.stage_readouts
        ro_head = self.readout_mol if use_mol_ro else self.readout
        for l, stage in enumerate(self.stages_list):
            k = self._lot_slot.get(l)
            lot_b = self.lot_swiglu_projs[k](e_lot) if (e_lot is not None and k is not None) else None
            k_e = self._lot_e_slot.get(l)
            if use_sparse_sr:
                s_in = torch.cat([state.reshape(B * N, -1), state.new_zeros((1, state.shape[-1]))], dim=0)
                v_in = torch.cat(
                    [vectors.reshape(B * N, self.vector_rank, 3), vectors.new_zeros((1, self.vector_rank, 3))],
                    dim=0,
                )
                lb_flat = (
                    torch.cat([lot_b.expand(B, N, -1).reshape(B * N, -1), lot_b.new_zeros((1, lot_b.shape[-1]))], dim=0)
                    if lot_b is not None
                    else None
                )
                if e_lot is not None and k_e is not None:
                    lbe = self.lot_energy_projs[k_e](e_lot)
                    lbe_flat = torch.cat([lbe.expand(B, N, -1).reshape(B * N, -1), lbe.new_zeros((1, lbe.shape[-1]))], dim=0)
                    s_out, v_out, d_e_flat = stage.forward_sparse(
                        s_in, v_in, g, lot_bias=lb_flat, lot_bias_energy=lbe_flat
                    )
                    d_e = d_e_flat[:-1].reshape(B, N, -1) * mask_f
                    lot_delta = d_e if lot_delta is None else lot_delta + d_e
                else:
                    s_out, v_out = stage.forward_sparse(s_in, v_in, g, lot_bias=lb_flat)
                state = s_out[:-1].reshape(B, N, -1) * mask_f
                vectors = v_out[:-1].reshape(B, N, self.vector_rank, 3) * mask_f.unsqueeze(-1)
            else:
                if e_lot is not None and k_e is not None:
                    state, vectors, d_e = stage(state, vectors, g, lot_bias=lot_b,
                                                lot_bias_energy=self.lot_energy_projs[k_e](e_lot))
                    lot_delta = d_e if lot_delta is None else lot_delta + d_e
                else:
                    state, vectors = stage(state, vectors, g, lot_bias=lot_b)
            if self.linear_mixers is not None:
                state = self.linear_mixers[l](state, mask)
            if l < len(stage_ro):
                atomic = atomic + stage_ro[l](state).squeeze(-1)
            if not static_no_elec and l < self.feedback_points:
                if self.split_mode:
                    state, q_fb = self._feedback_split_dense(
                        l,
                        state,
                        q_seed,
                        w_bond,
                        mask_f,
                        total_charge=total_charge,
                        w_lr=w_lr,
                        elec_f=elec_f,
                        seeded=(atom_charge is not None),
                    )
                elif self.island_mode:
                    s_fb, q_fb = self._feedback_island(l, state, q_isl, R, mask_f)
                    q_fb = q_fb * elec_f.view(B, 1)
                    state = (state + elec_f.view(B, 1, 1) * (s_fb - state)) * mask_f
                else:
                    s_fb, q_fb = self._feedback_dense(l, state, total_charge, mask_f)
                    q_fb = q_fb * elec_f.view(B, 1)
                    state = (state + elec_f.view(B, 1, 1) * (s_fb - state)) * mask_f
                fb_q.append(q_fb)
            if not static_no_elec and l in self.ic_index:
                s_ic = self.island_ctx[self.ic_index[l]](state, pool, q_ic)
                state = (state + elec_f.view(B, 1, 1) * (s_ic - state)) * mask_f
        for stage, a_s, a_v in zip(self.gated_stages_list, self.gate_s, self.gate_v):
            if use_sparse_sr:
                s_in = torch.cat([state.reshape(B * N, -1), state.new_zeros((1, state.shape[-1]))], dim=0)
                v_in = torch.cat(
                    [vectors.reshape(B * N, self.vector_rank, 3), vectors.new_zeros((1, self.vector_rank, 3))],
                    dim=0,
                )
                s_out, v_out = stage.forward_sparse(s_in, v_in, g)
                s_new = s_out[:-1].reshape(B, N, -1)
                v_new = v_out[:-1].reshape(B, N, self.vector_rank, 3)
            else:
                s_new, v_new = stage(state, vectors, g)
            state = (state + a_s * (s_new - state)) * mask_f
            vectors = vectors + a_v.view(1, 1, -1, 1) * (v_new - vectors)
        s_ro = state if lot_delta is None else state + lot_delta  # energy readout only; charges read `state`
        x_ro = s_ro if self.vector_rank == 0 else torch.cat([s_ro, _pair_inv(vectors, self.read_u, self.read_v)], dim=-1)
        if e_lot is not None:
            x_ro = x_ro + self.lot_readout_proj(e_lot)
        ro_out = ro_head(x_ro).squeeze(-1)
        if lot_g is not None:
            ro_out = ro_out + self.lot_readout_bias(lot_g)
        atomic = atomic + ro_out
        if self.null_probe is not None:
            atomic = atomic + self.null_probe(x_ro).squeeze(-1)
        atomic = atomic * mask_f.squeeze(-1)
        energy = atomic.sum(dim=-1) * self.output_scale
        if self.band is not None:
            if self.island_mode and not single_island:
                raise ValueError("band_energy needs single_island=True in island mode (one Fermi level per graph; "
                                 "per-island Fermi levels are not implemented)")
            energy = energy + self.band(state, edge, chi, pmf, mask_f, total_charge, spin)  # not scaled by output_scale
        if self.pair_spline_out is not None:
            coef = self._pair_spline_coef(z.unsqueeze(2).expand(B, N, N), z.unsqueeze(1).expand(B, N, N))
            energy = energy + 0.5 * ((coef * self.rbf(dist)).sum(-1) * chi * pmf).sum(dim=(-1, -2))

        q = None
        dipole = None
        e_elec = energy.new_zeros(B)
        if not static_no_elec and self.charge_head is not None:
            m_flat = mask_f.squeeze(-1)
            q_raw = self._bounded(self.charge_head(state).squeeze(-1), self.charge_bound) * m_flat
            if self.split_mode:
                f = F.softplus(self.nqe_head(state).squeeze(-1)) * m_flat
                phi = self.charge_head_transfer(state).squeeze(-1) * m_flat if self.charge_head_transfer is not None else None
                q = self._equilibrate_split_dense(
                    q_raw,
                    f,
                    q_seed,
                    w_bond,
                    m_flat,
                    total_charge=total_charge,
                    phi=phi,
                    w_lr=w_lr,
                    seeded=(atom_charge is not None),
                )
            elif self.island_mode:
                f = F.softplus(self.nqe_head(state).squeeze(-1)) * m_flat
                q = self._equilibrate(q_raw, f, q_isl, R, m_flat)
            elif self.nqe_head is None:
                q = (q_raw - q_raw.sum(-1, keepdim=True) / n_atoms + total_charge.view(B, 1) / n_atoms) * m_flat
            else:
                f = F.softplus(self.nqe_head(state).squeeze(-1)) * m_flat
                q = (q_raw + f * (total_charge.view(B, 1) - q_raw.sum(-1, keepdim=True)) / f.sum(-1, keepdim=True)) * m_flat
            if self.qeq_head is not None:
                q, e_elec = self._qeq_dense(state, z, dist, pmf, m_flat,
                                            q_isl if self.island_mode else total_charge.view(B, 1).expand(B, N), R)
            else:
                kern = self._coulomb_full(dist) if self.mesh_def else self._vacuum_coulomb_kernel(dist)
                pair = q.unsqueeze(2) * q.unsqueeze(1) * kern * pmf
                e_elec = 0.5 * self.coulomb_k * pair.sum(dim=(-1, -2))
            q = q * elec_f.view(B, 1)
            e_elec = e_elec * elec_f
            energy = energy + e_elec
            dipole = latent_dipole(q, pos, mask)

        if self.zbl_core:
            energy = energy + self._repulsion_energy_dense(z, dist_raw, pmf)

        forces = None
        if direct_forces:
            forces = torch.einsum("bnk,bnkc->bnc", self.force_head(state), vectors) * mask_f * self.output_scale
        elif compute_gradient_forces:
            grad = torch.autograd.grad(energy.sum(), pos, create_graph=self.training, retain_graph=self.training)[0]
            forces = -grad * mask_f
        return {"energy": energy, "forces": forces, "electrostatic_energy": e_elec, "charges": q,
                "dipole": dipole, "feedback_charges": torch.stack(fb_q) if fb_q else None,
                "dispersion_energy": energy.new_zeros(B), "homo": None, "lumo": None, "gap": None}

    def forward_sparse(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        n_graphs: int,
        edge_index: Tensor,
        edge_shift: Tensor,
        total_charge: Tensor | None = None,
        lr_edge_index: Tensor | None = None,
        lr_edge_shift: Tensor | None = None,
        compute_gradient_forces: bool = True,
        spin: Tensor | None = None,
        atom_charge: Tensor | None = None,
        group: Tensor | None = None,
        group_charge: Tensor | None = None,
        group_neutralize: bool = True,
        group_embed: bool = True,
        single_island: bool = False,
        pbc: bool = False,
        lot_id: Tensor | int | None = None,
        edge_diff: Tensor | None = None,
        r_cut: float | None = None,
        r_on: float | None = None,
        c3_taper: bool | None = None,
        use_pair_radius: bool | None = None,
        use_elec_context: bool | None = None,
        unwrap_dipole: bool = False,
        cell: Tensor | None = None,
        **_: object,
    ) -> dict[str, Tensor | None]:
        """Neighbour-list forward, O(N) instead of the dense O(N^2); same weights, same energies.

        single_island (island mode): the caller guarantees every graph is one island (as in the OMol25 training rows), so
        the island search (iterative, with host syncs) is skipped and the graph is the island.

        LoT-based electrostatic/context gate (`_elec_mask`): ON per structure for `omol_wb97mv` (`lot_id in self.elec_lot_ids`)
        with or without PBC; OFF for crystal LoTs (`lot_id in 0..5`). Falls back to `not pbc` when `lot_id` is None.

        edge_diff [E, 3] (optional): pre-computed r_recv - r_send - edge_shift tensor (used by Direct Edge-Vector
        Virial to differentiate energy directly with respect to edge vectors without [E, 3, 3] strain bmm's).
        """
        if self.band is not None:
            raise NotImplementedError("band_energy: dense path only")
        if self.linear_mixers is not None:
            raise NotImplementedError("attention_mixer='linear': dense path only (forward_sparse has no mixer)")
        if edge_diff is not None:
            if not pbc:
                raise ValueError("edge_diff direct virial mode requires pbc=True")
            pos = pos.new_empty((0, 3))  # Guard: any accidental direct read of pos[i] will fail immediately
        elif compute_gradient_forces and not pos.requires_grad:
            pos = pos.detach().requires_grad_(True)
        N = z.shape[0]
        B = int(n_graphs)
        rc = self.r_cut if r_cut is None else float(r_cut)
        ro = self.r_on if r_on is None else float(r_on)
        c3 = self.c3_taper if c3_taper is None else bool(c3_taper)
        use_pr = (self.pair_log_radius is not None) if use_pair_radius is None else (
            bool(use_pair_radius) and self.pair_log_radius is not None
        )

        # Geometry on edges; same conventions as the dense path (diff = r_i - r_j, eps in dist and unit)
        recv, send = edge_index[0], edge_index[1]
        if "gather" in self.fast_ops:  # index_select: same values, faster on MPS (see ACEStage.forward_sparse)
            at_r, at_s = (lambda t: t.index_select(0, recv)), (lambda t: t.index_select(0, send))
        else:
            at_r, at_s = (lambda t: t[recv]), (lambda t: t[send])
        diff = edge_diff if edge_diff is not None else (at_r(pos) - at_s(pos) - edge_shift)
        dist = (diff.pow(2).sum(-1) + 1e-8).sqrt()
        unit = diff / (dist.unsqueeze(-1) + 1e-7)
        if self.zbl_core:
            dist_raw = dist
            dist = self._clamp_sparse(z, dist, recv, send)
        if rc == self.r_cut and ro == self.r_on and c3 == self.c3_taper:
            chi = self.envelope(dist)
        else:
            chi = taper_envelope(dist, ro, rc, c3)

        lot_g = self._resolve_lot_ids(lot_id, B, z.device)
        e_lot_g = self.lot_embedding(lot_g) if lot_g is not None else None  # [B, d_lot]
        static_no_elec = self._static_no_elec(lot_id, pbc, use_elec_context)
        run_elec = not static_no_elec and (self.use_electrostatics or bool(self.ic_index) or bool(self.feedback_points))
        if run_elec:
            elec_mask_b = self._elec_mask(
                B, z.device, lot_id=(lot_id if isinstance(lot_id, int) else lot_g), pbc=pbc, use_elec_context=use_elec_context
            )
            elec_f_b = elec_mask_b.to(dist.dtype)  # [B]
            elec_f_atom = elec_f_b if B == 1 else elec_f_b[batch]  # [1] or [N]
            if not pbc and lr_edge_index is None and not self.mesh_def and bool(elec_mask_b.any()):
                raise ValueError(
                    "MenaACE.forward_sparse(pbc=False) requires an explicit lr_edge_index when electrostatics or "
                    "windowed IslandContext is active; refusing to silently reuse short-range edge_index."
                )
        else:
            elec_mask_b = None
            elec_f_b = None
            elec_f_atom = None

        state = self.embedding(z)
        G = B
        q_per_atom = None
        n_group = None
        w_bond = None
        lr_r = None
        lr_s = None
        r_lr_pre = None
        w_lr_edge = None
        use_mesh = bool(run_elec and self.mesh_def and (pbc or lr_edge_index is None))
        mctx = None
        win_sum = None
        if use_mesh:  # mesh backend: no long-range edges at all (lr_edge_index, if given, serves D3 only)
            mctx = MeshContext(pos, batch, B, cell, pbc, self.mesh_spacing, self.mesh_order)
            win_sum = (lambda x: mctx.conv(x, [self._ic_kernels[self._widest]]))
        if run_elec:
            if use_mesh:
                lr_r = lr_s = r_lr_pre = None
            elif lr_edge_index is not None:
                lr_r, lr_s = lr_edge_index[0], lr_edge_index[1]
                lr_es = lr_edge_shift if lr_edge_shift is not None else pos.new_zeros((lr_r.shape[0], 3))
                r_lr_pre = ((pos[lr_r] - pos[lr_s] - lr_es).pow(2).sum(-1) + 1e-8).sqrt()
                if self.zbl_core:
                    r_lr_pre = self._clamp_sparse(z, r_lr_pre, lr_r, lr_s)
            else:
                lr_r, lr_s = recv, send
                r_lr_pre = dist
            if self.split_mode:
                w_bond = self._bond_weights_sparse(z, dist, recv, send, r_cut=rc)
                q_per_atom = self._split_seed_sparse(
                    z, dist, recv, send, batch, B, total_charge, group, group_charge, atom_charge, dist.dtype
                )
                rc_elec = float(self.taper_elec.r_cut) if getattr(self, "taper_elec", None) is not None else 18.0
                if use_mesh:
                    w_lr_edge = None
                elif self.mesh_def:
                    w_lr_edge = self._ic_kernels[self._widest].real(r_lr_pre)
                else:
                    w_lr_edge = taper_envelope(r_lr_pre, max(rc, rc_elec - 1.0), rc_elec, True)
                if self.spin_embedding is not None:
                    e_sp = self._spin_embed_split_sparse(state, z, q_per_atom, w_bond, recv, send, batch, B, spin)
                    state = state + elec_f_atom.unsqueeze(-1) * e_sp
                if self.ic_index and self.ic_mode == "island":
                    if single_island:
                        group, G = batch, B
                    else:
                        group = self._islands_sparse(N, recv, send, dist.detach(), r_cut=rc)
                        G = int(group.max().item()) + 1 if N else 0
                    group_charge = dist.new_zeros(G).index_add_(0, group, q_per_atom.detach())
            else:
                if atom_charge is not None:
                    q_per_atom = atom_charge.to(dist.dtype)
                    if group is None:
                        group = batch
                        group_charge = graph_sum(dist, B, batch, q_per_atom.detach())
                    G = int(group_charge.shape[0])
                    n_group = dist.new_zeros(G).index_add_(0, group, dist.new_ones(N)).clamp_min(1.0)
                else:
                    if group is None:
                        group = batch
                        group_charge = total_charge if total_charge is not None else dist.new_zeros(B)
                    G = int(group_charge.shape[0])
                    n_group = dist.new_zeros(G).index_add_(0, group, dist.new_ones(N)).clamp_min(1.0)  # [G]
                    q_per_atom = (group_charge.to(dist.dtype) / n_group)[group]  # [N]
                q_embed = q_per_atom
                if not group_embed:
                    n_graph = graph_sum(dist, B, batch, dist.new_ones(N)).clamp_min(1.0)
                    q_graph = graph_sum(dist, B, batch, q_per_atom)  # = the graph's total charge
                    q_embed = (q_graph / n_graph)[batch]

                if self.island_mode:
                    if single_island:
                        group, G = batch, B
                    else:
                        group = self._islands_sparse(N, recv, send, dist.detach(), r_cut=rc)
                        G = int(group.max().item()) + 1 if N else 0
                    group_charge = dist.new_zeros(G).index_add_(0, group, q_per_atom.detach())
                    n_group = dist.new_zeros(G).index_add_(0, group, dist.new_ones(N)).clamp_min(1.0)
                    q_per_atom = (group_charge / n_group)[group]
                    group_neutralize = True
                    if self.spin_embedding is not None:
                        isl_graph = batch.new_zeros(G).scatter_reduce(0, group, batch, reduce="amax", include_self=False)
                        n_isl = torch.zeros(B, dtype=torch.long, device=z.device).index_add_(0, isl_graph, torch.ones_like(isl_graph))
                        electrons = (dist.new_zeros(G).index_add_(0, group, z.to(dist.dtype)) - group_charge).round().long()
                        rows = torch.where(n_isl[batch] == 1, self._spin_index(spin, B, z.device)[batch],
                                           (electrons.remainder(2) + 1)[group].clamp(0, self.max_spin))
                        if spin is None:
                            rows = torch.zeros_like(rows)
                        state = state + elec_f_atom.unsqueeze(-1) * self.spin_embedding(rows)
                else:
                    if self.charge_embedding is not None:
                        state = state + elec_f_atom.unsqueeze(-1) * self.charge_embedding(q_embed.unsqueeze(-1))
                    if self.spin_embedding is not None:
                        state = state + elec_f_atom.unsqueeze(-1) * self.spin_embedding(self._spin_index(spin, B, z.device))[batch]
        else:
            # Crystal LoTs (lot_id in 0..5): use legacy spin_embedding only when LoT conditioning is not supplied
            if e_lot_g is None and self.spin_embedding is not None:
                sp_emb = self.spin_embedding(self._spin_index(spin, B, z.device))
                state = state + (sp_emb if B == 1 else sp_emb[batch])

        if e_lot_g is not None:
            if self.lot_atom_proj is not None:
                lot_atom = self.lot_atom_proj(e_lot_g)
                state = state + (lot_atom if B == 1 else lot_atom[batch])

        # ChemicalEdgeWeights on edges: per-edge query_i . key_j, centred over each receiver's neighbours
        ew = self.edge_weights
        norm_state = F.rms_norm(state, (state.shape[-1],), eps=1e-6)
        key, query = ew.qk(norm_state).chunk(2, -1)
        modulation = torch.tanh(ew.scale * (at_r(query) * at_s(key)).sum(-1) * 0.25)
        degree = dist.new_zeros(N).index_add_(0, recv, chi)
        deg_denom = (degree + 1.0) if c3 else degree.clamp_min(1.0)
        mean_mod = dist.new_zeros(N).index_add_(0, recv, chi * modulation) / deg_denom
        fz = getattr(self, "diag_freeze", None)  # diagnostic only; never set in training
        if fz is not None:  # first call stores the per-atom degree / neighbour mean, later calls reuse them as constants
            if "degree" in fz.get("what", ()):
                degree = fz.setdefault("degree", degree.detach().clone())
            if "mean" in fz.get("what", ()):
                mean_mod = fz.setdefault("mean_mod", mean_mod.detach().clone())
        w = chi * (1.0 + 0.49 * (modulation - at_r(mean_mod)))
        rbf_dist = dist
        if use_pr:  # C-C-equivalent distance for the basis only (see pair_radius)
            pr = (
                self._baked_pair_radius
                if (
                    not self.training
                    and self._baked_pair_radius is not None
                    and self._baked_pair_radius.device == z.device
                    and self._baked_pair_radius.dtype == dist.dtype
                    and baked_current(self._baked_pr_version, self.pair_log_radius._version)
                )
                else self.pair_log_radius.exp()
            )
            rho = pr.index_select(0, z) if "gather" in self.fast_ops else pr[z]  # [N]
            s_ij = 1.52 / (at_r(rho) + at_s(rho))
            if self.pair_radius_warp:
                rbf_dist = (s_ij * dist) / (1.0 + (s_ij - 1.0) * (dist / rc)).clamp_min(1e-6)
            else:
                rbf_dist = dist * s_ij
        rbf = self.rbf(rbf_dist)
        if self.pair_basis_mlp is not None:
            rbf = self._pair_basis(rbf, z[recv], z[send])
        edge = rbf * w.unsqueeze(-1)  # [E, R]
        g = {"recv": recv, "send": send, "norm": torch.rsqrt(degree + 1.0).unsqueeze(-1), "edge": edge,
             "unit": unit, "y2": l2_pair(unit, unit)}
        if self.use_l3:
            g["y3"] = l3_harmonic(unit)
        if self.use_l4:
            g["y4"] = l4_harmonic(unit)
        if self.edge_heads:
            g["rbf"] = rbf
            g["chi"] = chi
            g["deg1"] = (degree + 1.0).unsqueeze(-1)

        vectors = state.new_zeros(N, self.vector_rank, 3)
        atomic = state.new_zeros(N)
        pool, q_ic = None, None
        lot_delta = None  # the token's energy-only correction to the scalar state (lot_energy_stages)
        if run_elec and self.ic_index:
            if self.ic_mode == "window":
                if self.mesh_def:
                    pool = MultiWindowPool([MeshWindowPool(mctx, k) if use_mesh
                                            else IslandPool.sparse_window(lr_r, lr_s, k.real(r_lr_pre), N)
                                            for k in self._ic_kernels], self._widest, sparse=True)
                else:
                    w_ic = taper_envelope(r_lr_pre, self.ic_r_on, self.ic_r_cut, True)
                    pool = IslandPool.sparse_window(lr_r, lr_s, w_ic, N)
                if atom_charge is not None:
                    seed_for_ic = atom_charge.to(dist.dtype)
                elif total_charge is not None:
                    n_b = graph_sum(dist, B, batch, dist.new_ones(N)).clamp_min(1.0)
                    seed_for_ic = (total_charge.to(dist.dtype).view(B) / n_b)[batch]
                else:
                    seed_for_ic = q_per_atom
                q_ic = pool.sum(seed_for_ic)
            else:
                pool = IslandPool.sparse(group, G)
                q_ic = group_charge.to(dist.dtype)[group]
        use_mol_ro = self._is_mol_domain(pbc=pbc, lot_id=lot_id, use_elec_context=use_elec_context)
        stage_ro = self.stage_readouts_mol if use_mol_ro else self.stage_readouts
        ro_head = self.readout_mol if use_mol_ro else self.readout
        for l, stage in enumerate(self.stages_list):
            k = self._lot_slot.get(l)
            if e_lot_g is not None and k is not None:
                lb = self.lot_swiglu_projs[k](e_lot_g)
                lot_b = lb if B == 1 else lb[batch]
            else:
                lot_b = None
            k_e = self._lot_e_slot.get(l)
            if e_lot_g is not None and k_e is not None:
                lbe = self.lot_energy_projs[k_e](e_lot_g)
                lot_be = lbe if B == 1 else lbe[batch]
                state, vectors, d_e = stage.forward_sparse(state, vectors, g, lot_bias=lot_b,
                                                           lot_bias_energy=lot_be)
                lot_delta = d_e if lot_delta is None else lot_delta + d_e
            else:
                state, vectors = stage.forward_sparse(state, vectors, g, lot_bias=lot_b)
            if l < len(stage_ro):
                atomic = atomic + stage_ro[l](state).squeeze(-1)
            if run_elec and l < self.feedback_points:
                if self.split_mode:
                    q_fb = self._feedback_split_sparse(
                        l,
                        state,
                        q_per_atom,
                        w_bond,
                        recv,
                        send,
                        batch,
                        B,
                        total_charge=total_charge,
                        lr_recv=lr_r,
                        lr_send=lr_s,
                        w_lr=w_lr_edge,
                        seeded=(atom_charge is not None),
                        win_sum=win_sum,
                    )
                else:
                    q_fb = self._feedback_charges_sparse(l, state, group, group_charge, q_per_atom, batch, B, group_neutralize)
                q_fb = q_fb * elec_f_atom
                state = state + elec_f_atom.unsqueeze(-1) * self.fb_embed[l](q_fb.unsqueeze(-1))
            if run_elec and l in self.ic_index:
                s_ic = self.island_ctx[self.ic_index[l]](state, pool, q_ic)
                state = state + elec_f_atom.unsqueeze(-1) * (s_ic - state)
        for stage, a_s, a_v in zip(self.gated_stages_list, self.gate_s, self.gate_v):
            s_new, v_new = stage.forward_sparse(state, vectors, g)
            state = state + a_s * (s_new - state)
            vectors = vectors + a_v.view(1, -1, 1) * (v_new - vectors)
        s_ro = state if lot_delta is None else state + lot_delta  # energy readout only; charges read `state`
        x_ro = s_ro if self.vector_rank == 0 else torch.cat([s_ro, _pair_inv(vectors, self.read_u, self.read_v)], dim=-1)
        if e_lot_g is not None:
            ro_proj = self.lot_readout_proj(e_lot_g)
            x_ro = x_ro + (ro_proj if B == 1 else ro_proj[batch])
        ro_out = ro_head(x_ro).squeeze(-1)
        if lot_g is not None:
            ro_bias = self.lot_readout_bias(lot_g).squeeze(-1)
            ro_out = ro_out + (ro_bias if B == 1 else ro_bias[batch])
        atomic = atomic + ro_out
        if self.null_probe is not None:
            atomic = atomic + self.null_probe(x_ro).squeeze(-1)
        energy = graph_sum(dist, B, batch, atomic) * self.output_scale
        if self.env_pair_out is not None:  # every ordered pair in the local list (both directions present)
            p_ = self.env_pair_proj(F.rms_norm(state, (state.shape[-1],), eps=1e-6))
            pi, pj = at_r(p_), at_s(p_)
            coef = self.env_pair_out(torch.cat([pi * pj, pi + pj], -1))  # [E, R]
            energy = energy + 0.5 * graph_sum(dist, B, at_r(batch), (coef * rbf).sum(-1) * chi)
        if self.pair_spline_out is not None:  # every ordered pair in the local list (both directions present)
            coef = self._pair_spline_coef(z[recv], z[send])
            energy = energy + 0.5 * graph_sum(dist, B, batch[recv], (coef * self.rbf(dist)).sum(-1) * chi)

        q = None
        dipole = None
        e_elec = energy.new_zeros(B)
        if run_elec and self.qeq_head is not None:
            raise NotImplementedError("qeq is implemented in the dense forward only")
        if run_elec and self.charge_head is not None:
            q_raw = self._bounded(self.charge_head(state).squeeze(-1), self.charge_bound)
            if self.split_mode:
                f = F.softplus(self.nqe_head(state).squeeze(-1))
                phi = self.charge_head_transfer(state).squeeze(-1) if self.charge_head_transfer is not None else None
                q = self._equilibrate_split_sparse(
                    q_raw,
                    f,
                    q_per_atom,
                    w_bond,
                    recv,
                    send,
                    batch,
                    B,
                    total_charge=total_charge,
                    phi=phi,
                    lr_recv=lr_r,
                    lr_send=lr_s,
                    w_lr=w_lr_edge,
                    seeded=(atom_charge is not None),
                    win_sum=win_sum,
                )
            elif self.nqe_head is None:
                if group_neutralize:
                    q_mean = (dist.new_zeros(G).index_add_(0, group, q_raw) / n_group)[group]
                else:
                    n_graph = graph_sum(dist, B, batch, dist.new_ones(N)).clamp_min(1.0)
                    q_mean = (graph_sum(dist, B, batch, q_raw) / n_graph)[batch]
                q = q_raw - q_mean + q_per_atom
            else:
                f = F.softplus(self.nqe_head(state).squeeze(-1))
                if group_neutralize:  # q_i = q_raw_i + f_i (Q_g - sum_g q_raw) / sum_g f, per charge group
                    s_q = dist.new_zeros(G).index_add_(0, group, q_raw)[group]
                    s_f = dist.new_zeros(G).index_add_(0, group, f)[group]
                    q = q_raw + f * (group_charge.to(dist.dtype)[group] - s_q) / s_f
                else:  # groups' formal charges seed q_per_atom; only the mean latent charge of the graph is removed
                    s_q = graph_sum(dist, B, batch, q_raw)[batch]
                    s_f = graph_sum(dist, B, batch, f)[batch]
                    q = q_raw - f * s_q / s_f + q_per_atom
            q = q * elec_f_atom
            li, lj, r = lr_r, lr_s, r_lr_pre
            alp_pbc = self.dsf_alpha if self.dsf_alpha > 0.0 else self.pbc_dsf_alpha
            if use_mesh:  # short part on the 6 A edges + erf(r / sigma) / r on the mesh (Ewald under PBC)
                e_c = graph_sum(energy, B, batch[recv], q[recv] * q[send] * self._coulomb_sr(dist))
                kern_for = ((lambda b: [self._coul_lr]) if pbc
                            else (lambda b: [self._coulomb_open_kernel(mctx.extent(b))]))
                y = mctx.conv(q, [self._coul_lr], kernel_for=kern_for)
                k0 = torch.stack([kern_for(b)[0].real(dist.new_zeros(())) for b in range(B)])
                e_c = e_c + graph_sum(energy, B, batch, q * (y - k0[batch] * q))
                e_elec = 0.5 * self.coulomb_k * e_c * elec_f_b
            else:
                if self.mesh_def:
                    pair = q[li] * q[lj] * self._coulomb_full(r)
                elif pbc and self.dsf_coulomb and alp_pbc > 0.0:
                    pair = q[li] * q[lj] * self._dsf_kernel(r, alpha=alp_pbc)
                else:
                    pair = q[li] * q[lj] * self._vacuum_coulomb_kernel(r)
                e_elec = 0.5 * self.coulomb_k * graph_sum(energy, B, batch[li], pair) * elec_f_b
            energy = energy + e_elec
            if edge_diff is None and not pbc:
                dipole = dist.new_zeros(B, 3).index_add_(0, batch, q.unsqueeze(-1) * pos)
            elif edge_diff is None and pbc and unwrap_dipole:
                dipole = self._unwrap_pbc_dipole(q, pos, diff, w_bond, dist, recv, send, batch, B)

        if self.zbl_core:
            energy = energy + self._repulsion_energy_sparse(z, dist_raw, recv, send, batch, B)

        forces = None
        if compute_gradient_forces and edge_diff is None:
            grad = torch.autograd.grad(energy.sum(), pos, create_graph=self.training, retain_graph=self.training)[0]
            forces = -grad
        return {"energy": energy, "forces": forces, "electrostatic_energy": e_elec, "charges": q,
                "dipole": dipole, "dispersion_energy": energy.new_zeros(B), "homo": None, "lumo": None, "gap": None}
