"""One model configuration for lam4. No variants: change a field, get a new hash.

Defaults are the released lam4 / lam5 stack.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    # geometry
    r_cut: float = 6.0
    r_on: float = 5.0            # C3 envelope starts here
    n_rbf: int = 32              # fixed Gaussian centres
    descriptor_floor: bool = True  # chi and the warp read the 5 eV NLH-floored edge length (earlier Mena reads the ZBL-clamped one); pad edges stay inert
    spline_grid: int = 768       # B-spline table knots; set from the measured error table of the C2 tables
    # trunk
    n_stages: int = 3
    d_scalar: int = 384
    swiglu_ratio: float = 2.0
    msg_dim: int = 128
    k_vec: int = 32              # carried l=1 channels
    k2: int = 16                 # transient l=2 channels
    k3: int = 8                  # transient l=3 channels
    vec_c: float = 2.0           # bounded vectors u = v / sqrt(1 + r^2 / c^2)
    state_norm: bool = True      # rms-normalise the scalar state at the end of every stage
    n_heads: int = 4             # per-stage bounded-QK normalised edge weight (zero-init: reduces to the unweighted sum)
    head_dim: int = 16           # channels per head of the q / k projections (q, k: [N, n_heads * head_dim])
    edge_gain_max: float = 1.0   # bounded-QK gain ceiling g_max * tanh(raw_gain / g_max) (1.0 in lam4, 2.0 in lam5)
    ttt_mix: int = 2             # angular branch B1: cubic l=2 invariants tr(M(C T)^3)
    vttv_rank: int = 8           # angular branch B1: v^T M(A T) M(B T) v invariants
    r1_vec: int = 32             # contraction v2 (lam5): l=1 pair invariants <U1 u, V1 u> (128 in lam5)
    r2_ten: int = 16             # contraction v2 (lam5): l=2 pair invariants <U2 T, V2 T> (64 in lam5)
    r3_tvv: int = 16             # contraction v2 (lam5): 3rd-order <T3 T, l2_pair(P3 u, Q3 u)> (64 in lam5)
    k3_read: int = 8             # contraction v2 (lam5): l=3 power spectrum & (O, v, T) read width (32 in lam5)
    inv_norm: bool = False       # contraction v2 (lam5): per-invariant scale buffer + state gate + T.v / vxv frame updates
    r_cut_near: float = 0.0      # split sensor cutoff (lam5): >0 ends l=2, l=3, tr2/lon2, and near s/v channels at r_cut_near (4.0 A)
    r_on_near: float = 3.0       # C3 envelope start for near-cutoff channels
    k_s_far: int = 32            # scalar channels running to r_cut when r_cut_near > 0 (remaining msg_dim - k_s_far end at r_cut_near)
    k_v_far: int = 16            # l=1 channels running to r_cut when r_cut_near > 0 (remaining k_vec - k_v_far end at r_cut_near)
    edge_weight: str = "heads"   # "heads" | "chem" (earlier Mena ChemicalEdgeWeights, equivalence test only)
    domains: tuple[str, ...] = ("crystal", "molecule")   # one node-MLP expert each, chosen per batch
    # conditioning
    n_tokens: int = 6            # lam4.registry.tokens.TOKENS
    cond_dim: int = 64           # c = emb(token)+emb(u)+emb(Q)+emb(S)
    q_max: int = 10
    s_max: int = 11
    # molecular physics (domain == molecule only)
    group_context: bool = True   # pooled [Q/4, n^(-1/3), (M-1)/4] after every stage + E_Q / E_S tables; never on periodic crystals
    # charges and long range (one kernel erf(r/sigma)/r for both domains)
    feedback_points: int = 2
    phi_basis: int = 8           # Gaussian basis of the potential feedback (earlier Mena phi_basis)
    fb_hidden: int = 64          # hidden width of the density feedback MLP([q, m])
    charge_bound: float = 2.0
    coulomb_sigma: float = 2.0
    polarization: bool = True
    aniso_polarizability: bool = False  # lam5: tensor alpha_i = alpha_iso + M(w(s_i) . tau_i) for Raman & anisotropic induced dipoles
    stage_e_weight: float = 0.0         # lam5: weight (relative to lam_e) on partial stage energy sums E^(0), E^(1) so stage terms don't cancel 6:1
    long_range_molecule: str = "exact"     # open molecules N <= cap in training
    long_range_crystal: str = "spme_gated"  # batched fixed-grid SPME with the per-atom screening gate ("spme" ungated, "none" off)
    charge_gate_bias_crystal: float = -3.0  # gate logit bias for crystals at init: g ~ 0.05, so run 1 starts as short-range lam1
    charge_gate_skip: float = 0.02          # inference: skip the mesh when sum_i g_i / N is below this (metals run at short-range speed)
    long_range_inference: str = "pme"      # single-box PME for large-N MD, gated by a mesh-vs-exact test on trained weights
    coulomb_cap: int = 350
    mesh_spacing: float = 0.8
    mesh_order: int = 6
    repulsion: str = "nlh_c2_shift_rmin"
    pair_read: bool = False      # omol4c pair read: F -5% but E +2-3% and an edge MLP; graftable later (zero-init)
    direct_force_head: bool = True  # training-only non-conservative head for the direct-force phase; never used at inference
    # lam6 unified environment-modulated trunk & factorized LoT (defaults preserve lam4/lam5 bit-exactness)
    ffn_mode: str = "domain_experts"   # "domain_experts" | "env_mod_768" | "env_mod_1152"
    env_mod_rank: int = 32
    contract_cumulative_t: bool = False  # lam6: contract bound_vectors(t_new, c) in Stage u2/v2/t3 & AngularBranch
    coord_inv_scale: bool = False        # lam6: exp(-(log_scale_0 + alpha * log1p(dsum_i))) continuous scaling
    coord_alpha_max: float = 2.5         # max exponent bound alpha in (0, coord_alpha_max) or (-coord_alpha_max, coord_alpha_max)
    coord_alpha_signed: bool = False     # lam6 Resume A: signed alpha = coord_alpha_max * tanh(alpha_raw)
    factorized_lot: bool = False         # lam6: (xc_id, paw_id, u_i) per-stage FiLM + Readout FiLM
    n_xc_tokens: int = 12
    n_paw_bases: int = 16
    direct_inv_bypass: bool = False      # lam6: zero-init Stage-2 invariant bypass directly into Readout h
    direct_inv_rank: int = 64
    expose_readout_phi: bool = False     # lam6: expose pre-head phi_i [N, d//2] for Tier B #4 Laplace OOD
    two_channel_spin: bool = False       # lam6: two-channel (u_rad_i >= 0, m_raw_i) spin descriptor head
    use_l2_efg: bool = False             # lam6: l=2 EFG FieldFeedback + EFG coupling
    sqe_pqeq: bool = False               # lam6: implicit PCG-PQEq + scaled SQE electrostatics/polarization
    sqe_k_fix: int = 24                  # lam6: fixed implicit PCG steps (24 recommended when BatchedSPME+EFG active)
    spme_batched: bool = False           # lam6: gated BatchedSPME on periodic cells
    cloud_overlap: bool = False          # lam6: Pauli (>=0) + l=1,2 anisotropic overlap + EFG head
    eda_calib_head: bool = False         # lam6: Tier B #4 zero-leak EDA_CalibHead + X-Ray support
    iso_atom_n1: bool = False            # lam6: (Z, lot, spin) N=1 delta switched on by 1 - smoothstep(dsum, d_lo, d_hi)
    iso_atom_d_lo: float = 0.05
    iso_atom_d_hi: float = 0.5
    spin_exchange: bool = False          # lam6 graft: size-consistent Hund + Heisenberg SpinExchangeHead (molecule only)
    quad_l2_head: bool = False           # lam6 graft: zero-init Stage-2 ten -> [xx,yy,zz,xy,xz,yz] atomic quadrupole
    gap_head: bool = False               # lam6 graft: intensive attention + softmin [gap_a, gap_b] eV head (molecule only)

    def __post_init__(self) -> None:
        if not 0.0 < self.r_on < self.r_cut:
            raise ValueError("need 0 < r_on < self.r_cut")
        if self.r_cut_near > 0.0 and not (0.0 < self.r_on_near < self.r_cut_near <= self.r_cut):
            raise ValueError("need 0 < r_on_near < r_cut_near <= r_cut when r_cut_near > 0")
        if self.r_cut_near > 0.0 and (self.k_s_far % self.n_heads or self.k_v_far % self.n_heads
                                      or not (0 <= self.k_s_far <= self.msg_dim) or not (0 <= self.k_v_far <= self.k_vec)):
            raise ValueError("k_s_far and k_v_far must be divisible by n_heads and within [0, msg_dim / k_vec]")
        if self.n_stages < 1 or self.n_heads < 1 or self.spline_grid < 16:
            raise ValueError("n_stages, n_heads >= 1 and spline_grid >= 16")
        if self.k_vec % self.n_heads or self.d_scalar % self.n_heads or self.msg_dim % self.n_heads:
            raise ValueError("n_heads must divide k_vec, msg_dim and d_scalar")
        if self.edge_weight not in ("heads", "chem"):
            raise ValueError("edge_weight in {heads, chem}")
        if self.repulsion not in ("nlh_c2_shift_rmin", "none"):
            raise ValueError("repulsion in {nlh_c2_shift_rmin, none}")
        for name, allowed in (("long_range_molecule", ("exact", "none")), ("long_range_crystal", ("spme_gated", "spme", "none")),
                              ("long_range_inference", ("pme", "exact", "none"))):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} in {allowed}")
        if not 0 <= self.feedback_points <= self.n_stages - 1:
            raise ValueError("feedback_points must be in [0, n_stages - 1]")
        if self.edge_gain_max <= 0.0:
            raise ValueError("edge_gain_max must be > 0")
        if self.ffn_mode not in ("domain_experts", "env_mod_768", "env_mod_1152"):
            raise ValueError("ffn_mode in {domain_experts, env_mod_768, env_mod_1152}")
        if self.env_mod_rank < 1 or self.direct_inv_rank < 1 or self.coord_alpha_max <= 0.0:
            raise ValueError("env_mod_rank, direct_inv_rank >= 1 and coord_alpha_max > 0")
        if self.n_xc_tokens < self.n_tokens or self.n_paw_bases < 1:
            raise ValueError("n_xc_tokens >= n_tokens and n_paw_bases >= 1")
        if self.sqe_k_fix < 1:
            raise ValueError("sqe_k_fix >= 1")
        if not 0.0 <= self.iso_atom_d_lo < self.iso_atom_d_hi:
            raise ValueError("need 0 <= iso_atom_d_lo < iso_atom_d_hi")

    def hash(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in _LAM4_EXT_DEFAULTS or v != _LAM4_EXT_DEFAULTS[k]}
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @classmethod
    def lam5(cls, **overrides) -> ModelConfig:
        """lam5 preset: Contraction v2 (~370 reads + scale buffer + frame updates),
        conservative-only training, anisotropic tensor polarisability (Raman) and partial stage energy loss (r_cut_near=0.0 under static E)."""
        kw = dict(r1_vec=128, r2_ten=64, r3_tvv=64, k3_read=32, ttt_mix=16, vttv_rank=32, inv_norm=True,
                  r_cut_near=0.0, r_on_near=3.0, k_s_far=32, k_v_far=16, direct_force_head=False,
                  aniso_polarizability=True, stage_e_weight=0.05, edge_gain_max=2.0)
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def lam6(cls, ffn_mode: str = "env_mod_1152", **overrides) -> ModelConfig:
        """lam6 preset: full 6.0 A l<=3 cutoff, Design C continuous env_mod (or Design A fallback), cumulative t_new contraction,
        continuous log1p(dsum) invariant scaling, factorized (xc, paw, u_i) conditioning/readout, DirectInvBypass & two-channel spin."""
        if overrides.get("r_cut_near", 0.0) > 0.0:
            raise ValueError("LAM-6 locks full 6.0 A cutoff for all channels (r_cut_near must be 0.0)")
        kw = dict(r_cut=6.0, r_on=5.0, n_stages=3, r1_vec=128, r2_ten=64, r3_tvv=64, k3_read=32, ttt_mix=16, vttv_rank=32,
                  inv_norm=True, r_cut_near=0.0, r_on_near=3.0, k_s_far=32, k_v_far=16, direct_force_head=False,
                  aniso_polarizability=True, stage_e_weight=0.05, edge_gain_max=2.0, long_range_crystal="spme_gated",
                  ffn_mode=ffn_mode, env_mod_rank=32, contract_cumulative_t=True, coord_inv_scale=True,
                  coord_alpha_max=2.5, factorized_lot=True, n_xc_tokens=12, n_paw_bases=16,
                  direct_inv_bypass=True, direct_inv_rank=64, expose_readout_phi=True, two_channel_spin=True,
                  iso_atom_n1=True, iso_atom_d_lo=0.05, iso_atom_d_hi=0.5)
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def lam6_cloud(cls, ffn_mode: str = "env_mod_1152", **overrides) -> ModelConfig:
        """LAM-6 cloud-overlap preset without SQE/PQEq/SPME/EDA (ExactCoulomb + CloudOverlap + Polarization)."""
        kw = dict(cloud_overlap=True, sqe_pqeq=False, spme_batched=False, eda_calib_head=False)
        kw.update(overrides)
        return cls.lam6(ffn_mode=ffn_mode, **kw)

    @classmethod
    def lam6_physics(cls, ffn_mode: str = "env_mod_1152", **overrides) -> ModelConfig:
        """Full LAM-6 Gate U2+P3 / D1+E1 preset: Design C trunk + l=2 EFG + PCG-SQE/PQEq + BatchedSPME + CloudOverlap + EDA."""
        kw = dict(use_l2_efg=True, sqe_pqeq=True, sqe_k_fix=12, spme_batched=True, cloud_overlap=True, eda_calib_head=True)
        kw.update(overrides)
        return cls.lam6(ffn_mode=ffn_mode, **kw)


_LAM4_EXT_DEFAULTS = {
    "r1_vec": 32, "r2_ten": 16, "r3_tvv": 16, "k3_read": 8, "inv_norm": False,
    "r_cut_near": 0.0, "r_on_near": 3.0, "k_s_far": 32, "k_v_far": 16,
    "aniso_polarizability": False, "stage_e_weight": 0.0, "edge_gain_max": 1.0,
    "ffn_mode": "domain_experts", "env_mod_rank": 32, "contract_cumulative_t": False,
    "coord_inv_scale": False, "coord_alpha_max": 2.5, "coord_alpha_signed": False, "factorized_lot": False,
    "n_xc_tokens": 12, "n_paw_bases": 16, "direct_inv_bypass": False,
    "direct_inv_rank": 64, "expose_readout_phi": False, "two_channel_spin": False,
    "use_l2_efg": False, "sqe_pqeq": False, "sqe_k_fix": 24,
    "spme_batched": False, "cloud_overlap": False, "eda_calib_head": False,
    "iso_atom_n1": False, "iso_atom_d_lo": 0.05, "iso_atom_d_hi": 0.5,
    "spin_exchange": False, "quad_l2_head": False, "gap_head": False,
}


def lam5_config(**overrides) -> ModelConfig:
    return ModelConfig.lam5(**overrides)


def lam6_config(ffn_mode: str = "env_mod_1152", **overrides) -> ModelConfig:
    return ModelConfig.lam6(ffn_mode=ffn_mode, **overrides)


def lam6_cloud_config(ffn_mode: str = "env_mod_1152", **overrides) -> ModelConfig:
    return ModelConfig.lam6_cloud(ffn_mode=ffn_mode, **overrides)


def lam6_physics_config(ffn_mode: str = "env_mod_1152", **overrides) -> ModelConfig:
    return ModelConfig.lam6_physics(ffn_mode=ffn_mode, **overrides)
