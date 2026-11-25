"""lam4.physics: molecular & unified condensed-phase physics modules (lam5 backward-compatible + LAM-6).

lam5 modules: charges, coulomb_exact, feedback, group_context, polarization, repulsion.
LAM-6 extensions:
  - spme_batched: static-grid BatchedSPME (p=4 analytic splines, live d_cell stress, ScreeningGate, EFG, W_recip)
  - sqe_pqeq: singularity-free scaled SQE + variational fixed-K_fix implicit-adjoint PCG-PQEq
  - cloud_overlap: full-6.0 A learned equivariant Pauli (>=0) + l=1,2 cloud overlap + EFG head
  - xray_eda: Tier B #4 zero-leak EDA_CalibHead, EmpiricalBayesOOD, and Hardy/SPME Cauchy virial X-Ray
"""
from lam4.physics.charges import ChargeHead, SpinDensityHead, equilibrate
from lam4.physics.cloud_overlap import CloudOverlapHead
from lam4.physics.coulomb_exact import (COULOMB_K, DEFAULT_CAP, SIGMA, ExactCoulomb, SlotLayout, assert_capacity,
                                        coulomb_energy, make_slots_static, smeared_kernel)
from lam4.physics.feedback import DensityFeedback, FieldFeedback, soft_clip_vec
from lam4.physics.group_context import (ChargeEmbed, GroupContext, GroupReference, group_domain, group_gate, group_scalars,
                                        load_group_tables)
from lam4.physics.polarization import Polarization
from lam4.physics.repulsion import Repulsion
from lam4.physics.spme_batched import (
    DEFAULT_GRID_BUCKETS, BatchedSPME, ScreeningGate, good_fft_size, select_grid_bucket, select_spme_grid,
)
from lam4.physics.spin_exchange import SpinExchangeHead
from lam4.physics.sqe_pqeq import OpenMultipoleBackend, SQEPQEqHead, fragment_com_positions, register_triton_multipole_fn
from lam4.physics.xray_eda import (EmpiricalBayesOOD, EDA_CalibHead, atomic_strain_metrics, compute_eda_forces,
                                     eda_term_shares_and_stability, evaluate_sapt_ood_calibration, hardy_edge_virial)

__all__ = [
    "ChargeHead", "SpinDensityHead", "equilibrate", "COULOMB_K", "DEFAULT_CAP", "SIGMA", "ExactCoulomb", "SlotLayout",
    "assert_capacity", "coulomb_energy", "make_slots_static", "smeared_kernel", "DensityFeedback", "FieldFeedback",
    "soft_clip_vec", "ChargeEmbed", "GroupContext", "GroupReference", "group_domain", "group_gate", "group_scalars",
    "load_group_tables", "Polarization", "Repulsion",
    "BatchedSPME", "ScreeningGate", "DEFAULT_GRID_BUCKETS", "good_fft_size", "select_grid_bucket", "select_spme_grid",
    "OpenMultipoleBackend", "SQEPQEqHead", "fragment_com_positions", "register_triton_multipole_fn",
    "CloudOverlapHead", "SpinExchangeHead", "EDA_CalibHead", "EmpiricalBayesOOD", "compute_eda_forces", "hardy_edge_virial",
    "atomic_strain_metrics", "eda_term_shares_and_stability", "evaluate_sapt_ood_calibration",
]
