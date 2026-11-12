"""lam4.physics: the molecular-physics modules of the earlier Mena molecular stack, ported onto GraphBatch with inert padding.

Modules: charges (NQE charge head, spin-density head), coulomb_exact (static-slot ExactCoulomb, smeared erf(r/sigma)/r),
feedback (FieldFeedback, DensityFeedback), group_context (GroupContext, ChargeEmbed, GroupReference, group_gate; molecule
domain only, exact identity / zero for crystals), polarization (E_pol and induced dipoles), repulsion (NLH c2_shift_rmin + 5 eV descriptor floor, nlh_tables). Numerical equivalence with the
earlier Mena originals (tests/lam4/test_physics_port.py); each module docstring gives its signature and its earlier Mena state-dict key map.

NOT ported yet: batched fixed-grid SPME for periodic training, single-box PME for large-N inference (earlier Mena PMECoulomb is
not ported), D3 add-back (the trunk assembly is lam4.model, the readout lives in lam4.heads). The descriptor floor
(repulsion.floored_length) is wired into the edge geometry (lam4.core.edge_terms.edge_geometry). The group
context, E_Q/E_S tables and charge embedding are gated to domain == "molecule"; the per-element E0 reference is a registry task.
"""
from lam4.physics.charges import ChargeHead, SpinDensityHead, equilibrate
from lam4.physics.coulomb_exact import (COULOMB_K, DEFAULT_CAP, SIGMA, ExactCoulomb, SlotLayout, assert_capacity,
                                        coulomb_energy, make_slots_static, smeared_kernel)
from lam4.physics.feedback import DensityFeedback, FieldFeedback, soft_clip_vec
from lam4.physics.group_context import (ChargeEmbed, GroupContext, GroupReference, group_domain, group_gate, group_scalars,
                                        load_group_tables)
from lam4.physics.polarization import Polarization
from lam4.physics.repulsion import Repulsion

__all__ = [
    "ChargeHead", "SpinDensityHead", "equilibrate", "COULOMB_K", "DEFAULT_CAP", "SIGMA", "ExactCoulomb", "SlotLayout",
    "assert_capacity", "coulomb_energy", "make_slots_static", "smeared_kernel", "DensityFeedback", "FieldFeedback",
    "soft_clip_vec", "ChargeEmbed", "GroupContext", "GroupReference", "group_domain", "group_gate", "group_scalars",
    "load_group_tables", "Polarization", "Repulsion",
]
