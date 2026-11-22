"""lam4.heads: the energy readout (stage readouts + final MLP, ref_id path, isolated-atom reference), its per-eval cancellation diagnostic,
and the training-only direct-force head."""
from lam4.heads.force_head import DirectForceHead
from lam4.heads.iso_calib import calibrate_iso_reference, iso_smoothstep
from lam4.heads.readout import N_REF, DirectInvBypass, Readout, SpinDescriptorHead, pair_inv
from lam4.heads.readout_diag import QCLASSES, SIZE_EDGES, cancellation_diagnostic

__all__ = [
    "DirectForceHead", "DirectInvBypass", "N_REF", "Readout", "SpinDescriptorHead",
    "calibrate_iso_reference", "iso_smoothstep",
    "pair_inv", "QCLASSES", "SIZE_EDGES", "cancellation_diagnostic",
]
