"""lam4.heads: the energy readout (stage readouts + final MLP, ref_id path, isolated-atom reference), its per-eval cancellation diagnostic,
and the training-only direct-force head."""
from lam4.heads.force_head import DirectForceHead
from lam4.heads.readout import N_REF, Readout, pair_inv
from lam4.heads.readout_diag import QCLASSES, SIZE_EDGES, cancellation_diagnostic

__all__ = ["DirectForceHead", "N_REF", "Readout", "pair_inv", "QCLASSES", "SIZE_EDGES", "cancellation_diagnostic"]
