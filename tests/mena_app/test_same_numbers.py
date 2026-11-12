"""Every operating system must give the same numbers: energies and forces of both shipped models on a fixed molecule and a
fixed crystal, against references computed on macOS (float64, CPU). The tolerance covers maths-library rounding only;
a wrong energy-reference table or level of theory is off by 0.01 eV or more."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk, molecule

from mena_app import calculator, models

# model, system: (energy eV, max |force| eV/Å, force on atom 0 along x eV/Å)
REFERENCE = {
    ("lam5-v1", "H2O"): (-2079.5201922361757, 4.007078106208074, -0.5269921176349707),
    ("lam5-v1", "Si"): (-10.64697351632294, 1.5470947492283862, -1.5470947492283862),
    ("lam-400k", "H2O"): (-2079.5252396625087, 3.9982709380188064, -0.5249331841446465),
    ("lam-400k", "Si"): (-10.666065381885446, 1.4151816780797446, -1.4151816780797446),
}
TOL_E, TOL_F = 1e-4, 1e-4


def _installed(model_id: str) -> bool:
    return any(m["id"] == model_id and m.get("available") for m in models.all_models())


@pytest.mark.parametrize("model_id,system", sorted(REFERENCE))
def test_energies_and_forces_match_the_reference(model_id, system):
    if not _installed(model_id):
        pytest.skip(f"{model_id} weights not installed")
    atoms = molecule("H2O") if system == "H2O" else bulk("Si", "diamond", 5.43)
    atoms.rattle(0.05, seed=1)
    atoms.calc = calculator(model_id, dtype="float64")
    e, f = atoms.get_potential_energy(), atoms.get_forces()
    e_ref, fmax_ref, f0_ref = REFERENCE[(model_id, system)]
    assert abs(e - e_ref) < TOL_E, f"energy {e!r} vs {e_ref!r}"
    assert abs(float(np.abs(f).max()) - fmax_ref) < TOL_F and abs(float(f[0, 0]) - f0_ref) < TOL_F
