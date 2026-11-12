"""Symmetry-reduced phonons must give what ASE's full 6N + 1 displacement method gives (cheap analytic potential)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.phonons import Phonons

from mena_app import phonon_symmetry, structures
from mena_app.tasks.phonons import _LivePhonons


class _Ctx:
    def check(self):
        pass

    def progress(self, *a, **k):
        pass


def _calc():
    return LennardJones(sigma=2.0, epsilon=0.1, rc=6.0, smooth=True)


def _bands(ph, prim):
    ph.read(acoustic=True)
    path = prim.cell.bandpath(npoints=40)
    return ph.get_band_structure(path, verbose=False).energies[0]


DELTA = 0.002
CASES = [
    ("si", 3), ("mgo", 3), ("srtio3", 3), ("tio2", 2), ("diamond", 1),
    ("hcp_mg", 3), ("wurtzite", 2), ("shifted_nacl", 3),
]


def _atoms(name):
    if name == "hcp_mg":
        return bulk("Mg", "hcp", a=3.2, c=5.2)
    if name == "wurtzite":
        return bulk("ZnO", "wurtzite", a=3.25, c=5.2)
    if name == "shifted_nacl":  # atoms outside [0, 1): the supercell is built from unwrapped positions
        a = structures.load({"format": "builtin", "text": "nacl"})
        a.positions += a.cell[0] * -1.3 + a.cell[2] * 0.7
        return a
    return structures.load({"format": "builtin", "text": name})


@pytest.mark.parametrize("name,k", CASES)
def test_symmetric_matches_full(name, k, tmp_path):
    prim = _atoms(name)
    plan = phonon_symmetry.symmetry_plan(prim, k)
    assert plan is not None and plan.n_evaluations < 6 * len(prim) + 1
    full = Phonons(prim, _calc(), supercell=(k, k, k), delta=DELTA, name=str(tmp_path / "full"))
    full.run()
    sym = _LivePhonons(_Ctx(), plan.n_evaluations, prim, _calc(), supercell=(k, k, k), delta=DELTA,
                       name=str(tmp_path / "sym"), plan=plan)
    sym.measure()
    sym.run()
    assert sym.done == plan.n_evaluations  # only the irreducible supercells reached the calculator
    b_full, b_sym = _bands(full, prim), _bands(sym, prim)
    # Cubic cells: rotations map axes onto axes, so the same supercells are evaluated and only the zero-frequency
    # acoustic modes at Γ differ (1e-10 eV, sqrt of rounding). Hexagonal cells: a 120° rotation infers y from x, and the
    # two central differences carry different O(delta^2) anharmonic error (measured: x100 smaller at delta/10), so they
    # agree to within the finite-difference error of the full method itself.
    assert np.abs(b_full - b_sym).max() < 1e-8 + 1e-4 * np.abs(b_full).max()


def test_no_symmetry_means_no_plan():
    a = bulk("Si", cubic=True)
    a.rattle(0.05, seed=3)
    assert phonon_symmetry.symmetry_plan(a, 2) is None
