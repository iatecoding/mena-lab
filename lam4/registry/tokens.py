"""Trunk conditioning tokens: one per FUNCTIONAL, not per dataset.

lam1 used one LoT id for two jobs: which physics the trunk is conditioned on, and which energy
reference convention applies. lam4 splits them. The trunk token below says "which functional";
the reference convention (E0 table and column) is a separate per-dataset field in `sources.py`.
+U is a structure property (flag derived from composition by the data pipeline), fed to the
conditioning vector like Q and S, so it is not a token.
"""
from __future__ import annotations

TOKENS: tuple[str, ...] = (
    "pbe",         # PBE family: MPtrj, sAlex, OMat24, MatPES-PBE, OC22, OMC25 (D3(0) stripped)
    "r2scan",      # MatPES r2SCAN
    "rpbe",        # OC20
    "wb97mv",      # OMol25
    "wb97xd",      # OpenRxn (legacy lam4/lam5)
    "wb97m_d3bj",  # SPICE
)
TOKEN_ID: dict[str, int] = {t: i for i, t in enumerate(TOKENS)}

# Factorized LAM-6 XC functionals (rows 0..5 match TOKENS bit-for-bit; zero dataset-ID leakage).
XC_TOKENS: tuple[str, ...] = TOKENS + (
    "wb97x",       # Transition1x (wB97X / 6-31G*)
    "b3lyp_d3",    # RGD1 (B3LYP-D3 / TZVP)
    "pbe0",        # QCML (PBE0+MBD / FHI-aims tight)
    "tpssh_d3bj",  # tmQM (TPSSh-D3BJ / def2-SVP)
    "sapt0",       # SPLINTER (SAPT0 EDA anchor)
)
XC_ID: dict[str, int] = {t: i for i, t in enumerate(XC_TOKENS)}

# Factorized LAM-6 PAW POTCAR / basis-set identifiers (decouples PAW/basis offsets from XC).
PAW_BASES: tuple[str, ...] = (
    "vasp_pbe52",    # MPtrj, sAlex (VASP PAW_PBE ENCUT=520)
    "vasp_pbe54",    # OMat24, OMC25, OC20, OC25 (VASP 5.4 POTCARs)
    "vasp_pbe64",    # MatPES-PBE, MatPES-r2SCAN (VASP 6.4 ENCUT=680)
    "aims_tight",    # MAD-1.6 (r2SCAN), QCML (PBE0 FHI-aims tight tier-2)
    "def2_tzvpd",    # OMol25, OPoly26 (ORCA 6 wB97M-V)
    "def2_tzvppd",   # SPICE 2.0.1 (Psi4 wB97M-D3BJ)
    "pop_631gd",     # Transition1x (6-31G(d)) / legacy OpenRxn
    "def2_tzvp",     # RGD1 (TZVP)
    "def2_svp",      # tmQM (def2-SVP)
    "aug_cc_pvddz",  # SPLINTER (aug-cc-pV(D+d)Z)
    "jun_cc_pvdz",   # SPLINTER (jun-cc-pVDZ)
)
PAW_ID: dict[str, int] = {b: i for i, b in enumerate(PAW_BASES)}


def token_id(name: str) -> int:
    try:
        return TOKEN_ID[name]
    except KeyError:
        raise KeyError(f"unknown trunk token {name!r}; known: {TOKENS}") from None


def xc_id(name: str) -> int:
    try:
        return XC_ID[name]
    except KeyError:
        raise KeyError(f"unknown XC functional {name!r}; known: {XC_TOKENS}") from None


def paw_id(name: str) -> int:
    try:
        return PAW_ID[name]
    except KeyError:
        raise KeyError(f"unknown PAW/basis {name!r}; known: {PAW_BASES}") from None


def lot_cell_id(xc: str | int, paw: str | int, u: int, n_paw: int = 16) -> int:
    """Canonical integer index `(xc_id * n_paw + paw_id) * 2 + u` for a physical `(xc, paw, +U)` theory cell."""
    x = xc_id(xc) if isinstance(xc, str) else int(xc)
    p = paw_id(paw) if isinstance(paw, str) else int(paw)
    return (x * int(n_paw) + p) * 2 + int(bool(u))
