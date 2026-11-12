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
    "wb97xd",      # OpenRxn
    "wb97m_d3bj",  # SPICE
)
TOKEN_ID: dict[str, int] = {t: i for i, t in enumerate(TOKENS)}


def token_id(name: str) -> int:
    try:
        return TOKEN_ID[name]
    except KeyError:
        raise KeyError(f"unknown trunk token {name!r}; known: {TOKENS}") from None
