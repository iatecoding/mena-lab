"""Source registry for lam4: a typed view over the legacy `lam/data/lot_registry.yaml`.

The legacy YAML stays the single source of truth for E0 files, units and D3 parameters. This
module adds what lam4 needs on top and refuses to load anything it cannot classify:
  trunk token per functional, +U as a structure property, domain (crystal | molecule),
  D3 add-back as a property of the source (independent of any token or electrostatics gate),
  and the default pool share where it has been decided.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lam4.registry.tokens import TOKEN_ID

CRYSTAL, MOLECULE = "crystal", "molecule"
# OMC25 (molecular_crystals) uses the molecular expert (domain = MOLECULE) with periodic boundary conditions (pbc = True; spec section 12).
_DOMAIN_OF_LEGACY = {"bulk": CRYSTAL, "catalysis": CRYSTAL, "molecular_crystals": MOLECULE, "molecules": MOLECULE}
# u_flag: 0 / 1 fixed by the source (MatPES is pure PBE / r2SCAN, no Hubbard U on any composition), or None = a source whose
# +U depends on the structure (none today; MP-style composition rules would go in the data pipeline).
# d3: None, or the dispersion variant to ADD BACK at inference (labels are stored with it stripped).
# share: fraction of the domain's batches; None = not decided yet.
_OVERLAY: dict[str, dict[str, Any]] = {
    "mp_pbe": dict(token="pbe", u=0, d3=None, share=None),
    "mp_pbeU": dict(token="pbe", u=1, d3=None, share=None),
    "omat_pbe": dict(token="pbe", u=0, d3=None, share=None),
    "omat_pbeU": dict(token="pbe", u=1, d3=None, share=None),
    "matpes_pbe": dict(token="pbe", u=0, d3=None, share=None),
    "matpes_r2scan": dict(token="r2scan", u=0, d3=None, share=None),
    "omol_wb97mv": dict(token="wb97mv", u=0, d3="bj_wb97m", share=None),
    # OC20 is a different functional (RPBE) so it gets its own token. OC22 is PBE+U with VASP slab settings:
    # it shares the PBE token but keeps its own reference column; the gate watches its residual (spec section 6).
    "oc20_rpbe": dict(token="rpbe", u=0, d3=None, share=0.03),
    "oc22_pbeU": dict(token="pbe", u=1, d3=None, share=0.05),
    "omc25_pbe": dict(token="pbe", u=0, d3="zero_pbe", share=None),
    "openrxn_wb97xd": dict(token="wb97xd", u=0, d3=None, share=None),
    "spice_wb97md3": dict(token="wb97m_d3bj", u=0, d3="bj_wb97m", share=None),
}

REFERENCE_BY_LOT: dict[tuple[str, int], str] = {
    ("pbe", 0): "mp_pbe",
    ("pbe", 1): "mp_pbeU",
    ("r2scan", 0): "matpes_r2scan",
    ("rpbe", 0): "oc20_rpbe",
    ("wb97mv", 0): "omol_wb97mv",
    ("wb97xd", 0): "openrxn_wb97xd",
    ("wb97m_d3bj", 0): "spice_wb97md3",
}


def reference_key_of(name: str) -> str:
    ov = _OVERLAY[name]
    return REFERENCE_BY_LOT[(ov["token"], int(ov["u"] or 0))]


@dataclass(frozen=True)
class Source:
    name: str                 # legacy LoT key; also the reference-convention key
    domain: str               # crystal | molecule: picks the per-batch expert
    pbc: bool
    token: str                # trunk conditioning (per functional)
    token_id: int
    u_flag: int | None        # None: derived from composition per structure
    d3_addback: str | None    # None | "bj_wb97m" | "zero_pbe"
    e0_file: str | None
    e0_column: int
    energy_unit: str
    ev_to_kcal: float
    share: float | None

    @property
    def reference_key(self) -> str:
        return REFERENCE_BY_LOT[(self.token, int(self.u_flag or 0))]


def _d3_policy_is_subtract(raw: Any) -> bool:
    return str(raw).strip().lower().startswith("subtract")


def build_sources(legacy: dict[str, Any] | None = None) -> dict[str, Source]:
    """Build lam4 sources from the legacy registry (loaded lazily so tests can inject a dict)."""
    if legacy is None:
        from lam.data.registry import load_lot_registry
        legacy = load_lot_registry()
    lots = legacy["levels_of_theory"]
    missing = sorted(set(lots) - set(_OVERLAY))
    if missing:
        raise KeyError(f"legacy LoTs without a lam4 overlay (classify them in sources.py): {missing}")
    out: dict[str, Source] = {}
    for name, cfg in lots.items():
        ov = _OVERLAY[name]
        domain = _DOMAIN_OF_LEGACY.get(cfg["domain"])
        if domain is None:
            raise ValueError(f"{name}: unknown legacy domain {cfg['domain']!r}")
        wants_d3 = _d3_policy_is_subtract(cfg.get("d3_policy", "none"))
        if wants_d3 != (ov["d3"] is not None):
            raise ValueError(f"{name}: legacy d3_policy={cfg.get('d3_policy')!r} disagrees with overlay d3={ov['d3']!r}")
        out[name] = Source(
            name=name, domain=domain, pbc=bool(cfg["pbc"]), token=ov["token"], token_id=TOKEN_ID[ov["token"]],
            u_flag=ov["u"], d3_addback=ov["d3"], e0_file=cfg.get("e0_file"), e0_column=int(cfg.get("e0_column", 0)),
            energy_unit=str(cfg["energy_unit"]), ev_to_kcal=float(cfg["ev_to_kcal"]), share=ov["share"],
        )
    return out


def domain_of(sources: dict[str, Source], name: str) -> str:
    return sources[name].domain
