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

from lam4.registry.tokens import PAW_ID, TOKEN_ID, XC_ID, lot_cell_id

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
    # OC20 is RPBE so it gets its own token. OC22 is VASP PBE + MP Hubbard U on {V,Cr,Mn,Fe,Co,Ni,Mo,W}+O/F (~32%)
    # and pure PBE without +U (~68%, routed to mp_pbe u=0); oc22_pbeU specifically covers the +U oxide subset under ("pbe", 1).
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
    key = _LOT_ALIASES.get(name, name)
    ov = _OVERLAY.get(key)
    if ov is not None:
        return REFERENCE_BY_LOT[(ov["token"], int(ov["u"] or 0))]
    spec = LAM6_LOT_SPECS.get(key)
    if spec is not None:
        u = 0 if spec.u_policy == "zero" else 1
        return REFERENCE_BY_LOT.get((spec.xc, u), "matpes_r2scan" if spec.xc in ("r2scan", "pbe0") else "mp_pbe")
    raise KeyError(name)


_LEGACY_PAW: dict[str, str] = {
    "mp_pbe": "vasp_pbe52",
    "mp_pbeU": "vasp_pbe52",
    "omat_pbe": "vasp_pbe54",
    "omat_pbeU": "vasp_pbe54",
    "matpes_pbe": "vasp_pbe64",
    "matpes_r2scan": "vasp_pbe64",
    "omol_wb97mv": "def2_tzvpd",
    "oc20_rpbe": "vasp_pbe54",
    "oc22_pbeU": "vasp_pbe54",
    "omc25_pbe": "vasp_pbe54",
    "openrxn_wb97xd": "pop_631gd",
    "spice_wb97md3": "def2_tzvppd",
}


@dataclass(frozen=True)
class Source:
    name: str                 # legacy LoT key; also the reference-convention key
    domain: str               # crystal | molecule: picks the per-batch expert (lam4/lam5)
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
    xc: str = ""
    xc_id: int = 0
    paw_basis: str = ""
    paw_id: int = 0

    @property
    def reference_key(self) -> str:
        k = (self.token, int(self.u_flag or 0))
        return REFERENCE_BY_LOT[k] if k in REFERENCE_BY_LOT else reference_key_of(self.name)


@dataclass(frozen=True)
class FactorizedLotSpec:
    """Factorized physical theory specification (zero dataset-ID leakage)."""

    name: str
    xc: str
    paw_basis: str
    pbc: bool
    domain_legacy: str
    u_policy: str             # "zero" | "mp2020_auto" | "fixed_one"
    d3_addback: str | None = None
    energy_unit: str = "eV"

    @property
    def xc_id(self) -> int:
        return XC_ID[self.xc]

    @property
    def paw_id(self) -> int:
        return PAW_ID[self.paw_basis]


# Clean-license Wave-2 LAM-6 physical LoTs (excludes OpenRxn26; OC22 replaced by OC25/MAD-1.6).
LAM6_LOT_SPECS: dict[str, FactorizedLotSpec] = {
    "mp_pbe": FactorizedLotSpec("mp_pbe", "pbe", "vasp_pbe52", True, CRYSTAL, "zero"),
    "mp_pbeU": FactorizedLotSpec("mp_pbeU", "pbe", "vasp_pbe52", True, CRYSTAL, "fixed_one"),
    "omat_pbe": FactorizedLotSpec("omat_pbe", "pbe", "vasp_pbe54", True, CRYSTAL, "zero"),
    "omat_pbeU": FactorizedLotSpec("omat_pbeU", "pbe", "vasp_pbe54", True, CRYSTAL, "mp2020_auto"),
    "matpes_pbe": FactorizedLotSpec("matpes_pbe", "pbe", "vasp_pbe64", True, CRYSTAL, "zero"),
    "matpes_r2scan": FactorizedLotSpec("matpes_r2scan", "r2scan", "vasp_pbe64", True, CRYSTAL, "zero"),
    "omc25_pbe": FactorizedLotSpec("omc25_pbe", "pbe", "vasp_pbe54", True, MOLECULE, "zero", "zero_pbe"),
    "oc20_rpbe": FactorizedLotSpec("oc20_rpbe", "rpbe", "vasp_pbe54", True, CRYSTAL, "zero"),
    "oc25_rpbe": FactorizedLotSpec("oc25_rpbe", "rpbe", "vasp_pbe54", True, CRYSTAL, "zero"),
    "mad16_r2scan": FactorizedLotSpec("mad16_r2scan", "r2scan", "aims_tight", True, CRYSTAL, "zero"),
    "mad16_pbe": FactorizedLotSpec("mad16_pbe", "pbe", "aims_tight", True, CRYSTAL, "zero"),
    "mad16_r2scan_mol": FactorizedLotSpec("mad16_r2scan_mol", "r2scan", "aims_tight", False, MOLECULE, "zero"),
    "omol_wb97mv": FactorizedLotSpec("omol_wb97mv", "wb97mv", "def2_tzvpd", False, MOLECULE, "zero", "bj_wb97m", "kcal/mol"),
    "opoly26_wb97mv": FactorizedLotSpec("opoly26_wb97mv", "wb97mv", "def2_tzvpd", False, MOLECULE, "zero", "bj_wb97m", "kcal/mol"),
    "spice_wb97md3": FactorizedLotSpec("spice_wb97md3", "wb97m_d3bj", "def2_tzvppd", False, MOLECULE, "zero", "bj_wb97m", "kcal/mol"),
    "t1x_wb97x": FactorizedLotSpec("t1x_wb97x", "wb97x", "pop_631gd", False, MOLECULE, "zero", None, "kcal/mol"),
    "rgd1_b3lyp": FactorizedLotSpec("rgd1_b3lyp", "b3lyp_d3", "def2_tzvp", False, MOLECULE, "zero", None, "kcal/mol"),
    "qcml_pbe0": FactorizedLotSpec("qcml_pbe0", "pbe0", "aims_tight", False, MOLECULE, "zero", None, "kcal/mol"),
    "tmqm_tpssh": FactorizedLotSpec("tmqm_tpssh", "tpssh_d3bj", "def2_svp", False, MOLECULE, "zero", None, "kcal/mol"),
    "splinter_aug": FactorizedLotSpec("splinter_aug", "sapt0", "aug_cc_pvddz", False, MOLECULE, "zero", None, "kcal/mol"),
    "splinter_sapt0": FactorizedLotSpec("splinter_sapt0", "sapt0", "jun_cc_pvdz", False, MOLECULE, "zero", None, "kcal/mol"),
}

_LOT_ALIASES: dict[str, str] = {
    "mad_r2scan": "mad16_r2scan",
    "opoly_wb97mv": "opoly26_wb97mv",
    "splinter_jun": "splinter_sapt0",
}


def build_lam6_sources() -> dict[str, FactorizedLotSpec]:
    return dict(LAM6_LOT_SPECS)


def lot_ids(name: str) -> tuple[int, int]:
    """Return `(xc_id, paw_id)` for any known LAM-4/5/6 source name."""
    key = _LOT_ALIASES.get(name, name)
    spec = LAM6_LOT_SPECS.get(key)
    if spec is not None:
        return spec.xc_id, spec.paw_id
    ov = _OVERLAY.get(key)
    if ov is not None:
        paw = _LEGACY_PAW.get(key, "vasp_pbe52")
        return XC_ID.get(ov["token"], 0), PAW_ID.get(paw, 0)
    return (0, 0)


def resolve_lot_cell(spec: str | tuple[str | int, str | int, int] | FactorizedLotSpec | Source, n_paw: int = 16) -> int:
    """Resolve a source name, alias, `(xc, paw, u)` tuple, `FactorizedLotSpec`, or `Source` to its canonical `lot_cell_id`."""
    if isinstance(spec, (FactorizedLotSpec, Source)):
        return lot_cell_id(spec.xc_id, spec.paw_id, int(getattr(spec, "u_flag", 0) or (getattr(spec, "u_policy", "zero") != "zero")), n_paw)
    if isinstance(spec, tuple):
        return lot_cell_id(spec[0], spec[1], int(spec[2]), n_paw)
    spec_key = _LOT_ALIASES.get(spec, spec)
    if spec_key in LAM6_LOT_SPECS:
        s = LAM6_LOT_SPECS[spec_key]
        return lot_cell_id(s.xc_id, s.paw_id, 0 if s.u_policy == "zero" else 1, n_paw)
    if spec_key in _OVERLAY:
        ov = _OVERLAY[spec_key]
        return lot_cell_id(ov["token"], _LEGACY_PAW[spec_key], int(ov["u"] or 0), n_paw)
    raise KeyError(f"cannot resolve physical LoT cell from {spec!r}")


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
        paw = _LEGACY_PAW[name]
        out[name] = Source(
            name=name, domain=domain, pbc=bool(cfg["pbc"]), token=ov["token"], token_id=TOKEN_ID[ov["token"]],
            u_flag=ov["u"], d3_addback=ov["d3"], e0_file=cfg.get("e0_file"), e0_column=int(cfg.get("e0_column", 0)),
            energy_unit=str(cfg["energy_unit"]), ev_to_kcal=float(cfg["ev_to_kcal"]), share=ov["share"],
            xc=ov["token"], xc_id=XC_ID[ov["token"]], paw_basis=paw, paw_id=PAW_ID[paw],
        )
    return out


def domain_of(sources: dict[str, Source], name: str) -> str:
    return sources[name].domain
