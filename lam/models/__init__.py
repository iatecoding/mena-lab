"""lam.models: Unified Mena-LAM model package."""

from lam.models.dispersion import D3BJ, WB97M_D3BJ
from lam.models.island_context import IslandContext, IslandPool
from lam.models.mena_ace import ACEStage, AngularBranch, L4Branch, MenaACE, l2_pair, l2_pair_mm, latent_dipole
from lam.models.mena_lam import (
    EV,
    ID_TO_LOT_NAME,
    LOT_ID_TO_NAME,
    LOT_NAMES,
    LOT_NAME_TO_ID,
    LOT_TO_ID,
    MEV,
    ElementPrior,
    MenaLAM,
    MenaLAMCalculator,
    WithD3,
    build_graph,
    cell_volume,
    compile_for_inference,
    use_compiled_stages,
    enable_low_memory,
    graft_materials_checkpoint,
    graft_molecules_checkpoint,
    load_mena_lam_checkpoint,
    mena_lam_kwargs,
    resolve_inference_device,
)

_LAZY_MUON_EXPORTS = frozenset({"BatchedMuon", "newton_schulz_batched", "split_muon_adam_params"})


def __getattr__(name: str):
    if name in _LAZY_MUON_EXPORTS:
        from lam.models import muon_batched

        return getattr(muon_batched, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACEStage",
    "AngularBranch",
    "BatchedMuon",
    "D3BJ",
    "EV",
    "ElementPrior",
    "ID_TO_LOT_NAME",
    "IslandContext",
    "IslandPool",
    "L4Branch",
    "LOT_ID_TO_NAME",
    "LOT_NAMES",
    "LOT_NAME_TO_ID",
    "LOT_TO_ID",
    "MEV",
    "MenaACE",
    "MenaLAM",
    "MenaLAMCalculator",
    "WB97M_D3BJ",
    "WithD3",
    "build_graph",
    "cell_volume",
    "compile_for_inference",
    "use_compiled_stages",
    "enable_low_memory",
    "graft_materials_checkpoint",
    "graft_molecules_checkpoint",
    "l2_pair",
    "l2_pair_mm",
    "latent_dipole",
    "load_mena_lam_checkpoint",
    "mena_lam_kwargs",
    "newton_schulz_batched",
    "resolve_inference_device",
    "split_muon_adam_params",
]

