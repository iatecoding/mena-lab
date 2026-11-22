"""Python wrapper for strict-IEEE FP32 SMEM-tiled CUDA ExactCoulomb kernels."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lam4.kernels.cuda_coulomb_src import CUDA_COULOMB_SRC
from lam4.kernels.cuda_ext import cuda_ext_ready, load_cuda_ext

_CPP_DECLS = r"""
void cuda_coulomb_fwd_launch(
    torch::Tensor pos, torch::Tensor q, torch::Tensor mask, torch::Tensor g_arr, torch::Tensor gptr,
    torch::Tensor bsph, torch::Tensor phi, torch::Tensor fld, int64_t n_atoms,
    double inv_sig, double c0, double inv_sig2, double far_pad,
    bool is_fast64, bool is_grouped, bool need_fld);

void cuda_coulomb_bwd_launch(
    torch::Tensor pos, torch::Tensor q, torch::Tensor mask, torch::Tensor g_arr, torch::Tensor gptr,
    torch::Tensor bsph, torch::Tensor gphi, torch::Tensor gfld, torch::Tensor dpos, torch::Tensor dq,
    int64_t n_atoms, double inv_sig, double c0, double inv_sig2, double far_pad,
    bool is_fast64, bool is_grouped, bool has_gfld, bool need_dpos, bool need_dq);
"""


def _ext() -> Any | None:
    return load_cuda_ext(
        "lam4_cuda_coulomb",
        _CPP_DECLS,
        CUDA_COULOMB_SRC,
        ["cuda_coulomb_fwd_launch", "cuda_coulomb_bwd_launch"],
    )


def _require_ext() -> Any:
    ext = _ext()
    if ext is None:
        raise RuntimeError("lam4_cuda_coulomb CUDA extension unavailable")
    return ext


def has_cuda_coulomb(dev: torch.device | str | None = None) -> bool:
    return cuda_ext_ready("LAM4_CUDA_COULOMB", "lam4_cuda_coulomb", dev, _ext)


def cuda_coulomb_fwd(
    pos: Tensor, q: Tensor, mask: Tensor, g_arr: Tensor, gptr: Tensor, bsph: Tensor,
    phi: Tensor, fld: Tensor, n_atoms: int,
    inv_sig: float, c0: float, inv_sig2: float, far_pad: float,
    *, is_fast64: bool, is_grouped: bool, need_fld: bool,
) -> None:
    _require_ext().cuda_coulomb_fwd_launch(
        pos, q, mask, g_arr, gptr, bsph, phi, fld, int(n_atoms),
        float(inv_sig), float(c0), float(inv_sig2), float(far_pad),
        bool(is_fast64), bool(is_grouped), bool(need_fld),
    )


def cuda_coulomb_bwd(
    pos: Tensor, q: Tensor, mask: Tensor, g_arr: Tensor, gptr: Tensor, bsph: Tensor,
    gphi: Tensor, gfld: Tensor, dpos: Tensor, dq: Tensor, n_atoms: int,
    inv_sig: float, c0: float, inv_sig2: float, far_pad: float,
    *, is_fast64: bool, is_grouped: bool, has_gfld: bool, need_dpos: bool, need_dq: bool,
) -> None:
    _require_ext().cuda_coulomb_bwd_launch(
        pos, q, mask, g_arr, gptr, bsph, gphi, gfld, dpos, dq, int(n_atoms),
        float(inv_sig), float(c0), float(inv_sig2), float(far_pad),
        bool(is_fast64), bool(is_grouped), bool(has_gfld), bool(need_dpos), bool(need_dq),
    )
