"""Python wrapper for fused strict-IEEE FP32 CUDA Stage node & SwiGLU kernels."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lam4.kernels.cuda_ext import cuda_ext_ready, load_cuda_ext
from lam4.kernels.cuda_node_bwd_src import CUDA_NODE_BWD_SRC
from lam4.kernels.cuda_node_src import CUDA_NODE_FWD_SRC

_CPP_DECLS = r"""
torch::Tensor cuda_node_inv_fwd(torch::Tensor vec, torch::Tensor agg, torch::Tensor ten, torch::Tensor w, int64_t flags, double c2);
std::vector<torch::Tensor> cuda_node_inv_bwd(torch::Tensor vec, torch::Tensor agg, torch::Tensor ten, torch::Tensor w, torch::Tensor g, int64_t flags, double c2);
std::vector<torch::Tensor> cuda_node_vec_fwd(torch::Tensor vec, torch::Tensor agg, torch::Tensor ten, torch::Tensor gates, torch::Tensor w, int64_t flags, double c2);
std::vector<torch::Tensor> cuda_node_vec_bwd(torch::Tensor vec, torch::Tensor agg, torch::Tensor ten, torch::Tensor gates, torch::Tensor w, torch::Tensor g_v, torch::Tensor g_t, int64_t flags, double c2);
torch::Tensor cuda_swiglu_fwd(torch::Tensor gv);
torch::Tensor cuda_swiglu_bwd(torch::Tensor gv, torch::Tensor gy);
torch::Tensor cuda_env_swiglu_proj_fwd(torch::Tensor gv, torch::Tensor h_env, torch::Tensor w_t, torch::Tensor bias);
std::vector<torch::Tensor> cuda_env_swiglu_proj_bwd(torch::Tensor gv, torch::Tensor h_env, torch::Tensor w_t, torch::Tensor bias, torch::Tensor gy);
"""

_FUNCS = [
    "cuda_node_inv_fwd",
    "cuda_node_inv_bwd",
    "cuda_node_vec_fwd",
    "cuda_node_vec_bwd",
    "cuda_swiglu_fwd",
    "cuda_swiglu_bwd",
    "cuda_env_swiglu_proj_fwd",
    "cuda_env_swiglu_proj_bwd",
]


def _ext() -> Any | None:
    return load_cuda_ext("lam4_cuda_node", _CPP_DECLS, CUDA_NODE_FWD_SRC + "\n" + CUDA_NODE_BWD_SRC, _FUNCS)


def _require_ext() -> Any:
    ext = _ext()
    if ext is None:
        raise RuntimeError("lam4_cuda_node CUDA extension unavailable")
    return ext


def has_cuda_node(dev: torch.device | str | None = None) -> bool:
    return cuda_ext_ready("LAM4_CUDA_NODE", "lam4_cuda_node", dev, _ext)


def cuda_node_inv_fwd(vec: Tensor, agg: Tensor, ten: Tensor, w: Tensor, flags: int, c2: float) -> Tensor:
    t_arg = ten.contiguous() if ten.numel() else agg.new_zeros(1)
    return _require_ext().cuda_node_inv_fwd(vec.contiguous(), agg.contiguous(), t_arg, w, int(flags), float(c2))


def cuda_node_inv_bwd(vec: Tensor, agg: Tensor, ten: Tensor, w: Tensor, g: Tensor, flags: int, c2: float) -> tuple[Tensor, Tensor, Tensor]:
    t_arg = ten.contiguous() if ten.numel() else agg.new_zeros(1)
    res = _require_ext().cuda_node_inv_bwd(vec.contiguous(), agg.contiguous(), t_arg, w, g.contiguous(), int(flags), float(c2))
    return res[0], res[1], res[2]


def cuda_node_vec_fwd(vec: Tensor, agg: Tensor, ten: Tensor, gates: Tensor, w: Tensor, flags: int, c2: float) -> tuple[Tensor, Tensor]:
    t_arg = ten.contiguous() if ten.numel() else agg.new_zeros(1)
    res = _require_ext().cuda_node_vec_fwd(vec.contiguous(), agg.contiguous(), t_arg, gates.contiguous(), w, int(flags), float(c2))
    return res[0], res[1]


def cuda_node_vec_bwd(vec: Tensor, agg: Tensor, ten: Tensor, gates: Tensor, w: Tensor, g_v: Tensor, g_t: Tensor, flags: int, c2: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    t_arg = ten.contiguous() if ten.numel() else agg.new_zeros(1)
    res = _require_ext().cuda_node_vec_bwd(vec.contiguous(), agg.contiguous(), t_arg, gates.contiguous(), w, g_v.contiguous(), g_t.contiguous(), int(flags), float(c2))
    return res[0], res[1], res[2], res[3]


def cuda_swiglu_fwd(gv: Tensor) -> Tensor:
    return _require_ext().cuda_swiglu_fwd(gv.contiguous())


def cuda_swiglu_bwd(gv: Tensor, gy: Tensor) -> Tensor:
    return _require_ext().cuda_swiglu_bwd(gv.contiguous(), gy.contiguous())


def cuda_env_swiglu_proj_fwd(gv: Tensor, h_env: Tensor, w_t: Tensor, bias: Tensor) -> Tensor:
    return _require_ext().cuda_env_swiglu_proj_fwd(gv.contiguous(), h_env.contiguous(), w_t.contiguous(), bias.contiguous())


def cuda_env_swiglu_proj_bwd(gv: Tensor, h_env: Tensor, w_t: Tensor, bias: Tensor, gy: Tensor) -> tuple[Tensor, Tensor]:
    res = _require_ext().cuda_env_swiglu_proj_bwd(gv.contiguous(), h_env.contiguous(), w_t.contiguous(), bias.contiguous(), gy.contiguous())
    return res[0], res[1]
