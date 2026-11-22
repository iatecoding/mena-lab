"""Fused per-atom node kernels of a Stage on MPS: invariant block and vector / tensor stream update (custom ops + analytic backward).

    lam4::node_inv_fwd(vec, agg, ten, w_a, flags, c2) -> all_inv [N, 368 | 464]      (Stage._post_impl inv list + AngularBranch.features)
    lam4::node_vec_fwd(vec, agg, ten, gates, w_b, flags, c2) -> (v_out [N, 32, 3], t_out [N, 16, 5])   (post_vector_tensor_update)
Both have registered fake impls (the whole-model torch.compile runner traces them with dynamic N) and first-order autograd through the
`*_bwd` ops. There is no double backward: Stage uses them only when `node_post_ok` holds (compiled MPS FP32, eval, frozen weights,
no create_graph, LAM4_MPS_NODE_FUSED not 0); every other path (CPU, CUDA, FP64, training, Hessians, eager) runs the plain PyTorch code.
flags: 1 LATER layout, 2 has ten (cumulative l=2 stream), 4 contract_cumulative_t. c2 = vec_c^2 of bound_vectors.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lam4.kernels.metal_node_bwd_src import _INV_BWD_SRC
from lam4.kernels.metal_node_flags import MetalShader, fused_node_device
from lam4.kernels.metal_node_src import _INV_SRC, _Y3_BWD, _Y3_FWD, METAL_NODE_COMMON
from lam4.kernels.metal_node_vec_src import _VEC_SRC

F_LATER, F_TEN, F_CUMT = 1, 2, 4
WA_SIZE, WB_SIZE = 23552, 3840
_NODE_SHADER = MetalShader(
    "metal_node",
    lambda: METAL_NODE_COMMON + _INV_SRC.replace("@Y3_FWD@", _Y3_FWD).replace("@Y3_BWD@", _Y3_BWD) + _INV_BWD_SRC + _VEC_SRC,
)


def has_metal_node() -> bool:
    return _NODE_SHADER.available()


def _lib() -> Any:
    return _NODE_SHADER.lib()


def _out_dim(flags: int) -> int:
    return 408 if flags & F_LATER else 360


def _inv_dim(flags: int) -> int:
    return 464 if flags & F_LATER else 368


def _ten_arg(ten: Tensor, ref: Tensor) -> Tensor:
    return ten.contiguous() if ten.numel() else ref.new_zeros(1)


# ---------------------------------------------------------------- invariant block
@torch.library.custom_op("lam4::node_inv_fwd", mutates_args=())
def node_inv_fwd(vec: Tensor, agg: Tensor, ten: Tensor, w: Tensor, flags: int, c2: float) -> Tensor:
    if agg.device.type == "cuda":
        from lam4.kernels.cuda_node import cuda_node_inv_fwd
        return cuda_node_inv_fwd(vec, agg, ten, w, flags, c2)
    n = int(agg.shape[0])
    out = agg.new_empty((n, _inv_dim(flags)))
    if n:
        _lib().node_inv_fwd(vec.contiguous(), agg.contiguous(), _ten_arg(ten, agg), w, out, n, int(flags), float(c2),
                            threads=(n * 128, 1, 1), group_size=(128, 1, 1))
    return out


@node_inv_fwd.register_fake
def _(vec, agg, ten, w, flags, c2):
    return agg.new_empty((agg.shape[0], _inv_dim(flags)))


@torch.library.custom_op("lam4::node_inv_bwd", mutates_args=())
def node_inv_bwd(vec: Tensor, agg: Tensor, ten: Tensor, w: Tensor, g: Tensor, flags: int, c2: float) -> tuple[Tensor, Tensor, Tensor]:
    if agg.device.type == "cuda":
        from lam4.kernels.cuda_node import cuda_node_inv_bwd
        return cuda_node_inv_bwd(vec, agg, ten, w, g, flags, c2)
    n = int(agg.shape[0])
    g_vec, g_agg = vec.new_empty((n, 32, 3)), agg.new_empty((n, _out_dim(flags)))
    g_ten = ten.new_empty(ten.shape)
    if n:
        _lib().node_inv_bwd(vec.contiguous(), agg.contiguous(), _ten_arg(ten, agg), w, g.contiguous(), g_vec, g_agg,
                            g_ten if ten.numel() else agg.new_zeros(1), n, int(flags), float(c2),
                            threads=(n * 128, 1, 1), group_size=(128, 1, 1))
    return g_vec, g_agg, g_ten


@node_inv_bwd.register_fake
def _(vec, agg, ten, w, g, flags, c2):
    return vec.new_empty((agg.shape[0], 32, 3)), agg.new_empty((agg.shape[0], _out_dim(flags))), ten.new_empty(ten.shape)


def _inv_setup(ctx, inputs, output):
    vec, agg, ten, w, flags, c2 = inputs
    ctx.save_for_backward(vec, agg, ten, w)
    ctx.flags, ctx.c2 = flags, c2


def _inv_backward(ctx, g):
    vec, agg, ten, w = ctx.saved_tensors
    g_vec, g_agg, g_ten = node_inv_bwd(vec, agg, ten, w, g, ctx.flags, ctx.c2)
    return g_vec, g_agg, (g_ten if ten.numel() else None), None, None, None


torch.library.register_autograd("lam4::node_inv_fwd", _inv_backward, setup_context=_inv_setup)


# ---------------------------------------------------------------- vector / tensor update
@torch.library.custom_op("lam4::node_vec_fwd", mutates_args=())
def node_vec_fwd(vec: Tensor, agg: Tensor, ten: Tensor, gates: Tensor, w: Tensor, flags: int, c2: float) -> tuple[Tensor, Tensor]:
    if agg.device.type == "cuda":
        from lam4.kernels.cuda_node import cuda_node_vec_fwd
        return cuda_node_vec_fwd(vec, agg, ten, gates, w, flags, c2)
    n = int(agg.shape[0])
    v_out, t_out = vec.new_empty((n, 32, 3)), agg.new_empty((n, 16, 5))
    if n:
        _lib().node_vec_fwd(vec.contiguous(), agg.contiguous(), _ten_arg(ten, agg), gates.contiguous(), w, v_out, t_out,
                            n, int(flags), float(c2), threads=(n * 64, 1, 1), group_size=(64, 1, 1))
    return v_out, t_out


@node_vec_fwd.register_fake
def _(vec, agg, ten, gates, w, flags, c2):
    return vec.new_empty((agg.shape[0], 32, 3)), agg.new_empty((agg.shape[0], 16, 5))


@torch.library.custom_op("lam4::node_vec_bwd", mutates_args=())
def node_vec_bwd(vec: Tensor, agg: Tensor, ten: Tensor, gates: Tensor, w: Tensor, g_v: Tensor, g_t: Tensor, flags: int,
                 c2: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if agg.device.type == "cuda":
        from lam4.kernels.cuda_node import cuda_node_vec_bwd
        return cuda_node_vec_bwd(vec, agg, ten, gates, w, g_v, g_t, flags, c2)
    n = int(agg.shape[0])
    g_vec, g_agg, g_gates = vec.new_empty((n, 32, 3)), agg.new_empty((n, _out_dim(flags))), gates.new_empty((n, 96))
    g_ten = ten.new_empty(ten.shape)
    if n:
        _lib().node_vec_bwd(vec.contiguous(), agg.contiguous(), _ten_arg(ten, agg), gates.contiguous(), w, g_v.contiguous(), g_t.contiguous(),
                            g_vec, g_agg, g_ten if ten.numel() else agg.new_zeros(1), g_gates, n, int(flags), float(c2),
                            threads=(n * 64, 1, 1), group_size=(64, 1, 1))
    return g_vec, g_agg, g_ten, g_gates


@node_vec_bwd.register_fake
def _(vec, agg, ten, gates, w, g_v, g_t, flags, c2):
    n = agg.shape[0]
    return vec.new_empty((n, 32, 3)), agg.new_empty((n, _out_dim(flags))), ten.new_empty(ten.shape), gates.new_empty((n, 96))


def _vec_setup(ctx, inputs, output):
    vec, agg, ten, gates, w, flags, c2 = inputs
    ctx.save_for_backward(vec, agg, ten, gates, w)
    ctx.flags, ctx.c2 = flags, c2


def _vec_backward(ctx, g_v, g_t):
    vec, agg, ten, gates, w = ctx.saved_tensors
    g_v = torch.zeros_like(vec) if g_v is None else g_v
    g_t = agg.new_zeros((agg.shape[0], 16, 5)) if g_t is None else g_t
    g_vec, g_agg, g_ten, g_gates = node_vec_bwd(vec, agg, ten, gates, w, g_v, g_t, ctx.flags, ctx.c2)
    return g_vec, g_agg, (g_ten if ten.numel() else None), g_gates, None, None, None


torch.library.register_autograd("lam4::node_vec_fwd", _vec_backward, setup_context=_vec_setup)


# ---------------------------------------------------------------- packing / dispatch helpers
@torch.no_grad()
def pack_node_weights(st) -> tuple[Tensor, Tensor] | None:
    """(w_a [23552], w_b [3840]) FP32 packs of a Stage with the lam5 / lam6 dims, every block transposed to [in][rows]; None otherwise."""
    if not node_dims_ok(st):
        return None
    a = st.ang
    dev = st.u1.weight.device

    def blk(mods, n_in):
        return torch.cat([m.weight.detach().float() for m in mods], 0).t().contiguous().reshape(-1)

    def opt(m, n):
        return m.weight.detach().float().t().contiguous().reshape(-1) if m is not None else torch.zeros(n * n, device=dev)
    w_a = torch.cat([
        blk((st.u1, st.v1, st.p3, st.q3), 32), blk((st.u2, st.v2, st.t3), 16), blk((a.ttt_c, a.vttv_a, a.vttv_b, a.a3), 16),
        blk((a.vttv_p, a.vttv_q, a.p3), 32), blk((a.u3, a.v3, a.w3), 8),
        opt(st.u_cv, 32), opt(st.v_cv, 32), opt(st.u_ct, 16), opt(st.v_ct, 16),
    ]).contiguous()
    w_b = torch.cat([blk((st.vec_self, st.tv_v, st.vv_p, st.vv_q), 32), blk((st.ten_self, st.tv_t), 16)]).contiguous()
    assert w_a.numel() == WA_SIZE and w_b.numel() == WB_SIZE
    return w_a, w_b


def node_dims_ok(st) -> bool:
    L, a = st.layout, st.ang
    return bool(
        st.inv_norm and (L.k_s, L.k_v, L.k_t, L.k_o) == (128, 32, 16, 8)
        and (st.u1.out_features, st.u2.out_features, st.t3.out_features) == (128, 64, 64)
        and (a.kt, a.kr, a.k3, a.ko) == (16, 32, 32, 8)
        and (not L.later or (L.k_tr, L.k_lon, L.k_tr2, L.k_lon2) == (32, 32, 16, 16))
    )


def node_post_ok(st, s: Tensor, bk: dict | None, allow: bool) -> bool:
    """The fused node kernels replace the plain Stage post block: MPS or CUDA FP32 inference (no double backward)."""
    return bool(
        allow and bk is not None and "node_wa" in bk and fused_node_device(s) and not st.training and st._inv_hook is None
        and not st.u1.weight.requires_grad and bk["node_wa"].device.type == s.device.type
        and (s.device.type != "mps" or has_metal_node())
    )


def stage_flags(st, ten: Tensor | None) -> int:
    return (F_LATER if st.layout.later else 0) | (F_TEN if ten is not None else 0) | (F_CUMT if st.contract_cumulative_t else 0)


def fused_inv(st, vec: Tensor, agg: Tensor, ten: Tensor | None, bk: dict) -> Tensor:
    t_arg = ten if ten is not None else agg.new_zeros(0)
    return node_inv_fwd(vec, agg, t_arg, bk["node_wa"], stage_flags(st, ten), float(st.vec_c) ** 2)


def fused_vec_update(st, s: Tensor, vec: Tensor, agg: Tensor, ten: Tensor | None, bk: dict) -> tuple[Tensor, Tensor]:
    """v_out [N, 32, 3], t_out [N, 16, 5] of post_vector_tensor_update (baked inv_norm branch) from the post-SwiGLU state s."""
    gates = torch.tanh(torch.nn.functional.linear(st.vec_gate_norm(s), bk["w_gates4"], bk["b_gates4"]))
    t_arg = ten if ten is not None else agg.new_zeros(0)
    return node_vec_fwd(vec, agg, t_arg, gates, bk["node_wb"], stage_flags(st, ten), float(st.vec_c) ** 2)
