"""muon_batched.py: torch.optim.Muon with the Newton-Schulz orthogonalisation batched over matrices of the same shape.

torch.optim.Muon (torch 2.11-2.12) loops over the parameters one by one: per matrix ~25 small kernels (momentum lerps,
bf16 cast, norm, 5 x (matmul, 2 addmm), weight decay, add). MenaACE d384 has 66 hidden matrices in a handful of shapes,
so one step launches ~1,650 kernels that are each too small to fill an A100. Here the momentum update, the
orthogonalisation and the parameter update run once per shape group (stack -> bmm), with the same arithmetic as
torch's `_zeropower_via_newtonschulz` (bf16, Frobenius pre-normalisation per matrix, quintic coefficients, transpose
for tall matrices) and the same lr adjustment, weight decay and Nesterov momentum.

The state layout (`state[p]["momentum_buffer"]`) and the param-group keys are torch's, so checkpoints of runs that used
torch.optim.Muon load into this optimizer and vice versa. tests/test_muon_batched.py checks the update against torch's.
"""

from __future__ import annotations

import math
import torch

try:
    from torch.optim._muon import _adjust_lr
except ImportError:

    def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
        """Fallback matching PyTorch's `torch.optim._muon._adjust_lr`."""
        a_dim, b_dim = param_shape[:2]
        if adjust_lr_fn is None or adjust_lr_fn == "original":
            adjusted_ratio = math.sqrt(max(1.0, a_dim / b_dim))
        elif adjust_lr_fn == "match_rms_adamw":
            adjusted_ratio = 0.2 * math.sqrt(max(a_dim, b_dim))
        else:
            adjusted_ratio = 1.0
        return lr * adjusted_ratio


def split_muon_adam_params(model: torch.nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split trainable parameters into 2D hidden matrices for Muon and remaining parameters for AdamW."""
    muon_params: list[torch.Tensor] = []
    adam_params: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and min(p.shape) > 1 and "embedding" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


_MuonBase = getattr(torch.optim, "Muon", torch.optim.Optimizer)


def newton_schulz_batched(g: torch.Tensor, coeffs: tuple[float, float, float], steps: int, eps: float) -> torch.Tensor:
    """[k, A, B] -> [k, A, B] bf16: torch's `_zeropower_via_newtonschulz` applied to each matrix of the stack."""
    a, b, c = coeffs
    x = g.bfloat16()
    tall = g.size(-2) > g.size(-1)
    if tall:
        x = x.transpose(-2, -1)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        x = a * x + torch.baddbmm(gram, gram, gram, beta=b, alpha=c) @ x
    return x.transpose(-2, -1) if tall else x


class BatchedMuon(_MuonBase):
    """Drop-in torch.optim.Muon; step() groups the parameters of each param group by shape."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
            by_shape: dict[tuple, list[torch.Tensor]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    raise ValueError("Param gradient must be a 2D matrix")
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                by_shape.setdefault(tuple(p.shape), []).append(p)
            for shape, ps in by_shape.items():
                grads = [p.grad for p in ps]
                bufs = [self.state[p]["momentum_buffer"] for p in ps]
                g = torch.stack(grads)
                buf = torch.stack(bufs)
                b_norm = buf.norm(dim=(-2, -1), keepdim=True)
                g_norm = g.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
                g = torch.where((b_norm > 1e-5) & (g_norm > 5.0 * b_norm), g * (5.0 * b_norm / g_norm), g)
                buf.lerp_(g, 1 - mom)
                for b_dst, b_src in zip(bufs, buf.unbind(0)):
                    b_dst.copy_(b_src)
                upd = g.lerp(buf, mom) if group["nesterov"] else buf
                upd = newton_schulz_batched(upd, group["ns_coefficients"], group["ns_steps"], group["eps"])
                adj = _adjust_lr(lr, group["adjust_lr_fn"], torch.Size(shape))
                if wd != 0:
                    torch._foreach_mul_(ps, 1 - lr * wd)
                torch._foreach_add_(ps, list(upd.to(ps[0].dtype).unbind(0)), alpha=-adj)
        return loss
