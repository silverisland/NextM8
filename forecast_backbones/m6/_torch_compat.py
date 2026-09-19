# -*- coding: utf-8 -*-
# forecast_backbones/m6/_torch_compat.py
"""torch < 2.4 没有 nn.RMSNorm，而 timesfm3 3.0.2 的 transformer.py 直接写 nn.RMSNorm(...)。
ensure_rmsnorm() 在缺失时注入一个与 torch.nn.RMSNorm 同语义的实现：eps=None → finfo(dtype).eps，
elementwise_affine 权重名 weight（state_dict 键与官方一致，真权重 safetensors 可直接 load）。
torch ≥ 2.4 上是空操作。必须在 import timesfm3.torch.* 之前调用。"""
from __future__ import annotations

import torch
import torch.nn as nn


def ensure_rmsnorm() -> bool:
    if hasattr(nn, "RMSNorm"):
        return False

    class RMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            self.weight = (nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
                           if elementwise_affine else None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
            dims = tuple(range(-len(self.normalized_shape), 0))
            y = x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + eps)
            return y * self.weight if self.weight is not None else y

        def extra_repr(self) -> str:
            return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"

    nn.RMSNorm = RMSNorm
    return True
