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


def ensure_finite_running_stats() -> bool:
    """timesfm3.torch.util.update_running_stats 对恰好零方差的 patch 做 torch.sqrt(0)：前向安全（用
    torch.where 避了 0/0），反向在 0 处导数为 +inf，遇到该分支上游梯度恰为 0 时 0*inf=nan——光伏夜间连续
    整段恰为 0.0，使这种零方差 patch 成为常态而非边角情形。替换成两处 sqrt 都套 _safe_sqrt 的版本：双重
    torch.where，不 clamp。前向逐位不变（v>0 时精确是 sqrt(v)，v<=0 时精确是 0，与未打补丁的 timesfm3
    逐位一致——route A 靠这个保证与 master 位级相同；早前用 torch.sqrt(v.clamp_min(eps)) 的版本对 v 落在
    (0, 1e-12) 时会把值钳到 sqrt(1e-12)=1e-6，不是逐位不变，且恰好落在 timesfm3 `_make_safe_for_division`
    的 1e-6 判定边界上，已弃用）。只有反向变了：v<=0 处 torch.sqrt 只作用在与 v 无关的常量 1 上（梯度不经
    v 回传），外层 torch.where 把 v<=0 的元素路由到常数 0 分支，避免 0*inf=nan。对 update_running_stats 的调用方（timesfm3.torch.util 自身、
    timesfm3.torch.cpm_revin_refine 等经 `from . import util` 按属性调用的模块）都用属性替换即时生效；
    额外扫一遍 sys.modules，把任何以 `from .util import update_running_stats` 方式按值导入过原函数的
    timesfm3.torch.* 模块也一并重绑定。补丁函数上挂 `_m6_original` 指回原始未打补丁的函数，供逐位对比
    用。必须在 import timesfm3.torch.* 之前调用（与 ensure_rmsnorm 顺序一致，紧跟其后）。"""
    import sys

    try:
        from timesfm3.torch import util as _util
    except ImportError:
        return False

    orig = getattr(_util, "update_running_stats", None)
    if orig is None:
        return False
    if getattr(orig, "_m6_eps_patched", False):
        return False

    def _safe_sqrt(v):
        safe = torch.where(v > 0, v, torch.ones_like(v))
        return torch.where(v > 0, torch.sqrt(safe), torch.zeros_like(v))

    def _patched(n, mu, sigma, x, mask):
        is_legit = ~mask
        is_legit_f = is_legit.float()
        inc_n = is_legit_f.sum(dim=-1)
        x_masked = torch.where(is_legit, x, torch.zeros_like(x))
        inc_sum = x_masked.sum(dim=-1)
        inc_mu = torch.where(inc_n == 0, torch.zeros_like(inc_sum), inc_sum / inc_n)
        x_diff_sq = torch.where(is_legit, (x - inc_mu.unsqueeze(-1)) ** 2, torch.zeros_like(x))
        inc_var = torch.where(inc_n == 0, torch.zeros_like(inc_sum), x_diff_sq.sum(dim=-1) / inc_n)
        inc_sigma = _safe_sqrt(inc_var)
        new_n = n + inc_n
        new_mu = torch.where(new_n == 0, torch.zeros_like(mu), (n * mu + inc_mu * inc_n) / new_n)
        var_sum = (n * sigma * sigma + inc_n * inc_sigma * inc_sigma
                  + n * (mu - new_mu) * (mu - new_mu) + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu))
        safe_new_n = torch.where(new_n == 0, torch.ones_like(new_n), new_n)
        new_var = torch.where(new_n == 0, torch.zeros_like(sigma), var_sum / safe_new_n)
        new_sigma = _safe_sqrt(new_var)
        return new_n, new_mu, new_sigma

    _patched._m6_eps_patched = True
    _patched._m6_original = orig
    _util.update_running_stats = _patched
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("timesfm3.torch") or name == "timesfm3.torch.util":
            continue
        if getattr(mod, "update_running_stats", None) is orig:
            setattr(mod, "update_running_stats", _patched)
    return True
