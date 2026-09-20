# -*- coding: utf-8 -*-
# forecast_backbones/m6/backbone.py
"""timesfm3 的可微 torch 路径。TimesFM3Torch.decode 被 @torch.no_grad() 装饰，predict_batch 再套
inference_mode，两者都不能训；decode 用 functools.wraps 包装，decode.__wrapped__ 就是未装饰的原函数，
而 forward 本身全程可微。predict_quantiles 在 numpy 里照搬 predict_batch 的前处理，在 torch 里照搬后处理，
中间只调 decode.__wrapped__，输出 [B,H,9] 带梯度。限制：单目标变量；horizon 须为 output_patch_len 整数倍
（与 predict_batch(padding_mode="none") 在协变量宽度上的隐含要求一致）；输入经插值后必须全有限。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from ._torch_compat import ensure_rmsnorm, ensure_finite_running_stats
    from .decision import MODEL_LEVELS
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from _torch_compat import ensure_rmsnorm, ensure_finite_running_stats
    from decision import MODEL_LEVELS

ensure_rmsnorm()                                    # 必须在 import timesfm3.torch 之前
ensure_finite_running_stats()                       # 同上：零方差 patch 的 sqrt 反向 NaN 垫片
from timesfm3.torch import util as _util            # noqa: E402
if not getattr(getattr(_util, "update_running_stats", None), "_m6_eps_patched", False):
    raise RuntimeError(
        "ensure_finite_running_stats() 垫片没有真正打上：timesfm3.torch.util.update_running_stats 缺失"
        "或已改名（当前安装的 timesfm3 build 与 m6 假定的 3.0.2 API 不一致）。光伏夜间连续整段恰为 0.0，"
        "让零方差 patch 在训练里是常态而非边角情形，垫片缺失只会在某次夜间 batch 恰好命中零方差分支的反向"
        "传播时才炸出 NaN——必须在 import 期就拦下来，不能等训练中途才发现。")
from timesfm3.torch.model import TimesFM3Torch      # noqa: E402
from timesfm3.torch.timesfm3_forecaster import linear_interpolation, _znorm_stats, _is_nonnegative  # noqa: E402

_GLOBAL_CONTEXT_POINTS = 15360                      # timesfm3_forecaster.py:362-368


def _interp_rows(a: np.ndarray | None) -> np.ndarray | None:
    if a is None:
        return None
    a = np.array(a, dtype=np.float32, copy=True)
    flat = a.reshape(-1, a.shape[-1])
    flat = linear_interpolation(flat)
    return np.asarray(flat, np.float32).reshape(a.shape)


def _znorm_rows(a: np.ndarray | None) -> np.ndarray | None:
    """协变量逐行独立 znorm（predict_batch 对 po/pf 的做法；统计量不回传）。"""
    if a is None:
        return None
    out = a.copy()
    flat = out.reshape(-1, out.shape[-1])
    for r in range(flat.shape[0]):
        mu, sigma = _znorm_stats(flat[r]); flat[r] = (flat[r] - mu) / sigma
    return out


class TorchBackbone(nn.Module):
    def __init__(self, model: TimesFM3Torch):
        super().__init__()
        if [float(q) for q in model.quantiles] != [float(q) for q in MODEL_LEVELS]:
            raise ValueError(f"底座 quantiles={list(model.quantiles)} 与 MODEL_LEVELS={list(MODEL_LEVELS)} 不一致")
        self.model = model

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_evaluator(cls, evaluator) -> "TorchBackbone":
        return cls(evaluator.model)

    @classmethod
    def tiny_for_tests(cls, seed: int = 0) -> "TorchBackbone":
        from timesfm3.torch import configs
        torch.manual_seed(seed)
        rb = configs.ResidualBlockConfig(hidden_dims=32, output_dims=32, use_bias=False, activation="relu", dropout=0.0)
        tc = configs.StackedTransformersConfig(num_layers=2, use_remat=False, transformer=configs.TransformerConfig(
            model_dims=32, hidden_dims=32, num_heads=4, attention_norm="rms", feedforward_norm="rms", qk_norm="rms",
            use_rope_seq=True, use_rope_var=True, use_bias=False, ff_activation="relu", deterministic=True, use_sdpa=True))
        m = TimesFM3Torch(input_patch_len=8, output_patch_len=16, quantiles=list(MODEL_LEVELS),
                          residual_block_config=rb, transformer_config=tc, use_stitching=True,
                          use_linear_detrending=True, use_iterative_cpm_revin=True, use_frozen_running_stats=False)
        m.eval()
        with torch.no_grad():                        # pre_transformer_resblock 懒初始化：先跑一次定形
            m.decode(target=torch.zeros(1, 1, 16), horizon=16, mask=torch.zeros(1, 16, dtype=torch.bool))
        return cls(m)

    # ------------------------------------------------------------------ 属性
    @property
    def input_patch_len(self) -> int:
        return int(self.model.input_patch_len)

    @property
    def output_patch_len(self) -> int:
        return int(self.model.output_patch_len)

    @property
    def global_context(self) -> int:
        return math.ceil(_GLOBAL_CONTEXT_POINTS / self.input_patch_len) * self.input_patch_len

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ------------------------------------------------------------------ 可微推理
    def predict_quantiles(self, contexts, past_future=None, past_only=None, *, horizon: int,
                          use_znorm: bool = False, use_symmetric_averaging: bool = True,
                          make_positive: bool = True, sort_quantiles: bool = True) -> torch.Tensor:
        ctx = np.asarray(contexts, np.float32)
        if ctx.ndim == 2:
            ctx = ctx[:, None, :]
        if ctx.ndim != 3 or ctx.shape[1] != 1:
            raise ValueError(f"contexts 须为 [B,L] 或 [B,1,L]（得到 {ctx.shape}）")
        B, _, L = ctx.shape
        if horizon % self.output_patch_len != 0:
            raise ValueError(f"horizon={horizon} 须为 output_patch_len={self.output_patch_len} 的整数倍")
        pf = None if past_future is None else np.asarray(past_future, np.float32)
        po = None if past_only is None else np.asarray(past_only, np.float32)
        if pf is not None and (pf.ndim != 3 or pf.shape[0] != B or pf.shape[-1] != L + horizon):
            raise ValueError(f"past_future 须为 [B,C,L+H]=[{B},C,{L + horizon}]（得到 {None if pf is None else pf.shape}）")
        if po is not None and (po.ndim != 3 or po.shape[0] != B or po.shape[-1] != L):
            raise ValueError(f"past_only 须为 [B,C,L]=[{B},C,{L}]（得到 {po.shape}）")
        orig = ctx[:, 0, :].copy()                  # make_positive 按原始输入判定
        ctx, pf, po = _interp_rows(ctx), _interp_rows(pf), _interp_rows(po)
        for name, a in (("contexts", ctx), ("past_future", pf), ("past_only", po)):
            if a is not None and not np.isfinite(a).all():
                raise ValueError(f"{name} 插值后仍含非有限值（整行 NaN 需调用方先处理）")
        # 1) znorm（目标行统计量留作反变换）
        mu = np.zeros(B, np.float32); sigma = np.ones(B, np.float32)
        if use_znorm:
            ctx = ctx.copy()
            for i in range(B):
                m, s = _znorm_stats(ctx[i, 0]); mu[i], sigma[i] = m, s; ctx[i, 0] = (ctx[i, 0] - m) / s
            pf, po = _znorm_rows(pf), _znorm_rows(po)
        # 2) 对称平均：[x, -x] 交错成 2B
        if use_symmetric_averaging:
            ctx = np.stack([ctx, -ctx], axis=1).reshape(2 * B, 1, L)
            pf = None if pf is None else np.stack([pf, -pf], axis=1).reshape(2 * B, pf.shape[1], pf.shape[-1])
            po = None if po is None else np.stack([po, -po], axis=1).reshape(2 * B, po.shape[1], po.shape[-1])
        Bq = ctx.shape[0]
        # 3) 上下文补到 patch 整数倍（左补零 + mask=True），超 global_context 左截
        ipl = self.input_patch_len
        ctx_len = max(min(math.ceil(L / ipl) * ipl, self.global_context), ipl)
        mask = np.zeros((Bq, L), dtype=bool)
        if L > ctx_len:
            ctx, mask = ctx[..., -ctx_len:], mask[:, -ctx_len:]
            po = None if po is None else po[..., -ctx_len:]
            pf = None if pf is None else pf[..., -(ctx_len + horizon):]
        elif L < ctx_len:
            pad = ctx_len - L
            ctx = np.pad(ctx, [(0, 0), (0, 0), (pad, 0)])
            mask = np.pad(mask, [(0, 0), (pad, 0)], constant_values=True)
            po = None if po is None else np.pad(po, [(0, 0), (0, 0), (pad, 0)])
            pf = None if pf is None else np.pad(pf, [(0, 0), (0, 0), (pad, 0)])
        dev = self.device
        t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dtype=dt)
        # 4) 未装饰的 decode（可微）
        out = TimesFM3Torch.decode.__wrapped__(
            self.model, target=t(ctx), horizon=int(horizon),
            past_only_covariates=None if po is None else t(po),
            past_future_covariates=None if pf is None else t(pf),
            mask=t(mask, torch.bool))                # [Bq, V, horizon, 9]
        out = out[:, 0]                              # 目标变量
        # 5) 后处理，顺序与 predict_batch 相同
        if sort_quantiles:
            out = torch.sort(out, dim=-1).values
        if use_symmetric_averaging:
            out = (out[0::2] - out[1::2].flip(-1)) / 2
        if use_znorm:
            out = out * t(sigma)[:, None, None] + t(mu)[:, None, None]
        if make_positive:
            nonneg = torch.from_numpy(np.asarray(_is_nonnegative(orig), dtype=bool)).to(dev)
            out = torch.where(nonneg[:, None, None], out.clamp_min(0.0), out)
        return out[:, :horizon]
