# -*- coding: utf-8 -*-
# forecast_backbones/m8/transformer_models.py
"""M8 "PVRex"：带 NWP 协变量的混合递推分位预报模型（从零训练，纯 PyTorch）。

契约与 m6 相同：forward(batch_dict, labels=None) -> TransformersAdaptedModelOutput，
.logits [B,pred_len] = 提交曲线（decision.point 选点，已夜间归零与物理钳位），分位等辅助输出进 .aux（schema 与 m6 对齐，
可直接喂 m6 的 recompute_points / ladder_report / Calibrator）。labels 给定时 .loss = pinball（监督步 = 未来段 + CPM 掩码段，
掩夜间与缺测）。⚠️ 推理输入不读 observe_power_future。

权重：train.weights_path 存在即在构造时严格加载（推理）；训练入口把它改指向从不写入的哨兵路径，保证从零起步。"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from ...interfaces.modeloutput import TransformersAdaptedModelOutput
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from interfaces.modeloutput import TransformersAdaptedModelOutput

try:
    from .config import load_config
    from .features import FeatureBuilder
    from .model import PVRexCore
    from ..m6.features import StationMeta
    from ..m6.decision import enforce_monotone, physics_clip, select_point, level_value, p_star
    from ..m6.losses import pinball_loss
except ImportError:
    from forecast_backbones.m8.config import load_config
    from forecast_backbones.m8.features import FeatureBuilder
    from forecast_backbones.m8.model import PVRexCore
    from forecast_backbones.m6.features import StationMeta
    from forecast_backbones.m6.decision import enforce_monotone, physics_clip, select_point, level_value, p_star
    from forecast_backbones.m6.losses import pinball_loss


def load_station_meta(path) -> StationMeta:
    p = Path(path)
    if not p.exists():
        if not p.is_absolute():
            fallback = Path(__file__).parent.parent / "m6" / "station_meta.csv"
            if fallback.exists():
                warnings.warn(f"station_meta {p} 不存在（相对当前工作目录解析失败）：回退到 {fallback}")
                return StationMeta.from_csv(fallback)
        warnings.warn(f"station_meta {p} 不存在：全部站按缺坐标退化")
        import pandas as pd
        return StationMeta(pd.DataFrame(columns=list(StationMeta.COLS)))
    return StationMeta.from_csv(p)


def save_weights(model: nn.Module, path) -> None:
    from safetensors.torch import save_file
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path))


def load_weights(model: nn.Module, path) -> None:
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(path)), strict=True)


class TransformersAdaptedM8Model(nn.Module):
    """必须是本文件第一个 nn.Module 子类（registry 自动发现）。"""

    def __init__(self, model_config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.cfg = load_config(model_config)
        ds = self.cfg["dataset"]
        self.target_col, self.label_col = ds["pv_col"], ds["label_col"]
        self.context_len, self.pred_len = int(ds["context_len"]), int(ds["pred_len"])
        self.station_meta = load_station_meta(ds["station_meta"])
        self.fe = FeatureBuilder(self.cfg, self.station_meta)
        self.levels = tuple(float(q) for q in self.cfg["model"]["quantiles"])
        self.core = PVRexCore(len(self.fe.channel_names), self.fe.n_static, self.fe.n_slots, self.context_len, self.pred_len,
                              self.cfg["model"], self.fe.target_channels, self.fe.cov_channels)
        seed = int((self.cfg["run"].get("seeds") or [0])[0])
        self._rng = np.random.default_rng(seed)
        self._progress_desc = None
        self.weights_loaded = False
        wp = Path(str(self.cfg["train"]["weights_path"]))
        if wp.exists():
            load_weights(self, wp); self.weights_loaded = True
        self.eval()                                       # 默认推理态；训练脚本显式 .train()

    # ------------------------------------------------------------------ 框架契约
    @property
    def config(self) -> dict:
        return self.cfg

    @staticmethod
    def compute_metrics(eval_pred) -> dict:
        return {}

    def reseed(self, seed: int) -> None:
        """多卡时每个进程换不同的增强随机流。"""
        self._rng = np.random.default_rng(int(seed))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ forward
    def forward(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        L, H = self.context_len, self.pred_len
        y_future = None
        if labels is not None:
            y = labels[self.label_col] if isinstance(labels, dict) else labels
            y_future = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        inp = self.fe.build(batch_dict, training=self.training, y_future=y_future, rng=self._rng)
        dev = next(self.parameters()).device
        x = torch.as_tensor(inp["x"], dtype=torch.float32, device=dev)
        static = torch.as_tensor(inp["static"], dtype=torch.float32, device=dev)
        slot = torch.as_tensor(inp["slot"], dtype=torch.long, device=dev)
        q_norm = self.core(x, static, slot)                                       # [B, T, K]，归一化单位
        if q_norm.dtype != torch.float32:
            q_norm = q_norm.float()
        if self.fe.resid_col is not None:                                         # 残差学习：输出 = 提示基准 + 网络输出
            q_norm = q_norm + torch.as_tensor(inp["base"], dtype=torch.float32, device=dev)[:, :, None]
        scale_t = torch.as_tensor(inp["scale"], dtype=torch.float32, device=dev)
        q_fut = q_norm[:, L:, :] * scale_t[:, None, None]                         # [B, H, K] 真实单位
        ctx = inp["ctx"]
        night_full = torch.as_tensor(ctx.night, device=dev)
        y_mask = x[..., self.fe.idx["y_mask"]]
        loss = None
        if labels is not None:
            loss = self._loss_from_parts(q_norm, y_mask, night_full, torch.as_tensor(inp["target"], device=dev))

        # 决策层（numpy，与 m6 同一套函数）
        q_raw = q_fut.detach().float().cpu().numpy().astype(float)
        ph, pt = self.cfg["decision"]["physics"], self.cfg["decision"]["point"]
        if pt.get("tail", "flat") != "flat":
            raise ValueError(f"decision.point.tail: only 'flat' is implemented (got {pt.get('tail')!r})")
        night_fut = ctx.night[:, L:]
        night_physics = night_fut if ph["night_zero"] else np.zeros_like(night_fut)
        q_cal = physics_clip(enforce_monotone(q_raw), night_physics, ctx.p_clear[:, L:], ctx.cap[:, None],
                             float(ph["clear_sky_cap_ratio"]), bool(ph["capacity_cap"]))
        shrink = float(pt.get("shrink_factor") or 1.0)
        point = select_point(q_cal, self.levels, ctx.cap, pt["default"], list(pt.get("by_step") or ()), shrink,
                             float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
        median = level_value(q_cal, self.levels, 0.5)
        pstar = p_star(q_cal, self.levels, ctx.cap[:, None], float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
        logits = torch.as_tensor(point, dtype=torch.float32, device=dev)
        aux = {
            "quantiles": q_fut,
            "quantiles_cal": torch.as_tensor(q_cal, dtype=torch.float32, device=dev),
            "levels_cal": list(self.levels),
            "p_star": torch.as_tensor(pstar, dtype=torch.float32, device=dev),
            "median": torch.as_tensor(median, dtype=torch.float32, device=dev),
            "night": night_physics, "night_loss": night_fut,
            "elev_deg": ctx.elev_deg[:, L:], "p_clear": ctx.p_clear[:, L:],
            "var_bin": np.zeros(len(ctx.cap), int), "var_signal": ctx.var_signal, "station_id": ctx.station_id,
            "cap": ctx.cap, "degraded": ctx.degraded,
            # compute_loss（FrameworkTrainer 路径）需要的中间量
            "_q_norm": q_norm, "_y_mask": y_mask, "_night_full": night_full,
            "_target_hist": torch.as_tensor(inp["target"][:, :L], device=dev),
            "_scale": scale_t, "_amp": torch.as_tensor(inp["amp"], dtype=torch.float32, device=dev),
        }
        return TransformersAdaptedModelOutput(loss=loss, logits=logits, predictions=logits, aux=aux)

    def _loss_from_parts(self, q_norm: torch.Tensor, y_mask: torch.Tensor, night_full: torch.Tensor,
                         target_full: torch.Tensor) -> torch.Tensor:
        mask = (y_mask > 0.5) & torch.isfinite(target_full) & (~night_full)
        return pinball_loss(q_norm, torch.nan_to_num(target_full), self.levels, mask.to(q_norm.dtype))

    def compute_loss(self, outputs, labels, num_items_in_batch=None) -> torch.Tensor:
        """FrameworkTrainer 路径：forward 未收到 labels 时在这里补算（与 forward 内算法一致）。"""
        if outputs.loss is not None:
            return outputs.loss
        aux = outputs.aux
        y = labels[self.label_col] if isinstance(labels, dict) else labels
        y = torch.as_tensor(np.asarray(y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y, np.float32),
                            device=aux["_q_norm"].device)[:, : self.pred_len]
        y_fut = y * aux["_amp"][:, None] / aux["_scale"][:, None]
        target_full = torch.cat([aux["_target_hist"], y_fut], dim=1)
        return self._loss_from_parts(aux["_q_norm"], aux["_y_mask"], aux["_night_full"], target_full)
