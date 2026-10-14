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
    from .solar import StationMeta
    from .decision import enforce_monotone, physics_clip, select_point, level_value, p_star
    from .losses import pinball_loss
except ImportError:
    from forecast_backbones.m8.config import load_config
    from forecast_backbones.m8.features import FeatureBuilder
    from forecast_backbones.m8.model import PVRexCore
    from forecast_backbones.m8.solar import StationMeta
    from forecast_backbones.m8.decision import enforce_monotone, physics_clip, select_point, level_value, p_star
    from forecast_backbones.m8.losses import pinball_loss


def load_station_meta(path, cols: dict | None = None) -> StationMeta:
    """dataset.station_meta（任意路径、任意文件名）+ dataset.station_meta_cols（列名映射，见 StationMeta.apply_cols）。
    路径不存在直接报错；只有仓库内的默认相对路径（forecast_backbones/...）在别的工作目录下解析失败时才回退到 m8 自带那份。"""
    p = Path(path)
    if not p.exists():
        fallback = Path(__file__).parent / "station_meta.csv"
        if not p.is_absolute() and p.parts[:1] == ("forecast_backbones",) and fallback.exists():
            warnings.warn(f"station_meta {p} 不存在（相对当前工作目录解析失败）：回退到 {fallback}")
            return StationMeta.from_csv(fallback, cols)
        raise FileNotFoundError(f"station_meta {p} 不存在（dataset.station_meta；相对路径按当前工作目录解析）")
    return StationMeta.from_csv(p, cols)


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
        self.station_meta = load_station_meta(ds["station_meta"], ds.get("station_meta_cols"))
        self.fe = FeatureBuilder(self.cfg, self.station_meta)
        self.levels = tuple(float(q) for q in self.cfg["model"]["quantiles"])
        self.core = PVRexCore(len(self.fe.channel_names), self.fe.n_static, self.fe.n_slots, self.context_len, self.pred_len,
                              self.cfg["model"])
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

    def set_known_stations(self, station_ids) -> list[int]:
        """训练入口调用：记录训练集里的站号，推理时名单外的站走未知站槽。同步进 cfg（随 model_config.yaml 落盘）。"""
        ids = sorted({int(s) for s in station_ids})
        self.cfg["dataset"]["known_stations"] = ids
        self.fe.known_stations = np.array(ids, dtype=np.int64) if ids else None
        return ids

    def reseed(self, seed: int) -> None:
        """多卡时每个进程换不同的增强随机流。"""
        self._rng = np.random.default_rng(int(seed))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ forward
    def _run_core(self, inp: dict):
        """特征 → 网络 → (q_norm [B,T,K] 归一化单位, scale_t [B], x [B,T,C])。"""
        dev = next(self.parameters()).device
        x = torch.as_tensor(inp["x"], dtype=torch.float32, device=dev)
        static = torch.as_tensor(inp["static"], dtype=torch.float32, device=dev)
        slot = torch.as_tensor(inp["slot"], dtype=torch.long, device=dev)
        q_norm = self.core(x, static, slot)                                       # 归一化单位
        if q_norm.dtype != torch.float32:
            q_norm = q_norm.float()
        if self.fe.resid_col is not None:                                         # 残差学习：输出 = 提示基准 + 网络输出
            q_norm = q_norm + torch.as_tensor(inp["base"], dtype=torch.float32, device=dev)[:, :, None]
        scale_t = torch.as_tensor(inp["scale"], dtype=torch.float32, device=dev)
        return q_norm, scale_t, x

    def decide(self, q_raw: np.ndarray, night_fut: np.ndarray, p_clear_fut: np.ndarray, cap: np.ndarray, decision: dict | None = None):
        """决策层（numpy，decision.py；没有可训练参数，所以评估时可换规则不用重训）。
        q_raw [B,H,K] 真实单位的原始分位 → (point [B,H], q_cal, median, pstar, night_physics)。decision 缺省用 cfg['decision']。"""
        dec = decision if decision is not None else self.cfg["decision"]
        ph, pt = dec["physics"], dec["point"]
        if pt.get("tail", "flat") != "flat":
            raise ValueError(f"decision.point.tail: only 'flat' is implemented (got {pt.get('tail')!r})")
        # night_zero=false 时决策层对夜间不做任何强制归零：夜间掩码不用，晴空上限也不在 p_clear<=0 的步生效
        # （否则 upper = ratio·0 = 0 会把夜间悄悄钳成 0，与 M1–M7 不做夜间后处理的口径仍然不齐；2026-10-12 之前就是这样）
        nz, cn = bool(ph["night_zero"]), bool(ph.get("clip_negative", True))
        night_physics = night_fut if nz else np.zeros_like(night_fut)
        ratio, capcap = float(ph["clear_sky_cap_ratio"]), bool(ph["capacity_cap"])
        q_cal = physics_clip(enforce_monotone(q_raw), night_physics, p_clear_fut, cap[:, None], ratio, capcap,
                             night_ceiling=nz, clip_negative=cn)
        shrink = float(pt.get("shrink_factor") or 1.0)
        point = select_point(q_cal, self.levels, cap, pt["default"], list(pt.get("by_step") or ()), shrink,
                             float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
        median = level_value(q_cal, self.levels, 0.5)
        pstar = p_star(q_cal, self.levels, cap[:, None], float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
        return point, q_cal, median, pstar, night_physics

    @torch.no_grad()
    def predict_raw(self, batch_dict: dict):
        """推理态：→ (q_fut [B,H,K] 真实单位 numpy, ctx)。不过决策层，供集成先平均再统一决策。"""
        was = self.training
        self.eval()
        try:
            inp = self.fe.build(batch_dict, training=False, y_future=None, rng=self._rng)
            q_norm, scale_t, _ = self._run_core(inp)
            L = self.context_len
            return (q_norm[:, L:, :] * scale_t[:, None, None]).detach().float().cpu().numpy().astype(float), inp["ctx"]
        finally:
            self.train(was)

    def forward(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        L, H = self.context_len, self.pred_len
        y_future = None
        if labels is not None:
            y = labels[self.label_col] if isinstance(labels, dict) else labels
            y_future = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        inp = self.fe.build(batch_dict, training=self.training, y_future=y_future, rng=self._rng)
        dev = next(self.parameters()).device
        q_norm, scale_t, x = self._run_core(inp)
        q_fut = q_norm[:, L:, :] * scale_t[:, None, None]                         # [B, H, K] 真实单位
        ctx = inp["ctx"]
        night_full = torch.as_tensor(ctx.night, device=dev)
        cap_t = torch.as_tensor(ctx.cap, dtype=torch.float32, device=dev)
        y_mask = x[..., self.fe.idx["y_mask"]]
        loss = None
        if labels is not None:
            loss = self._loss_from_parts(q_norm, y_mask, night_full, torch.as_tensor(inp["target"], device=dev))

        point, q_cal, median, pstar, night_physics = self.decide(
            q_fut.detach().float().cpu().numpy().astype(float), ctx.night[:, L:], ctx.p_clear[:, L:], ctx.cap)
        logits = torch.as_tensor(point, dtype=torch.float32, device=dev)
        aux = {
            "quantiles": q_fut,
            "quantiles_cal": torch.as_tensor(q_cal, dtype=torch.float32, device=dev),
            "levels_cal": list(self.levels),
            "p_star": torch.as_tensor(pstar, dtype=torch.float32, device=dev),
            "median": torch.as_tensor(median, dtype=torch.float32, device=dev),
            "night": night_physics, "night_loss": ctx.night[:, L:],
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
        """pinball（9 档分位），监督步 = y_mask 为 1 且目标有限的步；夜间步按 train.loss.night_weight 加权（1 = 同权，0 = 不监督）。
        全部在归一化单位里算。"""
        lc = self.cfg["train"].get("loss") or {}
        mask = ((y_mask > 0.5) & torch.isfinite(target_full)).to(q_norm.dtype)                      # [B, T]
        nw = float(lc.get("night_weight", 1.0))
        if nw != 1.0:                                                                                 # 0 = 夜间不监督（旧行为）
            mask = mask * torch.where(night_full, torch.full_like(mask, nw), torch.ones_like(mask))
        return pinball_loss(q_norm, torch.nan_to_num(target_full), self.levels, mask)

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
