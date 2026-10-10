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


def quantile_value_t(q: torch.Tensor, levels, tau: float) -> torch.Tensor:
    """q [..., K] 在分位 tau 处的值（torch，可反传）：命中档位直接取，否则相邻两档线性插值，越界取端点。"""
    lv = [float(l) for l in levels]
    if tau in lv:
        return q[..., lv.index(tau)]
    below = [i for i, l in enumerate(lv) if l < tau]
    if not below:
        return q[..., 0]
    j = below[-1]
    if j + 1 >= len(lv):
        return q[..., -1]
    wgt = (tau - lv[j]) / (lv[j + 1] - lv[j])
    return q[..., j] * (1.0 - wgt) + q[..., j + 1] * wgt


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
                              self.cfg["model"], self.fe.target_channels, self.fe.cov_channels, chan_idx=self.fe.idx)
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
        """特征 → 网络 → (q_norm [B,T,K] 归一化单位, m_norm [B,T] 或 None, scale_t [B], x [B,T,C])。"""
        dev = next(self.parameters()).device
        x = torch.as_tensor(inp["x"], dtype=torch.float32, device=dev)
        static = torch.as_tensor(inp["static"], dtype=torch.float32, device=dev)
        slot = torch.as_tensor(inp["slot"], dtype=torch.long, device=dev)
        q_norm, m_norm = self.core(x, static, slot)                               # 归一化单位
        if q_norm.dtype != torch.float32:
            q_norm = q_norm.float()
        if m_norm is not None and m_norm.dtype != torch.float32:
            m_norm = m_norm.float()
        if self.fe.resid_col is not None:                                         # 残差学习：输出 = 提示基准 + 网络输出
            base = torch.as_tensor(inp["base"], dtype=torch.float32, device=dev)
            q_norm = q_norm + base[:, :, None]
            if m_norm is not None:
                m_norm = m_norm + base
        scale_t = torch.as_tensor(inp["scale"], dtype=torch.float32, device=dev)
        return q_norm, m_norm, scale_t, x

    def decide(self, q_raw: np.ndarray, night_fut: np.ndarray, p_clear_fut: np.ndarray, cap: np.ndarray, decision: dict | None = None,
               extra: dict | None = None):
        """决策层（numpy，与 m6 同一套函数；没有可训练参数，所以评估时可换规则不用重训）。
        q_raw [B,H,K] 真实单位的原始分位 → (point [B,H], q_cal, median, pstar, night_physics)。decision 缺省用 cfg['decision']。
        extra: {"mean_head": [B,H] 真实单位} —— by_step 里 method=mean_head 的步直接用它（经同一套物理裁剪），优先于其它规则。"""
        dec = decision if decision is not None else self.cfg["decision"]
        ph, pt = dec["physics"], dec["point"]
        if pt.get("tail", "flat") != "flat":
            raise ValueError(f"decision.point.tail: only 'flat' is implemented (got {pt.get('tail')!r})")
        if pt["default"] == "mean_head":
            raise ValueError("decision.point.default 不能是 mean_head（请写进 by_step，如 {steps: [55, 151], method: mean_head}）")
        night_physics = night_fut if ph["night_zero"] else np.zeros_like(night_fut)
        ratio, capcap = float(ph["clear_sky_cap_ratio"]), bool(ph["capacity_cap"])
        q_cal = physics_clip(enforce_monotone(q_raw), night_physics, p_clear_fut, cap[:, None], ratio, capcap)
        shrink = float(pt.get("shrink_factor") or 1.0)
        rules = list(pt.get("by_step") or ())
        mh_rules = [r for r in rules if r["method"] == "mean_head"]
        point = select_point(q_cal, self.levels, cap, pt["default"], [r for r in rules if r["method"] != "mean_head"], shrink,
                             float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
        if mh_rules:
            mh = None if extra is None else extra.get("mean_head")
            if mh is None:
                raise ValueError("decision.point.by_step 用了 mean_head，但模型没有 mean_head 输出（需 model.mean_head: true 训练）")
            mh = physics_clip(np.asarray(mh, float)[..., None], night_physics, p_clear_fut, cap[:, None], ratio, capcap)[..., 0]
            for r in mh_rules:
                lo, hi = int(r["steps"][0]), int(r["steps"][1])
                point[:, lo:hi] = mh[:, lo:hi]
        median = level_value(q_cal, self.levels, 0.5)
        pstar = p_star(q_cal, self.levels, cap[:, None], float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
        return point, q_cal, median, pstar, night_physics

    @torch.no_grad()
    def predict_raw_full(self, batch_dict: dict):
        """推理态：→ (q_fut [B,H,K] 真实单位 numpy, m_fut [B,H] 或 None, ctx)。不过决策层，供集成先平均再统一决策。"""
        was = self.training
        self.eval()
        try:
            inp = self.fe.build(batch_dict, training=False, y_future=None, rng=self._rng)
            q_norm, m_norm, scale_t, _ = self._run_core(inp)
            L = self.context_len
            q_fut = (q_norm[:, L:, :] * scale_t[:, None, None]).detach().float().cpu().numpy().astype(float)
            m_fut = (m_norm[:, L:] * scale_t[:, None]).detach().float().cpu().numpy().astype(float) if m_norm is not None else None
            return q_fut, m_fut, inp["ctx"]
        finally:
            self.train(was)

    @torch.no_grad()
    def predict_raw(self, batch_dict: dict):
        """推理态：→ (q_fut [B,H,K] 真实单位 numpy, ctx)。"""
        q_fut, _, ctx = self.predict_raw_full(batch_dict)
        return q_fut, ctx

    def forward(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        L, H = self.context_len, self.pred_len
        y_future = None
        if labels is not None:
            y = labels[self.label_col] if isinstance(labels, dict) else labels
            y_future = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        inp = self.fe.build(batch_dict, training=self.training, y_future=y_future, rng=self._rng)
        dev = next(self.parameters()).device
        q_norm, m_norm, scale_t, x = self._run_core(inp)
        q_fut = q_norm[:, L:, :] * scale_t[:, None, None]                         # [B, H, K] 真实单位
        m_fut = m_norm[:, L:] * scale_t[:, None] if m_norm is not None else None  # [B, H] 真实单位（mean_head）
        ctx = inp["ctx"]
        night_full = torch.as_tensor(ctx.night, device=dev)
        cap_t = torch.as_tensor(ctx.cap, dtype=torch.float32, device=dev)
        y_mask = x[..., self.fe.idx["y_mask"]]
        loss = None
        if labels is not None:
            loss = self._loss_from_parts(q_norm, m_norm, y_mask, night_full, torch.as_tensor(inp["target"], device=dev), scale_t, cap_t)

        extra = {"mean_head": m_fut.detach().float().cpu().numpy().astype(float)} if m_fut is not None else None
        point, q_cal, median, pstar, night_physics = self.decide(
            q_fut.detach().float().cpu().numpy().astype(float), ctx.night[:, L:], ctx.p_clear[:, L:], ctx.cap, extra=extra)
        logits = torch.as_tensor(point, dtype=torch.float32, device=dev)
        aux = {
            "quantiles": q_fut,
            "mean_head": m_fut,
            "quantiles_cal": torch.as_tensor(q_cal, dtype=torch.float32, device=dev),
            "levels_cal": list(self.levels),
            "p_star": torch.as_tensor(pstar, dtype=torch.float32, device=dev),
            "median": torch.as_tensor(median, dtype=torch.float32, device=dev),
            "night": night_physics, "night_loss": ctx.night[:, L:],
            "elev_deg": ctx.elev_deg[:, L:], "p_clear": ctx.p_clear[:, L:],
            "var_bin": np.zeros(len(ctx.cap), int), "var_signal": ctx.var_signal, "station_id": ctx.station_id,
            "cap": ctx.cap, "degraded": ctx.degraded,
            # compute_loss（FrameworkTrainer 路径）需要的中间量
            "_q_norm": q_norm, "_m_norm": m_norm, "_y_mask": y_mask, "_night_full": night_full, "_cap": cap_t,
            "_target_hist": torch.as_tensor(inp["target"][:, :L], device=dev),
            "_scale": scale_t, "_amp": torch.as_tensor(inp["amp"], dtype=torch.float32, device=dev),
        }
        return TransformersAdaptedModelOutput(loss=loss, logits=logits, predictions=logits, aux=aux)

    def _loss_from_parts(self, q_norm: torch.Tensor, m_norm, y_mask: torch.Tensor, night_full: torch.Tensor,
                         target_full: torch.Tensor, scale_t: torch.Tensor, cap_t: torch.Tensor) -> torch.Tensor:
        """pinball（可按步加权）+ 可选指标对齐项（见 config train.loss）。全部在归一化单位里算；
        南网分母 max(y, 0.2·cap) 在归一化单位里是 max(y_norm, 0.2·cap/scale)。"""
        L = self.context_len
        lc = self.cfg["train"].get("loss") or {}
        hw, ma = (lc.get("horizon_weights") or {}), (lc.get("metric_aux") or {})
        mask = ((y_mask > 0.5) & torch.isfinite(target_full) & (~night_full)).to(q_norm.dtype)      # [B, T]
        y = torch.nan_to_num(target_full)
        n_u, uw = int(hw.get("ultra_steps", 16)), float(hw.get("ultra_weight", 1.0))
        w = torch.ones(mask.shape[1], dtype=q_norm.dtype, device=q_norm.device)
        if n_u > 0 and uw != 1.0:
            w[L:L + n_u] = uw
        loss = pinball_loss(q_norm, y, self.levels, mask * w[None, :])
        a_l1, a_l2 = float(ma.get("ultra_l1", 0.0)), float(ma.get("short_l2", 0.0))
        a_m = float(ma.get("mean_head_weight", 1.0)) if m_norm is not None else 0.0
        if a_l1 <= 0 and a_l2 <= 0 and a_m <= 0:
            return loss
        den = torch.maximum(y, float(ma.get("floor_ratio", 0.2)) * (cap_t / scale_t)[:, None])
        if a_l1 > 0 and n_u > 0:
            med = quantile_value_t(q_norm, self.levels, 0.5)
            mk = mask[:, L:L + n_u]
            loss = loss + a_l1 * ((torch.abs(y - med) / den)[:, L:L + n_u] * mk).sum() / mk.sum().clamp_min(1.0)
        if a_m > 0:
            loss = loss + a_m * ((((y - m_norm) / den) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
        if a_l2 > 0:
            lo, hi = (int(v) for v in ma.get("short_steps", [55, 151]))
            pt = m_norm if m_norm is not None else quantile_value_t(q_norm, self.levels, 0.5)
            mk = mask[:, L + lo:L + hi]
            loss = loss + a_l2 * ((((y - pt) / den) ** 2)[:, L + lo:L + hi] * mk).sum() / mk.sum().clamp_min(1.0)
        return loss

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
        return self._loss_from_parts(aux["_q_norm"], aux["_m_norm"], aux["_y_mask"], aux["_night_full"], target_full, aux["_scale"], aux["_cap"])
