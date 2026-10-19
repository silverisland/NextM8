# -*- coding: utf-8 -*-
"""Timestral-209M 零样本封装（2026-10-19，替换原 PatchTST-FM-r2；仿照 M5 legacy 零样本，API 已对 timestral 1.0.0 源码核对）。

契约（与 M5 legacy 完全一致，按 pipeline 约定）：
  - torch.nn.Module；forward(batch_dict, labels=None)，推理只传一个位置参数；
    labels 仅为接口兼容保留，**传了也忽略**，恒不算 loss；
  - 返回 TransformersAdaptedModelOutput：.logits / .predictions 为 [B, pred_len]；
    分位数 [B, pred_len, Q] 放在 .aux["quantiles"]（Q=len(quantile_levels)，默认 9 档
    0.1~0.9，与 M5 同构，conformal / transformer_trainer.evaluate_model 直接复用）；
  - **点预测**：logits 默认取中位数（Timestral 无均值输出；与 M5 默认相同）；
    配置 timestral.predict_quantile 可让 logits 落在 quantile_levels 里的某一档
    （写分位值如 0.3，或写 quantile_levels 的索引如 2）；aux["quantiles"] 始终保留完整档位；
  - batch_dict 扁平 dict，key=parquet 列名，value=[B, L]（tensor/ndarray/list 均可，CPU）；
  - **设备自理**：默认 auto（有 cuda 用 cuda）；输出 tensor 在设备上；
  - **NaN 自理**：默认 "interp"（与 M5 相同，便于和 M5 零样本对齐比较）；另有 "zero" / "raise"，
    以及 "native"（NaN 原样交给 Timestral，由它的观测掩码当缺失值处理；整行全 NaN 仍置 0）。

协变量（与 M5 的 timesfm3 段同构）：
  timestral.past_future_cols -> [[历史列, 未来列], ...]：历史 context_len + 未来 pred_len 拼起来，
                                作为 known_covariates [B, C, L+H] 喂模型（NWP 辐照 / 温度走这里）；
  timestral.past_only_cols   -> [历史列, ...]：仅历史，作为 past_covariates [B, C, L]。
  缺列的协变量对只提示一次并跳过（同 M5）。Timestral 不支持静态协变量。

⚠️ 推理输入中 observe_power_future 不在本 adapter 读取范围内（同 M5，读了即未来真值泄漏）。

配置（config 文件驱动，与 M5 同构；m7/config/configs.yaml 是推理配置唯一真源）：
    dataset.pv_col                  -> 目标/历史功率列（默认 observe_power）
    timestral.checkpoint            -> 本地权重目录（含 config.json + model.safetensors）或 HF repo id
    timestral.context_len           -> 历史上下文长度（默认 672；模型上限 32768）
    timestral.pred_len              -> 预测长度（默认 192；≤1024 一次解码，更长按块自回归）
    timestral.device                -> "auto"|"cuda"|"cpu"
    timestral.per_core_batch_size   -> 单次喂模型的样本数（forward 内部再切块）
    timestral.past_future_cols / past_only_cols
    timestral.quantile_levels       -> 输出的分位档（须是模型 21 档的子集，默认 0.1~0.9）
    timestral.return_quantiles
    timestral.predict_quantile      -> logits 落在哪一档；None=中位数（默认）
    timestral.nan_policy            -> "interp"|"zero"|"raise"|"native"
    timestral.make_positive         -> 预测与分位钳到 ≥0（PV 功率建议 true，默认）
    timestral.symmetric_averaging   -> ±context 对称平均（timestral 默认开，同 M5 的 use_symmetric_averaging）
    timestral.eager_init            -> 构造时即加载权重（默认 true；测试里设 false 换替身）

安装（目标机）：
  pip install timestral                     # 或 pip install git+https://github.com/mistralai/timestral
  （timestral 声明 torch>=2.4、python>=3.10；环境 torch 较老时用 --no-deps 装，避免 pip 升级 torch）
  权重 mistralai/Timestral-209M（~840MB）：
    huggingface-cli download mistralai/Timestral-209M --local-dir /data/tangqingang/Timestral-209M
  ⚠️ 权重 license 为 Modified MIT：上月全球合并营收 > 2000 万美元的公司不得行使任何权利（需向 Mistral 申请商业授权）。

已核对的 API（timestral 1.0.0 src/timestral/forecaster.py）：
  TimestralForecaster.from_pretrained(path_or_repo_id, device="cpu")
  .quantile_levels -> 21 档 (0.01, 0.05, 0.1, ..., 0.95, 0.99)；.supports_known_covariates
  .forecast(context [B,V,T]（NaN=缺失）, prediction_length=H,
            past_covariates [B,C,T], known_covariates [B,C,T+H],
            symmetric_averaging=True, clip_nonnegative=True)
    -> Forecast(quantiles [B,V,H,Q] 已按分位排序, quantile_levels)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

try:  # 若被作为子包导入（registry 方式：forecast_backbones.m7.transformer_models）
    from ...interfaces.modeloutput import TransformersAdaptedModelOutput
except ImportError:  # 独立运行 __main__ / 顶层 sys.path 方式
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from interfaces.modeloutput import TransformersAdaptedModelOutput


DEFAULT_CFG = {
    "dataset": {"pv_col": "observe_power"},
    "timestral": {
        "checkpoint": "mistralai/Timestral-209M",
        "context_len": 672,
        "pred_len": 192,
        "device": "auto",           # "auto" | "cuda" | "cpu"
        "per_core_batch_size": 64,
        "past_future_cols": [
            ["general_ghi", "general_ghi_predict"],
            ["general_temp", "general_temp_predict"],
        ],
        "past_only_cols": [],
        # 与 M5 的 9 档对齐，下游 conformal / evaluate_model 的 q_0.1..q_0.9 列无需改动。
        "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "return_quantiles": True,
        "predict_quantile": None,   # None=中位数
        "nan_policy": "interp",     # "interp"（默认）| "zero" | "raise" | "native"
        "make_positive": True,      # 预测钳制在正值域（光伏功率 ≥0）
        "symmetric_averaging": True,
        "eager_init": True,
    },
}


def _merge(base: dict, override: dict) -> dict:
    """递归合并 config：override 里 dict 键才往下钻，否则整体替换（同 M5）。"""
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class TransformersAdaptedM7Model(nn.Module):
    """把 Timestral 包成 pipeline 标准 nn.Module（推理侧，零样本/无 loss）。
    必须是本文件第一个 nn.Module 子类（registry 自动发现）；别名 TransformersAdaptedTimestralModel 见文件末尾。"""

    def __init__(self, model_config: Optional[dict] = None, **kwargs):
        super().__init__()

        mc = model_config or {}
        override = {k: mc[k] for k in ("dataset", "timestral") if isinstance(mc.get(k), dict)}
        cfg = _merge(DEFAULT_CFG, override)
        self.target_col = cfg["dataset"].get("pv_col", DEFAULT_CFG["dataset"]["pv_col"])
        t = cfg["timestral"]
        self.checkpoint = t["checkpoint"]
        self.context_len = int(t["context_len"])
        self.pred_len = int(t["pred_len"])
        self.device_pref = t["device"]
        self.per_core_batch_size = int(t["per_core_batch_size"])
        self.past_future_cols = [tuple(p) for p in (t.get("past_future_cols") or [])]
        self.past_only_cols = list(t.get("past_only_cols") or [])
        self.quantile_levels = [float(q) for q in t["quantile_levels"]]
        self.return_quantiles = bool(t["return_quantiles"])
        self.predict_quantile = t["predict_quantile"]
        self.nan_policy = t["nan_policy"]
        if self.nan_policy not in ("interp", "zero", "raise", "native"):
            raise ValueError(f"timestral.nan_policy 非法: {self.nan_policy!r}")
        self.make_positive = bool(t["make_positive"])
        self.symmetric_averaging = bool(t["symmetric_averaging"])
        # 构造期就校验 predict_quantile，配置写错尽早报错。
        self._child_idx = self._quantile_index(self.predict_quantile, self.quantile_levels)
        self._backbone = None
        self._backbone_device = None
        self._q_idx = None              # quantile_levels 在模型 21 档里的索引，加载 backbone 时解析
        self._median_idx = None
        self._warned_pairs = set()      # 已提示过缺列的协变量对，只提示一次
        if bool(t["eager_init"]):
            self._ensure_backbone(self._resolve_device())

    # ------------------------------------------------------------------ 框架契约
    @property
    def context_length(self) -> int:                      # pipelines/rolling_wrapper_inference 读这两个名字
        return self.context_len

    @property
    def prediction_length(self) -> int:
        return self.pred_len

    # ------------------------------------------------------------------ 设备
    def _resolve_device(self) -> str:
        if self.device_pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_pref

    # ------------------------------------------------------------------ backbone
    def _ensure_backbone(self, want: str):
        """懒加载 TimestralForecaster；顶层只 import 本模块时不触发 timestral 导入/权重下载。"""
        if self._backbone is not None and self._backbone_device == want:
            return
        from timestral import TimestralForecaster   # 懒 import
        backbone = TimestralForecaster.from_pretrained(self.checkpoint, device=want)
        self._set_backbone(backbone, want)

    def _set_backbone(self, backbone, device: str):
        """挂载 backbone 并解析分位档索引（测试里可直接传替身）。

        backbone 不注册为子模块：与 M5 一致，adapter 的 state_dict 为空，外部 .to(device) 不搬权重，
        设备只由 timestral.device 决定。
        """
        levels = list(backbone.quantile_levels)
        self._q_idx = self._resolve_quantile_indices(self.quantile_levels, levels)
        self._median_idx = self._resolve_quantile_indices([0.5], levels)[0]
        if self.past_future_cols and not backbone.supports_known_covariates:
            raise ValueError("该 Timestral checkpoint 不支持未来已知协变量，请清空 timestral.past_future_cols")
        self.__dict__["_backbone"] = backbone
        self._backbone_device = device

    @staticmethod
    def _resolve_quantile_indices(wanted: List[float], available: List[float]) -> List[int]:
        """按容差把配置的分位档映射到模型 21 档的索引（避免浮点精确比较）。"""
        avail = np.asarray(available, dtype=np.float64)
        idx = []
        for q in wanted:
            j = int(np.argmin(np.abs(avail - q)))
            if abs(avail[j] - q) > 1e-6:
                raise ValueError(f"timestral.quantile_levels 里的 {q} 不在模型分位档中: {available}")
            idx.append(j)
        return idx

    # ------------------------------------------------------------------ NaN
    @staticmethod
    def _interp_nan(arr: np.ndarray) -> np.ndarray:
        """逐行沿时间轴线性内插；端点 NaN 用最近有效值延伸；整行 NaN 置 0（同 M5）。"""
        out = arr.copy()
        idx = np.arange(arr.shape[1])
        for i in range(arr.shape[0]):
            row = out[i]
            m = np.isfinite(row)
            if m.all():
                continue
            if not m.any():
                out[i] = 0.0
                continue
            row[~m] = np.interp(idx[~m], idx[m], row[m])
        return out

    # ------------------------------------------------------------------ 取列
    def _col(self, batch_dict: dict, col: str) -> np.ndarray:
        if col not in batch_dict:
            raise KeyError(
                f"batch_dict 缺列 {col!r}；现有 keys={sorted(batch_dict)[:20]}..."
            )
        v = batch_dict[col]
        arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:                       # 单样本容错 → [1, L]
            arr = arr[None, :]
        if not np.isfinite(arr).all():
            if self.nan_policy == "interp":
                arr = self._interp_nan(arr)
            elif self.nan_policy == "zero":
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            elif self.nan_policy == "native":
                arr = np.where(np.isinf(arr), np.nan, arr)
                arr[np.isnan(arr).all(axis=1)] = 0.0    # 整行缺失时没有可观测值，只能置 0
            else:
                bad = int((~np.isfinite(arr)).sum())
                raise ValueError(f"列 {col!r} 含 {bad} 个非有限值（nan_policy='raise'）")
        return arr

    # ------------------------------------------------------------------ 协变量
    def _covariates(self, batch_dict: dict):
        """→ (known [B,C,L+H] | None, past [B,C,L] | None)。"""
        L, H = self.context_len, self.pred_len
        pf = []
        for hist_c, fut_c in self.past_future_cols:
            if hist_c not in batch_dict or fut_c not in batch_dict:
                if (hist_c, fut_c) not in self._warned_pairs:
                    print(f"[timestral] 协变量对 {(hist_c, fut_c)} 在 batch 里缺列，本次推理跳过（只提示一次）")
                    self._warned_pairs.add((hist_c, fut_c))
                continue
            h = self._col(batch_dict, hist_c)[:, -L:]
            f = self._col(batch_dict, fut_c)[:, :H]
            pf.append(np.concatenate([h, f], axis=1))                           # [B, L+H]
        po = [self._col(batch_dict, c)[:, -L:] for c in self.past_only_cols]    # [B, L]
        known = np.stack(pf, axis=1) if pf else None
        past = np.stack(po, axis=1) if po else None
        return known, past

    # ------------------------------------------------------------------ 分位数选择
    @staticmethod
    def _quantile_index(predict_quantile, levels: List[float]) -> Optional[int]:
        """把 timestral.predict_quantile 映射到 levels 的索引。

        - None / "none"       -> None（用中位数）
        - "median"            -> 0.5 所在索引
        - (0,1) 内的 float    -> 该分位值所在索引（须在 levels 中）
        - 整数（或整数字符串）-> 直接当 levels 索引
        其余取值 raise ValueError，避免静默用错档位。
        """
        if predict_quantile is None:
            return None
        if isinstance(predict_quantile, bool):
            raise ValueError(f"timestral.predict_quantile 不接受 bool: {predict_quantile!r}")
        v = predict_quantile
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("", "none"):
                return None
            if s == "median":
                v = 0.5
            else:
                try:
                    v = int(s) if s.isdigit() else float(s)
                except ValueError:
                    raise ValueError(
                        f"timestral.predict_quantile 无法解析: {predict_quantile!r}；"
                        f"应为 {levels} 中的分位值 / 其索引 / median / None"
                    )
        if isinstance(v, float) and 0.0 < v < 1.0:
            hits = [i for i, q in enumerate(levels) if abs(q - v) < 1e-6]
            if not hits:
                raise ValueError(f"timestral.predict_quantile={v} 不在 quantile_levels {levels} 中")
            return hits[0]
        idx = int(v)
        if idx < 0 or idx >= len(levels):
            raise ValueError(
                f"timestral.predict_quantile 超出 quantile_levels 范围: {predict_quantile!r} -> 索引 {idx}"
            )
        return idx

    # ------------------------------------------------------------------ forward
    def forward(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        # labels 仅为接口兼容，忽略。
        if self._backbone is None:
            raise RuntimeError("backbone未加载")
        dev = torch.device(self._backbone_device)
        tgt = self._col(batch_dict, self.target_col)[:, -self.context_len:]     # [B, L]
        known, past = self._covariates(batch_dict)

        chunks = []
        bs = self.per_core_batch_size
        for s in range(0, tgt.shape[0], bs):
            fc = self._backbone.forecast(
                tgt[s: s + bs, None, :],                                          # [b, 1, L] 单目标
                prediction_length=self.pred_len,
                past_covariates=None if past is None else past[s: s + bs],
                known_covariates=None if known is None else known[s: s + bs],
                symmetric_averaging=self.symmetric_averaging,
                clip_nonnegative=self.make_positive,
            )
            chunks.append(torch.as_tensor(fc.quantiles)[:, 0].float())          # [b, H, 21]
        full = torch.cat(chunks, dim=0).to(dev)
        if self.make_positive:          # clip_nonnegative 只钳历史全非负的序列；PV 功率一律钳
            full = full.clamp_min(0.0)
        quantiles = full[:, :, self._q_idx].contiguous()                          # [B, H, Q]

        if self._child_idx is not None:
            logits = quantiles[:, :, self._child_idx]
        else:
            logits = full[:, :, self._median_idx]
        logits = logits.contiguous()

        aux = {"quantiles": quantiles} if self.return_quantiles else None
        return TransformersAdaptedModelOutput(logits=logits, predictions=logits, aux=aux)


# 语义别名：按模型名引用
TransformersAdaptedTimestralModel = TransformersAdaptedM7Model
