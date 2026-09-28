# -*- coding: utf-8 -*-
"""PatchTST-FM-r2 零样本封装（2026-09-28，仿照 M5/TimesFM-3，流程已对 granite-tsfm 源码核对）。

契约（与 M5 完全一致，按 pipeline 约定）：
  - torch.nn.Module；forward(batch_dict, labels=None)，推理只传一个位置参数；
    labels 仅为接口兼容保留，**传了也忽略**，恒不算 loss；
  - 返回 TransformersAdaptedModelOutput：.logits / .predictions 为 [B, pred_len]；
    分位数 [B, pred_len, Q] 放在 .aux["quantiles"]（Q=len(quantile_levels)，默认 9 档
    0.1~0.9，与 M5 同构，conformal / transformer_trainer.evaluate_model 直接复用）；
  - **点预测**：logits 默认取模型自带点预测 prediction_outputs（对 99 档分位函数积分得到的均值）；
    配置 patchtst_fm.predict_quantile 可让 logits 落在 quantile_levels 里的某一档
    （写分位值如 0.5，或写 quantile_levels 的索引如 4）；aux["quantiles"] 始终保留完整档位；
  - batch_dict 扁平 dict，key=parquet 列名，value=[B, L]（tensor/ndarray/list 均可，CPU）；
  - **设备自理**：默认 auto（有 cuda 用 cuda）；输出 tensor 在设备上；
  - **NaN 自理**：默认 "interp"（与 M5 相同）；另有 "zero" / "raise"，以及 "native"
    （NaN 原样交给模型，由 past_observed_mask 当缺失值处理；整行全 NaN 仍置 0）。

⚠️ 与 M5 的关键差异：**PatchTST-FM 不支持协变量**。
    模型按通道独立处理（源码 rearrange "B T N -> (B N) T"），输入层只吃 [值, 缺失掩码]，
    PatchTSTFMForPrediction.forward 的 future_values 仍是注释掉的 "future use"。
    因此本 adapter 只读 dataset.pv_col 一列历史功率，general_ghi_predict 等 NWP 列一律不用；
    与 M5 对比时，应与 M5 的 past_future_cols: [] 消融结果对齐比较才公平。

⚠️ 推理输入中 observe_power_future 不在本 adapter 读取范围内（同 M5，读了即未来真值泄漏）。

配置（config 文件驱动，与 M5 同构；m7/config/configs.yaml 是推理配置唯一真源）：
    dataset.pv_col                  -> 目标/历史功率列（默认 observe_power）
    patchtst_fm.checkpoint          -> 权重路径/HF repo（默认 HF repo，服务器上请指向本地目录）
    patchtst_fm.context_len         -> 历史上下文长度（默认 672；context_len+pred_len ≤ 8192）
    patchtst_fm.pred_len            -> 预测长度（默认 192）
    patchtst_fm.device              -> "auto"|"cuda"|"cpu"
    patchtst_fm.per_core_batch_size -> 单次喂模型的样本数（forward 内部再切块）
    patchtst_fm.quantile_levels     -> 输出的分位档（须是模型 99 档 0.01~0.99 的子集，默认 0.1~0.9）
    patchtst_fm.return_quantiles
    patchtst_fm.predict_quantile    -> logits 落在哪一档；None=模型点预测（均值，默认）
    patchtst_fm.nan_policy          -> "interp"|"zero"|"raise"|"native"
    patchtst_fm.make_positive       -> 预测与分位钳到 ≥0（PV 功率建议 true，默认）
    patchtst_fm.eager_init          -> 构造时即加载权重（默认 true；测试里设 false 换替身）

安装（目标机）：
  pip install "granite-tsfm>=0.3.9"         # 提供 tsfm_public.PatchTSTFMForPrediction
  权重 ibm-granite/granite-timeseries-patchtst-fm-r2（~1.5GB，Apache-2.0 / OpenMDW-1.0，可商用）：
    huggingface-cli download ibm-granite/granite-timeseries-patchtst-fm-r2 \
        --local-dir /data/tangqingang/granite-timeseries-patchtst-fm-r2

已核对的 API（granite-tsfm 源码 tsfm_public/models/patchtst_fm/modeling_patchtst_fm.py）：
  PatchTSTFMForPrediction.from_pretrained(path)
  forward(past_values: [B, T, N] tensor（NaN=缺失）, prediction_length=H, return_dict=True)
    -> prediction_outputs: [B, H, N]（分位函数积分得到的均值）
       quantile_outputs:   [B, 99, H, N]（config.quantile_levels = 0.01..0.99）
  context 不足 config.context_length(8192) 时模型内部左侧 pad（pad_mask），无需手动补齐。
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
    "patchtst_fm": {
        "checkpoint": "ibm-granite/granite-timeseries-patchtst-fm-r2",
        "context_len": 672,
        "pred_len": 192,
        "device": "auto",           # "auto" | "cuda" | "cpu"
        "per_core_batch_size": 64,
        # 与 M5 的 9 档对齐，下游 conformal / evaluate_model 的 q_0.1..q_0.9 列无需改动。
        "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "return_quantiles": True,
        # logits 落在哪一档（分位值如 0.5，或 quantile_levels 的索引如 4）；None=模型点预测（均值）。
        "predict_quantile": None,
        "nan_policy": "interp",     # "interp"（默认）| "zero" | "raise" | "native"
        "make_positive": True,      # 预测钳制在正值域（光伏功率 ≥0）
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
    """把 PatchTST-FM-r2 包成 pipeline 标准 nn.Module（推理侧，零样本/无 loss）。
    必须是本文件第一个 nn.Module 子类（registry 自动发现）；别名 TransformersAdaptedPatchTSTFMModel 见文件末尾。"""

    def __init__(self, model_config: Optional[dict] = None, **kwargs):
        super().__init__()

        mc = model_config or {}
        override = {k: mc[k] for k in ("dataset", "patchtst_fm") if isinstance(mc.get(k), dict)}
        cfg = _merge(DEFAULT_CFG, override)
        self.target_col = cfg["dataset"].get("pv_col", DEFAULT_CFG["dataset"]["pv_col"])
        t = cfg["patchtst_fm"]
        self.checkpoint = t["checkpoint"]
        self.context_len = int(t["context_len"])
        self.pred_len = int(t["pred_len"])
        self.device_pref = t["device"]
        self.per_core_batch_size = int(t["per_core_batch_size"])
        self.quantile_levels = [float(q) for q in t["quantile_levels"]]
        self.return_quantiles = bool(t["return_quantiles"])
        self.predict_quantile = t["predict_quantile"]
        self.nan_policy = t["nan_policy"]
        if self.nan_policy not in ("interp", "zero", "raise", "native"):
            raise ValueError(f"patchtst_fm.nan_policy 非法: {self.nan_policy!r}")
        self.make_positive = bool(t["make_positive"])
        # 构造期就校验 predict_quantile，配置写错尽早报错。
        self._child_idx = self._quantile_index(self.predict_quantile, self.quantile_levels)
        self._backbone = None
        self._backbone_device = None
        self._q_idx = None              # quantile_levels 在模型 99 档里的索引，加载 backbone 时解析
        if bool(t["eager_init"]):
            self._ensure_backbone(self._resolve_device())

    # ------------------------------------------------------------------ 设备
    def _resolve_device(self) -> str:
        if self.device_pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_pref

    # ------------------------------------------------------------------ backbone
    def _ensure_backbone(self, want: str):
        """懒加载 PatchTSTFMForPrediction；顶层只 import 本模块时不触发 granite-tsfm 导入/权重下载。"""
        if self._backbone is not None and self._backbone_device == want:
            return
        from tsfm_public import PatchTSTFMForPrediction   # 懒 import
        backbone = PatchTSTFMForPrediction.from_pretrained(self.checkpoint)
        self._set_backbone(backbone, want)

    def _set_backbone(self, backbone, device: str):
        """挂载 backbone 并解析分位档索引（测试里可直接传小模型）。

        backbone 不注册为子模块：与 M5 一致，adapter 的 state_dict 为空，外部 .to(device) 不搬权重，
        设备只由 patchtst_fm.device 决定。
        """
        backbone = backbone.to(device).eval()
        self._q_idx = self._resolve_quantile_indices(self.quantile_levels, list(backbone.config.quantile_levels))
        self.__dict__["_backbone"] = backbone
        self._backbone_device = device

    @staticmethod
    def _resolve_quantile_indices(wanted: List[float], available: List[float]) -> List[int]:
        """按容差把配置的分位档映射到模型 99 档的索引（避免浮点精确比较）。"""
        avail = np.asarray(available, dtype=np.float64)
        idx = []
        for q in wanted:
            j = int(np.argmin(np.abs(avail - q)))
            if abs(avail[j] - q) > 1e-6:
                raise ValueError(f"patchtst_fm.quantile_levels 里的 {q} 不在模型分位档中: {available}")
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
                arr[np.isnan(arr).all(axis=1)] = 0.0    # 整行缺失时模型内部均值也是 NaN，只能置 0
            else:
                bad = int((~np.isfinite(arr)).sum())
                raise ValueError(f"列 {col!r} 含 {bad} 个非有限值（nan_policy='raise'）")
        return arr

    # ------------------------------------------------------------------ 分位数选择
    @staticmethod
    def _quantile_index(predict_quantile, levels: List[float]) -> Optional[int]:
        """把 patchtst_fm.predict_quantile 映射到 levels 的索引。

        - None / "none"       -> None（用模型点预测）
        - "median"            -> 0.5 所在索引
        - (0,1) 内的 float    -> 该分位值所在索引（须在 levels 中）
        - 整数（或整数字符串）-> 直接当 levels 索引
        其余取值 raise ValueError，避免静默用错档位。
        """
        if predict_quantile is None:
            return None
        if isinstance(predict_quantile, bool):
            raise ValueError(f"patchtst_fm.predict_quantile 不接受 bool: {predict_quantile!r}")
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
                        f"patchtst_fm.predict_quantile 无法解析: {predict_quantile!r}；"
                        f"应为 {levels} 中的分位值 / 其索引 / median / None"
                    )
        if isinstance(v, float) and 0.0 < v < 1.0:
            hits = [i for i, q in enumerate(levels) if abs(q - v) < 1e-6]
            if not hits:
                raise ValueError(f"patchtst_fm.predict_quantile={v} 不在 quantile_levels {levels} 中")
            return hits[0]
        idx = int(v)
        if idx < 0 or idx >= len(levels):
            raise ValueError(
                f"patchtst_fm.predict_quantile 超出 quantile_levels 范围: {predict_quantile!r} -> 索引 {idx}"
            )
        return idx

    # ------------------------------------------------------------------ forward
    def forward(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        # labels 仅为接口兼容，忽略。
        if self._backbone is None:
            raise RuntimeError("backbone未加载")
        dev = torch.device(self._backbone_device)
        # target：取后 context_len 个点 → [B, 672]；单变量 → past_values [B, 672, 1]
        tgt = self._col(batch_dict, self.target_col)[:, -self.context_len:]
        x = torch.as_tensor(tgt, dtype=torch.float32, device=dev).unsqueeze(-1)

        points, quants = [], []
        with torch.no_grad():
            for s in range(0, x.shape[0], self.per_core_batch_size):
                out = self._backbone(
                    past_values=x[s: s + self.per_core_batch_size],
                    prediction_length=self.pred_len,
                    return_dict=True,
                )
                points.append(out.prediction_outputs[..., 0].float())                   # [b, H]
                quants.append(out.quantile_outputs[:, self._q_idx, :, 0].float())       # [b, Q, H]
        fc = torch.cat(points, dim=0)
        quantiles = torch.cat(quants, dim=0).permute(0, 2, 1).contiguous()              # [B, H, Q]
        if self.make_positive:
            fc = fc.clamp_min(0.0)
            quantiles = quantiles.clamp_min(0.0)

        if self._child_idx is not None:
            fc = quantiles[:, :, self._child_idx]
        logits = fc.to(dev)

        aux = {"quantiles": quantiles} if self.return_quantiles else None
        return TransformersAdaptedModelOutput(logits=logits, predictions=logits, aux=aux)


# 语义别名：按模型名引用
TransformersAdaptedPatchTSTFMModel = TransformersAdaptedM7Model
