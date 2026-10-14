# 冻结副本：2026-10-14 重构前的 forecast_backbones/m5/transformer_models.py（逐字），只供 test_legacy_contract 对照，不要改。
# -*- coding: utf-8 -*-
"""TimesFM-3 零样本封装（2026-09-07，流程已对官方 API 核对）。

契约（按 pipeline 约定）：
  - torch.nn.Module；forward(batch_dict, labels=None)，推理只传一个位置参数；
    labels 仅为接口兼容保留，**传了也忽略**，恒不算 loss；
  - 返回 TransformersAdaptedModelOutput（pipeline 共享输出类，字段含
    loss/logits/predictions/hidden_states/embeddings/aux），
    .logits 非 None 且为 [B, pred_len] tensor；.predictions 同 logits；
    分位数 [B, pred_len, 9] 放在 .aux["quantiles"]（class 无 quantiles 字段，
    遵循仓库约定把辅助输出塞 aux；无分位头时为 None），供后续 conformal/南网 p* 用；
  - **分位选择**：logits 默认取中位数（0.5，来自点预测 forecast 的现状不变）；
    配置 timesfm3.predict_quantile 可让 logits 落在 9 档里的某一档
    （选值 0.1~0.9，或索引 0~8），灵活应变（如偏保守用 0.1、偏乐观用 0.9）。
    aux["quantiles"] 始终保留完整 9 档，不受该开关影响。
  - batch_dict 扁平 dict，key=parquet 列名，value=[B, L]（tensor/ndarray/list 均可，CPU）；
    一行一个样本，timestamp_win=起报时刻，observe_power=纯历史 672，_predict 列=未来 192；
  - **设备自理**：默认 auto（有 cuda 用 cuda），config 可强制 "cuda"/"cpu"；
    输出 tensor 在设备上（pipeline 推理会 .cpu() 收集，行为与 m1~m4 一致）；
  - **NaN 自理**：默认逐行沿时间轴线性内插（端点用最近有效值延伸），整行全 NaN 置 0；
    config 可切 "zero"（一律置 0）或 "raise"（发现即报错，排查数据时用）。

⚠️ 推理输入中 observe_power_future 不在本 adapter 读取范围内。
    eval_station_script.infer_station 默认把 observe_power_future 从推理输入剔除
    （drop_label_cols=True），若你绕过它直接传 batch_dict，务必不要读该列——一旦
    读取即为未来真值泄漏。

配置（config 文件驱动，与 m1~m4 的 config 模式对齐）：
  __init__(model_config) 读取 config 里的 dataset 段与 timesfm3 段，
  两者缺省值都回落到本模块的 DEFAULT_CFG（保证空 config 也能 0-shot 推理）：
    dataset.pv_col              -> 目标/历史功率列（默认 observe_power）
    timesfm3.context_len        -> 历史上下文长度（默认 672）
    timesfm3.pred_len           -> 预测长度（默认 192）
    timesfm3.checkpoint         -> 权重路径/HF repo（默认本地 3.0 pytorch 权重）
    timesfm3.device             -> "auto"|"cuda"|"cpu"
    timesfm3.past_future_cols   -> [[历史列, 未来列], ...] 未来已知协变量（两段拼 context+pred 喂模型）
    timesfm3.past_only_cols     -> [历史列, ...] 仅历史协变量（context 长）
    timesfm3.per_core_batch_size
    timesfm3.return_quantiles
    timesfm3.predict_quantile   -> logits 落在哪一档分位数：0.1~0.9（或索引 0~8）；None=中位数（默认）
    timesfm3.nan_policy         -> "interp"|"zero"|"raise"
    timesfm3.make_positive      -> 是否把预测钳制在正值域（PV 功率建议 true，默认）
    timesfm3.use_znorm          -> 是否对每个样本做 z 归一化后预测再逆归一化（默认 false）
    timesfm3.use_symmetric_averaging -> 是否对 ±target 对称平均增强鲁棒性（默认 true）
  例如 m5 的 config/configs.yaml（与 m4 同构，是推理配置唯一真源；train_script.py 按这个路径自动定位）：
    dataset: {pv_col: observe_power}
    timesfm3:
      context_len: 672
      pred_len: 192
      checkpoint: /data/tangqingang/timesfm-3.0-pytorch
      past_future_cols:
        - [general_ghi, general_ghi_predict]
        - [general_temp, general_temp_predict]
      past_only_cols: []
      device: auto
      per_core_batch_size: 32
      return_quantiles: true
      nan_policy: interp
      make_positive: true
      use_znorm: false
      use_symmetric_averaging: true

安装（目标机）：
  pip install "timesfm[torch]"
  权重 google/timesfm-3.0-pytorch：已下载到本机 /data/tangqingang/timesfm-3.0-pytorch，
  本机 DEFAULT_CFG 的 checkpoint 指向该本地路径（不走网络 HF 下载）。
  ⚠️ v3 权重 license 为非商用非生产（timesfm-non-commercial-license-v1.0）——本次仅零样本验证实验。

已核对的 API（timesfm3 包，本机 2026-09-07 实测）：
  ModelConfig(checkpoint_path, per_core_batch_size, device, ...)
    -> 注意 device 单位是 "cpu"/"cuda"（非张量 device 对象）
  TimesFM3Evaluator(ModelConfig(...))
  predict_batch(contexts: list[np.ndarray], horizon, *, past_only_covariates,
                past_future_covariates, return_quantiles, ...) -> Iterator[ForecastOutput]
    contexts: list of [L, 1]（单变量！本 adapter 用 (L,1) 按需转）
    past_future_covariates: list of [L]（每样本一个，shape 由极坐标决定）
  ForecastOutput(ts_id, forecast: ndarray, quantiles: ndarray | None)
  ⚠️ predict_batch 默认 make_positive=True，纯推零样本 OK；点预测单位与输入一致。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:  # 若被作为子包导入（registry 方式：forecast_backbones.m5.transformer_models）
    from ...interfaces.modeloutput import TransformersAdaptedModelOutput
except ImportError:  # 独立运行 __main__ / 顶层 sys.path 方式
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from interfaces.modeloutput import TransformersAdaptedModelOutput


DEFAULT_CFG = {
    "dataset": {"pv_col": "observe_power"},
    "timesfm3": {
        "checkpoint": "/data/tangqingang/timesfm-3.0-pytorch",
        "context_len": 672,
        "pred_len": 192,
        "device": "auto",           # "auto" | "cuda" | "cpu"
        # 未来已知协变量：[历史列(context_len), 未来列(pred_len)] 成对，二者拼成
        # context_len + pred_len 喂 past_future_covariates。
        # ⚠️ 必须与 parquet 实际列名一致，否则 forward 会 KeyError。
        # 本仓库 parquet 无 GHI_solargis / temp_solargis；如以后有再加：
        #   ["GHI_solargis", "ghi_solargis_predict"],  ["temp_solargis", "temp_solargis_predict"]
        "past_future_cols": [
            ["general_ghi", "general_ghi_predict"],
            ["general_temp", "general_temp_predict"],
        ],
        # 只有历史、未来未知的协变量列（context_len），暂无就留空。
        "past_only_cols": [],
        "per_core_batch_size": 32,
        "return_quantiles": True,
        # logits 落在哪一档分位数（0.1~0.9 或索引 0~8）；None/0.5=中位数（默认）。
        # 需要在 9 档里灵活应变时设成具体档位；aux["quantiles"] 始终保留完整 9 档。
        # ⚠️ 只在 return_quantiles=True 时生效；设为非中位档但 return_quantiles=False
        #    会无 9 档可切，此时回落到 forecast 点预测。
        "predict_quantile": None,
        "nan_policy": "interp",     # "interp"（线性内插，默认）| "zero" | "raise"
        # predict_batch 行为开关（直接透传给 timesfm3.predict_batch）：
        "make_positive": True,               # 预测钳制在正值域（光伏功率 ≥0）
        "use_znorm": False,                  # 逐样本 z 归一化（预测后逆归一化）
        "use_symmetric_averaging": True,     # 对 ±target 对称平均，增强鲁棒性
    },
}


def _merge(base: dict, override: dict) -> dict:
    """递归合并 config：override 里 dict 键才往下钻，否则整体替换。

    传进来的 model_config 是 yaml.safe_load 的顶层 dict（可能含 run/model 等
    与 adapter 无关的段），_merge 只按 DEFAULT_CFG 的已知键取子段并合并。
    """
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class TransformersAdaptedM5Model(nn.Module):
    """把 TimesFM-3 包成 pipeline 标准 nn.Module（推理侧，零样本/无 loss）。
    必须是本文件第一个 nn.Module 子类（registry 自动发现）；旧名 TransformersAdaptedTimesFM3Model 在文件末尾保留为别名。"""

    def __init__(self, model_config: Optional[dict] = None, **kwargs):
        super().__init__()

        # config 文件驱动：config 里的 dataset/timesfm3 段优先，缺省回落到 DEFAULT_CFG。
        # _merge 对 dict 逐键合并：config 只覆盖它出现的键，未出现的键沿用 DEFAULT_CFG。
        # model_config 可能带 run/model 等与 adapter 无关的顶层段，这里只挑
        # DEFAULT_CFG 认识的段做合并，其余不影响推理行为。
        mc = model_config or {}
        override = {k: mc[k] for k in ("dataset", "timesfm3") if isinstance(mc.get(k), dict)}
        cfg = _merge(DEFAULT_CFG, override)
        ds = cfg.get("dataset", {}) or {}
        self.target_col = ds.get("pv_col", DEFAULT_CFG["dataset"]["pv_col"])
        t = cfg.get("timesfm3", {}) or {}
        self.checkpoint = t.get("checkpoint", DEFAULT_CFG["timesfm3"]["checkpoint"])
        self.context_len = int(t.get("context_len", DEFAULT_CFG["timesfm3"]["context_len"]))
        self.pred_len = int(t.get("pred_len", DEFAULT_CFG["timesfm3"]["pred_len"]))
        self.device_pref = t.get("device", "auto")
        self.past_future_cols = [tuple(p) for p in t.get("past_future_cols", [])] if t.get("past_future_cols") else []
        self.past_only_cols = list(t.get("past_only_cols", []) or [])
        self.per_core_batch_size = int(t.get("per_core_batch_size", DEFAULT_CFG["timesfm3"]["per_core_batch_size"]))
        self.return_quantiles = bool(t.get("return_quantiles", DEFAULT_CFG["timesfm3"]["return_quantiles"]))
        self.predict_quantile = t.get("predict_quantile", DEFAULT_CFG["timesfm3"]["predict_quantile"])
        self.nan_policy = t.get("nan_policy", DEFAULT_CFG["timesfm3"]["nan_policy"])
        # predict_batch 行为开关（config 可调，缺省回落到 DEFAULT_CFG）
        self.make_positive = bool(t.get("make_positive", DEFAULT_CFG["timesfm3"]["make_positive"]))
        self.use_znorm = bool(t.get("use_znorm", DEFAULT_CFG["timesfm3"]["use_znorm"]))
        self.use_symmetric_averaging = bool(t.get("use_symmetric_averaging", DEFAULT_CFG["timesfm3"]["use_symmetric_averaging"]))
        self._backbone = None
        self._backbone_device = None
        self._warned_pairs = set()      # 已提示过缺列的协变量对，只提示一次
        if bool(t.get("eager_init",True)):
            self._ensure_backbone(self._resolve_device())

    # ------------------------------------------------------------------ 设备
    def _resolve_device(self) -> str:
        if self.device_pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_pref

    # ------------------------------------------------------------------ backbone
    def _ensure_backbone(self, want: str):
        """懒加载 TimesFM3 evaluator；顶层只 import 本模块时不触发权重下载。"""
        if self._backbone is not None and self._backbone_device == want:
            return
        from timesfm3 import ModelConfig, TimesFM3Evaluator   # 懒 import
        self._backbone = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=self.checkpoint,
            per_core_batch_size=self.per_core_batch_size,
            device=want,
        ))
        self._backbone_device = want

    # ------------------------------------------------------------------ NaN
    @staticmethod
    def _interp_nan(arr: np.ndarray) -> np.ndarray:
        """逐行沿时间轴线性内插；端点 NaN 用最近有效值延伸（np.interp 的钳位行为）；整行 NaN 置 0。"""
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
            else:
                bad = int((~np.isfinite(arr)).sum())
                raise ValueError(f"列 {col!r} 含 {bad} 个非有限值（nan_policy='raise'）")
        return arr

    # ------------------------------------------------------------------ 分位数选择
    @staticmethod
    def _quantile_index(predict_quantile) -> Optional[int]:
        """把 timesfm3.predict_quantile 映射到 9 档索引（0~8）。

        - None            -> None（不切换，沿用点预测 == 中位数）
        - "median"/0.5    -> 4（中位数）
        - 0.1~0.9（float/数字字符串）-> 对应档位索引（0.1→0, 0.5→4, 0.9→8）
        - 0~8（int/数字字符串）      -> 直接当索引
        - 其余取值 raise ValueError，避免静默用错档位。

        TimesFM-3 的 9 档固定为 [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]（见
        checkpoint 的 config.json：quantiles 字段）。
        """
        if predict_quantile is None:
            return None
        if isinstance(predict_quantile, str):
            s = predict_quantile.strip().lower()
            if s in ("", "none"):
                return None
            if s == "median":
                return 4
            try:
                v = float(s)
            except ValueError:
                raise ValueError(
                    f"timesfm3.predict_quantile 无法解析: {predict_quantile!r}；"
                    "应为 0.1~0.9 / 0~8 / median / None"
                )
        else:
            if isinstance(predict_quantile, bool):
                raise ValueError(f"timesfm3.predict_quantile 不接受 bool: {predict_quantile!r}")
            v = predict_quantile
        # 0.1~0.9 档位 -> 索引；其余（0~8）当索引。
        if isinstance(v, float) and 0.0 < v < 1.0:
            idx = int(round(v * 10)) - 1
        else:
            idx = int(v)
        if idx < 0 or idx > 8:
            raise ValueError(
                f"timesfm3.predict_quantile 超出 9 档范围: {predict_quantile!r} -> 索引 {idx}"
            )
        return idx

    # ------------------------------------------------------------------ forward
    def forward(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        # labels 仅为接口兼容，忽略。
        if self._backbone is None:
            raise RuntimeError(
                "backbone未加载"
            )
        dev=torch.device(self._backbone_device)
        # target：取后 context_len 个点 → [B, 672]；单变量 → feeds [B, 672, None]
        tgt = self._col(batch_dict, self.target_col)[:, -self.context_len:]
        B = tgt.shape[0]

        # 未来已知协变量：历史(672) + 未来(192) 拼成 864
        pf = {}
        for hist_c, fut_c in self.past_future_cols:
            if hist_c not in batch_dict or fut_c not in batch_dict:
                if (hist_c, fut_c) not in self._warned_pairs:
                    print(f"[timesfm3] 协变量对 {(hist_c, fut_c)} 在 batch 里缺列，本次推理跳过（只提示一次）")
                    self._warned_pairs.add((hist_c, fut_c))
                continue
            h = self._col(batch_dict, hist_c)[:, -self.context_len:]
            f = self._col(batch_dict, fut_c)[:, : self.pred_len]
            pf[(hist_c, fut_c)] = np.concatenate([h, f], axis=1)                 # [B,864]
        # 历史协变量：仅历史 672
        po = {c: self._col(batch_dict, c)[:, -self.context_len:] for c in self.past_only_cols}

        inputs, pf_list, po_list = [], [], []
        for i in range(B):
            # timesfm3 约定 axis0=变量数、axis1=时间；单目标 → (1, 672)
            inputs.append(tgt[i][None, :].astype(np.float32))                    # (1,672)
            if pf:
                pf_list.append(np.stack([pf[k][i] for k in pf]))                 # (C,864)
            if po:
                po_list.append(np.stack([po[c][i] for c in po]))                 # (C,672)

        kwargs = dict(
            horizon=self.pred_len,
            return_quantiles=self.return_quantiles,
            make_positive=self.make_positive,
            use_znorm=self.use_znorm,
            use_symmetric_averaging=self.use_symmetric_averaging,
        )
        if pf_list:
            kwargs["past_future_covariates"] = pf_list
        if po_list:
            kwargs["past_only_covariates"] = po_list
        outs = self._backbone.predict_batch(inputs, **kwargs)
        outs = list(outs)

        # forecast: (variates, horizon) 或 (horizon,) → [B, pred_len]
        fc = np.stack([np.asarray(o.forecast) for o in outs])
        if fc.ndim == 3:                      # (B, C, H) → 取第 0 变体
            fc = fc[:, 0, :]
        elif fc.ndim == 2:
            pass
        else:
            raise ValueError(f"无法解析 forecast 形状: {fc.shape}")

        # 完整 9 档分位 [B, pred_len, 9]，始终保留（不受 predict_quantile 影响）。
        quantiles = None
        q0 = getattr(outs[0], "quantiles", None)
        if self.return_quantiles and q0 is not None:
            qs = []
            for o in outs:
                q = np.asarray(o.quantiles)                                     # (variates,H,9) 或 (H,9)
                qs.append(q[0] if q.ndim == 3 else q)
            quantiles = torch.as_tensor(np.stack(qs), dtype=torch.float32, device=dev)  # [B,192,9]

        # 分位选择：默认 None -> 沿用点预测（中位数，与 m1~m4 现状一致）；
        # predict_quantile 设在 0.1~0.9 / 0~8 时，logits 落到对应档位。
        child_idx = self._quantile_index(self.predict_quantile)
        if child_idx is not None and quantiles is not None:
            fc = quantiles[:, :, child_idx].detach().cpu().numpy()
        logits = torch.as_tensor(fc, dtype=torch.float32, device=dev)

        # 仓库输出类没有 quantiles 字段，遵循约定放进 aux（隐形辅助输出）。
        aux = {"quantiles": quantiles} if quantiles is not None else None
        return TransformersAdaptedModelOutput(logits=logits, predictions=logits, aux=aux)


# 兼容旧名：历史代码/配置按这个名字引用
TransformersAdaptedTimesFM3Model = TransformersAdaptedM5Model
