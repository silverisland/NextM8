# -*- coding: utf-8 -*-
"""M5：TimesFM-3。两种模式，按 model_config 里有没有 dataset.station_meta 切换（2026-10-14）：

  legacy（没有 dataset.station_meta：线上 nanwang_online、eval_pretrain、experiments/nwp_covariate 的零样本配置）
      行为与重构前逐位相同，下面"零样本封装"的契约原样有效；tests/legacy_m5_forward.py 冻结了旧代码作对照。
  full（给了 dataset.station_meta：forecast_backbones/m5/train.py / evaluate.py / station_report.py 用）
      配置走 config.load_config（默认值见 config.py / config/finetune.yaml）；站表只取每站固定容量 cap_mw 与站号；
      协变量 = features.past_future（默认与线上相同：ghi、温度）+ 可选几何通道（features.derived，默认关）；
      底座：finetune.enabled → torch 可微路径挂 LoRA（并解冻 output_head）；否则零样本（finetune.inference_path：torch | evaluator）；
      决策层与 M8 相同（单调化 → 钳位 → median / p*），默认无晴空上限；labels 给定时 .loss = 9 档 pinball（默认容量归一）；
      aux 与 M8 同 schema。权重文件（train.weights_path）只存 LoRA + output_head。

────────────────────────────── 以下为 legacy 模式的原文档 ──────────────────────────────
TimesFM-3 零样本封装（2026-09-07，流程已对官方 API 核对）。

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
import warnings
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


def _dep(name: str):
    """full 模式的依赖按需导入：legacy 模式（线上）不 import pvlib / timesfm3.torch。"""
    import importlib
    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ImportError:
            pass
    return importlib.import_module(f"forecast_backbones.m5.{name}")


def _full_mode(model_config) -> bool:
    """给了 dataset.station_meta → full 模式（训练 / 评估）；否则 legacy（线上零样本，行为与重构前相同）。"""
    ds = model_config.get("dataset") if isinstance(model_config, dict) else None
    return isinstance(ds, dict) and bool(ds.get("station_meta"))


def load_station_meta(path, cols: dict | None = None):
    """dataset.station_meta（任意路径）+ dataset.station_meta_cols（列名映射）。路径不存在直接报错；
    只有仓库内默认相对路径（forecast_backbones/...）在别的工作目录下解析失败时才回退到 m5 自带那份。"""
    StationMeta = _dep("solar").StationMeta
    p = Path(path)
    if not p.exists():
        fallback = Path(__file__).parent / "station_meta.csv"
        if not p.is_absolute() and p.parts[:1] == ("forecast_backbones",) and fallback.exists():
            warnings.warn(f"station_meta {p} 不存在（相对当前工作目录解析失败）：回退到 {fallback}")
            return StationMeta.from_csv(fallback, cols)
        raise FileNotFoundError(f"station_meta {p} 不存在（dataset.station_meta；相对路径按当前工作目录解析）")
    return StationMeta.from_csv(p, cols)


def save_weights(model, path) -> None:
    """只存 LoRA + output_head（键名相对 backbone.model，safetensors）；底座不存。"""
    lora = _dep("lora")
    if getattr(model, "backbone", None) is None:
        raise RuntimeError("save_weights 需要 torch 路径（full 模式且 finetune.enabled 或 inference_path=torch）")
    lora.save_lora_weights(Path(path), lora.lora_state_dict(model.backbone.model))


def load_weights(model, path) -> None:
    lora = _dep("lora")
    blocks = list((model.cfg.get("finetune") or {}).get("unfreeze_modules") or [])
    lora.load_lora_state_dict(model.backbone.model, lora.load_lora_weights(Path(path)), ["output_head", *blocks])


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
        if _full_mode(model_config):
            self.mode = "full"
            self._init_full(model_config)
            return
        self.mode = "legacy"
        if ((model_config or {}).get("finetune") or {}).get("enabled"):
            raise ValueError("finetune.enabled 需要 full 模式：请在配置里给 dataset.station_meta（站表提供每站容量 cap_mw 与站号）")

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
        if self.mode == "full":
            _dep("_torch_compat").ensure_rmsnorm()            # torch < 2.4：须在 import timesfm3.torch 之前
        from timesfm3 import ModelConfig, TimesFM3Evaluator   # 懒 import
        self._backbone = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=self.checkpoint,
            per_core_batch_size=self.per_core_batch_size,
            device=want,
        ))
        self._backbone_device = want
        if self.mode == "full" and self.torch_path and self.backbone is None:
            self._attach_torch_path()

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

    # ================================================================== full 模式：构造 / 框架契约 / 权重
    def _init_full(self, model_config: dict) -> None:
        self.cfg = _dep("config").load_config(model_config)
        ds, t3, fe, ft = (self.cfg[k] for k in ("dataset", "timesfm3", "features", "finetune"))
        self.target_col, self.label_col = ds["pv_col"], ds["label_col"]
        self.context_len, self.pred_len = int(ds["context_len"]), int(ds["pred_len"])
        # 与 legacy 同名的属性：_resolve_device / _ensure_backbone / _col 两种模式共用
        self.checkpoint, self.device_pref = t3["checkpoint"], t3.get("device", "auto")
        self.per_core_batch_size, self.nan_policy = int(t3["per_core_batch_size"]), t3["nan_policy"]
        self.make_positive, self.use_znorm = bool(t3["make_positive"]), bool(t3["use_znorm"])
        self.use_symmetric_averaging = bool(t3["use_symmetric_averaging"])
        self.return_quantiles, self.predict_quantile = True, None
        self.past_future_cols = [tuple(p) for p in fe["past_future"]]
        self.past_only_cols = list(fe["past_only"])
        self._warned_pairs = set()
        self.station_meta = load_station_meta(ds["station_meta"], ds.get("station_meta_cols"))
        self.ctx_builder = _dep("rowctx").RowContextBuilder(self.cfg, self.station_meta)
        self.levels = tuple(float(q) for q in _dep("decision").MODEL_LEVELS)
        self.torch_path = bool(ft["enabled"]) or str(ft["inference_path"]) == "torch"
        self.backbone = None                      # torch 路径的 TorchBackbone 子模块（_attach_torch_path 赋值）
        self._backbone, self._backbone_device = None, None
        self.weights_loaded = False
        self._cond = None                         # adapters.CondState（prompt / 静态嵌入 / 门控 LoRA 开时）
        self._progress_desc = None
        if bool(t3.get("eager_init", True)):
            self._ensure_backbone(self._resolve_device())

    @property
    def config(self) -> dict:
        """full 模式：train.py 把合并后的 cfg 落成 model_config.yaml。legacy 模式没有这个属性（与重构前一致）。"""
        if self.mode != "full":
            raise AttributeError("config")
        return self.cfg

    def set_known_stations(self, station_ids) -> list[int]:
        """训练入口调用：记下训练集里的站号（只用于报表 seen / unseen，TimesFM 本身不用），同步进 cfg 随 model_config.yaml 落盘。"""
        ids = sorted({int(s) for s in station_ids})
        self.cfg["dataset"]["known_stations"] = ids
        return ids

    def reseed(self, seed: int) -> None:
        """train.py 契约；M5 没有随机增强，空操作。"""

    def n_params(self) -> int:
        """可训参数数（LoRA + output_head；零样本为 0）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _attach_torch_path(self) -> None:
        """从已加载的 evaluator 取底层 nn.Module 建 TorchBackbone；finetune.enabled 时挂 LoRA、冻结底座、加载 train.weights_path。"""
        lora = _dep("lora")
        self.backbone = _dep("backbone").TorchBackbone.from_evaluator(self._backbone)
        ft = self.cfg["finetune"]
        if ft["enabled"]:
            lo, root = ft["lora"], self.backbone.model
            blocks = list(ft.get("unfreeze_modules") or [])
            known = {n for n, _ in root.named_modules()}
            bad = [b for b in blocks if b not in known]
            if bad:
                raise ValueError(f"finetune.unfreeze_modules 里的模块不存在：{bad}（顶层可选：{sorted(n for n in known if n and '.' not in n)}）")
            lora.inject_lora(root, list(lo["target_modules"]), int(lo["rank"]), float(lo["alpha"]), float(lo["dropout"]),
                             int(lo.get("experts", 1) or 1))
            lora.inject_ia3(root, list((ft.get("ia3") or {}).get("target_modules") or []))
            self._cond = _dep("adapters").attach(root, ft, n_slots=1 + len(self.station_meta.ids()),
                                                 n_z=_dep("envelope").N_REGIME)
            lora.freeze_all_but_lora(root, bool(ft["unfreeze_output_head"]), unfreeze_modules=blocks)
            if not any(p.requires_grad for p in root.parameters()):
                raise ValueError("finetune.enabled=true 但没有任何可训参数：lora.target_modules 为空、unfreeze_output_head=false、"
                                 "unfreeze_modules 为空——至少开一项（只想零样本请设 finetune.enabled=false）")
            # 可训整块参数（output_head、unfreeze_modules）的预训练值留一份，WiSE-FT 插值要用；必须在加载微调权重之前取
            self._dense_pre = {n: p.detach().clone() for n, p in root.named_parameters()
                               if p.requires_grad and not lora._is_lora_name(n)}
            wp = Path(str(self.cfg["train"]["weights_path"]))
            if wp.exists():
                load_weights(self, wp)
                self.weights_loaded = True
        else:
            self.backbone.requires_grad_(False)
        self.backbone.eval()               # LoRA 的 dropout 默认 train 态：落回推理态，训练脚本显式 .train()

    def wise(self, alpha: float) -> None:
        """W2 WiSE-FT（评估用，可反复调）：LoRA 增量 × alpha，可训整块参数（output_head、unfreeze_modules）=
        alpha·微调 + (1−alpha)·预训练。alpha=1 = 原微调结果，alpha=0 = 零样本。第一次调用时把当前（微调后）的值存为插值端点。"""
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"WiSE alpha 必须在 [0, 1]（得到 {alpha}）")
        lora = _dep("lora")
        if getattr(self, "backbone", None) is None or not hasattr(self, "_dense_pre"):
            raise ValueError("WiSE 只适用于微调过的模型（finetune.enabled=true：LoRA / head / 解冻块）；零样本 run 没有可插值的增量")
        params = dict(self.backbone.model.named_parameters())
        if not hasattr(self, "_dense_ft"):
            self._dense_ft = {n: params[n].detach().clone() for n in self._dense_pre}
        for m in self.backbone.modules():
            if isinstance(m, lora.LoRALinear):
                m._scaling_ft = getattr(m, "_scaling_ft", m.scaling)
                m.scaling = alpha * m._scaling_ft
        with torch.no_grad():
            for n, ft_val in self._dense_ft.items():
                params[n].copy_(alpha * ft_val + (1.0 - alpha) * self._dense_pre[n].to(ft_val.device))
        self.wise_alpha = alpha

    # ================================================================== full 模式：推理 / 决策 / 损失
    def _covariates(self, batch_dict: dict, ctx):
        """→ (past_future [B,C,L+H] | None, past_only [B,C,L] | None)。缺列的协变量对只提示一次并跳过（与 legacy 同）。"""
        L, H = self.context_len, self.pred_len
        pf = []
        for hist_c, fut_c in self.past_future_cols:
            if hist_c not in batch_dict or fut_c not in batch_dict:
                if (hist_c, fut_c) not in self._warned_pairs:
                    print(f"[timesfm3] 协变量对 {(hist_c, fut_c)} 在 batch 里缺列，本次推理跳过（只提示一次）")
                    self._warned_pairs.add((hist_c, fut_c))
                continue
            pf.append(np.concatenate([self._col(batch_dict, hist_c)[:, -L:], self._col(batch_dict, fut_c)[:, :H]], axis=1))
        d = self.cfg["features"]["derived"]
        if d.get("solar_elevation"):
            pf.append(ctx.elev_deg.astype(np.float32))
        if d.get("clear_sky_power"):
            pf.append(ctx.p_clear.astype(np.float32))
        if d.get("history_envelope"):
            pf.append(self._envelopes(batch_dict, ctx)["power_env"].astype(np.float32))
        if d.get("envelope_nwp"):
            e = self._envelopes(batch_dict, ctx)
            pf.append((e["power_env"] * e["ratio"]).astype(np.float32))
        po = [self._col(batch_dict, c)[:, -L:] for c in self.past_only_cols]
        return (np.stack(pf, axis=1).astype(np.float32) if pf else None,
                np.stack(po, axis=1).astype(np.float32) if po else None)

    def _envelopes(self, batch_dict: dict, ctx) -> dict:
        """C1 / C2 / 形势特征共用的包络（envelope.py），一次前向只算一次（缓存挂在 ctx 上）。
        → power_env [B,T] 功率历史日形包络；ratio [B,T] = clip(GHI_nwp / GHI 历史包络, 0, clip_max)；
          k_fut [B,H] 未来晴空指数（上限 1.5）、day_fut [B,H] 白天步。"""
        cached = getattr(ctx, "_m5_env", None)
        if cached is not None:
            return cached
        ev, en = _dep("envelope"), self.cfg["features"]["envelope"]
        L, H = self.context_len, self.pred_len
        n_day = 1440 // int(self.cfg["dataset"]["step_min"])
        q, sm = float(en["quantile"]), int(en["smooth"])
        hist_c, fut_c = en["ghi"]
        missing = [c for c in (hist_c, fut_c) if c not in batch_dict]
        if missing:
            raise KeyError(f"features.envelope.ghi 的列 {missing} 不在数据里：历史包络 × NWP（envelope_nwp）与门控 / 路由 LoRA 的形势特征要用它，"
                           f"改 features.envelope.ghi 成数据里的 GHI 列对")
        y = np.asarray(self._col(batch_dict, self.target_col)[:, -L:], float)
        ghi_h = np.asarray(self._col(batch_dict, hist_c)[:, -L:], float)
        ghi = np.concatenate([ghi_h, np.asarray(self._col(batch_dict, fut_c)[:, :H], float)], axis=1)
        power_env = ev.envelope_rows(y, ctx.slot, n_day, q, sm)
        ghi_env = ev.envelope_rows(ghi_h, ctx.slot, n_day, q, sm)
        floor = float(en["ghi_floor"])
        out = {"power_env": power_env, "ratio": ev.nwp_ratio(ghi, ghi_env, floor, float(en["clip_max"])),
               "k_fut": ev.nwp_ratio(ghi[:, L:], ghi_env[:, L:], floor, 1.5), "day_fut": ghi_env[:, L:] > floor}
        ctx._m5_env = out
        return out

    def _set_cond(self, batch_dict: dict, ctx) -> None:
        """适配器的逐样本条件量写进 CondState（adapters.py）：静态特征、prompt 逐站行号、形势特征 → LoRA 门 / 路由权重。"""
        mods = getattr(self.backbone.model, _dep("lora").ADAPTER_ROOT)
        ad = _dep("adapters")
        dev = self.backbone.device
        sid = np.asarray(ctx.station_id, int)
        static = slot = z = lora_w = None
        if "static_embed" in mods:
            static = torch.as_tensor(ad.static_features(self.station_meta, lambda s: self.ctx_builder.station_caps(np.array([s]))[0], sid),
                                     device=dev)
        if "prompt" in mods and mods["prompt"].station is not None:
            known = set(int(s) for s in (self.cfg["dataset"].get("known_stations") or []))
            ids = self.station_meta.ids()
            slot = torch.as_tensor([1 + ids.index(int(s)) if int(s) in known and int(s) in ids else 0 for s in sid],
                                   dtype=torch.long, device=dev)
        if "lora_cond" in mods:
            e = self._envelopes(batch_dict, ctx)
            has = np.array([self.station_meta.has_coords(int(s)) for s in sid], float)
            z = torch.as_tensor(_dep("envelope").regime_features(e["k_fut"], e["day_fut"], has), device=dev)
            lora_w = mods["lora_cond"](z)
        self._cond.set(len(sid), static=static, slot=slot, z=z, lora_w=lora_w)

    def _quantiles(self, tgt: np.ndarray, pf, po) -> torch.Tensor:
        """→ [B,H,9] float32 真实单位。torch 路径可微（训练时带梯度）；evaluator 路径走 predict_batch（无梯度）。"""
        if self.torch_path:
            q = self.backbone.predict_quantiles(tgt, pf, po, horizon=self.pred_len, use_znorm=self.use_znorm,
                                                use_symmetric_averaging=self.use_symmetric_averaging,
                                                make_positive=self.make_positive)
            return q if q.dtype == torch.float32 else q.float()
        kwargs = dict(horizon=self.pred_len, return_quantiles=True, make_positive=self.make_positive,
                      use_znorm=self.use_znorm, use_symmetric_averaging=self.use_symmetric_averaging)
        if pf is not None:
            kwargs["past_future_covariates"] = list(pf)
        if po is not None:
            kwargs["past_only_covariates"] = list(po)
        qs = []
        for o in self._backbone.predict_batch([t[None, :] for t in tgt], **kwargs):
            if getattr(o, "quantiles", None) is None:
                raise ValueError("TimesFM-3 evaluator 没有返回分位数")
            q = np.asarray(o.quantiles)
            qs.append(q[0] if q.ndim == 3 else q)
        return torch.as_tensor(np.stack(qs), dtype=torch.float32, device=torch.device(self._backbone_device or "cpu"))

    def _raw(self, batch_dict: dict):
        if self._backbone is None:
            raise RuntimeError("backbone 未加载：timesfm3.eager_init=False 时需先 _ensure_backbone(device)")
        tgt = self._col(batch_dict, self.target_col)[:, -self.context_len:]
        ctx = self.ctx_builder.build(batch_dict)
        pf, po = self._covariates(batch_dict, ctx)
        if self._cond is not None:
            self._set_cond(batch_dict, ctx)
        return self._quantiles(tgt, pf, po), ctx

    @torch.no_grad()
    def predict_raw(self, batch_dict: dict):
        """推理态：→ (q [B,H,9] 真实单位 numpy, ctx)。不过决策层，供 evaluate.py 集成先平均分位再统一决策。"""
        if self.mode != "full":
            raise RuntimeError("predict_raw 只在 full 模式可用（配置里给 dataset.station_meta）")
        was = {m: m.training for m in self.modules()}      # 逐模块快照：主干在 eval、整体 training=True 的新建模型也要原样还原
        self.eval()
        try:
            q, ctx = self._raw(batch_dict)
            return q.detach().float().cpu().numpy().astype(float), ctx
        finally:
            for m, t in was.items():
                m.training = t

    def decide(self, q_raw: np.ndarray, night_fut: np.ndarray, p_clear_fut: np.ndarray, cap: np.ndarray, decision: dict | None = None):
        """决策层（numpy，没有可训参数，评估时可换规则不用重训）。decision 缺省用 cfg['decision']。
        → (point [B,H], q_cal, median, pstar, night_physics)。clear_sky_cap_ratio=None（默认）时无晴空上限。"""
        dm = _dep("decision")
        dec = decision if decision is not None else self.cfg["decision"]
        ph, pt = dec["physics"], dec["point"]
        tail = str(pt.get("tail") or "flat")
        nz, cn = bool(ph["night_zero"]), bool(ph.get("clip_negative", True))
        night_physics = night_fut if nz else np.zeros_like(night_fut)
        ratio = ph.get("clear_sky_cap_ratio")
        q_mono = dm.scale_spread(dm.enforce_monotone(q_raw), self.levels, pt.get("spread_lo"), pt.get("spread_hi"))
        q_cal = dm.physics_clip(q_mono, night_physics, p_clear_fut, cap[:, None],
                                None if ratio is None else float(ratio), bool(ph["capacity_cap"]),
                                night_ceiling=nz, clip_negative=cn)
        shrink = float(pt.get("shrink_factor") or 1.0)
        point = dm.select_point(q_cal, self.levels, cap, pt["default"], list(pt.get("by_step") or ()), shrink,
                                float(pt["floor_ratio"]), float(pt["tau_grid_step"]), tail)
        median = dm.level_value(q_cal, self.levels, 0.5)
        pstar = dm.p_star(q_cal, self.levels, cap[:, None], float(pt["floor_ratio"]), float(pt["tau_grid_step"]), tail)
        return point, q_cal, median, pstar, night_physics

    def _forward_full(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        L = self.context_len
        q_t, ctx = self._raw(batch_dict)                                          # [B,H,9]；torch 路径训练时带梯度
        dev = q_t.device
        night_fut, p_clear_fut = ctx.night[:, L:], ctx.p_clear[:, L:]
        point, q_cal, median, pstar, night_physics = self.decide(
            q_t.detach().float().cpu().numpy().astype(float), night_fut, p_clear_fut, ctx.cap)
        logits = torch.as_tensor(point, dtype=torch.float32, device=dev)
        aux = {
            "quantiles": q_t,
            "quantiles_cal": torch.as_tensor(q_cal, dtype=torch.float32, device=dev),
            "levels_cal": list(self.levels),
            "p_star": torch.as_tensor(pstar, dtype=torch.float32, device=dev),
            "median": torch.as_tensor(median, dtype=torch.float32, device=dev),
            "night": night_physics, "night_loss": night_fut,
            "elev_deg": ctx.elev_deg[:, L:], "p_clear": p_clear_fut,
            "var_bin": np.zeros(len(ctx.cap), int), "var_signal": ctx.var_signal, "station_id": ctx.station_id,
            "cap": ctx.cap, "degraded": ctx.degraded,
        }
        loss = None if labels is None else self._loss(q_t, labels, night_fut, ctx.cap)
        return TransformersAdaptedModelOutput(loss=loss, logits=logits, predictions=logits, aux=aux)

    def _loss(self, q_t: torch.Tensor, labels, night_fut: np.ndarray, cap: np.ndarray) -> torch.Tensor:
        """9 档 pinball，监督未来 pred_len 步；标签非有限的步掩掉；夜间步按 train.loss.night_weight 加权；
        normalize_by_cap（默认）时 q、y 都除以该行 cap。"""
        y = labels[self.label_col] if isinstance(labels, dict) else labels
        y = torch.as_tensor(np.asarray(y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y, np.float32),
                            device=q_t.device)[:, : self.pred_len]
        lc = self.cfg["train"]["loss"]
        mask = torch.isfinite(y).to(q_t.dtype)
        nw = float(lc.get("night_weight", 1.0))
        if nw != 1.0:
            night = torch.as_tensor(np.asarray(night_fut, bool), device=q_t.device)
            mask = mask * torch.where(night, torch.full_like(mask, nw), torch.ones_like(mask))
        y = torch.nan_to_num(y)
        if bool(lc.get("normalize_by_cap", True)):
            c = torch.as_tensor(np.asarray(cap, np.float32), device=q_t.device)
            q_t, y = q_t / c[:, None, None], y / c[:, None]
        return _dep("losses").pinball_loss(q_t, y, self.levels, mask)

    def compute_loss(self, outputs, labels, num_items_in_batch=None) -> torch.Tensor:
        if outputs.loss is not None:
            return outputs.loss
        aux = outputs.aux
        return self._loss(aux["quantiles"], labels, aux["night_loss"], aux["cap"])

    # ------------------------------------------------------------------ forward
    def forward(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        if self.mode == "full":
            return self._forward_full(batch_dict, labels)
        # labels 仅为接口兼容，忽略。（以下为 legacy 模式，逐字沿用重构前代码）
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
