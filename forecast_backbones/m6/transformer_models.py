# -*- coding: utf-8 -*-
# forecast_backbones/m6/transformer_models.py
"""M6：TimesFM-3 底座 + 逐站 conformal 分位校准 + 南网口径决策层（路线 A：evaluator 路径，零样本无梯度）。

契约与 m5 相同：forward(batch_dict, labels=None) -> TransformersAdaptedModelOutput，
.logits [B,pred_len] = 提交曲线（默认按 decision.point 选 p*/p*_L1，已夜间归零与物理钳位），
分位等辅助输出进 .aux。labels 给定时 .loss = pinball（本路径无梯度，仅报告）。
配置：load_config(model_config) 与 DEFAULT_CFG 深合并；run/model/dataset/timesfm3 四段与 m5 同名兼容。
⚠️ 推理输入不读 observe_power_future。

校准装载（calibration.enabled=True 时）：
  - table_path 指向的文件不存在 → warnings.warn 并本次不校准（calibrator=None，aux["quantiles_cal"]
    走未校准路径，等价于 calibration.enabled=False 时的输出）；
  - 文件存在但同名 `<table_path>.json` sidecar 缺失，或 sidecar 记录的拟合期配置（levels/model_levels/
    normalization/elev_edges/lead_edges/use_variability/n_min_factor）与本次 cfg 不一致 → `Calibrator.load`
    抛 ValueError，直接从 `__init__` 冒出（需要重新 fit，不吞掉该错误）。"""
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
    from .features import StationMeta, build_row_context
    from .masks import loss_mask, missing_mask
    from .decision import enforce_monotone, physics_clip, select_point, level_value, p_star, MODEL_LEVELS
    from .calibration import Calibrator, CalibrationConfig
    from .losses import pinball_loss
    from ._arrays import interp_nan
    from ._torch_compat import ensure_rmsnorm
except ImportError:
    from config import load_config
    from features import StationMeta, build_row_context
    from masks import loss_mask, missing_mask
    from decision import enforce_monotone, physics_clip, select_point, level_value, p_star, MODEL_LEVELS
    from calibration import Calibrator, CalibrationConfig
    from losses import pinball_loss
    from _arrays import interp_nan
    from _torch_compat import ensure_rmsnorm

# 已经就哪些 past_future 协变量对发过"batch_dict 缺列"警告——跨调用去重，同一进程内每对只提示一次
# （与 m5 的 self._warned_pairs 语义相同，这里用模块级集合而不是实例级）。
_WARNED_MISSING_PAIRS: set[tuple[str, str]] = set()


def _lazy_torch_path_imports():
    """惰性导入 torch 路径依赖（backbone.py 一 import 就拉 timesfm3）：路线 A 的 Global Constraint
    要求 transformer_models.py 模块级导入不依赖 timesfm3，所以只在 _attach_torch_path 里调用本函数。"""
    try:
        from .backbone import TorchBackbone
        from .lora import inject_lora, freeze_all_but_lora, load_lora_state_dict, set_lora_enabled, trainable_summary
    except ImportError:
        from backbone import TorchBackbone
        from lora import inject_lora, freeze_all_but_lora, load_lora_state_dict, set_lora_enabled, trainable_summary
    return TorchBackbone, inject_lora, freeze_all_but_lora, load_lora_state_dict, set_lora_enabled, trainable_summary


def _lazy_routing_imports():
    """惰性导入 routing.py（本身不依赖 timesfm3，惰性只是与 _lazy_torch_path_imports 风格一致）。"""
    try:
        from .routing import save_routing, load_routing
    except ImportError:
        from routing import save_routing, load_routing
    return save_routing, load_routing


class TransformersAdaptedM6Model(nn.Module):
    """必须是本文件第一个 nn.Module 子类（registry 自动发现）。"""

    def __init__(self, model_config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.cfg = load_config(model_config)
        ds, t3 = self.cfg["dataset"], self.cfg["timesfm3"]
        self.target_col = ds["pv_col"]
        self.label_col = ds["label_col"]
        self.context_len = int(t3["context_len"]); self.pred_len = int(t3["pred_len"])
        self.past_future_cols = [tuple(p) for p in self.cfg["features"]["past_future"]]
        self.past_only_cols = list(self.cfg["features"]["past_only"])
        self.nan_policy = t3["nan_policy"]
        self.station_meta = self._load_station_meta(ds["station_meta"])
        self.shrink_factor = float(self.cfg["decision"]["point"].get("shrink_factor") or 1.0)
        self.calibrator: Optional[Calibrator] = None
        self._calibration_table_missing = False
        c = self.cfg["calibration"]
        if c["enabled"]:
            p = Path(c["table_path"])
            if p.exists():
                self.calibrator = Calibrator.load(p, CalibrationConfig.from_cfg(self.cfg))
            else:
                self._calibration_table_missing = True
                warnings.warn(
                    f"calibration.enabled 但 {p} 不存在 → 本次不校准：p* 将在未校准的 9 个原生 "
                    f"MODEL_LEVELS 上计算，该分位集合系统性偏高（README 备注）；decision.point.default "
                    f"与 by_step 里配置成 p_star/p_star_l1 的方法本次已回退为 median（cfg 本身不改）"
                )
        self._backbone = None; self._backbone_device = None
        self.device_pref = t3.get("device", "auto")
        ft = self.cfg["finetune"]
        self.torch_path = bool(ft["enabled"]) or str(ft.get("inference_path", "evaluator")) == "torch"
        self.backbone = None                       # torch 路径：TorchBackbone 子模块（_attach_torch_path 赋值）
        self.lora_loaded = False
        self.station_routing = None                # {station_id: "lora"|"zeroshot"}；finetune.enabled 且路由表存在时加载
        if ft["enabled"]:
            rp = Path(str(ft["routing_path"]))
            if rp.exists():
                self.station_routing = _lazy_routing_imports()[1](rp)
            else:
                warnings.warn(f"finetune.enabled=True 但站级路由表 {rp} 不存在：不做逐站路由"
                              f"（station_routing 保持 None）")
        if bool(t3.get("eager_init", True)):
            self._ensure_backbone(self._resolve_device())

    # ------------------------------------------------------------------ 装载
    @staticmethod
    def _load_station_meta(path) -> StationMeta:
        p = Path(path)
        if not p.exists():
            if not p.is_absolute():
                fallback = Path(__file__).parent / "station_meta.csv"
                if fallback.exists():
                    warnings.warn(f"station_meta {p} 不存在（相对路径，相对当前工作目录解析失败）："
                                  f"回退到包内默认文件 {fallback}")
                    return StationMeta.from_csv(fallback)
            warnings.warn(f"station_meta {p} 不存在：全部站按缺坐标退化")
            import pandas as pd
            return StationMeta(pd.DataFrame(columns=list(StationMeta.COLS)))
        return StationMeta.from_csv(p)

    def set_calibrator(self, cal: Optional[Calibrator]):
        self.calibrator = cal

    def _effective_point_methods(self) -> tuple[str, list]:
        """decision.point.default / by_step 的“生效版”：cfg 本身永不改写。仅当本次因校准表缺失而
        退化未校准（_calibration_table_missing 且当前确实没有 calibrator——set_calibrator 补上后
        不再回退）时，把配置成 p_star/p_star_l1 的方法换成 median：未校准的 9 档 MODEL_LEVELS 上跑
        p*/p*_L1 系统性偏高，此时不如直接用中位数。calibration.enabled=False（主动关闭校准，非
        缺表）时不受影响，仍按配置跑 p_star/p_star_l1。"""
        pt = self.cfg["decision"]["point"]
        default, by_step = pt["default"], list(pt.get("by_step") or ())
        if not (self._calibration_table_missing and self.calibrator is None):
            return default, by_step
        fb = lambda method: "median" if method in ("p_star", "p_star_l1") else method
        return fb(default), [dict(rule, method=fb(rule["method"])) for rule in by_step]

    def _resolve_device(self) -> str:
        if self.device_pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_pref

    def _ensure_backbone(self, want: str):
        ensure_rmsnorm()          # 幂等；torch < 2.4 时必须在 import timesfm3.torch.* 之前打上，backbone.py 未必已 import 过
        if self._backbone is not None and self._backbone_device == want:
            return
        from timesfm3 import ModelConfig, TimesFM3Evaluator            # 懒 import
        t3 = self.cfg["timesfm3"]
        self._backbone = TimesFM3Evaluator(ModelConfig(checkpoint_path=t3["checkpoint"],
                                                       per_core_batch_size=int(t3["per_core_batch_size"]),
                                                       device=want))
        self._backbone_device = want
        if self.torch_path and self.backbone is None:
            self._attach_torch_path()

    def _attach_torch_path(self) -> None:
        """从已加载的 evaluator 取底层 nn.Module 建 TorchBackbone；finetune.enabled 时挂 LoRA、冻结、加载权重。"""
        TorchBackbone, inject_lora, freeze_all_but_lora, load_lora_state_dict, _, trainable_summary = _lazy_torch_path_imports()
        if self._backbone is None:
            raise RuntimeError("evaluator 尚未加载，不能建 torch 路径")
        self.backbone = TorchBackbone.from_evaluator(self._backbone)
        ft = self.cfg["finetune"]
        if ft["enabled"]:
            lo = ft["lora"]
            names = inject_lora(self.backbone.model, list(lo["target_modules"]), int(lo["rank"]), float(lo["alpha"]), float(lo["dropout"]))
            freeze_all_but_lora(self.backbone.model, bool(ft["unfreeze_output_head"]))
            lp = Path(str(ft["lora_path"]))
            if lp.exists():
                load_lora_state_dict(self.backbone.model, torch.load(lp, map_location="cpu"))
                self.lora_loaded = True
            else:
                self.lora_loaded = False
                warnings.warn(f"finetune.enabled=True 但 LoRA 权重 {lp} 不存在：底座以零样本权重起步"
                              f"（训练前正常；推理时表示尚未微调）。已挂 {len(names)} 个 LoRA 模块，"
                              f"可训参数占比 {trainable_summary(self.backbone.model)['frac']:.4%}")
        # inject_lora 新建的 LoRALinear（及其内部 nn.Dropout）默认 training=True，与已加载 evaluator（构造时
        # 已 eval()）不一致：不显式摆回 eval 会让 dropout 在“看似零样本/未训练”的推理调用里静默生效，
        # 同一权重连续两次 forward 都不可复现。默认落回推理态，训练脚本按惯例（同 evaluate_model）显式 m.train()。
        self.backbone.eval()

    def _quantiles_torch(self, tgt: np.ndarray, pf_list, po_list, station_id) -> torch.Tensor:
        t3 = self.cfg["timesfm3"]
        pf = np.stack(pf_list).astype(np.float32) if pf_list else None            # [B,C,864]
        po = np.stack(po_list).astype(np.float32) if po_list else None            # [B,C,672]
        kw = dict(horizon=self.pred_len, use_znorm=bool(t3["use_znorm"]),
                  use_symmetric_averaging=bool(t3["use_symmetric_averaging"]), make_positive=bool(t3["make_positive"]))
        if not self.station_routing:
            return self.backbone.predict_quantiles(tgt, pf, po, **kw)
        set_lora_enabled = _lazy_torch_path_imports()[4]
        use_lora = np.array([self.station_routing.get(int(s), "zeroshot") == "lora" for s in np.asarray(station_id).reshape(-1)])
        try:
            if use_lora.all():
                set_lora_enabled(self.backbone.model, True)
                return self.backbone.predict_quantiles(tgt, pf, po, **kw)
            if not use_lora.any():
                set_lora_enabled(self.backbone.model, False)
                return self.backbone.predict_quantiles(tgt, pf, po, **kw)
            set_lora_enabled(self.backbone.model, True);  q_lora = self.backbone.predict_quantiles(tgt, pf, po, **kw)
            set_lora_enabled(self.backbone.model, False); q_zs = self.backbone.predict_quantiles(tgt, pf, po, **kw)
            sel = torch.as_tensor(use_lora, device=q_lora.device)[:, None, None]
            return torch.where(sel, q_lora, q_zs)
        finally:
            set_lora_enabled(self.backbone.model, True)

    # ------------------------------------------------------------------ 取列（同 m5）
    def _col(self, batch_dict: dict, col: str) -> np.ndarray:
        if col not in batch_dict:
            raise KeyError(f"batch_dict 缺列 {col!r}；现有 keys={sorted(batch_dict)[:20]}...")
        v = batch_dict[col]
        arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if not np.isfinite(arr).all():
            if self.nan_policy == "interp":
                arr = interp_nan(arr).astype(np.float32)
            elif self.nan_policy == "zero":
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise ValueError(f"列 {col!r} 含非有限值（nan_policy='raise'）")
        return arr

    # ------------------------------------------------------------------ 协变量组装
    def _covariates(self, batch_dict: dict, ctx) -> tuple[list, list]:
        L, H = self.context_len, self.pred_len
        pf = []
        for hist_c, fut_c in self.past_future_cols:
            if hist_c in batch_dict and fut_c in batch_dict:
                h = self._col(batch_dict, hist_c)[:, -L:]; f = self._col(batch_dict, fut_c)[:, :H]
                pf.append(np.concatenate([h, f], axis=1))
            elif (hist_c, fut_c) not in _WARNED_MISSING_PAIRS:
                warnings.warn(f"past_future 协变量对 {(hist_c, fut_c)} 在 batch_dict 里缺列，"
                              f"本次推理跳过该协变量（只提示一次）")
                _WARNED_MISSING_PAIRS.add((hist_c, fut_c))
        d = self.cfg["features"]["derived"]
        if d.get("solar_elevation"):
            pf.append(ctx.elev_deg.astype(np.float32))
        if d.get("clear_sky_power"):
            pf.append(ctx.p_clear.astype(np.float32))
        nm = self.cfg["features"]["nwp_members"]
        if nm["enabled"]:
            mem = []
            for c in nm["cols"]:
                h = self._col(batch_dict, c)[:, -L:]; f = self._col(batch_dict, c + nm["future_suffix"])[:, :H]
                mem.append(np.concatenate([h, f], axis=1))
            pf.extend(mem)
            if nm["add_spread"]:
                pf.append(np.stack(mem, 0).std(0).astype(np.float32))
        po = [self._col(batch_dict, c)[:, -L:] for c in self.past_only_cols]
        B = ctx.station_id.shape[0]
        pf_list = [np.stack([a[i] for a in pf]) for i in range(B)] if pf else []
        po_list = [np.stack([a[i] for a in po]) for i in range(B)] if po else []
        return pf_list, po_list

    # ------------------------------------------------------------------ 底座
    def _run_backbone(self, tgt: np.ndarray, pf_list, po_list) -> np.ndarray:
        """→ q_raw [B,H,9]。无分位时用点预测复制 9 份并 warn。"""
        t3 = self.cfg["timesfm3"]
        kwargs = dict(horizon=self.pred_len, return_quantiles=bool(t3["return_quantiles"]),
                      make_positive=bool(t3["make_positive"]), use_znorm=bool(t3["use_znorm"]),
                      use_symmetric_averaging=bool(t3["use_symmetric_averaging"]))
        if pf_list:
            kwargs["past_future_covariates"] = pf_list
        if po_list:
            kwargs["past_only_covariates"] = po_list
        outs = list(self._backbone.predict_batch([t[None, :] for t in tgt], **kwargs))
        qs = []
        for o in outs:
            q = getattr(o, "quantiles", None)
            if q is None:
                fc = np.asarray(o.forecast); fc = fc[0] if fc.ndim == 2 else fc
                q = np.repeat(fc[:, None], len(MODEL_LEVELS), axis=1)
            else:
                q = np.asarray(q); q = q[0] if q.ndim == 3 else q
            qs.append(q)
        if any(getattr(o, "quantiles", None) is None for o in outs):
            warnings.warn("底座未返回分位数：p* 退化为点预测")
        q_raw = np.stack(qs).astype(np.float32)
        if q_raw.shape[-1] != len(MODEL_LEVELS):
            raise ValueError(f"底座返回了 {q_raw.shape[-1]} 个分位，期望 {len(MODEL_LEVELS)} 个"
                             f"（MODEL_LEVELS）；q_raw.shape={q_raw.shape}")
        return q_raw

    # ------------------------------------------------------------------ forward
    def forward(self, batch_dict: dict, labels=None) -> TransformersAdaptedModelOutput:
        if self._backbone is None:
            raise RuntimeError("backbone 未加载：eager_init=False 时需先 _ensure_backbone(device)")
        dev = torch.device(self._backbone_device or "cpu")
        L, H = self.context_len, self.pred_len
        tgt = self._col(batch_dict, self.target_col)[:, -L:]
        ctx = build_row_context(batch_dict, self.cfg, self.station_meta)
        pf_list, po_list = self._covariates(batch_dict, ctx)
        if self.torch_path:
            q_raw_t = self._quantiles_torch(tgt, pf_list, po_list, ctx.station_id)   # [B,H,9] tensor；训练时带梯度
            if q_raw_t.dtype != torch.float32:
                q_raw_t = q_raw_t.float()          # bf16 autocast → fp32：aux["quantiles"]/loss 口径统一，不随 autocast 变
            q_raw = q_raw_t.detach().float().cpu().numpy()   # .float() 兜底：bf16 张量 .numpy() 会抛 TypeError
            dev = q_raw_t.device
        else:
            q_raw = self._run_backbone(tgt, pf_list, po_list)                          # [B,H,9]
            q_raw_t = torch.as_tensor(q_raw, dtype=torch.float32, device=dev)
        B = q_raw.shape[0]
        # 校准
        if self.calibrator is not None:
            lead = np.tile(np.arange(H), B)
            qc = self.calibrator.apply(q_raw.reshape(B * H, -1), np.repeat(ctx.station_id, H),
                                       ctx.elev_deg[:, L:].reshape(-1), np.repeat(ctx.var_bin, H),
                                       lead, ctx.p_clear[:, L:].reshape(-1))
            q_cal = qc.reshape(B, H, -1); levels = tuple(self.calibrator.cfg.levels)
        else:
            q_cal = q_raw.astype(float); levels = tuple(MODEL_LEVELS)
        q_cal = enforce_monotone(q_cal)
        ph = self.cfg["decision"]["physics"]
        # night_physics 只喂给 physics_clip（提交曲线的展示/钳位用），受 decision.physics.night_zero 开关控制；
        # night_loss 恒等于真实夜间（ctx.night_future），不受该开关影响，专供训练损失掩码使用——
        # 两者不得共用同一个变量，否则 night_zero=False 会静默把夜间步纳入损失。
        night_physics = ctx.night_future if ph["night_zero"] else np.zeros_like(ctx.night_future)
        night_loss = ctx.night_future
        q_cal = physics_clip(q_cal, night_physics, ctx.p_clear[:, L:], ctx.cap[:, None],
                             float(ph["clear_sky_cap_ratio"]), bool(ph["capacity_cap"]))
        pt = self.cfg["decision"]["point"]
        if pt.get("tail", "flat") != "flat":
            raise ValueError(f"decision.point.tail: only 'flat' is implemented (got {pt.get('tail')!r})")
        eff_default, eff_by_step = self._effective_point_methods()
        point = select_point(q_cal, levels, ctx.cap, eff_default, eff_by_step,
                             self.shrink_factor, float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
        median = level_value(q_cal, levels, 0.5)
        pstar = p_star(q_cal, levels, ctx.cap[:, None], float(pt["floor_ratio"]), float(pt["tau_grid_step"]))
        logits = torch.as_tensor(point, dtype=torch.float32, device=dev)
        aux = {
            "quantiles": q_raw_t,
            "quantiles_cal": torch.as_tensor(q_cal, dtype=torch.float32, device=dev),
            "levels_cal": list(levels),
            "p_star": torch.as_tensor(pstar, dtype=torch.float32, device=dev),
            "median": torch.as_tensor(median, dtype=torch.float32, device=dev),
            "night": night_physics, "night_loss": night_loss,
            "elev_deg": ctx.elev_deg[:, L:], "p_clear": ctx.p_clear[:, L:],
            "var_bin": ctx.var_bin, "var_signal": ctx.var_signal, "station_id": ctx.station_id,
            "cap": ctx.cap, "degraded": ctx.degraded,
        }
        loss = None
        if labels is not None:
            loss = self._loss_from_labels(aux["quantiles"], labels, night_loss)
        return TransformersAdaptedModelOutput(loss=loss, logits=logits, predictions=logits, aux=aux)

    def _loss_from_labels(self, q_raw_t: torch.Tensor, labels, night: np.ndarray) -> torch.Tensor:
        y = labels[self.label_col] if isinstance(labels, dict) else labels
        y = torch.as_tensor(np.asarray(y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y, np.float32),
                            device=q_raw_t.device)[:, : self.pred_len]
        m = loss_mask(night, None, missing_mask(y.cpu().numpy()))
        y = torch.nan_to_num(y)
        return pinball_loss(q_raw_t, y, MODEL_LEVELS, torch.as_tensor(m, device=q_raw_t.device))

    def compute_loss(self, outputs, labels, num_items_in_batch=None) -> torch.Tensor:
        return self._loss_from_labels(outputs.aux["quantiles"], labels, outputs.aux["night_loss"])
