# -*- coding: utf-8 -*-
# forecast_backbones/m8/transformer_trainer.py
"""M8 推理/评估入口。evaluate_model 与 m1–m6 同签名：
    evaluate_model(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame
result_df 与 m6 同 schema（timestamp_win, station, station_id, cap, observe_power_predict, observe_power_future,
p_star, median, night, elev_deg, p_clear, var_bin, var_signal, degraded, q_0.1..q_0.9）。
M8_build_training_args：框架契约（train_script.build_train_args）。"""
from __future__ import annotations


import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import TrainingArguments

try:
    from .config import load_config
    from .decision import quantile_cols
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m8.config import load_config
    from forecast_backbones.m8.decision import quantile_cols


from pipelines.forecasting.training import build_training_args as build_training_args


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def evaluate_model(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame:
    cfg = getattr(model, "cfg", None) or load_config(config)
    bs = int(cfg["eval"].get("batch_size", 256))
    ts_col, label_col, H = cfg["dataset"]["ts_col"], cfg["dataset"]["label_col"], int(cfg["dataset"]["pred_len"])
    levels = tuple(float(q) for q in cfg["model"]["quantiles"])
    qcols = quantile_cols(levels)
    loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=HF_collate_fn)
    show = bool(cfg["train"]["schedule"].get("progress_bar", True))
    desc = str(getattr(model, "_progress_desc", None) or "[m8] 推理")
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch", disable=not show, mininterval=2.0, dynamic_ncols=True):
            if isinstance(batch, dict) and "batch_dict" in batch:
                bd, labels = batch["batch_dict"], batch.get("labels")
            else:
                bd, labels = batch, None
            bd = {k: v for k, v in bd.items() if k != label_col}          # 绝不喂未来真值
            truth = _np(labels[label_col]) if labels is not None and label_col in labels else None
            out = model(bd)
            q = _np(out.aux["quantiles"]); logits = _np(out.logits)
            pstar = _np(out.aux["p_star"]); med = _np(out.aux["median"])
            ts = _np(bd[ts_col]).astype(np.int64).reshape(-1)
            for i in range(logits.shape[0]):
                sid = int(out.aux["station_id"][i])
                rec = {
                    "timestamp_win": pd.Timestamp(int(ts[i]), unit="s"),
                    "station": model.station_meta.name(sid), "station_id": sid, "cap": float(out.aux["cap"][i]),
                    "observe_power_predict": logits[i].astype(np.float32),
                    "observe_power_future": (truth[i][:H].astype(np.float32) if truth is not None else np.full(H, np.nan, np.float32)),
                    "p_star": pstar[i].astype(np.float32), "median": med[i].astype(np.float32),
                    "night": out.aux["night"][i].astype(bool), "elev_deg": out.aux["elev_deg"][i].astype(np.float32),
                    "p_clear": out.aux["p_clear"][i].astype(np.float32),
                    "var_bin": int(out.aux["var_bin"][i]), "var_signal": float(out.aux["var_signal"][i]),
                    "degraded": bool(out.aux["degraded"][i]),
                }
                for j, c in enumerate(qcols):
                    rec[c] = q[i, :, j].astype(np.float32)
                rows.append(rec)
    return pd.DataFrame(rows)


def M8_build_training_args(model_config, result_dir, max_steps=None):
    """框架契约（train_script.build_train_args）。M8 的训练不走 pipelines.train_model（需要按验证集早停、存最优），
    train_script 对 m8 直接调 train.train_m8；这里只把 TrainingArguments 的构造按同一签名暴露出来。"""
    return build_training_args(load_config(model_config), result_dir, max_steps)
