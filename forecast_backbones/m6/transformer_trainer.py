# -*- coding: utf-8 -*-
# forecast_backbones/m6/transformer_trainer.py
"""M6 推理/评估入口。evaluate_model 与 m1–m4 同签名：
    evaluate_model(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame
result_df 与 pipelines/nanwang_metric.py 的 schema 兼容（timestamp_win, station, observe_power_predict,
observe_power_future），并追加原始分位 q_0.1..q_0.9、p_star、median、night、elev_deg、p_clear、var_bin 等列。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .calibration import quantile_cols
    from .decision import MODEL_LEVELS
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from calibration import quantile_cols
    from decision import MODEL_LEVELS


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def evaluate_model(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame:
    cfg = getattr(model, "cfg", None) or config
    bs = int(cfg["eval"].get("batch_size", 64))
    ts_col = cfg["dataset"]["ts_col"]; label_col = cfg["dataset"]["label_col"]; H = int(cfg["timesfm3"]["pred_len"])
    qcols = quantile_cols(MODEL_LEVELS)
    loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=HF_collate_fn)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict) and "batch_dict" in batch:
                bd, labels = batch["batch_dict"], batch.get("labels")
            else:
                bd, labels = batch, None
            bd = {k: v for k, v in bd.items() if k != label_col}          # 绝不喂未来真值
            truth = None
            if labels is not None and label_col in labels:
                truth = _np(labels[label_col])
            out = model(bd)
            q = _np(out.aux["quantiles"]); logits = _np(out.logits)
            pstar = _np(out.aux["p_star"]); med = _np(out.aux["median"])
            ts = _np(bd[ts_col]).astype(np.int64).reshape(-1)
            B = logits.shape[0]
            for i in range(B):
                sid = int(out.aux["station_id"][i])
                rec = {
                    "timestamp_win": pd.Timestamp(int(ts[i]), unit="s"),
                    "station": model.station_meta.name(sid), "station_id": sid,
                    "cap": float(out.aux["cap"][i]),
                    "observe_power_predict": logits[i].astype(np.float32),
                    "observe_power_future": (truth[i][:H].astype(np.float32) if truth is not None
                                             else np.full(H, np.nan, np.float32)),
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
