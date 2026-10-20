# -*- coding: utf-8 -*-
# forecast_backbones/m7/transformer_trainer.py
"""M7（Timestral-209M 零样本）的框架契约入口：evaluate_model(model, test_dataset, HF_collate_fn, config) -> DataFrame。
2026-10-20 起自带一份（自 m5/transformer_trainer.py 的 legacy 路径复制），不再引用 M5；结果表列与 M5 legacy 完全一致
（timestamp_win / station_id / observe_power_predict / observe_power_future / q_0.1..q_0.9）。
registry.build_evaluator("m7") 只要求本文件与 evaluate_model 存在。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

MODEL_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QUANTILE_COLS = [f"q_{float(l):g}" for l in MODEL_LEVELS]

__all__ = ["MODEL_LEVELS", "QUANTILE_COLS", "evaluate_model"]


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def evaluate_model(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame:
    """逐 batch 零样本推理。列名从 config.dataset 取（ts_col / label_col / station_id_col），缺省与 M5 一致；
    绝不把 label_col 喂给模型。站标识列若被 collate 丢掉（字符串站名）则结果表里没有 station_id 列。"""
    cfg = config or {}
    ds_cfg = cfg.get("dataset") or {}
    ts_col = ds_cfg.get("ts_col", "timestamp_win")
    label_col = ds_cfg.get("label_col", "observe_power_future")
    sid_col = ds_cfg.get("station_id_col", "station_id")
    bs = int((cfg.get("eval") or {}).get("batch_size", 64))
    H = int(model.pred_len)
    loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=HF_collate_fn)
    rows = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict) and "batch_dict" in batch:
                    bd, labels = batch["batch_dict"], batch.get("labels")
                else:
                    bd, labels = batch, None
                bd = {k: v for k, v in bd.items() if k != label_col}
                truth = _np(labels[label_col]) if labels is not None and label_col in labels else None
                out = model(bd)
                logits = _np(out.logits)
                q = _np(out.aux["quantiles"]) if out.aux and out.aux.get("quantiles") is not None else None
                ts = _np(bd[ts_col]).astype(np.int64).reshape(-1)
                sid = _np(bd[sid_col]).reshape(-1) if sid_col in bd else None
                for i in range(logits.shape[0]):
                    rec = {"timestamp_win": pd.Timestamp(int(ts[i]), unit="s")}
                    if sid is not None:
                        rec["station_id"] = int(sid[i])
                    rec["observe_power_predict"] = logits[i].astype(np.float32)
                    rec["observe_power_future"] = (truth[i][:H].astype(np.float32) if truth is not None
                                                  else np.full(H, np.nan, np.float32))
                    if q is not None:
                        for j, c in enumerate(QUANTILE_COLS):
                            rec[c] = q[i, :, j].astype(np.float32)
                    rows.append(rec)
    finally:
        model.train(was_training)
    return pd.DataFrame(rows)
