"""Point-forecast evaluation with M8 row selection and metric implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from pipelines.forecasting.metrics import (
    nrmse_pct,
    score_nanwang_result,
    seen_unseen_scores,
)
from pipelines.forecasting.rows import issue_grid_row_index, scored_row_index
from pipelines.forecasting.training import build_training_args

from .config import load_config
from .decision import quantile_cols
from .features import as_numpy


@torch.no_grad()
def evaluate_model(model, test_dataset, HF_collate_fn, config):
    cfg = model.cfg
    ds = cfg["dataset"]
    loader = DataLoader(
        test_dataset,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        collate_fn=HF_collate_fn,
    )
    rows = []
    was_training = model.training
    model.eval()
    try:
        for batch in tqdm(
            loader,
            desc="[m4_rum] inference",
            disable=not cfg["train"]["schedule"]["progress_bar"],
        ):
            bd, labels = (
                (batch["batch_dict"], batch.get("labels"))
                if "batch_dict" in batch
                else (batch, None)
            )
            bd = {k: v for k, v in bd.items() if k != ds["label_col"]}
            out = model(bd)
            prediction, cap = as_numpy(out.logits), as_numpy(out.aux["cap"])
            truth = as_numpy(labels[ds["label_col"]]) if labels is not None else None
            if truth is not None and (
                truth.ndim != 2 or truth.shape[1] < ds["pred_len"]
            ):
                raise ValueError("Evaluation labels must cover 192 steps")
            timestamps = as_numpy(bd[ds["ts_col"]]).reshape(-1)
            diagnostics = {}
            if "quantiles" in out.aux:
                q = as_numpy(out.aux["quantiles"])
                diagnostics = {
                    name: as_numpy(out.aux[name]) for name in ("median", "p_star")
                }
                diagnostics.update(
                    {
                        name: q[..., j]
                        for j, name in enumerate(quantile_cols(model.levels))
                    }
                )
            for i, sid in enumerate(out.aux["station_id"]):
                row = {
                    "timestamp_win": pd.Timestamp(int(timestamps[i]), unit="s"),
                    "station": model.station_meta.name(int(sid)),
                    "station_id": int(sid),
                    "cap": float(cap[i]),
                    "observe_power_predict": prediction[i].astype(np.float32),
                    "observe_power_future": truth[i, : ds["pred_len"]].astype(
                        np.float32
                    )
                    if truth is not None
                    else np.full(ds["pred_len"], np.nan, np.float32),
                }
                for key in (
                    "history_unavailable",
                    "hint_unavailable",
                    "future_weather_unavailable",
                ):
                    row[key] = bool(out.aux[key][i])
                row.update(
                    {
                        key: value[i].astype(np.float32)
                        for key, value in diagnostics.items()
                    }
                )
                rows.append(row)
    finally:
        model.train(was_training)
    return pd.DataFrame(rows)


def inference_rows(ds, cfg, full=False):
    if full:
        return np.arange(len(ds))
    d = cfg["dataset"]
    every = cfg["train"]["early_stop"]["nrmse_issue_every_min"]
    return np.union1d(
        scored_row_index(ds, d["station_id_col"], d["ts_col"]),
        issue_grid_row_index(ds, d["ts_col"], every),
    )


def score_result(model, result):
    """Score Rum point predictions with the shared metric implementation."""
    cfg = model.cfg
    score, table, _ = score_nanwang_result(model, result, cfg)
    seen, unseen, count = seen_unseen_scores(table)
    every = cfg["train"]["early_stop"]["nrmse_issue_every_min"]
    if result.empty:
        hourly = result
    else:
        ts = result["timestamp_win"]
        hourly = result[(ts.dt.hour * 60 + ts.dt.minute) % every == 0]
    return {
        "nanwang_short_D+1": score,
        "nrmse_pct": nrmse_pct(hourly),
        "nanwang_seen": seen,
        "nanwang_unseen": unseen,
        "n_unseen_stations": count,
    }, table


def validate(model, dataset, collate, full=False):
    indices = inference_rows(dataset, model.cfg, full)
    if not len(indices):
        raise ValueError("No scored issue rows in evaluation dataset")
    result = evaluate_model(
        model, Subset(dataset, indices.tolist()), collate, model.cfg
    )
    scores, table = score_result(model, result)
    return scores, table, result


def M4_RUM_build_training_args(model_config, result_dir, max_steps=None):
    return build_training_args(load_config(model_config), result_dir, max_steps)
