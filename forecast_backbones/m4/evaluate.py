"""Evaluate saved M4 runs with M8 D+1, 192-step NRMSE and optional full reports."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipelines.forecasting.config import deep_merge, raw_override
from pipelines.forecasting.data import dataset_with_station_id, expand_parquets
from pipelines.transformer_dataset import UnifiedDataset

from .transformer_models import TransformersAdaptedM4Model
from .transformer_trainer import score_result, validate


def load_member(run, device="cpu"):
    run = Path(run)
    config, weights = run / "model_config.yaml", run / "model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError(
            f"{run} must contain model_config.yaml and model.safetensors"
        )
    cfg = deep_merge(raw_override(config), {"train": {"weights_path": str(weights)}})
    model = TransformersAdaptedM4Model(cfg).to(device)
    if not model.weights_loaded:
        raise RuntimeError(f"Failed to load {weights}")
    return model.eval()


def evaluate(
    runs,
    test_paths,
    out_dir=None,
    full=False,
    batch_size=None,
    device=None,
    progress=True,
):
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not runs:
        raise ValueError("At least one run is required")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_dir) if out_dir else Path(runs[0]) / "eval_test"
    out.mkdir(parents=True, exist_ok=True)
    reports = out / "reports"
    reports.mkdir(exist_ok=True)
    paths = expand_parquets(test_paths)
    if not paths:
        raise ValueError("No test parquet files")
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    result, first, member_rows, predictions = None, None, [], []
    for run in runs:
        model = load_member(run, device)
        model.cfg["train"]["schedule"]["progress_bar"] = progress
        if batch_size is not None:
            model.cfg["eval"]["batch_size"] = batch_size
        ds = dataset_with_station_id(
            paths, model.cfg, model.station_meta, out / ".data"
        )
        if len(ds) == 0:
            raise ValueError("Empty test dataset")
        scores, _, frame = validate(model, ds, collate, full)
        coverage = {
            f"{name}_rate": float(frame[name].mean())
            for name in (
                "history_unavailable",
                "hint_unavailable",
                "future_weather_unavailable",
            )
        }
        member_rows.append({"run": str(run), **scores, **coverage})
        if result is None:
            first, result = model, frame.copy()
        else:
            keys = ["timestamp_win", "station", "station_id", "cap"]
            if not result[keys].equals(frame[keys]) or not np.allclose(
                np.stack(result.observe_power_future),
                np.stack(frame.observe_power_future),
                equal_nan=True,
            ):
                raise ValueError(
                    "Ensemble members disagree on rows, capacities or labels"
                )
        prediction = np.stack(frame.observe_power_predict)
        predictions.append(prediction)
        if len(runs) > 1:
            result[f"predict_m{len(predictions) - 1}"] = list(prediction)
    result["observe_power_predict"] = list(
        np.mean(predictions, axis=0).astype(np.float32)
    )
    scores, table = score_result(first, result)
    summary = {
        **scores,
        "n_members": len(runs),
        "n_scored_rows": len(result),
        "full": full,
        "point_method": "direct",
    }
    result.to_parquet(out / "result_test.parquet", index=False)
    table.to_csv(reports / "station_table.csv", index=False)
    pd.DataFrame(member_rows).to_csv(reports / "members.csv", index=False)
    if full:
        from pipelines.forecasting.reports import (
            summary_helpers,
            write_station_report,
        )

        extract, encoder, _ = summary_helpers()
        rows = []
        known = first.cfg["dataset"]["known_stations"]
        for sid, group in result.groupby("station_id", sort=True):
            station = first.station_meta.name(int(sid))
            # Keep generated filenames inside the report directory.
            if station in (".", "..") or "/" in station or "\\" in station:
                raise ValueError(
                    f"Station name cannot be used as a report filename: {station!r}"
                )
            _, row = write_station_report(
                group,
                station,
                float(group.cap.iloc[0]),
                out / "metrics_ens",
                extract,
                encoder,
            )
            rows.append(
                {**row, "station_id": int(sid), "seen": not known or int(sid) in known}
            )
        station_metrics = pd.DataFrame(rows)
        station_metrics.to_csv(reports / "station_metrics.csv", index=False)
        for key in (
            "ods_short_rmse_acc",
            "ods_ultra_short_rmse_acc",
            "nanwang_short_D+1",
            "nanwang_ultra_short",
        ):
            if key in station_metrics:
                # Keep the station-day D+1 metric above; label station means explicitly.
                summary[f"station_mean_{key}"] = pd.to_numeric(
                    station_metrics[key], errors="coerce"
                ).mean()
        # M8's full-evaluation summary uses station means, whereas M4's
        # nanwang_short_D+1 above is a station-day mean used for checkpointing.
        # Export an explicitly comparable table without changing the old keys.
        comparable = {
            "n_members": len(runs),
            "n_stations": len(station_metrics),
            "aggregation": "station_mean_of_monthly_report",
        }
        for key in (
            "nanwang_short_D+1",
            "nanwang_ultra_short",
            "ods_short_rmse_acc",
            "ods_ultra_short_rmse_acc",
        ):
            comparable[key] = summary.get(f"station_mean_{key}", float("nan"))
        pd.DataFrame([comparable]).to_csv(
            reports / "m8_comparable_summary.csv", index=False
        )
    pd.DataFrame([summary]).to_csv(reports / "summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    return {"summary": summary, "station_table": table, "result": result}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate M4 using the M8 metric protocol"
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--out")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Infer every issue row and export M8 station workbooks",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--no-progress", action="store_true")
    a = parser.parse_args(argv)
    evaluate(
        a.runs,
        a.test,
        a.out,
        a.full,
        a.batch_size,
        a.device,
        not a.no_progress,
    )


if __name__ == "__main__":
    main()
