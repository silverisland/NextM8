"""Evaluate saved Rum runs with M8 D+1, 192-step NRMSE and optional full reports."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipelines.forecasting.config import deep_merge, raw_override
from pipelines.forecasting.data import dataset_with_station_id, expand_parquets
from pipelines.transformer_dataset import UnifiedDataset

from .decision import decide, quantile_cols
from .transformer_models import TransformersAdaptedM4RumModel
from .transformer_trainer import score_result, validate


def load_member(run, device="cpu", point=None):
    run = Path(run)
    config, weights = run / "model_config.yaml", run / "model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError(
            f"{run} must contain model_config.yaml and model.safetensors"
        )
    cfg = deep_merge(raw_override(config), {"train": {"weights_path": str(weights)}})
    if point is not None:
        if cfg.get("train", {}).get("loss", "mse") != "pinball":
            raise ValueError("--point requires a checkpoint trained with pinball loss")
        cfg = deep_merge(cfg, {"decision": {"point": {"default": point}}})
    model = TransformersAdaptedM4RumModel(cfg).to(device)
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
    point=None,
):
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
    quantile_sum = None
    for run in runs:
        model = load_member(run, device, point)
        if first is not None:
            if first.core.is_quantile != model.core.is_quantile:
                raise ValueError(
                    "Cannot ensemble point and pinball checkpoints together"
                )
            if model.core.is_quantile and (
                first.levels != model.levels
                or first.cfg["decision"] != model.cfg["decision"]
            ):
                raise ValueError(
                    "Pinball ensemble members must share quantiles "
                    "and decision settings"
                )
        model.cfg["train"]["schedule"]["progress_bar"] = progress
        if batch_size is not None:
            model.cfg["eval"]["batch_size"] = batch_size
        ds = dataset_with_station_id(
            paths, model.cfg, model.station_meta, out / ".data"
        )
        if len(ds) == 0:
            raise ValueError("Empty test dataset")
        scores, _, frame = validate(model, ds, collate, full)
        member_rows.append({"run": str(run), **scores})
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
        if model.core.is_quantile:
            q = np.stack(
                [np.stack(frame[name]) for name in quantile_cols(model.levels)], axis=-1
            )
            quantile_sum = q.copy() if quantile_sum is None else quantile_sum + q
        if len(runs) > 1:
            result[f"predict_m{len(predictions) - 1}"] = list(prediction)
    if quantile_sum is not None:
        # Average raw quantiles, then decide; p_star(mean(q)) != mean(p_star(q)).
        q_mean = quantile_sum / len(runs)
        selected = {key: [] for key in ("point", "median", "p_star")}
        size = first.cfg["eval"]["batch_size"]
        for start in range(0, len(result), size):
            selection = decide(
                torch.from_numpy(q_mean[start : start + size]),
                first.levels,
                torch.tensor(
                    result.cap.to_numpy()[start : start + size], dtype=torch.float32
                ),
                first.cfg["decision"]["point"],
            )
            for key in selected:
                selected[key].append(selection[key].numpy())
        for key, values in selected.items():
            column = "observe_power_predict" if key == "point" else key
            result[column] = list(np.concatenate(values).astype(np.float32))
        for j, name in enumerate(quantile_cols(first.levels)):
            result[name] = list(q_mean[..., j])
        (reports / "decision.json").write_text(
            json.dumps({"quantiles": first.levels, **first.cfg["decision"]}, indent=2),
            encoding="utf-8",
        )
    else:
        result["observe_power_predict"] = list(
            np.mean(predictions, axis=0).astype(np.float32)
        )
    scores, table = score_result(first, result)
    summary = {
        **scores,
        "n_members": len(runs),
        "n_scored_rows": len(result),
        "full": full,
        "point_method": first.cfg["decision"]["point"]["default"]
        if first.core.is_quantile
        else "direct",
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
    pd.DataFrame([summary]).to_csv(reports / "summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    return {"summary": summary, "station_table": table, "result": result}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate M4-Rum using the M8 metric protocol"
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
    parser.add_argument(
        "--point",
        choices=("median", "p_star"),
        help="Override pinball checkpoint point selection",
    )
    a = parser.parse_args(argv)
    evaluate(
        a.runs,
        a.test,
        a.out,
        a.full,
        a.batch_size,
        a.device,
        not a.no_progress,
        a.point,
    )


if __name__ == "__main__":
    main()
