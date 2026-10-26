"""Unlabelled production inference through the project's unified_inference."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from pipelines.transformer_dataset import UnifiedDataset

from .evaluate import load_member
from .transformer_trainer import evaluate_model


def predict_run(
    run, inputs, *, device=None, station_meta=None, batch_size=32, progress=True
):
    """DataFrame / parquet input -> point forecasts, quantiles and diagnostics; no labels required."""
    model = load_member(
        run,
        device or ("cuda" if torch.cuda.is_available() else "cpu"),
        station_meta=station_meta,
    )
    cfg = model.cfg
    cfg["eval"]["batch_size"] = batch_size
    cfg["train"]["schedule"]["progress_bar"] = progress
    ds_cfg = cfg["dataset"]
    collate = functools.partial(
        UnifiedDataset.collate_fn, for_hf_trainer=True, drop_str_type=False
    )
    with TemporaryDirectory(prefix="m4_rum_infer_") as scratch:
        if model.station_meta.from_data or ds_cfg["station_key"] is not None:
            if isinstance(inputs, pd.DataFrame):
                inputs = model.station_meta.prepare_frame(inputs)
            else:
                from pipelines.forecasting.data import expand_parquets

                paths = expand_parquets(
                    [inputs] if isinstance(inputs, (str, Path)) else inputs
                )
                inputs = model.station_meta.prepare_parquets(paths, cfg, scratch)
        entities = [ds_cfg["station_id_col"], "station"]
        if ds_cfg["station_key"] and ds_cfg["station_key"] not in entities:
            entities.append(ds_cfg["station_key"])
        dataset = UnifiedDataset(
            inputs,
            timestamp_cols=[ds_cfg["ts_col"]],
            entity_cols=entities,
            for_hf_trainer=True,
        )
        result = evaluate_model(model, dataset, collate, cfg)
    return result.drop(columns="observe_power_future")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument(
        "--out", required=True, help="New parquet file; refuses to overwrite"
    )
    p.add_argument("--station-meta", "--station_meta")
    p.add_argument("--device")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-progress", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    # Resolve directories/globs consistently with evaluation, preserving sorted row order.
    from pipelines.forecasting.data import expand_parquets

    result = predict_run(
        a.run,
        expand_parquets(a.input),
        device=a.device,
        station_meta=a.station_meta,
        batch_size=a.batch_size,
        progress=not a.no_progress,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"Saved {len(result)} predictions to {out}")


if __name__ == "__main__":
    main()
