"""M8's legacy API delegates to public infrastructure without changing its model."""

import json

import numpy as np
import pandas as pd
import pytest
import torch


def test_legacy_exports_use_public_implementations():
    from forecast_backbones.m8 import (
        config,
        data,
        rows,
        solar,
        station_report,
        train,
        transformer_models,
    )
    from pipelines.forecasting import checkpoints, reports, stations, training
    from pipelines.forecasting import config as public_config
    from pipelines.forecasting import data as public_data
    from pipelines.forecasting import rows as public_rows

    assert config.deep_merge is public_config.deep_merge
    assert data._dataset_with_station_id is public_data.dataset_with_station_id
    assert solar.StationMeta is stations.StationMeta
    assert rows.scored_row_index is public_rows.scored_row_index
    assert train.WeightEma is training.WeightEma
    assert train._M8Trainer is training.ForecastTrainer
    assert issubclass(train.EvalCallback, training.EvaluationCallback)
    assert transformer_models.load_weights is checkpoints.load_weights
    assert station_report.write_station_report is reports.write_station_report


def test_m8_training_and_evaluation_after_extraction(tmp_path):
    from forecast_backbones.m8.evaluate import evaluate
    from forecast_backbones.m8.train import train_m8

    torch.set_num_threads(1)
    meta = tmp_path / "stations.csv"
    meta.write_text("station_id,station,lat,lon,alt_m,cap_mw\n1,A,,,,100\n")
    splits = tmp_path / "splits"
    splits.mkdir()
    for name, date in (("train", "2025-06-01"), ("valid", "2025-07-01")):
        pd.DataFrame(
            [
                {
                    "station_id": 1,
                    "timestamp_win": pd.Timestamp(date + " 10:00"),
                    "observe_power": np.full(672, 50, np.float32),
                    "observe_power_future": np.full(192, 50, np.float32),
                    "general_ghi": np.full(672, 500, np.float32),
                    "general_ghi_predict": np.full(192, 500, np.float32),
                    "general_temp": np.full(672, 20, np.float32),
                    "general_temp_predict": np.full(192, 20, np.float32),
                }
            ]
        ).to_parquet(splits / f"{name}.parquet", index=False)
    (splits / "split_manifest.json").write_text(json.dumps({"plan": "test"}))
    cfg = {
        "dataset": {"station_meta": str(meta)},
        "model": {
            "d_model": 16,
            "d_ff": 32,
            "head_hidden": 16,
            "n_heads": 2,
            "blocks": ["conv"],
            "pos_in_patch_dim": 4,
        },
        "train": {
            "schedule": {
                "batch_size": 1,
                "max_steps": 2,
                "eval_every_steps": 1,
                "warmup_steps": 0,
                "progress_bar": False,
            }
        },
        "eval": {"batch_size": 1},
    }
    out = tmp_path / "m8"
    trained = train_m8(cfg, splits_dir=splits, out_dir=out)
    scored = evaluate([out], [str(splits / "valid.parquet")], progress=False)
    score = trained["summary"].iloc[0]["nanwang_short_D+1"]
    assert np.isfinite(score)
    assert scored["summary"]["nanwang_short_D+1"] == pytest.approx(score)
