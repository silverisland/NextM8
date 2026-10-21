import json

import numpy as np
import pandas as pd
import pytest
import yaml

from forecast_backbones.m4_rum.evaluate import evaluate, load_member
from forecast_backbones.m4_rum.train import train_m4_rum
from forecast_backbones.m4_rum.transformer_trainer import score_result
from pipelines.forecasting.config import deep_merge
from pipelines.forecasting.metrics import nrmse_pct, score_nanwang_result


@pytest.mark.parametrize("metric", ["nanwang_short_D+1", "nrmse"])
def test_train_evaluate_checkpoint_and_unseen(cfg, splits, tmp_path, metric):
    cfg = deep_merge(cfg, {"train": {"early_stop": {"metric": metric}}})
    out = tmp_path / "trained"
    result = train_m4_rum(cfg, splits_dir=splits, out_dir=out)
    assert result["summary"]["best_step"] in (1, 2)
    assert result["summary"]["n_unseen_stations"] == 1
    saved = yaml.safe_load((out / "model_config.yaml").read_text())
    assert saved["dataset"]["known_stations"] == [1]
    assert saved["features"]["cov_present"] == cfg.get("features", {}).get(
        "past_future",
        [
            ["general_ghi", "general_ghi_predict"],
            ["general_temp", "general_temp_predict"],
        ],
    )
    for step in (1, 2):
        assert load_member(out / "checkpoints" / f"step_{step:05d}").weights_loaded
    report = evaluate(
        [out],
        [str(splits / "valid.parquet")],
        out_dir=tmp_path / "eval",
        progress=False,
    )
    assert np.isfinite(report["summary"]["nanwang_short_D+1"])
    assert np.isfinite(report["summary"]["nrmse_pct"])
    model = load_member(out)
    expected, _, _ = score_nanwang_result(model, report["result"], model.cfg)
    assert report["summary"]["nanwang_short_D+1"] == expected
    assert report["summary"]["nrmse_pct"] == nrmse_pct(report["result"])
    log = json.loads((out / "train_log.json").read_text())
    assert len(log["history"]) == 2


def test_full_train_named_stations_and_full_report(cfg, splits, tmp_path):
    frame = (
        pd.read_parquet(splits / "train.parquet")
        .drop(columns="station_id")
        .assign(station="A")
    )
    data = tmp_path / "partition" / "date=2025-06-08"
    data.mkdir(parents=True)
    frame.to_parquet(data / "data.parquet", index=False)
    out = tmp_path / "full"
    result = train_m4_rum(cfg, train_parquet=[str(data.parent)], out_dir=out)
    assert result["summary"]["n_train_rows"] == len(frame)
    assert not (out / "result_val.parquet").exists()
    assert np.isnan(result["summary"]["nanwang_short_D+1"])
    ev = tmp_path / "full_eval"
    report = evaluate(
        [out, out], [str(splits / "valid.parquet")], ev, full=True, progress=False
    )
    assert len(report["result"]) == len(pd.read_parquet(splits / "valid.parquet"))
    assert np.array_equal(
        np.stack(report["result"].observe_power_predict),
        np.stack(report["result"].predict_m0),
    )
    assert (ev / "reports" / "station_metrics.csv").is_file()
    for station in ("A", "B"):
        workbook = ev / "metrics_ens" / f"{station}-metrics.xlsx"
        assert (
            workbook.is_file()
            and (ev / "metrics_ens" / f"{station}-summary.json").is_file()
        )
        assert set(pd.ExcelFile(workbook).sheet_names) == {
            "ods_ultra_short",
            "ods_short",
            "ultra_short",
            "short",
            "48hours",
            "nanwang",
        }


def test_registry_and_unified_inference(cfg, batch):
    from forecast_backbones.m4_rum.transformer_models import (
        TransformersAdaptedM4RumModel,
    )
    from pipelines.model_registry import build_model

    model = build_model("m4_rum", cfg)
    assert isinstance(model, TransformersAdaptedM4RumModel)
    from pipelines.pipeline_wrapper_inference import unified_inference

    frame = pd.DataFrame({k: list(v.numpy()) for k, v in batch["batch_dict"].items()})
    # Public wrapper accepts pandas datetime timestamps, like M8 station_report.
    frame["timestamp_win"] = pd.to_datetime(frame.timestamp_win, unit="s")
    result = unified_inference(
        pipeline_backbone=model,
        ds_input=frame,
        entity_cols=["timestamp_win"],
        batch_size=2,
    )
    prediction = np.stack(result[type(model).__name__])
    assert prediction.shape == (2, 192) and np.isfinite(prediction).all()
    np.testing.assert_allclose(
        prediction,
        model(batch["batch_dict"]).logits.detach().numpy(),
        rtol=1e-6,
    )


def test_d1_slice_matches_m8(cfg, batch):
    from forecast_backbones.m4_rum.transformer_models import (
        TransformersAdaptedM4RumModel,
    )

    model = TransformersAdaptedM4RumModel(cfg)
    truth = np.full(192, 50.0, dtype=np.float32)
    prediction = truth.copy()
    prediction[:55], prediction[151:] = 0, 0
    frame = pd.DataFrame(
        [
            {
                "timestamp_win": pd.Timestamp("2025-06-08 10:00"),
                "station": "A",
                "station_id": 1,
                "cap": 100.0,
                "observe_power_predict": prediction,
                "observe_power_future": truth,
            }
        ]
    )
    scores, _ = score_result(model, frame)
    assert scores["nanwang_short_D+1"] == pytest.approx(100.0)
    assert scores["nrmse_pct"] > 0


def test_refit_merges_splits_and_saves_ema(cfg, splits, tmp_path):
    cfg = deep_merge(cfg, {"train": {"ema_decay": 0.9}})
    out = tmp_path / "refit"
    trained = train_m4_rum(cfg, splits_dir=splits, out_dir=out, refit=True)
    saved = yaml.safe_load((out / "model_config.yaml").read_text())
    assert saved["dataset"]["known_stations"] == [1, 2]
    expected = sum(
        len(pd.read_parquet(splits / f"{name}.parquet")) for name in ("train", "valid")
    )
    assert trained["summary"]["n_train_rows"] == expected
    final = load_member(out).state_dict()
    last = load_member(out / "checkpoints" / "step_00002").state_dict()
    for key in final:
        np.testing.assert_array_equal(final[key].numpy(), last[key].numpy())
