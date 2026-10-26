"""真实小步训练后，生产 demo 与评估输出逐点一致。"""

import shutil

import numpy as np
import pandas as pd
import pytest

from forecast_backbones.m4 import demo_unified_inference as demo
from forecast_backbones.m4.train import train_m4
from forecast_backbones.m4.transformer_models import TransformersAdaptedM4Model


def test_trained_demo_station_names_ids_and_relocated_csv(
    cfg, splits, tmp_path, monkeypatch
):
    # 与生产一致：训练时明确将 hint 列设为 tsfm_predict。
    cfg["features"] = {"hint_col": "tsfm_predict"}
    for name in ("train", "valid"):
        path = splits / f"{name}.parquet"
        frame = pd.read_parquet(path).rename(columns={"chronos": "tsfm_predict"})
        frame.to_parquet(path, index=False)
    # 自定义 CSV 表头，迁移时必须保留列名映射。
    csv = tmp_path / "stations_custom.csv"
    pd.read_csv(cfg["dataset"]["station_meta"]).rename(
        columns={"cap_mw": "rated_mw"}
    ).to_csv(csv, index=False)
    cfg["dataset"].update(
        station_meta=str(csv), station_meta_cols={"cap_mw": "rated_mw"}
    )
    run = tmp_path / "trained"
    trained = train_m4(cfg, splits_dir=splits, out_dir=run)
    frame = pd.read_parquet(splits / "valid.parquet").iloc[[34, 10, 11]].copy()
    frame["station"] = frame.station_id.map({1: "A", 2: "B"})
    frame = frame.drop(columns="observe_power_future").rename(
        columns={
            "general_ghi": "general_GHI",
            "general_temp": "general_TEMP",
            "general_ghi_predict": "general_GHI_predict",
            "general_temp_predict": "general_TEMP_predict",
        }
    )
    # 非数值业务附加列、损坏标签、无关容量列不能影响预测。
    frame["business_note"] = "ignored"
    frame["observe_power_future"] = "unused"
    frame["cap_power_on"] = [1.0, [999.0] * 192, [2.0] * 192]
    relocated = tmp_path / "relocated.csv"
    shutil.move(str(csv), relocated)
    model = demo.load_member(run, station_meta=relocated)
    assert model.cfg["dataset"]["station_meta_cols"] == {"cap_mw": "rated_mw"}
    original, calls = demo.unified_inference, []

    def spy(**kwargs):
        assert kwargs["error_policy"] == "raise"
        assert "observe_power_future" not in kwargs["ds_input"]
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(demo, "unified_inference", spy)
    expected = trained["result"].set_index(["timestamp_win", "station"])
    for inputs in (frame, frame.drop(columns="station_id")):
        result = demo.predict_dataframe(model, inputs, batch_size=2)
        assert result.station.tolist() == ["B", "A", "A"]
        assert result.timestamp_win.tolist() == frame.timestamp_win.tolist()
        target = expected.loc[list(zip(frame.timestamp_win, frame.station))]
        np.testing.assert_allclose(
            np.stack(result.observe_power_predict),
            np.stack(target.observe_power_predict),
            rtol=1e-5,
            atol=1e-4,
        )
        assert np.stack(result.observe_power_predict).shape == (3, 192)
    assert len(calls) == 2

    source, dest = tmp_path / "live.parquet", tmp_path / "predictions.parquet"
    frame.drop(
        columns=["station_id", "observe_power_future", "cap_power_on"]
    ).to_parquet(source, index=False)
    args = [
        "--run",
        str(run),
        "--station-meta",
        str(relocated),
        "--input",
        str(source),
        "--out",
        str(dest),
        "--batch-size",
        "2",
    ]
    demo.main(args)
    np.testing.assert_allclose(
        np.stack(pd.read_parquet(dest).observe_power_predict),
        np.stack(result.observe_power_predict),
    )
    with pytest.raises(FileExistsError):
        demo.main(args)
    with pytest.raises(ValueError, match="Unknown/conflicting"):
        demo.predict_dataframe(
            model, frame.drop(columns="station_id").assign(station="missing")
        )
    for size in (0, -1, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            demo.predict_dataframe(model, frame, batch_size=size)


def test_demo_rejects_unloaded_model(cfg):
    with pytest.raises(ValueError, match="load_member"):
        demo.predict_dataframe(TransformersAdaptedM4Model(cfg), pd.DataFrame())
