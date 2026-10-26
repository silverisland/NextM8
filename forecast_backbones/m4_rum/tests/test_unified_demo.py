"""生产直接调用示例与评估预测路径的一致性。"""

import numpy as np
import pandas as pd
import pytest

from forecast_backbones.m4_rum import demo_unified_inference as demo
from forecast_backbones.m4_rum.infer import predict_run
from forecast_backbones.m4_rum.tests.conftest import make_frame
from forecast_backbones.m4_rum.tests.test_unified_production import make_config
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from pipelines.forecasting.checkpoints import save_weights
from pipelines.forecasting.config import dump_config


@pytest.mark.parametrize("source", ["csv_id", "csv_station", "data_capacity"])
def test_demo_matches_evaluation_and_reuses_model(cfg, tmp_path, source, monkeypatch):
    config = make_config(cfg, "query_decoder")
    frame = make_frame(days=1, every=120).iloc[[17, 5, 6]].copy()
    frame["station"] = frame.station_id.map({1: "A", 2: "B"})
    if source != "csv_id":
        config["dataset"]["station_key"] = "station"
        frame = frame.drop(columns="station_id")
    if source == "data_capacity":
        config["dataset"].update(station_meta=None, capacity_col="installed_capacity")
        # 首值规则：第一行标量，第二行后续容量不同，第三行为常值列表。
        frame["installed_capacity"] = [100.0, [120.0] + [150.0] * 191, [80.0] * 192]
    model = TransformersAdaptedM4RumModel(config)
    run = tmp_path / "saved"
    run.mkdir()
    save_weights(model, run / "model.safetensors")
    dump_config(model.cfg, run / "model_config.yaml")

    unlabelled = frame.drop(columns="observe_power_future")
    expected = predict_run(run, unlabelled, device="cpu", batch_size=2, progress=False)
    loaded = demo.load_member(run, device="cpu")
    original = demo.unified_inference
    calls = []

    def spy(**kwargs):
        assert "observe_power_future" not in kwargs["ds_input"]
        assert kwargs["error_policy"] == "raise"
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(demo, "unified_inference", spy)
    # 标签列即使损坏也不影响生产预测；尾批及输入次序均需保留。
    result = demo.predict_dataframe(
        loaded, unlabelled.assign(observe_power_future="unused"), batch_size=2
    )
    repeated = demo.predict_dataframe(loaded, unlabelled, batch_size=2)
    assert len(calls) == 2
    assert result.station_id.tolist() == expected.station_id.tolist()
    assert result.station.tolist() == ["B", "A", "A"]
    assert result.timestamp_win.tolist() == frame.timestamp_win.tolist()
    for actual in (result, repeated):
        np.testing.assert_allclose(
            np.stack(actual.observe_power_predict),
            np.stack(expected.observe_power_predict),
            atol=1e-4,
            rtol=1e-5,
        )
    if source != "csv_id":
        assert "station_id" not in unlabelled  # 输入表未被就地修改。

    if source == "csv_station":
        input_path, output_path = (
            tmp_path / "input.parquet",
            tmp_path / "output.parquet",
        )
        unlabelled.to_parquet(input_path, index=False)
        demo.main(
            [
                "--run",
                str(run),
                "--input",
                str(input_path),
                "--out",
                str(output_path),
                "--batch-size",
                "2",
            ]
        )
        saved = pd.read_parquet(output_path)
        np.testing.assert_allclose(
            np.stack(saved.observe_power_predict),
            np.stack(expected.observe_power_predict),
        )
        with pytest.raises(FileExistsError):
            demo.main(
                [
                    "--run",
                    str(run),
                    "--input",
                    str(input_path),
                    "--out",
                    str(output_path),
                ]
            )
