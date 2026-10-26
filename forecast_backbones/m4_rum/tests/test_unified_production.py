"""Actual production engine, raw quantiles, strict checkpoint and input compatibility."""

import functools

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from forecast_backbones.m4_rum.decision import quantile_cols
from forecast_backbones.m4_rum.evaluate import load_member
from forecast_backbones.m4_rum.infer import predict_run
from forecast_backbones.m4_rum.tests.conftest import make_frame
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from forecast_backbones.m4_rum.transformer_trainer import evaluate_model
from pipelines import pipeline_wrapper_inference as wrapper
from pipelines.forecasting.checkpoints import save_weights
from pipelines.forecasting.config import deep_merge, dump_config
from pipelines.transformer_dataset import UnifiedDataset


def make_config(cfg, architecture):
    return deep_merge(
        cfg,
        {
            "dataset": {"known_stations": [1]},
            "train": {"loss": "pinball", "weights_path": None},
            "model": {
                "architecture": architecture,
                "station_embedding": architecture == "query_decoder",
                "n_station_slots": 8,
                "decoder_layers": 1,
                "extensions": {
                    key: architecture == "query_decoder"
                    for key in (
                        "residual_head",
                        "weather_day_reference",
                        "local_weather",
                        "hint_gate",
                    )
                },
            },
            "decision": {
                "point": {
                    "default": "p_star",
                    "clip_negative": True,
                    "capacity_cap": True,
                }
            },
        },
    )


@pytest.mark.parametrize("architecture", ["legacy", "query_decoder"])
def test_evaluation_uses_unified_once_and_retains_quantiles_truth_order(
    cfg, architecture, monkeypatch
):
    model = TransformersAdaptedM4RumModel(make_config(cfg, architecture))
    model.cfg["eval"]["batch_size"] = 2
    frame = make_frame(days=1, every=120).iloc[[5, 17, 6]]
    ds = UnifiedDataset(
        frame,
        timestamp_cols=["timestamp_win"],
        label_cols=["observe_power_future"],
        for_hf_trainer=True,
    )
    ds = Subset(ds, [2, 0, 1])
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    batch = collate([ds[i] for i in range(3)])
    with torch.no_grad():
        expected = model(batch["batch_dict"])
    original = wrapper.unified_inference
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        assert kwargs["error_policy"] == "raise"
        return original(**kwargs)

    monkeypatch.setattr(wrapper, "unified_inference", spy)
    forward_calls = []

    def record(_model, args):
        assert "observe_power_future" not in args[0]
        forward_calls.append(1)

    handle = model.register_forward_pre_hook(record)
    model.train()
    result = evaluate_model(model, ds, collate, model.cfg)
    handle.remove()
    assert model.training and len(calls) == 1 and len(forward_calls) == 2
    np.testing.assert_allclose(
        np.stack(result.observe_power_predict),
        expected.logits.numpy(),
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_array_equal(
        np.stack(result.observe_power_future), batch["labels"]["observe_power_future"]
    )
    for j, name in enumerate(quantile_cols(model.levels)):
        np.testing.assert_allclose(
            np.stack(result[name]),
            expected.aux["quantiles"][..., j],
            atol=1e-4,
            rtol=1e-5,
        )
    assert result.station_id.tolist() == [1, 1, 2]


def test_saved_checkpoint_unlabelled_named_stations_aliases_and_relocated_table(
    cfg, tmp_path
):
    model = TransformersAdaptedM4RumModel(make_config(cfg, "query_decoder"))
    run = tmp_path / "saved"
    run.mkdir()
    save_weights(model, run / "model.safetensors")
    old_meta = cfg["dataset"]["station_meta"]
    model.cfg["dataset"]["station_meta"] = "/missing/server/stations.csv"
    dump_config(model.cfg, run / "model_config.yaml")
    frame = (
        make_frame(days=1, every=120).iloc[[5, 17]].drop(columns="observe_power_future")
    )
    ref = predict_run(run, frame, device="cpu", station_meta=old_meta, progress=False)
    online = frame.rename(
        columns={
            "timesfm3": "tsfm_predict",
            "general_ghi": "general_GHI",
            "general_ghi_predict": "general_GHI_predict",
        }
    )
    online["station"] = online.station_id.map({1: "A", 2: "B"})
    online = online.drop(columns="station_id")
    result = predict_run(
        run, online, device="cpu", station_meta=old_meta, progress=False
    )
    assert "observe_power_future" not in result
    np.testing.assert_array_equal(
        np.stack(result.observe_power_predict), np.stack(ref.observe_power_predict)
    )
    assert result.station_id.tolist() == [1, 2]
    # Unknown station metadata must fail rather than producing a scored subset.
    with pytest.raises((ValueError, KeyError)):
        predict_run(
            run,
            frame.assign(station_id=999),
            device="cpu",
            station_meta=old_meta,
            progress=False,
        )
    (run / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        load_member(run, station_meta=old_meta)
