import copy

import numpy as np
import pytest
import torch

from forecast_backbones.m4_rum.config import load_config
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from pipelines.forecasting.config import deep_merge


def test_forward_backward_and_no_label_leak(cfg, batch):
    model = TransformersAdaptedM4RumModel(cfg)
    output = model(**batch)
    assert output.logits.shape == (2, 192)
    assert torch.isfinite(output.loss)
    changed = copy.deepcopy(batch)
    changed["batch_dict"]["observe_power_future"] = torch.full((2, 192), 1e6)
    changed["labels"]["observe_power_future"] = torch.full((2, 192), float("nan"))
    assert torch.equal(output.logits, model(changed["batch_dict"]).logits)
    with pytest.raises(ValueError, match="no finite"):
        model(**changed)
    output.loss.backward()
    for part in (
        model.core.encoder,
        model.core.context_projection,
        model.core.weather,
        model.core.head,
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters()
        )
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in part.parameters()
        )


def test_hint_is_feature_not_residual(cfg, batch):
    model = TransformersAdaptedM4RumModel(cfg)
    features = model.fe.build(batch["batch_dict"], "cpu")
    features.hint.requires_grad_()
    features.future.requires_grad_()
    model.core(features).sum().backward()
    assert torch.all(features.hint.grad.abs().sum(0) > 0)
    assert torch.all(features.future.grad.abs().sum((0, 2)) > 0)
    with torch.no_grad():
        model.core.head[-1].weight.zero_()
        model.core.head[-1].bias.zero_()
    assert torch.count_nonzero(model(**batch).logits) == 0
    assert features.hint.abs().sum() > 0


def test_all_missing_history_hint_weather_is_finite(cfg, batch):
    model = TransformersAdaptedM4RumModel(cfg).train()
    for key in (
        "observe_power",
        "general_ghi",
        "general_temp",
        "general_ghi_predict",
        "general_temp_predict",
        "timesfm3",
    ):
        batch["batch_dict"][key] = torch.full_like(
            batch["batch_dict"][key], float("nan")
        )
    output = model(**batch)
    assert torch.isfinite(output.logits).all()
    assert (
        output.aux["history_unavailable"].all() and output.aux["hint_unavailable"].all()
    )
    assert output.aux["future_weather_unavailable"].all()
    output.loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_branch_dropout_removes_all_history_and_hint_information(cfg, batch):
    cfg = deep_merge(cfg, {"train": {"history_dropout": 1.0, "hint_dropout": 1.0}})
    model = TransformersAdaptedM4RumModel(cfg).train()
    initial = model(**batch).logits
    changed = copy.deepcopy(batch)
    for key in ("observe_power", "general_ghi", "general_temp", "timesfm3"):
        changed["batch_dict"][key] = torch.full_like(
            changed["batch_dict"][key], float("nan")
        )
    torch.testing.assert_close(initial, model(**changed).logits, rtol=0, atol=0)


def test_station_not_network_input_and_capacity_from_table(cfg, batch):
    model = TransformersAdaptedM4RumModel(cfg)
    base = model(**batch).logits
    changed = copy.deepcopy(batch["batch_dict"])
    changed["station_id"] = torch.full((2,), 2)
    changed["cap_power_on"] = torch.full((2,), 9999.0)
    torch.testing.assert_close(base, model(changed).logits, rtol=0, atol=0)
    changed["station_id"] = torch.full((2,), 3)
    for key in ("observe_power", "timesfm3"):
        changed[key] = changed[key] * 2
    torch.testing.assert_close(base * 2, model(changed).logits)
    changed.pop("station_id")
    changed["station"] = ["C", "C"]
    torch.testing.assert_close(base * 2, model(changed).logits)


def test_time_window_padding_aliases_and_short_future(cfg, batch):
    model = TransformersAdaptedM4RumModel(cfg)
    bd = batch["batch_dict"]
    initial = model(bd).logits
    bd["tsfm_predict"] = bd.pop("timesfm3")
    bd["GENERAL_GHI"] = bd.pop("general_ghi")
    torch.testing.assert_close(initial, model(bd).logits, rtol=0, atol=0)
    f = model.fe.build(bd, "cpu")
    # First sample issues at 10:00: future index 55 is midnight; 191 is D+2 10:00.
    torch.testing.assert_close(
        f.future_time[0, 55, :2], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0
    )
    assert np.isclose(f.future_time[0, 191, 0], 0.5, atol=1e-6)
    bd["observe_power"] = bd["observe_power"][:, -10:]
    f = model.fe.build(bd, "cpu")
    assert not f.history_mask[:, :-10, 0].any()
    bd["tsfm_predict"] = bd["tsfm_predict"][:, :96]
    with pytest.raises(ValueError, match="192"):
        model(bd)


def test_hint_missing_and_absent_weather_pair(cfg, batch):
    model = TransformersAdaptedM4RumModel(cfg)
    bd = batch["batch_dict"]
    del bd["timesfm3"]
    with pytest.warns(UserWarning, match="no-hint"):
        output = model(bd)
    assert output.aux["hint_unavailable"].all()
    model.train()
    with pytest.raises(ValueError, match="hint"):
        model(bd)
    with pytest.raises(ValueError, match="timesfm3"):
        model.fe.check_columns(bd.keys())


@pytest.mark.parametrize(
    "override",
    [
        {"features": {"hint_residual": True}},
        {"dataset": {"pred_len": 96}},
        {"model": {"patch_len": 17}},
        {"model": {"n_heads": 3}},
        {"features": {"weather_scales": [0.0, 1.0]}},
        {"features": {"hint_col": "observe_power_future"}},
        {"features": {"col_aliases": {"timesfm3": ["observe_power_future"]}}},
    ],
)
def test_invalid_configuration(override):
    with pytest.raises(ValueError):
        load_config(override)


def test_default_architecture_and_single_row(cfg, batch):
    model = TransformersAdaptedM4RumModel(
        {"dataset": cfg["dataset"], "train": {"weights_path": None}}
    )
    bd = {k: v[:1] for k, v in batch["batch_dict"].items()}
    assert model(bd).logits.shape == (1, 192)
    assert model.n_params() == 310529


def test_disabled_weather_pair_and_timestamp_validation(cfg, batch):
    cfg = deep_merge(
        cfg,
        {
            "features": {
                "cov_present": [
                    ["general_ghi", "general_ghi_predict"],
                ]
            }
        },
    )
    model = TransformersAdaptedM4RumModel(cfg)
    bd = batch["batch_dict"]
    initial = model(bd).logits
    bd["general_temp_predict"] = bd["general_temp_predict"] + 1000
    torch.testing.assert_close(initial, model(bd).logits, rtol=0, atol=0)
    bd["timestamp_win"] = bd["timestamp_win"] * 1000
    with pytest.raises(ValueError, match="seconds"):
        model(bd)
