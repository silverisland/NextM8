"""Smoothing must remain as-of, mask preserving, and outside evaluation truth."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from forecast_backbones.m4_rum.config import load_config
from forecast_backbones.m4_rum.evaluate import evaluate, load_member
from forecast_backbones.m4_rum.smoothing import masked_moving_average
from forecast_backbones.m4_rum.train import train_m4_rum
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from pipelines.forecasting.config import deep_merge

ROOT = Path(__file__).resolve().parents[1] / "config"


def smooth_config(
    cfg, history=3, target=3, alignment="centered", architecture="query_decoder"
):
    return deep_merge(
        cfg,
        {
            "model": {"architecture": architecture, "decoder_layers": 1},
            "preprocess": {
                "pv_smoothing": {
                    "history_window": history,
                    "target_window": target,
                    "alignment": alignment,
                }
            },
            "train": {"loss": "pinball"},
            "decision": {"point": {"default": "p_star"}},
        },
    )


def numpy_smooth(values, window, alignment):
    assert alignment == "centered"
    out = np.where(np.isfinite(values), values, 0)
    k = window // 2
    for b in range(len(values)):
        for t in range(k, values.shape[1] - k):
            segment = values[b, t - k : t + k + 1]
            if np.isfinite(segment).all():
                out[b, t] = segment.sum() / window
    return out


@pytest.mark.parametrize("alignment", ["centered"])
def test_rolling_matches_independent_oracle_and_preserves_missing(alignment):
    x = torch.tensor(
        [
            [1.0, float("nan"), 5.0, 7.0, 9.0],
            [float("nan")] * 5,
            [100.0, 200.0, 300.0, float("inf"), 500.0],
        ]
    )
    original = x.clone()
    valid = torch.isfinite(x)
    actual = masked_moving_average(x, valid, 3, alignment)
    expected = torch.from_numpy(numpy_smooth(x.numpy(), 3, alignment))
    torch.testing.assert_close(actual, expected)
    assert (actual[~valid] == 0).all() and torch.isfinite(actual).all()
    torch.testing.assert_close(x, original, equal_nan=True)


@pytest.mark.parametrize("window", [3, 5, 9])
def test_centered_mean_keeps_exact_boundary_values_and_rows_separate(window):
    x = torch.tensor(
        [
            [1.0, 4.0, 20.0, 3.0, 5.0, 100.0, 7.0, 8.0, 2.0, 10.0, 11.0],
            [3.0, 2.0, 1.0, 5.0, 6.0, 8.0, 90.0, 2.0, 4.0, 1.0, 0.0],
        ]
    )
    valid = torch.ones_like(x, dtype=torch.bool)
    before = masked_moving_average(x, valid, window)
    k = window // 2
    torch.testing.assert_close(before[:, :k], x[:, :k], atol=0, rtol=0)
    torch.testing.assert_close(before[:, -k:], x[:, -k:], atol=0, rtol=0)
    for t in range(k, x.shape[1] - k):
        torch.testing.assert_close(before[:, t], x[:, t - k : t + k + 1].mean(1))
    x[1] *= 10000
    after = masked_moving_average(x, valid, window)
    torch.testing.assert_close(before[0], after[0], atol=0, rtol=0)


def test_missing_neighbor_keeps_current_value_not_partial_mean():
    x = torch.tensor([[1.0, 2.0, 100.0, float("nan"), 5.0, 6.0, 7.0]])
    result = masked_moving_average(x, torch.isfinite(x), 5)
    torch.testing.assert_close(
        result, torch.tensor([[1.0, 2.0, 100.0, 0.0, 5.0, 6.0, 7.0]])
    )


@pytest.mark.parametrize("alignment", ["centered"])
def test_history_only_changes_pv_and_never_reads_labels(cfg, batch, alignment):
    model = TransformersAdaptedM4RumModel(smooth_config(cfg, alignment=alignment))
    baseline = TransformersAdaptedM4RumModel(smooth_config(cfg, history=1, target=1))
    raw = copy.deepcopy(batch["batch_dict"])
    raw["observe_power"][:, :3] = float("nan")
    saved = {k: v.clone() for k, v in raw.items()}
    f = model.fe.build(raw, "cpu", training=True)
    unchanged = baseline.fe.build(raw, "cpu")
    torch.testing.assert_close(f.history[..., 1:], unchanged.history[..., 1:])
    assert torch.equal(f.history_mask, unchanged.history_mask)
    expected = numpy_smooth((raw["observe_power"] / 100).numpy(), 3, alignment)
    torch.testing.assert_close(f.history[..., 0], torch.from_numpy(expected))
    for key in ("future", "hint", "history_time", "future_time", "capacity"):
        torch.testing.assert_close(
            getattr(f, key), getattr(unchanged, key), atol=0, rtol=0
        )
    raw["observe_power_future"] = torch.full((2, 192), 1e9)
    inference = model.fe.build(raw, "cpu", training=False)
    torch.testing.assert_close(f.history, inference.history, atol=0, rtol=0)
    for key, value in saved.items():
        if key != "observe_power_future":
            torch.testing.assert_close(raw[key], value, equal_nan=True)


@pytest.mark.parametrize("architecture", ["legacy", "query_decoder"])
@pytest.mark.parametrize("alignment", ["centered"])
def test_target_smoothing_only_changes_training_loss(
    cfg, batch, architecture, alignment
):
    model = TransformersAdaptedM4RumModel(
        smooth_config(cfg, alignment=alignment, architecture=architecture)
    )
    labels = torch.zeros(2, 192)
    labels[:, 10] = 90
    labels[0, 20] = float("nan")
    saved = labels.clone()
    valid = torch.isfinite(labels)
    for training in (True, False):
        model.train(training)
        output = model(batch["batch_dict"], labels)
        truth = labels.numpy() / 100
        target = numpy_smooth(truth, 3, alignment) if training else np.nan_to_num(truth)
        q = output.aux["quantiles"] / 100
        error = torch.from_numpy(target)[..., None] - q
        lv = torch.tensor(model.levels)
        per_step = torch.maximum(lv * error, (lv - 1) * error).mean(-1)
        expected = ((per_step * valid).sum(1) / valid.sum(1)).mean()
        torch.testing.assert_close(output.loss, expected)
        if training:
            output.loss.backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
    torch.testing.assert_close(labels, saved, equal_nan=True)
    first = model(batch["batch_dict"], labels).logits
    second = model(batch["batch_dict"], torch.full_like(labels, 1e6)).logits
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_disabled_smoothing_keeps_old_weights_and_output(cfg, batch):
    original = TransformersAdaptedM4RumModel(cfg)
    disabled = TransformersAdaptedM4RumModel(
        deep_merge(
            cfg,
            {"preprocess": {"pv_smoothing": {"history_window": 1, "target_window": 1}}},
        )
    )
    disabled.load_state_dict(original.state_dict(), strict=True)
    torch.testing.assert_close(
        original(**batch).logits, disabled(**batch).logits, atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "override",
    [
        {"history_window": 0},
        {"target_window": 2},
        {"history_window": True},
        {"target_window": 193},
        {"history_window": 673},
        {"alignment": "unknown"},
        {"alignment": "trailing"},
    ],
)
def test_invalid_smoothing_configs(override):
    with pytest.raises(ValueError, match="pv_smoothing"):
        load_config({"preprocess": {"pv_smoothing": override}})


def test_smoothing_recipes_only_change_preprocessing():
    baseline = load_config(ROOT / "d1_sweep/q6_ema099.yaml")
    baseline.pop("run")
    baseline["train"]["weights_path"] = None
    files = sorted((ROOT / "smoothing_sweep").glob("*.yaml"))
    assert len(files) == 8
    for path in files:
        c = load_config(path)
        c.pop("run")
        c["train"]["weights_path"] = None
        if path.stem.startswith("s00"):
            assert c == baseline
        c["preprocess"] = baseline["preprocess"]
        assert c == baseline


def test_smoothing_training_reload_and_full_eval_preserve_raw_truth(
    cfg, splits, tmp_path
):
    out = tmp_path / "smooth_run"
    config = deep_merge(
        smooth_config(cfg, history=5, target=5, alignment="centered"),
        {
            "train": {"ema_decay": 0.99},
            "model": {
                "extensions": {
                    k: True
                    for k in (
                        "residual_head",
                        "weather_day_reference",
                        "local_weather",
                        "hint_gate",
                    )
                }
            },
        },
    )
    train_m4_rum(config, splits_dir=splits, out_dir=out)
    model = load_member(out)
    assert model.cfg["preprocess"]["pv_smoothing"]["history_window"] == 5
    assert (
        load_member(out / "checkpoints/step_00001").cfg["preprocess"]
        == model.cfg["preprocess"]
    )
    source = pd.read_parquet(splits / "valid.parquet")
    result = evaluate(
        [out],
        [str(splits / "valid.parquet")],
        tmp_path / "eval",
        full=True,
        progress=False,
    )
    for _, row in result["result"].iterrows():
        truth = (
            source[
                (source.station_id == row.station_id)
                & (source.timestamp_win == row.timestamp_win)
            ]
            .iloc[0]
            .observe_power_future
        )
        np.testing.assert_array_equal(row.observe_power_future, truth)
    assert (tmp_path / "eval/reports/m8_comparable_summary.csv").is_file()


def test_smoothing_is_fp32_under_autocast():
    x = torch.rand(2, 192)
    valid = torch.ones_like(x, dtype=torch.bool)
    expected = masked_moving_average(x, valid, 9)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = masked_moving_average(x, valid, 9)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
