"""Quantile training and point decisions remain entirely inside Rum."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from forecast_backbones.m4_rum.config import load_config
from forecast_backbones.m4_rum.evaluate import evaluate, load_member
from forecast_backbones.m4_rum.train import train_m4_rum
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from pipelines.forecasting.checkpoints import save_weights
from pipelines.forecasting.config import deep_merge, dump_config
from pipelines.transformer_dataset import UnifiedDataset


@pytest.fixture
def quantile_cfg(cfg):
    return deep_merge(
        cfg,
        {
            "train": {"loss": "pinball"},
            "model": {"quantiles": [0.1, 0.5, 0.9]},
            "decision": {"point": {"default": "p_star"}},
        },
    )


def test_pinball_loss_raw_quantiles_capacity_masks_and_gradients(
    quantile_cfg, batch, monkeypatch
):
    model = TransformersAdaptedM4RumModel(quantile_cfg)
    batch["batch_dict"]["station_id"][1] = 3  # Second station has twice the capacity.
    levels = [0.1, 0.5, 0.9]
    values = [1.1, -0.1, 0.4]  # Crossing and negative values must survive in the loss.
    raw = torch.tensor(values).repeat(2, 192, 1).requires_grad_()
    monkeypatch.setattr(model.core, "forward", lambda _: raw)
    target = torch.full((2, 192), float("nan"))
    target[0, :2], target[1, 0] = torch.tensor([25.0, 75.0]), 100.0
    output = model(batch["batch_dict"], target)

    expected_loss, expected_grad = 0.0, torch.zeros_like(raw)
    for row, truths in enumerate(([0.25, 0.75], [0.5])):
        for step, truth in enumerate(truths):
            for k, (tau, value) in enumerate(zip(levels, values)):
                error = truth - value
                weight = 1 / (2 * len(truths) * len(levels))
                expected_loss += max(tau * error, (tau - 1) * error) * weight
                expected_grad[row, step, k] = (-tau if error > 0 else 1 - tau) * weight
    assert output.loss.item() == pytest.approx(expected_loss)
    output.loss.backward()
    torch.testing.assert_close(raw.grad, expected_grad)
    capacity = torch.tensor([100.0, 200.0])[:, None, None]
    torch.testing.assert_close(output.aux["quantiles"], raw * capacity)
    torch.testing.assert_close(
        output.aux["quantiles_cal"], raw.sort(dim=-1).values * capacity
    )
    assert output.aux["levels_cal"] == levels
    assert (output.aux["quantiles"] < 0).any()
    torch.testing.assert_close(output.logits, output.aux["p_star"])
    model.cfg["decision"]["point"]["default"] = "median"
    median_output = model(batch["batch_dict"], target)
    torch.testing.assert_close(median_output.loss, output.loss)
    torch.testing.assert_close(median_output.logits, median_output.aux["median"])
    assert not torch.equal(output.logits, median_output.logits)
    with pytest.raises(ValueError, match="no finite"):
        model(batch["batch_dict"], torch.full_like(target, float("nan")))


def test_every_quantile_head_learns_without_label_leak(quantile_cfg, batch):
    model = TransformersAdaptedM4RumModel(quantile_cfg)
    output = model(**batch)
    assert output.logits.shape == (2, 192)
    assert output.aux["quantiles"].shape == (2, 192, 3)
    changed = copy.deepcopy(batch["batch_dict"])
    changed["observe_power_future"] = torch.full((2, 192), 1e6)
    torch.testing.assert_close(output.logits, model(changed).logits, rtol=0, atol=0)
    output.loss.backward()
    head_gradient = model.core.head[-1].weight.grad
    assert head_gradient.shape[0] == 3
    assert (head_gradient.abs().sum(dim=1) > 0).all()
    for part in (model.core.encoder, model.core.weather):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters()
        )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


@pytest.mark.parametrize("loss", ["mse", "mae"])
def test_existing_point_checkpoints_keep_shapes_and_losses(cfg, batch, tmp_path, loss):
    legacy = TransformersAdaptedM4RumModel(cfg)
    candidate = TransformersAdaptedM4RumModel(
        deep_merge(cfg, {"train": {"loss": loss}, "model": {"quantiles": [0.2, 0.8]}})
    )
    candidate.load_state_dict(legacy.state_dict(), strict=True)
    assert candidate.n_params() == legacy.n_params()
    output = candidate(**batch)
    torch.testing.assert_close(output.logits, legacy(**batch).logits, rtol=0, atol=0)
    error = (output.logits - batch["labels"]["observe_power_future"]) / 100.0
    expected = error.square().mean() if loss == "mse" else error.abs().mean()
    torch.testing.assert_close(output.loss, expected)
    assert "quantiles" not in output.aux
    default = TransformersAdaptedM4RumModel(
        {"dataset": cfg["dataset"], "train": {"weights_path": None, "loss": loss}}
    )
    assert default.n_params() == 310529
    dump_config(candidate.cfg, tmp_path / "model_config.yaml")
    save_weights(candidate, tmp_path / "model.safetensors")
    torch.testing.assert_close(load_member(tmp_path)(**batch).logits, output.logits)
    for point in ("median", "p_star"):
        with pytest.raises(ValueError):
            load_member(tmp_path, point=point)


def test_pinball_training_reload_and_inference_point_override(
    quantile_cfg, splits, tmp_path, batch, collate
):
    out = tmp_path / "pinball_run"
    trained = train_m4_rum(
        quantile_cfg,
        out_dir=out,
        train_parquet=[str(splits / "train.parquet")],
        valid_parquet=[str(splits / "valid.parquet")],
    )
    assert trained["summary"]["best_step"] in (1, 2)
    saved = yaml.safe_load((out / "model_config.yaml").read_text())
    assert saved["train"]["loss"] == "pinball"
    assert saved["model"]["quantiles"] == [0.1, 0.5, 0.9]
    assert saved["decision"]["point"]["default"] == "p_star"
    for step in (1, 2):
        checkpoint = load_member(out / "checkpoints" / f"step_{step:05d}")
        assert checkpoint.weights_loaded
        assert torch.isfinite(checkpoint(**batch).loss)
    original = load_member(out)
    original_output = original(**batch)
    source = pd.read_parquet(splits / "valid.parquet")
    for point in ("median", "p_star"):
        restored = load_member(out, point=point)
        assert restored.cfg["train"]["loss"] == "pinball"
        restored_output = restored(**batch)
        torch.testing.assert_close(restored_output.loss, original_output.loss)
        torch.testing.assert_close(restored_output.logits, original_output.aux[point])
        report = evaluate(
            [out],
            [str(splits / "valid.parquet")],
            tmp_path / point,
            progress=False,
            point=point,
            full=point == "p_star",
        )
        assert np.isfinite(report["summary"]["nanwang_short_D+1"])
        if point == "p_star":
            assert (tmp_path / point / "reports" / "station_metrics.csv").is_file()
        row = report["result"].iloc[0]
        selected = source[
            (source.timestamp_win == row.timestamp_win)
            & (source.station_id == row.station_id)
        ]
        dataset = UnifiedDataset(
            selected,
            timestamp_cols=["timestamp_win"],
            entity_cols=[],
            label_cols=["observe_power_future"],
            for_hf_trainer=True,
        )
        prediction = restored(**collate([dataset[0]])).logits.detach().numpy()[0]
        np.testing.assert_allclose(
            row.observe_power_predict, prediction, rtol=1e-5, atol=1e-5
        )
    # Inference overrides never rewrite the persisted training configuration.
    assert yaml.safe_load((out / "model_config.yaml").read_text()) == saved


def _constant_member(config, path, values):
    model = TransformersAdaptedM4RumModel(config)
    with torch.no_grad():
        model.core.head[-1].weight.zero_()
        model.core.head[-1].bias.copy_(torch.tensor(values))
    dump_config(model.cfg, path / "model_config.yaml")
    save_weights(model, path / "model.safetensors")
    return model


def test_close_quantiles_keep_distinct_report_columns(
    quantile_cfg, splits, tmp_path, batch
):
    levels = [0.10000001, 0.10000002, 0.5]
    config = deep_merge(quantile_cfg, {"model": {"quantiles": levels}})
    run = tmp_path / "close_levels"
    model = _constant_member(config, run, [0.2, 0.5, 0.9])
    report = evaluate(
        [run], [str(splits / "valid.parquet")], tmp_path / "report", progress=False
    )
    for level, power in zip(levels, [20, 50, 90]):
        np.testing.assert_allclose(np.stack(report["result"][f"q_{level}"]), power)
    prediction = np.stack(report["result"].observe_power_predict)
    expected = model(**batch).logits.detach().numpy()[0]
    np.testing.assert_allclose(
        prediction, np.broadcast_to(expected, prediction.shape), rtol=1e-6
    )


def test_quantile_ensemble_combines_distributions_before_pstar(
    quantile_cfg, splits, tmp_path, batch
):
    paths = [tmp_path / "member_a", tmp_path / "member_b"]
    values = torch.tensor([[-0.2, 0.3, 0.9], [0.8, 1.2, 2.0]])
    members = [
        _constant_member(quantile_cfg, path, value)
        for path, value in zip(paths, values.tolist())
    ]
    average = _constant_member(
        quantile_cfg, tmp_path / "average", values.mean(0).tolist()
    )
    expected = average(**batch).logits.detach().numpy()[0]
    mean_points = torch.stack([m(**batch).logits for m in members]).mean(0)
    assert not np.allclose(expected, mean_points.detach().numpy()[0])
    report = evaluate(
        paths,
        [str(splits / "valid.parquet")],
        tmp_path / "ensemble",
        progress=False,
        point="p_star",
    )
    actual = np.stack(report["result"].observe_power_predict)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape))
    for k, level in enumerate([0.1, 0.5, 0.9]):
        np.testing.assert_allclose(
            np.stack(report["result"][f"q_{level}"]),
            values.mean(0)[k].item() * 100,
            rtol=1e-6,
        )
    # Distribution averaging requires matching output heads, levels and decisions.
    for kind, override, biases in (
        (
            "point",
            {"train": {"loss": "mse"}, "decision": {"point": {"default": "median"}}},
            [0.4],
        ),
        ("levels", {"model": {"quantiles": [0.2, 0.5, 0.8]}}, [0.1, 0.5, 0.9]),
        ("floor", {"decision": {"point": {"floor_ratio": 0.3}}}, [0.1, 0.5, 0.9]),
    ):
        path = tmp_path / kind
        _constant_member(deep_merge(quantile_cfg, override), path, biases)
        with pytest.raises(ValueError):
            evaluate(
                [paths[0], path],
                [str(splits / "valid.parquet")],
                tmp_path / f"invalid_{kind}",
                progress=False,
            )


@pytest.mark.parametrize(
    "override",
    [
        {"train": {"loss": "unknown"}},
        {"model": {"quantiles": []}},
        {"model": {"quantiles": [0.5, 0.1, 0.9]}},
        {"model": {"quantiles": [0.1, 0.1, 0.9]}},
        {"model": {"quantiles": [0.0, 0.5, 0.9]}},
        {"model": {"quantiles": [0.1, 0.5, 1.0]}},
        {"model": {"quantiles": [0.1, float("nan"), 0.9]}},
        {"model": {"quantiles": [0.499999999, 0.500000001]}},
        {"model": {"quantiles": [0.1, 1 - 1e-10]}},
        {"decision": {"point": {"default": "unknown"}}},
        {"decision": {"point": {"floor_ratio": 0.0}}},
        {"decision": {"point": {"floor_ratio": float("nan")}}},
        {"decision": {"point": {"tau_grid_step": 0.0}}},
        {"decision": {"point": {"tau_grid_step": float("nan")}}},
        {"decision": {"point": {"tail": "unknown"}}},
    ],
)
def test_invalid_pinball_configuration(quantile_cfg, override):
    with pytest.raises(ValueError):
        load_config(deep_merge(quantile_cfg, override))
