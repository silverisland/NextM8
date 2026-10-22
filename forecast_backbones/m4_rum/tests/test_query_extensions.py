"""Architecture ablations: information paths, missing inputs, and EMA checkpoints."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from forecast_backbones.m4_rum.config import load_config
from forecast_backbones.m4_rum.evaluate import evaluate, load_member
from forecast_backbones.m4_rum.query_extensions import (
    LocalWeatherEncoder,
    WeatherDayReference,
)
from forecast_backbones.m4_rum.train import train_m4_rum
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from pipelines.forecasting.config import deep_merge

ROOT = Path(__file__).resolve().parents[1] / "config"
RECIPES = sorted((ROOT / "arch_sweep").glob("*.yaml"))
SWITCHES = ("residual_head", "weather_day_reference", "local_weather", "hint_gate")


def variant(cfg, extensions):
    return deep_merge(
        cfg,
        {
            "model": {
                "architecture": "query_decoder",
                "decoder_layers": 1,
                "extensions": extensions,
            },
            "train": {"loss": "pinball"},
            "decision": {"point": {"default": "p_star"}},
        },
    )


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.stem)
def test_all_recipe_shapes_gradients_and_no_label_leak(cfg, batch, recipe):
    recipe_cfg = load_config(recipe)
    c = variant(cfg, recipe_cfg["model"]["extensions"])
    c["model"]["station_embedding"] = recipe_cfg["model"]["station_embedding"]
    model = TransformersAdaptedM4RumModel(c)
    out = model(**batch)
    assert out.aux["quantiles"].shape == (2, 192, 9)
    assert torch.isfinite(out.loss) and torch.isfinite(out.logits).all()
    out.loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    changed = copy.deepcopy(batch["batch_dict"])
    changed["observe_power_future"] = torch.full((2, 192), 1e6)
    torch.testing.assert_close(out.logits, model(changed).logits, rtol=0, atol=0)


def test_day_reference_weather_matching_fallback_and_phase():
    module = WeatherDayReference(1, 8)
    history = torch.zeros(1, 192, 2)
    history[:, :96, 0], history[:, 96:, 0] = 2, 8
    history[:, 96:, 1] = 10
    mask = torch.ones_like(history, dtype=torch.bool)
    future = torch.zeros(1, 192, 1)
    fm = torch.ones_like(future, dtype=torch.bool)
    low = module(history, mask, future, fm)
    high = module(history, mask, future + 10, fm)
    torch.testing.assert_close(low[..., 0], torch.full((1, 192), 2.0))
    torch.testing.assert_close(high[..., 0], torch.full((1, 192), 8.0))
    fallback = module(history, mask, future, ~fm)
    torch.testing.assert_close(fallback[..., 0], torch.full((1, 192), 5.0))
    assert (fallback[..., 3] == 0).all()
    missing = module(history, ~mask, future, fm)
    assert (missing == 0).all() and torch.isfinite(missing).all()
    # Invalid power must not enter the reference even if its weather matches.
    mask[:, :96, 0] = False
    history[:, :96, 0] = float("nan")
    torch.testing.assert_close(
        module(history, mask, future, fm)[..., 0], torch.full((1, 192), 8.0)
    )
    history = torch.zeros(1, 192, 2)
    history[0, :, 0] = torch.arange(192)
    phase = module(history, torch.ones_like(mask), future, fm)
    assert phase[0, 0, 0] == 48 and phase[0, 95, 0] == 143
    torch.testing.assert_close(phase[:, :96], phase[:, 96:])


def test_local_branch_has_bounded_temporal_receptive_field():
    module = LocalWeatherEncoder(9, 16, [3, 9], 0.0)
    inputs = torch.randn(1, 192, 9, requires_grad=True)
    module(inputs)[0, 50, 0].backward()
    grad = inputs.grad[0].abs().sum(-1)
    assert grad[46:55].sum() > 0
    assert grad[:46].sum() == 0 and grad[55:].sum() == 0
    assert grad[49] > 0 and grad[51] > 0  # Both sides are known forecasts.


def test_hint_gate_is_masked_and_shared_across_quantiles(cfg, batch):
    model = TransformersAdaptedM4RumModel(variant(cfg, {"hint_gate": True}))
    f = model.fe.build(batch["batch_dict"], "cpu")
    gate = model.core.hint_gate
    actual = model.core(f)
    model.core.hint_gate = None
    direct = model.core(f)
    delta = actual - direct
    torch.testing.assert_close(
        delta, delta[..., :1].expand_as(delta), atol=1e-6, rtol=1e-6
    )
    assert (delta >= -1e-6).all() and (delta <= f.hint[..., None] + 1e-6).all()
    f.hint_mask[:] = False
    direct = model.core(f)
    model.core.hint_gate = gate
    torch.testing.assert_close(model.core(f), direct, atol=0, rtol=0)


def test_residual_head_initially_preserves_baseline_and_has_gradients(cfg, batch):
    torch.manual_seed(21)
    original = TransformersAdaptedM4RumModel(variant(cfg, {}))
    torch.manual_seed(21)
    residual = TransformersAdaptedM4RumModel(variant(cfg, {"residual_head": True}))
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, residual.state_dict()[key], atol=0, rtol=0)
    torch.testing.assert_close(
        original(**batch).logits, residual(**batch).logits, atol=0, rtol=0
    )
    residual(**batch).loss.backward()
    assert residual.core.head_skip.weight.grad.abs().sum() > 0


def test_all_extensions_respect_history_hint_dropout_and_missing_weather(cfg, batch):
    config = deep_merge(
        variant(cfg, dict.fromkeys(SWITCHES, True)),
        {"train": {"history_dropout": 1.0, "hint_dropout": 1.0}},
    )
    model = TransformersAdaptedM4RumModel(config).train()
    first = model(**batch)
    changed = copy.deepcopy(batch)
    for name in ("observe_power", "general_ghi", "general_temp", "timesfm3"):
        changed["batch_dict"][name].fill_(float("nan"))
    torch.testing.assert_close(first.logits, model(**changed).logits, atol=0, rtol=0)
    for name in ("general_ghi_predict", "general_temp_predict"):
        changed["batch_dict"][name].fill_(float("nan"))
    output = model(**changed)
    output.loss.backward()
    assert torch.isfinite(output.logits).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_all_extensions_bfloat16_autocast(cfg, batch):
    model = TransformersAdaptedM4RumModel(
        variant(cfg, dict.fromkeys(SWITCHES, True))
    ).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(**batch)
    out.loss.backward()
    assert torch.isfinite(out.logits).all() and torch.isfinite(out.loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


@pytest.mark.parametrize(
    "extensions",
    [
        {"residual_head": "yes"},
        {"weather_day_reference": 1},
        {"local_weather": None},
        {"hint_gate": 1},
        {"day_hidden": 0},
        {"hint_gate_hidden": True},
        {"local_kernels": []},
        {"local_kernels": [4]},
        {"local_kernels": [3, 3]},
        {"local_kernels": [193]},
    ],
)
def test_invalid_extension_config(cfg, extensions):
    with pytest.raises(ValueError):
        load_config(variant(cfg, extensions))


def test_legacy_rejects_enabled_extensions(cfg):
    with pytest.raises(ValueError, match="query_decoder"):
        load_config(deep_merge(cfg, {"model": {"extensions": {"hint_gate": True}}}))


def test_recipe_matrix_and_q6_control():
    assert len(RECIPES) == 19
    baseline = load_config(ROOT / "d1_sweep/q6_ema099.yaml")
    normalize = lambda c: {k: v for k, v in c.items() if k != "run"}
    baseline["train"]["weights_path"] = None
    combinations = set()
    for path in RECIPES:
        c = load_config(path)
        c["train"]["weights_path"] = None
        for key in ("dataset", "features", "train", "decision", "eval"):
            assert c[key] == baseline[key]
        if path.stem.startswith("a00"):
            assert normalize(c) == normalize(baseline)
        if int(path.stem[1:3]) < 16:
            combinations.add(tuple(c["model"]["extensions"][k] for k in SWITCHES))
    assert len(combinations) == 16


def test_all_branches_ema_training_strict_reload_and_full_report(
    cfg, splits, tmp_path, batch
):
    config = deep_merge(
        variant(cfg, dict.fromkeys(SWITCHES, True)),
        {
            "model": {"station_embedding": True, "n_station_slots": 16},
            "train": {"ema_decay": 0.99},
        },
    )
    out = tmp_path / "all_branches"
    trained = train_m4_rum(config, splits_dir=splits, out_dir=out)
    assert trained["summary"]["best_step"] in (1, 2)
    model, again = load_member(out), load_member(out)
    torch.testing.assert_close(
        model(**batch).logits, again(**batch).logits, rtol=0, atol=0
    )
    assert load_member(out / "checkpoints/step_00001").core.hint_gate is not None
    assert model.cfg["train"]["ema_decay"] == 0.99
    assert model.core.station_slots([1, 2], "cpu").tolist() == [2, 0]
    report = evaluate(
        [out],
        [str(splits / "valid.parquet")],
        tmp_path / "eval",
        full=True,
        progress=False,
    )
    assert np.isfinite(report["summary"]["nanwang_short_D+1"])
    comparable = pd.read_csv(tmp_path / "eval/reports/m8_comparable_summary.csv")
    assert "nanwang_short_D+1" in comparable
