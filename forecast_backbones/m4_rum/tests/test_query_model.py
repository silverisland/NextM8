"""New information paths, legacy compatibility, and real save/evaluate workflow."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from forecast_backbones.m4_rum.config import load_config
from forecast_backbones.m4_rum.decision import decide
from forecast_backbones.m4_rum.evaluate import evaluate, load_member
from forecast_backbones.m4_rum.train import train_m4_rum
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from pipelines.forecasting.config import deep_merge


def query_config(cfg):
    return deep_merge(
        cfg,
        {
            "model": {"architecture": "query_decoder", "decoder_layers": 1},
            "train": {"loss": "pinball"},
            "decision": {
                "point": {
                    "default": "p_star",
                    "clip_negative": True,
                    "capacity_cap": True,
                    "by_step": [{"steps": [0, 16], "method": "p_star_l1"}],
                }
            },
        },
    )


def test_future_steps_interact_only_in_query_decoder(cfg, batch):
    for architecture in ("legacy", "query_decoder"):
        torch.manual_seed(12)
        model = TransformersAdaptedM4RumModel(
            deep_merge(cfg, {"model": {"architecture": architecture}})
        )
        features = model.fe.build(batch["batch_dict"], "cpu")
        features.future.requires_grad_()
        model.core(features)[0, 0].backward()
        other_steps = features.future.grad[0, 1:].abs().sum()
        if architecture == "legacy":
            assert other_steps == 0  # A structural limitation, independent of weights.
        else:
            assert other_steps > 0
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.core.encoder.parameters()
            )


def test_query_raw_loss_gradients_and_no_label_leak(cfg, batch):
    model = TransformersAdaptedM4RumModel(query_config(cfg))
    output = model(**batch)
    assert output.aux["quantiles"].shape == (2, 192, 9)
    changed = copy.deepcopy(batch["batch_dict"])
    changed["observe_power_future"] = torch.full((2, 192), 1e6)
    torch.testing.assert_close(output.logits, model(changed).logits, rtol=0, atol=0)
    raw = output.aux["quantiles"] / 100
    error = batch["labels"]["observe_power_future"][..., None] / 100 - raw
    levels = torch.tensor(model.levels)
    torch.testing.assert_close(
        output.loss, torch.maximum(levels * error, (levels - 1) * error).mean()
    )
    output.loss.backward()
    for part in (model.core.encoder, model.core.decoder, model.core.head):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters()
        )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_daily_profiles_are_aligned_masked_and_never_use_future(cfg, batch):
    model = TransformersAdaptedM4RumModel(query_config(cfg))
    f = model.fe.build(batch["batch_dict"], "cpu")
    history = torch.zeros_like(f.history)
    history[..., 0] = torch.arange(672)[None]
    mask = torch.ones_like(f.history_mask)
    profile = model.core.daily_profile(history, mask)
    assert profile[0, 0, 0] == 576  # S+15min minus 24h
    assert profile[0, 95, 0] == 671  # S+24h minus 24h
    torch.testing.assert_close(profile[:, :96], profile[:, 96:])
    assert profile[0, 0, 2] == torch.arange(0, 672, 96).float().mean()
    mask[:, -96:, 0] = False
    masked = model.core.daily_profile(history, mask)
    assert (masked[..., :2] == 0).all()
    assert masked[0, 0, 2] == torch.arange(0, 576, 96).float().mean()
    assert masked[0, 0, 3] == pytest.approx(6 / 7)


def test_query_missing_inputs_and_dropout_do_not_leak(cfg, batch):
    config = deep_merge(
        query_config(cfg), {"train": {"history_dropout": 1.0, "hint_dropout": 1.0}}
    )
    model = TransformersAdaptedM4RumModel(config).train()
    first = model(**batch).logits
    changed = copy.deepcopy(batch)
    for name in ("observe_power", "general_ghi", "general_temp", "timesfm3"):
        changed["batch_dict"][name].fill_(float("nan"))
    torch.testing.assert_close(first, model(**changed).logits, rtol=0, atol=0)
    for name in ("general_ghi_predict", "general_temp_predict"):
        changed["batch_dict"][name].fill_(float("nan"))
    out = model(**changed)
    assert torch.isfinite(out.logits).all() and torch.isfinite(out.loss)
    out.loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_segmented_decisions_match_weighted_median_and_preserve_raw():
    raw = torch.tensor([-10.0, 20.0, 130.0], dtype=torch.float64).repeat(2, 192, 1)
    saved = raw.clone()
    levels = [0.1, 0.5, 0.9]
    cfg = {
        "default": "p_star",
        "clip_negative": True,
        "capacity_cap": True,
        "by_step": [{"steps": [0, 16], "method": "p_star_l1"}],
    }
    selected = decide(raw, levels, torch.tensor([100.0, 100.0]), cfg)
    y = np.interp(np.arange(0.0025, 1, 0.005), levels, [0.0, 20.0, 100.0])
    weights = 1 / np.maximum(y, 20)
    median = y[np.searchsorted(weights.cumsum(), weights.sum() / 2)]
    np.testing.assert_allclose(selected["point"][:, :16].numpy(), median)
    torch.testing.assert_close(selected["point"][:, 16:], selected["p_star"][:, 16:])
    torch.testing.assert_close(raw, saved, rtol=0, atol=0)
    assert (
        selected["quantiles_cal"].min() == 0 and selected["quantiles_cal"].max() == 100
    )


def test_matched_recipes_only_change_network():
    root = Path(__file__).resolve().parents[1] / "config"
    control, query = [
        load_config(root / name)
        for name in ("pinball_control.yaml", "query_decoder.yaml")
    ]
    for key in ("dataset", "features", "train", "decision", "eval"):
        if key == "train":
            control[key]["weights_path"] = query[key]["weights_path"] = None
        assert control[key] == query[key]
    query["model"]["architecture"] = "legacy"
    assert query["model"] == control["model"]


def test_query_train_checkpoint_and_full_evaluation(cfg, splits, tmp_path, batch):
    out = tmp_path / "query"
    trained = train_m4_rum(query_config(cfg), splits_dir=splits, out_dir=out)
    assert trained["summary"]["best_step"] in (1, 2)
    model = load_member(out)
    assert model.cfg["model"]["architecture"] == "query_decoder"
    snapshot = load_member(out / "checkpoints" / "step_00001")
    assert snapshot.core.is_quantile
    report = evaluate(
        [out],
        [str(splits / "valid.parquet")],
        tmp_path / "eval",
        full=True,
        progress=False,
    )
    assert np.isfinite(report["summary"]["nanwang_short_D+1"])
    comparable = pd.read_csv(
        tmp_path / "eval" / "reports" / "m8_comparable_summary.csv"
    )
    table = pd.read_csv(tmp_path / "eval" / "reports" / "station_metrics.csv")
    np.testing.assert_allclose(
        comparable["nanwang_short_D+1"].iloc[0],
        pd.to_numeric(table["nanwang_short_D+1"], errors="coerce").mean(),
        equal_nan=True,
    )
    member_table = pd.read_csv(tmp_path / "eval" / "reports" / "members.csv")
    assert member_table["hint_unavailable_rate"].iloc[0] == 0
    median = load_member(out, point="median")
    assert median.cfg["decision"]["point"]["by_step"] == []
    torch.testing.assert_close(median(**batch).logits, model(**batch).aux["median"])
    legacy = TransformersAdaptedM4RumModel(
        deep_merge(query_config(cfg), {"model": {"architecture": "legacy"}})
    )
    with pytest.raises(RuntimeError):
        legacy.load_state_dict(model.state_dict(), strict=True)
