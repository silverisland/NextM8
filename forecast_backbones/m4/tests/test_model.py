import copy

import pytest
import torch

from forecast_backbones.m4.config import load_config
from forecast_backbones.m4.transformer_models import TransformersAdaptedM4Model


@pytest.mark.parametrize("loss", ["mae", "mse", "huber"])
@pytest.mark.parametrize("capacity_column", ["cap_power_on", "custom_capacity", None])
def test_original_main_loss_and_backward(cfg, batch, loss, capacity_column):
    cfg["train"]["loss"] = loss
    cfg["dataset"]["label_norm"] = [capacity_column] if capacity_column else []
    model = TransformersAdaptedM4Model(cfg).train()
    labels = batch["labels"]["observe_power_future"]
    bd = dict(batch["batch_dict"])
    if capacity_column:
        bd[capacity_column] = torch.tensor([50.0, 200.0])
        capacity = bd[capacity_column][:, None]
    else:
        capacity = torch.maximum(labels, torch.ones_like(labels))
    out = model(bd, labels)
    criterion = {
        "mae": torch.nn.L1Loss,
        "mse": torch.nn.MSELoss,
        "huber": torch.nn.HuberLoss,
    }[loss]()
    expected = criterion(
        (out.logits / capacity).unsqueeze(1), (labels / capacity).unsqueeze(1)
    )
    torch.testing.assert_close(out.loss, expected, rtol=0, atol=0)
    out.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_missing_capacity_column_uses_original_target_fallback(cfg, batch):
    model = TransformersAdaptedM4Model(cfg)
    bd = {k: v for k, v in batch["batch_dict"].items() if k != "cap_power_on"}
    labels = batch["labels"]["observe_power_future"]
    out = model(bd, labels)
    capacity = torch.maximum(labels, torch.ones_like(labels))
    torch.testing.assert_close(
        out.loss,
        torch.nn.functional.l1_loss(out.logits / capacity, labels / capacity),
        rtol=0,
        atol=0,
    )


def test_labels_and_row_capacity_never_enter_predictions(cfg, batch):
    model = TransformersAdaptedM4Model(cfg)
    bd = batch["batch_dict"]
    expected = model(bd).logits
    other = dict(
        bd,
        observe_power_future=torch.full((2, 192), 9999.0),
        cap_power_on=torch.tensor([1e6, 0.0]),
    )
    torch.testing.assert_close(model(other).logits, expected, rtol=0, atol=0)
    assert expected.shape == (2, 192)


def test_feature_availability_and_causal_windows(cfg, batch):
    model = TransformersAdaptedM4Model(cfg)
    bd = dict(batch["batch_dict"])
    bd["observe_power"] = bd["observe_power"][:, -100:]
    bd["general_ghi"] = torch.full_like(bd["general_ghi"], float("nan"))
    bd["chronos"] = torch.full_like(bd["chronos"], float("nan"))
    result = model(bd)
    assert torch.isfinite(result.logits).all()
    assert result.aux["hint_unavailable"].all()
    del bd["chronos"]
    with pytest.raises(ValueError, match="requires hint"):
        model(bd)


def test_configs_and_weights_are_independent(cfg):
    first = TransformersAdaptedM4Model(cfg)
    second = TransformersAdaptedM4Model(cfg)
    original = second.model.head.pred.weight.detach().clone()
    with torch.no_grad():
        first.model.head.pred.weight.fill_(42)
    torch.testing.assert_close(second.model.head.pred.weight, original)
    first.cfg["features"]["past_future"][0][0] = "changed"
    assert second.cfg["features"]["past_future"][0][0] == "general_ghi"
    assert load_config()["run"]["out_dir"] == "runs/m4/patchtst"
    with pytest.raises(ValueError, match="run.model_name"):
        load_config({"run": {"model_name": "m8"}})


@pytest.mark.parametrize(
    "override",
    [
        {"features": {"hint_col": "observe_power_future"}},
        {"features": {"col_aliases": {"chronos": ["OBSERVE_POWER_FUTURE"]}}},
        {"model": {"d_model": 15, "n_heads": 4}},
        {"model": {"patch_len": 0}},
        {"train": {"schedule": {"max_steps": -1}}},
        {"model": {"architecture": "query_decoder"}},
    ],
)
def test_invalid_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        load_config(override)


def test_explicit_missing_weights_fail(cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    cfg["train"]["weights_path"] = str(tmp_path / "missing.safetensors")
    with pytest.raises(FileNotFoundError, match="weights not found"):
        TransformersAdaptedM4Model(cfg)
