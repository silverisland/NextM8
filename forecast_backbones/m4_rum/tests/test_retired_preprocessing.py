"""No PV smoothing at runtime; legacy no-op checkpoint configurations still load."""

import copy

import pytest
import torch

from forecast_backbones.m4_rum.config import load_config
from forecast_backbones.m4_rum.evaluate import load_member
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from pipelines.forecasting.checkpoints import save_weights
from pipelines.forecasting.config import deep_merge, dump_config, raw_override


@pytest.mark.parametrize("architecture", ["legacy", "query_decoder"])
def test_old_disabled_block_loads_without_changing_weights_predictions_or_loss(
    cfg, batch, tmp_path, architecture
):
    config = deep_merge(cfg, {"model": {"architecture": architecture}})
    model = TransformersAdaptedM4RumModel(config)
    original = model(**batch)
    old_config = deep_merge(
        model.cfg,
        {
            "preprocess": {
                "pv_smoothing": {
                    "history_window": 1,
                    "target_window": 1,
                    "alignment": "centered",
                }
            }
        },
    )
    unchanged = copy.deepcopy(old_config)
    assert "preprocess" not in load_config(old_config)
    assert old_config == unchanged
    save_weights(model, tmp_path / "model.safetensors")
    dump_config(old_config, tmp_path / "model_config.yaml")
    loaded = load_member(tmp_path)
    result = loaded(**batch)
    torch.testing.assert_close(result.logits, original.logits, atol=0, rtol=0)
    torch.testing.assert_close(result.loss, original.loss, atol=0, rtol=0)
    dump_config(loaded.cfg, tmp_path / "new_config.yaml")
    assert "preprocess" not in raw_override(tmp_path / "new_config.yaml")
    # Training loss uses raw labels and feature preparation uses raw history.
    loaded.train()
    output = loaded(**batch)
    cap = output.aux["cap"][:, None]
    truth = batch["labels"]["observe_power_future"]
    torch.testing.assert_close(
        output.loss, ((output.logits - truth) / cap).square().mean()
    )
    features = loaded.fe.build(batch["batch_dict"], "cpu", training=True)
    torch.testing.assert_close(
        features.history[:, :, 0], batch["batch_dict"]["observe_power"] / cap
    )


@pytest.mark.parametrize(
    "options",
    [
        {"history_window": 3},
        {"target_window": 5},
        {"history_window": True},
        {"alignment": "trailing"},
    ],
)
def test_actual_smoothing_checkpoints_are_not_silently_reinterpreted(options):
    with pytest.raises(ValueError, match="PV smoothing has been removed"):
        load_config({"preprocess": {"pv_smoothing": options}})


def test_unknown_retired_options_are_not_ignored():
    with pytest.raises(ValueError, match="Unknown retired"):
        load_config({"preprocess": {"pv_smoothing": {"unknown": 1}}})
    assert "preprocess" not in load_config()
