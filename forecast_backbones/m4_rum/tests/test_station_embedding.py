"""Station conditioning, unseen-station safety, and saved training membership."""

import copy
from pathlib import Path

import pytest
import torch

from forecast_backbones.m4_rum.config import load_config
from forecast_backbones.m4_rum.evaluate import evaluate, load_member
from forecast_backbones.m4_rum.train import train_m4_rum
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from pipelines.forecasting.config import deep_merge


def station_config(cfg, **train):
    return deep_merge(
        cfg,
        {
            "dataset": {"known_stations": [1, 2]},
            "model": {
                "architecture": "query_decoder",
                "decoder_layers": 1,
                "station_embedding": True,
                "n_station_slots": 16,
            },
            "train": {"loss": "pinball", "slot_dropout": 0.0, **train},
            "decision": {"point": {"default": "p_star"}},
        },
    )


def test_station_identity_changes_prediction_and_receives_gradients(cfg, batch):
    torch.manual_seed(10)
    model = TransformersAdaptedM4RumModel(station_config(cfg))
    f = model.fe.build(batch["batch_dict"], "cpu")
    f.station_id[:] = 1
    first = model.core(f)
    f.station_id[:] = 2
    second = model.core(f)
    assert not torch.allclose(first, second)
    (first.square().mean() + second.square().mean()).backward()
    grad = model.core.station_embedding.weight.grad
    assert torch.isfinite(grad).all()
    assert grad[2].abs().sum() > 0 and grad[3].abs().sum() > 0
    assert grad[0].abs().sum() == 0


def test_unseen_hash_collision_and_dynamic_training_membership(cfg, batch):
    model = TransformersAdaptedM4RumModel(station_config(cfg))
    # 17 collides with known station 1, but MUST remain unknown.
    assert model.core.station_slots([1, 2, 17, 999], "cpu").tolist() == [2, 3, 0, 0]
    f = model.fe.build(batch["batch_dict"], "cpu")
    f.station_id[:] = 17
    first = model.core(f)
    f.station_id[:] = 999
    torch.testing.assert_close(first, model.core(f), rtol=0, atol=0)
    for known in (None, []):
        model.cfg["dataset"]["known_stations"] = known
        assert model.core.station_slots([1, 2], "cpu").tolist() == [0, 0]
    # train_m4_rum discovers the training membership after constructing the model.
    model.cfg["dataset"]["known_stations"] = [2]
    assert model.core.station_slots([1, 2], "cpu").tolist() == [0, 3]


def test_station_dropout_trains_unknown_slot_and_is_disabled_at_eval(cfg, batch):
    model = TransformersAdaptedM4RumModel(station_config(cfg, slot_dropout=1.0))
    model.train()
    assert model.core.station_slots([1, 2], "cpu").tolist() == [0, 0]
    model(**batch).loss.backward()
    grad = model.core.station_embedding.weight.grad
    assert grad[0].abs().sum() > 0
    assert grad[1:].abs().sum() == 0
    model.eval()
    assert model.core.station_slots([1, 2], "cpu").tolist() == [2, 3]


def test_disabled_embedding_preserves_old_state_and_seeded_base_weights(cfg, batch):
    old = deep_merge(
        cfg, {"model": {"architecture": "query_decoder"}, "train": {"loss": "pinball"}}
    )
    torch.manual_seed(42)
    original = TransformersAdaptedM4RumModel(old)
    torch.manual_seed(42)
    enabled = TransformersAdaptedM4RumModel(
        deep_merge(old, {"model": {"station_embedding": True}})
    )
    assert not any("station_embedding" in k for k in original.state_dict())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, enabled.state_dict()[key], rtol=0, atol=0)
    restored = TransformersAdaptedM4RumModel(old)
    restored.load_state_dict(original.state_dict(), strict=True)
    torch.testing.assert_close(
        original(**batch).logits, restored(**batch).logits, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "override",
    [
        {"model": {"station_embedding": "true"}},
        {"model": {"n_station_slots": 0}},
        {"model": {"n_station_slots": True}},
        {"model": {"architecture": "legacy"}},
        {"train": {"slot_dropout": -0.1}},
        {"train": {"slot_dropout": float("nan")}},
        {"train": {"slot_dropout": True}},
        {"dataset": {"known_stations": [1, 1]}},
        {"dataset": {"known_stations": [True]}},
    ],
)
def test_invalid_station_configuration(cfg, override):
    with pytest.raises(ValueError):
        load_config(deep_merge(station_config(cfg), override))


def test_embedding_training_checkpoint_and_unseen_evaluation(
    cfg, splits, tmp_path, batch
):
    out = tmp_path / "station_run"
    train_m4_rum(station_config(cfg, slot_dropout=0.3), splits_dir=splits, out_dir=out)
    model = load_member(out)
    assert model.cfg["dataset"]["known_stations"] == [
        1
    ]  # Validation station 2 excluded.
    assert model.core.station_slots([1, 2], "cpu").tolist() == [2, 0]
    snapshot = load_member(out / "checkpoints" / "step_00001")
    assert snapshot.core.station_slots([1, 2], "cpu").tolist() == [2, 0]
    again = load_member(out)
    torch.testing.assert_close(
        model(**batch).logits, again(**batch).logits, rtol=0, atol=0
    )
    report = evaluate(
        [out],
        [str(splits / "valid.parquet")],
        tmp_path / "eval",
        full=True,
        progress=False,
    )
    assert report["summary"]["n_unseen_stations"] == 1
    assert (tmp_path / "eval/reports/m8_comparable_summary.csv").is_file()


def test_capacity_and_station_recipes_form_matched_comparisons():
    root = Path(__file__).resolve().parents[1] / "config/d1_sweep"
    names = ["q0_baseline", "q7_station", "q8_width256", "q9_width256_station"]
    configs = [load_config(root / f"{name}.yaml") for name in names]
    for c in configs:
        c.pop("run")
        c["train"]["weights_path"] = None
    for a, b in ((0, 1), (2, 3)):
        disabled = copy.deepcopy(configs[b])
        disabled["model"]["station_embedding"] = False
        assert configs[a] == disabled
    large = copy.deepcopy(configs[2])
    large["model"].update(d_model=128, d_ff=256)
    assert configs[0] == large
