"""Exercise actual Trainer stopping for the published M4 schedules."""

import json
from pathlib import Path

import pytest

from forecast_backbones.m4.config import load_config
from forecast_backbones.m4.train import M4EvalCallback, train_m4
from pipelines.forecasting.config import deep_merge
from pipelines.forecasting.training import build_training_args

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
RECIPES = ("epoch_earlystop", "step_earlystop", "step_ema", "refit_steps")


def test_recipe_budgets_and_model_independence(tmp_path):
    defaults = load_config()
    configs = [load_config(CONFIG_DIR / f"{name}.yaml") for name in RECIPES]
    assert len({c["run"]["out_dir"] for c in configs}) == len(RECIPES)
    for name, config in zip(RECIPES, configs):
        assert config["model"] == defaults["model"]
        assert config["train"]["loss"] == "quantile"
        args = build_training_args(config, str(tmp_path / name))
        if name == "epoch_earlystop":
            assert args.max_steps == -1
            assert args.num_train_epochs == 100
            assert config["train"]["schedule"]["eval_every_steps"] is None
        else:
            assert args.max_steps == 6000
    assert configs[2]["train"]["ema_decay"] == 0.99
    assert all(configs[i]["train"]["ema_decay"] is None for i in (0, 1, 3))


@pytest.mark.parametrize(
    "recipe,expected_steps",
    [("epoch_earlystop", [6, 12, 18]), ("step_earlystop", [2, 4, 6])],
)
def test_actual_early_stop_cadence(
    cfg, splits, tmp_path, monkeypatch, recipe, expected_steps
):
    config = deep_merge(load_config(CONFIG_DIR / f"{recipe}.yaml"), cfg)
    # Keep the published mode while shortening the experiment to a tiny CPU run.
    epoch_mode = recipe == "epoch_earlystop"
    config = deep_merge(
        config,
        {
            "train": {
                "schedule": {
                    "max_steps": None if epoch_mode else 30,
                    "epochs_max": 10,
                    "eval_every_steps": None if epoch_mode else 2,
                    "save_every": 0,
                },
                "early_stop": {"patience": 2},
            }
        },
    )
    # A plateau must stop after the first best score plus two non-improvements.
    monkeypatch.setattr(M4EvalCallback, "_score", lambda self: 50.0)
    out = tmp_path / recipe
    trained = train_m4(config, splits_dir=splits, out_dir=out)
    history = json.loads((out / "train_log.json").read_text())["history"]
    assert [entry["step"] for entry in history] == expected_steps
    assert [entry["bad_evals"] for entry in history] == [0, 1, 2]
    assert trained["summary"]["best_step"] == expected_steps[0]
    assert (out / "model.safetensors").is_file()
    assert not (out / "checkpoints").exists()
    if epoch_mode:
        assert [entry["epoch"] for entry in history] == [1, 2, 3]
