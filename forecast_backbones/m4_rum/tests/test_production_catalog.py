"""Retained experiment coverage, paired deltas and independent replay failures."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from forecast_backbones.m4_rum import production
from forecast_backbones.m4_rum.config import load_config
from forecast_backbones.m4_rum.train import train_m4_rum
from forecast_backbones.m4_rum.tests.conftest import make_frame
from pipelines.forecasting.config import dump_config


def test_catalog_preserves_every_active_recipe_and_no_smoothing():
    catalog = production.read_catalog()
    config_root = Path(production.__file__).parent / "config"
    assert {
        str((production.ROOT / x["config"]).resolve()) for x in catalog.values()
    } == {str(p.resolve()) for p in config_root.rglob("*.yaml")}
    assert len(catalog) == 35
    assert catalog["a0"]["equivalent_to"] == "q6"
    assert not {"a19", "a20", "a21"} & catalog.keys()
    assert "a17_epoch" in catalog


def test_deltas_only_pair_same_seed_rows_and_valid_scores():
    metric = production.METRICS[0]
    rows = [
        {
            "experiment": "q6",
            "seed": 0,
            "status": "ok",
            "row_signature": "x",
            metric: 76.88,
        },
        {
            "experiment": "a17",
            "seed": 0,
            "status": "ok",
            "row_signature": "x",
            metric: 77.33,
        },
        {
            "experiment": "other",
            "seed": 0,
            "status": "ok",
            "row_signature": "y",
            metric: 99.0,
        },
        {
            "experiment": "a17",
            "seed": 1,
            "status": "ok",
            "row_signature": "x",
            metric: 99.0,
        },
        {
            "experiment": "failed",
            "seed": 0,
            "status": "failed",
            "error": "missing weights",
        },
    ]
    production.add_deltas(rows, "q6")
    assert rows[1][metric + "_delta_pp"] == pytest.approx(0.45)
    for row in rows[2:]:
        assert not row["comparable_to_baseline"] and row[metric + "_delta_pp"] is None


def test_source_provenance_without_git_executable(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(production.subprocess, "run", no_git)
    version = production.code_version()
    assert version["commit"] is None and version["worktree_status"] is None
    assert version["source_sha256"]["forecast_backbones/m4_rum/production.py"]


def test_replay_each_checkpoint_independently_and_record_failure(
    cfg, splits, tmp_path, monkeypatch
):
    cfg["train"].update(loss="pinball", weights_path=None)
    checkpoint = tmp_path / "trained"
    train_m4_rum(cfg, splits_dir=splits, out_dir=checkpoint)
    recipe = tmp_path / "recipe.yaml"
    dump_config(cfg, recipe)
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "experiments": [
                    {
                        "id": name,
                        "group": "test",
                        "description": "test",
                        "config": str(recipe),
                    }
                    for name in ("q6", "a17", "broken")
                ],
            }
        )
    )
    mapping = tmp_path / "runs.yaml"
    mapping.write_text(
        yaml.safe_dump(
            {
                "q6": str(checkpoint),
                "a17": str(checkpoint),
                "broken": str(tmp_path / "missing"),
            }
        )
    )
    out = tmp_path / "comparison"
    replay = tmp_path / "replay.parquet"
    # Monthly summaries need >=3 valid days; ultra-short needs 15-min issue cadence.
    make_frame(start="2025-06-20", days=5, every=15).to_parquet(replay, index=False)
    monkeypatch.setattr(production, "code_version", lambda: {"commit": "test"})
    argv = [
        "--mode",
        "evaluate",
        "--catalog",
        str(catalog),
        "--experiments",
        "q6",
        "a17",
        "broken",
        "--run-map",
        str(mapping),
        "--out",
        str(out),
        "--test",
        str(replay),
        "--device",
        "cpu",
        "--no-progress",
    ]
    with pytest.raises(SystemExit) as exc:
        production.main(argv)
    assert exc.value.code == 1
    result = pd.read_csv(out / "comparison.csv")
    assert result.status.tolist() == ["ok", "ok", "failed"]
    assert result.loc[1, "n_members"] == 1  # Not an ensemble of candidates.
    for metric in production.METRICS:
        assert result.loc[1, metric + "_delta_pp"] == 0
    assert not result.loc[2, "comparable_to_baseline"]
    frozen = json.loads((out / "q6/seed_0/checkpoint_manifest.json").read_text())
    assert frozen["weights_sha256"] == production.sha256(
        checkpoint / "model.safetensors"
    )
    with pytest.raises(ValueError, match="empty"):
        production.main(argv)


def test_training_command_uses_selected_python_and_keeps_recipe(
    cfg, splits, tmp_path, monkeypatch
):
    recipe = tmp_path / "recipe.yaml"
    dump_config(cfg, recipe)
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "experiments": [
                    {
                        "id": "q6",
                        "group": "test",
                        "description": "test",
                        "config": str(recipe),
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(production, "code_version", lambda: {})
    commands = []
    monkeypatch.setattr(
        production.subprocess, "run", lambda command, **kw: commands.append(command)
    )
    argv = [
        "--mode",
        "train",
        "--catalog",
        str(catalog),
        "--experiments",
        "q6",
        "--seeds",
        "2",
        "--run-root",
        str(tmp_path / "runs"),
        "--out",
        str(tmp_path / "logs"),
        "--train",
        str(splits / "train.parquet"),
        "--valid",
        str(splits / "valid.parquet"),
    ]
    rows = production.run(production.build_parser().parse_args(argv))
    assert rows[0]["status"] == "ok"
    command = commands[0]
    assert command[command.index("--venv-python") + 1] == production.sys.executable
    requested = load_config(command[command.index("--config") + 1])
    assert requested["model"] == load_config(recipe)["model"]
    assert requested["run"]["seeds"] == [2]
    assert requested["train"]["weights_path"] is None
    # Existing trained data must not be overwritten by subsequent queues.
    run = Path(rows[0]["run"])
    run.mkdir(parents=True)
    (run / "model.safetensors").write_bytes(b"keep")
    argv[argv.index("--out") + 1] = str(tmp_path / "second_logs")
    rows = production.run(production.build_parser().parse_args(argv))
    assert rows[0]["status"] == "failed" and "overwrite" in rows[0]["error"]
    assert (run / "model.safetensors").read_bytes() == b"keep"
    assert len(commands) == 1
