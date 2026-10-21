"""Separate datasets must preserve validation, station identity and input boundaries."""

import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from safetensors.torch import load_file

from forecast_backbones.m4_rum.config import load_config
from forecast_backbones.m4_rum.train import main, train_m4_rum
from pipelines.forecasting.data import load_train_valid
from pipelines.forecasting.rows import row_stations
from pipelines.forecasting.stations import StationMeta, load_station_meta


def test_train_separate_file_and_validation_directory(cfg, splits, tmp_path):
    validation = tmp_path / "validation files" / "date=2025-06-20"
    validation.mkdir(parents=True)
    shutil.copyfile(splits / "valid.parquet", validation / "data.parquet")
    # An empty partition is harmless when the validation dataset has other rows.
    pd.read_parquet(splits / "valid.parquet").iloc[:0].to_parquet(
        validation / "empty.parquet", index=False
    )
    out = tmp_path / "trained"
    result = train_m4_rum(
        cfg,
        train_parquet=[str(splits / "train.parquet")],
        valid_parquet=[str(validation.parent)],
        out_dir=out,
    )

    summary = result["summary"]
    expected = {
        name: len(pd.read_parquet(splits / f"{name}.parquet"))
        for name in ("train", "valid")
    }
    assert summary["n_train_rows"] == expected["train"]
    assert summary["n_val_rows"] == expected["valid"]
    assert np.isfinite(summary["nanwang_short_D+1"])
    assert np.isfinite(summary["nrmse_pct"])
    assert summary["best_step"] in (1, 2)
    assert len(result["history"]) == 2
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert manifest["refit"] is False
    assert manifest["counts"] == {"train": expected["train"], "val": expected["valid"]}
    saved = yaml.safe_load((out / "model_config.yaml").read_text())
    assert saved["dataset"]["known_stations"] == [1]
    assert len(pd.read_parquet(out / "result_val.parquet")) == expected["valid"]
    best = load_file(out / "model.safetensors")
    periodic = load_file(
        out / "checkpoints" / f"step_{summary['best_step']:05d}" / "model.safetensors"
    )
    assert best.keys() == periodic.keys()
    for key in best:
        np.testing.assert_array_equal(best[key].numpy(), periodic[key].numpy())


def test_reused_named_station_caches_stay_separate(cfg, splits, tmp_path):
    normalized = load_config(cfg)
    meta = StationMeta.from_csv(normalized["dataset"]["station_meta"])
    paths, counts = [], []
    for name, station in (("train", "A"), ("valid", "B")):
        path = tmp_path / name / "date=2025-06-20" / "data.parquet"
        path.parent.mkdir(parents=True)
        frame = pd.read_parquet(splits / f"{name}.parquet")
        frame.drop(columns="station_id").assign(station=station).to_parquet(
            path, index=False
        )
        paths.append(path)
        counts.append(len(frame))
    inputs = ([str(paths[0].parent.parent)], [str(paths[1].parent / "*.parquet")])

    for reuse in (False, True):
        train, valid, manifest = load_train_valid(
            *inputs, normalized, meta, tmp_path / "cache", reuse=reuse
        )
        assert (len(train), len(valid)) == tuple(counts)
        assert set(row_stations(train, "station_id")) == {1}
        assert set(row_stations(valid, "station_id")) == {2}
        assert manifest["train_parquet"] == [str(paths[0].resolve())]
        assert manifest["valid_parquet"] == [str(paths[1].resolve())]


@pytest.mark.parametrize(
    "inputs",
    [
        {"valid_parquet": ["valid.parquet"]},
        {"splits_dir": "splits", "valid_parquet": ["valid.parquet"]},
        {
            "train_parquet": ["train.parquet"],
            "valid_parquet": ["valid.parquet"],
            "refit": True,
        },
    ],
)
def test_invalid_validation_options_fail_before_loading(cfg, inputs):
    with pytest.raises(ValueError):
        train_m4_rum(cfg, **inputs)


@pytest.mark.parametrize("symlink", [False, True])
def test_reject_same_file_in_training_and_validation(cfg, splits, tmp_path, symlink):
    path = splits / "train.parquet"
    valid = tmp_path / "validation.parquet" if symlink else path
    if symlink:
        valid.symlink_to(path)
    normalized = load_config(cfg)
    meta = StationMeta.from_csv(normalized["dataset"]["station_meta"])
    with pytest.raises(ValueError):
        load_train_valid(
            [str(path)], [str(valid)], normalized, meta, tmp_path / "cache"
        )


@pytest.mark.parametrize("empty_file", [False, True])
def test_reject_empty_validation_data(cfg, splits, tmp_path, empty_file):
    valid = tmp_path / "empty validation"
    valid.mkdir()
    if empty_file:
        pd.read_parquet(splits / "valid.parquet").iloc[:0].to_parquet(
            valid / "data.parquet", index=False
        )
    normalized = load_config(cfg)
    meta = StationMeta.from_csv(normalized["dataset"]["station_meta"])
    with pytest.raises((ValueError, FileNotFoundError), match="valid"):
        load_train_valid(
            [str(splits / "train.parquet")],
            [str(valid)],
            normalized,
            meta,
            tmp_path / "cache",
        )


def test_cli_accepts_multiple_validation_paths(monkeypatch):
    captured = {}

    def capture(*args, **kwargs):
        captured.update(inspect.signature(train_m4_rum).bind(*args, **kwargs).arguments)

    monkeypatch.setattr("forecast_backbones.m4_rum.train.train_m4_rum", capture)
    main(
        [
            "--train-parquet",
            "/data/train files",
            "--valid-parquet",
            "/data/validation files/*.parquet",
            "/data/extra validation.parquet",
        ]
    )
    assert captured["valid_parquet"] == [
        "/data/validation files/*.parquet",
        "/data/extra validation.parquet",
    ]


@pytest.mark.parametrize("recipe", ["configs.yaml", "pinball.yaml"])
def test_cli_station_meta_replaces_online_column_mapping(cfg, monkeypatch, recipe):
    captured = {}

    def capture(config, *args, **kwargs):
        captured.update(load_config(config)["dataset"])

    monkeypatch.setattr("forecast_backbones.m4_rum.train.train_m4_rum", capture)
    main(
        [
            "--config",
            str(Path(__file__).resolve().parents[1] / "config" / recipe),
            "--train-parquet",
            "/data/train",
            "--station-meta",
            cfg["dataset"]["station_meta"],
        ]
    )
    assert captured["station_meta"] == cfg["dataset"]["station_meta"]
    assert captured["station_meta_cols"] is None
    meta = load_station_meta(captured["station_meta"], captured["station_meta_cols"])
    assert meta.id_of(" A ") == 1
    assert meta.get(1)["cap_mw"] == 100
