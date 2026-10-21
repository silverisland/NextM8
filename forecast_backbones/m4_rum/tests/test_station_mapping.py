"""Rum uses the public station-table mapping in inference and training loaders."""

import numpy as np
import pandas as pd
import pytest
import torch

from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from pipelines.forecasting.data import dataset_with_station_id
from pipelines.forecasting.rows import row_stations
from pipelines.forecasting.stations import StationMeta

CHINESE_COLS = {
    "station_id": "编号",
    "station": "站名",
    "lat": "纬度",
    "lon": "经度",
    "alt_m": None,
    "cap_mw": "容量",
}


@pytest.mark.parametrize(
    "columns,explicit,encoding",
    [
        (None, False, "utf-8"),
        (StationMeta.INFO_CSV_COLS, False, "utf-8"),
        (StationMeta.INFO_CSV_COLS, True, "utf-8"),
        (CHINESE_COLS, True, "utf-8-sig"),
        (CHINESE_COLS, True, "gbk"),
    ],
    ids=["standard", "info-auto", "info-explicit", "chinese-bom", "chinese-gbk"],
)
def test_station_csv_mapping_reaches_rum_features(
    cfg, batch, tmp_path, columns, explicit, encoding
):
    table = pd.DataFrame(
        {
            "station_id": [1, 2],
            "station": [" 光伏甲 ", "1002.0"],
            "lat": [30.0, 31.0],
            "lon": [110.0, 111.0],
            "alt_m": [0.0, 0.0],
            "cap_mw": [100.0, 200.0],
        }
    )
    if columns:
        table = table.drop(columns=[k for k, v in columns.items() if v is None])
        table = table.rename(columns={k: v for k, v in columns.items() if v})
    path = tmp_path / "station_meta.csv"
    table.to_csv(path, index=False, encoding=encoding)
    cfg["dataset"].update(
        station_meta=str(path), station_meta_cols=columns if explicit else None
    )
    model = TransformersAdaptedM4RumModel(cfg)
    inputs = dict(batch["batch_dict"])
    inputs.pop("station_id")
    inputs["station"] = ["光伏甲", " 1002 "]
    features = model.fe.build(inputs, "cpu")
    np.testing.assert_array_equal(features.station_id, [1, 2])
    torch.testing.assert_close(features.capacity[:, 0], torch.tensor([100.0, 200.0]))
    torch.testing.assert_close(
        features.history[:, :, 0], inputs["observe_power"] / features.capacity
    )
    assert model.station_meta.get(2)["alt_m"] == 0.0
    assert model.station_meta.has_coords(2)


@pytest.mark.parametrize(
    "column,values",
    [("station_id", [1, 1, 3]), ("station", ["1002", "1002.0", "C"])],
)
def test_duplicate_station_table_rejected(cfg, column, values):
    path = cfg["dataset"]["station_meta"]
    table = pd.read_csv(path)
    table[column] = values
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="重复"):
        TransformersAdaptedM4RumModel(cfg)


@pytest.mark.parametrize("names", [["unknown", "A"], ["A", "1"]])
def test_unknown_or_conflicting_station_names_rejected(cfg, batch, names):
    model = TransformersAdaptedM4RumModel(cfg)
    inputs = dict(batch["batch_dict"])
    inputs.pop("station_id")
    inputs["station"] = names
    with pytest.raises(ValueError, match="Unknown/conflicting"):
        model(inputs)


@pytest.mark.parametrize(
    "column,values",
    [
        ("station", [" A ", "B"]),
        ("station", ["1.0", " 2 "]),
        ("station_id", [" A ", "B"]),
        ("station_id", ["1.0", " 2 "]),
    ],
    ids=["station-name", "station-id-fallback", "text-id-name", "text-id-number"],
)
def test_training_loader_converts_station_identity(
    cfg, splits, tmp_path, collate, column, values
):
    model = TransformersAdaptedM4RumModel(cfg)
    frame = (
        pd.read_parquet(splits / "train.parquet").iloc[:2].drop(columns="station_id")
    )
    frame[column] = values
    source = tmp_path / "named.parquet"
    frame.to_parquet(source, index=False)
    dataset = dataset_with_station_id(
        [str(source)], model.cfg, model.station_meta, tmp_path / "cache"
    )
    np.testing.assert_array_equal(row_stations(dataset, "station_id"), [1, 2])
    inputs = collate([dataset[0], dataset[1]])
    output = model(**inputs)
    np.testing.assert_array_equal(output.aux["station_id"], [1, 2])
    assert torch.isfinite(output.loss)


def test_training_loader_rejects_unknown_station(cfg, splits, tmp_path):
    model = TransformersAdaptedM4RumModel(cfg)
    source = tmp_path / "unknown.parquet"
    frame = pd.read_parquet(splits / "train.parquet").iloc[:1]
    frame.drop(columns="station_id").assign(station="unknown").to_parquet(source)
    with pytest.raises(ValueError, match="找不到"):
        dataset_with_station_id(
            [str(source)], model.cfg, model.station_meta, tmp_path / "cache"
        )
