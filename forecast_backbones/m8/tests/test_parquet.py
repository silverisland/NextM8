"""m8 自带的拼表 / 站号归一（原先借自 pipelines.time_splits，2026-10-14 搬进来）：行为逐条守住。"""
import numpy as np
import pandas as pd
import pytest
from forecast_backbones.m8._parquet import station_keys, concat_parquets


def test_station_keys_normalizes_numeric_forms():
    assert station_keys(np.array([1001, 1002])).tolist() == ["1001", "1002"]
    assert station_keys(np.array([1001.0, 1002.0])).tolist() == ["1001", "1002"]
    assert station_keys(np.array(["1001.0", "雅砻江", " 7 "], dtype=object)).tolist() == ["1001", "雅砻江", "7"]   # 带空格的数字也归一
    with pytest.raises(ValueError, match="NaN"):
        station_keys(np.array([1.0, np.nan]))


def test_concat_parquets_reads_folder_files_in_order(tmp_path):
    a = pd.DataFrame({"station": ["A", "A"], "v": [1.0, 2.0]}); b = pd.DataFrame({"station": ["B"], "v": [3.0]})
    a.to_parquet(tmp_path / "a.parquet", index=False); b.to_parquet(tmp_path / "b.parquet", index=False)
    df = concat_parquets([tmp_path / "a.parquet", tmp_path / "b.parquet"], station_col="station")
    assert df["station"].tolist() == ["A", "A", "B"] and df["v"].tolist() == [1.0, 2.0, 3.0]
    assert len(concat_parquets([tmp_path / "a.parquet"])) == 2                              # 单文件原样返回


def test_concat_parquets_rejects_column_mismatch(tmp_path):
    pd.DataFrame({"station": ["A"], "v": [1.0]}).to_parquet(tmp_path / "a.parquet", index=False)
    pd.DataFrame({"station": ["B"], "w": [1.0]}).to_parquet(tmp_path / "b.parquet", index=False)
    with pytest.raises(ValueError, match="不一致"):
        concat_parquets([tmp_path / "a.parquet", tmp_path / "b.parquet"])


def test_concat_parquets_unifies_station_column_types_only(tmp_path):
    pd.DataFrame({"station": [1001], "v": [1.0]}).to_parquet(tmp_path / "a.parquet", index=False)
    pd.DataFrame({"station": ["雅砻江"], "v": [2.0]}).to_parquet(tmp_path / "b.parquet", index=False)
    df = concat_parquets([tmp_path / "a.parquet", tmp_path / "b.parquet"], station_col="station")
    assert df["station"].tolist() == ["1001", "雅砻江"]
    pd.DataFrame({"station": ["A"], "v": ["x"]}).to_parquet(tmp_path / "c.parquet", index=False)
    with pytest.raises(ValueError, match="'v'"):
        concat_parquets([tmp_path / "a.parquet", tmp_path / "c.parquet"], station_col="station")
