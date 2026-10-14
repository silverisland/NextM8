"""站表列名映射（2026-10-14）：dataset.station_meta_cols = {内部名: 你表里的列名}，表头随意；值为 null 的列视作没有（lat/lon/alt_m 可以）。
显式给的站表路径不存在必须报错，不能静默回退到仓库那份。"""
import pytest
from pathlib import Path
from forecast_backbones.m8.config import load_config
from forecast_backbones.m8.solar import StationMeta
from forecast_backbones.m8.transformer_models import load_station_meta, TransformersAdaptedM8Model

CN = {"station_id": "场站编号", "station": "场站名称", "lat": "纬度", "lon": "经度", "alt_m": "海拔", "cap_mw": "装机容量"}


def _write(tmp_path, header, rows, name="stations.csv"):
    p = Path(tmp_path) / name
    p.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")
    return p


def test_mapping_renames_columns(tmp_path):
    p = _write(tmp_path, "场站编号,场站名称,纬度,经度,海拔,装机容量,备注",
               ["1001,中广核和田,36.95,80.05,1568,100,x", "1002,黄河共和,35.59,100.19,,500,y"])
    meta = StationMeta.from_csv(p, cols=CN)
    assert meta.get(1001)["cap_mw"] == 100 and meta.get(1001)["lat"] == 36.95 and meta.has_coords(1002)
    assert meta.get(1002)["alt_m"] == 0.0                                   # 海拔空 → 0
    assert meta.resolve(["中广核和田"])[0] == {"中广核和田": 1001}         # 站名按映射后的列找
    assert meta.name(1002) == "黄河共和"


def test_null_means_column_absent(tmp_path):
    p = _write(tmp_path, "sid,name,cap", ["7,A,50"])
    meta = StationMeta.from_csv(p, cols={"station_id": "sid", "station": "name", "cap_mw": "cap", "lat": None, "lon": None, "alt_m": None})
    assert meta.get(7)["cap_mw"] == 50 and not meta.has_coords(7) and meta.get(7)["alt_m"] == 0.0


def test_partial_mapping_keeps_other_names(tmp_path):
    p = _write(tmp_path, "station_id,station,lat,lon,alt_m,capacity", ["3,S3,30,110,100,80"])
    meta = StationMeta.from_csv(p, cols={"cap_mw": "capacity"})
    assert meta.get(3)["cap_mw"] == 80 and meta.has_coords(3)


def test_missing_mapped_column_names_both(tmp_path):
    p = _write(tmp_path, "场站编号,场站名称,纬度,经度,海拔", ["1,A,1,1,1"])
    with pytest.raises(KeyError, match="装机容量"):
        StationMeta.from_csv(p, cols=CN)
    with pytest.raises(ValueError, match="station_meta_cols"):
        StationMeta.from_csv(p, cols={"capacity": "装机容量"})              # 内部名写错


def test_model_reads_mapping_from_config(tmp_path):
    p = _write(tmp_path, "场站编号,场站名称,纬度,经度,海拔,装机容量", ["1001,S1001,36.95,80.05,1568,100", "1002,S1002,35.59,100.19,2767,500"])
    cfg = load_config({"dataset": {"station_meta": str(p), "station_meta_cols": CN},
                       "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "pos_in_patch_dim": 8}})
    m = TransformersAdaptedM8Model(cfg)
    assert m.station_meta.get(1002)["cap_mw"] == 500 and m.station_meta.has_coords(1001)


def test_explicit_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_station_meta(str(tmp_path / "nope.csv"))
    cfg = load_config({"dataset": {"station_meta": str(tmp_path / "nope.csv")}})
    with pytest.raises(FileNotFoundError):
        TransformersAdaptedM8Model(cfg)
