"""线上推理脚本 single_model_scene_infer_m8.py 的列口径（2026-10-19）：协变量大写 general_GHI / general_TEMP、插值列叫 tsfm_predict、
站列是 plant_guangfu1358 这种站名、站表是 info.csv 表头。与训练口径（小写、timesfm3、station_id）喂进模型的输入必须逐值相同。"""
import warnings

import numpy as np
import pandas as pd
import pytest

from forecast_backbones.m8.config import load_config
from forecast_backbones.m8.features import FeatureBuilder
from forecast_backbones.m8.solar import StationMeta
from forecast_backbones.m8.train import expand_parquets, report_inputs
from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
from forecast_backbones.m8.tests.conftest import CAP, make_frame

ONLINE = {"general_ghi": "general_GHI", "general_ghi_predict": "general_GHI_predict",
          "general_temp": "general_TEMP", "general_temp_predict": "general_TEMP_predict", "timesfm3": "tsfm_predict"}


def _info_csv(tmp_path):
    p = tmp_path / "info.csv"
    p.write_text("plantid,plantname,plant_pointname,GCCAPACITY,LONGITUDE,LATITUDE\n"
                 f"1001,光伏甲,plant_guangfu1001,{CAP[1001]},101.0,30.0\n"
                 f"1002,光伏乙,plant_guangfu1002,{CAP[1002]},101.0,30.0\n"
                 "2701,风电丙,plant_fengdian2701,99.0,,\n", encoding="utf-8")
    return p


def _cfg(small_cfg, meta):
    return {**small_cfg, "dataset": {**small_cfg["dataset"], "station_meta": str(meta)},
            "features": {"hint_col": "timesfm3", "hint_residual": True}}


def _frames():
    tr = make_frame(n_days=1, issue_every_min=240).rename(columns={"timesfm_predict": "timesfm3"})
    on = tr.rename(columns=ONLINE)
    on["station"] = on["station_id"].map(lambda s: f"plant_guangfu{s}")
    return tr, on


def test_info_csv_header_auto_detected(tmp_path):
    m = StationMeta.from_csv(_info_csv(tmp_path))
    assert m.id_of("plant_guangfu1001") == 1001 and m.get(1002)["cap_mw"] == CAP[1002]
    assert m.get(1001)["lat"] == 30.0 and m.get(1001)["alt_m"] == 0.0 and not m.has_coords(2701)


def test_online_columns_give_identical_input(small_cfg, tmp_path):
    """大写协变量 + tsfm_predict + 只有站名（没有 station_id）→ 与训练口径逐值相同的模型输入。"""
    cfg = load_config(_cfg(small_cfg, _info_csv(tmp_path)))
    fb = FeatureBuilder(cfg, StationMeta.from_csv(cfg["dataset"]["station_meta"]))
    tr, on = _frames()

    def bd(df, drop=()):
        d = {c: (np.stack(df[c].to_numpy()) if isinstance(df[c].iloc[0], np.ndarray) else df[c].to_numpy()) for c in df.columns}
        d["timestamp_win"] = df["timestamp_win"].astype("int64").to_numpy() // 10**9
        d["station"] = list(df["station"]) if "station" in df else None
        return {k: v for k, v in d.items() if v is not None and k not in drop}

    a = fb.build(bd(tr), training=False)
    with pytest.warns(UserWarning, match="general_GHI"):
        b = fb.build(bd(on, drop=("station_id",)), training=False)
    for k in ("x", "static", "slot", "base"):
        np.testing.assert_array_equal(a[k], b[k])
    assert (b["x"][:, :, fb.idx["hint_missing"]] == 0).all()


def test_unified_inference_with_online_template(small_cfg, tmp_path):
    """按线上脚本的调用方式走 unified_inference：没有一行兜底成 None，预测与训练口径相同。"""
    from pipelines.pipeline_wrapper_inference import unified_inference
    m = TransformersAdaptedM8Model(_cfg(small_cfg, _info_csv(tmp_path))).eval()
    tr, on = _frames()
    feats_tr = ["observe_power", "general_ghi", "general_ghi_predict", "general_temp", "general_temp_predict", "timesfm3"]
    feats_on = ["observe_power", "general_GHI", "general_GHI_predict", "general_TEMP", "general_TEMP_predict", "tsfm_predict"]
    ref = unified_inference(m, tr, entity_cols=("timestamp_win", "station_id"), feature_cols=feats_tr)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = unified_inference(m, on, entity_cols=("timestamp_win", "station", "station_id"), feature_cols=feats_on)
    assert not [x for x in w if "兜底" in str(x.message)]
    assert out["TransformersAdaptedM8Model"].notna().all()
    np.testing.assert_allclose(np.stack(out["TransformersAdaptedM8Model"]), np.stack(ref["TransformersAdaptedM8Model"]), atol=1e-5)
    # 线上脚本 rolling（短期）路径只给 station 站名、不给 station_id：模型自己按站表映射
    out2 = unified_inference(m, on.drop(columns=["station_id"]), entity_cols=("timestamp_win", "station"), feature_cols=feats_on)
    np.testing.assert_allclose(np.stack(out2["TransformersAdaptedM8Model"]), np.stack(ref["TransformersAdaptedM8Model"]), atol=1e-5)


def test_unknown_station_name_raises(small_cfg, tmp_path):
    cfg = load_config(_cfg(small_cfg, _info_csv(tmp_path)))
    fb = FeatureBuilder(cfg, StationMeta.from_csv(cfg["dataset"]["station_meta"]))
    _, on = _frames()
    d = {c: np.stack(on[c].to_numpy()) for c in ["observe_power", "general_GHI", "general_GHI_predict", "general_TEMP",
                                                 "general_TEMP_predict", "tsfm_predict"]}
    d["timestamp_win"] = on["timestamp_win"].astype("int64").to_numpy() // 10**9
    d["station"] = ["plant_guangfu9999"] * len(on)
    with pytest.raises(ValueError, match="plant_guangfu9999"):
        fb.build(d, training=False)


def test_case_variants_ambiguous_raise(small_cfg):
    fb = FeatureBuilder(load_config(small_cfg), StationMeta.from_csv(small_cfg["dataset"]["station_meta"]))
    with pytest.raises(ValueError, match="大小写"):
        fb.resolve_col(["general_GHI", "general_Ghi"], "general_ghi")


def test_inference_missing_trained_cov_uses_cov_dropout_fallback(small_cfg, batch):
    """训练时有的 NWP 对推理时缺列：全部 NWP 通道置 0、cov_missing=1（和训练 cov_dropout 一样），而不是只把那一路悄悄置 0。"""
    cfg = load_config({**small_cfg, "features": {"cov_present": [["general_ghi", "general_ghi_predict"],
                                                                  ["general_temp", "general_temp_predict"]]}})
    fb = FeatureBuilder(cfg, StationMeta.from_csv(cfg["dataset"]["station_meta"]))
    bd = {k: v for k, v in batch["batch_dict"].items() if k != "general_temp_predict"}
    with pytest.warns(UserWarning, match="cov_dropout 兜底"):
        inp = fb.build(bd, training=False)
    assert (inp["x"][:, :, fb.idx["cov_missing"]] == 1).all()
    assert (inp["x"][:, :, fb.nwp_channels] == 0).all()


def test_report_inputs_lists_resolved_columns(small_cfg, tmp_path, capsys):
    m = TransformersAdaptedM8Model(_cfg(small_cfg, _info_csv(tmp_path)))
    _, on = _frames()
    present = report_inputs(m, list(on.columns))
    log = capsys.readouterr().out
    assert present == [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]]
    assert "general_ghi → 数据列 general_GHI ✓" in log and "timesfm3 → 数据列 tsfm_predict ✓" in log
    assert "GHI_solargis ✗" in log
    with pytest.raises(ValueError, match="timesfm3"):
        report_inputs(m, [c for c in on.columns if c != "tsfm_predict"], verbose=False)


def test_expand_parquets_recurses_into_date_partitions(tmp_path):
    for d in ("2026-06-01", "2026-06-02"):
        (tmp_path / "train" / f"date={d}").mkdir(parents=True)
        pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "train" / f"date={d}" / "data_0.parquet")
    (tmp_path / "train" / ".data").mkdir()
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "train" / ".data" / "tmp.parquet")
    files = expand_parquets([tmp_path / "train"])
    assert [f.split("date=")[1][:10] for f in files] == ["2026-06-01", "2026-06-02"]
