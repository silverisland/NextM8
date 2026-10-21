"""线上口径（nanwang_online/eval_scripts_template/m7）：info.csv 站表、大写列名、站名 → station_id、
超短期 unified_inference / 短期 480 步 rolling_inference，决策层 p*。替身 backbone，不依赖 timestral。"""
import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.pipeline_wrapper_inference import unified_inference
from pipelines.rolling_wrapper_inference import rolling_inference
from forecast_backbones.m7.transformer_models import TransformersAdaptedM7Model
from forecast_backbones.m7.tests.test_transformer_trainer import L, H, _FakeForecaster

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "nanwang_online/eval_scripts_template/m7/single_model_scene_infer_m7.py"
CAPS = {"plant_guangfu1780": 178.87, "plant_guangfu1781": 129.2}
FEATS = ["observe_power", "general_GHI", "general_GHI_predict", "general_TEMP", "general_TEMP_predict"]


def _online_model(point="p_star"):
    m = TransformersAdaptedM7Model({
        "timestral": {"eager_init": False, "per_core_batch_size": 4},
        "dataset": {"station_meta": "forecast_backbones/m7/info.csv", "station_meta_cols": None},
        "decision": {"point": point, "levels": "model", "grid": 200, "tail": "flat", "floor_ratio": 0.2},
    })
    fake = _FakeForecaster()
    m._set_backbone(fake, "cpu")
    return m, fake


def _online_frame(model, fut_len=H, n=3):
    """线上轨迹表的样子：站名 plant_guangfu*、列名大写；功率幅度远超容量，测 cap 钳位。"""
    rows = []
    for st in CAPS:
        for i in range(n):
            rows.append({"timestamp_win": pd.Timestamp("2026-10-01 10:00") + pd.Timedelta(minutes=15 * i), "station": st,
                         "observe_power": np.linspace(0, 400, L).astype(np.float32),
                         "general_GHI": np.full(L, 3.0, np.float32), "general_GHI_predict": np.full(fut_len, 1.0, np.float32),
                         "general_TEMP": np.full(L, 20.0, np.float32), "general_TEMP_predict": np.full(fut_len, 25.0, np.float32)})
    df = pd.DataFrame(rows)
    df["station_id"] = df["station"].map(model.station_meta.id_of).astype("int64")      # 模板里的映射
    return df


def test_info_csv_station_meta_and_caps():
    m, _ = _online_model()
    assert {m.station_meta.id_of(s) for s in CAPS} == {1780, 1781}
    for st, cap in CAPS.items():
        assert m.station_cap(st) == pytest.approx(cap)


def test_uppercase_columns_are_fed_and_reported():
    m, fake = _online_model()
    lines, res = m.input_report(["station", "station_id"] + FEATS)
    assert res == {"observe_power": "observe_power", "general_ghi": "general_GHI", "general_ghi_predict": "general_GHI_predict",
                   "general_temp": "general_TEMP", "general_temp_predict": "general_TEMP_predict"}
    assert any("general_ghi → 数据列 general_GHI ✓" in ln for ln in lines) and any("p_star" in ln for ln in lines)
    df = _online_frame(m)
    with pytest.warns(UserWarning, match="general_GHI"):
        unified_inference(m, df, timestamp_cols=("timestamp_win",), entity_cols=("timestamp_win", "station", "station_id"),
                          feature_cols=FEATS)
    known = fake.calls[0]["known"]
    assert known is not None and known.shape[1:] == (2, L + H)              # 两对协变量都喂进去了
    np.testing.assert_allclose(known[:, 0, :L], 3.0); np.testing.assert_allclose(known[:, 1, L:], 25.0)


def test_mixed_case_duplicate_column_raises():
    m, _ = _online_model()
    with pytest.raises(ValueError, match="大小写变体"):
        m.resolve_col(["general_GHI", "general_ghi_x", "GENERAL_GHI"], "general_ghi")


def _cap_of(out, col):
    return np.array([CAPS[s] for s in out["station"]])[:, None], np.stack(out[col].to_numpy())


def test_ultra_short_unified_inference_pstar():
    m, _ = _online_model()
    df = _online_frame(m)
    col = type(m).__name__
    out = unified_inference(m, df, timestamp_cols=("timestamp_win",), entity_cols=("timestamp_win", "station", "station_id"),
                            feature_cols=FEATS)
    cap, p = _cap_of(out, col)
    assert p.shape == (len(df), H) and np.all(p <= cap + 1e-3) and np.all(p >= 0)
    med = np.stack(unified_inference(_online_model("median")[0], df, timestamp_cols=("timestamp_win",),
                                     entity_cols=("timestamp_win", "station", "station_id"), feature_cols=FEATS)[col].to_numpy())
    assert np.all(p <= med + 1e-3) and np.any(p < med - 1e-3)            # p* 压低端，比中位数低


def test_short_rolling_inference_480_pstar():
    m, _ = _online_model()
    df = _online_frame(m, fut_len=480)
    cfg = {"past_covariate": ["general_GHI", "general_TEMP"], "future_covariate": ["general_GHI_predict", "general_TEMP_predict"],
           "past_target": ["observe_power"], "future_target": [], "future_len": 480}
    out = rolling_inference(m, df, config=cfg, timestamp_cols=("timestamp_win",),
                            entity_cols=("timestamp_win", "station", "station_id"),
                            feature_cols=["general_GHI", "general_GHI_predict", "general_TEMP", "general_TEMP_predict", "observe_power"])
    cap, p = _cap_of(out, type(m).__name__)
    assert p.shape == (len(df), 480) and np.isfinite(p).all() and np.all(p <= cap + 1e-3)


def test_template_compiles_and_sets_pstar():
    py_compile.compile(str(TEMPLATE), doraise=True)
    src = TEMPLATE.read_text(encoding="utf-8")
    assert "M7_CONFIG = m7_dir / 'config/z1_zeroshot_pstar.yaml'" in src and "config_m7['decision']" not in src and "TransformersAdaptedM7Model" in src and "info.csv" in src
    assert "weights_path" not in src and "load_state_dict" not in src            # 零样本：不加载微调权重


def test_z1_is_configs_plus_pstar_and_info_csv():
    """z1_zeroshot_pstar.yaml = configs.yaml + decision.point p_star + 线上站表 info.csv，其余逐键相同；能直接建模型。"""
    import yaml
    cfg = ROOT / "forecast_backbones/m7/config"
    a = yaml.safe_load((cfg / "configs.yaml").read_text(encoding="utf-8"))
    b = yaml.safe_load((cfg / "z1_zeroshot_pstar.yaml").read_text(encoding="utf-8"))
    assert a["decision"]["point"] is None and b["decision"]["point"] == "p_star"
    assert b["dataset"]["station_meta"] == "forecast_backbones/m7/info.csv"
    a["decision"]["point"], a["dataset"]["station_meta"] = b["decision"]["point"], b["dataset"]["station_meta"]
    assert a == b
    m = TransformersAdaptedM7Model({**b, "timestral": {**b["timestral"], "eager_init": False}})
    assert m.decision_point == "p_star" and m.decision_levels == "model" and m.decision_grid == 200
    assert m.station_cap("plant_guangfu1780") == pytest.approx(178.87)
