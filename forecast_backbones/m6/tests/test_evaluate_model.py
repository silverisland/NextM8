import functools
import numpy as np
import pandas as pd
from pipelines.transformer_dataset import UnifiedDataset
from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
from forecast_backbones.m6.transformer_trainer import evaluate_model
from forecast_backbones.m6.calibration import quantile_cols
from forecast_backbones.m6.decision import MODEL_LEVELS
from .conftest import FakeEvaluator, make_synthetic_frame, write_station_meta

def _ds(df):
    return UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[],
                          label_cols=["observe_power_future"], for_hf_trainer=True)

def test_evaluate_model_schema(tmp_path):
    df = make_synthetic_frame(n_days=1, issue_every_min=180)
    cfg = {"timesfm3": {"eager_init": False}, "dataset": {"station_meta": str(write_station_meta(tmp_path))},
           "calibration": {"enabled": False}, "eval": {"batch_size": 4}}
    m = TransformersAdaptedM6Model(cfg); m._backbone = FakeEvaluator(); m._backbone_device = "cpu"
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    res = evaluate_model(m, _ds(df), collate, m.cfg)
    assert len(res) == len(df)
    for c in ["timestamp_win", "station", "station_id", "cap", "observe_power_predict", "observe_power_future",
              "p_star", "median", "night", "elev_deg", "p_clear", "var_bin", "var_signal", "degraded"] + quantile_cols(MODEL_LEVELS):
        assert c in res.columns, c
    r = res.iloc[0]
    assert len(r["observe_power_predict"]) == 192 and len(r["q_0.5"]) == 192
    assert r["station"] in ("S1001", "S1002") and isinstance(r["timestamp_win"], pd.Timestamp)
    assert np.allclose(r["observe_power_future"], df.iloc[0]["observe_power_future"])
    assert set(res["station_id"].unique()) == {1001, 1002}

def test_evaluate_model_without_labels_gives_nan_truth(tmp_path):
    df = make_synthetic_frame(n_days=1, issue_every_min=360).drop(columns=["observe_power_future"])
    cfg = {"timesfm3": {"eager_init": False}, "dataset": {"station_meta": str(write_station_meta(tmp_path))},
           "calibration": {"enabled": False}}
    m = TransformersAdaptedM6Model(cfg); m._backbone = FakeEvaluator(); m._backbone_device = "cpu"
    ds = UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], for_hf_trainer=False)
    res = evaluate_model(m, ds, UnifiedDataset.collate_fn, m.cfg)
    assert np.isnan(res.iloc[0]["observe_power_future"]).all()
