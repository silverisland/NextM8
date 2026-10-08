import json
import numpy as np
import pandas as pd
import pytest
from .conftest import write_splits, write_station_meta


@pytest.mark.parametrize("metric", ["nanwang_after_pstar", "nrmse"])
def test_train_m8_end_to_end(tmp_path, metric):
    from forecast_backbones.m8.train import train_m8
    meta = write_station_meta(tmp_path)
    splits = write_splits(tmp_path, n_train_days=3, n_val_days=2)
    cfg = {"dataset": {"station_meta": str(meta)},
           "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv", "deltanet"], "pos_in_patch_dim": 8},
           "train": {"schedule": {"batch_size": 8, "max_steps": 6, "eval_every_steps": 3, "warmup_steps": 1, "logging_steps": 1, "progress_bar": False},
                     "early_stop": {"metric": metric, "patience": 2}},
           "eval": {"batch_size": 16}}
    out = tmp_path / "run"
    r = train_m8(cfg, splits_dir=splits, out_dir=out, tag="smoke")
    assert (out / "model.safetensors").exists() and (out / "model_config.yaml").exists()
    log = json.loads((out / "train_log.json").read_text())
    assert len(log["history"]) >= 2 and log["best_step"] is not None
    s = pd.read_csv(out / "reports" / "summary.csv")
    assert s.loc[0, "tag"] == "smoke" and s.loc[0, "nanwang_short_D+1"] == s.loc[0, "nanwang_short_D+1"]   # 非 NaN
    assert (out / "result_val.parquet").exists() and (out / "reports" / "station_table.csv").exists()
    assert not (out / ".fresh" / "model.safetensors").exists()


def test_train_m8_reports_unseen_station(tmp_path):
    """验证集里有训练集没有的站 → model_config 记录 known_stations，summary 分开已见 / 未见站分数。"""
    import yaml
    from forecast_backbones.m8.train import train_m8
    meta = write_station_meta(tmp_path)
    splits = write_splits(tmp_path, n_train_days=3, n_val_days=2, train_stations=(1001,), val_stations=(1001, 1002))
    cfg = {"dataset": {"station_meta": str(meta)},
           "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv"], "pos_in_patch_dim": 8},
           "train": {"schedule": {"batch_size": 8, "max_steps": 4, "eval_every_steps": 2, "warmup_steps": 1, "logging_steps": 1, "progress_bar": False},
                     "early_stop": {"patience": 2}}, "eval": {"batch_size": 16}}
    out = tmp_path / "run"
    r = train_m8(cfg, splits_dir=splits, out_dir=out, tag="unseen")
    saved = yaml.safe_load((out / "model_config.yaml").read_text(encoding="utf-8"))
    assert saved["dataset"]["known_stations"] == [1001]
    s = pd.read_csv(out / "reports" / "summary.csv").iloc[0]
    assert s["n_unseen_stations"] == 1 and np.isfinite(s["nanwang_unseen"]) and np.isfinite(s["nanwang_seen"])
    tab = pd.read_csv(out / "reports" / "station_table.csv")
    assert set(tab.loc[tab["seen"] == False, "station_id"]) == {1002}
    assert all("nanwang_unseen" in h for h in r["history"])
