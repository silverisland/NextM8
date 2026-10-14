import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("timesfm3")
from .conftest import full_cfg, make_frame, write_splits

M5 = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runs(tmp_path_factory, tiny_ckpt_dir):
    """一个 LoRA run、一个零样本 run（验证集里 1002 是训练没见过的站）。"""
    from forecast_backbones.m5.train import train_m5
    tmp = tmp_path_factory.mktemp("ev")
    splits = write_splits(tmp, n_train_days=2, n_val_days=2, train_stations=(1001,), val_stations=(1001, 1002))
    train_m5(full_cfg(tmp, tiny_ckpt_dir, train={"schedule": {"lr": 1.0e-2}}), splits_dir=splits, out_dir=tmp / "l1", tag="l1")
    train_m5(full_cfg(tmp, tiny_ckpt_dir, finetune={"enabled": False}), splits_dir=splits, out_dir=tmp / "z0", tag="z0")
    return tmp, splits


def test_evaluate_matches_training_eval_for_lora_and_zero_shot(runs, tmp_path):
    from forecast_backbones.m5.evaluate import evaluate
    tmp, splits = runs
    for name in ("l1", "z0"):
        r = evaluate([tmp / name], [str(splits / "valid.parquet")], out_dir=tmp_path / name, progress=False)
        tr = pd.read_csv(tmp / name / "reports" / "summary.csv").iloc[0]
        assert abs(r["summary"]["nanwang_short_D+1"] - tr["nanwang_short_D+1"]) < 1e-3, name
        assert abs(r["summary"]["nrmse_pct"] - tr["nrmse_pct"]) < 1e-3, name
        assert r["summary"]["n_unseen_stations"] == 1 and r["summary"]["point_rule"] == "median"


def test_ensemble_and_decision_override(runs, tmp_path):
    from forecast_backbones.m5.evaluate import evaluate
    tmp, splits = runs
    e = evaluate([tmp / "l1", tmp / "z0"], [str(splits / "valid.parquet")], out_dir=tmp_path / "ens",
                 decision_yaml=str(M5 / "config" / "variants" / "g2_cs_ceiling.yaml"), progress=False)
    assert e["summary"]["n_members"] == 2 and len(e["members"]) == 3 and np.isfinite(e["summary"]["nanwang_short_D+1"])
    dec = json.loads((tmp_path / "ens" / "reports" / "decision.json").read_text(encoding="utf-8"))
    assert dec["physics"]["clear_sky_cap_ratio"] == 1.05
    p = evaluate([tmp / "l1"], [str(splits / "valid.parquet")], out_dir=tmp_path / "p",
                 decision_yaml=str(M5 / "config" / "variants" / "d2_pstar_point.yaml"), progress=False)
    assert p["summary"]["point_rule"] == "p_star"


def test_lora_run_without_weights_is_rejected(runs, tmp_path):
    from forecast_backbones.m5.evaluate import evaluate
    tmp, splits = runs
    bad = tmp_path / "bad"
    shutil.copytree(tmp / "l1", bad)
    (bad / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="LoRA"):
        evaluate([bad], [str(splits / "valid.parquet")], out_dir=tmp_path / "x", progress=False)


def test_station_report_native(runs, tmp_path):
    from forecast_backbones.m5.station_report import evaluate_full
    tmp, _ = runs
    tdir = tmp_path / "test"; tdir.mkdir()
    for s in (1001, 1002):
        df = make_frame(n_days=3, stations=(s,), issue_every_min=15, seed=5 + s, start="2025-06-24")
        df.insert(1, "station", f"S{s}")
        df.to_parquet(tdir / f"station={s}.parquet", index=False)
    r = evaluate_full([tmp / "l1"], [str(tdir)], out_dir=tmp_path / "f", engine="native", progress=False)
    tab = r["station_table"]
    assert list(tab["station"]) == ["S1001", "S1002"] and tab["seen"].tolist() == [True, False]
    assert np.isfinite(tab[["nanwang_short_D+1", "nanwang_ultra_short"]].astype(float).to_numpy()).all()
    assert (tmp_path / "f" / "metrics_ens" / "S1001-metrics.xlsx").exists()
