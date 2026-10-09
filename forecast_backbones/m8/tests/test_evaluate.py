import numpy as np
import pandas as pd
import pytest
from .conftest import write_splits, write_station_meta


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """训一个很小的模型，供评估测试复用。"""
    from forecast_backbones.m8.train import train_m8
    tmp = tmp_path_factory.mktemp("ev")
    meta = write_station_meta(tmp)
    splits = write_splits(tmp, n_train_days=3, n_val_days=2, train_stations=(1001,), val_stations=(1001, 1002))
    cfg = {"dataset": {"station_meta": str(meta)},
           "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv"], "pos_in_patch_dim": 8},
           "train": {"schedule": {"batch_size": 8, "max_steps": 4, "eval_every_steps": 2, "warmup_steps": 1, "logging_steps": 1, "progress_bar": False},
                     "early_stop": {"patience": 2}}, "eval": {"batch_size": 16}}
    out = tmp / "run"
    train_m8(cfg, splits_dir=splits, out_dir=out, tag="ev")
    return out, splits / "valid.parquet"


def test_single_run_matches_training_eval(trained, tmp_path):
    """同一权重、同一验证集、同一决策规则 → 评估脚本的分数和训练终评一致。"""
    from forecast_backbones.m8.evaluate import evaluate
    run, test = trained
    r = evaluate([run], [str(test)], out_dir=tmp_path / "e", progress=False)
    s = r["summary"]
    tr = pd.read_csv(run / "reports" / "summary.csv").iloc[0]
    assert s["n_members"] == 1 and s["point_rule"] == "median"
    assert abs(s["nanwang_short_D+1"] - tr["nanwang_short_D+1"]) < 1e-3     # 不同 batch 组成下的浮点噪声 ~1e-6
    assert abs(s["nrmse_pct"] - tr["nrmse_pct"]) < 1e-3
    assert s["n_unseen_stations"] == 1 and np.isfinite(s["nanwang_unseen"])
    assert (tmp_path / "e" / "result_test.parquet").exists() and (tmp_path / "e" / "reports" / "station_table.csv").exists()


def test_decision_override_without_retrain(trained, tmp_path):
    from forecast_backbones.m8.evaluate import evaluate
    import json
    run, test = trained
    d2 = "forecast_backbones/m8/config/variants/d2_pstar_point.yaml"
    r = evaluate([run], [str(test)], out_dir=tmp_path / "p", decision_yaml=d2, progress=False)
    assert r["summary"]["point_rule"].startswith("p_star[")
    dec = json.loads((tmp_path / "p" / "reports" / "decision.json").read_text(encoding="utf-8"))
    assert dec["point"]["default"] == "p_star" and len(dec["point"]["by_step"]) == 2
    res = r["result"]
    assert np.allclose(np.stack(res["observe_power_predict"])[:, 55:151], np.stack(res["p_star"])[:, 55:151], atol=1e-4)
    r2 = evaluate([run], [str(test)], out_dir=tmp_path / "m", point="median", progress=False)
    assert r2["summary"]["point_rule"] == "median"


def test_ensemble_of_identical_members_equals_single(trained, tmp_path):
    from forecast_backbones.m8.evaluate import evaluate
    run, test = trained
    one = evaluate([run], [str(test)], out_dir=tmp_path / "one", progress=False)
    two = evaluate([run, run], [str(test)], out_dir=tmp_path / "two", progress=False)
    assert two["summary"]["n_members"] == 2
    assert abs(two["summary"]["nanwang_short_D+1"] - one["summary"]["nanwang_short_D+1"]) < 1e-6
    mt = two["members"]
    assert list(mt["member"]) == [run.name, run.name, "ensemble"]
    assert np.allclose(mt["nanwang_short_D+1"].to_numpy(), one["summary"]["nanwang_short_D+1"])
    assert (tmp_path / "two" / "reports" / "members.csv").exists()
    assert "predict_m0" in two["result"].columns and "predict_m0" not in one["result"].columns


def test_cli_main(trained, tmp_path):
    from forecast_backbones.m8.evaluate import main
    run, test = trained
    main(["--runs", str(run), "--test", str(test), "--out", str(tmp_path / "cli"), "--point", "p_star", "--no-progress", "--tag", "t"])
    s = pd.read_csv(tmp_path / "cli" / "reports" / "summary.csv").iloc[0]
    assert s["tag"] == "t" and s["point_rule"] == "p_star"
