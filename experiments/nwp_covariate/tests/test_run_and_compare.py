import json

import numpy as np
import pandas as pd
import pytest

import compare
from run_experiments import m5_config, resolve_experiment, run_experiment


def test_resolve_extends_add_and_ablation(base_cfg):
    c3a = resolve_experiment(base_cfg, "C3a")
    assert c3a["past_future"] == [["ghi_cons", "ghi_cons_predict"], ["t2m_ec", "t2m_ec_predict"]]
    c4 = resolve_experiment(base_cfg, "C4_lcc")
    assert c4["past_future"] == c3a["past_future"] + [["lcc_ec", "lcc_ec_predict"]]
    assert c4["parent"] == "C3a"
    c3e = resolve_experiment(base_cfg, "C3e")
    assert c3e["past_future"][0] == ["ghi_obs", "ghi_cons_predict"]
    c3f = resolve_experiment(base_cfg, "C3f")
    assert c3f["past_only"] == ["ghi_obs", "temp_obs"] and c3f["past_future"] == c3a["past_future"]
    abl = resolve_experiment(base_cfg, "C5_drop_ec")
    assert abl["ablation"] == {"drop_sources": ["ec"], "apply_fallback": True}
    assert abl["past_future"] == c3a["past_future"]          # C5 extends C3a
    with pytest.raises(KeyError):
        resolve_experiment(base_cfg, "nope")


def test_m5_config_only_touches_timesfm3_cols(base_cfg):
    exp = resolve_experiment(base_cfg, "C1")
    mc = m5_config(base_cfg, exp)
    assert mc["run"]["model_name"] == "m5"
    assert mc["timesfm3"]["past_future_cols"] == [["ghi_solargis", "ghi_solargis_predict"], ["temp_solargis", "temp_solargis_predict"]]
    assert mc["timesfm3"]["nan_policy"] == "raise" and mc["timesfm3"]["pred_len"] == 192


def test_run_mock_and_compare_end_to_end(ds_dir, tmp_path):
    cfg, raw = ds_dir
    cfg["run"]["out_dir"] = str(tmp_path / "runs")
    cfg["evaluation"]["paired_test"]["bootstrap"] = 200
    s0 = run_experiment(cfg, "C0", "ultra_short", backend="mock")
    s1 = run_experiment(cfg, "C1", "ultra_short", backend="mock")
    s3 = run_experiment(cfg, "C3c", "ultra_short", backend="mock")
    sa = run_experiment(cfg, "C5_drop_ec", "ultra_short", backend="mock")
    for s in (s0, s1, s3, sa):
        assert not s.get("skipped"), s
        assert s["daily_days"]["nanwang_short_D+1"] == 3      # 3 个 10:00 起报 -> 3 个 D+1 日指标
        assert s["fleet_mean"]["nanwang_short_D+1"] is not None
    # 消融用了独立特征库；配置文件写在实验目录
    assert sa["feature_store"].endswith("abl_C5_drop_ec")
    mc = json.load(open(tmp_path / "runs" / "C3c" / "summary.json"))
    assert mc["past_future_cols"] == [["p_phys", "p_phys_predict"]]
    assert (tmp_path / "runs" / "C3c" / "config" / "configs.yaml").exists()
    # 物理代理（有 GHI 信息）应明显好于纯持平的 C0
    assert s3["fleet_mean"]["nanwang_short_D+1"] > s0["fleet_mean"]["nanwang_short_D+1"] + 5
    # 幂等：再跑一次直接 SKIP
    assert run_experiment(cfg, "C1", "ultra_short", backend="mock")["id"] == "C1"
    # 主指标按类型选：ultra_short 的主指标没有 15min 起报 -> 空；这里改成 short 口径做比较
    cfg["evaluation"]["primary_by_type"]["ultra_short"] = "nanwang_short_D+1"
    tab = compare.run(cfg, "ultra_short", ids=["C0", "C1", "C3c", "C5_drop_ec"])
    assert set(tab["id"]) == {"C0", "C3c", "C5_drop_ec"}
    r = tab.set_index("id")
    assert r.loc["C0", "primary_mean_gain"] < 0
    assert r.loc["C5_drop_ec", "vs"] == "C3a" or r.loc["C5_drop_ec", "vs"] == "C1"   # parent C5 未跑 -> 回退 baseline
    assert (tmp_path / "runs" / "report.md").exists() and (tmp_path / "runs" / "comparison.csv").exists()
    txt = (tmp_path / "runs" / "report.md").read_text(encoding="utf-8")
    assert "baseline = **C1**" in txt
