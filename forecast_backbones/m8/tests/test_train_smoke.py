import json
import numpy as np
import pandas as pd
import pytest
from .conftest import write_splits, write_station_meta


@pytest.mark.parametrize("metric", ["nanwang_after_pstar", "nrmse", "nanwang_ultra_proxy"])
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
    assert np.isfinite(s.loc[0, "nanwang_ultra_proxy"]) and 0 <= s.loc[0, "nanwang_ultra_proxy"] <= 100
    assert (out / "result_val.parquet").exists() and (out / "reports" / "station_table.csv").exists()
    assert not (out / ".fresh" / "model.safetensors").exists()
    if metric == "nanwang_ultra_proxy":
        h = log["history"][0]
        assert metric in h and np.isfinite(h[metric]) and "nrmse" in h


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


def test_train_m8_refit(tmp_path):
    """全量重训：train ∪ valid 全当训练行，不验证不早停，固定步数后存最终权重；评估脚本能直接加载。"""
    import yaml
    from forecast_backbones.m8.train import train_m8
    from forecast_backbones.m8.evaluate import evaluate
    meta = write_station_meta(tmp_path)
    splits = write_splits(tmp_path, n_train_days=3, n_val_days=2, train_stations=(1001,), val_stations=(1001, 1002))
    cfg = {"dataset": {"station_meta": str(meta)},
           "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv"], "pos_in_patch_dim": 8},
           "train": {"schedule": {"batch_size": 8, "max_steps": 4, "eval_every_steps": 2, "warmup_steps": 1, "logging_steps": 1, "progress_bar": False},
                     "early_stop": {"patience": 2}}, "eval": {"batch_size": 16}}
    with pytest.raises(ValueError, match="max_steps"):
        train_m8(cfg, splits_dir=splits, out_dir=tmp_path / "bad", tag="bad", max_steps=0, refit=True)
    out = tmp_path / "refit"
    r = train_m8(cfg, splits_dir=splits, out_dir=out, tag="refit", refit=True)
    assert (out / "model.safetensors").exists() and not (out / "result_val.parquet").exists()
    saved = yaml.safe_load((out / "model_config.yaml").read_text(encoding="utf-8"))
    assert saved["dataset"]["known_stations"] == [1001, 1002]                    # valid 里的 1002 并进了训练
    s = pd.read_csv(out / "reports" / "summary.csv").iloc[0]
    assert s["best_step"] == 4 and s["n_val_rows"] == 0 and np.isnan(s["nanwang_short_D+1"]) and s["n_train_rows"] > 0
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert log["refit"] is True and log["best_step"] == 4 and log["history"] and all("train_loss" in h for h in log["history"])
    assert r["station_table"] is None and json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))["refit"] is True
    e = evaluate([out], [str(splits / "valid.parquet")], out_dir=tmp_path / "ev", progress=False)
    assert e["summary"]["n_unseen_stations"] == 0 and np.isfinite(e["summary"]["nanwang_short_D+1"])
    # valid.parquet 为 0 行的 refit 切分：不加 refit 报错提示；加了照常训
    empty = tmp_path / "empty"; empty.mkdir()
    pd.read_parquet(splits / "train.parquet").to_parquet(empty / "train.parquet", index=False)
    pd.read_parquet(splits / "valid.parquet").iloc[:0].to_parquet(empty / "valid.parquet", index=False)
    (empty / "split_manifest.json").write_text((splits / "split_manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="refit"):
        train_m8(cfg, splits_dir=empty, out_dir=tmp_path / "e1", tag="e1")
    train_m8(cfg, splits_dir=empty, out_dir=tmp_path / "e2", tag="e2", refit=True)
    assert (tmp_path / "e2" / "model.safetensors").exists()
    assert yaml.safe_load((tmp_path / "e2" / "model_config.yaml").read_text(encoding="utf-8"))["dataset"]["known_stations"] == [1001]


def test_train_m8_full_from_ready_parquets(tmp_path):
    """全量训练（2026-10-14）：训练 parquet 已经切好、一站一个文件时，--train-parquet 直接吃文件夹 / 通配符 / 文件列表，
    不走 time_splits、不切块、不剔除任何行、没有验证集，固定步数后存 checkpoint。站列是站名文本也能按站表映射。"""
    import pyarrow.parquet as pq
    from forecast_backbones.m8.train import train_m8
    from .conftest import make_frame
    meta = write_station_meta(tmp_path)
    d = tmp_path / "trainset"; d.mkdir()
    n_rows = 0
    for s in (1001, 1002):
        df = make_frame(n_days=2, stations=(s,), issue_every_min=120, seed=s)
        df = df.drop(columns=["station_id"]).assign(station=f"S{s}")          # 真实数据：站列是站名文本
        df.to_parquet(d / f"S{s}.parquet", index=False); n_rows += len(df)
    cfg = {"dataset": {"station_meta": str(meta)},
           "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv", "deltanet"], "pos_in_patch_dim": 8},
           "train": {"schedule": {"batch_size": 8, "max_steps": 4, "warmup_steps": 1, "logging_steps": 1, "progress_bar": False}},
           "eval": {"batch_size": 16}}
    out = tmp_path / "run"
    r = train_m8(cfg, train_parquet=[str(d)], out_dir=out, tag="full")           # 目录 → 其下全部 parquet
    assert (out / "model.safetensors").exists() and (out / "model_config.yaml").exists()
    man = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert man["refit"] is True and man["splits_dir"] is None and len(man["train_parquet"]) == 2
    assert man["counts"] == {"train": n_rows, "val": 0}                           # 一行不少、没有验证行
    s = pd.read_csv(out / "reports" / "summary.csv")
    assert s.loc[0, "tag"] == "full" and np.isnan(s.loc[0, "nanwang_short_D+1"])     # 没有验证集，分数列 NaN
    import yaml
    assert yaml.safe_load((out / "model_config.yaml").read_text(encoding="utf-8"))["dataset"]["known_stations"] == [1001, 1002]
    # 通配符 / 文件列表同样可以；splits_dir 与 train_parquet 二选一
    r2 = train_m8(cfg, train_parquet=[str(d / "S1001.parquet"), str(d / "S1002.parquet")], out_dir=tmp_path / "run2", tag="full2")
    assert (tmp_path / "run2" / "model.safetensors").exists()
    with pytest.raises(ValueError, match="二选一"):
        train_m8(cfg, out_dir=tmp_path / "run3")
    with pytest.raises(ValueError, match="二选一"):
        train_m8(cfg, splits_dir=str(d), train_parquet=[str(d)], out_dir=tmp_path / "run4")
