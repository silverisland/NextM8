import json
import re
import zipfile

import numpy as np
import pandas as pd
import pytest
from .conftest import write_splits, write_station_meta, make_frame

MAIN = ("nanwang_short_D+1", "nanwang_ultra_short", "ods_short_rmse_acc", "ods_ultra_short_rmse_acc")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """很小的模型 + 两个单站测试文件（15 分钟起报，超短期要 16 个连续 lead）；1002 训练没见过。"""
    from forecast_backbones.m8.train import train_m8
    tmp = tmp_path_factory.mktemp("sr")
    meta = write_station_meta(tmp)
    splits = write_splits(tmp, n_train_days=3, n_val_days=2, train_stations=(1001,), val_stations=(1001, 1002))
    cfg = {"dataset": {"station_meta": str(meta)},
           "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv"], "pos_in_patch_dim": 8},
           "train": {"schedule": {"batch_size": 8, "max_steps": 4, "eval_every_steps": 2, "warmup_steps": 1, "logging_steps": 1, "progress_bar": False},
                     "early_stop": {"patience": 2}}, "eval": {"batch_size": 16}}
    out = tmp / "run"
    train_m8(cfg, splits_dir=splits, out_dir=out, tag="sr")
    tdir = tmp / "test"
    tdir.mkdir()
    for s in (1001, 1002):
        df = make_frame(n_days=3, stations=(s,), issue_every_min=15, seed=5 + s, start="2025-06-24")
        df.insert(1, "station", f"S{s}")
        df.to_parquet(tdir / f"station={s}.parquet", index=False)
    return out, tdir


def _sheets(xlsx):
    return set(re.findall(r'<sheet [^>]*name="([^"]+)"', zipfile.ZipFile(xlsx).read("xl/workbook.xml").decode("utf-8")))


def test_full_report_native(trained, tmp_path):
    from forecast_backbones.m8.station_report import evaluate_full
    from forecast_backbones.m8.evaluate import evaluate
    run, tdir = trained
    r = evaluate_full([run], [str(tdir)], out_dir=tmp_path / "f", engine="native", progress=False)
    ens = tmp_path / "f" / "metrics_ens"
    for st in ("S1001", "S1002"):
        assert _sheets(ens / f"{st}-metrics.xlsx") == {"ods_ultra_short", "ods_short", "ultra_short", "short", "48hours", "nanwang"}
        s = json.loads((ens / f"{st}-summary.json").read_text(encoding="utf-8"))
        assert s["station"] == st and s["rows"] > 0 and set(MAIN) <= set(s) and all(s[k] is not None for k in MAIN)
    tab = r["station_table"]
    assert list(tab["station"]) == ["S1001", "S1002"] and tab["seen"].tolist() == [True, False]
    assert np.isfinite(tab[list(MAIN)].astype(float).to_numpy()).all()
    assert (tab["nanwang_short_D+1"] <= 100).all() and (tab["ods_short_rmse_acc"] <= 1).all()
    fleet = r["summary"]
    assert fleet["engine"] == "native" and fleet["n_unseen_stations"] == 1 and fleet["n_stations"] == 2
    assert (tmp_path / "f" / "reports" / "summary.csv").exists() and (tmp_path / "f" / "reports" / "station_metrics.csv").exists()
    # 与舰队评估（10:00 行、同一决策）的南网 D+1 一致：单月时月度 mean 就是日均值
    e = evaluate([run], [str(tdir / "station=1001.parquet")], out_dir=tmp_path / "e", progress=False)
    assert abs(float(tab.loc[tab["station"] == "S1001", "nanwang_short_D+1"].iloc[0]) - e["summary"]["nanwang_short_D+1"]) < 0.01   # 月度表取 2 位小数


def test_full_report_unified_matches_native(trained, tmp_path, monkeypatch):
    """走真实的 pipelines.pipeline_wrapper_inference.unified_inference：单成员与 native 逐位一致；同成员集成恒等；决策覆盖生效。"""
    from forecast_backbones.m8 import station_report
    run, tdir = trained
    f = str(tdir / "station=1001.parquet")
    nat = station_report.evaluate_full([run], [f], out_dir=tmp_path / "n", engine="native", progress=False)["station_table"]
    uni = station_report.evaluate_full([run], [f], out_dir=tmp_path / "u", progress=False)                 # 默认 unified
    assert uni["summary"]["engine"] == "unified"
    for k in MAIN:
        assert abs(float(uni["station_table"][k].iloc[0]) - float(nat[k].iloc[0])) < 1e-3
    two = station_report.evaluate_full([run, run], [f], out_dir=tmp_path / "u2", engine="unified", progress=False)["station_table"]
    assert abs(float(two["nanwang_short_D+1"].iloc[0]) - float(nat["nanwang_short_D+1"].iloc[0])) < 1e-3
    med = station_report.evaluate_full([run], [f], out_dir=tmp_path / "m", engine="unified", point="median", progress=False)["station_table"]
    ps = station_report.evaluate_full([run], [f], out_dir=tmp_path / "p", engine="unified", point="p_star", progress=False)["station_table"]
    assert float(med["nanwang_short_D+1"].iloc[0]) != float(ps["nanwang_short_D+1"].iloc[0])
    # 带时区的时间列：两条路都按 UnifiedDataset 的约定（UTC 去时区）处理，结果仍一致
    tz = tmp_path / "tz.parquet"
    dft = pd.read_parquet(f)
    dft.assign(timestamp_win=pd.to_datetime(dft["timestamp_win"]).dt.tz_localize("Asia/Shanghai")).to_parquet(tz, index=False)
    ntz = station_report.evaluate_full([run], [str(tz)], out_dir=tmp_path / "ntz", engine="native", progress=False)["station_table"]
    utz = station_report.evaluate_full([run], [str(tz)], out_dir=tmp_path / "utz", engine="unified", progress=False)["station_table"]
    assert abs(float(utz["nanwang_short_D+1"].iloc[0]) - float(ntz["nanwang_short_D+1"].iloc[0])) < 1e-3
    # 线上接口不可用：unified 报错、auto 退 native
    monkeypatch.setattr(station_report, "unified_fn", lambda: None)
    with pytest.raises(ImportError, match="unified"):
        station_report.evaluate_full([run], [f], out_dir=tmp_path / "x", engine="unified", progress=False)
    assert station_report.evaluate_full([run], [f], out_dir=tmp_path / "a", engine="auto", progress=False)["summary"]["engine"] == "native"


def test_frame_with_station_id_variants(trained):
    from forecast_backbones.m8.station_report import frame_with_station_id
    from forecast_backbones.m8.evaluate import load_member
    run, tdir = trained
    m = load_member(run, "cpu")
    df = pd.read_parquet(tdir / "station=1002.parquet")
    a = frame_with_station_id(df.drop(columns=["station_id"]), m.cfg, m.station_meta)                       # 只有站名列
    b = frame_with_station_id(df.assign(station_id=df["station_id"].astype(str)), m.cfg, m.station_meta)   # 文本数字
    c = frame_with_station_id(df.drop(columns=["station"]), m.cfg, m.station_meta)                          # 没有站名列
    assert (a["station_id"] == 1002).all() and (b["station_id"] == 1002).all() and (c["station"] == "S1002").all()
    assert str(b["station_id"].dtype) == "int64"


def test_cli_main(trained, tmp_path):
    from forecast_backbones.m8.station_report import main
    run, tdir = trained
    main(["--runs", str(run), "--test", str(tdir / "station=1001.parquet"), "--out", str(tmp_path / "cli"), "--engine", "native",
          "--no-progress", "--tag", "t"])
    s = pd.read_csv(tmp_path / "cli" / "reports" / "summary.csv").iloc[0]
    assert s["tag"] == "t" and s["engine"] == "native" and np.isfinite(s["nanwang_ultra_short"])
