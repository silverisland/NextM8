import json

import numpy as np
import pandas as pd
import pytest
import yaml

from pipelines.time_splits import (SplitPlan, StationSplitPlan, verify_leak_free, LEGACY_KEYS, write_splits,
                                   load_split_dict, load_test_starts, main, suggest_val_blocks, read_parquets)

SPLIT = {"train_start": None,
         "val_blocks": [["2025-03-10", "2025-03-24"], ["2025-07-16", "2025-07-30"]],
         "test_start": "2025-10-01", "test_end": None,
         "cal_days": 45, "purge_days": 2, "embargo_days": 7}


def _grid(start="2025-01-01", end="2025-12-01"):
    return pd.date_range(start, end, freq="15min", inclusive="left").to_numpy()


def test_from_dict_builds_intervals_in_days():
    p = SplitPlan.from_dict(SPLIT)
    assert p.val_blocks == ((pd.Timestamp("2025-03-10"), pd.Timestamp("2025-03-24")),
                            (pd.Timestamp("2025-07-16"), pd.Timestamp("2025-07-30")))
    assert p.cal_interval == (pd.Timestamp("2025-08-15"), pd.Timestamp("2025-09-29"))   # test_start − 2d − 45d
    ex = p.train_exclusions
    assert ex[0] == (pd.Timestamp("2025-03-08"), pd.Timestamp("2025-04-02"))            # −purge, +purge+embargo
    assert ex[1] == (pd.Timestamp("2025-07-14"), pd.Timestamp("2025-08-08"))
    assert ex[2] == (pd.Timestamp("2025-08-13"), None)                                   # cal_start − purge → +inf
    assert p.inputs_disjoint_after is True                                               # 9 days × 96 ≥ 864
    assert p.coverage_note is None                                                       # Q1 and Q3
    d = p.to_dict()
    assert d["val_blocks"] == [["2025-03-10", "2025-03-24"], ["2025-07-16", "2025-07-30"]]
    assert d["cal"] == ["2025-08-15", "2025-09-29"] and d["test_start"] == "2025-10-01" and d["test_end"] is None


def test_assign_roles_and_counts_on_full_grid():
    p = SplitPlan.from_dict(SPLIT)
    ts = _grid()
    role, block = p.assign(ts)
    n = lambda r: int((role == r).sum())
    assert n("val") == 28 * 96 and set(block[role == "val"]) == {0, 1}
    assert int(((role == "val") & (block == 0)).sum()) == 14 * 96
    assert n("cal") == 45 * 96
    assert n("test") == 61 * 96                                                          # 10-01 … 12-01
    # train = everything up to 08-13 minus the two exclusion windows (25 days each)
    days_before_cal_purge = (pd.Timestamp("2025-08-13") - pd.Timestamp("2025-01-01")).days
    assert n("train") == (days_before_cal_purge - 2 * 25) * 96
    assert n("dropped") == len(ts) - n("train") - n("val") - n("cal") - n("test")
    assert block[role != "val"].max() == -1
    verify_leak_free(ts[role == "train"], [ts[(role == "val") & (block == i)] for i in range(2)], ts[role == "cal"], p)


def test_train_start_and_shift_days_apply_to_everything():
    p = SplitPlan.from_dict({**SPLIT, "train_start": "2025-02-01"}, shift_days=3)
    assert p.train_start == pd.Timestamp("2025-02-04")
    assert p.val_blocks[0] == (pd.Timestamp("2025-03-13"), pd.Timestamp("2025-03-27"))
    assert p.test_start == pd.Timestamp("2025-10-04")
    role, _ = p.assign(_grid())
    assert (role[_grid() < np.datetime64("2025-02-04")] == "dropped").all()


def test_nat_rows_are_dropped():
    ts = _grid("2025-01-01", "2025-01-03").astype("datetime64[ns]")
    ts[5] = np.datetime64("NaT")
    role, _ = SplitPlan.from_dict(SPLIT).assign(ts)
    assert role[5] == "dropped" and (role[[0, 1, 2]] == "train").all()


@pytest.mark.parametrize("bad, msg", [
    ({**SPLIT, "purge_days": 1}, "purge_days"),
    ({**SPLIT, "embargo_days": 6}, "embargo_days"),                                     # 块后共 2+6=8 天 < L+H=9 天
    ({**SPLIT, "val_blocks": [["2025-03-10", "2025-03-11"]]}, "不足 2 天"),
    ({**SPLIT, "val_blocks": [["2025-03-10", "2025-03-24"], ["2025-03-20", "2025-04-01"]]}, "重叠"),
    ({**SPLIT, "val_blocks": [["2025-03-10 06:00", "2025-03-24"]]}, "00:00"),
    ({**SPLIT, "val_blocks": [["2025-09-10", "2025-09-24"]]}, "purge"),                  # 09-24 + 2d > cal_start 08-15
    ({**SPLIT, "test_end": "2025-09-01"}, "test_end"),
    ({**SPLIT, "val_blocks": []}, "val_blocks"),
    ({**SPLIT, "cal_days": -1}, "cal_days"),
    ({**SPLIT, "train_start": "2025-04-01"}, "train_start"),                             # 第一个验证块 03-10 早于 train_start
])
def test_from_dict_rejects_bad_plans(bad, msg):
    with pytest.raises(ValueError, match=msg):
        SplitPlan.from_dict(bad)


def test_from_dict_rejects_legacy_keys():
    legacy = {"train_end": "2025-05-31", "val_start": "2025-07-16", "val_end": "2025-08-31",
              "test_start": "2025-09-01", "cal_days": 45}
    with pytest.raises(ValueError) as e:
        SplitPlan.from_dict(legacy)
    assert "train_end" in str(e.value) and "val_blocks" in str(e.value)
    assert "forecast_backbones/m6/README.md" in str(e.value)                             # 指向仓库里真实存在的文档
    assert set(LEGACY_KEYS) == {"train_end", "val_start", "val_end", "rolling_folds"}


def test_no_cal_when_cal_days_zero():
    p = SplitPlan.from_dict({**SPLIT, "cal_days": 0})
    assert p.cal_interval is None
    assert p.train_exclusions[-1] == (pd.Timestamp("2025-09-29"), None)                  # test_start − purge
    role, _ = p.assign(_grid())
    assert int((role == "cal").sum()) == 0
    verify_leak_free(_grid()[role == "train"], [], None, p)


def test_coverage_note_when_single_quarter():
    p = SplitPlan.from_dict({**SPLIT, "val_blocks": [["2025-07-01", "2025-07-15"], ["2025-08-01", "2025-08-08"]]})
    assert p.coverage_note is not None and "季度" in p.coverage_note


def test_verify_leak_free_catches_target_overlap():
    p = SplitPlan.from_dict(SPLIT)
    ts = _grid()
    role, block = p.assign(ts)
    val0 = ts[(role == "val") & (block == 0)]
    leaky_train = np.append(ts[role == "train"], val0[0] - np.timedelta64(24, "h"))       # 1 day before block: label overlaps
    with pytest.raises(AssertionError, match="验证块 0"):
        verify_leak_free(leaky_train, [val0, ts[(role == "val") & (block == 1)]], ts[role == "cal"], p)
    leaky_after = np.append(ts[role == "train"], val0[-1] + np.timedelta64(24, "h"))      # 1 day after block: label overlaps
    with pytest.raises(AssertionError, match="验证块 0"):
        verify_leak_free(leaky_after, [val0, ts[(role == "val") & (block == 1)]], ts[role == "cal"], p)
    cal = ts[role == "cal"]
    leaky_cal = np.append(ts[role == "train"], cal[0] - np.timedelta64(24, "h"))
    with pytest.raises(AssertionError, match="校准段"):
        verify_leak_free(leaky_cal, [val0, ts[(role == "val") & (block == 1)]], cal, p)


def test_verify_leak_free_catches_context_overlap_after_block():
    """块后第 3 天起报的训练行：标签 (t, t+H] 与块标签已不重叠，但它 7 天的上下文仍盖住块的标签段。"""
    p = SplitPlan.from_dict(SPLIT)
    ts = _grid()
    role, block = p.assign(ts)
    val0, val1 = ts[(role == "val") & (block == 0)], ts[(role == "val") & (block == 1)]
    cal, tr = ts[role == "cal"], ts[role == "train"]
    leaky_ctx = np.append(tr, val0[-1] + np.timedelta64(3, "D"))
    with pytest.raises(AssertionError, match="上下文"):
        verify_leak_free(leaky_ctx, [val0, val1], cal, p)
    with pytest.raises(AssertionError, match="验证块 0"):
        verify_leak_free(leaky_ctx, [val0, val1], cal, p)
    verify_leak_free(np.append(tr, val0[-1] + np.timedelta64(9, "D")), [val0, val1], cal, p)   # 块后满 9 天即干净


def test_verify_leak_free_checks_test_bound_even_when_cal_rows_are_empty():
    """校准段行数为 0（cal 区间落在数据之外）时，测试段那一侧的边界检查不能被跳过。"""
    p = SplitPlan.from_dict(SPLIT)
    ts = _grid()
    role, block = p.assign(ts)
    blocks = [ts[(role == "val") & (block == i)] for i in range(2)]
    empty_cal = ts[role == "cal"][:0]
    leaky_test = np.append(ts[role == "train"], np.datetime64("2025-09-30T00:00:00"))     # test_start − 1 天
    with pytest.raises(AssertionError, match="测试段"):
        verify_leak_free(leaky_test, blocks, empty_cal, p)
    verify_leak_free(ts[role == "train"], blocks, empty_cal, p)


def _frame(start="2025-01-01", end="2025-12-01", stations=("A", "B")):
    ts = pd.date_range(start, end, freq="15min", inclusive="left")
    rows = []
    for s in stations:
        rows.append(pd.DataFrame({"timestamp_win": ts, "station": s,
                                  "observe_power": [np.zeros(4, np.float32)] * len(ts),
                                  "observe_power_future": [np.zeros(2, np.float32)] * len(ts)}))
    df = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=0).reset_index(drop=True)   # 乱序输入
    return df


def test_write_splits_files_and_manifest(tmp_path):
    df = _frame()
    plan = SplitPlan.from_dict(SPLIT)
    man = write_splits(df, plan, tmp_path / "out", source="raw.parquet")
    out = tmp_path / "out"
    for f in ("train.parquet", "valid.parquet", "cal.parquet", "test.parquet", "split_manifest.json"):
        assert (out / f).exists()
    tr, va, ca, te = (pd.read_parquet(out / f) for f in ("train.parquet", "valid.parquet", "cal.parquet", "test.parquet"))
    assert len(va) == 2 * 28 * 96 and set(va["split_block"]) == {0, 1}
    assert "split_block" not in tr.columns and "_role" not in tr.columns
    assert len(ca) == 2 * 45 * 96 and len(te) == 2 * 61 * 96
    assert tr["timestamp_win"].is_monotonic_increasing                                   # 按时间排序落盘
    assert tr["timestamp_win"].max() < pd.Timestamp("2025-08-13")
    m = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert m == man
    assert m["plan"]["val_blocks"] == SPLIT["val_blocks"] and m["source"] == "raw.parquet"
    assert m["counts"]["train"] == len(tr) and m["counts"]["val"] == len(va) and m["counts"]["dropped"] > 0
    assert m["blocks"][0] == {"index": 0, "start": "2025-03-10", "end": "2025-03-24", "rows": 14 * 96 * 2, "months": ["2025-03"]}
    per = {"train": len(tr) // 2, "val": len(va) // 2, "cal": len(ca) // 2, "test": len(te) // 2, "val_blocks": [0, 1]}
    assert m["stations"] == {"A": per, "B": per} and m["by_station"] is False and m["station_col"] == "station"
    assert m["n_nat"] == 0 and m["inputs_disjoint_after"] is True and m["coverage_note"] is None


def test_write_splits_without_cal_and_with_station_id_and_nat(tmp_path):
    df = _frame().rename(columns={"station": "station_id"})
    df.loc[0, "timestamp_win"] = pd.NaT
    man = write_splits(df, SplitPlan.from_dict({**SPLIT, "cal_days": 0}), tmp_path / "o")
    assert not (tmp_path / "o" / "cal.parquet").exists() and "cal" not in man["counts"]
    assert man["n_nat"] == 1 and set(man["stations"]) == {"A", "B"}


def test_load_split_dict_from_yaml_or_m6_config(tmp_path):
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(SPLIT), encoding="utf-8")
    (tmp_path / "m6.yaml").write_text(yaml.safe_dump({"finetune": {"split": SPLIT}, "run": {"out_dir": "${x}"}}), encoding="utf-8")
    assert load_split_dict(split_yaml=tmp_path / "s.yaml") == SPLIT
    assert load_split_dict(m6_config=tmp_path / "m6.yaml") == SPLIT
    with pytest.raises(ValueError, match="二选一"):
        load_split_dict()
    with pytest.raises(ValueError, match="二选一"):
        load_split_dict(split_yaml=tmp_path / "s.yaml", m6_config=tmp_path / "m6.yaml")
    (tmp_path / "bad.yaml").write_text("run: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finetune.split"):
        load_split_dict(m6_config=tmp_path / "bad.yaml")


def test_load_split_dict_rejects_incomplete_split(tmp_path):
    """CLI 不替用户猜 purge/embargo/cal：缺哪个键就点名哪个键。"""
    partial = {k: v for k, v in SPLIT.items() if k != "purge_days"}
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(partial), encoding="utf-8")
    (tmp_path / "m6.yaml").write_text(yaml.safe_dump({"finetune": {"split": partial}}), encoding="utf-8")
    for kw in ({"split_yaml": tmp_path / "s.yaml"}, {"m6_config": tmp_path / "m6.yaml"}):
        with pytest.raises(ValueError, match="purge_days"):
            load_split_dict(**kw)
    (tmp_path / "empty.yaml").write_text(yaml.safe_dump({"test_start": "2025-10-01"}), encoding="utf-8")
    with pytest.raises(ValueError) as e:
        load_split_dict(split_yaml=tmp_path / "empty.yaml")
    for k in ("val_blocks", "cal_days", "purge_days", "embargo_days"):
        assert k in str(e.value)
    assert "test_start" not in str(e.value)


def test_write_splits_removes_stale_cal_parquet(tmp_path):
    """同一目录先切出 cal.parquet、再改成 cal_days=0 重切：旧的 cal.parquet 必须删掉，不能留着被误读。"""
    df, out = _frame(), tmp_path / "o"
    write_splits(df, SplitPlan.from_dict(SPLIT), out)
    assert (out / "cal.parquet").exists()
    man = write_splits(df, SplitPlan.from_dict({**SPLIT, "cal_days": 0}), out)
    assert not (out / "cal.parquet").exists() and "cal" not in man["counts"]


def test_cli_end_to_end(tmp_path, capsys):
    raw = tmp_path / "raw.parquet"; _frame().to_parquet(raw, index=False)
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(SPLIT), encoding="utf-8")
    man = main(["--parquet", str(raw), "--out-dir", str(tmp_path / "ds"), "--split-yaml", str(tmp_path / "s.yaml"),
                "--shift-days", "1"])
    assert man["plan"]["val_blocks"][0] == ["2025-03-11", "2025-03-25"] and man["shift_days"] == 1
    assert (tmp_path / "ds" / "valid.parquet").exists()
    out = capsys.readouterr().out
    assert "train=" in out and "val=" in out


def test_cli_takes_geometry_from_m6_config(tmp_path):
    """--m6-config 时 L/H/step_min 默认取该 yaml 的 timesfm3.context_len / pred_len 与 dataset.step_min；显式 flag 仍胜出。"""
    raw = tmp_path / "raw.parquet"; _frame().to_parquet(raw, index=False)
    (tmp_path / "m6.yaml").write_text(yaml.safe_dump({"timesfm3": {"pred_len": 384}, "finetune": {"split": SPLIT}}),
                                      encoding="utf-8")
    with pytest.raises(ValueError, match="purge_days"):                                  # H=384 → purge 至少 4 天，SPLIT 给 2
        main(["--parquet", str(raw), "--out-dir", str(tmp_path / "a"), "--m6-config", str(tmp_path / "m6.yaml")])
    man = main(["--parquet", str(raw), "--out-dir", str(tmp_path / "b"), "--m6-config", str(tmp_path / "m6.yaml"),
                "--H", "192"])                                                           # 显式 --H 覆盖 yaml
    assert man["plan"]["H"] == 192 and man["plan"]["L"] == 672 and man["plan"]["step_min"] == 15
    (tmp_path / "m6b.yaml").write_text(
        yaml.safe_dump({"timesfm3": {"context_len": 672, "pred_len": 384}, "dataset": {"step_min": 15},
                        "finetune": {"split": {**SPLIT, "purge_days": 4}}}), encoding="utf-8")
    man2 = main(["--parquet", str(raw), "--out-dir", str(tmp_path / "c"), "--m6-config", str(tmp_path / "m6b.yaml")])
    assert man2["plan"]["H"] == 384 and man2["plan"]["L"] == 672 and man2["plan"]["purge_days"] == 4


# ---------------- 按站不同 test_start ----------------
SPLIT_NO_TEST = {k: v for k, v in SPLIT.items() if k != "test_start"}


def test_station_plan_filters_blocks_per_station_and_keeps_global_indices():
    """A 站测试 10-01 容得下两个块；B 站测试 06-01（校准段起点 04-15）只容得下 3 月那块，7 月块被跳过并记录。"""
    sp = StationSplitPlan.from_dict(SPLIT_NO_TEST, {"A": "2025-10-01", "B": "2025-06-01"})
    assert sp.val_blocks == SplitPlan.from_dict(SPLIT).val_blocks
    assert sp.block_index == {"A": (0, 1), "B": (0,)} and sp.skipped_blocks == {"A": [], "B": [1]}
    assert sp.plans["B"].cal_interval == (pd.Timestamp("2025-04-15"), pd.Timestamp("2025-05-30"))
    ts = _grid(); n = len(ts)
    role, block = sp.assign(np.concatenate([ts, ts]), np.array(["A"] * n + ["B"] * n))
    ra, rb, ba, bb = role[:n], role[n:], block[:n], block[n:]
    assert set(ba[ra == "val"]) == {0, 1} and set(bb[rb == "val"]) == {0}                # B 的块序号仍是全局的 0
    assert int((rb == "val").sum()) == 14 * 96 and int((rb == "test").sum()) == 183 * 96  # 06-01 … 12-01
    assert int((rb == "cal").sum()) == 45 * 96
    assert ts[rb == "train"].max() < np.datetime64("2025-04-13")                          # cal_start − purge
    d = sp.to_dict()
    assert d["by_station"] is True and d["test_start"] == {"A": "2025-10-01", "B": "2025-06-01"}
    assert d["skipped_val_blocks"] == {"B": [1]} and d["val_blocks"] == SPLIT["val_blocks"]
    sp.verify(np.concatenate([ts, ts]), np.array(["A"] * n + ["B"] * n), role, block)    # 不抛


def test_station_plan_rejects_unknown_station_and_infeasible_cases():
    sp = StationSplitPlan.from_dict(SPLIT_NO_TEST, {"A": "2025-10-01"})
    with pytest.raises(ValueError, match="没有 test_start"):
        sp.assign(_grid()[:10], np.array(["A"] * 5 + ["C"] * 5))
    with pytest.raises(ValueError, match="一个验证块都容不下"):                           # B 测试 03-01：两块都在后面
        StationSplitPlan.from_dict(SPLIT_NO_TEST, {"A": "2025-10-01", "B": "2025-03-01"})
    with pytest.raises(ValueError, match="没有任何站容得下"):                             # 大家都容不下 7 月块
        StationSplitPlan.from_dict(SPLIT_NO_TEST, {"A": "2025-06-01", "B": "2025-06-01"})
    with pytest.raises(ValueError, match="test_starts 为空"):
        StationSplitPlan.from_dict(SPLIT_NO_TEST, {})


def test_station_plan_verify_reports_station_on_leak():
    sp = StationSplitPlan.from_dict(SPLIT_NO_TEST, {"A": "2025-10-01", "B": "2025-06-01"})
    ts = _grid(); n = len(ts)
    role, block = sp.assign(np.concatenate([ts, ts]), np.array(["A"] * n + ["B"] * n))
    bad = role.copy(); bad[n + int(np.searchsorted(ts, np.datetime64("2025-05-31")))] = "train"   # B 站塞一行到测试段前
    with pytest.raises(AssertionError, match="站 B"):
        sp.verify(np.concatenate([ts, ts]), np.array(["A"] * n + ["B"] * n), bad, block)


def _test_files(tmp_path, starts: dict, station_col="station"):
    paths = []
    for st, day in starts.items():
        ts = pd.date_range(day, periods=4 * 96, freq="15min")
        p = tmp_path / f"test_{st}.parquet"
        pd.DataFrame({"timestamp_win": ts, station_col: st, "observe_power": [np.zeros(4, np.float32)] * len(ts)}
                     ).to_parquet(p, index=False)
        paths.append(str(p))
    return paths


def test_load_test_starts_from_parquets_floors_to_day(tmp_path):
    paths = _test_files(tmp_path, {"A": "2025-10-01 06:15", "B": "2025-06-01 00:00"})
    assert load_test_starts(paths) == {"A": "2025-10-01", "B": "2025-06-01"}
    assert load_test_starts([str(tmp_path / "test_*.parquet")]) == {"A": "2025-10-01", "B": "2025-06-01"}
    with pytest.raises(FileNotFoundError):
        load_test_starts([str(tmp_path / "nope_*.parquet")])
    pd.DataFrame({"timestamp_win": pd.date_range("2025-07-01", periods=3, freq="15min")}).to_parquet(tmp_path / "x.parquet")
    with pytest.raises(ValueError, match="站标识列"):
        load_test_starts([str(tmp_path / "x.parquet")])


def test_cli_by_station_from_test_parquets(tmp_path, capsys):
    raw = tmp_path / "raw.parquet"; _frame().to_parquet(raw, index=False)
    (tmp_path / "s.yaml").write_text(yaml.safe_dump({**SPLIT_NO_TEST, "test_start": None}), encoding="utf-8")
    _test_files(tmp_path, {"A": "2025-10-01", "B": "2025-06-01"})
    man = main(["--parquet", str(raw), "--out-dir", str(tmp_path / "ds"), "--split-yaml", str(tmp_path / "s.yaml"),
                "--test-parquets", str(tmp_path / "test_*.parquet")])
    assert man["by_station"] is True and man["plan"]["test_start"] == {"A": "2025-10-01", "B": "2025-06-01"}
    assert man["stations"]["A"]["val_blocks"] == [0, 1] and man["stations"]["B"]["val_blocks"] == [0]
    tr = pd.read_parquet(tmp_path / "ds" / "train.parquet"); va = pd.read_parquet(tmp_path / "ds" / "valid.parquet")
    assert tr[tr["station"] == "B"]["timestamp_win"].max() < pd.Timestamp("2025-04-13")
    assert tr[tr["station"] == "A"]["timestamp_win"].max() < pd.Timestamp("2025-08-13")
    assert set(va[va["station"] == "B"]["split_block"]) == {0} and set(va[va["station"] == "A"]["split_block"]) == {0, 1}
    out = capsys.readouterr().out
    assert "逐站覆盖" in out and "split_resolved.yaml" in out
    rep = {r["station"]: r for r in man["station_report"]}
    assert rep["B"]["test_start"] == "2025-06-01" and rep["B"]["skipped_blocks"] == [1] and rep["B"]["val_blocks"] == [0]
    assert rep["A"]["skipped_blocks"] == [] and rep["A"]["val_days"] == 28 and rep["B"]["val_days"] == 14
    # split 里同时写了 test_start → 冲突报错；两个按站开关同时给 → 报错
    (tmp_path / "s2.yaml").write_text(yaml.safe_dump(SPLIT), encoding="utf-8")
    with pytest.raises(ValueError, match="删掉其一"):
        main(["--parquet", str(raw), "--out-dir", str(tmp_path / "e"), "--split-yaml", str(tmp_path / "s2.yaml"),
              "--test-parquets", str(tmp_path / "test_*.parquet")])
    with pytest.raises(ValueError, match="二选一"):
        main(["--parquet", str(raw), "--out-dir", str(tmp_path / "e"), "--split-yaml", str(tmp_path / "s.yaml"),
              "--test-parquets", "x", "--test-start-by-station", "y"])


def test_cli_by_station_from_mapping_file_and_missing_station(tmp_path):
    raw = tmp_path / "raw.parquet"; _frame().to_parquet(raw, index=False)
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(SPLIT_NO_TEST), encoding="utf-8")          # 干脆不写 test_start
    (tmp_path / "m.json").write_text(json.dumps({"A": "2025-10-01", "B": "2025-06-01"}), encoding="utf-8")
    man = main(["--parquet", str(raw), "--out-dir", str(tmp_path / "ds"), "--split-yaml", str(tmp_path / "s.yaml"),
                "--test-start-by-station", str(tmp_path / "m.json")])
    assert man["plan"]["skipped_val_blocks"] == {"B": [1]}
    (tmp_path / "m2.yaml").write_text(yaml.safe_dump({"A": "2025-10-01"}), encoding="utf-8")   # 少了 B：B 视为没有测试集
    man = main(["--parquet", str(raw), "--out-dir", str(tmp_path / "e"), "--split-yaml", str(tmp_path / "s.yaml"),
                "--test-start-by-station", str(tmp_path / "m2.yaml")])
    assert man["plan"]["stations_without_test"] == ["B"]
    assert set(pd.read_parquet(tmp_path / "e" / "test.parquet")["station"]) == {"A"}
    with pytest.raises(ValueError, match="缺少切分键"):                                        # 不按站时 test_start 仍必填
        main(["--parquet", str(raw), "--out-dir", str(tmp_path / "e"), "--split-yaml", str(tmp_path / "s.yaml")])


# ---------------- 跨度不一的站：自动放块 / 无验证行报错 / 逐站报告 / 复现 ----------------
# A 15 个月、B 从 2025-03-01 起、C 不到半年；测试段各取尾部
SPANS = {"A": ("2024-06-01", "2025-09-01"), "B": ("2025-03-01", "2026-01-20"), "C": ("2025-08-01", "2025-12-20")}
FIXED = {"train_start": None, "val_blocks": [["2025-01-10", "2025-01-31"], ["2025-04-10", "2025-05-01"], ["2025-07-16", "2025-08-06"]],
         "test_end": None, "cal_days": 0, "purge_days": 2, "embargo_days": 7}
AUTO = {**FIXED, "val_blocks": "auto", "n_val_blocks": 4, "val_block_days": 14}


def _days(a, b):
    return pd.date_range(a, b, freq="D", inclusive="left").to_numpy().astype("datetime64[D]")


def _span_frame(spans=SPANS, freq="15min"):
    return pd.concat([pd.DataFrame({"timestamp_win": pd.date_range(a, b, freq=freq, inclusive="left"), "station": s,
                                    "observe_power": [np.zeros(4, np.float32)] * len(pd.date_range(a, b, freq=freq, inclusive="left")),
                                    "observe_power_future": [np.zeros(2, np.float32)] * len(pd.date_range(a, b, freq=freq, inclusive="left"))})
                      for s, (a, b) in spans.items()], ignore_index=True)


def test_suggest_val_blocks_covers_stations_with_different_spans():
    days = {s: _days(a, b) for s, (a, b) in SPANS.items()}
    tests = {s: pd.Timestamp(b) for s, (a, b) in SPANS.items()}
    blocks, notes = suggest_val_blocks(days, tests, n_blocks=4, block_days=14, purge_days=2, embargo_days=7, cal_days=0)
    assert len(blocks) == 4 and notes == [] and blocks == sorted(blocks)
    assert all(e - s == pd.Timedelta(days=14) for s, e in blocks)
    assert all(s1 - e0 >= pd.Timedelta(days=11) for (_, e0), (s1, _) in zip(blocks, blocks[1:]))   # 剔除区不重叠
    lim = {s: t - pd.Timedelta(days=4) for s, t in tests.items()}
    for s, (a, b) in SPANS.items():                                                         # 每站至少被一块覆盖
        assert any(e <= lim[s] and s0 >= pd.Timestamp(a) for s0, e in blocks), s
    assert suggest_val_blocks(days, tests, 4, 14, 2, 7, 0) == (blocks, notes)               # 确定性
    months = {b.month for b, _ in blocks}
    assert len({(m - 1) // 3 for m in months}) >= 3                                         # 至少跨 3 个季度


def test_suggest_val_blocks_skips_data_gap_and_rejects_short_range():
    d = np.concatenate([_days("2025-01-01", "2025-04-01"), _days("2025-07-10", "2026-01-01")])   # 4–7 月整段缺数
    blocks, notes = suggest_val_blocks({"A": d}, {"A": pd.Timestamp("2026-01-05")}, 4, 14, 2, 7, 0)
    assert len(blocks) == 3 and any("不到半块" in n for n in notes) and "要求 4 块，实际放了 3 块" in notes
    assert not any(pd.Timestamp("2025-04-01") <= s < pd.Timestamp("2025-07-10") for s, _ in blocks)
    with pytest.raises(ValueError, match="放不下 4 块"):
        suggest_val_blocks({"A": _days("2025-01-01", "2025-02-01")}, {"A": pd.Timestamp("2025-02-05")}, 4, 14, 2, 7, 0)
    with pytest.raises(ValueError, match="没有 test_start"):
        suggest_val_blocks({"A": d, "Z": d}, {"A": pd.Timestamp("2026-01-05")}, 4, 14, 2, 7, 0)


def test_station_without_val_rows_raises_unless_allowed(tmp_path):
    spans = {**SPANS, "C": ("2025-08-10", "2025-12-20")}                     # C 的数据全落在固定块之后
    df = _span_frame(spans, freq="60min")
    starts = {s: b for s, (a, b) in spans.items()}
    with pytest.raises(ValueError, match="一行验证数据都没有：C"):
        write_splits(df, StationSplitPlan.from_dict(FIXED, starts), tmp_path / "a")
    assert not (tmp_path / "a" / "train.parquet").exists()                     # 先报错、不落半成品
    man = write_splits(df, StationSplitPlan.from_dict({**FIXED, "allow_stations_without_val": True}, starts), tmp_path / "b")
    rep = {r["station"]: r for r in man["station_report"]}
    assert rep["C"]["val_days"] == 0 and rep["C"]["train_days"] > 100 and rep["C"]["train_share"] > 0.9
    assert rep["A"]["val_blocks"] == [0, 1, 2] and man["plan"]["allow_stations_without_val"] is True


def test_station_plan_allows_station_whose_test_is_before_every_block():
    starts = {"A": "2025-10-01", "B": "2025-02-01"}                           # B：1 月块结束 01-31 晚于 02-01 − 4 天
    with pytest.raises(ValueError, match="allow_stations_without_val"):
        StationSplitPlan.from_dict(FIXED, starts)
    sp = StationSplitPlan.from_dict({**FIXED, "allow_stations_without_val": True}, starts)
    assert sp.block_index["B"] == () and sp.plans["B"].val_blocks == () and sp.plans["B"].coverage_note is None
    ts = _grid("2024-12-01", "2025-04-01"); n = len(ts)
    role, block = sp.assign(np.concatenate([ts, ts]), np.array(["A"] * n + ["B"] * n))
    rb = role[n:]
    assert (rb == "val").sum() == 0 and ts[rb == "train"].max() < np.datetime64("2025-01-30")      # test_start − purge
    sp.verify(np.concatenate([ts, ts]), np.array(["A"] * n + ["B"] * n), role, block)


def test_auto_is_rejected_where_it_cannot_be_resolved():
    with pytest.raises(ValueError, match="split_resolved.yaml"):
        SplitPlan.from_dict({**SPLIT, "val_blocks": "auto"})
    with pytest.raises(ValueError, match="split_resolved.yaml"):
        StationSplitPlan.from_dict({**SPLIT_NO_TEST, "val_blocks": "auto"}, {"A": "2025-10-01"})


def _write_station_files(tmp_path, spans=SPANS):
    (tmp_path / "train").mkdir(); (tmp_path / "test").mkdir()
    for s, (a, b) in spans.items():
        _span_frame({s: (a, b)}, freq="60min").to_parquet(tmp_path / "train" / f"{s}.parquet", index=False)
        pd.DataFrame({"timestamp_win": pd.date_range(b, periods=48, freq="60min"), "station": s}
                     ).to_parquet(tmp_path / "test" / f"{s}.parquet", index=False)


def test_cli_auto_blocks_multi_file_dry_run_and_resolved_yaml_reproduces(tmp_path, capsys):
    _write_station_files(tmp_path)
    (tmp_path / "auto.yaml").write_text(yaml.safe_dump(AUTO), encoding="utf-8")
    base = ["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "auto.yaml"),
            "--test-parquets", str(tmp_path / "test" / "*.parquet")]
    dry = main(base + ["--dry-run"])                                             # 不给 --out-dir 也行
    assert not (tmp_path / "ds").exists() and len(dry["source"]) == 3
    assert all(r["val_days"] > 0 for r in dry["station_report"])
    out = capsys.readouterr().out
    assert "val_blocks: auto →" in out and "dry-run，未写文件" in out and "逐站覆盖" in out
    man = main(base + ["--out-dir", str(tmp_path / "ds")])
    assert man["plan"]["val_blocks"] == dry["plan"]["val_blocks"] and len(man["plan"]["val_blocks"]) == 4
    resolved = yaml.safe_load((tmp_path / "ds" / "split_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["val_blocks"] == man["plan"]["val_blocks"]
    assert resolved["test_start"] == {s: b for s, (a, b) in SPANS.items()}
    # 只用 split_resolved.yaml（不带 --test-parquets）重跑：逐字节同一切分
    man2 = main(["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "ds" / "split_resolved.yaml"),
                 "--out-dir", str(tmp_path / "ds2")])
    for f in ("train.parquet", "valid.parquet", "test.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "ds" / f), pd.read_parquet(tmp_path / "ds2" / f))
    assert man2["counts"] == man["counts"]
    with pytest.raises(ValueError, match="shift-days"):
        main(base + ["--out-dir", str(tmp_path / "e"), "--shift-days", "3"])
    with pytest.raises(ValueError, match="dry-run"):
        main(base)                                                               # 既无 --out-dir 也无 --dry-run
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({k: v for k, v in AUTO.items() if k != "val_block_days"}), encoding="utf-8")
    with pytest.raises(ValueError, match="val_block_days"):
        main(["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "bad.yaml"),
              "--test-parquets", str(tmp_path / "test" / "*.parquet"), "--dry-run"])


def test_read_parquets_rejects_mismatched_columns(tmp_path):
    pd.DataFrame({"timestamp_win": [pd.Timestamp("2025-01-01")], "station": ["A"], "x": [1.0]}).to_parquet(tmp_path / "a.parquet")
    pd.DataFrame({"timestamp_win": [pd.Timestamp("2025-01-01")], "station": ["B"]}).to_parquet(tmp_path / "b.parquet")
    with pytest.raises(ValueError, match="缺少 \\['x'\\]"):
        read_parquets([str(tmp_path / "*.parquet")])
    df, paths = read_parquets([str(tmp_path / "a.parquet")])
    assert len(df) == 1 and paths == [str(tmp_path / "a.parquet")]
    with pytest.raises(FileNotFoundError):
        read_parquets([str(tmp_path / "nope*.parquet")])



def test_read_parquets_unifies_station_types_across_files(tmp_path, capsys):
    """一站一个文件时，有的文件站号存成整数、有的存成文本：统一成文本站号再拼，否则落盘时 pyarrow 报类型错。"""
    spans = {"1001": ("2024-06-01", "2025-09-01"), "B": ("2025-03-01", "2026-01-20")}
    _write_station_files(tmp_path, spans)
    f = tmp_path / "train" / "1001.parquet"; x = pd.read_parquet(f); x["station"] = 1001; x.to_parquet(f, index=False)
    df, _ = read_parquets([str(tmp_path / "train" / "*.parquet")])
    assert set(df["station"]) == {"1001", "B"} and "类型不一致" in capsys.readouterr().out
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(AUTO), encoding="utf-8")
    man = main(["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "s.yaml"),
                "--test-parquets", str(tmp_path / "test" / "*.parquet"), "--out-dir", str(tmp_path / "ds")])
    tr = pd.read_parquet(tmp_path / "ds" / "train.parquet")
    assert set(tr["station"]) == {"1001", "B"} and man["counts"]["train"] > 0

    g = tmp_path / "train" / "B.parquet"; y = pd.read_parquet(g)
    y["observe_power"] = y["observe_power"].map(lambda v: "x"); y.to_parquet(g, index=False)   # 非站号列类型不一致：报错
    with pytest.raises(ValueError, match="列 'observe_power' 在各文件里类型不一致"):
        read_parquets([str(tmp_path / "train" / "*.parquet")])

# ---------------- 数据太短的站只进训练 ----------------
SHORT = {**SPANS, "D": ("2025-07-20", "2025-09-10")}                 # D 测试起点前 52 天，且压着 8 月的自动块


def test_short_station_is_train_only_and_does_not_move_auto_blocks(tmp_path, capsys):
    _write_station_files(tmp_path, SHORT)
    cfg = {**AUTO, "min_station_days_for_val": 90}
    (tmp_path / "auto.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    base = ["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "auto.yaml"),
            "--test-parquets", str(tmp_path / "test" / "*.parquet")]
    man = main(base + ["--out-dir", str(tmp_path / "ds")])
    assert "站 D 测试起点前只有 52 天数据" in capsys.readouterr().out
    # 放块与没有 D 时完全一样：D 不参与放块
    ref_blocks, _ = suggest_val_blocks({s: _days(a, b) for s, (a, b) in SPANS.items()},
                                       {s: pd.Timestamp(b) for s, (a, b) in SPANS.items()}, 4, 14, 2, 7, 0)
    assert man["plan"]["val_blocks"] == [[a.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")] for a, e in ref_blocks]
    assert man["plan"]["train_only_stations"] == ["D"]
    rep = {r["station"]: r for r in man["station_report"]}
    assert rep["D"]["train_only"] is True and rep["D"]["val_days"] == 0 and rep["D"]["val_blocks"] == []
    assert rep["D"]["train_days"] == 50 and rep["D"]["dropped_days"] == 2                  # 只剔测试起点前 purge 2 天
    assert all(not rep[s]["train_only"] and rep[s]["val_days"] > 0 for s in SPANS)
    tr = pd.read_parquet(tmp_path / "ds" / "train.parquet"); va = pd.read_parquet(tmp_path / "ds" / "valid.parquet")
    d_tr = tr[tr["station"] == "D"]["timestamp_win"]
    assert ((d_tr >= "2025-08-01") & (d_tr < "2025-08-15")).any()                          # 验证块日期里的 D 行仍是训练
    assert (va["station"] != "D").all() and d_tr.max() < pd.Timestamp("2025-09-08")
    # split_resolved.yaml 写死名单、去掉阈值，原样重跑逐字节一致
    resolved = yaml.safe_load((tmp_path / "ds" / "split_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["train_only_stations"] == ["D"] and "min_station_days_for_val" not in resolved
    main(["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "ds" / "split_resolved.yaml"),
          "--out-dir", str(tmp_path / "ds2")])
    for f in ("train.parquet", "valid.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "ds" / f), pd.read_parquet(tmp_path / "ds2" / f))


def test_explicit_train_only_stations_and_errors(tmp_path):
    _write_station_files(tmp_path)
    run = lambda cfg, extra=(): (
        (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8"),
        main(["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "c.yaml"),
              "--test-parquets", str(tmp_path / "test" / "*.parquet"), "--dry-run", *extra]))[1]
    man = run({**FIXED, "train_only_stations": ["C"]})                                   # 固定块 + 点名
    rep = {r["station"]: r for r in man["station_report"]}
    assert rep["C"]["train_only"] and rep["C"]["val_days"] == 0 and rep["A"]["val_days"] > 0
    with pytest.raises(ValueError, match="训练数据里没有"):
        run({**FIXED, "train_only_stations": ["Z"]})
    with pytest.raises(ValueError, match="没有站可以做验证"):
        run({**AUTO, "min_station_days_for_val": 10_000})
    with pytest.raises(ValueError, match="train_only_stations 需要按站切分"):
        SplitPlan.from_dict({**SPLIT, "train_only_stations": ["A"]})
    with pytest.raises(ValueError, match="split_resolved.yaml"):
        SplitPlan.from_dict({**SPLIT, "min_station_days_for_val": 90})
    with pytest.raises(ValueError, match="没有 test_start"):
        StationSplitPlan.from_dict({**FIXED, "train_only_stations": ["Q"]}, {"A": "2025-10-01"})


def test_single_test_start_with_threshold_switches_to_by_station(tmp_path):
    df = pd.concat([_frame(stations=("A",)),
                    _frame("2025-09-01", "2025-12-01", stations=("S",))], ignore_index=True)   # S 测试起点前只有 30 天
    raw = tmp_path / "raw.parquet"; df.to_parquet(raw, index=False)
    (tmp_path / "s.yaml").write_text(yaml.safe_dump({**SPLIT, "min_station_days_for_val": 60}), encoding="utf-8")
    man = main(["--parquet", str(raw), "--split-yaml", str(tmp_path / "s.yaml"), "--out-dir", str(tmp_path / "ds")])
    assert man["by_station"] is True and man["plan"]["test_start"] == {"A": "2025-10-01", "S": "2025-10-01"}
    rep = {r["station"]: r for r in man["station_report"]}
    assert rep["S"]["train_only"] and rep["S"]["train_days"] == 28 and rep["S"]["test_days"] == 61
    assert rep["A"]["val_days"] == 28 and not rep["A"]["train_only"]


def test_unmatched_stations_are_ignored_or_kept_to_their_end(tmp_path, capsys):
    """E 有训练数据、没有测试文件 → 不留测试段、数据用到最后一天、照常参加验证；
    T 只有测试文件（测试起点早到一个验证块都容不下）→ 切分时忽略，不报错。"""
    spans = {**SPANS, "E": ("2024-09-01", "2027-06-01")}                                  # E 的数据比所有站的测试起点都晚
    _write_station_files(tmp_path, spans)
    (tmp_path / "test" / "E.parquet").unlink()
    pd.DataFrame({"timestamp_win": pd.date_range("2024-07-01", periods=48, freq="60min"), "station": "T"}
                 ).to_parquet(tmp_path / "test" / "T.parquet", index=False)
    run = lambda split, out, *extra: (
        (tmp_path / "s.yaml").write_text(yaml.safe_dump(split), encoding="utf-8"),
        main(["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "s.yaml"),
              "--out-dir", str(tmp_path / out), *extra]))[1]
    man = run(AUTO, "ds", "--test-parquets", str(tmp_path / "test" / "*.parquet"))
    printed = capsys.readouterr().out
    assert "站 T 只在测试集里" in printed and "站 E 没有测试集" in printed
    assert man["ignored_test_only_stations"] == ["T"] and man["plan"]["stations_without_test"] == ["E"]
    assert man["plan"]["test_start"]["E"] == "2027-06-03"                                  # 最后一天 05-31 + 1 + purge 2
    assert all(pd.Timestamp(e) <= pd.Timestamp("2026-01-16") for _, e in man["plan"]["val_blocks"])   # 不越过 B 的 01-20 − 4 天
    assert "之后的数据不参与放块" in printed
    rep = {r["station"]: r for r in man["station_report"]}
    assert "T" not in rep and rep["E"]["test_start"] == "无测试集"
    assert rep["E"]["val_days"] > 0 and rep["E"]["test_days"] == 0
    last = pd.Timestamp("2027-05-31 23:00")
    tr = pd.concat([pd.read_parquet(tmp_path / "ds" / f) for f in ("train.parquet", "valid.parquet")])
    assert tr.loc[tr.station == "E", "timestamp_win"].max() == last                       # 末尾一行都没剔

    resolved = (tmp_path / "ds" / "split_resolved.yaml").read_text(encoding="utf-8")
    assert "stations_without_test" in resolved and "占位值" in resolved
    (tmp_path / "r.yaml").write_text(resolved, encoding="utf-8")
    again = main(["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "r.yaml"),
                  "--out-dir", str(tmp_path / "ds2")])                                     # 原样复现
    assert again["plan"] == man["plan"] and again["counts"] == man["counts"]
    for f in ("train.parquet", "valid.parquet", "test.parquet"):
        assert pd.read_parquet(tmp_path / "ds" / f).equals(pd.read_parquet(tmp_path / "ds2" / f)), f

    man = run({**AUTO, "train_only_stations": ["E"]}, "only", "--test-parquets", str(tmp_path / "test" / "*.parquet"))
    rep = {r["station"]: r for r in man["station_report"]}
    assert rep["E"]["train_only"] and rep["E"]["val_days"] == 0 and rep["E"]["dropped_days"] == 0
    with pytest.raises(ValueError, match="需要按站切分"):
        SplitPlan.from_dict({**SPLIT, "stations_without_test": ["A"]})
    with pytest.raises(ValueError, match="没有占位 test_start"):
        StationSplitPlan.from_dict({**FIXED, "stations_without_test": ["Q"]}, {"A": "2025-10-01"})


def test_val_only_stations_hold_out_whole_station(tmp_path, capsys):
    """val_only_stations：整站只进验证（块内行带块序号、块外行 −1），不进训练；CLI 与 resolved yaml 都带这个键。"""
    spans = {**SPANS, "C": ("2025-08-10", "2025-12-20")}
    df = _span_frame(spans, freq="60min")
    starts = {s: b for s, (a, b) in spans.items()}
    man = write_splits(df, StationSplitPlan.from_dict({**FIXED, "allow_stations_without_val": True}, starts), tmp_path / "a",
                       val_only=["C"])
    rep = {r["station"]: r for r in man["station_report"]}
    assert rep["C"]["val_only"] and not rep["A"]["val_only"] and man["val_only_stations"] == ["C"]
    assert rep["C"]["train_days"] == 0 and rep["C"]["val_days"] > 100 and rep["C"]["test_days"] == 0
    va = pd.read_parquet(tmp_path / "a" / "valid.parquet")
    assert (va.loc[va.station == "C", "split_block"] == -1).all()          # C 的数据不在任何固定块里
    assert set(va.loc[va.station == "A", "split_block"]) == {0, 1, 2}
    tr = pd.read_parquet(tmp_path / "a" / "train.parquet")
    assert "C" not in set(tr.station)
    with pytest.raises(ValueError, match="在数据里没有"):
        write_splits(df, StationSplitPlan.from_dict({**FIXED, "allow_stations_without_val": True}, starts), tmp_path / "b", val_only=["Z"])

    _write_station_files(tmp_path, spans)
    (tmp_path / "s.yaml").write_text(yaml.safe_dump({**AUTO, "allow_stations_without_val": True, "val_only_stations": ["C"]}), encoding="utf-8")
    man = main(["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "s.yaml"),
                "--out-dir", str(tmp_path / "ds"), "--test-parquets", str(tmp_path / "test" / "*.parquet")])
    assert "整站只进验证" in capsys.readouterr().out
    rep = {r["station"]: r for r in man["station_report"]}
    assert rep["C"]["val_only"] and rep["C"]["train_days"] == 0 and rep["C"]["val_days"] > 0
    resolved = yaml.safe_load((tmp_path / "ds" / "split_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["val_only_stations"] == ["C"]
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({**AUTO, "val_only_stations": ["C"], "train_only_stations": ["C"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="同时写在"):
        main(["--parquet", str(tmp_path / "train" / "*.parquet"), "--split-yaml", str(tmp_path / "bad.yaml"), "--dry-run",
              "--test-parquets", str(tmp_path / "test" / "*.parquet")])
