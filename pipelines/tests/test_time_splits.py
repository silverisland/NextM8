import json

import numpy as np
import pandas as pd
import pytest
import yaml

from pipelines.time_splits import SplitPlan, verify_leak_free, LEGACY_KEYS, write_splits, load_split_dict, main

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
    assert m["stations"] == {"A": {"train": len(tr) // 2, "val": len(va) // 2, "cal": len(ca) // 2, "test": len(te) // 2},
                             "B": {"train": len(tr) // 2, "val": len(va) // 2, "cal": len(ca) // 2, "test": len(te) // 2}}
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
