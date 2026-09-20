# -*- coding: utf-8 -*-
# pipelines/time_splits.py
"""时间轴切分（与模型无关）：验证块列表 + 尾段测试 + 校准段 + purge/embargo。
每行是一个起报时刻 timestamp_win；训练行标签向前伸 H 步、上下文向后伸 L 步，相邻行几乎共用同一段标签，
所以切分只能按时间轴切，并在验证/校准/测试块前后丢掉一段训练行：
  块前 purge_days（≥ H 折算天数）：训练行标签不伸进块；
  块后 purge_days + embargo_days 合计须 ≥ L + H 折算天数（默认 2 + 7 = 9 天 = 672 + 192 步）：
  前 H 步保证训练行标签不与块标签重叠，再往后的 L 步保证训练行 7 天上下文不含块的标签。
校准段固定放在 test_start − purge 之前 cal_days 天；训练行到校准段前 purge 为止。"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LEGACY_KEYS = ("train_end", "val_start", "val_end", "rolling_folds")
ROLES = ("train", "val", "cal", "test", "dropped")


def _day(x, name: str) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    if t != t.normalize():
        raise ValueError(f"{name}={x!r} 不是整天边界（须为 00:00）")
    return t


def _dt64(t: pd.Timestamp) -> np.datetime64:
    return pd.Timestamp(t).to_datetime64().astype("datetime64[ns]")


@dataclass(frozen=True)
class SplitPlan:
    val_blocks: tuple                       # ((start, end), ...) 半开、整天、已排序、不重叠
    test_start: pd.Timestamp
    test_end: pd.Timestamp | None           # None = 数据末尾
    cal_days: int                           # 0 = 无校准段
    purge_days: int
    embargo_days: int
    train_start: pd.Timestamp | None = None
    L: int = 672
    H: int = 192
    step_min: int = 15

    @classmethod
    def from_dict(cls, split: dict, shift_days: int = 0, L: int = 672, H: int = 192, step_min: int = 15) -> "SplitPlan":
        legacy = [k for k in LEGACY_KEYS if k in split]
        if legacy:
            raise ValueError(f"split 含旧键 {legacy}：请改为 val_blocks=[[start,end],...] + test_start/test_end + "
                             f"cal_days + purge_days/embargo_days（见 forecast_backbones/m6/README.md 的切分说明）")
        sh = pd.Timedelta(days=int(shift_days))
        purge, embargo, cal = int(split.get("purge_days", 2)), int(split.get("embargo_days", 7)), int(split.get("cal_days", 0))
        steps_per_day = 1440 / int(step_min)
        min_purge = math.ceil(int(H) / steps_per_day)
        if purge < min_purge:
            raise ValueError(f"purge_days={purge} 小于 H={H} 步折算的 {min_purge} 天：训练行标签会伸进验证/校准/测试块")
        if embargo < 0:
            raise ValueError(f"embargo_days={embargo} 不能为负")
        min_after = math.ceil((int(L) + int(H)) / steps_per_day)
        if purge + embargo < min_after:
            raise ValueError(f"purge_days={purge} + embargo_days={embargo} 合计 {purge + embargo} 天，"
                             f"小于 L={L} + H={H} 步折算的 {min_after} 天：块后的训练行上下文仍会含验证块的标签")
        if cal < 0:
            raise ValueError(f"cal_days={cal} 不能为负")
        raw = list(split.get("val_blocks") or [])
        if not raw:
            raise ValueError("val_blocks 为空：至少需要一个验证块 [start, end)")
        blocks = []
        for i, pair in enumerate(raw):
            if len(pair) != 2:
                raise ValueError(f"val_blocks[{i}]={pair!r} 须是 [start, end]")
            s, e = _day(pair[0], f"val_blocks[{i}][0]") + sh, _day(pair[1], f"val_blocks[{i}][1]") + sh
            if e - s < pd.Timedelta(days=2):
                raise ValueError(f"val_blocks[{i}]=[{s.date()}, {e.date()}) 不足 2 天，凑不出可打分的完整天")
            blocks.append((s, e))
        blocks.sort()
        for (s0, e0), (s1, e1) in zip(blocks, blocks[1:]):
            if e0 > s1:
                raise ValueError(f"验证块重叠：[{s0.date()}, {e0.date()}) 与 [{s1.date()}, {e1.date()})")
        test_start = _day(split["test_start"], "test_start") + sh
        test_end = (_day(split["test_end"], "test_end") + sh) if split.get("test_end") else None
        if test_end is not None and test_end <= test_start:
            raise ValueError(f"test_end={test_end.date()} 不晚于 test_start={test_start.date()}")
        train_start = (_day(split["train_start"], "train_start") + sh) if split.get("train_start") else None
        if train_start is not None and blocks[0][0] < train_start:
            raise ValueError(f"第一个验证块起点 {blocks[0][0].date()} 早于 train_start {train_start.date()}："
                             f"val_blocks 必须落在训练数据区间之内")
        plan = cls(tuple(blocks), test_start, test_end, cal, purge, embargo, train_start, int(L), int(H), int(step_min))
        c = plan.cal_interval
        guard = c[0] if c is not None else test_start - plan.purge
        last_end = blocks[-1][1]
        if last_end + plan.purge > guard:
            what = "校准段起点" if c is not None else "test_start − purge"
            raise ValueError(f"最后一个验证块结束 {last_end.date()} + purge {purge} 天 晚于{what} {guard.date()}")
        return plan

    # ---- 区间 ----
    @property
    def purge(self) -> pd.Timedelta:
        return pd.Timedelta(days=self.purge_days)

    @property
    def embargo(self) -> pd.Timedelta:
        return pd.Timedelta(days=self.embargo_days)

    @property
    def cal_interval(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if self.cal_days <= 0:
            return None
        end = self.test_start - self.purge
        return end - pd.Timedelta(days=self.cal_days), end

    @property
    def train_exclusions(self) -> list[tuple[pd.Timestamp, pd.Timestamp | None]]:
        """训练行被剔除的半开区间；末项右端 None 表示直到数据末尾。"""
        ex = [(s - self.purge, e + self.purge + self.embargo) for s, e in self.val_blocks]
        c = self.cal_interval
        ex.append(((c[0] - self.purge) if c is not None else (self.test_start - self.purge), None))
        return ex

    @property
    def inputs_disjoint_after(self) -> bool:
        """块后剔除长度是否 ≥ L + H 步：为真时任何训练行上下文都不含验证块标签。
        from_dict 已把这条当硬约束拦在构造期，所以对任何构造出来的 plan 恒为真；保留它是为了让 manifest 里直接读得到这条事实。"""
        return (self.purge_days + self.embargo_days) * 1440 / self.step_min >= self.L + self.H

    @property
    def coverage_note(self) -> str | None:
        quarters = {(s.year, (s.month - 1) // 3 + 1) for s, _ in self.val_blocks}
        if len(quarters) >= 2:
            return None
        return ("全部验证块落在同一日历季度：早停与选型只看到一个季节的行为，建议把验证块分散到不同季节"
                f"（当前 {sorted(quarters)}）")

    # ---- 行标注 ----
    def assign(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ts: datetime64 数组（起报时刻）。返回 (role, block)：role ∈ ROLES；block 为验证块序号，非验证行 −1。"""
        t = np.asarray(ts).astype("datetime64[ns]")
        nat = np.isnat(t)
        role = np.full(len(t), "train", dtype=object)
        block = np.full(len(t), -1, dtype=np.int64)

        def _in(a, b) -> np.ndarray:
            m = t >= _dt64(a)
            if b is not None:
                m &= t < _dt64(b)
            return m & ~nat

        for a, b in self.train_exclusions:
            role[_in(a, b)] = "dropped"
        for i, (s, e) in enumerate(self.val_blocks):
            m = _in(s, e)
            role[m] = "val"
            block[m] = i
        c = self.cal_interval
        if c is not None:
            role[_in(*c)] = "cal"
        role[_in(self.test_start, self.test_end)] = "test"
        if self.train_start is not None:
            role[(t < _dt64(self.train_start)) & ~nat & (role == "train")] = "dropped"
        role[nat] = "dropped"
        return role, block

    def to_dict(self) -> dict:
        iso = lambda x: None if x is None else pd.Timestamp(x).strftime("%Y-%m-%d")
        c = self.cal_interval
        return {"train_start": iso(self.train_start),
                "val_blocks": [[iso(s), iso(e)] for s, e in self.val_blocks],
                "cal": None if c is None else [iso(c[0]), iso(c[1])],
                "test_start": iso(self.test_start), "test_end": iso(self.test_end),
                "cal_days": self.cal_days, "purge_days": self.purge_days, "embargo_days": self.embargo_days,
                "L": self.L, "H": self.H, "step_min": self.step_min,
                "train_exclusions": [[iso(a), iso(b)] for a, b in self.train_exclusions],
                "inputs_disjoint_after": self.inputs_disjoint_after, "coverage_note": self.coverage_note}


def verify_leak_free(ts_train, ts_val_blocks, ts_cal, plan: SplitPlan) -> None:
    """独立于 assign 的算术复核（按实际行，不按块边界）：
    标签侧：训练行 t 的标签 (t, t+H] 不得与任何验证行 / 校准行的标签 (v, v+H] 相交，即
    对每个验证块：not (min(v) − H < t < max(v) + H)；对校准段：not (t > min(c) − H)。
    上下文侧：块后起报的训练行 t 的上下文 [t − (L−1), t] 不得盖到块的标签段，即
    对每个验证块：not (t > max(v) 且 t − (L−1) ≤ max(v) + H)。不通过则 AssertionError。
    测试段的边界（t 不得 > test_start − H）无论有没有校准段都检查。
    失败时显式 raise AssertionError（不用裸 assert），以免 -O / PYTHONOPTIMIZE 下检查被整段剥离而静默通过。"""
    tr = np.asarray(ts_train).astype("datetime64[ns]")
    Hd = np.timedelta64(int(plan.H * plan.step_min), "m")
    Ld_minus_1 = np.timedelta64(int((plan.L - 1) * plan.step_min), "m")
    for i, tv in enumerate(ts_val_blocks):
        tv = np.asarray(tv).astype("datetime64[ns]")
        if tv.size == 0 or tr.size == 0:
            continue
        bad = tr[(tr > tv.min() - Hd) & (tr < tv.max() + Hd)]
        if bad.size != 0:
            raise AssertionError(f"验证块 {i} 与 {bad.size} 个训练行标签重叠，例如 {bad.min()} … {bad.max()}")
        bad = tr[(tr > tv.max()) & (tr - Ld_minus_1 <= tv.max() + Hd)]
        if bad.size != 0:
            raise AssertionError(f"验证块 {i} 的标签落进 {bad.size} 个块后训练行的上下文，例如 {bad.min()} … {bad.max()}")
    if plan.cal_interval is not None and ts_cal is not None:
        tc = np.asarray(ts_cal).astype("datetime64[ns]")
        if tc.size and tr.size:
            bad = tr[tr > tc.min() - Hd]
            if bad.size != 0:
                raise AssertionError(f"校准段与 {bad.size} 个训练行标签重叠，例如 {bad.min()}")
    bad = tr[tr > _dt64(plan.test_start) - Hd] if tr.size else tr
    if bad.size != 0:
        raise AssertionError(f"测试段与 {bad.size} 个训练行标签重叠，例如 {bad.min()}")


def write_splits(df: pd.DataFrame, plan: SplitPlan, out_dir, ts_col: str = "timestamp_win",
                 source=None, shift_days: int = 0) -> dict:
    """按 plan 把原始行表落成 train/valid[/cal]/test.parquet（按时间稳定排序）+ split_manifest.json。返回 manifest。"""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(df[ts_col])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    ts = ts.to_numpy().astype("datetime64[ns]")
    role, block = plan.assign(ts)
    order = np.argsort(ts, kind="stable")
    df, ts, role, block = df.iloc[order].reset_index(drop=True), ts[order], role[order], block[order]
    val_ts = [ts[(role == "val") & (block == i)] for i in range(len(plan.val_blocks))]
    verify_leak_free(ts[role == "train"], val_ts, ts[role == "cal"] if plan.cal_interval is not None else None, plan)

    files = {"train": "train.parquet", "val": "valid.parquet", "cal": "cal.parquet", "test": "test.parquet"}
    counts = {}
    for r, name in files.items():
        m = role == r
        if r == "cal" and plan.cal_interval is None:
            (out / name).unlink(missing_ok=True)        # 同一目录改用无校准段重切：删掉上一轮留下的 cal.parquet
            continue
        part = df[m].reset_index(drop=True)
        if r == "val":
            part = part.assign(split_block=block[m].astype(np.int64))
        part.to_parquet(out / name, index=False)
        counts[r] = int(m.sum())
    counts["dropped"] = int((role == "dropped").sum())
    months = lambda t: sorted({str(x)[:7] for x in t.astype("datetime64[M]")})
    blocks = [{"index": i, "start": plan.to_dict()["val_blocks"][i][0], "end": plan.to_dict()["val_blocks"][i][1],
               "rows": int(tv.size), "months": months(tv)} for i, tv in enumerate(val_ts)]
    st_col = "station_id" if "station_id" in df.columns else ("station" if "station" in df.columns else None)
    stations = {}
    if st_col is not None:
        for s, grp in pd.Series(role).groupby(df[st_col].astype(str).to_numpy()):
            vc = grp.value_counts()
            stations[str(s)] = {r: int(vc.get(r, 0)) for r in counts if r != "dropped"}
    manifest = {"generated_by": "pipelines.time_splits", "source": None if source is None else str(source),
                "shift_days": int(shift_days), "ts_col": ts_col, "plan": plan.to_dict(), "counts": counts,
                "blocks": blocks, "stations": stations, "n_nat": int(np.isnat(ts).sum()),
                "inputs_disjoint_after": plan.inputs_disjoint_after, "coverage_note": plan.coverage_note}
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


REQUIRED_SPLIT_KEYS = ("val_blocks", "test_start", "cal_days", "purge_days", "embargo_days")


def load_split_dict(split_yaml=None, m6_config=None) -> dict:
    """--split-yaml：顶层就是 split 字典；--m6-config：读 M6 model_config.yaml 的 finetune.split。二选一。
    两条路都要求 split 字典写全 REQUIRED_SPLIT_KEYS：CLI 不替用户默认 purge/embargo/cal 的天数。"""
    if (split_yaml is None) == (m6_config is None):
        raise ValueError("--split-yaml 与 --m6-config 二选一")
    if split_yaml is not None:
        split = yaml.safe_load(Path(split_yaml).read_text(encoding="utf-8")) or {}
        where = split_yaml
    else:
        cfg = yaml.safe_load(Path(m6_config).read_text(encoding="utf-8")) or {}
        try:
            split = cfg["finetune"]["split"]
        except (KeyError, TypeError):
            raise ValueError(f"{m6_config} 缺 finetune.split") from None
        where = f"{m6_config} 的 finetune.split"
    missing = [k for k in REQUIRED_SPLIT_KEYS if k not in (split or {})]
    if missing:
        raise ValueError(f"{where} 缺少切分键 {missing}：这些键必须逐个写明，CLI 不替它们取默认值")
    return split


def _m6_geometry(m6_config) -> dict:
    """从 M6 model_config.yaml 里取 L/H/step_min（缺哪个就不给哪个），供 CLI 当默认值用。
    这里直接读 yaml 而不 import forecast_backbones.m6.config：pipelines 不依赖任何具体模型目录。"""
    if m6_config is None:
        return {}
    cfg = yaml.safe_load(Path(m6_config).read_text(encoding="utf-8")) or {}
    tf, ds = cfg.get("timesfm3") or {}, cfg.get("dataset") or {}
    got = {"L": tf.get("context_len"), "H": tf.get("pred_len"), "step_min": ds.get("step_min")}
    return {k: int(v) for k, v in got.items() if v is not None}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="按时间轴切 train/valid[/cal]/test.parquet（验证块 purge/embargo，见 split_manifest.json）")
    ap.add_argument("--parquet", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-yaml", default=None); ap.add_argument("--m6-config", default=None)
    ap.add_argument("--ts-col", default="timestamp_win"); ap.add_argument("--shift-days", type=int, default=0)
    # 默认 None：解析后按 显式 flag > --m6-config 里的几何 > 内置默认 的顺序定 L/H/step_min
    ap.add_argument("--L", type=int, default=None); ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--step-min", type=int, default=None)
    a = ap.parse_args(argv)
    split = load_split_dict(a.split_yaml, a.m6_config)
    geom = _m6_geometry(a.m6_config)
    pick = lambda flag, key, fallback: flag if flag is not None else geom.get(key, fallback)
    L, H, step_min = pick(a.L, "L", 672), pick(a.H, "H", 192), pick(a.step_min, "step_min", 15)
    plan = SplitPlan.from_dict(split, shift_days=a.shift_days, L=L, H=H, step_min=step_min)
    df = pd.read_parquet(a.parquet)
    man = write_splits(df, plan, a.out_dir, ts_col=a.ts_col, source=a.parquet, shift_days=a.shift_days)
    c = man["counts"]
    print(f"[time_splits] train={c['train']} val={c['val']} cal={c.get('cal', 0)} test={c['test']} dropped={c['dropped']} "
          f"n_nat={man['n_nat']} → {a.out_dir}")
    for b in man["blocks"]:
        print(f"[time_splits] val block {b['index']}: [{b['start']}, {b['end']}) rows={b['rows']} months={b['months']}")
    if man["coverage_note"]:
        print(f"[time_splits] 提示：{man['coverage_note']}")
    return man


if __name__ == "__main__":
    main()
