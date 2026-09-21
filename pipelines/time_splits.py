# -*- coding: utf-8 -*-
# pipelines/time_splits.py
"""时间轴切分（与模型无关）：验证块列表 + 尾段测试 + 校准段 + purge/embargo。
每行是一个起报时刻 timestamp_win；训练行标签向前伸 H 步、上下文向后伸 L 步，相邻行几乎共用同一段标签，
所以切分只能按时间轴切，并在验证/校准/测试块前后丢掉一段训练行：
  块前 purge_days（≥ H 折算天数）：训练行标签不伸进块；
  块后 purge_days + embargo_days 合计须 ≥ L + H 折算天数（默认 2 + 7 = 9 天 = 672 + 192 步）：
  前 H 步保证训练行标签不与块标签重叠，再往后的 L 步保证训练行 7 天上下文不含块的标签。
校准段固定放在 test_start − purge 之前 cal_days 天；训练行到校准段前 purge 为止。
验证块是全站共用的日历日期（同一天不会一站当训练、另一站当验证）；val_blocks: auto 时按各站实际数据自动放块
（suggest_val_blocks），落盘后 split_resolved.yaml 记下实际用到的固定日期，原样传回即可复现。"""
from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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


def _reject_auto(split: dict) -> None:
    if isinstance(split.get("val_blocks"), str):
        raise ValueError(f"val_blocks={split['val_blocks']!r} 须先按数据解析成固定日期：pipelines.time_splits 的命令行会自动做，"
                         f"解析结果写在输出目录的 split_resolved.yaml（M6 等直接读配置的地方请抄那里的 val_blocks）")
    if int(split.get("min_station_days_for_val") or 0) > 0:
        raise ValueError("min_station_days_for_val 须先按数据解析成站名单：pipelines.time_splits 的命令行会自动做，"
                         "解析结果写在输出目录 split_resolved.yaml 的 train_only_stations（M6 等直接读配置的地方请抄那份名单）")


# 只进训练的站：不参加验证、没有校准段、不参与自动放块；数据除测试起点前 purge 天与测试段外全部是训练行
PER_STATION_KEYS = ("train_only_stations", "min_station_days_for_val", "stations_without_test")


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
    allow_stations_without_val: bool = False   # True：落盘时允许个别站一行验证数据都没有（其数据全进训练）

    @classmethod
    def from_dict(cls, split: dict, shift_days: int = 0, L: int = 672, H: int = 192, step_min: int = 15,
                  allow_empty_blocks: bool = False) -> "SplitPlan":
        """allow_empty_blocks 只给 StationSplitPlan 用：某站一个块都容不下且 allow_stations_without_val 为真时，
        该站的 plan 没有验证块（只剩训练/测试的划分）。"""
        _reject_auto(split)
        if split.get("train_only_stations") or split.get("stations_without_test"):
            raise ValueError("train_only_stations 需要按站切分（stations_without_test 同理）：请按站给 test_start（--test-parquets / --test-start-by-station / "
                             "test_start 写 {站: 日期}），或用 pipelines.time_splits 命令行（会自动转成按站切分）")
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
        if not raw and not allow_empty_blocks:
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
        if train_start is not None and blocks and blocks[0][0] < train_start:
            raise ValueError(f"第一个验证块起点 {blocks[0][0].date()} 早于 train_start {train_start.date()}："
                             f"val_blocks 必须落在训练数据区间之内")
        plan = cls(tuple(blocks), test_start, test_end, cal, purge, embargo, train_start, int(L), int(H), int(step_min),
                   bool(split.get("allow_stations_without_val", False)))
        if not blocks:
            return plan
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
        if len(quarters) >= 2 or not quarters:
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
                "allow_stations_without_val": self.allow_stations_without_val,
                "train_exclusions": [[iso(a), iso(b)] for a, b in self.train_exclusions],
                "inputs_disjoint_after": self.inputs_disjoint_after, "coverage_note": self.coverage_note}


def station_keys(values) -> np.ndarray:
    """站标识归一成字符串：数值型且全是整数（含 1001.0 这种被物化成浮点的 id）→ '1001'；其他 → str。"""
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64).astype(str)
    if arr.dtype.kind == "f":
        if np.isnan(arr).any():
            raise ValueError("站标识列含 NaN")
        if np.all(arr == np.round(arr)):
            return arr.astype(np.int64).astype(str)
        return arr.astype(str)
    out = arr.astype(str)
    # 字符串形态的 '1001.0' 同样归一，与数值列保持一致
    fixed = []
    for x in out.tolist():
        try:
            f = float(x)
            fixed.append(str(int(f)) if f == int(f) and x.strip() != "" else x)
        except ValueError:
            fixed.append(x)
    return np.asarray(fixed, dtype=str)


def _station_key(k) -> str:
    return str(station_keys(np.asarray([k]))[0])


def _feasible_blocks(blocks, test_start: pd.Timestamp, cal_days: int, purge_days: int) -> list[int]:
    """在给定 test_start 下能容下的验证块（全局序号）：块结束 + purge ≤ 校准段起点（无校准段则 ≤ test_start − purge）。"""
    purge = pd.Timedelta(days=int(purge_days))
    guard = test_start - purge - pd.Timedelta(days=int(cal_days))
    return [i for i, (s, e) in enumerate(blocks) if e + purge <= guard]


@dataclass(frozen=True)
class StationSplitPlan:
    """按站不同 test_start 的切分：验证块、purge/embargo、cal_days 全站共用，每站各自一个 SplitPlan。
    某站 test_start 早到容不下某个验证块时，该站跳过那个块（其他站照常），跳过情况记在 skipped_blocks；
    一个块没有任何站容得下、或一个站一个块都容不下，都直接报错。block 序号一律是全局序号。"""
    reference: SplitPlan                      # 用最晚的 test_start 构造，承载全局验证块与校验
    plans: dict                               # station(str) -> SplitPlan（val_blocks 是该站容得下的子集）
    block_index: dict                         # station -> (全局序号, ...)，与该站 plan.val_blocks 逐项对应
    skipped_blocks: dict                      # station -> [跳过的全局序号]
    train_only: frozenset = frozenset()       # 只进训练的站（split.train_only_stations）
    no_test: frozenset = frozenset()          # 没有测试集的站（split.stations_without_test）：test_start 是占位值

    @classmethod
    def from_dict(cls, split: dict, test_starts: dict, shift_days: int = 0, L: int = 672, H: int = 192,
                  step_min: int = 15) -> "StationSplitPlan":
        _reject_auto(split)
        if not test_starts:
            raise ValueError("test_starts 为空：按站切分至少要给一个站的 test_start")
        starts = {_station_key(k): _day(v, f"test_start[{k}]") for k, v in test_starts.items()}
        base = {k: v for k, v in split.items() if k != "test_start" and k not in PER_STATION_KEYS}
        train_only = frozenset(_station_key(x) for x in (split.get("train_only_stations") or []))
        unknown = sorted(train_only - set(starts))
        if unknown:
            raise ValueError(f"train_only_stations 里这些站没有 test_start：{unknown}（名字写错了，或测试集里没有这个站）")
        no_test = frozenset(_station_key(x) for x in (split.get("stations_without_test") or []))
        unknown = sorted(no_test - set(starts))
        if unknown:
            raise ValueError(f"stations_without_test 里这些站没有占位 test_start：{unknown}（请用 reconcile_test_starts 生成）")
        eligible = {s: t for s, t in starts.items() if s not in train_only}
        if not eligible:
            raise ValueError("所有站都被列为只进训练（train_only_stations / min_station_days_for_val），没有站可以做验证")
        latest = max(eligible.values())
        raw_blocks = [(_day(a, "val_blocks"), _day(b, "val_blocks")) for a, b in (split.get("val_blocks") or [])]
        raw_blocks.sort()
        purge_days, cal_days = int(base.get("purge_days", 2)), int(base.get("cal_days", 0))
        orphan = [i for i in range(len(raw_blocks)) if i not in _feasible_blocks(raw_blocks, latest, cal_days, purge_days)]
        if orphan:
            raise ValueError(f"验证块 {orphan} 没有任何站容得下（最晚的 test_start 也只有 {latest.date()}）：请删掉或提前这些块")
        ref = SplitPlan.from_dict({**base, "test_start": latest}, shift_days=shift_days, L=L, H=H, step_min=step_min)
        plans, index, skipped = {}, {}, {}
        for st, t0 in starts.items():
            if st in train_only:
                # 只进训练：没有验证块，也没有校准段（cal_days=0），测试起点前 purge 天之外全是训练行
                plans[st] = SplitPlan.from_dict({**base, "val_blocks": [], "test_start": t0, "cal_days": 0},
                                                shift_days=shift_days, L=L, H=H, step_min=step_min, allow_empty_blocks=True)
                index[st], skipped[st] = (), []
                continue
            keep = _feasible_blocks(raw_blocks, t0, ref.cal_days, ref.purge_days)
            if not keep and not ref.allow_stations_without_val:
                raise ValueError(f"站 {st} 的 test_start={t0.date()} 太早，一个验证块都容不下"
                                 f"（块结束 + purge {ref.purge_days} 天须 ≤ test_start − purge − cal_days {ref.cal_days} 天）：\n"
                                 f"请把该站的验证块提前、改用 val_blocks: auto，或写 allow_stations_without_val: true")
            sub = {**base, "val_blocks": [[str(raw_blocks[i][0].date()), str(raw_blocks[i][1].date())] for i in keep],
                   "test_start": t0}
            plans[st] = SplitPlan.from_dict(sub, shift_days=shift_days, L=L, H=H, step_min=step_min,
                                            allow_empty_blocks=True)
            index[st] = tuple(keep)
            skipped[st] = [i for i in range(len(raw_blocks)) if i not in keep]
        return cls(ref, plans, index, skipped, train_only, no_test)

    @property
    def val_blocks(self) -> tuple:
        return self.reference.val_blocks

    @property
    def stations(self) -> list:
        return sorted(self.plans)

    def assign(self, ts: np.ndarray, station: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ts: 起报时刻；station: 与 ts 等长的站标识。逐站套各自 plan；block 回填成全局序号。未知站直接报错。"""
        t = np.asarray(ts).astype("datetime64[ns]")
        st = station_keys(station)
        if len(t) != len(st):
            raise ValueError(f"ts ({len(t)}) 与 station ({len(st)}) 长度不一致")
        unknown = sorted(set(st.tolist()) - set(self.plans))
        if unknown:
            raise ValueError(f"这些站没有 test_start：{unknown}（每个站都必须给出自己的测试起点）")
        role = np.full(len(t), "dropped", dtype=object)
        block = np.full(len(t), -1, dtype=np.int64)
        for s, plan in self.plans.items():
            m = st == s
            if not m.any():
                continue
            r, b = plan.assign(t[m])
            g = np.array(self.block_index[s] + (-1,), dtype=np.int64)     # 局部序号 → 全局；−1 保持 −1
            role[m], block[m] = r, g[b]
        return role, block

    def verify(self, ts: np.ndarray, station: np.ndarray, role: np.ndarray, block: np.ndarray) -> None:
        """逐站调用 verify_leak_free（按该站实际行）。"""
        t = np.asarray(ts).astype("datetime64[ns]")
        st = station_keys(station)
        for s, plan in self.plans.items():
            m = st == s
            if not m.any():
                continue
            tv = [t[m & (role == "val") & (block == i)] for i in self.block_index[s]]
            tc = t[m & (role == "cal")] if plan.cal_interval is not None else None
            try:
                verify_leak_free(t[m & (role == "train")], tv, tc, plan)
            except AssertionError as e:
                raise AssertionError(f"站 {s}：{e}") from None

    def to_dict(self) -> dict:
        d = self.reference.to_dict()
        iso = lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")
        d["test_start"] = {s: iso(p.test_start) for s, p in self.plans.items()}
        d["cal"] = {s: (None if p.cal_interval is None else [iso(p.cal_interval[0]), iso(p.cal_interval[1])])
                    for s, p in self.plans.items()}
        d["train_exclusions"] = {s: [[iso(a), None if b is None else iso(b)] for a, b in p.train_exclusions]
                                 for s, p in self.plans.items()}
        d["by_station"] = True
        d["val_blocks_by_station"] = {s: list(self.block_index[s]) for s in self.plans}
        d["skipped_val_blocks"] = {s: v for s, v in self.skipped_blocks.items() if v}
        d["train_only_stations"] = sorted(self.train_only)
        d["stations_without_test"] = sorted(self.no_test)
        return d


def assign_rows(plan, ts: np.ndarray, station=None) -> tuple[np.ndarray, np.ndarray]:
    """统一入口：SplitPlan 只看时间；StationSplitPlan 还要站标识。"""
    if isinstance(plan, StationSplitPlan):
        if station is None:
            raise ValueError("按站切分需要站标识列")
        return plan.assign(ts, station)
    return plan.assign(ts)


def verify_rows(plan, ts: np.ndarray, role: np.ndarray, block: np.ndarray, station=None) -> None:
    """统一入口的独立复核：SplitPlan 汇总全表；StationSplitPlan 逐站。"""
    t = np.asarray(ts).astype("datetime64[ns]")
    if isinstance(plan, StationSplitPlan):
        plan.verify(t, station, role, block)
        return
    tv = [t[(role == "val") & (block == i)] for i in range(len(plan.val_blocks))]
    verify_leak_free(t[role == "train"], tv, t[role == "cal"] if plan.cal_interval is not None else None, plan)


def suggest_val_blocks(days_by_station: dict, test_starts: dict, n_blocks: int, block_days: int, purge_days: int,
                       embargo_days: int, cal_days: int, train_start=None) -> tuple[list, list]:
    """按数据自动放全站共用的验证块。days_by_station：{站: 有数据的日期数组}；test_starts：{站: 测试起点}。
    可用区间 = [所有站最早有数据的一天, 所有站里最晚的 测试起点 − 2×purge − cal_days)，等分成 n_blocks 段，
    每段里挑一个 block_days 天的块，使"块内有数据且容得下这个块的站·天数"最大，同分取最靠段中心的；
    相邻两块至少隔 2×purge + embargo 天（两块的剔除区不重叠）。某段里最好的块也凑不够半块（block_days/2 个站·天）
    就不放块（这一段基本没有数据）。
    返回 (blocks, notes)：blocks = [(start, end), ...]，notes = 需要提示用户的话。"""
    n, b = int(n_blocks), int(block_days)
    if n < 1:
        raise ValueError(f"n_val_blocks={n} 须 ≥ 1")
    if b < 2:
        raise ValueError(f"val_block_days={b} 须 ≥ 2")
    one = np.timedelta64(1, "D")
    lim = {str(s): np.datetime64(pd.Timestamp(t).date(), "D") - np.timedelta64(2 * int(purge_days) + int(cal_days), "D")
           for s, t in test_starts.items()}
    ts0 = None if train_start is None else np.datetime64(pd.Timestamp(train_start).date(), "D")
    usable = {}
    for s, d in days_by_station.items():
        if str(s) not in lim:
            raise ValueError(f"站 {s} 没有 test_start")
        d = np.unique(np.asarray(d).astype("datetime64[D]"))
        d = d[d < lim[str(s)]]
        if ts0 is not None:
            d = d[d >= ts0]
        if d.size:
            usable[str(s)] = d
    if not usable:
        raise ValueError("没有任何站在各自测试起点之前有可放验证块的数据")
    g0 = min(d[0] for d in usable.values())
    g1 = max(lim[s] for s in usable)
    span = int((g1 - g0) / one)
    if span < n * b:
        raise ValueError(f"可放验证块的区间 [{g0}, {g1}) 只有 {span} 天，放不下 {n} 块 × {b} 天："
                         f"减少 n_val_blocks 或 val_block_days")
    starts = np.arange(0, span - b + 1)
    score = np.zeros(starts.size, dtype=np.int64)
    for s, d in usable.items():
        ind = np.zeros(span, dtype=np.int64)
        ind[((d - g0) / one).astype(np.int64)] = 1
        cs = np.concatenate([[0], np.cumsum(ind)])
        score += (cs[starts + b] - cs[starts]) * ((starts + b) <= int((lim[s] - g0) / one))
    edges = [round(k * span / n) for k in range(n + 1)]
    gap = 2 * int(purge_days) + int(embargo_days)
    chosen, notes, prev_end = [], [], None
    for k in range(n):
        cand = starts[(starts >= edges[k]) & (starts + b <= edges[k + 1])]
        if prev_end is not None:
            cand = cand[cand >= prev_end + gap]
        seg = f"[{g0 + edges[k] * one}, {g0 + edges[k + 1] * one})"
        if cand.size == 0 or score[cand].max() < math.ceil(b / 2):
            notes.append(f"第 {k + 1} 段 {seg} 里可用数据不到半块，这一段不放验证块")
            continue
        best = cand[score[cand] == score[cand].max()]
        centre = (edges[k] + edges[k + 1] - b) / 2
        a = int(best[np.argmin(np.abs(best - centre))])
        chosen.append((a, a + b)); prev_end = a + b
    if not chosen:
        raise ValueError("自动放块失败：每一段都没有可用数据")
    if len(chosen) < n:
        notes.append(f"要求 {n} 块，实际放了 {len(chosen)} 块")
    blocks = [(pd.Timestamp(g0 + a * one), pd.Timestamp(g0 + e * one)) for a, e in chosen]
    return blocks, notes


def resolve_auto_blocks(ts: np.ndarray, station: np.ndarray, test_starts: dict, split: dict) -> tuple[list, list]:
    """val_blocks: auto → 固定日期列表 [[start, end], ...]（字符串）。需要 split 里的 n_val_blocks / val_block_days。"""
    missing = [k for k in ("n_val_blocks", "val_block_days") if k not in split]
    if missing:
        raise ValueError(f"val_blocks: auto 需要同时写 {missing}")
    t = np.asarray(ts).astype("datetime64[ns]")
    st = station_keys(station)
    ok = ~np.isnat(t)
    day = t[ok].astype("datetime64[D]"); st = st[ok]
    days = {s: day[st == s] for s in sorted(set(st.tolist()))}
    starts = {_station_key(k): v for k, v in test_starts.items()}
    lacking = sorted(set(days) - set(starts))
    if lacking:
        raise ValueError(f"这些站没有 test_start：{lacking}（每个站都必须给出自己的测试起点）")
    blocks, notes = suggest_val_blocks(days, {s: starts[s] for s in days}, int(split["n_val_blocks"]),
                                       int(split["val_block_days"]), int(split["purge_days"]),
                                       int(split["embargo_days"]), int(split["cal_days"]),
                                       train_start=split.get("train_start"))
    return [[a.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")] for a, e in blocks], notes


def reconcile_test_starts(ts: np.ndarray, station: np.ndarray, test_starts: dict, purge_days: int) -> tuple[dict, list, list]:
    """把按站的 test_start 与训练数据里实际出现的站对齐。返回 (starts, test_only, no_test)：
    test_only = 只在测试集里、训练数据里没有的站 → 从 starts 去掉（它们没有任何训练行，切分用不上）；
    no_test = 只在训练数据里、没有测试集的站 → 给占位 test_start = 该站最后一天 + 1 天 + purge_days，
    这样它没有测试行、末尾也不剔除任何行（test_start − purge 正好是最后一天的次日），可用数据一直到末尾。"""
    t = np.asarray(ts).astype("datetime64[ns]")
    st = station_keys(station)
    starts = {_station_key(k): v for k, v in test_starts.items()}
    present = sorted(set(st[~np.isnat(t)].tolist()))
    test_only = sorted(set(starts) - set(present))
    no_test = sorted(set(present) - set(starts))
    out = {s: starts[s] for s in present if s in starts}
    for s in no_test:
        last = pd.Timestamp(t[(st == s) & ~np.isnat(t)].max()).normalize()
        out[s] = (last + pd.Timedelta(days=1 + int(purge_days))).strftime("%Y-%m-%d")
    return out, test_only, no_test


def days_before_test(ts: np.ndarray, station: np.ndarray, test_starts: dict) -> dict:
    """每站在自己测试起点之前有数据的天数（有行的日期数，数据中间有缺口时不按日历跨度算）。"""
    t = np.asarray(ts).astype("datetime64[ns]")
    st = station_keys(station)
    starts = {_station_key(k): _dt64(_day(v, f"test_start[{k}]")) for k, v in test_starts.items()}
    out = {}
    for s in sorted(set(st.tolist())):
        if s not in starts:
            continue
        m = (st == s) & ~np.isnat(t) & (t < starts[s])
        out[s] = int(np.unique(t[m].astype("datetime64[D]")).size)
    return out


def station_report(ts: np.ndarray, station: np.ndarray, role: np.ndarray, block: np.ndarray, plan) -> pd.DataFrame:
    """逐站覆盖：数据起止、测试起点、参加的验证块、各角色天数（有该角色行的日期数）、训练/验证占测试前行数的比例。"""
    day = np.asarray(ts).astype("datetime64[D]")
    rows = []
    for s in sorted(set(station.tolist())):
        m = station == s
        d, r, bk = day[m], role[m], block[m]
        ok = ~np.isnat(d)
        nd = lambda mask: int(np.unique(d[mask & ok]).size)
        t0 = plan.plans[s].test_start if isinstance(plan, StationSplitPlan) else plan.test_start
        pre = int((r != "test").sum())
        rows.append({"station": s, "data_start": str(d[ok].min()) if ok.any() else None,
                     "data_end": str(d[ok].max()) if ok.any() else None,
                     "test_start": ("无测试集" if isinstance(plan, StationSplitPlan) and s in plan.no_test
                                    else pd.Timestamp(t0).strftime("%Y-%m-%d")),
                     "val_blocks": sorted(int(x) for x in set(bk[r == "val"].tolist())),
                     "skipped_blocks": list(plan.skipped_blocks.get(s, [])) if isinstance(plan, StationSplitPlan) else [],
                     "train_only": bool(isinstance(plan, StationSplitPlan) and s in plan.train_only),
                     "train_days": nd(r == "train"), "val_days": nd(r == "val"), "cal_days": nd(r == "cal"),
                     "dropped_days": nd(r == "dropped"), "test_days": nd(r == "test"),
                     "train_share": round(float((r == "train").sum()) / pre, 3) if pre else 0.0,
                     "val_share": round(float((r == "val").sum()) / pre, 3) if pre else 0.0})
    return pd.DataFrame(rows)


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


def has_cal(plan) -> bool:
    if isinstance(plan, StationSplitPlan):
        return any(p.cal_interval is not None for p in plan.plans.values())
    return plan.cal_interval is not None


def _station_col(df: pd.DataFrame, station_col=None):
    if station_col is not None:
        if station_col not in df.columns:
            raise ValueError(f"站标识列 {station_col!r} 不在数据里（列：{list(df.columns)[:12]}…）")
        return station_col
    return "station_id" if "station_id" in df.columns else ("station" if "station" in df.columns else None)


def _ts_array(col: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(col)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts.to_numpy().astype("datetime64[ns]")


def write_splits(df: pd.DataFrame, plan, out_dir, ts_col: str = "timestamp_win",
                 source=None, shift_days: int = 0, station_col=None, write: bool = True, extra: dict | None = None) -> dict:
    """按 plan（SplitPlan 或 StationSplitPlan）把原始行表落成 train/valid[/cal]/test.parquet（按时间稳定排序）
    + split_manifest.json。返回 manifest（含逐站覆盖 station_report）。按站切分时站标识列取 station_col，
    缺省自动认 station_id / station。write=False 只算不落盘（--dry-run）。
    有站一行验证数据都没有时报错，除非 plan.allow_stations_without_val。"""
    out = Path(out_dir) if out_dir is not None else None
    if write:
        out.mkdir(parents=True, exist_ok=True)
    ts = _ts_array(df[ts_col])
    st_col = _station_col(df, station_col)
    if isinstance(plan, StationSplitPlan) and st_col is None:
        raise ValueError("按站切分需要站标识列：数据里没有 station_id / station，请用 station_col 指定")
    station = station_keys(df[st_col].to_numpy()) if st_col is not None else None
    role, block = assign_rows(plan, ts, station)
    order = np.argsort(ts, kind="stable")
    df, ts, role, block = df.iloc[order].reset_index(drop=True), ts[order], role[order], block[order]
    station = None if station is None else station[order]
    verify_rows(plan, ts, role, block, station)
    val_ts = [ts[(role == "val") & (block == i)] for i in range(len(plan.val_blocks))]
    ref = plan.reference if isinstance(plan, StationSplitPlan) else plan
    report = station_report(ts, station, role, block, plan) if station is not None else None
    if report is not None and not ref.allow_stations_without_val:
        bad = report[(report["val_days"] == 0) & ~report["train_only"]]
        if len(bad):
            detail = "；".join(f"{r.station}（数据 {r.data_start}…{r.data_end}，测试起点 {r.test_start}）"
                              for r in bad.itertuples(index=False))
            raise ValueError(f"这些站一行验证数据都没有：{detail}。它们的数据段不落在任何验证块里。"
                             f"办法：val_blocks 改成 auto 按数据自动放块、调整块日期，"
                             f"把这些站写进 train_only_stations / 调高 min_station_days_for_val（只进训练），"
                             f"或写 allow_stations_without_val: true（放行）")

    files = {"train": "train.parquet", "val": "valid.parquet", "cal": "cal.parquet", "test": "test.parquet"}
    counts = {}
    for r, name in files.items():
        m = role == r
        if r == "cal" and not has_cal(plan):
            if write:
                (out / name).unlink(missing_ok=True)    # 同一目录改用无校准段重切：删掉上一轮留下的 cal.parquet
            continue
        if write:
            part = df[m].reset_index(drop=True)
            if r == "val":
                part = part.assign(split_block=block[m].astype(np.int64))
            part.to_parquet(out / name, index=False)
        counts[r] = int(m.sum())
    counts["dropped"] = int((role == "dropped").sum())
    months = lambda t: sorted({str(x)[:7] for x in t.astype("datetime64[M]")})
    blocks = [{"index": i, "start": plan.to_dict()["val_blocks"][i][0], "end": plan.to_dict()["val_blocks"][i][1],
               "rows": int(tv.size), "months": months(tv)} for i, tv in enumerate(val_ts)]
    stations = {}
    if station is not None:
        for s, grp in pd.Series(role).groupby(station):
            vc = grp.value_counts()
            stations[str(s)] = {r: int(vc.get(r, 0)) for r in counts if r != "dropped"}
            stations[str(s)]["val_blocks"] = sorted(int(b) for b in set(block[(station == s) & (role == "val")].tolist()))
    manifest = {"generated_by": "pipelines.time_splits",
                "source": None if source is None else ([str(x) for x in source] if isinstance(source, (list, tuple)) else str(source)),
                "shift_days": int(shift_days), "ts_col": ts_col, "station_col": st_col,
                "by_station": isinstance(plan, StationSplitPlan),
                "plan": plan.to_dict(), "counts": counts,
                "blocks": blocks, "stations": stations, "n_nat": int(np.isnat(ts).sum()),
                "inputs_disjoint_after": ref.inputs_disjoint_after, "coverage_note": ref.coverage_note,
                "station_report": [] if report is None else report.to_dict("records"), **(extra or {})}
    if write:
        (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


REQUIRED_SPLIT_KEYS = ("val_blocks", "test_start", "cal_days", "purge_days", "embargo_days")


def load_split_dict(split_yaml=None, m6_config=None, require_test_start: bool = True) -> dict:
    """--split-yaml：顶层就是 split 字典；--m6-config：读 M6 config/configs.yaml 的 finetune.split。二选一。
    两条路都要求 split 字典写全 REQUIRED_SPLIT_KEYS：CLI 不替用户默认 purge/embargo/cal 的天数。
    require_test_start=False（按站给 test_start 时）允许 yaml 里不写 test_start 或写 null。"""
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
    need = REQUIRED_SPLIT_KEYS if require_test_start else tuple(k for k in REQUIRED_SPLIT_KEYS if k != "test_start")
    missing = [k for k in need if k not in (split or {})]
    if missing:
        raise ValueError(f"{where} 缺少切分键 {missing}：这些键必须逐个写明，CLI 不替它们取默认值")
    return split


def _m6_geometry(m6_config) -> dict:
    """从 M6 config/configs.yaml 里取 L/H/step_min（缺哪个就不给哪个），供 CLI 当默认值用。
    这里直接读 yaml 而不 import forecast_backbones.m6.config：pipelines 不依赖任何具体模型目录。"""
    if m6_config is None:
        return {}
    cfg = yaml.safe_load(Path(m6_config).read_text(encoding="utf-8")) or {}
    tf, ds = cfg.get("timesfm3") or {}, cfg.get("dataset") or {}
    got = {"L": tf.get("context_len"), "H": tf.get("pred_len"), "step_min": ds.get("step_min")}
    return {k: int(v) for k, v in got.items() if v is not None}


def expand_paths(specs, what: str = "--parquet") -> list[str]:
    """路径 / 通配符 / 列表 → 去重后按字典序排好的文件列表；任何一项没匹配到文件直接报错。"""
    items = list(specs) if isinstance(specs, (list, tuple)) else [specs]
    paths = []
    for pat in items:
        hit = sorted(glob.glob(str(pat)))
        if not hit:
            raise FileNotFoundError(f"{what} {pat!r} 没匹配到文件")
        paths.extend(hit)
    return sorted(dict.fromkeys(paths))


def read_parquets(specs) -> tuple[pd.DataFrame, list[str]]:
    """读若干 parquet 拼成一张表（一站一个文件时用）。各文件列集合必须一致，否则报错（不静默补 NaN）。"""
    paths = expand_paths(specs, "--parquet")
    frames = [pd.read_parquet(p) for p in paths]
    cols0 = set(frames[0].columns)
    for pth, f in zip(paths[1:], frames[1:]):
        if set(f.columns) != cols0:
            raise ValueError(f"{pth} 的列与 {paths[0]} 不一致：多出 {sorted(set(f.columns) - cols0)}，"
                             f"缺少 {sorted(cols0 - set(f.columns))}")
    return (frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)), paths


def load_test_starts(test_parquets, ts_col: str = "timestamp_win", station_col=None) -> dict:
    """从若干测试集 parquet（可含通配符）推每站的 test_start：该站最早起报时刻所在的整天。返回 {station: 'YYYY-MM-DD'}。"""
    paths = expand_paths(test_parquets, "--test-parquets")
    starts: dict = {}
    for p in paths:
        cols = pq.read_schema(p).names
        sc = _station_col(pd.DataFrame(columns=cols), station_col)
        if sc is None:
            raise ValueError(f"{p} 没有站标识列（station_id / station），无法按站推 test_start；可用 --station-col 指定")
        if ts_col not in cols:
            raise ValueError(f"{p} 没有时间列 {ts_col!r}")
        df = pd.read_parquet(p, columns=[ts_col, sc])
        ts = pd.to_datetime(df[ts_col])
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
        for st, t0 in ts.groupby(station_keys(df[sc].to_numpy())).min().items():
            if pd.isna(t0):
                raise ValueError(f"{p} 里站 {st} 的时间全是 NaT")
            day = pd.Timestamp(t0).normalize()
            starts[str(st)] = min(starts.get(str(st), day), day)
    return {k: v.strftime("%Y-%m-%d") for k, v in sorted(starts.items())}


def load_test_starts_file(path) -> dict:
    """yaml/json 映射文件：{station: 'YYYY-MM-DD'}。"""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path} 须是非空的 {{station: date}} 映射")
    return {_station_key(k): str(v) for k, v in data.items()}


def _resolved_yaml(plan, source_yaml, n_files: int, shift_days: int) -> str:
    """实际用到的固定切分配置：val_blocks / test_start 都是具体日期（已含 --shift-days），原样 --split-yaml 即可复现。"""
    d = plan.to_dict()
    ref = plan.reference if isinstance(plan, StationSplitPlan) else plan
    out = {"train_start": d["train_start"], "val_blocks": d["val_blocks"], "test_start": d["test_start"],
           "test_end": d["test_end"], "cal_days": ref.cal_days, "purge_days": ref.purge_days,
           "embargo_days": ref.embargo_days}
    if ref.allow_stations_without_val:
        out["allow_stations_without_val"] = True
    if isinstance(plan, StationSplitPlan) and plan.train_only:
        out["train_only_stations"] = sorted(plan.train_only)
    if isinstance(plan, StationSplitPlan) and plan.no_test:
        out["stations_without_test"] = sorted(plan.no_test)
    head = (f"# 由 pipelines.time_splits 生成：源配置 {source_yaml}，数据 {n_files} 个文件"
            + (f"，已含 --shift-days {shift_days}" if shift_days else "") + "。\n"
            "# val_blocks / test_start 是这次实际用到的固定日期；原样 --split-yaml 本文件（不带 --test-parquets）即可复现同一切分。\n"
            + ("# stations_without_test 里的站没有测试集，它们的 test_start 是占位值（最后一天 + 1 + purge_days），不产生测试行、不剔除末尾数据。\n"
               if isinstance(plan, StationSplitPlan) and plan.no_test else ""))
    return head + yaml.safe_dump(out, allow_unicode=True, sort_keys=False)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="按时间轴切 train/valid[/cal]/test.parquet（验证块 purge/embargo，见 split_manifest.json）")
    ap.add_argument("--parquet", required=True, nargs="+", help="原始行表（可通配符、可多个；一站一个文件时直接给通配符）")
    ap.add_argument("--out-dir", default=None, help="输出目录；--dry-run 时可不给")
    ap.add_argument("--split-yaml", default=None); ap.add_argument("--m6-config", default=None)
    ap.add_argument("--ts-col", default="timestamp_win"); ap.add_argument("--shift-days", type=int, default=0)
    # 默认 None：解析后按 显式 flag > --m6-config 里的几何 > 内置默认 的顺序定 L/H/step_min
    ap.add_argument("--L", type=int, default=None); ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--step-min", type=int, default=None)
    # 按站不同 test_start：--test-parquets / --test-start-by-station / 切分配置里 test_start 写 {站: 日期}，三选一
    ap.add_argument("--test-parquets", nargs="+", default=None,
                    help="现有测试集 parquet（可通配符，可多个）；每站 test_start = 该站在这些文件里最早起报时刻所在的整天")
    ap.add_argument("--test-start-by-station", default=None, help="yaml/json 映射文件 {station: 'YYYY-MM-DD'}")
    ap.add_argument("--station-col", default=None, help="站标识列名；缺省自动认 station_id / station")
    ap.add_argument("--dry-run", action="store_true", help="只打印验证块与逐站覆盖报告，不写任何文件")
    a = ap.parse_args(argv)
    if a.out_dir is None and not a.dry_run:
        raise ValueError("缺 --out-dir（只看报告请加 --dry-run）")
    if a.test_parquets is not None and a.test_start_by_station is not None:
        raise ValueError("--test-parquets 与 --test-start-by-station 二选一")
    flags = a.test_parquets is not None or a.test_start_by_station is not None
    split = load_split_dict(a.split_yaml, a.m6_config, require_test_start=not flags)
    geom = _m6_geometry(a.m6_config)
    pick = lambda flag, key, fallback: flag if flag is not None else geom.get(key, fallback)
    L, H, step_min = pick(a.L, "L", 672), pick(a.H, "H", 192), pick(a.step_min, "step_min", 15)

    df, paths = read_parquets(a.parquet)
    if a.ts_col not in df.columns:
        raise ValueError(f"数据里没有时间列 {a.ts_col!r}")
    ts = _ts_array(df[a.ts_col])
    st_col = _station_col(df, a.station_col)
    station = station_keys(df[st_col].to_numpy()) if st_col is not None else None

    t0 = split.get("test_start")
    if flags and t0 is not None:
        raise ValueError(f"split 里已写 test_start={t0!r}，又按站给了 test_start：删掉其一")
    if a.test_parquets is not None:
        starts = load_test_starts(a.test_parquets, a.ts_col, a.station_col)
    elif a.test_start_by_station is not None:
        starts = load_test_starts_file(a.test_start_by_station)
    elif isinstance(t0, dict):
        starts = {_station_key(k): str(v) for k, v in t0.items()}
    else:
        starts = None
    if starts is not None and station is None:
        raise ValueError("按站给了 test_start，但数据里没有站标识列（station_id / station），请用 --station-col 指定")

    test_only: list = []
    if starts is not None:
        starts, test_only, no_test = reconcile_test_starts(ts, station, starts, int(split["purge_days"]))
        for s in test_only:
            print(f"[time_splits] 站 {s} 只在测试集里、训练数据里没有：切分时忽略（它在测试里是模型没见过的站）")
        for s in no_test:
            print(f"[time_splits] 站 {s} 没有测试集：不留测试段，数据一直用到它的最后一天，照常参加训练/验证")
        split = {**split, "stations_without_test": sorted(set(split.get("stations_without_test") or []) | set(no_test))}

    explicit = [_station_key(x) for x in (split.get("train_only_stations") or [])]
    min_days = int(split.get("min_station_days_for_val") or 0)
    if min_days < 0:
        raise ValueError(f"min_station_days_for_val={min_days} 不能为负")
    if explicit or min_days:
        if station is None:
            raise ValueError("train_only_stations / min_station_days_for_val 需要站标识列（station_id / station），请用 --station-col 指定")
        if starts is None:                                    # 单一 test_start → 转成按站切分，各站同一个起点
            starts = {s: t0 for s in sorted(set(station.tolist()))}
        missing = sorted(set(explicit) - set(station.tolist()))
        if missing:
            raise ValueError(f"train_only_stations 里这些站在训练数据里没有：{missing}")
        nd = days_before_test(ts, station, starts)
        short = sorted(s for s, n in nd.items() if n < min_days) if min_days else []
        for s in short:
            print(f"[time_splits] 站 {s} 测试起点前只有 {nd[s]} 天数据（< min_station_days_for_val={min_days}）：只进训练")
        for s in sorted(set(explicit) - set(short)):
            print(f"[time_splits] 站 {s} 在 train_only_stations 里：只进训练")
        split = {**{k: v for k, v in split.items() if k != "min_station_days_for_val"},
                 "train_only_stations": sorted(set(explicit) | set(short))}
    train_only = set(split.get("train_only_stations") or [])
    if train_only and station is not None and set(station.tolist()) <= train_only:
        raise ValueError(f"所有站都被列为只进训练（train_only_stations / min_station_days_for_val={min_days}），"
                         f"没有站可以做验证：调低阈值或少列几个站")

    if isinstance(split.get("val_blocks"), str):
        if split["val_blocks"] != "auto":
            raise ValueError(f"val_blocks 只能是日期列表或 auto，收到 {split['val_blocks']!r}")
        if a.shift_days:
            raise ValueError("val_blocks: auto 不能与 --shift-days 同用：第二折请用第一折落的 split_resolved.yaml 再加 --shift-days")
        if starts is not None:
            per, st_auto = {s: v for s, v in starts.items() if s not in train_only}, station
        elif station is not None:
            per, st_auto = {s: t0 for s in set(station.tolist())}, station
        else:
            per, st_auto = {"__all__": t0}, np.full(len(ts), "__all__")
        keep_rows = ~np.isin(st_auto, sorted(train_only)) if train_only else np.ones(len(ts), bool)
        blocks, notes = resolve_auto_blocks(ts[keep_rows], st_auto[keep_rows], per, split)   # 只进训练的站不参与放块
        split = {**split, "val_blocks": blocks}
        print(f"[time_splits] val_blocks: auto → {blocks}")
        for n in notes:
            print(f"[time_splits] 提示：{n}")

    if starts is not None:
        plan = StationSplitPlan.from_dict(split, starts, shift_days=a.shift_days, L=L, H=H, step_min=step_min)
    else:
        plan = SplitPlan.from_dict(split, shift_days=a.shift_days, L=L, H=H, step_min=step_min)
    man = write_splits(df, plan, a.out_dir, ts_col=a.ts_col, source=paths, shift_days=a.shift_days,
                       station_col=a.station_col, write=not a.dry_run, extra={"ignored_test_only_stations": test_only})
    c = man["counts"]
    print(f"[time_splits] train={c['train']} val={c['val']} cal={c.get('cal', 0)} test={c['test']} dropped={c['dropped']} "
          f"n_nat={man['n_nat']}" + (" （dry-run，未写文件）" if a.dry_run else f" → {a.out_dir}"))
    for b in man["blocks"]:
        print(f"[time_splits] val block {b['index']}: [{b['start']}, {b['end']}) rows={b['rows']} months={b['months']}")
    if man["station_report"]:
        rep_df = pd.DataFrame(man["station_report"]).rename(columns={
            "station": "站", "data_start": "数据起", "data_end": "数据止", "test_start": "测试起点", "val_blocks": "参加的块", "skipped_blocks": "跳过的块(测试太早)", "train_only": "只进训练",
            "train_days": "训练天", "val_days": "验证天", "cal_days": "校准天", "dropped_days": "剔除天", "test_days": "测试天",
            "train_share": "训练占比", "val_share": "验证占比"})
        if not has_cal(plan):
            rep_df = rep_df.drop(columns=["校准天"])
        print("[time_splits] 逐站覆盖（天数 = 有该角色行的日期数；占比 = 占测试前行数）：")
        print(rep_df.to_string(index=False))
    if man["coverage_note"]:
        print(f"[time_splits] 提示：{man['coverage_note']}")
    if not a.dry_run:
        src = a.split_yaml or f"{a.m6_config} 的 finetune.split"
        (Path(a.out_dir) / "split_resolved.yaml").write_text(_resolved_yaml(plan, src, len(paths), a.shift_days),
                                                             encoding="utf-8")
        print(f"[time_splits] 实际用到的固定切分配置 → {Path(a.out_dir) / 'split_resolved.yaml'}")
    return man


if __name__ == "__main__":
    main()
