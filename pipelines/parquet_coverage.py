# -*- coding: utf-8 -*-
# pipelines/parquet_coverage.py
"""逐文件 / 逐站数据覆盖体检：起止日期、跨度、每月行数与天数、15 分钟网格完整度、最大缺口。
只读时间列和站列（不碰功率数组列），几十个大文件也只要几秒。切分前先跑一遍，再决定验证块放哪、哪些站只进训练。

用法：
  python -m pipelines.parquet_coverage data/raw                      # 目录（不递归）
  python -m pipelines.parquet_coverage "data/raw/*.parquet" a.parquet --recursive --out-dir reports/coverage
  可选：--ts-col timestamp_win  --station-col station_id  --step-min 15

stdout：逐文件逐站汇总表、每月行数矩阵、每月有数据的站数；--out-dir（默认 <第一个目录>/_coverage）写四个 CSV：
  coverage_summary.csv            一行一个（文件, 站）
  coverage_monthly_rows.csv       每月行数矩阵
  coverage_monthly_days.csv       每月有数据的天数矩阵
  coverage_stations_per_month.csv 每月有数据的站数 / 行数（放验证块时看这个）"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from .time_splits import station_keys
except ImportError:
    from time_splits import station_keys

STATION_COLS = ("station_id", "station")


def expand(specs, recursive: bool) -> list[str]:
    out = []
    for s in specs:
        p = Path(s)
        if p.is_dir():
            out += sorted(str(x) for x in (p.rglob("*.parquet") if recursive else p.glob("*.parquet")))
        else:
            hit = sorted(glob.glob(s, recursive=recursive))
            if not hit:
                raise FileNotFoundError(f"{s!r} 没匹配到文件")
            out += [h for h in hit if h.endswith(".parquet")]
    out = sorted(dict.fromkeys(out))
    if not out:
        raise FileNotFoundError(f"{specs} 下没有 .parquet")
    return out


def to_datetime64(col: pd.Series) -> np.ndarray:
    """时间列：时间戳 / 文本 / epoch（秒、毫秒、微秒、纳秒按量级判断）→ datetime64[ns]（去 tz）。"""
    if pd.api.types.is_numeric_dtype(col):
        v = col.astype("float64")
        m = np.nanmedian(v)
        unit = "ns" if m > 1e17 else "us" if m > 1e14 else "ms" if m > 1e11 else "s"
        ts = pd.to_datetime(v, unit=unit, errors="coerce")
    else:
        ts = pd.to_datetime(col, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts.to_numpy().astype("datetime64[ns]")


def read_keys(path: str, ts_col: str, station_col: str | None) -> tuple[np.ndarray, np.ndarray, str]:
    names = pq.read_schema(path).names
    if ts_col not in names:
        raise ValueError(f"{path} 没有时间列 {ts_col!r}（列：{names[:12]}…）")
    sc = station_col if station_col else next((c for c in STATION_COLS if c in names), None)
    if station_col and station_col not in names:
        raise ValueError(f"{path} 没有站列 {station_col!r}")
    tbl = pq.read_table(path, columns=[ts_col] + ([sc] if sc else []))
    df = tbl.to_pandas()
    ts = to_datetime64(df[ts_col])
    st = station_keys(df[sc].to_numpy()) if sc else np.full(len(df), Path(path).stem)
    return ts, st, (sc or "<文件名>")


def summarize(ts: np.ndarray, step_min: int) -> dict:
    ok = ~np.isnat(ts)
    t = ts[ok]
    if t.size == 0:
        return {"n_rows": int(ts.size), "n_nat": int((~ok).sum())}
    day = t.astype("datetime64[D]")
    days = np.unique(day)
    gaps = np.diff(days).astype(int) - 1 if days.size > 1 else np.array([0])
    span = int((days[-1] - days[0]).astype(int)) + 1
    per_day = 1440 // int(step_min)
    return {"n_rows": int(ts.size), "n_nat": int((~ok).sum()),
            "start": str(days[0]), "end": str(days[-1]), "span_days": span, "days_with_data": int(days.size),
            "missing_days": span - int(days.size), "max_gap_days": int(gaps.max()) if gaps.size else 0,
            "rows_per_day": round(t.size / days.size, 1),
            "grid_pct": round(100.0 * t.size / (days.size * per_day), 1),
            "dup_ts": int(t.size - np.unique(t).size)}


def monthly(ts: np.ndarray) -> tuple[pd.Series, pd.Series]:
    t = pd.DatetimeIndex(ts[~np.isnat(ts)])
    per = t.to_period("M")
    rows = pd.Series(1, index=per).groupby(level=0).sum()
    days = pd.Series(t.normalize(), index=per).groupby(level=0).nunique()
    return rows, days


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="parquet 行表覆盖体检（逐文件逐站：起止、每月行数/天数、网格完整度、缺口）")
    ap.add_argument("paths", nargs="+", help="目录、文件或通配符（可多个）")
    ap.add_argument("--ts-col", default="timestamp_win")
    ap.add_argument("--station-col", default=None, help="站列名；缺省自动认 station_id / station，都没有则以文件名当站")
    ap.add_argument("--step-min", type=int, default=15, help="起报间隔（分钟），用来算网格完整度；每天 1440/step_min 行为 100%%")
    ap.add_argument("--out-dir", default=None, help="CSV 输出目录；默认 <第一个目录>/_coverage")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    a = ap.parse_args(argv)

    files = expand(a.paths, a.recursive)
    rows, mrows, mdays = [], {}, {}
    for f in files:
        ts, st, sc = read_keys(f, a.ts_col, a.station_col)
        for s in sorted(set(st.tolist())):
            m = st == s
            key = (Path(f).name, s)
            rec = {"file": key[0], "station": s, "station_col": sc, **summarize(ts[m], a.step_min)}
            rows.append(rec)
            r, d = monthly(ts[m])
            mrows[key], mdays[key] = r, d
    summ = pd.DataFrame(rows)
    idx = pd.MultiIndex.from_tuples(list(mrows), names=["file", "station"])
    M_rows = pd.DataFrame(mrows).T.reindex(idx).fillna(0).astype(int)
    M_days = pd.DataFrame(mdays).T.reindex(idx).fillna(0).astype(int)
    M_rows = M_rows[sorted(M_rows.columns)]; M_days = M_days[sorted(M_days.columns)]
    M_rows.columns = [str(c) for c in M_rows.columns]; M_days.columns = [str(c) for c in M_days.columns]
    per_month = pd.DataFrame({"stations_with_data": (M_rows > 0).sum(0), "rows": M_rows.sum(0),
                              "stations_full_month": (M_days >= 28).sum(0)})

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)
    print(f"== 文件 {len(files)} 个，（文件, 站）{len(summ)} 组；时间列 {a.ts_col!r}；网格 {1440 // a.step_min} 行/天 = 100%")
    print(summ.drop(columns=["station_col"]).to_string(index=False))
    if "span_days" in summ:
        b = pd.cut(summ["span_days"], [0, 92, 183, 366, 10_000], labels=["≤3月", "3–6月", "6–12月", ">12月"])
        print("== 跨度分布（组数）：" + "，".join(f"{k}: {v}" for k, v in b.value_counts().sort_index().items()))
        print(f"== 全局起止：{summ['start'].min()} → {summ['end'].max()}；"
              f"缺口>30天的组：{int((summ['max_gap_days'] > 30).sum())}；网格<90%的组：{int((summ['grid_pct'] < 90).sum())}；"
              f"有重复时间戳的组：{int((summ['dup_ts'] > 0).sum())}")
    print("== 每月行数（行 = 文件|站）：")
    print(M_rows.to_string())
    print("== 每月有数据的站数 / 总行数 / 当月≥28天的站数：")
    print(per_month.T.to_string())
    if not a.no_csv:
        out = Path(a.out_dir) if a.out_dir else (Path(a.paths[0]) if Path(a.paths[0]).is_dir() else Path(files[0]).parent) / "_coverage"
        out.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out / "coverage_summary.csv", index=False)
        M_rows.to_csv(out / "coverage_monthly_rows.csv"); M_days.to_csv(out / "coverage_monthly_days.csv")
        per_month.to_csv(out / "coverage_stations_per_month.csv")
        print(f"== CSV → {out}")
    return {"summary": summ, "monthly_rows": M_rows, "monthly_days": M_days, "per_month": per_month}


if __name__ == "__main__":
    main()
