# -*- coding: utf-8 -*-
# tools/parquet_coverage.py —— parquet 行表覆盖体检：逐文件逐站的起止日期、跨度、每月行数/天数、15 分钟网格完整度、最大缺口。
# 自包含脚本，只需要 pandas + pyarrow，可以单独拷到任何机器上跑。只读时间列和站列，不碰功率数组列。
# 站列自动认 station_id / station，都没有就用文件名当站。
#
#   python tools/parquet_coverage.py /path/to/parquet_folder
#   python tools/parquet_coverage.py /path/to/folder --ts-col 时间列名 --out-dir reports/coverage
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def station_key(x):
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    t = str(x).strip()
    try:
        f = float(t)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return t


def to_dt(col):
    if pd.api.types.is_numeric_dtype(col):
        v = col.astype("float64"); m = np.nanmedian(v)
        unit = "ns" if m > 1e17 else "us" if m > 1e14 else "ms" if m > 1e11 else "s"
        ts = pd.to_datetime(v, unit=unit, errors="coerce")
    else:
        ts = pd.to_datetime(col, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts.to_numpy().astype("datetime64[ns]")


def summarize(ts, step_min):
    t = ts[~np.isnat(ts)]
    if t.size == 0:
        return {"n_rows": int(ts.size), "n_nat": int(ts.size)}
    days = np.unique(t.astype("datetime64[D]"))
    gaps = np.diff(days).astype(int) - 1 if days.size > 1 else np.array([0])
    span = int((days[-1] - days[0]).astype(int)) + 1
    return {"n_rows": int(ts.size), "n_nat": int(ts.size - t.size),
            "start": str(days[0]), "end": str(days[-1]), "span_days": span, "days_with_data": int(days.size),
            "missing_days": span - int(days.size), "max_gap_days": int(gaps.max()),
            "rows_per_day": round(t.size / days.size, 1),
            "grid_pct": round(100.0 * t.size / (days.size * (1440 // step_min)), 1),
            "dup_ts": int(t.size - np.unique(t).size)}


def monthly(ts):
    t = pd.DatetimeIndex(ts[~np.isnat(ts)]); per = t.to_period("M")
    rows = pd.Series(1, index=per).groupby(level=0).sum()
    days = pd.Series(t.normalize(), index=per).groupby(level=0).nunique()
    return rows, days


def main():
    ap = argparse.ArgumentParser(description="parquet 行表覆盖体检")
    ap.add_argument("paths", nargs="+", help="目录、文件或通配符")
    ap.add_argument("--ts-col", default="timestamp_win")
    ap.add_argument("--station-col", default=None)
    ap.add_argument("--step-min", type=int, default=15)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--out-dir", default=None, help="可选：把结果写成 CSV")
    a = ap.parse_args()

    files = []
    for s in a.paths:
        p = Path(s)
        if p.is_dir():
            files += sorted(str(x) for x in (p.rglob("*.parquet") if a.recursive else p.glob("*.parquet")))
        else:
            files += sorted(glob.glob(s, recursive=a.recursive))
    files = sorted(dict.fromkeys(files))
    assert files, f"{a.paths} 下没有 parquet"

    rows, mrows, mdays = [], {}, {}
    for f in files:
        names = pq.read_schema(f).names
        assert a.ts_col in names, f"{f} 没有时间列 {a.ts_col!r}，现有列：{names[:15]}"
        sc = a.station_col or next((c for c in ("station_id", "station") if c in names), None)
        df = pq.read_table(f, columns=[a.ts_col] + ([sc] if sc else [])).to_pandas()
        ts = to_dt(df[a.ts_col])
        st = np.array([station_key(x) for x in df[sc]]) if sc else np.full(len(df), Path(f).stem)
        for s in sorted(set(st.tolist())):
            m = st == s; key = (Path(f).name, s)
            rows.append({"file": key[0], "station": s, **summarize(ts[m], a.step_min)})
            mrows[key], mdays[key] = monthly(ts[m])

    summ = pd.DataFrame(rows)
    idx = pd.MultiIndex.from_tuples(list(mrows), names=["file", "station"])
    M_rows = pd.DataFrame(mrows).T.reindex(idx).fillna(0).astype(int)
    M_days = pd.DataFrame(mdays).T.reindex(idx).fillna(0).astype(int)
    M_rows = M_rows[sorted(M_rows.columns)]; M_days = M_days[sorted(M_days.columns)]
    M_rows.columns = [str(c) for c in M_rows.columns]; M_days.columns = [str(c) for c in M_days.columns]
    per_month = pd.DataFrame({"stations_with_data": (M_rows > 0).sum(0), "rows": M_rows.sum(0),
                              "stations_full_month": (M_days >= 28).sum(0)})

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 1000)
    print(f"== 文件 {len(files)} 个，（文件, 站）{len(summ)} 组；网格 {1440 // a.step_min} 行/天 = 100%")
    print(summ.to_string(index=False))
    if "span_days" in summ:
        b = pd.cut(summ["span_days"], [0, 92, 183, 366, 10_000], labels=["≤3月", "3–6月", "6–12月", ">12月"])
        print("== 跨度分布：" + "，".join(f"{k}: {v}" for k, v in b.value_counts().sort_index().items()))
        print(f"== 全局起止 {summ['start'].min()} → {summ['end'].max()}；缺口>30天的组 {int((summ['max_gap_days'] > 30).sum())}；"
              f"网格<90%的组 {int((summ['grid_pct'] < 90).sum())}；有重复时间戳的组 {int((summ['dup_ts'] > 0).sum())}")
    print("== 每月行数（行 = 文件|站）："); print(M_rows.to_string())
    print("== 每月有数据的站数 / 总行数 / 当月≥28天的站数："); print(per_month.T.to_string())
    if a.out_dir:
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out / "coverage_summary.csv", index=False)
        M_rows.to_csv(out / "coverage_monthly_rows.csv"); M_days.to_csv(out / "coverage_monthly_days.csv")
        per_month.to_csv(out / "coverage_stations_per_month.csv")
        print(f"== CSV → {out}")


if __name__ == "__main__":
    main()
