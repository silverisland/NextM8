# -*- coding: utf-8 -*-
# forecast_backbones/m8/diagnose_test.py
"""M8 测试集诊断：只输出聚合统计（逐站一行、逐小时平均曲线、逐日分数），不含任何原始行，可以直接贴出来讨论。

  python -m forecast_backbones.m8.diagnose_test --result runs/m8/refit/ens_eval/result_test.parquet \\
      [--other m4/result.parquet] [--test "datasets/test/*.parquet"] [--station 白马湖光伏电站 ...] [--out <dir>]

--result  evaluate.py 落的 result_test.parquet（含 median / p_star / 分位列 / cap / degraded 时诊断更全）
--other   另一个模型同格式的结果（timestamp_win, station, observe_power_predict, observe_power_future；parquet/csv/pkl），
          例如 M4 的逐站结果；按站名对齐，南网分数只比两边都有的起报日
--test    测试 parquet（一站一文件）：统计输入侧——起报节奏、10:00 行数、cap_power_on 取值、NWP 未来列缺失率、历史功率缺失/全零率
--station 要出逐小时曲线与逐日分数的站（默认：南网 D+1 最差的两个站）
--pred-col <列>        把 --result 当作测试 parquet（文件 / 目录 / glob），直接拿数据里的某个 192 步预测列当提交曲线打分，
                       例如 --pred-col timesfm3 = TimesFM 零样本基线，不用跑任何模型；容量取 --station-meta 站表
--other-pred-col <列>  同上，作用于 --other（如 --result <M8 结果> --other "datasets/test/*.parquet" --other-pred-col timesfm3，
                       按公共起报日逐站比 M8 与 TimesFM）
--night-zero           两边的提交曲线都先做夜间抹零再打分（太阳高度角<=0 的步置 0，缺坐标站按 06:15–20:00 钟表窗），
                       模拟上线后对所有模型统一的夜间后处理：这样比出来的差距只来自白天。
                       结果里有 elev_deg 列（M8 的 result_test.parquet）直接用；没有就按 --station-meta 的坐标算

逐站表的列：
  nanwang_d1          该模型提交曲线的南网 D+1（起报日均值，日分数负值 cap 到 0，与月度表同口径）
  nanwang_d1_median / nanwang_d1_pstar   同一结果文件里用 median / p* 曲线打分（看决策规则在测试集上是帮还是害）
  nanwang_d1_rescaled 把预测整体乘以 (观测总量/预测总量) 后的分数——只纠正尺度能拿回多少（RevIN 类模型天然带这个自适应）
  bias_ratio          D+1 打分点上 预测总量/观测总量（1 = 不偏；>1 高估）
  peak_over_cap       测试期观测最大值 / cap（远小于 1 说明 cap 与实际出力尺度不符）
  nrmse_pct           全部 192 步 (y-p)/cap 的 RMSE×100（与训练报表同口径）
  days_lt40           日分数 < 40 的起报日占比（是少数灾难日还是整体偏低）"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ...pipelines.nanwang_metric import NanwangMetric
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from pipelines.nanwang_metric import NanwangMetric

STEP_MIN = 15
D1_START, D1_END = 55, 151          # 10:00 起报行的 D+1 全天 = predict[55..150]


# ------------------------------------------------------------------ 读结果
def _arr(col: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, float).reshape(-1) for v in col.to_numpy()])


def _expand(t: str) -> list[str]:
    """文件 / 目录 / glob → parquet 路径列表。"""
    if any(ch in t for ch in "*?["):
        return sorted(glob.glob(t))
    return [str(q) for q in sorted(Path(t).glob("*.parquet"))] if Path(t).is_dir() else [t]


DEFAULT_META = Path(__file__).resolve().parent.parent / "m6" / "station_meta.csv"


def result_from_data(paths: list[str], pred_col: str, meta_path=None) -> pd.DataFrame:
    """测试 parquet（一站一文件）里的某个 192 步预测列 → 与 evaluate.py 同格式的结果表（cap 取站表 cap_mw）。
    用途：给 timesfm3 之类的外部预测列打一个与 M8 完全同口径的基线分。"""
    try:
        from .station_report import frame_with_station_id
        from .transformer_models import load_station_meta
        from .features import station_cap_mw
    except ImportError:
        from forecast_backbones.m8.station_report import frame_with_station_id
        from forecast_backbones.m8.transformer_models import load_station_meta
        from forecast_backbones.m8.features import station_cap_mw
    meta = load_station_meta(meta_path or DEFAULT_META)
    cfg = {"dataset": {"station_id_col": "station_id"}}
    frames = []
    for f in paths:
        df = pd.read_parquet(f)
        miss = [c for c in ("timestamp_win", pred_col, "observe_power_future") if c not in df.columns]
        if miss:
            raise KeyError(f"{f} 缺列 {miss}")
        df = frame_with_station_id(df, cfg, meta)
        sid = df["station_id"].astype(int)
        caps = {int(s): station_cap_mw(meta, int(s)) for s in sid.unique()}
        frames.append(pd.DataFrame({
            "timestamp_win": df["timestamp_win"].to_numpy(),
            "station": (df["station"] if "station" in df.columns else sid).astype(str).to_numpy(),
            "station_id": sid.to_numpy(), "cap": sid.map(caps).astype(float).to_numpy(),
            "observe_power_predict": [np.asarray(v, float).reshape(-1) for v in df[pred_col].to_numpy()],
            "observe_power_future": [np.asarray(v, float).reshape(-1) for v in df["observe_power_future"].to_numpy()],
        }))
    if not frames:
        raise FileNotFoundError(f"没有 parquet：{paths}")
    return pd.concat(frames, ignore_index=True)


def night_zero_frame(df: pd.DataFrame, meta_path=None, tz: str = "Asia/Shanghai") -> pd.DataFrame:
    """把 192 步提交曲线（observe_power_predict，以及 median / p_star 若有）在夜间置 0：模拟上线后统一的夜间抹零后处理。
    夜间 = 目标时刻太阳高度角 <= 0（第 k 步 = timestamp_win + (k+1)·15min），与 M8 的 is_night 同口径；
    优先用结果里的 elev_deg 列，没有就按 station_meta 坐标算（缺坐标站用钟表窗 06:15–20:00）。"""
    df = df.copy()
    cols = [c for c in ("observe_power_predict", "median", "p_star") if c in df.columns]
    if "elev_deg" in df.columns:
        night = [np.asarray(e, float) <= 0 for e in df["elev_deg"].to_numpy()]
    else:
        try:
            from .station_report import frame_with_station_id
            from .transformer_models import load_station_meta
            from .features import SolarCache
        except ImportError:
            from forecast_backbones.m8.station_report import frame_with_station_id
            from forecast_backbones.m8.transformer_models import load_station_meta
            from forecast_backbones.m8.features import SolarCache
        meta = load_station_meta(meta_path or DEFAULT_META)
        df = frame_with_station_id(df, {"dataset": {"station_id_col": "station_id"}}, meta)
        sc = SolarCache(meta, tz, STEP_MIN)
        t0 = pd.to_datetime(df["timestamp_win"]).to_numpy().astype("datetime64[s]").astype(np.int64)
        offs = (np.arange(1, 193) * STEP_MIN * 60).astype(np.int64)
        night = [None] * len(df)
        for sid, idx in df.groupby("station_id").indices.items():
            ts = (t0[idx][:, None] + offs[None, :]).astype("datetime64[s]")
            e, _, _ = sc.geometry(int(sid), ts)
            for j, i in enumerate(idx):
                night[i] = e[j] <= 0
    for c in cols:
        df[c] = [np.where(n[:len(v)], 0.0, np.asarray(v, float)) for v, n in zip(df[c].to_numpy(), night)]
    return df


def load_result(path, pred_col: str | None = None, meta_path=None) -> pd.DataFrame:
    """evaluate.py / station_report.py 落的结果表；pred_col 给定时 path 是测试 parquet（文件 / 目录 / glob），见 result_from_data。"""
    if pred_col:
        df = result_from_data(_expand(str(path)), pred_col, meta_path)
    else:
        p = Path(path)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix in (".pkl", ".pickle"):
            df = pd.read_pickle(p)
        else:
            df = pd.read_csv(p)
            for c in ("observe_power_predict", "observe_power_future"):
                if df[c].dtype == object and isinstance(df[c].iloc[0], str):
                    df[c] = df[c].map(lambda s: np.asarray(json.loads(s), float))
    need = ["timestamp_win", "station", "observe_power_predict", "observe_power_future"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{path} 缺列 {miss}；需要 {need}")
    df = df.copy()
    df["timestamp_win"] = pd.to_datetime(df["timestamp_win"])
    if getattr(df["timestamp_win"].dt, "tz", None) is not None:
        df["timestamp_win"] = df["timestamp_win"].dt.tz_localize(None)
    df["station"] = df["station"].astype(str).str.strip()
    return df.sort_values(["station", "timestamp_win"]).reset_index(drop=True)


def match_stations(a: list[str], b: list[str]) -> dict:
    """按站名对齐两份结果：先精确，再互相包含（"白马湖" ↔ "白马湖光伏电站"）。"""
    m = {}
    for s in a:
        if s in b:
            m[s] = s
            continue
        c = [t for t in b if s in t or t in s]
        if len(c) == 1:
            m[s] = c[0]
    return m


# ------------------------------------------------------------------ D+1 打分行
def d1_rows(g: pd.DataFrame) -> pd.DataFrame:
    """每个起报日取 timestamp_win <= 10:00 的最晚一行（与 NanwangMetric.short_d1_daily 同规则），
    并给出该行 D+1 全天在 192 步里的下标范围。"""
    out = []
    for d, gd in g.groupby(g["timestamp_win"].dt.normalize()):
        gd = gd[gd["timestamp_win"] <= d + pd.Timedelta(hours=10)]
        if gd.empty:
            continue
        r = gd.iloc[-1]
        t0 = r["timestamp_win"]
        k = np.arange(192)
        tgt = t0 + pd.to_timedelta((k + 1) * STEP_MIN, unit="min")
        sel = (tgt >= d + pd.Timedelta(days=1)) & (tgt < d + pd.Timedelta(days=2))
        if sel.sum() == 0:
            continue
        out.append({"date": d, "row": r.name, "lo": int(np.argmax(sel)), "hi": int(np.argmax(sel) + sel.sum())})
    return pd.DataFrame(out, columns=["date", "row", "lo", "hi"])


def d1_segments(df: pd.DataFrame, rows: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (obs [n,96], pred [n,96], target_hour [96])；长度不足 96 的日子跳过。"""
    ys, ps = [], []
    for r in rows.itertuples(index=False):
        row = df.loc[r.row]
        y = np.asarray(row["observe_power_future"], float)[r.lo:r.hi]
        p = np.asarray(row[col], float)[r.lo:r.hi]
        if len(y) == 96 and len(p) == 96:
            ys.append(y); ps.append(p)
    if not ys:
        return np.zeros((0, 96)), np.zeros((0, 96)), np.arange(96) // 4
    return np.stack(ys), np.stack(ps), np.arange(96) // 4


def nanwang_daily(y: np.ndarray, p: np.ndarray, cap: float) -> np.ndarray:
    """逐日南网 D+1，负值 cap 到 0（与月度表同口径）。"""
    out = []
    for yi, pi in zip(y, p):
        v = NanwangMetric.nanwang_official(yi, pi, cap)
        out.append(np.nan if v is None else max(0.0, v))
    return np.asarray(out, float)


def nrmse_pct(df: pd.DataFrame, col: str, cap: float) -> float:
    y, p = _arr(df["observe_power_future"]), _arr(df[col])
    r = (y - p) / cap
    ok = np.isfinite(r)
    return float(100 * np.sqrt(np.mean(r[ok] ** 2))) if ok.any() else np.nan


# ------------------------------------------------------------------ 逐站汇总
def station_summary(res: pd.DataFrame, other: pd.DataFrame | None, caps: dict) -> tuple[pd.DataFrame, dict]:
    rows, detail = [], {}
    omap = match_stations(sorted(res["station"].unique()), sorted(other["station"].unique())) if other is not None else {}
    for st, g in res.groupby("station", sort=True):
        cap = caps[st]
        rr = d1_rows(g)
        y, p, hour = d1_segments(g, rr, "observe_power_predict")
        d = nanwang_daily(y, p, cap)
        yall = _arr(g["observe_power_future"])
        row = {"station": st, "cap": cap, "rows": len(g), "d1_days": len(d),
               "degraded_frac": float(g["degraded"].mean()) if "degraded" in g else np.nan,
               "peak_over_cap": float(np.nanmax(yall) / cap) if np.isfinite(yall).any() else np.nan,
               "obs_mean_over_cap": float(np.nanmean(y) / cap) if len(y) else np.nan,
               "bias_ratio": float(np.nansum(p) / np.nansum(y)) if len(y) and np.nansum(y) > 0 else np.nan,
               "nanwang_d1": float(np.nanmean(d)) if len(d) else np.nan,
               "d1_median_day": float(np.nanmedian(d)) if len(d) else np.nan,
               "days_lt40": float(np.mean(d < 40)) if len(d) else np.nan,
               "nrmse_pct": nrmse_pct(g, "observe_power_predict", cap)}
        for c, k in (("median", "nanwang_d1_median"), ("p_star", "nanwang_d1_pstar")):
            if c in g.columns:
                _, pc, _ = d1_segments(g, rr, c)
                row[k] = float(np.nanmean(nanwang_daily(y, pc, cap))) if len(pc) else np.nan
        if len(y) and np.nansum(p) > 0:
            s = np.nansum(y) / np.nansum(p)
            row["nanwang_d1_rescaled"] = float(np.nanmean(nanwang_daily(y, p * s, cap)))
        daily = pd.DataFrame({"date": rr["date"].dt.date.astype(str).to_numpy()[:len(d)] if len(rr) else [], "nanwang_d1": d})
        if len(y):
            daily["obs_peak_over_cap"] = np.nanmax(y, axis=1) / cap
            daily["pred_peak_over_cap"] = np.nanmax(p, axis=1) / cap
        hourly = pd.DataFrame({"hour": np.arange(24),
                               "obs_over_cap": [float(np.nanmean(y[:, hour == h]) / cap) if len(y) else np.nan for h in range(24)],
                               "pred_over_cap": [float(np.nanmean(p[:, hour == h]) / cap) if len(y) else np.nan for h in range(24)],
                               "mae_over_cap": [float(np.nanmean(np.abs(p - y)[:, hour == h]) / cap) if len(y) else np.nan for h in range(24)]})
        if st in omap:
            og = other[other["station"] == omap[st]]
            orr = d1_rows(og)
            oy, op, _ = d1_segments(og, orr, "observe_power_predict")
            od = nanwang_daily(oy, op, cap)
            odaily = pd.DataFrame({"date": orr["date"].dt.date.astype(str).to_numpy()[:len(od)] if len(orr) else [], "other_nanwang_d1": od})
            if len(oy):
                odaily["other_pred_peak_over_cap"] = np.nanmax(op, axis=1) / cap
            both = daily.merge(odaily, on="date", how="inner")
            row.update({"other_station": omap[st], "other_rows": len(og), "other_d1_days": len(od),
                        "other_nanwang_d1_all": float(np.nanmean(od)) if len(od) else np.nan,
                        "common_days": len(both),
                        "nanwang_d1_common": float(both["nanwang_d1"].mean()) if len(both) else np.nan,
                        "other_nanwang_d1_common": float(both["other_nanwang_d1"].mean()) if len(both) else np.nan,
                        "other_bias_ratio": float(np.nansum(op) / np.nansum(oy)) if len(oy) and np.nansum(oy) > 0 else np.nan,
                        "other_nrmse_pct": nrmse_pct(og, "observe_power_predict", cap)})
            daily = daily.merge(odaily, on="date", how="outer").sort_values("date")
            if len(oy):
                hourly["other_pred_over_cap"] = [float(np.nanmean(op[:, hour == h]) / cap) for h in range(24)]
                hourly["other_mae_over_cap"] = [float(np.nanmean(np.abs(op - oy)[:, hour == h]) / cap) for h in range(24)]
        rows.append(row)
        detail[st] = (hourly, daily)
    unmatched = [s for s in res["station"].unique() if other is not None and s not in omap]
    if unmatched:
        print(f"[diag] --other 里对不上站名的站：{unmatched}；--other 的站名有：{sorted(other['station'].unique())}")
    return pd.DataFrame(rows), detail


# ------------------------------------------------------------------ 输入侧（测试 parquet）
def input_stats(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        df = pd.read_parquet(p)
        ts = pd.to_datetime(df["timestamp_win"])
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
        st = str(df["station"].iloc[0]) if "station" in df.columns else (str(df["station_id"].iloc[0]) if "station_id" in df.columns else Path(p).stem)
        row = {"file": Path(p).name, "station": st, "rows": len(df), "start": str(ts.min().date()), "end": str(ts.max().date()),
               "rows_at_10:00": int(((ts.dt.hour == 10) & (ts.dt.minute == 0)).sum()),
               "rows_per_day": round(len(df) / max(ts.dt.normalize().nunique(), 1), 1),
               "minute_set": ",".join(str(m) for m in sorted(ts.dt.minute.unique())[:6])}
        if "cap_power_on" in df.columns:
            c = pd.to_numeric(df["cap_power_on"], errors="coerce")
            row.update({"cap_min": float(c.min()), "cap_max": float(c.max()), "cap_nunique": int(c.nunique())})
        for col in [c for c in df.columns if c.endswith("_predict") and c != "observe_power_predict"]:
            a = _arr(df[col])
            row[f"nan%_{col}"] = round(100 * float(np.mean(~np.isfinite(a))), 1)
        if "observe_power" in df.columns:
            h = _arr(df["observe_power"])
            row["hist_len"] = h.shape[1]
            row["nan%_hist"] = round(100 * float(np.mean(~np.isfinite(h))), 1)
            row["zero%_hist_rows"] = round(100 * float(np.mean(np.nanmax(np.where(np.isfinite(h), h, 0), axis=1) <= 0)), 1)
            row["hist_peak"] = float(np.nanmax(h))
        if "observe_power_future" in df.columns:
            f = _arr(df["observe_power_future"])
            row["nan%_future"] = round(100 * float(np.mean(~np.isfinite(f))), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True)
    ap.add_argument("--other", default=None)
    ap.add_argument("--test", nargs="*", default=None)
    ap.add_argument("--station", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cap", type=float, default=None, help="结果里没有 cap 列时统一用这个容量（或 --cap-json）")
    ap.add_argument("--cap-json", default=None, help='{"站名": 容量} 的 json 文件')
    ap.add_argument("--pred-col", default=None, help="把 --result 当测试 parquet，用这个 192 步列当提交曲线（如 timesfm3）")
    ap.add_argument("--other-pred-col", default=None, help="把 --other 当测试 parquet，用这个列当对方的提交曲线")
    ap.add_argument("--station-meta", default=None, help=f"--pred-col / --other-pred-col / --night-zero 用的站表（默认 {DEFAULT_META}）")
    ap.add_argument("--night-zero", action="store_true", help="两边提交曲线先夜间抹零再打分（模拟上线后统一的夜间后处理）")
    ap.add_argument("--tz", default="Asia/Shanghai")
    a = ap.parse_args(argv)
    res = load_result(a.result, a.pred_col, a.station_meta)
    other = load_result(a.other, a.other_pred_col, a.station_meta) if a.other else None
    if a.night_zero:
        res = night_zero_frame(res, a.station_meta, a.tz)
        other = night_zero_frame(other, a.station_meta, a.tz) if other is not None else None
        print("[diag] 夜间抹零：两边提交曲线在太阳高度角<=0 的步置 0 后再打分（缺坐标站按钟表窗）")
    if "cap" in res.columns:
        caps = res.groupby("station")["cap"].first().astype(float).to_dict()
    elif a.cap_json:
        caps = {str(k): float(v) for k, v in json.loads(Path(a.cap_json).read_text(encoding="utf-8")).items()}
    elif a.cap is not None:
        caps = {s: a.cap for s in res["station"].unique()}
    else:
        raise SystemExit("结果里没有 cap 列：请给 --cap 或 --cap-json")
    out = Path(a.out) if a.out else Path(a.result).parent / "diagnose"
    out.mkdir(parents=True, exist_ok=True)
    table, detail = station_summary(res, other, caps)
    table.to_csv(out / "station_diag.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\n=== 逐站诊断（station_diag.csv）===")
    print(table.to_string(index=False))
    if a.test:
        paths = sorted(sum([_expand(t) for t in a.test], []))
        inp = input_stats(paths)
        inp.to_csv(out / "input_stats.csv", index=False)
        print("\n=== 输入侧统计（input_stats.csv）===")
        print(inp.to_string(index=False))
    sts = a.station or list(table.sort_values("nanwang_d1")["station"].head(2))
    for st in sts:
        hit = [s for s in detail if s == st or st in s]
        if not hit:
            print(f"[diag] 结果里没有站 {st!r}"); continue
        hourly, daily = detail[hit[0]]
        hourly.to_csv(out / f"{hit[0]}-hourly.csv", index=False)
        daily.to_csv(out / f"{hit[0]}-daily.csv", index=False)
        print(f"\n=== {hit[0]} 逐小时平均（D+1 打分点，单位 cap）===")
        print(hourly[hourly["obs_over_cap"].fillna(0) + hourly["pred_over_cap"].fillna(0) > 0.005].to_string(index=False))
        print(f"\n=== {hit[0]} 逐起报日 ===")
        print(daily.to_string(index=False))
    print(f"\n[diag] 产物：{out}")


if __name__ == "__main__":
    main()
