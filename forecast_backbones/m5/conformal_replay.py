# -*- coding: utf-8 -*-
# forecast_backbones/m5/conformal_replay.py
"""M5 离线 conformal 测试入口，两步（cwd=仓库根）：

1) predict（GPU）：逐站读测试集 station=*.parquet，每站每天只推理 <= 10:00 的最晚一行（南网短期打分用的那行），
   落盘原始 9 档分位 + 真值 + 太阳高度/晴空功率/夜间/多变度档，一站一个 {站}.parquet；已存在的站跳过。
    python -m forecast_backbones.m5.conformal_replay predict \
        --station_dir datasets/test --out_dir runs/m5_conformal/raw
   多卡：每张卡起一个进程，加 --shard i --num_shards N（见 eval_shell/conformal_m5.sh）。
2) replay（CPU，几分钟）：读 raw 目录，逐站逐日滚动拟合并打分，写报表。
    python -m forecast_backbones.m5.conformal_replay replay \
        --raw_dir runs/m5_conformal/raw --out_dir runs/m5_conformal/replay --k 5 15 30

站点经纬度来自 --station_meta（默认 M6 的 station_meta.csv）；查不到坐标的站退化为按钟点的伪几何（晴空功率 = 装机容量、
高度角分档失效），会打印警告。多变度档、决策层（物理钳位、p*）的参数取 --m6_config。"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast_backbones.m5.conformal import Q_COLS, ReplayConfig, replay, summarize
from forecast_backbones.m6.config import load_config
from forecast_backbones.m6.features import StationMeta, build_row_context, norm_station_name
from forecast_backbones.m6.rows import SHORT_ISSUE_TIME

M5_CONFIG = ROOT / "forecast_backbones" / "m5" / "config" / "configs.yaml"
M6_CONFIG = ROOT / "forecast_backbones" / "m6" / "config" / "configs.yaml"
STATION_META = ROOT / "forecast_backbones" / "m6" / "station_meta.csv"


def station_of(path: Path, df: pd.DataFrame) -> str:
    m = re.match(r"station=(.+)\.parquet$", path.name)
    if m:
        return norm_station_name(m.group(1))
    if "station" in df.columns:
        return norm_station_name(df["station"].iloc[0])
    return path.stem


def scored_rows(df: pd.DataFrame, ts_col: str = "timestamp_win", issue_time: str = SHORT_ISSUE_TIME) -> pd.DataFrame:
    """每个起报日取 timestamp_win <= issue_time 的最晚一行（与 NanwangMetric.short_d1_daily 选行一致）。"""
    ts = pd.to_datetime(df[ts_col])
    day = ts.dt.normalize()
    keep = (ts - day) <= pd.Timedelta(f"{issue_time}:00")
    f = df[keep.to_numpy()].assign(**{ts_col: ts[keep]}).sort_values(ts_col, kind="stable")
    return f.groupby(f[ts_col].dt.normalize(), sort=True).tail(1).reset_index(drop=True)


def _first_finite(v) -> float:
    a = np.asarray(v, float).reshape(-1)
    a = a[np.isfinite(a)]
    return float(a[0]) if a.size else float("nan")


def row_caps(rows: pd.DataFrame, cap_col: str, meta_cap, fallback: float) -> np.ndarray:
    """逐行装机容量：数据的 cap 列 → station_meta 的 cap_mw → fallback。"""
    cap = rows[cap_col].map(_first_finite).to_numpy(float) if cap_col in rows.columns else np.full(len(rows), np.nan)
    if meta_cap is not None and np.isfinite(meta_cap):
        cap = np.where(np.isfinite(cap), cap, float(meta_cap))
    return np.where(np.isfinite(cap), cap, float(fallback))


def predict_station_raw(model, station_path, m6cfg: dict, meta: StationMeta, cap_fallback: float,
                        batch_size: int = 64) -> pd.DataFrame:
    """单站：选行 → M5 推理原始 9 档分位 → 按 M6 口径算太阳几何/晴空功率/夜间/多变度档 → 原始分位表。
    真值列只写进结果表，不进模型输入。"""
    ds = m6cfg["dataset"]; ts_col, label_col, cap_col = ds["ts_col"], ds["label_col"], ds["cap_col"]
    L = int(m6cfg["timesfm3"]["context_len"])
    station_path = Path(station_path)
    df = pd.read_parquet(station_path)
    st = station_of(station_path, df)
    rows = scored_rows(df, ts_col)
    if rows.empty:
        raise ValueError(f"{station_path}：没有 <= {SHORT_ISSUE_TIME} 的起报行")
    if label_col not in rows.columns:
        raise KeyError(f"{station_path} 缺真值列 {label_col!r}")
    need = [model.target_col, *[c for pair in model.past_future_cols for c in pair], *model.past_only_cols,
            m6cfg["variability"]["ghi_col"]]
    need = list(dict.fromkeys(need))
    missing = [c for c in need if c not in rows.columns]
    if missing:
        raise KeyError(f"{station_path} 缺列 {missing}")
    sid = meta.id_of(st)
    if sid is None or not meta.has_coords(sid):
        warnings.warn(f"站 {st} 在 station_meta 里没有坐标：按钟点伪几何处理（晴空功率 = 装机容量，高度角分档失效）")
        sid = -1
    cap = row_caps(rows, cap_col, (meta.get(sid) or {}).get("cap_mw") if sid >= 0 else None, cap_fallback)
    epoch = ((rows[ts_col] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
    out = []
    for a in range(0, len(rows), batch_size):
        chunk = rows.iloc[a:a + batch_size]
        b = slice(a, a + len(chunk))
        bd = {c: np.stack([np.asarray(v, np.float32) for v in chunk[c].to_numpy()]) for c in need}
        bd[ts_col] = epoch[b]
        res = model(bd)
        q = res.aux["quantiles"].detach().cpu().numpy()                                  # [B,H,9]
        ctx = build_row_context({**bd, ds["station_id_col"]: np.full(len(chunk), sid), cap_col: cap[b]}, m6cfg, meta)
        for i in range(len(chunk)):
            rec = {"station": st, "timestamp_win": chunk[ts_col].iloc[i], "cap": float(cap[b][i]),
                   "observe_power_future": np.asarray(chunk[label_col].iloc[i], np.float32)[:q.shape[1]],
                   "elev_deg": ctx.elev_deg[i, L:].astype(np.float32), "p_clear": ctx.p_clear[i, L:].astype(np.float32),
                   "night": ctx.night_future[i].astype(bool), "var_bin": int(ctx.var_bin[i]),
                   "degraded": bool(ctx.degraded[i])}
            for j, c in enumerate(Q_COLS):
                rec[c] = q[i, :, j].astype(np.float32)
            out.append(rec)
    return pd.DataFrame(out)


def shard_paths(paths, shard: int, num_shards: int) -> list:
    """多卡并行：按站排序后轮流分给 num_shards 个进程，本进程取第 shard 份。"""
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard={shard} 须在 [0, {num_shards}) 内")
    return list(paths)[shard::num_shards]


def run_predict(a) -> None:
    import torch
    from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model

    m5cfg = yaml.safe_load(Path(a.m5_config).read_text(encoding="utf-8")) or {}
    if a.device:
        m5cfg.setdefault("timesfm3", {})["device"] = a.device
    m6cfg = load_config(a.m6_config)
    meta = StationMeta.from_csv(a.station_meta)
    paths = sorted(Path(a.station_dir).glob("station=*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{a.station_dir} 下没有 station=*.parquet")
    paths = shard_paths(paths, a.shard, a.num_shards)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = TransformersAdaptedM5Model(m5cfg)
    model.eval()
    with torch.no_grad():
        for p in paths:
            st = norm_station_name(p.name[len("station="):-len(".parquet")])
            dst = out_dir / f"{st}.parquet"
            if dst.exists():
                print(f"SKIP {st}（已存在 {dst}）")
                continue
            raw = predict_station_raw(model, p, m6cfg, meta, a.cap_power, a.batch_size)
            raw.to_parquet(dst, index=False)
            print(f"OK   {st}：{len(raw)} 个起报日 {raw['timestamp_win'].min().date()} ~ {raw['timestamp_win'].max().date()}")


def load_clusters(path) -> dict | None:
    """station,cluster 两列的 csv；不给 = 全部站一类。表里没有的站自成一类（按站名）。"""
    if not path:
        return None
    df = pd.read_csv(path, dtype={"station": str, "cluster": str})
    return {norm_station_name(s): str(c) for s, c in zip(df["station"], df["cluster"])}


def run_replay(a) -> None:
    paths = sorted(Path(a.raw_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{a.raw_dir} 下没有原始分位表")
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    raw["station"] = raw["station"].map(norm_station_name)
    clusters = load_clusters(a.clusters)
    if clusters is not None:
        clusters = {s: clusters.get(s, f"_self_{s}") for s in raw["station"].unique()}
    cfg = ReplayConfig(window_days=a.window_days, lag_days=a.lag_days, m=a.m, ks=tuple(a.k), pool=a.pool)
    decision = load_config(a.m6_config)["decision"]
    daily, coverage = replay(raw, cfg, decision, clusters)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "daily.csv", index=False)
    tables = summarize(daily, coverage)
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
    (out / "replay_config.json").write_text(json.dumps(
        {"window_days": cfg.window_days, "lag_days": cfg.lag_days, "m": cfg.m, "ks": list(cfg.ks), "pool": cfg.pool,
         "levels": list(cfg.levels), "fit_steps": list(cfg.fit_steps), "clusters": clusters,
         "stations": sorted(raw["station"].unique())}, ensure_ascii=False, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tables["summary"].to_string(index=False))
        print()
        print(tables["cold_start"].to_string(index=False))
    print(f"\n报表：{out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="M5 离线滚动 conformal（收缩加权）测试")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="逐站推理并落盘原始分位表（GPU）")
    p.add_argument("--station_dir", required=True, help="测试集目录，含 station=*.parquet")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--m5_config", default=str(M5_CONFIG))
    p.add_argument("--m6_config", default=str(M6_CONFIG), help="取多变度档与站表路径口径")
    p.add_argument("--station_meta", default=str(STATION_META), help="station,station_id,lat,lon,alt_m,cap_mw")
    p.add_argument("--cap_power", type=float, default=465.0, help="数据与站表都没有容量时的回退值")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None, help="auto|cuda|cpu，默认用 m5 配置里的")
    p.add_argument("--shard", type=int, default=0, help="多卡并行：本进程是第几份（从 0 起）")
    p.add_argument("--num_shards", type=int, default=1, help="多卡并行：一共分几份")
    r = sub.add_parser("replay", help="滚动回放 + 报表（CPU）")
    r.add_argument("--raw_dir", required=True)
    r.add_argument("--out_dir", required=True)
    r.add_argument("--clusters", default=None, help="station,cluster 两列 csv；不给 = 全部站一类")
    r.add_argument("--k", type=float, nargs="+", default=[15.0], help="收缩强度，可给多个一起比")
    r.add_argument("--window_days", type=int, default=45)
    r.add_argument("--lag_days", type=int, default=2)
    r.add_argument("--m", type=float, default=3.0, help="D_min(τ) = ceil(m / min(τ, 1−τ))")
    r.add_argument("--pool", choices=["season", "calendar"], default="season")
    r.add_argument("--m6_config", default=str(M6_CONFIG), help="取决策层（物理钳位、p*）参数")
    a = ap.parse_args(argv)
    run_predict(a) if a.cmd == "predict" else run_replay(a)


if __name__ == "__main__":
    main()
