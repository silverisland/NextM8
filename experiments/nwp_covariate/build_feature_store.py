# -*- coding: utf-8 -*-
"""build_feature_store.py —— DS 轨迹表 → 特征库（按站 parquet，DS 同格式 + 标准名列）。

流程：读 DS parquet（多起报时刻合并）→ 站点/时段/省级行过滤 → columns 映射 + 单位统一 →
消融（可选：拔源 / 掩尾 / NWP 延迟）→ derived 衍生量 → fallback 补全链 → 落盘。

用法（cwd=仓库根）：
    python experiments/nwp_covariate/build_feature_store.py --config experiments/nwp_covariate/config.yaml \
        [--predict_type ultra_short] [--out_dir ...] [--ablation '{"drop_sources": ["ec"]}']
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops import BuildContext, Pair, convert_kind, run_op  # noqa: E402
from fallback import apply_fallback  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("nwp_cov.build")


# ----------------------------------------------------------------------------- config / io
def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _abs(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def ds_files(cfg: dict, predict_type: str) -> list:
    d = cfg["data"]
    if d.get("input_mode") == "ods":
        pat = str(_abs(cfg["ods"]["out_dir"].format(predict_type=predict_type)) / "date=*" / "time=*" / "*.parquet")
    else:
        pat = d["ds_trajectory_glob"].format(predict_type=predict_type)
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"没有匹配到 DS parquet: {pat}")
    return files


def time_range(cfg: dict, predict_type: str):
    tr = cfg["data"].get("time_range")
    if tr:
        return pd.Timestamp(tr["start"]), pd.Timestamp(tr["end"])
    rt = cfg["report_times"][predict_type]
    return pd.Timestamp(rt["start"]), pd.Timestamp(rt["end"])


def load_ds(cfg: dict, predict_type: str, files: Optional[list] = None) -> pd.DataFrame:
    d = cfg["data"]
    files = files or ds_files(cfg, predict_type)
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop(columns=[c for c in ("update_time",) if c in df.columns])
    df["timestamp_win"] = pd.to_datetime(df["timestamp_win"])
    s, e = time_range(cfg, predict_type)
    df = df[(df["timestamp_win"] >= s) & (df["timestamp_win"] <= e)]
    if d.get("drop_province_row", True):
        area = (cfg.get("ods") or {}).get("area", "province_")
        df = df[~df["station"].astype(str).str.startswith("province_") & (df["station"] != area)]
    st = d.get("stations", "all")
    if isinstance(st, list):
        df = df[df["station"].isin(st)]
    df = df.sort_values(["station", "timestamp_win"]).drop_duplicates(["station", "timestamp_win"], keep="last")
    log.info("DS 表: %d 文件, %d 行, %d 站, %s .. %s", len(files), len(df), df["station"].nunique(),
             df["timestamp_win"].min(), df["timestamp_win"].max())
    return df.reset_index(drop=True)


def load_station_meta(path: Optional[str]) -> pd.DataFrame:
    """station, lat, lon, alt_m。兼容 info.csv 风格（plant_pointname/LATITUDE/LONGITUDE）与 m6 station_meta.csv。"""
    if not path or not _abs(path).exists():
        return pd.DataFrame(columns=["station", "lat", "lon", "alt_m"])
    m = pd.read_csv(_abs(path))
    low = {c.lower(): c for c in m.columns}
    def pick(*names):
        for n in names:
            if n in low:
                return m[low[n]]
        return None
    st = pick("station", "plant_pointname", "plantid")
    lat = pick("lat", "latitude"); lon = pick("lon", "lng", "longitude"); alt = pick("alt_m", "alt", "altitude")
    out = pd.DataFrame({"station": st.astype(str), "lat": pd.to_numeric(lat, errors="coerce"),
                        "lon": pd.to_numeric(lon, errors="coerce"),
                        "alt_m": pd.to_numeric(alt, errors="coerce") if alt is not None else np.nan})
    return out.drop_duplicates("station")


# ----------------------------------------------------------------------------- 数组列
def stack_col(df: pd.DataFrame, col: str, length: int, take: str) -> np.ndarray:
    """object/list 列 -> [B, length]。take='tail' 取末尾 length 点（历史），'head' 取前 length 点（未来）。
    长度不足补 NaN；空/None 整行 NaN。"""
    out = np.full((len(df), length), np.nan, dtype=np.float32)
    for i, v in enumerate(df[col].tolist()):
        if v is None or np.ndim(v) == 0:
            continue
        a = np.asarray(v, dtype=np.float32).ravel()
        if a.size == 0:
            continue
        if take == "tail":
            a = a[-length:]; out[i, -a.size:] = a
        else:
            a = a[:length]; out[i, :a.size] = a
    return out


def cap_values(df: pd.DataFrame, cap_col: str, default: float) -> np.ndarray:
    if cap_col not in df.columns:
        return np.full(len(df), float(default))
    out = np.full(len(df), float(default))
    for i, v in enumerate(df[cap_col].tolist()):
        if v is None:
            continue
        if np.ndim(v) > 0:
            a = np.asarray(v, dtype=float).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                out[i] = a[-1]
        else:
            try:
                f = float(v)
                if np.isfinite(f) and f > 0:
                    out[i] = f
            except (TypeError, ValueError):
                pass
    return out


# ----------------------------------------------------------------------------- 主构建
def build(cfg: dict, predict_type: Optional[str] = None, ablation: Optional[dict] = None,
          files: Optional[list] = None) -> tuple:
    """-> (dict station -> DataFrame, report dict)。DataFrame 是 DS 同格式（list 列）+ 标准名列。"""
    d = cfg["data"]
    predict_type = predict_type or d["predict_type"]
    C, P = int(d["context_len"]), int(d["pred_len"])
    ablation = ablation or {}
    df = load_ds(cfg, predict_type, files)
    if df.empty:
        raise ValueError("过滤后 DS 表为空")

    meta = load_station_meta(d.get("station_meta"))
    if ablation.get("drop_station_meta"):
        meta = meta.iloc[0:0]
    mm = df[["station"]].astype(str).merge(meta, on="station", how="left")
    cap = cap_values(df, d.get("cap_col", "cap_power_on"), cfg.get("run", {}).get("cap_power_default", 465))
    ctx = BuildContext(ts_win=df["timestamp_win"].values.astype("datetime64[s]"), context_len=C, pred_len=P,
                       lat=mm["lat"].to_numpy(float), lon=mm["lon"].to_numpy(float),
                       alt_m=mm["alt_m"].to_numpy(float), cap=cap,
                       tz_offset_h=float(d.get("tz_offset_h", 8.0)))
    report = {"predict_type": predict_type, "rows": int(len(df)), "stations": int(df["station"].nunique()),
              "ablation": ablation, "mapped": [], "missing_raw": [], "derived_ok": [], "derived_fail": {},
              "fallback": [], "warnings": []}

    # 1. 目标与真值
    tcol, lcol = d["target"], d["label"]
    ctx.cols[tcol] = Pair(stack_col(df, tcol, C, "tail"), np.full((len(df), P), np.nan, np.float32))
    label_fut = stack_col(df, lcol, P, "head") if lcol in df.columns else None

    # 2. 源字段映射 + 单位
    suffix = cfg["columns"].get("future_suffix", "_predict")
    drop_sources = set(ablation.get("drop_sources", []) or [])
    source_of: Dict[str, str] = {}
    for sname, src in cfg["columns"]["sources"].items():
        if sname in drop_sources:
            report["warnings"].append(f"消融：整源拔除 {sname}")
            continue
        prefix = src.get("prefix", "") or ""
        hist_only = bool(src.get("history_only", False))
        for std, spec in (src.get("fields") or {}).items():
            raw_h = prefix + spec["raw"]
            raw_f = raw_h + suffix
            if raw_h not in df.columns or (not hist_only and raw_f not in df.columns):
                report["missing_raw"].append({"std": std, "raw": raw_h, "optional": bool(spec.get("optional"))})
                continue
            h = stack_col(df, raw_h, C, "tail")
            f = stack_col(df, raw_f, P, "head") if not hist_only else np.full((len(df), P), np.nan, np.float32)
            pair = convert_kind(Pair(h, f), spec.get("kind", "raw"), int(src.get("resolution_min", 15)))
            ctx.cols[std] = pair
            source_of[std] = sname
            report["mapped"].append(std)

    # 3. 掩尾 / NWP 延迟消融（只作用于 NWP 源字段，不动目标与真值）
    nwp_names = [n for n in source_of if source_of[n] != "obs"]
    if ablation.get("mask_future_tail"):
        k = int(ablation["mask_future_tail"])
        for n in nwp_names:
            ctx.cols[n].fut[:, -k:] = np.nan
    if ablation.get("mask_hist_tail"):
        k = int(ablation["mask_hist_tail"])
        for n in nwp_names:
            ctx.cols[n].hist[:, -k:] = np.nan
    if ablation.get("nwp_lag_days"):
        _apply_nwp_lag(df, ctx, nwp_names, int(ablation["nwp_lag_days"]), report)

    # 4. 衍生量
    for item in cfg.get("derived", []) or []:
        try:
            ctx.cols[item["name"]] = run_op(ctx, item["op"], item.get("args"))
            report["derived_ok"].append(item["name"])
        except KeyError as e:
            report["derived_fail"][item["name"]] = str(e)

    # 5. 补全链
    fb = cfg.get("fallback", {}) or {}
    apply = bool(ablation.get("apply_fallback", True))
    if apply:
        for name, chain in (fb.get("chains") or {}).items():
            if not ctx.has(name) and not any(ctx.has(c) or c.startswith("const:") or c in ("ghi_clear_clock",) for c in chain):
                continue
            rep = apply_fallback(ctx, name, list(chain), fb)
            if rep.n_interp or rep.filled_by or rep.n_dropped:
                report["fallback"].append(rep.as_dict())
    report["warnings"].extend(ctx.warnings)

    # 6. 组装输出
    out = df[["timestamp_win", "station"]].copy()
    out["cap_power_on"] = cap
    out["lat"], out["lon"] = ctx.lat, ctx.lon
    out[tcol] = list(ctx.cols[tcol].hist.astype(np.float32))
    if label_fut is not None:
        out[lcol] = list(label_fut.astype(np.float32))
    hist_only_names = {std for s in cfg["columns"]["sources"].values() if s.get("history_only")
                       for std in (s.get("fields") or {})}
    for name, pair in ctx.cols.items():
        if name == tcol:
            continue
        out[name] = list(pair.hist.astype(np.float32))
        if name not in hist_only_names:
            out[name + suffix] = list(pair.fut.astype(np.float32))
    frames = {st: g.reset_index(drop=True) for st, g in out.groupby("station", sort=True)}
    return frames, report


def _apply_nwp_lag(df: pd.DataFrame, ctx: BuildContext, names: list, lag_days: int, report: dict):
    """用 lag_days 天前同站起报的 NWP 顶替当前行：前一行的 hist/fut 拼起来整体右移 lag 步。
    覆盖不到的部分置 NaN（超短期 192 点未来覆盖不了 96 步延迟，会大面积 NaN → 交给补全链）。"""
    lag = lag_days * 96
    key = {(s, t): i for i, (s, t) in enumerate(zip(df["station"], df["timestamp_win"]))}
    prev_idx = np.array([key.get((s, t - pd.Timedelta(days=lag_days)), -1)
                         for s, t in zip(df["station"], df["timestamp_win"])])
    C, P = ctx.context_len, ctx.pred_len
    n_missing = int((prev_idx < 0).sum())
    for n in names:
        full = ctx.cols[n].full().astype(float)
        new = np.full_like(full, np.nan)
        ok = prev_idx >= 0
        src = full[prev_idx[ok]]
        # 前一行时间轴比当前早 lag 步：当前位置 j 对应前一行位置 j + lag
        L = full.shape[1]
        take = min(L - lag, L)
        if take > 0:
            new[ok, :take] = src[:, lag:lag + take]
        ctx.cols[n] = Pair.from_full(new, C)
    report["warnings"].append(f"消融：NWP 延迟 {lag_days} 天，{n_missing} 行找不到前一天的起报")


# ----------------------------------------------------------------------------- 落盘
def write_feature_store(frames: dict, out_dir: Path, report: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    for st, g in frames.items():
        g.to_parquet(out_dir / f"station={st}.parquet", index=False)
    with open(out_dir / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("特征库写入 %s：%d 站", out_dir, len(frames))


def feature_store_dir(cfg: dict, predict_type: str, ablation_id: Optional[str] = None) -> Path:
    base = _abs(cfg["data"]["feature_store"].format(predict_type=predict_type))
    return base / f"abl_{ablation_id}" if ablation_id else base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--predict_type", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--ablation", default=None, help='JSON，如 \'{"drop_sources": ["ec"]}\'')
    ap.add_argument("--ablation_id", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    pt = a.predict_type or cfg["data"]["predict_type"]
    abl = json.loads(a.ablation) if a.ablation else None
    frames, report = build(cfg, pt, abl)
    out = Path(a.out_dir) if a.out_dir else feature_store_dir(cfg, pt, a.ablation_id)
    write_feature_store(frames, out, report)
    print(json.dumps({k: report[k] for k in ("rows", "stations", "mapped", "missing_raw", "derived_ok",
                                              "derived_fail", "warnings")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
