# -*- coding: utf-8 -*-
# forecast_backbones/m8/station_report.py
"""M8 逐站完整评测（输出与 M1–M7 的 metrics_ens 同构，可直接进 make_models_summary 汇总）：
  每个测试 parquet（一站一文件）→ 全部起报行推理 → SolarMetricCalculator 五张表
  （ods_ultra_short / ods_short / ultra_short / short / 48hours）+ NanwangMetric 月度表（nanwang_ultra_short / nanwang_short_D+1）
  → <out>/metrics_ens/{站}-metrics.xlsx + {站}-summary.json；
  汇总 <out>/reports/station_metrics.csv（四个主指标逐站）与 summary.csv（跨站均值、seen/unseen）。

  python -m forecast_backbones.m8.station_report --runs runs/m8/refit/e1_final_s0 [更多 run = 集成] \\
         --test "datasets/test/*.parquet" --out runs/m8/refit/full_eval [--engine unified|native|auto]

推理引擎 --engine（默认 unified）：
  unified  pipelines.pipeline_wrapper_inference.unified_inference（线上同一接口；M8 的 forward(batch_dict) 与 m6 契约相同，
           .logits 就是决策后的提交曲线）。每个成员各跑一遍，多成员取点预测平均。
  native   仓库内 evaluate.run_inference：成员的原始分位逐分位平均后再过一次决策层（训练期验证走的同一条路）。
  auto     unified 能 import 就用 unified，否则 native。
单成员时两种引擎结果逐位相同（同一个 forward、同一套决策）；决策规则都可用 --point / --decision-yaml 覆盖。
summary.json 的键：仓库根的 eval_station_script.extract_summary 能 import 时用它（与其它模型逐键一致），
否则用本模块的回退实现（键名 <sheet>_<指标>，如 ods_short_rmse_acc；南网两项 nanwang_short_D+1 / nanwang_ultra_short）。
容量：每站一个固定值，只取 station_meta.csv 的 cap_mw（模型归一化与两套指标的分母是同一个数）；数据里的 cap_power_on 不用。"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .evaluate import expand_paths, load_member, decision_override, describe_point, run_inference
    from .features import station_cap_mw
    from .data import _dataset_with_station_id
    from .solar import attach_station_id, norm_station_name
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m8.evaluate import expand_paths, load_member, decision_override, describe_point, run_inference
    from forecast_backbones.m8.features import station_cap_mw
    from forecast_backbones.m8.data import _dataset_with_station_id
    from forecast_backbones.m8.solar import attach_station_id, norm_station_name
try:
    from ._parquet import station_keys
except ImportError:
    from forecast_backbones.m8._parquet import station_keys
try:
    from ...pipelines.transformer_dataset import UnifiedDataset
    from ...pipelines.metric import SolarMetricCalculator, MIN_DAYS
    from ...pipelines.nanwang_metric import NanwangMetric
except ImportError:
    from pipelines.transformer_dataset import UnifiedDataset
    from pipelines.metric import SolarMetricCalculator, MIN_DAYS
    from pipelines.nanwang_metric import NanwangMetric

RESULT_COLS = ["timestamp_win", "station", "observe_power_predict", "observe_power_future"]


def station_key(station_path, station_id_col: str = "station_id") -> str:
    """单站 parquet 的站号：优先 station 列，没有再用 station_id 列；文件里不止一个站直接报错。"""
    import pyarrow.parquet as pq
    names = pq.read_schema(str(station_path)).names
    col = "station" if "station" in names else station_id_col if station_id_col in names else None
    if col is None:
        raise ValueError(f"{station_path} 既没有 station 列也没有 {station_id_col!r} 列")
    keys = pd.unique(station_keys(pd.read_parquet(station_path, columns=[col])[col].to_numpy()))
    if len(keys) != 1:
        raise ValueError(f"{station_path} 里有 {len(keys)} 个站 {list(keys)[:5]}，单站评测只接受一个站")
    return str(keys[0])

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ODS_SHEETS = ("ods_ultra_short", "ods_short")
ODS_METRICS = ("rmse_acc", "mae_acc", "ACC_acc")
DS_SHEETS = ("ultra_short", "short", "48hours")
DS_METRICS = ("RMSE_SCORE", "RMSE", "MAE")
NANWANG_ROWS = ("nanwang_ultra_short", "nanwang_short_D+1")
MAIN_METRICS = ("nanwang_short_D+1", "nanwang_ultra_short", "ods_short_rmse_acc", "ods_ultra_short_rmse_acc")


# ------------------------------------------------------------------ 可选依赖（服务器上有、仓库里没有）
def unified_fn():
    """线上推理接口；本机没有返回 None。"""
    try:
        from pipelines.pipeline_wrapper_inference import unified_inference
        return unified_inference
    except ImportError:
        return None


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return str(o)
        return super().default(o)


def summary_helpers():
    """-> (extract_summary, JSON encoder, 来源)。eval_station_script 在仓库根（服务器）时用它，键与 M1–M7 逐个一致。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from eval_station_script import extract_summary, NumpyJSONEncoder
        return extract_summary, NumpyJSONEncoder, "eval_station_script"
    except ImportError:
        return fallback_extract_summary, _NumpyEncoder, "fallback"


def _month_mean(table: pd.DataFrame, rows, day_row: str = "day_count") -> dict:
    """ODS/DS 月表（.T 后，列 = 月份位置编号）→ {行名: 月均值}。有 mean 列（generate_report 给 ODS 表加的，已剔小月）直接用；
    否则剔除样本天数 < MIN_DAYS 的月份后各月等权。"""
    out = {}
    if "mean" in table.columns:
        for r in rows:
            v = table.at[r, "mean"] if r in table.index else np.nan
            out[r] = None if pd.isna(v) else float(v)
        return out
    months = [c for c in table.columns if c != "mean"]
    if day_row in table.index:
        days = pd.to_numeric(table.loc[day_row, months], errors="coerce")
        months = [c for c in months if pd.notna(days[c]) and float(days[c]) >= MIN_DAYS]
    for r in rows:
        if r not in table.index or not months:
            out[r] = None
            continue
        v = pd.to_numeric(table.loc[r, months], errors="coerce").mean()
        out[r] = None if pd.isna(v) else float(v)
    return out


def main_metrics(report: dict, nanwang_table: pd.DataFrame) -> dict:
    """station_metrics.csv 的指标列：直接从表里取，不依赖 extract_summary 的键名。"""
    out = {}
    for k in ODS_SHEETS:
        for m, v in _month_mean(report[k], ODS_METRICS).items():
            out[f"{k}_{m}"] = v
    for r in NANWANG_ROWS:
        v = nanwang_table.at[r, "mean"] if r in nanwang_table.index and "mean" in nanwang_table.columns else np.nan
        out[r] = None if pd.isna(v) else float(v)
    return out


def fallback_extract_summary(report: dict, nanwang_tables: dict) -> dict:
    """<sheet>_<指标> 的扁平字典：ODS 两张表 + DS 三张表的月均值，再加南网两项（该站月度表的 mean 列）。"""
    s = {}
    for k in ODS_SHEETS:
        for m, v in _month_mean(report[k], ODS_METRICS).items():
            s[f"{k}_{m}"] = v
    for k in DS_SHEETS:
        for m, v in _month_mean(report[k], DS_METRICS).items():
            s[f"{k}_{m}"] = v
    tab = next(iter(nanwang_tables.values())) if nanwang_tables else None
    for r in NANWANG_ROWS:
        v = tab.at[r, "mean"] if tab is not None and r in tab.index and "mean" in tab.columns else np.nan
        s[r] = None if pd.isna(v) else float(v)
    return s


# ------------------------------------------------------------------ 推理
def frame_with_station_id(df: pd.DataFrame, cfg: dict, meta) -> pd.DataFrame:
    """与 data._dataset_with_station_id 同一口径，但作用在 DataFrame 上（unified_inference 吃 DataFrame）：
    station_id 列是文本 → 数字文本直接转整数，站名按 station_meta 映射；没有 station_id 列 → 按 station 列映射；
    没有 station 列 → 用 station_meta 的站名补上（unified_inference 的 entity_cols 要用）。"""
    id_col = str(cfg["dataset"]["station_id_col"])
    if id_col in df.columns and not pd.api.types.is_numeric_dtype(df[id_col]):
        vals = df[id_col].map(norm_station_name)
        num = pd.to_numeric(vals, errors="coerce")
        if num.notna().all():
            df = df.assign(**{id_col: num.astype(np.int64)})
        else:
            df = attach_station_id(df.drop(columns=[id_col]).assign(_name=vals), meta, id_col=id_col, name_col="_name").drop(columns=["_name"])
    elif id_col not in df.columns:
        df = attach_station_id(df, meta, id_col=id_col, name_col="station")
    else:
        df = df.assign(**{id_col: df[id_col].astype(np.int64)})
    if "station" not in df.columns:
        df = df.assign(station=df[id_col].map(lambda s: meta.name(int(s))))
    return df


def _naive(ts: pd.Series) -> pd.Series:
    """与 UnifiedDataset._to_datetime64 同一约定：带时区的先转 UTC 再去掉时区；无时区原样（仓库数据都是无时区本地时间）。"""
    return ts.dt.tz_convert("UTC").dt.tz_localize(None) if getattr(ts.dt, "tz", None) is not None else ts


def _row_caps(df: pd.DataFrame, cfg: dict, meta) -> np.ndarray:
    """与 FeatureBuilder 同一口径：每站固定容量 = station_meta 的 cap_mw（不看数据里的 cap_power_on）。"""
    id_col = str(cfg["dataset"]["station_id_col"])
    caps = {int(s): station_cap_mw(meta, int(s)) for s in df[id_col].unique()}
    return np.array([caps[int(s)] for s in df[id_col]], float)


def predict_native(members, path, decision: dict, batch_size: int, scratch_dir, progress: bool) -> pd.DataFrame:
    """仓库内引擎：该站全部起报行 → run_inference（成员分位平均 → 决策）。"""
    m0 = members[0]
    ds = _dataset_with_station_id([str(path)], m0.cfg, m0.station_meta, scratch_dir=scratch_dir, reuse=False)
    if len(ds) == 0:
        raise ValueError(f"{path} 没有可推理的行")
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    return run_inference(members, ds, range(len(ds)), collate, decision, batch_size, progress)


def predict_unified(members, path, decision: dict, batch_size: int, unified) -> pd.DataFrame:
    """线上接口引擎：每个成员 unified_inference 一遍（.logits = 决策后的提交曲线），多成员取点预测平均。
    与 eval_pretrain.infer_station_pretrain 同一调用方式；推理输入先删掉未来真值列。"""
    m0 = members[0]
    cfg = m0.cfg
    ds_cfg = cfg["dataset"]
    ts_col, label_col, H = str(ds_cfg["ts_col"]), str(ds_cfg["label_col"]), int(ds_cfg["pred_len"])
    df = frame_with_station_id(pd.read_parquet(path), cfg, m0.station_meta)
    keys = [ts_col, "station"]
    infer_df = df.drop(columns=[c for c in (label_col, "observe_power_true") if c in df.columns])
    ts_in = _naive(pd.to_datetime(df[ts_col])).to_numpy("datetime64[s]")
    st_in = df["station"].astype(str).to_numpy()
    preds = []
    for m in members:
        m.cfg["decision"] = copy.deepcopy(decision)                      # forward 里 decide() 读 cfg['decision']
        out = unified(pipeline_backbone=m, ds_input=infer_df, timestamp_cols=[ts_col], entity_cols=keys,
                      feature_cols=None, batch_size=batch_size, prediction_to_numpy=True)
        col = type(m).__name__
        if col not in out.columns:
            raise KeyError(f"unified_inference 结果缺预测列 {col!r}，实际列：{list(out.columns)}")
        # 行序由 unified_inference 保证（shuffle=False、按批回填）；这里按位置对齐，再用实体列逐行核一遍
        if len(out) != len(df):
            raise ValueError(f"unified_inference 返回 {len(out)} 行，输入 {len(df)} 行（{path}）")
        bad_st = (out["station"].astype(str).to_numpy() != st_in)
        bad_ts = (pd.to_datetime(out[ts_col]).to_numpy("datetime64[s]") != ts_in)
        if bad_st.any() or bad_ts.any():
            raise ValueError(f"unified_inference 的输出与 {path} 的行对不上：站名不同 {int(bad_st.sum())} 行、时间不同 {int(bad_ts.sum())} 行")
        if out[col].isna().any():
            raise ValueError(f"{path}：{int(out[col].isna().sum())} 行没有预测——批量推理失败时 unified_inference 逐样本兜底、"
                             f"仍失败的行给 None，看前面的 warning 找原因")
        preds.append(np.stack([np.asarray(v, np.float32).reshape(-1)[:H] for v in out[col]]))
    point = np.mean(preds, axis=0).astype(np.float32)
    truth = (np.stack([np.asarray(v, np.float32).reshape(-1)[:H] for v in df[label_col]]) if label_col in df.columns
             else np.full((len(df), H), np.nan, np.float32))
    return pd.DataFrame({ts_col: ts_in.astype("datetime64[ns]"), "station": df["station"].to_numpy(),
                         "station_id": df[str(ds_cfg["station_id_col"])].astype(int).to_numpy(),
                         "cap": _row_caps(df, cfg, m0.station_meta),
                         "observe_power_predict": list(point), "observe_power_future": list(truth)})


# ------------------------------------------------------------------ 指标与写文件
def write_station_report(res: pd.DataFrame, st: str, cap: float, out_dir, extract_summary, encoder) -> tuple[dict, dict]:
    """与 eval_shell/m6_station_script.run_station 同一套：五张 solar 表 + nanwang 表合写 {站}-metrics.xlsx，{站}-summary.json。
    -> (summary.json 的字典, station_metrics.csv 的一行)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = res[RESULT_COLS].copy().assign(station=st)
    result_df["timestamp_win"] = pd.to_datetime(result_df["timestamp_win"])
    nanwang_tables = NanwangMetric().monthly_table(result_df, gccap=cap)
    xlsx = out_dir / f"{st}-metrics.xlsx"
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        ev = SolarMetricCalculator(model_version=st, cap_power=cap, save_dir=str(out_dir), time_col="timestamp_win",
                                   groundtruth_col="observe_power_future", predict_col="observe_power_predict", station_col="station")
        report = ev.generate_report(result_df, writer=writer)
        nanwang_tables[st].to_excel(writer, sheet_name="nanwang")
    summary = extract_summary(report, nanwang_tables)
    summary["station"], summary["cap_power"], summary["rows"] = st, cap, int(len(result_df))
    (out_dir / f"{st}-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, cls=encoder), encoding="utf-8")
    return summary, {"station": st, "cap": cap, "rows": int(len(result_df)), **main_metrics(report, nanwang_tables[st])}


def evaluate_full(runs, test_paths, out_dir=None, engine: str = "unified", point: str | None = None, decision_yaml: str | None = None,
                  batch_size: int | None = None, device: str | None = None, tag: str | None = None, progress: bool = True) -> dict:
    t0 = time.time()
    runs = [str(r) for r in runs]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    members = [load_member(r, device) for r in runs]
    m0 = members[0]
    cfg = m0.cfg
    unified = unified_fn()
    if engine == "auto":
        engine = "unified" if unified is not None else "native"
    if engine not in ("unified", "native"):
        raise ValueError(f"--engine {engine!r} 不在 (auto, unified, native)")
    if engine == "unified" and unified is None:
        raise ImportError("--engine unified 需要 pipelines.pipeline_wrapper_inference.unified_inference（服务器上的线上推理接口），"
                          "本机 import 不到；改用 --engine native")
    extract_summary, encoder, src = summary_helpers()
    out = Path(out_dir) if out_dir else (Path(runs[0]) / "full_eval" if len(runs) == 1 else Path(runs[0]).parent / "ensemble_full_eval")
    ens, rep, scratch = out / "metrics_ens", out / "reports", out / ".data"
    for d in (ens, rep, scratch):
        d.mkdir(parents=True, exist_ok=True)
    paths = expand_paths(test_paths)
    decision = decision_override(cfg["decision"], point, decision_yaml)
    bs = int(batch_size or cfg["eval"].get("batch_size", 256))
    known = {int(s) for s in (cfg["dataset"].get("known_stations") or [])}
    id_col = str(cfg["dataset"]["station_id_col"])
    print(f"[m8-full] 成员 {len(members)} 个，站文件 {len(paths)} 个，引擎 {engine}，决策 {describe_point(decision)}，"
          f"summary 键来自 {src}，device={device}", flush=True)
    rows, summaries, frames = [], {}, []
    for p in paths:
        st = station_key(p, id_col)
        t1 = time.time()
        res = (predict_unified(members, p, decision, bs, unified) if engine == "unified"
               else predict_native(members, p, decision, bs, scratch, progress))
        if res["station_id"].nunique() != 1:
            raise ValueError(f"{p} 映射成了多个 station_id {sorted(res['station_id'].unique())[:5]}，逐站评测只接受一站一文件")
        cap = station_cap_mw(m0.station_meta, int(res["station_id"].iloc[0]))      # 指标分母 = 站表容量，与模型归一化同一个数
        if not np.allclose(res["cap"].to_numpy(float), cap):
            raise AssertionError(f"{p}：推理结果里的 cap 与站表不一致（{res['cap'].unique()[:3]} vs {cap}）")
        summ, row = write_station_report(res, st, cap, ens, extract_summary, encoder)
        row["seen"] = (int(res["station_id"].iloc[0]) in known) if known else True
        row["minutes"] = round((time.time() - t1) / 60.0, 3)
        rows.append(row)
        summaries[st] = summ
        frames.append(res.assign(station=st))
        print(f"[m8-full] 站 {st} cap={cap:g} 行 {len(res)}：" +
              "  ".join(f"{k}={row[k]:.4g}" if row[k] is not None else f"{k}=nan" for k in MAIN_METRICS), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(rep / "station_metrics.csv", index=False)
    # 原始逐行预测（全部起报行、已过决策层），与 evaluate.py 的 result_test.parquet 同列，可直接喂 diagnose_test.py
    pd.concat(frames, ignore_index=True).to_parquet(out / "result_test.parquet", index=False)
    tag = tag or ("+".join(Path(r).name for r in runs) if len(runs) <= 3 else f"ens{len(runs)}_{Path(runs[0]).name}")
    num = {k: pd.to_numeric(table[k], errors="coerce") for k in MAIN_METRICS}
    fleet = {"tag": tag, "engine": engine, "n_members": len(members), "point_rule": describe_point(decision),
             "n_stations": int(len(table)), "n_rows": int(table["rows"].sum())}
    for k in MAIN_METRICS:
        fleet[k] = float(num[k].mean()) if num[k].notna().any() else float("nan")
    seen = table["seen"].astype(bool)
    fleet["nanwang_short_D+1_seen"] = float(num["nanwang_short_D+1"][seen].mean()) if seen.any() else float("nan")
    fleet["nanwang_short_D+1_unseen"] = float(num["nanwang_short_D+1"][~seen].mean()) if (~seen).any() else float("nan")
    fleet["n_unseen_stations"] = int((~seen).sum())
    fleet.update({"summary_keys": src, "eval_minutes": (time.time() - t0) / 60.0, "members": "|".join(runs), "test": "|".join(paths)})
    pd.DataFrame([fleet]).to_csv(rep / "summary.csv", index=False)
    (rep / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(table.to_string(index=False))
        print(pd.DataFrame([{k: v for k, v in fleet.items() if k not in ("members", "test")}]).to_string(index=False))
    print(f"[m8-full] 产物：{ens}/{{站}}-metrics.xlsx + {{站}}-summary.json / {rep}/station_metrics.csv / {rep}/summary.csv / "
          f"{out}/result_test.parquet（逐行预测，供 diagnose_test.py）")
    return {"summary": fleet, "station_table": table, "summaries": summaries}


def main(argv=None):
    ap = argparse.ArgumentParser(description="M8 逐站完整评测：ods_short / ods_ultra_short / nanwang_short_D+1 / nanwang_ultra_short（支持集成与 unified_inference）")
    ap.add_argument("--runs", nargs="+", required=True, help="train_m8 的产物目录，多个 = 集成")
    ap.add_argument("--test", nargs="+", required=True, help="测试 parquet：目录 / 文件 / 通配符（一站一文件）")
    ap.add_argument("--out", default=None, help="输出目录（默认 单成员 <run>/full_eval，多成员 <父目录>/ensemble_full_eval）")
    ap.add_argument("--engine", default="unified", choices=["unified", "native", "auto"])
    ap.add_argument("--point", default=None, help="覆盖 decision.point.default（median / p_star / p_star_l1），并清空 by_step")
    ap.add_argument("--decision-yaml", default=None, help="合并该 yaml 的 decision 段")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args(argv)
    evaluate_full(a.runs, a.test, a.out, a.engine, a.point, a.decision_yaml, a.batch_size, a.device, a.tag, not a.no_progress)


if __name__ == "__main__":
    main()
