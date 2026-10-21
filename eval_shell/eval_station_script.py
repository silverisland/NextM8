"""
eval_station_script.py —— 逐站回测入口：模型 checkpoint -> 单站推理 -> 该站指标 -> xlsx + summary.json。

与 eval_with_metric.py（多站合并一次评测）不同，本脚本完全仿照参考版
/data/100pretrain_code 的逐站回测语义：

    1. 每个站用 data/test/station=<站>.parquet 作为推理输入（含该站真值
       observe_power_future + 装机容量 cap_power_on）；
    2. unified_inference 对整表（实为单站）推理，得到 observe_power_predict；
    3. SolarMetricCalculator 逐站回测：cap 取该站真实 cap_power_on，
       rmse_acc = 1 - RMSE/cap_power（ODS 按日等权平均后按月，DS 按样本），
       输出 ods_ultra_short / ods_short / ultra_short / short / 48hours 五个 sheet；
    4. NanwangMetric 单站月度表（每站一个 sheet，无跨站 fleet 汇总）。

产出：
    runs/{MODEL}/{RUN_ID}/metrics_ens/{STATION}-metrics.xlsx   # 该站全套指标（含南网月度表 nanwang sheet）
    runs/{MODEL}/{RUN_ID}/metrics_ens/{STATION}-summary.json  # 供汇总对比的关键指标

指标在 summary.json 中统一抽取（与回测只读已推理结果、不重新训练一致）：
    ods_ultra_short.rmse_acc.mean / mae_acc / ACC_acc
    ods_short.rmse_acc.mean / mae_acc / ACC_acc
    南网 ult_sh/ult_short 月度 mean（若有）

用法：
    python eval_station_script.py \
        --model_name m1 \
        --checkpoint_dir runs/m1/pretrain_100m_v1/model_result \
        --station_path data/test/station=白马湖光伏电站.parquet \
        --out_dir runs/m1/pretrain_100m_v1/metrics_ens \
        --cap_power 60 \
        --batch_size 512
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

BASE_PATH = str(Path(__file__).resolve().parent)


def load_model_from_checkpoint(model_name, checkpoint_dir, device):
    from pipelines.model_registry import build_model
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "model_config.yaml"
    safetensors_path = checkpoint_dir / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少模型配置文件: {config_path}")
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"缺少模型权重文件: {safetensors_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    model = build_model(model_name=model_name, model_config=model_config)
    state_dict = load_file(str(safetensors_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint 与模型 {model_name} 结构不匹配: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model, model_config


_INFO_CSV_CACHE: dict | None = None


def _cap_info_path() -> Path:
    """仓库根 info.csv（station,cap_power_on 两列，17 站权威装机容量）。"""
    return Path(__file__).resolve().parent / "info.csv"


def load_cap_from_info_csv() -> dict[str, float]:
    """读取仓库根 info.csv -> {station(str): cap(float)}，作为每站装机容量的唯一权威来源。

    与逐站 parquet 的 cap_power_on（有的字段随时间变化、非恒定）不同，info.csv 是每站
    一个确定的装机容量。station 列含纯数字与中文站名，统一转 str 归一化，方便按站名查表。
    """
    global _INFO_CSV_CACHE
    if _INFO_CSV_CACHE is None:
        info = pd.read_csv(_cap_info_path())
        info = info.dropna(subset=["station"])
        info["station"] = info["station"].astype(str)
        _INFO_CSV_CACHE = dict(zip(
            info["station"],
            pd.to_numeric(info["cap_power_on"], errors="coerce"),
        ))
    return _INFO_CSV_CACHE


def station_cap_from_df(df, fallback):
    """单站装机容量：优先取仓库根 info.csv（每站权威值），查不到再回退全局 fallback。

    不再使用 parquet 自带的 cap_power_on（该列在部分站随时间变化、非恒定，且首行
    值不可靠）；info.csv 是每站装机容量的唯一权威来源。
    """
    st = None
    if "station" in df.columns:
        st = str(df["station"].iloc[0])
    cap = load_cap_from_info_csv().get(st)
    if cap is not None and pd.notna(cap) and float(cap) > 0:
        return float(cap)
    return float(fallback)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        if hasattr(o, "item"):
            return o.item()
        return str(o)


def extract_summary(report, nanwang_tables) -> dict:
    """从 generate_report 结果中抽关键指标 -> summary dict。"""
    summary = {}

    def _grab(sheet, row, col="mean"):
        df = report.get(sheet)
        if df is None or row not in df.index:
            return None
        if col not in df.columns:
            # DS 类表没有 mean 列，用数值列取均值（多数只有一列 0）
            num = pd.to_numeric(df.loc[row], errors="coerce").dropna()
            return float(num.mean()) if len(num) else None
        v = df.loc[row, col]
        return None if pd.isna(v) else float(v)

    for sheet, rows in (("ods_ultra_short", ("rmse_acc", "mae_acc", "ACC_acc")),
                        ("ods_short", ("rmse_acc", "mae_acc", "ACC_acc"))):
        for r in rows:
            v = _grab(sheet, r, "mean")
            if v is not None:
                summary[f"{sheet}.{r}.mean"] = v
    # 南网月表：每站（传共享 writer 时 nanwang_tables 是 {站: DataFrame}）；
    # 这里只可能在单站调用，取唯一站的 mean。
    for st, tab in (nanwang_tables or {}).items():
        for r in ("nanwang_ultra_short", "nanwang_short_D+1"):
            if r in tab.index and "mean" in tab.columns:
                v = tab.loc[r, "mean"]
                if not pd.isna(v):
                    summary[f"nanwang.{r}.mean"] = float(v)
    return summary


def infer_station(model_name, checkpoint_dir, station_path, cap_power,
                  batch_size=512, device="cuda", drop_label_cols=True):
    """单站推理纯函数，返回 (st, cap, result_df)，不做指标计算。

    -> (st, cap, result_df)
        st:        站名（result_df['station'] 首值）
        cap:       该站真实装机容量（float，取该站 parquet 的 cap_power_on）
        result_df: 含 timestamp_win / station / observe_power_predict /
                   observe_power_future（预测列为 192 长的数组）

    仅负责 加载模型 -> unified_inference -> 拼 observe_power_future，
    与指标/cap 计算逻辑分离，供逐站并行驱动复用。

    drop_label_cols=True（默认）: 推理前从输入中剔除 label/目标列
    （observe_power_future 等），防止未来真值作为特征混入 batch_dict。
    预测列的未来真值之后由本函数从 station_df 重新 merge 回 result_df，
    指标不受影响。
    """
    from pipelines.pipeline_wrapper_inference import unified_inference

    station_path = Path(station_path)
    station_df = pd.read_parquet(station_path)
    station_df["timestamp_win"] = pd.to_datetime(station_df["timestamp_win"])

    # ---- 1. 加载模型 ----
    model, model_config = load_model_from_checkpoint(model_name, checkpoint_dir, device)

    # ---- 2. 推理（输入=该站 parquet）----
    # 与 eval_with_metric 一致：优先用 config 里训练的 batch_size 兜底（m3/Moirai 系）。
    # 各模型 config 中 batch_size 存放位置不同：m1/m2_new/m4 在顶层，m3 在
    # hyperparameter 下——两处都读，取较小的作为上限，避免 8 卡并行时 OOM。
    bs = batch_size
    cfg_batch_sizes = []
    if isinstance(model_config, dict):
        hp = model_config.get("hyperparameter", {}) or {}
        for v in (hp.get("batch_size"), model_config.get("batch_size")):
            if isinstance(v, (int, float)) and v > 0:
                cfg_batch_sizes.append(v)
    if cfg_batch_sizes:
        bs = min(bs, min(cfg_batch_sizes))

    # 防泄漏：推理输入不带 label/未来真值列（feature_cols=None 会把它们收进
    # batch_dict）。删除的是投喂给模型的 DataFrame 副本，不影响后续 merge 真值。
    infer_df = station_df
    if drop_label_cols:
        label_cols = [c for c in station_df.columns if c in (
            "observe_power_future",
            "observe_power_true",  # 未来如有其他真值列名，加在这里
        )]
        if label_cols:
            infer_df = station_df.drop(columns=label_cols)

    ds_output = unified_inference(
        pipeline_backbone=model,
        ds_input=str(station_path) if infer_df is station_df else infer_df,
        timestamp_cols=["timestamp_win"],
        entity_cols=["timestamp_win", "station"],
        feature_cols=None,
        batch_size=bs,
        prediction_to_numpy=True,
    )
    pred_col = type(model).__name__
    if pred_col not in ds_output.columns:
        raise KeyError(f"推理结果缺少预测列 {pred_col!r}, 实际列: {list(ds_output.columns)}")

    result_df = ds_output.rename(columns={pred_col: "observe_power_predict"})
    result_df["timestamp_win"] = pd.to_datetime(result_df["timestamp_win"])
    if "observe_power_future" not in result_df.columns:
        result_df = result_df.merge(
            station_df[["timestamp_win", "station", "observe_power_future"]],
            on=["timestamp_win", "station"], how="left",
        )

    # 站名 + 该站真实 cap（参考版逐站回测：cap 用该站容量）
    st = str(result_df["station"].iloc[0])
    cap = station_cap_from_df(station_df, cap_power)
    print(f"[station] {st}  cap_power={cap}  行数={len(result_df)}")
    return st, cap, result_df


def run_station(model_name, checkpoint_dir, station_path, out_dir,
                cap_power, batch_size=512, device="cuda",
                skip_infer=False, predict_parquet=None):
    """对单个站点完成 推理 -> 指标 -> xlsx + summary.json。

    skip_infer=True: 不加载/推理，直接读 predict_parquet（须含
    observe_power_predict 等列）作为预测结果算指标写 xlsx。
    """
    from pipelines.metric import SolarMetricCalculator
    from pipelines.nanwang_metric import NanwangMetric

    station_path = Path(station_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    station_df = pd.read_parquet(station_path)
    station_df["timestamp_win"] = pd.to_datetime(station_df["timestamp_win"])

    if skip_infer:
        # ---- 1s. 短路径：跳过模型加载/推理，直接读已有推理结果 ----
        if predict_parquet is None:
            raise ValueError("--no-infer 需要提供 --predict-parquet")
        result_df = pd.read_parquet(predict_parquet)
        result_df["timestamp_win"] = pd.to_datetime(result_df["timestamp_win"])
        if "observe_power_future" not in result_df.columns:
            result_df = result_df.merge(
                station_df[["timestamp_win", "station", "observe_power_future"]],
                on=["timestamp_win", "station"], how="left",
            )

        # 站名 + 该站真实 cap（参考版逐站回测：cap 用该站容量）
        st = str(result_df["station"].iloc[0])
        cap = station_cap_from_df(station_df, cap_power)
        print(f"[station] {st}  cap_power={cap}  行数={len(result_df)}")
    else:
        # ---- 1.2. 推理（纯函数）：加载模型 -> unified_inference -> 拼真值 ----
        st, cap, result_df = infer_station(
            model_name, checkpoint_dir, station_path, cap_power,
            batch_size=batch_size, device=device,
        )

    # ---- 3+4. solar 指标 + 南网月度表，合并写 {站}-metrics.xlsx ----
    # 南网月度表先算（out_xlsx=None -> 不落独立文件，只返回 {站: DataFrame}），
    # 随后作为 "nanwang" sheet 追加进同一份 metrics.xlsx，一站一个 Excel。
    nm = NanwangMetric()
    nanwang_tables = nm.monthly_table(result_df, gccap=cap)

    xlsx_path = out_dir / f"{st}-metrics.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        evaluator = SolarMetricCalculator(
            model_version=st,
            cap_power=cap,
            save_dir=str(out_dir),
            time_col="timestamp_win",
            groundtruth_col="observe_power_future",
            predict_col="observe_power_predict",
            station_col="station",       # 单站 cap 由 cap_power 直接给定，不分组
        )
        report = evaluator.generate_report(result_df, writer=writer)
        # 南网月度表作为独立 sheet（sheet 名固定 "nanwang"）追加
        nanwang_tables[st].to_excel(writer, sheet_name="nanwang")

    # ---- 5. summary.json ----
    summary = extract_summary(report, nanwang_tables)
    summary["station"], summary["cap_power"], summary["rows"] = st, cap, int(len(result_df))
    summary_path = out_dir / f"{st}-summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)

    print(f"[summary] {summary_path}")
    for k, v in summary.items():
        print(f"    {k} = {v}")
    print(f"完成：{xlsx_path}（含 nanwang sheet）")
    return summary


def main():
    parser = argparse.ArgumentParser(description="逐站回测：checkpoint -> 单站推理 -> 指标")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--station_path", type=str, required=True,
                        help="data/test/station=<站>.parquet")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="runs/{MODEL}/{RUN_ID}/metrics_ens")
    parser.add_argument("--cap_power", type=float, default=465.0, help="回退容量")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-infer", action="store_true",
                        help="只算指标写 xlsx（读取已推理结果），跳过模型加载与推理")
    parser.add_argument("--predict-parquet", type=str, default=None,
                        help="--no-infer 时使用的已有推理结果 parquet（须含 observe_power_predict）")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_station(
        args.model_name, Path(args.checkpoint_dir), args.station_path,
        Path(args.out_dir), args.cap_power, args.batch_size, device,
        skip_infer=args.no_infer, predict_parquet=args.predict_parquet,
    )


if __name__ == "__main__":
    main()