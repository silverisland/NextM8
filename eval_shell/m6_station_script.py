# -*- coding: utf-8 -*-
"""m6_station_script.py —— M6（TimesFM-3 + LoRA + conformal 校准）单站 192 步评测入口。

M6 的 checkpoint_dir 是 train_m6 / train_m6.sh 的产物目录（runs/m6/<run_id>）：
  model_config.yaml     训练配置（推理时 run.out_dir 指向本目录）
  model.safetensors     LoRA + output_head 权重（底座仍从 timesfm3.checkpoint 读）；旧 run 为 lora_weights.pt
  calibration.parquet   conformal 校准表（calibration.enabled 时）
全部站都用这份 LoRA，模型全部冻结（旧 run 目录里的 station_routing.json 不再读）。
M6 的 safetensors 只存微调增量，不能交给 eval_station_script.load_model_from_checkpoint 做 strict load，
所以与 m5_station_script.py 一样自己装模型：
  产物目录 -> forecast_backbones.m6.station_eval.load_trained_model -> 单站推理（与训练期验证同一条 evaluate_model）

指标与写文件（SolarMetricCalculator / NanwangMetric / extract_summary / NumpyJSONEncoder / station_cap_from_df）
与 m5_station_script.py 逐行一致，输出与 M1-M5 的 metrics_ens 同构，可直接进 make_models_summary 汇总。

产出：
    out_dir/{STATION}-metrics.xlsx   # 6 sheet + nanwang sheet
    out_dir/{STATION}-summary.json   # 键与单模型一致

用法（cwd=仓库根）：
    python eval_shell/m6_station_script.py \
        --checkpoint_dir runs/m6/lora_20260922_1200 \
        --station_path datasets/test/station=1015.parquet \
        --out_dir runs/m6/lora_20260922_1200/metrics_ens \
        --cap_power 465

可选历史尾部缩放（--tail_scale/--tail_len）：与 m5_station_script.py 相同，只缩放推理输入里
observe_power 历史序列的末尾 --tail_len 个点，不改磁盘上的 parquet。
"""

import argparse
import json
import sys
import time
from pathlib import Path

# eval_shell 无 __init__.py；eval_station_script 与 forecast_backbones 都在仓库根。
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from eval_station_script import (
    NumpyJSONEncoder,
    extract_summary,
    station_cap_from_df,
)
from forecast_backbones.m6.station_eval import RESULT_COLS, load_trained_model, predict_station


def infer_station_m6(checkpoint_dir, station_path, cap_power, batch_size=None, device="cuda",
                     config=None, tail_scale=None, tail_len=4, scratch_dir=None):
    """单站 M6 推理 -> (st, cap, result_df)。result_df 列与 m5_station_script.infer_station_m5 一致：
    timestamp_win / station / observe_power_predict / observe_power_future，数据带 cap_power_on 时一并带上。"""
    import pandas as pd

    station_path = Path(station_path)
    station_df = pd.read_parquet(station_path)

    model, cfg = load_trained_model(checkpoint_dir, config=config, device=device, batch_size=batch_size)
    t0 = time.time()
    st, res = predict_station(model, station_path, tail_scale=tail_scale, tail_len=tail_len, scratch_dir=scratch_dir)
    print(f"[m6] 站 {st}（station_id={int(res['station_id'].iloc[0])}）推理 {len(res)} 行，"
          f"用时 {time.time() - t0:.1f}s")

    result_df = res[RESULT_COLS].copy()
    result_df["timestamp_win"] = pd.to_datetime(result_df["timestamp_win"])
    if "cap_power_on" in station_df.columns:
        src_ts = pd.to_datetime(station_df["timestamp_win"]).dt.floor("s")
        if len(src_ts) != len(result_df) or (src_ts.dt.tz is None and
                                             not (src_ts.to_numpy() == result_df["timestamp_win"].to_numpy()).all()):
            raise ValueError(f"推理结果（{len(result_df)} 行）与 {station_path}（{len(station_df)} 行）的行序对不上")
        result_df["cap_power_on"] = station_df["cap_power_on"].to_numpy()

    cap = station_cap_from_df(station_df, cap_power)
    print(f"[station] {st}  cap_power={cap}  行数={len(result_df)}")
    return st, cap, result_df


def run_station(checkpoint_dir, station_path, out_dir, cap_power, batch_size=None, device="cuda",
                config=None, tail_scale=None, tail_len=4):
    """对单个站点完成 M6 推理 -> 指标 -> xlsx + summary.json。"""
    from pipelines.metric import SolarMetricCalculator
    from pipelines.nanwang_metric import NanwangMetric
    import pandas as pd

    station_path = Path(station_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    st, cap, result_df = infer_station_m6(
        checkpoint_dir, station_path, cap_power,
        batch_size=batch_size, device=device, config=config,
        tail_scale=tail_scale, tail_len=tail_len, scratch_dir=out_dir,
    )

    # ---- solar 指标 + 南网月度表，合并写 {站}-metrics.xlsx ----
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
            station_col="station",
        )
        report = evaluator.generate_report(result_df, writer=writer)
        nanwang_tables[st].to_excel(writer, sheet_name="nanwang")

    # ---- summary.json ----
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
    parser = argparse.ArgumentParser(
        description="M6（TimesFM-3 + LoRA）逐站 192 步回测：训练产物 -> 推理 -> 指标"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="M6 训练产物目录 runs/m6/<run_id>（含 model_config.yaml / model.safetensors）")
    parser.add_argument("--station_path", type=str, required=True,
                        help="datasets/test/station=<站>.parquet")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="runs/m6/<run_id>/metrics_ens")
    parser.add_argument("--cap_power", type=float, default=465.0, help="回退容量")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="推理 batch；不给用配置里的 eval.batch_size")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="产物目录没有 model_config.yaml（旧 run）时给训练用的 yaml")
    parser.add_argument("--tail_scale", type=float, default=None,
                        help="对推理输入 observe_power 末尾 tail_len 个点做内存级缩放"
                             "（不落盘），如 0.1=末尾4点×0.1；None=不缩放（默认）")
    parser.add_argument("--tail_len", type=int, default=4,
                        help="尾部缩放的点数（默认 4）")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_station(
        Path(args.checkpoint_dir), Path(args.station_path), Path(args.out_dir),
        args.cap_power, args.batch_size, device, config=args.config,
        tail_scale=args.tail_scale, tail_len=args.tail_len,
    )


if __name__ == "__main__":
    main()
