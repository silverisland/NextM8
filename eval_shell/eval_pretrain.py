# -*- coding: utf-8 -*-
"""eval_pretrain.py —— 零样本基础模型（M5 TimesFM-3 / M7 PatchTST-FM-r2 …）单站 192 步评测入口。
（原 m5_station_script.py，推广为按 --model 选模型的公用脚本。）

背景：M1-M4 的 eval_station_script.py 走 load_model_from_checkpoint,要求
model.safetensors + strict load_state_dict；零样本 adapter 没有训练
checkpoint、不 load_state_dict（权重由 adapter 内部按配置里的 checkpoint 字段加载：
M5 为 timesfm3.checkpoint，M7 为 patchtst_fm.checkpoint）。因此与 mF_mean 的
fusion_station_script.py 同理：
  配置档 -> pipelines.model_registry.import_model_class(<model>)(config) -> model.to(device).eval()
新增零样本模型只要满足 M5/M7 的 adapter 契约（Model(model_config)、forward(batch_dict)
返回 logits [B,192]），无需改本脚本。

其余推理 + 指标 + 写文件逻辑（unified_inference / SolarMetricCalculator /
NanwangMetric / extract_summary / NumpyJSONEncoder / station_cap_from_df）逐行复用
eval_station_script.py，输出与 M1-M4 的 metrics_ens 完全同构，可直接进
make_models_summary 汇总。

产出：
    out_dir/{STATION}-metrics.xlsx   # 6 sheet + nanwang sheet
    out_dir/{STATION}-summary.json   # 键与单模型一致

用法（cwd=仓库根）：
    python eval_shell/eval_pretrain.py \
        --model m7 \
        --station_path dataset/test/station=1015.parquet \
        --out_dir runs/m7/pretrain_100m_v1/metrics_ens \
        --cap_power 465 --batch_size 512 \
        --tail_scale 0.1 --tail_len 4

模型与配置解析：
    --checkpoint_dir 缺省为 forecast_backbones/<model>；
    配置文件优先 <checkpoint_dir>/model_config.yaml，不存在则用 <checkpoint_dir>/config/configs.yaml；
    --model 缺省时取配置里的 run.model_name。

可选历史尾部缩放（--tail_scale/--tail_len）：
    仅对**推理输入**的 observe_power 历史序列做**内存级**缩放，不改动磁盘 parquet。
    --tail_scale <factor>：把每条历史序列末尾 --tail_len（默认 4）个点整体乘以 factor
    （如 0.1 = 末尾 4 点×0.1）。不传 --tail_scale 则行为与旧版完全一致（无缩放）。
    
huggingface-cli download ibm-granite/granite-timeseries-patchtst-fm-r2 --local-dir /data/tangqingang/granite-timeseries-patchtst-fm-r2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# eval_shell 无 __init__.py；cwd=仓库根时 eval_station_script 是顶层模块。
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

# 复用单模型评测的既有纯函数/工具，不复制重写。
from eval_station_script import (
    NumpyJSONEncoder,
    extract_summary,
    station_cap_from_df,
)

CONFIG_CANDIDATES = ("model_config.yaml", "config/configs.yaml")


def default_checkpoint_dir(model_name):
    return REPO_ROOT / "forecast_backbones" / model_name


def load_model_config(checkpoint_dir):
    """按 CONFIG_CANDIDATES 顺序找配置文件 -> (model_config, config_path)。"""
    checkpoint_dir = Path(checkpoint_dir)
    for name in CONFIG_CANDIDATES:
        config_path = checkpoint_dir / name
        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}, config_path
    raise FileNotFoundError(
        f"{checkpoint_dir} 下缺少模型配置文件（找过 {', '.join(CONFIG_CANDIDATES)}）"
    )


def load_pretrain_model(model_name, checkpoint_dir, device):
    """从配置直接构建零样本 adapter（无 safetensors 加载）-> (model, model_config)。

    与 eval_station_script.load_model_from_checkpoint 的区别：零样本模型没有可加载的
    state_dict；checkpoint_dir 只提供配置（M5 的 timesfm3 段 / M7 的 patchtst_fm 段），
    基础模型权重由 adapter 自己加载。model_name 为 None 时取配置里的 run.model_name。
    构建后 to(device).eval()。
    """
    from pipelines.model_registry import import_model_class

    model_config, config_path = load_model_config(checkpoint_dir)
    if model_name is None:
        model_name = (model_config.get("run") or {}).get("model_name")
        if not model_name:
            raise ValueError(f"未指定 --model，且 {config_path} 里没有 run.model_name")
    print(f"[model] {model_name}  config={config_path}")
    model = import_model_class(model_name)(model_config)
    model.to(device)
    model.eval()
    return model, model_config


def scale_observe_power_tail(station_df, tail_len=4, factor=None):
    """内存级缩放 observe_power 历史序列的末尾 tail_len 个点（不改磁盘数据）。

    station_df 是逐行 "一行一个样本" 的表，observe_power 每行是一条 672 点
    历史序列（list/np.ndarray）。factor 为 None 时原样返回（无缩放）。
    缩放的是最后 tail_len 个点：x[-tail_len:] *= factor。
    """
    if factor is None:
        return station_df
    factor = float(factor)
    if "observe_power" not in station_df.columns:
        raise KeyError("station_df 缺 observe_power 列，无法做尾部缩放")
    seqs = station_df["observe_power"].to_numpy()
    scaled = []
    for s in seqs:
        arr = np.asarray(s, dtype=np.float64)
        arr = arr.copy()
        if len(arr) > 0 and tail_len > 0:
            arr[-tail_len:] *= factor
        scaled.append(arr)
    station_df = station_df.copy()
    station_df["observe_power"] = scaled
    return station_df


def infer_station_pretrain(model_name, checkpoint_dir, station_path, cap_power,
                           batch_size=512, device="cuda", drop_label_cols=True,
                           tail_scale=None, tail_len=4):
    """单站零样本推理纯函数 -> (st, cap, result_df)。

    与 eval_station_script.py::infer_station 一一对应，仅把
    load_model_from_checkpoint 换成 load_pretrain_model。

    tail_scale/tail_len: 可选，对推理输入 observe_power 末尾 tail_len 个点
    做内存级缩放（见 scale_observe_power_tail），不落盘。
    """
    from pipelines.pipeline_wrapper_inference import unified_inference
    import pandas as pd

    station_path = Path(station_path)
    station_df = pd.read_parquet(station_path)
    station_df["timestamp_win"] = pd.to_datetime(station_df["timestamp_win"])

    # ---- 0. 可选：内存级缩放 observe_power 末尾 tail_len 个点 ----
    if tail_scale is not None:
        station_df = scale_observe_power_tail(station_df, tail_len=tail_len, factor=tail_scale)

    # ---- 1. 加载模型（从 config 构建，非 checkpoint）----
    model, model_config = load_pretrain_model(model_name, checkpoint_dir, device)

    bs = batch_size
    if isinstance(model_config, dict):
        cfg_batch_sizes = []
        hp = model_config.get("hyperparameter", {}) or {}
        for v in (hp.get("batch_size"), model_config.get("batch_size")):
            if isinstance(v, (int, float)) and v > 0:
                cfg_batch_sizes.append(v)
        if cfg_batch_sizes:
            bs = min(bs, min(cfg_batch_sizes))

    # 防泄漏：推理输入不带 label/未来真值列（feature_cols=None 会把它们收进
    # batch_dict）。删除的是投喂给模型的 DataFrame 副本,不影响后续 merge 真值。
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
    pred_col = type(model).__name__  # 如 TransformersAdaptedM5Model / TransformersAdaptedM7Model
    if pred_col not in ds_output.columns:
        raise KeyError(
            f"推理结果缺少预测列 {pred_col!r}, 实际列: {list(ds_output.columns)}"
        )

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
                tail_scale=None, tail_len=4):
    """对单个站点完成零样本推理 -> 指标 -> xlsx + summary.json。"""
    from pipelines.metric import SolarMetricCalculator
    from pipelines.nanwang_metric import NanwangMetric
    import pandas as pd

    station_path = Path(station_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    st, cap, result_df = infer_station_pretrain(
        model_name, checkpoint_dir, station_path, cap_power,
        batch_size=batch_size, device=device,
        tail_scale=tail_scale, tail_len=tail_len,
    )

    # ---- 3+4. solar 指标 + 南网月度表，合并写 {站}-metrics.xlsx ----
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
    parser = argparse.ArgumentParser(
        description="零样本基础模型（M5/M7…）逐站 192 步回测：config -> 推理 -> 指标"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="registry 模型名，如 m5 / m7；缺省取配置里的 run.model_name",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="配置目录（含 model_config.yaml 或 config/configs.yaml），默认 forecast_backbones/<model>",
    )
    parser.add_argument("--station_path", type=str, required=True,
                        help="dataset/test/station=<站>.parquet")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="runs/<model>/pretrain_100m_v1/metrics_ens")
    parser.add_argument("--cap_power", type=float, default=465.0, help="回退容量")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tail_scale", type=float, default=None,
                        help="对推理输入 observe_power 末尾 tail_len 个点做内存级缩放"
                             "（不落盘），如 0.1=末尾4点×0.1；None=不缩放（默认）")
    parser.add_argument("--tail_len", type=int, default=4,
                        help="尾部缩放的点数（默认 4）")
    args = parser.parse_args()

    if args.checkpoint_dir is None and args.model is None:
        parser.error("--model 与 --checkpoint_dir 至少给一个")
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else default_checkpoint_dir(args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_station(
        args.model, checkpoint_dir, Path(args.station_path), Path(args.out_dir),
        args.cap_power, args.batch_size, device,
        tail_scale=args.tail_scale, tail_len=args.tail_len,
    )


if __name__ == "__main__":
    main()
