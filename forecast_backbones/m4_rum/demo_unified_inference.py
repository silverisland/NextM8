"""直接使用 unified_inference 的生产接入示例。

在 NextM8 根目录运行：
  /path/to/python -m forecast_backbones.m4_rum.demo_unified_inference \
      --run /path/to/run --input /path/to/input.parquet \
      --out /path/to/predictions.parquet --device cuda:0

服务启动时调用 load_member 一次；请求中复用模型并调用 predict_dataframe。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipelines.forecasting.data import expand_parquets
from pipelines.pipeline_wrapper_inference import unified_inference

from .evaluate import load_member


def predict_dataframe(model, input_df: pd.DataFrame, *, batch_size=32, progress=False):
    """已加载的 M4-Rum + 窗口表 -> 实体列及 observe_power_predict。

    每行输出一条完整预测曲线，保持输入行序；不要求未来真实功率。
    模型 forward 已完成容量反归一化、分位数选点及配置指定的裁剪。
    """
    if not model.weights_loaded:
        raise ValueError("请先用 load_member 加载训练产物")
    if not isinstance(input_df, pd.DataFrame) or input_df.empty:
        raise ValueError("input_df 必须是非空 pandas.DataFrame")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    ds = model.cfg["dataset"]
    # 在进入 Dataset 前去掉标签，生产预测不依赖标签的存在、长度或数值。
    frame = input_df.drop(columns=[ds["label_col"]], errors="ignore")
    # 复用训练侧规则：按 station/ID 匹配；无 CSV 时读取配置容量列并取列表首值。
    # 生成的整数 ID 作为 entity 保留，避免作为浮点特征时发生精度损失。
    frame = model.station_meta.prepare_frame(frame)
    entities = list(
        dict.fromkeys(
            col
            for col in (
                ds["ts_col"],
                ds["station_id_col"],
                "station",
                ds["station_key"],
            )
            if col is not None and col in frame.columns
        )
    )

    # 这里直接调用项目统一推理框架，不调用 eval_shell 或独立评估循环。
    result = unified_inference(
        pipeline_backbone=model,
        ds_input=frame,
        timestamp_cols=[ds["ts_col"]],
        entity_cols=entities,
        batch_size=batch_size,
        num_workers=0,
        timestamp_output="epoch_s",
        prediction_to_numpy=True,
        exclude_input_cols=[ds["label_col"]],
        error_policy="raise",
        progress=progress,
    )
    # unified_inference 默认以模型类名命名预测列，转为项目业务列名。
    return result.rename(columns={type(model).__name__: "observe_power_predict"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="包含 model_config.yaml 和 model.safetensors 的目录",
    )
    parser.add_argument(
        "--input", nargs="+", required=True, help="Parquet 文件、目录或通配符"
    )
    parser.add_argument("--out", required=True, help="新建输出 Parquet，不覆盖已有文件")
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda:0 等；默认 cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--station-meta", help="可选：迁移后的标准列名站点 CSV 路径")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    paths = expand_parquets(args.input)
    if not paths:
        raise ValueError("未找到输入 Parquet 文件")

    # 生产服务中放在初始化阶段，后续请求复用此模型。
    model = load_member(args.run, device=args.device, station_meta=args.station_meta)
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    result = predict_dataframe(model, frame, batch_size=args.batch_size, progress=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"Saved {len(result)} predictions to {out}")


if __name__ == "__main__":
    main()
