"""M4 生产推理 demo：加载一次模型，直接调用项目 unified_inference。

在仓库根目录使用指定解释器运行：
python -m forecast_backbones.m4.demo_unified_inference --run /path/to/run
--input /path/to/input.parquet --out /path/to/predictions.parquet --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipelines.forecasting.data import expand_parquets
from pipelines.pipeline_wrapper_inference import unified_inference

from .evaluate import load_member


def predict_dataframe(model, input_df: pd.DataFrame, *, batch_size=32, progress=False):
    """已加载 M4 + 无标签窗口表 -> 实体列及 observe_power_predict。

    默认每行输出 192 点，保持输入行序；模型已按站表容量反归一化。
    沿用 checkpoint 的 hint 列和别名，不在推理时更换 hint 来源。
    """
    if not model.weights_loaded:
        raise ValueError("请先用 load_member 加载训练产物")
    if not isinstance(input_df, pd.DataFrame) or input_df.empty:
        raise ValueError("input_df 必须是非空 pandas.DataFrame")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    ds, fe = model.cfg["dataset"], model.cfg["features"]
    if ds["ts_col"] not in input_df:
        raise ValueError(f"缺少起报时间列 {ds['ts_col']!r}")
    if ds["station_id_col"] not in input_df and "station" not in input_df:
        raise ValueError("需要 station 或配置指定的站号列")
    # 去除标签；业务表中的其他附加列也不必进入数值特征转换。
    frame = input_df.drop(columns=[ds["label_col"]], errors="ignore")
    entities = list(
        dict.fromkeys(
            col
            for col in (ds["ts_col"], ds["station_id_col"], "station")
            if col in frame
        )
    )
    inputs = [ds["pv_col"], fe["hint_col"]] + [
        c for pair in fe["past_future"] for c in pair
    ]
    features = list(
        dict.fromkeys(
            resolved
            for name in inputs
            if (resolved := model.fe.resolve_col(frame.columns, name)) is not None
        )
    )
    result = unified_inference(
        pipeline_backbone=model,
        ds_input=frame,
        timestamp_cols=[ds["ts_col"]],
        entity_cols=entities,
        feature_cols=features,
        batch_size=batch_size,
        num_workers=0,
        timestamp_output="epoch_s",
        prediction_to_numpy=True,
        exclude_input_cols=[ds["label_col"]],
        error_policy="raise",
        progress=progress,
    )
    # 不增加裁剪或 p_star 等额外处理，保持与 M4 评估预测一致。
    return result.rename(columns={type(model).__name__: "observe_power_predict"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="包含 model_config.yaml、model.safetensors 的训练目录",
    )
    parser.add_argument(
        "--input", nargs="+", required=True, help="Parquet 文件、目录或通配符"
    )
    parser.add_argument("--out", required=True, help="新建输出 Parquet，不覆盖已有文件")
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda:0 等；默认 cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--station-meta", help="可选：迁移后的 CSV 路径，保留 checkpoint 的列名映射"
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    paths = expand_parquets(args.input)
    if not paths:
        raise ValueError("未找到输入 Parquet 文件")
    # 服务中将加载放在初始化阶段，随后反复调用 predict_dataframe 即可。
    model = load_member(args.run, device=args.device, station_meta=args.station_meta)
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    result = predict_dataframe(model, frame, batch_size=args.batch_size, progress=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"Saved {len(result)} predictions to {out}")


if __name__ == "__main__":
    main()
