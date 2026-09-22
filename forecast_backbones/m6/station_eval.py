# -*- coding: utf-8 -*-
# forecast_backbones/m6/station_eval.py
"""训练产物目录 → 推理模型；单站测试 parquet → 预测结果表。供 eval_shell/m6_station_script.py 逐站评测用。
产物目录即 train_m6 的 out_dir：model_config.yaml、model.safetensors（旧 run 为 lora_weights.pt），
calibration.enabled 时还有 calibration.parquet 与同名 .json。全部站都用这份 LoRA + output_head，模型全部冻结。
推理与训练期验证走同一条路：_dataset_with_station_id 建数据集（文本站号按 station_meta 转整数）→ evaluate_model。"""
from __future__ import annotations

import functools
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import load_config, deep_merge
    from .transformer_trainer import _raw_override, evaluate_model
    from .train import WEIGHTS_FILE, CONFIG_FILE, _dataset_with_station_id
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from config import load_config, deep_merge
    from transformer_trainer import _raw_override, evaluate_model
    from train import WEIGHTS_FILE, CONFIG_FILE, _dataset_with_station_id
try:
    from ...pipelines.transformer_dataset import UnifiedDataset
    from ...pipelines.time_splits import station_keys
except ImportError:
    from pipelines.transformer_dataset import UnifiedDataset
    from pipelines.time_splits import station_keys

LEGACY_WEIGHTS_FILE = "lora_weights.pt"
RESULT_COLS = ["timestamp_win", "station", "observe_power_predict", "observe_power_future"]


def load_trained_model(run_dir, config=None, device=None, batch_size=None, progress_bar=False):
    """run_dir：train_m6 的 out_dir。config：不用 run_dir/model_config.yaml 时给训练用的 yaml。
    权重、校准表都从 run_dir 读，缺任何一个直接报错（不静默退回零样本或未校准）。
    返回的模型全部参数 requires_grad=False、全部子模块 eval 态（dropout 关闭）。返回 (model, cfg)。"""
    try:
        from .transformer_models import TransformersAdaptedM6Model
    except ImportError:
        from transformer_models import TransformersAdaptedM6Model
    run_dir = Path(run_dir)
    src = Path(config) if config else run_dir / CONFIG_FILE
    if not src.is_file():
        raise FileNotFoundError(f"缺模型配置 {src}：旧 run 没有 model_config.yaml 时用 --config 指定训练用的 yaml")
    weights = run_dir / WEIGHTS_FILE
    if not weights.exists() and (run_dir / LEGACY_WEIGHTS_FILE).exists():
        weights = run_dir / LEGACY_WEIGHTS_FILE
    if not weights.exists():
        raise FileNotFoundError(f"{run_dir} 缺训练产物 {weights.name}（旧 run 为 {LEGACY_WEIGHTS_FILE}）")
    over = {"run": {"out_dir": str(run_dir)},
            "finetune": {"enabled": True, "lora_path": str(weights), "schedule": {"progress_bar": bool(progress_bar)}}}
    if device:
        over["timesfm3"] = {"device": str(device)}
    if batch_size:
        over["eval"] = {"batch_size": int(batch_size)}
    cfg = load_config(deep_merge(_raw_override(src), over))
    table = Path(cfg["calibration"]["table_path"])
    if cfg["calibration"]["enabled"] and not table.exists():
        raise FileNotFoundError(f"calibration.enabled 但校准表 {table} 不存在")
    model = TransformersAdaptedM6Model(cfg)
    if model.backbone is None:
        model._ensure_backbone(model._resolve_device())
    if not model.lora_loaded:
        raise RuntimeError(f"LoRA 权重 {weights} 没有加载上")
    model.requires_grad_(False)
    model.eval()
    return model, cfg


def scale_history_tail(df: pd.DataFrame, col: str, tail_len: int = 4, factor: float | None = None) -> pd.DataFrame:
    """把 col 每行历史序列的最后 tail_len 个点乘以 factor（返回副本）；factor 为 None 原样返回。"""
    if factor is None:
        return df
    if col not in df.columns:
        raise KeyError(f"数据缺 {col!r} 列，无法做尾部缩放")
    out = []
    for s in df[col].to_numpy():
        a = np.asarray(s, dtype=np.float64).copy()
        if len(a) and tail_len > 0:
            a[-tail_len:] *= float(factor)
        out.append(a)
    return df.assign(**{col: out})


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


def predict_station(model, station_path, tail_scale=None, tail_len: int = 4, scratch_dir=None) -> tuple[str, pd.DataFrame]:
    """单站推理 → (站号, evaluate_model 的结果表)。结果表的 station 列改成站号（与文件名一致）。
    站号转换与尾部缩放要写临时 parquet，放在 scratch_dir 下的临时目录里，推理完即删。"""
    cfg = model.cfg
    station_path = Path(station_path)
    st = station_key(station_path, str(cfg["dataset"]["station_id_col"]))
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    if scratch_dir is not None:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".m6_eval_", dir=scratch_dir) as tmp:
        src = station_path
        if tail_scale is not None:
            src = Path(tmp) / station_path.name
            scale_history_tail(pd.read_parquet(station_path), cfg["dataset"]["pv_col"], tail_len, tail_scale).to_parquet(src, index=False)
        ds = _dataset_with_station_id([str(src)], cfg, model.station_meta, tmp)
        model._progress_desc = f"[m6] 站 {st} 推理"
        try:
            res = evaluate_model(model, ds, collate, cfg)
        finally:
            model._progress_desc = None
    if res.empty:
        raise ValueError(f"{station_path} 没有可推理的行")
    if res["station_id"].nunique() != 1:
        raise ValueError(f"站 {st} 的数据映射成了多个 station_id {sorted(res['station_id'].unique())[:5]}")
    return st, res.assign(station=st)
