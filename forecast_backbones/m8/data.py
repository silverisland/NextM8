# forecast_backbones/m8/data.py（2026-10-13 起自带一份，不再引用 m6）
"""训练 parquet（文件夹里的若干文件，或切分目录里的 train/valid.parquet）→ 按 M8 口径建 UnifiedDataset（arrow 后端）。
拼表与站号归一自带（_parquet.py），不依赖 pipelines.time_splits。"""
from __future__ import annotations


import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .solar import attach_station_id, norm_station_name
    from ._parquet import station_keys, concat_parquets, _arrow_kind
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m8.solar import attach_station_id, norm_station_name
    from forecast_backbones.m8._parquet import station_keys, concat_parquets, _arrow_kind
try:
    from ...pipelines.transformer_dataset import UnifiedDataset
except ImportError:
    from pipelines.transformer_dataset import UnifiedDataset


from pipelines.forecasting.data import dataset_with_station_id as _dataset_with_station_id
from pipelines.forecasting.data import check_presplit_dir as _check_presplit_dir


def report_inputs(model, columns, verbose: bool = True) -> list[list[str]]:
    """（2026-10-20 从 train.py 挪来：南网 m8_train_cli.py 不 import train.py，也就不依赖 pipelines/nanwang_metric.py）
    训练开始前核对"配置要的列 ↔ 训练数据里的列"并打进日志；插值 / future_only 列缺了直接报错。
    返回训练数据里实际有的 past_future 对（写进 model_config.yaml 的 features.cov_present，推理时据此判断 NWP 缺列兜底）。"""
    lines, res = model.fe.input_report(columns)
    if verbose:
        print("[m8] ======== 输入列核对（配置列 → 训练数据列；✗ = 数据里没有）========")
        for ln in lines:
            print(f"[m8]   {ln}")
    missing_fo = [c for c, _ in model.fe.future_only if res.get(c) is None]
    if missing_fo:
        raise ValueError(f"训练数据里没有插值 / future_only 列 {missing_fo}：检查 features.hint_col（或 train_m8.sh --hint_col）；"
                         f"数据列 {sorted(map(str, columns))}")
    present = [[h, f] for h, f in model.fe.pairs if res.get(h) and res.get(f)]
    if verbose:
        absent = [[h, f] for h, f in model.fe.pairs if [h, f] not in present]
        print(f"[m8]   → 实际使用的 NWP 协变量对 {len(present)}/{len(model.fe.pairs)}：{present}"
              + (f"；缺列、通道恒 0：{absent}" if absent else ""))
        if not present and model.fe.pairs:
            print("[m8]   ⚠️ 一对 NWP 协变量都没对上：模型等于不看天气预报，请检查 features.past_future 的列名", flush=True)
    return present


def unified_feature_cols(fe, columns) -> list[str]:
    """南网训练入口（train_shell/m8_train_cli.py）建 UnifiedDataset 用的 feature_cols：配置要的列（历史功率、past_future 对、
    past_only、插值 / future_only）按 fe.resolve_col（精确 → col_aliases → 大小写不敏感）对到数据里的列名，数据里没有的跳过
    （NWP 对缺列 = 通道恒 0；插值列缺列由 report_inputs 报错）。历史功率列缺了直接报错。"""
    columns = [str(c) for c in columns]
    if fe.resolve_col(columns, fe.pv_col) is None:
        raise KeyError(f"数据里没有历史功率列 {fe.pv_col!r}（dataset.pv_col）；数据列 {sorted(columns)}")
    want = [fe.pv_col] + [c for pair in fe.pairs for c in pair] + list(fe.past_only) + [c for c, _ in fe.future_only]
    out: list[str] = []
    for c in want:
        k = fe.resolve_col(columns, c)
        if k is not None and k not in out:
            out.append(k)
    return out


def check_hint_units(fe, parquet_file, max_rows: int = 2000) -> str | None:
    """插值列（TimesFM）与历史功率是否同单位的粗检（2026-10-20）：模型把两者都除以站容量（features.build 里 scale=target）。
    插值列若是容量归一值（0–1）而功率是 MW，训练时提示通道 ≈ 0，线上 tsfm_predict（M5 对 observe_power 的预测，MW）一来就大上百倍；
    反之亦然。取前 max_rows 行两列的 99 分位比较，相差 10 倍以上返回提示文本，否则 None（不用插值列 / 缺列 / 全 0 也返回 None）。"""
    hint = fe.cfg["features"].get("hint_col")
    if not hint:
        return None
    import pyarrow.parquet as pq
    cols = pq.read_schema(parquet_file).names
    hk, pk = fe.resolve_col(cols, str(hint)), fe.resolve_col(cols, fe.pv_col)
    if hk is None or pk is None:
        return None
    df = pd.read_parquet(parquet_file, columns=[hk, pk]).head(max_rows)

    def q99(s):
        parts = [np.asarray(x, float).ravel() for x in s if x is not None]
        v = np.concatenate(parts) if parts else np.array([])
        v = v[np.isfinite(v)]
        return float(np.quantile(v, 0.99)) if v.size else float("nan")

    h, p = q99(df[hk]), q99(df[pk])
    if not (h > 0 and p > 0) or 0.1 <= h / p <= 10.0:
        return None
    return (f"插值列 {hk!r} 的 99 分位 {h:.4g} 与历史功率 {pk!r} 的 {p:.4g} 相差 {max(h / p, p / h):.0f} 倍：两者须同单位（MW）。"
            f"线上 tsfm_predict 是 M5 对 observe_power 的预测（MW），训练列若是容量归一值，模型学到的提示通道到线上会大上百倍")


def station_ids_from_names(station_meta, names) -> list[int]:
    """数据 station 列（站名）→ 训练集里出现过的整数站号（升序）；站表里找不到或有冲突直接报错（与 FeatureBuilder.build 同一映射）。"""
    names = [n for n in dict.fromkeys(names) if norm_station_name(n) is not None]
    mapping, unknown, conflicts = station_meta.resolve(names)
    if unknown or conflicts:
        raise ValueError(f"站名映射失败：station_meta 里找不到 {unknown[:10]} {'；'.join(conflicts)}"
                         f"（station_meta 的 station 列要写数据里的站名，或 --station-meta 换站表）")
    return sorted({int(mapping[norm_station_name(n)]) for n in names})
