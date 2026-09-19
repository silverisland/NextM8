# forecast_backbones/m6/config.py
"""M6 配置：yaml → dict，与 DEFAULT_CFG 深合并，解析 ${run.out_dir}。
段式与 m5 兼容：run / model / dataset / timesfm3 四段逐字同名；其余为 M6 新增。"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CFG: dict = {
    "run": {"model_name": "m6", "seeds": [0, 1, 2], "out_dir": "runs/m6_baseline"},
    "model": {},
    "dataset": {
        "pv_col": "observe_power",
        "label_col": "observe_power_future",
        "ts_col": "timestamp_win",
        "station_id_col": "station_id",
        "cap_col": "cap_power_on",
        "station_meta": "forecast_backbones/m6/station_meta.csv",
        "info_csv": "info.csv",                    # 路线 A 不读
        "tz": "Asia/Shanghai",
        "step_min": 15,
    },
    "timesfm3": {
        "checkpoint": "/data/tangqingang/timesfm-3.0-pytorch",
        "context_len": 672,
        "pred_len": 192,
        "device": "auto",
        "per_core_batch_size": 64,
        "return_quantiles": True,
        "nan_policy": "interp",
        "make_positive": True,
        "use_znorm": False,
        "use_symmetric_averaging": True,
        "eager_init": True,
    },
    "features": {
        "past_future": [
            ["general_ghi", "general_ghi_predict"],
            ["general_temp", "general_temp_predict"],
            ["GHI_solargis", "ghi_solargis_predict"],
            ["temp_solargis", "temp_solargis_predict"],
        ],
        "derived": {"solar_elevation": True, "clear_sky_power": True},
        "nwp_members": {
            "enabled": False,
            "cols": [f"SSRD_pos_{i}" for i in range(1, 10)],
            "future_suffix": "_predict",
            "add_spread": True,
        },
        "past_only": [],
        "target_transform": "none",                # 路线 A 不读
    },
    "variability": {"signal": "k_nwp", "bins": 3, "k_nwp_edges": [0.35, 0.75],
                    "nwp_spread_edges": None, "src_disagree_edges": None, "recent_kt_std_edges": None,
                    "ghi_col": "general_ghi_predict"},
    "data": {
        "issue_times_eval": "all",                 # 路线 A 不读
        "issue_times_train": "all",                # 路线 A 不读
        "scored_steps": {"short": [55, 151], "ultra": [0, 16]},
    },
    "finetune": {                                   # 路线 B（计划 2）：LoRA 微调；路线 A 只读 enabled/inference_path
        "enabled": False,
        "inference_path": "evaluator",              # evaluator | torch；enabled=True 时训练强制走 torch 路径
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.05,
                 # timesfm3 3.0.2 真名（transformer.py:188-193）；seq_attn 与 var_attn 两处都匹配
                 "target_modules": ["query_proj", "key_proj", "value_proj", "out_proj"]},
        "unfreeze_output_head": True,
        "lora_path": "${run.out_dir}/lora_weights.pt",          # 训练产物；推理时存在即加载
        "routing_path": "${run.out_dir}/station_routing.json",  # 逐站 lora|zeroshot 路由表；存在即按表选路
        "loss": {"type": "pinball", "levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                 "aux_nanwang_head": {"enabled": False, "weight": 0.5}},   # 辅助头：计划 3 消融，本计划不读
        "masks": {"night": "solar_elevation_le_0", "curtailment_col": None,
                  "curtailment_autodetect": {"enabled": False, "plateau_ratio": 0.05,
                                             "k_nwp_min": 0.7, "min_run": 4},
                  "missing": True},
        "schedule": {"lr": 1.0e-4, "weight_decay": 0.01, "epochs_max": 10, "batch_size": 16,
                     "grad_clip": 1.0, "precision": "bf16", "logging_steps": 20},
        "split": {"train_start": None, "train_end": "2025-05-31", "cal_days": 45,
                  "val_start": "2025-07-16", "val_end": "2025-08-31", "test_start": "2025-09-01",
                  "rolling_folds": 2},               # rolling_folds 只作说明；折间平移用 train_m6(shift_days=)
        "early_stop": {"metric": "nanwang_after_pstar", "patience": 3},
        "guards": {"per_station_fallback_to_zeroshot": True, "min_gain_pct": 0.0},
    },
    "calibration": {
        "enabled": True,
        "table_path": "${run.out_dir}/calibration.parquet",
        "levels": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99],
        "normalization": "clear_sky",
        "strata": {"solar_elevation_edges": [20, 45], "variability": True,
                   "lead_edges": [4, 8, 16, 55, 103]},
        "fit_steps": [[0, 16], [55, 151]],
        "window_days": 45,
        "n_min_factor": 10,
        "fallback": "hierarchical",                 # 路线 A 不读
        "aci": {"enabled": False, "gamma": 0.01},   # 路线 A 不读
    },
    "decision": {
        "physics": {"night_zero": True, "clear_sky_cap_ratio": 1.05, "capacity_cap": True},
        "point": {
            "default": "p_star",
            "by_step": [{"steps": [0, 16], "method": "p_star_l1"},
                        {"steps": [55, 151], "method": "p_star"}],
            "floor_ratio": 0.2,
            "tau_grid_step": 0.005,
            "tail": "flat",
            "shrink_factor": None,
        },
    },
    "stacking": {"enabled": False, "members": ["m1", "m2", "m3", "m4", "m6"],           # 整段路线 A 不读
                 "method": "nnls_bins", "bins": ["variability", "lead"], "window_days": 45,  # ——计划 3
                 "gate_min_gain_over_m6": "seed_std"},
    "eval": {
        "metrics": ["nanwang_short_D+1", "nanwang_ultra_short", "nrmse", "crps", "coverage", "pit"],  # 路线 A 不读
        "buckets": ["variability", "solar_elevation", "station", "lead"],              # 路线 A 不读
        "report_dir": "${run.out_dir}/reports",
        "batch_size": 64,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """dict 递归合并；非 dict（含 list）整体替换。与 m5 的 _merge 语义一致。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def interpolate(cfg: dict) -> dict:
    """把字符串里的 ${run.out_dir} 替换成 cfg['run']['out_dir']。只支持这一个占位符。"""
    out_dir = str(cfg.get("run", {}).get("out_dir", ""))

    def _walk(x):
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, str) and "${run.out_dir}" in x:
            return x.replace("${run.out_dir}", out_dir)
        return x

    return _walk(cfg)


def load_config(src: str | Path | dict | None) -> dict:
    """src: yaml 路径 / 已加载 dict / None（纯默认）。返回合并+插值后的完整配置。"""
    if src is None:
        override = {}
    elif isinstance(src, dict):
        override = src
    else:
        override = yaml.safe_load(Path(src).read_text(encoding="utf-8")) or {}
    return interpolate(deep_merge(DEFAULT_CFG, override))
