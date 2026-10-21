"""Rum defaults using public YAML utilities and independent architecture settings."""

from __future__ import annotations

import math

import numpy as np

from pipelines.forecasting.config import deep_merge, interpolate, raw_override

DEFAULT_CFG = {
    "run": {
        "model_name": "m4_rum",
        "tag": "rum",
        "seeds": [0],
        "out_dir": "runs/m4_rum/rum",
    },
    "dataset": {
        "pv_col": "observe_power",
        "label_col": "observe_power_future",
        "ts_col": "timestamp_win",
        "station_id_col": "station_id",
        "station_meta": "pipelines/forecasting/assets/station_meta.csv",
        "station_meta_cols": None,
        "known_stations": None,
        "context_len": 672,
        "pred_len": 192,
        "step_min": 15,
    },
    "features": {
        "past_future": [
            ["general_ghi", "general_ghi_predict"],
            ["general_temp", "general_temp_predict"],
        ],
        "weather_offsets": [0.0, 15.0],
        "weather_scales": [1000.0, 15.0],
        "hint_col": "timesfm3",
        "col_aliases": {"timesfm3": ["tsfm_predict"]},
        "cov_present": None,
    },
    "model": {
        "patch_len": 16,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 256,
        "context_dim": 64,
        "weather_dim": 64,
        "head_hidden": 128,
        "head_dim": 64,
        "dropout": 0.1,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "train": {
        "history_dropout": 0.2,
        "hint_dropout": 0.3,
        "loss": "mse",
        "ema_decay": None,
        "weights_path": "${run.out_dir}/model.safetensors",
        "schedule": {
            "lr": 3e-4,
            "weight_decay": 0.05,
            "batch_size": 64,
            "max_steps": 6000,
            "epochs_max": 20,
            "warmup_steps": 300,
            "grad_clip": 1.0,
            "precision": "fp32",
            "logging_steps": 50,
            "eval_every_steps": 250,
            "save_every": 500,
            "sampling": "random",
            "progress_bar": True,
            "num_workers": 0,
        },
        "early_stop": {
            "metric": "nanwang_short_D+1",
            "patience": 8,
            "nrmse_issue_every_min": 60,
            "smooth_k": 1,
        },
    },
    "decision": {
        "point": {
            "default": "median",
            "floor_ratio": 0.2,
            "tau_grid_step": 0.005,
            "tail": "flat",
        },
    },
    "eval": {"batch_size": 128},
}


def _check_keys(value, template, prefix=""):
    for key, item in value.items():
        path = f"{prefix}{key}"
        if key not in template:
            raise ValueError(f"Unknown M4-Rum configuration: {path}")
        # These dictionaries contain user-defined column names.
        if path == "features.col_aliases" or template[key] is None:
            continue
        if isinstance(template[key], dict):
            if not isinstance(item, dict):
                raise ValueError(f"{path} must be a mapping")
            _check_keys(item, template[key], path + ".")


def load_config(config=None):
    override = raw_override(config)
    _check_keys(override, DEFAULT_CFG)
    cfg = interpolate(deep_merge(DEFAULT_CFG, override))
    ds, model, fe, train = (cfg[k] for k in ("dataset", "model", "features", "train"))
    if ds["pred_len"] != 192 or ds["step_min"] != 15:
        raise ValueError("M4-Rum uses 192 steps, starting 15 minutes after issue time")
    for key in (
        "patch_len",
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "context_dim",
        "weather_dim",
        "head_hidden",
        "head_dim",
    ):
        if not isinstance(model[key], int) or model[key] <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
    if ds["context_len"] <= 0 or ds["context_len"] % model["patch_len"]:
        raise ValueError("context_len must be positive and divisible by patch_len")
    if model["d_model"] % model["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    pairs = fe["past_future"]
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("past_future must contain history/future column pairs")
    for key in ("weather_offsets", "weather_scales"):
        if len(fe[key]) != len(pairs) or not all(math.isfinite(v) for v in fe[key]):
            raise ValueError(f"{key} must have one finite value per weather pair")
    if any(v <= 0 for v in fe["weather_scales"]):
        raise ValueError("weather_scales must be positive")
    names = [c for pair in pairs for c in pair] + [
        fe["hint_col"],
        ds["pv_col"],
        ds["label_col"],
    ]
    if any(not isinstance(c, str) or not c for c in names) or len(set(names)) != len(
        names
    ):
        raise ValueError(
            "Weather, hint, historical power and label columns must be distinct"
        )
    label = ds["label_col"].lower()
    for name in names[:-1]:
        candidates = [name] + fe["col_aliases"].get(name, [])
        if any(c.lower() == label for c in candidates):
            raise ValueError(
                "Input columns and aliases must not refer to future labels"
            )
    for key in ("history_dropout", "hint_dropout"):
        if not 0 <= train[key] <= 1:
            raise ValueError(f"train.{key} must be in [0, 1]")
    if not 0 <= model["dropout"] < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if train["loss"] not in ("mse", "mae", "pinball"):
        raise ValueError("train.loss must be mse, mae or pinball")
    levels = model["quantiles"]
    if (
        not isinstance(levels, (list, tuple))
        or len(levels) < 2
        or any(
            isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 < q < 1
            for q in levels
        )
        or any(a >= b for a, b in zip(levels, levels[1:]))
    ):
        raise ValueError("model.quantiles must be strictly increasing in (0,1), K >= 2")
    levels_fp32 = np.asarray(levels, dtype=np.float32)
    if (
        np.any(levels_fp32 <= 0)
        or np.any(levels_fp32 >= 1)
        or np.any(np.diff(levels_fp32) <= 0)
    ):
        raise ValueError("model.quantiles must remain distinct and in (0,1) in float32")
    point = cfg["decision"]["point"]
    if point["default"] not in ("median", "p_star") or point["tail"] != "flat":
        raise ValueError("decision.point supports median/p_star with flat tails")
    for key in ("floor_ratio", "tau_grid_step"):
        value = point[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"decision.point.{key} must be finite and positive")
    if not 1e-4 <= point["tau_grid_step"] <= 1:
        raise ValueError("decision.point.tau_grid_step must be in [0.0001, 1]")
    if point["default"] == "p_star" and train["loss"] != "pinball":
        raise ValueError("p_star requires train.loss=pinball and a quantile head")
    if train["early_stop"]["metric"] not in ("nanwang_short_D+1", "nrmse"):
        raise ValueError("early_stop.metric must be nanwang_short_D+1 or nrmse")
    if train["early_stop"]["patience"] < 1:
        raise ValueError("early_stop.patience must be positive")
    if train["early_stop"]["nrmse_issue_every_min"] <= 0:
        raise ValueError("nrmse_issue_every_min must be positive")
    for key in ("batch_size", "epochs_max", "lr", "grad_clip"):
        if train["schedule"][key] <= 0:
            raise ValueError(f"schedule.{key} must be positive")
    if cfg["eval"]["batch_size"] <= 0:
        raise ValueError("eval.batch_size must be positive")
    every = train["schedule"]["eval_every_steps"]
    if every is not None and every <= 0:
        raise ValueError("eval_every_steps must be positive or null (epoch end)")
    if train["schedule"]["sampling"] not in ("random", "station_day"):
        raise ValueError("schedule.sampling must be random or station_day")
    if train["schedule"]["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("schedule.precision must be fp32, fp16 or bf16")
    if (train["schedule"]["save_every"] or 0) < 0:
        raise ValueError("schedule.save_every must be nonnegative")
    return cfg
