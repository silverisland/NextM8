"""M4-only defaults and validation; no sibling model dependencies."""

from __future__ import annotations

import math

from pipelines.forecasting.config import deep_merge, interpolate, raw_override

DEFAULT_CFG = {
    "run": {
        "model_name": "m4",
        "tag": "patchtst",
        "seeds": [2025],
        "out_dir": "runs/m4/patchtst",
    },
    "dataset": {
        "pv_col": "observe_power",
        "label_col": "observe_power_future",
        "label_norm": ["cap_power_on"],
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
        "hint_col": "chronos",
        "col_aliases": {"timesfm3": ["tsfm_predict"]},
        "cov_present": None,
    },
    "model": {
        "patch_len": 96,
        "stride": 96,
        "padding_patch": None,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
        "d_ff": 1024,
        "dropout": 0.1,
        "fc_dropout": 0.2,
        "head_dropout": 0.1,
        "embedding_dim": 256,
        "activation": "relu",
        "norm": "BatchNorm",
        "revin": True,
        "affine": False,
        "subtract_last": False,
    },
    "train": {
        "loss": "mae",
        "ema_decay": None,
        "weights_path": None,
        "schedule": {
            "lr": 1e-4,
            "weight_decay": 0.05,
            "batch_size": 64,
            "max_steps": 6000,
            "epochs_max": 100,
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
            "patience": 10,
            "nrmse_issue_every_min": 60,
            "smooth_k": 1,
        },
    },
    "eval": {"batch_size": 128},
}


def _check_keys(value, template, prefix=""):
    if not isinstance(value, dict):
        raise TypeError(f"{prefix or 'config'} must be a mapping")
    for key, item in value.items():
        path = prefix + key
        if key not in template:
            raise ValueError(f"Unknown M4 configuration: {path}")
        if path == "features.col_aliases" or template[key] is None:
            continue
        if isinstance(template[key], dict):
            _check_keys(item, template[key], path + ".")


def load_config(config=None):
    override = raw_override(config)
    if "inference" in override or "optimization" in override:
        raise ValueError(
            "Legacy M4 config: use config/configs.yaml and retrain; old checkpoints are not the standalone workflow format"
        )
    _check_keys(override, DEFAULT_CFG)
    cfg = interpolate(deep_merge(DEFAULT_CFG, override))
    ds, model, fe, train = (cfg[k] for k in ("dataset", "model", "features", "train"))
    if cfg["run"]["model_name"] != "m4":
        raise ValueError("M4 requires run.model_name=m4; other models cannot be loaded")
    if ds["pred_len"] != 192 or ds["step_min"] != 15:
        raise ValueError("M4 uses 192 future steps at 15-minute intervals")
    for name, value in [("context_len", ds["context_len"])] + [
        (k, model[k])
        for k in (
            "patch_len",
            "stride",
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "embedding_dim",
        )
    ]:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    total = ds["context_len"] + ds["pred_len"]
    if model["patch_len"] > total or model["stride"] > model["patch_len"]:
        raise ValueError("Patches must fit the input and stride must not leave gaps")
    if model["padding_patch"] not in (None, "end"):
        raise ValueError("padding_patch must be null or end")
    if (
        model["padding_patch"] is None
        and (total - model["patch_len"]) % model["stride"]
    ):
        raise ValueError(
            "Patches must cover the final input position; use padding_patch=end"
        )
    if model["d_model"] % model["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    for key in ("dropout", "fc_dropout", "head_dropout"):
        if not 0 <= model[key] < 1:
            raise ValueError(f"model.{key} must be in [0, 1)")
    for key in ("revin", "affine", "subtract_last"):
        if type(model[key]) is not bool:
            raise ValueError(f"model.{key} must be boolean")
    if model["activation"] not in ("relu", "gelu") or model["norm"] not in (
        "BatchNorm",
        "LayerNorm",
    ):
        raise ValueError(
            "activation must be relu/gelu and norm must be BatchNorm/LayerNorm"
        )
    pairs = fe["past_future"]
    if not pairs or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in pairs):
        raise ValueError("past_future must contain history/future column pairs")
    for key in ("weather_offsets", "weather_scales"):
        if len(fe[key]) != len(pairs) or not all(math.isfinite(v) for v in fe[key]):
            raise ValueError(f"{key} needs one finite value per weather pair")
    if any(v <= 0 for v in fe["weather_scales"]):
        raise ValueError("weather_scales must be positive")
    if fe["cov_present"] is not None and any(p not in pairs for p in fe["cov_present"]):
        raise ValueError("cov_present must be a subset of past_future")
    names = [c for pair in pairs for c in pair] + [
        fe["hint_col"],
        ds["pv_col"],
        ds["label_col"],
    ]
    if any(not isinstance(c, str) or not c for c in names):
        raise ValueError("Feature columns must be nonempty strings")
    if len({c.lower() for c in names}) != len(names):
        raise ValueError("Weather, hint, history and label columns must be distinct")
    aliases = fe["col_aliases"]
    if not isinstance(aliases, dict) or any(
        not isinstance(v, list) or any(not isinstance(c, str) or not c for c in v)
        for v in aliases.values()
    ):
        raise ValueError("col_aliases must map names to lists of column names")
    for name in names[:-1]:
        if any(c.lower() == ds["label_col"].lower() for c in aliases.get(name, [])):
            raise ValueError("Input aliases must not refer to future labels")
    if train["loss"] not in ("mae", "mse", "huber"):
        raise ValueError("train.loss must be mae, mse or huber")
    label_norm = ds["label_norm"]
    if label_norm is not None and (
        not isinstance(label_norm, list)
        or any(not isinstance(col, str) or not col for col in label_norm)
    ):
        raise ValueError("dataset.label_norm must be null or a list of column names")
    if train["ema_decay"] is not None and not 0 <= train["ema_decay"] <= 1:
        raise ValueError("ema_decay must be in [0, 1]")
    schedule, early = train["schedule"], train["early_stop"]
    for key in ("batch_size", "logging_steps"):
        if type(schedule[key]) is not int or schedule[key] <= 0:
            raise ValueError(f"schedule.{key} must be a positive integer")
    for key in ("epochs_max", "lr", "grad_clip"):
        if not math.isfinite(schedule[key]) or schedule[key] <= 0:
            raise ValueError(f"schedule.{key} must be positive and finite")
    for key in ("max_steps", "save_every", "warmup_steps", "num_workers"):
        value = schedule[key]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"schedule.{key} must be a nonnegative integer")
    every = schedule["eval_every_steps"]
    if every is not None and (type(every) is not int or every <= 0):
        raise ValueError("eval_every_steps must be positive or null")
    if schedule["sampling"] not in ("random", "station_day"):
        raise ValueError("sampling must be random or station_day")
    if schedule["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if early["metric"] not in ("nanwang_short_D+1", "nrmse"):
        raise ValueError("early_stop.metric must be nanwang_short_D+1 or nrmse")
    for key in ("patience", "smooth_k", "nrmse_issue_every_min"):
        if type(early[key]) is not int or early[key] <= 0:
            raise ValueError(f"early_stop.{key} must be a positive integer")
    if type(cfg["eval"]["batch_size"]) is not int or cfg["eval"]["batch_size"] <= 0:
        raise ValueError("eval.batch_size must be a positive integer")
    return cfg
