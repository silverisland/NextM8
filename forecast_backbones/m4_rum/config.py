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
        "station_key": None,
        "capacity_col": "cap_power_on",
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
        "architecture": "legacy",
        "decoder_layers": 2,
        "station_embedding": False,
        "n_station_slots": 4096,
        "extensions": {
            "residual_head": False,
            "weather_day_reference": False,
            "day_hidden": 32,
            "local_weather": False,
            "local_kernels": [3, 9],
            "hint_gate": False,
            "hint_gate_hidden": 32,
        },
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
        "slot_dropout": 0.3,
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
            "by_step": [],
            "clip_negative": False,
            "capacity_cap": False,
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
    override = dict(raw_override(config))
    # Old non-smoothing checkpoints stored this no-op block. Accept and remove
    # it without retaining any preprocessing in newly saved configurations.
    retired = override.pop("preprocess", {})
    if not isinstance(retired, dict) or set(retired) - {"pv_smoothing"}:
        raise ValueError("Unknown retired preprocess configuration")
    smoothing = retired.get("pv_smoothing", {})
    if not isinstance(smoothing, dict) or set(smoothing) - {
        "history_window",
        "target_window",
        "alignment",
    }:
        raise ValueError("Unknown retired pv_smoothing configuration")
    if (
        any(
            type(smoothing.get(key, 1)) is not int or smoothing.get(key, 1) != 1
            for key in ("history_window", "target_window")
        )
        or smoothing.get("alignment", "centered") != "centered"
    ):
        raise ValueError(
            "PV smoothing has been removed. Historical smoothing checkpoints "
            "require the matching historical code version; do not silently disable smoothing."
        )
    _check_keys(override, DEFAULT_CFG)
    cfg = interpolate(deep_merge(DEFAULT_CFG, override))
    ds, model, fe, train = (cfg[k] for k in ("dataset", "model", "features", "train"))
    if ds["station_key"] is not None and (
        not isinstance(ds["station_key"], str) or not ds["station_key"]
    ):
        raise ValueError("dataset.station_key must be a column name or null")
    if not isinstance(ds["capacity_col"], str) or not ds["capacity_col"]:
        raise ValueError("dataset.capacity_col must be a column name")
    if ds["station_key"] == ds["station_id_col"]:
        raise ValueError(
            "station_key must differ from station_id_col (reserved for internal integer IDs)"
        )
    if ds["pred_len"] != 192 or ds["step_min"] != 15:
        raise ValueError("M4-Rum uses 192 steps, starting 15 minutes after issue time")
    if model["architecture"] not in ("legacy", "query_decoder"):
        raise ValueError("model.architecture must be legacy or query_decoder")
    if type(model["station_embedding"]) is not bool:
        raise ValueError("model.station_embedding must be boolean")
    if type(model["n_station_slots"]) is not int or model["n_station_slots"] <= 0:
        raise ValueError("model.n_station_slots must be a positive integer")
    if model["station_embedding"] and model["architecture"] != "query_decoder":
        raise ValueError("station_embedding requires model.architecture=query_decoder")
    ext = model["extensions"]
    switches = ("residual_head", "weather_day_reference", "local_weather", "hint_gate")
    for key in switches:
        if type(ext[key]) is not bool:
            raise ValueError(f"model.extensions.{key} must be boolean")
    if any(ext[key] for key in switches) and model["architecture"] != "query_decoder":
        raise ValueError("model.extensions require model.architecture=query_decoder")
    for key in ("day_hidden", "hint_gate_hidden"):
        if type(ext[key]) is not int or ext[key] <= 0:
            raise ValueError(f"model.extensions.{key} must be a positive integer")
    kernels = ext["local_kernels"]
    if (
        not isinstance(kernels, list)
        or not kernels
        or any(
            type(k) is not int or k < 3 or k > ds["pred_len"] or k % 2 == 0
            for k in kernels
        )
        or len(set(kernels)) != len(kernels)
    ):
        raise ValueError(
            "model.extensions.local_kernels must be unique odd integers in [3, pred_len]"
        )
    known = ds["known_stations"]
    if known is not None and (
        not isinstance(known, list)
        or any(type(sid) is not int for sid in known)
        or len(set(known)) != len(known)
    ):
        raise ValueError(
            "dataset.known_stations must be null or a list of unique integers"
        )
    if type(model["decoder_layers"]) is not int or model["decoder_layers"] <= 0:
        raise ValueError("model.decoder_layers must be a positive integer")
    if model["architecture"] == "query_decoder" and ds["context_len"] % 96:
        raise ValueError("query_decoder context_len must contain complete 96-step days")
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
    for key in ("history_dropout", "hint_dropout", "slot_dropout"):
        if (
            isinstance(train[key], bool)
            or not isinstance(train[key], (int, float))
            or not 0 <= train[key] <= 1
        ):
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
    methods = ("median", "p_star", "p_star_l1")
    if point["default"] not in methods or point["tail"] != "flat":
        raise ValueError(
            "decision.point supports median/p_star/p_star_l1 with flat tails"
        )
    if not isinstance(point["by_step"], list):
        raise ValueError("decision.point.by_step must be a list")  # noqa: TRY004 -- configuration errors use ValueError
    occupied = set()
    for rule in point["by_step"]:
        if not isinstance(rule, dict) or set(rule) != {"steps", "method"}:
            raise ValueError("Each by_step rule requires steps and method")
        bounds = rule["steps"]
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(type(x) is not int for x in bounds)
            or not 0 <= bounds[0] < bounds[1] <= ds["pred_len"]
            or rule["method"] not in methods
        ):
            raise ValueError("Invalid by_step bounds or method")
        indices = set(range(*bounds))
        if indices & occupied:
            raise ValueError("by_step rules must not overlap")
        occupied.update(indices)
    for key in ("clip_negative", "capacity_cap"):
        if type(point[key]) is not bool:
            raise ValueError(f"decision.point.{key} must be boolean")
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
    if (
        point["default"] != "median"
        or point["by_step"]
        or point["clip_negative"]
        or point["capacity_cap"]
    ) and train["loss"] != "pinball":
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
