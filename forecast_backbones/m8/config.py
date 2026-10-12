# -*- coding: utf-8 -*-
# forecast_backbones/m8/config.py
"""M8 配置：yaml → dict，与 DEFAULT_CFG 深合并，解析 ${run.out_dir}。
deep_merge / interpolate / dump_config 直接复用 m6.config（语义相同，不另写一份）。"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

try:
    from ..m6.config import deep_merge, interpolate, dump_config
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m6.config import deep_merge, interpolate, dump_config

__all__ = ["DEFAULT_CFG", "load_config", "deep_merge", "interpolate", "dump_config", "raw_override"]

DEFAULT_CFG: dict = {
    "run": {"model_name": "m8", "seeds": [0], "out_dir": "runs/m8_baseline"},
    "dataset": {
        "pv_col": "observe_power",
        "label_col": "observe_power_future",
        "ts_col": "timestamp_win",
        "station_id_col": "station_id",
        "station_meta": "forecast_backbones/m6/station_meta.csv",   # 与 m6 共用一份站表（station_id, station, lat, lon, alt_m, cap_mw）
        "tz": "Asia/Shanghai",
        "step_min": 15,
        # 训练入口自动写入：训练集里出现过的整数站号。推理时不在名单里的站走"未知站"槽位 0（见 train.slot_dropout）
        "known_stations": None,
        "context_len": 672,
        "pred_len": 192,
    },
    "features": {
        # (历史列, 未来列) 对；两列都在 batch_dict 里才用，缺列只提示一次并跳过（该通道为 0）
        "past_future": [
            ["general_ghi", "general_ghi_predict"],
            ["general_temp", "general_temp_predict"],
            ["GHI_solargis", "ghi_solargis_predict"],
            ["temp_solargis", "temp_solargis_predict"],
        ],
        "past_only": [],
        # 只有未来段的"提示"通道（如 TimesFM 单变量预测 timesfm_predict）：历史段恒 0，未来段按 scale 归一；
        # scale=target 表示除以每行的目标尺度（容量或近期峰值，并随幅度调制同乘），fixed 表示按 scales 表的固定常数
        "future_only": [],                                 # 例：[{"col": "timesfm_predict", "scale": "target"}]
        # 残差学习：模型输出 = 该 future_only 列（归一后）+ 网络输出；须是 future_only 里 scale=target 的列
        "residual_base_col": None,
        "nwp_members": {"enabled": False, "cols": [f"SSRD_pos_{i}" for i in range(1, 10)], "future_suffix": "_predict"},
        # 列名（小写）含某个键时按 [shift, scale] 归一：(x - shift) / scale；都不含用 default
        "scales": {"ghi": [0.0, 1000.0], "ssrd": [0.0, 1000.0], "temp": [15.0, 15.0], "t2m": [15.0, 15.0], "default": [0.0, 1.0]},
        "clear_sky_index_col": "general_ghi_predict",       # 驱动 k_nwp / nwp_resid 的 GHI 类未来列（其历史列按 past_future 配对找）
        "derived": {"solar": True, "time_encoding": True, "nwp_resid": True,
                    # 按站自适应晴空出力比（2026-10-12）：历史可见段近正午点 y / p_clear 的高分位，用它缩放 p_clear 通道与决策层晴空上限。
                    # p_clear 按水平面 GHI 算（1000 W/m² = 满容量），对倾斜 / 跟踪 / 双面阵列在冬季高纬会偏低 30–60%，
                    # 固定 1.05 倍上限会把峰值裁掉（华能北方润达 12 月：上限 0.51·cap，观测峰 0.78·cap）。缺坐标的退化站比值恒 1
                    "clear_sky_ratio": {"enabled": False, "quantile": 0.98, "near_noon_frac": 0.6, "lo": 1.0, "hi": 2.5}},
        # coords / log_cap：静态向量里的经纬高 / log 容量。它们能唯一标识训练站，关掉可查已见/未见站差距是否来自按站记忆
        "static": {"station_embedding": True, "n_station_slots": 4096, "context_stats": True, "coords": True, "log_cap": True},
    },
    "model": {
        "layout": "fused",               # fused | split（目标流 / 协变量流双流 + 跨流混合）
        "head_mode": "step_cov",         # step_cov（逐步拼协变量的输出头）| token（TiRex 式 token 头）
        "patch_len": 16,                 # 须整除 context_len 与 pred_len：8/16/24/32/48
        "d_model": 256,
        "n_heads": 4,
        "d_ff": 1024,
        "blocks": ["conv", "deltanet", "conv", "deltanet"],   # conv | deltanet | slstm | attention
        "dropout": 0.1,
        "bidirectional": False,
        "short_conv_kernel": 4,
        "deltanet_gated": True,
        "head_hidden": 256,
        "pos_in_patch_dim": 32,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # ---- 超短期 / 指标对齐模块（默认全关 = 原模型）----
        "mean_head": False,                                   # 额外一路点预测，按南网相对 L2 训练（给 D+1 段用：decision by_step method=mean_head）
        "recent_skip": {"enabled": False, "k": 16},           # 输出头直连最近 k 步观测 + 晴空指数持久化锚（NLinear / scaled-persistence 式）
        "ultra_head": {"enabled": False, "n_steps": 16, "hidden": 256},   # 前 n 步的直出 MLP 头（TabM/TiDE 式），与主头按步门控融合
    },
    "train": {
        "cpm": {"enabled": True, "p_row": 0.5, "min_frac": 0.05, "max_frac": 0.25, "max_runs": 3},
        "cov_dropout": 0.1,
        "hint_dropout": 0.3,
        # 训练时按此概率把站点嵌入槽位换成"未知站"槽 0，让模型学会不依赖站点嵌入；推理时不在 known_stations 里的站走槽 0
        "slot_dropout": 0.3,                   # future_only 通道整行清零的概率（与 cov_dropout 独立抽样；残差基准同步为 0）
        "amp_mod": {"p": 0.3, "lo": 0.6, "hi": 1.0},
        "schedule": {"lr": 5.0e-4, "weight_decay": 0.05, "batch_size": 64, "max_steps": 20000, "epochs_max": 20,
                     "warmup_steps": 500, "grad_clip": 1.0, "precision": "fp32", "logging_steps": 50,
                     "eval_every_steps": 500, "sampling": "random", "progress_bar": True, "num_workers": 0},
        # 损失：pinball（9 档分位）+ 可选的超短期加权 / 指标对齐项
        #   horizon_weights.ultra_weight：前 ultra_steps 步的 pinball 权重（1 = 不加权）
        #   metric_aux.ultra_l1：前 ultra_steps 步 中位数 的南网式相对 L1（|y-p| / max(y, 0.2·cap)）
        #   metric_aux.short_l2：short_steps 段 点预测（有 mean_head 用它，否则中位数）的南网式相对 L2
        #   metric_aux.mean_head_weight：mean_head 开着时，它在全部监督步上的相对 L2 权重
        #   night_weight：夜间步在损失里的权重。1 = 与白天同权（默认，模型自己学夜间为 0，与 M1–M7 不做夜间后处理的口径一致）；
        #                 0 = 夜间不监督（2026-10-12 之前的旧行为，须配合 decision.physics.night_zero: true 用）
        "loss": {"horizon_weights": {"ultra_steps": 16, "ultra_weight": 1.0}, "night_weight": 1.0,
                 "metric_aux": {"ultra_l1": 0.0, "short_l2": 0.0, "short_steps": [55, 151], "floor_ratio": 0.2, "mean_head_weight": 1.0}},
        # early_stop.metric：nanwang_after_pstar（D+1）| nrmse | nanwang_ultra_proxy（整点行上的超短期代理分）
        #                    | nanwang_combo（combo_weights 加权的 D+1 与超短期代理分）
        "early_stop": {"metric": "nanwang_after_pstar", "patience": 5, "nrmse_issue_every_min": 60,
                       "combo_weights": {"short": 0.5, "ultra": 0.5}},
        "weights_path": "${run.out_dir}/model.safetensors",   # 存在即在构造时加载（推理）；训练入口会改指向哨兵路径
    },
    "decision": {
        # night_zero 默认关（2026-10-12）：其它模型都不做夜间归零，指标又全点含夜间，开着会把 M8 的分数虚高；
        # 模型靠 is_night 输入通道 + 夜间进损失自己学 0。开它必须同时把 train.loss.night_weight 设回 0 重训
        # clip_negative：负值截到 0（负功率物理上不可能）。它对夜间有一点帮助（略负的夜间输出变成 0 = 全对），
        # 想和 M1–M7 严格裸对比可在 decision yaml 里设 false，不用重训
        "physics": {"night_zero": False, "clear_sky_cap_ratio": 1.05, "capacity_cap": True, "clip_negative": True},
        "point": {"default": "median", "by_step": [], "floor_ratio": 0.2, "tau_grid_step": 0.005,
                  "tail": "flat", "shrink_factor": None},
    },
    "eval": {"batch_size": 256, "report_dir": "${run.out_dir}/reports"},
}


def raw_override(config_path) -> dict:
    """yaml 路径 / dict / None → 尚未合并、未插值的 override dict（同 m6.transformer_trainer._raw_override）。"""
    if config_path is None:
        return {}
    if isinstance(config_path, dict):
        return config_path
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}


def normalize_future_only(entries) -> list[dict]:
    """features.future_only 的三种写法统一成 [{"col", "scale"}]：字符串、[col, scale]、{col, scale}。"""
    out = []
    for e in entries or []:
        if isinstance(e, str):
            d = {"col": e, "scale": "target"}
        elif isinstance(e, dict) and "col" in e:
            d = {"col": str(e["col"]), "scale": str(e.get("scale", "target"))}
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            d = {"col": str(e[0]), "scale": str(e[1])}
        else:
            raise ValueError(f"features.future_only 条目须是列名、[列名, scale] 或 {{col, scale}}，得到 {e!r}")
        if d["scale"] not in ("target", "fixed"):
            raise ValueError(f"features.future_only[{d['col']}].scale 须是 target 或 fixed，得到 {d['scale']!r}")
        out.append(d)
    if len({d["col"] for d in out}) != len(out):
        raise ValueError("features.future_only 列名重复")
    return out


def load_config(src) -> dict:
    cfg = interpolate(deep_merge(DEFAULT_CFG, raw_override(src)))
    ds, m = cfg["dataset"], cfg["model"]
    fe = cfg["features"]
    fe["future_only"] = normalize_future_only(fe.get("future_only"))
    rb = fe.get("residual_base_col")
    if rb is not None and not any(d["col"] == rb and d["scale"] == "target" for d in fe["future_only"]):
        raise ValueError(f"features.residual_base_col={rb!r} 须是 features.future_only 里 scale=target 的列")
    P = int(m["patch_len"])
    if int(ds["context_len"]) % P or int(ds["pred_len"]) % P:
        raise ValueError(f"model.patch_len={P} 须整除 dataset.context_len={ds['context_len']} 与 pred_len={ds['pred_len']}")
    if int(m["d_model"]) % int(m["n_heads"]):
        raise ValueError(f"model.d_model={m['d_model']} 须被 n_heads={m['n_heads']} 整除")
    bad = [b for b in m["blocks"] if b not in ("conv", "deltanet", "slstm", "attention")]
    if bad:
        raise ValueError(f"model.blocks 含未知块类型 {bad}；可选 conv/deltanet/slstm/attention")
    lv = [float(q) for q in m["quantiles"]]
    if lv != sorted(lv) or not all(0.0 < q < 1.0 for q in lv):
        raise ValueError(f"model.quantiles 须是 (0,1) 内递增的列表，得到 {lv}")
    return cfg
