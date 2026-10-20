# -*- coding: utf-8 -*-
# forecast_backbones/m5/config.py
"""M5（TimesFM-3 + LoRA 微调）的训练 / 评估配置：yaml → dict，与 DEFAULT_CFG 深合并，解析 ${run.out_dir}。
只服务 full 模式（必须有 dataset.station_meta）；线上零样本（legacy）配置由 transformer_models 里的 DEFAULT_CFG 合并，不经过这里。
config/*.yaml 是各配方（只写与 DEFAULT_CFG 不同的字段；2026-10-20 起同 M8 平铺在 config/ 下）。"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

__all__ = ["DEFAULT_CFG", "REMOVED_KEYS", "LORA_TARGETS", "load_config", "deep_merge", "interpolate", "dump_config", "raw_override"]


def deep_merge(base: dict, override: dict) -> dict:
    """dict 递归合并；非 dict（含 list）整体替换。"""
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


def dump_config(cfg: dict, path) -> None:
    """配置写成 yaml（元组转列表，键序不变）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plain = json.loads(json.dumps(cfg, ensure_ascii=False, default=str))
    Path(path).write_text(yaml.safe_dump(plain, allow_unicode=True, sort_keys=False), encoding="utf-8")


def raw_override(config_path) -> dict:
    """yaml 路径 / dict / None → 尚未合并、未插值的 override dict。"""
    if config_path is None:
        return {}
    if isinstance(config_path, dict):
        return config_path
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}


LORA_TARGETS = ["query_proj", "key_proj", "value_proj", "out_proj"]
LR_SCHEDULERS = ("cosine", "constant", "constant_with_warmup", "linear")

DEFAULT_CFG: dict = {
    "run": {"model_name": "m5", "seeds": [0], "out_dir": "runs/m5_baseline"},
    "dataset": {
        "pv_col": "observe_power",
        "label_col": "observe_power_future",
        "ts_col": "timestamp_win",
        "station_id_col": "station_id",
        # 训练数据根目录（南网 m5_train_cli.py 读其下 train/**/*.parquet；--src-dir / --train-path 可覆盖）。家里走切分目录，不用它
        "src_dir": None,
        # 只取每站固定容量 cap_mw（损失归一与两套指标的分母）与站号（seen / unseen 报表）；经纬度只在几何变体（g1 / g2）里用
        "station_meta": "forecast_backbones/m5/station_meta.csv",
        "station_meta_cols": None,          # 站表列名映射 {内部名: 你表里的列名}，同 M8；null 时线上 info.csv 表头自动识别
        "tz": "Asia/Shanghai",
        "step_min": 15,
        "known_stations": None,             # 训练入口自动写入训练集站号（只用于报表）
        "context_len": 672,
        "pred_len": 192,
    },
    "timesfm3": {
        "checkpoint": "/data/tangqingang/timesfm-3.0-pytorch",
        "device": "auto",
        "per_core_batch_size": 64,          # 只影响 evaluator 路径
        "nan_policy": "interp",
        "make_positive": True,
        "use_znorm": False,
        "use_symmetric_averaging": True,
        "eager_init": True,
    },
    "features": {
        "past_future": [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]],
        "past_only": [],
        # 列名别名 {配置列名: 数据列名或列表}（2026-10-20，同 M8）；没写的列先精确、再大小写不敏感匹配（general_ghi ↔ general_GHI）
        "col_aliases": {},
        # 训练数据里实际有的 past_future 对：训练入口自动写入 model_config.yaml，推理时不在里面的对有列也不喂（null = 全喂）
        "cov_present": None,
        # D8：几何协变量默认关，输入与线上 M5 相同；g1 变体打开（太阳高度角 / 晴空出力各一个 past_future 通道）
        # history_envelope（C1）：功率历史日形包络一条通道；envelope_nwp（C2）：包络 × clip(GHI_nwp / GHI 历史包络)；都不需要坐标
        "derived": {"solar_elevation": False, "clear_sky_power": False, "history_envelope": False, "envelope_nwp": False},
        # 包络参数（C1 / C2 与门控 / 路由 LoRA 的形势特征共用）：上下文 7 天同一时刻槽的 quantile 分位、环形平滑 smooth 步；
        # ghi = [历史列, 未来列]（C2 与形势特征用）；ghi_floor：GHI 包络低于它的步当夜间（比值记 0）；clip_max：C2 比值上限
        "envelope": {"quantile": 0.9, "smooth": 3, "ghi": ["general_ghi", "general_ghi_predict"], "ghi_floor": 20.0, "clip_max": 1.2},
    },
    "finetune": {
        "enabled": True,
        "inference_path": "torch",          # enabled=false 时零样本走哪条：torch（与 LoRA 同一可微路径）| evaluator（predict_batch）
        "method": "lora",                   # 目前只有 lora；后续 PEFT 变体的开关位
        # experts>1：HydraLoRA 式（共享 A、K 个 B，按形势特征路由）；gate：Gated-LoRA（按形势特征逐样本调 LoRA 强度，初值 1）
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": list(LORA_TARGETS), "experts": 1, "gate": False},
        "unfreeze_output_head": True,
        # 整块解冻（全参数可训，与 LoRA 并存）：如 ["pre_transformer_resblock"]（读 NWP 前瞻的输入残差块）
        "unfreeze_modules": [],
        # 以下适配器全部默认关（adapters.py）；关时模型与原 LoRA 逐位相同
        "ia3": {"target_modules": []},      # (IA)³ 逐通道缩放（靶点规则同 LoRA，不能与 LoRA 靶点重叠）
        "prompt": {"n_tokens": 0, "per_station": False, "station_dropout": 0.25, "init_std": 0.02},   # 虚拟变量 soft prompt
        "static_embed": {"enabled": False, "dropout": 0.25},   # 零初始化静态加性嵌入（经纬度 / 海拔 / log 容量 / has_coords）
        "resblock_adapter": {"rank": 0},    # 残差块输出上的非线性适配器（0 = 关）
        "init_from": None,                  # LP-FT 热启动：训练前从这个权重文件拷入同名可训参数（如 head_only run 的 model.safetensors）；{seed} 换成本 run 种子
    },
    "train": {
        # batch_size：全局 batch（所有卡合计），每卡 = batch_size / 卡数，须整除；单卡时与旧口径相同
        "schedule": {"lr": 1.0e-4, "weight_decay": 0.01, "batch_size": 16, "max_steps": 4000, "epochs_max": 10,
                     "warmup_steps": 0, "grad_clip": 1.0, "precision": "fp32", "logging_steps": 20,
                     "eval_every_steps": 250, "sampling": "random", "progress_bar": True, "num_workers": 0,
                     "lr_scheduler": "cosine",      # cosine | constant | constant_with_warmup | linear（HF lr_scheduler_type）
                     # 南网 m5_train_cli.py：每 N 步存一份 checkpoint（hf_trainer_output/checkpoint-N/，带 model_config.yaml），
                     # 全量训练没有验证集，事后用线上脚本在训练期之后的时段上挑步；0 / null = 不存
                     "save_every": None},
        # 优化器参数分组（全 null = HF 默认单组）：lora_plus_ratio = LoRA+（lora_B 的 lr = lr × 该值）；
        # head_lr / adapter_lr = output_head / 其余可训参数（解冻块、(IA)³、prompt 等）单独的 lr
        "optim": {"lora_plus_ratio": None, "head_lr": None, "adapter_lr": None},
        # night_weight：夜间步在损失里的权重（1 = 同权）。normalize_by_cap：q、y 都除以该行 cap 再算 pinball（false = 真实单位）
        "loss": {"night_weight": 1.0, "normalize_by_cap": True},
        "early_stop": {"metric": "nanwang_after_pstar", "patience": 5, "nrmse_issue_every_min": 60, "smooth_k": 1},
        "ema_decay": None,                  # 只影子可训参数（LoRA + output_head）
        "weights_path": "${run.out_dir}/model.safetensors",   # 存在即在构造时加载（只含 LoRA + output_head）
    },
    "decision": {
        # D8：clear_sky_cap_ratio=null → 无晴空上限，只钳到容量；g2 变体 = 1.05（与 M8 同口径）
        "physics": {"night_zero": False, "clear_sky_cap_ratio": None, "capacity_cap": True, "clip_negative": True},
        "point": {"default": "median", "by_step": [], "floor_ratio": 0.2, "tau_grid_step": 0.005,
                  "tail": "flat", "shrink_factor": None, "spread_lo": None, "spread_hi": None},
    },
    "eval": {"batch_size": 64, "report_dir": "${run.out_dir}/reports"},
}

# 旧（M6 / legacy M5）键 → 新位置或删除原因；出现即报错，不让 deep_merge 静默收下
REMOVED_KEYS: dict = {
    ("calibration",): "M5 不做 conformal 校准（2026-10-14 删除）",
    ("stacking",): "M5 不做堆叠",
    ("variability",): "M5 不分多变度档",
    ("data",): "切分由 train.py 的 --splits-dir / --train-parquet 给",
    ("finetune", "lora_path"): "train.weights_path",
    ("finetune", "loss"): "train.loss",
    ("finetune", "masks"): "train.loss.night_weight",
    ("finetune", "schedule"): "train.schedule",
    ("finetune", "split"): "train.py 的 --splits-dir",
    ("finetune", "early_stop"): "train.early_stop",
    ("features", "nwp_members"): "M5 不用 NWP 集合成员",
    ("features", "target_transform"): "train.loss.normalize_by_cap",
    ("timesfm3", "context_len"): "dataset.context_len",
    ("timesfm3", "pred_len"): "dataset.pred_len",
    ("timesfm3", "past_future_cols"): "features.past_future",
    ("timesfm3", "past_only_cols"): "features.past_only",
    ("timesfm3", "return_quantiles"): "full 模式恒返回 9 档分位",
    ("timesfm3", "predict_quantile"): "decision.point.default（如 quantile:0.9）",
}


def _check_removed(override: dict) -> None:
    hit = []
    for path, hint in REMOVED_KEYS.items():
        d = override
        for k in path:
            if not isinstance(d, dict) or k not in d:
                break
            d = d[k]
        else:
            hit.append(f"{'.'.join(path)} → {hint}")
    if hit:
        raise ValueError("M5 配置里有已删除 / 已搬家的键：" + "；".join(hit))


def load_config(src) -> dict:
    override = raw_override(src)
    _check_removed(override)
    cfg = interpolate(deep_merge(DEFAULT_CFG, override))
    if not cfg["dataset"].get("station_meta"):
        raise ValueError("M5 训练 / 评估（full 模式）必须给 dataset.station_meta：站表提供每站容量 cap_mw 与站号")
    ft = cfg["finetune"]
    if ft["method"] != "lora":
        raise ValueError(f"finetune.method 目前只支持 lora，得到 {ft['method']!r}")
    if ft["inference_path"] not in ("torch", "evaluator"):
        raise ValueError(f"finetune.inference_path 须是 torch 或 evaluator，得到 {ft['inference_path']!r}")
    if int(ft["lora"]["rank"]) <= 0:
        raise ValueError(f"finetune.lora.rank 须 > 0，得到 {ft['lora']['rank']!r}")
    if ft["lora"].get("experts") is None or int(ft["lora"]["experts"]) < 1:
        raise ValueError(f"finetune.lora.experts 须 ≥ 1，得到 {ft['lora'].get('experts')!r}")
    if bool(ft["lora"].get("gate")) and int(ft["lora"].get("experts", 1) or 1) > 1:
        raise ValueError("finetune.lora.gate 与 experts>1 二选一（experts>1 的路由权重本身就是逐样本门）")
    sched = str(cfg["train"]["schedule"].get("lr_scheduler") or "cosine")
    if sched not in LR_SCHEDULERS:
        raise ValueError(f"train.schedule.lr_scheduler 须是 {LR_SCHEDULERS} 之一，得到 {sched!r}")
    ratio = cfg["decision"]["physics"].get("clear_sky_cap_ratio")
    if ratio is not None and not float(ratio) > 0:
        raise ValueError(f"decision.physics.clear_sky_cap_ratio 须为 null（无晴空上限）或正数，得到 {ratio!r}")
    return cfg
