""" M8 训练 CLI（南网服务器全量训练用；结构与 train_scripts/m6_b_train_cli.py 一致）
用法实例:

    cd <仓库>/train_shell                  # 南网放在 <仓库>/train_scripts/ 下同样可用：BASE_PATH = 本文件上两级
    python m8_train_cli.py --config j1_final2.yaml --ckpt-dir j1_final2

    # accelerate 多卡:
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 -- \
        m8_train_cli.py --config j1_final2.yaml --ckpt-dir j1_final2

    # 冒烟:
    python m8_train_cli.py --config k3_resid.yaml --ckpt-dir smoke --max-steps 2

命令行参数与 m6_b 模板相同；其余设置全在配置 forecast_backbones/m8/config/<文件名>（j1_final2 / k3_resid / k1–k6 …）里，
每个 yaml 只写与 config.py 的 DEFAULT_CFG 不同的字段：
  dataset.src_dir                     训练数据根目录（读其下 train/**/*.parquet；--src-dir / --train-path 覆盖）
  dataset.station_meta / station_meta_cols   站表（线上 info.csv）及其列名映射；相对路径按仓库根目录解析
  dataset.ts_col / pv_col / label_col，features.past_future / hint_col   训练数据列名
  train.schedule.*                    超参（lr / batch_size / max_steps / warmup_steps / weight_decay / grad_clip / precision）
  train.schedule.save_every           每 N 步存一份 checkpoint 到 hf_trainer_output/checkpoint-N/（带 model_config.yaml，事后挑步）；0 / null = 只留模板的按 epoch 存
换超参 = 换 / 新增一个 yaml。家里用 train_m8.sh（有验证集、早停）调参，选定的配置原样拿到这里全量训练。

与 m6_b 模板的差异（M8 必需）：
  1. 训练超参取 M8 配置的 train.schedule（lr / batch_size / max_steps / warmup_steps / weight_decay / grad_clip / precision /
     epochs_max），不读 optimization 段：同一个 yaml 在家里和南网训出来是同一套超参（k4/k5/k6 改的就是这些）。
  2. --max-steps 缺省 -1 = 用配置的 train.schedule.max_steps（j1_final2 = 6000，cosine 按它退火；配置为 null 才按 epochs_max）；
     > 0 时覆盖（冒烟设 2）。模板里 -1 是"按 epochs"，M8 按 epoch 跑满会把 cosine 调度拉长几十倍，所以这里改成跟配置走。
  3. feature_cols 按配置自动对列（大小写 / col_aliases 不敏感）；collate 保留 station 站名（drop_str_type=False）：
     M8 按站名查站表取容量与坐标，丢掉站名会直接报错。
  4. 训练集站号 known_stations、数据里实际有的 NWP 对 cov_present 写进 model_config.yaml；其 train.weights_path 指向同目录的
     model.safetensors（线上模板会再显式覆盖）。
  5. 构造模型时 weights_path 指向从不写入的哨兵路径，保证从零训练（runs/m8/<run> 下留着旧权重也不会被加载）。
  6. 不支持 train.ema_decay 与 sampling=station_day（pipelines.train_model 不接回调 / 采样器），配置里开了直接报错。
"""

import os, argparse, sys
import torch
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BASE_PATH = str(Path(__file__).resolve().parent.parent)
sys.path.append(BASE_PATH)

from pipelines.transformer_dataset import UnifiedDataset
from pipelines.transformer_trainer import train_model

from forecast_backbones.m8.config import DEFAULT_CFG, load_config, deep_merge, dump_config
from forecast_backbones.m8.data import unified_feature_cols, station_ids_from_names, report_inputs, check_hint_units

WEIGHTS_FILE = "model.safetensors"


def build_training_args(config_dict: dict, result_dir):
    """config_dict = load_config 之后的完整 M8 配置。超参全部来自 train.schedule。"""
    from transformers import TrainingArguments
    s = config_dict["train"]["schedule"]
    cuda = torch.cuda.is_available()
    precision = str(s.get("precision", "fp32"))
    save_every = int(s.get("save_every") or 0)
    training_args = TrainingArguments(
        output_dir = str(Path(result_dir) / "hf_trainer_output"),
        num_train_epochs = float(s["epochs_max"]),
        per_device_train_batch_size = int(s["batch_size"]),
        per_device_eval_batch_size = int(s["batch_size"]),
        max_steps = int(s["max_steps"]) if s.get("max_steps") else -1,

        gradient_accumulation_steps = 1,

        optim = "adamw_torch",

        # save_every > 0：每 N 步存 checkpoint-N/ 且全部保留（训完补 model_config.yaml，事后挑步）；否则与模板相同
        save_strategy = "steps" if save_every > 0 else "epoch",
        **({"save_steps": save_every} if save_every > 0 else {}),
        save_total_limit = None if save_every > 0 else 1,

        logging_strategy = "steps",
        logging_steps = int(s.get("logging_steps", 50)),
        logging_first_step = True,

        learning_rate = float(s["lr"]),
        weight_decay = float(s["weight_decay"]),
        lr_scheduler_type = "cosine",
        warmup_steps = int(s.get("warmup_steps", 0)),
        max_grad_norm = float(s["grad_clip"]),
        seed = int((config_dict["run"].get("seeds") or [0])[0]),

        remove_unused_columns = False,
        dataloader_drop_last = False,
        dataloader_num_workers = int(s.get("num_workers", 0)),
        use_cpu = not cuda,
        fp16 = precision == "fp16" and cuda,
        bf16 = precision == "bf16" and cuda and torch.cuda.is_bf16_supported(),
        report_to = "none",
        logging_dir = None,
        run_name = None,

        do_train = True,
        do_eval = False,
    )

    return training_args


def self_contained(cfg: dict, ckpt_dir) -> dict:
    """model_config.yaml 原地可加载：run.out_dir = 该目录，train.weights_path = ${run.out_dir}/model.safetensors。"""
    return deep_merge(cfg, {"run": {"out_dir": str(Path(ckpt_dir).resolve())},
                            "train": {"weights_path": "${run.out_dir}/" + WEIGHTS_FILE}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="m8 独立训练脚本（全量训练，无验证集）")
    parser.add_argument("--config", required=True, help="YAML 路径或 forecast_backbones/m8/config/ 目录内的文件名（如 j1_final2.yaml）")
    parser.add_argument("--ckpt-dir", required=True, help="runs/m8 下的运行名，也可传绝对路径")
    parser.add_argument("--src-dir", default=None, help="数据根目录，读其下 train/；默认取配置 dataset.src_dir")
    parser.add_argument("--train-path", help="可选：直接指定训练 Parquet 目录")
    parser.add_argument("--max-steps", type=int, default=-1, help="默认 -1 = 按配置 train.schedule.max_steps；冒烟可设为 2")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    config_file = args.config

    MODEL_NAME = "m8"
    MODEL_CONFIG_PATH = (Path(BASE_PATH) / "forecast_backbones" / MODEL_NAME / "config" / config_file)
    if not MODEL_CONFIG_PATH.is_file():
        parser.error(f"配置不存在: {MODEL_CONFIG_PATH}")

    RUN_ID = ckpt_dir
    RUN_BASE = Path(BASE_PATH) / "runs"
    RUN_PATH = RUN_BASE / MODEL_NAME / RUN_ID
    RUN_BASE.mkdir(exist_ok=True, parents=True)
    IS_MAIN = int(os.environ.get("RANK", 0)) == 0

    # ================================ build config and model ================================
    import yaml
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f) or {}
    cfg = load_config(model_config)                                   # 与 DEFAULT_CFG 深合并 + 合法性校验
    src_dir = args.src_dir or cfg["dataset"].get("src_dir")
    if not args.train_path and not src_dir:
        parser.error("没有训练数据位置：配置写 dataset.src_dir，或给 --src-dir / --train-path")
    DATASET_PATH = Path(args.train_path) if args.train_path else Path(src_dir) / "train"
    if not DATASET_PATH.is_dir():
        parser.error(f"训练数据目录不存在: {DATASET_PATH}")
    if cfg["train"].get("ema_decay") is not None:
        parser.error("train.ema_decay 需要 EMA 回调，本脚本不支持：用 train_m8.sh --train_parquet，或把 ema_decay 设为 null")
    if str(cfg["train"]["schedule"].get("sampling") or "random") != "random":
        parser.error("train.schedule.sampling=station_day 需要自定义采样器，本脚本只支持 random")

    from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
    model = TransformersAdaptedM8Model(deep_merge(cfg, {"train": {"weights_path": str(RUN_PATH / ".fresh" / WEIGHTS_FILE)}}))

    # ================================ build the training args ================================
    training_args = build_training_args(cfg, result_dir = RUN_PATH)
    if args.max_steps > 0:
        training_args.max_steps = args.max_steps
    if training_args.n_gpu > 1:
        parser.error(f"单进程看到 {training_args.n_gpu} 张卡：HF 会退到 DataParallel，M8 的 dict batch 不支持。"
                     f"单卡用 CUDA_VISIBLE_DEVICES=0，多卡用 accelerate launch")
    model.reseed(int(training_args.seed) * 1000 + int(os.environ.get("RANK", 0)))     # 每个进程不同的增强随机流

    # ================================ build the training dataset ================================
    ds_cfg = cfg["dataset"]
    parquet_files = sorted(DATASET_PATH.glob("**/*.parquet"))
    if not parquet_files:
        parser.error(f"训练数据目录下没有 parquet: {DATASET_PATH}")
    data_cols = pq.read_schema(parquet_files[0]).names
    if "station" not in data_cols:
        parser.error(f"训练数据没有 station 站名列（M8 按站名查站表）：{parquet_files[0]}")
    feature_cols = unified_feature_cols(model.fe, data_cols)

    train_dataset = UnifiedDataset(
        str(DATASET_PATH / "**" / "*.parquet"),
        timestamp_cols=[ds_cfg["ts_col"]],
        entity_cols=[ds_cfg["ts_col"], "station"],
        feature_cols=feature_cols,
        label_cols=[ds_cfg["label_col"]],
        for_hf_trainer=True,
    )

    # 训练集站号（推理时名单外的站走未知站槽）与数据里实际有的 NWP 对（推理时缺列兜底），都写进 model_config.yaml
    known = model.set_known_stations(station_ids_from_names(model.station_meta, np.unique(np.asarray(train_dataset.column("station"), dtype=object))))
    cov_present = report_inputs(model, list(data_cols), verbose=IS_MAIN)
    model.cfg["features"]["cov_present"] = cov_present
    unit_msg = check_hint_units(model.fe, parquet_files[0])           # 插值列与功率是否同单位（都要是 MW）
    if unit_msg and IS_MAIN:
        print(f"[m8] ⚠️ {unit_msg}", flush=True)
    if IS_MAIN:
        print(f"[m8] config={MODEL_CONFIG_PATH.name} train={len(train_dataset)} 行 / 站 {len(known)} 个 / feature_cols={feature_cols}")
        print(f"[m8] lr={training_args.learning_rate} batch={training_args.per_device_train_batch_size} "
              f"max_steps={training_args.max_steps} warmup={training_args.warmup_steps} wd={training_args.weight_decay} "
              f"params={model.n_params():,} → {RUN_PATH}", flush=True)

    # ================================ build the collate function ================================
    from functools import partial
    custom_collate_fn = partial(
        UnifiedDataset.collate_fn,
        timestamp_output = "epoch_s",
        drop_str_type = False,                                         # 保留 station 站名：M8 按站名查站表
        for_hf_trainer=True
    )

    model.train()
    trainer = train_model(
        training_config = training_args,
        model = model,
        train_dataset = train_dataset,
        eval_dataset = train_dataset,
        collate_fn = custom_collate_fn
    )

    MODEL_SAVE_PATH = Path(RUN_PATH) / "model_result"
    trainer.save_model(MODEL_SAVE_PATH)
    trainer.accelerator.wait_for_everyone()

    if trainer.accelerator.is_main_process:
        cfg_dump = deep_merge(deep_merge(DEFAULT_CFG, model_config),
                              {"dataset": {"known_stations": known}, "features": {"cov_present": cov_present}})
        dump_config(self_contained(cfg_dump, MODEL_SAVE_PATH), MODEL_SAVE_PATH / "model_config.yaml")
        ckpts = sorted(Path(training_args.output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        for ck in ckpts:
            dump_config(self_contained(cfg_dump, ck), ck / "model_config.yaml")
        print(f"[m8] 产物：{MODEL_SAVE_PATH}/{{{WEIGHTS_FILE}, model_config.yaml}}"
              + (f"；中间 checkpoint {len(ckpts)} 个：{training_args.output_dir}/checkpoint-*" if ckpts else ""))
