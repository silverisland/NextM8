""" M5 训练 CLI（南网服务器全量训练用；结构与 train_scripts/m6_b_train_cli.py、m8_train_cli.py 一致）
用法实例:

    cd <仓库>/train_shell                  # 南网放在 <仓库>/train_scripts/ 下同样可用：BASE_PATH = 本文件上两级
    python m5_train_cli.py --config f1_final.yaml --ckpt-dir f1_final

    # accelerate 多卡（train.schedule.batch_size 是全局 batch，每卡 = batch_size / 卡数）:
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 -- \
        m5_train_cli.py --config f1_final.yaml --ckpt-dir f1_final

    # 冒烟:
    python m5_train_cli.py --config f1_final.yaml --ckpt-dir smoke --max-steps 2

命令行参数与 m6_b / m8 模板相同；其余设置全在配置 forecast_backbones/m5/config/<文件名>（f1_final / f2–f5 …）里，
每个 yaml 只写与 config.py 的 DEFAULT_CFG 不同的字段：
  dataset.src_dir                     训练数据根目录（读其下 train/**/*.parquet；--src-dir / --train-path 覆盖）
  dataset.station_meta / station_meta_cols   站表（线上 info.csv）及其列名映射；相对路径按仓库根目录解析
  dataset.ts_col / pv_col / label_col，features.past_future   训练数据列名（大小写不敏感）
  timesfm3.checkpoint                 TimesFM-3 底座权重目录（南网服务器上的路径）
  train.schedule.*                    超参（lr / lr_scheduler / batch_size / max_steps / warmup_steps / weight_decay / grad_clip / precision）
  finetune.lora.*                     LoRA（rank / alpha / dropout / 靶点）
  train.schedule.save_every           每 N 步存一份 checkpoint 到 hf_trainer_output/checkpoint-N/（带 model_config.yaml，事后挑步）
换超参 = 换 / 新增一个 yaml。家里用 train_m5.sh（有验证集、早停）调参，选定的配置原样拿到这里全量训练。

与 m8_train_cli.py 的差异（M5 必需）：
  1. 学习率调度取 train.schedule.lr_scheduler（f 系列是 constant；m8 写死 cosine）；每卡 batch = 全局 batch_size / WORLD_SIZE
     （与家里 train_m5.sh 同口径，卡数变了配方不变）。
  2. 模型构造前先定种子（lora_A 在构造里随机初始化），并先取 training_args.device（多卡时先绑卡，底座才不会全加载到 0 号卡）。
  3. model.save_trainable_only()：HF Trainer 存的是 model.state_dict()；M5 底座 3.3 亿参数，不开这个每份 checkpoint 约 1.3 GB，
     而且键名对不上 M5 的权重加载。开了之后 checkpoint-N / model_result 的 model.safetensors 只含 LoRA（键名相对 backbone.model，
     与家里 train.py 存的同格式），model_config.yaml 的 train.weights_path 指向它即可原地加载。
     save_only_model：checkpoint-N 不存 optimizer.pt / scheduler.pt / rng_state（不能断点续训；全量训练只事后挑步，用不到）。
     产物每份 = model.safetensors（LoRA r8 q/k/v/o，约 13 MB）+ model_config.yaml + HF 自带的 training_args.bin / trainer_state.json。
  4. 不支持：零样本（finetune.enabled=false）、train.ema_decay、sampling=station_day、train.optim 参数分组（LoRA+ / head_lr /
     adapter_lr）、finetune.init_from（LP-FT）——pipelines.train_model 不接回调 / 采样器 / 自定义优化器，配置里开了直接报错。
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

from forecast_backbones.m5.config import DEFAULT_CFG, load_config, deep_merge, dump_config
from forecast_backbones.m5.data import unified_feature_cols, station_ids_from_names
from forecast_backbones.m5.train import report_inputs
from forecast_backbones.m5.transformer_trainer import per_device_batch

WEIGHTS_FILE = "model.safetensors"


def build_training_args(config_dict: dict, result_dir):
    """config_dict = load_config 之后的完整 M5 配置。超参全部来自 train.schedule。"""
    from transformers import TrainingArguments
    s = config_dict["train"]["schedule"]
    cuda = torch.cuda.is_available()
    precision = str(s.get("precision", "fp32"))
    save_every = int(s.get("save_every") or 0)
    training_args = TrainingArguments(
        output_dir = str(Path(result_dir) / "hf_trainer_output"),
        num_train_epochs = float(s["epochs_max"]),
        per_device_train_batch_size = per_device_batch(s),             # 全局 batch / 卡数
        per_device_eval_batch_size = per_device_batch(s),
        max_steps = int(s["max_steps"]) if s.get("max_steps") else -1,

        gradient_accumulation_steps = 1,

        optim = "adamw_torch",

        # save_every > 0：每 N 步存 checkpoint-N/ 且全部保留（训完补 model_config.yaml，事后挑步）；否则与模板相同
        save_strategy = "steps" if save_every > 0 else "epoch",
        **({"save_steps": save_every} if save_every > 0 else {}),
        save_total_limit = None if save_every > 0 else 1,
        save_only_model = True,                                        # checkpoint 只要权重（事后挑步用），不存优化器 / 调度器 / RNG 状态

        logging_strategy = "steps",
        logging_steps = int(s.get("logging_steps", 50)),
        logging_first_step = True,

        learning_rate = float(s["lr"]),
        weight_decay = float(s["weight_decay"]),
        lr_scheduler_type = str(s.get("lr_scheduler") or "cosine"),
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
        ddp_timeout = 7200,

        do_train = True,
        do_eval = False,
    )

    return training_args


def self_contained(cfg: dict, ckpt_dir) -> dict:
    """model_config.yaml 原地可加载：run.out_dir = 该目录，train.weights_path = ${run.out_dir}/model.safetensors。"""
    return deep_merge(cfg, {"run": {"out_dir": str(Path(ckpt_dir).resolve())},
                            "train": {"weights_path": "${run.out_dir}/" + WEIGHTS_FILE}})


def unsupported(cfg: dict) -> list[str]:
    """pipelines.train_model 做不到的配置项（家里 train_m5.sh 支持）→ 报错说明。"""
    out = []
    if not cfg["finetune"]["enabled"]:
        out.append("finetune.enabled=false（零样本不用训练，直接评估）")
    if cfg["train"].get("ema_decay") is not None:
        out.append("train.ema_decay（需要 EMA 回调）")
    if str(cfg["train"]["schedule"].get("sampling") or "random") != "random":
        out.append("train.schedule.sampling=station_day（需要自定义采样器）")
    if any(v is not None for v in (cfg["train"].get("optim") or {}).values()):
        out.append("train.optim（LoRA+ / head_lr / adapter_lr 需要自定义优化器参数组）")
    if cfg["finetune"].get("init_from"):
        out.append("finetune.init_from（LP-FT 热启动）")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="m5 独立训练脚本（全量训练，无验证集）")
    parser.add_argument("--config", required=True, help="YAML 路径或 forecast_backbones/m5/config/ 目录内的文件名（如 f1_final.yaml）")
    parser.add_argument("--ckpt-dir", required=True, help="runs/m5 下的运行名，也可传绝对路径")
    parser.add_argument("--src-dir", default=None, help="数据根目录，读其下 train/；默认取配置 dataset.src_dir")
    parser.add_argument("--train-path", help="可选：直接指定训练 Parquet 目录")
    parser.add_argument("--max-steps", type=int, default=-1, help="默认 -1 = 按配置 train.schedule.max_steps；冒烟可设为 2")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    config_file = args.config

    MODEL_NAME = "m5"
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
    from transformers import set_seed
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f) or {}
    cfg = load_config(model_config)                                   # 与 DEFAULT_CFG 深合并 + 合法性校验
    src_dir = args.src_dir or cfg["dataset"].get("src_dir")
    if not args.train_path and not src_dir:
        parser.error("没有训练数据位置：配置写 dataset.src_dir，或给 --src-dir / --train-path")
    DATASET_PATH = Path(args.train_path) if args.train_path else Path(src_dir) / "train"
    if not DATASET_PATH.is_dir():
        parser.error(f"训练数据目录不存在: {DATASET_PATH}")
    bad = unsupported(cfg)
    if bad:
        parser.error("本脚本（pipelines.train_model）不支持：" + "；".join(bad) + "。要用就走 train_m5.sh --train_parquet")
    if not Path(str(cfg["timesfm3"]["checkpoint"])).exists():
        parser.error(f"TimesFM-3 底座权重目录不存在: {cfg['timesfm3']['checkpoint']}（改配置 timesfm3.checkpoint 成本机路径）")

    training_args = build_training_args(cfg, result_dir = RUN_PATH)
    if args.max_steps > 0:
        training_args.max_steps = args.max_steps
    _ = training_args.device                                          # 多进程时在这里建进程组、绑卡；须先于模型构造
    if training_args.n_gpu > 1:
        parser.error(f"单进程看到 {training_args.n_gpu} 张卡：HF 会退到 DataParallel，M5 的 dict batch 不支持。"
                     f"单卡用 CUDA_VISIBLE_DEVICES=0，多卡用 accelerate launch")
    set_seed(int(training_args.seed))                                 # lora_A 在构造里随机初始化：须先于构造定种子

    from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model
    model = TransformersAdaptedM5Model(deep_merge(cfg, {"train": {"weights_path": str(RUN_PATH / ".fresh" / WEIGHTS_FILE)}}))
    model.save_trainable_only()                                       # checkpoint 只存 LoRA，不存 3.3 亿参数的底座

    # ================================ build the training dataset ================================
    ds_cfg = cfg["dataset"]
    parquet_files = sorted(DATASET_PATH.glob("**/*.parquet"))
    if not parquet_files:
        parser.error(f"训练数据目录下没有 parquet: {DATASET_PATH}")
    data_cols = pq.read_schema(parquet_files[0]).names
    if "station" not in data_cols:
        parser.error(f"训练数据没有 station 站名列（M5 按站名查站表取容量）：{parquet_files[0]}")
    feature_cols = unified_feature_cols(model, data_cols)

    train_dataset = UnifiedDataset(
        str(DATASET_PATH / "**" / "*.parquet"),
        timestamp_cols=[ds_cfg["ts_col"]],
        entity_cols=[ds_cfg["ts_col"], "station"],
        feature_cols=feature_cols,
        label_cols=[ds_cfg["label_col"]],
        for_hf_trainer=True,
    )

    # 训练集站号（只用于报表 seen / unseen）与数据里实际有的 NWP 对 cov_present（推理时不在里面的对不喂），都写进 model_config.yaml
    known = model.set_known_stations(station_ids_from_names(model.station_meta, np.unique(np.asarray(train_dataset.column("station"), dtype=object))))
    cov_present = report_inputs(model, list(data_cols), verbose=IS_MAIN)
    model.cfg["features"]["cov_present"] = cov_present
    model.cov_present = {tuple(p) for p in cov_present}
    if IS_MAIN:
        lo = cfg["finetune"]["lora"]
        print(f"[m5] config={MODEL_CONFIG_PATH.name} train={len(train_dataset)} 行 / 站 {len(known)} 个 / feature_cols={feature_cols}")
        print(f"[m5] LoRA rank={lo['rank']} alpha={lo['alpha']} dropout={lo['dropout']} targets={'+'.join(lo['target_modules'])} "
              f"解冻 output_head={cfg['finetune']['unfreeze_output_head']}；可训参数 {model.n_params():,} / 总参数 {model.n_total_params():,}")
        print(f"[m5] lr={training_args.learning_rate} sched={training_args.lr_scheduler_type} "
              f"batch={training_args.per_device_train_batch_size}/卡 × {training_args.world_size} 卡 "
              f"max_steps={training_args.max_steps} warmup={training_args.warmup_steps} wd={training_args.weight_decay} "
              f"save_every={int(cfg['train']['schedule'].get('save_every') or 0)} → {RUN_PATH}", flush=True)

    # ================================ build the collate function ================================
    from functools import partial
    custom_collate_fn = partial(
        UnifiedDataset.collate_fn,
        timestamp_output = "epoch_s",
        drop_str_type = False,                                         # 保留 station 站名：M5 按站名查站表
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
        print(f"[m5] 产物：{MODEL_SAVE_PATH}/{{{WEIGHTS_FILE}, model_config.yaml}}"
              + (f"；中间 checkpoint {len(ckpts)} 个：{training_args.output_dir}/checkpoint-*" if ckpts else ""))
