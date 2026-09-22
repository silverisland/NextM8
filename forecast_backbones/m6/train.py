# -*- coding: utf-8 -*-
# forecast_backbones/m6/train.py
"""M6 路线 B 训练入口（spec §5.2）：
  1. finetune.split 走 SplitPlan 切 train / cal / 多个验证块（块前后 purge/embargo 剔训练行；shift_days 整体后移做滚动折）
  2. HF Trainer：损失走模型 compute_loss（pinball 9 档 + 夜间/缺失掩码），LoRA + output_head 可训；
     训练预算 finetune.schedule.max_steps 个优化器步，训练行按 sampling 随机抽
  3. NanwangEvalCallback：开训前零样本基线，之后每 eval_every_steps 步对每个验证块 cal→δ、块→校准→p*→南网短期，
     块均值作早停指标 + 最优权重；guards.split_val_days 时只用偶数起报日
  4. 训后：加载最优 LoRA；零样本 vs LoRA 逐站按天配对 bootstrap（split_val_days 时只用奇数起报日）→ station_routing.json
  5. 路由后的模型重拟合校准表 → calibration.parquet（与推理时一致）
  6. 报表：reports/ladder_lora.csv（阶梯第 4 步，全部验证日）、reports/station_ladder.csv；
     权重：model.safetensors（LoRA + output_head，键名相对 backbone.model）；推理配置：model_config.yaml
     状态：train_log.json、split_manifest.json
用 HF Trainer 而不是 pipelines.transformer_trainer.train_model：后者 import 的 transformer_utils 本仓库没有，
语义（forward 返 loss、collate 出 batch_dict/labels）在这里逐条保持一致。

多卡：用 accelerate launch / torchrun 起多个进程（train_m6.sh 默认用全部可见 GPU）。每个进程各持一份模型、
各吃不同的训练批（每步共 进程数×batch_size 行）；验证打分、存盘、训后路由与报表只在主进程做，其余进程训完即返回。

训练模型的构造必须是真正的零样本起点：把 finetune.lora_path / routing_path 都改指向 run.out_dir 下一个
从不写入的 `.fresh/` 哨兵路径，而不是直接用 cfg 里解析出的真实 model.safetensors / station_routing.json——
后两者是本函数自己的输出路径，若同一 out_dir 曾跑过一次训练，构造期会静默把上一轮的 LoRA 权重和路由表
读回来，epoch 0 记录的"零样本基线"和训练期的逐站前向就都不再是零样本/无路由的了。哨兵路径只影响构造期
的装载判断，不影响本函数最终把新产物写到 out/model.safetensors、out/station_routing.json 的行为。"""
from __future__ import annotations

import argparse
import functools
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments

try:
    from .config import DEFAULT_CFG, load_config, deep_merge, dump_config
    from .transformer_trainer import _raw_override, evaluate_model
    from .nanwang_callback import NanwangEvalCallback, score_split
    from .routing import route_by_days, save_routing
    from .lora import load_lora_state_dict, load_lora_weights, set_lora_enabled, trainable_summary
    from .features import attach_station_id
    from .rows import StationDaySampler, day_parity
    from . import _dist
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from config import DEFAULT_CFG, load_config, deep_merge, dump_config
    from transformer_trainer import _raw_override, evaluate_model
    from nanwang_callback import NanwangEvalCallback, score_split
    from routing import route_by_days, save_routing
    from lora import load_lora_state_dict, load_lora_weights, set_lora_enabled, trainable_summary
    from features import attach_station_id
    from rows import StationDaySampler, day_parity
    import _dist
try:
    from ...pipelines.transformer_dataset import UnifiedDataset
    from ...pipelines.time_splits import (SplitPlan, StationSplitPlan, assign_rows, verify_rows, has_cal,
                                          load_test_starts, station_keys, reconcile_test_starts, concat_parquets, _arrow_kind)
except ImportError:
    from pipelines.transformer_dataset import UnifiedDataset
    from pipelines.time_splits import (SplitPlan, StationSplitPlan, assign_rows, verify_rows, has_cal,
                                       load_test_starts, station_keys, reconcile_test_starts, concat_parquets, _arrow_kind)


def build_plan(split: dict, shift_days: int = 0, L: int = 672, H: int = 192, step_min: int = 15,
               ts=None, station=None):
    """split.test_start 是日期 → SplitPlan；是 {station: 日期} 映射，或 split.test_parquets 给了测试集文件
    （每站 test_start = 该站在测试集里最早起报时刻的整天）→ StationSplitPlan。两者都给则报错。
    给了 ts/station 时按数据对齐：只在测试里的站忽略，没有测试集的站数据用到末尾（reconcile_test_starts）。"""
    t0, files = split.get("test_start"), split.get("test_parquets")
    if files:
        if t0 is not None:
            raise ValueError("finetune.split 同时给了 test_start 与 test_parquets：删掉其一")
        starts = load_test_starts(list(files) if isinstance(files, (list, tuple)) else [files],
                                  ts_col=split.get("ts_col", "timestamp_win"), station_col=split.get("station_col"))
    elif isinstance(t0, dict):
        starts = t0
    else:
        return SplitPlan.from_dict(split, shift_days=shift_days, L=L, H=H, step_min=step_min)
    if ts is not None and station is not None:
        starts, test_only, no_test = reconcile_test_starts(ts, station, starts, int(split.get("purge_days", 2)))
        for s in test_only:
            print(f"[m6] 站 {s} 只在测试集里、训练数据里没有：切分时忽略")
        for s in no_test:
            print(f"[m6] 站 {s} 没有测试集：数据一直用到它的最后一天")
        split = {**split, "stations_without_test": sorted(set(split.get("stations_without_test") or []) | set(no_test))}
    return StationSplitPlan.from_dict(split, starts, shift_days=shift_days, L=L, H=H, step_min=step_min)


def make_splits(ds_all, split: dict, shift_days: int = 0, L: int = 672, H: int = 192, step_min: int = 15,
                station_col: str = "station_id", need_cal: bool = True):
    """按 SplitPlan / StationSplitPlan 在起报时间轴上切 train / cal / val（全部验证块合并）/ val_blocks（逐块列表）。
    train 行已剔除每个验证块前 purge、后 purge+embargo，以及校准段前 purge 之后的一切；返回末项为 plan。
    按站切分时某站容不下的验证块该站不出行，其他站照常；块序号是全局的。
    need_cal=False（calibration.enabled=false，不做 conformal）：不要求校准段，返回的 cal 为 None。"""
    ts = ds_all._ts_as_datetime64(ds_all.timestamp_cols[0])
    station = None
    if split.get("test_parquets") or isinstance(split.get("test_start"), dict):
        known = set(ds_all.timestamp_cols) | set(ds_all.entity_cols) | set(ds_all.feature_cols) | set(ds_all.label_cols)
        if station_col not in known:
            raise ValueError(f"按站切分需要站标识列 {station_col!r}，数据里没有（列：{sorted(known)[:12]}…）")
        station = station_keys(ds_all.column(station_col))
    plan = build_plan(split, shift_days=shift_days, L=L, H=H, step_min=step_min, ts=ts, station=station)
    if need_cal and not has_cal(plan):
        raise ValueError("M6 训练需要校准段：finetune.split.cal_days 须 ≥ 1（不做 conformal 时写 calibration.enabled: false）")
    if not need_cal and has_cal(plan):
        print("[m6] calibration.enabled=false：切分里的校准段不使用，这些行也不进训练；想让它们进训练请把 cal_days 改成 0")
    role, block = assign_rows(plan, ts, station)

    def _sel(mask, name):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError(f"切分后 {name} 段为空（plan={plan.to_dict()}）")
        return ds_all.select(idx)

    is_val = role == "val"
    ds_train, ds_val_all = _sel(role == "train", "train"), _sel(is_val, "val")
    ds_cal = _sel(role == "cal", "cal") if need_cal else None
    ds_val_blocks = [_sel(is_val & (block == i), f"val_blocks[{i}]") for i in range(len(plan.val_blocks))]
    verify_rows(plan, ts, role, block, station)
    return ds_train, ds_cal, ds_val_all, ds_val_blocks, plan


def _expand_paths(spec) -> list[str]:
    """parquet 路径：单个路径 / 通配符 / 列表 → 排好序的文件列表；一个都没匹配到直接报错。"""
    import glob as _glob
    items = list(spec) if isinstance(spec, (list, tuple)) else [spec]
    out = []
    for it in items:
        hit = sorted(_glob.glob(str(it)))
        if not hit and Path(str(it)).exists():
            hit = [str(it)]
        if not hit:
            raise FileNotFoundError(f"parquet 路径 {it!r} 没匹配到文件")
        out.extend(hit)
    return out


def _dataset_with_station_id(paths, cfg: dict, station_meta, scratch_dir, reuse: bool = False) -> UnifiedDataset:
    """按 M6 口径建 UnifiedDataset（arrow 后端，make_splits/select 需要）。数据没有 station_id 列而只有站名列时，
    按 station_meta 映射补上并落到 scratch_dir 下的临时 parquet 再装载（arrow 后端不能就地加列）。
    reuse=True：临时 parquet 已存在就直接读（多卡时主进程写好，其余进程不再重复转换）。"""
    ds_cfg = cfg["dataset"]; id_col = str(ds_cfg["station_id_col"])
    import pyarrow.parquet as pq
    schemas = [pq.read_schema(p) for p in paths]
    cols = schemas[0].names
    # 数据自带的 station_id 列是文本（站名，或文本存的数字）：collate 会丢掉文本列，所有行会静默落成站 0，所以先转成整数编号
    text_id = id_col in cols and any(_arrow_kind(sc.field(id_col).type) == "文本" for sc in schemas if id_col in sc.names)
    tmp = Path(scratch_dir) / f"{Path(paths[0]).stem}_with_{id_col}.parquet"
    if (id_col not in cols or text_id) and reuse and tmp.exists():
        paths = [str(tmp)]
    elif id_col not in cols or text_id:
        name_col = id_col if text_id else "station"
        df = concat_parquets(paths, station_col=name_col)
        if text_id:
            vals = pd.Series(station_keys(df[id_col].to_numpy()), index=df.index)
            as_int = pd.to_numeric(vals, errors="coerce")
            if as_int.notna().all():
                df[id_col] = as_int.astype(np.int64)
                how = f"{id_col!r} 列是文本存的整数编号：直接转成整数"
            else:
                df = attach_station_id(df.drop(columns=[id_col]).assign(_station_name=vals), station_meta,
                                       id_col=id_col, name_col="_station_name").drop(columns=["_station_name"])
                how = f"{id_col!r} 列是站名文本：按 station_meta 的 station 列映射成整数编号"
        else:
            df = attach_station_id(df, station_meta, id_col=id_col, name_col="station")
            how = f"数据无 {id_col!r} 列：按 station_meta 从站名映射补上"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp, index=False)
        print(f"[m6] {how}，写到 {tmp}")
        paths = [str(tmp)]
    return UnifiedDataset(paths if len(paths) > 1 else paths[0], timestamp_cols=[ds_cfg["ts_col"]], entity_cols=[],
                          label_cols=[ds_cfg["label_col"]], timestamp_output="epoch_s", epoch_unit="s",
                          for_hf_trainer=True)


def _check_presplit_dir(splits_dir, need_cal: bool = True) -> Path:
    d = Path(splits_dir)
    need = ["train.parquet", "valid.parquet", "split_manifest.json"] + (["cal.parquet"] if need_cal else [])
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        hint = ("：M6 做 conformal 需要校准段，用 pipelines.time_splits 切分时 cal_days 须 ≥ 1；"
                "不做 conformal 请在配置里写 calibration.enabled: false") if "cal.parquet" in missing else ""
        raise FileNotFoundError(f"{d} 缺 {missing}{hint}")
    return d


def load_presplit(splits_dir, cfg: dict, station_meta, scratch_dir, reuse: bool = False):
    """读 pipelines.time_splits 落好的目录：train.parquet / valid.parquet（含 split_block）/ cal.parquet /
    split_manifest.json。返回 (ds_train, ds_cal, ds_val_all, ds_val_blocks, manifest)。块序号沿用 manifest 的全局序号。
    calibration.enabled=false 时不读 cal.parquet（有也不读），ds_cal 为 None。reuse 见 _dataset_with_station_id。"""
    need_cal = bool(cfg["calibration"]["enabled"])
    d = _check_presplit_dir(splits_dir, need_cal)
    man = json.loads((d / "split_manifest.json").read_text(encoding="utf-8"))
    load = functools.partial(_dataset_with_station_id, cfg=cfg, station_meta=station_meta, scratch_dir=scratch_dir, reuse=reuse)
    ds_train = load([str(d / "train.parquet")])
    ds_cal = load([str(d / "cal.parquet")]) if need_cal else None
    ds_val = load([str(d / "valid.parquet")])
    blk = np.asarray(ds_val.column("split_block")).astype(int) if "split_block" in ds_val.feature_cols \
        else None
    if blk is None:
        raise ValueError(f"{d / 'valid.parquet'} 没有 split_block 列：不是 pipelines.time_splits 落的验证集")
    n_blocks = len(man["plan"]["val_blocks"])
    ds_val_blocks = []
    for i in range(n_blocks):
        idx = np.flatnonzero(blk == i)
        if idx.size == 0:
            raise ValueError(f"valid.parquet 里验证块 {i} 没有行（manifest 声明了 {n_blocks} 个块）")
        ds_val_blocks.append(ds_val.select(idx))
    for name, ds in (("train", ds_train), ("cal", ds_cal)):
        if ds is not None and len(ds) == 0:
            raise ValueError(f"{d / (name + '.parquet')} 为空")
    return ds_train, ds_cal, ds_val, ds_val_blocks, man


def build_training_args(cfg: dict, out_dir, max_steps=None) -> TrainingArguments:
    """max_steps：None = 用 finetune.schedule.max_steps；0（或配置里 null）= 不限步数，按 epochs_max 训。"""
    s = cfg["finetune"]["schedule"]
    steps = s.get("max_steps") if max_steps is None else max_steps
    cuda = torch.cuda.is_available()
    # transformers >= 4.46 删掉了 evaluation_strategy，改名 eval_strategy；按 TrainingArguments.__init__
    # 实际签名探测该传哪个，不硬编码某个 transformers 版本的键名。
    eval_strategy_kw = ("eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
                        else "evaluation_strategy")
    return TrainingArguments(
        output_dir=str(Path(out_dir) / "hf"), per_device_train_batch_size=int(s["batch_size"]),
        num_train_epochs=float(s["epochs_max"]), learning_rate=float(s["lr"]), weight_decay=float(s["weight_decay"]),
        max_grad_norm=float(s["grad_clip"]), bf16=(str(s["precision"]) == "bf16" and cuda), use_cpu=not cuda,
        save_strategy="no", logging_steps=int(s.get("logging_steps", 20)), report_to=[],
        remove_unused_columns=False, disable_tqdm=not bool(s.get("progress_bar", True)), dataloader_pin_memory=cuda,
        max_steps=(int(steps) if steps else -1), seed=int((cfg["run"].get("seeds") or [0])[0]),
        # 多卡时主进程转换数据、打验证分期间其余进程等在集合通信上；默认 30 分钟超时不够保险
        ddp_timeout=7200,
        **{eval_strategy_kw: "no"})


def _station_table(zs: pd.DataFrame, lo: pd.DataFrame, routing: dict, detail: pd.DataFrame | None = None) -> pd.DataFrame:
    t = zs.rename(columns={"accuracy_pct": "acc_zeroshot"})[["station_id", "station", "acc_zeroshot"]].merge(
        lo.rename(columns={"accuracy_pct": "acc_lora"})[["station_id", "acc_lora"]], on="station_id", how="outer")
    t["route"] = t["station_id"].astype(int).map(routing).fillna("zeroshot")
    if detail is not None:
        t = t.merge(detail.drop(columns=["route"]), on="station_id", how="left")
    return t.sort_values("station_id").reset_index(drop=True)


SAMPLING = ("random", "station_day")
WEIGHTS_FILE, CONFIG_FILE = "model.safetensors", "model_config.yaml"


class _M6Trainer(Trainer):
    """finetune.schedule.sampling=station_day 时换成按（站, 起报日）等概率抽行的采样器；random 用 HF 默认的均匀随机采样。"""

    def __init__(self, *a, sampling: str = "random", station_col: str = "station_id", ts_col: str = "timestamp_win", **kw):
        super().__init__(*a, **kw)
        self._sampling, self._station_col, self._ts_col = sampling, station_col, ts_col

    def _get_train_sampler(self):
        if self._sampling == "station_day":
            return StationDaySampler(self.train_dataset, self._station_col, self._ts_col, seed=self.args.seed)
        return super()._get_train_sampler()


def _on_days(daily: pd.DataFrame, parity) -> pd.DataFrame:
    return daily if parity is None or not len(daily) else daily[day_parity(daily["date"]) == int(parity)]


def train_m6(config_path, parquet_path=None, shift_days: int = 0, out_dir=None, max_steps=None,
             splits_dir=None) -> dict:
    """两种喂数方式二选一：parquet_path（原始行表，按 finetune.split 在内存里切）或 splits_dir
    （pipelines.time_splits 落好的 train/valid/cal.parquet + split_manifest.json，与 M1–M4 共用同一份切分）。
    calibration.enabled=false：不做 conformal——不要校准段，早停与逐站路由按未校准分位上的点预测
    （decision.point 配置）打分，不写 calibration.parquet，阶梯报表三行改叫 *_uncalibrated。
    max_steps：覆盖 finetune.schedule.max_steps，0 = 不限步数。多卡时非主进程训完即返回 {}。"""
    if (parquet_path is None) == (splits_dir is None):
        raise ValueError("parquet_path 与 splits_dir 二选一")
    if out_dir is not None:
        config_path = deep_merge(_raw_override(config_path), {"run": {"out_dir": str(out_dir)}})
    cfg = load_config(config_path)
    need_cal = bool(cfg["calibration"]["enabled"])
    if splits_dir is not None:
        if shift_days:
            raise ValueError("splits_dir 模式下 shift_days 无效：整体后移请在 pipelines.time_splits 切分时用 --shift-days")
        _check_presplit_dir(splits_dir, need_cal)           # 先于模型构造：缺文件不必先加载底座
    ft = cfg["finetune"]
    if not ft["enabled"]:
        raise ValueError("train_m6 需要 finetune.enabled=true")
    sch, guards = ft["schedule"], ft["guards"]
    sampling = str(sch.get("sampling") or "random")
    if sampling not in SAMPLING:
        raise ValueError(f"finetune.schedule.sampling={sampling!r} 不在 {SAMPLING}")
    out = Path(cfg["run"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    report_dir = Path(cfg["eval"]["report_dir"]); report_dir.mkdir(parents=True, exist_ok=True)
    try:
        from .transformer_models import TransformersAdaptedM6Model
    except ImportError:
        from transformer_models import TransformersAdaptedM6Model
    # 训练模型：原始分位（校准在回调里拟合）；inference_path 强制 torch；lora_path/routing_path 指向
    # 从不写入的哨兵路径，保证构造期永远是零样本、无路由起点（不会读回同一 out_dir 里上一轮训练的产物）。
    fresh = out / ".fresh"
    train_cfg = {**cfg, "calibration": {**cfg["calibration"], "enabled": False},
                 "finetune": {**ft, "inference_path": "torch",
                              "lora_path": str(fresh / WEIGHTS_FILE),
                              "routing_path": str(fresh / "station_routing.json")}}
    args = build_training_args(train_cfg, out, max_steps)
    _ = args.device                  # 多进程时在这里建进程组、把本进程绑到自己的 GPU；须先于模型构造
    if args.n_gpu > 1:
        raise ValueError(f"单进程看到 {args.n_gpu} 张卡：HF 会退到 DataParallel，M6 的 dict batch 不支持。"
                         f"多卡请用 train_m6.sh 或 accelerate launch / torchrun 起多进程；单卡请设 CUDA_VISIBLE_DEVICES")
    main = _dist.is_main()
    model = TransformersAdaptedM6Model(train_cfg)
    if model.backbone is None:
        model._ensure_backbone(model._resolve_device())
    scratch = out / ".data"
    station_col, ts_col = str(cfg["dataset"]["station_id_col"]), str(cfg["dataset"]["ts_col"])

    def _load(reuse: bool):
        if splits_dir is not None:
            ds_train, ds_cal, ds_val, ds_val_blocks, man = load_presplit(splits_dir, cfg, model.station_meta, scratch, reuse=reuse)
            plan_dict, coverage_note = man["plan"], man.get("coverage_note")
            info = {"generated_by": "forecast_backbones.m6.train", "source": "pipelines.time_splits",
                    "splits_dir": str(Path(splits_dir).resolve()), "plan": plan_dict,
                    "shift_days": int(man.get("shift_days", 0))}
        else:
            ds_all = _dataset_with_station_id(_expand_paths(parquet_path), cfg, model.station_meta, scratch, reuse=reuse)
            ds_train, ds_cal, ds_val, ds_val_blocks, plan = make_splits(
                ds_all, ft["split"], shift_days, L=int(cfg["timesfm3"]["context_len"]), H=int(cfg["timesfm3"]["pred_len"]),
                step_min=int(cfg["dataset"]["step_min"]), station_col=station_col, need_cal=need_cal)
            plan_dict, coverage_note = plan.to_dict(), (plan.reference if isinstance(plan, StationSplitPlan) else plan).coverage_note
            # generated_by 放第一个键：同名的 split_manifest.json 也可能由 pipelines.time_splits 的 CLI 落，
            # 两份 manifest 的字段集合不同，读的人先看这一行就知道该按哪套字段解析。
            info = {"generated_by": "forecast_backbones.m6.train", "plan": plan_dict, "shift_days": int(shift_days)}
        info["counts"] = {"train": len(ds_train), "cal": 0 if ds_cal is None else len(ds_cal), "val": len(ds_val),
                          "val_blocks": [len(b) for b in ds_val_blocks]}
        return ds_train, ds_cal, ds_val, ds_val_blocks, info, coverage_note

    # 站号要转换时主进程先把转换好的 parquet 写进 scratch，其余进程等它写完直接读，不重复转换
    if main:
        loaded = _load(reuse=False)
    _dist.barrier()
    if not main:
        loaded = _load(reuse=True)
    ds_train, ds_cal, ds_val, ds_val_blocks, split_info, coverage_note = loaded
    plan_dict = split_info["plan"]
    if main:
        (out / "split_manifest.json").write_text(json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8")
        # 推理配置：合并了默认值、未解析 ${run.out_dir} 的配置，推理时把 run.out_dir 指向产物目录即可整体搬走
        dump_config(deep_merge(DEFAULT_CFG, _raw_override(config_path)), out / CONFIG_FILE)
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    sel_parity, route_parity = (0, 1) if bool(guards.get("split_val_days", True)) else (None, None)
    eval_every = sch.get("eval_every_steps")
    if main:
        world, bs = _dist.world_size(), int(args.per_device_train_batch_size)
        if not need_cal:
            print("[m6-lora] calibration.enabled=false：不做 conformal，早停与逐站路由按未校准分位上的点预测打分")
        print(f"[m6-lora] train={len(ds_train)} cal={0 if ds_cal is None else len(ds_cal)} val={len(ds_val)} rows in {len(ds_val_blocks)} blocks "
              f"{plan_dict['val_blocks']}; trainable={trainable_summary(model.backbone.model)}")
        budget = f"{args.max_steps} 步" if args.max_steps > 0 else f"不限步数（{sch['epochs_max']} 个 epoch）"
        print(f"[m6-lora] {world} 个进程 × 每卡 batch {bs} = 每步 {world * bs} 行；预算 {budget}；"
              f"验证{f'每 {eval_every} 步一次' if eval_every else '每个 epoch 末一次'}，patience={ft['early_stop']['patience']} 次；"
              f"训练行采样 {sampling}")
        if sel_parity is not None:
            print("[m6-lora] 早停选权重只用偶数起报日，逐站路由只用奇数起报日（guards.split_val_days）")
        if coverage_note:
            print(f"[m6-lora] 提示：{coverage_note}")
    best_path = out / WEIGHTS_FILE
    cb = NanwangEvalCallback(model, ds_cal, ds_val_blocks, collate, train_cfg, best_path=best_path,
                             patience=int(ft["early_stop"]["patience"]), log_path=out / "train_log.json",
                             eval_every_steps=eval_every, day_parity=sel_parity)
    trainer = _M6Trainer(model=model, args=args, data_collator=collate, train_dataset=ds_train, callbacks=[cb],
                         sampling=sampling, station_col=station_col, ts_col=ts_col)
    trainer.train()
    if not main:
        return {}
    if cb.best_epoch is None or not best_path.exists():
        raise ValueError(
            f"训练全程未产出任何可用的最优 LoRA 权重：验证块 {plan_dict['val_blocks']} 上的南网短期打分"
            f"（NanwangEvalCallback 每次验证对每块各调一次 score_split，取块均值）自始至终都是非有限值"
            f"（没有一块能凑出完整可评分的一天），cb.best_epoch={cb.best_epoch!r}，{best_path} 不存在。"
            f"请检查各验证块是否覆盖足够多的完整天。")
    # 最优权重
    load_lora_state_dict(model.backbone.model, load_lora_weights(best_path))
    model.lora_loaded = True
    # 阶梯 4：零样本 vs LoRA（各自校准）→ 逐站路由 → 路由模型重拟合校准表
    model.station_routing = None
    set_lora_enabled(model.backbone.model, False); zs_score, zs_tab, _, zs_daily = score_split(model, ds_cal, ds_val, collate, train_cfg)
    set_lora_enabled(model.backbone.model, True);  lo_score, lo_tab, _, lo_daily = score_split(model, ds_cal, ds_val, collate, train_cfg)
    stations = pd.concat([zs_tab["station_id"], lo_tab["station_id"]])
    boot = guards.get("bootstrap") or {}
    rule = {"route_day_parity": route_parity, "min_gain_pct": float(guards.get("min_gain_pct", 0.0)),
            "n_boot": int(boot.get("n", 0)), "alpha": float(boot.get("alpha", 0.1)),
            "min_days": int(boot.get("min_days", 1)), "seed": int(boot.get("seed", 0))}
    detail = None
    if bool(guards["per_station_fallback_to_zeroshot"]):
        routing, detail = route_by_days(_on_days(zs_daily, route_parity), _on_days(lo_daily, route_parity), stations,
                                        min_gain_pct=rule["min_gain_pct"], n_boot=rule["n_boot"], alpha=rule["alpha"],
                                        min_days=rule["min_days"], seed=rule["seed"])
    else:
        routing = {int(s): "lora" for s in stations}
    save_routing(out / "station_routing.json", routing)
    model.station_routing = routing
    routed_score, _, calib, _ = score_split(model, ds_cal, ds_val, collate, train_cfg)
    if calib is not None:
        calib.save(cfg["calibration"]["table_path"])
    res_val = evaluate_model(model, ds_val, collate, train_cfg)
    res_val.to_parquet(out / "result_val_lora.parquet", index=False)
    n_lora = sum(v == "lora" for v in routing.values()); n_zs = len(routing) - n_lora
    tag = "calibrated_pstar" if need_cal else "uncalibrated"
    ladder = pd.DataFrame([{"variant": f"zeroshot_{tag}", "nanwang_short_D+1": zs_score, "n_stations_lora": 0, "n_stations_zeroshot": len(routing)},
                           {"variant": f"lora_{tag}", "nanwang_short_D+1": lo_score, "n_stations_lora": len(routing), "n_stations_zeroshot": 0},
                           {"variant": f"routed_{tag}", "nanwang_short_D+1": routed_score, "n_stations_lora": n_lora, "n_stations_zeroshot": n_zs}])
    ladder.to_csv(report_dir / "ladder_lora.csv", index=False)
    station_tab = _station_table(zs_tab, lo_tab, routing, detail)
    station_tab.to_csv(report_dir / "station_ladder.csv", index=False)
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    log.update({"final": ladder.to_dict("records"), "routing": {str(k): v for k, v in routing.items()},
                "routing_rule": rule, "routing_detail": None if detail is None else detail.to_dict("records"),
                "shift_days": int(shift_days), "split": split_info})
    (out / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(ladder.to_string(index=False)); print(station_tab.to_string(index=False))
    print(f"[m6-lora] 产物：{best_path} / {out / CONFIG_FILE} / {out / 'station_routing.json'} / "
          + (f"{cfg['calibration']['table_path']} / " if calib is not None else "")
          + f"{out / 'split_manifest.json'} / {report_dir}")
    return {"ladder": ladder, "routing": routing, "routing_detail": detail, "history": cb.history, "best": cb.best,
            "split": split_info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="M6 路线 B：LoRA 微调 + 站级回退 + 阶梯 4 报表")
    ap.add_argument("--config", default="forecast_backbones/m6/config/configs.yaml")
    ap.add_argument("--parquet", default=None, help="原始行表（可通配符），按 config 的 finetune.split 在内存里切")
    ap.add_argument("--splits-dir", default=None, help="pipelines.time_splits 落好的目录（train/valid/cal.parquet + manifest）")
    ap.add_argument("--out", default=None); ap.add_argument("--shift-days", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="覆盖 finetune.schedule.max_steps；0 = 不限步数（按 epochs_max）")
    a = ap.parse_args()
    train_m6(a.config, a.parquet, shift_days=a.shift_days, out_dir=a.out, max_steps=a.max_steps, splits_dir=a.splits_dir)
