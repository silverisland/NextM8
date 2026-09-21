# -*- coding: utf-8 -*-
# forecast_backbones/m6/train.py
"""M6 路线 B 训练入口（spec §5.2）：
  1. finetune.split 走 SplitPlan 切 train / cal / 多个验证块（块前后 purge/embargo 剔训练行；shift_days 整体后移做滚动折）
  2. HF Trainer：损失走模型 compute_loss（pinball 9 档 + 夜间/缺失掩码），LoRA + output_head 可训
  3. NanwangEvalCallback：epoch 0 零样本基线，每 epoch 对每个验证块 cal→δ、块→校准→p*→南网短期，块均值作早停指标 + 最优权重
  4. 训后：加载最优 LoRA；零样本 vs LoRA 逐站比较（各自校准）→ station_routing.json
  5. 路由后的模型重拟合校准表 → calibration.parquet（与推理时一致）
  6. 报表：reports/ladder_lora.csv（阶梯第 4 步）、reports/station_ladder.csv；状态：lora_weights.pt、train_log.json、split_manifest.json
用 HF Trainer 而不是 pipelines.transformer_trainer.train_model：后者 import 的 transformer_utils 本仓库没有，
语义（forward 返 loss、collate 出 batch_dict/labels）在这里逐条保持一致。

训练模型的构造必须是真正的零样本起点：把 finetune.lora_path / routing_path 都改指向 run.out_dir 下一个
从不写入的 `.fresh/` 哨兵路径，而不是直接用 cfg 里解析出的真实 lora_weights.pt / station_routing.json——
后两者是本函数自己的输出路径，若同一 out_dir 曾跑过一次训练，构造期会静默把上一轮的 LoRA 权重和路由表
读回来，epoch 0 记录的"零样本基线"和训练期的逐站前向就都不再是零样本/无路由的了。哨兵路径只影响构造期
的装载判断，不影响本函数最终把新产物写到 out/lora_weights.pt、out/station_routing.json 的行为。"""
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
    from .config import load_config, deep_merge
    from .transformer_trainer import _raw_override, evaluate_model
    from .nanwang_callback import NanwangEvalCallback, score_split
    from .routing import decide_routing, save_routing
    from .lora import load_lora_state_dict, set_lora_enabled, trainable_summary
    from .features import attach_station_id
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from config import load_config, deep_merge
    from transformer_trainer import _raw_override, evaluate_model
    from nanwang_callback import NanwangEvalCallback, score_split
    from routing import decide_routing, save_routing
    from lora import load_lora_state_dict, set_lora_enabled, trainable_summary
    from features import attach_station_id
try:
    from ...pipelines.transformer_dataset import UnifiedDataset
    from ...pipelines.time_splits import (SplitPlan, StationSplitPlan, assign_rows, verify_rows, has_cal,
                                          load_test_starts, station_keys)
except ImportError:
    from pipelines.transformer_dataset import UnifiedDataset
    from pipelines.time_splits import (SplitPlan, StationSplitPlan, assign_rows, verify_rows, has_cal,
                                       load_test_starts, station_keys)


def build_plan(split: dict, shift_days: int = 0, L: int = 672, H: int = 192, step_min: int = 15):
    """split.test_start 是日期 → SplitPlan；是 {station: 日期} 映射，或 split.test_parquets 给了测试集文件
    （每站 test_start = 该站在测试集里最早起报时刻的整天）→ StationSplitPlan。两者都给则报错。"""
    t0, files = split.get("test_start"), split.get("test_parquets")
    if files:
        if t0 is not None:
            raise ValueError("finetune.split 同时给了 test_start 与 test_parquets：删掉其一")
        starts = load_test_starts(list(files) if isinstance(files, (list, tuple)) else [files],
                                  ts_col=split.get("ts_col", "timestamp_win"), station_col=split.get("station_col"))
        return StationSplitPlan.from_dict(split, starts, shift_days=shift_days, L=L, H=H, step_min=step_min)
    if isinstance(t0, dict):
        return StationSplitPlan.from_dict(split, t0, shift_days=shift_days, L=L, H=H, step_min=step_min)
    return SplitPlan.from_dict(split, shift_days=shift_days, L=L, H=H, step_min=step_min)


def make_splits(ds_all, split: dict, shift_days: int = 0, L: int = 672, H: int = 192, step_min: int = 15,
                station_col: str = "station_id"):
    """按 SplitPlan / StationSplitPlan 在起报时间轴上切 train / cal / val（全部验证块合并）/ val_blocks（逐块列表）。
    train 行已剔除每个验证块前 purge、后 purge+embargo，以及校准段前 purge 之后的一切；返回末项为 plan。
    按站切分时某站容不下的验证块该站不出行，其他站照常；块序号是全局的。"""
    plan = build_plan(split, shift_days=shift_days, L=L, H=H, step_min=step_min)
    if not has_cal(plan):
        raise ValueError("M6 训练需要校准段：finetune.split.cal_days 须 ≥ 1")
    ts = ds_all._ts_as_datetime64(ds_all.timestamp_cols[0])
    station = None
    if isinstance(plan, StationSplitPlan):
        known = set(ds_all.timestamp_cols) | set(ds_all.entity_cols) | set(ds_all.feature_cols) | set(ds_all.label_cols)
        if station_col not in known:
            raise ValueError(f"按站切分需要站标识列 {station_col!r}，数据里没有（列：{sorted(known)[:12]}…）")
        station = station_keys(ds_all.column(station_col))
    role, block = assign_rows(plan, ts, station)

    def _sel(mask, name):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError(f"切分后 {name} 段为空（plan={plan.to_dict()}）")
        return ds_all.select(idx)

    is_val = role == "val"
    ds_train, ds_cal, ds_val_all = _sel(role == "train", "train"), _sel(role == "cal", "cal"), _sel(is_val, "val")
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


def _dataset_with_station_id(paths, cfg: dict, station_meta, scratch_dir) -> UnifiedDataset:
    """按 M6 口径建 UnifiedDataset（arrow 后端，make_splits/select 需要）。数据没有 station_id 列而只有站名列时，
    按 station_meta 映射补上并落到 scratch_dir 下的临时 parquet 再装载（arrow 后端不能就地加列）。"""
    ds_cfg = cfg["dataset"]; id_col = str(ds_cfg["station_id_col"])
    import pyarrow.parquet as pq
    cols = pq.read_schema(paths[0]).names
    if id_col not in cols:
        df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        df = attach_station_id(df, station_meta, id_col=id_col, name_col="station")
        scratch = Path(scratch_dir); scratch.mkdir(parents=True, exist_ok=True)
        tmp = scratch / f"{Path(paths[0]).stem}_with_{id_col}.parquet"
        df.to_parquet(tmp, index=False)
        print(f"[m6] 数据无 {id_col!r} 列：按 station_meta 从站名映射补上，写到 {tmp}")
        paths = [str(tmp)]
    return UnifiedDataset(paths if len(paths) > 1 else paths[0], timestamp_cols=[ds_cfg["ts_col"]], entity_cols=[],
                          label_cols=[ds_cfg["label_col"]], timestamp_output="epoch_s", epoch_unit="s",
                          for_hf_trainer=True)


def _check_presplit_dir(splits_dir) -> Path:
    d = Path(splits_dir)
    need = ["train.parquet", "valid.parquet", "cal.parquet", "split_manifest.json"]
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"{d} 缺 {missing}：M6 训练需要校准段，用 pipelines.time_splits 切分时 cal_days 须 ≥ 1")
    return d


def load_presplit(splits_dir, cfg: dict, station_meta, scratch_dir):
    """读 pipelines.time_splits 落好的目录：train.parquet / valid.parquet（含 split_block）/ cal.parquet /
    split_manifest.json。返回 (ds_train, ds_cal, ds_val_all, ds_val_blocks, manifest)。块序号沿用 manifest 的全局序号。"""
    d = _check_presplit_dir(splits_dir)
    man = json.loads((d / "split_manifest.json").read_text(encoding="utf-8"))
    ds_train = _dataset_with_station_id([str(d / "train.parquet")], cfg, station_meta, scratch_dir)
    ds_cal = _dataset_with_station_id([str(d / "cal.parquet")], cfg, station_meta, scratch_dir)
    ds_val = _dataset_with_station_id([str(d / "valid.parquet")], cfg, station_meta, scratch_dir)
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
        if len(ds) == 0:
            raise ValueError(f"{d / (name + '.parquet')} 为空")
    return ds_train, ds_cal, ds_val, ds_val_blocks, man


def build_training_args(cfg: dict, out_dir, max_steps=None) -> TrainingArguments:
    s = cfg["finetune"]["schedule"]
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
        remove_unused_columns=False, disable_tqdm=True, dataloader_pin_memory=cuda,
        max_steps=(-1 if max_steps is None else int(max_steps)), seed=int((cfg["run"].get("seeds") or [0])[0]),
        **{eval_strategy_kw: "no"})


def _station_table(zs: pd.DataFrame, lo: pd.DataFrame, routing: dict) -> pd.DataFrame:
    t = zs.rename(columns={"accuracy_pct": "acc_zeroshot"})[["station_id", "station", "acc_zeroshot"]].merge(
        lo.rename(columns={"accuracy_pct": "acc_lora"})[["station_id", "acc_lora"]], on="station_id", how="outer")
    t["route"] = t["station_id"].astype(int).map(routing).fillna("zeroshot")
    return t.sort_values("station_id").reset_index(drop=True)


def train_m6(config_path, parquet_path=None, shift_days: int = 0, out_dir=None, max_steps=None,
             splits_dir=None) -> dict:
    """两种喂数方式二选一：parquet_path（原始行表，按 finetune.split 在内存里切）或 splits_dir
    （pipelines.time_splits 落好的 train/valid/cal.parquet + split_manifest.json，与 M1–M4 共用同一份切分）。"""
    if (parquet_path is None) == (splits_dir is None):
        raise ValueError("parquet_path 与 splits_dir 二选一")
    if splits_dir is not None:
        if shift_days:
            raise ValueError("splits_dir 模式下 shift_days 无效：整体后移请在 pipelines.time_splits 切分时用 --shift-days")
        _check_presplit_dir(splits_dir)           # 先于模型构造：缺文件不必先加载底座
    if out_dir is not None:
        config_path = deep_merge(_raw_override(config_path), {"run": {"out_dir": str(out_dir)}})
    cfg = load_config(config_path)
    ft = cfg["finetune"]
    if not ft["enabled"]:
        raise ValueError("train_m6 需要 finetune.enabled=true")
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
                              "lora_path": str(fresh / "lora_weights.pt"),
                              "routing_path": str(fresh / "station_routing.json")}}
    model = TransformersAdaptedM6Model(train_cfg)
    if model.backbone is None:
        model._ensure_backbone(model._resolve_device())
    scratch = out / ".data"
    if splits_dir is not None:
        ds_train, ds_cal, ds_val, ds_val_blocks, man = load_presplit(splits_dir, cfg, model.station_meta, scratch)
        plan_dict, coverage_note = man["plan"], man.get("coverage_note")
        split_info = {"generated_by": "forecast_backbones.m6.train", "source": "pipelines.time_splits",
                      "splits_dir": str(Path(splits_dir).resolve()), "plan": plan_dict,
                      "shift_days": int(man.get("shift_days", 0)),
                      "counts": {"train": len(ds_train), "cal": len(ds_cal), "val": len(ds_val),
                                 "val_blocks": [len(b) for b in ds_val_blocks]}}
    else:
        ds_all = _dataset_with_station_id(_expand_paths(parquet_path), cfg, model.station_meta, scratch)
        ds_train, ds_cal, ds_val, ds_val_blocks, plan = make_splits(
            ds_all, ft["split"], shift_days, L=int(cfg["timesfm3"]["context_len"]), H=int(cfg["timesfm3"]["pred_len"]),
            step_min=int(cfg["dataset"]["step_min"]), station_col=str(cfg["dataset"]["station_id_col"]))
        plan_dict, coverage_note = plan.to_dict(), (plan.reference if isinstance(plan, StationSplitPlan) else plan).coverage_note
        # generated_by 放第一个键：同名的 split_manifest.json 也可能由 pipelines.time_splits 的 CLI 落，
        # 两份 manifest 的字段集合不同，读的人先看这一行就知道该按哪套字段解析。
        split_info = {"generated_by": "forecast_backbones.m6.train",
                      "plan": plan_dict, "shift_days": int(shift_days),
                      "counts": {"train": len(ds_train), "cal": len(ds_cal), "val": len(ds_val),
                                 "val_blocks": [len(b) for b in ds_val_blocks]}}
    (out / "split_manifest.json").write_text(json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8")
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    print(f"[m6-lora] train={len(ds_train)} cal={len(ds_cal)} val={len(ds_val)} rows in {len(ds_val_blocks)} blocks "
          f"{plan_dict['val_blocks']}; trainable={trainable_summary(model.backbone.model)}")
    if coverage_note:
        print(f"[m6-lora] 提示：{coverage_note}")
    best_path = out / "lora_weights.pt"
    cb = NanwangEvalCallback(model, ds_cal, ds_val_blocks, collate, train_cfg, best_path=best_path,
                             patience=int(ft["early_stop"]["patience"]), log_path=out / "train_log.json")
    trainer = Trainer(model=model, args=build_training_args(train_cfg, out, max_steps), data_collator=collate,
                      train_dataset=ds_train, callbacks=[cb])
    trainer.train()
    if cb.best_epoch is None or not best_path.exists():
        raise ValueError(
            f"训练全程未产出任何可用的最优 LoRA 权重：验证块 {plan_dict['val_blocks']} 上的南网短期打分"
            f"（NanwangEvalCallback 每个 epoch 对每块各调一次 score_split，取块均值）自始至终都是非有限值"
            f"（没有一块能凑出完整可评分的一天），cb.best_epoch={cb.best_epoch!r}，{best_path} 不存在。"
            f"请检查各验证块是否覆盖足够多的完整天。")
    # 最优权重
    load_lora_state_dict(model.backbone.model, torch.load(best_path, map_location="cpu"))
    model.lora_loaded = True
    # 阶梯 4：零样本 vs LoRA（各自校准）→ 逐站路由 → 路由模型重拟合校准表
    model.station_routing = None
    set_lora_enabled(model.backbone.model, False); zs_score, zs_tab, _ = score_split(model, ds_cal, ds_val, collate, train_cfg)
    set_lora_enabled(model.backbone.model, True);  lo_score, lo_tab, _ = score_split(model, ds_cal, ds_val, collate, train_cfg)
    guards = ft["guards"]
    if bool(guards["per_station_fallback_to_zeroshot"]):
        routing = decide_routing(_station_table(zs_tab, lo_tab, {})[["station_id", "acc_zeroshot", "acc_lora"]],
                                 float(guards.get("min_gain_pct", 0.0)))
    else:
        routing = {int(s): "lora" for s in zs_tab["station_id"]}
    save_routing(out / "station_routing.json", routing)
    model.station_routing = routing
    routed_score, _, calib = score_split(model, ds_cal, ds_val, collate, train_cfg)
    calib.save(cfg["calibration"]["table_path"])
    res_val = evaluate_model(model, ds_val, collate, train_cfg)
    res_val.to_parquet(out / "result_val_lora.parquet", index=False)
    n_lora = sum(v == "lora" for v in routing.values()); n_zs = len(routing) - n_lora
    ladder = pd.DataFrame([{"variant": "zeroshot_calibrated_pstar", "nanwang_short_D+1": zs_score, "n_stations_lora": 0, "n_stations_zeroshot": len(routing)},
                           {"variant": "lora_calibrated_pstar", "nanwang_short_D+1": lo_score, "n_stations_lora": len(routing), "n_stations_zeroshot": 0},
                           {"variant": "routed_calibrated_pstar", "nanwang_short_D+1": routed_score, "n_stations_lora": n_lora, "n_stations_zeroshot": n_zs}])
    ladder.to_csv(report_dir / "ladder_lora.csv", index=False)
    station_tab = _station_table(zs_tab, lo_tab, routing)
    station_tab.to_csv(report_dir / "station_ladder.csv", index=False)
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    log.update({"final": ladder.to_dict("records"), "routing": {str(k): v for k, v in routing.items()},
                "shift_days": int(shift_days), "split": split_info})
    (out / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(ladder.to_string(index=False)); print(station_tab.to_string(index=False))
    print(f"[m6-lora] 产物：{best_path} / {out / 'station_routing.json'} / {cfg['calibration']['table_path']} / "
          f"{out / 'split_manifest.json'} / {report_dir}")
    return {"ladder": ladder, "routing": routing, "history": cb.history, "best": cb.best, "split": split_info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="M6 路线 B：LoRA 微调 + 站级回退 + 阶梯 4 报表")
    ap.add_argument("--config", default="forecast_backbones/m6/config/configs.yaml")
    ap.add_argument("--parquet", default=None, help="原始行表（可通配符），按 config 的 finetune.split 在内存里切")
    ap.add_argument("--splits-dir", default=None, help="pipelines.time_splits 落好的目录（train/valid/cal.parquet + manifest）")
    ap.add_argument("--out", default=None); ap.add_argument("--shift-days", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    train_m6(a.config, a.parquet, shift_days=a.shift_days, out_dir=a.out, max_steps=a.max_steps, splits_dir=a.splits_dir)
