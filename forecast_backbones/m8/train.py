# -*- coding: utf-8 -*-
# forecast_backbones/m8/train.py
"""M8 训练入口：
  1. 读 pipelines.time_splits 落好的切分目录（train.parquet / valid.parquet / split_manifest.json；不需要 cal.parquet）
  2. HF Trainer：损失在模型 forward 里（pinball，未来段 + CPM 掩码段，掩夜间/缺测）；预算 train.schedule.max_steps 个优化器步
  3. EvalCallback：每 eval_every_steps 步在验证集上打分（nanwang_after_pstar：每站每天 10:00 前最晚一行 → 南网短期 D+1；
     或 nrmse：整点起报行 → 192 步容量归一 RMSE%），patience 按验证次数计，最优权重存 model.safetensors（完整 state_dict）
  4. 训后：加载最优权重 → 验证集推理 → result_val.parquet（默认只存整点起报行）→ reports/summary.csv、reports/station_table.csv
  5. --refit（配方选定后给最终模型）：train ∪ valid 全当训练行，不验证、不早停，固定 max_steps 步后存最终权重；summary 里没有验证分数
  7. --save-every N / train.schedule.save_every（2026-10-19）：每 N 个优化步存一份完整 checkpoint 到 checkpoints/step_XXXXX/
     （model.safetensors + model_config.yaml，开了 EMA 存 EMA 权重），可直接给线上推理脚本加载。全量训练没有验证集、不能早停时，
     训完用推理脚本在训练期之后的时段上逐个评测、挑最优步 = 事后早停
  6. --train-parquet（2026-10-14）：训练 parquet 已经切好时直接吃文件夹 / 通配符 / 文件列表，不走 time_splits、不切块、不剔除任何行、
     没有验证集（等于 refit）。与 --splits-dir 二选一
多卡：accelerate launch / torchrun 起多进程，每进程各持一份模型吃不同的批；打分、存盘、报表只在主进程做。
训练模型的构造必须从零起步：train.weights_path 改指向 out_dir 下从不写入的 .fresh/ 哨兵路径。"""
from __future__ import annotations


import argparse
import contextlib
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainerCallback

try:
    from .config import DEFAULT_CFG, load_config, deep_merge, dump_config, raw_override, future_only_cols
    from .features import station_cap_mw
    from .transformer_trainer import evaluate_model, build_training_args
    from .transformer_models import TransformersAdaptedM8Model, save_weights, load_weights
    from .data import _dataset_with_station_id, _check_presplit_dir, report_inputs
    from .rows import SHORT_ISSUE_TIME, scored_row_index, issue_grid_row_index, StationDaySampler, row_stations
    from . import _dist
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m8.config import DEFAULT_CFG, load_config, deep_merge, dump_config, raw_override, future_only_cols
    from forecast_backbones.m8.features import station_cap_mw
    from forecast_backbones.m8.transformer_trainer import evaluate_model, build_training_args
    from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model, save_weights, load_weights
    from forecast_backbones.m8.data import _dataset_with_station_id, _check_presplit_dir, report_inputs
    from forecast_backbones.m8.rows import SHORT_ISSUE_TIME, scored_row_index, issue_grid_row_index, StationDaySampler, row_stations
    from forecast_backbones.m8 import _dist
try:
    from ...pipelines.transformer_dataset import UnifiedDataset
    from ...pipelines.nanwang_metric import NanwangMetric
except ImportError:
    from pipelines.transformer_dataset import UnifiedDataset
    from pipelines.nanwang_metric import NanwangMetric

WEIGHTS_FILE, CONFIG_FILE = "model.safetensors", "model_config.yaml"
EARLY_STOP_METRICS = {"nanwang_after_pstar": True, "nrmse": False, "nanwang_ultra_proxy": True}   # 指标 → 是否越高越好


from pipelines.forecasting.metrics import seen_unseen_scores as seen_unseen_scores
from pipelines.forecasting.metrics import nrmse_pct as nrmse_pct
from pipelines.forecasting.training import WeightEma as WeightEma
from pipelines.forecasting.training import EmaCallback as EmaCallback
from pipelines.forecasting.training import CheckpointCallback as CheckpointCallback
from pipelines.forecasting.training import EvaluationCallback
from pipelines.forecasting.training import ForecastTrainer as _M8Trainer
from pipelines.forecasting.data import load_presplit as load_presplit
from pipelines.forecasting.data import expand_parquets as expand_parquets
from pipelines.forecasting.data import load_full as load_full
from pipelines.forecasting.data import _refit_rows as _refit_rows


def ultra_proxy_pct(res: pd.DataFrame) -> float:
    """超短期代理分（整点起报行）：每个目标时刻取能拿到的 lead（整点行只有 1h/2h/3h/4h 四个），
    100·(1 − mean |y−p| / max(y, 0.2·cap))，负值 cap 到 0。与南网超短期同一相对误差与日窗，
    区别是 lead 子集、按目标时刻平均而不是 /96；用于验证期选 checkpoint，不是正式超短期分（正式分要 15 分钟起报行，走 --full）。"""
    if res is None or res.empty:
        return float("nan")
    nm = NanwangMetric()
    df = nm._prep(res[["timestamp_win", "station", "observe_power_predict", "observe_power_future"]])
    caps = res.groupby("station")["cap"].first()
    parts = []
    for st, g in df.groupby("station", sort=True):
        s = nm._ultra_per_target(g, float(caps[st]), full_leads_only=False)
        if len(s):
            parts.append(s.to_numpy(float))
    if not parts:
        return float("nan")
    return float(max(0.0, 100.0 * (1.0 - float(np.concatenate(parts).mean()))))
TABLE_COLS = ["station_id", "station", "accuracy_pct", "n_days"]


# ------------------------------------------------------------------ 打分
def _subset(ds, idx):
    return torch.utils.data.Subset(ds, [int(i) for i in idx])


def score_nanwang(model, ds_val, collate, cfg, res: pd.DataFrame | None = None) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """每站每个起报日只推理 10:00 前最晚一行 → 南网短期 D+1 日准确率均值；返回 (均值, 逐站表, 结果表)。"""
    ds_cfg = cfg["dataset"]
    if res is None:
        idx = scored_row_index(ds_val, str(ds_cfg["station_id_col"]), str(ds_cfg["ts_col"]))
        if len(idx) == 0:
            return float("nan"), pd.DataFrame(columns=TABLE_COLS), pd.DataFrame()
        was = model.training
        try:
            res = evaluate_model(model, _subset(ds_val, idx), collate, cfg)
        finally:
            model.train(was)
    from pipelines.forecasting.metrics import score_nanwang_result
    return score_nanwang_result(model, res, cfg)


def score_nrmse(model, ds_val, collate, cfg, issue_every_min: int = 60) -> tuple[float, pd.DataFrame]:
    idx = issue_grid_row_index(ds_val, str(cfg["dataset"]["ts_col"]), issue_every_min)
    if len(idx) == 0:
        return float("nan"), pd.DataFrame()
    was = model.training
    try:
        res = evaluate_model(model, _subset(ds_val, idx), collate, cfg)
    finally:
        model.train(was)
    return nrmse_pct(res), res


# ------------------------------------------------------------------ 回调


class EvalCallback(EvaluationCallback):
    def _score(self) -> float:
        self.model._progress_desc = "[m8] 验证"
        self.last_extra = {}
        try:
            if self.metric == "nrmse":
                return score_nrmse(self.model, self.ds_val, self.collate, self.cfg, self.issue_every_min)[0]
            if self.metric == "nanwang_ultra_proxy":
                # 整点行一次推理：超短期代理分，nrmse 顺带记进 history
                nr, res_h = score_nrmse(self.model, self.ds_val, self.collate, self.cfg, self.issue_every_min)
                up = ultra_proxy_pct(res_h)
                self.last_extra = {"nanwang_ultra_proxy": up, "nrmse": nr}
                return up
            score, table, _ = score_nanwang(self.model, self.ds_val, self.collate, self.cfg)
            _, unseen, n_unseen = seen_unseen_scores(table)
            self.last_unseen = float(unseen) if n_unseen else None
            return score
        finally:
            self.model._progress_desc = None


# ------------------------------------------------------------------ 数据


def _summary_row(cfg, tag, out, model, best_step, n_evals, minutes, n_val, n_train,
                 nw=float("nan"), nr=float("nan"), seen=float("nan"), unseen=float("nan"), n_unseen=0, ultra=float("nan")) -> dict:
    """reports/summary.csv 的一行（列序固定；collect_runs 按列名取）。refit 时分数列全是 NaN。"""
    return {"tag": tag or cfg["run"].get("tag") or out.name, "nanwang_short_D+1": nw, "nrmse_pct": nr, "nanwang_ultra_proxy": ultra,
            "nanwang_seen": seen, "nanwang_unseen": unseen, "n_unseen_stations": n_unseen,
            "best_step": best_step, "n_evals": n_evals, "train_minutes": minutes,
            "params": model.n_params(), "blocks": "+".join(cfg["model"]["blocks"]) or "none",
            "patch_len": cfg["model"]["patch_len"], "d_model": cfg["model"]["d_model"],
            "target_scale": cfg["features"].get("target_scale", "cap"), "cpm": cfg["train"]["cpm"]["enabled"],
            "cov_dropout": cfg["train"].get("cov_dropout", 0.0),
            "future_only": "+".join(c for c, _ in future_only_cols(cfg["features"])[0]) or "",
            "residual": future_only_cols(cfg["features"])[1] or "",
            "ema_decay": float("nan") if cfg["train"].get("ema_decay") is None else float(cfg["train"]["ema_decay"]), "smooth_k": int(cfg["train"]["early_stop"].get("smooth_k", 1) or 1),
            "n_val_rows": n_val, "n_train_rows": n_train}


# ------------------------------------------------------------------ 主流程
def train_m8(config_path, splits_dir=None, out_dir=None, max_steps=None, tag: str | None = None, refit: bool = False,
             train_parquet=None, save_every: int | None = None) -> dict:
    """splits_dir：time_splits 落好的切分目录（有验证集，可早停）；train_parquet：现成的训练 parquet，全量训练、无验证（等于 refit）。二选一。"""
    if (splits_dir is None) == (train_parquet is None):
        raise ValueError("--splits-dir 与 --train-parquet 二选一：切分目录（带验证集）或现成的训练 parquet（全量、无验证）")
    if train_parquet is not None:
        refit = True
    if out_dir is not None:
        config_path = deep_merge(raw_override(config_path), {"run": {"out_dir": str(out_dir)}})
    cfg = load_config(config_path)
    out = Path(cfg["run"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    if out_dir is not None:
        # 调用方传了 out_dir 时报表目录跟着走（传入的若是已插值的 dict，eval.report_dir 里还是旧的 run.out_dir）
        cfg["eval"]["report_dir"] = str(out / "reports")
    report_dir = Path(cfg["eval"]["report_dir"]); report_dir.mkdir(parents=True, exist_ok=True)
    sch = cfg["train"]["schedule"]
    if save_every is not None:
        sch["save_every"] = int(save_every)
    save_every = int(sch.get("save_every") or 0)
    if save_every < 0:
        raise ValueError(f"train.schedule.save_every 须 >= 0（0 / null = 不存中间 checkpoint），得到 {save_every}")
    sampling = str(sch.get("sampling") or "random")
    if sampling not in ("random", "station_day"):
        raise ValueError(f"train.schedule.sampling={sampling!r} 不在 ('random', 'station_day')")
    if splits_dir is not None:
        _check_presplit_dir(splits_dir, need_cal=False)
    train_cfg = {**cfg, "train": {**cfg["train"], "weights_path": str(out / ".fresh" / WEIGHTS_FILE)}}
    args = build_training_args(train_cfg, out, max_steps)
    if refit and int(args.max_steps) <= 0:
        raise ValueError("全量重训（refit）没有验证集、不能早停：train.schedule.max_steps（或 --max-steps）必须 > 0，"
                         "按该配方在验证切分上各种子 best_step 的上限给（c8 调度约 5000）")
    _ = args.device                        # 多进程时在这里建进程组、绑卡；须先于模型构造
    if args.n_gpu > 1:
        raise ValueError(f"单进程看到 {args.n_gpu} 张卡：HF 会退到 DataParallel，dict batch 不支持。多卡请用 accelerate launch / torchrun")
    main = _dist.is_main()
    model = TransformersAdaptedM8Model(train_cfg)
    model.reseed(int(args.seed) * 1000 + _dist.rank())
    model.to(args.device)
    scratch = out / ".data"
    station_col, ts_col = str(cfg["dataset"]["station_id_col"]), str(cfg["dataset"]["ts_col"])
    def _load(reuse: bool):
        if train_parquet is not None:
            return load_full(train_parquet, cfg, model.station_meta, scratch, reuse=reuse)
        return load_presplit(splits_dir, cfg, model.station_meta, scratch, reuse=reuse, refit=refit)
    if main:
        loaded = _load(reuse=False)
    _dist.barrier()
    if not main:
        loaded = _load(reuse=True)
    ds_train, ds_val, man = loaded
    cov_present = report_inputs(model, list(ds_train.column_names) + [str(cfg["dataset"]["label_col"])], verbose=main)
    train_cfg["features"]["cov_present"] = cov_present
    known = model.set_known_stations(row_stations(ds_train, station_col).tolist())     # 验证/测试里名单外的站走未知站槽
    train_cfg["dataset"]["known_stations"] = known
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    eval_every = sch.get("eval_every_steps")
    if main:
        cfg_dump = deep_merge(deep_merge(DEFAULT_CFG, raw_override(config_path)),
                              {"dataset": {"known_stations": known}, "features": {"cov_present": cov_present}})
        dump_config(cfg_dump, out / CONFIG_FILE)
        if train_parquet is not None:
            print(f"[m8] 全量训练：{len(man['train_parquet'])} 个 parquet、{len(ds_train)} 行、站 {len(known)} 个，一行不删、无验证集，"
                  f"固定 {args.max_steps} 步后存最终权重")
        elif refit:
            print(f"[m8] 全量重训（refit）：训练集站 {len(known)} 个（train ∪ valid），不验证、不早停，固定 {args.max_steps} 步后存最终权重")
        else:
            n_unseen_val = len(set(row_stations(ds_val, station_col).tolist()) - set(known))
            print(f"[m8] 训练集站 {len(known)} 个；验证集里训练没见过的站 {n_unseen_val} 个（走未知站槽，报表里 seen=False）")
        (out / "split_manifest.json").write_text(json.dumps({"generated_by": "forecast_backbones.m8.train",
            "splits_dir": None if splits_dir is None else str(Path(splits_dir).resolve()),
            "train_parquet": man.get("train_parquet"), "plan": man.get("plan"), "refit": bool(refit),
            "counts": {"train": len(ds_train), "val": 0 if ds_val is None else len(ds_val)}}, indent=2, ensure_ascii=False), encoding="utf-8")
        world, bs = _dist.world_size(), int(args.per_device_train_batch_size)
        budget = f"{args.max_steps} 步" if args.max_steps > 0 else f"不限步数（{sch['epochs_max']} 个 epoch）"
        print(f"[m8] tag={tag or cfg['run'].get('tag')} params={model.n_params():,} channels={len(model.fe.channel_names)} "
              f"blocks={cfg['model']['blocks']}")
        how = ("无验证、不早停（refit）" if refit else
               f"验证{f'每 {eval_every} 步' if eval_every else '每个 epoch 末'}，早停指标 {cfg['train']['early_stop']['metric']}，"
               f"patience={cfg['train']['early_stop']['patience']}")
        print(f"[m8] train={len(ds_train)} val={0 if ds_val is None else len(ds_val)} rows；{world} 个进程 × 每卡 batch {bs}；"
              f"预算 {budget}；{how}；采样 {sampling}；device={args.device}", flush=True)
    best_path = out / WEIGHTS_FILE
    ema_decay = cfg["train"].get("ema_decay")
    ema = WeightEma(model, float(ema_decay)) if ema_decay is not None else None
    if main and ema is not None:
        print(f"[m8] 权重 EMA 开：decay={ema.decay}；验证与最优权重都用 EMA 权重")
    cb = None if refit else EvalCallback(model, ds_val, collate, train_cfg, best_path=best_path,
                                         patience=int(cfg["train"]["early_stop"]["patience"]),
                                         log_path=out / "train_log.json", eval_every_steps=eval_every, ema=ema)
    ckpt = CheckpointCallback(model, out, save_every, cfg_dump, ema) if (main and save_every > 0) else None
    if ckpt is not None:
        print(f"[m8] 每 {save_every} 步存 checkpoint → {out / 'checkpoints'}/step_XXXXX/（{'EMA 权重' if ema is not None else '当前权重'}）；"
              f"最后一步也存一份", flush=True)
    callbacks = ([EmaCallback(ema, model)] if ema is not None else []) + ([ckpt] if ckpt else []) + ([cb] if cb else [])   # EMA 更新排在存盘 / 验证前
    model.train()
    trainer = _M8Trainer(model=model, args=args, data_collator=collate, train_dataset=ds_train, callbacks=callbacks,
                         sampling=sampling, station_col=station_col, ts_col=ts_col)
    t0 = time.time()
    trainer.train()
    if not main:
        return {}
    row = functools.partial(_summary_row, cfg, tag, out, model, n_train=len(ds_train))
    if refit:
        # 全量重训：没有验证分数；最终权重 = 第 max_steps 步的权重（cosine 调度在这一步退火到 0）；开了 EMA 就是 EMA 权重
        if ema is not None:
            ema.load_into(model)
        save_weights(model, best_path)
        if ckpt is not None and int(trainer.state.global_step) not in ckpt.saved:                    # 最后一步也进 checkpoints/，挑选时一并评测
            ckpt.ema = None                                                                          # EMA 已写进模型
            ckpt.save(int(trainer.state.global_step))
        if ckpt is not None:
            print(f"[m8] 中间 checkpoint {len(ckpt.saved)} 个：{out / 'checkpoints'}（步数 {ckpt.saved}）")
        hist = [{"step": int(h["step"]), "train_loss": float(h["loss"])} for h in trainer.state.log_history if "loss" in h]
        summary = pd.DataFrame([row(best_step=int(args.max_steps), n_evals=0, minutes=(time.time() - t0) / 60.0, n_val=0)])
        summary.to_csv(report_dir / "summary.csv", index=False)
        (out / "train_log.json").write_text(json.dumps({"history": hist, "best": None, "best_step": int(args.max_steps),
                                                        "patience": None, "metric": None, "refit": True, "ema_decay": ema_decay,
                                                        "checkpoints": [] if ckpt is None else [f"checkpoints/step_{t:05d}" for t in ckpt.saved],
                                                        "final": summary.to_dict("records")[0]},
                                                       indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(summary.to_string(index=False))
        print(f"[m8] 产物：{best_path} / {out / CONFIG_FILE} / {report_dir}（全量重训没有验证分数；测试集分数用 evaluate.py / eval_m8.sh）")
        return {"summary": summary, "station_table": None, "history": hist, "best": None}
    if cb.best_step is None or not best_path.exists():
        raise ValueError(f"训练全程没有产出可用的最优权重：早停指标 {cb.metric} 在验证集上自始至终非有限，{best_path} 不存在。"
                         f"请检查验证集是否覆盖完整天、station_meta 是否填好。")
    load_weights(model, best_path)
    model.eval()
    # 终评：南网 D+1（10:00 行）+ nrmse（整点行）；result_val 默认存整点行
    nw_score, station_tab, _ = score_nanwang(model, ds_val, collate, train_cfg)
    nr, res_hourly = score_nrmse(model, ds_val, collate, train_cfg, int(cfg["train"]["early_stop"].get("nrmse_issue_every_min", 60)))
    res_hourly.to_parquet(out / "result_val.parquet", index=False)
    nw_seen, nw_unseen, n_unseen = seen_unseen_scores(station_tab)
    summary = pd.DataFrame([row(best_step=cb.best_step, n_evals=len(cb.history), minutes=(time.time() - t0) / 60.0, n_val=len(ds_val),
                                nw=nw_score, nr=nr, seen=nw_seen, unseen=nw_unseen, n_unseen=n_unseen, ultra=ultra_proxy_pct(res_hourly))])
    summary.to_csv(report_dir / "summary.csv", index=False)
    station_tab.to_csv(report_dir / "station_table.csv", index=False)
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    log["final"] = summary.to_dict("records")[0]
    (out / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(summary.to_string(index=False)); print(station_tab.to_string(index=False))
    print(f"[m8] 产物：{best_path} / {out / CONFIG_FILE} / {out / 'result_val.parquet'} / {report_dir}")
    return {"summary": summary, "station_table": station_tab, "history": cb.history, "best": cb.best}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="M8 PVRex 训练")
    ap.add_argument("--config", default="forecast_backbones/m8/config/j1_final2.yaml")
    ap.add_argument("--splits-dir", default=None, help="pipelines.time_splits 落好的目录（train/valid.parquet + split_manifest.json）")
    ap.add_argument("--train-parquet", nargs="+", default=None,
                    help="现成的训练 parquet：文件夹 / 通配符 / 文件列表，全量训练、不切块、不剔除行、无验证集（等于 --refit）。与 --splits-dir 二选一")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖 train.schedule.max_steps；0 = 不限步数（按 epochs_max）")
    ap.add_argument("--tag", default=None, help="写进 reports/summary.csv 的变体名")
    ap.add_argument("--save-every", type=int, default=None,
                    help="每 N 个优化步存一份完整 checkpoint 到 <out>/checkpoints/step_XXXXX/（覆盖 train.schedule.save_every；0 = 不存）")
    ap.add_argument("--refit", action="store_true",
                    help="全量重训：train ∪ valid 全当训练行，不验证、不早停，固定 max_steps 步后存最终权重（配方选定后给最终模型）")
    a = ap.parse_args()
    train_m8(a.config, a.splits_dir, out_dir=a.out, max_steps=a.max_steps, tag=a.tag, refit=a.refit, train_parquet=a.train_parquet,
             save_every=a.save_every)
