# -*- coding: utf-8 -*-
# forecast_backbones/m5/train.py
"""M5 训练入口（TimesFM-3 + LoRA；框架自 M8）：
  1. 读 pipelines.time_splits 落好的切分目录（train.parquet / valid.parquet / split_manifest.json）
  2. HF Trainer：损失在模型 forward 里（9 档 pinball，未来 192 步，默认容量归一）；只训 LoRA + output_head
  3. EvalCallback：第 0 步（LoRA 的 B=0 = 零样本）先打一次分计入 history 并作为初始最优，之后每 eval_every_steps 步打分；
     早停 / 最优按 train.early_stop；最优权重存 model.safetensors（只含 LoRA + output_head）
  4. 训后：加载最优权重 → 验证集推理 → result_val.parquet → reports/summary.csv（含 nanwang_zeroshot / gain）、reports/station_table.csv
  5. finetune.enabled=false（零样本 z0）：不训练，第 0 步打分即终评，不写权重
  6. --refit / --train-parquet：与 M8 相同（无验证、固定步数后存最终权重）；零样本不支持
训练模型的构造从零样本起步：train.weights_path 改指向 out_dir 下从不写入的 .fresh/ 哨兵路径。"""
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
from transformers import Trainer, TrainerCallback, set_seed

try:
    from .config import DEFAULT_CFG, load_config, deep_merge, dump_config, raw_override
    from .solar import station_cap_mw
    from .transformer_trainer import evaluate_model, build_training_args
    from .transformer_models import TransformersAdaptedM5Model, save_weights, load_weights
    from .data import _dataset_with_station_id, _check_presplit_dir
    from .rows import SHORT_ISSUE_TIME, scored_row_index, issue_grid_row_index, StationDaySampler, row_stations
    from . import _dist
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m5.config import DEFAULT_CFG, load_config, deep_merge, dump_config, raw_override
    from forecast_backbones.m5.solar import station_cap_mw
    from forecast_backbones.m5.transformer_trainer import evaluate_model, build_training_args
    from forecast_backbones.m5.transformer_models import TransformersAdaptedM5Model, save_weights, load_weights
    from forecast_backbones.m5.data import _dataset_with_station_id, _check_presplit_dir
    from forecast_backbones.m5.rows import SHORT_ISSUE_TIME, scored_row_index, issue_grid_row_index, StationDaySampler, row_stations
    from forecast_backbones.m5 import _dist
try:
    from ...pipelines.transformer_dataset import UnifiedDataset
    from ...pipelines.nanwang_metric import NanwangMetric
except ImportError:
    from pipelines.transformer_dataset import UnifiedDataset
    from pipelines.nanwang_metric import NanwangMetric

WEIGHTS_FILE, CONFIG_FILE = "model.safetensors", "model_config.yaml"
EARLY_STOP_METRICS = {"nanwang_after_pstar": True, "nrmse": False, "nanwang_ultra_proxy": True}   # 指标 → 是否越高越好


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
    if res.empty:
        return float("nan"), pd.DataFrame(columns=TABLE_COLS), res
    gccap = {model.station_meta.name(int(s)): station_cap_mw(model.station_meta, int(s)) for s in res["station_id"].unique()}   # 每站固定容量
    sd = NanwangMetric().short_d1_daily(res, gccap)
    name_to_id = {model.station_meta.name(int(s)): int(s) for s in res["station_id"].unique()}
    if len(sd) == 0:
        return float("nan"), pd.DataFrame(columns=TABLE_COLS), res
    g = sd.groupby("station")["accuracy_pct"].agg(["mean", "count"]).reset_index()
    table = pd.DataFrame({"station_id": g["station"].map(name_to_id).astype(int), "station": g["station"],
                          "accuracy_pct": g["mean"].astype(float), "n_days": g["count"].astype(int)})
    known = cfg["dataset"].get("known_stations")
    table["seen"] = True if not known else table["station_id"].isin([int(s) for s in known])
    # 诊断列：这站的打分行有多少走了退化几何（缺坐标）；这站打分行上的 nrmse
    by_sid = {int(s): g for s, g in res.groupby("station_id")}
    table["degraded_frac"] = table["station_id"].map(lambda s: float(by_sid[s]["degraded"].mean()) if "degraded" in res else 0.0).round(3)
    table["nrmse_pct"] = table["station_id"].map(lambda s: nrmse_pct(by_sid[s])).round(2)
    return float(sd["accuracy_pct"].mean()), table, res


def seen_unseen_scores(table: pd.DataFrame) -> tuple[float, float, int]:
    """逐站表 → (已见站站日均值, 未见站站日均值, 未见站数)；按 n_days 加权等于站日均值。"""
    if table.empty or "seen" not in table:
        return float("nan"), float("nan"), 0
    def wmean(t):
        return float((t["accuracy_pct"] * t["n_days"]).sum() / t["n_days"].sum()) if len(t) and t["n_days"].sum() else float("nan")
    seen, unseen = table[table["seen"]], table[~table["seen"]]
    return wmean(seen), wmean(unseen), int(len(unseen))


def nrmse_pct(res: pd.DataFrame) -> float:
    if res.empty:
        return float("nan")
    y = np.stack(res["observe_power_future"].to_numpy()).astype(float)
    p = np.stack(res["observe_power_predict"].to_numpy()).astype(float)
    r = (y - p) / res["cap"].to_numpy(float)[:, None]
    ok = np.isfinite(r)
    return float(100.0 * np.sqrt(np.mean(r[ok] ** 2))) if ok.any() else float("nan")


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
class WeightEma:
    """权重指数滑动平均（train.ema_decay）：只影子 requires_grad 的参数（LoRA + output_head），冻结的底座不复制。
    shadow ← d_n·shadow + (1−d_n)·θ，d_n = min(decay, (1+n)/(10+n))；applied(model) 期间这些参数换成 shadow，退出复原。"""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay, self.n, self.last_decay = float(decay), 0, float("nan")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"train.ema_decay 须在 [0, 1]，得到 {decay!r}")
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
        if not self.shadow:
            raise ValueError("train.ema_decay 开了，但模型没有可训参数")

    def _params(self, model: torch.nn.Module) -> dict:
        return {k: p for k, p in model.named_parameters() if k in self.shadow}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        self.last_decay = d
        for k, p in self._params(model).items():
            s = self.shadow[k]
            s.mul_(d).add_(p.detach().to(s.device), alpha=1.0 - d)

    @contextlib.contextmanager
    def applied(self, model: torch.nn.Module):
        params = self._params(model)
        backup = {k: p.detach().clone() for k, p in params.items()}
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(self.shadow[k])
        try:
            yield
        finally:
            with torch.no_grad():
                for k, p in params.items():
                    p.copy_(backup[k])

    @torch.no_grad()
    def load_into(self, model: torch.nn.Module) -> None:
        """把 shadow 永久写进模型（refit 的最终权重）。"""
        for k, p in self._params(model).items():
            p.copy_(self.shadow[k])


class EmaCallback(TrainerCallback):
    """每个优化步后更新 EMA；须排在 EvalCallback 前面（同一步的验证用更新后的 shadow）。"""

    def __init__(self, ema: WeightEma, model: torch.nn.Module):
        self.ema, self.model = ema, model

    def on_step_end(self, args, state, control, **kwargs):
        self.ema.update(self.model)
        return control


class EvalCallback(TrainerCallback):
    def __init__(self, model, ds_val, collate, cfg, best_path, patience: int, log_path, eval_every_steps=None, ema: WeightEma | None = None):
        self.model, self.ds_val, self.collate, self.cfg, self.ema = model, ds_val, collate, cfg, ema
        self.best_path, self.log_path, self.patience = Path(best_path), Path(log_path), int(patience)
        self.eval_every_steps = int(eval_every_steps) if eval_every_steps else None
        es = cfg["train"]["early_stop"]
        self.metric = str(es.get("metric", "nanwang_after_pstar"))
        if self.metric not in EARLY_STOP_METRICS:
            raise ValueError(f"train.early_stop.metric={self.metric!r} 未知，可选 {sorted(EARLY_STOP_METRICS)}")
        self.higher_better = EARLY_STOP_METRICS[self.metric]
        self.issue_every_min = int(es.get("nrmse_issue_every_min", 60))
        self.smooth_k = int(es.get("smooth_k", 1) or 1)
        if self.smooth_k < 1:
            raise ValueError(f"train.early_stop.smooth_k 须 ≥ 1，得到 {self.smooth_k}")
        self.best = -np.inf if self.higher_better else np.inf
        self.best_step, self.bad_evals, self.history, self.raw_scores = None, 0, [], []
        self.last_unseen = None
        self.last_extra: dict = {}
        self.zeroshot = None                   # 第 0 步（还没走过优化步 = 零样本）的南网 D+1

    def _score(self) -> float:
        self.model._progress_desc = "[m5] 验证"
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

    def _record(self, step: int, epoch: float, score: float, loss: float | None) -> bool:
        """score = 本次原始验证分；best / 早停按最近 smooth_k 次原始分（非有限的不算）的均值判。"""
        self.raw_scores.append(float(score))
        window = [s for s in self.raw_scores[-self.smooth_k:] if np.isfinite(s)]
        sel = float(np.mean(window)) if window else float("nan")
        better = sel > self.best if self.higher_better else sel < self.best
        if np.isfinite(sel) and better:
            self.best, self.best_step, self.bad_evals = float(sel), int(step), 0
            save_weights(self.model, self.best_path)
        else:
            self.bad_evals += 1
        self.history.append({"step": int(step), "epoch": float(epoch), self.metric: float(score),
                             **({f"{self.metric}_smooth": sel} if self.smooth_k > 1 else {}), "train_loss": loss,
                             **({"nanwang_unseen": self.last_unseen} if self.last_unseen is not None else {}),
                             **{k: (float(v) if v == v else None) for k, v in self.last_extra.items()},
                             "best": self.best, "bad_evals": self.bad_evals, "time": time.time()})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps({"history": self.history, "best": self.best, "best_step": self.best_step,
                                             "patience": self.patience, "metric": self.metric, "smooth_k": self.smooth_k,
                                             "ema_decay": None if self.ema is None else self.ema.decay,
                                             "higher_better": self.higher_better}, indent=2, ensure_ascii=False), encoding="utf-8")
        sm = f" smooth{self.smooth_k}={sel:.4f}" if self.smooth_k > 1 else ""
        print(f"[m5] step {step} (epoch {epoch:.3g}): {self.metric}={score:.4f}{sm} best={self.best:.4f}@step {self.best_step} "
              f"bad={self.bad_evals}/{self.patience}", flush=True)
        return self.bad_evals >= self.patience

    def _evaluate(self, state, control):
        loss = next((h["loss"] for h in reversed(state.log_history) if "loss" in h), None)
        if _dist.is_main():
            with (self.ema.applied(self.model) if self.ema is not None else contextlib.nullcontext()):   # 验证与存最优都用 EMA 权重
                score = self._score()
                if self.zeroshot is None:
                    self.zeroshot = (float(score) if self.metric == "nanwang_after_pstar"
                                     else float(score_nanwang(self.model, self.ds_val, self.collate, self.cfg)[0]))
                stop = self._record(int(state.global_step or 0), float(state.epoch or 0.0), score, loss)
        else:
            stop = False
        if _dist.broadcast_from_main(stop) and control is not None:
            control.should_training_stop = True

    def on_train_begin(self, args, state, control, **kwargs):
        """第 0 步（LoRA 的 B=0，输出与零样本逐位相同）先打一次分，计入 history 并作为初始最优；patience 从它起算。"""
        self._evaluate(state, control)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        every = self.eval_every_steps
        if every and (state.global_step % every == 0 or state.global_step >= state.max_steps):
            self._evaluate(state, control)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.eval_every_steps:
            self._evaluate(state, control)
        return control


class _M5Trainer(Trainer):
    def __init__(self, *a, sampling: str = "random", station_col: str = "station_id", ts_col: str = "timestamp_win", **kw):
        super().__init__(*a, **kw)
        self._sampling, self._station_col, self._ts_col = sampling, station_col, ts_col

    def _get_train_sampler(self, *a, **kw):          # transformers>=4.46 多传一个 dataset 位置参数，4.40 不传：两种签名都接
        if self._sampling == "station_day":
            return StationDaySampler(self.train_dataset, self._station_col, self._ts_col, seed=self.args.seed)
        return super()._get_train_sampler(*a, **kw)


# ------------------------------------------------------------------ 数据
def load_presplit(splits_dir, cfg: dict, station_meta, scratch_dir, reuse: bool = False, refit: bool = False):
    """refit=True：全量重训，train.parquet ∪ valid.parquet 全部当训练行（valid 可以是 0 行），不建验证集（返回 ds_val=None）。"""
    import pyarrow.parquet as pq
    d = _check_presplit_dir(splits_dir, need_cal=False)
    man = json.loads((d / "split_manifest.json").read_text(encoding="utf-8"))
    load = functools.partial(_dataset_with_station_id, cfg=cfg, station_meta=station_meta, scratch_dir=scratch_dir, reuse=reuse)
    if refit:
        ds_train = load([str(_refit_rows(d, scratch_dir, reuse))])
        if len(ds_train) == 0:
            raise ValueError(f"{d} 的 train ∪ valid 为空")
        return ds_train, None, man
    if pq.read_metadata(d / "valid.parquet").num_rows == 0:
        raise ValueError(f"{d} 的 valid.parquet 为 0 行：这是全量重训切分，训练请加 --refit（不验证、不早停、固定步数）")
    ds_train, ds_val = load([str(d / "train.parquet")]), load([str(d / "valid.parquet")])
    if len(ds_train) == 0 or len(ds_val) == 0:
        raise ValueError(f"{d} 的 train/valid 为空")
    return ds_train, ds_val, man


def expand_parquets(items) -> list[str]:
    """目录 → 其下全部 *.parquet；通配符 → 展开；文件 → 原样。按路径排序，去重。"""
    import glob
    out: list[str] = []
    for it in items or ():
        p = Path(str(it))
        if p.is_dir():
            out += sorted(str(f) for f in p.glob("*.parquet"))
        else:
            hits = sorted(glob.glob(str(it)))
            if not hits:
                raise FileNotFoundError(f"--train-parquet {it!r} 没有匹配到文件")
            out += hits
    return sorted(dict.fromkeys(out))


def load_full(train_parquet, cfg: dict, station_meta, scratch_dir, reuse: bool = False):
    """全量训练：现成的训练 parquet（文件夹 / 通配符 / 文件列表，一站一个文件也行）全部当训练行，一行不删、没有验证集。
    站列是站名文本时按 station_meta 映射成整数站号（_dataset_with_station_id）。返回 (ds_train, None, manifest)。"""
    files = expand_parquets(train_parquet)
    if not files:
        raise FileNotFoundError(f"--train-parquet {train_parquet!r} 下没有 parquet 文件")
    ds = _dataset_with_station_id(files, cfg, station_meta, scratch_dir=scratch_dir, reuse=reuse)
    if len(ds) == 0:
        raise ValueError(f"--train-parquet {files} 里没有行")
    man = {"generated_by": "forecast_backbones.m5.train --train-parquet", "plan": None, "train_parquet": files}
    return ds, None, man


def _refit_rows(d: Path, scratch_dir, reuse: bool) -> Path:
    """train.parquet ∪ valid.parquet（去掉 split_block 列）→ scratch_dir/refit_rows.parquet；valid 为 0 行时直接用 train.parquet。
    reuse=True（多卡的非主进程）：主进程已写好就直接用。"""
    import pyarrow.parquet as pq
    if pq.read_metadata(d / "valid.parquet").num_rows == 0:
        return d / "train.parquet"
    out = Path(scratch_dir) / "refit_rows.parquet"
    if reuse and out.exists():
        return out
    tr = pd.read_parquet(d / "train.parquet")
    va = pd.read_parquet(d / "valid.parquet").drop(columns=["split_block"], errors="ignore")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([tr, va], ignore_index=True).to_parquet(out, index=False)
    return out


def _summary_row(cfg, tag, out, model, best_step, n_evals, minutes, n_val, n_train,
                 nw=float("nan"), nr=float("nan"), seen=float("nan"), unseen=float("nan"), n_unseen=0, ultra=float("nan"),
                 zs=float("nan")) -> dict:
    """reports/summary.csv 的一行（列序固定；collect_runs 按列名取）。refit 时分数列全是 NaN。
    nanwang_zeroshot = 第 0 步（零样本）的南网 D+1；gain = 最终 D+1 − 零样本。"""
    ft, lo = cfg["finetune"], cfg["finetune"]["lora"]
    on = bool(ft["enabled"])
    ratio = cfg["decision"]["physics"].get("clear_sky_cap_ratio")
    return {"tag": tag or cfg["run"].get("tag") or out.name, "nanwang_short_D+1": nw, "nanwang_zeroshot": zs, "gain": nw - zs,
            "nrmse_pct": nr, "nanwang_ultra_proxy": ultra,
            "nanwang_seen": seen, "nanwang_unseen": unseen, "n_unseen_stations": n_unseen,
            "best_step": best_step, "n_evals": n_evals, "train_minutes": minutes,
            "method": ft["method"] if on else "zero_shot", "rank": int(lo["rank"]) if on else 0,
            "alpha": float(lo["alpha"]) if on else float("nan"),
            "target_modules": "+".join(lo["target_modules"]) if on else "",
            "unfreeze_head": bool(ft["unfreeze_output_head"]) and on,
            "n_trainable": model.n_params(), "lr": float(cfg["train"]["schedule"]["lr"]),
            "derived": "+".join(k for k, v in cfg["features"]["derived"].items() if v),
            "cs_ceiling": float("nan") if ratio is None else float(ratio),
            "ema_decay": float("nan") if cfg["train"].get("ema_decay") is None else float(cfg["train"]["ema_decay"]),
            "smooth_k": int(cfg["train"]["early_stop"].get("smooth_k", 1) or 1),
            "n_val_rows": n_val, "n_train_rows": n_train,
            "batch_global": int(cfg["train"]["schedule"]["batch_size"]), "n_gpus": _dist.world_size()}


def _zero_shot_report(model, ds_val, collate, train_cfg, cfg, tag, out, report_dir, n_train) -> dict:
    """finetune.enabled=false：不训练，第 0 步（零样本）打分即终评；不写权重文件。"""
    t0 = time.time()
    model.eval()
    nw, station_tab, _ = score_nanwang(model, ds_val, collate, train_cfg)
    nr, res_hourly = score_nrmse(model, ds_val, collate, train_cfg, int(cfg["train"]["early_stop"].get("nrmse_issue_every_min", 60)))
    res_hourly.to_parquet(out / "result_val.parquet", index=False)
    seen, unseen, n_unseen = seen_unseen_scores(station_tab)
    up = ultra_proxy_pct(res_hourly)
    summary = pd.DataFrame([_summary_row(cfg, tag, out, model, best_step=0, n_evals=1, minutes=(time.time() - t0) / 60.0,
                                         n_val=len(ds_val), n_train=n_train, nw=nw, nr=nr, seen=seen, unseen=unseen,
                                         n_unseen=n_unseen, ultra=up, zs=nw)])
    summary.to_csv(report_dir / "summary.csv", index=False)
    station_tab.to_csv(report_dir / "station_table.csv", index=False)
    metric = str(cfg["train"]["early_stop"].get("metric", "nanwang_after_pstar"))
    score = {"nanwang_after_pstar": nw, "nrmse": nr, "nanwang_ultra_proxy": up}[metric]
    hist = [{"step": 0, "epoch": 0.0, "nanwang_after_pstar": nw, metric: score, "train_loss": None,
             "best": score, "bad_evals": 0, "time": time.time()}]
    (out / "train_log.json").write_text(json.dumps({"history": hist, "best": score, "best_step": 0, "metric": metric,
                                                    "zero_shot": True, "final": summary.to_dict("records")[0]},
                                                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(summary.to_string(index=False)); print(station_tab.to_string(index=False))
    print(f"[m5] 零样本：没有训练、没有权重文件；产物 {out / CONFIG_FILE} / {out / 'result_val.parquet'} / {report_dir}")
    return {"summary": summary, "station_table": station_tab, "history": hist, "best": score}


# ------------------------------------------------------------------ 主流程
def train_m5(config_path, splits_dir=None, out_dir=None, max_steps=None, tag: str | None = None, refit: bool = False,
             train_parquet=None) -> dict:
    """splits_dir：time_splits 落好的切分目录（有验证集，可早停）；train_parquet：现成的训练 parquet，全量训练、无验证（等于 refit）。二选一。
    finetune.enabled=false（零样本 z0）：不训练，第 0 步打分即终评，不写权重。"""
    if (splits_dir is None) == (train_parquet is None):
        raise ValueError("--splits-dir 与 --train-parquet 二选一：切分目录（带验证集）或现成的训练 parquet（全量、无验证）")
    if train_parquet is not None:
        refit = True
    if out_dir is not None:
        config_path = deep_merge(raw_override(config_path), {"run": {"out_dir": str(out_dir)}})
    cfg = load_config(config_path)
    metric = str(cfg["train"]["early_stop"].get("metric", "nanwang_after_pstar"))
    if metric not in EARLY_STOP_METRICS:      # 提前校验：零样本 run 不建 EvalCallback，否则要到打分后才 KeyError
        raise ValueError(f"train.early_stop.metric={metric!r} 未知，可选 {sorted(EARLY_STOP_METRICS)}")
    finetune = bool(cfg["finetune"]["enabled"])
    if refit and not finetune:
        raise ValueError("零样本（finetune.enabled: false）没有可训练参数：不需要 --refit / --train-parquet，测试集直接用 evaluate.py")
    out = Path(cfg["run"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    if out_dir is not None:
        cfg["eval"]["report_dir"] = str(out / "reports")
    report_dir = Path(cfg["eval"]["report_dir"]); report_dir.mkdir(parents=True, exist_ok=True)
    sch = cfg["train"]["schedule"]
    sampling = str(sch.get("sampling") or "random")
    if sampling not in ("random", "station_day"):
        raise ValueError(f"train.schedule.sampling={sampling!r} 不在 ('random', 'station_day')")
    if splits_dir is not None:
        _check_presplit_dir(splits_dir, need_cal=False)
    train_cfg = {**cfg, "train": {**cfg["train"], "weights_path": str(out / ".fresh" / WEIGHTS_FILE)}}
    args = build_training_args(train_cfg, out, max_steps)
    if refit and int(args.max_steps) <= 0:
        raise ValueError("全量重训（refit）没有验证集、不能早停：train.schedule.max_steps（或 --max-steps）必须 > 0，"
                         "按该配方在验证切分上各种子 best_step 的上限给")
    _ = args.device                        # 多进程时在这里建进程组、绑卡；须先于模型构造
    if args.n_gpu > 1:
        raise ValueError(f"单进程看到 {args.n_gpu} 张卡：HF 会退到 DataParallel，dict batch 不支持。多卡请用 accelerate launch / torchrun")
    main = _dist.is_main()
    set_seed(int(args.seed))               # LoRA 的 lora_A 在构造里随机初始化，须先于构造定种子（否则 run.seeds 管不到）
    model = TransformersAdaptedM5Model(train_cfg)
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
    known = model.set_known_stations(row_stations(ds_train, station_col).tolist())
    train_cfg["dataset"]["known_stations"] = known
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    eval_every = sch.get("eval_every_steps")
    if main:
        dump_config(deep_merge(deep_merge(DEFAULT_CFG, raw_override(config_path)), {"dataset": {"known_stations": known}}), out / CONFIG_FILE)
        (out / "split_manifest.json").write_text(json.dumps({"generated_by": "forecast_backbones.m5.train",
            "splits_dir": None if splits_dir is None else str(Path(splits_dir).resolve()),
            "train_parquet": man.get("train_parquet"), "plan": man.get("plan"), "refit": bool(refit),
            "counts": {"train": len(ds_train), "val": 0 if ds_val is None else len(ds_val)}}, indent=2, ensure_ascii=False), encoding="utf-8")
        ft, lo = cfg["finetune"], cfg["finetune"]["lora"]
        what = (f"LoRA rank={lo['rank']} alpha={lo['alpha']} dropout={lo['dropout']} targets={'+'.join(lo['target_modules'])} "
                f"解冻 output_head={ft['unfreeze_output_head']}" if finetune else f"零样本（{ft['inference_path']} 路径，不训练）")
        print(f"[m5] tag={tag or cfg['run'].get('tag')} {what}；可训参数 {model.n_params():,} / 总参数 {model.n_total_params():,}", flush=True)
        how = ("无验证、不早停（refit）" if refit else
               f"验证{f'每 {eval_every} 步' if eval_every else '每个 epoch 末'}（含第 0 步零样本），早停指标 "
               f"{cfg['train']['early_stop']['metric']}，patience={cfg['train']['early_stop']['patience']}")
        print(f"[m5] 训练集站 {len(known)} 个；train={len(ds_train)} val={0 if ds_val is None else len(ds_val)} rows；"
              f"全局 batch {int(sch['batch_size'])}（{_dist.world_size()} 卡 × 每卡 {int(args.per_device_train_batch_size)}）；预算 {args.max_steps} 步；{how}；采样 {sampling}；device={args.device}", flush=True)
    if not finetune:
        return _zero_shot_report(model, ds_val, collate, train_cfg, cfg, tag, out, report_dir, len(ds_train)) if main else {}
    best_path = out / WEIGHTS_FILE
    ema_decay = cfg["train"].get("ema_decay")
    ema = WeightEma(model, float(ema_decay)) if ema_decay is not None else None
    if main and ema is not None:
        print(f"[m5] 权重 EMA 开：decay={ema.decay}（只影子 LoRA + output_head）；验证与最优权重都用 EMA 权重")
    cb = None if refit else EvalCallback(model, ds_val, collate, train_cfg, best_path=best_path,
                                         patience=int(cfg["train"]["early_stop"]["patience"]),
                                         log_path=out / "train_log.json", eval_every_steps=eval_every, ema=ema)
    callbacks = ([EmaCallback(ema, model)] if ema is not None else []) + ([cb] if cb else [])
    model.train()
    trainer = _M5Trainer(model=model, args=args, data_collator=collate, train_dataset=ds_train, callbacks=callbacks,
                         sampling=sampling, station_col=station_col, ts_col=ts_col)
    t0 = time.time()
    trainer.train()
    if not main:
        return {}
    row = functools.partial(_summary_row, cfg, tag, out, model, n_train=len(ds_train))
    if refit:
        if ema is not None:
            ema.load_into(model)
        save_weights(model, best_path)
        hist = [{"step": int(h["step"]), "train_loss": float(h["loss"])} for h in trainer.state.log_history if "loss" in h]
        summary = pd.DataFrame([row(best_step=int(args.max_steps), n_evals=0, minutes=(time.time() - t0) / 60.0, n_val=0)])
        summary.to_csv(report_dir / "summary.csv", index=False)
        (out / "train_log.json").write_text(json.dumps({"history": hist, "best": None, "best_step": int(args.max_steps),
                                                        "patience": None, "metric": None, "refit": True, "ema_decay": ema_decay,
                                                        "final": summary.to_dict("records")[0]},
                                                       indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(summary.to_string(index=False))
        print(f"[m5] 产物：{best_path} / {out / CONFIG_FILE} / {report_dir}（全量重训没有验证分数；测试集分数用 evaluate.py / eval_m5.sh）")
        return {"summary": summary, "station_table": None, "history": hist, "best": None}
    if cb.best_step is None or not best_path.exists():
        raise ValueError(f"训练全程没有产出可用的最优权重：早停指标 {cb.metric} 在验证集上自始至终非有限，{best_path} 不存在。"
                         f"请检查验证集是否覆盖完整天、station_meta 是否填好。")
    load_weights(model, best_path)
    model.eval()
    nw_score, station_tab, _ = score_nanwang(model, ds_val, collate, train_cfg)
    nr, res_hourly = score_nrmse(model, ds_val, collate, train_cfg, int(cfg["train"]["early_stop"].get("nrmse_issue_every_min", 60)))
    res_hourly.to_parquet(out / "result_val.parquet", index=False)
    nw_seen, nw_unseen, n_unseen = seen_unseen_scores(station_tab)
    summary = pd.DataFrame([row(best_step=cb.best_step, n_evals=len(cb.history), minutes=(time.time() - t0) / 60.0, n_val=len(ds_val),
                                nw=nw_score, nr=nr, seen=nw_seen, unseen=nw_unseen, n_unseen=n_unseen,
                                ultra=ultra_proxy_pct(res_hourly), zs=float(cb.zeroshot))])
    summary.to_csv(report_dir / "summary.csv", index=False)
    station_tab.to_csv(report_dir / "station_table.csv", index=False)
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    log["final"] = summary.to_dict("records")[0]
    (out / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(summary.to_string(index=False)); print(station_tab.to_string(index=False))
    print(f"[m5] 产物：{best_path} / {out / CONFIG_FILE} / {out / 'result_val.parquet'} / {report_dir}")
    return {"summary": summary, "station_table": station_tab, "history": cb.history, "best": cb.best}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="M5 TimesFM-3 + LoRA 微调（finetune.enabled=false 时只做零样本打分）")
    ap.add_argument("--config", default="forecast_backbones/m5/config/finetune.yaml")
    ap.add_argument("--splits-dir", default=None, help="pipelines.time_splits 落好的目录（train/valid.parquet + split_manifest.json）")
    ap.add_argument("--train-parquet", nargs="+", default=None,
                    help="现成的训练 parquet：文件夹 / 通配符 / 文件列表，全量训练、不切块、不剔除行、无验证集（等于 --refit）。与 --splits-dir 二选一")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖 train.schedule.max_steps；0 = 不限步数（按 epochs_max）")
    ap.add_argument("--tag", default=None, help="写进 reports/summary.csv 的变体名")
    ap.add_argument("--refit", action="store_true",
                    help="全量重训：train ∪ valid 全当训练行，不验证、不早停，固定 max_steps 步后存最终权重（配方选定后给最终模型）")
    a = ap.parse_args()
    train_m5(a.config, a.splits_dir, out_dir=a.out, max_steps=a.max_steps, tag=a.tag, refit=a.refit, train_parquet=a.train_parquet)
