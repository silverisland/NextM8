"""Shared HF schedule, EMA, checkpointing and metric-driven early stopping.

EvaluationCallback subclasses supply _score; this layer never loads a model.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from . import distributed as _dist
from .checkpoints import save_weights
from .config import deep_merge, dump_config
from .rows import StationDaySampler

WEIGHTS_FILE, CONFIG_FILE = "model.safetensors", "model_config.yaml"
EARLY_STOP_METRICS = {
    "nanwang_after_pstar": True,
    "nanwang_short_D+1": True,
    "nrmse": False,
    "nanwang_ultra_proxy": True,
}


class WeightEma:
    """权重指数滑动平均（train.ema_decay）：每个优化步后 shadow ← d_n·shadow + (1−d_n)·θ，d_n = min(decay, (1+n)/(10+n))
    （前几步 d 小，shadow 快速跟上随机初始化之后的权重）；浮点张量平均，其它缓冲直接拷贝。
    applied(model) 期间模型权重换成 shadow（验证、存最优），退出复原。"""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay, self.n, self.last_decay = float(decay), 0, float("nan")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"train.ema_decay 须在 [0, 1]，得到 {decay!r}")
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        self.last_decay = d
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if s.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().to(s.device), alpha=1.0 - d)
            else:
                s.copy_(v.detach().to(s.device))

    @contextlib.contextmanager
    def applied(self, model: torch.nn.Module):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)

    def load_into(self, model: torch.nn.Module) -> None:
        """把 shadow 永久写进模型（refit 的最终权重）。"""
        model.load_state_dict(self.shadow, strict=True)


class EmaCallback(TrainerCallback):
    """每个优化步后更新 EMA；须排在 EvalCallback 前面（同一步的验证用更新后的 shadow）。"""

    def __init__(self, ema: WeightEma, model: torch.nn.Module):
        self.ema, self.model = ema, model

    def on_step_end(self, args, state, control, **kwargs):
        self.ema.update(self.model)
        return control


class CheckpointCallback(TrainerCallback):
    """每 every 个优化步存一份自包含 checkpoint：<out>/checkpoints/step_XXXXX/{model.safetensors, model_config.yaml}。
    model_config.yaml 的 run.out_dir 指向该目录本身，train.weights_path = ${run.out_dir}/model.safetensors，原地就能加载；
    开了 EMA 存 EMA 权重（与最终权重同口径）。须排在 EmaCallback 后面（同一步存更新后的 shadow）。只在主进程调用。"""

    def __init__(
        self,
        model: torch.nn.Module,
        out: Path,
        every: int,
        cfg_dump: dict,
        ema: "WeightEma | None" = None,
    ):
        self.model, self.root, self.every, self.cfg_dump, self.ema = (
            model,
            Path(out) / "checkpoints",
            int(every),
            cfg_dump,
            ema,
        )
        self.saved: list[int] = []

    def save(self, step: int) -> Path:
        d = self.root / f"step_{int(step):05d}"
        d.mkdir(parents=True, exist_ok=True)
        if self.ema is not None:
            with self.ema.applied(self.model):
                save_weights(self.model, d / WEIGHTS_FILE)
        else:
            save_weights(self.model, d / WEIGHTS_FILE)
        cfg = deep_merge(
            self.cfg_dump,
            {
                "run": {"out_dir": str(d.resolve())},
                "train": {"weights_path": "${run.out_dir}/" + WEIGHTS_FILE},
            },
        )
        dump_config(cfg, d / CONFIG_FILE)
        self.saved.append(int(step))
        return d

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step > 0 and step % self.every == 0 and step not in self.saved:
            d = self.save(step)
            print(f"[forecast] checkpoint step {step} → {d}", flush=True)
        return control


class ForecastTrainer(Trainer):
    def __init__(
        self,
        *a,
        sampling: str = "random",
        station_col: str = "station_id",
        ts_col: str = "timestamp_win",
        **kw,
    ):
        super().__init__(*a, **kw)
        self._sampling, self._station_col, self._ts_col = sampling, station_col, ts_col

    def _get_train_sampler(
        self, *a, **kw
    ):  # transformers>=4.46 多传一个 dataset 位置参数，4.40 不传：两种签名都接
        if self._sampling == "station_day":
            return StationDaySampler(
                self.train_dataset, self._station_col, self._ts_col, seed=self.args.seed
            )
        return super()._get_train_sampler(*a, **kw)


def build_training_args(cfg: dict, out_dir, max_steps=None) -> TrainingArguments:
    """max_steps：None = 用 train.schedule.max_steps；0（或配置里 null）= 不限步数，按 epochs_max 训。"""
    s = cfg["train"]["schedule"]
    steps = s.get("max_steps") if max_steps is None else max_steps
    cuda = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if cuda and local_rank >= 0:
        count = torch.cuda.device_count()
        if local_rank >= count:
            raise ValueError(
                f"LOCAL_RANK={local_rank} exceeds the {count} visible CUDA devices; "
                "check CUDA_VISIBLE_DEVICES and the launcher's process count"
            )
        # Bind before capability probes and HF/Accelerate process-group setup.
        torch.cuda.set_device(local_rank)
    bf16 = (
        str(s.get("precision", "fp32")) == "bf16"
        and cuda
        and torch.cuda.is_bf16_supported()
    )
    fp16 = str(s.get("precision", "fp32")) == "fp16" and cuda
    eval_kw = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    return TrainingArguments(
        output_dir=str(Path(out_dir) / "hf"),
        per_device_train_batch_size=int(s["batch_size"]),
        num_train_epochs=float(s["epochs_max"]),
        learning_rate=float(s["lr"]),
        weight_decay=float(s["weight_decay"]),
        max_grad_norm=float(s["grad_clip"]),
        bf16=bf16,
        fp16=fp16,
        use_cpu=not cuda,
        lr_scheduler_type="cosine",
        warmup_steps=int(s.get("warmup_steps", 0)),
        save_strategy="no",
        logging_steps=int(s.get("logging_steps", 50)),
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=not bool(s.get("progress_bar", True)),
        dataloader_pin_memory=cuda,
        dataloader_num_workers=int(s.get("num_workers", 0)),
        max_steps=(int(steps) if steps else -1),
        seed=int((cfg["run"].get("seeds") or [0])[0]),
        ddp_timeout=7200,
        **{eval_kw: "no"},
    )


class EvaluationCallback(TrainerCallback):
    def __init__(
        self,
        model,
        ds_val,
        collate,
        cfg,
        best_path,
        patience: int,
        log_path,
        eval_every_steps=None,
        ema: WeightEma | None = None,
    ):
        self.model, self.ds_val, self.collate, self.cfg, self.ema = (
            model,
            ds_val,
            collate,
            cfg,
            ema,
        )
        self.best_path, self.log_path, self.patience = (
            Path(best_path),
            Path(log_path),
            int(patience),
        )
        self.eval_every_steps = int(eval_every_steps) if eval_every_steps else None
        es = cfg["train"]["early_stop"]
        self.metric = str(es.get("metric", "nanwang_after_pstar"))
        if self.metric not in EARLY_STOP_METRICS:
            raise ValueError(
                f"train.early_stop.metric={self.metric!r} 未知，可选 {sorted(EARLY_STOP_METRICS)}"
            )
        self.higher_better = EARLY_STOP_METRICS[self.metric]
        self.issue_every_min = int(es.get("nrmse_issue_every_min", 60))
        self.smooth_k = int(es.get("smooth_k", 1) or 1)
        if self.smooth_k < 1:
            raise ValueError(f"train.early_stop.smooth_k 须 ≥ 1，得到 {self.smooth_k}")
        self.best = -np.inf if self.higher_better else np.inf
        self.best_step, self.bad_evals, self.history, self.raw_scores = None, 0, [], []
        self.last_unseen = None
        self.last_extra: dict = {}

    def _score(self) -> float:
        raise NotImplementedError("Provide a model-specific validation scorer")

    def _record(
        self, step: int, epoch: float, score: float, loss: float | None
    ) -> bool:
        """score = 本次原始验证分；best / 早停按最近 smooth_k 次原始分（非有限的不算）的均值判。"""
        self.raw_scores.append(float(score))
        window = [s for s in self.raw_scores[-self.smooth_k :] if np.isfinite(s)]
        sel = float(np.mean(window)) if window else float("nan")
        better = sel > self.best if self.higher_better else sel < self.best
        if np.isfinite(sel) and better:
            self.best, self.best_step, self.bad_evals = float(sel), int(step), 0
            save_weights(self.model, self.best_path)
        else:
            self.bad_evals += 1
        self.history.append(
            {
                "step": int(step),
                "epoch": float(epoch),
                self.metric: float(score),
                **({f"{self.metric}_smooth": sel} if self.smooth_k > 1 else {}),
                "train_loss": loss,
                **(
                    {"nanwang_unseen": self.last_unseen}
                    if self.last_unseen is not None
                    else {}
                ),
                **{
                    k: (float(v) if v == v else None)
                    for k, v in self.last_extra.items()
                },
                "best": self.best,
                "bad_evals": self.bad_evals,
                "time": time.time(),
            }
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            json.dumps(
                {
                    "history": self.history,
                    "best": self.best,
                    "best_step": self.best_step,
                    "patience": self.patience,
                    "metric": self.metric,
                    "smooth_k": self.smooth_k,
                    "ema_decay": None if self.ema is None else self.ema.decay,
                    "higher_better": self.higher_better,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        sm = f" smooth{self.smooth_k}={sel:.4f}" if self.smooth_k > 1 else ""
        print(
            f"[forecast] step {step} (epoch {epoch:.3g}): {self.metric}={score:.4f}{sm} best={self.best:.4f}@step {self.best_step} "
            f"bad={self.bad_evals}/{self.patience}",
            flush=True,
        )
        return self.bad_evals >= self.patience

    def _evaluate(self, state, control):
        loss = next(
            (h["loss"] for h in reversed(state.log_history) if "loss" in h), None
        )
        if _dist.is_main():
            with (
                self.ema.applied(self.model)
                if self.ema is not None
                else contextlib.nullcontext()
            ):  # 验证与存最优都用 EMA 权重
                stop = self._record(
                    int(state.global_step or 0),
                    float(state.epoch or 0.0),
                    self._score(),
                    loss,
                )
        else:
            stop = False
        if _dist.broadcast_from_main(stop) and control is not None:
            control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        every = self.eval_every_steps
        if every and (
            state.global_step % every == 0 or state.global_step >= state.max_steps
        ):
            self._evaluate(state, control)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.eval_every_steps:
            self._evaluate(state, control)
        return control
