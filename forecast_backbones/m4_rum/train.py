"""Rum training with M8 splits, sampling, scheduling and checkpoint selection."""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import pandas as pd
from transformers import set_seed

from pipelines.forecasting import distributed as _dist
from pipelines.forecasting.checkpoints import load_weights, save_weights
from pipelines.forecasting.config import deep_merge, dump_config, raw_override
from pipelines.forecasting.data import load_full, load_presplit, load_train_valid
from pipelines.forecasting.rows import row_stations
from pipelines.forecasting.training import (
    CheckpointCallback,
    EmaCallback,
    EvaluationCallback,
    ForecastTrainer,
    WeightEma,
    build_training_args,
)
from pipelines.transformer_dataset import UnifiedDataset

from .config import load_config
from .transformer_models import TransformersAdaptedM4RumModel
from .transformer_trainer import validate


class RumEvalCallback(EvaluationCallback):
    """Rum validation scorer using the public checkpoint-selection protocol."""

    def _score(self):
        scores, _, _ = validate(self.model, self.ds_val, self.collate)
        self.last_extra = {
            "nanwang_short_D+1": scores["nanwang_short_D+1"],
            "nrmse": scores["nrmse_pct"],
        }
        self.last_unseen = (
            scores["nanwang_unseen"] if scores["n_unseen_stations"] else None
        )
        return scores["nrmse_pct" if self.metric == "nrmse" else self.metric]


def train_m4_rum(
    config=None,
    splits_dir=None,
    out_dir=None,
    max_steps=None,
    tag=None,
    refit=False,
    train_parquet=None,
    save_every=None,
    valid_parquet=None,
):
    if (splits_dir is None) == (train_parquet is None):
        raise ValueError("Choose exactly one of --splits-dir and --train-parquet")
    if valid_parquet is not None and (train_parquet is None or refit):
        raise ValueError("--valid-parquet requires --train-parquet and forbids --refit")
    override = raw_override(config)
    if out_dir is not None:
        override = deep_merge(override, {"run": {"out_dir": str(out_dir)}})
    if save_every is not None:
        override = deep_merge(
            override, {"train": {"schedule": {"save_every": save_every}}}
        )
    cfg = load_config(override)
    out = Path(cfg["run"]["out_dir"]).resolve()
    cfg["run"]["out_dir"] = str(out)
    if max_steps is not None:
        cfg["train"]["schedule"]["max_steps"] = max_steps
    if tag is not None:
        cfg["run"]["tag"] = tag
    refit = refit or (train_parquet is not None and valid_parquet is None)
    args = build_training_args(cfg, out)
    if refit and args.max_steps <= 0:
        raise ValueError("Refit requires positive max_steps")
    _ = args.device
    if args.n_gpu > 1:
        raise ValueError(
            "Use torchrun / accelerate for multiple GPUs; DataParallel is unsupported"
        )
    main = _dist.is_main()
    out.mkdir(parents=True, exist_ok=True)
    # Never accidentally resume an existing run; seed before parameter construction.
    set_seed(args.seed)
    model = TransformersAdaptedM4RumModel(
        deep_merge(cfg, {"train": {"weights_path": None}})
    ).to(args.device)

    def load(reuse):
        if valid_parquet is not None:
            return load_train_valid(
                train_parquet,
                valid_parquet,
                cfg,
                model.station_meta,
                out / ".data",
                reuse,
            )
        if train_parquet is not None:
            return load_full(
                train_parquet, cfg, model.station_meta, out / ".data", reuse
            )
        return load_presplit(
            splits_dir, cfg, model.station_meta, out / ".data", reuse, refit
        )

    if main:
        loaded = load(False)
    _dist.barrier()
    if not main:
        loaded = load(True)
    ds_train, ds_val, manifest = loaded
    cfg["features"]["cov_present"] = model.fe.check_columns(ds_train.column_names)
    known = sorted(
        set(row_stations(ds_train, cfg["dataset"]["station_id_col"]).tolist())
    )
    cfg["dataset"]["known_stations"] = known
    cfg["train"]["weights_path"] = "${run.out_dir}/model.safetensors"
    model.cfg["dataset"]["known_stations"] = known
    collate = functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)
    if main:
        dump_config(cfg, out / "model_config.yaml")
        (out / "split_manifest.json").write_text(
            json.dumps(
                {
                    "generated_by": "forecast_backbones.m4_rum.train",
                    "refit": refit,
                    "source": manifest,
                    "counts": {
                        "train": len(ds_train),
                        "val": 0 if ds_val is None else len(ds_val),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[m4_rum] params={model.n_params():,}, train={len(ds_train)}, "
            f"val={0 if ds_val is None else len(ds_val)}, refit={refit}"
        )
    schedule = cfg["train"]["schedule"]
    decay = cfg["train"]["ema_decay"]
    ema = WeightEma(model, decay) if decay is not None else None
    cb = (
        None
        if refit
        else RumEvalCallback(
            model=model,
            ds_val=ds_val,
            collate=collate,
            cfg=cfg,
            best_path=out / "model.safetensors",
            patience=cfg["train"]["early_stop"]["patience"],
            log_path=out / "train_log.json",
            eval_every_steps=schedule["eval_every_steps"],
            ema=ema,
        )
    )
    checkpoint = (
        CheckpointCallback(model, out, schedule["save_every"], cfg, ema)
        if main and schedule["save_every"]
        else None
    )
    callbacks = (
        ([EmaCallback(ema, model)] if ema else [])
        + ([checkpoint] if checkpoint else [])
        + ([cb] if cb else [])
    )
    model.train()
    trainer = ForecastTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        data_collator=collate,
        callbacks=callbacks,
        sampling=schedule["sampling"],
        station_col=cfg["dataset"]["station_id_col"],
        ts_col=cfg["dataset"]["ts_col"],
    )
    start = time.time()
    trainer.train()
    if not main:
        return {}
    step = int(trainer.state.global_step)
    if checkpoint and step not in checkpoint.saved:
        checkpoint.save(step)
    table, result = None, None
    scores = {
        k: float("nan")
        for k in ("nanwang_short_D+1", "nrmse_pct", "nanwang_seen", "nanwang_unseen")
    }
    if refit:
        if ema:
            ema.load_into(model)
        save_weights(model, out / "model.safetensors")
        history, best_step = trainer.state.log_history, step
        log = {"history": history, "refit": True, "best_step": best_step}
    else:
        if cb.best_step is None:
            raise ValueError(
                "No finite validation score; check issue rows and target coverage"
            )
        load_weights(model, out / "model.safetensors")
        scores, table, result = validate(model, ds_val, collate)
        result.to_parquet(out / "result_val.parquet", index=False)
        history, best_step = cb.history, cb.best_step
        log = json.loads((out / "train_log.json").read_text())
    summary = {
        "tag": cfg["run"]["tag"],
        **scores,
        "params": model.n_params(),
        "best_step": best_step,
        "n_train_rows": len(ds_train),
        "n_val_rows": 0 if ds_val is None else len(ds_val),
        "train_minutes": (time.time() - start) / 60,
    }
    reports = out / "reports"
    reports.mkdir(exist_ok=True)
    pd.DataFrame([summary]).to_csv(reports / "summary.csv", index=False)
    if table is not None:
        table.to_csv(reports / "station_table.csv", index=False)
    log.update(
        final=summary, checkpoints=[] if checkpoint is None else checkpoint.saved
    )
    (out / "train_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "summary": summary,
        "station_table": table,
        "history": history,
        "result": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train M4-Rum with the M8 data/evaluation protocol"
    )
    parser.add_argument("--config", default=None)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--splits-dir")
    inputs.add_argument("--train-parquet", nargs="+")
    parser.add_argument(
        "--valid-parquet",
        nargs="+",
        help="Validation files/directories/globs; requires --train-parquet",
    )
    parser.add_argument("--out")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--tag")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument(
        "--station-meta", help="Override station table and clear column mapping"
    )
    parser.add_argument("--hint-col", help="Override foundation forecast column")
    a = parser.parse_args(argv)
    cfg = raw_override(a.config)
    if a.station_meta:
        cfg = deep_merge(
            cfg,
            {"dataset": {"station_meta": a.station_meta, "station_meta_cols": None}},
        )
    if a.hint_col:
        cfg = deep_merge(cfg, {"features": {"hint_col": a.hint_col}})
    train_m4_rum(
        cfg,
        a.splits_dir,
        a.out,
        a.max_steps,
        a.tag,
        a.refit,
        a.train_parquet,
        a.save_every,
        a.valid_parquet,
    )


if __name__ == "__main__":
    main()
