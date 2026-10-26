"""Retained recipes and independent production replay/retraining comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.forecasting.config import dump_config, raw_override
from pipelines.forecasting.data import expand_parquets

from .config import load_config
from .evaluate import evaluate

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path(__file__).with_name("production") / "experiments.yaml"
METRICS = ("station_mean_nanwang_short_D+1", "station_mean_nanwang_ultra_short")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
    )


def code_version():
    def git(*args):
        try:
            p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        except FileNotFoundError:
            # Deployment bundles may contain source files without the Git CLI.
            return None
        return p.stdout.strip() if p.returncode == 0 else None

    sources = [
        *Path(__file__).parent.glob("*.py"),
        *ROOT.joinpath("pipelines/forecasting").glob("*.py"),
        *ROOT.joinpath("pipelines").glob("*.py"),
        *ROOT.joinpath("interfaces").glob("*.py"),
    ]
    sources += [
        ROOT / name
        for name in (
            "train_shell/train_m4_rum.sh",
            "train_shell/compare_m4_rum.sh",
            "eval_shell/eval_m4_rum.sh",
            "eval_shell/infer_m4_rum.sh",
        )
        if (ROOT / name).is_file()
    ]
    return {
        "commit": git("rev-parse", "HEAD"),
        "worktree_status": git("status", "--short"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)},
    }


def input_files(paths):
    """File inventory, not a claim of a full dataset content hash."""
    return [
        {
            "path": str(Path(p).resolve()),
            "bytes": Path(p).stat().st_size,
            "mtime_ns": Path(p).stat().st_mtime_ns,
        }
        for p in paths
    ]


def row_signature(frame):
    """Compare ordered rows, scoring capacities and raw truth across candidates."""
    h = hashlib.sha256()
    for _, row in frame.iterrows():
        h.update(
            np.asarray(
                [pd.Timestamp(row.timestamp_win).value, int(row.station_id)],
                dtype="<i8",
            ).tobytes()
        )
        h.update(np.asarray([row.cap], dtype="<f8").tobytes())
        truth = np.asarray(row.observe_power_future, dtype="<f4").copy()
        truth[np.isnan(truth)] = np.nan
        h.update(truth.tobytes())
    return h.hexdigest()


def read_catalog(path=CATALOG):
    data = raw_override(path)
    if data.get("version") != 1:
        raise ValueError("Unsupported experiment catalog version")
    entries = data["experiments"]
    result = {}
    for entry in entries:
        name = entry["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in result:
            raise ValueError(f"Invalid or duplicate experiment ID: {name}")
        config = (ROOT / entry["config"]).resolve()
        if not config.is_file():
            raise FileNotFoundError(config)
        load_config(config)  # Also rejects retired smoothing recipes.
        result[name] = entry
    return result


def add_deltas(rows, baseline):
    by_seed = {
        r["seed"]: r
        for r in rows
        if r["experiment"] == baseline and r["status"] == "ok"
    }
    for row in rows:
        ref = by_seed.get(row["seed"])
        comparable = (
            row["status"] == "ok"
            and ref is not None
            and row.get("row_signature") == ref.get("row_signature")
        )
        row["comparable_to_baseline"] = comparable
        for metric in METRICS:
            value, base = row.get(metric), (ref or {}).get(metric)
            row[metric + "_delta_pp"] = (
                value - base
                if comparable
                and value is not None
                and base is not None
                and np.isfinite(value)
                and np.isfinite(base)
                else None
            )
    return rows


def checkpoint_path(args, mapping, name, seed):
    if mapping is not None:
        if name not in mapping:
            raise ValueError(f"run-map missing experiment {name}")
        item = mapping[name]
        if isinstance(item, dict):
            item = item.get(str(seed), item.get(seed))
        elif len(args.seeds) != 1:
            raise ValueError("Multiple seeds require per-seed mappings in run-map")
        if not item:
            raise ValueError(f"run-map missing {name} seed {seed}")
        return Path(item).resolve()
    return Path(args.run_root).resolve() / name / f"seed_{seed}"


def station_comparison(rows, baseline):
    tables = {}
    for row in rows:
        if row["status"] == "ok":
            path = Path(row["report_dir"]) / "evaluation/reports/station_metrics.csv"
            tables[row["experiment"], row["seed"]] = pd.read_csv(path).set_index(
                "station_id"
            )
    records = []
    for row in rows:
        table = tables.get((row["experiment"], row["seed"]))
        if table is None:
            continue
        ref = tables.get((baseline, row["seed"]))
        for sid, station in table.iterrows():
            result = {
                "experiment": row["experiment"],
                "seed": row["seed"],
                "station_id": sid,
                "comparable_to_baseline": row["comparable_to_baseline"],
            }
            for metric in ("nanwang_short_D+1", "nanwang_ultra_short"):
                value = station.get(metric, np.nan)
                result[metric] = value
                base = (
                    ref.loc[sid].get(metric, np.nan)
                    if ref is not None and sid in ref.index
                    else np.nan
                )
                result[metric + "_delta_pp"] = (
                    value - base if row["comparable_to_baseline"] else np.nan
                )
            records.append(result)
    return pd.DataFrame(records)


def run(args):
    catalog = read_catalog(args.catalog)
    names = list(catalog) if args.experiments == ["all"] else args.experiments
    if len(names) != len(set(names)) or set(names) - catalog.keys():
        raise ValueError("Experiments must be unique catalog IDs (or 'all')")
    if args.mode == "list":
        for name in names:
            e = catalog[name]
            print(f"{name:14} {e['group']:12} {e['config']}  {e['description']}")
        return []
    if not args.out or (not args.run_root and not args.run_map):
        raise ValueError(
            "--out and --run-root (or --run-map for evaluate) are required"
        )
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if args.mode == "train" and (args.run_map or not args.train or not args.valid):
        raise ValueError(
            "train requires --train and --valid; --run-map is evaluation-only"
        )
    if args.mode == "evaluate" and (not args.test or args.baseline not in names):
        raise ValueError(
            "evaluate requires --test and baseline included in --experiments"
        )
    if args.mode == "evaluate" and args.hint_col:
        raise ValueError(
            "Use checkpoint hint mapping for evaluation; rename upstream columns instead"
        )
    if args.no_station_meta and args.station_meta:
        raise ValueError("--no-station-meta and --station-meta are mutually exclusive")
    if args.mode == "evaluate" and (
        args.no_station_meta or args.capacity_col or args.station_key
    ):
        raise ValueError(
            "Capacity source and station key must come from the saved checkpoint for evaluation"
        )
    if args.station_meta and not Path(args.station_meta).is_file():
        raise FileNotFoundError(args.station_meta)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Comparison output must be empty: {out}")
    mapping = raw_override(args.run_map) if args.run_map else None
    files = {
        key: [str(Path(p).resolve()) for p in expand_parquets(value)]
        for key, value in (
            ("train", args.train),
            ("valid", args.valid),
            ("test", args.test),
        )
        if value
    }
    if args.mode == "train" and set(files["train"]) & set(files["valid"]):
        raise ValueError("Training and validation files overlap")
    out.mkdir(parents=True, exist_ok=True)
    inventory = {k: input_files(v) for k, v in files.items()}
    write_json(
        out / "comparison_manifest.json",
        {
            "mode": args.mode,
            "experiments": [catalog[n] for n in names],
            "seeds": args.seeds,
            "baseline": args.baseline,
            "input_files": inventory,
            "catalog_sha256": sha256(args.catalog),
            "run_map": mapping,
            "station_meta_override": args.station_meta,
            "no_station_meta": args.no_station_meta,
            "capacity_col_override": args.capacity_col,
            "station_key_override": args.station_key,
            "hint_col_override": args.hint_col,
            "gpus": args.gpus,
            "python": sys.executable,
            "inference_engine": "pipelines.pipeline_wrapper_inference.unified_inference",
            "code": code_version(),
        },
    )
    rows = []
    for name in names:
        for seed in args.seeds:
            record = {"experiment": name, "seed": seed, "status": "failed"}
            started = time.perf_counter()
            trial = out / name / f"seed_{seed}"
            trial.mkdir(parents=True)
            try:
                if {k: input_files(v) for k, v in files.items()} != inventory:
                    raise ValueError(
                        "Input files changed during comparison; use a frozen replay dataset"
                    )
                checkpoint = checkpoint_path(args, mapping, name, seed)
                record["run"] = str(checkpoint)
                recipe = ROOT / catalog[name]["config"]
                record["recipe_sha256"] = sha256(recipe)
                if args.mode == "train":
                    if checkpoint.exists() and any(checkpoint.iterdir()):
                        raise ValueError(
                            f"Refusing to overwrite training run: {checkpoint}"
                        )
                    cfg = load_config(recipe)
                    cfg["run"].update(
                        seeds=[seed], tag=f"{name}_s{seed}", out_dir=str(checkpoint)
                    )
                    cfg["train"]["weights_path"] = None
                    if args.station_meta:
                        cfg["dataset"].update(
                            station_meta=str(Path(args.station_meta).resolve()),
                            station_meta_cols=None,
                        )
                    if args.hint_col:
                        cfg["features"]["hint_col"] = args.hint_col
                    if args.no_station_meta:
                        cfg["dataset"].update(station_meta=None, station_meta_cols=None)
                    if args.capacity_col:
                        cfg["dataset"]["capacity_col"] = args.capacity_col
                    if args.station_key:
                        cfg["dataset"]["station_key"] = args.station_key
                    dump_config(cfg, trial / "requested_config.yaml")
                    meta_path = (
                        Path(cfg["dataset"]["station_meta"]).resolve()
                        if cfg["dataset"]["station_meta"] is not None
                        else None
                    )
                    write_json(
                        trial / "station_meta_manifest.json",
                        {
                            "path": str(meta_path) if meta_path else None,
                            "sha256": sha256(meta_path) if meta_path else None,
                            "columns": cfg["dataset"]["station_meta_cols"],
                            "capacity_source": "station_meta"
                            if meta_path
                            else "data_first_value",
                            "capacity_col": cfg["dataset"]["capacity_col"],
                            "station_key": cfg["dataset"]["station_key"],
                        },
                    )
                    command = [
                        "bash",
                        str(ROOT / "train_shell/train_m4_rum.sh"),
                        "--venv-python",
                        sys.executable,
                        "--gpus",
                        args.gpus,
                        "--port",
                        str(args.port),
                        "--config",
                        str(trial / "requested_config.yaml"),
                        "--train_parquet",
                        *files["train"],
                        "--valid_parquet",
                        *files["valid"],
                    ]
                    if args.allow_cpu:
                        command.append("--allow-cpu")
                    if args.cpu_threads:
                        command.extend(["--cpu_threads", str(args.cpu_threads)])
                    write_json(trial / "command.json", command)
                    print(
                        f"Training {name} seed={seed}; log: {trial / 'train.log'}",
                        flush=True,
                    )
                    with (trial / "train.log").open("w") as log:
                        subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                else:
                    print(f"Evaluating {name} seed={seed}: {checkpoint}", flush=True)
                    config_path, weights_path = (
                        checkpoint / "model_config.yaml",
                        checkpoint / "model.safetensors",
                    )
                    # Bind the comparison to the actual checkpoint, not the catalog recipe.
                    actual = load_config(config_path)
                    metadata_source = (
                        args.station_meta or actual["dataset"]["station_meta"]
                    )
                    metadata_path = (
                        Path(metadata_source).resolve()
                        if metadata_source is not None
                        else None
                    )
                    write_json(
                        trial / "checkpoint_manifest.json",
                        {
                            "config": actual,
                            "config_sha256": sha256(config_path),
                            "weights_sha256": sha256(weights_path),
                            "station_meta": str(metadata_path)
                            if metadata_path
                            else None,
                            "station_meta_sha256": sha256(metadata_path)
                            if metadata_path
                            else None,
                            "capacity_source": "station_meta"
                            if metadata_path
                            else "data_first_value",
                            "saved_seed": actual["run"].get("seeds"),
                        },
                    )
                    if actual["run"].get("seeds", [0]) != [seed]:
                        raise ValueError("Checkpoint seed differs from comparison seed")
                    report = evaluate(
                        [str(checkpoint)],
                        files["test"],
                        trial / "evaluation",
                        full=True,
                        batch_size=args.batch_size,
                        device=args.device,
                        progress=not args.no_progress,
                        station_meta=args.station_meta,
                    )
                    if input_files(files["test"]) != inventory["test"]:
                        raise ValueError("Test files changed during inference")
                    record.update(report["summary"])
                    record["missing_metrics"] = ",".join(
                        key
                        for key in METRICS
                        if not np.isfinite(record.get(key, np.nan))
                    )
                    record["row_signature"] = row_signature(report["result"])
                    members = pd.read_csv(trial / "evaluation/reports/members.csv")
                    record.update(
                        {
                            k: members[k].iloc[0]
                            for k in members.columns
                            if k.endswith("_rate")
                        }
                    )
                    del report
                record["status"] = "ok"
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"FAILED {name} seed={seed}: {record['error']}",
                    file=sys.stderr,
                    flush=True,
                )
            record["seconds"] = time.perf_counter() - started
            record["report_dir"] = str(trial)
            write_json(trial / "status.json", record)
            rows.append(record)
            pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    if args.mode == "evaluate":
        add_deltas(rows, args.baseline)
        station_comparison(rows, args.baseline).to_csv(
            out / "station_comparison.csv", index=False
        )
        for row in rows:
            write_json(Path(row["report_dir"]) / "status.json", row)
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    return rows


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("list", "train", "evaluate"), default="list")
    p.add_argument("--catalog", default=str(CATALOG))
    p.add_argument("--experiments", nargs="+", default=["q6", "a16", "a17"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--baseline", default="q6")
    p.add_argument("--run-root")
    p.add_argument(
        "--run-map", help="YAML mapping ID to checkpoint directory or {seed: directory}"
    )
    p.add_argument(
        "--out",
        help="Empty directory for this comparison's manifests, logs and reports",
    )
    p.add_argument("--train", nargs="+")
    p.add_argument("--valid", nargs="+")
    p.add_argument("--test", nargs="+")
    p.add_argument("--station-meta", "--station_meta")
    p.add_argument("--no-station-meta", action="store_true")
    p.add_argument("--capacity-col")
    p.add_argument("--station-key")
    p.add_argument("--hint-col", "--hint_col")
    p.add_argument("--gpus", default="0")
    p.add_argument("--port", type=int, default=29700)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--cpu-threads", type=int)
    p.add_argument("--device")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv=None):
    rows = run(build_parser().parse_args(argv))
    if any(r["status"] != "ok" or r.get("missing_metrics") for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
