"""CPU 接口冒烟测试：生成生产格式 mock，训练 a17，评估并通过 unified_inference 推理。

从仓库根目录用指定 Python 执行 -m forecast_backbones.m4_rum.smoke_production
--out /path/to/new_directory。保留原始数据、配置、命令、日志、权重及检查结果。
保持 a17 网络和损失，只缩短训练预算；mock 分数不代表模型精度。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from pipelines.forecasting.config import dump_config

from .config import load_config
from .demo_unified_inference import predict_dataframe
from .evaluate import load_member


ROOT = Path(__file__).resolve().parents[2]
CASES = ("csv_station", "capacity_scalar", "capacity_list")


def mock_frame(start, stations, mode, capacity_col):
    rows = []
    for number in stations:
        cap = 80.0 + number * 20
        for issue in pd.date_range(start, periods=24, freq="1h"):
            history = pd.date_range(end=issue, periods=672, freq="15min")
            future = pd.date_range(
                start=issue + pd.Timedelta(minutes=15), periods=192, freq="15min"
            )

            def sun(times):
                hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
                return np.maximum(np.sin((hour - 6) * np.pi / 12), 0).astype(np.float32)

            past, ahead = sun(history), sun(future)
            power = cap * 0.8 * past
            # hint 用历史最后一天持续预测，不从未来真实标签复制。
            hint = np.tile(power[-96:], 2)
            row_cap = cap + (5.0 if issue.hour % 2 else 0.0)
            capacity = (
                ([row_cap] + [row_cap * 1.5] * 191)
                if mode == "capacity_list"
                else row_cap
            )
            rows.append(
                {
                    "timestamp_win": issue,
                    "station": f"plant_mock_{number}",
                    "observe_power": power,
                    "observe_power_future": cap * 0.8 * ahead,
                    "general_GHI": 900 * past,
                    "general_TEMP": 15 + 10 * past,
                    "general_GHI_predict": 900 * ahead,
                    "general_TEMP_predict": 15 + 10 * ahead,
                    "tsfm_predict": hint,
                    # CSV 模式故意提供不一致的数据容量，核对实际使用 CSV。
                    capacity_col: 1.0 if mode == "csv_station" else capacity,
                }
            )
    return pd.DataFrame(rows)


def run_case(case, root):
    folder = root / case
    folder.mkdir()
    cap_col = "installed_capacity" if case == "capacity_list" else "cap_power_on"
    train = mock_frame("2025-06-08", (1, 2), case, cap_col)
    valid = mock_frame("2025-06-20", (1, 2, 3), case, cap_col)
    # 已见站 + 未见站，故意打乱行序并使用不足一个 batch 的尾批。
    live = valid.iloc[[58, 10, 34, 59, 11]].drop(columns="observe_power_future")
    for name, frame in (("train", train), ("valid", valid), ("live", live)):
        frame.to_parquet(folder / f"{name}.parquet", index=False)
    csv = folder / "stations.csv"
    pd.DataFrame(
        {
            "station": [f"plant_mock_{i}" for i in (1, 2, 3)],
            "rated_mw": [100.0, 120.0, 140.0],
        }
    ).to_csv(csv, index=False)

    cfg = load_config(
        ROOT / "forecast_backbones/m4_rum/config/arch_sweep/a17_all_large_station.yaml"
    )
    cfg["dataset"].update(
        station_meta=str(csv) if case == "csv_station" else None,
        station_meta_cols={"station": "station", "cap_mw": "rated_mw"}
        if case == "csv_station"
        else None,
        station_key="station",
        capacity_col=cap_col,
        known_stations=None,
    )
    # 列表容量案例同时走每 epoch 验证，其他案例按 step 验证。
    epoch = case == "capacity_list"
    cfg["train"]["schedule"].update(
        max_steps=12 if epoch else 3,
        epochs_max=2,
        batch_size=8,
        warmup_steps=0,
        eval_every_steps=None if epoch else 1,
        save_every=6 if epoch else 1,
        logging_steps=1,
        progress_bar=False,
    )
    cfg["eval"]["batch_size"] = 8
    cfg["train"]["weights_path"] = None
    run = folder / "run"
    cfg["run"].update(out_dir=str(run), seeds=[0])
    dump_config(cfg, folder / "config.yaml")
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    env.setdefault("HF_HOME", str(root / ".hf"))
    env.setdefault("HF_DATASETS_CACHE", str(root / ".hf/datasets"))
    commands = [
        [
            "bash",
            "train_shell/train_m4_rum.sh",
            "--venv-python",
            sys.executable,
            "--allow-cpu",
            "--cpu-threads",
            "1",
            "--config",
            str(folder / "config.yaml"),
            "--train-parquet",
            str(folder / "train.parquet"),
            "--valid-parquet",
            str(folder / "valid.parquet"),
            "--out",
            str(run),
        ],
        [
            "bash",
            "eval_shell/eval_m4_rum.sh",
            "--venv-python",
            sys.executable,
            "--runs",
            str(run),
            "--test",
            str(folder / "valid.parquet"),
            "--device",
            "cpu",
            "--full",
            "--no-progress",
            "--out",
            str(folder / "evaluation"),
        ],
        [
            sys.executable,
            "-m",
            "forecast_backbones.m4_rum.demo_unified_inference",
            "--run",
            str(run),
            "--input",
            str(folder / "live.parquet"),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--out",
            str(folder / "predictions.parquet"),
        ],
    ]
    (folder / "commands.json").write_text(
        json.dumps(commands, indent=2, ensure_ascii=False)
    )
    for stage, command in zip(("train", "evaluate", "infer"), commands):
        print(f"[{case}] {stage}", flush=True)
        log = folder / f"{stage}.log"
        with log.open("w") as output:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(f"{stage} failed, see {log}")

    model = load_member(run, device="cpu")
    predicted = pd.read_parquet(folder / "predictions.parquet")
    evaluated = pd.read_parquet(folder / "evaluation/result_test.parquet")
    expected = live[["timestamp_win", "station"]].merge(
        evaluated, on=["timestamp_win", "station"], validate="one_to_one", sort=False
    )
    actual = np.stack(predicted.observe_power_predict)
    assert actual.shape == (len(live), 192) and np.isfinite(actual).all()
    assert predicted.station.tolist() == live.station.tolist()
    assert predicted.timestamp_win.tolist() == live.timestamp_win.tolist()
    np.testing.assert_allclose(
        actual, np.stack(expected.observe_power_predict), rtol=1e-5, atol=1e-4
    )
    np.testing.assert_allclose(
        actual,
        np.stack(predict_dataframe(model, live, batch_size=2).observe_power_predict),
        rtol=1e-5,
        atol=1e-4,
    )
    capacities = model.station_meta.capacities(live, model.station_meta.identity(live))
    expected_caps = (
        live.station.map(
            {"plant_mock_1": 100, "plant_mock_2": 120, "plant_mock_3": 140}
        ).to_numpy()
        if case == "csv_station"
        else np.array([v[0] if isinstance(v, list) else v for v in live[cap_col]])
    )
    np.testing.assert_array_equal(capacities, expected_caps)
    assert (actual >= 0).all() and (actual <= capacities[:, None] + 1e-4).all()
    slots = model.core.station_slots(model.station_meta.identity(live), "cpu").numpy()
    assert (slots[live.station.eq("plant_mock_3")] == 0).all()
    assert (slots[~live.station.eq("plant_mock_3")] > 0).all()
    log = json.loads((run / "train_log.json").read_text())
    best_step = int(log["final"]["best_step"])
    best = load_member(run / "checkpoints" / f"step_{best_step:05d}")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, best.state_dict()[name], rtol=0, atol=0)
    assert np.isfinite(log["final"]["nanwang_short_D+1"])
    return {
        "case": case,
        "status": "passed",
        "params": model.n_params(),
        "train_rows": len(train),
        "validation_rows": len(valid),
        "prediction_shape": list(actual.shape),
        "best_step": best_step,
        "validation_count": len(log["history"]),
        "validation_cadence": "epoch" if epoch else "step",
        "max_abs_eval_infer_difference": float(
            np.max(np.abs(actual - np.stack(expected.observe_power_predict)))
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    results = []
    for case in CASES:
        try:
            results.append(run_case(case, root))
        except Exception as error:
            results.append({"case": case, "status": "failed", "error": str(error)})
        (root / "summary.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False)
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    if any(row["status"] != "passed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
