"""Small, causal synthetic fixtures; no pretrained model or real data required."""

import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.transformer_dataset import UnifiedDataset

torch.set_num_threads(1)


def make_frame(start="2025-06-08", days=2, stations=(1, 2), every=60):
    rows = []
    for sid in stations:
        for issue in pd.date_range(
            start, periods=days * 1440 // every, freq=f"{every}min"
        ):
            history = pd.date_range(end=issue, periods=672, freq="15min")
            future = pd.date_range(
                start=issue + pd.Timedelta(minutes=15), periods=192, freq="15min"
            )

            def irradiance(times):
                hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
                return np.maximum(np.sin((hour - 6) * np.pi / 12), 0).astype(np.float32)

            hist, fut = irradiance(history), irradiance(future)
            power = 80 * hist
            # Repeat the last observed daily cycle for all future target slots.
            hint = np.tile(power[-96:], 2)
            rows.append(
                {
                    "timestamp_win": issue,
                    "station_id": sid,
                    "cap_power_on": 1.0,
                    "observe_power": power,
                    "observe_power_future": 80 * fut,
                    "general_ghi": 1000 * hist,
                    "general_ghi_predict": 1000 * fut,
                    "general_temp": 15 + hist * 10,
                    "general_temp_predict": 15 + fut * 10,
                    "chronos": hint,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def cfg(tmp_path):
    meta = tmp_path / "stations.csv"
    meta.write_text(
        "station_id,station,lat,lon,alt_m,cap_mw\n1,A,,,,100\n2,B,,,,100\n3,C,,,,200\n"
    )
    return {
        "dataset": {"station_meta": str(meta)},
        "model": {
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "head_dropout": 0.0,
            "embedding_dim": 16,
        },
        "run": {"out_dir": str(tmp_path / "run")},
        "train": {
            "schedule": {
                "batch_size": 4,
                "max_steps": 2,
                "warmup_steps": 0,
                "eval_every_steps": 1,
                "save_every": 1,
                "logging_steps": 1,
                "progress_bar": False,
            },
        },
        "eval": {"batch_size": 16},
    }


@pytest.fixture
def collate():
    return functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)


@pytest.fixture
def batch(collate):
    frame = make_frame(days=1, every=120).iloc[[5, 6]]
    ds = UnifiedDataset(
        frame,
        timestamp_cols=["timestamp_win"],
        entity_cols=[],
        label_cols=["observe_power_future"],
        for_hf_trainer=True,
    )
    return collate([ds[0], ds[1]])


@pytest.fixture
def splits(tmp_path):
    path = tmp_path / "splits"
    path.mkdir()
    make_frame(days=1, stations=(1,)).to_parquet(path / "train.parquet", index=False)
    make_frame(start="2025-06-20", days=1).to_parquet(
        path / "valid.parquet", index=False
    )
    (path / "split_manifest.json").write_text(json.dumps({"plan": "synthetic test"}))
    return path
