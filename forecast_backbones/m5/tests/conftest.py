import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]          # NextGenPVForecast/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import functools
import json
import numpy as np
import pandas as pd
import pytest

CAP = {1001: 500.0, 1002: 300.0}
COORDS = {1001: (30.0, 101.0, 4600.0), 1002: (30.0, 101.0, 4600.0)}
L, H = 672, 192


def write_station_meta(tmp_path, drop_coords_for=()):
    p = Path(tmp_path) / "station_meta.csv"
    lines = ["station_id,station,lat,lon,alt_m,cap_mw"]
    for s, (la, lo, al) in COORDS.items():
        lines.append(f"{s},S{s},,,,{CAP[s]}" if s in drop_coords_for else f"{s},S{s},{la},{lo},{al},{CAP[s]}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _bell(ts: pd.DatetimeIndex) -> np.ndarray:
    h = ts.hour + ts.minute / 60.0
    return np.where((h >= 7) & (h <= 19), np.sin(np.pi * (h - 7) / 12.0), 0.0).clip(min=0)


def make_frame(n_days=3, stations=(1001, 1002), issue_every_min=60, seed=0, start="2025-06-08", sigma=0.15, nan_frac=0.0):
    """物理尺度的合成行表（与 M8 测试同一构造）：GHI ≈ 1000·bell·k_day W/m²，功率 = cap·0.85·bell·k_day·(1+ε)，温度 20 + 8·bell。"""
    rng = np.random.default_rng(seed)
    rows = []
    grid = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=7), periods=(7 + n_days + 2) * 96, freq="15min")
    bell = _bell(grid)
    for s in stations:
        cap = CAP[s]
        k_day = np.repeat(rng.uniform(0.4, 1.0, len(grid) // 96 + 1), 96)[: len(grid)]
        ghi_true = 1000.0 * bell * k_day
        power = cap * 0.85 * bell * k_day * (1 + np.clip(rng.normal(0, sigma, len(grid)), -0.8, 0.8))
        power = np.where(bell > 0, power, 0.0)
        bias = np.repeat(rng.uniform(-0.1, 0.1, len(grid) // 96 + 1), 96)[: len(grid)]
        ghi_nwp = np.clip(ghi_true * (1 + bias), 0, None)
        temp = 20.0 + 8.0 * bell
        if nan_frac > 0:
            power = power.copy(); power[rng.random(len(grid)) < nan_frac] = np.nan
        issues = pd.date_range(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=n_days) - pd.Timedelta(minutes=15),
                               freq=f"{issue_every_min}min")
        for S in issues:
            i = grid.get_loc(S)
            if i - L + 1 < 0 or i + H >= len(grid):
                continue
            hist = slice(i - L + 1, i + 1); fut = slice(i + 1, i + 1 + H)
            rows.append({
                "timestamp_win": S, "station_id": s, "cap_power_on": cap,
                "observe_power": power[hist].astype(np.float32), "observe_power_future": power[fut].astype(np.float32),
                "general_ghi": ghi_nwp[hist].astype(np.float32), "general_ghi_predict": ghi_nwp[fut].astype(np.float32),
                "general_temp": temp[hist].astype(np.float32), "general_temp_predict": temp[fut].astype(np.float32),
            })
    return pd.DataFrame(rows)


def write_splits(tmp_path, n_train_days=4, n_val_days=2, seed=0, train_stations=(1001, 1002), val_stations=(1001, 1002)):
    """train/valid.parquet + split_manifest.json（与 pipelines.time_splits 产物同构的最小集）。"""
    d = Path(tmp_path) / "splits"; d.mkdir(parents=True, exist_ok=True)
    tr = make_frame(n_days=n_train_days, stations=train_stations, issue_every_min=60, seed=seed, start="2025-06-08")
    va = make_frame(n_days=n_val_days, stations=val_stations, issue_every_min=60, seed=seed + 1, start="2025-06-20")
    va["split_block"] = 0
    tr.to_parquet(d / "train.parquet", index=False); va.to_parquet(d / "valid.parquet", index=False)
    (d / "split_manifest.json").write_text(json.dumps({"generated_by": "pipelines.time_splits",
        "plan": {"val_blocks": [["2025-06-20", "2025-06-22"]], "test_start": "2025-07-01"}}), encoding="utf-8")
    return d


def full_cfg(tmp_path, ckpt, **over):
    """full 模式最小配置：小 TimesFM-3 权重 + 两站站表 + 很短的训练预算；over 按段深合并。"""
    from forecast_backbones.m5.config import deep_merge
    base = {"dataset": {"station_meta": str(write_station_meta(tmp_path))},
            "timesfm3": {"checkpoint": str(ckpt), "device": "cpu", "per_core_batch_size": 8},
            "run": {"out_dir": str(Path(tmp_path) / "run")},
            "train": {"schedule": {"batch_size": 4, "max_steps": 4, "eval_every_steps": 2, "warmup_steps": 1,
                                   "logging_steps": 1, "progress_bar": False},
                      "early_stop": {"patience": 3}},
            "eval": {"batch_size": 8}}
    return deep_merge(base, over)


@pytest.fixture(scope="session")
def tiny_ckpt_dir(tmp_path_factory):
    """随机小 TimesFM-3（8/16 patch、32 维、2 层）存成 HF 目录一次，全部测试复用同一份权重。"""
    pytest.importorskip("timesfm3")
    from forecast_backbones.m5.backbone import TorchBackbone
    d = tmp_path_factory.mktemp("tiny_tfm3")
    TorchBackbone.tiny_for_tests(seed=0).model.save_pretrained(str(d))
    return d


@pytest.fixture
def tiny_evaluator(tiny_ckpt_dir):
    """每个测试一个全新的 evaluator（后续测试会往 evaluator.model 注入 LoRA，不能跨测试共享实例）。"""
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    return TimesFM3Evaluator(ModelConfig(checkpoint_path=str(tiny_ckpt_dir), per_core_batch_size=4, device="cpu"))


@pytest.fixture
def collate():
    from pipelines.transformer_dataset import UnifiedDataset
    return functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)


@pytest.fixture
def dataset():
    from pipelines.transformer_dataset import UnifiedDataset
    df = make_frame(n_days=2, issue_every_min=120)
    return UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"], for_hf_trainer=True)


@pytest.fixture
def batch(dataset, collate):
    import torch
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, collate_fn=collate, shuffle=False)
    return next(iter(loader))
