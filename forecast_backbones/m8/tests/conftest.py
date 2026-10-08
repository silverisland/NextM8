import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]          # NextGenPVForecast/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import functools
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
    """物理尺度的合成行表：GHI ≈ 1000·bell·k_day W/m²，功率 = cap·0.85·bell·k_day·(1+ε)，温度 20 + 8·bell。
    NWP 预报 = 真实 GHI 加一点平滑偏差（模拟 NWP 误差）。"""
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
        # 模拟 TimesFM 单变量预测：昨天同时刻功率 × (1+噪声)（只依赖历史，无未来泄漏）
        tf_hint = np.clip(np.roll(np.nan_to_num(power), 96) * (1 + rng.normal(0, 0.1, len(grid))), 0, None)
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
                "timesfm_predict": tf_hint[fut].astype(np.float32),
            })
    return pd.DataFrame(rows)


def write_splits(tmp_path, n_train_days=4, n_val_days=2, seed=0):
    """train/valid.parquet + split_manifest.json（与 pipelines.time_splits 产物同构的最小集）。"""
    d = Path(tmp_path) / "splits"; d.mkdir(parents=True, exist_ok=True)
    tr = make_frame(n_days=n_train_days, issue_every_min=60, seed=seed, start="2025-06-08")
    va = make_frame(n_days=n_val_days, issue_every_min=60, seed=seed + 1, start="2025-06-20")
    va["split_block"] = 0
    tr.to_parquet(d / "train.parquet", index=False); va.to_parquet(d / "valid.parquet", index=False)
    (d / "split_manifest.json").write_text(json.dumps({"generated_by": "pipelines.time_splits",
        "plan": {"val_blocks": [["2025-06-20", "2025-06-22"]], "test_start": "2025-07-01"}}), encoding="utf-8")
    return d


@pytest.fixture
def station_meta(tmp_path):
    return write_station_meta(tmp_path)


@pytest.fixture
def small_cfg(station_meta, tmp_path):
    return {"dataset": {"station_meta": str(station_meta)},
            "model": {"d_model": 32, "d_ff": 64, "head_hidden": 32, "n_heads": 2, "blocks": ["conv", "deltanet"], "pos_in_patch_dim": 8},
            "run": {"out_dir": str(tmp_path / "run")}}


@pytest.fixture
def dataset():
    from pipelines.transformer_dataset import UnifiedDataset
    df = make_frame(n_days=2, issue_every_min=120)
    return UnifiedDataset(df, timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"], for_hf_trainer=True)


@pytest.fixture
def collate():
    from pipelines.transformer_dataset import UnifiedDataset
    return functools.partial(UnifiedDataset.collate_fn, for_hf_trainer=True)


@pytest.fixture
def batch(dataset, collate):
    import torch
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, collate_fn=collate, shuffle=False)
    return next(iter(loader))
