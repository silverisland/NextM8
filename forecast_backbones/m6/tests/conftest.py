import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]          # NextGenPVForecast/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import types
import numpy as np
import pandas as pd
import pytest

CAP = {1001: 500.0, 1002: 300.0}
LAT = {1001: (30.0, 101.0, 4600.0), 1002: (30.0, 101.0, 4600.0)}


def write_station_meta(tmp_path, drop_coords_for=()):
    p = tmp_path / "station_meta.csv"
    lines = ["station_id,station,lat,lon,alt_m,cap_mw"]
    for s, (la, lo, al) in LAT.items():
        if s in drop_coords_for:
            lines.append(f"{s},S{s},,,,{CAP[s]}")
        else:
            lines.append(f"{s},S{s},{la},{lo},{al},{CAP[s]}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _bell(ts: pd.DatetimeIndex, cap: float) -> np.ndarray:
    h = ts.hour + ts.minute / 60.0
    return np.where((h >= 6) & (h <= 19), cap * np.sin(np.pi * (h - 6) / 13.0), 0.0).clip(min=0)


def make_synthetic_frame(n_days=3, stations=(1001, 1002), issue_every_min=15, seed=0, L=672, H=192,
                         start="2025-06-08", sigma=0.3):
    """每行一个起报：历史 L、未来 H。a_clean = bell·k_day；真值 = a_clean·(1+ε)；
    general_ghi(_predict) = 10·a_clean，让 FakeEvaluator 的中位数正好等于 a_clean。"""
    rng = np.random.default_rng(seed)
    rows = []
    grid = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=7), periods=(7 + n_days + 2) * 96, freq="15min")
    for s in stations:
        cap = CAP[s]
        k_day = np.repeat(rng.uniform(0.4, 1.0, len(grid) // 96 + 1), 96)[: len(grid)]
        a_clean = _bell(grid, cap) * k_day
        eps = np.clip(rng.normal(0, sigma, len(grid)), -0.9, 0.9)
        truth = a_clean * (1 + eps)
        ghi = 10.0 * a_clean
        issues = pd.date_range(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=n_days) - pd.Timedelta(minutes=15),
                               freq=f"{issue_every_min}min")
        for S in issues:
            i = grid.get_loc(S)
            if i - L + 1 < 0 or i + H >= len(grid):
                continue
            hist = slice(i - L + 1, i + 1); fut = slice(i + 1, i + 1 + H)
            rows.append({
                "timestamp_win": S, "station_id": s, "cap_power_on": cap,
                "observe_power": truth[hist].astype(np.float32),
                "observe_power_future": truth[fut].astype(np.float32),
                "general_ghi": ghi[hist].astype(np.float32), "general_ghi_predict": ghi[fut].astype(np.float32),
                "general_temp": np.full(L, 20.0, np.float32), "general_temp_predict": np.full(H, 20.0, np.float32),
                "GHI_solargis": ghi[hist].astype(np.float32), "ghi_solargis_predict": ghi[fut].astype(np.float32),
                "temp_solargis": np.full(L, 20.0, np.float32), "temp_solargis_predict": np.full(H, 20.0, np.float32),
            })
    return pd.DataFrame(rows)


class FakeEvaluator:
    """TimesFM3Evaluator 替身：中位数 = 0.1×第一协变量末 H 段 + 1e-6×context 均值 + 1e-6×其余协变量末 H 段之和
    （依赖 contexts 与全部协变量槽位，不只依赖第一个，用于测试侧的未来真值泄漏侦测）；
    分位 = 中位数 × [0.6..1.4]。self.calls：每次 predict_batch 调用记一条，值为该次收到的全部数组
    （contexts + past_future_covariates 各元素 + past_only_covariates 各元素），供测试做投毒溯源
    （检查任何一个数组是否含 labels 里的未来真值）。"""
    FACTORS = np.linspace(0.6, 1.4, 9)

    def __init__(self, with_quantiles=True):
        self.with_quantiles = with_quantiles
        self.calls = []

    def predict_batch(self, contexts, horizon, past_future_covariates=None, past_only_covariates=None,
                      return_quantiles=True, **kw):
        contexts = list(contexts)
        pf = list(past_future_covariates) if past_future_covariates is not None else None
        po = list(past_only_covariates) if past_only_covariates is not None else None
        recorded = [np.array(a, copy=True) for a in contexts]
        if pf is not None:
            recorded += [np.array(a, copy=True) for a in pf]
        if po is not None:
            recorded += [np.array(a, copy=True) for a in po]
        self.calls.append(recorded)
        outs = []
        for i, ctx in enumerate(contexts):
            ctx_arr = np.asarray(ctx, dtype=np.float64)
            if pf is not None:
                pf_i = np.asarray(pf[i], dtype=np.float64)
                med = 0.1 * pf_i[0, -horizon:] + 1e-6 * ctx_arr.mean()
                if pf_i.shape[0] > 1:
                    med = med + 1e-6 * pf_i[1:, -horizon:].sum(axis=0)
            else:
                med = np.tile(ctx_arr.reshape(-1)[-96:], 2)[:horizon] + 1e-6 * ctx_arr.mean()
            med = np.maximum(med, 0.0).astype(np.float32)
            q = (med[:, None] * self.FACTORS[None, :]).astype(np.float32) if (self.with_quantiles and return_quantiles) else None
            outs.append(types.SimpleNamespace(ts_id=i, forecast=med, quantiles=q))
        return iter(outs)


@pytest.fixture(scope="session")
def tiny_ckpt_dir(tmp_path_factory):
    """随机小 TimesFM3（8/16 patch、32 维、2 层）存成 HF 目录一次，供本会话全部测试复用同一份权重。"""
    pytest.importorskip("timesfm3")
    from forecast_backbones.m6.backbone import TorchBackbone
    d = tmp_path_factory.mktemp("tiny_tfm3")
    TorchBackbone.tiny_for_tests(seed=0).model.save_pretrained(str(d))
    return d


@pytest.fixture
def tiny_evaluator(tiny_ckpt_dir):
    """每个测试从 tiny_ckpt_dir 重新构造一个 TimesFM3Evaluator：evaluator 路径与 torch 路径共用同一份
    存盘权重（一致性测试不受权重来源影响），但 evaluator 实例本身逐测试全新——后续任务会向
    evaluator.model 注入/开关 LoRA 并训练，session 级共享实例会让那些变更泄漏进其他一致性测试。"""
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    return TimesFM3Evaluator(ModelConfig(checkpoint_path=str(tiny_ckpt_dir), per_core_batch_size=4, device="cpu"))
