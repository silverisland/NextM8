# -*- coding: utf-8 -*-
# forecast_backbones/m5/envelope.py
"""窗内历史日形包络（2026-10-18，M5 C1 / C2 与 Gated-LoRA / HydraLoRA 的形势特征共用；纯 numpy，不需要坐标）。
history_envelope 与 M8 特征构造里的同名函数逐位相同（M5 自带一份，不跨包 import）。

  C1 包络通道   E[t]   = 上下文 7 天里同一时刻槽功率的 q 分位（环形平滑），按时刻槽铺满上下文 + 未来 → 一条 past_future 协变量；
  C2 NWP 日形   P[t]   = E[t] × clip(GHI_nwp[t] / GHI_env[t], 0, clip_max)，GHI_env 是同一窗口 GHI 历史列的同法包络
                        （"今天的 NWP 比这几天的晴空 GHI 低多少，就把功率包络压多少"，非仿射，RevIN 抹不掉）；
  形势特征 z    未来窗（白天步）晴空指数 k = clip(GHI_nwp / GHI_env, 0, 1.5) 的 [均值, 标准差, 平均逐步变化] + has_coords。
包络只用上下文（历史）里的值，未来部分按时刻槽复用历史包络，不碰未来真值。"""
from __future__ import annotations

import warnings

import numpy as np

N_REGIME = 4                                      # 形势特征维数：k_mean / k_std / k_dif / has_coords


def history_envelope(y: np.ndarray, slot0: np.ndarray, n_day: int, quantile: float, smooth: int) -> tuple[np.ndarray, np.ndarray]:
    """窗内历史的按时刻槽日包络。y [B, L] 原始单位、不可见处 NaN；slot0 [B] 历史第一步的时刻槽号；L 须是 n_day 的整数倍。
    → (env [B, n_day] 每槽 quantile 分位（全缺的槽 = 0）并做宽 smooth 的环形滑动平均, days_with_data [B] 有可见点的天数)。"""
    B, L = y.shape
    if L % n_day != 0:
        raise ValueError(f"history_envelope 要求 context_len 是一天步数的整数倍：L={L}, n_day={n_day}")
    D = L // n_day
    idx = (np.arange(L)[None, :] - np.asarray(slot0, np.int64)[:, None]) % L          # rolled[b, p] = y[b, (p - slot0) % L] → 槽号 p % n_day
    rolled = np.take_along_axis(np.asarray(y, float), idx, axis=1).reshape(B, D, n_day)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        env = np.nanquantile(rolled, float(quantile), axis=1) if np.isfinite(rolled).any() else np.full((B, n_day), np.nan)
    env = np.where(np.isfinite(env), env, 0.0)
    k = int(smooth)
    if k > 1:
        w = np.ones(k) / k
        env = np.stack([np.convolve(np.concatenate([e[-(k // 2):], e, e[: k // 2]]), w, mode="valid") for e in env], 0)
    days = np.isfinite(rolled).any(axis=2).sum(axis=1)
    return np.clip(env, 0.0, None), days


def envelope_rows(hist: np.ndarray, slot: np.ndarray, n_day: int, quantile: float, smooth: int) -> np.ndarray:
    """hist [B, L]（NaN = 缺）、slot [B, T] 每步时刻槽（T = L + H）→ 包络按时刻槽铺到 [B, T]。"""
    env, _ = history_envelope(hist, slot[:, 0], n_day, quantile, smooth)
    return np.take_along_axis(env, np.asarray(slot, np.int64), axis=1)


def nwp_ratio(ghi_full: np.ndarray, ghi_env: np.ndarray, floor: float, clip_max: float) -> np.ndarray:
    """GHI_nwp / GHI_env，包络低于 floor（夜间 / 晨昏）处记 0；NaN 记 0。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ghi_env > floor, np.asarray(ghi_full, float) / np.maximum(ghi_env, floor), 0.0)
    return np.clip(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), 0.0, float(clip_max))


def regime_features(k_fut: np.ndarray, day_fut: np.ndarray, has_coords: np.ndarray) -> np.ndarray:
    """k_fut [B, H] 晴空指数、day_fut [B, H] bool 白天步 → z [B, N_REGIME] float32（没有白天步的行 = 0）。"""
    k = np.where(day_fut, k_fut, np.nan)
    n = day_fut.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(k, axis=1)
        std = np.nanstd(k, axis=1)
        dif = np.nanmean(np.abs(np.diff(k, axis=1)), axis=1)
    z = np.stack([mean, std, dif, np.asarray(has_coords, float)], axis=1)
    z[n == 0, :3] = 0.0
    return np.nan_to_num(z, nan=0.0).astype(np.float32)
