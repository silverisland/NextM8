# -*- coding: utf-8 -*-
# forecast_backbones/m7/decision.py（2026-10-20，M7 自带，不引用 m5 / m6）
"""M7 决策层：Timestral 的 21 档分位 → 排序 → 钳到 [0, cap] → 插值成 N 点等概率网格 → 上报点。纯 numpy。

分位函数 Q(τ) 在已知档位之间线性插值，网格取 τ_i = (i + 0.5) / N（i = 0..N-1）。
档位外（τ < 最低档 / τ > 最高档）按 tail 处理："flat" 取端点值；"linear" 沿最外两档斜率外推，再钳回 [0, cap]。

上报点（point）：
  median    : Q(0.5)
  mean      : 网格均值 ≈ ∫Q(τ)dτ（分布均值，RMSE 最优点）
  p_star    : 南网短期（RMSE 型，分母 max(y, 0.2C)）最优点 Σ y·w / Σ w，w = 1/max(y, floor·C)²
  p_star_l1 : 南网超短期（MAE 型）最优点 = w = 1/max(y, floor·C) 的加权中位数
  quantile:τ: Q(τ)

注意：网格是对同一组档位做线性插值，N 再大也不会多出信息；N 只决定离散积分的精度（200 → 300 变化在 1e-4 量级）。
真正多出信息的是档位本身：21 档比 M5 的 9 档多了 0.01 / 0.05 / 0.95 / 0.99 两端，p* 的权重压在低端，下尾直接影响结果。"""
from __future__ import annotations

import numpy as np

POINTS = ("median", "mean", "p_star", "p_star_l1")
TAILS = ("flat", "linear")


def taus(n_grid: int) -> np.ndarray:
    """N 点等概率网格的中点 τ_i = (i + 0.5) / N。"""
    n = int(n_grid)
    if n < 2:
        raise ValueError(f"decision.grid 至少 2（得到 {n_grid}）")
    return (np.arange(n) + 0.5) / n


def interp_quantiles(q, levels, at, tail: str = "flat") -> np.ndarray:
    """q [..., K]（沿末维非降）在 levels 上的分位 → [..., len(at)] 在 τ = at 处的值（档间线性插值）。"""
    if tail not in TAILS:
        raise ValueError(f"decision.tail 只能是 {TAILS}（得到 {tail!r}）")
    q = np.asarray(q, float)
    lv = np.asarray(levels, float)
    if lv.ndim != 1 or lv.size < 2 or q.shape[-1] != lv.size:
        raise ValueError(f"分位末维 {q.shape[-1]} 与档位数 {lv.size} 不一致（至少 2 档）")
    if np.any(np.diff(lv) <= 0):
        raise ValueError(f"档位必须严格递增：{list(lv)}")
    at = np.asarray(at, float)
    j = np.clip(np.searchsorted(lv, at, side="right") - 1, 0, lv.size - 2)
    w = (at - lv[j]) / (lv[j + 1] - lv[j])
    if tail == "flat":
        w = np.clip(w, 0.0, 1.0)
    return q[..., j] + (q[..., j + 1] - q[..., j]) * w


def prepare(q, cap) -> np.ndarray:
    """排序（修交叉）+ 钳到 [0, cap]。cap 可广播到 q.shape[:-1]；cap=None 只截负值。"""
    q = np.sort(np.asarray(q, float), axis=-1)
    hi = np.inf if cap is None else np.broadcast_to(np.asarray(cap, float), q.shape[:-1])[..., None]
    return np.clip(q, 0.0, hi)


def grid(q, levels, cap, n_grid: int = 200, tail: str = "flat") -> np.ndarray:
    """[..., K] → [..., N] 等概率网格上的分位函数值；linear 外推出的值钳回 [0, cap]（flat 时 q 已钳过，原样）。"""
    y = interp_quantiles(q, levels, taus(n_grid), tail)
    if tail == "linear":
        hi = np.inf if cap is None else np.broadcast_to(np.asarray(cap, float), y.shape[:-1])[..., None]
        y = np.clip(y, 0.0, hi)
    return y


def _floor(y, cap, floor_ratio: float) -> np.ndarray:
    if cap is None:
        raise ValueError("p_star / p_star_l1 需要装机容量 cap（站表 cap_mw）")
    return floor_ratio * np.broadcast_to(np.asarray(cap, float), y.shape[:-1])[..., None]


def p_star(y, cap, floor_ratio: float = 0.2) -> np.ndarray:
    """y [..., N] 网格 → [...]：Σ y w / Σ w，w = 1/max(y, floor·C)²。"""
    w = 1.0 / np.maximum(y, _floor(y, cap, floor_ratio)) ** 2
    return (y * w).sum(-1) / w.sum(-1)


def p_star_l1(y, cap, floor_ratio: float = 0.2) -> np.ndarray:
    """y [..., N]（沿末维非降）→ [...]：w = 1/max(y, floor·C) 的加权中位数（累计权重首次 ≥ 一半处的 y）。"""
    w = 1.0 / np.maximum(y, _floor(y, cap, floor_ratio))
    cw = np.cumsum(w, axis=-1)
    k = (cw >= 0.5 * cw[..., -1:]).argmax(axis=-1)
    return np.take_along_axis(y, k[..., None], axis=-1)[..., 0]


def point(q, levels, cap, method: str = "p_star", n_grid: int = 200, tail: str = "flat",
          floor_ratio: float = 0.2) -> np.ndarray:
    """q [..., K] 原始分位（真实单位）→ [...] 上报点。q 先过 prepare（排序 + [0, cap]）。"""
    q = prepare(q, cap)
    if method == "median":
        return interp_quantiles(q, levels, [0.5], tail)[..., 0]
    if isinstance(method, str) and method.startswith("quantile:"):
        return interp_quantiles(q, levels, [float(method.split(":", 1)[1])], tail)[..., 0]
    y = grid(q, levels, cap, n_grid, tail)
    if method == "mean":
        return y.mean(-1)
    if method == "p_star":
        return p_star(y, cap, floor_ratio)
    if method == "p_star_l1":
        return p_star_l1(y, cap, floor_ratio)
    raise ValueError(f"未知 decision.point: {method!r}（可选 {POINTS} 或 quantile:<τ>）")
