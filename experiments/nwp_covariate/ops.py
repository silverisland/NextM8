# -*- coding: utf-8 -*-
"""ops.py —— 特征算子库（单位换算 + 衍生量），全部是纯 numpy 函数，对 [B, L] 数组逐行操作。

约定：每个标准量是一对数组 (hist [B, context_len], fut [B, pred_len])，用 Pair 表示。
衍生算子签名统一为 op(ctx, **args) -> Pair，ctx 是 BuildContext（时间戳、坐标、容量、已有列）。
args 里引用其他标准量时写列名字符串，算子内部用 ctx.get(name) 取 Pair。

config.derived 里 `{name, op, args}` 的 op 就是 OPS 注册表里的键。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

STEP_MIN = 15
HAURWITZ_A = 1098.0
HAURWITZ_B = 0.059


# ============================================================================ 数据结构
@dataclass
class Pair:
    hist: np.ndarray            # [B, context_len] float32
    fut: np.ndarray             # [B, pred_len]    float32；history_only 的量 fut 为全 NaN

    def full(self) -> np.ndarray:
        return np.concatenate([self.hist, self.fut], axis=1)

    @staticmethod
    def from_full(arr: np.ndarray, context_len: int) -> "Pair":
        return Pair(arr[:, :context_len].astype(np.float32), arr[:, context_len:].astype(np.float32))

    def copy(self) -> "Pair":
        return Pair(self.hist.copy(), self.fut.copy())


@dataclass
class BuildContext:
    """一批起报窗的公共上下文。"""
    ts_win: np.ndarray                          # [B] datetime64[s]，起报时刻 S
    context_len: int
    pred_len: int
    lat: np.ndarray                             # [B] 站点纬度（缺 -> NaN）
    lon: np.ndarray                             # [B]
    alt_m: np.ndarray                           # [B]
    cap: np.ndarray                             # [B] 装机容量
    tz_offset_h: float = 8.0
    cols: Dict[str, Pair] = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def get(self, name: str) -> Pair:
        if name not in self.cols:
            raise KeyError(f"标准量 {name!r} 不存在；已有: {sorted(self.cols)[:30]}")
        return self.cols[name]

    def has(self, name: str) -> bool:
        return name in self.cols

    def warn(self, msg: str):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def timestamps(self) -> np.ndarray:
        """[B, context_len+pred_len] datetime64[s]：历史最后一点 = S；未来第 i 点 = S + (i+1)·step。"""
        return window_timestamps(self.ts_win, self.context_len, self.pred_len)


def window_timestamps(ts_win, context_len: int, pred_len: int, step_min: int = STEP_MIN) -> np.ndarray:
    s = np.asarray(pd.to_datetime(ts_win).values.astype("datetime64[s]")).astype("int64").reshape(-1, 1)
    offs = np.concatenate([np.arange(-(context_len - 1), 1), np.arange(1, pred_len + 1)]) * (step_min * 60)
    return (s + offs[None, :]).astype("datetime64[s]")


# ============================================================================ 单位换算（按 columns.*.kind）
def _diff_rate(full: np.ndarray, step_sec: float) -> np.ndarray:
    """自起报累积量 -> 速率：相邻差分 / 秒。第 0 点用第 1 点补。负差分（新起报周期重置）置 0。"""
    d = np.diff(full, axis=1, prepend=full[:, :1])
    d[:, 0] = d[:, 1] if full.shape[1] > 1 else 0.0
    d = np.where(d < 0, 0.0, d)
    return d / step_sec


def convert_kind(pair: Pair, kind: str, resolution_min: int = STEP_MIN, step_min: int = STEP_MIN) -> Pair:
    """把原始单位统一到：辐照 W/m²、温度 °C、云量 0–1、降水 mm/步长。历史与未来拼起来处理再拆开。"""
    full = pair.full().astype(np.float64)
    if kind in ("wm2", "celsius", "frac", "ms", "raw", "code"):
        out = full
    elif kind == "kelvin":
        out = full - 273.15
    elif kind == "pct":
        out = full / 100.0
    elif kind == "accum_jm2":
        # ECMWF 自起报累积 J/m²。DS 层若已把 1h 累积内插到 15min，差分 / (15·60) 得 W/m²。
        out = _diff_rate(full, step_min * 60.0)
    elif kind == "mm_accum":
        out = _diff_rate(full, 1.0)                      # 累积 mm -> 每步长 mm
    elif kind == "avg6h":
        # GFS 6h 周期重置的时段均值 A_n（窗口 0–n h）：h_n = n·A_n − (n−1)·A_{n−1}。
        # 需要按小时节奏反平均；15min 网格上按 4 点一小时处理。
        out = _deaverage_6h(full, step_min)
    else:
        raise ValueError(f"未知 kind={kind!r}")
    return Pair.from_full(out, pair.hist.shape[1])


def _deaverage_6h(full: np.ndarray, step_min: int) -> np.ndarray:
    per_h = 60 // step_min
    B, L = full.shape
    out = full.copy()
    # 以小时为单位的窗口序号 n = 1..6 循环；假设序列第 0 点对应某周期的第 1 小时（无法从数据得知，
    # 真实接入 GFS 时按起报时刻对齐 n 的相位：见 config.columns.sources.gfs.phase_hours）。
    for j in range(per_h, L):
        n = ((j // per_h) % 6) + 1
        if n == 1:
            continue
        out[:, j] = n * full[:, j] - (n - 1) * full[:, j - per_h]
    return np.maximum(out, 0.0)


# ============================================================================ 基础算子
def _pair_map(fn: Callable[[np.ndarray], np.ndarray], *pairs: Pair) -> Pair:
    fulls = [p.full().astype(np.float64) for p in pairs]
    out = fn(*fulls)
    return Pair.from_full(out, pairs[0].hist.shape[1])


def op_sub(ctx: BuildContext, a: str, b: str) -> Pair:
    return _pair_map(lambda x, y: x - y, ctx.get(a), ctx.get(b))


def op_add(ctx: BuildContext, a: str, b: str) -> Pair:
    return _pair_map(lambda x, y: x + y, ctx.get(a), ctx.get(b))


def op_mul(ctx: BuildContext, a: str, b: str) -> Pair:
    return _pair_map(lambda x, y: x * y, ctx.get(a), ctx.get(b))


def op_scale(ctx: BuildContext, x: str, k: float = 1.0, offset: float = 0.0) -> Pair:
    return _pair_map(lambda v: v * k + offset, ctx.get(x))


def op_ratio(ctx: BuildContext, num: str, den: str, min_den: float = 1e-6, fill: float = 0.0) -> Pair:
    def f(n, d):
        ok = d >= min_den
        return np.where(ok, n / np.where(ok, d, 1.0), fill)
    return _pair_map(f, ctx.get(num), ctx.get(den))


def op_gt(ctx: BuildContext, x: str, thr: float) -> Pair:
    return _pair_map(lambda v: (v > thr).astype(np.float64), ctx.get(x))


def op_wind_speed(ctx: BuildContext, u: str, v: str) -> Pair:
    return _pair_map(lambda a, b: np.sqrt(a * a + b * b), ctx.get(u), ctx.get(v))


def op_accum_diff(ctx: BuildContext, x: str, step_sec: float = 1.0) -> Pair:
    return _pair_map(lambda v: _diff_rate(v, step_sec), ctx.get(x))


# ============================================================================ 太阳几何 / 晴空
def _spencer(doy: np.ndarray):
    B = 2.0 * np.pi * (doy - 1.0) / 365.0
    decl = (0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B) - 0.006758 * np.cos(2 * B)
            + 0.000907 * np.sin(2 * B) - 0.002697 * np.cos(3 * B) + 0.001480 * np.sin(3 * B))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
                    - 0.014615 * np.cos(2 * B) - 0.040849 * np.sin(2 * B))
    return decl, eot


def cos_zenith(ts: np.ndarray, lat: np.ndarray, lon: np.ndarray, tz_offset_h: float = 8.0) -> np.ndarray:
    """ts [B, L] datetime64（墙钟）、lat/lon [B] -> cosθz [B, L]（地平线下为负）。Spencer(1971)。"""
    t = pd.DatetimeIndex(ts.reshape(-1))
    doy = t.dayofyear.to_numpy(float).reshape(ts.shape)
    clock_h = (t.hour.to_numpy(float) + t.minute.to_numpy(float) / 60.0).reshape(ts.shape)
    decl, eot = _spencer(doy)
    phi = np.deg2rad(lat.astype(float))[:, None]
    solar_h = clock_h + (4.0 * (lon.astype(float)[:, None] - 15.0 * tz_offset_h) + eot) / 60.0
    omega = np.deg2rad(15.0 * (solar_h - 12.0))
    return np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega)


def clear_sky_ghi(ts: np.ndarray, lat: np.ndarray, lon: np.ndarray, tz_offset_h: float = 8.0) -> np.ndarray:
    """Haurwitz 晴空 GHI W/m²，夜间 0。"""
    cz = cos_zenith(ts, lat, lon, tz_offset_h)
    out = np.zeros_like(cz)
    day = cz > 0
    out[day] = HAURWITZ_A * cz[day] * np.exp(-HAURWITZ_B / cz[day])
    return out


def clock_clear_sky(ts: np.ndarray, day_start: str = "06:15", day_end: str = "20:00", level: float = 800.0) -> np.ndarray:
    """缺坐标站的退化晴空：白天窗内常数 level，窗外 0。"""
    t = pd.DatetimeIndex(ts.reshape(-1))
    tod = (t.hour * 60 + t.minute).to_numpy().reshape(ts.shape)
    lo = int(day_start[:2]) * 60 + int(day_start[3:]); hi = int(day_end[:2]) * 60 + int(day_end[3:])
    return np.where((tod >= lo) & (tod <= hi), level, 0.0)


def op_clear_sky_ghi(ctx: BuildContext, day_start: str = "06:15", day_end: str = "20:00") -> Pair:
    ts = ctx.timestamps()
    have = np.isfinite(ctx.lat) & np.isfinite(ctx.lon)
    out = np.zeros(ts.shape)
    if have.any():
        out[have] = clear_sky_ghi(ts[have], ctx.lat[have], ctx.lon[have], ctx.tz_offset_h)
    if (~have).any():
        ctx.warn(f"{int((~have).sum())} 行缺坐标，晴空 GHI 退化为时钟几何")
        out[~have] = clock_clear_sky(ts[~have], day_start, day_end)
    return Pair.from_full(out, ctx.context_len)


def op_kt(ctx: BuildContext, ghi: str, clear: str, min_clear: float = 50.0, night_fill: float = 0.0,
          kt_max: float = 1.3) -> Pair:
    """晴空指数：clear >= min_clear 处 ghi/clear（截到 kt_max），否则 night_fill。"""
    def f(g, c):
        ok = c >= min_clear
        kt = np.where(ok, g / np.where(ok, c, 1.0), night_fill)
        return np.clip(kt, 0.0, kt_max)
    return _pair_map(f, ctx.get(ghi), ctx.get(clear))


def op_cloud_ghi(ctx: BuildContext, clear: str, tcc: str, a: float = 0.75, b: float = 3.4) -> Pair:
    """Kasten-Czeplak：GHI = GHI_clear × (1 − a·N^b)，N 为 0–1 云量。"""
    return _pair_map(lambda c, n: c * (1.0 - a * np.clip(n, 0, 1) ** b), ctx.get(clear), ctx.get(tcc))


def op_cell_temp(ctx: BuildContext, ghi: str, temp: str, wind: Optional[str] = None, model: str = "faiman",
                 noct: float = 45.0, u0: float = 25.0, u1: float = 6.84) -> Pair:
    """组件温度。faiman: T_amb + GHI/(u0 + u1·ws)；noct: T_amb + (NOCT−20)/800·GHI。"""
    g, t = ctx.get(ghi).full().astype(float), ctx.get(temp).full().astype(float)
    if model == "faiman":
        if wind is not None and ctx.has(wind):
            ws = ctx.get(wind).full().astype(float)
        else:
            ctx.warn("cell_temp: 无风速列，faiman 按 ws=2 m/s")
            ws = np.full_like(g, 2.0)
        tc = t + g / (u0 + u1 * np.nan_to_num(ws, nan=2.0))
    elif model == "noct":
        tc = t + (noct - 20.0) / 800.0 * g
    else:
        raise ValueError(f"cell_temp.model={model!r}")
    return Pair.from_full(tc, ctx.context_len)


def op_p_phys(ctx: BuildContext, ghi: str, t_cell: Optional[str] = None, gamma: float = -0.004,
              cap: Optional[str] = None, t_ref: float = 25.0) -> Pair:
    """物理功率代理：cap × GHI/1000 × [1 + γ (T_cell − 25)]，与 observe_power 同量纲。"""
    g = ctx.get(ghi).full().astype(float)
    eff = 1.0
    if t_cell is not None and ctx.has(t_cell):
        eff = 1.0 + gamma * (ctx.get(t_cell).full().astype(float) - t_ref)
    c = ctx.cap.astype(float)[:, None]
    p = c * np.clip(g / 1000.0, 0.0, 1.2) * eff
    return Pair.from_full(np.clip(p, 0.0, None), ctx.context_len)


# ============================================================================ 多源聚合
def _members(ctx: BuildContext, members: list) -> list:
    have = [m for m in members if ctx.has(m)]
    miss = [m for m in members if not ctx.has(m)]
    if miss:
        ctx.warn(f"consensus/spread: 成员缺失 {miss}，用剩余 {have}")
    if not have:
        raise KeyError(f"consensus/spread: 成员全部缺失 {members}")
    return have


def op_consensus(ctx: BuildContext, members: list, mode: str = "equal", reference: Optional[str] = "ghi_obs",
                 min_valid: int = 1, **_ignored) -> Pair:
    """多源共识。equal: 均值；median: 中位数；skill: 按各成员历史段对 reference 的 RMSE 倒数加权
    （逐行，reference 缺失时退化为 equal）。NaN 成员逐点跳过，有效成员数 < min_valid 处置 NaN。"""
    have = _members(ctx, members)
    stack = np.stack([ctx.get(m).full().astype(float) for m in have])        # [M, B, L]
    valid = np.isfinite(stack)
    if mode == "median":
        out = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
    else:
        if mode == "skill" and reference is not None and ctx.has(reference):
            ref = ctx.get(reference).hist.astype(float)                       # [B, C]
            w = []
            for m in have:
                h = ctx.get(m).hist.astype(float)
                ok = np.isfinite(h) & np.isfinite(ref)
                rmse = np.sqrt(np.where(ok.sum(1) > 0, np.nansum(np.where(ok, (h - ref) ** 2, 0), 1) / np.maximum(ok.sum(1), 1), np.nan))
                w.append(1.0 / np.maximum(rmse, 1e-3))
            w = np.stack(w)                                                   # [M, B]
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.where(w.sum(0, keepdims=True) > 0, w, 1.0)
            w = w[:, :, None]
        else:
            if mode == "skill":
                ctx.warn(f"consensus: reference {reference!r} 不存在，skill 退化为 equal")
            w = np.ones((len(have), 1, 1))
        wv = np.where(valid, w, 0.0)
        out = np.nansum(np.where(valid, stack, 0.0) * wv, axis=0) / np.where(wv.sum(0) > 0, wv.sum(0), np.nan)
    nvalid = valid.sum(0)
    out = np.where(nvalid >= min_valid, out, np.nan)
    return Pair.from_full(out, ctx.context_len)


def op_spread(ctx: BuildContext, members: list) -> Pair:
    have = _members(ctx, members)
    stack = np.stack([ctx.get(m).full().astype(float) for m in have])
    out = np.nanstd(stack, axis=0) if len(have) > 1 else np.zeros(stack.shape[1:])
    return Pair.from_full(out, ctx.context_len)


# ============================================================================ 注册表
OPS: Dict[str, Callable[..., Pair]] = {
    "sub": op_sub, "add": op_add, "mul": op_mul, "scale": op_scale, "ratio": op_ratio, "gt": op_gt,
    "wind_speed": op_wind_speed, "accum_diff": op_accum_diff,
    "clear_sky_ghi": op_clear_sky_ghi, "kt": op_kt, "cloud_ghi": op_cloud_ghi,
    "cell_temp": op_cell_temp, "p_phys": op_p_phys,
    "consensus": op_consensus, "spread": op_spread,
}


def run_op(ctx: BuildContext, op: str, args: Optional[dict] = None) -> Pair:
    if op not in OPS:
        raise KeyError(f"未知算子 {op!r}；可用: {sorted(OPS)}")
    args = dict(args or {})
    # "${data.cap_col}" 这类占位在 p_phys 里不需要列名，直接忽略 cap 参数
    args.pop("cap", None) if op == "p_phys" else None
    return OPS[op](ctx, **args)
