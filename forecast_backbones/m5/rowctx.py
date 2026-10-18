# -*- coding: utf-8 -*-
# forecast_backbones/m5/rowctx.py
"""每批的站点上下文：站号 → 固定容量（站表 cap_mw）；窗口时间戳 → 太阳高度角 / 晴空 GHI / 晴空功率 / 夜间。
M5 默认（D8）只用 cap 与 station_id；几何量只进 aux 供报表，features.derived 或 decision.physics.clear_sky_cap_ratio 打开时
才影响预测。SolarCache 自 M8 复制（pvlib 按 (站, 日) 缓存 96 步）；RowCtx 字段与 M8 同名，evaluate.run_inference 原样可用。"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .solar import window_timestamps, solar_geometry, clock_pseudo_geometry, StationMeta, station_cap_mw
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m5.solar import window_timestamps, solar_geometry, clock_pseudo_geometry, StationMeta, station_cap_mw

_DEGRADED_WARNED: set = set()


def _to_np(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


class SolarCache:
    """(station_id, 日期) → 该日 15 分钟网格上的 (elev_deg[96], ghi_clear[96])。时间戳不在网格上的行直接算、不缓存。"""

    def __init__(self, station_meta: StationMeta, tz: str, step_min: int = 15, max_entries: int = 400_000):
        self.meta, self.tz, self.step_s = station_meta, tz, int(step_min) * 60
        self.n_day = 86400 // self.step_s
        self.max_entries = int(max_entries)
        self._elev: dict = {}
        self._ghi: dict = {}

    def geometry(self, sid: int, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """ts [n, T] datetime64[s]（naive 本地墙钟）→ (elev[n,T], ghi_clear[n,T], degraded)。"""
        if not self.meta.has_coords(sid):
            e, g = clock_pseudo_geometry(ts)
            return e, g, True
        m = self.meta.get(sid)
        epoch = ts.astype("datetime64[s]").astype(np.int64)
        if (epoch % self.step_s).any():
            e, g = solar_geometry(ts, m["lat"], m["lon"], m["alt_m"], self.tz)
            return e, g, False
        days = (epoch // 86400).astype(np.int64)
        slots = ((epoch - days * 86400) // self.step_s).astype(np.int64)
        uniq = np.unique(days)
        missing = [int(d) for d in uniq if (sid, int(d)) not in self._elev]
        if missing:
            if len(self._elev) > self.max_entries:
                self._elev.clear(); self._ghi.clear()
            grid = (np.asarray(missing, np.int64)[:, None] * 86400 + np.arange(self.n_day)[None, :] * self.step_s)
            e, g = solar_geometry(grid.astype("datetime64[s]"), m["lat"], m["lon"], m["alt_m"], self.tz)
            for i, d in enumerate(missing):
                self._elev[(sid, d)] = e[i].astype(np.float32); self._ghi[(sid, d)] = g[i].astype(np.float32)
        etab = np.stack([self._elev[(sid, int(d))] for d in uniq]); gtab = np.stack([self._ghi[(sid, int(d))] for d in uniq])
        idx = np.searchsorted(uniq, days)
        return etab[idx, slots].astype(float), gtab[idx, slots].astype(float), False


@dataclass
class RowCtx:
    station_id: np.ndarray      # [B] int
    cap: np.ndarray             # [B]
    elev_deg: np.ndarray        # [B, T]
    ghi_clear: np.ndarray       # [B, T]
    p_clear: np.ndarray         # [B, T]
    night: np.ndarray           # [B, T] bool（elev <= 0）
    k_steps: np.ndarray         # [B, H]，M5 恒 0（M8 的 NWP 晴空指数，这里不算）
    var_signal: np.ndarray      # [B]，M5 恒 0
    degraded: np.ndarray        # [B] bool：站表缺坐标，走钟表几何
    cs_ratio: np.ndarray = None # [B]，M5 恒 1
    envelope_used: np.ndarray = None   # [B]，M5 恒 False
    slot: np.ndarray = None     # [B, T] int：每步的一天内时刻槽号（0..n_day-1，本地墙钟），历史包络按它对齐


class RowContextBuilder:
    def __init__(self, cfg: dict, station_meta: StationMeta):
        ds = cfg["dataset"]
        self.L, self.H = int(ds["context_len"]), int(ds["pred_len"])
        self.ts_col, self.sid_col, self.step_min = ds["ts_col"], ds["station_id_col"], int(ds["step_min"])
        self.meta = station_meta
        self.solar = SolarCache(station_meta, ds["tz"], self.step_min)
        self._cap_cache: dict[int, float] = {}

    def station_caps(self, sid: np.ndarray) -> np.ndarray:
        """[B] 每行所属站的固定容量（站表 cap_mw）；站表没有的站直接报错。"""
        for s in np.unique(sid):
            if int(s) not in self._cap_cache:
                self._cap_cache[int(s)] = station_cap_mw(self.meta, int(s))
        return np.array([self._cap_cache[int(s)] for s in sid], float)

    def build(self, batch_dict: dict) -> RowCtx:
        if self.sid_col not in batch_dict:
            raise ValueError(f"batch_dict 缺 {self.sid_col!r} 列：M5 full 模式按站查 station_meta 取容量，必须带数值 station_id")
        ts_epoch = _to_np(batch_dict[self.ts_col]).astype(np.int64).reshape(-1)
        sid = _to_np(batch_dict[self.sid_col]).astype(int).reshape(-1)
        B, T = len(sid), self.L + self.H
        cap = self.station_caps(sid)
        ts = window_timestamps(ts_epoch, self.L, self.H, self.step_min)
        elev = np.empty((B, T)); ghi = np.empty((B, T)); degraded = np.zeros(B, bool)
        for s in np.unique(sid):
            rows = np.flatnonzero(sid == s)
            e, g, d = self.solar.geometry(int(s), ts[rows])
            elev[rows] = e; ghi[rows] = g; degraded[rows] = d
        if degraded.any():
            new = [int(s) for s in np.unique(sid[degraded]) if int(s) not in _DEGRADED_WARNED]
            if new:
                warnings.warn(f"station_meta 缺坐标，几何退化为钟表几何（p_clear = 装机容量, elev = ±10）：station_id={new}"
                              f"（M5 默认不用几何，只影响 aux 与 g1 / g2 变体）")
                _DEGRADED_WARNED.update(new)
        p_clear = np.where(degraded[:, None], np.where(ghi > 0, cap[:, None], 0.0), ghi / 1000.0 * cap[:, None])
        tsm = ts.astype("datetime64[m]")
        slot = ((tsm - tsm.astype("datetime64[D]")).astype(np.int64) // self.step_min).astype(np.int64)
        return RowCtx(station_id=sid, cap=cap, elev_deg=elev, ghi_clear=ghi, p_clear=p_clear, night=elev <= 0.0,
                      k_steps=np.zeros((B, self.H)), var_signal=np.zeros(B), degraded=degraded,
                      cs_ratio=np.ones(B), envelope_used=np.zeros(B, bool), slot=slot)
