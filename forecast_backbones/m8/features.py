# -*- coding: utf-8 -*-
# forecast_backbones/m8/features.py
"""M8 特征层：每批把 batch_dict 组装成逐步通道 x[B,T,C]、静态向量 static[B,S]、站槽号 slot[B]、监督目标 target[B,T]。
纯 numpy，不依赖 torch。太阳几何（pvlib）按 (station_id, 日期) 缓存 96 步数组，训练吞吐不受 pvlib 拖累。
T = context_len + pred_len；历史段最后一点 = 起报时刻 S；未来第 i 点 = S + (i+1)·step（与 m6 / nanwang_metric 对齐）。

训练期增强（training=True 时）：幅度调制（历史与标签同乘 u）、CPM 连续 patch 掩码（历史 token 整块置为不可见，并在
target 里保留真值供计损）、协变量 dropout（整行清零全部 NWP 通道并置 cov_missing=1）。"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..m6.features import window_timestamps, solar_geometry, clock_pseudo_geometry, StationMeta
    from ..m6._arrays import interp_nan
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m6.features import window_timestamps, solar_geometry, clock_pseudo_geometry, StationMeta
    from forecast_backbones.m6._arrays import interp_nan
try:
    from .config import normalize_future_only
except ImportError:
    from forecast_backbones.m8.config import normalize_future_only

_WARNED_MISSING: set = set()
_DEGRADED_WARNED: set = set()


def _to_np(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def _col(batch_dict: dict, col: str, keep_nan: bool) -> np.ndarray:
    """[B, n] float64；1 维输入视为 B=1。keep_nan=False 时逐行线性插值填补非有限值（整行无效填 0）。"""
    if col not in batch_dict:
        raise KeyError(f"batch_dict 缺列 {col!r}；现有 keys={sorted(batch_dict)[:20]}...")
    a = _to_np(batch_dict[col]).astype(np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if not keep_nan and not np.isfinite(a).all():
        a = interp_nan(a)
    return a


def _scalar_per_row(v) -> np.ndarray:
    a = _to_np(v).astype(float)
    if a.ndim == 2:
        return interp_nan(a)[:, 0]
    return a.reshape(-1)


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
        days = (epoch // 86400).astype(np.int64)                    # 日序号（本地墙钟）
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
    k_steps: np.ndarray         # [B, H] NWP 晴空指数（未来段）
    var_signal: np.ndarray      # [B] 未来白天 k_nwp 均值
    degraded: np.ndarray        # [B] bool


def cpm_token_mask(rng: np.random.Generator, B: int, n_tok: int, p_row: float, min_frac: float, max_frac: float,
                   max_runs: int) -> np.ndarray:
    """TiRex 式连续 patch 掩码：每行以 p_row 概率抽 U(min_frac,max_frac) 比例的历史 token，拆成 1..max_runs 段连续放置。"""
    mask = np.zeros((B, n_tok), bool)
    for b in range(B):
        if rng.random() >= p_row:
            continue
        n = int(max(1, round(rng.uniform(min_frac, max_frac) * n_tok)))
        runs = int(rng.integers(1, max(1, min(max_runs, n)) + 1))
        cuts = np.sort(rng.choice(np.arange(1, n), size=runs - 1, replace=False)) if runs > 1 else np.array([], int)
        lengths = np.diff(np.concatenate([[0], cuts, [n]]))
        for ln in lengths:
            ln = int(ln)
            start = int(rng.integers(0, n_tok - ln + 1))
            mask[b, start:start + ln] = True
    return mask


class FeatureBuilder:
    """构造期固定通道表（channel_names）与静态维数（n_static）；build() 每批组装。"""

    BASE = ["y", "y_mask", "is_future"]
    SOLAR = ["k_nwp", "elev", "p_clear", "is_night"]
    TIME = ["tod_sin", "tod_cos", "doy_sin", "doy_cos"]

    def __init__(self, cfg: dict, station_meta: StationMeta):
        self.cfg = cfg
        ds, fe = cfg["dataset"], cfg["features"]
        self.L, self.H = int(ds["context_len"]), int(ds["pred_len"])
        self.T = self.L + self.H
        self.P = int(cfg["model"]["patch_len"])
        self.n_hist_tok = self.L // self.P
        self.pv_col, self.cap_col = ds["pv_col"], ds["cap_col"]
        self.ts_col, self.sid_col = ds["ts_col"], ds["station_id_col"]
        self.step_min = int(ds["step_min"])
        self.pairs = [tuple(p) for p in fe["past_future"]]
        self.past_only = list(fe["past_only"])
        self.future_only = [(d["col"], d["scale"]) for d in normalize_future_only(fe.get("future_only"))]
        self.resid_col = fe.get("residual_base_col")
        if self.resid_col is not None and (self.resid_col, "target") not in self.future_only:
            raise ValueError(f"features.residual_base_col={self.resid_col!r} 须是 future_only 里 scale=target 的列")
        self.members = fe["nwp_members"]
        self.scales = fe["scales"]
        self.derived = fe["derived"]
        self.static_cfg = fe["static"]
        self.csi_col = fe.get("clear_sky_index_col")
        self.meta = station_meta
        self.solar = SolarCache(station_meta, ds["tz"], self.step_min)
        names = list(self.BASE)
        names += [f"pf:{h}" for h, _ in self.pairs]
        names += [f"po:{c}" for c in self.past_only]
        names += [f"fo:{c}" for c, _ in self.future_only]
        if self.future_only:
            names.append("hint_missing")
        names += list(self.SOLAR)
        if self.derived.get("time_encoding", True):
            names += list(self.TIME)
        if self.derived.get("nwp_resid", True):
            names.append("nwp_resid")
        names.append("cov_missing")
        if self.members["enabled"]:
            names += ["mem_mean", "mem_std"]
        self.channel_names = names
        self.idx = {n: i for i, n in enumerate(names)}
        self.nwp_channels = [i for n, i in self.idx.items() if n.startswith("pf:") or n in ("k_nwp", "nwp_resid", "mem_mean", "mem_std")]
        # split 布局：目标流 = 功率及其掩码 / 偏差；协变量流 = 其余（两流都带 is_future 与 cov_missing 标志）
        tgt = {"y", "y_mask", "is_future", "nwp_resid", "cov_missing"}
        self.target_channels = [i for n, i in self.idx.items() if n in tgt]
        self.cov_channels = [i for n, i in self.idx.items() if n not in tgt or n in ("is_future", "cov_missing")]
        self.target_scale = str(fe.get("target_scale", "cap"))          # cap | recent_peak
        if self.target_scale not in ("cap", "recent_peak"):
            raise ValueError(f"features.target_scale 须是 cap 或 recent_peak，得到 {self.target_scale!r}")
        self.n_static = 5 + (3 if self.static_cfg.get("context_stats", True) else 0)
        self.n_slots = int(self.static_cfg.get("n_station_slots", 4096)) if self.static_cfg.get("station_embedding", True) else 0

    # ------------------------------------------------------------------ 小工具
    def _scale(self, col: str) -> tuple[float, float]:
        lc = col.lower()
        for key, (shift, scale) in self.scales.items():
            if key != "default" and key in lc:
                return float(shift), float(scale)
        shift, scale = self.scales.get("default", [0.0, 1.0])
        return float(shift), float(scale)

    def _pair(self, batch_dict: dict, hist_c: str, fut_c: str):
        """→ [B, T] 原始单位（已插值）；缺列返回 None（只提示一次）。"""
        if hist_c in batch_dict and fut_c in batch_dict:
            h = _col(batch_dict, hist_c, keep_nan=False)[:, -self.L:]
            f = _col(batch_dict, fut_c, keep_nan=False)[:, :self.H]
            return np.concatenate([h, f], axis=1)
        if (hist_c, fut_c) not in _WARNED_MISSING:
            warnings.warn(f"past_future 协变量对 {(hist_c, fut_c)} 在 batch_dict 里缺列，该通道置 0（只提示一次）")
            _WARNED_MISSING.add((hist_c, fut_c))
        return None

    # ------------------------------------------------------------------ 主流程
    def build(self, batch_dict: dict, training: bool = False, y_future=None, rng: np.random.Generator | None = None) -> dict:
        L, H, T, P = self.L, self.H, self.T, self.P
        rng = rng if rng is not None else np.random.default_rng()
        ts_epoch = _to_np(batch_dict[self.ts_col]).astype(np.int64).reshape(-1)
        B = ts_epoch.shape[0]
        sid = _to_np(batch_dict[self.sid_col]).astype(int).reshape(-1) if self.sid_col in batch_dict else np.zeros(B, int)
        if self.cap_col in batch_dict:
            cap = _scalar_per_row(batch_dict[self.cap_col])
        else:
            cap = np.array([(self.meta.get(s) or {}).get("cap_mw", np.nan) for s in sid], float)
        if not np.isfinite(cap).all() or (cap <= 0).any():
            bad = sorted(int(s) for s in np.unique(sid[~np.isfinite(cap) | (cap <= 0)]))
            raise ValueError(f"cap 非有限或非正（station_meta 与 {self.cap_col!r} 均无有效值）：station_id={bad}")
        ts = window_timestamps(ts_epoch, L, H, self.step_min)                         # [B, T] datetime64[s]

        # 目标：历史保留 NaN；幅度调制同乘历史与标签
        y_hist = _col(batch_dict, self.pv_col, keep_nan=True)[:, -L:]
        amp = np.ones(B)
        am = self.cfg["train"]["amp_mod"]
        if training and float(am.get("p", 0)) > 0:
            hit = rng.random(B) < float(am["p"])
            amp = np.where(hit, rng.uniform(float(am["lo"]), float(am["hi"]), B), 1.0)
        y_hist = y_hist * amp[:, None]
        # 目标尺度：cap（装机容量）或 recent_peak（Reverso min-max 的 PV 版：近 7 天实测峰值，下限 0.3·cap）
        if self.target_scale == "recent_peak":
            with np.errstate(all="ignore"):
                peak = np.nanmax(np.where(np.isfinite(y_hist), y_hist, -np.inf), axis=1)
            scale = np.maximum(0.3 * cap, np.where(np.isfinite(peak), peak, 0.0))
        else:
            scale = cap.copy()
        y_norm = y_hist / scale[:, None]                                            # [B, L]，NaN 处未观测
        observed = np.isfinite(y_norm)

        # 太阳几何
        elev = np.empty((B, T)); ghi = np.empty((B, T)); degraded = np.zeros(B, bool)
        for s in np.unique(sid):
            rows = np.flatnonzero(sid == s)
            e, g, d = self.solar.geometry(int(s), ts[rows])
            elev[rows] = e; ghi[rows] = g; degraded[rows] = d
        if degraded.any():
            new = [int(s) for s in np.unique(sid[degraded]) if int(s) not in _DEGRADED_WARNED]
            if new:
                warnings.warn(f"station_meta 缺坐标，退化为钟表几何（p_clear = 装机容量, elev = ±10）：station_id={new}")
                _DEGRADED_WARNED.update(new)
        p_clear = np.where(degraded[:, None], np.where(ghi > 0, cap[:, None], 0.0), ghi / 1000.0 * cap[:, None])
        night = elev <= 0.0

        # 协变量
        C = len(self.channel_names)
        x = np.zeros((B, T, C), np.float32)
        ghi_pred_full = None
        for hist_c, fut_c in self.pairs:
            arr = self._pair(batch_dict, hist_c, fut_c)
            if arr is None:
                continue
            if fut_c == self.csi_col or (ghi_pred_full is None and ("ghi" in fut_c.lower() or "ssrd" in fut_c.lower())):
                if fut_c == self.csi_col or ghi_pred_full is None:
                    ghi_pred_full = arr
            sh, sc = self._scale(hist_c)
            x[:, :, self.idx[f"pf:{hist_c}"]] = (arr - sh) / sc
        for c in self.past_only:
            if c in batch_dict:
                sh, sc = self._scale(c)
                x[:, :L, self.idx[f"po:{c}"]] = (_col(batch_dict, c, keep_nan=False)[:, -L:] - sh) / sc
        if self.members["enabled"]:
            suf = self.members["future_suffix"]
            mem = np.stack([np.concatenate([_col(batch_dict, c, False)[:, -L:], _col(batch_dict, c + suf, False)[:, :H]], 1)
                            for c in self.members["cols"]], 0)                      # [M, B, T]
            x[:, :, self.idx["mem_mean"]] = mem.mean(0) / 1000.0
            x[:, :, self.idx["mem_std"]] = mem.std(0) / 1000.0
        if ghi_pred_full is None:
            ghi_pred_full = np.zeros((B, T))
        k_nwp = np.clip(np.where(ghi > 50.0, ghi_pred_full / np.maximum(ghi, 1e-6), 0.0), 0.0, 1.5)

        # CPM：历史 token 整块不可见
        y_mask_hist = ~observed
        if training and self.cfg["train"]["cpm"]["enabled"]:
            cp = self.cfg["train"]["cpm"]
            tok = cpm_token_mask(rng, B, self.n_hist_tok, float(cp["p_row"]), float(cp["min_frac"]), float(cp["max_frac"]), int(cp["max_runs"]))
            y_mask_hist = y_mask_hist | np.repeat(tok, P, axis=1)
        visible = ~y_mask_hist
        y_vis = np.where(visible, np.nan_to_num(y_norm), 0.0)

        x[:, :L, self.idx["y"]] = y_vis
        x[:, :L, self.idx["y_mask"]] = y_mask_hist
        x[:, L:, self.idx["y_mask"]] = 1.0
        x[:, L:, self.idx["is_future"]] = 1.0
        x[:, :, self.idx["k_nwp"]] = k_nwp
        x[:, :, self.idx["elev"]] = elev / 90.0
        x[:, :, self.idx["p_clear"]] = p_clear / cap[:, None]
        x[:, :, self.idx["is_night"]] = night
        if "tod_sin" in self.idx:
            t = pd.DatetimeIndex(ts.reshape(-1))
            tod = ((t.hour * 60 + t.minute) / 1440.0).to_numpy().reshape(B, T)
            doy = (t.dayofyear / 365.25).to_numpy().reshape(B, T)
            x[:, :, self.idx["tod_sin"]] = np.sin(2 * np.pi * tod); x[:, :, self.idx["tod_cos"]] = np.cos(2 * np.pi * tod)
            x[:, :, self.idx["doy_sin"]] = np.sin(2 * np.pi * doy); x[:, :, self.idx["doy_cos"]] = np.cos(2 * np.pi * doy)
        if "nwp_resid" in self.idx:
            day_vis = visible & ~night[:, :L]
            x[:, :L, self.idx["nwp_resid"]] = np.where(day_vis, y_vis - ghi_pred_full[:, :L] / 1000.0 * cap[:, None] / scale[:, None], 0.0)

        # 协变量 dropout：整行清零 NWP 通道
        cov_missing = np.zeros(B, bool)
        pdrop = float(self.cfg["train"].get("cov_dropout", 0.0))
        if training and pdrop > 0:
            cov_missing = rng.random(B) < pdrop
            if cov_missing.any():
                x[np.ix_(np.flatnonzero(cov_missing), np.arange(T), self.nwp_channels)] = 0.0
        x[:, :, self.idx["cov_missing"]] = cov_missing[:, None]

        # 只有未来段的提示通道（如 TimesFM 预测）：历史段恒 0，CPM 掩掉的历史不会从这里漏回来；
        # hint_dropout 与 cov_dropout 独立抽样，整组清零并标 hint_missing；残差基准 base 跟随实际喂入的值
        base = np.zeros((B, T), np.float32)
        if self.future_only:
            hint_missing = np.zeros(B, bool)
            ph = float(self.cfg["train"].get("hint_dropout", 0.0))
            if training and ph > 0:
                hint_missing = rng.random(B) < ph
            absent = [c for c, _ in self.future_only if c not in batch_dict]
            if absent:
                for c in absent:
                    if c not in _WARNED_MISSING:
                        warnings.warn(f"future_only 列 {c!r} 在 batch_dict 里缺列，该组通道置 0 并标 hint_missing=1（只提示一次）")
                        _WARNED_MISSING.add(c)
                hint_missing[:] = True
            for col, spec in self.future_only:
                if col in absent:
                    continue
                arr = _col(batch_dict, col, keep_nan=False)[:, :H]
                if spec == "target":
                    val = arr * amp[:, None] / scale[:, None]
                else:
                    sh, sc = self._scale(col)
                    val = (arr - sh) / sc
                x[:, L:, self.idx[f"fo:{col}"]] = np.where(hint_missing[:, None], 0.0, val)
            x[:, :, self.idx["hint_missing"]] = hint_missing[:, None]
            if self.resid_col is not None:
                base[:, L:] = x[:, L:, self.idx[f"fo:{self.resid_col}"]]

        # 静态向量与站槽
        static = np.zeros((B, self.n_static), np.float32)
        lat = np.array([(self.meta.get(s) or {}).get("lat", np.nan) for s in sid], float)
        lon = np.array([(self.meta.get(s) or {}).get("lon", np.nan) for s in sid], float)
        alt = np.array([(self.meta.get(s) or {}).get("alt_m", 0.0) for s in sid], float)
        static[:, 0] = np.log1p(cap) / 7.0
        if self.target_scale == "recent_peak":
            static[:, 0] = np.log1p(scale) / 7.0
        static[:, 1] = np.nan_to_num(lat) / 90.0
        static[:, 2] = np.nan_to_num(lon) / 180.0
        static[:, 3] = np.nan_to_num(alt) / 5000.0
        static[:, 4] = degraded
        if self.n_static > 5:
            day_vis = visible & ~night[:, :L]
            cnt = np.maximum(day_vis.sum(1), 1)
            static[:, 5] = np.where(visible, y_vis, 0.0).max(1)
            static[:, 6] = np.where(day_vis, y_vis, 0.0).sum(1) / cnt
            static[:, 7] = np.where(day_vis, k_nwp[:, :L], 0.0).sum(1) / cnt
        slot = (np.mod(sid, self.n_slots) + 1).astype(np.int64) if self.n_slots > 0 else np.zeros(B, np.int64)

        # 监督目标：历史 = 真值（NaN 未观测；是否计损由 y_mask 通道决定）；未来 = 标签（给了才有）
        target = np.full((B, T), np.nan, np.float32)
        target[:, :L] = y_norm
        if y_future is not None:
            yf = np.asarray(_to_np(y_future), float)
            if yf.ndim == 1:
                yf = yf[None, :]
            target[:, L:] = yf[:, :H] * amp[:, None] / scale[:, None]

        fut_day = ~night[:, L:]
        var_signal = np.where(fut_day, k_nwp[:, L:], 0.0).sum(1) / np.maximum(fut_day.sum(1), 1)
        ctx = RowCtx(station_id=sid, cap=cap, elev_deg=elev, ghi_clear=ghi, p_clear=p_clear, night=night,
                     k_steps=k_nwp[:, L:], var_signal=var_signal, degraded=degraded)
        return {"x": x, "static": static, "slot": slot, "target": target, "amp": amp.astype(np.float32),
                "scale": scale.astype(np.float32), "base": base, "ctx": ctx}
