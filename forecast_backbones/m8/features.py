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
    from .solar import window_timestamps, solar_geometry, clock_pseudo_geometry, StationMeta, station_cap_mw
    from ._arrays import interp_nan
    from .config import normalize_future_only
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from forecast_backbones.m8.solar import window_timestamps, solar_geometry, clock_pseudo_geometry, StationMeta, station_cap_mw
    from forecast_backbones.m8._arrays import interp_nan
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
    cs_ratio: np.ndarray = None # [B] 按站自适应晴空出力比（features.derived.clear_sky_ratio 关时恒 1）
    envelope_used: np.ndarray = None   # [B] bool：该行的几何（p_clear 通道 / elev / 夜间）来自历史日形包络（features.derived.history_envelope）


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
        self.pv_col = ds["pv_col"]
        self._cap_cache: dict[int, float] = {}
        self.ts_col, self.sid_col = ds["ts_col"], ds["station_id_col"]
        self.step_min = int(ds["step_min"])
        self.pairs = [tuple(p) for p in fe["past_future"]]
        self.past_only = list(fe["past_only"])
        self.future_only = [(d["col"], d["scale"]) for d in normalize_future_only(fe.get("future_only"))]
        self.resid_col = fe.get("residual_base_col")
        if self.resid_col is not None and (self.resid_col, "target") not in self.future_only:
            raise ValueError(f"features.residual_base_col={self.resid_col!r} 须是 future_only 里 scale=target 的列")
        self.scales = fe["scales"]
        self.derived = fe["derived"]
        self.static_cfg = fe["static"]
        self.csi_col = fe.get("clear_sky_index_col")
        self.envelope_cfg = dict(self.derived.get("history_envelope") or {})
        if bool(self.envelope_cfg.get("enabled", False)) and self.L % (1440 // self.step_min) != 0:
            raise ValueError(f"features.derived.history_envelope 要求 context_len({self.L}) 是一天步数({1440 // self.step_min})的整数倍")
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
        self.channel_names = names
        self.idx = {n: i for i, n in enumerate(names)}
        self.nwp_channels = [i for n, i in self.idx.items() if n.startswith("pf:") or n in ("k_nwp", "nwp_resid")]
        self.target_scale = str(fe.get("target_scale", "cap"))          # cap | recent_peak
        if self.target_scale not in ("cap", "recent_peak"):
            raise ValueError(f"features.target_scale 须是 cap 或 recent_peak，得到 {self.target_scale!r}")
        self.n_static = 5 + (3 if self.static_cfg.get("context_stats", True) else 0)
        self.n_slots = int(self.static_cfg.get("n_station_slots", 4096)) if self.static_cfg.get("station_embedding", True) else 0
        ks = ds.get("known_stations")
        self.known_stations = None if not ks else np.array(sorted({int(s) for s in ks}), dtype=np.int64)

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

    def station_caps(self, sid: np.ndarray) -> np.ndarray:
        """[B] 每行所属站的固定容量（站表 cap_mw）；站表没有的站直接报错。"""
        for s in np.unique(sid):
            if int(s) not in self._cap_cache:
                self._cap_cache[int(s)] = station_cap_mw(self.meta, int(s))
        return np.array([self._cap_cache[int(s)] for s in sid], float)

    # ------------------------------------------------------------------ 主流程
    def build(self, batch_dict: dict, training: bool = False, y_future=None, rng: np.random.Generator | None = None) -> dict:
        L, H, T, P = self.L, self.H, self.T, self.P
        rng = rng if rng is not None else np.random.default_rng()
        ts_epoch = _to_np(batch_dict[self.ts_col]).astype(np.int64).reshape(-1)
        B = ts_epoch.shape[0]
        if self.sid_col not in batch_dict:
            raise ValueError(f"batch_dict 缺 {self.sid_col!r} 列：M8 按站查 station_meta 取容量与坐标，必须带数值 station_id")
        sid = _to_np(batch_dict[self.sid_col]).astype(int).reshape(-1)
        cap = self.station_caps(sid)
        ts = window_timestamps(ts_epoch, L, H, self.step_min)                         # [B, T] datetime64[s]

        # 目标：历史保留 NaN；幅度调制同乘历史与标签
        y_hist = _col(batch_dict, self.pv_col, keep_nan=True)[:, -L:]
        y_hist_raw = y_hist.copy()                                                  # 幅度调制前的原值（历史日形包络用）
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
                how = ("用窗内历史日形包络代替钟表几何（features.derived.history_envelope 开）" if bool(self.envelope_cfg.get("enabled", False))
                       else "退化为钟表几何（p_clear = 装机容量, elev = ±10；可开 features.derived.history_envelope 用历史日形）")
                warnings.warn(f"station_meta 缺坐标，{how}：station_id={new}")
                _DEGRADED_WARNED.update(new)
        p_clear = np.where(degraded[:, None], np.where(ghi > 0, cap[:, None], 0.0), ghi / 1000.0 * cap[:, None])
        night = elev <= 0.0
        p_clear_in = p_clear                                                        # 进通道的晴空功率；决策层用 p_clear

        # CPM：历史 token 整块不可见（先于协变量算，历史日形包络与晴空出力比都只看可见段）
        y_mask_hist = ~observed
        if training and self.cfg["train"]["cpm"]["enabled"]:
            cp = self.cfg["train"]["cpm"]
            tok = cpm_token_mask(rng, B, self.n_hist_tok, float(cp["p_row"]), float(cp["min_frac"]), float(cp["max_frac"]), int(cp["max_runs"]))
            y_mask_hist = y_mask_hist | np.repeat(tok, P, axis=1)
        visible = ~y_mask_hist
        y_vis = np.where(visible, np.nan_to_num(y_norm), 0.0)

        # 缺坐标站的历史日形包络：替代钟表几何的 p_clear 通道 / elev / 夜间；决策层 p_clear 仍是白天 = 容量
        envelope_used = np.zeros(B, bool)
        he = self.envelope_cfg
        if bool(he.get("enabled", False)) and degraded.any():
            n_day = 1440 // self.step_min
            tt = pd.DatetimeIndex(ts.reshape(-1))
            slot_all = ((tt.hour * 60 + tt.minute) // self.step_min).to_numpy().reshape(B, T).astype(np.int64)
            env_slot, days = history_envelope(np.where(visible, y_hist_raw, np.nan), slot_all[:, 0], n_day,
                                              float(he.get("quantile", 0.9)), int(he.get("smooth", 3)))
            peak = env_slot.max(axis=1)
            envelope_used = degraded & (peak >= float(he.get("peak_frac", 0.1)) * cap) & (days >= int(he.get("min_days", 3)))
            if envelope_used.any():
                r = np.flatnonzero(envelope_used)
                env_t = np.take_along_axis(env_slot[r], slot_all[r], axis=1)         # [n, T]
                day = env_t > float(he.get("night_frac", 0.02)) * cap[r, None]
                p_clear_in = p_clear.copy(); p_clear = p_clear.copy()
                p_clear_in[r] = env_t
                p_clear[r] = np.where(day, cap[r, None], 0.0)
                elev[r] = np.where(day, 90.0 * env_t / peak[r, None], -10.0)
                ghi[r] = np.where(day, 800.0, 0.0)
                night[r] = ~day

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
        if ghi_pred_full is None:
            ghi_pred_full = np.zeros((B, T))
        k_nwp = np.clip(np.where(ghi > 50.0, ghi_pred_full / np.maximum(ghi, 1e-6), 0.0), 0.0, 1.5)

        # 按站自适应晴空出力比：历史可见段里近正午（p_clear > near_noon_frac·窗内最大）点的 y / p_clear 高分位，
        # 夹到 [lo, hi]；缩放 p_clear 本身（通道 + ctx，决策层的晴空上限随之抬高）。k_nwp 是辐照比，不动
        cs = self.derived.get("clear_sky_ratio") or {}
        cs_ratio = np.ones(B)
        if bool(cs.get("enabled", False)):
            pc_h = p_clear[:, :L]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                thr = float(cs.get("near_noon_frac", 0.6)) * np.max(pc_h, axis=1, keepdims=True)
                ok = visible & (pc_h > thr) & (pc_h > 0) & ~degraded[:, None]
                ratio = np.where(ok, np.nan_to_num(y_hist) / np.maximum(pc_h, 1e-6), np.nan)
                r = np.nanquantile(ratio, float(cs.get("quantile", 0.98)), axis=1) if ok.any() else np.full(B, np.nan)
            cs_ratio = np.clip(np.where(np.isfinite(r), r, 1.0), float(cs.get("lo", 1.0)), float(cs.get("hi", 2.5)))
            p_clear = p_clear * cs_ratio[:, None]
            p_clear_in = p_clear_in * cs_ratio[:, None]

        x[:, :L, self.idx["y"]] = y_vis
        x[:, :L, self.idx["y_mask"]] = y_mask_hist
        x[:, L:, self.idx["y_mask"]] = 1.0
        x[:, L:, self.idx["is_future"]] = 1.0
        x[:, :, self.idx["k_nwp"]] = k_nwp
        x[:, :, self.idx["elev"]] = elev / 90.0
        x[:, :, self.idx["p_clear"]] = p_clear_in / cap[:, None]
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
        # 坐标与容量能唯一标识训练站（17 个站 17 组数），模型可以按它们记住"这是哪个站"；关掉只留逐步太阳几何（物理量）
        if self.static_cfg.get("log_cap", True):
            static[:, 0] = np.log1p(scale if self.target_scale == "recent_peak" else cap) / 7.0
        if self.static_cfg.get("coords", True):
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
        if self.n_slots > 0:
            if self.known_stations is not None:                                   # 名单外的站 → 未知站槽 0
                slot = np.where(np.isin(sid, self.known_stations), slot, 0)
            p_slot = float(self.cfg["train"].get("slot_dropout", 0.0))
            if training and p_slot > 0:
                slot = np.where(rng.random(B) < p_slot, 0, slot)

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
                     k_steps=k_nwp[:, L:], var_signal=var_signal, degraded=degraded, cs_ratio=cs_ratio, envelope_used=envelope_used)
        return {"x": x, "static": static, "slot": slot, "target": target, "amp": amp.astype(np.float32),
                "scale": scale.astype(np.float32), "base": base, "ctx": ctx}
