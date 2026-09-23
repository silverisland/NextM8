# -*- coding: utf-8 -*-
# forecast_backbones/m5/conformal.py
"""M5 离线滚动 conformal 回放（收缩加权方案）。纯 numpy/pandas，不碰模型。

输入：conformal_replay.py predict 落盘的原始分位表，每站每个起报日一行（当日 <= 10:00 的最晚一行），列
    station, timestamp_win, cap, observe_power_future[192], q_0.1..q_0.9[192], elev_deg[192], p_clear[192], night[192], var_bin
逐站逐日回放：校准第 d 天时只用实测已到齐的点（起报日 <= d − lag_days），拟合完校准第 d 天，再按南网 D+1 打分。
分数 s = (y − q_{τ0}) / p_clear；δ = 第 ⌈(n+1)τ⌉ 小的分数；q̃_τ = q_{τ0} + δ·p_clear（τ0 = 离 τ 最近的模型分位档）。
格子层级：(elev,var,lead) → (elev,lead) → (elev) → 全体；每档取站日数 >= D_min(τ) = ⌈m / min(τ,1−τ)⌉ 的最细层。
本站 δ：本站起报日落在 [d−lag−W+1, d−lag] 的点。
类 δ：同类各站的点。pool=calendar：同一绝对时间窗；pool=season：按一年中的第几天（不看年份）落在同一窗的点，
    本站自己的点只取绝对时间窗内的（不用本站未来的实测）。
收缩：在类 δ 选中的那一层上，δ = w·本站 δ + (1−w)·类 δ，w = D/(D+k)，D = 本站在该格的起报日数，D < D_min(τ) 时 w = 0；
    类在每一层都不够时退回只用本站；本站也不够就不校准（δ = 0）。
对照组：raw（不校准）、station（只用本站）、pool（只用类，即新站的处境）、shrink_k{k}（收缩加权）。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_backbones.m6.calibration import bin_elev, bin_lead, nearest_model_level, quantile_cols
from forecast_backbones.m6.decision import CAL_LEVELS, MODEL_LEVELS, enforce_monotone, physics_clip, select_point
from pipelines.nanwang_metric import NanwangMetric

Q_COLS = quantile_cols(MODEL_LEVELS)
ARRAY_COLS = ("observe_power_future", "elev_deg", "p_clear", "night", *Q_COLS)
RAW_COLS = ("station", "timestamp_win", "cap", "var_bin", *ARRAY_COLS)
LEVEL_DIMS = (("e", "v", "l"), ("e", "l"), ("e",), ())
DAY_BUCKETS = ((0, 0), (1, 6), (7, 14), (15, 29), (30, 10 ** 6))
_EPS = 1e-6


@dataclass
class ReplayConfig:
    window_days: int = 45
    lag_days: int = 2
    m: float = 3.0
    ks: tuple = (15.0,)
    pool: str = "season"                      # season | calendar
    levels: tuple = CAL_LEVELS
    elev_edges: tuple = (20.0, 45.0)
    lead_edges: tuple = (4, 8, 16, 55, 103)
    fit_steps: tuple = (55, 151)              # 10:00 起报的 D+1 全天

    def __post_init__(self):
        if self.pool not in ("season", "calendar"):
            raise ValueError(f"pool 只能是 season 或 calendar：{self.pool!r}")
        if self.lag_days < 2:
            raise ValueError(f"lag_days={self.lag_days}：10:00 起报的 D+1 实测要到第 2 天才齐，lag_days 至少 2")
        if self.window_days < 1:
            raise ValueError(f"window_days 须 >= 1：{self.window_days}")
        if list(self.levels) != sorted(self.levels):
            raise ValueError(f"levels 必须升序：{self.levels}")

    def d_min(self) -> np.ndarray:
        return np.array([math.ceil(self.m / min(t, 1.0 - t)) for t in self.levels], int)

    def model_idx(self) -> np.ndarray:
        return np.array([list(MODEL_LEVELS).index(nearest_model_level(t, MODEL_LEVELS)) for t in self.levels], int)


def _stack(raw: pd.DataFrame, col: str, dtype=float) -> np.ndarray:
    return np.stack([np.asarray(v, dtype) for v in raw[col].to_numpy()])


def _day_number(ts: pd.Series) -> np.ndarray:
    """起报日 → 自 1970-01-01 起的天数。"""
    d = pd.to_datetime(ts).dt.normalize()
    return ((d - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)).to_numpy(np.int64)


def _cell_codes(e, v, l, dims, radix) -> np.ndarray:
    parts = {"e": e, "v": v, "l": l}
    c = np.zeros(np.shape(e), np.int64)
    for d in dims:
        c = c * radix[d] + np.asarray(parts[d], np.int64)
    return c


class ReplayData:
    """原始分位表 → 目标行数组（每行一个站日）+ 拟合点数组（每个白天、真值有限的 D+1 点一行）。"""

    def __init__(self, raw: pd.DataFrame, cfg: ReplayConfig):
        missing = [c for c in RAW_COLS if c not in raw.columns]
        if missing:
            raise KeyError(f"原始分位表缺列：{missing}")
        raw = raw.assign(timestamp_win=pd.to_datetime(raw["timestamp_win"]), station=raw["station"].astype(str))
        raw = raw.sort_values(["station", "timestamp_win"]).drop_duplicates(["station", "timestamp_win"], keep="last")
        raw = raw.assign(_day=_day_number(raw["timestamp_win"]))
        raw = raw.drop_duplicates(["station", "_day"], keep="last").reset_index(drop=True)
        self.cfg, self.raw = cfg, raw
        self.stations = sorted(raw["station"].unique())
        code = {s: i for i, s in enumerate(self.stations)}
        self.st = raw["station"].map(code).to_numpy(np.int64)
        self.day = raw["_day"].to_numpy(np.int64)
        self.doy = (pd.to_datetime(raw["timestamp_win"]).dt.dayofyear.to_numpy(np.int64) - 1)
        self.cap = raw["cap"].to_numpy(float)
        self.var_bin = raw["var_bin"].to_numpy(np.int64)
        self.y = _stack(raw, "observe_power_future")
        self.q = np.stack([_stack(raw, c) for c in Q_COLS], axis=-1)                   # [N,H,9]
        self.elev = _stack(raw, "elev_deg")
        self.p_clear = _stack(raw, "p_clear")
        self.night = _stack(raw, "night", bool)
        N, H = self.y.shape
        self.radix = {"e": len(cfg.elev_edges) + 1, "v": int(max(self.var_bin.max(initial=0), 0)) + 1,
                      "l": len(cfg.lead_edges) + 1}
        e_bin = bin_elev(self.elev, cfg.elev_edges)                                   # [N,H]
        l_bin = np.broadcast_to(bin_lead(np.arange(H), cfg.lead_edges), (N, H))
        v_bin = np.broadcast_to(self.var_bin[:, None], (N, H))
        self.step_codes = [_cell_codes(e_bin, v_bin, l_bin, dims, self.radix) for dims in LEVEL_DIMS]   # 每层 [N,H]

        lo, hi = cfg.fit_steps
        fit = np.zeros(H, bool); fit[lo:hi] = True
        ok = fit[None, :] & np.isfinite(self.y) & ~self.night & (self.p_clear > 0) & np.isfinite(self.q).all(-1)
        r, t = np.nonzero(ok)
        self.p_row, self.p_st, self.p_day, self.p_doy = r, self.st[r], self.day[r], self.doy[r]
        self.p_codes = [c[r, t] for c in self.step_codes]
        self.p_scores = (self.y[r, t][:, None] - self.q[r, t]) / np.maximum(self.p_clear[r, t], _EPS)[:, None]   # [n,9]
        self.fit_mask = ok                                                              # 覆盖率也在这些点上算


def cell_deltas(scores, codes, groups, taus, jidx):
    """按格子码分组 → (格子码 [G] 升序, δ [G,T], 站日数 [G])。δ = 组内第 ⌈(n+1)τ⌉ 小的分数（n = 点数，截到 [1,n]），
    分数取 τ 对应的模型分位档 jidx；站日数 = 组内 groups 的不同取值个数。"""
    taus = np.asarray(taus, float)
    if len(codes) == 0:
        return np.zeros(0, np.int64), np.zeros((0, len(taus))), np.zeros(0, int)
    order = np.argsort(codes, kind="stable")
    uc, start = np.unique(codes[order], return_index=True)
    end = np.r_[start[1:], len(order)]
    delta = np.empty((len(uc), len(taus))); nd = np.empty(len(uc), int)
    for g, (a, b) in enumerate(zip(start, end)):
        idx = order[a:b]; n = b - a
        k = np.array([min(max(int(math.ceil((n + 1) * t)), 1), n) for t in taus]) - 1
        s = np.sort(scores[idx], axis=0)
        delta[g] = s[k, jidx]
        nd[g] = len(np.unique(groups[idx]))
    return uc, delta, nd


def _lookup(uc, delta, nd, codes):
    """格子码 [S] → (δ [S,T]，缺格为 NaN；站日数 [S]，缺格为 0)。"""
    S, T = len(codes), delta.shape[1]
    if len(uc) == 0:
        return np.full((S, T), np.nan), np.zeros(S, int)
    pos = np.minimum(np.searchsorted(uc, codes), len(uc) - 1)
    hit = uc[pos] == codes
    d = np.where(hit[:, None], delta[pos], np.nan)
    return d, np.where(hit, nd[pos], 0)


def _choose(d, n, dmin):
    """d [层,S,T]、n [层,S] → 每格取站日数够 dmin 的最细层：(δ [S,T]，未选中为 0；所选层 [S,T]，-1 = 无)。"""
    _, S, T = d.shape
    out = np.zeros((S, T)); lvl = np.full((S, T), -1)
    for i in range(d.shape[0]):
        ok = (lvl < 0) & (n[i][:, None] >= dmin[None, :]) & np.isfinite(d[i])
        out[ok] = d[i][ok]; lvl[ok] = i
    return out, lvl


class Replayer:
    def __init__(self, data: ReplayData, clusters: dict | None = None):
        self.data, self.cfg = data, data.cfg
        clusters = clusters or {}
        names = [str(clusters.get(s, "all")) for s in data.stations]
        cl_code = {c: i for i, c in enumerate(sorted(set(names)))}
        self.cluster_of = np.array([cl_code[c] for c in names], int)                  # 站码 → 类码
        self.p_cluster = self.cluster_of[data.p_st]
        self.taus = np.asarray(self.cfg.levels, float)
        self.dmin = self.cfg.d_min()
        self.jidx = self.cfg.model_idx()

    def windows(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """第 i 个目标行（站 s、起报日 d）的本站点掩码与类点掩码。"""
        D, c = self.data, self.cfg
        s, d, doy = D.st[i], D.day[i], D.doy[i]
        hi, lo = d - c.lag_days, d - c.lag_days - c.window_days + 1
        in_abs = (D.p_day >= lo) & (D.p_day <= hi)
        own = (D.p_st == s) & in_abs
        same_cluster = self.p_cluster == self.cluster_of[s]
        if c.pool == "calendar":
            pool = same_cluster & in_abs
        else:
            off = (doy - D.p_doy) % 365
            in_season = (off >= c.lag_days) & (off <= c.lag_days + c.window_days - 1)
            pool = same_cluster & np.where(D.p_st == s, in_abs, in_season)
        return own, pool

    def _level_deltas(self, mask, groups, i):
        D = self.data
        ds, ns = [], []
        for lv in range(len(LEVEL_DIMS)):
            uc, dl, nd = cell_deltas(D.p_scores[mask], D.p_codes[lv][mask], groups[mask], self.taus, self.jidx)
            d, n = _lookup(uc, dl, nd, D.step_codes[lv][i])
            ds.append(d); ns.append(n)
        return np.stack(ds), np.stack(ns)                                              # [层,H,T], [层,H]

    def deltas(self, i: int) -> tuple[dict, dict]:
        """第 i 个目标行各对照组的 δ [H,T]，以及诊断量（本站天数、类站日数）。"""
        D = self.data
        own, pool = self.windows(i)
        od, on = self._level_deltas(own, D.p_day, i)
        pd_, pn = self._level_deltas(pool, D.p_st * 1_000_000 + D.p_day, i)
        d_station, _ = _choose(od, on, self.dmin)
        d_pool, lvl_pool = _choose(pd_, pn, self.dmin)
        H = od.shape[1]
        has_pool = lvl_pool >= 0
        li = np.where(has_pool, lvl_pool, 0)
        rows = np.arange(H)[:, None]
        own_at = np.nan_to_num(od[li, rows, np.arange(len(self.taus))[None, :]], nan=0.0)
        days_at = on[li, rows]                                                          # [H,T]
        out = {"station": d_station, "pool": d_pool}
        for k in self.cfg.ks:
            w = np.where(days_at >= self.dmin[None, :], days_at / (days_at + float(k)), 0.0)
            out[f"shrink_k{float(k):g}"] = np.where(has_pool, w * own_at + (1.0 - w) * d_pool, d_station)
        diag = {"own_days": int(len(np.unique(D.p_day[own]))),
                "pool_station_days": int(len(np.unique((D.p_st * 1_000_000 + D.p_day)[pool])))}
        return out, diag


def _points(q_all, levels, data: ReplayData, rows, decision: dict) -> np.ndarray:
    """[n,H,K] 分位 → 物理钳位 → 南网口径上报点 [n,H]。"""
    ph, pt = decision["physics"], decision["point"]
    cap = data.cap[rows]
    qc = physics_clip(enforce_monotone(q_all), data.night[rows], data.p_clear[rows], cap[:, None],
                      float(ph["clear_sky_cap_ratio"]), bool(ph["capacity_cap"]))
    return select_point(qc, levels, cap, pt["default"], pt.get("by_step") or (), 1.0,
                        float(pt["floor_ratio"]), float(pt["tau_grid_step"]))


def _coverage_rows(q_sorted, levels, data: ReplayData, rows, arm) -> list[dict]:
    """拟合窗内（白天、真值有限）点上 y <= q_τ 的比例，按站累加计数。"""
    out = []
    ok = data.fit_mask[rows]
    y = data.y[rows]
    for st in np.unique(data.st[rows]):
        m = data.st[rows] == st
        okm = ok[m]; n = int(okm.sum())
        if n == 0:
            continue
        below = (y[m][..., None] <= q_sorted[m]) & okm[..., None]
        for j, t in enumerate(levels):
            out.append({"arm": arm, "station": data.stations[st], "tau": float(t),
                        "n_below": int(below[..., j].sum()), "n": n})
    return out


def replay(raw: pd.DataFrame, cfg: ReplayConfig, decision: dict, clusters: dict | None = None,
           progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐站逐日回放。返回 (daily：station/date/phase/own_days/pool_station_days/arm/accuracy_pct，
    coverage：arm/station/tau/n_below/n)。phase：每站按起报日排序，前一半 first_half（挑 k 用），后一半 second_half（报分用）。"""
    data = ReplayData(raw, cfg)
    rp = Replayer(data, clusters)
    N, H, _ = data.q.shape
    arms = ["station", "pool", *[f"shrink_k{float(k):g}" for k in cfg.ks]]
    qcal = {a: np.empty((N, H, len(cfg.levels))) for a in arms}
    diag = []
    base = data.q[:, :, rp.jidx]                                                        # [N,H,T]
    norm = np.maximum(data.p_clear, _EPS)
    it = range(N)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="[m5-conformal] 回放", unit="站日", mininterval=2.0)
    for i in it:
        ds, dg = rp.deltas(i)
        for a in arms:
            qcal[a][i] = base[i] + ds[a] * norm[i][:, None]
        diag.append(dg)

    nm = NanwangMetric()
    daily, cov = [], []
    rows = np.arange(N)
    variants = {"raw": (data.q, MODEL_LEVELS), **{a: (qcal[a], tuple(cfg.levels)) for a in arms}}
    frame = data.raw[["station", "timestamp_win", "observe_power_future"]]
    gccap = {s: float(c) for s, c in zip(data.raw["station"], data.cap)}
    for arm, (q, levels) in variants.items():
        q_sorted = enforce_monotone(q)
        cov.extend(_coverage_rows(q_sorted, levels, data, rows, arm))
        pts = _points(q, levels, data, rows, decision)
        sd = nm.short_d1_daily(frame.assign(observe_power_predict=list(pts.astype(np.float32))), gccap)
        daily.append(sd.assign(arm=arm)[["station", "date", "arm", "accuracy_pct"]])
    daily = pd.concat(daily, ignore_index=True)
    meta = data.raw[["station", "timestamp_win"]].assign(date=lambda f: f["timestamp_win"].dt.normalize())
    meta = meta.join(pd.DataFrame(diag))
    rank = meta.groupby("station")["date"].rank(method="first")
    size = meta.groupby("station")["date"].transform("size")
    meta["phase"] = np.where(rank <= np.ceil(size / 2), "first_half", "second_half")
    daily = daily.merge(meta[["station", "date", "phase", "own_days", "pool_station_days"]], on=["station", "date"], how="left")
    return daily, pd.DataFrame(cov)


def _bucket(days: pd.Series) -> pd.Series:
    labels = [f"{a}" if a == b else (f"{a}+" if b >= 10 ** 6 else f"{a}-{b}") for a, b in DAY_BUCKETS]
    out = pd.Series(pd.NA, index=days.index, dtype="object")
    for (a, b), lab in zip(DAY_BUCKETS, labels):
        out[(days >= a) & (days <= b)] = lab
    return pd.Categorical(out, categories=labels, ordered=True)


def summarize(daily: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """by_station：站 × 段 × 组的日均准确率与相对 raw 的提升；summary：先站内平均再站间平均；
    cold_start：按本站当天可用天数分桶的逐日提升；coverage：各组各档全体覆盖率与站间平均绝对偏差。"""
    raw = daily[daily["arm"] == "raw"][["station", "date", "accuracy_pct"]].rename(columns={"accuracy_pct": "raw_pct"})
    d = daily.merge(raw, on=["station", "date"], how="inner").assign(gain=lambda f: f["accuracy_pct"] - f["raw_pct"])
    both = pd.concat([d, d.assign(phase="all")], ignore_index=True)
    by_station = (both.groupby(["phase", "arm", "station"])
                  .agg(accuracy_pct=("accuracy_pct", "mean"), gain=("gain", "mean"), n_days=("date", "size"))
                  .reset_index())
    summary = (by_station.groupby(["phase", "arm"])
               .agg(accuracy_pct=("accuracy_pct", "mean"), gain=("gain", "mean"),
                    n_stations=("station", "size"), n_worse=("gain", lambda g: int((g < -1e-9).sum())),
                    n_days=("n_days", "sum"))
               .reset_index())
    cold = (d.assign(own_days_bucket=_bucket(d["own_days"]))
            .groupby(["arm", "own_days_bucket"], observed=True)
            .agg(gain=("gain", "mean"), accuracy_pct=("accuracy_pct", "mean"), n_station_days=("date", "size"))
            .reset_index())
    cv = coverage.assign(coverage=lambda f: f["n_below"] / f["n"])
    tot = cv.groupby(["arm", "tau"]).agg(n_below=("n_below", "sum"), n=("n", "sum")).reset_index()
    tot["coverage"] = tot["n_below"] / tot["n"]
    dev = cv.assign(abs_dev=lambda f: (f["coverage"] - f["tau"]).abs()).groupby(["arm", "tau"])["abs_dev"].mean()
    tot = tot.merge(dev.rename("station_mean_abs_dev").reset_index(), on=["arm", "tau"])
    return {"summary": summary, "by_station": by_station, "cold_start": cold, "coverage": tot,
            "coverage_by_station": cv}
