"""逐站 Mondrian 单侧 conformal 分位校准。
分数 s = (y - q_{τ0}) / norm，δ_{s,g,τ} = 第 ⌈(n+1)τ⌉ 小分数；q̃_τ = q_{τ0} + δ·norm。
层级：0=(station,elev,var,lead) 1=(station,elev,lead) 2=(station,elev) 3=(station) 4=()；
apply 时每档取 n ≥ n_min(τ) 的最细层。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .decision import MODEL_LEVELS, CAL_LEVELS
except ImportError:
    from decision import MODEL_LEVELS, CAL_LEVELS

KEY_LEVELS = (
    ("station_id", "elev_bin", "var_bin", "lead_bin"),
    ("station_id", "elev_bin", "lead_bin"),
    ("station_id", "elev_bin"),
    ("station_id",),
    (),
)
_ALL_KEYS = ("station_id", "elev_bin", "var_bin", "lead_bin")
_EPS = 1e-6
_META_FIELDS = ("levels", "model_levels", "normalization", "elev_edges", "lead_edges",
                "use_variability", "n_min_factor")


@dataclass
class CalibrationConfig:
    levels: Sequence[float] = CAL_LEVELS
    model_levels: Sequence[float] = MODEL_LEVELS
    normalization: str = "clear_sky"          # clear_sky | model_spread | none
    elev_edges: Sequence[float] = (20.0, 45.0)
    lead_edges: Sequence[int] = (4, 8, 16, 55, 103)
    use_variability: bool = True
    n_min_factor: float = 10.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "CalibrationConfig":
        c = cfg["calibration"]; s = c["strata"]
        return cls(levels=tuple(c["levels"]), model_levels=MODEL_LEVELS,
                   normalization=c["normalization"], elev_edges=tuple(s["solar_elevation_edges"]),
                   lead_edges=tuple(s["lead_edges"]), use_variability=bool(s["variability"]),
                   n_min_factor=float(c["n_min_factor"]))


def _cfg_meta(cfg: CalibrationConfig) -> dict:
    """cfg 中影响 apply 数值结果的字段，用于 save/load 的一致性校验。"""
    return {
        "levels": [float(l) for l in cfg.levels],
        "model_levels": [float(l) for l in cfg.model_levels],
        "normalization": cfg.normalization,
        "elev_edges": [float(e) for e in cfg.elev_edges],
        "lead_edges": [int(e) for e in cfg.lead_edges],
        "use_variability": bool(cfg.use_variability),
        "n_min_factor": float(cfg.n_min_factor),
    }


def _meta_path(path: str | Path) -> Path:
    return Path(str(path) + ".json")


def quantile_cols(levels) -> list[str]:
    return [f"q_{float(l):g}" for l in levels]


def nearest_model_level(tau: float, model_levels) -> float:
    return float(min(model_levels, key=lambda m: abs(m - tau)))


def n_min(tau: float, factor: float) -> int:
    return int(math.ceil(factor / min(tau, 1.0 - tau)))


def bin_elev(elev_deg, edges) -> np.ndarray:
    return np.digitize(np.asarray(elev_deg, float), np.asarray(edges, float)).astype(int)


def bin_lead(lead, edges) -> np.ndarray:
    return np.digitize(np.asarray(lead, int), np.asarray(edges, int), right=False).astype(int)


class Calibrator:
    def __init__(self, cfg: CalibrationConfig):
        self.cfg = cfg
        self.table: pd.DataFrame = pd.DataFrame()
        self._lookup: dict[int, pd.DataFrame] = {}
        self.last_levels: np.ndarray | None = None       # 上次 apply：[N,L] 每格用的层号，-1=未校准
        self.last_stats: dict | None = None               # 上次 apply 的可观测性统计

    # ------------------------------------------------------------------ 内部
    @property
    def is_fitted(self) -> bool:
        return len(self.table) > 0

    def _norm(self, q_raw: np.ndarray, p_clear: np.ndarray) -> np.ndarray:
        if self.cfg.normalization == "clear_sky":
            return np.maximum(np.asarray(p_clear, float), _EPS)
        if self.cfg.normalization == "model_spread":
            return np.maximum(q_raw[:, -1] - q_raw[:, 0], _EPS)
        return np.ones(len(q_raw), float)

    def _keys(self, station_id, elev_deg, var_bin, lead) -> pd.DataFrame:
        k = pd.DataFrame({
            "station_id": np.asarray(station_id, int),
            "elev_bin": bin_elev(elev_deg, self.cfg.elev_edges),
            "var_bin": np.asarray(var_bin, int) if self.cfg.use_variability else 0,
            "lead_bin": bin_lead(lead, self.cfg.lead_edges),
        })
        return k

    def _model_idx(self, tau: float) -> int:
        return list(self.cfg.model_levels).index(nearest_model_level(tau, self.cfg.model_levels))

    # ------------------------------------------------------------------ fit
    def fit(self, table: pd.DataFrame) -> "Calibrator":
        qc = quantile_cols(self.cfg.model_levels)
        q_raw = table[qc].to_numpy(float)
        y = table["y"].to_numpy(float)
        norm = self._norm(q_raw, table["p_clear"].to_numpy(float))
        keys = self._keys(table["station_id"], table["elev_deg"], table["var_bin"], table["lead"])
        rows = []
        for level, kcols in enumerate(KEY_LEVELS):
            groups = keys.groupby(list(kcols)).indices if kcols else {(): np.arange(len(keys))}
            for gkey, idx in groups.items():
                gkey = gkey if isinstance(gkey, tuple) else (gkey,)
                n = len(idx)
                for tau in self.cfg.levels:
                    s = (y[idx] - q_raw[idx, self._model_idx(tau)]) / norm[idx]
                    k = min(max(int(math.ceil((n + 1) * tau)), 1), n)
                    delta = float(np.partition(s, k - 1)[k - 1])
                    rec = {c: -1 for c in _ALL_KEYS}
                    rec.update(dict(zip(kcols, [int(v) for v in gkey])))
                    rec.update(level=level, tau=float(tau), delta=delta, n=int(n))
                    rows.append(rec)
        self.table = pd.DataFrame(rows)
        self._build_lookup()
        return self

    def _build_lookup(self):
        self._lookup = {}
        for level, kcols in enumerate(KEY_LEVELS):
            t = self.table[self.table["level"] == level]
            wide = t.pivot_table(index=list(kcols) if kcols else None, columns="tau",
                                 values=["delta", "n"]) if kcols else None
            if kcols:
                self._lookup[level] = wide
            else:
                self._lookup[level] = t.set_index("tau")[["delta", "n"]]

    # ------------------------------------------------------------------ apply
    def apply(self, q_raw, station_id, elev_deg, var_bin, lead, p_clear) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Calibrator 未拟合")
        q_raw = np.asarray(q_raw, float)
        N = len(q_raw); L = len(self.cfg.levels)
        norm = self._norm(q_raw, p_clear)
        keys = self._keys(station_id, elev_deg, var_bin, lead)
        taus = np.asarray(self.cfg.levels, float)
        nmin = np.array([n_min(t, self.cfg.n_min_factor) for t in taus])
        delta = np.full((N, L), np.nan); done = np.zeros((N, L), bool)
        levels_used = np.full((N, L), -1, dtype=int)
        for level, kcols in enumerate(KEY_LEVELS):
            if kcols:
                # pivot 的列 = 升序 tau，与 self.cfg.levels（也升序）一一对应；merge 后列序 = key 列 + tau 列
                wide = self._lookup[level]
                d = keys[list(kcols)].merge(wide["delta"].reset_index(), how="left", on=list(kcols)) \
                    .iloc[:, len(kcols):].to_numpy(float)
                n = keys[list(kcols)].merge(wide["n"].reset_index(), how="left", on=list(kcols)) \
                    .iloc[:, len(kcols):].to_numpy(float)
            else:
                g = self._lookup[level].reindex(taus)
                d = np.broadcast_to(g["delta"].to_numpy(float), (N, L)).copy()
                n = np.broadcast_to(g["n"].to_numpy(float), (N, L)).copy()
            ok = (~done) & np.isfinite(d) & (np.nan_to_num(n, nan=0.0) >= nmin[None, :])
            delta[ok] = d[ok]; levels_used[ok] = level; done |= ok
        delta = np.nan_to_num(delta, nan=0.0)                       # 全体层也不够时不校准（levels_used 仍为 -1）
        base = np.stack([q_raw[:, self._model_idx(t)] for t in taus], axis=1)
        out = base + delta * norm[:, None]
        self.last_levels = levels_used
        uncal = levels_used == -1
        self.last_stats = {
            "n": int(N),
            "uncalibrated_cells": int(uncal.sum()),
            "uncalibrated_frac": float(uncal.mean()) if N * L > 0 else 0.0,
            "per_tau_uncalibrated_frac": {float(t): float(uncal[:, j].mean())
                                          for j, t in enumerate(taus)} if N > 0 else {float(t): 0.0 for t in taus},
        }
        return np.sort(out, axis=1)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path):
        """落盘 delta 长表 + 同名 `<path>.json` sidecar（拟合时影响 apply 数值的 cfg 字段），供 load 一致性校验。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(path, index=False)
        _meta_path(path).write_text(json.dumps(_cfg_meta(self.cfg)), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, cfg: CalibrationConfig) -> "Calibrator":
        """cfg 须与 save 时的配置一致（levels/model_levels/normalization/elev_edges/lead_edges/
        use_variability/n_min_factor 逐项比对），否则抛 ValueError 指明首个不一致字段；缺 sidecar 同样抛错（需重新 fit）。"""
        mp = _meta_path(path)
        if not mp.exists():
            raise ValueError(f"缺少校准元数据 {mp}：无法校验 cfg 一致性，需重新 fit")
        saved = json.loads(mp.read_text(encoding="utf-8"))
        want = _cfg_meta(cfg)
        for k in _META_FIELDS:
            if saved.get(k) != want[k]:
                raise ValueError(f"calibration 元数据字段 {k!r} 与拟合时不一致："
                                 f"saved={saved.get(k)!r} vs cfg={want[k]!r}，需重新 fit")
        c = cls(cfg)
        c.table = pd.read_parquet(path)
        c._build_lookup()
        return c


def result_to_long(result_df: pd.DataFrame, fit_steps, model_levels=MODEL_LEVELS) -> pd.DataFrame:
    """evaluate_model 的 result_df（每行 192 步数组）→ 校准拟合长表（每行一步）。
    只保留 fit_steps 区间内、非夜间、真值有限、p_clear>0 的 (行, 步)。"""
    qc = quantile_cols(model_levels)
    keep = np.zeros(len(result_df.iloc[0]["observe_power_future"]), bool)
    for lo, hi in fit_steps:
        keep[int(lo):int(hi)] = True
    steps = np.flatnonzero(keep)
    parts = []
    # 用 to_dict("records") 而不是 itertuples：后者会把 'q_0.1' 这类非标识符列名改成位置字段
    for r in result_df.to_dict("records"):
        y = np.asarray(r["observe_power_future"], float)[steps]
        night = np.asarray(r["night"], bool)[steps]; pc = np.asarray(r["p_clear"], float)[steps]
        ok = np.isfinite(y) & ~night & (pc > 0)
        if not ok.any():
            continue
        d = {"station_id": int(r["station_id"]), "lead": steps[ok], "elev_deg": np.asarray(r["elev_deg"], float)[steps][ok],
             "var_bin": int(r["var_bin"]), "y": y[ok], "p_clear": pc[ok]}
        for c in qc:
            d[c] = np.asarray(r[c], float)[steps][ok]
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["station_id", "lead", "elev_deg", "var_bin", "y", "p_clear", *qc])
