"""逐站 Mondrian 单侧 conformal 分位校准。
分数 s = (y - q_{τ0}) / norm，δ_{s,g,τ} = 第 ⌈(n+1)τ⌉ 小分数；q̃_τ = q_{τ0} + δ·norm。
层级：0=(station,elev,var,lead) 1=(station,elev,lead) 2=(station,elev) 3=(station) 4=()；
apply 时每档取 n ≥ n_min(τ) 的最细层。"""
from __future__ import annotations

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
            delta[ok] = d[ok]; done |= ok
        delta = np.nan_to_num(delta, nan=0.0)                       # 全体层也不够时不校准
        base = np.stack([q_raw[:, self._model_idx(t)] for t in taus], axis=1)
        out = base + delta * norm[:, None]
        return np.sort(out, axis=1)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(path, index=False)

    @classmethod
    def load(cls, path: str | Path, cfg: CalibrationConfig) -> "Calibrator":
        c = cls(cfg)
        c.table = pd.read_parquet(path)
        c._build_lookup()
        return c
