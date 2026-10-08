# -*- coding: utf-8 -*-
"""fallback.py —— 缺失补全链（config.fallback.chains）。构建特征库与缺失消融共用，将来可原样搬到线上。

策略（逐行、按协变量对）：
  1. 一行里该量的 (hist+fut) 非有限值比例 <= max_missing_frac：行内线性内插，不换源。
  2. 超过阈值：按链顺序找第一个"自身缺失比例 <= 阈值"的替代量，整行（历史+未来一起）换成它 —— 同源原则。
  3. 链走完仍没有：记为 dropped，该行置 NaN。上层（run_experiments）对 dropped 比例过高的协变量对整体移除。

链元素写法：
  标准名                 已有列，如 ghi_gdfs
  "const:25"             常数
  "skt_ec-8" / "x+1.5"   已有列加减常数
  ghi_kt_recent          晴空 GHI × 最近 kt_recent_days 天白天 observe_power/p_clear 的中位数
  ghi_clear_clock        时钟几何晴空（缺坐标兜底）
  tcc_from_kt            由 kt 反推云量 ((1-kt)/0.75)^(1/3.4)
  temp_climatology       历史段同时刻均值铺到未来（日循环气候态）
  dhi_erbs               Erbs 分解（需 ghi_cons/kt_cons 与 ghi_clear）
  drop                   显式终止：后面不再尝试
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ops import BuildContext, Pair, clock_clear_sky

_EXPR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*([+-])\s*([0-9.]+)$")


@dataclass
class FallbackReport:
    name: str
    n_rows: int
    n_interp: int = 0
    filled_by: Dict[str, int] = field(default_factory=dict)
    n_dropped: int = 0

    def as_dict(self) -> dict:
        return {"name": self.name, "n_rows": self.n_rows, "n_interp": self.n_interp,
                "filled_by": dict(self.filled_by), "n_dropped": self.n_dropped}


# ----------------------------------------------------------------------------- 工具
def missing_frac(full: np.ndarray, context_len: Optional[int] = None) -> np.ndarray:
    """逐行缺失比例。给了 context_len 时按历史段 / 未来段分别算并取较大者：
    未来 192 点缺一半不能靠整段 864 点摊薄成 11% 而被内插糊过去。"""
    if context_len is None or context_len >= full.shape[1]:
        return 1.0 - np.isfinite(full).mean(axis=1)
    h = 1.0 - np.isfinite(full[:, :context_len]).mean(axis=1)
    f = 1.0 - np.isfinite(full[:, context_len:]).mean(axis=1)
    return np.maximum(h, f)


def interp_rows(full: np.ndarray) -> np.ndarray:
    """逐行线性内插非有限值；端点用最近有效值；整行无效保持 NaN。"""
    out = full.astype(float).copy()
    idx = np.arange(out.shape[1])
    for i in range(out.shape[0]):
        m = np.isfinite(out[i])
        if m.all() or not m.any():
            continue
        out[i, ~m] = np.interp(idx[~m], idx[m], out[i, m])
    return out


# ----------------------------------------------------------------------------- 特殊替代量
def _ghi_kt_recent(ctx: BuildContext, target: str = "observe_power", clear: str = "ghi_clear",
                   min_clear: float = 50.0, days: int = 7) -> Optional[np.ndarray]:
    """每行：用自己历史段最近 days 天的白天 observe_power / (cap·ghi_clear/1000) 中位数当 kt，乘晴空 GHI。"""
    if not (ctx.has(target) and ctx.has(clear)):
        return None
    p = ctx.get(target).hist.astype(float)
    c_full = ctx.get(clear).full().astype(float)
    c_hist = c_full[:, :ctx.context_len]
    n = min(ctx.context_len, days * 96)
    p_clear = ctx.cap.astype(float)[:, None] * c_hist / 1000.0
    ok = (c_hist >= min_clear) & np.isfinite(p) & (p_clear > 0)
    ok[:, :-n] = False
    kt = np.where(ok, p / np.where(ok, p_clear, 1.0), np.nan)
    kt_med = np.nanmedian(np.where(ok, kt, np.nan), axis=1)
    kt_med = np.where(np.isfinite(kt_med), np.clip(kt_med, 0.05, 1.2), 0.6)
    return c_full * kt_med[:, None]


def _temp_climatology(ctx: BuildContext, name: str) -> Optional[np.ndarray]:
    """历史段按一天 96 个时刻求均值，铺满历史+未来。"""
    if not ctx.has(name):
        return None
    h = ctx.get(name).hist.astype(float)
    B, C = h.shape
    L = C + ctx.pred_len
    # 历史最后一点是起报时刻 S；时刻槽 = (j - (C-1)) mod 96 相对 S 的偏移
    slot_hist = (np.arange(C) - (C - 1)) % 96
    clim = np.full((B, 96), np.nan)
    for s in range(96):
        col = h[:, slot_hist == s]
        with np.errstate(all="ignore"):
            clim[:, s] = np.nanmean(col, axis=1)
    slot_full = (np.arange(L) - (C - 1)) % 96
    return clim[:, slot_full]


def _tcc_from_kt(ctx: BuildContext, kt: str = "kt_cons", a: float = 0.75, b: float = 3.4) -> Optional[np.ndarray]:
    if not ctx.has(kt):
        return None
    k = np.clip(ctx.get(kt).full().astype(float), 0.0, 1.0)
    return np.clip(((1.0 - k) / a), 0.0, 1.0) ** (1.0 / b)


def _dhi_erbs(ctx: BuildContext, ghi: str = "ghi_cons", kt: str = "kt_cons") -> Optional[np.ndarray]:
    if not (ctx.has(ghi) and ctx.has(kt)):
        return None
    g = ctx.get(ghi).full().astype(float)
    k = np.clip(ctx.get(kt).full().astype(float), 0.0, 1.3)
    df = np.where(k <= 0.22, 1.0 - 0.09 * k,
                  np.where(k <= 0.80, 0.9511 - 0.1604 * k + 4.388 * k**2 - 16.638 * k**3 + 12.336 * k**4, 0.165))
    return g * df


SPECIALS = {
    "ghi_kt_recent": lambda ctx, cfg: _ghi_kt_recent(ctx, days=int(cfg.get("kt_recent_days", 7))),
    "ghi_clear_clock": lambda ctx, cfg: clock_clear_sky(ctx.timestamps()),
    "tcc_from_kt": lambda ctx, cfg: _tcc_from_kt(ctx),
    "dhi_erbs": lambda ctx, cfg: _dhi_erbs(ctx),
}


def resolve_item(ctx: BuildContext, item: str, name: str, cfg: dict) -> Optional[np.ndarray]:
    """链元素 -> 全长数组 [B, C+P]，不可用返回 None。"""
    if item == "drop":
        return None
    if item.startswith("const:"):
        return np.full((len(ctx.ts_win), ctx.context_len + ctx.pred_len), float(item[6:]))
    if item == "temp_climatology":
        return _temp_climatology(ctx, name)
    if item in SPECIALS:
        return SPECIALS[item](ctx, cfg)
    m = _EXPR.match(item)
    if m:
        base, sign, num = m.groups()
        if not ctx.has(base):
            return None
        v = ctx.get(base).full().astype(float)
        return v + float(num) if sign == "+" else v - float(num)
    if ctx.has(item):
        return ctx.get(item).full().astype(float)
    return None


# ----------------------------------------------------------------------------- 主入口
def apply_fallback(ctx: BuildContext, name: str, chain: List[str], cfg: dict) -> FallbackReport:
    """对标准量 name 施加补全链，就地更新 ctx.cols[name]，返回报告。name 完全不存在时整列从链构造。"""
    thr = float(cfg.get("max_missing_frac", 0.3))
    B, L = len(ctx.ts_win), ctx.context_len + ctx.pred_len
    rep = FallbackReport(name=name, n_rows=B)
    full = ctx.get(name).full().astype(float) if ctx.has(name) else np.full((B, L), np.nan)

    mf = missing_frac(full, ctx.context_len)
    need = mf > thr
    small = (mf > 0) & ~need
    if small.any():
        full[small] = interp_rows(full[small])
        rep.n_interp = int(small.sum())

    todo = need.copy()
    for item in chain:
        if not todo.any():
            break
        if item == "drop":
            break
        cand = resolve_item(ctx, item, name, cfg)
        if cand is None:
            continue
        cmf = missing_frac(cand, ctx.context_len)
        take = todo & (cmf <= thr)
        if take.any():
            sub = cand[take]
            sub = interp_rows(sub)
            full[take] = sub
            rep.filled_by[item] = int(take.sum())
            todo &= ~take
    if todo.any():
        full[todo] = np.nan
        rep.n_dropped = int(todo.sum())
    ctx.cols[name] = Pair.from_full(full, ctx.context_len)
    return rep
