# -*- coding: utf-8 -*-
"""screen.py —— 免模型筛选（阶段 0）。

线性尺（自实现 XReg）：对每个起报窗，在历史 context_len 点上做岭回归 observe_power ~ X_hist，
把系数用到 X_fut 上，与 observe_power_future 算样本外 R² / nRMSE（按容量归一）。
  - single：每个候选单独作 X，得该候选的线性信息量；
  - delta：base 特征 + 候选，得在 base 基础上的增量 ΔR²。
非线性尺（可选）：LightGBM 按 lead 展平，站内时间切分，看去掉该特征后的 ΔRMSE（训练机装了 lightgbm 才启用）。

用法（cwd=仓库根）：
    python experiments/nwp_covariate/screen.py --config experiments/nwp_covariate/config.yaml [--predict_type ultra_short]
产出：${run.out_dir}/screening.csv（+ .xlsx 若有 xlsxwriter/openpyxl），按站 × 候选给 R²、nRMSE、ΔR²，以及跨站汇总。
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_feature_store import feature_store_dir, load_config, _abs  # noqa: E402

log = logging.getLogger("nwp_cov.screen")
ID_COLS = {"timestamp_win", "station", "cap_power_on", "lat", "lon"}


# ----------------------------------------------------------------------------- 岭回归 XReg
def ridge_fit_predict(Xh: np.ndarray, yh: np.ndarray, Xf: np.ndarray, alpha: float) -> np.ndarray:
    """单窗：Xh [C,k], yh [C], Xf [P,k] -> yf_hat [P]。列标准化 + 截距 + 岭；NaN 行剔除。"""
    ok = np.isfinite(yh) & np.isfinite(Xh).all(1)
    if ok.sum() < max(10, Xh.shape[1] + 2):
        return np.full(Xf.shape[0], np.nan)
    Xh, yh = Xh[ok], yh[ok]
    mu, sd = Xh.mean(0), Xh.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Zh = (Xh - mu) / sd
    A = Zh.T @ Zh + alpha * np.eye(Zh.shape[1])
    beta = np.linalg.solve(A, Zh.T @ (yh - yh.mean()))
    Zf = (np.nan_to_num(Xf) - mu) / sd
    return Zf @ beta + yh.mean()


def window_scores(Xh, yh, Xf, yf, cap, alpha) -> tuple:
    yhat = ridge_fit_predict(Xh, yh, Xf, alpha)
    ok = np.isfinite(yf) & np.isfinite(yhat)
    if ok.sum() < 8:
        return np.nan, np.nan
    e = yf[ok] - yhat[ok]
    ss_tot = ((yf[ok] - yf[ok].mean()) ** 2).sum()
    r2 = 1 - (e ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    nrmse = np.sqrt((e ** 2).mean()) / max(cap, 1e-6) * 100
    return r2, nrmse


def stack(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=float) for v in df[col].to_numpy()])


def candidates_of(df: pd.DataFrame, target: str, label: str, suffix: str) -> list:
    return sorted(c for c in df.columns if c + suffix in df.columns and c not in ID_COLS and c not in (target, label))


def screen_station(df: pd.DataFrame, cfg: dict, cands: list) -> pd.DataFrame:
    d = cfg["data"]; sc = cfg["screening"]["ridge"]
    suffix = cfg["columns"].get("future_suffix", "_predict")
    alpha = float(sc.get("alpha", 1.0))
    base = [b for b in (sc.get("base") or []) if b in df.columns]
    yh_all, yf_all = stack(df, d["target"]), stack(df, d["label"])
    caps = df["cap_power_on"].to_numpy(float)
    feats = {c: (stack(df, c), stack(df, c + suffix)) for c in cands + base}
    rows = []
    for c in cands:
        r2s, nr, dr2 = [], [], []
        for i in range(len(df)):
            Xh = feats[c][0][i][:, None]; Xf = feats[c][1][i][:, None]
            r2, n = window_scores(Xh, yh_all[i], Xf, yf_all[i], caps[i], alpha); r2s.append(r2); nr.append(n)
            if base and c not in base:
                Bh = np.column_stack([feats[b][0][i] for b in base]); Bf = np.column_stack([feats[b][1][i] for b in base])
                rb, _ = window_scores(Bh, yh_all[i], Bf, yf_all[i], caps[i], alpha)
                rbc, _ = window_scores(np.column_stack([Bh, Xh]), yh_all[i], np.column_stack([Bf, Xf]), yf_all[i], caps[i], alpha)
                dr2.append(rbc - rb)
        rows.append({"station": df["station"].iloc[0], "candidate": c, "n_windows": len(df),
                     "r2_single": np.nanmean(r2s), "nrmse_single_pct": np.nanmean(nr),
                     "delta_r2_over_base": np.nanmean(dr2) if dr2 else np.nan,
                     "base": "+".join(base)})
    return pd.DataFrame(rows)


def screen_lgbm(frames: dict, cfg: dict, cands: list) -> pd.DataFrame:
    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        log.warning("lightgbm 未安装，跳过非线性尺")
        return pd.DataFrame()
    import lightgbm as lgb
    d = cfg["data"]; suffix = cfg["columns"].get("future_suffix", "_predict")
    n_est = int(cfg["screening"]["lgbm"].get("n_estimators", 400))
    rows = []
    for st, df in frames.items():
        df = df.sort_values("timestamp_win")
        yf = stack(df, d["label"])
        X = np.stack([stack(df, c + suffix) for c in cands], axis=-1)        # [B, P, k]
        B, P, k = X.shape
        lead = np.tile(np.arange(P), (B, 1))[..., None]
        Xf = np.concatenate([X, lead], axis=-1).reshape(B * P, k + 1)
        y = yf.reshape(-1)
        split = int(B * 0.7) * P
        ok = np.isfinite(y) & np.isfinite(Xf).all(1)
        tr = ok.copy(); tr[split:] = False; te = ok.copy(); te[:split] = False
        if tr.sum() < 100 or te.sum() < 50:
            continue
        def fit_rmse(cols):
            m = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=0.05, num_leaves=31, verbose=-1)
            m.fit(Xf[tr][:, cols], y[tr])
            p = m.predict(Xf[te][:, cols])
            return np.sqrt(np.mean((p - y[te]) ** 2)), m
        all_cols = list(range(k + 1))
        rmse_all, m_all = fit_rmse(all_cols)
        gain = dict(zip(cands + ["lead"], m_all.booster_.feature_importance("gain")))
        for j, c in enumerate(cands):
            rmse_wo, _ = fit_rmse([x for x in all_cols if x != j])
            rows.append({"station": st, "candidate": c, "lgbm_gain": gain[c], "lgbm_delta_rmse_drop": rmse_wo - rmse_all})
    return pd.DataFrame(rows)


def run(cfg: dict, predict_type: str, fs_dir: Path | None = None) -> pd.DataFrame:
    fs_dir = fs_dir or feature_store_dir(cfg, predict_type)
    files = sorted(glob.glob(str(fs_dir / "station=*.parquet")))
    if not files:
        raise FileNotFoundError(f"特征库为空: {fs_dir}")
    frames = {Path(f).stem.split("=", 1)[1]: pd.read_parquet(f) for f in files}
    d = cfg["data"]; suffix = cfg["columns"].get("future_suffix", "_predict")
    first = next(iter(frames.values()))
    spec = cfg["screening"]["ridge"].get("candidates", "all_derived")
    cands = candidates_of(first, d["target"], d["label"], suffix) if spec == "all_derived" else [c for c in spec if c in first.columns]
    log.info("候选 %d 个: %s", len(cands), cands)
    parts = [screen_station(df, cfg, cands) for df in frames.values()]
    ridge = pd.concat(parts, ignore_index=True)
    out = ridge
    if (cfg["screening"].get("lgbm") or {}).get("enabled"):
        lg = screen_lgbm(frames, cfg, cands)
        if not lg.empty:
            out = ridge.merge(lg, on=["station", "candidate"], how="left")
    summary = (out.groupby("candidate").agg(r2_single=("r2_single", "mean"), nrmse_single_pct=("nrmse_single_pct", "mean"),
                                            delta_r2_over_base=("delta_r2_over_base", "mean"),
                                            win_frac=("delta_r2_over_base", lambda s: float((s > 0).mean())))
               .sort_values("delta_r2_over_base", ascending=False).reset_index())
    out_path = _abs(cfg["screening"].get("out", "screening.xlsx").replace("${run.out_dir}", cfg["run"]["out_dir"].replace("${run.name}", cfg["run"]["name"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    out.to_csv(csv_path, index=False)
    summary.to_csv(csv_path.with_name(csv_path.stem + "_summary.csv"), index=False)
    try:
        with pd.ExcelWriter(out_path) as w:
            summary.to_excel(w, sheet_name="summary", index=False)
            out.to_excel(w, sheet_name="by_station", index=False)
    except (ImportError, ValueError):
        log.info("无 xlsx 写入引擎，只写 CSV")
    log.info("筛选结果 -> %s", csv_path)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--predict_type", default=None)
    ap.add_argument("--feature_store", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    pt = a.predict_type or cfg["data"]["predict_type"]
    s = run(cfg, pt, Path(a.feature_store) if a.feature_store else None)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
