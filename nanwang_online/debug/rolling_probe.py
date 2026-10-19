# -*- coding: utf-8 -*-
"""短期滚动推理（pipelines/rolling_wrapper_inference.rolling_inference）对齐探针。不改原文件，只调用它。

造一份线上口径的模拟 parquet（station 站名、general_GHI / general_TEMP 大写、tsfm_predict、observe_power），
每个序列的值 = 该点相对起报时刻的 15 分钟步号：历史 -H..-1，未来 0..479。于是模型每个窗口收到的东西一眼能看出是哪几步。

  python nanwang_online/debug/rolling_probe.py                 # 原版 rolling_inference，逐窗口打印 + 检查
  python nanwang_online/debug/rolling_probe.py --fixed         # 换成本文件里的修正版切片再跑一遍（仍不改原文件）
  python nanwang_online/debug/rolling_probe.py --hist_len 960  # 历史比 context(672) 长时的第二个问题
  python nanwang_online/debug/rolling_probe.py --ckpt runs/m8/xxx/j1_final2_s0   # 用真实 M8：原版 vs 修正版输出逐点比较

探针模型的"预测" = 它收到的历史功率最后一步 + 1, +2, ...（像持续性外推）。对齐正确时最终输出应该恰好是 0..479；
历史功率和气象错位多少步，输出那一段就偏多少。只靠未来气象出预测的模型看不出这个错位——这就是曲线看着正常的原因。
"""
import argparse
import math
import sys
from operator import itemgetter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pipelines.rolling_wrapper_inference as rw  # noqa: E402

C, P, F = 672, 192, 480                           # M8 context_len / pred_len，线上短期 PRED_LEN
HIS, PRED = ["general_GHI", "general_TEMP"], ["general_GHI_predict", "general_TEMP_predict"]
ROLLING_CONFIG = {"past_covariate": HIS, "future_covariate": PRED, "past_target": ["observe_power"],
                  "future_target": ["tsfm_predict"], "future_len": F}          # 与线上脚本 short 分支一字不差
FEATS = ["general_GHI", "general_GHI_predict", "general_TEMP", "general_TEMP_predict", "observe_power", "tsfm_predict"]
ENTITY = ("timestamp_win", "station", "station_id")


def make_index_parquet(path, hist_len=C, stations=("plant_guangfu1001", "plant_guangfu1002")):
    """值 = 步号的模拟 parquet。observe_power 历史和气象历史一样长（线上若不一样长，用 --power_len 单独改）。"""
    his, fut = np.arange(-hist_len, 0, dtype="float32"), np.arange(0, F, dtype="float32")
    rows = [dict(timestamp_win=pd.Timestamp("2026-10-01 10:00:00"), station=s, station_id=1001 + i,
                 observe_power=his.copy(), general_GHI=his.copy(), general_TEMP=his.copy(),
                 general_GHI_predict=fut.copy(), general_TEMP_predict=fut.copy(), tsfm_predict=fut.copy())
            for i, s in enumerate(stations)]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


class Probe(torch.nn.Module):
    """满足 unified_inference 约定的假模型：forward(batch_dict) → .logits；记下每个窗口收到的步号范围。"""
    context_length, prediction_length = C, P

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, bd, labels=None):
        g = lambda k: bd[k][0].float().numpy()                                  # 只看第一行（各站相同）
        rng = lambda a: (int(a[0]), int(a[-1]))
        self.calls.append({"历史气象": rng(g("general_GHI")), "历史功率": rng(g("observe_power")),
                           "未来气象": rng(g("general_GHI_predict")), "插值列": rng(g("tsfm_predict"))})
        last_p = bd["observe_power"][:, -1:].float()
        return SimpleNamespace(logits=last_p + 1 + torch.arange(P, dtype=torch.float32))


def build_sliding_windows_fixed(df, config, full_covariate, context_length, prediction_length, total_remain, remaining,
                                entity_cols):
    """修正版（只在本探针里替换，不动原文件）：所有切片都按"本窗口从未来第 e 步起报"算，e = r·P，最后一个窗口 e = F-P。
    气象用负下标（不依赖历史长度 = context）；功率 _full 里已经拼了 r·P 个预测，最后一个窗口要退回 r·P-e 步。"""
    Fl, P_, C_ = config["future_len"], prediction_length, context_length
    is_last = remaining == total_remain - 1
    e = Fl - P_ if is_last else remaining * P_
    end = Fl - e                                                                 # 起报点离全序列末尾多少步
    back = remaining * P_ - e                                                    # 功率 _full 末尾比起报点多出的步数
    get_past = itemgetter(slice(-end - C_, -end))
    get_future = itemgetter(slice(-end, -end + P_ or None))
    get_ptgt = itemgetter(slice(-back - C_, -back or None))
    get_tgt = itemgetter(slice(e, e + P_))
    cur = df[list(entity_cols)].copy()
    for past_col, full_col in zip(config["past_covariate"], full_covariate):
        cur[past_col] = df[full_col].map(get_past)
    for fut_col, full_col in zip(config["future_covariate"], full_covariate):
        cur[fut_col] = df[full_col].map(get_future)
    tgt = config["past_target"][0]
    cur[tgt] = df[tgt + "_full"].map(get_ptgt)
    for col in config["future_target"]:
        cur[col] = df[col].map(get_tgt)
    return cur


def run(model, df, fixed):
    orig = rw.build_sliding_windows
    if fixed:
        rw.build_sliding_windows = build_sliding_windows_fixed
    try:
        return rw.rolling_inference(model, df.copy(), config=ROLLING_CONFIG, timestamp_cols=("timestamp_win",),
                                    entity_cols=ENTITY, feature_cols=FEATS)
    finally:
        rw.build_sliding_windows = orig


def probe(args):
    path = make_index_parquet(Path(args.out), hist_len=args.hist_len)
    df = pd.read_parquet(path)
    print(f"模拟 parquet: {path}  历史 {args.hist_len} 步（-{args.hist_len}..-1），未来 {F} 步（0..{F - 1}）；"
          f"context={C} pred={P}，滚 {math.ceil(F / P)} 个窗口；{'修正版' if args.fixed else '原版'}切片\n")
    m = Probe()
    out = run(m, df, args.fixed)
    ok_all = True
    for r, c in enumerate(m.calls):
        start = F - P if r == len(m.calls) - 1 else r * P                         # 本窗口应该从未来第几步起报
        want = {"历史气象": (start - C, start - 1), "历史功率": (start - C, start - 1),
                "未来气象": (start, start + P - 1), "插值列": (start, start + P - 1)}
        print(f"窗口 {r}{'（最后一个）' if r == len(m.calls) - 1 else ''}: 应从第 {start} 步起报")
        for k, v in c.items():
            ok = v == want[k]
            ok_all &= ok
            print(f"   {k}: 步 {v[0]:>5} .. {v[1]:>4}   {'✓' if ok else f'✗ 应为 {want[k][0]} .. {want[k][1]}，差 {v[1] - want[k][1]:+d} 步'}")
    y = np.asarray(out[type(m).__name__].iloc[0])
    bad = np.flatnonzero(y != np.arange(F))
    print(f"\n最终输出 {len(y)} 步: 前 3 = {y[:3].astype(int).tolist()}，后 3 = {y[-3:].astype(int).tolist()}（应为 0..{F - 1}）")
    if len(bad):
        print(f"✗ 第 {bad[0]}..{bad[-1]} 步（{len(bad)} 个点）用错了历史功率，偏 {int(y[bad[0]] - bad[0]):+d} 步"
              f"（10:00 起报 → 第 384 步 = D+4 10:15，第 479 步 = D+5 10:00）")
    print("全部对齐 ✓" if ok_all and not len(bad) else "存在错位 ✗")


def compare_ckpt(args):
    """真实 M8：用合成晴空曲线跑原版 / 修正版，逐点比较。预期第 0..383 步完全相同，差异只出现在 384..479（只有最后一个窗口受影响）。
    合成数据每天形状差不多、夜里又归零，错开一天的历史功率差别不大，所以这里的差值偏小；用真实数据会更明显。"""
    import yaml
    from forecast_backbones.m8.transformer_models import TransformersAdaptedM8Model
    d = Path(args.ckpt)
    cfg = yaml.safe_load((d / "model_config.yaml").read_text(encoding="utf-8"))
    cfg["train"]["weights_path"] = str(d / "model.safetensors")
    if args.station_meta:
        cfg["dataset"]["station_meta"], cfg["dataset"]["station_meta_cols"] = args.station_meta, None
    m = TransformersAdaptedM8Model(cfg).eval()
    assert m.weights_loaded, f"权重没加载：{d}"
    meta = m.station_meta
    ids = [s for s in meta.ids() if ((meta.get(s) or {}).get("cap_mw") or 0) > 0][:3]
    t0 = pd.Timestamp("2026-10-01 10:00:00")
    t = t0 + pd.to_timedelta(np.arange(-args.hist_len, F) * 15, unit="min")
    hod = (t.hour + t.minute / 60).to_numpy()
    day = np.clip(np.sin((hod - 6.5) / 12 * np.pi), 0, None)
    rng = np.random.default_rng(0)
    rows = []
    for s in ids:
        cap = meta.get(s)["cap_mw"]
        ghi = (900 * day * (0.8 + 0.2 * rng.random(len(t)))).astype("float32")
        temp = (22 + 6 * day).astype("float32")
        pw = (cap * 0.75 * ghi / 900).astype("float32")
        H = args.hist_len
        rows.append(dict(timestamp_win=t0, station=meta.name(s) or str(s), station_id=s, observe_power=pw[:H],
                         general_GHI=ghi[:H], general_TEMP=temp[:H], general_GHI_predict=ghi[H:], general_TEMP_predict=temp[H:],
                         tsfm_predict=pw[H:]))
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    col = type(m).__name__
    a = np.stack(run(m, df, False)[col].to_numpy())
    b = np.stack(run(m, df, True)[col].to_numpy())
    diff = np.abs(a - b).max(axis=0)
    nz = np.flatnonzero(diff > 1e-6)
    print(f"\n真实 M8 {d}，{len(ids)} 个站，模拟数据存在 {args.out}")
    print(f"原版 vs 修正版：最大差 {diff.max():.4f} MW；不同的步 = "
          f"{'无' if not len(nz) else f'{nz[0]}..{nz[-1]}（{len(nz)} 个）'}；第 0..383 步最大差 {diff[:384].max():.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hist_len", type=int, default=C, help="模拟数据的历史长度（默认 672 = context）")
    ap.add_argument("--fixed", action="store_true", help="换成修正版切片（只在本进程里替换）")
    ap.add_argument("--ckpt", help="真实 M8 checkpoint 目录（含 model_config.yaml / model.safetensors）：原版 vs 修正版输出比较")
    ap.add_argument("--station_meta", help="配合 --ckpt：覆盖 dataset.station_meta（如线上 info.csv）")
    ap.add_argument("--out", default=str(Path(__file__).with_name("mock_short.parquet")), help="模拟 parquet 存到哪")
    a = ap.parse_args()
    compare_ckpt(a) if a.ckpt else probe(a)
