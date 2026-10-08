# -*- coding: utf-8 -*-
"""make_ds.py —— 按 config.report_times 逐起报时刻调用 DataSetInterface.prepare_infer_data()，
把 ODS→DS 的全字段轨迹表落成 parquet（镜像 multi_infer.get_feature_store 的循环，去掉线上副作用）。

只在能访问原始气象源的机器上运行（部署机，或拷了 vendor/ods_transform 且挂了数据盘的训练机）。
产物目录：{ods.out_dir}/date=YYYY-MM-DD/time=HH:MM/{YYYYMMDDHHMM}.parquet，
之后 build_feature_store.py 用 data.input_mode=ods 读取。

用法（cwd=仓库根）：
    python experiments/nwp_covariate/make_ds.py --config experiments/nwp_covariate/config.yaml \
        --predict_type ultra_short [--start ... --end ... --step 15min] [--overwrite]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger("nwp_cov.make_ds")


# ----------------------------------------------------------------------------- config
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def report_time_list(start: str, end: str, step: str) -> list[str]:
    """闭区间等步长起报时刻列表，格式 'YYYY-MM-DD HH:MM:SS'（与 multi_infer 一致）。"""
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=step)
    return [t.strftime("%Y-%m-%d %H:%M:%S") for t in idx]


def out_path(out_dir: Path, report_time: str) -> Path:
    t = pd.Timestamp(report_time)
    return out_dir / f"date={t:%Y-%m-%d}" / f"time={t:%H:%M}" / f"{t:%Y%m%d%H%M}.parquet"


# ----------------------------------------------------------------------------- import DataSetInterface
def import_dataset_interface(ods_cfg: dict):
    """优先 vendor 目录（离线复用），其次部署机 sys.path。返回 DataSetInterface 类。"""
    vendor = REPO_ROOT / ods_cfg.get("vendor_dir", "experiments/nwp_covariate/vendor")
    if (vendor / "ods_transform").is_dir():
        sys.path.insert(0, str(vendor))
        log.info("使用 vendor 目录的 ods_transform: %s", vendor)
    for p in ods_cfg.get("sys_path", []) or []:
        if p not in sys.path:
            sys.path.append(p)
    from ods_transform.dataset_inferface import DataSetInterface  # noqa: WPS433
    return DataSetInterface


# ----------------------------------------------------------------------------- station table
def load_area_stations(info_csv: str, area: str) -> pd.DataFrame:
    """与 multi_infer 相同的站表构造：plantId / lng / lat / cap_power_on，按省份与站名前缀过滤。"""
    stations_df = pd.read_csv(info_csv)
    stations_df = stations_df[["plant_pointname", "LONGITUDE", "LATITUDE", "GCCAPACITY", "province_pointname"]]
    stations_df = stations_df.rename(columns={
        "plant_pointname": "plantId", "LONGITUDE": "lng", "LATITUDE": "lat", "GCCAPACITY": "cap_power_on"})
    area_type = "_solar" if area.endswith("_solar") else "_wind"
    province = area.replace(area_type, "")
    suffix = "plant_guangfu" if area.endswith("_solar") else "plant_fengdian"
    st = stations_df[stations_df["province_pointname"] == province]
    st = st[st["plantId"].str.startswith(suffix)]
    return st[["plantId", "lng", "lat", "cap_power_on"]].copy()


def clean_ods_cache(infer_dir: str, area: str):
    for suffix in ("short", "ultra_short"):
        p = os.path.join(infer_dir, "ods", f"{area}_{suffix}_ods_infer_cache.parquet")
        if os.path.exists(p):
            os.remove(p)
            log.info("缓存已删除: %s", p)


# ----------------------------------------------------------------------------- main loop
def make_ds(cfg: dict, predict_type: str, start=None, end=None, step=None, overwrite=False) -> list[Path]:
    ods = cfg["ods"]
    rt = cfg["report_times"][predict_type]
    times = report_time_list(start or rt["start"], end or rt["end"], step or rt["step"])
    out_dir = REPO_ROOT / ods["out_dir"].format(predict_type=predict_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("predict_type=%s 起报时刻 %d 个: %s .. %s -> %s", predict_type, len(times), times[0], times[-1], out_dir)

    DataSetInterface = import_dataset_interface(ods)
    area_df = load_area_stations(ods["info_csv"], ods["area"])
    log.info("area=%s 站数=%d", ods["area"], len(area_df))

    written, failed = [], []
    for i, t in enumerate(times, 1):
        dst = out_path(out_dir, t)
        if dst.exists() and not overwrite:
            log.info("[%d/%d] SKIP 已存在 %s", i, len(times), dst)
            written.append(dst)
            continue
        t0 = time.time()
        try:
            if ods.get("clean_cache_each_call", True):
                clean_ods_cache(ods["save_infer_data_dir"], ods["area"])
            df, ok = DataSetInterface(
                save_history_data_dir=ods["save_history_data_dir"],
                save_infer_data_dir=ods["save_infer_data_dir"],
                logger=log,
            ).prepare_infer_data(
                forecast_time=t,
                predict_type=predict_type,
                data_path=ods["data_path"],
                area_name=ods["area"],
                plants_df=area_df,
                predict_seq_lens=int(ods["predict_seq_lens"][predict_type]),
                history_len=int(ods["history_len"]),
                only_power=bool(ods.get("only_power", False)),
            )
            if not ok or df is None or len(df) == 0:
                raise RuntimeError("prepare_infer_data 返回 success_flag=False 或空表")
            df = df.drop(columns=[c for c in ("update_time",) if c in df.columns])
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)
            written.append(dst)
            log.info("[%d/%d] OK %s shape=%s cols=%d %.1fs", i, len(times), dst.name, df.shape, df.shape[1], time.time() - t0)
        except Exception as e:  # noqa: BLE001 —— 回测态单个起报失败不应中断整批
            failed.append((t, repr(e)))
            log.warning("[%d/%d] FAIL report_time=%s: %r", i, len(times), t, e)
    log.info("完成: 成功 %d, 失败 %d", len(written), len(failed))
    for t, e in failed:
        log.info("  失败 %s -> %s", t, e)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--predict_type", choices=["short", "ultra_short"], default=None)
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--step")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    pt = a.predict_type or cfg["data"]["predict_type"]
    make_ds(cfg, pt, a.start, a.end, a.step, a.overwrite)


if __name__ == "__main__":
    main()
