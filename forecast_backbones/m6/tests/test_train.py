import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("timesfm3")
import forecast_backbones.m6.transformer_models as tm_mod
from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
from forecast_backbones.m6.train import make_splits, build_training_args, train_m6
from forecast_backbones.m6.routing import load_routing
from pipelines.transformer_dataset import UnifiedDataset
from pipelines.time_splits import SplitPlan, StationSplitPlan
from .conftest import make_synthetic_frame, write_station_meta, CAP


def _patch_tiny_backbone(monkeypatch, evaluator):
    def _ensure(self, want):
        self._backbone = evaluator; self._backbone_device = "cpu"
        if self.torch_path and self.backbone is None:
            self._attach_torch_path()
    monkeypatch.setattr(TransformersAdaptedM6Model, "_ensure_backbone", _ensure)


def _inputs(tmp_path, n_days=12):
    df = make_synthetic_frame(n_days=n_days, issue_every_min=60, seed=0, start="2025-06-08")   # 06-08 … 06-20
    parquet = tmp_path / "data.parquet"; df.to_parquet(parquet, index=False)
    out = tmp_path / "run"
    cfg = {"dataset": {"station_meta": str(write_station_meta(tmp_path))},
           "run": {"out_dir": str(out)}, "eval": {"batch_size": 64},
           "finetune": {"enabled": True, "inference_path": "torch",
                        "schedule": {"batch_size": 4, "epochs_max": 1, "lr": 1e-3, "precision": "fp32", "logging_steps": 1},
                        # train [06-08,06-10) | purge | val [06-12,06-14) | purge | cal [06-16,06-17) | purge | test [06-19,06-20)
                        "split": {"train_start": None, "val_blocks": [["2025-06-12", "2025-06-14"]],
                                  "test_start": "2025-06-19", "test_end": None,
                                  "cal_days": 1, "purge_days": 2, "embargo_days": 7},
                        "early_stop": {"patience": 1}}}
    return df, parquet, out, cfg


def test_make_splits_boundaries_and_shift(tmp_path):
    df, parquet, _, cfg = _inputs(tmp_path)
    ds = UnifiedDataset(str(parquet), timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        timestamp_output="epoch_s", epoch_unit="s", for_hf_trainer=True)
    per_day = len(df) // 12
    tr, cal, val_all, blocks, plan = make_splits(ds, cfg["finetune"]["split"])
    assert len(tr) == 2 * per_day and len(cal) == per_day and len(val_all) == 2 * per_day
    assert len(blocks) == 1 and len(blocks[0]) == len(val_all)
    assert plan.to_dict()["cal"] == ["2025-06-16", "2025-06-17"]
    ts = lambda d: pd.to_datetime(d._ts_as_datetime64("timestamp_win"))
    assert ts(tr).max() < pd.Timestamp("2025-06-10") and ts(val_all).min() >= pd.Timestamp("2025-06-12")
    tr2, cal2, val2, blocks2, plan2 = make_splits(ds, cfg["finetune"]["split"], shift_days=1)      # 整体后移 1 天
    assert len(tr2) == 3 * per_day and len(cal2) == per_day and len(val2) == 2 * per_day          # train 起点不动，上限后移 1 天
    assert ts(tr2).min() >= pd.Timestamp("2025-06-08") and ts(tr2).max() < pd.Timestamp("2025-06-11")
    assert plan2.test_start == pd.Timestamp("2025-06-20")
    with pytest.raises(ValueError, match="段为空"):
        make_splits(ds, {**cfg["finetune"]["split"], "val_blocks": [["2025-07-01", "2025-07-03"]], "test_start": "2025-07-10"})  # 块与校准段都落在数据之外
    with pytest.raises(ValueError, match="cal_days"):
        make_splits(ds, {**cfg["finetune"]["split"], "cal_days": 0})
    with pytest.raises(ValueError, match="旧键"):
        make_splits(ds, {"train_end": "2025-06-11", "val_start": "2025-06-12", "val_end": "2025-06-14", "cal_days": 1, "test_start": "2025-06-14"})


def test_make_splits_with_two_real_val_blocks(tmp_path):
    """两个真验证块（06-05…06-07 与 06-18…06-20）：块 0 后的 9 天剔除一直吃到 06-16，正是块 1 的 purge 起点，
    两块之间连一行训练数据都不剩；训练行只剩 [06-01, 06-03)，校准段 [06-25, 06-27)。不需要模型。"""
    df = make_synthetic_frame(n_days=30, issue_every_min=60, seed=0, start="2025-06-01")   # 06-01 … 07-01，每小时一个起报
    parquet = tmp_path / "two_blocks.parquet"; df.to_parquet(parquet, index=False)
    ds = UnifiedDataset(str(parquet), timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        timestamp_output="epoch_s", epoch_unit="s", for_hf_trainer=True)
    split = {"train_start": None,
             "val_blocks": [["2025-06-05", "2025-06-07"], ["2025-06-18", "2025-06-20"]],
             "test_start": "2025-06-29", "test_end": None,
             "cal_days": 2, "purge_days": 2, "embargo_days": 7}
    tr, cal, val_all, blocks, plan = make_splits(ds, split)                               # 内部已跑过 verify_leak_free
    per_day = 2 * 24                                                                      # 2 站 × 每小时一个起报 = 48 行/天
    assert len(blocks) == 2 and [len(b) for b in blocks] == [2 * per_day, 2 * per_day]
    assert len(val_all) == sum(len(b) for b in blocks)
    ts = lambda d: pd.to_datetime(d._ts_as_datetime64("timestamp_win"))
    s0, s1 = set(ts(blocks[0])), set(ts(blocks[1]))
    assert not (s0 & s1) and max(s0) < min(s1)
    assert len(tr) == 2 * per_day                                                         # 只剩 [06-01, 06-03) 两天
    assert ts(tr).min() >= pd.Timestamp("2025-06-01") and ts(tr).max() < pd.Timestamp("2025-06-03")
    assert len(cal) == 2 * per_day and plan.to_dict()["cal"] == ["2025-06-25", "2025-06-27"]


def test_build_training_args_cpu_defaults(tmp_path):
    _, _, out, cfg = _inputs(tmp_path)
    from forecast_backbones.m6.config import load_config
    args = build_training_args(load_config(cfg), out, max_steps=3)
    assert args.max_steps == 3 and args.per_device_train_batch_size == 4 and args.remove_unused_columns is False
    assert args.evaluation_strategy.value == "no" and args.save_strategy.value == "no" and args.report_to == []
    assert args.bf16 is False and args.max_grad_norm == 1.0 and args.use_cpu is (not torch.cuda.is_available())


def test_train_m6_end_to_end_writes_all_artifacts(tmp_path, tiny_evaluator, monkeypatch):
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):                    # 训练前无权重：构造期 warn 一次
        res = train_m6(cfg, str(parquet), max_steps=2)
    # 产物
    assert (out / "lora_weights.pt").exists() and (out / "station_routing.json").exists()
    assert (out / "calibration.parquet").exists() and (out / "calibration.parquet.json").exists()
    assert (out / "train_log.json").exists() and (out / "result_val_lora.parquet").exists()
    ladder = pd.read_csv(out / "reports" / "ladder_lora.csv")
    assert list(ladder["variant"]) == ["zeroshot_calibrated_pstar", "lora_calibrated_pstar", "routed_calibrated_pstar"]
    assert np.isfinite(ladder["nanwang_short_D+1"]).all()
    st = pd.read_csv(out / "reports" / "station_ladder.csv")
    assert set(st["station_id"]) == set(CAP) and set(st["route"]) <= {"lora", "zeroshot"}
    routing = load_routing(out / "station_routing.json")
    assert routing == dict(zip(st["station_id"].astype(int), st["route"]))
    # 路由后的全体得分 ≥ 两者逐站取优的下界：不低于零样本行（逐站不输零样本）
    z = ladder.set_index("variant")["nanwang_short_D+1"]
    assert z["routed_calibrated_pstar"] >= z["zeroshot_calibrated_pstar"] - 1e-9
    log = json.loads((out / "train_log.json").read_text())
    assert log["history"][0]["epoch"] == 0.0 and len(log["history"]) >= 2 and "best" in log
    assert set(res) >= {"ladder", "routing", "history", "best"}
    assert res["ladder"]["variant"].tolist() == ladder["variant"].tolist() and res["routing"] == routing
    man = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert man["generated_by"] == "forecast_backbones.m6.train"                  # 与 pipelines.time_splits 的 manifest 区分开
    assert man["plan"]["val_blocks"] == [["2025-06-12", "2025-06-14"]] and man["plan"]["cal"] == ["2025-06-16", "2025-06-17"]
    assert man["counts"]["train"] > 0 and man["counts"]["val_blocks"] == [man["counts"]["val"]]
    assert log["split"] == man and res["split"] == man
    assert log["history"][0]["blocks"] and len(log["history"][0]["blocks"]) == 1


def test_trained_artifacts_load_into_inference_model(tmp_path, tiny_evaluator, monkeypatch):
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        train_m6(cfg, str(parquet), max_steps=2)
    from forecast_backbones.m6.config import load_config
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # 权重、路由、校准表齐全 → 零 warning
        m = TransformersAdaptedM6Model(load_config(cfg))                     # calibration.enabled 默认 True → 读训练产物
    assert m.lora_loaded is True and m.calibrator is not None and m.station_routing is not None
    from .test_model_route_a import _batch
    bd, _ = _batch(df, [len(df) - 4, len(df) - 3, len(df) - 2, len(df) - 1])
    with torch.no_grad():
        o = m(bd)
    assert o.logits.shape == (4, 192) and torch.isfinite(o.logits).all()


def test_train_m6_raises_clear_error_when_no_finite_score_ever_recorded(tmp_path, tiny_evaluator, monkeypatch):
    """南网打分全程非有限（没有一天能凑出完整可评分的一天）→ NanwangEvalCallback 从未记录到
    best_epoch/存过权重；train_m6 必须在 trainer.train() 之后立刻拦下来，报出点名 val 窗口的
    ValueError，而不是让后面的 torch.load(best_path) 摔出 FileNotFoundError。"""
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    import forecast_backbones.m6.nanwang_callback as nc
    monkeypatch.setattr(nc, "score_split", lambda *a, **k: (float("nan"), None, None))
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        with pytest.raises(ValueError) as exc:
            train_m6(cfg, str(parquet), max_steps=1)
    msg = str(exc.value)
    assert "2025-06-12" in msg and "2025-06-14" in msg and "非有限" in msg
    assert not (out / "lora_weights.pt").exists()


def test_second_run_in_same_out_dir_starts_fresh(tmp_path, tiny_evaluator, monkeypatch):
    """跨任务约束 1 的回归测试：同一 out_dir 里已经有上一轮训练的 lora_weights.pt / station_routing.json
    时，第二轮训练的模型构造仍须是零样本、无路由起点——不能把上一轮的产物读回来当成本轮的起点。"""
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        train_m6(cfg, str(parquet), max_steps=1)
    assert (out / "lora_weights.pt").exists() and (out / "station_routing.json").exists()

    # 第二轮换一份全新底座（同一份 tiny 权重目录重新构造）：避免两次构造共享同一个 evaluator.model 对象，
    # 让第一轮训练留在内存里的 LoRA 权重把"零样本起点"判定悄悄冲掉——真实场景里每轮训练都会从磁盘
    # checkpoint 重新加载一份全新底座，这里也应如此，spy 才测的是 train.py 自己的哨兵路径逻辑。
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    evaluator2 = TimesFM3Evaluator(ModelConfig(checkpoint_path=tiny_evaluator.config.checkpoint_path,
                                               per_core_batch_size=4, device="cpu"))
    _patch_tiny_backbone(monkeypatch, evaluator2)

    RealModel = tm_mod.TransformersAdaptedM6Model
    recorded = []

    class _SpyModel(RealModel):
        def __init__(self, model_config=None, **kw):
            super().__init__(model_config, **kw)
            recorded.append({"lora_path": self.cfg["finetune"]["lora_path"],
                             "routing_path": self.cfg["finetune"]["routing_path"],
                             "lora_loaded": self.lora_loaded, "station_routing": self.station_routing})

    monkeypatch.setattr(tm_mod, "TransformersAdaptedM6Model", _SpyModel)
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        train_m6(cfg, str(parquet), max_steps=1)

    assert len(recorded) == 1, "第二轮训练模型只应构造一次"
    rec = recorded[0]
    assert not Path(rec["lora_path"]).exists() and not Path(rec["routing_path"]).exists()
    assert rec["lora_loaded"] is False and rec["station_routing"] is None
    log2 = json.loads((out / "train_log.json").read_text())
    assert log2["history"][0]["epoch"] == 0.0


def test_make_splits_by_station_test_start(tmp_path):
    """站 1001 测试 06-28 容得下两个块；站 1002 测试 06-13（校准段起点 06-10）只容得下第一块，第二块只剩 1001 的行。"""
    df = make_synthetic_frame(n_days=30, issue_every_min=60, seed=0, start="2025-06-01")        # 06-01 … 07-01
    parquet = tmp_path / "d.parquet"; df.to_parquet(parquet, index=False)
    ds_all = UnifiedDataset(str(parquet), timestamp_cols=["timestamp_win"], entity_cols=[],
                            label_cols=["observe_power_future"], timestamp_output="epoch_s", epoch_unit="s",
                            for_hf_trainer=True)
    split = {"train_start": None, "val_blocks": [["2025-06-05", "2025-06-07"], ["2025-06-14", "2025-06-16"]],
             "test_start": {1001: "2025-06-28", 1002: "2025-06-13"}, "test_end": None,
             "cal_days": 1, "purge_days": 2, "embargo_days": 7}
    ds_train, ds_cal, ds_val, blocks, plan = make_splits(ds_all, split)
    assert isinstance(plan, StationSplitPlan) and plan.skipped_blocks == {"1001": [], "1002": [1]}
    st = lambda ds: set(np.asarray(ds.column("station_id")).astype(int).tolist())
    ts = lambda ds: pd.to_datetime(ds._ts_as_datetime64("timestamp_win"))
    assert st(blocks[0]) == {1001, 1002} and st(blocks[1]) == {1001}
    assert len(ds_val) == len(blocks[0]) + len(blocks[1])
    tr = pd.DataFrame({"ts": ts(ds_train), "st": np.asarray(ds_train.column("station_id")).astype(int)})
    assert tr[tr.st == 1002].ts.max() < pd.Timestamp("2025-06-03")                             # 块 0 前 purge
    assert tr[tr.st == 1001].ts.max() < pd.Timestamp("2025-06-23")                             # cal_start 06-25 − purge
    ca = pd.DataFrame({"ts": ts(ds_cal), "st": np.asarray(ds_cal.column("station_id")).astype(int)})
    assert ca[ca.st == 1002].ts.min() >= pd.Timestamp("2025-06-10") and ca[ca.st == 1001].ts.min() >= pd.Timestamp("2025-06-25")
    d = plan.to_dict()
    assert d["by_station"] is True and d["test_start"] == {"1001": "2025-06-28", "1002": "2025-06-13"}
    _, _, _, _, plan2 = make_splits(ds_all, {**split, "test_start": {1001: "2025-06-28"}})     # 少给 1002：视为没有测试集
    assert plan2.no_test == {"1002"} and set(plan2.plans) == {"1001", "1002"}
    last = pd.Timestamp(df.loc[df.station_id == 1002, "timestamp_win"].max()).normalize()
    assert plan2.to_dict()["test_start"]["1002"] == (last + pd.Timedelta(days=3)).strftime("%Y-%m-%d")


def test_build_plan_from_test_parquets(tmp_path):
    from forecast_backbones.m6.train import build_plan
    for sid, day in ((1001, "2025-06-28 08:00"), (1002, "2025-06-13")):
        pd.DataFrame({"timestamp_win": pd.date_range(day, periods=8, freq="15min"), "station_id": sid}
                     ).to_parquet(tmp_path / f"test_{sid}.parquet", index=False)
    split = {"train_start": None, "val_blocks": [["2025-06-05", "2025-06-07"]], "test_start": None, "test_end": None,
             "cal_days": 1, "purge_days": 2, "embargo_days": 7, "test_parquets": [str(tmp_path / "test_*.parquet")]}
    plan = build_plan(split)
    assert isinstance(plan, StationSplitPlan) and plan.to_dict()["test_start"] == {"1001": "2025-06-28", "1002": "2025-06-13"}
    with pytest.raises(ValueError, match="删掉其一"):
        build_plan({**split, "test_start": "2025-06-28"})
    assert isinstance(build_plan({**split, "test_parquets": None, "test_start": "2025-06-28"}), SplitPlan)


# ---------------- 框架契约 + 预切分目录 ----------------
def test_m6_framework_contract_hooks(tmp_path):
    from forecast_backbones.m6.transformer_trainer import M6_build_training_args
    from forecast_backbones.m6.config import load_config
    from pipelines.model_registry import import_model_class, build_evaluator
    from forecast_backbones.m6.transformer_trainer import evaluate_model
    assert import_model_class("m6") is TransformersAdaptedM6Model and build_evaluator("m6") is evaluate_model
    cfg = {"finetune": {"schedule": {"batch_size": 4, "epochs_max": 1, "lr": 1e-3, "precision": "fp32"}}}
    ta = M6_build_training_args(cfg, tmp_path / "run", max_steps=3)
    assert ta.per_device_train_batch_size == 4 and ta.max_steps == 3 and str(ta.output_dir).startswith(str(tmp_path / "run"))
    m = TransformersAdaptedM6Model({"timesfm3": {"eager_init": False}, "dataset": {"station_meta": str(write_station_meta(tmp_path))},
                                    "calibration": {"enabled": False}})
    assert m.config is m.cfg and m.config["run"]["model_name"] == "m6" and m.compute_metrics(None) == {}
    assert (Path(__file__).resolve().parents[1] / "config" / "configs.yaml").exists()      # train_script 的自动定位路径
    assert load_config(Path(__file__).resolve().parents[1] / "config" / "configs.yaml")["run"]["model_name"] == "m6"


def test_dataset_with_station_id_maps_station_names(tmp_path):
    from forecast_backbones.m6.train import _dataset_with_station_id
    from forecast_backbones.m6.features import StationMeta
    from forecast_backbones.m6.config import load_config
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    named = df.assign(station=df["station_id"].map(lambda s: f"S{s}")).drop(columns=["station_id"])
    p = tmp_path / "named.parquet"; named.to_parquet(p, index=False)
    meta = StationMeta.from_csv(write_station_meta(tmp_path))
    cfg = load_config({"dataset": {"station_meta": str(tmp_path / "station_meta.csv")}})
    ds = _dataset_with_station_id([str(p)], cfg, meta, tmp_path / "scratch")
    assert sorted(set(np.asarray(ds.column("station_id")).astype(int))) == [1001, 1002] and len(ds) == len(df)
    assert (tmp_path / "scratch" / "named_with_station_id.parquet").exists()
    ds2 = _dataset_with_station_id([str(p)], cfg, meta, tmp_path / "scratch")           # 幂等
    assert len(ds2) == len(ds)
    bad = named.assign(station="Unknown"); bp = tmp_path / "bad.parquet"; bad.to_parquet(bp, index=False)
    with pytest.raises(ValueError, match="没有对应的 station_id"):
        _dataset_with_station_id([str(bp)], cfg, meta, tmp_path / "scratch")



def test_dataset_with_station_id_handles_int_and_text_station_files(tmp_path):
    """两个站文件：一个站名存成整数 1001，一个存成文本 S1002；拼接前统一成文本，不再在写临时 parquet 时崩。"""
    from forecast_backbones.m6.train import _dataset_with_station_id
    from forecast_backbones.m6.features import StationMeta
    from forecast_backbones.m6.config import load_config
    from forecast_backbones.m6.tests.conftest import LAT, CAP
    meta_csv = tmp_path / "station_meta.csv"
    meta_csv.write_text("station_id,station,lat,lon,alt_m,cap_mw\n"
                        + "".join(f"{s},{'1001' if s == 1001 else 'S' + str(s)},{LAT[s][0]},{LAT[s][1]},{LAT[s][2]},{CAP[s]}\n"
                                  for s in (1001, 1002)), encoding="utf-8")
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    paths = []
    for sid, name in ((1001, 1001), (1002, "S1002")):
        part = df[df.station_id == sid].drop(columns=["station_id"]).assign(station=name)
        pth = tmp_path / f"{sid}.parquet"; part.to_parquet(pth, index=False); paths.append(str(pth))
    cfg = load_config({"dataset": {"station_meta": str(meta_csv)}})
    ds = _dataset_with_station_id(paths, cfg, StationMeta.from_csv(meta_csv), tmp_path / "scratch")
    assert sorted(set(np.asarray(ds.column("station_id")).astype(int))) == [1001, 1002] and len(ds) == len(df)


def test_dataset_with_text_station_id_column_maps_through_station_meta(tmp_path):
    """数据自带的 station_id 列是文本（站名，如 雅砻江）：按 station_meta 映射成整数编号，
    否则 collate 会丢掉文本列，所有行静默落成同一个站 0。文本存的纯数字编号直接转整数。"""
    from forecast_backbones.m6.train import _dataset_with_station_id
    from forecast_backbones.m6.features import StationMeta
    from forecast_backbones.m6.config import load_config
    from forecast_backbones.m6.tests.conftest import LAT, CAP
    meta_csv = tmp_path / "station_meta.csv"
    meta_csv.write_text("station_id,station,lat,lon,alt_m,cap_mw\n"
                        + "".join(f"{s},{'雅砻江' if s == 1001 else '1002'},{LAT[s][0]},{LAT[s][1]},{LAT[s][2]},{CAP[s]}\n"
                                  for s in (1001, 1002)), encoding="utf-8")
    meta = StationMeta.from_csv(meta_csv)
    cfg = load_config({"dataset": {"station_meta": str(meta_csv)}})
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    named = df.assign(station_id=df["station_id"].map({1001: "雅砻江", 1002: "1002"}))
    p = tmp_path / "named_id.parquet"; named.to_parquet(p, index=False)
    ds = _dataset_with_station_id([str(p)], cfg, meta, tmp_path / "scratch")
    got = pd.DataFrame({"sid": np.asarray(ds.column("station_id")).astype(int)})
    assert sorted(set(got["sid"])) == [1001, 1002] and len(ds) == len(df)
    digits = df.assign(station_id=df["station_id"].astype(str))                    # 文本存的整数编号
    q = tmp_path / "digit_id.parquet"; digits.to_parquet(q, index=False)
    ds2 = _dataset_with_station_id([str(q)], cfg, meta, tmp_path / "scratch")
    assert sorted(set(np.asarray(ds2.column("station_id")).astype(int))) == [1001, 1002]
    bad = df.assign(station_id=df["station_id"].map({1001: "雅砻江", 1002: "没这个站"}))
    r = tmp_path / "bad_id.parquet"; bad.to_parquet(r, index=False)
    with pytest.raises(ValueError, match="没这个站.*没有对应的 station_id|没有对应的 station_id.*没这个站"):
        _dataset_with_station_id([str(r)], cfg, meta, tmp_path / "scratch")

def test_train_m6_from_presplit_dir(tmp_path, tiny_evaluator, monkeypatch):
    """与 M1–M4 共用 pipelines.time_splits 落的目录：train/valid/cal.parquet + split_manifest.json。"""
    from pipelines.time_splits import SplitPlan, write_splits
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    splits = tmp_path / "splits"
    man = write_splits(df, SplitPlan.from_dict(cfg["finetune"]["split"]), splits, source=str(parquet))
    with pytest.raises(ValueError, match="二选一"):
        train_m6(cfg, str(parquet), splits_dir=str(splits))
    with pytest.raises(ValueError, match="shift_days 无效"):
        train_m6(cfg, splits_dir=str(splits), shift_days=1)
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        res = train_m6(cfg, splits_dir=str(splits), max_steps=2)
    for f in ("lora_weights.pt", "station_routing.json", "calibration.parquet", "train_log.json", "split_manifest.json"):
        assert (out / f).exists()
    info = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert info["source"] == "pipelines.time_splits" and info["plan"] == man["plan"] and info["splits_dir"] == str(splits.resolve())
    assert info["counts"]["train"] == man["counts"]["train"] and info["counts"]["val_blocks"] == [man["counts"]["val"]]
    assert res["split"] == info and len(res["history"]) >= 2
    (splits / "cal.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="cal_days"):
        train_m6(cfg, splits_dir=str(splits), max_steps=1)



def test_train_m6_without_conformal_uses_m1_m4_split(tmp_path, tiny_evaluator, monkeypatch):
    """calibration.enabled=false：不做 conformal，不要 cal.parquet，直接用 M1–M4 的切分目录（cal_days 0）；
    早停与逐站路由按未校准分位上的点预测打分，不写 calibration.parquet。"""
    from pipelines.time_splits import SplitPlan, write_splits
    _patch_tiny_backbone(monkeypatch, tiny_evaluator)
    df, parquet, out, cfg = _inputs(tmp_path)
    split = {**cfg["finetune"]["split"], "cal_days": 0}
    splits = tmp_path / "splits"
    write_splits(df, SplitPlan.from_dict(split), splits, source=str(parquet))
    assert not (splits / "cal.parquet").exists()
    with pytest.raises(FileNotFoundError, match="calibration.enabled: false"):
        train_m6(cfg, splits_dir=str(splits), max_steps=1)                  # 默认做校准：缺 cal.parquet 报错并指出开关
    nocal = {**cfg, "calibration": {"enabled": False}}
    with pytest.warns(UserWarning, match="LoRA 权重|站级路由"):
        res = train_m6(nocal, splits_dir=str(splits), max_steps=2)
    for f in ("lora_weights.pt", "station_routing.json", "train_log.json", "split_manifest.json"):
        assert (out / f).exists()
    assert not (out / "calibration.parquet").exists()
    assert list(res["ladder"]["variant"]) == ["zeroshot_uncalibrated", "lora_uncalibrated", "routed_uncalibrated"]
    assert res["split"]["counts"]["cal"] == 0 and len(res["history"]) >= 2

    ds = UnifiedDataset(str(parquet), timestamp_cols=["timestamp_win"], entity_cols=[], label_cols=["observe_power_future"],
                        timestamp_output="epoch_s", epoch_unit="s", for_hf_trainer=True)
    with pytest.raises(ValueError, match="cal_days 须 ≥ 1"):
        make_splits(ds, split)
    tr, cal, val_all, blocks, _ = make_splits(ds, split, need_cal=False)      # 原始行表模式同样可以不要校准段
    assert cal is None and len(tr) > 0 and len(blocks) == 1


def test_lora_no_conformal_config_differs_from_default_only_where_intended():
    from forecast_backbones.m6.config import load_config
    base = load_config("forecast_backbones/m6/config/configs.yaml")
    c = load_config("forecast_backbones/m6/config/lora_no_conformal.yaml")
    assert c["finetune"]["enabled"] is True and c["calibration"]["enabled"] is False
    assert c["decision"]["point"]["default"] == "median" and c["decision"]["point"]["by_step"] == []
    assert c["features"]["past_future"] == [["general_ghi", "general_ghi_predict"], ["general_temp", "general_temp_predict"]]
    for k in ("dataset", "timesfm3"):
        assert c[k] == base[k], k
    assert {**c["eval"], "report_dir": None} == {**base["eval"], "report_dir": None}
    assert c["eval"]["report_dir"] == "runs/m6_lora_no_conformal/reports"
    assert {k: v for k, v in c["finetune"].items() if k not in ("enabled", "split", "lora_path", "routing_path")} == \
           {k: v for k, v in base["finetune"].items() if k not in ("enabled", "split", "lora_path", "routing_path")}

def test_build_plan_rejects_unresolved_auto_blocks():
    from forecast_backbones.m6.train import build_plan
    split = {"train_start": None, "val_blocks": "auto", "n_val_blocks": 3, "val_block_days": 14, "test_start": "2025-10-01",
             "test_end": None, "cal_days": 1, "purge_days": 2, "embargo_days": 7}
    with pytest.raises(ValueError, match="split_resolved.yaml"):
        build_plan(split)
