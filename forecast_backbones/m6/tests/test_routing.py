import json
import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("timesfm3")
from forecast_backbones.m6.routing import decide_routing, save_routing, load_routing
from forecast_backbones.m6.lora import LoRALinear, set_lora_enabled
from .conftest import make_synthetic_frame
from .test_model_route_a import _batch
from .test_model_route_b import _model_b


def test_decide_routing_rules():
    t = pd.DataFrame({"station_id": [1, 2, 3, 4], "acc_zeroshot": [80.0, 80.0, 80.0, np.nan], "acc_lora": [81.0, 80.0, 79.0, 90.0]})
    assert decide_routing(t) == {1: "lora", 2: "zeroshot", 3: "zeroshot", 4: "zeroshot"}
    assert decide_routing(t, min_gain_pct=1.5) == {1: "zeroshot", 2: "zeroshot", 3: "zeroshot", 4: "zeroshot"}


def test_routing_json_round_trip_and_validation(tmp_path):
    p = tmp_path / "station_routing.json"
    save_routing(p, {1001: "lora", 1002: "zeroshot"})
    assert json.loads(p.read_text())["1001"] == "lora"
    assert load_routing(p) == {1001: "lora", 1002: "zeroshot"}
    p.write_text(json.dumps({"1001": "maybe"}))
    with pytest.raises(ValueError, match="maybe"):
        load_routing(p)


def _lora_model_with_nonzero_B(tmp_path, evaluator, routing_path=None):
    with pytest.warns(UserWarning):
        m = _model_b(tmp_path, evaluator)
    with torch.no_grad():
        for x in m.backbone.model.modules():
            if isinstance(x, LoRALinear):
                x.lora_B.normal_(std=0.5)
    if routing_path is not None:
        m.station_routing = load_routing(routing_path)
    return m


def test_mixed_batch_routes_per_row(tmp_path, tiny_evaluator):
    df = make_synthetic_frame(n_days=1, issue_every_min=360)          # 两站交替
    rows = [int(df.index[df["station_id"] == s][k]) for k in (0, 1) for s in sorted(df["station_id"].unique())]
    bd, _ = _batch(df, rows)
    sids = bd["station_id"].tolist(); assert len(set(sids)) == 2
    m = _lora_model_with_nonzero_B(tmp_path, tiny_evaluator)
    with torch.no_grad():
        q_lora = m(bd).aux["quantiles"].numpy()
        set_lora_enabled(m.backbone.model, False); q_zs = m(bd).aux["quantiles"].numpy(); set_lora_enabled(m.backbone.model, True)
    assert not np.allclose(q_lora, q_zs)
    a, b = sorted(set(sids))
    p = tmp_path / "station_routing.json"; save_routing(p, {a: "lora", b: "zeroshot"})
    m.station_routing = load_routing(p)
    with torch.no_grad():
        q = m(bd).aux["quantiles"].numpy()
    for i, s in enumerate(sids):
        assert np.allclose(q[i], (q_lora if s == a else q_zs)[i], atol=1e-5)
    assert all(x.enabled for x in m.backbone.model.modules() if isinstance(x, LoRALinear))   # 调用后恢复


def test_uniform_batch_single_pass_and_unknown_station_defaults_to_zeroshot(tmp_path, tiny_evaluator, monkeypatch):
    df = make_synthetic_frame(n_days=1, issue_every_min=360)
    bd, _ = _batch(df, [0, 2])                                          # 同一站两行
    sid = int(bd["station_id"][0]); assert int(bd["station_id"][1]) == sid
    m = _lora_model_with_nonzero_B(tmp_path, tiny_evaluator)
    calls = []
    orig = m.backbone.predict_quantiles
    monkeypatch.setattr(m.backbone, "predict_quantiles", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    m.station_routing = {sid: "zeroshot"}
    with torch.no_grad():
        m(bd)
    assert len(calls) == 1                                              # 全零样本：单 pass
    m.station_routing = {9999: "zeroshot"}                              # 表里没有的站（R6）→ 默认 zeroshot
    calls.clear()
    with torch.no_grad():
        q_unknown = m(bd).aux["quantiles"].numpy()
    assert len(calls) == 1
    m.station_routing = {sid: "zeroshot"}                               # 同一站显式 zeroshot 作对照
    with torch.no_grad():
        q_explicit_zeroshot = m(bd).aux["quantiles"].numpy()
    assert np.allclose(q_unknown, q_explicit_zeroshot)                  # 未知站路由结果 = 零样本，不是 LoRA（B 非零，二者本应有别）


def test_routing_loaded_at_construction_when_file_exists(tmp_path, tiny_evaluator):
    from forecast_backbones.m6.transformer_models import TransformersAdaptedM6Model
    from .conftest import write_station_meta
    p = tmp_path / "station_routing.json"; save_routing(p, {1001: "zeroshot", 1002: "lora"})
    cfg = {"timesfm3": {"eager_init": False}, "dataset": {"station_meta": str(write_station_meta(tmp_path))},
           "calibration": {"enabled": False},
           "finetune": {"enabled": True, "inference_path": "torch", "lora_path": str(tmp_path / "none.pt"), "routing_path": str(p)}}
    m = TransformersAdaptedM6Model(cfg)
    assert m.station_routing == {1001: "zeroshot", 1002: "lora"}
