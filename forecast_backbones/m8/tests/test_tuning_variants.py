"""k 系列调参变体（2026-10-14）：每个 yaml = j1_final2 + 恰好一处改动；决策层扫描 yaml 只改 clear_sky_cap_ratio。"""
import pytest
from forecast_backbones.m8.config import load_config
from forecast_backbones.m8.evaluate import decision_override
from forecast_backbones.m8.tests.conftest import ROOT

V = ROOT / "forecast_backbones/m8/config/variants"
D = ROOT / "forecast_backbones/m8/config/decision"
EXPECTED = {
    "k1_hint01":  {"train.hint_dropout": 0.1},
    "k2_hint00":  {"train.hint_dropout": 0.0},
    "k3_resid":   {"features.hint_residual": True},
    "k4_lr2e4":   {"train.schedule.lr": 2e-4},
    "k5_lr5e4":   {"train.schedule.lr": 5e-4},
    "k6_regular": {"train.schedule.weight_decay": 0.1, "model.dropout": 0.2},
}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v if not isinstance(v, list) else repr(v)
    return out


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_k_variant_is_j1_plus_exactly_one_change(name):
    j1 = _flat(load_config(str(V / "j1_final2.yaml")))
    k = _flat(load_config(str(V / f"{name}.yaml")))
    derived = ("eval.report_dir", "train.weights_path")                                     # 由 ${run.out_dir} 插值得来
    diff = {key: k.get(key) for key in set(j1) | set(k)
            if not key.startswith("run.") and key not in derived and j1.get(key) != k.get(key)}
    assert diff == EXPECTED[name], f"{name} 与 j1 的差异应恰好是 {EXPECTED[name]}，实际 {diff}"


@pytest.mark.parametrize("ratio", [1.0, 1.15, 1.3])
def test_decision_sweep_only_moves_cap_ratio(ratio):
    base = load_config(str(V / "j1_final2.yaml"))["decision"]
    dec = decision_override(base, decision_yaml=str(D / f"cap_ratio_{int(round(ratio * 100))}.yaml"))
    assert dec["physics"]["clear_sky_cap_ratio"] == ratio
    assert dec["point"] == base["point"] and dec["physics"]["night_zero"] == base["physics"]["night_zero"]
