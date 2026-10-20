"""M7 决策层（m7/decision.py）+ adapter 的 decision 段。不依赖 timestral（替身 backbone 见 test_transformer_trainer）。"""
import numpy as np
import pytest
import torch

from forecast_backbones.m7 import decision
from forecast_backbones.m7.transformer_models import TransformersAdaptedM7Model
from forecast_backbones.m7.tests.test_transformer_trainer import LEVELS21, L, H, _FakeForecaster, _frame

LEVELS9 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _rand_q(levels, B=7, T=5, cap=100.0, seed=0):
    rng = np.random.default_rng(seed)
    q = np.sort(rng.uniform(-0.1 * cap, 1.2 * cap, (B, T, len(levels))), -1)   # 带越界值，测钳位
    q[0, 0] = q[0, 0][::-1]                                                     # 一行交叉，测排序
    return q


@pytest.mark.parametrize("tail", decision.TAILS)
@pytest.mark.parametrize("method", ["median", "p_star", "p_star_l1"])
def test_matches_m5_on_9_levels(method, tail):
    """9 档 / 200 点 / 同 tail 下与 M5 决策层（sort → clip [0,cap] → 点）数值一致：M7 是重写，不是引用。"""
    from forecast_backbones.m5 import decision as m5d
    q, cap = _rand_q(LEVELS9), 100.0
    q5 = m5d.physics_clip(m5d.enforce_monotone(q), np.zeros(q.shape[:2], bool), np.full(q.shape[:2], np.nan), cap,
                          clear_sky_cap_ratio=None)
    ref = m5d.select_point(q5, LEVELS9, np.full(q.shape[0], cap), method, tail=tail)
    got = decision.point(q, LEVELS9, cap, method, n_grid=200, tail=tail)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


def test_grid_density_barely_matters():
    """同一组档位插值：200 → 300 点只改离散误差（相对 cap < 1e-3），1000 点作参照。"""
    q, cap = _rand_q(LEVELS21, B=50, T=20), 100.0
    ref = decision.point(q, LEVELS21, cap, "p_star", n_grid=1000)
    for n in (200, 300):
        assert np.max(np.abs(decision.point(q, LEVELS21, cap, "p_star", n_grid=n) - ref)) < 1e-3 * cap


def test_points_basic_properties():
    cap = 100.0
    q = np.tile(np.linspace(10, 90, len(LEVELS21)), (2, 3, 1))
    med = decision.point(q, LEVELS21, cap, "median")
    np.testing.assert_allclose(med, np.interp(0.5, LEVELS21, q[0, 0]))
    ps = decision.point(q, LEVELS21, cap, "p_star")
    mean = decision.point(q, LEVELS21, cap, "mean")
    assert np.all(ps < mean)                          # p* 的 1/y² 权重压在低端
    const = np.full((1, 1, len(LEVELS21)), 37.0)
    for m in ("median", "mean", "p_star", "p_star_l1", "quantile:0.9"):
        np.testing.assert_allclose(decision.point(const, LEVELS21, cap, m), 37.0)
    assert np.all(decision.point(np.full((1, 1, 21), 500.0), LEVELS21, cap, "mean") == cap)   # 钳到 cap
    assert np.all(decision.point(np.full((1, 1, 21), -5.0), LEVELS21, cap, "p_star") == 0.0)  # 截负值


def test_per_row_cap_broadcast():
    q = np.full((2, 4, 21), 80.0)
    out = decision.point(q, LEVELS21, np.array([[50.0], [100.0]]), "median")
    np.testing.assert_allclose(out, [[50.0] * 4, [80.0] * 4])


def test_errors():
    q = np.zeros((1, 1, 21))
    with pytest.raises(ValueError):
        decision.point(q, LEVELS21, 100.0, "bogus")
    with pytest.raises(ValueError):
        decision.point(q, LEVELS21, None, "p_star")
    with pytest.raises(ValueError):
        decision.point(q, LEVELS21, 100.0, "p_star", tail="cubic")
    with pytest.raises(ValueError):
        decision.point(q, LEVELS9, 100.0, "median")


# ------------------------------------------------------------------ adapter

def _model(dec=None, **t):
    cfg = {"timestral": {"eager_init": False, "per_core_batch_size": 4, **t}}
    if dec is not None:
        cfg["decision"] = dec
    m = TransformersAdaptedM7Model(cfg)
    m._set_backbone(_FakeForecaster(), "cpu")
    return m


def _batch(stations=(1001, 1002), names=False):
    df = _frame(n=3, stations=stations)
    bd = {c: torch.as_tensor(np.stack(df[c].to_numpy())) for c in
          ("observe_power", "general_ghi", "general_ghi_predict", "general_temp", "general_temp_predict")}
    if names:
        from forecast_backbones.m5.solar import StationMeta
        meta = StationMeta.from_csv("forecast_backbones/m6/station_meta.csv")
        bd["station"] = [meta.get(int(s))["station"] for s in df["station_id"]]
    else:
        bd["station_id"] = torch.as_tensor(df["station_id"].to_numpy())
    return bd, df


def _expected(m, df, method, levels="model", **kw):
    full = _FakeForecaster().forecast(np.stack(df["observe_power"].to_numpy())[:, None, -L:], H).quantiles[:, 0]
    full = full.clamp_min(0.0).numpy().astype(np.float64)
    caps = {1001: 100.0, 1002: 500.0}
    cap = np.array([caps[s] for s in df["station_id"]])[:, None]
    q, lv = (full, LEVELS21) if levels == "model" else (full[:, :, [LEVELS21.index(x) for x in LEVELS9]], LEVELS9)
    return decision.point(q, lv, cap, method, **kw)


def test_decision_off_is_old_behaviour():
    bd, _ = _batch()
    old = _model().forward(bd)
    new = _model({"point": None}).forward(bd)
    assert torch.equal(old.logits, new.logits)
    assert torch.equal(old.aux["quantiles"], new.aux["quantiles"])


@pytest.mark.parametrize("dec,kw", [
    ({"point": "median"}, {}),
    ({"point": "mean"}, {"n_grid": 200}),
    ({"point": "p_star", "levels": "output"}, {"n_grid": 200}),
    ({"point": "p_star"}, {"n_grid": 200}),
    ({"point": "p_star", "grid": 300}, {"n_grid": 300}),
    ({"point": "p_star_l1", "tail": "linear"}, {"tail": "linear"}),
])
def test_decision_on_matches_decision_module(dec, kw):
    bd, df = _batch()
    m = _model(dec)
    out = m.forward(bd)
    exp = _expected(m, df, dec["point"], dec.get("levels", "model"), **kw)
    np.testing.assert_allclose(out.logits.numpy(), exp, rtol=1e-5, atol=1e-4)
    assert torch.equal(out.logits, out.predictions)
    assert out.aux["quantiles"].shape[-1] == 9                         # aux 档位不变
    assert float(out.logits[: len(df) // 2].max()) <= 100.0 + 1e-4     # 1001 站 cap=100 生效


def test_station_names_resolve_same_as_ids():
    bd_id, _ = _batch()
    bd_nm, _ = _batch(names=True)
    m = _model({"point": "p_star"})
    assert torch.equal(m.forward(bd_id).logits, m.forward(bd_nm).logits)


def test_unknown_station_raises():
    bd, _ = _batch(stations=(999999,))
    with pytest.raises(ValueError):
        _model({"point": "p_star"}).forward(bd)


@pytest.mark.parametrize("dec,t", [
    ({"point": "bogus"}, {}),
    ({"point": "p_star", "levels": "all"}, {}),
    ({"point": "p_star", "tail": "cubic"}, {}),
    ({"point": "p_star", "grid": 1}, {}),
    ({"point": "p_star"}, {"predict_quantile": 0.3}),
])
def test_bad_decision_config_raises_at_init(dec, t):
    with pytest.raises(ValueError):
        TransformersAdaptedM7Model({"timestral": {"eager_init": False, **t}, "decision": dec})


def test_shipped_decision_yamls_build():
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "config"
    base = yaml.safe_load(open(root / "configs.yaml", encoding="utf-8"))
    assert base["decision"]["point"] is None
    names = sorted(p.stem for p in (root / "decision").glob("*.yaml"))
    assert names == ["a1_median", "a2_mean", "a3_pstar_9", "a4_pstar_21", "a5_pstar_21_g300"]
    for n in names:
        dec = yaml.safe_load(open(root / "decision" / f"{n}.yaml", encoding="utf-8"))["decision"]
        m = TransformersAdaptedM7Model({"timestral": {**base["timestral"], "eager_init": False},
                                        "dataset": base["dataset"], "decision": {**base["decision"], **dec}})
        assert m.decision_point == dec["point"]
