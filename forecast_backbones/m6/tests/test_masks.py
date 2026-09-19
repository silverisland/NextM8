import numpy as np
from forecast_backbones.m6.masks import (night_from_elevation, night_from_clock,
                                         curtail_from_rule, missing_mask, loss_mask)

def test_night_from_elevation():
    assert night_from_elevation(np.array([-1.0, 0.0, 0.1])).tolist() == [True, True, False]

def test_night_from_clock():
    ts = np.array([np.datetime64("2025-06-15T06:00"), np.datetime64("2025-06-15T06:15"),
                   np.datetime64("2025-06-15T20:00"), np.datetime64("2025-06-15T20:15")])
    assert night_from_clock(ts).tolist() == [True, False, False, True]

def test_curtail_rule_requires_run_length():
    p_clear = np.full((1, 8), 100.0); k = np.full((1, 8), 0.9)
    y = np.array([[80, 1, 1, 1, 1, 80, 1, 80]], float)      # 一段 4 步平台 + 两段 1 步
    m = curtail_from_rule(y, p_clear, k, plateau_ratio=0.05, k_min=0.7, min_run=4)
    assert m.tolist() == [[False, True, True, True, True, False, False, False]]

def test_curtail_rule_ignores_cloudy_or_night():
    p_clear = np.array([[100.0] * 4 + [0.0] * 4]); y = np.ones((1, 8))
    k_cloudy = np.full((1, 8), 0.3)
    assert not curtail_from_rule(y, p_clear, k_cloudy).any()
    k_sunny = np.full((1, 8), 0.9)
    assert not curtail_from_rule(y, p_clear, k_sunny)[0, 4:].any()

def test_loss_mask_union():
    night = np.array([[True, False, False, False]])
    curt = np.array([[False, True, False, False]])
    miss = np.array([[False, False, True, False]])
    assert loss_mask(night, curt, miss).tolist() == [[0.0, 0.0, 0.0, 1.0]]
    assert loss_mask(night).tolist() == [[0.0, 1.0, 1.0, 1.0]]
    assert missing_mask(np.array([[1.0, np.nan]])).tolist() == [[False, True]]
