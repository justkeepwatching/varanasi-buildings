import math

import numpy as np

from src.change_detection import (GAIN, LOSS, NOT_ANALYSED, PERSISTENT, STABLE_NON, change_classes,
                                  change_statistics, confident_change_classes, pixel_area_km2)


def test_change_class_logic():
    e = np.array([0, 0, 1, 1, 1])
    l = np.array([0, 1, 0, 1, 1])
    v = np.array([True, True, True, True, False])
    assert list(change_classes(e, l, v)) == [STABLE_NON, GAIN, LOSS, PERSISTENT, NOT_ANALYSED]


def test_pixel_area():
    assert pixel_area_km2(10_000, 10) == 1.0  # 10,000 px * 100 m^2 = 1 km^2


def test_statistics_consistency():
    rng = np.random.default_rng(0)
    e, l = rng.random((2, 100, 100)) > 0.5
    v = np.ones((100, 100), bool)
    v[:10] = False
    s = change_statistics(change_classes(e, l, v), 10)
    assert math.isclose(s["valid_analysed_area_km2"], 0.9)
    assert math.isclose(s["later_built_up_km2"] - s["earlier_built_up_km2"], s["net_change_km2"])
    assert math.isclose(s["earlier_built_up_km2"], s["persistent_km2"] + s["gross_loss_km2"])
    assert math.isclose(s["later_built_up_km2"], s["persistent_km2"] + s["gross_gain_km2"])
    # gross gain can never exceed the area that was not built up before
    assert s["gross_gain_km2"] <= s["valid_analysed_area_km2"] - s["earlier_built_up_km2"] + 1e-12


def test_high_confidence_suppresses_flicker():
    pe = np.array([0.45, 0.10, 0.90, 0.55])
    pl = np.array([0.55, 0.90, 0.10, 0.45])
    v = np.ones(4, bool)
    c = confident_change_classes(pe, pl, v, threshold=0.5, margin=0.2)
    assert list(c) == [STABLE_NON, GAIN, LOSS, PERSISTENT]
