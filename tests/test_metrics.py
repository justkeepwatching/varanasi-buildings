import math

import numpy as np

from src.metrics import (confusion_counts, f1_iou_identity_holds, global_scores, per_tile_mean_scores,
                         scores_from_counts)


def test_known_example():
    ref = np.array([[1, 1, 0, 0]])
    pred = np.array([[1, 0, 1, 0]])
    c = confusion_counts(pred, ref)
    assert c == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    s = scores_from_counts(c)
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    assert math.isclose(s["iou"], 1 / 3) and math.isclose(s["f1"], 0.5)
    assert math.isclose(s["f1"], 2 * s["iou"] / (1 + s["iou"]))


def test_nodata_excluded():
    ref = np.array([1, 0, 1])
    pred = np.array([1, 1, 0])
    valid = np.array([True, False, False])
    assert confusion_counts(pred, ref, valid) == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}


def test_identity_holds_on_random_maps():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p, r = rng.random((2, 32, 32)) > 0.6
        s = scores_from_counts(confusion_counts(p, r))
        assert f1_iou_identity_holds(s)


def test_global_vs_per_tile_differ():
    # tile A: perfect on 1 pixel; tile B: 1/4 IoU on many pixels
    ref = [np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1])]
    pred = [np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])]
    valid = [np.ones(4, bool)] * 2
    g = global_scores(pred, ref, valid)
    t = per_tile_mean_scores(pred, ref, valid)
    assert math.isclose(g["iou"], 2 / 5)          # (1+1) / (1+1 + 3)
    assert math.isclose(t["iou"], (1 + 0.25) / 2)  # mean of per-tile IoUs


def test_empty_tile_is_undefined_not_zero():
    s = scores_from_counts(confusion_counts(np.zeros(4), np.zeros(4)))
    assert math.isnan(s["iou"]) and s["accuracy"] == 1.0
    t = per_tile_mean_scores([np.zeros(4)], [np.zeros(4)], [np.ones(4, bool)])
    assert t["iou_n_tiles_defined"] == 0


def test_all_background_baseline_has_high_accuracy_zero_recall():
    ref = np.zeros(100)
    ref[:5] = 1
    s = scores_from_counts(confusion_counts(np.zeros(100), ref))
    assert s["accuracy"] == 0.95 and s["recall"] == 0.0 and s["iou"] == 0.0
