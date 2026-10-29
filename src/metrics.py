"""Pixel-level segmentation metrics computed from confusion-matrix counts.

Definitions (foreground = building):
    precision = TP / (TP + FP)        how many predicted building pixels are right
    recall    = TP / (TP + FN)        how many reference building pixels we found
    F1        = 2TP / (2TP + FP + FN) harmonic mean of precision and recall
    IoU       = TP / (TP + FP + FN)   overlap / union of predicted and reference
For the SAME counts, F1 = 2 IoU / (1 + IoU) holds exactly; we check this.

Aggregation:
* "global" metrics sum TP/FP/FN/TN over all pixels of all tiles first and
  then compute the ratios (every pixel weighs the same) - our headline numbers.
* "per-tile mean" metrics compute a score per tile and average them; tiles
  whose score is undefined (e.g. no reference and no predicted buildings,
  0/0) are excluded from that average and counted separately.
Pixels flagged as nodata (valid == False) are excluded from every count.
"""
from __future__ import annotations

import math

import numpy as np


def confusion_counts(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> dict:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    valid = np.ones_like(ref, bool) if valid is None else valid.astype(bool)
    return {
        "tp": int(np.sum(pred & ref & valid)),
        "fp": int(np.sum(pred & ~ref & valid)),
        "fn": int(np.sum(~pred & ref & valid)),
        "tn": int(np.sum(~pred & ~ref & valid)),
    }


def _div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def scores_from_counts(c: dict) -> dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    iou = _div(tp, tp + fp + fn)
    iou_bg = _div(tn, tn + fp + fn)  # IoU of the background class
    return {
        **c,
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "iou": iou,
        # background-inclusive metrics (inflated by the many easy background pixels)
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "iou_background": iou_bg,
        "mean_iou_2class": (iou + iou_bg) / 2 if not (math.isnan(iou) or math.isnan(iou_bg)) else float("nan"),
    }


def f1_iou_identity_holds(scores: dict, tol: float = 1e-9) -> bool:
    f1, iou = scores["f1"], scores["iou"]
    if math.isnan(f1) or math.isnan(iou):
        return True
    return abs(f1 - 2 * iou / (1 + iou)) < tol


def global_scores(preds, refs, valids) -> dict:
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for p, r, v in zip(preds, refs, valids):
        for k, n in confusion_counts(p, r, v).items():
            total[k] += n
    s = scores_from_counts(total)
    assert f1_iou_identity_holds(s), "F1/IoU identity violated - counts are inconsistent"
    return s


def per_tile_mean_scores(preds, refs, valids) -> dict:
    per = [scores_from_counts(confusion_counts(p, r, v)) for p, r, v in zip(preds, refs, valids)]
    out = {}
    for k in ("precision", "recall", "f1", "iou"):
        vals = [s[k] for s in per if not math.isnan(s[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_n_tiles_defined"] = len(vals)
    out["n_tiles"] = len(per)
    return out
