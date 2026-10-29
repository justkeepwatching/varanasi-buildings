"""Evaluate the trained U-Net and baselines on the held-out TEST tiles.

Protocol (no test-set tuning):
1. Probability threshold is chosen on the VALIDATION tiles (highest
   validation IoU over config.evaluation.threshold_search).
2. That single threshold is applied once to the test tiles.
3. Baselines are evaluated on the same test pixels:
   * all-background: predicts "no building" everywhere (shows how
     uninformative accuracy is under class imbalance);
   * random forest: per-pixel classifier on the 6 bands + NDVI/NDBI/NDWI
     indices, trained on a pixel sample from the TRAINING tiles only.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from .config import PATHS
from .dataset import TileDataset
from .metrics import confusion_counts, global_scores, per_tile_mean_scores, scores_from_counts
from .model import load_checkpoint
from .train import get_device, load_tiles, predict_probs, subset

log = logging.getLogger(__name__)


def pixel_features(images: np.ndarray) -> np.ndarray:
    """Per-pixel features for the non-neural baseline: bands + 3 spectral indices.

    Band order: B02 B03 B04 B08 B11 B12.
    NDVI (vegetation) = (B08-B04)/(B08+B04); NDBI (built-up) = (B11-B08)/(B11+B08);
    NDWI (water) = (B03-B08)/(B03+B08).
    """
    b2, b3, b4, b8, b11, b12 = (images[:, i] for i in range(6))
    eps = 1e-6
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndbi = (b11 - b8) / (b11 + b8 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    feats = np.stack([b2, b3, b4, b8, b11, b12, ndvi, ndbi, ndwi], axis=-1)  # (n, h, w, 9)
    return feats


def random_forest_baseline(tr, te, seed: int, n_samples: int = 200_000):
    (tr_x, tr_y, tr_v), (te_x, te_y, te_v) = tr, te
    f_tr = pixel_features(tr_x)[tr_v]
    y_tr = tr_y[tr_v]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_tr), size=min(n_samples, len(y_tr)), replace=False)
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1,
                                class_weight="balanced", random_state=seed)
    rf.fit(f_tr[idx], y_tr[idx])
    f_te = pixel_features(te_x)
    prob = rf.predict_proba(f_te.reshape(-1, f_te.shape[-1]))[:, 1].reshape(te_y.shape)
    return prob, rf


def choose_threshold(val_probs, val_y, val_v, candidates) -> tuple[float, pd.DataFrame]:
    rows = []
    for t in candidates:
        s = global_scores(val_probs >= t, val_y, val_v)
        rows.append({"threshold": t, "val_iou": s["iou"], "val_f1": s["f1"],
                     "val_precision": s["precision"], "val_recall": s["recall"]})
    df = pd.DataFrame(rows)
    best = float(df.loc[df.val_iou.idxmax(), "threshold"])
    return best, df


def evaluate(cfg: dict) -> dict:
    device = get_device()
    model, ckpt = load_checkpoint(PATHS.models / "unet_best.pt", device)
    stats = ckpt["band_stats"]
    images, masks, valids, manifest = load_tiles()
    tr, _ = subset((images, masks, valids), manifest, "train")
    va, _ = subset((images, masks, valids), manifest, "val")
    te, te_m = subset((images, masks, valids), manifest, "test")

    def probs_for(arrs):
        return predict_probs(model, DataLoader(TileDataset(*arrs, stats), batch_size=64), device)

    va_p = probs_for(va)
    thr, sweep = choose_threshold(va_p, va[1], va[2], cfg["evaluation"]["threshold_search"])
    sweep.to_csv(PATHS.metrics / "threshold_sweep_validation.csv", index=False)

    te_p = probs_for(te)
    te_pred = te_p >= thr
    unet_global = global_scores(te_pred, te[1], te[2])
    unet_tile = per_tile_mean_scores(te_pred, te[1], te[2])
    unet_at_05 = global_scores(te_p >= 0.5, te[1], te[2])

    bg_global = global_scores(np.zeros_like(te[1]), te[1], te[2])
    rf_p, _ = random_forest_baseline(tr, te, cfg["project"]["seed"])
    rf_global = global_scores(rf_p >= 0.5, te[1], te[2])

    np.savez_compressed(PATHS.processed / "test_predictions.npz", probs=te_p.astype(np.float32),
                        rf_probs=rf_p.astype(np.float32), tile_ids=te_m.tile_id.values)

    # per-tile CSV for inspection
    per_tile = []
    for i, tid in enumerate(te_m.tile_id):
        s =scores_from_counts(confusion_counts(te_pred[i], te[1][i], te[2][i]))
        per_tile.append({"tile_id": tid, **{k: s[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou")}})
    pd.DataFrame(per_tile).to_csv(PATHS.metrics / "test_per_tile_metrics.csv", index=False)

    metrics = {
        "status": "MEASURED on held-out spatial test blocks",
        "n_test_tiles": len(te_m),
        "n_test_blocks": int(te_m.block_id.nunique()),
        "test_reference_building_pixel_share": float(te[1][te[2]].mean()),
        "threshold": {"value": thr, "selected_on": "validation split (max global IoU)"},
        "unet_global": unet_global,
        "unet_per_tile_mean": unet_tile,
        "unet_global_at_threshold_0.5": unet_at_05,
        "baseline_all_background": bg_global,
        "baseline_random_forest_pixelwise": rf_global,
        "checkpoint_epoch": ckpt["epoch"],
        "notes": [
            "Global = TP/FP/FN/TN summed over all valid test pixels, then ratios.",
            "Per-tile mean = average of per-tile scores; undefined tiles (0/0) excluded.",
            "Foreground metrics: precision/recall/f1/iou. Background-inclusive: accuracy, iou_background, mean_iou_2class.",
            "Reference = rasterized Open Buildings V3 (an imperfect, differently-dated label).",
        ],
    }
    (PATHS.metrics / "metrics.json").write_text(json.dumps(metrics, indent=2))

    rows = []
    for name, s in (("U-Net (val-selected threshold)", unet_global), ("U-Net (threshold 0.5)", unet_at_05),
                    ("Random forest (pixel)", rf_global), ("All-background", bg_global)):
        rows.append({"model": name, **{k: s[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou", "accuracy")}})
    pd.DataFrame(rows).to_csv(PATHS.metrics / "test_metrics_summary.csv", index=False)
    log.info("Test IoU %.4f F1 %.4f (threshold %.2f)", unet_global["iou"], unet_global["f1"], thr)
    return metrics
