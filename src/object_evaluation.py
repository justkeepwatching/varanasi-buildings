"""Object-level precision and recall with an explicit, documented matching rule.

Matching rule (region level, the primary object metric):
* Candidate pairs are predicted / reference polygons that intersect.
* Pair score = polygon IoU = area(P ∩ R) / area(P ∪ R).
* Greedy ONE-TO-ONE matching: pairs are sorted by IoU (highest first) and
  accepted only if IoU >= threshold and neither polygon is already matched.
  So no polygon is ever counted twice.
* TP = matched pairs; FP = unmatched predictions; FN = unmatched references.
  precision = TP / (TP + FP), recall = TP / (TP + FN).
* One-to-many situations are reported, not hidden: a "merge" is a predicted
  polygon that intersects 2+ reference polygons, a "split" is a reference
  polygon intersecting 2+ predictions. At most one of those pairs can match.

Footprint-level detection rates (secondary, many-to-one allowed):
* building detection rate = share of Open Buildings footprints with
  >= `footprint_overlap` of their area inside predicted regions;
* region support rate = share of predicted regions whose area is covered by
  footprints for >= `region_cover` (same 0.25 rule used for rasterization).
These are NOT one-to-one precision/recall and are labelled differently.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd


def candidate_pairs(pred: gpd.GeoDataFrame, ref: gpd.GeoDataFrame) -> pd.DataFrame:
    if len(pred) == 0 or len(ref) == 0:
        return pd.DataFrame(columns=["pred_idx", "ref_idx", "iou"])
    p = pred.geometry.reset_index(drop=True)
    r = ref.geometry.reset_index(drop=True)
    pi, ri = r.sindex.query(p, predicate="intersects")
    inter = p.iloc[pi].intersection(r.iloc[ri], align=False).area.values
    union = p.iloc[pi].area.values + r.iloc[ri].area.values - inter
    return pd.DataFrame({"pred_idx": pi, "ref_idx": ri, "iou": inter / np.maximum(union, 1e-9)})


def greedy_one_to_one(pairs: pd.DataFrame, iou_threshold: float) -> list[tuple[int, int, float]]:
    matches, used_p, used_r = [], set(), set()
    for row in pairs.sort_values("iou", ascending=False).itertuples():
        if row.iou < iou_threshold:
            break
        if row.pred_idx in used_p or row.ref_idx in used_r:
            continue
        matches.append((int(row.pred_idx), int(row.ref_idx), float(row.iou)))
        used_p.add(row.pred_idx)
        used_r.add(row.ref_idx)
    return matches


def object_scores(pred: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, iou_threshold: float) -> dict:
    assert pred.crs == ref.crs, "prediction and reference CRS differ"
    pairs = candidate_pairs(pred, ref)
    matches = greedy_one_to_one(pairs, iou_threshold)
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(ref) - tp
    merges = int((pairs.groupby("pred_idx").size() > 1).sum()) if len(pairs) else 0
    splits = int((pairs.groupby("ref_idx").size() > 1).sum()) if len(pairs) else 0
    return {
        "matching": f"greedy one-to-one, polygon IoU >= {iou_threshold}",
        "n_pred": len(pred), "n_ref": len(ref),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "mean_iou_of_matches": float(np.mean([m[2] for m in matches])) if matches else float("nan"),
        "pred_merging_multiple_refs": merges,
        "refs_split_across_multiple_preds": splits,
    }


def footprint_detection_rates(pred: gpd.GeoDataFrame, footprints: gpd.GeoDataFrame,
                              footprint_overlap: float = 0.5, region_cover: float = 0.25) -> dict:
    assert pred.crs == footprints.crs
    if len(pred) == 0 or len(footprints) == 0:
        return {"n_footprints": len(footprints), "n_regions": len(pred)}
    pred_union = pred.geometry.union_all()
    fp_area = footprints.geometry.area
    inside = footprints.geometry.intersection(pred_union).area
    detected = (inside / fp_area) >= footprint_overlap

    fp_union = footprints.geometry.union_all()
    covered = pred.geometry.intersection(fp_union).area / pred.geometry.area
    supported = covered >= region_cover
    return {
        "rule": f"footprint detected if >= {footprint_overlap:.0%} of it lies in predicted regions; "
                f"region supported if footprints cover >= {region_cover:.0%} of it (many-to-one allowed)",
        "n_footprints": int(len(footprints)),
        "building_detection_rate": float(detected.mean()),
        "n_regions": int(len(pred)),
        "region_support_rate": float(supported.mean()),
    }
