"""Post-classification change detection between two building-coverage maps.

Each pixel of the analysed area falls in exactly one class:
    0 = stable non-building     (earlier 0, later 0)
    1 = potential gain          (earlier 0, later 1)
    2 = potential loss          (earlier 1, later 0)
    3 = persistent building     (earlier 1, later 1)
  255 = not analysed            (nodata in either period)

"Potential" because a classification difference can be a real change OR a
classification error in either year (haze, season, sensor noise). Errors in
the two maps add up, so change maps are noisier than either single map.

The optional "high-confidence" variant only counts a gain if the probability
went from < threshold - margin to > threshold + margin (and vice versa for
losses), which suppresses flicker of pixels sitting near the threshold.

Areas: on our UTM grid every pixel is pixel_size^2 = 100 m^2 (planar). We
also report a geodesic value using the grid's geodesic/planar ratio from
study_area.area_report, since UTM is not exactly equal-area.
"""
from __future__ import annotations

import numpy as np

STABLE_NON, GAIN, LOSS, PERSISTENT, NOT_ANALYSED = 0, 1, 2, 3, 255
CLASS_NAMES = {STABLE_NON: "stable_non_building", GAIN: "potential_gain", LOSS: "potential_loss",
               PERSISTENT: "persistent_building"}


def change_classes(earlier: np.ndarray, later: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cross-tabulate two binary maps into the change classes above."""
    e = earlier.astype(bool)
    l = later.astype(bool)
    out = np.full(e.shape, NOT_ANALYSED, np.uint8)
    out[valid & ~e & ~l] = STABLE_NON
    out[valid & ~e & l] = GAIN
    out[valid & e & ~l] = LOSS
    out[valid & e & l] = PERSISTENT
    return out


def confident_change_classes(p_early, p_late, valid, threshold: float, margin: float) -> np.ndarray:
    lo, hi = threshold - margin, threshold + margin
    out = change_classes(p_early >= threshold, p_late >= threshold, valid)
    gain = out == GAIN
    loss = out == LOSS
    # demote uncertain changes to their "stable" counterpart
    out[gain & ~((p_early < lo) & (p_late > hi))] = STABLE_NON
    out[loss & ~((p_early > hi) & (p_late < lo))] = PERSISTENT
    return out


def pixel_area_km2(n_pixels: int, pixel_size_m: float) -> float:
    return n_pixels * pixel_size_m**2 / 1e6


def change_statistics(classes: np.ndarray, pixel_size_m: float, geodesic_ratio: float = 1.0,
                      total_area_km2: float | None = None) -> dict:
    """Areas (km^2) and shares of the analysed area for every change quantity."""
    n = {name: int(np.sum(classes == code)) for code, name in CLASS_NAMES.items()}
    n_valid = sum(n.values())
    a = lambda px: pixel_area_km2(px, pixel_size_m)  # noqa: E731
    earlier = n["potential_loss"] + n["persistent_building"]
    later = n["potential_gain"] + n["persistent_building"]
    stats = {
        "valid_analysed_area_km2": a(n_valid),
        "not_analysed_area_km2": a(int(np.sum(classes == NOT_ANALYSED))),
        "earlier_built_up_km2": a(earlier),
        "later_built_up_km2": a(later),
        "gross_gain_km2": a(n["potential_gain"]),
        "gross_loss_km2": a(n["potential_loss"]),
        "net_change_km2": a(n["potential_gain"]) - a(n["potential_loss"]),
        "persistent_km2": a(n["persistent_building"]),
        "pixel_counts": n,
    }
    for k in ("earlier_built_up_km2", "later_built_up_km2", "gross_gain_km2", "gross_loss_km2",
              "net_change_km2", "persistent_km2"):
        stats[k.replace("_km2", "_pct_of_valid")] = 100 * stats[k] / stats["valid_analysed_area_km2"] if n_valid else float("nan")
        stats[k.replace("_km2", "_km2_geodesic")] = stats[k] * geodesic_ratio
    if total_area_km2:
        stats["study_area_km2"] = total_area_km2
    return stats
