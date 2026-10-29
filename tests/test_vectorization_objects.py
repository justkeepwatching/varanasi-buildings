import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from shapely.geometry import box

from src.object_evaluation import footprint_detection_rates, greedy_one_to_one, object_scores
from src.vectorization import mask_to_polygons

T = Affine(10, 0, 700000, 0, -10, 2800000)
CRS = "EPSG:32644"


def test_polygonize_valid_crs_and_area():
    m = np.zeros((20, 20), np.uint8)
    m[2:5, 2:6] = 1      # 12 px region
    m[10, 10] = 1        # single pixel
    m[11, 11] = 1        # diagonal neighbour -> joined only with 8-connectivity
    g8 = mask_to_polygons(m, T, CRS, connectivity=8)
    g4 = mask_to_polygons(m, T, CRS, connectivity=4)
    assert g8.crs == CRS and g8.geometry.is_valid.all()
    assert len(g8) == 2 and len(g4) == 3
    assert sorted(g8.area_m2.round()) == [200, 1200]


def test_min_region_pixels_drops_speckle():
    m = np.zeros((10, 10), np.uint8)
    m[0, 0] = 1
    m[5:8, 5:8] = 1
    g = mask_to_polygons(m, T, CRS, connectivity=8, min_pixels=2)
    assert len(g) == 1 and g.area_m2.iloc[0] == 900


def _gdf(geoms):
    return gpd.GeoDataFrame(geometry=geoms, crs=CRS)


def test_one_to_one_no_double_counting():
    ref = _gdf([box(0, 0, 10, 10)])
    pred = _gdf([box(0, 0, 10, 10), box(1, 0, 10, 10)])  # both overlap the single reference
    s = object_scores(pred, ref, 0.5)
    assert s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 0
    assert s["refs_split_across_multiple_preds"] == 1


def test_merge_counts_as_one_match_at_most():
    ref = _gdf([box(0, 0, 10, 10), box(10, 0, 20, 10)])
    pred = _gdf([box(0, 0, 20, 10)])  # one prediction covering two references (IoU 0.5 each)
    s = object_scores(pred, ref, 0.5)
    assert s["tp"] == 1 and s["fn"] == 1 and s["pred_merging_multiple_refs"] == 1


def test_threshold_respected():
    ref = _gdf([box(0, 0, 10, 10)])
    pred = _gdf([box(5, 0, 15, 10)])  # IoU = 50/150 = 0.333
    assert object_scores(pred, ref, 0.5)["tp"] == 0
    assert object_scores(pred, ref, 0.3)["tp"] == 1


def test_greedy_prefers_highest_iou():
    import pandas as pd
    pairs = pd.DataFrame({"pred_idx": [0, 0, 1], "ref_idx": [0, 1, 0], "iou": [0.6, 0.9, 0.8]})
    assert sorted(greedy_one_to_one(pairs, 0.5)) == [(0, 1, 0.9), (1, 0, 0.8)]


def test_footprint_detection_rates():
    pred = _gdf([box(0, 0, 20, 20)])
    fps = _gdf([box(1, 1, 5, 5), box(30, 30, 35, 35)])
    r = footprint_detection_rates(pred, fps)
    assert r["building_detection_rate"] == 0.5
    assert r["region_support_rate"] == 0.0  # 16/400 = 4% < 25%
