import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from src.imagery_preprocessing import check_on_grid, write_geotiff
from src.labels import coverage_to_mask, rasterize_coverage
from src.study_area import area_report, core_grid, wide_grid


def test_core_is_pixel_aligned_inside_wide(cfg):
    wide, core = wide_grid(cfg), core_grid(cfg)
    r, c = core.window_in(wide)
    assert 0 <= r and r + core.height <= wide.height and 0 <= c and c + core.width <= wide.width
    assert wide.crs == core.crs == "EPSG:32644" and wide.pixel_size == core.pixel_size == 10


def test_area_report_core_close_to_25km2(cfg):
    rep = area_report(core_grid(cfg))
    assert rep["planar_area_km2"] == 25.0
    # geodesic area differs from planar by < 0.2% in this part of UTM 44N
    assert abs(rep["geodesic_area_km2"] - 25.0) / 25.0 < 0.002


def test_written_raster_matches_grid(tmp_path, synthetic_grid):
    data = np.zeros((2, synthetic_grid.height, synthetic_grid.width), np.float32)
    p = tmp_path / "x.tif"
    write_geotiff(p, data, synthetic_grid, -9999.0, ["a", "b"])
    check_on_grid(p, synthetic_grid)


def test_misaligned_raster_is_rejected(tmp_path, synthetic_grid):
    from dataclasses import replace

    from rasterio.transform import Affine
    shifted = replace(synthetic_grid, transform=Affine(10, 0, 700005, 0, -10, 2800000))
    p = tmp_path / "s.tif"
    write_geotiff(p, np.zeros((1, shifted.height, shifted.width), np.uint8), shifted, None)
    with pytest.raises(AssertionError):
        check_on_grid(p, synthetic_grid)


def test_rasterize_coverage_fraction_rule(synthetic_grid):
    t = synthetic_grid.transform
    x0, y0 = t.c, t.f
    polys = gpd.GeoDataFrame(geometry=[
        box(x0, y0 - 10, x0 + 10, y0),            # covers pixel (0,0) completely
        box(x0 + 20, y0 - 4, x0 + 24, y0),        # 4 m x 4 m = 16% of pixel (0,2)
        box(x0 + 40, y0 - 6, x0 + 46, y0),        # 6 m x 6 m = 36% of pixel (0,4)
    ], crs=synthetic_grid.crs)
    frac = rasterize_coverage(polys, synthetic_grid, supersample=5)
    assert frac.shape == (synthetic_grid.height, synthetic_grid.width)
    assert math.isclose(frac[0, 0], 1.0)
    assert math.isclose(frac[0, 2], 0.16, abs_tol=1e-6)  # 2x2 of the 5x5 sub-pixels
    assert math.isclose(frac[0, 4], 0.36, abs_tol=1e-6)  # 3x3 of 25
    m = coverage_to_mask(frac, 0.25)
    assert m[0, 0] == 1 and m[0, 2] == 0 and m[0, 4] == 1
    assert frac.sum() == pytest.approx(1 + 0.16 + 0.36)
