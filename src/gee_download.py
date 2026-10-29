"""Sentinel-2 composites and Open Buildings V3 footprints via Google Earth Engine.

This is the Earth Engine acquisition path. It needs, once per machine:

    .\\.venv\\Scripts\\earthengine.exe authenticate

and a Google Cloud project registered for Earth Engine, whose id goes in
config.yaml -> imagery.gee_project. Credentials are stored by the Earth Engine
client in your user profile, never inside this project.

Differences from the Planetary Computer path (documented, not hidden):
* Collection COPERNICUS/S2_SR_HARMONIZED, in which Google has already
  removed the baseline-04.00 +1000 offset, so we only divide by 10000.
* Cloud masking uses Google's Cloud Score+ product (cs_cdf >= 0.60 = clear)
  instead of the SCL layer. Cloud Score+ is generally more reliable than SCL.
* Earth Engine resamples on request: the 20 m bands are resampled bilinearly
  onto our 10 m grid via `.resample('bilinear')`.
"""
from __future__ import annotations

import logging

import numpy as np

from .imagery_preprocessing import nan_to_nodata, write_geotiff
from .study_area import GridSpec

log = logging.getLogger(__name__)

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLEAR_THRESHOLD = 0.60
GEE_BAND = {"B02": "B2", "B03": "B3", "B04": "B4", "B08": "B8", "B11": "B11", "B12": "B12"}
CHUNK = 512  # pixels per side per request; keeps each request well below the 48 MB limit


def initialize(project: str | None):
    """Initialise Earth Engine; raises a clear error if not authenticated."""
    import ee

    if not project:
        raise RuntimeError("Set imagery.gee_project in config/config.yaml to your Cloud project id.")
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine is not authenticated. Run: .\\.venv\\Scripts\\earthengine.exe authenticate"
        ) from exc
    return ee


def _region(ee, grid: GridSpec):
    return ee.Geometry.Rectangle(list(grid.bounds), proj=grid.crs, geodesic=False)


def composite_image(ee, cfg: dict, grid: GridSpec, period: str):
    icfg = cfg["imagery"]
    start, end = icfg["periods"][period]
    region = _region(ee, grid)
    bands = [GEE_BAND[b] for b in icfg["bands"]]
    s2 = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(region)
        .filterDate(start, ee.Date(end).advance(1, "day"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", icfg["max_scene_cloud_pct"]))
        .linkCollection(ee.ImageCollection(CS_PLUS), ["cs_cdf"])
    )

    def mask(img):
        clear = img.select("cs_cdf").gte(CLEAR_THRESHOLD)
        return img.select(bands).updateMask(clear).divide(icfg["reflectance_scale"]).resample("bilinear")

    n_scenes = s2.size().getInfo()
    comp = s2.map(mask).median().select(bands, icfg["bands"]).toFloat()
    return comp, n_scenes


def download_image(ee, image, grid: GridSpec, n_bands: int) -> np.ndarray:
    """Fetch pixels on exactly our grid with ee.data.computePixels, in chunks."""
    out = np.full((n_bands, grid.height, grid.width), np.nan, np.float32)
    t = grid.transform
    for r0 in range(0, grid.height, CHUNK):
        for c0 in range(0, grid.width, CHUNK):
            h, w = min(CHUNK, grid.height - r0), min(CHUNK, grid.width - c0)
            request = {
                "expression": image,
                "fileFormat": "NUMPY_NDARRAY",
                "grid": {
                    "dimensions": {"width": w, "height": h},
                    "affineTransform": {
                        "scaleX": t.a, "shearX": 0, "translateX": t.c + c0 * t.a,
                        "shearY": 0, "scaleY": t.e, "translateY": t.f + r0 * t.e,
                    },
                    "crsCode": grid.crs,
                },
            }
            arr = ee.data.computePixels(request)  # structured array, one field per band
            for i, name in enumerate(arr.dtype.names):
                block = arr[name].astype(np.float32)
                out[i, r0:r0 + h, c0:c0 + w] = block
    return out


def build_composite(cfg: dict, grid: GridSpec, period: str, out_path) -> dict:
    ee = initialize(cfg["imagery"]["gee_project"])
    comp, n_scenes = composite_image(ee, cfg, grid, period)
    data = download_image(ee, comp.unmask(-9999), grid, len(cfg["imagery"]["bands"]))
    data[data == -9999] = np.nan
    meta = {
        "source": f"Google Earth Engine {S2_COLLECTION}",
        "period": period,
        "date_range": cfg["imagery"]["periods"][period],
        "scene_cloud_filter_pct": cfg["imagery"]["max_scene_cloud_pct"],
        "pixel_mask": f"Cloud Score+ cs_cdf >= {CLEAR_THRESHOLD}",
        "composite": "median",
        "n_scenes": n_scenes,
        "valid_fraction": float(np.isfinite(data[0]).mean()),
    }
    nodata = cfg["imagery"]["nodata"]
    write_geotiff(out_path, nan_to_nodata(data, nodata), grid, nodata, cfg["imagery"]["bands"], meta)
    return meta


def download_open_buildings(cfg: dict, grid: GridSpec):
    """Open Buildings V3 polygons inside the grid, as a GeoDataFrame (EPSG:4326)."""
    import geopandas as gpd

    ee = initialize(cfg["imagery"]["gee_project"])
    fc = (
        ee.FeatureCollection(cfg["labels"]["gee_collection"])
        .filterBounds(_region(ee, grid))
        .filter(ee.Filter.gte("confidence", cfg["labels"]["min_confidence"]))
    )
    # computeFeatures pages through results; large areas can take minutes.
    gdf = ee.data.computeFeatures({"expression": fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
