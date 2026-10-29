"""Sentinel-2 L2A composites from Microsoft Planetary Computer (no account needed).

This is the "no-account fallback" acquisition path. It produces the same
GeoTIFF structure as the Earth Engine path in gee_download.py.

Steps for one period:
1. STAC search for Sentinel-2 L2A scenes over the study area and date range,
   keeping only scenes whose *scene-level* cloud cover is below a threshold.
2. For the lowest-cloud acquisition dates, read each band directly onto our
   10 m UTM grid with GDAL's warper (rasterio WarpedVRT):
     * 10 m bands: nearest neighbour; when the scene is already on our UTM zone
       and 10 m lattice this is an exact pixel copy (no resampling at all).
     * 20 m bands (B11, B12): bilinear, because they must be upsampled.
     * SCL (20 m class map): nearest neighbour - never interpolate categories.
3. Mask cloudy / shadowed pixels with the Scene Classification Layer (SCL).
4. Convert digital numbers to surface reflectance, handling the -1000 offset
   that ESA introduced with processing baseline 04.00 (Jan 2022).
5. Median-composite all clear observations per pixel.

Scene filtering vs pixel masking: the cloud-% filter throws away whole scenes
that are mostly cloudy (cheap, coarse). Pixel masking removes the remaining
individual cloudy / shadow pixels inside the scenes we kept (fine-grained).
Both are needed: a 10% cloudy scene can still have a cloud over our city.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import date

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping

from .imagery_preprocessing import median_composite, nan_to_nodata, write_geotiff
from .study_area import GridSpec

log = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
)


def search_scenes(grid: GridSpec, start: str, end: str, max_cloud: float):
    """Return STAC items intersecting the grid, filtered by scene cloud %."""
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    aoi = transform_geom(grid.crs, "EPSG:4326", mapping(box(*grid.bounds)))
    search = catalog.search(
        collections=[COLLECTION],
        intersects=aoi,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


def pick_dates(items, max_dates: int) -> dict[date, list]:
    """Group items by acquisition date and keep the clearest dates.

    One date can have several items (adjacent MGRS tiles both covering part
    of the area); they are kept together so the date is fully covered.
    """
    # Planetary Computer can hold the same acquisition twice (original processing
    # and an ESA reprocessing, e.g. baseline 04.00 and 05.10). Keep only the
    # newest processing per (date, MGRS tile) so each pixel has one source.
    newest = {}
    for it in items:
        key = (it.datetime.date(), it.properties.get("s2:mgrs_tile"))
        if key not in newest or it.id > newest[key].id:  # id ends with the processing timestamp
            newest[key] = it
    by_date = defaultdict(list)
    for it in newest.values():
        by_date[it.datetime.date()].append(it)
    ranked = sorted(by_date, key=lambda d: np.mean([i.properties["eo:cloud_cover"] for i in by_date[d]]))
    return {d: by_date[d] for d in sorted(ranked[:max_dates])}


def _boa_offset(item) -> float:
    """Return the additive offset for converting DN to reflectance.

    Since processing baseline 04.00 (25 Jan 2022) L2A DNs carry +1000, so
    reflectance = (DN - 1000) / 10000. Older products have no offset.
    """
    baseline = item.properties.get("s2:processing_baseline")
    try:
        return -1000.0 if float(baseline) >= 4.0 else 0.0
    except (TypeError, ValueError):
        # Fall back on the date the change took effect if the property is missing.
        return -1000.0 if item.datetime.date() >= date(2022, 1, 25) else 0.0


def _read_on_grid(href: str, grid: GridSpec, resampling: Resampling) -> np.ndarray:
    """Read one single-band COG onto the analysis grid (0 = nodata)."""
    for attempt in range(3):
        try:
            with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
                same_lattice = (
                    src.crs.to_string() == grid.crs
                    and src.transform.a == grid.pixel_size
                    and (grid.transform.c - src.transform.c) % grid.pixel_size == 0
                    and (grid.transform.f - src.transform.f) % grid.pixel_size == 0
                )
                rs = Resampling.nearest if same_lattice else resampling
                with WarpedVRT(
                    src,
                    crs=grid.crs,
                    transform=grid.transform,
                    width=grid.width,
                    height=grid.height,
                    resampling=rs,
                    nodata=0,
                ) as vrt:
                    return vrt.read(1)
        except rasterio.errors.RasterioIOError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def read_item(item, grid: GridSpec, cfg: dict) -> np.ndarray:
    """Read one scene as (bands, rows, cols) float32 reflectance, NaN where masked."""
    import planetary_computer

    item = planetary_computer.sign(item)  # fresh short-lived URL signature
    icfg = cfg["imagery"]
    scl = _read_on_grid(item.assets["SCL"].href, grid, Resampling.nearest)
    bad = np.isin(scl, icfg["scl_invalid_classes"])
    offset = _boa_offset(item)
    out = np.empty((len(icfg["bands"]), grid.height, grid.width), dtype=np.float32)
    for i, band in enumerate(icfg["bands"]):
        rs = Resampling.nearest if band in icfg["bands_10m"] else Resampling.bilinear
        dn = _read_on_grid(item.assets[band].href, grid, rs).astype(np.float32)
        refl = (dn + offset) / icfg["reflectance_scale"]
        refl[(dn == 0) | bad] = np.nan
        out[i] = refl
    return out


def build_composite(cfg: dict, grid: GridSpec, period: str, out_path) -> dict:
    """Search, read, mask and composite one period; write a GeoTIFF; return metadata."""
    icfg = cfg["imagery"]
    start, end = icfg["periods"][period]
    items = search_scenes(grid, start, end, icfg["max_scene_cloud_pct"])
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {period} ({start}..{end}).")
    dates = pick_dates(items, icfg["max_scenes"])
    log.info("%s: %d candidate items, using %d dates", period, len(items), len(dates))

    stack, used, failed = [], [], []
    for d, day_items in dates.items():
        day = np.full((len(icfg["bands"]), grid.height, grid.width), np.nan, np.float32)
        for it in day_items:
            try:
                arr = read_item(it, grid, cfg)
            except Exception as exc:  # recorded in the metadata, never silent
                log.warning("Failed to read %s: %s", it.id, exc)
                failed.append({"id": it.id, "error": str(exc)[:200]})
                continue
            fill = np.isnan(day) & np.isfinite(arr)  # first valid value wins
            day[fill] = arr[fill]
            used.append(
                {
                    "id": it.id,
                    "date": str(d),
                    "cloud_cover": it.properties["eo:cloud_cover"],
                    "processing_baseline": it.properties.get("s2:processing_baseline"),
                    "boa_offset": _boa_offset(it),
                }
            )
        stack.append(day)
        log.info("  %s read (%d items)", d, len(day_items))

    if not stack:
        raise RuntimeError(f"Every scene failed to read for {period}.")
    comp, count = median_composite(np.stack(stack), icfg["min_valid_observations"])
    meta = {
        "source": "Microsoft Planetary Computer sentinel-2-l2a",
        "period": period,
        "date_range": [start, end],
        "scene_cloud_filter_pct": icfg["max_scene_cloud_pct"],
        "pixel_mask": f"SCL classes {icfg['scl_invalid_classes']}",
        "composite": "median",
        "dates_used": sorted({u["date"] for u in used}),
        "items_used": used,
        "items_failed": failed,
        "valid_fraction": float(np.isfinite(comp[0]).mean()),
    }
    write_geotiff(out_path, nan_to_nodata(comp, icfg["nodata"]), grid, icfg["nodata"], icfg["bands"], meta)
    write_geotiff(
        out_path.with_name(out_path.stem + "_validcount.tif"), count, grid, None, ["clear_observations"]
    )
    return meta
