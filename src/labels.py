"""Google Open Buildings V3 reference labels: download and rasterization.

Source: Google Research, Open Buildings V3 (https://sites.research.google/open-buildings/),
licence CC BY-4.0 / ODbL (attribution required). Footprints were detected by
Google from 50 cm satellite imagery; the dataset covers Africa, South Asia,
South-East Asia, Latin America and the Caribbean, including India.

No-account path: Google publishes the polygons as gzipped CSV files split by
S2 geometry level-6 cells. We find the cell(s) covering our area with the
s2sphere library and stream each file, keeping only rows inside our area -
the files can be over 1 GB so they are never stored in full.

These polygons are a REFERENCE, not perfect ground truth: they miss some
buildings, merge/split others, and come from imagery of a different date
than our Sentinel-2 composites.
"""
from __future__ import annotations

import csv
import gzip
import io
import logging

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rasterio import features
from rasterio.transform import Affine
from shapely import wkt

from .study_area import GridSpec

log = logging.getLogger(__name__)

# Column order of the *_no_header CSV files (same as the documented header).
OB_COLUMNS = ["latitude", "longitude", "area_in_meters", "confidence", "geometry", "full_plus_code"]


def s2_level6_tokens(lonlat_bounds: tuple[float, float, float, float]) -> list[str]:
    """S2 level-6 cell tokens covering a lon/lat bounding box."""
    import s2sphere

    w, s, e, n = lonlat_bounds
    rect = s2sphere.LatLngRect.from_point_pair(
        s2sphere.LatLng.from_degrees(s, w), s2sphere.LatLng.from_degrees(n, e)
    )
    coverer = s2sphere.RegionCoverer()
    coverer.min_level = coverer.max_level = 6
    return sorted(c.to_token() for c in coverer.get_covering(rect))


def stream_filter_csv(url: str, lonlat_bounds, min_conf: float, chunk_rows: int = 200_000) -> pd.DataFrame:
    """Stream a gzipped Open Buildings CSV and keep rows whose centroid is in the box."""
    w, s, e, n = lonlat_bounds
    kept, total = [], 0
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        text = io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw), encoding="utf-8")
        reader = pd.read_csv(text, header=None, names=OB_COLUMNS, chunksize=chunk_rows,
                             quoting=csv.QUOTE_MINIMAL)
        for chunk in reader:
            total += len(chunk)
            m = chunk.longitude.between(w, e) & chunk.latitude.between(s, n) & (chunk.confidence >= min_conf)
            if m.any():
                kept.append(chunk[m])
            if total % 5_000_000 < chunk_rows:
                log.info("  scanned %.1f M rows, kept %d", total / 1e6, sum(len(k) for k in kept))
    df = pd.concat(kept) if kept else pd.DataFrame(columns=OB_COLUMNS)
    df.attrs["rows_scanned"] = total
    return df


def download_open_buildings(cfg: dict, grid: GridSpec, lonlat_bounds) -> tuple[gpd.GeoDataFrame, dict]:
    """No-account download of footprints for the grid; returns (gdf in EPSG:4326, metadata)."""
    lcfg = cfg["labels"]
    # Pad the box by ~200 m so buildings straddling the edge keep their full outline.
    pad = 0.002
    box_ll = (lonlat_bounds[0] - pad, lonlat_bounds[1] - pad, lonlat_bounds[2] + pad, lonlat_bounds[3] + pad)
    tokens = s2_level6_tokens(box_ll)
    frames, files = [], []
    for tok in tokens:
        url = f"{lcfg['bucket_url']}/{tok}_buildings.csv.gz"
        log.info("Streaming %s", url)
        df = stream_filter_csv(url, box_ll, lcfg["min_confidence"])
        frames.append(df)
        files.append({"url": url, "rows_scanned": df.attrs["rows_scanned"], "rows_kept": len(df)})
    df = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(
        df.drop(columns="geometry"), geometry=df["geometry"].map(wkt.loads), crs="EPSG:4326"
    )
    meta = {"source": "Google Open Buildings V3 (public GCS CSV)", "s2_level6_cells": tokens, "files": files}
    return gdf, meta


def clean_footprints(gdf: gpd.GeoDataFrame, crs: str) -> tuple[gpd.GeoDataFrame, dict]:
    """Project to the analysis CRS and repair or drop invalid geometries."""
    gdf = gdf.to_crs(crs)
    n_in = len(gdf)
    invalid = ~gdf.geometry.is_valid
    gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)  # standard repair
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]
    return gdf.reset_index(drop=True), {"input": n_in, "repaired": int(invalid.sum()), "kept": len(gdf)}


def rasterize_coverage(gdf: gpd.GeoDataFrame, grid: GridSpec, supersample: int) -> np.ndarray:
    """Fraction (0-1) of each grid pixel covered by building footprints.

    Rule: burn polygons onto a grid `supersample` times finer (2 m for a 10 m
    grid with supersample=5) using the pixel-centre rule, then average each
    5 x 5 block back to 10 m with OpenCV INTER_AREA (an exact block mean for an
    integer factor). This is much less biased than either "all-touched"
    (overestimates small buildings) or plain centre-sampling at 10 m
    (misses buildings that don't cover a pixel centre).
    """
    t = grid.transform
    fine = Affine(t.a / supersample, 0, t.c, 0, t.e / supersample, t.f)
    shape = (grid.height * supersample, grid.width * supersample)
    burned = features.rasterize(
        ((g, 1) for g in gdf.geometry), out_shape=shape, transform=fine, fill=0, dtype="uint8"
    )
    frac = cv2.resize(burned.astype(np.float32), (grid.width, grid.height), interpolation=cv2.INTER_AREA)
    return np.clip(frac, 0.0, 1.0)


def coverage_to_mask(frac: np.ndarray, threshold: float) -> np.ndarray:
    """Binary building mask: 1 where at least `threshold` of the pixel is roof."""
    return (frac >= threshold).astype(np.uint8)
