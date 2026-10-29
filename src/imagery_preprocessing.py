"""Shared raster helpers: writing GeoTIFFs, compositing, scaling and normalization.

Both acquisition paths (Planetary Computer and Google Earth Engine) end up
calling `write_geotiff` so their outputs have identical structure:
float32 surface reflectance, nodata = -9999, one named band per spectral band,
and acquisition metadata stored as GeoTIFF tags.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import rasterio

from .study_area import GridSpec


def write_geotiff(
    path: Path,
    data: np.ndarray,
    grid: GridSpec,
    nodata: float | int | None,
    band_names: list[str] | None = None,
    tags: dict | None = None,
) -> None:
    """Write a (bands, rows, cols) array on `grid` as a tiled, compressed GeoTIFF."""
    if data.ndim == 2:
        data = data[None]
    assert data.shape[1:] == (grid.height, grid.width), "array does not match grid"
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=data.shape[0],
        dtype=data.dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if band_names:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
        if tags:
            # GeoTIFF tags are strings; complex values are stored as JSON.
            dst.update_tags(**{k: v if isinstance(v, str) else json.dumps(v) for k, v in tags.items()})


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Read every band of a raster; returns (array, profile)."""
    with rasterio.open(path) as src:
        return src.read(), src.profile.copy()


def check_on_grid(path: Path, grid: GridSpec) -> None:
    """Assert that a raster file lies exactly on the analysis grid."""
    with rasterio.open(path) as src:
        assert src.crs.to_string() == grid.crs, f"{path.name}: CRS {src.crs} != {grid.crs}"
        assert (src.width, src.height) == (grid.width, grid.height), f"{path.name}: size mismatch"
        assert src.transform.almost_equals(grid.transform), f"{path.name}: transform mismatch"


def median_composite(stack: np.ndarray, min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median over time, ignoring NaNs (masked cloud/shadow pixels).

    stack: (time, bands, rows, cols) float32 with NaN where a pixel was masked.
    Returns (composite[bands, rows, cols], valid_count[rows, cols]).
    Pixels with fewer than `min_obs` clear observations become NaN, because a
    median of one or two looks is easily contaminated by undetected haze.
    """
    valid_count = np.isfinite(stack[:, 0]).sum(axis=0).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN pixels
        comp = np.nanmedian(stack, axis=0).astype(np.float32)
    comp[:, valid_count < min_obs] = np.nan
    return comp, valid_count


def nan_to_nodata(arr: np.ndarray, nodata: float) -> np.ndarray:
    out = arr.copy()
    out[~np.isfinite(out)] = nodata
    return out


def valid_mask_from_image(img: np.ndarray, nodata: float) -> np.ndarray:
    """A pixel is valid only if every band holds a finite, non-nodata value."""
    return np.all(np.isfinite(img) & (img != nodata), axis=0)


def compute_band_stats(images: np.ndarray, valid: np.ndarray) -> dict:
    """Per-band mean/std over valid pixels of the TRAINING tiles only.

    images: (n, bands, h, w); valid: (n, h, w) bool.
    Computing these on validation/test data would leak information.
    """
    means, stds = [], []
    for b in range(images.shape[1]):
        vals = images[:, b][valid]
        means.append(float(vals.mean()))
        stds.append(float(vals.std() + 1e-6))
    return {"mean": means, "std": stds}


def normalize(img: np.ndarray, stats: dict, valid: np.ndarray | None = None) -> np.ndarray:
    """Z-score each band; invalid pixels are set to 0 (= the band mean)."""
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    out = (img.astype(np.float32) - mean) / std
    if valid is not None:
        out[:, ~valid] = 0.0
    return out
