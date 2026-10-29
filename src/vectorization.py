"""Convert binary building rasters into vector polygons (GeoPandas / Shapely).

Method:
1. (optional) OpenCV morphological opening to remove 1-pixel speckle.
2. OpenCV connected components (4- or 8-connectivity) to drop regions smaller
   than `min_region_pixels`.
3. rasterio.features.shapes (GDAL's GDALPolygonize) traces each connected
   region's pixel edges into a polygon in the raster's CRS.
4. Shapely make_valid repairs any self-touching rings (8-connected pixels that
   meet only at a corner produce them); empty / non-areal results are dropped.

IMPORTANT: at 10 m, adjacent houses are merged into one connected region, so
each polygon is a "contiguous built-up region", NOT an individual building.
The `object_type` column states this explicitly.
"""
from __future__ import annotations

import cv2
import geopandas as gpd
import numpy as np
from rasterio import features
from shapely import make_valid
from shapely.geometry import shape


def clean_mask(mask: np.ndarray, connectivity: int, min_pixels: int, opening_kernel: int = 0) -> np.ndarray:
    m = (mask == 1).astype(np.uint8)
    if opening_kernel and opening_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_kernel, opening_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if min_pixels > 1:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=connectivity)
        small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < min_pixels)
        small = small[small != 0]  # label 0 is background
        m[np.isin(labels, small)] = 0
    return m


def mask_to_polygons(mask: np.ndarray, transform, crs, connectivity: int = 8, min_pixels: int = 1,
                     opening_kernel: int = 0, object_type: str = "contiguous built-up region") -> gpd.GeoDataFrame:
    m = clean_mask(mask, connectivity, min_pixels, opening_kernel)
    geoms = []
    for geom, val in features.shapes(m, mask=m.astype(bool), transform=transform, connectivity=connectivity):
        if val != 1:
            continue
        g = make_valid(shape(geom))
        if g.geom_type == "GeometryCollection":  # keep only the areal parts
            parts = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
            g = parts[0] if len(parts) == 1 else (make_valid(gpd.GeoSeries(parts).union_all()) if parts else None)
        if g is not None and not g.is_empty and g.area > 0:
            geoms.append(g)
    gdf = gpd.GeoDataFrame({"object_type": [object_type] * len(geoms)}, geometry=geoms, crs=crs)
    gdf["area_m2"] = gdf.geometry.area
    gdf["region_id"] = np.arange(len(gdf))
    assert gdf.geometry.is_valid.all()
    return gdf
