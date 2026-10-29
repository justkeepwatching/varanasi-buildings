"""Define the study-area boundaries and the 10 m analysis grid.

Two nested square areas are used, both on the same UTM pixel grid:

* the WIDE area (15.36 km x 15.36 km) supplies training / validation / test
  tiles for the U-Net;
* the CORE area (5 km x 5 km) is the ~25 km^2 reporting area for building
  coverage and 2020 -> 2025 change.

Because the core grid is an exact pixel window of the wide grid, rasters of
the two areas can be cropped into each other without any resampling.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from pyproj import Geod, Transformer
from rasterio.transform import Affine
from shapely.geometry import box


@dataclass(frozen=True)
class GridSpec:
    """A north-up raster grid: CRS + affine transform + size in pixels."""

    crs: str
    transform: Affine
    width: int
    height: int

    @property
    def pixel_size(self) -> float:
        return self.transform.a

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in CRS units (metres)."""
        left, top = self.transform.c, self.transform.f
        right = left + self.width * self.transform.a
        bottom = top + self.height * self.transform.e
        return left, bottom, right, top

    def window_in(self, parent: "GridSpec") -> tuple[int, int]:
        """(row_off, col_off) of this grid inside `parent`; asserts exact alignment."""
        assert self.crs == parent.crs and self.pixel_size == parent.pixel_size
        col = (self.transform.c - parent.transform.c) / parent.pixel_size
        row = (parent.transform.f - self.transform.f) / parent.pixel_size
        assert col == int(col) and row == int(row), "grids are not pixel-aligned"
        return int(row), int(col)


def _center_utm(cfg: dict) -> tuple[float, float]:
    """Centre point projected to UTM and snapped to the pixel grid."""
    sa = cfg["study_area"]
    to_utm = Transformer.from_crs("EPSG:4326", sa["crs"], always_xy=True)
    x, y = to_utm.transform(*sa["center_lonlat"])
    px = sa["pixel_size_m"]
    return round(x / px) * px, round(y / px) * px


def wide_grid(cfg: dict) -> GridSpec:
    sa = cfg["study_area"]
    px = sa["pixel_size_m"]
    n = sa["wide_size_tiles"] * cfg["tiling"]["tile_size"]
    xc, yc = _center_utm(cfg)
    half = n * px / 2
    return GridSpec(sa["crs"], Affine(px, 0, xc - half, 0, -px, yc + half), n, n)


def core_grid(cfg: dict) -> GridSpec:
    sa = cfg["study_area"]
    px = sa["pixel_size_m"]
    n = int(sa["core_size_m"] / px)
    xc, yc = _center_utm(cfg)
    half = sa["core_size_m"] / 2
    grid = GridSpec(sa["crs"], Affine(px, 0, xc - half, 0, -px, yc + half), n, n)
    grid.window_in(wide_grid(cfg))  # asserts that core is aligned with wide
    return grid


def grid_polygon_gdf(grid: GridSpec, name: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"name": [name]}, geometry=[box(*grid.bounds)], crs=grid.crs)


def area_report(grid: GridSpec) -> dict:
    """Planar (UTM) and geodesic (WGS84 ellipsoid) area of a grid's footprint.

    UTM distorts distances slightly (scale factor 0.9996 at the central
    meridian, larger away from it), so the true on-the-ground area differs a
    little from the planar value. We report both rather than assume.
    """
    gdf = grid_polygon_gdf(grid, "x")
    planar_km2 = gdf.geometry.area.iloc[0] / 1e6
    lonlat = gdf.to_crs("EPSG:4326").geometry.iloc[0]
    geod_area, _ = Geod(ellps="WGS84").geometry_area_perimeter(lonlat.segmentize(0.001))
    geodesic_km2 = abs(geod_area) / 1e6
    return {
        "width_px": grid.width,
        "height_px": grid.height,
        "pixel_size_m": grid.pixel_size,
        "planar_area_km2": round(planar_km2, 4),
        "geodesic_area_km2": round(geodesic_km2, 4),
        # multiply planar pixel areas by this to obtain geodesic areas
        "geodesic_to_planar_ratio": round(geodesic_km2 / planar_km2, 6),
        "bounds_utm": [round(v, 2) for v in grid.bounds],
        "bounds_lonlat": [round(v, 6) for v in lonlat.bounds],
    }


def write_boundaries(cfg: dict, out_dir) -> dict:
    """Save both boundaries as GeoJSON (WGS84) and return their area reports."""
    out = {}
    for name, grid in (("wide", wide_grid(cfg)), ("core", core_grid(cfg))):
        grid_polygon_gdf(grid, name).to_crs("EPSG:4326").to_file(
            out_dir / f"study_area_{name}.geojson", driver="GeoJSON"
        )
        out[name] = area_report(grid)
    return out
