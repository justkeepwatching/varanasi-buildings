"""Cut aligned image / mask rasters into 64 x 64 tiles and split them spatially.

Key rules (all configurable in config.yaml -> tiling / split):
* Tiles are non-overlapping (stride = tile size) and start at the grid's
  top-left corner, so each tile's footprint is exactly 640 m x 640 m at 10 m.
* Edge policy "discard": a partial tile at the right/bottom edge is not used.
* Tiles with valid-pixel fraction < min_valid_fraction are discarded.
* Image and mask tiles are cut with the SAME pixel window, so they cannot
  drift apart; nothing is resized.
* Train/val/test assignment is by geographic BLOCK (block_size x block_size
  tiles), not by individual tile, to limit spatial leakage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from .study_area import GridSpec


def tile_origins(height: int, width: int, tile: int, stride: int) -> list[tuple[int, int]]:
    """Top-left (row, col) of every FULL tile; partial edge tiles are excluded."""
    return [(r, c) for r in range(0, height - tile + 1, stride) for c in range(0, width - tile + 1, stride)]


def cut_tiles(image, valid, mask, frac, grid: GridSpec, cfg: dict, source: str):
    """Cut tiles and build a manifest row for each kept tile.

    image: (bands, H, W) reflectance; valid: (H, W) bool; mask: (H, W) uint8 0/1;
    frac: (H, W) building coverage fraction.
    Returns (images, masks, valids, manifest DataFrame).
    """
    tcfg = cfg["tiling"]
    ts = tcfg["tile_size"]
    assert image.shape[1:] == mask.shape == valid.shape == (grid.height, grid.width)
    assert set(np.unique(mask)).issubset({0, 1}), "mask must be binary"

    imgs, msks, vals, rows = [], [], [], []
    n_total = n_lowvalid = n_empty = 0
    for r, c in tile_origins(grid.height, grid.width, ts, tcfg["stride"]):
        n_total += 1
        v = valid[r:r + ts, c:c + ts]
        vf = float(v.mean())
        if vf < tcfg["min_valid_fraction"]:
            n_lowvalid += 1
            continue
        m = mask[r:r + ts, c:c + ts] * v  # nodata pixels never count as buildings
        if not tcfg["keep_empty_tiles"] and m.sum() == 0:
            n_empty += 1
            continue
        t = window_transform(Window(c, r, ts, ts), grid.transform)
        left, top = t.c, t.f
        imgs.append(image[:, r:r + ts, c:c + ts])
        msks.append(m.astype(np.uint8))
        vals.append(v)
        rows.append(
            dict(
                tile_id=f"r{r:04d}_c{c:04d}",
                row=r, col=c, height=ts, width=ts,
                minx=left, maxy=top, maxx=left + ts * t.a, miny=top + ts * t.e,
                crs=grid.crs, source=source,
                valid_fraction=round(vf, 4),
                building_pixel_fraction=round(float(m[v].mean()) if v.any() else 0.0, 4),
                building_coverage_mean=round(float(frac[r:r + ts, c:c + ts][v].mean()) if v.any() else 0.0, 4),
            )
        )
    manifest = pd.DataFrame(rows)
    manifest.attrs.update(candidates=n_total, dropped_low_valid=n_lowvalid, dropped_empty=n_empty)
    images = np.stack(imgs).astype(np.float32)
    masks = np.stack(msks)
    valids = np.stack(vals)
    assert images.shape[1:] == (image.shape[0], ts, ts) and masks.shape[1:] == (ts, ts)
    return images, masks, valids, manifest


def assign_spatial_split(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Assign whole geographic blocks to train / val / test.

    Blocks are ranked by building density and grouped into strata of
    ~1/min(fraction) blocks; inside each stratum one block goes to test, one to
    val and the rest to train (order shuffled with the project seed). This
    keeps urban and rural blocks represented in every split while every
    block - and therefore every tile - belongs to exactly one split.
    """
    ts = cfg["tiling"]["tile_size"]
    bs = cfg["split"]["block_size_tiles"] * ts
    fr = cfg["split"]["fractions"]
    rng = np.random.default_rng(cfg["project"]["seed"])

    m = manifest.copy()
    m["block_id"] = (m.row // bs).astype(str) + "_" + (m.col // bs).astype(str)
    density = m.groupby("block_id").building_pixel_fraction.mean().sort_values()
    group = max(3, int(round(1 / min(fr["val"], fr["test"]))))
    split_of = {}
    blocks = list(density.index)
    for i in range(0, len(blocks), group):
        stratum = list(blocks[i:i + group])
        rng.shuffle(stratum)
        labels = ["test", "val"] + ["train"] * (len(stratum) - 2)
        if len(stratum) < 3:  # a tiny leftover stratum goes to training
            labels = ["train"] * len(stratum)
        split_of.update(zip(stratum, labels))
    m["split"] = m.block_id.map(split_of)
    return m


def write_tile_geotiffs(images, masks, manifest: pd.DataFrame, out_dir: Path, band_names) -> None:
    """Save every tile as a small georeferenced GeoTIFF (image + mask)."""
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(manifest.itertuples()):
        tr = rasterio.transform.from_origin(rec.minx, rec.maxy, (rec.maxx - rec.minx) / rec.width, (rec.maxy - rec.miny) / rec.height)
        common = dict(driver="GTiff", height=rec.height, width=rec.width, crs=rec.crs, transform=tr)
        with rasterio.open(out_dir / "images" / f"{rec.tile_id}.tif", "w", count=images.shape[1],
                           dtype="float32", nodata=-9999.0, **common) as dst:
            dst.write(images[i])
            for b, name in enumerate(band_names, 1):
                dst.set_band_description(b, name)
        with rasterio.open(out_dir / "masks" / f"{rec.tile_id}.tif", "w", count=1, dtype="uint8", **common) as dst:
            dst.write(masks[i][None])
