import numpy as np
import pandas as pd

from src.tiling import assign_spatial_split, cut_tiles, tile_origins


def test_tile_origins_non_overlapping_and_full_only():
    o = tile_origins(130, 200, 64, 64)  # 2 rows x 3 cols of full tiles; edges discarded
    assert len(o) == 6
    assert o[0] == (0, 0) and o[-1] == (64, 128)
    assert all(r + 64 <= 130 and c + 64 <= 200 for r, c in o)


def test_cut_tiles_dimensions_count_and_alignment(cfg, synthetic_grid, synthetic_scene):
    img, valid, mask, frac = synthetic_scene
    images, masks, valids, man = cut_tiles(img, valid, mask, frac, synthetic_grid, cfg, "synthetic")
    # 6 candidate tiles; the right-hand column (cols 128-191) is ~97% nodata -> dropped
    assert man.attrs["candidates"] == 6
    assert len(man) == 4 and man.attrs["dropped_low_valid"] == 2
    assert images.shape == (4, 6, 64, 64) and masks.shape == (4, 64, 64)
    assert set(np.unique(masks)) <= {0, 1}
    # image and mask tiles come from the same window
    t0 = man.iloc[0]
    assert (t0.row, t0.col) == (0, 0)
    assert np.array_equal(masks[0], mask[0:64, 0:64])
    assert np.array_equal(images[0], img[:, 0:64, 0:64])
    # georeferenced bounds = 640 m x 640 m
    assert t0.maxx - t0.minx == 640 and t0.maxy - t0.miny == 640
    assert t0.minx == synthetic_grid.transform.c and t0.maxy == synthetic_grid.transform.f


def test_nodata_pixels_never_labelled_building(cfg, synthetic_grid, synthetic_scene):
    img, valid, mask, frac = synthetic_scene
    mask = mask.copy()
    mask[:, 100:] = 1  # buildings under nodata
    _, masks, valids, _ = cut_tiles(img, valid, mask, frac, synthetic_grid, cfg, "synthetic")
    assert not np.any(masks.astype(bool) & ~valids)


def test_spatial_split_keeps_blocks_together(cfg):
    rows = [dict(tile_id=f"{r}_{c}", row=r * 64, col=c * 64, building_pixel_fraction=(r + c) / 30)
            for r in range(16) for c in range(16)]
    man = assign_spatial_split(pd.DataFrame(rows), cfg)
    assert man.groupby("block_id").split.nunique().max() == 1  # one split per block
    assert set(man.split) == {"train", "val", "test"}
    share = man.split.value_counts(normalize=True)
    assert share["train"] > 0.55 and share["test"] > 0.08
