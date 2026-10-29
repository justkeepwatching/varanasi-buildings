"""Shared SYNTHETIC test fixtures.

Everything here is artificial data that exercises the software. It is never
written to outputs/ and must never be presented as Varanasi results.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.study_area import GridSpec  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def synthetic_grid():
    """A 128 x 192 px synthetic 10 m grid in UTM 44N (2 x 3 tiles)."""
    return GridSpec("EPSG:32644", Affine(10, 0, 700000, 0, -10, 2800000), width=192, height=128)


@pytest.fixture
def synthetic_scene(synthetic_grid):
    """Synthetic 6-band image, a square 'building' block and a nodata strip."""
    rng = np.random.default_rng(0)
    h, w = synthetic_grid.height, synthetic_grid.width
    img = rng.uniform(0.02, 0.3, (6, h, w)).astype(np.float32)
    mask = np.zeros((h, w), np.uint8)
    mask[10:30, 20:50] = 1
    valid = np.ones((h, w), bool)
    valid[:, 130:] = False  # last tile column mostly nodata
    img[:, ~valid] = -9999.0
    frac = mask.astype(np.float32)
    return img, valid, mask, frac
