"""Sliding-window inference over a whole composite, preserving georeferencing.

The model was trained on 64 x 64 tiles. To map a large raster we slide a
64 x 64 window with stride 32 (50% overlap) and average the overlapping
probabilities with a smooth weight that favours window centres. Predictions
near a window's border see less context and are less reliable; overlapping
and centre-weighting removes the visible "seams" that a non-overlapping
tiling would leave. The image is reflect-padded so border pixels also get
full windows. Output pixels where the imagery is nodata are set to nodata.

Caveat at 10 m: a typical Varanasi house (~50-150 m^2) is smaller than 1-2
pixels, so the model maps "building-dominated pixels" / built-up texture,
not individual roofs or precise roof edges.
"""
from __future__ import annotations

import numpy as np
import rasterio
import torch

from .imagery_preprocessing import normalize, valid_mask_from_image, write_geotiff
from .study_area import GridSpec

PROB_NODATA = -1.0
MASK_NODATA = 255


def _window_weight(size: int) -> np.ndarray:
    """2-D weight peaking at the window centre (never exactly zero)."""
    w1 = np.hanning(size + 2)[1:-1]
    return np.outer(w1, w1).astype(np.float32) + 1e-3


@torch.no_grad()
def predict_raster(model, image: np.ndarray, valid: np.ndarray, stats: dict, device,
                   window: int = 64, stride: int = 32, batch: int = 64) -> np.ndarray:
    _, h, w = image.shape
    x = normalize(image, stats, valid)
    pad = window
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    H, W = x.shape[1:]
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    wt = _window_weight(window)
    origins = [(r, c) for r in range(0, H - window + 1, stride) for c in range(0, W - window + 1, stride)]
    model.eval()
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        xb = torch.from_numpy(np.stack([x[:, r:r + window, c:c + window] for r, c in chunk])).to(device)
        pb = torch.sigmoid(model(xb)).cpu().numpy()[:, 0]
        for (r, c), p in zip(chunk, pb):
            acc[r:r + window, c:c + window] += p * wt
            wsum[r:r + window, c:c + window] += wt
    prob = (acc / np.maximum(wsum, 1e-6))[pad:pad + h, pad:pad + w]
    prob[~valid] = np.nan
    return prob


def run_inference(model, stats, image_path, grid: GridSpec, threshold: float, device, cfg, out_prefix):
    """Predict one composite; write probability + binary GeoTIFFs; return arrays."""
    with rasterio.open(image_path) as src:
        img = src.read().astype(np.float32)
        nodata = src.nodata
    valid = valid_mask_from_image(img, nodata)
    prob = predict_raster(model, img, valid, stats, device, cfg["inference"]["window"], cfg["inference"]["stride"])
    binary = np.where(valid, (prob >= threshold).astype(np.uint8), MASK_NODATA).astype(np.uint8)
    prob_out = np.where(valid, prob, PROB_NODATA).astype(np.float32)
    tags = {"model": "U-Net", "threshold": threshold, "source_image": image_path.name}
    write_geotiff(out_prefix.with_name(out_prefix.name + "_probability.tif"), prob_out, grid, PROB_NODATA,
                  ["building_probability"], tags)
    write_geotiff(out_prefix.with_name(out_prefix.name + "_binary.tif"), binary, grid, MASK_NODATA,
                  ["building_mask"], tags)
    return prob, binary, valid
