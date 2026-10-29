"""PyTorch Dataset that serves normalized image tiles with their masks."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .imagery_preprocessing import normalize


class TileDataset(Dataset):
    """Returns (image, mask, valid) tensors for one 64 x 64 tile.

    image: (C, 64, 64) float32, z-scored with TRAINING-set band statistics
    mask:  (1, 64, 64) float32, 1 = building, 0 = background
    valid: (1, 64, 64) float32, 0 where the imagery had no data (ignored by the loss)

    Augmentation (training only): random 90-degree rotations and flips. These
    are applied identically to image, mask and valid, so alignment is kept.
    They are physically reasonable for a nadir satellite view; we do not use
    colour or scale augmentation, which would change the meaning of 10 m pixels.
    """

    def __init__(self, images, masks, valids, stats: dict, augment: bool = False, seed: int = 0):
        assert len(images) == len(masks) == len(valids)
        self.x = np.stack([normalize(img, stats, v) for img, v in zip(images, valids)])
        self.y = masks.astype(np.float32)[:, None]
        self.v = valids.astype(np.float32)[:, None]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        x, y, v = self.x[i], self.y[i], self.v[i]
        if self.augment:
            k = int(self.rng.integers(4))
            x, y, v = (np.rot90(a, k, axes=(1, 2)) for a in (x, y, v))
            if self.rng.random() < 0.5:
                x, y, v = (a[:, :, ::-1] for a in (x, y, v))
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (x, y, v))
