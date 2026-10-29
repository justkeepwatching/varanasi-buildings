import numpy as np
import pytest
import torch

from src.dataset import TileDataset
from src.model import UNet, load_checkpoint, save_checkpoint
from src.train import MaskedBCEDiceLoss


def test_unet_output_shape():
    m = UNet(in_channels=6, base_channels=8, depth=4)
    y = m(torch.randn(2, 6, 64, 64))
    assert y.shape == (2, 1, 64, 64)


def test_unet_rejects_bad_size():
    with pytest.raises(AssertionError):
        UNet(6, 8, 4)(torch.randn(1, 6, 60, 60))


def test_checkpoint_roundtrip(tmp_path, cfg):
    cfg = {**cfg, "model": {**cfg["model"], "base_channels": 8}}
    m = UNet(6, 8, cfg["model"]["depth"], cfg["model"]["dropout"]).eval()
    x = torch.randn(1, 6, 64, 64)
    save_checkpoint(tmp_path / "m.pt", m, cfg, {"epoch": 1})
    m2, ck = load_checkpoint(tmp_path / "m.pt")
    assert ck["epoch"] == 1
    assert torch.allclose(m(x), m2(x))


def test_loss_ignores_invalid_pixels():
    loss = MaskedBCEDiceLoss(pos_weight=1.0, dice_weight=0.5)
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    valid = torch.ones(1, 1, 4, 4)
    base = loss(logits, target, valid)
    target2 = target.clone()
    target2[..., 0, 0] = 1
    valid2 = valid.clone()
    valid2[..., 0, 0] = 0  # the changed pixel is nodata -> must not change the loss
    assert torch.isclose(base, loss(logits, target2, valid2), atol=0.05)


def test_augmentation_keeps_image_mask_aligned():
    img = np.zeros((1, 6, 64, 64), np.float32)
    img[0, :, 5, 9] = 10.0
    mask = np.zeros((1, 64, 64), np.uint8)
    mask[0, 5, 9] = 1
    ds = TileDataset(img, mask, np.ones((1, 64, 64), bool), {"mean": [0] * 6, "std": [1] * 6}, augment=True)
    for _ in range(10):
        x, y, v = ds[0]
        assert torch.equal(x[0] == 10.0, y[0] == 1)
