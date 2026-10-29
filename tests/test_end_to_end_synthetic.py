"""SYNTHETIC end-to-end smoke test: tiles -> U-Net training step -> inference ->
change -> vectorization -> object metrics. Checks the software path only; the
numbers have no meaning for Varanasi."""
import numpy as np
import torch

from src.change_detection import change_classes, change_statistics
from src.inference import predict_raster
from src.metrics import global_scores
from src.model import UNet
from src.object_evaluation import object_scores
from src.tiling import cut_tiles
from src.train import MaskedBCEDiceLoss
from src.vectorization import mask_to_polygons


def test_synthetic_pipeline(cfg, synthetic_grid, synthetic_scene):
    torch.manual_seed(0)
    img, valid, mask, frac = synthetic_scene
    img = img.copy()
    img[3, mask == 1] = 0.9  # make "buildings" bright in band 4 so they are learnable
    images, masks, valids, man = cut_tiles(img, valid, mask, frac, synthetic_grid, cfg, "synthetic")
    stats = {"mean": [0.15] * 6, "std": [0.1] * 6}
    model = UNet(6, 8, 4)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    loss_fn = MaskedBCEDiceLoss(5.0, 0.5)
    x = torch.from_numpy(((np.where(valids[:, None], images, 0.15) - 0.15) / 0.1).astype(np.float32))
    y = torch.from_numpy(masks[:, None].astype(np.float32))
    v = torch.from_numpy(valids[:, None].astype(np.float32))
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(model(x), y, v)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    # 30 steps are too few for BatchNorm running statistics to converge; settle
    # them with gradient-free forward passes (real training runs thousands of steps).
    with torch.no_grad():
        for _ in range(60):
            model(x)

    prob = predict_raster(model, img, valid, stats, "cpu", 64, 32)
    assert prob.shape == mask.shape and np.isnan(prob[~valid]).all()
    pred = (np.nan_to_num(prob) >= 0.5).astype(np.uint8)
    s = global_scores([pred], [mask], [valid])
    assert s["iou"] > 0.5  # learnable synthetic pattern

    ch = change_classes(np.zeros_like(mask), pred, valid)
    st = change_statistics(ch, 10)
    assert np.isclose(st["gross_gain_km2"], pred[valid].sum() * 100 / 1e6)

    pg = mask_to_polygons(pred, synthetic_grid.transform, synthetic_grid.crs)
    rg = mask_to_polygons(mask, synthetic_grid.transform, synthetic_grid.crs)
    o = object_scores(pg, rg, 0.5)
    assert o["n_ref"] == 1 and o["tp"] <= 1
