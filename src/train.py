"""Train the U-Net on the training tiles; select the checkpoint on validation tiles.

Loss = BCEWithLogits(pos_weight) + dice_weight * soft Dice, both over valid
pixels only.
* BCE with pos_weight: per-pixel classification loss; pos_weight = (#background
  / #building) pixels in the training set (capped at 10) up-weights the rarer
  building class so the model is not rewarded for predicting background everywhere.
* Soft Dice: directly optimizes overlap (closely related to F1/IoU) and is
  insensitive to the large number of easy background pixels.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import PATHS, set_seed
from .dataset import TileDataset
from .imagery_preprocessing import compute_band_stats
from .metrics import global_scores
from .model import build_model, save_checkpoint

log = logging.getLogger(__name__)


def load_tiles():
    """Load the tile arrays and manifest produced by the tiling stage."""
    data = np.load(PATHS.tiles / "tiles.npz")
    manifest = pd.read_csv(PATHS.tiles / "manifest.csv")
    assert len(manifest) == len(data["images"]), "manifest and tile arrays disagree"
    return data["images"], data["masks"], data["valids"], manifest


def subset(arrays, manifest, split):
    idx = np.flatnonzero(manifest.split.values == split)
    return [a[idx] for a in arrays], manifest.iloc[idx].reset_index(drop=True)


class MaskedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float, dice_weight: float):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.dice_weight = dice_weight

    def forward(self, logits, target, valid):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        n_valid = valid.sum().clamp(min=1.0)
        bce = (bce * valid).sum() / n_valid
        prob = torch.sigmoid(logits) * valid
        t = target * valid
        dice = 1 - (2 * (prob * t).sum() + 1.0) / (prob.sum() + t.sum() + 1.0)
        return bce + self.dice_weight * dice


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _, _ in loader:
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy()[:, 0])
    return np.concatenate(out)


def train(cfg: dict) -> dict:
    tcfg = cfg["training"]
    set_seed(cfg["project"]["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = get_device()

    images, masks, valids, manifest = load_tiles()
    (tr_x, tr_y, tr_v), tr_m = subset((images, masks, valids), manifest, "train")
    (va_x, va_y, va_v), va_m = subset((images, masks, valids), manifest, "val")
    if len(tr_x) < 20 or len(va_x) < 3:
        raise RuntimeError(f"Too few tiles to train (train={len(tr_x)}, val={len(va_x)}).")

    # Normalization statistics from TRAINING tiles only (no leakage).
    stats = compute_band_stats(tr_x, tr_v)
    (PATHS.models / "band_stats.json").write_text(json.dumps(stats, indent=2))

    pos = float(tr_y[tr_v].sum())
    neg = float(tr_v.sum() - pos)
    pos_weight = float(min(neg / max(pos, 1.0), 10.0))
    log.info("train tiles=%d val tiles=%d building pixel share=%.3f pos_weight=%.2f",
             len(tr_x), len(va_x), pos / (pos + neg), pos_weight)

    g = torch.Generator().manual_seed(cfg["project"]["seed"])
    train_dl = DataLoader(TileDataset(tr_x, tr_y, tr_v, stats, tcfg["augment"], cfg["project"]["seed"]),
                          batch_size=tcfg["batch_size"], shuffle=True, generator=g, num_workers=tcfg["num_workers"])
    val_ds = TileDataset(va_x, va_y, va_v, stats)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    loss_fn = MaskedBCEDiceLoss(pos_weight, tcfg["dice_weight"]).to(device)

    best_iou, best_epoch, bad_epochs, history = -1.0, -1, 0, []
    ckpt_path = PATHS.models / "unet_best.pt"
    t0 = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for x, y, v in train_dl:
            x, y, v = x.to(device), y.to(device), v.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y, v)
            if not torch.isfinite(loss):  # fail loudly instead of skipping the batch
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
            n += len(x)

        # validation loss + IoU at the default 0.5 threshold
        model.eval()
        va_loss, m = 0.0, 0
        with torch.no_grad():
            for x, y, v in val_dl:
                x, y, v = x.to(device), y.to(device), v.to(device)
                va_loss += loss_fn(model(x), y, v).item() * len(x)
                m += len(x)
        probs = predict_probs(model, val_dl, device)
        val = global_scores(probs >= cfg["evaluation"]["default_threshold"], va_y, va_v)
        sched.step(val["iou"])
        rec = dict(epoch=epoch, train_loss=tr_loss / n, val_loss=va_loss / m, val_iou=val["iou"],
                   val_f1=val["f1"], lr=opt.param_groups[0]["lr"])
        history.append(rec)
        log.info("epoch %3d  train %.4f  val %.4f  val IoU %.4f  F1 %.4f", epoch,
                 rec["train_loss"], rec["val_loss"], rec["val_iou"], rec["val_f1"])

        if val["iou"] > best_iou:
            best_iou, best_epoch, bad_epochs = val["iou"], epoch, 0
            save_checkpoint(ckpt_path, model, cfg, {"epoch": epoch, "val_iou": best_iou, "band_stats": stats})
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["early_stopping_patience"]:
                log.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    pd.DataFrame(history).to_csv(PATHS.metrics / "training_history.csv", index=False)
    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "n_parameters": n_params,
        "train_tiles": len(tr_x),
        "val_tiles": len(va_x),
        "pos_weight": pos_weight,
        "train_building_pixel_share": pos / (pos + neg),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_iou_at_0.5": best_iou,
        "training_seconds": round(time.time() - t0, 1),
        "selection_rule": "highest validation IoU (global, threshold 0.5)",
    }
    (PATHS.metrics / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
