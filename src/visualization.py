"""Matplotlib figures for the report. Every figure is drawn from files produced
by the pipeline; nothing here invents data.

Colour roles (validated categorical palette, first three slots):
blue = train / persistent building, orange = validation / gain,
aqua = test / loss. Neutral grey = background / stable. Every map has a legend.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL, INK, MUTED = "#e4e3df", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 110, "savefig.dpi": 160,
    "axes.spines.top": False, "axes.spines.right": False,
})


def rgb(image: np.ndarray, nodata: float = -9999.0, lo=2, hi=98) -> np.ndarray:
    """True-colour (B04, B03, B02) display image with a 2-98 percentile stretch."""
    arr = image[[2, 1, 0]].astype(np.float32).copy()
    bad = np.any((arr == nodata) | ~np.isfinite(arr), axis=0)
    out = np.zeros(arr.shape[1:] + (3,), np.float32)
    for i in range(3):
        band = arr[i]
        p1, p2 = np.percentile(band[~bad], [lo, hi]) if (~bad).any() else (0, 1)
        out[..., i] = np.clip((band - p1) / (p2 - p1 + 1e-9), 0, 1) ** 0.9
    out[bad] = 1.0
    return out


def _extent_km(grid):
    left, bottom, right, top = grid.bounds
    return [left / 1000, right / 1000, bottom / 1000, top / 1000]


def _km_axes(ax):
    ax.set_xlabel("UTM 44N easting (km)")
    ax.set_ylabel("northing (km)")


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def study_area_map(image, wide, core, path, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb(image), extent=_extent_km(wide))
    l, b, r, t = (v / 1000 for v in core.bounds)
    ax.add_patch(Rectangle((l, b), r - l, t - b, fill=False, ec=ORANGE, lw=2))
    l, b, r, t = (v / 1000 for v in wide.bounds)
    ax.add_patch(Rectangle((l, b), r - l, t - b, fill=False, ec=BLUE, lw=2))
    ax.legend(handles=[Patch(fc="none", ec=BLUE, lw=2, label="Wide area (model training / testing)"),
                       Patch(fc="none", ec=ORANGE, lw=2, label="Core area (coverage & change reporting)")],
              loc="lower left", framealpha=0.9)
    ax.set_title(title)
    _km_axes(ax)
    _save(fig, path)


def imagery_panels(images: dict, grid, path, title):
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for ax, (name, img) in zip(np.atleast_1d(axes), images.items()):
        ax.imshow(rgb(img), extent=_extent_km(grid))
        ax.set_title(f"Sentinel-2 median composite {name} (true colour)")
        _km_axes(ax)
    fig.suptitle(title)
    _save(fig, path)


def label_map(image, frac, mask, grid, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    ext = _extent_km(grid)
    axes[0].imshow(rgb(image), extent=ext)
    axes[0].set_title("Sentinel-2 2023 composite")
    im = axes[1].imshow(frac, cmap="Blues", vmin=0, vmax=1, extent=ext)
    axes[1].set_title("Roof coverage fraction per 10 m pixel")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="fraction of pixel covered by footprints")
    axes[2].imshow(mask, cmap=ListedColormap([NEUTRAL, BLUE]), vmin=0, vmax=1, extent=ext, interpolation="nearest")
    axes[2].set_title("Binary reference mask")
    axes[2].legend(handles=[Patch(fc=BLUE, label="building"), Patch(fc=NEUTRAL, label="background")], loc="lower left")
    for ax in axes:
        _km_axes(ax)
    fig.suptitle(title)
    _save(fig, path)


def tile_split_map(image, manifest, grid, path, title, all_origins=None, tile=64):
    colors = {"train": BLUE, "val": ORANGE, "test": AQUA}
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb(image), extent=_extent_km(grid))
    px = grid.pixel_size / 1000
    kept = set(zip(manifest.row, manifest.col))
    for r, c in (all_origins or []):
        if (r, c) not in kept:
            x0, y1 = grid.bounds[0] / 1000 + c * px, grid.bounds[3] / 1000 - r * px
            ax.add_patch(Rectangle((x0, y1 - tile * px), tile * px, tile * px, fc="none", ec="black", lw=0.6, ls=":"))
    for rec in manifest.itertuples():
        ax.add_patch(Rectangle((rec.minx / 1000, rec.miny / 1000), (rec.maxx - rec.minx) / 1000,
                               (rec.maxy - rec.miny) / 1000, fc=colors[rec.split], alpha=0.35,
                               ec="white", lw=0.6))
    counts = manifest.split.value_counts()
    handles = [Patch(fc=colors[s], alpha=0.6, label=f"{s}: {counts.get(s, 0)} tiles") for s in ("train", "val", "test")]
    if all_origins and len(all_origins) > len(kept):
        handles.append(Patch(fc="none", ec="black", ls=":", label="discarded (nodata)"))
    ax.legend(handles=handles, loc="lower left", framealpha=0.9)
    ax.set_title(title)
    _km_axes(ax)
    _save(fig, path)


def training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.epoch, history.train_loss, color=BLUE, lw=2, label="training loss")
    axes[0].plot(history.epoch, history.val_loss, color=ORANGE, lw=2, label="validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE + 0.5 Dice loss")
    axes[0].legend()
    axes[0].set_title("Loss curves")
    axes[1].plot(history.epoch, history.val_iou, color=BLUE, lw=2, label="validation IoU")
    axes[1].plot(history.epoch, history.val_f1, color=ORANGE, lw=2, label="validation F1")
    best = history.val_iou.idxmax()
    axes[1].scatter([history.epoch[best]], [history.val_iou[best]], s=60, color=BLUE, ec="white", zorder=3)
    axes[1].annotate(f"best epoch {history.epoch[best]}\nIoU {history.val_iou[best]:.3f}",
                     (history.epoch[best], history.val_iou[best]), xytext=(10, -30),
                     textcoords="offset points", color=INK)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score (threshold 0.5)")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].set_title("Validation metrics")
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, path)


def test_examples(images, masks, probs, preds, valids, tile_ids, path, threshold):
    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.8 * n))
    axes = np.atleast_2d(axes)
    titles = ["Sentinel-2 (true colour)", "Reference (Open Buildings)", "U-Net probability",
              f"U-Net mask (p >= {threshold:.2f})"]
    for i in range(n):
        inval = ~valids[i]
        axes[i, 0].imshow(rgb(images[i]))
        axes[i, 1].imshow(np.ma.masked_where(inval, masks[i]), cmap=ListedColormap([NEUTRAL, BLUE]), vmin=0, vmax=1, interpolation="nearest")
        im = axes[i, 2].imshow(np.ma.masked_where(inval, probs[i]), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 3].imshow(np.ma.masked_where(inval, preds[i]), cmap=ListedColormap([NEUTRAL, BLUE]), vmin=0, vmax=1, interpolation="nearest")
        axes[i, 0].set_ylabel(tile_ids[i], fontsize=8)
        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(titles[j])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="U-Net probability")
    fig.suptitle("Held-out TEST tiles (640 m x 640 m each)")
    _save(fig, path)


def coverage_pair(m_early, m_late, grid, labels, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    cmap = ListedColormap([NEUTRAL, BLUE])
    for ax, m, lab in zip(axes, (m_early, m_late), labels):
        ax.imshow(np.ma.masked_equal(m, 255), cmap=cmap, vmin=0, vmax=1, extent=_extent_km(grid), interpolation="nearest")
        ax.set_title(lab)
        _km_axes(ax)
    axes[1].legend(handles=[Patch(fc=BLUE, label="classified building"), Patch(fc=NEUTRAL, label="non-building")],
                   loc="lower left")
    fig.suptitle(title)
    _save(fig, path)


def change_map(classes, grid, path, title, stats):
    cmap = ListedColormap([NEUTRAL, ORANGE, AQUA, BLUE])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(np.ma.masked_equal(classes, 255), cmap=cmap, vmin=0, vmax=3, extent=_extent_km(grid), interpolation="nearest")
    ax.legend(handles=[
        Patch(fc=BLUE, label=f"persistent building ({stats['persistent_km2']:.2f} km²)"),
        Patch(fc=ORANGE, label=f"potential gain ({stats['gross_gain_km2']:.2f} km²)"),
        Patch(fc=AQUA, label=f"potential loss ({stats['gross_loss_km2']:.2f} km²)"),
        Patch(fc=NEUTRAL, label="stable non-building"),
    ], loc="lower left", framealpha=0.95)
    ax.set_title(title)
    _km_axes(ax)
    _save(fig, path)


def vector_overlay(image_rgb_extent, pred_gdf, ref_gdf, path, title, extent_m):
    fig, ax = plt.subplots(figsize=(8, 8))
    if image_rgb_extent is not None:
        ax.imshow(image_rgb_extent[0], extent=[v / 1 for v in image_rgb_extent[1]])
    ref_gdf.boundary.plot(ax=ax, color=AQUA, lw=1.2)
    pred_gdf.boundary.plot(ax=ax, color=ORANGE, lw=1.0)
    ax.set_xlim(extent_m[0], extent_m[1])
    ax.set_ylim(extent_m[2], extent_m[3])
    ax.legend(handles=[Patch(fc="none", ec=AQUA, label=f"reference regions ({len(ref_gdf)})"),
                       Patch(fc="none", ec=ORANGE, label=f"predicted regions ({len(pred_gdf)})")], loc="lower left")
    ax.set_title(title)
    ax.set_xlabel("UTM 44N easting (m)")
    ax.set_ylabel("northing (m)")
    _save(fig, path)


def metric_bars(rows: dict, path, title):
    """Grouped horizontal bars: one group per metric, one bar per model."""
    metrics = ["precision", "recall", "f1", "iou"]
    models = list(rows)
    colors = [BLUE, ORANGE, AQUA][: len(models)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    h = 0.8 / len(models)
    for j, (m, col) in enumerate(zip(models, colors)):
        raw = [rows[m][k] for k in metrics]
        vals = [0.0 if v != v else v for v in raw]  # NaN (0/0, e.g. precision of all-background) drawn as 0
        y = np.arange(len(metrics)) + j * h
        ax.barh(y, vals, height=h * 0.9, color=col, label=m)
        for yy, v, r in zip(y, vals, raw):
            ax.text(v + 0.01, yy, "undefined" if r != r else f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(np.arange(len(metrics)) + h * (len(models) - 1) / 2)
    ax.set_yticklabels([k.upper() if k in ("f1", "iou") else k for k in metrics])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("score on held-out test pixels (global counts)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(title)
    _save(fig, path)
