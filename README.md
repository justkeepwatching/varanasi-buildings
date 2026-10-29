# Building Coverage Mapping & Change Detection | Varanasi

A reproducible pipeline that maps building-dominated pixels in **Sentinel-2** imagery at a **10 m** grid with a
**PyTorch U-Net**, trains it against **Google Open Buildings V3** footprints, evaluates it on spatially held-out
data, and compares **2020 vs 2025** building coverage for central Varanasi (Uttar Pradesh, India).

> **Measured results** are in [`outputs/reports/final_report.md`](outputs/reports/final_report.md) (generated
> from `outputs/metrics/*.json`). Status of every stage: [`PROJECT_STATUS.md`](PROJECT_STATUS.md).
> Interview study guide: [`INTERVIEW_PREPARATION.md`](INTERVIEW_PREPARATION.md).

---

## 1. What the project does and why it matters

Catastrophe (cat) risk models estimate losses from floods, earthquakes and storms by combining a **hazard**
(e.g. flood depth), **vulnerability** (how buildings are damaged) and **exposure** (where buildings are, and how
much they are worth). In fast-growing Indian cities, exposure inventories go out of date quickly. This project
shows a low-cost way to (a) map where buildings are from free satellite data and (b) flag where building
coverage appears to have grown, so exposure datasets can be prioritised for updating.

## 2. Workflow

```
study area (UTM 44N, 10 m grid)
   │
   ├─ Sentinel-2 L2A, Feb–May 2020 / 2023 / 2025 ──► cloud & shadow masking ──► median composites (6 bands)
   │       (Planetary Computer, or Google Earth Engine)
   ├─ Open Buildings V3 footprints ──► repair geometries ──► 2 m rasterization ──► roof-fraction ──► binary mask
   │
   ├─ 64×64 non-overlapping tiles (640 m) + manifest ──► block-wise spatial split (train / val / test)
   ├─ U-Net training (BCE + Dice, pos_weight) ──► best checkpoint by validation IoU
   ├─ evaluation on test blocks (threshold chosen on validation) + baselines (all-background, random forest)
   ├─ sliding-window inference over 2020, 2023, 2025 composites ──► probability & binary GeoTIFFs
   ├─ change detection 2020→2025 (gain / loss / persistent; areas in km²) ──► 21 km² claim investigation
   ├─ vectorization (GDAL polygonize) ──► object-level precision / recall (IoU ≥ 0.5, one-to-one)
   └─ maps, figures, final_report.md
```

Two nested areas are used (see `config/config.yaml` → `study_area`):

| Area | Size | Purpose |
|---|---|---|
| **Wide** | 15.36 × 15.36 km (1536 × 1536 px, 576 tiles) | train / validate / test the U-Net |
| **Core** | 5 × 5 km (500 × 500 px, 25.00 km² planar) | report building coverage and 2020→2025 change |

The core is an exact pixel window of the wide grid, so no resampling is ever needed between them.

## 3. Folder structure

```
config/config.yaml          every parameter (dates, bands, thresholds, split, model, training)
src/                        library code, one module per stage
  config.py                 load YAML, folder paths, seeding
  study_area.py             UTM grids, boundaries, planar + geodesic areas
  pc_download.py            Sentinel-2 via Microsoft Planetary Computer (no account)
  gee_download.py           Sentinel-2 + Open Buildings via Google Earth Engine
  imagery_preprocessing.py  GeoTIFF writing, compositing, grid checks, normalization
  labels.py                 Open Buildings download, cleaning, rasterization rule
  tiling.py                 64×64 tiles, manifest, block-wise spatial split
  dataset.py                PyTorch Dataset + augmentation
  model.py                  U-Net
  train.py                  training loop, loss, early stopping
  metrics.py                TP/FP/FN/TN, precision, recall, F1, IoU (global & per-tile)
  evaluate.py               threshold selection on val, test metrics, baselines
  inference.py              sliding-window georeferenced inference
  change_detection.py       change classes and area statistics
  vectorization.py          raster → polygons (OpenCV + GDAL polygonize + Shapely)
  object_evaluation.py      one-to-one polygon matching, object precision/recall
  visualization.py          maps and figures
  reporting.py              builds outputs/reports/final_report.md from metric files
scripts/run_pipeline.py     runs all stages (or selected ones)
scripts/verify_environment.py  prints versions, GPU, Earth Engine auth status
tests/                      pytest suite (synthetic fixtures only)
data/{raw,interim,processed,external}   downloaded and intermediate data (git-ignored)
models/                     unet_best.pt, band_stats.json
outputs/{maps,figures,metrics,vectors,rasters,reports}
```

## 4. Setup (Windows PowerShell)

Python 3.13 is used through the `py` launcher (the `python` on this machine's PATH is an MSYS2 build that is
not suitable for geospatial wheels).

```powershell
cd C:\Users\anime\Desktop\varanasi-buildings
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# PyTorch with CUDA 12.6 (works with NVIDIA driver >= 560). For CPU only use the default index instead.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python scripts\verify_environment.py
```

If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or call
`.\.venv\Scripts\python.exe` directly instead of activating.

GDAL is used through the copy bundled inside the `rasterio` and `pyogrio` wheels (no separate GDAL install).

## 5. Data sources and access

| Data | Access path A (default, no account) | Access path B (Earth Engine) |
|---|---|---|
| Sentinel-2 L2A surface reflectance | Microsoft Planetary Computer STAC `sentinel-2-l2a` | `COPERNICUS/S2_SR_HARMONIZED` |
| Cloud masking | Scene Classification Layer (SCL) | Cloud Score+ `cs_cdf ≥ 0.60` |
| Building footprints | Open Buildings V3 public CSV (`gs://open-buildings-data/v3/polygons_s2_level_6_gzip_no_header`) | `GOOGLE/Research/open-buildings/v3/polygons` |

Choose the path in `config/config.yaml` → `imagery.source` (`planetary_computer` or `gee`).
Open Buildings V3 is licensed CC BY-4.0 / ODbL; cite *Sirko et al., "Continental-scale building detection from
high resolution satellite imagery", 2021* and Google Research Open Buildings.

### Earth Engine authentication (path B only)

1. Register a noncommercial Cloud project: <https://code.earthengine.google.com/register>.
2. Put its id in `config/config.yaml` → `imagery.gee_project`.
3. Run once and complete the browser login:
   ```powershell
   .\.venv\Scripts\earthengine.exe authenticate
   ```
4. Set `imagery.source: gee` and run the pipeline with `--force` to re-download.
Credentials stay in `%USERPROFILE%\.config\earthengine\`, never in this project.

## 6. Running

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q                                   # automated tests
python scripts\run_pipeline.py                        # everything (downloads skipped if present)
python scripts\run_pipeline.py --stages train evaluate infer change vectors figures report
```

Stages: `boundaries imagery labels tiles train evaluate infer change vectors figures report`. A log is written to
`outputs/pipeline.log`.

## 7. Method details

### 7.1 Imagery
* **Bands:** B02 (blue), B03 (green), B04 (red), B08 (NIR) at native 10 m; B11, B12 (SWIR) at native 20 m,
  bilinearly resampled to 10 m. SWIR helps to separate concrete/roofs from bare soil.
* **Scene filter vs pixel mask:** scenes with ≥ 20% cloud are skipped entirely (coarse, cheap); inside the kept
  scenes, pixels flagged by SCL as cloud (8, 9), cirrus (10), shadow (3), saturated (1), snow (11) or nodata (0)
  are masked (fine-grained). Both are needed.
* **Offset:** since processing baseline 04.00 (25 Jan 2022) L2A digital numbers carry a +1000 offset;
  reflectance = (DN − 1000) / 10000 for those scenes and DN / 10000 before. Mixing them without correction would
  create fake "change" between 2020 and 2025.
* **Composite:** per-pixel median of the 12 clearest dates in the Feb–May window (same season every year);
  pixels with < 2 clear looks become nodata.
* **Grid:** EPSG:32644 (UTM 44N), 10 m. Sentinel-2 10 m bands in the same UTM zone already lie on the same 10 m
  lattice, so they are copied without resampling ("reproject only when necessary").

### 7.2 Labels
Polygons with confidence ≥ 0.70 are projected to UTM, invalid geometries are repaired (`buffer(0)`), then burned
on a 2 m sub-grid and averaged to 10 m with OpenCV `INTER_AREA`, giving the **fraction of each pixel covered by
roofs**. A pixel is labelled *building* when this fraction is ≥ 0.25. Footprints are imperfect: missing buildings,
false positives, merged rows of houses, and a date (~2021–2023) that differs from both 2020 and 2025.

### 7.3 Tiles and split
Non-overlapping 64×64 tiles (640 m × 640 m nominal), partial edge tiles discarded, tiles with < 95% valid
pixels discarded, empty tiles kept (they are real negatives). Each tile's bounds, valid fraction and building
share are in `data/processed/tiles/manifest.csv`, and every tile is also saved as a georeferenced GeoTIFF.
Tiles are grouped into 4×4-tile (2.56 km) **blocks**; whole blocks go to train/val/test (≈70/15/15), stratified by
building density with seed 42. Random tile-level splits would put neighbouring, near-identical tiles in train and
test and overstate accuracy.

### 7.4 U-Net
Encoder of 4 levels (32→64→128→256 channels, each two 3×3 conv + BatchNorm + ReLU, then 2×2 max-pool), a 512-
channel bottleneck at 4×4, a decoder with transposed convolutions and **skip connections**, and a 1×1 conv head
giving one **logit** per pixel. Probabilities = sigmoid(logit). Trained from scratch (no pretrained weights; the
6-band input does not match ImageNet RGB models).

### 7.5 Training
AdamW (lr 1e-3, weight decay 1e-4), batch 16, up to 80 epochs, ReduceLROnPlateau, early stopping (patience 15)
on validation IoU. Loss = BCE-with-logits (pos_weight = background/building ratio from the training set, capped at
10) + 0.5 × soft Dice; nodata pixels are excluded from the loss. Augmentation: random 90° rotations and flips.

### 7.6 Metrics
With TP, FP, FN, TN counted over valid test pixels:

* precision = TP / (TP + FP), recall = TP / (TP + FN)
* F1 = 2TP / (2TP + FP + FN), IoU = TP / (TP + FP + FN), and F1 = 2·IoU / (1 + IoU)

**Global** metrics sum counts over all test pixels first; **per-tile mean** metrics average tile scores
(tiles with undefined 0/0 scores excluded). Foreground metrics describe the building class; accuracy and
2-class mean IoU include the background and look better than they are under class imbalance. The probability
threshold is chosen on **validation** data only. Baselines: all-background, and a per-pixel random forest on bands
+ NDVI / NDBI / NDWI.

### 7.7 Inference
64×64 windows with stride 32 over reflect-padded composites; overlapping probabilities are averaged with a
centre-weighted (Hann) window to avoid seams; nodata stays nodata. Outputs are GeoTIFFs on the analysis grid.
At 10 m the model cannot resolve individual small houses or roof edges.

### 7.8 Change detection
Post-classification comparison: stable non-building, potential gain, potential loss, persistent building.
Areas = pixel count × 100 m² (planar UTM), with a geodesic correction factor also reported. A "high-confidence"
variant requires the probability to cross threshold ± 0.2. Gains are classification differences, not confirmed
construction.

### 7.9 Vectorization and object metrics
Masks → OpenCV connected components (speckle < 2 px removed) → `rasterio.features.shapes` (GDAL polygonize,
8-connectivity) → Shapely `make_valid`. At 10 m, polygons are **contiguous built-up regions**, not buildings.
Objects are matched one-to-one greedily by polygon IoU ≥ 0.5; TP / FP / FN and merges / splits are reported.
A secondary, footprint-level detection rate (many-to-one allowed) is reported separately.

## 8. Outputs

| Path | Content |
|---|---|
| `outputs/maps/01_study_area.png` … `09_vectorized_test_block.png` | maps |
| `outputs/figures/` | example imagery, training curves, test examples, metric summary |
| `outputs/metrics/metrics.json`, `test_metrics_summary.csv` | pixel-level test metrics |
| `outputs/metrics/object_metrics.json` | object-level metrics |
| `outputs/metrics/change_statistics.json`, `resume_21km2_investigation.csv` | change areas, 21 km² check |
| `outputs/rasters/*.tif` | probability, binary and change GeoTIFFs (open in QGIS) |
| `outputs/vectors/*.gpkg` | vectorized regions (open in QGIS) |
| `outputs/reports/final_report.md` | the full measured report |
| `models/unet_best.pt` | trained checkpoint |

## 9. Troubleshooting

* **`rasterio` import error about DLLs:** make sure you are using `.venv\Scripts\python.exe`, not MSYS2 `python`.
* **Planetary Computer read timeouts:** re-run the `imagery` stage; failed scenes are listed in
  `data/interim/s2_<year>_meta.json` → `items_failed`.
* **CUDA out of memory (4 GB GPU):** lower `training.batch_size` in the config.
* **Earth Engine `not authenticated`:** run `earthengine authenticate` and set `imagery.gee_project`.

## 10. Limitations and ethics

* Labels are model-derived (Open Buildings) and a single snapshot; there is no independent 2020/2025 reference.
* 10 m resolution: many houses are smaller than a pixel; dense old-city fabric merges into large regions.
* Year-to-year composite differences (haze, sensor, season) cause spurious change.
* Building maps can be misused (e.g. for surveillance or eviction targeting of informal settlements); outputs
  here are aggregate coverage for risk-exposure modelling and are not suitable for decisions about individual
  properties.
