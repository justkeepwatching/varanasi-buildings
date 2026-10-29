# PROJECT STATUS

_Last updated: 2026-10-29. Evidence: `outputs/pipeline.log`, `outputs/metrics/*.json`, pytest run (32 passed)._

## Legend
✅ implemented **and executed** on real data · 🧪 implemented and tested on synthetic data only ·
⏸ implemented, **not executed** · ⛔ blocked by an external requirement

## Stages

| Stage | Status | Evidence / notes |
|---|---|---|
| Environment (.venv, Python 3.13.5, torch 2.14.1+cu126, GDAL 3.12.2 via rasterio) | ✅ | `scripts/verify_environment.py`; CUDA on RTX 3050 4 GB |
| Study area (wide 235.93 km², core 25.00 km² planar / 24.996 km² geodesic) | ✅ | `outputs/metrics/study_area.json`, `data/external/study_area_*.geojson` |
| Sentinel-2 via Planetary Computer, 2020 / 2023 / 2025 (Feb–May, 12 dates each) | ✅ | `data/interim/s2_<year>_meta.json`: 0 failed items, ~100% valid pixels |
| Sentinel-2 via Google Earth Engine (`src/gee_download.py`) | ⏸ ⛔ | needs `earthengine authenticate` + Cloud project id (not available) |
| Open Buildings V3 via public CSV (cell 398f, 462,166 footprints ≥ 0.70 conf.) | ✅ | `data/raw/open_buildings_v3_meta.json` (retrieved 2026-10-29) |
| Open Buildings V3 via Earth Engine | ⏸ ⛔ | same Earth Engine requirement |
| Label rasterization (2 m supersample, coverage ≥ 0.25) | ✅ | `outputs/metrics/label_statistics.json`; rule unit-tested |
| Tiling (576 tiles, 0 discarded) + block split 416 / 80 / 80 | ✅ | `data/processed/tiles/manifest.csv`, `outputs/metrics/tiling_summary.json` |
| U-Net training (46 epochs, best epoch 31, 63 s on GPU) | ✅ | `models/unet_best.pt`, `outputs/metrics/training_history.csv` |
| Test evaluation + baselines | ✅ | `outputs/metrics/metrics.json`, `test_metrics_summary.csv` |
| Inference 2020 / 2023 / 2025 (wide area) | ✅ | `outputs/rasters/unet_<year>_wide_{probability,binary}.tif` |
| Change detection 2020 → 2025 | ✅ (unvalidated) | `outputs/metrics/change_statistics.json`; no independent change reference exists |
| Vectorization + object metrics | ✅ | `outputs/vectors/*.gpkg`, `outputs/metrics/object_metrics.json` |
| Maps / figures | ✅ | `outputs/maps/`, `outputs/figures/` (all from real outputs) |
| Final report | ✅ | `outputs/reports/final_report.md` (auto-generated from metric files) |
| README / INTERVIEW_PREPARATION | ✅ | project root |

## Measured headline results (from the metric files)

| Quantity | Measured | Resume claim |
|---|---|---|
| Tiles | 576 used (wide); a 25 km² area holds 49 | ≈60 |
| Pixel IoU (test, global) | **0.615** | 0.69 |
| Pixel F1 (test, global) | **0.762** | 0.82 |
| Random forest baseline IoU / F1 | 0.545 / 0.706 | — |
| Core built-up 2020 → 2025 | 11.07 → 12.13 km² | — |
| Core gross gain / net change | 2.69 / 1.06 km² | 21 km² "growth" |
| Wide (236 km²) gross gain / net | 21.60 / 11.79 km² | (numerical coincidence, see report §8) |
| Object precision / recall (region, IoU ≥ 0.5) | **0.386 / 0.290** | 0.85 / 0.79 |
| Footprint detection rate | 0.815 | — |

## Tests (`python -m pytest -q`): 32 passed, 0 failed
`test_tiling` (4), `test_metrics` (6), `test_change_detection` (4), `test_geospatial_alignment` (5),
`test_vectorization_objects` (7), `test_model` (5), `test_end_to_end_synthetic` (1 — synthetic data only).

## Bugs found and fixed during execution
* `TileDataset.__getitem__` returned a generator → DataLoader failed; now returns a tuple.
* Planetary Computer lists 2023 acquisitions twice (baseline 04.00 and a 2024 reprocess 05.10); now only the newest
  processing per (date, MGRS tile) is used. 2020 and 2025 had no duplicates, so their composites are unaffected.
* Transient HTTP 503 errors from Planetary Computer were recovered by GDAL retries (0 failed items).

## Blocked / open
* ⛔ **Earth Engine path**: run `.\.venv\Scripts\earthengine.exe authenticate`, set `imagery.gee_project` and
  `imagery.source: gee` in `config/config.yaml`, then
  `python scripts\run_pipeline.py --stages imagery labels --force` and re-run later stages.
* Change maps have **no validation data**; an accuracy assessment with independent high-resolution samples is needed
  before calling gains "construction".
* Single training run (seed 42). No spatial cross-validation or confidence intervals yet.
* Git repository not initialised (not requested).
