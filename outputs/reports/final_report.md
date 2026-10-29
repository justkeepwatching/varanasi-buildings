# Building Coverage Mapping & Change Detection | Varanasi — Final Report

_Generated 2026-10-29 19:56 by `scripts/run_pipeline.py --stages report`. All numbers below are read from files in `outputs/metrics/`; nothing is typed in by hand._

## 1. Study area

- **Core reporting area:** 500×500 px at 10 m = 25.0 km² planar (UTM 44N), 24.9956 km² geodesic. Lon/lat bounds [82.949808, 25.29207, 83.000187, 25.337862].
- **Wide training area:** 1536×1536 px = 235.9296 km² planar, 235.8879 km² geodesic.
- A 64×64 tile covers 640 m × 640 m = 0.4096 km². The core holds **49 full tiles**; the wide area holds 576.

## 2. Data

- **Sentinel-2 2020:** Microsoft Planetary Computer sentinel-2-l2a; window 2020-02-01 → 2020-05-31; 12 acquisition dates (2020-02-05, 2020-02-10, 2020-02-15, 2020-02-25, 2020-03-06, 2020-03-16, 2020-03-31, 2020-04-05, 2020-04-10, 2020-05-05, 2020-05-15, 2020-05-25); scene filter < 20% cloud; pixel mask: SCL classes [0, 1, 3, 8, 9, 10, 11]; median composite; valid pixels 100.0%.
- **Sentinel-2 2023:** Microsoft Planetary Computer sentinel-2-l2a; window 2023-02-01 → 2023-05-31; 12 acquisition dates (2023-02-04, 2023-02-09, 2023-02-14, 2023-02-19, 2023-03-01, 2023-03-06, 2023-03-11, 2023-04-05, 2023-04-15, 2023-04-20, 2023-05-20, 2023-05-30); scene filter < 20% cloud; pixel mask: SCL classes [0, 1, 3, 8, 9, 10, 11]; median composite; valid pixels 100.0%.
- **Sentinel-2 2025:** Microsoft Planetary Computer sentinel-2-l2a; window 2025-02-01 → 2025-05-31; 12 acquisition dates (2025-02-03, 2025-02-13, 2025-02-18, 2025-02-23, 2025-03-10, 2025-03-15, 2025-03-25, 2025-03-30, 2025-04-01, 2025-04-19, 2025-04-21, 2025-04-24); scene filter < 20% cloud; pixel mask: SCL classes [0, 1, 3, 8, 9, 10, 11]; median composite; valid pixels 100.0%.
- **Open Buildings V3** (confidence ≥ 0.7): 462,166 footprints in the wide area, median footprint 57 m²; 71.4% are smaller than one 10 m pixel. Total roof area 39.63 km²; rasterized mask (coverage ≥ 0.25) 56.85 km² (24.1% of pixels).

## 3. Tiles and spatial split

- 576 candidate full tiles → **576 kept** (0 dropped for < 95% valid pixels).
- Tiles per split: {'train': 416, 'val': 80, 'test': 80}; blocks per split (4×4-tile blocks): {'test': 5, 'train': 26, 'val': 5}.
- Mean building-pixel share per split: {'test': 0.2158, 'train': 0.2506, 'val': 0.2159}.

## 4. Training

- Device: cuda NVIDIA GeForce RTX 3050 Laptop GPU; 7,763,905 parameters; 46 epochs (best epoch 31, early stopping patience 15); 62.6 s. pos_weight 2.99.
- Best validation IoU (threshold 0.5): 0.602.

## 5. Pixel-level test metrics (held-out spatial test blocks)

Threshold 0.60, selected on the validation split. 80 test tiles in 5 blocks; 21.6% of test pixels are reference buildings.

| Model | Precision | Recall | F1 | IoU | Accuracy |
|---|---|---|---|---|---|
| U-Net (val-selected threshold) | 0.692 | 0.847 | 0.762 | 0.615 | 0.886 |
| U-Net (threshold 0.5) | 0.661 | 0.883 | 0.756 | 0.608 | 0.877 |
| Random forest, per pixel | 0.607 | 0.843 | 0.706 | 0.545 | 0.848 |
| All-background | n/a | 0.000 | 0.000 | 0.000 | 0.784 |

Per-tile mean (U-Net): IoU 0.518 over 79 tiles with a defined IoU, F1 0.668. Background-inclusive 2-class mean IoU: 0.737. Identity check F1 = 2·IoU/(1+IoU): 0.7616 vs F1 0.7616.

## 6. Building coverage and change, 2020 → 2025

**Core area** (valid analysed 25.00 km² of 25.00 km²)

| Quantity | All changes (km²) | % of valid | High-confidence (km²) |
|---|---|---|---|
| Built-up 2020 | 11.071 | 44.3% | 11.071 |
| Built-up 2025 | 12.132 | 48.5% | 11.925 |
| Gross potential gain | 2.690 | 10.8% | 1.170 |
| Gross potential loss | 1.629 | 6.5% | 0.316 |
| Net change | 1.061 | 4.2% | 0.854 |
| Persistent built-up | 9.441 | 37.8% | 10.755 |

**Wide area** (valid analysed 235.92 km² of 235.93 km²)

| Quantity | All changes (km²) | % of valid | High-confidence (km²) |
|---|---|---|---|
| Built-up 2020 | 60.651 | 25.7% | 60.651 |
| Built-up 2025 | 72.446 | 30.7% | 68.142 |
| Gross potential gain | 21.596 | 9.2% | 9.135 |
| Gross potential loss | 9.801 | 4.2% | 1.644 |
| Net change | 11.795 | 5.0% | 7.491 |
| Persistent built-up | 50.850 | 21.6% | 59.007 |

Context (core): Open Buildings V3 rasterized mask 9.77 km², actual roof area 6.66 km²; U-Net 2023 map 11.76 km².

Gains/losses are **classification differences**, not confirmed construction or demolition.

## 7. Object-level evaluation (test tiles)

Objects = connected regions of building pixels (8-connected, 10 m) - NOT individual buildings. Matching: greedy one-to-one, polygon IoU >= 0.5.

- Predicted regions 2269, reference regions 3020: TP 875, FP 1394, FN 2145 → **precision 0.386, recall 0.290**. Merges 446, splits 371.
- Sensitivity at IoU ≥ 0.25: precision 0.658, recall 0.494.
- Footprint level (footprint detected if >= 50% of it lies in predicted regions; region supported if footprints cover >= 25% of it (many-to-one allowed)): building detection rate 0.815 of 53,287 footprints; region support rate 0.816.

## 8. Resume claims vs measured results

| Resume claim | Measured here | Verdict |
|---|---|---|
| ≈60 non-overlapping 64×64 tiles | core holds 49 tiles; 576 tiles used for training/val/test in the wide area | ≈60 tiles ≈ one 25 km² area; too few for a reliable spatially separated test set on its own |
| IoU 69% | 61.5% | not reproduced |
| F1 82% | 76.2% | not reproduced; the two resume figures are consistent with each other (69% IoU ⇔ 81.7% F1) |
| 21 km² built-up growth | core gross gain 2.69 km²; closest quantity to 21 km²: 'wide gross gain' = 21.60 km² | the maximum gain physically possible in the core is 13.93 km², so 21 km² cannot be growth inside a 25 km² area; see `resume_21km2_investigation.csv` |

**On the 21 km² figure.** Inside the 25 km² core, built-up area was 11.07 km² (2020) and 12.13 km² (2025). The only measured quantity near 21 km² is the *gross* potential gain over the 236 km² wide area (21.60 km²). That area's size was a training choice made in this project, the quantity is unvalidated classification change, its net change is 11.79 km², and its high-confidence gain is only 9.13 km². The numerical closeness is therefore not evidence that the resume figure is reproduced. The most likely readings of the original claim are (a) total built-up area, or (b) gross gain over a study area much larger than 25 km².
| Object precision 85% / recall 79% | 38.6% / 29.0% (region-level, IoU ≥ 0.5) | not reproduced |

## 9. Limitations

- Labels: Open Buildings V3 is a single snapshot derived from ~2021-2023 high-resolution imagery; it misses some buildings, contains false detections, and is not dated per building. It is a reference, not ground truth.
- There is no independent building reference for 2020 or 2025, so the change maps cannot be validated; differences include errors from both years' classifications.
- 10 m pixels are larger than many Varanasi buildings; the model maps building-dominated pixels/urban fabric, not individual roofs. Object metrics therefore describe contiguous built-up regions.
- Composites from different years differ in atmosphere (haze, aerosols), sensor (S2A/B/C) and processing baseline; this creates spurious change.
- Spatial blocks reduce but do not remove spatial autocorrelation between train and test (blocks touch at their edges).
- Metrics depend on the label rasterization rule (coverage threshold 0.25) and confidence filter (0.70).
