# Interview Preparation — Building Coverage Mapping & Change Detection | Varanasi

This guide teaches the background from zero, explains every design decision in this repository, and gives
practice questions. **Use the measured numbers from `outputs/reports/final_report.md`**, not the resume figures.
Only claim what the code and logs show; if asked about a step that was not executed (e.g. the Earth Engine path),
say it is implemented but was not the path that produced the results.

---

## Part A — Concepts from zero

### A1. Remote sensing and Sentinel-2
*Remote sensing* means measuring the Earth from a distance — here, sunlight reflected from the surface and
recorded by a satellite camera. **Sentinel-2** (European Space Agency, Copernicus programme) is a pair/trio of
satellites (2A, 2B, 2C) that image every land area about every 5 days. Each image measures 13 *spectral bands*
(colours of light, including invisible infrared). Data are free and open.

* **Level-1C** = top-of-atmosphere reflectance; **Level-2A** = *surface reflectance*, corrected for atmosphere.
  We use L2A because we compare years, and the atmosphere changes from day to day.
* **Reflectance** = fraction of incoming light reflected (0–1). Stored as integers ×10000.

### A2. Spectral bands and spatial resolution
| Band | Wavelength | Native pixel | Why useful here |
|---|---|---|---|
| B02, B03, B04 | blue, green, red | 10 m | true colour, roofs vs vegetation |
| B08 | near-infrared (NIR) | 10 m | vegetation is very bright in NIR; buildings are not |
| B11, B12 | short-wave infrared (SWIR) | 20 m | concrete/roofs vs bare soil and water |

*Spatial resolution* = ground size of one pixel. At 10 m, one pixel = 100 m². A typical small Varanasi house of
50–100 m² is smaller than one pixel, so we map "building-dominated pixels", not individual houses.
Indices: **NDVI** = (NIR−Red)/(NIR+Red) (vegetation), **NDBI** = (SWIR−NIR)/(SWIR+NIR) (built-up),
**NDWI** = (Green−NIR)/(Green+NIR) (water).

### A3. Google Earth Engine (GEE)
A cloud platform with petabytes of satellite data and a server-side API (JavaScript / Python). You describe a
computation (filter, mask, composite) and Google's servers run it. Needs a Google account and a registered
Cloud project. In this project, `src/gee_download.py` implements the GEE path; the default run used Microsoft
Planetary Computer (same Sentinel-2 L2A data, accessed as Cloud-Optimized GeoTIFFs through a STAC catalogue)
because it needs no account. Be precise in the interview about which path produced your results.

### A4. Open Buildings V3 and label quality
Google Research detected ~1.8 billion building footprints in Africa, South/South-East Asia, Latin America and the
Caribbean from 50 cm imagery using deep learning. Each polygon has a **confidence** score (0.65–1). V3 was released
in 2023. Caveats: (1) it is itself a model output with errors (missed buildings, false positives, merged rows of
terraced houses); (2) imagery dates vary by area (roughly 2021–2023) and are not given per building; (3) it is a
single snapshot, so it cannot label 2020 or 2025 directly. We therefore call it a *reference*, not ground truth.

### A5. Raster vs vector
* **Raster** = grid of pixels (satellite image, probability map). Stored as GeoTIFF.
* **Vector** = geometric shapes (points, lines, polygons) with attributes (building footprints). Stored as
  GeoPackage/GeoJSON/Shapefile.
We convert vector→raster (*rasterization*) to create training masks, and raster→vector (*polygonization*) to
produce GIS-ready outputs and object-level metrics.

### A6. CRS and reprojection
A *coordinate reference system* tells you what the numbers in coordinates mean. **EPSG:4326** = longitude/latitude
in degrees (not suitable for area, a degree of longitude shrinks towards the poles). **EPSG:32644** = UTM zone
44 North, coordinates in metres, nearly true distances within the zone — Varanasi (~83°E) lies in zone 44
(78–84°E). *Reprojection* converts data between CRSs; for rasters this needs *resampling* (nearest neighbour for
categories, bilinear for continuous values) and slightly alters values, so we reproject only when necessary.
UTM is conformal, not equal-area: areas have a small scale error (here ≈0.1%), so we also report geodesic area.

### A7. The libraries and why each was used
| Library | Used for | Why |
|---|---|---|
| **GDAL** | the engine under rasterio/pyogrio: COG reading over HTTP, warping (WarpedVRT), polygonize | industry standard for raster/vector formats and reprojection |
| **Rasterio** | reading/writing GeoTIFFs with CRS + affine transform, windows, rasterize, shapes | Pythonic GDAL wrapper for rasters |
| **GeoPandas / Shapely** | polygons, CRS conversion, validity repair, spatial joins, IoU between polygons | pandas-style vector analysis |
| **OpenCV** | area-averaging the 2 m label grid to 10 m (`INTER_AREA`), connected components, morphology | fast, well-tested image operations |
| **PyTorch** | U-Net, training on GPU | flexible, standard deep-learning library |
| **scikit-learn** | random-forest baseline | simple non-neural reference model |
| **pystac-client / planetary-computer** | finding and signing Sentinel-2 scenes | open STAC standard, no account |

### A8. Tiling and data leakage
CNNs are trained on fixed-size patches. We cut **64×64-pixel tiles** (640 m × 640 m). *Non-overlapping* tiles
avoid the same pixels appearing twice. **Data leakage** = information from the test set influencing training.
In geospatial data, neighbouring tiles look alike (*spatial autocorrelation*), so a random tile split gives
over-optimistic test scores. We split by **geographic blocks** (4×4 tiles) so each block is entirely train, val
or test. Normalization statistics are computed from training tiles only, and the decision threshold is chosen on
validation tiles only — both are additional leakage guards.

### A9. CNNs and U-Net
A *convolutional neural network* learns small filters (e.g. 3×3) that detect patterns (edges, textures) and
stacks them into increasingly abstract features. **Semantic segmentation** = a class for every pixel.
**U-Net** (Ronneberger et al., 2015):
* **Encoder**: conv blocks + max-pooling; resolution 64→32→16→8→4; channels grow 32→512. Captures context.
* **Bottleneck**: the most compressed representation.
* **Decoder**: transposed convolutions double the resolution back to 64.
* **Skip connections**: concatenate encoder features with decoder features at the same resolution, restoring
  fine spatial detail lost by pooling — critical for sharp boundaries.
* **Head**: 1×1 convolution → one *logit* per pixel; sigmoid → probability.
* **BatchNorm** normalizes activations for stable training; **ReLU** adds non-linearity; **Dropout** in the
  bottleneck reduces overfitting.

### A10. Training, validation, test; loss; optimizer
* **Train set** fits weights; **validation set** chooses checkpoint/threshold/early stopping; **test set** is
  touched once for the final score.
* **Loss**: BCE-with-logits measures per-pixel classification error; `pos_weight` up-weights the minority class.
  **Dice loss** = 1 − Dice coefficient; directly targets overlap and ignores the mass of easy background pixels.
* **Optimizer**: AdamW — adaptive learning rates per weight (Adam) + decoupled weight decay (regularization).
* **Epoch** = one pass over the training data. **Early stopping** stops when validation IoU stops improving.
* **Overfitting**: training loss keeps falling while validation loss rises — watch the curves in
  `outputs/figures/05_training_curves.png`.

### A11. Metrics
Confusion matrix for the building class:

|  | reference building | reference background |
|---|---|---|
| **predicted building** | TP | FP |
| **predicted background** | FN | TN |

* Precision = TP/(TP+FP): "when I say building, how often am I right?"
* Recall = TP/(TP+FN): "of the real buildings, how many did I find?"
* F1 = 2PR/(P+R) = 2TP/(2TP+FP+FN)
* IoU (Jaccard) = TP/(TP+FP+FN) — always ≤ F1; F1 = 2·IoU/(1+IoU).
  So **69% IoU corresponds to 81.7% F1** — the resume's 69/82 pair is internally consistent.
* Accuracy = (TP+TN)/all — misleading under **class imbalance**: an all-background model gets high accuracy and
  zero recall (see the all-background baseline row in the report).
* Global vs per-tile averaging give different numbers; always say which.

### A12. Change detection and areas
*Post-classification comparison*: classify each year independently, then cross-tabulate (gain, loss, persistent,
stable). Errors in both maps add up, so a 5% error per map can create many false "changes". Area = pixel count ×
100 m²; 10,000 pixels = 1 km². Distinguish **total built-up in 2025** from **gain since 2020**: gross gain counts
only pixels that went 0→1; net change = gains − losses.

### A13. Polygonization and object-level evaluation
Polygonization traces connected groups of pixels into polygons (*4-connectivity*: edge neighbours only;
*8-connectivity*: diagonals too). Object-level metrics count objects instead of pixels: a predicted polygon is a
true positive if it matches a reference polygon with IoU ≥ 0.5. Matching must be **one-to-one** so one big
prediction covering five references counts as at most one TP. At 10 m, polygons are blocks of houses, so object
metrics depend heavily on how regions merge.

### A14. Relevance to catastrophe risk modelling
Exposure = what is at risk: location, number, type, value of buildings. Cat models (e.g. for Ganga floods or
earthquakes in the Indo-Gangetic plain) aggregate exposure to grids or admin units. A 10 m building-coverage map
gives built-up area per grid cell, which can be combined with occupancy/construction-type assumptions and
replacement costs per m² to produce a gridded exposure layer. Change maps show where exposure grew, flagging
where portfolios or industry exposure databases are stale. Limitations for that use: no building height (floor
area), no occupancy/construction type, no individual buildings; usually combined with census, OpenStreetMap,
GHSL, and Open Buildings 2.5D (heights).

---

## Part B — Every important decision (what / why / alternative / trade-off / how to say it)

| Decision | What it does | Why | Alternative | Trade-off | How to explain |
|---|---|---|---|---|---|
| UTM 44N, 10 m grid | single projected grid for all data | metres → exact pixel areas; matches S2 native lattice | EPSG:4326 or Web Mercator | degree pixels are not square / not equal-area | "I fixed one metric grid first so every layer is pixel-aligned by construction." |
| Wide + core areas | train on 236 km², report on 25 km² | 49 tiles in 25 km² is too few for a spatial train/val/test split | train only on ~60 tiles | more data vs scope creep | "60 tiles would give ~9 test tiles; I trained wider and kept reporting on the 25 km² core." |
| Feb–May composites | same season each year | limits phenology/monsoon/cloud differences | single best scene | single scenes have clouds/haze | "Seasonal consistency reduces spurious change." |
| Median composite | per-pixel median of clear looks | robust to residual cloud/shadow outliers | mean, greenest-pixel, best-pixel | median can blur real within-season change | "Median rejects outliers that masks miss." |
| SCL masking (PC path) / Cloud Score+ (GEE) | remove cloudy pixels | L2A classification available with the data | s2cloudless, Fmask | SCL misses thin haze | "Scene filter is coarse; pixel mask is fine-grained." |
| Baseline-04.00 offset | subtract 1000 from post-2022 DNs | otherwise 2025 looks ~0.1 brighter than 2020 | use harmonized collection (GEE) | must track metadata | "I harmonized processing baselines so change isn't an artefact." |
| 6 bands incl. SWIR | spectral input | SWIR separates built-up vs soil | RGB only (pretrained models) | can't use ImageNet weights | "Multispectral input matters more than pretrained weights at 10 m." |
| 2 m supersampled rasterization, 0.25 threshold | label rule | unbiased roof fraction | all_touched, centre-pixel | threshold choice affects metrics | "I computed coverage fraction, then chose an explicit, documented threshold." |
| Confidence ≥ 0.70 | label filter | Google's guidance: 0.65–0.75 | 0.65 (more recall) / 0.75 | precision vs completeness of labels | — |
| 64×64 non-overlapping tiles | training samples | matches U-Net 4-level downsampling; no duplicate pixels | overlapping tiles, 128×128 | fewer samples vs more context | "Non-overlap prevents duplicated pixels and leakage." |
| Block split, stratified | leakage control | spatial autocorrelation | random split; spatial CV | fewer independent test blocks | "Whole 2.56 km blocks go to one split." |
| U-Net from scratch | segmentation model | standard, works with small data via skips | DeepLabV3, pretrained ResNet encoder, RF | overfitting risk on small data | "U-Net's skip connections keep detail; no pretrained 6-band weights exist." |
| BCE(pos_weight) + Dice | loss | class imbalance + overlap objective | focal loss, Lovász | Dice is unstable on empty tiles alone | "BCE gives stable gradients; Dice targets IoU-like overlap." |
| Threshold on validation | decision rule | never tune on test | fixed 0.5 | small val set → noisy threshold | "Report both val-tuned and 0.5." |
| Sliding window stride 32 + Hann weights | inference | removes tile seams | non-overlapping inference | 4× compute | — |
| High-confidence change margin | change filter | removes threshold flicker | morphological filtering, multi-date | misses subtle real change | — |
| Greedy one-to-one IoU ≥ 0.5 matching | object metrics | standard (COCO-style), no double counting | Hungarian matching, centroid matching | greedy ≈ optimal when IoU ≥ 0.5 (matches are unique) | "Above IoU 0.5 a prediction can match at most one reference, so greedy is safe." |

Note: for IoU > 0.5 two disjoint references cannot both have IoU > 0.5 with the same prediction, which is why
greedy and Hungarian matching agree at that threshold.

---

## Part C — Practice questions (with answers)

### Beginner
1. **What is Sentinel-2?** ESA's free multispectral satellite mission, 13 bands, 10–60 m pixels, ~5-day revisit.
2. **Why 10 m?** It's Sentinel-2's finest native resolution (B02/B03/B04/B08); finer would mean inventing detail.
3. **What does L2A mean?** Atmospherically corrected surface reflectance (vs L1C top-of-atmosphere).
4. **Why use NIR?** Vegetation reflects NIR strongly; buildings don't — separates trees from roofs.
5. **Why SWIR?** Distinguishes impervious surfaces, soil and moisture; NDBI uses it.
6. **What is a GeoTIFF?** A TIFF image plus CRS, affine transform and nodata metadata.
7. **What is an affine transform?** Six numbers mapping pixel (row, col) to map (x, y): origin + pixel size (+rotation).
8. **Raster vs vector?** Pixels on a grid vs geometric shapes with attributes.
9. **What is a CRS?** Definition of what coordinates mean (datum, projection, units).
10. **Why not compute areas in EPSG:4326?** Degrees are not constant distances; areas would be wrong.
11. **What is a tile?** A fixed-size sub-image (here 64×64 px = 640 m) used as one training sample.
12. **What is semantic segmentation?** Assigning a class to every pixel.
13. **What's a logit?** The un-squashed model output; sigmoid(logit) = probability.
14. **What is IoU?** Overlap ÷ union of predicted and reference building pixels = TP/(TP+FP+FN).
15. **What is F1?** Harmonic mean of precision and recall.
16. **Why not accuracy?** Buildings are a minority; predicting all background scores high accuracy, zero recall.
17. **What is Open Buildings?** Google's ML-derived building footprints for the Global South, with confidence scores.
18. **What is a cloud mask?** A per-pixel flag removing clouds/shadows before compositing.

### Intermediate
19. **Scene cloud filter vs pixel mask?** Filter discards mostly-cloudy scenes; mask removes cloudy pixels in kept scenes. Use both.
20. **Why a median composite?** Robust to remaining cloud/shadow outliers and noise; fills gaps from multiple dates.
21. **What was the processing-baseline 04.00 issue?** From Jan 2022 ESA added +1000 to L2A DNs; ignoring it biases post-2022 reflectance and creates fake change.
22. **How did you align labels with imagery?** Both written on the same UTM grid (CRS, transform, size); asserted with `check_on_grid` before tiling.
23. **How were footprints rasterized?** 2 m sub-grid, pixel-centre rule, averaged to 10 m → roof fraction; ≥ 0.25 → building.
24. **Why not `all_touched=True`?** It marks every pixel a polygon touches → overestimates small buildings heavily at 10 m.
25. **How did you avoid leakage?** Block-wise spatial split, train-only normalization stats, threshold from validation, test used once.
26. **Why keep empty tiles?** They are real negatives (fields, river); dropping them biases the model to predict buildings.
27. **What does a skip connection do?** Passes encoder feature maps to the decoder at the same resolution to restore spatial detail.
28. **Why BatchNorm?** Stabilizes and speeds training by normalizing layer activations.
29. **Why pos_weight?** Balances BCE so missing a building costs more than a false alarm, countering imbalance.
30. **Why Dice loss?** It optimizes overlap directly and isn't dominated by easy background pixels.
31. **What is early stopping?** Stop when validation metric stops improving to avoid overfitting; keep best checkpoint.
32. **How did you pick the threshold?** Swept 0.30–0.70 on validation, chose max validation IoU, applied once to test.
33. **Global vs per-tile metrics?** Global sums counts over all pixels; per-tile averages tile scores — small tiles with few buildings weigh more in per-tile.
34. **Why compare against baselines?** To show the U-Net adds value beyond trivial (all-background) or classical (random forest) methods.
35. **What is post-classification change detection?** Classify each date, then compare maps pixel by pixel.
36. **Gross gain vs net change?** Gross = all 0→1 pixels; net = gains − losses.
37. **How do you compute area from pixels?** count × pixel area (100 m² at 10 m in UTM), optionally geodesic-corrected.

### Advanced
38. **Why is 21 km² of growth implausible in a 25 km² area?** It would mean 84% of the core changed from non-built to built in five years. Our 2020 map already classifies ~11 km² of the core as built-up, so at most ~14 km² could possibly have been gained. The only measured value near 21 km² is the *gross* classification gain over the 236 km² wide training area (~21.6 km²; net ~11.8 km², high-confidence gain ~9.1 km²). That is a coincidence of extent, not a reproduction. Say: "I re-measured it; within a 25 km² area the gross gain was ~2.7 km²." See `outputs/metrics/resume_21km2_investigation.csv`.
39. **Are 69% IoU and 82% F1 consistent?** Yes, approximately: F1 = 2·0.69/1.69 = 0.817.
40. **Why can't 60 tiles support a reliable test?** With block splitting you get ~9 test tiles from 1–2 blocks; the score's variance is huge and depends on which neighbourhood landed in test.
41. **How would you get confidence intervals?** Block bootstrap over test blocks, or spatial cross-validation across folds.
42. **Why does object precision differ from pixel precision?** Objects are counted, not weighted by area; one huge merged region and many tiny speckles count equally, and IoU ≥ 0.5 is strict for irregular 10 m blobs.
43. **One-to-many matches?** A merged prediction can match only one reference (others → FN); a split reference matches one prediction (others → FP). We report merges/splits explicitly.
44. **Why greedy matching is safe at IoU ≥ 0.5?** Two disjoint polygons can't both have IoU > 0.5 with the same polygon, so matches are unique.
45. **Temporal label mismatch?** Labels ~2021–23 vs imagery 2023; demolished/new buildings create label noise; applying the model to 2020 assumes buildings look the same spectrally.
46. **Domain shift between years?** Different sensors (S2A/B/C), atmosphere (April 2020 lockdown had cleaner air), season — handled partly by same-season medians and harmonized offsets; could add relative radiometric normalization on pseudo-invariant features.
47. **How to validate change without labels?** Visual interpretation of high-resolution imagery (Google Earth historical), stratified random sampling of change/no-change strata with area-adjusted estimates (Olofsson et al. 2014 good-practice).
48. **Why not train on 2020 too?** No 2020 labels; using V3 for 2020 would label buildings that didn't exist yet.
49. **How would you improve resolution?** Use PlanetScope (3 m) or high-res imagery, super-resolution with caution, or predict building fraction (regression) instead of a binary class.
50. **Why predict a binary mask and not a fraction?** Simpler metrics; but regression of roof fraction is arguably more honest at 10 m — a strong improvement idea.
51. **What spatial autocorrelation remains?** Blocks touch at edges; a buffer of excluded tiles between splits would remove it at the cost of data.
52. **What's a COG and STAC?** Cloud-Optimized GeoTIFF supports HTTP range reads of only the needed window; STAC is a JSON standard to search imagery catalogues.
53. **Why nearest-neighbour for SCL and bilinear for B11?** Classes can't be averaged; reflectance is continuous.
54. **How do you handle nodata in metrics?** Excluded from TP/FP/FN/TN and from the loss; tiles < 95% valid dropped.
55. **How does this feed a cat model?** Aggregate built-up area per grid cell → combine with occupancy, construction type, floor count and replacement cost per m² → total insured value per cell → loss with hazard and vulnerability curves.
56. **What would you change for production?** Higher-res imagery, building heights (Open Buildings 2.5D Temporal), multi-year time-series models, uncertainty maps, independent validation samples, MLOps for re-runs.

### Follow-up questions to expect
* "Show me where the threshold is chosen." → `src/evaluate.py::choose_threshold`.
* "Prove image and mask are aligned." → `check_on_grid` in `scripts/run_pipeline.py::stage_tiles`, and `tests/test_tiling.py`.
* "Why is your IoU lower than your resume?" → explain label noise, 10 m resolution, strict spatial test, and that the figure was re-measured honestly.
* "What does one pixel of 'gain' mean?" → a pixel classified non-building in 2020 and building in 2025; unverified.

### Common mistakes to avoid
* Saying "ground truth" for Open Buildings — say "reference labels".
* Calling 2025 total built-up "growth".
* Reporting accuracy instead of IoU/F1 for imbalanced classes.
* Tuning the threshold on the test set.
* Claiming polygon counts are building counts at 10 m.
* Claiming the GEE path was run if it wasn't.
* Quoting resume numbers that the code doesn't reproduce.

---

## Part D — Limitations and improvements
1. Labels are imperfect and single-date → add independent manual validation samples.
2. 10 m resolution → building fraction regression, higher-res data, or Open Buildings 2.5D Temporal for change.
3. Change not validated → stratified accuracy assessment with area-adjusted estimates.
4. Small test set → spatial k-fold cross-validation and block bootstrap confidence intervals.
5. Radiometric differences between years → multi-temporal models (e.g. Siamese/change U-Nets) or relative normalization.
