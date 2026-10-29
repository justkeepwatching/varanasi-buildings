"""Run the whole pipeline, or selected stages, in order.

Examples (PowerShell, from the project root, venv activated):
    python scripts/run_pipeline.py                      # every stage
    python scripts/run_pipeline.py --stages train evaluate
    python scripts/run_pipeline.py --stages imagery --force   # re-download

Stages: boundaries imagery labels tiles train evaluate infer change vectors figures report
Each stage reads the previous stages' files from disk, so stages can be re-run
independently. Downloads are skipped when their output already exists
unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

from src import change_detection as cd  # noqa: E402
from src import labels as lab  # noqa: E402
from src import visualization as viz  # noqa: E402
from src.config import PATHS, load_config  # noqa: E402
from src.imagery_preprocessing import check_on_grid, read_raster, valid_mask_from_image, write_geotiff  # noqa: E402
from src.study_area import area_report, core_grid, wide_grid, write_boundaries  # noqa: E402
from src.tiling import assign_spatial_split, cut_tiles, tile_origins, write_tile_geotiffs  # noqa: E402

log = logging.getLogger("pipeline")
STAGES = ["boundaries", "imagery", "labels", "tiles", "train", "evaluate", "infer", "change", "vectors",
          "figures", "report"]


def dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str))


def s2_path(period):
    return PATHS.interim / f"s2_{period}_wide.tif"


# ----------------------------------------------------------------------------- stages
def stage_boundaries(cfg, args):
    reports = write_boundaries(cfg, PATHS.external)
    reports["core_full_tiles_64px"] = len(tile_origins(core_grid(cfg).height, core_grid(cfg).width, 64, 64))
    reports["wide_full_tiles_64px"] = len(tile_origins(wide_grid(cfg).height, wide_grid(cfg).width, 64, 64))
    reports["tile_ground_size_m"] = cfg["tiling"]["tile_size"] * cfg["study_area"]["pixel_size_m"]
    dump(reports, PATHS.metrics / "study_area.json")
    log.info("core area: %s km² planar, %s km² geodesic", reports["core"]["planar_area_km2"],
             reports["core"]["geodesic_area_km2"])


def stage_imagery(cfg, args):
    grid = wide_grid(cfg)
    source = cfg["imagery"]["source"]
    if source == "gee":
        from src.gee_download import build_composite
    else:
        from src.pc_download import build_composite
    for period in cfg["imagery"]["periods"]:
        out = s2_path(period)
        if out.exists() and not args.force:
            log.info("%s exists, skipping (use --force to rebuild)", out.name)
            continue
        log.info("Building %s composite via %s", period, source)
        meta = build_composite(cfg, grid, period, out)
        meta["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
        dump(meta, PATHS.interim / f"s2_{period}_meta.json")
        check_on_grid(out, grid)


def stage_labels(cfg, args):
    grid = wide_grid(cfg)
    raw = PATHS.raw / "open_buildings_v3_wide.gpkg"
    if not raw.exists() or args.force:
        rep = area_report(grid)
        if cfg["imagery"]["source"] == "gee":
            from src.gee_download import download_open_buildings
            gdf, meta = download_open_buildings(cfg, grid), {"source": "Earth Engine " + cfg["labels"]["gee_collection"]}
        else:
            gdf, meta = lab.download_open_buildings(cfg, grid, rep["bounds_lonlat"])
        meta.update(retrieved_utc=datetime.now(timezone.utc).isoformat(), min_confidence=cfg["labels"]["min_confidence"],
                    n_polygons=len(gdf), licence="CC BY-4.0 / ODbL - Google Open Buildings V3")
        gdf.to_file(raw, driver="GPKG")
        dump(meta, PATHS.raw / "open_buildings_v3_meta.json")
    gdf = __import__("geopandas").read_file(raw)
    gdf, clean = lab.clean_footprints(gdf, grid.crs)
    gdf.to_file(PATHS.interim / "open_buildings_v3_wide_utm.gpkg", driver="GPKG")
    frac = lab.rasterize_coverage(gdf, grid, cfg["labels"]["supersample"])
    mask = lab.coverage_to_mask(frac, cfg["labels"]["coverage_threshold"])
    tags = {"rule": f"coverage >= {cfg['labels']['coverage_threshold']} on {cfg['labels']['supersample']}x supersampled grid"}
    write_geotiff(PATHS.interim / "labels_coverage_wide.tif", frac.astype(np.float32), grid, None, ["roof_fraction"], tags)
    write_geotiff(PATHS.interim / "labels_mask_wide.tif", mask, grid, None, ["building"], tags)
    stats = {
        **clean,
        "footprint_area_km2": float(gdf.geometry.area.sum() / 1e6),
        "footprint_area_median_m2": float(gdf.geometry.area.median()),
        "footprints_smaller_than_one_pixel_pct": float(100 * (gdf.geometry.area < 100).mean()),
        "coverage_sum_km2": float(frac.sum() * 100 / 1e6),
        "mask_building_pixels": int(mask.sum()),
        "mask_building_area_km2": float(mask.sum() * 100 / 1e6),
        "mask_building_share": float(mask.mean()),
    }
    dump(stats, PATHS.metrics / "label_statistics.json")
    log.info("labels: %s", stats)


def stage_tiles(cfg, args):
    grid = wide_grid(cfg)
    period = cfg["imagery"]["train_period"]
    for p in (s2_path(period), PATHS.interim / "labels_mask_wide.tif", PATHS.interim / "labels_coverage_wide.tif"):
        check_on_grid(p, grid)  # same CRS, size and transform -> pixel-aligned
    img, prof = read_raster(s2_path(period))
    mask = read_raster(PATHS.interim / "labels_mask_wide.tif")[0][0]
    frac = read_raster(PATHS.interim / "labels_coverage_wide.tif")[0][0]
    assert img.shape[0] == cfg["model"]["in_channels"], "band count does not match model input"
    valid = valid_mask_from_image(img, prof["nodata"])
    images, masks, valids, man = cut_tiles(img, valid, mask, frac, grid, cfg, s2_path(period).name)
    man = assign_spatial_split(man, cfg)
    PATHS.tiles.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PATHS.tiles / "tiles.npz", images=images, masks=masks, valids=valids)
    man.to_csv(PATHS.tiles / "manifest.csv", index=False)
    write_tile_geotiffs(images, masks, man, PATHS.tiles, cfg["imagery"]["bands"])
    summary = {
        "candidate_full_tiles": man.attrs["candidates"],
        "dropped_low_valid": man.attrs["dropped_low_valid"],
        "dropped_empty": man.attrs["dropped_empty"],
        "kept_tiles": len(man),
        "tiles_per_split": man.split.value_counts().to_dict(),
        "blocks_per_split": man.groupby("split").block_id.nunique().to_dict(),
        "building_pixel_share_per_split": man.groupby("split").building_pixel_fraction.mean().round(4).to_dict(),
        "tiles_with_no_buildings": int((man.building_pixel_fraction == 0).sum()),
        "tile_ground_size_m": cfg["tiling"]["tile_size"] * grid.pixel_size,
        "source_image": s2_path(period).name,
    }
    dump(summary, PATHS.metrics / "tiling_summary.json")
    log.info("tiles: %s", summary)


def stage_train(cfg, args):
    from src.train import train
    log.info("training summary: %s", train(cfg))


def stage_evaluate(cfg, args):
    from src.evaluate import evaluate
    m = evaluate(cfg)
    log.info("U-Net test: %s", {k: round(m["unet_global"][k], 4) for k in ("precision", "recall", "f1", "iou")})


def _threshold():
    return json.loads((PATHS.metrics / "metrics.json").read_text())["threshold"]["value"]


def stage_infer(cfg, args):
    from src.inference import run_inference
    from src.model import load_checkpoint
    from src.train import get_device
    device = get_device()
    model, ckpt = load_checkpoint(PATHS.models / "unet_best.pt", device)
    thr = _threshold()
    for period in cfg["imagery"]["periods"]:
        run_inference(model, ckpt["band_stats"], s2_path(period), wide_grid(cfg), thr, device, cfg,
                      PATHS.rasters / f"unet_{period}_wide")
        log.info("inference %s done", period)


def _read_core(path, cfg):
    wide, core = wide_grid(cfg), core_grid(cfg)
    r0, c0 = core.window_in(wide)
    with rasterio.open(path) as src:
        return src.read(1, window=rasterio.windows.Window(c0, r0, core.width, core.height))


def stage_change(cfg, args):
    e, l = cfg["imagery"]["change_periods"]
    thr, margin = _threshold(), cfg["change"]["confidence_margin"]
    out = {"periods": [e, l], "threshold": thr, "method": "post-classification comparison of U-Net maps",
           "status": "MEASURED from model predictions (not validated against independent change reference)"}
    for area, grid, reader in (("core", core_grid(cfg), lambda p: _read_core(p, cfg)),
                               ("wide", wide_grid(cfg), lambda p: rasterio.open(p).read(1))):
        pe = reader(PATHS.rasters / f"unet_{e}_wide_probability.tif")
        pl = reader(PATHS.rasters / f"unet_{l}_wide_probability.tif")
        valid = (pe >= 0) & (pl >= 0)
        ratio = area_report(grid)["geodesic_to_planar_ratio"]
        cls = cd.change_classes(pe >= thr, pl >= thr, valid)
        conf = cd.confident_change_classes(pe, pl, valid, thr, margin)
        total = grid.width * grid.height * grid.pixel_size**2 / 1e6
        out[area] = {"all_changes": cd.change_statistics(cls, grid.pixel_size, ratio, total),
                     f"high_confidence_margin_{margin}": cd.change_statistics(conf, grid.pixel_size, ratio, total)}
        write_geotiff(PATHS.rasters / f"change_{e}_{l}_{area}.tif", cls, grid, cd.NOT_ANALYSED, ["change_class"],
                      {"classes": cd.CLASS_NAMES})
        if area == "core":
            write_geotiff(PATHS.rasters / f"change_{e}_{l}_core_high_confidence.tif", conf, grid, cd.NOT_ANALYSED,
                          ["change_class"], {"classes": cd.CLASS_NAMES, "margin": margin})
            # reference (Open Buildings V3, ~2022-23) built-up in the core, for context
            ref = _read_core(PATHS.interim / "labels_mask_wide.tif", cfg)
            frac = _read_core(PATHS.interim / "labels_coverage_wide.tif", cfg)
            p23 = _read_core(PATHS.rasters / "unet_2023_wide_probability.tif", cfg)
            out["core_reference"] = {
                "open_buildings_mask_area_km2": float(ref.sum() * 100 / 1e6),
                "open_buildings_roof_area_km2": float(frac.sum() * 100 / 1e6),
                "unet_2023_built_up_km2": float(((p23 >= thr) & (p23 >= 0)).sum() * 100 / 1e6),
            }
    dump(out, PATHS.metrics / "change_statistics.json")
    investigate_21km2(cfg, out)


def investigate_21km2(cfg, change):
    """Compare the resume's '21 km² of built-up growth' with every candidate quantity."""
    claim = 21.0
    core, wide = change["core"]["all_changes"], change["wide"]["all_changes"]
    cands = {
        "core study area (total)": core["study_area_km2"],
        f"core built-up {change['periods'][0]}": core["earlier_built_up_km2"],
        f"core built-up {change['periods'][1]}": core["later_built_up_km2"],
        "core gross gain": core["gross_gain_km2"],
        "core net change": core["net_change_km2"],
        "core Open Buildings mask area": change["core_reference"]["open_buildings_mask_area_km2"],
        "wide study area (total)": wide["study_area_km2"],
        f"wide built-up {change['periods'][0]}": wide["earlier_built_up_km2"],
        f"wide built-up {change['periods'][1]}": wide["later_built_up_km2"],
        "wide gross gain": wide["gross_gain_km2"],
        "wide net change": wide["net_change_km2"],
    }
    rows = [{"quantity": k, "km2": round(v, 3), "difference_from_21_km2": round(v - claim, 3)} for k, v in cands.items()]
    df = pd.DataFrame(rows)
    df.to_csv(PATHS.metrics / "resume_21km2_investigation.csv", index=False)
    closest = df.iloc[(df.km2 - claim).abs().argsort()].iloc[0]
    dump({"claim_km2": claim, "closest_quantity": closest.quantity, "closest_km2": closest.km2,
          "max_possible_gain_in_core_km2": core["valid_analysed_area_km2"] - core["earlier_built_up_km2"],
          "candidates": rows}, PATHS.metrics / "resume_21km2_investigation.json")


def stage_vectors(cfg, args):
    import geopandas as gpd
    from shapely.geometry import box

    from src.object_evaluation import footprint_detection_rates, object_scores
    from src.vectorization import mask_to_polygons

    vcfg = cfg["vectorization"]
    wide = wide_grid(cfg)
    thr = _threshold()
    man = pd.read_csv(PATHS.tiles / "manifest.csv")
    tiles = np.load(PATHS.tiles / "tiles.npz")
    pred = np.load(PATHS.processed / "test_predictions.npz", allow_pickle=True)
    test = man[man.split == "test"].reset_index(drop=True)
    assert list(test.tile_id) == list(pred["tile_ids"]), "test tile order mismatch"
    idx = np.flatnonzero(man.split.values == "test")

    # place test-tile predictions / references on a wide-grid canvas (outside test = 0)
    ts = cfg["tiling"]["tile_size"]
    p_canvas = np.zeros((wide.height, wide.width), np.uint8)
    r_canvas = np.zeros_like(p_canvas)
    for k, rec in enumerate(test.itertuples()):
        v = tiles["valids"][idx[k]]
        p_canvas[rec.row:rec.row + ts, rec.col:rec.col + ts] = (pred["probs"][k] >= thr) & v
        r_canvas[rec.row:rec.row + ts, rec.col:rec.col + ts] = tiles["masks"][idx[k]] & v
    kw = dict(connectivity=vcfg["connectivity"], min_pixels=vcfg["min_region_pixels"], opening_kernel=vcfg["opening_kernel"])
    pred_gdf = mask_to_polygons(p_canvas, wide.transform, wide.crs, **kw)
    ref_gdf = mask_to_polygons(r_canvas, wide.transform, wide.crs, object_type="reference built-up region", **kw)
    pred_gdf.to_file(PATHS.vectors / "test_predicted_regions.gpkg", driver="GPKG")
    ref_gdf.to_file(PATHS.vectors / "test_reference_regions.gpkg", driver="GPKG")

    test_area = gpd.GeoSeries([box(r.minx, r.miny, r.maxx, r.maxy) for r in test.itertuples()], crs=wide.crs).union_all()
    fps = gpd.read_file(PATHS.interim / "open_buildings_v3_wide_utm.gpkg")
    fps = fps[fps.geometry.centroid.within(test_area)]

    iou_t = cfg["object_evaluation"]["iou_threshold"]
    result = {
        "status": "MEASURED on held-out test tiles",
        "object_definition": "connected regions of building pixels (8-connected, 10 m) - NOT individual buildings",
        "region_level_one_to_one": object_scores(pred_gdf, ref_gdf, iou_t),
        "region_level_one_to_one_iou_0.25": object_scores(pred_gdf, ref_gdf, 0.25),
        "footprint_level_detection_rates": footprint_detection_rates(pred_gdf, fps),
        "settings": {**kw, "iou_threshold": iou_t},
    }
    dump(result, PATHS.metrics / "object_metrics.json")

    # vector product for the core area (2025 map + gains), for GIS use
    core = core_grid(cfg)
    e, l = cfg["imagery"]["change_periods"]
    m_late = _read_core(PATHS.rasters / f"unet_{l}_wide_binary.tif", cfg)
    mask_to_polygons((m_late == 1).astype(np.uint8), core.transform, core.crs, **kw).to_file(
        PATHS.vectors / f"core_builtup_regions_{l}.gpkg", driver="GPKG")
    with rasterio.open(PATHS.rasters / f"change_{e}_{l}_core_high_confidence.tif") as src:
        ch = src.read(1)
    mask_to_polygons((ch == cd.GAIN).astype(np.uint8), core.transform, core.crs,
                     object_type="potential gain region (high-confidence)", **kw).to_file(
        PATHS.vectors / f"core_potential_gain_{e}_{l}.gpkg", driver="GPKG")
    log.info("object metrics: %s", result["region_level_one_to_one"])


def stage_figures(cfg, args):
    wide, core = wide_grid(cfg), core_grid(cfg)
    r0, c0 = core.window_in(wide)
    crop = lambda a: a[..., r0:r0 + core.height, c0:c0 + core.width]  # noqa: E731
    imgs = {p: read_raster(s2_path(p))[0] for p in cfg["imagery"]["periods"]}
    tp = cfg["imagery"]["train_period"]
    f = PATHS.figures
    viz.study_area_map(imgs[tp], wide, core, PATHS.maps / "01_study_area.png",
                       f"Study area, Varanasi (Sentinel-2 {tp} composite)")
    viz.imagery_panels({p: crop(i) for p, i in imgs.items()}, core, f / "02_example_imagery_core.png",
                       "Core area: dry-season (Feb-May) median composites")
    frac = read_raster(PATHS.interim / "labels_coverage_wide.tif")[0][0]
    mask = read_raster(PATHS.interim / "labels_mask_wide.tif")[0][0]
    viz.label_map(crop(imgs[tp]), crop(frac), crop(mask), core, PATHS.maps / "03_reference_labels_core.png",
                  f"Open Buildings V3 reference, rasterized (pixel = building if coverage >= {cfg['labels']['coverage_threshold']})")
    man = pd.read_csv(PATHS.tiles / "manifest.csv")
    viz.tile_split_map(imgs[tp], man, wide, PATHS.maps / "04_tiles_and_spatial_split.png",
                       "64 x 64 px tiles (640 m) and block-wise train / val / test split",
                       tile_origins(wide.height, wide.width, 64, 64))
    viz.training_curves(pd.read_csv(PATHS.metrics / "training_history.csv"), f / "05_training_curves.png")

    tiles = np.load(PATHS.tiles / "tiles.npz")
    pred = np.load(PATHS.processed / "test_predictions.npz", allow_pickle=True)
    idx = np.flatnonzero(man.split.values == "test")
    thr = _threshold()
    order = np.argsort(-man.building_pixel_fraction.values[idx])  # spread across densities
    pick = order[np.linspace(0, len(order) - 1, 5).astype(int)]
    viz.test_examples(tiles["images"][idx[pick]], tiles["masks"][idx[pick]], pred["probs"][pick],
                      pred["probs"][pick] >= thr, tiles["valids"][idx[pick]], list(pred["tile_ids"][pick]),
                      f / "06_test_examples.png", thr)
    e, l = cfg["imagery"]["change_periods"]
    me = _read_core(PATHS.rasters / f"unet_{e}_wide_binary.tif", cfg)
    ml = _read_core(PATHS.rasters / f"unet_{l}_wide_binary.tif", cfg)
    viz.coverage_pair(me, ml, core, [f"U-Net building coverage {e}", f"U-Net building coverage {l}"],
                      PATHS.maps / "07_coverage_early_late_core.png", "Core area building-coverage classification")
    ch = json.loads((PATHS.metrics / "change_statistics.json").read_text())
    for name, suffix in (("all_changes", ""), (f"high_confidence_margin_{cfg['change']['confidence_margin']}", "_high_confidence")):
        with rasterio.open(PATHS.rasters / f"change_{e}_{l}_core{suffix}.tif") as src:
            viz.change_map(src.read(1), core, PATHS.maps / f"08_change_{e}_{l}_core{suffix}.png",
                           f"Potential change {e} -> {l} ({name.replace('_', ' ')})", ch["core"][name])
    import geopandas as gpd
    pg = gpd.read_file(PATHS.vectors / "test_predicted_regions.gpkg")
    rg = gpd.read_file(PATHS.vectors / "test_reference_regions.gpkg")
    # zoom on the test block with the most reference buildings
    test = man[man.split == "test"]
    blk = test.groupby("block_id").building_pixel_fraction.mean().idxmax()
    b = test[test.block_id == blk]
    ext = (b.minx.min(), b.maxx.max(), b.miny.min(), b.maxy.max())
    viz.vector_overlay(None, pg, rg, PATHS.maps / "09_vectorized_test_block.png",
                       f"Vectorized regions, test block {blk} (10 m, connected pixels)", ext)
    m = json.loads((PATHS.metrics / "metrics.json").read_text())
    viz.metric_bars({"U-Net": m["unet_global"], "Random forest (pixel)": m["baseline_random_forest_pixelwise"],
                     "All-background": m["baseline_all_background"]}, f / "10_metric_summary.png",
                    "Pixel-level test metrics (building class)")


def stage_report(cfg, args):
    from src.reporting import write_report
    write_report(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--force", action="store_true", help="re-download data even if files exist")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    PATHS.make_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(PATHS.outputs / "pipeline.log", encoding="utf-8")])
    for s in STAGES:
        if s in args.stages:
            log.info("===== stage: %s =====", s)
            globals()[f"stage_{s}"](cfg, args)


if __name__ == "__main__":
    main()
