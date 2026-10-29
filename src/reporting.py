"""Generate outputs/reports/final_report.md from the metric files on disk.

Every number in the report is read from a JSON/CSV written by the pipeline;
the resume figures appear ONLY in the comparison table, labelled as claims.
"""
from __future__ import annotations

import json
from datetime import datetime

from .config import PATHS

RESUME = {
    "tiles": "≈60 non-overlapping 64×64 tiles",
    "iou": 0.69, "f1": 0.82, "growth_km2": 21.0,
    "obj_precision": 0.85, "obj_recall": 0.79,
}


def _j(name):
    p = PATHS.metrics / name
    return json.loads(p.read_text()) if p.exists() else None


def _f(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def write_report(cfg: dict) -> str:
    sa, lab, til, tr = _j("study_area.json"), _j("label_statistics.json"), _j("tiling_summary.json"), _j("training_summary.json")
    m, ch, ob, inv = _j("metrics.json"), _j("change_statistics.json"), _j("object_metrics.json"), _j("resume_21km2_investigation.json")
    metas = {p: json.loads((PATHS.interim / f"s2_{p}_meta.json").read_text())
             for p in cfg["imagery"]["periods"] if (PATHS.interim / f"s2_{p}_meta.json").exists()}
    e, l = cfg["imagery"]["change_periods"]
    L = []
    w = L.append
    w("# Building Coverage Mapping & Change Detection | Varanasi — Final Report\n")
    w(f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/run_pipeline.py --stages report`. "
      "All numbers below are read from files in `outputs/metrics/`; nothing is typed in by hand._\n")

    w("## 1. Study area\n")
    if sa:
        c, wd = sa["core"], sa["wide"]
        w(f"- **Core reporting area:** {c['width_px']}×{c['height_px']} px at 10 m = {c['planar_area_km2']} km² planar "
          f"(UTM 44N), {c['geodesic_area_km2']} km² geodesic. Lon/lat bounds {c['bounds_lonlat']}.")
        w(f"- **Wide training area:** {wd['width_px']}×{wd['height_px']} px = {wd['planar_area_km2']} km² planar, "
          f"{wd['geodesic_area_km2']} km² geodesic.")
        w(f"- A 64×64 tile covers {sa['tile_ground_size_m']} m × {sa['tile_ground_size_m']} m = 0.4096 km². "
          f"The core holds **{sa['core_full_tiles_64px']} full tiles**; the wide area holds {sa['wide_full_tiles_64px']}.\n")

    w("## 2. Data\n")
    for p, meta in metas.items():
        w(f"- **Sentinel-2 {p}:** {meta['source']}; window {meta['date_range'][0]} → {meta['date_range'][1]}; "
          f"{len(meta.get('dates_used', [])) or meta.get('n_scenes')} acquisition dates "
          f"({', '.join(meta.get('dates_used', []))}); scene filter < {meta['scene_cloud_filter_pct']}% cloud; "
          f"pixel mask: {meta['pixel_mask']}; median composite; valid pixels {100 * meta['valid_fraction']:.1f}%.")
    if lab:
        w(f"- **Open Buildings V3** (confidence ≥ {cfg['labels']['min_confidence']}): {lab['kept']:,} footprints in the wide area, "
          f"median footprint {lab['footprint_area_median_m2']:.0f} m²; {lab['footprints_smaller_than_one_pixel_pct']:.1f}% are smaller "
          f"than one 10 m pixel. Total roof area {lab['footprint_area_km2']:.2f} km²; rasterized mask "
          f"(coverage ≥ {cfg['labels']['coverage_threshold']}) {lab['mask_building_area_km2']:.2f} km² "
          f"({100 * lab['mask_building_share']:.1f}% of pixels).\n")

    w("## 3. Tiles and spatial split\n")
    if til:
        w(f"- {til['candidate_full_tiles']} candidate full tiles → **{til['kept_tiles']} kept** "
          f"({til['dropped_low_valid']} dropped for < {100 * cfg['tiling']['min_valid_fraction']:.0f}% valid pixels).")
        w(f"- Tiles per split: {til['tiles_per_split']}; blocks per split (4×4-tile blocks): {til['blocks_per_split']}.")
        w(f"- Mean building-pixel share per split: {til['building_pixel_share_per_split']}.\n")

    w("## 4. Training\n")
    if tr:
        w(f"- Device: {tr['device']} {tr['gpu'] or ''}; {tr['n_parameters']:,} parameters; {tr['epochs_run']} epochs "
          f"(best epoch {tr['best_epoch']}, early stopping patience {cfg['training']['early_stopping_patience']}); "
          f"{tr['training_seconds']} s. pos_weight {tr['pos_weight']:.2f}.")
        w(f"- Best validation IoU (threshold 0.5): {_f(tr['best_val_iou_at_0.5'])}.\n")

    w("## 5. Pixel-level test metrics (held-out spatial test blocks)\n")
    if m:
        w(f"Threshold {m['threshold']['value']:.2f}, selected on the validation split. {m['n_test_tiles']} test tiles in "
          f"{m['n_test_blocks']} blocks; {100 * m['test_reference_building_pixel_share']:.1f}% of test pixels are reference buildings.\n")
        w("| Model | Precision | Recall | F1 | IoU | Accuracy |")
        w("|---|---|---|---|---|---|")
        for name, key in (("U-Net (val-selected threshold)", "unet_global"), ("U-Net (threshold 0.5)", "unet_global_at_threshold_0.5"),
                          ("Random forest, per pixel", "baseline_random_forest_pixelwise"), ("All-background", "baseline_all_background")):
            s = m[key]
            w(f"| {name} | {_f(s['precision'])} | {_f(s['recall'])} | {_f(s['f1'])} | {_f(s['iou'])} | {_f(s['accuracy'])} |")
        t = m["unet_per_tile_mean"]
        w(f"\nPer-tile mean (U-Net): IoU {_f(t['iou'])} over {t['iou_n_tiles_defined']} tiles with a defined IoU, F1 {_f(t['f1'])}. "
          f"Background-inclusive 2-class mean IoU: {_f(m['unet_global']['mean_iou_2class'])}. "
          f"Identity check F1 = 2·IoU/(1+IoU): {_f(2 * m['unet_global']['iou'] / (1 + m['unet_global']['iou']), 4)} vs F1 {_f(m['unet_global']['f1'], 4)}.\n")

    w(f"## 6. Building coverage and change, {e} → {l}\n")
    if ch:
        for area in ("core", "wide"):
            a = ch[area]["all_changes"]
            hc = ch[area][f"high_confidence_margin_{cfg['change']['confidence_margin']}"]
            w(f"**{area.capitalize()} area** (valid analysed {a['valid_analysed_area_km2']:.2f} km² of {a['study_area_km2']:.2f} km²)\n")
            w("| Quantity | All changes (km²) | % of valid | High-confidence (km²) |")
            w("|---|---|---|---|")
            for k, lab_ in (("earlier_built_up", f"Built-up {e}"), ("later_built_up", f"Built-up {l}"),
                            ("gross_gain", "Gross potential gain"), ("gross_loss", "Gross potential loss"),
                            ("net_change", "Net change"), ("persistent", "Persistent built-up")):
                w(f"| {lab_} | {a[k + '_km2']:.3f} | {a[k + '_pct_of_valid']:.1f}% | {hc[k + '_km2']:.3f} |")
            w("")
        r = ch["core_reference"]
        w(f"Context (core): Open Buildings V3 rasterized mask {r['open_buildings_mask_area_km2']:.2f} km², actual roof area "
          f"{r['open_buildings_roof_area_km2']:.2f} km²; U-Net 2023 map {r['unet_2023_built_up_km2']:.2f} km².\n")
        w("Gains/losses are **classification differences**, not confirmed construction or demolition.\n")

    w("## 7. Object-level evaluation (test tiles)\n")
    if ob:
        o = ob["region_level_one_to_one"]
        w(f"Objects = {ob['object_definition']}. Matching: {o['matching']}.\n")
        w(f"- Predicted regions {o['n_pred']}, reference regions {o['n_ref']}: TP {o['tp']}, FP {o['fp']}, FN {o['fn']} → "
          f"**precision {_f(o['precision'])}, recall {_f(o['recall'])}**. Merges {o['pred_merging_multiple_refs']}, splits {o['refs_split_across_multiple_preds']}.")
        o2 = ob["region_level_one_to_one_iou_0.25"]
        w(f"- Sensitivity at IoU ≥ 0.25: precision {_f(o2['precision'])}, recall {_f(o2['recall'])}.")
        fd = ob["footprint_level_detection_rates"]
        if "building_detection_rate" in fd:
            w(f"- Footprint level ({fd['rule']}): building detection rate {_f(fd['building_detection_rate'])} of {fd['n_footprints']:,} "
              f"footprints; region support rate {_f(fd['region_support_rate'])}.\n")

    w("## 8. Resume claims vs measured results\n")
    w("| Resume claim | Measured here | Verdict |")
    w("|---|---|---|")
    if sa and til:
        w(f"| {RESUME['tiles']} | core holds {sa['core_full_tiles_64px']} tiles; {til['kept_tiles']} tiles used for training/val/test in the wide area | "
          "≈60 tiles ≈ one 25 km² area; too few for a reliable spatially separated test set on its own |")
    if m:
        u = m["unet_global"]
        w(f"| IoU 69% | {100 * u['iou']:.1f}% | {'reproduced' if abs(u['iou'] - .69) < .03 else 'not reproduced'} |")
        w(f"| F1 82% | {100 * u['f1']:.1f}% | {'reproduced' if abs(u['f1'] - .82) < .03 else 'not reproduced'}; the two resume figures are consistent with each other (69% IoU ⇔ {200 * .69 / 1.69:.1f}% F1) |")
    if inv:
        w(f"| 21 km² built-up growth | core gross gain {ch['core']['all_changes']['gross_gain_km2']:.2f} km²; closest quantity to 21 km²: "
          f"'{inv['closest_quantity']}' = {inv['closest_km2']:.2f} km² | the maximum gain physically possible in the core is "
          f"{inv['max_possible_gain_in_core_km2']:.2f} km², so 21 km² cannot be growth inside a 25 km² area; see `resume_21km2_investigation.csv` |")
        wc = ch["wide"]["all_changes"]
        whc = ch["wide"][f"high_confidence_margin_{cfg['change']['confidence_margin']}"]
        w(f"\n**On the 21 km² figure.** Inside the 25 km² core, built-up area was {ch['core']['all_changes']['earlier_built_up_km2']:.2f} km² "
          f"({e}) and {ch['core']['all_changes']['later_built_up_km2']:.2f} km² ({l}). The only measured quantity near 21 km² is the "
          f"*gross* potential gain over the {wc['study_area_km2']:.0f} km² wide area ({wc['gross_gain_km2']:.2f} km²). That area's size was a "
          f"training choice made in this project, the quantity is unvalidated classification change, its net change is "
          f"{wc['net_change_km2']:.2f} km², and its high-confidence gain is only {whc['gross_gain_km2']:.2f} km². The numerical "
          "closeness is therefore not evidence that the resume figure is reproduced. The most likely readings of the original claim are "
          "(a) total built-up area, or (b) gross gain over a study area much larger than 25 km².")
    if ob:
        o = ob["region_level_one_to_one"]
        w(f"| Object precision 85% / recall 79% | {100 * o['precision']:.1f}% / {100 * o['recall']:.1f}% (region-level, IoU ≥ 0.5) | "
          f"{'reproduced' if abs(o['precision'] - .85) < .03 and abs(o['recall'] - .79) < .03 else 'not reproduced'} |")
    w("")

    w("## 9. Limitations\n")
    for s in [
        "Labels: Open Buildings V3 is a single snapshot derived from ~2021-2023 high-resolution imagery; it misses some buildings, contains false detections, and is not dated per building. It is a reference, not ground truth.",
        "There is no independent building reference for 2020 or 2025, so the change maps cannot be validated; differences include errors from both years' classifications.",
        "10 m pixels are larger than many Varanasi buildings; the model maps building-dominated pixels/urban fabric, not individual roofs. Object metrics therefore describe contiguous built-up regions.",
        "Composites from different years differ in atmosphere (haze, aerosols), sensor (S2A/B/C) and processing baseline; this creates spurious change.",
        "Spatial blocks reduce but do not remove spatial autocorrelation between train and test (blocks touch at their edges).",
        "Metrics depend on the label rasterization rule (coverage threshold 0.25) and confidence filter (0.70).",
    ]:
        w(f"- {s}")
    text = "\n".join(L) + "\n"
    (PATHS.reports / "final_report.md").write_text(text, encoding="utf-8")
    return text
