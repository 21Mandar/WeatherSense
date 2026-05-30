"""
Phase 2 Stage 5: full BDD100K benchmark across 3 YOLO models x 13 conditions.

Evaluates YOLOv8n, YOLOv8s, YOLOv8m on the full BDD100K validation set
(10,000 images) under:
    clean baseline
    rain {light, medium, heavy}
    fog  {light, medium, heavy}
    night {light, medium, heavy}
    motion_blur {light, medium, heavy}

Total: 3 models * 13 conditions * 10,000 images = 390,000 inferences.

Designed for an overnight run:
- Saves CSV after each (model, condition) pair completes (39 files total)
- Resumable: skips (model, condition) pairs whose CSV already exists
- Logs to both terminal and logs/stage5_run.log via tee (when invoked properly)

Recommended invocation:
    caffeinate -i python notebooks/full_benchmark.py 2>&1 | tee logs/stage5_run.log

Total runtime estimate (M1 Mac CPU): ~8 hours.
"""
import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.detection.bdd_dataset import BDDDataset, BDD_CLASSES
from src.detection.yolo_runner import YOLORunner
from src.evaluation.map_calculator import MAPEvaluator
from src.augmentations.rain import apply_rain
from src.augmentations.fog import apply_fog
from src.augmentations.night import apply_night
from src.augmentations.motion_blur import apply_motion_blur


# ----- Paths -----
BDD_ROOT = Path.home() / "datasets" / "bdd100k" / "archive"
IMAGES_DIR = BDD_ROOT / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
LABELS_JSON = BDD_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "stage5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----- Configuration -----
MODELS = ["yolov8n", "yolov8s", "yolov8m"]
DEVICE = None  # None = auto (CPU on M1)


# ----- Conditions -----
def apply_clean(image, intensity):
    """Identity for the clean baseline."""
    return image


CONDITIONS = [
    ("clean",              apply_clean,        "n/a"),
    ("rain_light",         apply_rain,         "light"),
    ("rain_medium",        apply_rain,         "medium"),
    ("rain_heavy",         apply_rain,         "heavy"),
    ("fog_light",          apply_fog,          "light"),
    ("fog_medium",         apply_fog,          "medium"),
    ("fog_heavy",          apply_fog,          "heavy"),
    ("night_light",        apply_night,        "light"),
    ("night_medium",       apply_night,        "medium"),
    ("night_heavy",        apply_night,        "heavy"),
    ("motion_blur_light",  apply_motion_blur,  "light"),
    ("motion_blur_medium", apply_motion_blur,  "medium"),
    ("motion_blur_heavy",  apply_motion_blur,  "heavy"),
]


# ----- Helpers -----
def csv_path_for(model_name, condition_label):
    return RESULTS_DIR / f"{model_name}_{condition_label}.csv"


def write_condition_csv(path, model_name, condition_label, intensity, n_images,
                        results, total_seconds):
    """Write a single (model, condition) result row to CSV."""
    fieldnames = (
        ["model", "condition", "intensity", "n_images", "runtime_seconds",
         "map_50", "map_50_95"] + BDD_CLASSES
    )
    row = {
        "model":           model_name,
        "condition":       condition_label,
        "intensity":       intensity,
        "n_images":        n_images,
        "runtime_seconds": f"{total_seconds:.1f}",
        "map_50":          f"{results['map_50']:.4f}",
        "map_50_95":       f"{results['map_50_95']:.4f}",
    }
    for cls in BDD_CLASSES:
        ap = results["per_class"][cls]["ap_50_95"]
        if np.isnan(ap) or ap < 0:
            row[cls] = ""
        else:
            row[cls] = f"{ap:.4f}"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def evaluate_condition(model_name, runner, condition_label, aug_fn, intensity,
                       gt_samples):
    """Run one (model, condition) evaluation. Returns the mAP results dict."""
    evaluator = MAPEvaluator()
    skipped = 0
    start = time.time()

    for i, gt in enumerate(gt_samples, start=1):
        # Load image fresh from disk each iteration.
        # SSD reads are fast on M1; this trades ~20-30% runtime for ~25x lower RAM.
        clean_image = cv2.imread(str(gt["image_path"]))
        if clean_image is None:
            skipped += 1
            continue

        augmented = aug_fn(clean_image, intensity)
        preds = runner.predict(augmented)
        evaluator.add_sample(preds, gt)

        # Progress every 1000 images
        if i % 1000 == 0:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (len(gt_samples) - i) / rate
            print(f"    [{i:>5}/{len(gt_samples)}] "
                  f"{elapsed:.0f}s elapsed, {rate:.1f} img/s, ETA {eta:.0f}s",
                  flush=True)


    inference_time = time.time() - start
    print(f"    Computing mAP via pycocotools ...", flush=True)
    t0 = time.time()
    results = evaluator.compute()
    compute_time = time.time() - t0
    print(f"    mAP@50 = {results['map_50']:.4f}  "
          f"mAP@50:95 = {results['map_50_95']:.4f}  "
          f"(infer {inference_time:.0f}s + compute {compute_time:.0f}s)",
          flush=True)

    return results, inference_time + compute_time


def main():
    overall_start = time.time()
    started_at = datetime.now()
    print(f"========================================================", flush=True)
    print(f"Phase 2 Stage 5: full benchmark", flush=True)
    print(f"Started at {started_at.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"========================================================", flush=True)

    # Load BDD val (all 10,000 images, no filter)
    print(f"\nLoading BDD val (all 10,000 images)...", flush=True)
    dataset = BDDDataset(images_dir=IMAGES_DIR, labels_json=LABELS_JSON)
    print(f"Dataset size: {len(dataset)}", flush=True)

    # Pre-load all GT + clean images into memory.
    # 10K images x ~250KB each = ~2.5 GB - fits easily on a 16GB M1.
    # Pre-load just GT metadata (lightweight - no images yet).
    # Loading 10K full images into RAM would need ~27 GB; we'll read each
    # image on demand from disk during each (model, condition) evaluation.
    print(f"\nPre-loading {len(dataset)} GT samples (metadata only) ...", flush=True)
    pre_start = time.time()
    gt_samples = []
    for idx in range(len(dataset)):
        gt_samples.append(dataset[idx])
    pre_elapsed = time.time() - pre_start
    print(f"Loaded {len(gt_samples)} GT entries in {pre_elapsed:.0f}s", flush=True)

    # Loop: model-major
    total_pairs = len(MODELS) * len(CONDITIONS)
    pair_index = 0
    skipped_pairs = 0

    for model_name in MODELS:
        print(f"\n{'#' * 60}", flush=True)
        print(f"Model: {model_name}", flush=True)
        print(f"{'#' * 60}", flush=True)

        runner = None  # lazy-load only if needed

        for condition_label, aug_fn, intensity in CONDITIONS:
            pair_index += 1
            out_csv = csv_path_for(model_name, condition_label)

            # Resume support: skip if already done
            if out_csv.exists():
                print(f"\n[{pair_index}/{total_pairs}] {model_name} / "
                      f"{condition_label}: SKIP (already done at {out_csv.name})",
                      flush=True)
                skipped_pairs += 1
                continue

            # Lazy-load model only when first non-skipped condition appears
            if runner is None:
                print(f"\nLoading {model_name} ...", flush=True)
                runner = YOLORunner(model_name, device=DEVICE)

            print(f"\n[{pair_index}/{total_pairs}] {model_name} / "
                  f"{condition_label} ({intensity})", flush=True)

            results, total_seconds = evaluate_condition(
                model_name, runner, condition_label, aug_fn, intensity, gt_samples
            )

            write_condition_csv(
                out_csv, model_name, condition_label, intensity,
                len(gt_samples), results, total_seconds
            )
            print(f"    Saved -> {out_csv.name}", flush=True)

            # Running ETA based on actual completed work
            completed_pairs = pair_index - skipped_pairs
            elapsed_total = time.time() - overall_start
            avg_seconds_per_pair = elapsed_total / completed_pairs if completed_pairs else 0
            remaining_pairs = total_pairs - pair_index
            eta_seconds = avg_seconds_per_pair * remaining_pairs
            eta_human = str(timedelta(seconds=int(eta_seconds)))
            print(f"    Overall ETA: {eta_human} ({remaining_pairs} pairs left)",
                  flush=True)

    overall_elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}", flush=True)
    print(f"All pairs complete in {overall_elapsed/3600:.2f} hours "
          f"({skipped_pairs} skipped)", flush=True)
    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"\nResults: {RESULTS_DIR}", flush=True)
    print(f"Files written: {len(list(RESULTS_DIR.glob('*.csv')))}", flush=True)


if __name__ == "__main__":
    main()