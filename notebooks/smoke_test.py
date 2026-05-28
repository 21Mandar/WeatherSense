"""
Phase 2 Stage 4: smoke test of the full evaluation pipeline.

For a fixed sample of 200 clear+daytime BDD val images, evaluates YOLOv8n
across 13 conditions:
    clean baseline
    rain {light, medium, heavy}
    fog  {light, medium, heavy}
    night {light, medium, heavy}
    motion_blur {light, medium, heavy}

The 'clean' result should match Stage 3's 0.2225 mAP@50 (same images,
same model, no augmentation).

Outputs:
- Terminal: per-condition mAP and per-class AP, plus a final degradation table
- CSV: results/phase2_smoke_test.csv with one row per condition

The CSV format is the same structure Stage 5 will produce at 10K-image scale,
so Stage 6 visualization code can be developed against this smaller dataset.

Total runtime estimate: ~30-40 minutes on M1 Mac CPU.
"""
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.detection.bdd_dataset import BDDDataset, BDD_CLASSES
from src.detection.yolo_runner import YOLORunner
from src.evaluation.map_calculator import MAPEvaluator, format_results_table
from src.augmentations.rain import apply_rain
from src.augmentations.fog import apply_fog
from src.augmentations.night import apply_night
from src.augmentations.motion_blur import apply_motion_blur


# Paths
BDD_ROOT = Path.home() / "datasets" / "bdd100k" / "archive"
IMAGES_DIR = BDD_ROOT / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
LABELS_JSON = BDD_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "phase2_smoke_test.csv"


# Configuration - matches Stage 3 sanity check so 'clean' row is comparable
N_IMAGES = 200
SEED = 42
MODEL_NAME = "yolov8n"
DEVICE = None  # None = auto (CPU on M1); pass "mps" for Apple GPU


# Identity function for the clean baseline - "no augmentation"
def apply_clean(image: np.ndarray, intensity: str) -> np.ndarray:
    return image


# All 13 evaluation conditions: (label, augmentation_fn, intensity)
# Order matters - matches the CSV row order
CONDITIONS = [
    ("clean",             apply_clean,        "n/a"),

    ("rain_light",        apply_rain,         "light"),
    ("rain_medium",       apply_rain,         "medium"),
    ("rain_heavy",        apply_rain,         "heavy"),

    ("fog_light",         apply_fog,          "light"),
    ("fog_medium",        apply_fog,          "medium"),
    ("fog_heavy",         apply_fog,          "heavy"),

    ("night_light",       apply_night,        "light"),
    ("night_medium",      apply_night,        "medium"),
    ("night_heavy",       apply_night,        "heavy"),

    ("motion_blur_light", apply_motion_blur,  "light"),
    ("motion_blur_medium",apply_motion_blur,  "medium"),
    ("motion_blur_heavy", apply_motion_blur,  "heavy"),
]


def evaluate_condition(label, aug_fn, intensity, sampled_data, runner):
    """
    Run inference + mAP for one condition.
    sampled_data is a list of (gt_dict, image_bgr) tuples - we pre-loaded
    GT and clean images once so we don't re-load them 13 times.
    """
    evaluator = MAPEvaluator()
    t_start = time.time()

    for gt, clean_image in sampled_data:
        augmented = aug_fn(clean_image, intensity)
        preds = runner.predict(augmented)
        evaluator.add_sample(preds, gt)

    inference_time = time.time() - t_start

    t0 = time.time()
    results = evaluator.compute()
    compute_time = time.time() - t0

    results["_meta"] = {
        "condition":   label,
        "intensity":   intensity,
        "n_images":    len(sampled_data),
        "infer_time_s":  inference_time,
        "compute_time_s": compute_time,
    }
    return results


def main():
    # Load dataset filtered to the clean baseline pool
    print("Loading BDD val (clear + daytime baseline pool)...")
    dataset = BDDDataset(
        images_dir=IMAGES_DIR,
        labels_json=LABELS_JSON,
        weather_filter=["clear"],
        timeofday_filter=["daytime"],
    )
    print(f"Baseline pool size: {len(dataset)}")

    # Sample N_IMAGES - same seed as Stage 3 so 'clean' result is comparable
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), min(N_IMAGES, len(dataset)))
    print(f"Sampling {len(indices)} images (seed={SEED})")

    # Pre-load GT + clean images once so we don't re-read from disk 13 times
    print("Pre-loading clean images and GT...")
    sampled_data = []
    for idx in indices:
        sample = dataset[idx]
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            print(f"  ⚠ Could not read {sample['image_path']}")
            continue
        sampled_data.append((sample, image))
    print(f"Loaded {len(sampled_data)} samples into memory")

    # Load YOLO
    print(f"\nLoading {MODEL_NAME}...")
    runner = YOLORunner(MODEL_NAME, device=DEVICE)

    # Run all 13 conditions
    all_results = []
    overall_start = time.time()

    for i, (label, aug_fn, intensity) in enumerate(CONDITIONS, start=1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(CONDITIONS)}] Condition: {label}")
        print(f"{'='*60}")

        results = evaluate_condition(label, aug_fn, intensity, sampled_data, runner)
        all_results.append(results)

        meta = results["_meta"]
        print(f"  Inference: {meta['infer_time_s']:.1f}s "
              f"({len(sampled_data) / meta['infer_time_s']:.1f} img/s)")
        print(f"  mAP@50    = {results['map_50']:.4f}")
        print(f"  mAP@50:95 = {results['map_50_95']:.4f}")

    overall_elapsed = time.time() - overall_start
    print(f"\nAll conditions complete in {overall_elapsed/60:.1f} minutes total")

    # Write CSV
    print(f"\nWriting CSV to {CSV_PATH} ...")
    fieldnames = ["condition", "intensity", "n_images", "map_50", "map_50_95"] + BDD_CLASSES

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {
                "condition":  r["_meta"]["condition"],
                "intensity":  r["_meta"]["intensity"],
                "n_images":   r["_meta"]["n_images"],
                "map_50":     f"{r['map_50']:.4f}",
                "map_50_95":  f"{r['map_50_95']:.4f}",
            }
            for cls in BDD_CLASSES:
                ap = r["per_class"][cls]["ap_50_95"]
                if np.isnan(ap):
                    row[cls] = ""
                elif ap < 0:
                    row[cls] = ""  # pycocotools sentinel - treat as missing
                else:
                    row[cls] = f"{ap:.4f}"
            writer.writerow(row)
    print(f"✓ CSV saved")

    # Final summary table
    print(f"\n{'='*70}")
    print("Phase 2 smoke test - degradation summary")
    print(f"{'='*70}")
    print(f"{'Condition':<22} {'mAP@50':>10} {'Δ vs clean':>12}")
    print("-" * 50)

    clean_map50 = all_results[0]["map_50"]
    for r in all_results:
        label = r["_meta"]["condition"]
        m50 = r["map_50"]
        delta = m50 - clean_map50
        delta_pct = (delta / clean_map50 * 100) if clean_map50 > 0 else 0
        delta_str = f"{delta:+.4f} ({delta_pct:+.1f}%)" if label != "clean" else ""
        print(f"{label:<22} {m50:>10.4f} {delta_str:>20}")

    print()
    print("Note: 'clean' row should match Stage 3 baseline (mAP@50 = 0.2225)")
    print("      Any large discrepancy suggests pipeline drift; check augmentation code")


if __name__ == "__main__":
    main()