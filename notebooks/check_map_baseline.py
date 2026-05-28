"""
Sanity check for the mAP calculator.

Runs YOLOv8n on a small sample of BDD val (clear + daytime) and computes mAP
using our MAPEvaluator. Verifies the result lands in a plausible range before
we trust the calculator for the full Stage 5 benchmark.

Verified baseline (200 clear+daytime BDD val images, YOLOv8n):
- mAP@50:    0.2225
- mAP@50:95: 0.1279
Per-class AP@50:95 in this run:
  car:           0.32   (dominant class)
  truck:         0.20
  pedestrian:    0.20
  bus:           0.19
  motorcycle:    0.08   (sparse class, noisy)
  bicycle:       0.08   (sparse class, noisy)
  traffic light: 0.07
  traffic sign:  0.02   (limited - only stop_sign maps from COCO)
  rider:         0.00   (no COCO mapping - documented)
  train:         n/a    (15 GT total in BDD val, rare in samples)

If overall mAP@50 is < 0.10 or > 0.50, something is likely wrong - we debug
before scaling up.
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.detection.bdd_dataset import BDDDataset
from src.detection.yolo_runner import YOLORunner
from src.evaluation.map_calculator import MAPEvaluator, format_results_table


BDD_ROOT = Path.home() / "datasets" / "bdd100k" / "archive"
IMAGES_DIR = BDD_ROOT / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
LABELS_JSON = BDD_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"


# Configuration
N_IMAGES = 200       # Small enough for ~5 min runtime on CPU; large enough for stable mAP
DEVICE = None        # None = auto (CPU on M1). Set "mps" for Apple GPU acceleration.
SEED = 42


def main():
    # Build the clean-baseline dataset
    print("Loading BDD val filtered to clear + daytime ...")
    dataset = BDDDataset(
        images_dir=IMAGES_DIR,
        labels_json=LABELS_JSON,
        weather_filter=["clear"],
        timeofday_filter=["daytime"],
    )
    print(f"Clean baseline pool size: {len(dataset)}")

    # Sample N images
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), min(N_IMAGES, len(dataset)))
    print(f"Sampling {len(indices)} images for sanity check")

    # Load YOLO
    print("\nLoading YOLOv8n ...")
    runner = YOLORunner("yolov8n", device=DEVICE)

    # Run inference + accumulate predictions
    evaluator = MAPEvaluator()

    print("\nRunning inference + accumulating predictions ...")
    t_start = time.time()
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            print(f"  ⚠ Could not read {sample['image_path']}")
            continue

        preds = runner.predict(image)
        evaluator.add_sample(preds, sample)

        # Progress every 25 images
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(indices) - i - 1) / rate
            print(f"  [{i + 1:>3}/{len(indices)}]  "
                  f"{elapsed:5.1f}s elapsed, {rate:.2f} img/s, ETA {eta:.0f}s")

    elapsed = time.time() - t_start
    print(f"\nInference complete: {elapsed:.1f}s total ({len(indices)/elapsed:.2f} img/s)")

    # Compute mAP
    print("\nComputing mAP (pycocotools backend - may take ~30s) ...")
    t0 = time.time()
    results = evaluator.compute()
    print(f"Computation took {time.time() - t0:.1f}s")

    # Pretty-print
    print()
    print(format_results_table(results, f"Clean baseline ({len(indices)} clear+daytime images)"))

    # Sanity check the headline number
    print()
    print("=== Sanity check ===")
    map_50 = results["map_50"]
    if 0.10 <= map_50 <= 0.50:
        print(f"  ✓ mAP@50 = {map_50:.4f} is in the plausible range [0.10, 0.50]")
        print(f"  ✓ Calculator appears to work correctly")
    elif map_50 < 0.10:
        print(f"  ✗ mAP@50 = {map_50:.4f} is suspiciously low (< 0.10)")
        print(f"    Possible issues: GT boxes wrong format, predictions empty,")
        print(f"    class mapping broken, or coordinate system mismatch")
    else:
        print(f"  ✗ mAP@50 = {map_50:.4f} is suspiciously high (> 0.50)")
        print(f"    Either we got lucky with the sample or something is wrong")


if __name__ == "__main__":
    main()