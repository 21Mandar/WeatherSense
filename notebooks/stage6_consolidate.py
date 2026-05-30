"""
Phase 2 Stage 6 (1/3): consolidate 39 per-(model, condition) CSVs into
one master long-format CSV.

Output: results/phase2_master_results.csv

Schema:
- model:           yolov8n / yolov8s / yolov8m
- weather_type:    clean / rain / fog / night / motion_blur
- intensity:       n/a / light / medium / heavy
- condition:       full label, e.g. "rain_heavy"
- n_images:        10000
- runtime_seconds: per-pair runtime
- map_50:          mAP @ IoU 0.50
- map_50_95:       mAP @ IoU 0.50:0.95
- degradation_pct: (clean_map50 - map50) / clean_map50 * 100   [0 for clean rows]
- pedestrian, rider, car, truck, bus, train, motorcycle, bicycle,
  traffic light, traffic sign:  per-class AP@50:95 values
"""
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.bdd_dataset import BDD_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE5_DIR = PROJECT_ROOT / "results" / "stage5"
MASTER_CSV = PROJECT_ROOT / "results" / "phase2_master_results.csv"


def parse_condition(condition_label):
    """Split 'rain_heavy' -> ('rain', 'heavy'). 'clean' -> ('clean', 'n/a')."""
    if condition_label == "clean":
        return "clean", "n/a"
    # All other conditions are of the form "<weather_type>_<intensity>"
    # weather_type can be multi-word, so rsplit at the last underscore
    weather_type, _, intensity = condition_label.rpartition("_")
    return weather_type, intensity


def main():
    if not STAGE5_DIR.exists():
        raise FileNotFoundError(f"Stage 5 results directory not found: {STAGE5_DIR}")

    csv_files = sorted(STAGE5_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {STAGE5_DIR}")

    if len(csv_files) != 39:
        print(f"  ⚠ Expected 39 files, found {len(csv_files)}. Continuing anyway.")

    # Read all rows
    all_rows = []
    for csv_path in csv_files:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    print(f"Loaded {len(all_rows)} result rows total")

    # Build clean baseline lookup: model -> clean mAP@50
    clean_baselines = {
        row["model"]: float(row["map_50"])
        for row in all_rows
        if row["condition"] == "clean"
    }
    print(f"Clean baselines: {clean_baselines}")

    # Write master CSV
    fieldnames = (
        ["model", "weather_type", "intensity", "condition",
         "n_images", "runtime_seconds",
         "map_50", "map_50_95", "degradation_pct"]
        + BDD_CLASSES
    )

    with open(MASTER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_rows:
            model = row["model"]
            condition = row["condition"]
            weather_type, intensity = parse_condition(condition)
            map_50 = float(row["map_50"])
            clean_map_50 = clean_baselines[model]
            if condition == "clean":
                degradation_pct = 0.0
            else:
                degradation_pct = (clean_map_50 - map_50) / clean_map_50 * 100.0

            out_row = {
                "model":           model,
                "weather_type":    weather_type,
                "intensity":       intensity,
                "condition":       condition,
                "n_images":        row["n_images"],
                "runtime_seconds": row["runtime_seconds"],
                "map_50":          f"{map_50:.4f}",
                "map_50_95":       f"{float(row['map_50_95']):.4f}",
                "degradation_pct": f"{degradation_pct:.2f}",
            }
            for cls in BDD_CLASSES:
                out_row[cls] = row.get(cls, "")
            writer.writerow(out_row)

    print(f"\n✓ Master CSV written: {MASTER_CSV}")
    print(f"  Rows: {len(all_rows)}")

    # Quick sanity preview
    print(f"\nSample rows (first 5):")
    with open(MASTER_CSV) as f:
        for i, line in enumerate(f):
            if i >= 6:
                break
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()