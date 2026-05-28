"""
Scene-stratified YOLO inference validation.

Produces two grids beyond the random-sample grid from check_yolo_inference.py:

Grid A: 4 weather conditions x 2 images = 8 panels
    Rows: clear, rainy, snowy, foggy
    Columns: 2 random samples per row

Grid B: 3 times of day x 2 images = 6 panels
    Rows: daytime, dawn/dusk, night
    Columns: 2 random samples per row

Predictions shown at conf >= 0.25 for readability; the full prediction set
is what Stage 3's mAP calculator will integrate over.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.detection.bdd_dataset import BDDDataset, BDD_CLASSES
from src.detection.yolo_runner import YOLORunner


BDD_ROOT = Path.home() / "datasets" / "bdd100k" / "archive"
IMAGES_DIR = BDD_ROOT / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
LABELS_JSON = BDD_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "yolo_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_COLORS = [
    (0, 0, 255), (0, 100, 255), (0, 255, 0), (0, 200, 200), (255, 200, 0),
    (255, 0, 255), (200, 100, 0), (200, 200, 200), (0, 255, 255), (255, 100, 100),
]

VIZ_CONF_THRESHOLD = 0.25
SAMPLES_PER_FILTER = 2


def draw_solid_box(image, box, color, label):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(image, label, (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_dashed_box(image, box, color, label):
    x1, y1, x2, y2 = box.astype(int)
    dash_len, gap_len = 8, 6
    x = x1
    while x < x2:
        cv2.line(image, (x, y1), (min(x + dash_len, x2), y1), color, 2)
        cv2.line(image, (x, y2), (min(x + dash_len, x2), y2), color, 2)
        x += dash_len + gap_len
    y = y1
    while y < y2:
        cv2.line(image, (x1, y), (x1, min(y + dash_len, y2)), color, 2)
        cv2.line(image, (x2, y), (x2, min(y + dash_len, y2)), color, 2)
        y += dash_len + gap_len
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y2), (x1 + tw + 4, y2 + th + 4), color, -1)
    cv2.putText(image, label, (x1 + 2, y2 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_image(image, gt, preds):
    out = image.copy()
    for box, cid in zip(gt["boxes"], gt["class_ids"]):
        draw_solid_box(out, box, CLASS_COLORS[cid], f"GT:{BDD_CLASSES[cid]}")
    keep = preds["confidences"] >= VIZ_CONF_THRESHOLD
    for box, cid, conf in zip(
        preds["boxes"][keep], preds["class_ids"][keep], preds["confidences"][keep]
    ):
        draw_dashed_box(out, box, CLASS_COLORS[cid], f"P:{BDD_CLASSES[cid]} {conf:.2f}")
    return out


def add_caption(panel, text, band_h=50, font_scale=0.65):
    h, w = panel.shape[:2]
    band = np.full((band_h, w, 3), (40, 40, 40), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x = (w - tw) // 2
    y = (band_h + th) // 2
    cv2.putText(band, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, panel])


def add_row_header(row_image, text, band_h=70):
    """Add a tall labeled band above an entire row to identify the filter."""
    h, w = row_image.shape[:2]
    band = np.full((band_h, w, 3), (20, 20, 60), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2)
    x = (w - tw) // 2
    y = (band_h + th) // 2
    cv2.putText(band, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([band, row_image])


def build_row_for_filter(runner, filter_name, filter_value, attr_name, seed):
    """
    Load BDD with the given filter, sample N images, return a horizontally
    stacked row of annotated panels.
    """
    print(f"\n--- {filter_name}: {filter_value} ---")
    kwargs = {
        "images_dir": IMAGES_DIR,
        "labels_json": LABELS_JSON,
        attr_name: [filter_value],
    }
    dataset = BDDDataset(**kwargs)

    if len(dataset) < SAMPLES_PER_FILTER:
        print(f"⚠ Only {len(dataset)} images match - using all of them")
        indices = list(range(len(dataset)))
    else:
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), SAMPLES_PER_FILTER)

    panels = []
    for idx in indices:
        sample = dataset[idx]
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            print(f"⚠ Could not read {sample['image_path']}")
            continue

        preds = runner.predict(image)
        n_gt = len(sample["boxes"])
        n_pred_above = int((preds["confidences"] >= VIZ_CONF_THRESHOLD).sum())
        n_pred_total = len(preds["boxes"])

        annotated = annotate_image(image, sample, preds)
        caption = (
            f"{sample['name']}  |  GT: {n_gt}  |  "
            f"Pred>={VIZ_CONF_THRESHOLD}: {n_pred_above}/{n_pred_total}"
        )
        panel = add_caption(annotated, caption)
        panels.append(panel)
        print(f"  {sample['name']}: GT={n_gt}, pred>={VIZ_CONF_THRESHOLD}: {n_pred_above}")

    if not panels:
        return None
    # Pad with blank if we got fewer than expected
    while len(panels) < SAMPLES_PER_FILTER:
        h, w = panels[0].shape[:2]
        panels.append(np.zeros((h, w, 3), dtype=np.uint8))

    row = np.hstack(panels)
    row_with_header = add_row_header(row, filter_value.upper())
    return row_with_header


def build_grid(runner, filter_values, attr_name, base_seed):
    """Build a stacked grid: one row per filter value."""
    rows = []
    for i, value in enumerate(filter_values):
        row = build_row_for_filter(
            runner, attr_name, value, attr_name, seed=base_seed + i
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return None

    # All rows should be same width; pad if not
    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded_rows.append(r)

    return np.vstack(padded_rows)


def main():
    print("Loading YOLOv8n...")
    runner = YOLORunner("yolov8n")

    # ---- Grid A: weather conditions ----
    print("\n" + "=" * 60)
    print("Grid A: YOLO performance across weather conditions")
    print("=" * 60)
    weather_values = ["clear", "rainy", "snowy", "foggy"]
    grid_a = build_grid(runner, weather_values, "weather_filter", base_seed=100)
    if grid_a is not None:
        out_path = OUT_DIR / "yolo_by_weather.jpg"
        cv2.imwrite(str(out_path), grid_a)
        print(f"\n✓ Grid A saved to {out_path}")

    # ---- Grid B: time of day ----
    print("\n" + "=" * 60)
    print("Grid B: YOLO performance across times of day")
    print("=" * 60)
    timeofday_values = ["daytime", "dawn/dusk", "night"]
    grid_b = build_grid(runner, timeofday_values, "timeofday_filter", base_seed=200)
    if grid_b is not None:
        out_path = OUT_DIR / "yolo_by_timeofday.jpg"
        cv2.imwrite(str(out_path), grid_b)
        print(f"\n✓ Grid B saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()