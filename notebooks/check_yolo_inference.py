"""
Visual validation for YOLOv8 inference wrapper.

Side-by-side: ground truth (solid boxes) vs YOLOv8n predictions (dashed boxes)
on the same 6 random BDD images we visualized in Stage 1. Lets us eyeball
whether the wrapper + COCO-to-BDD mapping work correctly before scaling up.

Predictions kept at conf >= 0.25 for visualization clarity. (mAP runs use
conf >= 0.001 to sweep the full PR curve.)
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


# Same color palette as Stage 1 for consistency
CLASS_COLORS = [
    (0, 0, 255),       # 0 pedestrian
    (0, 100, 255),     # 1 rider
    (0, 255, 0),       # 2 car
    (0, 200, 200),     # 3 truck
    (255, 200, 0),     # 4 bus
    (255, 0, 255),     # 5 train
    (200, 100, 0),     # 6 motorcycle
    (200, 200, 200),   # 7 bicycle
    (0, 255, 255),     # 8 traffic light
    (255, 100, 100),   # 9 traffic sign
]

VIZ_CONF_THRESHOLD = 0.25  # only show predictions above this for clarity


def draw_solid_box(image, box, color, label):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(image, label, (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_dashed_box(image, box, color, label):
    """Dashed bbox by drawing short line segments along each side."""
    x1, y1, x2, y2 = box.astype(int)
    dash_len = 8
    gap_len = 6
    # Top edge
    x = x1
    while x < x2:
        cv2.line(image, (x, y1), (min(x + dash_len, x2), y1), color, 2)
        x += dash_len + gap_len
    # Bottom edge
    x = x1
    while x < x2:
        cv2.line(image, (x, y2), (min(x + dash_len, x2), y2), color, 2)
        x += dash_len + gap_len
    # Left edge
    y = y1
    while y < y2:
        cv2.line(image, (x1, y), (x1, min(y + dash_len, y2)), color, 2)
        y += dash_len + gap_len
    # Right edge
    y = y1
    while y < y2:
        cv2.line(image, (x2, y), (x2, min(y + dash_len, y2)), color, 2)
        y += dash_len + gap_len
    # Label background and text (same as solid box)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y2), (x1 + tw + 4, y2 + th + 4), color, -1)
    cv2.putText(image, label, (x1 + 2, y2 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_image(image, gt, preds):
    """Draw GT solid + predictions dashed onto the image."""
    out = image.copy()

    # Ground truth (solid, top label)
    for box, cid in zip(gt["boxes"], gt["class_ids"]):
        draw_solid_box(out, box, CLASS_COLORS[cid], f"GT:{BDD_CLASSES[cid]}")

    # Predictions (dashed, bottom label) - only above viz threshold
    keep = preds["confidences"] >= VIZ_CONF_THRESHOLD
    for box, cid, conf in zip(
        preds["boxes"][keep], preds["class_ids"][keep], preds["confidences"][keep]
    ):
        label = f"P:{BDD_CLASSES[cid]} {conf:.2f}"
        draw_dashed_box(out, box, CLASS_COLORS[cid], label)

    return out


def add_caption(panel, text):
    h, w = panel.shape[:2]
    band_h = 50
    band = np.full((band_h, w, 3), (40, 40, 40), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    x = (w - tw) // 2
    y = (band_h + th) // 2
    cv2.putText(band, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, panel])


def main():
    print("Loading BDD dataset...")
    dataset = BDDDataset(images_dir=IMAGES_DIR, labels_json=LABELS_JSON)

    print("Loading YOLOv8n (first run will download ~6 MB)...")
    runner = YOLORunner("yolov8n")

    # Same seed as Stage 1 -> same 6 images
    random.seed(42)
    indices = random.sample(range(len(dataset)), 6)

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
            f"Pred (conf>={VIZ_CONF_THRESHOLD}): {n_pred_above}/{n_pred_total}  |  "
            f"{sample['scene'].get('weather','?')}/{sample['scene'].get('timeofday','?')}"
        )
        panel = add_caption(annotated, caption)
        panels.append(panel)

        print(f"  ✓ {sample['name']}: GT={n_gt}, pred={n_pred_above} (above thresh)")

    if len(panels) == 6:
        row1 = np.hstack(panels[:3])
        row2 = np.hstack(panels[3:])
        grid = np.vstack([row1, row2])
        out_path = OUT_DIR / "yolo_vs_gt_sample.jpg"
        cv2.imwrite(str(out_path), grid)
        print(f"\n✓ Grid saved to {out_path}")
    else:
        print(f"\n⚠ Only got {len(panels)} panels")


if __name__ == "__main__":
    main()
