"""
Visual validation for BDD100K loader.

Loads 6 random images from the BDD val set with their ground truth bounding
boxes drawn on top. Saves a 2x3 grid to outputs/bdd_check/ for inspection.

This is the Phase 1-style sanity check: we look at the output and verify boxes
land on the right objects with the right class labels before building anything
on top of the loader.
"""
import random
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.detection.bdd_dataset import BDDDataset, BDD_CLASSES


# Paths - edit if your BDD100K lives elsewhere
BDD_ROOT = Path.home() / "datasets" / "bdd100k" / "archive"
IMAGES_DIR = BDD_ROOT / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
LABELS_JSON = BDD_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

# Output location (inside project, gitignored)
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "bdd_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Distinct colors per class for readability (BGR for OpenCV)
CLASS_COLORS = [
    (0, 0, 255),       # pedestrian      - red
    (0, 100, 255),     # rider           - orange
    (0, 255, 0),       # car             - green
    (0, 200, 200),     # truck           - yellow-olive
    (255, 200, 0),     # bus             - sky blue
    (255, 0, 255),     # train           - magenta
    (200, 100, 0),     # motorcycle      - dark blue
    (200, 200, 200),   # bicycle         - light gray
    (0, 255, 255),     # traffic light   - yellow
    (255, 100, 100),   # traffic sign    - blue-ish
]


def draw_boxes(image: np.ndarray, boxes: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """Draw ground truth bounding boxes with class labels on the image."""
    out = image.copy()
    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS[class_id]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = BDD_CLASSES[class_id]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return out


def add_caption(panel: np.ndarray, text: str) -> np.ndarray:
    """Prepend a dark band with white text above the panel."""
    h, w = panel.shape[:2]
    band_h = 50
    band = np.full((band_h, w, 3), (40, 40, 40), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    x = (w - tw) // 2
    y = (band_h + th) // 2
    cv2.putText(band, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, panel])


def main():
    print("Initializing BDD dataset...")
    dataset = BDDDataset(images_dir=IMAGES_DIR, labels_json=LABELS_JSON)

    print(f"\nDataset has {len(dataset)} entries.")
    print("\nScene breakdown:")
    summary = dataset.scene_summary()
    for axis, counts in summary.items():
        print(f"  {axis}:")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {k}: {v}")

    # Pick 6 random images. Fixed seed for reproducibility.
    random.seed(42)
    indices = random.sample(range(len(dataset)), 6)

    panels = []
    for idx in indices:
        sample = dataset[idx]
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            print(f"⚠ Could not read {sample['image_path']}")
            continue

        annotated = draw_boxes(image, sample["boxes"], sample["class_ids"])

        # Caption with image name, count, and scene attrs
        n_objects = len(sample["boxes"])
        scene = sample["scene"]
        caption = (
            f"{sample['name']}  |  {n_objects} objects  |  "
            f"{scene.get('weather','?')}/{scene.get('timeofday','?')}/{scene.get('scene','?')}"
        )
        panel = add_caption(annotated, caption)
        panels.append(panel)

        print(f"  ✓ {sample['name']}: {n_objects} objects, {scene}")

    # Build 2x3 grid
    if len(panels) == 6:
        row1 = np.hstack(panels[:3])
        row2 = np.hstack(panels[3:])
        grid = np.vstack([row1, row2])
        out_path = OUT_DIR / "bdd_val_sample_with_boxes.jpg"
        cv2.imwrite(str(out_path), grid)
        print(f"\n✓ Grid saved to {out_path}")
    else:
        print(f"\n⚠ Only got {len(panels)} panels, expected 6")


if __name__ == "__main__":
    main()