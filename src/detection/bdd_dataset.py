"""
BDD100K validation set loader for WeatherSense Phase 2.

Reads the legacy-format BDD100K detection labels JSON, filters detection-only
annotations (those with box2d), and exposes a Pythonic interface for inference
and evaluation.

Each sample yields:
    image_path: Path to the .jpg file
    boxes:      np.ndarray of shape (N, 4) in [x1, y1, x2, y2] pixel coords
    class_ids:  np.ndarray of shape (N,) with BDD class IDs (0-9)
    scene:      dict with 'weather', 'timeofday', 'scene' keys

Supports filtering by scene attributes (e.g., clear-weather daytime images
as the clean baseline for synthetic augmentation comparisons).
"""
import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


# BDD100K detection categories, in canonical order.
# Note: this is 0-indexed; the BDD documentation lists them 1-indexed.
# We use 0-indexed to match common Python/PyTorch conventions.
BDD_CLASSES = [
    "pedestrian",      # 0
    "rider",           # 1
    "car",             # 2
    "truck",           # 3
    "bus",             # 4
    "train",           # 5
    "motorcycle",      # 6
    "bicycle",         # 7
    "traffic light",   # 8
    "traffic sign",    # 9
]

BDD_CLASS_TO_ID = {name: idx for idx, name in enumerate(BDD_CLASSES)}


class BDDDataset:
    """Lazy iterator over BDD100K validation images + annotations."""

    def __init__(
        self,
        images_dir: Path,
        labels_json: Path,
        weather_filter: Optional[list[str]] = None,
        timeofday_filter: Optional[list[str]] = None,
        scene_filter: Optional[list[str]] = None,
    ):
        """
        Args:
            images_dir:   Directory containing the .jpg images
            labels_json:  Path to bdd100k_labels_images_val.json
            weather_filter:   If set, only include images where attributes.weather
                              is in this list. E.g., ['clear', 'partly cloudy'].
            timeofday_filter: Same idea for attributes.timeofday.
                              E.g., ['daytime'].
            scene_filter:     Same idea for attributes.scene.
        """
        self.images_dir = Path(images_dir)
        self.labels_json = Path(labels_json)

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_json.is_file():
            raise FileNotFoundError(f"Labels JSON not found: {self.labels_json}")

        print(f"Loading BDD labels from {self.labels_json} ...")
        with open(self.labels_json) as f:
            all_entries = json.load(f)
        print(f"Loaded {len(all_entries)} entries.")

        # Apply scene-attribute filters
        self.entries = []
        for entry in all_entries:
            attrs = entry.get("attributes", {})
            if weather_filter and attrs.get("weather") not in weather_filter:
                continue
            if timeofday_filter and attrs.get("timeofday") not in timeofday_filter:
                continue
            if scene_filter and attrs.get("scene") not in scene_filter:
                continue
            self.entries.append(entry)

        if len(self.entries) < len(all_entries):
            print(
                f"After filtering: {len(self.entries)} entries "
                f"({len(self.entries) / len(all_entries) * 100:.1f}% of total)"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[dict]:
        for entry in self.entries:
            yield self._parse_entry(entry)

    def __getitem__(self, idx: int) -> dict:
        return self._parse_entry(self.entries[idx])

    def _parse_entry(self, entry: dict) -> dict:
        """Convert a raw JSON entry into a standardized dict."""
        boxes = []
        class_ids = []

        for label in entry.get("labels", []):
            # Filter to detection-only annotations (those with box2d).
            # Skip lane lines (poly2d) and drivable areas (poly2d).
            if "box2d" not in label:
                continue

            category = label.get("category")
            if category not in BDD_CLASS_TO_ID:
                # Unknown class - shouldn't happen but skip just in case
                continue

            b = label["box2d"]
            boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
            class_ids.append(BDD_CLASS_TO_ID[category])

        return {
            "image_path": self.images_dir / entry["name"],
            "name": entry["name"],
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            "class_ids": np.array(class_ids, dtype=np.int64) if class_ids else np.zeros((0,), dtype=np.int64),
            "scene": entry.get("attributes", {}),
        }

    def scene_summary(self) -> dict:
        """Return a count breakdown by weather / timeofday / scene."""
        from collections import Counter
        weather = Counter()
        timeofday = Counter()
        scene = Counter()
        for entry in self.entries:
            attrs = entry.get("attributes", {})
            weather[attrs.get("weather", "?")] += 1
            timeofday[attrs.get("timeofday", "?")] += 1
            scene[attrs.get("scene", "?")] += 1
        return {
            "weather": dict(weather),
            "timeofday": dict(timeofday),
            "scene": dict(scene),
        }