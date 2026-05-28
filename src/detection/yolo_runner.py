"""
YOLOv8 inference wrapper for WeatherSense Phase 2.

Wraps Ultralytics YOLOv8 with a standardized interface that:
- Loads a YOLOv8 model variant (n / s / m / l / x)
- Runs inference on an image
- Maps COCO predictions to BDD100K classes (9 active mappings)
- Discards predictions for COCO classes with no BDD equivalent
- Returns boxes + class_ids + confidences in a format directly comparable
  to BDDDataset's ground-truth output

Documented limitations:
- `rider` (BDD class 1) has no clean COCO equivalent and is never predicted.
  YOLO's `person` class maps to BDD `pedestrian` only.
- `traffic sign` (BDD class 9) is mapped only from COCO `stop sign`.
  YOLO will miss speed-limit signs, yield signs, lane signs, etc.
- 71 of COCO's 80 classes have no BDD equivalent and are discarded.
"""
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO


# COCO class ID -> BDD class ID
# Only the 9 mappable classes are listed; everything else is discarded.
COCO_TO_BDD = {
    0:  0,   # person        -> pedestrian (BDD 0)
    1:  7,   # bicycle       -> bicycle    (BDD 7)
    2:  2,   # car           -> car        (BDD 2)
    3:  6,   # motorcycle    -> motorcycle (BDD 6)
    5:  4,   # bus           -> bus        (BDD 4)
    6:  5,   # train         -> train      (BDD 5)
    7:  3,   # truck         -> truck      (BDD 3)
    9:  8,   # traffic light -> traffic light (BDD 8)
    11: 9,   # stop sign     -> traffic sign  (BDD 9)
}


# YOLOv8 model variant names accepted by Ultralytics.
# These auto-download on first use to ~/.config/Ultralytics/ or the working dir.
VALID_MODELS = {"yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"}


class YOLORunner:
    """
    Standardized YOLOv8 inference wrapper.

    Usage:
        runner = YOLORunner("yolov8n")
        result = runner.predict(image_bgr)
        # result = {
        #     "boxes":       np.ndarray (N, 4) in [x1, y1, x2, y2] pixel coords
        #     "class_ids":   np.ndarray (N,)  BDD class IDs (0-9)
        #     "confidences": np.ndarray (N,)  detection confidences in [0, 1]
        # }
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name:     One of {yolov8n, yolov8s, yolov8m, yolov8l, yolov8x}.
                            Auto-downloads on first use.
            conf_threshold: Minimum confidence to keep a prediction.
                            Default 0.001 keeps essentially everything so the
                            mAP calculation can sweep across the confidence
                            curve. Use a higher value (e.g. 0.25) only for
                            visualization.
            iou_threshold:  NMS IoU threshold. Default matches Ultralytics.
            device:         "cpu", "mps" (Apple Silicon), "cuda", or None
                            for auto-detect.
        """
        if model_name not in VALID_MODELS:
            raise ValueError(
                f"Unknown model {model_name!r}. "
                f"Expected one of {sorted(VALID_MODELS)}."
            )

        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Ultralytics expects the model file path or short name like 'yolov8n.pt'
        print(f"Loading {model_name}...")
        self.model = YOLO(f"{model_name}.pt")

    def predict(self, image: np.ndarray) -> dict:
        """
        Run inference on a single image and return BDD-mapped predictions.

        Args:
            image: BGR image as returned by cv2.imread, shape (H, W, 3).

        Returns:
            dict with keys 'boxes', 'class_ids', 'confidences'.
            All arrays are shape (N, ...) where N is the post-filter count.
            If no predictions survive the COCO->BDD mapping, arrays are empty.
        """
        # Ultralytics accepts BGR numpy arrays directly.
        # verbose=False to silence per-image logging spam.
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        # results is a list of length 1 (single image input -> single result)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return self._empty_result()

        # Extract on CPU as numpy
        coco_class_ids = r.boxes.cls.cpu().numpy().astype(int)
        confidences = r.boxes.conf.cpu().numpy().astype(np.float32)
        xyxy_boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)

        # Filter to only the COCO classes that map to BDD
        keep_mask = np.array(
            [cid in COCO_TO_BDD for cid in coco_class_ids],
            dtype=bool,
        )

        if not keep_mask.any():
            return self._empty_result()

        kept_boxes = xyxy_boxes[keep_mask]
        kept_confidences = confidences[keep_mask]
        kept_coco_ids = coco_class_ids[keep_mask]

        # Map COCO IDs to BDD IDs
        bdd_class_ids = np.array(
            [COCO_TO_BDD[cid] for cid in kept_coco_ids],
            dtype=np.int64,
        )

        return {
            "boxes": kept_boxes,
            "class_ids": bdd_class_ids,
            "confidences": kept_confidences,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "class_ids": np.zeros((0,), dtype=np.int64),
            "confidences": np.zeros((0,), dtype=np.float32),
        }