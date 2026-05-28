"""
mAP evaluator for WeatherSense Phase 2.

Wraps torchmetrics.detection.MeanAveragePrecision (which uses the official
pycocotools backend) with a WeatherSense-specific interface:

- Accepts predictions and ground truths in the formats produced by
  YOLORunner.predict() and BDDDataset[idx]
- Accumulates across many image samples
- Returns structured per-class + overall metrics keyed by BDD class names

Why we wrap rather than use torchmetrics directly:
- Our prediction/GT formats use numpy arrays; torchmetrics expects torch tensors
- We want results keyed by BDD class names (readable), not numeric IDs
- We want a clean reset() interface for evaluating multiple conditions in
  sequence (e.g., clean, fog-light, fog-medium, fog-heavy) without re-instantiating
"""
from typing import Optional

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

from src.detection.bdd_dataset import BDD_CLASSES


class MAPEvaluator:
    """
    Accumulate predictions vs ground truths and compute mAP metrics.

    Usage:
        evaluator = MAPEvaluator()
        for sample in dataset:
            image = cv2.imread(str(sample["image_path"]))
            preds = yolo_runner.predict(image)
            evaluator.add_sample(preds, sample)
        results = evaluator.compute()
        # results = {
        #     "map_50":      float,        # overall mAP@IoU=0.50
        #     "map_50_95":   float,        # overall mAP@IoU=0.50:0.95
        #     "per_class": {
        #         "pedestrian":    {"ap_50": float, "ap_50_95": float},
        #         "rider":         {...},
        #         ...
        #     }
        # }
    """

    def __init__(self, class_names: Optional[list[str]] = None):
        """
        Args:
            class_names: List of class names indexed by class ID.
                         Defaults to BDD_CLASSES (the 10 BDD detection classes).
        """
        self.class_names = class_names if class_names is not None else BDD_CLASSES
        self.num_classes = len(self.class_names)

        # class_metrics=True: report per-class AP, not just overall mAP
        # box_format='xyxy': matches both YOLO's xyxy output and BDD's box2d layout
        self.metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        backend="pycocotools",
        # Raise from default 100 - we feed YOLO predictions at conf=0.001
        # so per-image prediction counts are 100-300. Capping at 100 silently
        # drops lower-confidence predictions and produces -1.0 sentinels for
        # classes whose predictions all got truncated.
        
    )

    # COCO convention: cap at 100 predictions per image (highest confidence)
    MAX_PREDS_PER_IMAGE = 100

    def add_sample(self, preds: dict, ground_truth: dict) -> None:
        """
        Add a single image's predictions and GT to the running evaluation.

        Per COCO convention (and pycocotools default), we keep at most 100
        predictions per image, selected by highest confidence. This matches
        the behavior of the standard COCO evaluator and avoids triggering a
        known torchmetrics bug with non-default max_detection_thresholds.
        """
        boxes = preds["boxes"]
        scores = preds["confidences"]
        labels = preds["class_ids"]

        # Keep top-K by confidence if we have too many
        if len(scores) > self.MAX_PREDS_PER_IMAGE:
            # argsort descending by score, take top K
            top_k = np.argsort(-scores)[:self.MAX_PREDS_PER_IMAGE]
            boxes = boxes[top_k]
            scores = scores[top_k]
            labels = labels[top_k]

        pred_tensor = {
            "boxes":  torch.from_numpy(boxes).float(),
            "scores": torch.from_numpy(scores).float(),
            "labels": torch.from_numpy(labels).long(),
        }
        gt_tensor = {
            "boxes":  torch.from_numpy(ground_truth["boxes"]).float(),
            "labels": torch.from_numpy(ground_truth["class_ids"]).long(),
        }
        # torchmetrics expects a list of dicts (one per image in the batch).
        # We add one image at a time.
        self.metric.update(preds=[pred_tensor], target=[gt_tensor])

    def add_batch(self, preds_list: list[dict], gts_list: list[dict]) -> None:
        """Add multiple samples at once. Equivalent to looping add_sample."""
        if len(preds_list) != len(gts_list):
            raise ValueError(
                f"Mismatched batch sizes: {len(preds_list)} preds vs {len(gts_list)} GTs"
            )
        for p, g in zip(preds_list, gts_list):
            self.add_sample(p, g)

    def compute(self) -> dict:
        """
        Compute aggregated metrics across all added samples.

        Returns:
            dict with overall map_50, map_50_95, and per-class breakdowns.
        """
        raw = self.metric.compute()

        # Overall metrics
        results = {
            "map_50":    float(raw["map_50"].item()),
            "map_50_95": float(raw["map"].item()),
            "per_class": {},
        }

        # Per-class AP. torchmetrics returns:
        #   "map_per_class":     tensor of per-class mAP@50:95
        #   "mar_100_per_class": tensor of per-class mean Average Recall
        # Annoyingly, mAP@50 isn't directly exposed per-class - so we report
        # only AP@50:95 per class. (The overall map_50 still comes from torchmetrics.)
        per_class_50_95 = raw.get("map_per_class")
        classes_present = raw.get("classes")

        if per_class_50_95 is not None and classes_present is not None:
            # classes tensor contains the integer class IDs that had >=1 prediction or GT
            per_class_50_95 = per_class_50_95.cpu().numpy()
            classes_present = classes_present.cpu().numpy()

            for class_id, ap in zip(classes_present, per_class_50_95):
                if 0 <= class_id < self.num_classes:
                    name = self.class_names[int(class_id)]
                    results["per_class"][name] = {"ap_50_95": float(ap)}

        # Fill in classes that didn't appear (no predictions AND no GT) so
        # the table is always 10 rows for readability
        for name in self.class_names:
            if name not in results["per_class"]:
                results["per_class"][name] = {"ap_50_95": float("nan")}

        return results

    def reset(self) -> None:
        """Clear accumulators. Call between evaluation conditions."""
        self.metric.reset()


def format_results_table(results: dict, condition_name: str = "Results") -> str:
    """
    Pretty-print mAP results as a text table.

    Useful for the sanity-check script and for terminal output during the
    full Stage 5 benchmark.
    """
    lines = []
    lines.append(f"=== {condition_name} ===")
    lines.append(f"  mAP@50      = {results['map_50']:.4f}")
    lines.append(f"  mAP@50:95   = {results['map_50_95']:.4f}")
    lines.append("")
    lines.append(f"  {'Class':<15} {'AP@50:95':>10}")
    lines.append(f"  {'-'*15} {'-'*10}")
    for name, vals in results["per_class"].items():
        ap = vals["ap_50_95"]
        if np.isnan(ap):
            ap_str = "  n/a   "
        elif ap < 0:
            # pycocotools sentinel for "class had no GT in samples"
            ap_str = "no GT   "
        else:
            ap_str = f"{ap:.4f}"
        lines.append(f"  {name:<15} {ap_str:>10}")
    return "\n".join(lines)