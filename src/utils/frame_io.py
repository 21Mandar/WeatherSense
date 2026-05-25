"""Frame I/O helpers — load, resize, and save frames consistently."""
import cv2
import numpy as np
from pathlib import Path


TARGET_SIZE = (1280, 720)  # (width, height) — standardize all frames here


def resize_if_larger(frame: np.ndarray, target=TARGET_SIZE) -> np.ndarray:
    """
    Downsample frame to target size if it's larger.
    Preserves aspect ratio by resizing to fit within target dimensions.
    """
    h, w = frame.shape[:2]
    tw, th = target

    if w <= tw and h <= th:
        return frame

    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_first_frame(video_path, resize: bool = True):
    """Load the first frame of a video, optionally resizing."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None
    return resize_if_larger(frame) if resize else frame


def save_frame(frame: np.ndarray, out_path) -> None:
    """Save a frame to disk, creating parent dirs if needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)