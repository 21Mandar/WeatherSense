"""
Build a unified comparison grid showing all 4 augmentations applied to the
first frame of each clip, with the clean baseline.

Output layout per clip: 2x3 grid (clean + 4 augmentations + a blank cell),
with labels above each panel.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.utils.frame_io import load_first_frame, save_frame
from src.augmentations.rain import apply_rain
from src.augmentations.fog import apply_fog
from src.augmentations.night import apply_night
from src.augmentations.motion_blur import apply_motion_blur


RAW_DIR = Path("data/raw")
OUT_DIR = Path("outputs/aug_unified")
OUT_DIR.mkdir(parents=True, exist_ok=True)


LABEL_HEIGHT = 60       # pixel height of label band above each panel
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 1.0
LABEL_THICKNESS = 2
LABEL_COLOR = (255, 255, 255)        # white text
LABEL_BG = (40, 40, 40)              # dark grey band


def add_label(panel: np.ndarray, text: str) -> np.ndarray:
    """Prepend a dark band with centered white text above the panel."""
    h, w = panel.shape[:2]
    band = np.full((LABEL_HEIGHT, w, 3), LABEL_BG, dtype=np.uint8)

    (text_w, text_h), _ = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    x = (w - text_w) // 2
    y = (LABEL_HEIGHT + text_h) // 2
    cv2.putText(band, text, (x, y), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS, cv2.LINE_AA)

    return np.vstack([band, panel])


def make_blank_panel(reference: np.ndarray) -> np.ndarray:
    """Blank dark panel matching the reference panel's dimensions."""
    h, w = reference.shape[:2]
    return np.full((h, w, 3), LABEL_BG, dtype=np.uint8)


videos = sorted(RAW_DIR.glob("*.mp4"))
print(f"Processing {len(videos)} clip(s)\n")

for video_path in videos:
    frame = load_first_frame(video_path)
    if frame is None:
        print(f"✗ {video_path.name} — couldn't read frame")
        continue

    # Apply each augmentation at medium intensity
    rain    = apply_rain(frame, "medium")
    fog     = apply_fog(frame, "medium")
    night   = apply_night(frame, "medium")
    blur    = apply_motion_blur(frame, "medium")

    # Wrap each in a labeled panel
    p_clean = add_label(frame,  "Clean (Baseline)")
    p_rain  = add_label(rain,   "Rain (Medium)")
    p_fog   = add_label(fog,    "Fog (Medium)")
    p_night = add_label(night,  "Night (Medium)")
    p_blur  = add_label(blur,   "Motion Blur (Medium)")
    p_blank = add_label(make_blank_panel(frame), "")

    # 2x3 grid: row1 = clean, rain, fog ; row2 = night, blur, blank
    row1 = np.hstack([p_clean, p_rain, p_fog])
    row2 = np.hstack([p_night, p_blur, p_blank])
    grid = np.vstack([row1, row2])

    out_path = OUT_DIR / f"{video_path.stem}_all_augmentations.jpg"
    save_frame(grid, out_path)
    print(f"✓ {video_path.name} → {out_path}")

print("\nDone. Open outputs/aug_unified/ to inspect.")