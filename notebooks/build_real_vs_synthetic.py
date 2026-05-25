"""
Build the real-vs-synthetic comparison grid.

Each condition uses a synthetic baseline clip chosen to categorically match
the real-world clip's scene type, with intensity tuned to the real clip's
severity. This controls for scene variables so the comparison shows the
augmentation effect rather than scene differences.
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
REAL_DIR = Path("data/real_weather")
OUT_DIR = Path("outputs/real_vs_synthetic")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Each condition: (name, real_clip, clean_baseline_clip, augmentation_fn, intensity)
CONDITIONS = [
    ("Rain",        REAL_DIR / "real_rain.mp4",        RAW_DIR / "clip_5.mp4", apply_rain,        "heavy"),
    ("Fog",         REAL_DIR / "real_fog.mp4",         RAW_DIR / "clip_5.mp4", apply_fog,         "medium"),
    ("Night",       REAL_DIR / "real_night.mp4",       RAW_DIR / "clip_2.mp4", apply_night,       "medium"),
    ("Motion Blur", REAL_DIR / "real_motion_blur.mp4", RAW_DIR / "clip_3.mp4", apply_motion_blur, "heavy"),
]


LABEL_HEIGHT = 60
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 1.0
LABEL_THICKNESS = 2
LABEL_COLOR = (255, 255, 255)
LABEL_BG = (40, 40, 40)


def add_label(panel: np.ndarray, text: str) -> np.ndarray:
    """Prepend a dark band with centered white text above the panel."""
    h, w = panel.shape[:2]
    band = np.full((LABEL_HEIGHT, w, 3), LABEL_BG, dtype=np.uint8)

    (text_w, text_h), _ = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    x = (w - text_w) // 2
    y = (LABEL_HEIGHT + text_h) // 2
    cv2.putText(band, text, (x, y), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS, cv2.LINE_AA)

    return np.vstack([band, panel])


def match_size(frame: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize a frame to match a target (H, W) shape."""
    th, tw = target_shape[:2]
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


# Process each condition
rows = []

for name, real_path, clean_path, aug_fn, intensity in CONDITIONS:
    if not real_path.exists():
        print(f"⚠ {real_path.name} missing — skipping {name}")
        continue
    if not clean_path.exists():
        print(f"⚠ {clean_path.name} missing — skipping {name}")
        continue

    real_frame = load_first_frame(real_path)
    clean_frame = load_first_frame(clean_path)

    if real_frame is None or clean_frame is None:
        print(f"✗ Could not read frames for {name}")
        continue

    # Resize real frame to match the chosen clean baseline's dimensions
    real_frame = match_size(real_frame, clean_frame.shape)

    # Apply synthetic augmentation to the matched clean baseline
    synth_frame = aug_fn(clean_frame, intensity)

    # Wrap each panel with a label
    real_label  = f"Real {name}"
    synth_label = f"Synthetic {name} ({intensity.capitalize()}) | {clean_path.stem}"

    p_real  = add_label(real_frame,  real_label)
    p_synth = add_label(synth_frame, synth_label)

    # Horizontal pair
    pair = np.hstack([p_real, p_synth])

    out_path = OUT_DIR / f"{name.lower().replace(' ', '_')}_comparison.jpg"
    save_frame(pair, out_path)
    print(f"✓ {name} ({clean_path.name}, {intensity}) → {out_path}")

    rows.append(pair)


# Combined grid (all conditions stacked vertically)
if rows:
    combined = np.vstack(rows)
    combined_path = OUT_DIR / "all_conditions_comparison.jpg"
    save_frame(combined, combined_path)
    print(f"\n✓ Combined grid → {combined_path}")
else:
    print("\n⚠ No conditions processed.")

print("\nDone. Open outputs/real_vs_synthetic/ to inspect.")