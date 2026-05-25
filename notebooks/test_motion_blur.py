"""
Visual test — apply all three motion blur intensities to the first frame of
every clip and save side-by-side comparisons.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.frame_io import load_first_frame, save_frame
from src.augmentations.motion_blur import apply_motion_blur


RAW_DIR = Path("data/raw")
OUT_DIR = Path("outputs/aug_motion_blur")
OUT_DIR.mkdir(parents=True, exist_ok=True)

videos = sorted(RAW_DIR.glob("*.mp4"))
print(f"Processing {len(videos)} clip(s)\n")

for video_path in videos:
    frame = load_first_frame(video_path)
    if frame is None:
        print(f"✗ {video_path.name} — couldn't read frame")
        continue

    light = apply_motion_blur(frame, "light")
    medium = apply_motion_blur(frame, "medium")
    heavy = apply_motion_blur(frame, "heavy")

    top = np.hstack([frame, light])
    bot = np.hstack([medium, heavy])
    grid = np.vstack([top, bot])

    out_path = OUT_DIR / f"{video_path.stem}_motion_blur_grid.jpg"
    save_frame(grid, out_path)
    print(f"✓ {video_path.name} → {out_path}")

print("\nDone. Open outputs/aug_motion_blur/ to inspect.")