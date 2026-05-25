"""
Visual test — apply all three fog intensities to the first frame of every clip
and save side-by-side comparisons.
"""
import sys
from pathlib import Path

# Make `src/` importable from this script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.frame_io import load_first_frame, save_frame
from src.augmentations.fog import apply_fog


RAW_DIR = Path("data/raw")
OUT_DIR = Path("outputs/aug_fog")
OUT_DIR.mkdir(parents=True, exist_ok=True)

videos = sorted(RAW_DIR.glob("*.mp4"))
print(f"Processing {len(videos)} clip(s)\n")

for video_path in videos:
    frame = load_first_frame(video_path)
    if frame is None:
        print(f"✗ {video_path.name} — couldn't read frame")
        continue

    light = apply_fog(frame, "light")
    medium = apply_fog(frame, "medium")
    heavy = apply_fog(frame, "heavy")

    top = np.hstack([frame, light])
    bot = np.hstack([medium, heavy])
    grid = np.vstack([top, bot])

    out_path = OUT_DIR / f"{video_path.stem}_fog_grid.jpg"
    save_frame(grid, out_path)
    print(f"✓ {video_path.name} → {out_path}")

print("\nDone. Open outputs/aug_fog/ to inspect.")
