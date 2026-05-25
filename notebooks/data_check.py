"""
Verify videos load and frames extract cleanly.
"""
import cv2
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("outputs/data_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

videos = list(RAW_DIR.glob("*.mp4"))
print(f"Found {len(videos)} video(s) in {RAW_DIR}\n")

if not videos:
    print("No .mp4 files found.")
    exit()

for video_path in videos:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"✗ Could not open {video_path.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0

    print(f"{video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {duration:.1f}s")

    ret, frame = cap.read()
    if ret:
        out_path = OUT_DIR / f"{video_path.stem}_frame0.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"  ✓ Saved → {out_path}\n")
    else:
        print(f"  ✗ Failed to read first frame\n")

    cap.release()

print("Done.")