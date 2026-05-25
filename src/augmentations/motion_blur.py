"""Motion blur augmentation — directional kernel convolution to simulate
camera motion or fast vehicle movement."""
import cv2
import numpy as np


# Each preset controls:
#   kernel_size — length of the blur kernel in pixels (must be odd; larger = more blur)
#   angle_deg   — direction of blur in degrees (0 = horizontal, 90 = vertical)
PRESETS = {
    "light": {
        "kernel_size": 9,
        "angle_deg": 0,
    },
    "medium": {
        "kernel_size": 17,
        "angle_deg": 0,
    },
    "heavy": {
        "kernel_size": 27,
        "angle_deg": 0,
    },
}


def _make_motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    """
    Build a directional motion-blur kernel.

    A horizontal line of 1s, rotated to the target angle, then normalized.
    """
    if size % 2 == 0:
        size += 1  # Force odd

    # Start with a horizontal line in the center row
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0

    # Rotate the kernel by `angle_deg`
    center = (size / 2 - 0.5, size / 2 - 0.5)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))

    # Normalize so total weight = 1 (preserves brightness)
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return kernel


def apply_motion_blur(frame: np.ndarray, intensity: str = "medium") -> np.ndarray:
    """
    Apply synthetic motion blur to a frame using directional kernel convolution.

    Args:
        frame: BGR image (HxWx3, uint8) from cv2.
        intensity: "light", "medium", or "heavy".

    Returns:
        Augmented frame in BGR.
    """
    if intensity not in PRESETS:
        raise ValueError(f"intensity must be one of {list(PRESETS)}")

    preset = PRESETS[intensity]
    kernel = _make_motion_kernel(preset["kernel_size"], preset["angle_deg"])
    return cv2.filter2D(frame, ddepth=-1, kernel=kernel)