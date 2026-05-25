"""Rain augmentation using Albumentations."""
import cv2
import numpy as np
import albumentations as A


PRESETS = {
    "light": A.RandomRain(
        slant_range=(-10, 10),
        drop_length=10,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=2,
        brightness_coefficient=0.95,
        rain_type="default",
        p=1.0,
    ),
    "medium": A.RandomRain(
        slant_range=(-15, 15),
        drop_length=20,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=4,
        brightness_coefficient=0.85,
        rain_type="heavy",
        p=1.0,
    ),
    "heavy": A.RandomRain(
        slant_range=(-20, 20),
        drop_length=25,
        drop_width=2,
        drop_color=(190, 190, 190),
        blur_value=4,
        brightness_coefficient=0.9,
        rain_type="heavy",
        p=1.0,
    ),
}


def apply_rain(frame: np.ndarray, intensity: str = "medium") -> np.ndarray:
    """
    Apply synthetic rain to a frame.

    Args:
        frame: BGR image (HxWx3, uint8) from cv2.
        intensity: "light", "medium", or "heavy".

    Returns:
        Augmented frame in BGR.
    """
    if intensity not in PRESETS:
        raise ValueError(f"intensity must be one of {list(PRESETS)}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    augmented = PRESETS[intensity](image=rgb)["image"]
    return cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)