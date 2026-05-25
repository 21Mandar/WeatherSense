"""Fog augmentation — combines uniform haze, contrast reduction, and desaturation
for realistic atmospheric fog effect."""
import cv2
import numpy as np
import albumentations as A


# Each preset controls:
#   alb_fog_coef    — density of Albumentations' patchy fog overlay
#   haze_alpha      — strength of uniform white blend
#   contrast_factor — multiplier to reduce contrast (<1 flattens the scene)
#   saturation_mul  — multiplier on saturation channel (<1 desaturates)
PRESETS = {
    "light": {
        "alb_fog_coef": 0.15,
        "haze_alpha": 0.20,
        "contrast_factor": 0.90,
        "saturation_mul": 0.85,
    },
    "medium": {
        "alb_fog_coef": 0.30,
        "haze_alpha": 0.45,
        "contrast_factor": 0.70,
        "saturation_mul": 0.60,
    },
    "heavy": {
        "alb_fog_coef": 0.50,
        "haze_alpha": 0.65,
        "contrast_factor": 0.55,
        "saturation_mul": 0.40,
    },
}

# Atmospheric haze color — slightly cool grey-white, BGR
HAZE_COLOR = (215, 218, 220)


def _alb_fog(coef: float) -> A.RandomFog:
    return A.RandomFog(
        fog_coef_range=(coef, coef + 0.05),
        alpha_coef=0.15,
        p=1.0,
    )


def _uniform_haze(frame: np.ndarray, alpha: float) -> np.ndarray:
    """Blend frame with uniform haze color."""
    haze = np.full_like(frame, HAZE_COLOR, dtype=np.uint8)
    return cv2.addWeighted(frame, 1 - alpha, haze, alpha, 0)


def _reduce_contrast(frame: np.ndarray, factor: float) -> np.ndarray:
    """Pull pixel values toward the mean, reducing contrast."""
    mean = frame.mean(axis=(0, 1), keepdims=True)
    out = (frame.astype(np.float32) - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def _desaturate(frame: np.ndarray, mul: float) -> np.ndarray:
    """Reduce color saturation by multiplying the S channel in HSV space."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= mul
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_fog(frame: np.ndarray, intensity: str = "medium") -> np.ndarray:
    """
    Apply synthetic fog to a frame.

    Pipeline: patchy fog → uniform haze → contrast reduction → desaturation.
    This combination models atmospheric scattering more faithfully than
    Albumentations' RandomFog alone.

    Args:
        frame: BGR image (HxWx3, uint8) from cv2.
        intensity: "light", "medium", or "heavy".

    Returns:
        Augmented frame in BGR.
    """
    if intensity not in PRESETS:
        raise ValueError(f"intensity must be one of {list(PRESETS)}")

    preset = PRESETS[intensity]

    # 1. Patchy fog (Albumentations works in RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = _alb_fog(preset["alb_fog_coef"])(image=rgb)["image"]
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 2. Uniform haze blend
    out = _uniform_haze(out, preset["haze_alpha"])

    # 3. Contrast reduction
    out = _reduce_contrast(out, preset["contrast_factor"])

    # 4. Desaturation
    out = _desaturate(out, preset["saturation_mul"])

    return out