"""Night augmentation — gamma correction, blue tint, and sensor noise to
simulate low-light driving conditions."""
import cv2
import numpy as np


# Each preset controls:
#   brightness_scale — pre-gamma multiplier (<1 reduces overall brightness uniformly)
#   gamma            — power-law exponent (<1 darkens midtones)
#   blue_shift       — cool blue color shift
#   noise_std        — Gaussian noise standard deviation
#   saturation_mul   — saturation reduction
PRESETS = {
    "light": {
        "brightness_scale": 0.75,
        "gamma": 0.70,
        "blue_shift": 0.10,
        "noise_std": 4.0,
        "saturation_mul": 0.75,
    },
    "medium": {
        "brightness_scale": 0.50,
        "gamma": 0.55,
        "blue_shift": 0.18,
        "noise_std": 8.0,
        "saturation_mul": 0.55,
    },
    "heavy": {
        "brightness_scale": 0.30,
        "gamma": 0.45,
        "blue_shift": 0.25,
        "noise_std": 12.0,
        "saturation_mul": 0.35,
    },
}


def _apply_brightness_and_gamma(
    frame: np.ndarray, brightness_scale: float, gamma: float
) -> np.ndarray:
    """
    Combined brightness scaling + gamma. Brightness scaling darkens
    uniformly (incl. bright skies); gamma further darkens midtones.
    """
    inv = 1.0 / max(gamma, 1e-6)
    x = np.arange(256) / 255.0
    x = x * brightness_scale  # Uniform pre-darkening (kills bright skies)
    table = (x ** inv * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(frame, table)


def _apply_blue_shift(frame: np.ndarray, amount: float) -> np.ndarray:
    """Shift color balance toward cool blue (BGR: boost B, reduce R)."""
    out = frame.astype(np.float32)
    out[..., 0] = np.clip(out[..., 0] * (1 + amount), 0, 255)
    out[..., 2] = np.clip(out[..., 2] * (1 - amount * 0.5), 0, 255)
    return out.astype(np.uint8)


def _add_sensor_noise(frame: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to simulate high-ISO sensor noise."""
    noise = np.random.normal(0, std, frame.shape).astype(np.float32)
    out = frame.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _desaturate(frame: np.ndarray, mul: float) -> np.ndarray:
    """Reduce color saturation in HSV space."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= mul
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_night(frame: np.ndarray, intensity: str = "medium") -> np.ndarray:
    """
    Apply synthetic night/low-light conditions to a frame.

    Pipeline: brightness + gamma → desaturate → blue shift → sensor noise.
    Brightness scaling fixes the "bright sky stays bright" failure mode that
    gamma alone exhibits.

    Args:
        frame: BGR image (HxWx3, uint8) from cv2.
        intensity: "light" (dusk), "medium" (night), or "heavy" (deep night).

    Returns:
        Augmented frame in BGR.
    """
    if intensity not in PRESETS:
        raise ValueError(f"intensity must be one of {list(PRESETS)}")

    preset = PRESETS[intensity]

    out = _apply_brightness_and_gamma(frame, preset["brightness_scale"], preset["gamma"])
    out = _desaturate(out, preset["saturation_mul"])
    out = _apply_blue_shift(out, preset["blue_shift"])
    out = _add_sensor_noise(out, preset["noise_std"])

    return out