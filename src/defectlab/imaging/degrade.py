"""Inline camera model.

Regime A is the original lab capture; Regime B is a conveyor-mounted inline camera.
Regime B is the primary result: the lab images are saturated, so fusion has no headroom there.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np


class Regime(StrEnum):
    LAB = "lab"
    INLINE = "inline"


@dataclass(frozen=True, slots=True)
class InlineCamera:
    """Conveyor-mounted industrial CMOS under uncontrolled plant lighting."""

    severity: float = 1.0
    blur_kernel: int = 9
    gain_range: tuple[float, float] = (0.65, 1.35)
    offset_range: tuple[float, float] = (-30.0, 30.0)
    noise_sd: float = 12.0
    effective_size: int = 96


def degrade(image: np.ndarray, rng: np.random.Generator, camera: InlineCamera) -> np.ndarray:
    """Apply motion blur, lighting drift, sensor noise and resolution loss in order."""
    if camera.severity <= 0.0:
        return image
    working = image.astype(np.float32)
    working = _motion_blur(working, rng, camera)
    working = _lighting_drift(working, rng, camera)
    working = _sensor_noise(working, rng, camera)
    return _resolution_loss(working, camera)


def apply_regime(
    image: np.ndarray, regime: Regime, rng: np.random.Generator, camera: InlineCamera | None = None
) -> np.ndarray:
    if regime is Regime.LAB:
        return image
    return degrade(image, rng, camera or InlineCamera())


def _motion_blur(image: np.ndarray, rng: np.random.Generator, camera: InlineCamera) -> np.ndarray:
    """The part is moving under the camera, at an unknown angle."""
    size = camera.blur_kernel
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1.0
    centre = (size / 2 - 0.5, size / 2 - 0.5)
    rotation = cv2.getRotationMatrix2D(centre, float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    return cv2.filter2D(image, -1, kernel / kernel.sum())


def _lighting_drift(
    image: np.ndarray, rng: np.random.Generator, camera: InlineCamera
) -> np.ndarray:
    """No controlled lighting rig on the line."""
    gain = rng.uniform(*camera.gain_range)
    offset = rng.uniform(*camera.offset_range)
    return gain * image + offset


def _sensor_noise(image: np.ndarray, rng: np.random.Generator, camera: InlineCamera) -> np.ndarray:
    return image + rng.normal(0.0, camera.noise_sd * camera.severity, image.shape)


def _resolution_loss(image: np.ndarray, camera: InlineCamera) -> np.ndarray:
    """Shorter working distance and cheaper optics cost effective resolution."""
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    height, width = clipped.shape[:2]
    small = cv2.resize(clipped, (camera.effective_size,) * 2, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
