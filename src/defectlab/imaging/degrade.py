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


MIN_GAIN = 0.05
MIN_RESOLUTION = 16


@dataclass(frozen=True, slots=True)
class InlineCamera:
    """Conveyor-mounted industrial CMOS under uncontrolled plant lighting.

    Fields are the reference values at severity 1.0; severity scales every channel
    together, so 0.0 is a clean capture and values above 1.0 extrapolate.
    """

    severity: float = 1.0
    blur_kernel: int = 9
    gain_range: tuple[float, float] = (0.65, 1.35)
    offset_range: tuple[float, float] = (-30.0, 30.0)
    noise_sd: float = 12.0
    effective_size: int = 96

    def kernel_size(self) -> int:
        """Odd-sized kernel; 1 is a no-op, which is what severity 0 must give."""
        size = round(1 + (self.blur_kernel - 1) * self.severity)
        return max(1, size + 1 if size % 2 == 0 else size)

    def gain_bounds(self) -> tuple[float, float]:
        """Clamped positive: a negative gain would invert the image, not dim it."""
        scaled = (1.0 + (bound - 1.0) * self.severity for bound in self.gain_range)
        return tuple(max(MIN_GAIN, bound) for bound in scaled)

    def offset_bounds(self) -> tuple[float, float]:
        return tuple(bound * self.severity for bound in self.offset_range)

    def resolution(self, full: int) -> int:
        """Geometric decay, so extrapolating past severity 1 degrades without collapsing."""
        target = round(full * (self.effective_size / full) ** self.severity)
        return int(np.clip(target, MIN_RESOLUTION, full))


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
    size = camera.kernel_size()
    if size <= 1:
        return image
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
    gain = rng.uniform(*camera.gain_bounds())
    offset = rng.uniform(*camera.offset_bounds())
    return gain * image + offset


def _sensor_noise(image: np.ndarray, rng: np.random.Generator, camera: InlineCamera) -> np.ndarray:
    return image + rng.normal(0.0, camera.noise_sd * camera.severity, image.shape)


def _resolution_loss(image: np.ndarray, camera: InlineCamera) -> np.ndarray:
    """Shorter working distance and cheaper optics cost effective resolution."""
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    height, width = clipped.shape[:2]
    size = camera.resolution(min(height, width))
    if size >= min(height, width):
        return clipped
    small = cv2.resize(clipped, (size,) * 2, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
