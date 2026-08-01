"""Batched feature extraction with an on-disk cache.

Extraction is the slowest step in the pipeline, so it runs once per (backbone, regime)
and every downstream experiment reads the cache.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from .backbones import build, build_transform, configure_threads
from .degrade import InlineCamera, Regime, apply_regime
from .progress import LOG, Progress, clock


class ImageReadError(RuntimeError):
    """Raised when an image path cannot be decoded."""


def cache_path(root: Path, backbone: str, split: str, regime: Regime) -> Path:
    return root / f"{split}_{backbone}_{regime.value}.npy"


def extract_split(
    paths: Sequence[Path | str],
    regime: Regime,
    backbone: str,
    seed: int = 0,
    batch_size: int = 32,
    camera: InlineCamera | None = None,
) -> np.ndarray:
    """Forward every image once and stack the embeddings."""
    configure_threads()
    LOG.info("loading %s for %s (%d images)", backbone, regime.value, len(paths))
    model, spec = build(backbone)
    transform = build_transform(model, spec)
    rng = np.random.default_rng(seed)
    tracker = Progress(len(paths), f"{backbone}/{regime.value}")
    embeddings = []
    for batch in _iter_batches(paths, batch_size):
        embeddings.append(_embed(model, transform, batch, regime, rng, camera))
        tracker.update(len(batch))
    LOG.info("done %s in %s at %.1f img/s", tracker.label, clock(tracker.elapsed()), tracker.rate())
    return np.vstack(embeddings).astype(np.float32)


def extract_cached(
    paths: Sequence[Path | str],
    regime: Regime,
    backbone: str,
    destination: Path,
    seed: int = 0,
    batch_size: int = 32,
    camera: InlineCamera | None = None,
) -> np.ndarray:
    """Return cached features when present, otherwise extract and cache them."""
    if destination.exists():
        LOG.info("cache hit %s, skipping extraction", destination.name)
        return np.load(destination)
    features = extract_split(paths, regime, backbone, seed, batch_size, camera)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, features)
    return features


def load_grayscale(path: Path | str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ImageReadError(f"could not decode {path}")
    return image


def to_three_channel(image: np.ndarray) -> np.ndarray:
    """ImageNet-normalised backbones expect three channels."""
    return np.repeat(image[:, :, None], 3, axis=2)


def _embed(
    model,
    transform,
    batch: Sequence[Path | str],
    regime: Regime,
    rng: np.random.Generator,
    camera: InlineCamera | None,
) -> np.ndarray:
    import torch

    tensors = [_prepare(path, regime, rng, camera, transform) for path in batch]
    with torch.no_grad():
        return model(torch.stack(tensors)).numpy()


def _prepare(path: Path | str, regime: Regime, rng: np.random.Generator, camera, transform):
    from PIL import Image

    image = apply_regime(load_grayscale(path), regime, rng, camera)
    return transform(Image.fromarray(to_three_channel(image)))


def _iter_batches(paths: Sequence[Path | str], size: int) -> Iterator[Sequence[Path | str]]:
    for start in range(0, len(paths), size):
        yield paths[start : start + size]
