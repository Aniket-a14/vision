"""Joins simulated shots to real images by sampled label.

The join is label-only and deliberately so: the two channels stay independent,
which is what makes the degradation experiment interpretable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..twin import TwinConfig, run_line, score
from .images import ImageSplit


class InsufficientShotsError(RuntimeError):
    """Raised when the simulated run cannot cover every image label."""


def build_paired_frame(split: ImageSplit, config: TwinConfig, oversample: int = 4) -> pd.DataFrame:
    """Simulate a run, sample labels from physics, then attach matching images."""
    shots = score(run_line(len(split) * oversample, config), config, split.prevalence)
    chosen = _select_by_label(shots["label"].to_numpy(), split.labels)
    paired = shots.iloc[chosen].reset_index(drop=True)
    return _attach_images(paired, split)


def _select_by_label(shot_labels: np.ndarray, image_labels: np.ndarray) -> np.ndarray:
    """Walk the run in order, taking the next shot whose label matches each image."""
    pools = {value: iter(np.flatnonzero(shot_labels == value)) for value in (0, 1)}
    try:
        return np.array([next(pools[int(label)]) for label in image_labels])
    except StopIteration as exc:
        raise InsufficientShotsError("raise oversample: not enough shots of one label") from exc


def _attach_images(frame: pd.DataFrame, split: ImageSplit) -> pd.DataFrame:
    out = frame.copy()
    out["image_path"] = [str(path) for path in split.paths]
    out["split"] = split.split
    out["part_id"] = _part_ids(split)
    _assert_labels_align(out["label"].to_numpy(), split.labels)
    return out


def _part_ids(split: ImageSplit) -> list[str]:
    return [f"{split.split}-{index:05d}" for index in range(len(split))]


def _assert_labels_align(shot_labels: np.ndarray, image_labels: np.ndarray) -> None:
    if not np.array_equal(shot_labels, image_labels):
        raise InsufficientShotsError("label alignment failed after selection")
