"""Discovery and verification of the casting image dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

DEFECT_DIR: Final = "def_front"
OK_DIR: Final = "ok_front"

EXPECTED_COUNTS: Final[dict[tuple[str, str], int]] = {
    ("train", DEFECT_DIR): 3758,
    ("train", OK_DIR): 2875,
    ("test", DEFECT_DIR): 453,
    ("test", OK_DIR): 262,
}


@dataclass(frozen=True, slots=True)
class ImageSplit:
    split: str
    paths: tuple[Path, ...]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def prevalence(self) -> float:
        return float(self.labels.mean())


class DatasetLayoutError(RuntimeError):
    """Raised when the extracted Kaggle dataset does not match the published layout."""


def load_split(root: Path, split: str) -> ImageSplit:
    """Collect paths and binary labels for one split, defect first."""
    defect = _list_images(root / split / DEFECT_DIR)
    ok = _list_images(root / split / OK_DIR)
    paths = defect + ok
    labels = np.concatenate([np.ones(len(defect), dtype=np.int8), np.zeros(len(ok), dtype=np.int8)])
    return ImageSplit(split, tuple(paths), labels)


def verify_counts(root: Path, strict: bool = True) -> dict[tuple[str, str], int]:
    """Compare on-disk counts with the published dataset figures."""
    actual = {key: _count(root, key) for key in EXPECTED_COUNTS}
    if strict and actual != EXPECTED_COUNTS:
        raise DatasetLayoutError(f"expected {EXPECTED_COUNTS}, found {actual}")
    return actual


def _count(root: Path, key: tuple[str, str]) -> int:
    split, folder = key
    directory = root / split / folder
    return len(_list_images(directory)) if directory.is_dir() else 0


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise DatasetLayoutError(f"missing image directory: {directory}")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _EXTENSIONS)


_EXTENSIONS: Final = frozenset({".jpeg", ".jpg", ".png"})
