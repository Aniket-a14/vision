"""Builds the paired train and test tables and writes them to parquet."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from ..twin import TwinConfig
from .images import ImageSplit, load_split
from .pairing import build_paired_frame
from .schemas import PairedShotSchema

SPLITS: tuple[str, ...] = ("train", "test")


@dataclass(frozen=True, slots=True)
class BuiltDataset:
    train: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                _summarise(name, frame)
                for name, frame in (("train", self.train), ("test", self.test))
            ]
        )


def build(casting_root: Path, config: TwinConfig, oversample: int = 4) -> BuiltDataset:
    frames = {name: _build_split(casting_root, name, config, oversample) for name in SPLITS}
    return BuiltDataset(frames["train"], frames["test"])


def write(dataset: BuiltDataset, destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, frame in (("train", dataset.train), ("test", dataset.test)):
        path = destination / f"{name}_paired.parquet"
        frame.to_parquet(path, index=False)
        written[name] = path
    return written


def image_paths(frame: pd.DataFrame) -> list[str]:
    return frame["image_path"].tolist()


def _build_split(root: Path, name: str, config: TwinConfig, oversample: int) -> pd.DataFrame:
    split: ImageSplit = load_split(root, name)
    seeded = _seed_for(config, name)
    frame = build_paired_frame(split, seeded, oversample)
    return PairedShotSchema.validate(frame)


def _seed_for(config: TwinConfig, name: str) -> TwinConfig:
    """Train and test must come from independent RNG streams."""
    offset = 0 if name == "train" else 10_000
    return replace(config, seed=config.seed + offset)


def _summarise(name: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "split": name,
        "rows": len(frame),
        "prevalence": float(frame["label"].mean()),
        "lots": int(frame["lot_id"].nunique()),
        "shifts": int(frame["shift_id"].nunique()),
    }
