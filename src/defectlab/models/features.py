"""Feature assembly for the three model configurations.

Scaler and PCA are fitted on train only. Image and tabular blocks are scale-normalised
before concatenation, because a wide image block otherwise suppresses the tabular signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..twin import FEATURES


class Modality(StrEnum):
    VISION = "vision"
    PROCESS = "process"
    FUSION = "fusion"


@dataclass(frozen=True, slots=True)
class FeatureBlocks:
    train: np.ndarray
    test: np.ndarray
    names: list[str]


@dataclass(slots=True)
class ImageReducer:
    """Fitted on train embeddings only; transform is reused for test and serving."""

    scaler: StandardScaler
    pca: PCA
    balance: float = 1.0

    @property
    def n_components(self) -> int:
        return self.pca.n_components_

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(embeddings)) * self.balance


def fit_image_reducer(train_embeddings: np.ndarray, n_components: int) -> ImageReducer:
    scaler = StandardScaler().fit(train_embeddings)
    pca = PCA(n_components=n_components, random_state=42).fit(scaler.transform(train_embeddings))
    return ImageReducer(scaler, pca)


def balance_blocks(reducer: ImageReducer, image_block: np.ndarray, n_tabular: int) -> ImageReducer:
    """Equalise total variance per block so neither modality dominates by width alone."""
    image_energy = float(np.sqrt((image_block.var(axis=0)).sum()))
    if image_energy == 0.0:
        return reducer
    reducer.balance = float(np.sqrt(n_tabular) / image_energy)
    return reducer


def image_column_names(n_components: int) -> list[str]:
    return [f"img_pc{index:02d}" for index in range(n_components)]


def build_blocks(
    modality: Modality,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_embeddings: np.ndarray | None = None,
    test_embeddings: np.ndarray | None = None,
    n_components: int = 30,
) -> FeatureBlocks:
    if modality is Modality.PROCESS:
        return _process_only(train_frame, test_frame)
    reducer = _prepare_reducer(train_embeddings, n_components)
    train_image = reducer.transform(train_embeddings)
    test_image = reducer.transform(test_embeddings)
    names = image_column_names(reducer.n_components)
    if modality is Modality.VISION:
        return FeatureBlocks(train_image, test_image, names)
    return _fused(train_frame, test_frame, train_image, test_image, names)


def _prepare_reducer(train_embeddings: np.ndarray | None, n_components: int) -> ImageReducer:
    if train_embeddings is None:
        raise ValueError("image embeddings are required for vision and fusion modalities")
    reducer = fit_image_reducer(train_embeddings, n_components)
    raw = reducer.pca.transform(reducer.scaler.transform(train_embeddings))
    return balance_blocks(reducer, raw, len(FEATURES))


def _process_only(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> FeatureBlocks:
    names = list(FEATURES)
    return FeatureBlocks(_tabular(train_frame), _tabular(test_frame), names)


def _fused(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_image: np.ndarray,
    test_image: np.ndarray,
    image_names: list[str],
) -> FeatureBlocks:
    train = np.hstack([_tabular(train_frame), train_image])
    test = np.hstack([_tabular(test_frame), test_image])
    return FeatureBlocks(train, test, list(FEATURES) + image_names)


def _tabular(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(FEATURES)].to_numpy(dtype=np.float64)
