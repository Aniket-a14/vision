"""Group-aware splitting. Random splits leak lot- and die-level structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

GROUP_COLUMN = "lot_id"


class GroupLeakageError(RuntimeError):
    """Raised when a group appears on both sides of a split."""


def grouped_holdout(
    frame: pd.DataFrame, test_size: float = 0.25, seed: int = 42, group: str = GROUP_COLUMN
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole groups so chemistry never spans the split."""
    groups = frame[group].to_numpy()
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    held = rng.choice(unique, size=max(1, round(len(unique) * test_size)), replace=False)
    mask = np.isin(groups, held)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def grouped_folds(
    frame: pd.DataFrame, n_splits: int = 5, group: str = GROUP_COLUMN, stratify: bool = True
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits) if stratify else GroupKFold(n_splits)
    labels = frame["label"].to_numpy() if stratify else None
    return list(splitter.split(frame, labels, groups=frame[group].to_numpy()))


def assert_disjoint(
    frame: pd.DataFrame, train: np.ndarray, test: np.ndarray, group: str = GROUP_COLUMN
) -> None:
    overlap = set(frame.iloc[train][group]) & set(frame.iloc[test][group])
    if overlap:
        raise GroupLeakageError(f"groups on both sides of the split: {sorted(overlap)}")
