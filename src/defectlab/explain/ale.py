"""Accumulated local effects.

Partial dependence averages a feature over its whole marginal range, so with correlated
process parameters it scores combinations the machine cannot physically produce -- a 620 C
pour into a die already sitting at 320 C, say. ALE only moves a feature inside the bin a
shot already occupies, then accumulates those local moves, so every evaluated point stays
near the observed data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_BINS = 20
MIN_EDGES = 2

Predict = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class AleCurve:
    """One feature's centred effect, in the units the predictor returns."""

    feature: str
    centres: np.ndarray
    effect: np.ndarray
    counts: np.ndarray

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"value": self.centres, "effect": self.effect, "shots": self.counts}
        ).assign(feature=self.feature)

    def span(self) -> float:
        """How much the model moves across the feature's range; a crude importance."""
        return float(self.effect.max() - self.effect.min()) if len(self.effect) else 0.0


def ale(
    predict: Predict, features: np.ndarray, index: int, name: str, bins: int = DEFAULT_BINS
) -> AleCurve:
    """Accumulated local effect of one column, centred on its weighted mean."""
    edges = _edges(features[:, index], bins)
    if len(edges) < MIN_EDGES:
        return AleCurve(name, np.array([]), np.array([]), np.array([], dtype=int))
    assignment = _assign_bins(features[:, index], edges)
    deltas, counts = _local_deltas(predict, features, index, edges, assignment)
    accumulated = np.concatenate([[0.0], np.cumsum(deltas)])
    centres = _centres(edges)
    return AleCurve(name, centres, _centre(_at_centres(accumulated), counts), counts)


def _edges(column: np.ndarray, bins: int) -> np.ndarray:
    """Quantile edges, deduplicated; a near-constant feature yields fewer bins, not errors."""
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    return np.unique(np.quantile(column, quantiles))


def _assign_bins(column: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per row, clipped so the extreme values join the end bins."""
    raw = np.searchsorted(edges, column, side="left") - 1
    return np.clip(raw, 0, len(edges) - 2)


def _local_deltas(
    predict: Predict, features: np.ndarray, index: int, edges: np.ndarray, assignment: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean prediction change across each bin, using only the shots that fall inside it."""
    n_bins = len(edges) - 1
    deltas = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for bin_index in range(n_bins):
        rows = np.flatnonzero(assignment == bin_index)
        counts[bin_index] = len(rows)
        if len(rows) == 0:
            continue
        deltas[bin_index] = _bin_delta(predict, features, index, rows, edges, bin_index)
    return deltas, counts


def _bin_delta(
    predict: Predict,
    features: np.ndarray,
    index: int,
    rows: np.ndarray,
    edges: np.ndarray,
    bin_index: int,
) -> float:
    lower = _replaced(features[rows], index, edges[bin_index])
    upper = _replaced(features[rows], index, edges[bin_index + 1])
    return float(np.mean(predict(upper) - predict(lower)))


def _replaced(block: np.ndarray, index: int, value: float) -> np.ndarray:
    copy = block.copy()
    copy[:, index] = value
    return copy


def _centres(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2.0


def _at_centres(accumulated: np.ndarray) -> np.ndarray:
    """Accumulation is defined at bin edges; the curve is reported at bin centres."""
    return (accumulated[:-1] + accumulated[1:]) / 2.0


def _centre(effect: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Centre on the data, not on zero, so the curve reads against the average shot."""
    total = counts.sum()
    if total == 0:
        return effect
    return effect - float(np.sum(counts * effect) / total)


def ale_table(
    predict: Predict, features: np.ndarray, names: list[str], bins: int = DEFAULT_BINS
) -> pd.DataFrame:
    """Every named column's curve stacked into one tidy table."""
    curves = [ale(predict, features, index, name, bins) for index, name in enumerate(names)]
    return pd.concat([curve.frame() for curve in curves], ignore_index=True)
