"""Mondrian (class-conditional) conformal prediction.

Marginal conformal prediction under-covers the minority class badly at high imbalance,
so calibration quantiles are computed per class. Prediction sets may be empty (out of
distribution) or contain both labels (abstain).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

BOTH = frozenset({0, 1})


@dataclass(frozen=True, slots=True)
class PredictionSets:
    contains_defect: np.ndarray
    contains_ok: np.ndarray

    @property
    def is_ambiguous(self) -> np.ndarray:
        """Both labels plausible: defer to a human."""
        return self.contains_defect & self.contains_ok

    @property
    def is_empty(self) -> np.ndarray:
        """Neither label plausible: out of distribution."""
        return ~self.contains_defect & ~self.contains_ok

    @property
    def is_confident(self) -> np.ndarray:
        return self.contains_defect ^ self.contains_ok

    def abstention_rate(self) -> float:
        return float((self.is_ambiguous | self.is_empty).mean())


@dataclass(slots=True)
class MondrianConformal:
    """Split-conformal with per-class quantiles."""

    alpha: float = 0.1
    quantiles: dict[int, float] = field(default_factory=dict)

    def fit(
        self, calibration_labels: np.ndarray, calibration_scores: np.ndarray
    ) -> MondrianConformal:
        for label in (0, 1):
            mask = calibration_labels == label
            if not mask.any():
                raise ValueError(f"no calibration examples for class {label}")
            self.quantiles[label] = _conformal_quantile(
                _nonconformity(calibration_scores[mask], label), self.alpha
            )
        return self

    def predict_sets(self, scores: np.ndarray) -> PredictionSets:
        if not self.quantiles:
            raise RuntimeError("call fit before predict_sets")
        return PredictionSets(
            contains_defect=_nonconformity(scores, 1) <= self.quantiles[1],
            contains_ok=_nonconformity(scores, 0) <= self.quantiles[0],
        )


def coverage_by_class(labels: np.ndarray, sets: PredictionSets) -> dict[int, float]:
    """Fraction of each class whose true label is inside its prediction set."""
    included = {1: sets.contains_defect, 0: sets.contains_ok}
    return {label: float(included[label][labels == label].mean()) for label in (0, 1)}


def _nonconformity(scores: np.ndarray, label: int) -> np.ndarray:
    """One minus the score assigned to the candidate label."""
    return 1.0 - scores if label == 1 else scores


def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    n = len(residuals)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    return float(np.sort(residuals)[rank - 1])
