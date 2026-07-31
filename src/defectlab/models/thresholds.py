"""Cost-optimal and constrained decision thresholds.

Thresholds are chosen on calibrated probabilities. An escape costs far more than an
overkill, so the optimum sits well below 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True, slots=True)
class CostMatrix:
    escape: float = 250.0
    overkill: float = 4.0

    @property
    def ratio(self) -> float:
        return self.escape / self.overkill


@dataclass(frozen=True, slots=True)
class ThresholdSweep:
    grid: np.ndarray
    cost: np.ndarray
    alert_rate: np.ndarray

    def optimum(self) -> float:
        return float(self.grid[int(np.argmin(self.cost))])


def sweep(
    labels: np.ndarray, scores: np.ndarray, costs: CostMatrix, points: int = 199
) -> ThresholdSweep:
    grid = np.linspace(0.005, 0.995, points)
    totals = np.array([_total_cost(labels, scores, t, costs) for t in grid])
    alerts = np.array([float((scores >= t).mean()) for t in grid])
    return ThresholdSweep(grid, totals, alerts)


def cost_optimal(labels: np.ndarray, scores: np.ndarray, costs: CostMatrix) -> float:
    return sweep(labels, scores, costs).optimum()


def neyman_pearson(scores: np.ndarray, labels: np.ndarray, max_escape_rate: float) -> float:
    """Order-statistic bound guaranteeing the escape rate stays under the contract limit."""
    defect_scores = np.sort(scores[labels == 1])
    if defect_scores.size == 0:
        return 0.0
    index = int(np.floor(max_escape_rate * defect_scores.size))
    return float(defect_scores[min(index, defect_scores.size - 1)])


def alert_budget(scores: np.ndarray, max_alert_rate: float) -> float:
    """ISA-18.2 caps operator alarms; derive the threshold from that budget."""
    return float(np.quantile(scores, 1.0 - max_alert_rate))


def choose(
    labels: np.ndarray,
    scores: np.ndarray,
    costs: CostMatrix,
    max_escape_rate: float | None = None,
    max_alert_rate: float | None = None,
) -> float:
    """Cost optimum, tightened by any contractual or alarm-budget constraint."""
    threshold = cost_optimal(labels, scores, costs)
    if max_escape_rate is not None:
        threshold = min(threshold, neyman_pearson(scores, labels, max_escape_rate))
    if max_alert_rate is not None:
        threshold = max(threshold, alert_budget(scores, max_alert_rate))
    return float(np.clip(threshold, 0.005, 0.995))


def _total_cost(
    labels: np.ndarray, scores: np.ndarray, threshold: float, costs: CostMatrix
) -> float:
    predictions = (scores >= threshold).astype(int)
    _, fp, fn, _ = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return float(fn * costs.escape + fp * costs.overkill)
