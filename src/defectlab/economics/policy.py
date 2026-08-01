"""Choosing the operating point on money rather than on a metric.

`models.thresholds` minimises a cost matrix over the data as it arrives. This does the same
minimisation over the line as it actually runs: probabilities corrected to the deployment
prior, and the two error rates weighted by that prior instead of by the test mixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel
from .ledger import Ledger, inspect_everything, ledger, ship_everything

GRID_POINTS = 199
GRID_BOUNDS = (0.005, 0.995)


@dataclass(frozen=True, slots=True)
class Operating:
    """The chosen threshold and what it costs against the two do-nothing alternatives."""

    threshold: float
    gate: Ledger
    ship: Ledger
    inspect: Ledger

    @property
    def savings_vs_ship(self) -> float:
        return self.ship.total - self.gate.total

    @property
    def savings_vs_inspect(self) -> float:
        return self.inspect.total - self.gate.total

    def frame(self) -> pd.DataFrame:
        rows = {"gate": self.gate, "ship_all": self.ship, "inspect_all": self.inspect}
        return pd.DataFrame({name: entry.frame() for name, entry in rows.items()}).T


def cost_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    prevalence: float,
    costs: CostModel,
    shots: int = 1000,
) -> pd.DataFrame:
    """Cost per shot across the threshold grid, with the alert rate the line would see."""
    grid = np.linspace(*GRID_BOUNDS, GRID_POINTS)
    entries = [ledger(labels, scores, t, prevalence, costs, shots) for t in grid]
    return pd.DataFrame(
        {
            "threshold": grid,
            "per_shot": [entry.per_shot for entry in entries],
            "escape_rate": [entry.outcome.escape_rate for entry in entries],
            "overkill_rate": [entry.outcome.overkill_rate for entry in entries],
            "alert_rate": [entry.alert_rate for entry in entries],
        }
    )


def optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    prevalence: float,
    costs: CostModel,
    shots: int = 1000,
) -> float:
    curve = cost_curve(labels, scores, prevalence, costs, shots)
    return float(curve.loc[curve["per_shot"].idxmin(), "threshold"])


def operate(
    labels: np.ndarray,
    scores: np.ndarray,
    prevalence: float,
    costs: CostModel,
    shots: int = 1000,
) -> Operating:
    """The full economic verdict for one model: where to sit, and what it is worth."""
    threshold = optimal_threshold(labels, scores, prevalence, costs, shots)
    return Operating(
        threshold=threshold,
        gate=ledger(labels, scores, threshold, prevalence, costs, shots),
        ship=ship_everything(prevalence, costs, shots),
        inspect=inspect_everything(prevalence, costs, shots),
    )
