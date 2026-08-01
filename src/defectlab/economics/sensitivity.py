"""How much of the answer is the model, and how much is a number that was guessed.

The escape multiplier M and the deployment prevalence are both estimates, and the savings
figure is roughly linear in each. Reporting one headline number would hand the reader a
precision that neither input supports, so both are swept and the range is the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import ESCAPE_MULTIPLIER_RANGE, CostModel
from .policy import operate

MULTIPLIER_POINTS = 9
PREVALENCE_RANGE = (0.01, 0.06)
PREVALENCE_POINTS = 6


def multiplier_sweep(
    labels: np.ndarray,
    scores: np.ndarray,
    prevalence: float,
    costs: CostModel,
    multipliers: np.ndarray | None = None,
    shots: int = 1000,
) -> pd.DataFrame:
    """Vary only the external-failure multiplier; every other cost is held fixed."""
    grid = _default_multipliers() if multipliers is None else np.asarray(multipliers)
    rows = [_row(labels, scores, prevalence, costs.with_multiplier(m), shots) for m in grid]
    return pd.DataFrame(rows).assign(escape_multiplier=grid)


def prevalence_sweep(
    labels: np.ndarray,
    scores: np.ndarray,
    costs: CostModel,
    prevalences: np.ndarray | None = None,
    shots: int = 1000,
) -> pd.DataFrame:
    """Vary the line's defect rate. The scores are assumed already corrected to each prior."""
    grid = _default_prevalences() if prevalences is None else np.asarray(prevalences)
    rows = [_row(labels, scores, p, costs, shots) for p in grid]
    return pd.DataFrame(rows).assign(prevalence=grid)


def _row(
    labels: np.ndarray, scores: np.ndarray, prevalence: float, costs: CostModel, shots: int
) -> dict[str, float]:
    point = operate(labels, scores, prevalence, costs, shots)
    return {
        "threshold": point.threshold,
        "per_shot": point.gate.per_shot,
        "copq": point.gate.copq,
        "savings_vs_ship": point.savings_vs_ship,
        "savings_vs_inspect": point.savings_vs_inspect,
    }


def _default_multipliers() -> np.ndarray:
    return np.linspace(*ESCAPE_MULTIPLIER_RANGE, MULTIPLIER_POINTS)


def _default_prevalences() -> np.ndarray:
    return np.linspace(*PREVALENCE_RANGE, PREVALENCE_POINTS)
