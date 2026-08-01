"""Per-shot cost of running a gate at a given threshold.

Three inputs are kept apart deliberately: what the classifier does (per-class error rates),
what the line looks like (prevalence), and what things cost (`CostModel`). Counting raw
confusion-matrix cells would fuse the first two, and the test set runs near 50% defective
while the line runs at 3% -- so raw counts would price a factory that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass(frozen=True, slots=True)
class Outcome:
    """Per-class error rates, which is all a cost model needs from a classifier."""

    escape_rate: float
    overkill_rate: float

    @property
    def recall(self) -> float:
        return 1.0 - self.escape_rate


@dataclass(frozen=True, slots=True)
class Ledger:
    """Expected costs for one policy over `shots` parts. Counts are expectations, not integers."""

    outcome: Outcome
    prevalence: float
    shots: int
    costs: CostModel

    @property
    def defects(self) -> float:
        return self.shots * self.prevalence

    @property
    def escapes(self) -> float:
        return self.defects * self.outcome.escape_rate

    @property
    def caught(self) -> float:
        return self.defects - self.escapes

    @property
    def overkills(self) -> float:
        return self.shots * (1.0 - self.prevalence) * self.outcome.overkill_rate

    @property
    def alert_rate(self) -> float:
        """What the operator sees: flags per shot at the line's prior, not the test set's."""
        return (self.caught + self.overkills) / self.shots

    @property
    def prevention(self) -> float:
        return self.shots * self.costs.prevention_per_shot

    @property
    def appraisal(self) -> float:
        """Every flagged part is inspected, whether or not the flag was right."""
        return (self.caught + self.overkills) * self.costs.inspection

    @property
    def internal_failure(self) -> float:
        """A caught defect is still scrapped; detection saves the escape, not the part."""
        return self.caught * self.costs.scrap

    @property
    def external_failure(self) -> float:
        return self.escapes * self.costs.escape

    @property
    def total(self) -> float:
        return self.prevention + self.appraisal + self.internal_failure + self.external_failure

    @property
    def per_shot(self) -> float:
        return self.total / self.shots

    @property
    def copq(self) -> float:
        """Cost of poor quality: everything a perfect process would remove."""
        return self.appraisal + self.internal_failure + self.external_failure

    def frame(self) -> pd.Series:
        return pd.Series(
            {
                "prevention": self.prevention,
                "appraisal": self.appraisal,
                "internal_failure": self.internal_failure,
                "external_failure": self.external_failure,
                "total": self.total,
                "per_shot": self.per_shot,
                "copq": self.copq,
            }
        )


def outcome(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Outcome:
    """Error rates conditional on the true class, so the test prevalence divides out."""
    flagged = np.asarray(scores) >= threshold
    truth = np.asarray(labels).astype(bool)
    return Outcome(escape_rate=_rate(~flagged[truth]), overkill_rate=_rate(flagged[~truth]))


def ledger(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    prevalence: float,
    costs: CostModel,
    shots: int = 1000,
) -> Ledger:
    return Ledger(outcome(labels, scores, threshold), prevalence, shots, costs)


def ship_everything(prevalence: float, costs: CostModel, shots: int = 1000) -> Ledger:
    """No gate at all: every defect reaches the customer. The do-nothing baseline."""
    return Ledger(Outcome(1.0, 0.0), prevalence, shots, _uninstrumented(costs))


def inspect_everything(prevalence: float, costs: CostModel, shots: int = 1000) -> Ledger:
    """100% manual inspection: no escapes, but appraisal is paid on every shot."""
    return Ledger(Outcome(0.0, 1.0), prevalence, shots, _uninstrumented(costs))


def _uninstrumented(costs: CostModel) -> CostModel:
    """A baseline runs no model, so it does not carry the model's prevention cost."""
    return CostModel(costs.scrap, costs.inspection, costs.escape_multiplier, 0.0)


def _rate(errors: np.ndarray) -> float:
    """An empty class has no measurable error rate; report zero rather than dividing by it."""
    return float(np.mean(errors)) if errors.size else 0.0
