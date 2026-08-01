"""Taguchi quadratic loss: in spec is not the same as on target.

A shot poured 2 C off nominal and one poured 40 C off both pass a conformance check, and
they do not cost the same. The quadratic loss L = k (y - m)^2 prices the distance itself,
with k fixed by a single anchor: the loss A0 incurred at the tolerance D0.

D0 is a customer tolerance, and the twin does not define one -- its `lower` and `upper` are
machine limits, which is a different thing entirely. So the tolerance used here is
TOLERANCE_SIGMAS process standard deviations, the width a Cp = 1.0 process would just hold.
That is an assumption, stated here rather than buried inside a constant.

It also makes the absolute loss unreliable: a parameter drawn at its own declared spread has
expected loss A0 / TOLERANCE_SIGMAS^2 whatever A0 is, so summing ten parameters can exceed the
value of the part. Read `baseline_ratio`, not `mean_loss`. The ratio divides that artefact out
and leaves only what the line dynamics did -- drift, autocorrelation, lot structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..twin import FEATURES, spec
from .costs import CostModel

TOLERANCE_SIGMAS = 3.0


@dataclass(frozen=True, slots=True)
class QualityLoss:
    """One parameter's loss curve, anchored at the tolerance limit."""

    name: str
    target: float
    tolerance: float
    loss_at_tolerance: float

    @property
    def k(self) -> float:
        return self.loss_at_tolerance / self.tolerance**2

    def loss(self, values: np.ndarray) -> np.ndarray:
        return self.k * (np.asarray(values, dtype=np.float64) - self.target) ** 2


def for_parameter(
    name: str, loss_at_tolerance: float, sigmas: float = TOLERANCE_SIGMAS
) -> QualityLoss:
    bounds = spec(name)
    return QualityLoss(name, bounds.nominal, sigmas * bounds.spread, loss_at_tolerance)


def loss_table(
    frame: pd.DataFrame,
    costs: CostModel | None = None,
    sigmas: float = TOLERANCE_SIGMAS,
) -> pd.DataFrame:
    """Mean loss per parameter, ranked, against the loss an undisturbed draw would already incur."""
    anchor = (costs or CostModel()).scrap
    curves = [for_parameter(name, anchor, sigmas) for name in _columns(frame)]
    means = pd.Series({c.name: float(np.mean(c.loss(frame[c.name]))) for c in curves})
    table = means.sort_values(ascending=False).rename("mean_loss").to_frame()
    table["share"] = table["mean_loss"] / table["mean_loss"].sum()
    table["baseline_ratio"] = table["mean_loss"] / _baseline(anchor, sigmas)
    return table.reset_index(names="parameter")


def _baseline(loss_at_tolerance: float, sigmas: float) -> float:
    """Expected loss for a parameter drawn at exactly its declared spread: A0 / sigmas^2."""
    return loss_at_tolerance / sigmas**2


def total_loss(
    frame: pd.DataFrame,
    costs: CostModel | None = None,
    sigmas: float = TOLERANCE_SIGMAS,
) -> np.ndarray:
    """Per-shot off-target loss, summed over parameters. Additive by Taguchi's own convention."""
    anchor = (costs or CostModel()).scrap
    curves = [for_parameter(name, anchor, sigmas) for name in _columns(frame)]
    return np.sum([curve.loss(frame[curve.name]) for curve in curves], axis=0)


def _columns(frame: pd.DataFrame) -> list[str]:
    """Tool wear has no nominal to sit on, so it is excluded rather than scored against 25k."""
    return [name for name in FEATURES if name in frame.columns and name != "tool_wear_shots"]
