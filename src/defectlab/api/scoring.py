"""The served model, loaded once and held for the process lifetime.

Process telemetry only, no images. That is not a shortcut: a real cell has telemetry for every
shot but a photograph only for parts that reach the camera, so a per-shot scoring endpoint is
exactly where the process channel earns its place. The fusion model is the offline result; this
is the online one.

Risk is prior-corrected to the line's base rate on the way out. The training set runs near 57 %
defective and the line runs at 3 %, so an uncorrected probability is wrong by more than an order
of magnitude at the low end and every downstream cost decision inherits the error.

The threshold has to make the same journey. `FittedModel.threshold` is the cost optimum at the
*research* prevalence, so comparing a 3 %-scale risk against it puts the two sides of the
inequality on different scales -- which flagged 83 % of a nominal line before it was caught.
The served threshold is re-chosen on corrected scores at the deployment prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import settings
from ..twin import FEATURES, TwinConfig, run_line, score

TRAIN_SHOTS = 12000
RESEARCH_PREVALENCE = 0.567
MODEL_VERSION = "process-xgboost-2"
HOLDOUT_FRACTION = 0.25

# ISA-18.2 calls 12 alarms/hour the upper bound of a sustainable operator load.
ALARMS_PER_HOUR = 12.0
SECONDS_PER_SHOT = 60.0
MAX_ALERT_RATE = ALARMS_PER_HOUR * SECONDS_PER_SHOT / 3600.0


@dataclass(frozen=True, slots=True)
class Scorer:
    """A fitted gate, the prior it reports against, and the threshold that pairs with both."""

    model: object
    source_prevalence: float
    target_prevalence: float
    threshold: float
    version: str = MODEL_VERSION

    def risk(self, readings: dict[str, float]) -> float:
        from ..economics import shift

        frame = pd.DataFrame([readings])[list(FEATURES)].to_numpy()
        raw = self.model.score(frame)
        return float(shift(raw, self.source_prevalence, self.target_prevalence)[0])

    def prediction_set(self, readings: dict[str, float]) -> tuple[list[int], bool]:
        """Mondrian conformal set, plus whether the model is declining to choose.

        An empty set abstains too, and for a worse reason than an ambiguous one: the shot is
        outside anything the calibration set covered.
        """
        frame = pd.DataFrame([readings])[list(FEATURES)].to_numpy()
        sets = self.model.predict_sets(frame)
        labels = [
            label
            for label, present in ((0, sets.contains_ok[0]), (1, sets.contains_defect[0]))
            if bool(present)
        ]
        return labels, len(labels) != 1


def build(seed: int = 42, estimator: str = "xgboost") -> Scorer:
    """Fit on a fresh twin run. Deterministic given the seed, so a restart serves the same model."""
    from ..models.pipeline import FitConfig, fit

    config = TwinConfig(seed=seed)
    frame = score(run_line(TRAIN_SHOTS, config), config, target_prevalence=RESEARCH_PREVALENCE)
    features, labels = frame[list(FEATURES)].to_numpy(), frame["label"].to_numpy()
    train, holdout = _split(len(labels), seed)
    model = fit(features[train], labels[train], FitConfig(estimator=estimator))
    source = float(labels[train].mean())
    target = settings.target_defect_rate
    return Scorer(model, source, target, _operating_point(model, features, labels, holdout, source))


def _split(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The threshold is chosen on shots the estimator never saw, fitted or calibrated."""
    order = np.random.default_rng(seed).permutation(count)
    cut = int(count * (1.0 - HOLDOUT_FRACTION))
    return order[:cut], order[cut:]


def _operating_point(
    model, features: np.ndarray, labels: np.ndarray, holdout: np.ndarray, source: float
) -> float:
    """Cost optimum at the line's prior, then raised to fit the operator's alarm budget.

    The budget binds here and does not in the offline study. Process telemetry alone barely
    separates the classes, so with an escape at 100x an inspection the unconstrained optimum
    wants to inspect most of the line -- economically right, operationally unusable.
    """
    from ..economics import CostModel, optimal_threshold, shift
    from ..models.thresholds import alert_budget

    target = settings.target_defect_rate
    corrected = shift(model.score(features[holdout]), source, target)
    optimum = optimal_threshold(labels[holdout], corrected, target, CostModel())
    return float(max(optimum, alert_budget(corrected, MAX_ALERT_RATE)))
