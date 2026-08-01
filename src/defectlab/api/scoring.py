"""The served model, loaded once and held for the process lifetime.

Process telemetry only, no images. That is not a shortcut: a real cell has telemetry for every
shot but a photograph only for parts that reach the camera, so a per-shot scoring endpoint is
exactly where the process channel earns its place. The fusion model is the offline result; this
is the online one.

Risk is prior-corrected to the line's base rate on the way out. The training set runs near 57 %
defective and the line runs at 3 %, so an uncorrected probability is wrong by more than an order
of magnitude at the low end and every downstream cost decision inherits the error.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import settings
from ..twin import FEATURES, TwinConfig, run_line, score

TRAIN_SHOTS = 12000
RESEARCH_PREVALENCE = 0.567
MODEL_VERSION = "process-xgboost-1"


@dataclass(frozen=True, slots=True)
class Scorer:
    """A fitted gate plus the prior it reports against."""

    model: object
    source_prevalence: float
    target_prevalence: float
    version: str = MODEL_VERSION

    def risk(self, readings: dict[str, float]) -> float:
        from ..economics import shift

        frame = pd.DataFrame([readings])[list(FEATURES)].to_numpy()
        raw = self.model.score(frame)
        return float(shift(raw, self.source_prevalence, self.target_prevalence)[0])

    @property
    def threshold(self) -> float:
        return float(self.model.threshold)

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
    labels = frame["label"].to_numpy()
    model = fit(frame[list(FEATURES)].to_numpy(), labels, FitConfig(estimator=estimator))
    return Scorer(model, float(labels.mean()), settings.target_defect_rate)
