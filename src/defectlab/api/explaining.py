"""Per-shot explanation for the inspector panel.

An anchor needs a background sample to perturb against, and drawing one costs a twin run. It is
drawn once at first use and held, so the first explanation is slow and the rest are not.

The rule is grown against the *served* threshold, not against 0.5. An explanation of a decision
the gate did not make would be worse than no explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..economics import shift
from ..explain import anchor
from ..twin import FEATURES, TwinConfig, run_line
from .scoring import Scorer

BACKGROUND_SHOTS = 2000
ANCHOR_SAMPLES = 800


@dataclass(frozen=True, slots=True)
class Explainer:
    """A background sample plus the scorer whose decisions it explains."""

    scorer: Scorer
    background: np.ndarray

    def rule(self, readings: dict[str, float]) -> dict:
        instance = _row(readings)
        found = anchor(
            self._label,
            instance,
            self.background,
            list(FEATURES),
            samples=ANCHOR_SAMPLES,
        )
        return {
            "rule": found.rule(),
            "prediction": found.prediction,
            "precision": round(found.precision, 4),
            "coverage": round(found.coverage, 4),
            "predicates": [
                {"parameter": p.name, "lower": p.lower, "upper": p.upper} for p in found.predicates
            ],
        }

    def _label(self, features: np.ndarray) -> np.ndarray:
        """The gate's own verdict, on corrected scores at the served threshold."""
        raw = self.scorer.model.score(features)
        risk = shift(raw, self.scorer.source_prevalence, self.scorer.target_prevalence)
        return (risk >= self.scorer.threshold).astype(int)


def build(scorer: Scorer, seed: int = 42) -> Explainer:
    config = TwinConfig(seed=seed + 1)
    frame = run_line(BACKGROUND_SHOTS, config)
    return Explainer(scorer, frame[list(FEATURES)].to_numpy())


def _row(readings: dict[str, float]) -> np.ndarray:
    return pd.DataFrame([readings])[list(FEATURES)].to_numpy()[0]
