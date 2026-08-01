"""Does the advice survive the simulator being wrong?

The surrogate is fitted to the twin, so scoring advice against that same twin only proves the
surrogate copied it. The question worth asking is whether the advice still helps when the
twin's mechanism weights -- which are literature-calibrated estimates, not measurements -- are
wrong by a realistic margin.

So each trial redraws every weight by a multiplicative factor and re-evaluates the true
propensity before and after the move. A recommendation that lowers risk in almost every
perturbed world is telling us about the physics; one that only works at the nominal weights is
telling us about the fit.

The mechanism *indices* are unchanged by this: perturbing weights asks "what if shrinkage
matters more than we thought", not "what if the physics is different". That is the honest scope
of the test and it is narrower than full model misspecification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..twin import MECHANISM_WEIGHTS, TwinConfig
from ..twin.propensity import mechanism_indices, weighted_contributions
from .actions import apply
from .search import Recommendation

DEFAULT_SCALE = 0.35
DEFAULT_TRIALS = 200
SCALE_GRID = (0.2, 0.35, 0.5)


@dataclass(frozen=True, slots=True)
class Stability:
    """How often the advice still helped once the weights were disturbed."""

    trials: int
    helped: int
    scale: float
    median_margin_gain: float

    @property
    def rate(self) -> float:
        return self.helped / self.trials if self.trials else 0.0


def perturbed_weights(rng: np.random.Generator, scale: float) -> dict[str, float]:
    """Multiplicative, so a weight cannot change sign; a negative mechanism is not physical."""
    factors = rng.uniform(1.0 - scale, 1.0 + scale, len(MECHANISM_WEIGHTS))
    pairs = zip(MECHANISM_WEIGHTS.items(), factors, strict=True)
    return {name: weight * factor for (name, weight), factor in pairs}


def stability(
    recommendation: Recommendation,
    reading: dict[str, float],
    *,
    trials: int = DEFAULT_TRIALS,
    scale: float = DEFAULT_SCALE,
    seed: int = 0,
    config: TwinConfig | None = None,
) -> Stability:
    """Fraction of perturbed worlds in which the recommended move still lowers true risk."""
    gain = (config or TwinConfig()).signal_gain
    indices = _indices_before_after(reading, recommendation)
    rng = np.random.default_rng(seed)
    deltas = np.array(
        [_improvement(indices, perturbed_weights(rng, scale), gain) for _ in range(trials)]
    )
    return Stability(trials, int((deltas > 0.0).sum()), scale, float(np.median(deltas)))


def scale_sweep(
    recommendation: Recommendation,
    reading: dict[str, float],
    *,
    scales: tuple[float, ...] = SCALE_GRID,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
) -> pd.DataFrame:
    """Stability against how wrong the weights are allowed to be; one number would hide that."""
    rows = [stability(recommendation, reading, trials=trials, scale=s, seed=seed) for s in scales]
    return pd.DataFrame(
        {
            "scale": [row.scale for row in rows],
            "stability": [row.rate for row in rows],
            "median_margin_gain": [row.median_margin_gain for row in rows],
        }
    )


def _indices_before_after(reading: dict[str, float], recommendation: Recommendation):
    """Indices depend only on the parameters, so they are computed once and merely re-weighted."""
    frame = pd.DataFrame([reading, apply(reading, recommendation.actions)])
    return mechanism_indices(frame)


def _improvement(indices: pd.DataFrame, weights: dict[str, float], gain: float) -> float:
    """Measured on the margin, not in probability.

    The perturbed propensity carries no calibrated intercept, so a risky shot sits far out on
    the sigmoid where before and after both round to 1.0 and a real improvement reads as zero.
    The margin drop is scale-free and says how much headroom the move actually bought.
    """
    margins = (weighted_contributions(indices, weights) * gain).sum(axis=1).to_numpy()
    return float(margins[0] - margins[1])
