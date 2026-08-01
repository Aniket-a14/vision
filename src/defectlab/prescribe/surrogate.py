"""A fast stand-in for the twin, fitted on randomised interventional data.

Two things this is not, stated plainly because both are easy to overclaim.

It is not fitted on production data. Historian data is confounded: the line dynamics correlate
pour temperature with die temperature and both with time-in-shift, so a model fitted on it
learns associations that do not survive intervention. Training instead on
`sample_uniform_envelope` -- every parameter drawn independently across its full range -- is the
do-operator by construction, and it is the only reason the resulting advice can be read causally
at all.

It is also not evidence that the advice is right. The surrogate is fitted to the twin, so it
inherits every one of the twin's coefficients, and testing the advice against the same twin
would be circular. `robustness` is where that is addressed, by perturbing those coefficients
and asking whether the advice survives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from ..config import settings
from ..twin import FEATURES, TwinConfig, run_line
from ..twin.parameters import sample_uniform_envelope
from ..twin.propensity import calibrate_intercept, evaluate

DEFAULT_SHOTS = 20000
DEFAULT_SEED = 42
DEFAULT_BASE_RATE = settings.target_defect_rate


@dataclass(frozen=True, slots=True)
class Surrogate:
    """Maps a full parameter vector to the twin's deterministic defect logit."""

    model: object
    names: tuple[str, ...]
    intercept: float

    def logit(self, frame: pd.DataFrame) -> np.ndarray:
        """The margin, which is what the search optimises; probability saturates and goes flat."""
        raw = np.asarray(self.model.predict(frame[list(self.names)]), dtype=np.float64)
        return raw + self.intercept

    def risk(self, frame: pd.DataFrame) -> np.ndarray:
        return expit(self.logit(frame))

    def logit_of(self, reading: dict[str, float]) -> float:
        return float(self.logit(pd.DataFrame([reading]))[0])

    def risk_of(self, reading: dict[str, float]) -> float:
        return float(self.risk(pd.DataFrame([reading]))[0])


def fit(
    shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
    config: TwinConfig | None = None,
    base_rate: float = DEFAULT_BASE_RATE,
) -> Surrogate:
    """Fit on an interventional design. Noise is off: the target is the signal, not a coin flip."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    settings = config or TwinConfig(seed=seed)
    design = sample_uniform_envelope(shots, np.random.default_rng(seed))
    model = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
    model.fit(design[list(FEATURES)], _logit_of(design, settings))
    return Surrogate(model, FEATURES, _intercept(model, settings, shots, base_rate))


def _intercept(model, config: TwinConfig, shots: int, base_rate: float) -> float:
    """Calibrated on a *line* sample, not on the uniform design.

    The design spans the whole machine envelope, most of which the line never visits, so its
    prevalence is not the line's. Without this the surrogate reports risks pinned at 1.0 and
    every recommendation reads as changing nothing.
    """
    line = run_line(shots, config)
    raw = np.asarray(model.predict(line[list(FEATURES)]), dtype=np.float64)
    return calibrate_intercept(raw, base_rate)


def _logit_of(design: pd.DataFrame, config: TwinConfig) -> np.ndarray:
    """noise_sd = 0 makes this the twin's mean response, which is what a recommender needs."""
    result = evaluate(
        design, np.random.default_rng(0), noise_sd=0.0, signal_gain=config.signal_gain
    )
    return result.logit
