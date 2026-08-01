"""X-bar/R, I-MR and EWMA charts, each fitted on Phase I and applied to Phase II.

`fit_*` estimates limits and returns a frozen object; `apply` never touches them again. Sigma
always comes from *within*-subgroup variation (the mean range), never from the overall standard
deviation: if the process has already shifted, the overall sd contains the shift and would
widen the limits enough to hide it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import nelson
from .constants import D2_INDIVIDUALS, E2_INDIVIDUALS, factors
from .limits import ControlLimits, bounded, symmetric

DEFAULT_LAMBDA = 0.2
DEFAULT_EWMA_WIDTH = 3.0
MIN_POINTS = 2


@dataclass(frozen=True, slots=True)
class Shewhart:
    """A location chart and its paired dispersion chart, both frozen.

    `process_sigma` describes one part; `location.sigma` describes the statistic being plotted.
    For an X-bar chart those differ by sqrt(n), and the Nelson rules read the latter -- zone
    tests on a subgroup mean scaled by the individual sigma would never fire.
    """

    location: ControlLimits
    dispersion: ControlLimits
    subgroup: int
    process_sigma: float


def fit_xbar_r(phase_one: np.ndarray) -> Shewhart:
    """Phase I over subgroups shaped (m, n). Requires a stable window; it does not check."""
    groups = _as_subgroups(phase_one)
    constants = factors(groups.shape[1])
    mean_range = float(_ranges(groups).mean())
    process_sigma = mean_range / constants.d2
    location = symmetric(float(groups.mean()), process_sigma / np.sqrt(groups.shape[1]))
    dispersion = bounded(
        mean_range, process_sigma, constants.d3 * mean_range, constants.d4 * mean_range
    )
    return Shewhart(location, dispersion, groups.shape[1], process_sigma)


def apply_xbar_r(chart: Shewhart, subgroups: np.ndarray) -> pd.DataFrame:
    groups = _as_subgroups(subgroups, chart.subgroup)
    return _signals(groups.mean(axis=1), _ranges(groups), chart)


def fit_i_mr(phase_one: np.ndarray) -> Shewhart:
    """Individuals: no subgroups, so sigma comes from the mean moving range of successive pairs."""
    series = _as_series(phase_one)
    mean_range = float(_moving_range(series).mean())
    sigma = mean_range / D2_INDIVIDUALS
    centre = float(series.mean())
    location = ControlLimits(
        centre, sigma, centre - E2_INDIVIDUALS * mean_range, centre + E2_INDIVIDUALS * mean_range
    )
    dispersion = bounded(mean_range, sigma, 0.0, factors(MIN_POINTS).d4 * mean_range)
    return Shewhart(location, dispersion, 1, sigma)


def apply_i_mr(chart: Shewhart, series: np.ndarray) -> pd.DataFrame:
    values = _as_series(series)
    ranges = np.concatenate([[np.nan], np.abs(np.diff(values))])
    return _signals(values, ranges, chart)


@dataclass(frozen=True, slots=True)
class Ewma:
    """EWMA limits widen with the run, so they are a formula rather than two numbers."""

    centre: float
    sigma: float
    lam: float
    width: float

    def band(self, count: int) -> np.ndarray:
        steps = np.arange(1, count + 1)
        variance = (self.lam / (2.0 - self.lam)) * (1.0 - (1.0 - self.lam) ** (2 * steps))
        return self.width * self.sigma * np.sqrt(variance)

    def statistic(self, series: np.ndarray) -> np.ndarray:
        """Recursive by definition; the loop is the formula, not an unvectorised mistake."""
        values = _as_series(series)
        smoothed = np.empty(len(values))
        previous = self.centre
        for position, value in enumerate(values):
            previous = self.lam * value + (1.0 - self.lam) * previous
            smoothed[position] = previous
        return smoothed


def fit_ewma(
    phase_one: np.ndarray, lam: float = DEFAULT_LAMBDA, width: float = DEFAULT_EWMA_WIDTH
) -> Ewma:
    """Sigma from the moving range again, so a shift inside Phase I cannot inflate it."""
    series = _as_series(phase_one)
    sigma = float(_moving_range(series).mean()) / D2_INDIVIDUALS
    return Ewma(float(series.mean()), sigma, lam, width)


def apply_ewma(chart: Ewma, series: np.ndarray) -> pd.DataFrame:
    """EWMA detects small sustained shifts that Shewhart misses; only rule 1 applies to it."""
    smoothed = chart.statistic(series)
    band = chart.band(len(smoothed))
    return pd.DataFrame(
        {
            "ewma": smoothed,
            "lower": chart.centre - band,
            "upper": chart.centre + band,
            "signal": np.abs(smoothed - chart.centre) > band,
        }
    )


def _signals(location: np.ndarray, dispersion: np.ndarray, chart: Shewhart) -> pd.DataFrame:
    """Nelson runs on the location chart; the dispersion chart only tests its own limits."""
    frame = nelson.evaluate(chart.location.zones(location))
    frame.insert(0, "location", location)
    frame.insert(1, "dispersion", dispersion)
    frame["dispersion_signal"] = chart.dispersion.outside(np.nan_to_num(dispersion, nan=0.0))
    frame["signal"] = frame[list(nelson.RULES)].to_numpy().any(axis=1) | frame["dispersion_signal"]
    return frame


def _ranges(groups: np.ndarray) -> np.ndarray:
    return groups.max(axis=1) - groups.min(axis=1)


def _moving_range(series: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(series))


def _as_subgroups(values: np.ndarray, expected: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != MIN_POINTS:
        raise ValueError(f"subgroups must be 2-D (m, n); got shape {array.shape}")
    if expected is not None and array.shape[1] != expected:
        raise ValueError(f"chart was fitted for subgroups of {expected}; got {array.shape[1]}")
    return array


def _as_series(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    if len(array) < MIN_POINTS:
        raise ValueError("an individuals chart needs at least two points to form a moving range")
    return array
