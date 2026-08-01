"""Control limits, estimated once on Phase I and then frozen.

The freezing is the whole point. If limits are re-estimated as new data arrives, a slow drift
widens the limits at exactly the rate it moves the mean, and the chart never signals -- the
process is judged against whatever it has become rather than against what it was capable of.
Phase I fits the limits on a window certified stable; Phase II only ever applies them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGMA_LIMIT = 3.0


@dataclass(frozen=True, slots=True)
class ControlLimits:
    """Centre line and 3-sigma limits. Sigma is the within-subgroup estimate, not the overall sd."""

    centre: float
    sigma: float
    lower: float
    upper: float

    def zones(self, values: np.ndarray) -> np.ndarray:
        """Signed distance from centre in sigma units; every Nelson rule reads this."""
        if self.sigma == 0.0:
            return np.zeros(len(np.asarray(values)))
        return (np.asarray(values, dtype=np.float64) - self.centre) / self.sigma

    def outside(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return (array < self.lower) | (array > self.upper)


def symmetric(centre: float, sigma: float, width: float = SIGMA_LIMIT) -> ControlLimits:
    return ControlLimits(centre, sigma, centre - width * sigma, centre + width * sigma)


def bounded(centre: float, sigma: float, lower: float, upper: float) -> ControlLimits:
    """Range and moving-range limits are asymmetric and cannot go below zero."""
    return ControlLimits(centre, sigma, lower, upper)
