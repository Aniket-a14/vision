"""Prior correction from the research prevalence to the line's.

The training sets here run near 50% defective; a real HPDC cell scraps 2-4%. A model
calibrated at 50% reports probabilities an order of magnitude too high at the low end, and
every cost figure built on them is wrong by the same factor. Correcting the threshold
without correcting the probabilities would hide the error rather than fix it.

This is the Elkan / Saerens correction: rescale the odds by the ratio of priors. It assumes
label shift -- the class prior moves but p(x | y) does not. That holds by construction here,
because prevalence is set by oversampling the same physics, not by changing the process.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-9


def prevalence(labels: np.ndarray) -> float:
    return float(np.mean(labels))


def shift(scores: np.ndarray, source: float, target: float) -> np.ndarray:
    """Map probabilities calibrated at `source` prevalence onto a line running at `target`."""
    _check(source, "source")
    _check(target, "target")
    clipped = np.clip(np.asarray(scores, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    positive = clipped * (target / source)
    negative = (1.0 - clipped) * ((1.0 - target) / (1.0 - source))
    return positive / (positive + negative)


def _check(value: float, name: str) -> None:
    """A prior of 0 or 1 makes the odds ratio undefined, not merely extreme."""
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} prevalence must lie strictly in (0, 1); got {value}")
