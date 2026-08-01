"""The eight Nelson rules, evaluated on sigma zones.

Every rule flags the *last* point of the run that triggered it, which is the point an operator
is standing in front of when the alarm sounds. Flagging the whole run would multiply one event
into nine alarms and is the usual reason SPC dashboards get muted.

Rules 2-8 are pattern detectors, not outlier tests. Their false-alarm rates compound: running
all eight at once takes the in-control alarm rate well above rule 1's 0.27%. That is a
deliberate trade -- Nelson's rules exist to catch drifts that never breach 3 sigma -- but it is
the reason the alert budget in the economics layer is measured against the whole set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Zone boundaries. Zone C is within 1 sigma, B is 1-2, A is 2-3; the rules are written in them.
ONE_SIGMA = 1.0
TWO_SIGMA = 2.0
THREE_SIGMA = 3.0

RULES: tuple[str, ...] = (
    "beyond_3_sigma",
    "nine_one_side",
    "six_trending",
    "fourteen_alternating",
    "two_of_three_beyond_2",
    "four_of_five_beyond_1",
    "fifteen_hugging",
    "eight_avoiding",
)


def evaluate(zones: np.ndarray) -> pd.DataFrame:
    """One boolean column per rule, indexed like the input."""
    array = np.asarray(zones, dtype=np.float64)
    checks = (
        _beyond_3_sigma,
        _nine_one_side,
        _six_trending,
        _fourteen_alternating,
        _two_of_three_beyond_2,
        _four_of_five_beyond_1,
        _fifteen_hugging,
        _eight_avoiding,
    )
    return pd.DataFrame({name: check(array) for name, check in zip(RULES, checks, strict=True)})


def any_signal(zones: np.ndarray) -> np.ndarray:
    return evaluate(zones).to_numpy().any(axis=1)


def _beyond_3_sigma(zones: np.ndarray) -> np.ndarray:
    """Rule 1: a single point outside the limits."""
    return np.abs(zones) > THREE_SIGMA


def _nine_one_side(zones: np.ndarray) -> np.ndarray:
    """Rule 2: nine consecutive points on one side of centre -- a shift."""
    return _run(zones > 0.0, 9) | _run(zones < 0.0, 9)


def _six_trending(zones: np.ndarray) -> np.ndarray:
    """Rule 3: six points all rising or all falling. Six points is five differences."""
    steps = np.diff(zones, prepend=zones[0])
    return _run(steps > 0.0, 5) | _run(steps < 0.0, 5)


def _fourteen_alternating(zones: np.ndarray) -> np.ndarray:
    """Rule 4: fourteen points zig-zagging -- overcontrol, not natural variation.

    Fourteen points are thirteen differences, and thirteen differences give twelve sign changes.
    """
    steps = np.diff(zones, prepend=zones[0])
    alternates = np.zeros(len(zones), dtype=bool)
    alternates[2:] = steps[2:] * steps[1:-1] < 0.0
    return _run(alternates, 12)


def _two_of_three_beyond_2(zones: np.ndarray) -> np.ndarray:
    """Rule 5: two of three consecutive points past 2 sigma on the same side."""
    return _k_of_n(zones > TWO_SIGMA, 2, 3) | _k_of_n(zones < -TWO_SIGMA, 2, 3)


def _four_of_five_beyond_1(zones: np.ndarray) -> np.ndarray:
    """Rule 6: four of five consecutive points past 1 sigma on the same side."""
    return _k_of_n(zones > ONE_SIGMA, 4, 5) | _k_of_n(zones < -ONE_SIGMA, 4, 5)


def _fifteen_hugging(zones: np.ndarray) -> np.ndarray:
    """Rule 7: fifteen points inside 1 sigma. Too good -- usually stratified sampling."""
    return _run(np.abs(zones) < ONE_SIGMA, 15)


def _eight_avoiding(zones: np.ndarray) -> np.ndarray:
    """Rule 8: eight points all outside 1 sigma, either side -- a bimodal process."""
    return _run(np.abs(zones) > ONE_SIGMA, 8)


def _run(mask: np.ndarray, length: int) -> np.ndarray:
    """Flag position i when mask held for the `length` points ending there."""
    return _rolling_sum(mask, length) == length


def _k_of_n(mask: np.ndarray, k: int, window: int) -> np.ndarray:
    """Flag only when the triggering point itself qualifies, else the alarm lags the event."""
    return (_rolling_sum(mask, window) >= k) & mask


def _rolling_sum(mask: np.ndarray, window: int) -> np.ndarray:
    """Trailing count over `window` points; positions before a full window can never trigger."""
    if len(mask) < window:
        return np.full(len(mask), -1)
    counts = np.cumsum(np.concatenate([[0], mask.astype(int)]))
    rolled = np.full(len(mask), -1)
    rolled[window - 1 :] = counts[window:] - counts[: len(mask) - window + 1]
    return rolled
