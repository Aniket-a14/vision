"""Turns fitted control charts into the long `fact_spc` shape.

One row per shot per chart, with the limits carried on every row. Repeating the centre line and
limits is redundant in a database and exactly right for a report: a Power BI line chart draws
limits as three more series, and asking it to join them back on would be slower and more
fragile than storing four extra floats per row.

The charts are fitted on a Phase I prefix and applied to the whole run, which is the only
arrangement that lets a drift show up. Re-fitting on everything would absorb the drift.

The risk chart is drawn on the **logit** of the score, not the score. A Shewhart chart assumes
an approximately normal statistic, and a probability bounded on [0, 1] with most of its mass
near zero is not that: measured on one run the raw score has skew +4.25 and excess kurtosis
+18.9, and the chart signals on 36 % of shots -- points bunched inside 1 sigma trip rule 7 while
the tail simultaneously trips rule 1. The logit has skew +1.02 and kurtosis +1.19, and the
signal rate falls to 15 %. Transforming toward normality before charting is the textbook
remedy, and it is the same margin-not-probability rule the rest of this codebase follows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit

from ..spc import RULES, apply_i_mr, fit_i_mr
from ..spc.charts import Shewhart

PHASE_ONE_FRACTION = 0.4
RISK_CHART = "risk_residual"
RISK_CLIP = 1e-6

# Below this there is no correlation worth removing, and an AR(1) step would add machinery
# that explains nothing. Above it, an I-MR chart on the raw series is simply invalid.
MIN_AUTOCORRELATION = 0.10
MIN_POINTS_FOR_AR1 = 3
NO_RULE = ""

# The moving-range chart signals on its own and is not one of Nelson's eight, so a row can be
# out of control with no rule attached. The dashboard must never draw a signal it cannot name.
RANGE_RULE = "moving_range"


def risk_chart(risk: np.ndarray, phase_one: float = PHASE_ONE_FRACTION) -> pd.DataFrame:
    """An I-MR chart on the model's own output: the drift monitor for the deployed gate."""
    margin = logit(np.clip(np.asarray(risk, dtype=np.float64), RISK_CLIP, 1.0 - RISK_CLIP))
    cut = _cut(len(margin), phase_one)
    series = _decorrelate(margin, cut)
    chart = fit_i_mr(series[:cut])
    return _long(RISK_CHART, chart, apply_i_mr(chart, series))


def _decorrelate(margin: np.ndarray, cut: int) -> np.ndarray:
    """Chart the AR(1) residual, because the raw risk score violates independence.

    A production risk score inherits the line's slow structure -- lot chemistry, die thermal
    state, tool wear -- so consecutive shots are correlated (measured: lag-1 0.23, still 0.16 at
    lag 20). An I-MR chart estimates sigma from the range of *adjacent* points, which under
    positive autocorrelation is too small, so the limits come out too tight and the chart fires
    on a third of all shots, almost all of it rule 2 chasing a wandering mean.

    Charting the residual of a fitted AR(1) is the standard remedy (Montgomery ch. 10): what is
    left after the predictable part is removed is what a special cause would actually disturb.
    """
    phi = _ar1(margin[:cut])
    if phi < MIN_AUTOCORRELATION:
        return margin
    previous = np.concatenate([[float(np.mean(margin[:cut]))], margin[:-1]])
    return margin - phi * previous


def _ar1(series: np.ndarray) -> float:
    """Lag-1 coefficient estimated on Phase I only, so a later drift cannot be absorbed into it."""
    if len(series) < MIN_POINTS_FOR_AR1:
        return 0.0
    centred = series - series.mean()
    denominator = float(np.dot(centred, centred))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(centred[1:], centred[:-1]) / denominator)


def parameter_chart(
    series: np.ndarray, name: str, phase_one: float = PHASE_ONE_FRACTION
) -> pd.DataFrame:
    chart = fit_i_mr(series[: _cut(len(series), phase_one)])
    return _long(name, chart, apply_i_mr(chart, series))


def _long(name: str, chart: Shewhart, applied: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": np.arange(len(applied)),
            "chart": name,
            "value": applied["location"].to_numpy(),
            "centre": chart.location.centre,
            "lower": chart.location.lower,
            "upper": chart.location.upper,
            "signal": applied["signal"].astype(int).to_numpy(),
            "rule": _first_rule(applied),
        }
    )


def _first_rule(applied: pd.DataFrame) -> np.ndarray:
    """One rule per row: the lowest-numbered that fired, so the legend stays readable."""
    flags = applied[list(RULES)].to_numpy()
    named = np.where(flags.any(axis=1), np.array(RULES)[flags.argmax(axis=1)], NO_RULE)
    dispersion = applied["dispersion_signal"].to_numpy()
    return np.where((named == NO_RULE) & dispersion, RANGE_RULE, named)


def _cut(count: int, fraction: float) -> int:
    return max(int(count * fraction), 2)
