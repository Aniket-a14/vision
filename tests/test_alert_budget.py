"""An alarm budget only makes sense at a realistic base rate.

Applying ISA-18.2's 5% ceiling to a 50%-prevalence research dataset starves recall,
which is why the budget lives in the economics layer, after prior correction.
"""

import numpy as np
import pytest

from defectlab.models.thresholds import CostMatrix, choose

COSTS = CostMatrix(escape=250.0, overkill=4.0)
BUDGET = 0.05


def _scores(prevalence: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, prevalence, n)
    scores = np.where(labels == 1, rng.beta(5, 3, n), rng.beta(2, 6, n))
    return labels, np.clip(scores, 1e-6, 1 - 1e-6)


def _recall(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    return float((scores[labels == 1] >= threshold).mean())


def test_alert_budget_destroys_recall_at_research_prevalence():
    labels, scores = _scores(prevalence=0.5, n=20000, seed=1)
    unbudgeted = choose(labels, scores, COSTS)
    budgeted = choose(labels, scores, COSTS, max_alert_rate=BUDGET)
    assert _recall(labels, scores, budgeted) < 0.5 * _recall(labels, scores, unbudgeted)


def test_alert_budget_is_workable_at_realistic_prevalence():
    labels, scores = _scores(prevalence=0.03, n=40000, seed=2)
    budgeted = choose(labels, scores, COSTS, max_alert_rate=BUDGET)
    assert _recall(labels, scores, budgeted) > 0.5


def test_budget_caps_the_alarm_rate_either_way():
    for prevalence in (0.03, 0.5):
        labels, scores = _scores(prevalence, 20000, seed=3)
        threshold = choose(labels, scores, COSTS, max_alert_rate=BUDGET)
        assert (scores >= threshold).mean() <= BUDGET + 0.01


def test_pipeline_does_not_apply_a_budget_by_default():
    from defectlab.models.pipeline import DEFAULT_MAX_ALERT_RATE

    assert DEFAULT_MAX_ALERT_RATE is None


def test_expensive_escapes_still_lower_the_threshold_without_a_budget():
    labels, scores = _scores(prevalence=0.03, n=20000, seed=4)
    threshold = choose(labels, scores, COSTS)
    assert threshold < 0.5
    assert _recall(labels, scores, threshold) > 0.8


@pytest.mark.parametrize("budget", [0.01, 0.05, 0.2])
def test_tighter_budgets_raise_the_threshold(budget):
    labels, scores = _scores(prevalence=0.03, n=20000, seed=5)
    loose = choose(labels, scores, COSTS, max_alert_rate=0.5)
    tight = choose(labels, scores, COSTS, max_alert_rate=budget)
    assert tight >= loose
