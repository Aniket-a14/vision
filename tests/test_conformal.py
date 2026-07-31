import numpy as np
import pytest

from defectlab.models.conformal import MondrianConformal, coverage_by_class

ALPHA = 0.1
IMBALANCE = 0.004


def _imbalanced_scores(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Severe imbalance where defects are genuinely hard, as in real defect detection."""
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, IMBALANCE, n)
    scores = np.where(labels == 1, rng.beta(2.0, 2.0, n), rng.beta(1.5, 9.0, n))
    return labels, np.clip(scores, 1e-6, 1 - 1e-6)


def _true_label_nonconformity(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """What marginal split-conformal pools: one minus the score of the observed label."""
    return np.where(labels == 1, 1.0 - scores, scores)


def test_per_class_coverage_holds_under_extreme_imbalance():
    labels, scores = _imbalanced_scores(60000, seed=3)
    split = len(labels) // 2
    model = MondrianConformal(alpha=ALPHA).fit(labels[:split], scores[:split])
    coverage = coverage_by_class(labels[split:], model.predict_sets(scores[split:]))
    for label in (0, 1):
        assert coverage[label] >= 1 - ALPHA - 0.03, f"class {label} under-covered: {coverage}"


def test_marginal_calibration_under_covers_the_minority_class():
    """The reason calibration is class-conditional rather than pooled."""
    labels, scores = _imbalanced_scores(60000, seed=4)
    split = len(labels) // 2
    pooled = np.quantile(_true_label_nonconformity(labels[:split], scores[:split]), 1 - ALPHA)
    defects = labels[split:] == 1
    marginal_coverage = ((1.0 - scores[split:])[defects] <= pooled).mean()

    model = MondrianConformal(alpha=ALPHA).fit(labels[:split], scores[:split])
    mondrian_coverage = coverage_by_class(labels[split:], model.predict_sets(scores[split:]))[1]

    assert marginal_coverage < 1 - ALPHA, "expected marginal CP to under-cover defects"
    assert mondrian_coverage > marginal_coverage


def test_tighter_alpha_widens_the_prediction_sets():
    labels, scores = _imbalanced_scores(40000, seed=5)
    split = len(labels) // 2
    loose = MondrianConformal(alpha=0.2).fit(labels[:split], scores[:split])
    tight = MondrianConformal(alpha=0.01).fit(labels[:split], scores[:split])
    loose_sets = loose.predict_sets(scores[split:])
    tight_sets = tight.predict_sets(scores[split:])
    assert tight_sets.abstention_rate() >= loose_sets.abstention_rate()


def test_sets_partition_into_confident_ambiguous_and_empty():
    labels, scores = _imbalanced_scores(20000, seed=6)
    split = len(labels) // 2
    sets = (
        MondrianConformal(alpha=ALPHA)
        .fit(labels[:split], scores[:split])
        .predict_sets(scores[split:])
    )
    total = sets.is_confident.sum() + sets.is_ambiguous.sum() + sets.is_empty.sum()
    assert total == len(scores[split:])


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        MondrianConformal().predict_sets(np.array([0.5]))


def test_missing_calibration_class_is_an_error():
    with pytest.raises(ValueError, match="no calibration examples"):
        MondrianConformal().fit(np.zeros(10, dtype=int), np.linspace(0, 1, 10))
