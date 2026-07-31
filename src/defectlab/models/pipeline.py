"""End-to-end train, calibrate, conformalise and score for one ablation cell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from ..imaging import Regime
from .conformal import MondrianConformal, PredictionSets, coverage_by_class
from .estimators import build, positive_class_scores
from .evaluation import Scores, evaluate
from .features import Modality
from .thresholds import CostMatrix, choose

CALIBRATION_FRACTION = 0.25


@dataclass(frozen=True, slots=True)
class FittedModel:
    estimator: object
    conformal: MondrianConformal
    threshold: float

    def score(self, features: np.ndarray) -> np.ndarray:
        return positive_class_scores(self.estimator, features)

    def predict_sets(self, features: np.ndarray) -> PredictionSets:
        return self.conformal.predict_sets(self.score(features))


@dataclass(frozen=True, slots=True)
class AblationResult:
    modality: Modality
    regime: Regime
    estimator: str
    scores: Scores
    threshold: float
    abstention_rate: float
    conformal_coverage: dict[int, float]

    def as_row(self) -> dict[str, object]:
        return {
            "modality": self.modality.value,
            "regime": self.regime.value,
            "estimator": self.estimator,
            "threshold": self.threshold,
            "abstention_rate": self.abstention_rate,
            "coverage_defect": self.conformal_coverage[1],
            "coverage_ok": self.conformal_coverage[0],
            **self.scores.as_dict(),
        }


def fit(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    estimator: str = "xgboost",
    seed: int = 42,
    alpha: float = 0.1,
    costs: CostMatrix | None = None,
) -> FittedModel:
    """Fit, then calibrate and conformalise on a held-out slice of the training set."""
    fit_index, calibration_index = _split_calibration(len(train_labels), seed)
    model = _fit_calibrated(train_features[fit_index], train_labels[fit_index], estimator, seed)
    calibration_scores = positive_class_scores(model, train_features[calibration_index])
    calibration_labels = train_labels[calibration_index]
    conformal = MondrianConformal(alpha=alpha).fit(calibration_labels, calibration_scores)
    threshold = choose(calibration_labels, calibration_scores, costs or CostMatrix())
    return FittedModel(model, conformal, threshold)


def run_cell(
    modality: Modality,
    regime: Regime,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    estimator: str = "xgboost",
    seed: int = 42,
) -> AblationResult:
    model = fit(train_features, train_labels, estimator, seed)
    test_scores = model.score(test_features)
    sets = model.conformal.predict_sets(test_scores)
    return AblationResult(
        modality=modality,
        regime=regime,
        estimator=estimator,
        scores=evaluate(test_labels, test_scores, model.threshold),
        threshold=model.threshold,
        abstention_rate=sets.abstention_rate(),
        conformal_coverage=coverage_by_class(test_labels, sets),
    )


def _fit_calibrated(features: np.ndarray, labels: np.ndarray, estimator: str, seed: int):
    """Isotonic calibration first; cost-optimal thresholds need honest probabilities."""
    base = build(estimator, seed)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(features, labels)
    return calibrated


def _split_calibration(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(n * (1.0 - CALIBRATION_FRACTION))
    return order[:cut], order[cut:]
