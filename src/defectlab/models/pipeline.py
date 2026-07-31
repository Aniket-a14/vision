"""End-to-end train, calibrate, conformalise and score for one ablation cell."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from ..imaging import Regime
from .conformal import MondrianConformal, PredictionSets, coverage_by_class
from .estimators import build, positive_class_scores
from .evaluation import Scores, evaluate
from .features import Modality
from .thresholds import CostMatrix, choose

CALIBRATION_FRACTION = 0.25

# ISA-18.2 caps an operator at 6-12 alarms/hour, but that budget only makes sense once
# probabilities are prior-corrected to a realistic 2-4% base rate. The research datasets
# run at ~57% prevalence, where any alarm budget starves recall, so it stays off here and
# is applied in the economics layer instead.
DEFAULT_MAX_ALERT_RATE: float | None = None


@dataclass(frozen=True, slots=True)
class CellData:
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray


@dataclass(frozen=True, slots=True)
class FitConfig:
    estimator: str = "xgboost"
    seed: int = 42
    alpha: float = 0.1
    calibration_folds: int = 3
    max_alert_rate: float | None = DEFAULT_MAX_ALERT_RATE
    costs: CostMatrix = field(default_factory=CostMatrix)


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
            **asdict(self.scores),
        }


def fit(features: np.ndarray, labels: np.ndarray, config: FitConfig | None = None) -> FittedModel:
    """Fit, then calibrate and conformalise on a held-out slice of the training set."""
    settings = config or FitConfig()
    fit_index, calibration_index = _split_calibration(len(labels), settings.seed)
    model = _fit_calibrated(features[fit_index], labels[fit_index], settings)
    calibration_scores = positive_class_scores(model, features[calibration_index])
    calibration_labels = labels[calibration_index]
    conformal = MondrianConformal(alpha=settings.alpha).fit(calibration_labels, calibration_scores)
    threshold = choose(
        calibration_labels,
        calibration_scores,
        settings.costs,
        max_alert_rate=settings.max_alert_rate,
    )
    return FittedModel(model, conformal, threshold)


def run_cell(
    modality: Modality, regime: Regime, data: CellData, config: FitConfig | None = None
) -> AblationResult:
    settings = config or FitConfig()
    model = fit(data.train_features, data.train_labels, settings)
    test_scores = model.score(data.test_features)
    sets = model.conformal.predict_sets(test_scores)
    return AblationResult(
        modality=modality,
        regime=regime,
        estimator=settings.estimator,
        scores=evaluate(data.test_labels, test_scores, model.threshold),
        threshold=model.threshold,
        abstention_rate=sets.abstention_rate(),
        conformal_coverage=coverage_by_class(data.test_labels, sets),
    )


def _fit_calibrated(features: np.ndarray, labels: np.ndarray, settings: FitConfig):
    """Isotonic calibration first; cost-optimal thresholds need honest probabilities."""
    base = build(settings.estimator, settings.seed)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=settings.calibration_folds)
    calibrated.fit(features, labels)
    return calibrated


def _split_calibration(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(n * (1.0 - CALIBRATION_FRACTION))
    return order[:cut], order[cut:]
