"""Grouped SHAP attribution for a fitted defect model.

SHAP values are additive, so a group's contribution is the sum of its columns' values and
needs no separate estimator. Attribution is taken on the base learners' log-odds margin:
`CalibratedClassifierCV` wraps one learner per fold, and isotonic calibration is monotone,
so it rescales the margin without reordering it.

Values are path-dependent rather than interventional. shap 0.52 refuses the interventional
form on xgboost 3.3, reporting a categorical split that these all-numeric models do not
have. Path-dependent TreeSHAP splits credit along the trees' own paths, so correlated
process parameters share credit instead of one standing in for the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.pipeline import FittedModel
from .groups import assign

PERTURBATION = "tree_path_dependent"


@dataclass(frozen=True, slots=True)
class GroupedAttribution:
    """Per-row contributions in log-odds, one column per group."""

    groups: tuple[str, ...]
    values: np.ndarray
    base_value: float

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, columns=list(self.groups))

    def importance(self) -> pd.Series:
        """Global ranking: mean absolute contribution, largest first."""
        mean = np.abs(self.values).mean(axis=0)
        return pd.Series(mean, index=list(self.groups)).sort_values(ascending=False)

    def explain_row(self, row: int) -> pd.Series:
        """Why this part was flagged, signed, largest magnitude first."""
        series = pd.Series(self.values[row], index=list(self.groups))
        return series.reindex(series.abs().sort_values(ascending=False).index)


def explain(model, features: np.ndarray, names: list[str]) -> GroupedAttribution:
    """Grouped SHAP over a fitted pipeline; per-column values are summed within each group."""
    columns = _column_values(model, features)
    indices = assign(names)
    grouped = np.column_stack([columns.values[:, index].sum(axis=1) for index in indices.values()])
    return GroupedAttribution(tuple(indices), grouped, columns.base_value)


@dataclass(frozen=True, slots=True)
class _ColumnValues:
    values: np.ndarray
    base_value: float


def _column_values(model, features: np.ndarray) -> _ColumnValues:
    """Average across calibration folds; each fold holds a separately fitted learner."""
    import shap

    explainers = [
        shap.TreeExplainer(learner, feature_perturbation=PERTURBATION)
        for learner in _base_learners(model)
    ]
    values = np.mean([explainer.shap_values(features) for explainer in explainers], axis=0)
    base = float(np.mean([np.mean(explainer.expected_value) for explainer in explainers]))
    return _ColumnValues(np.asarray(values, dtype=np.float64), base)


def _base_learners(model) -> list:
    """Unwrap the fitted pipeline down to the trees SHAP can read."""
    if isinstance(model, FittedModel):
        return _base_learners(model.estimator)
    inner = getattr(model, "calibrated_classifiers_", None)
    if inner is None:
        return [model]
    return [fold.estimator for fold in inner]
