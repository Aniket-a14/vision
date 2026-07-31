"""The three tabular learners compared in the ablation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

Builder = Callable[[], Any]


def xgboost_classifier(seed: int = 42) -> Any:
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        n_jobs=8,
        random_state=seed,
        eval_metric="logloss",
    )


def xgboost_fast(seed: int = 42) -> Any:
    """Small budget for tests and sweeps where trend matters more than the last point of AUC."""
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.15,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        n_jobs=8,
        random_state=seed,
        eval_metric="logloss",
    )


def explainable_boosting_classifier(seed: int = 42) -> Any:
    """Glass-box GA2M; its shape functions are the explanation, with no post-hoc step."""
    from interpret.glassbox import ExplainableBoostingClassifier

    return ExplainableBoostingClassifier(interactions=8, random_state=seed)


def logistic_baseline(seed: int = 42) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))


BUILDERS: dict[str, Callable[[int], Any]] = {
    "xgboost": xgboost_classifier,
    "xgboost_fast": xgboost_fast,
    "ebm": explainable_boosting_classifier,
    "logistic": logistic_baseline,
}

DEFAULT_ESTIMATOR = "xgboost"


def build(name: str = DEFAULT_ESTIMATOR, seed: int = 42) -> Any:
    if name not in BUILDERS:
        raise KeyError(f"unknown estimator {name!r}; available: {sorted(BUILDERS)}")
    return BUILDERS[name](seed)


def positive_class_scores(model: Any, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1]
