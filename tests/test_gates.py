"""Gate thresholds from docs/04-execution-plan.md, enforced as tests."""

import pandas as pd
import pytest

from defectlab.data.splits import assert_disjoint, grouped_holdout
from defectlab.twin import FEATURES, TwinConfig, run_line, score

xgb = pytest.importorskip("xgboost")
metrics = pytest.importorskip("sklearn.metrics")

MAX_ABSOLUTE_CORRELATION = 0.35
AUC_BAND = (0.80, 0.88)


@pytest.fixture(scope="module")
def labelled() -> pd.DataFrame:
    config = TwinConfig(seed=7)
    return score(run_line(12000, config), config, target_prevalence=0.567)


def _fit_process_only(frame: pd.DataFrame):
    train, test = grouped_holdout(frame, test_size=0.3, seed=1)
    assert_disjoint(frame, train, test)
    features = frame[list(FEATURES)].to_numpy()
    labels = frame["label"].to_numpy()
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        n_jobs=8,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(features[train], labels[train])
    scores = model.predict_proba(features[test])[:, 1]
    return metrics.roc_auc_score(labels[test], scores)


def test_process_only_auc_is_realistic(labelled):
    """Above the band means the simulator is too easy to be credible."""
    auc = _fit_process_only(labelled)
    assert AUC_BAND[0] <= auc <= AUC_BAND[1], f"process-only AUC {auc:.3f} outside {AUC_BAND}"


def test_no_feature_leaks_the_label(labelled):
    correlations = labelled[list(FEATURES)].corrwith(labelled["label"]).abs()
    assert correlations.max() < MAX_ABSOLUTE_CORRELATION, correlations.idxmax()


def test_controllable_levers_carry_signal(labelled):
    """A prescriptive system needs its knobs to move risk."""
    from defectlab.twin.propensity import mechanism_indices, weighted_contributions

    spread = weighted_contributions(mechanism_indices(labelled)).std()
    for mechanism in ("shrinkage", "air_entrapment", "cold_shut"):
        assert spread[mechanism] > 0.1, f"{mechanism} is inert (std={spread[mechanism]:.3f})"


def test_grouped_holdout_keeps_lots_intact(labelled):
    train, test = grouped_holdout(labelled, test_size=0.3, seed=1)
    assert_disjoint(labelled, train, test)
    assert len(train) + len(test) == len(labelled)
