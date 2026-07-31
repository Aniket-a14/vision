"""Gate thresholds from docs/04-execution-plan.md, enforced as tests.

The twin is evaluated at the real dataset shape (6633 train / 715 test) with
independent RNG streams, because AUC depends on sample size and tuning against
an arbitrary size would not transfer.
"""

import pandas as pd
import pytest

from defectlab.twin import FEATURES, TwinConfig, run_line, score

xgb = pytest.importorskip("xgboost")
metrics = pytest.importorskip("sklearn.metrics")

MAX_ABSOLUTE_CORRELATION = 0.35
AUC_BAND = (0.80, 0.88)
TRAIN_ROWS, TEST_ROWS = 6633, 715
TRAIN_PREVALENCE, TEST_PREVALENCE = 0.5666, 0.6336
TEST_SEED_OFFSET = 10_000


def _split(seed: int, rows: int, prevalence: float) -> pd.DataFrame:
    config = TwinConfig(seed=seed)
    return score(run_line(rows, config), config, target_prevalence=prevalence)


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return _split(42, TRAIN_ROWS, TRAIN_PREVALENCE)


@pytest.fixture(scope="module")
def test() -> pd.DataFrame:
    return _split(42 + TEST_SEED_OFFSET, TEST_ROWS, TEST_PREVALENCE)


def _process_only_auc(train: pd.DataFrame, test: pd.DataFrame) -> float:
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
    model.fit(train[list(FEATURES)].to_numpy(), train["label"].to_numpy())
    scores = model.predict_proba(test[list(FEATURES)].to_numpy())[:, 1]
    return float(metrics.roc_auc_score(test["label"].to_numpy(), scores))


def test_process_only_auc_is_realistic(train, test):
    """Above the band means the simulator is too easy to be credible."""
    auc = _process_only_auc(train, test)
    assert AUC_BAND[0] <= auc <= AUC_BAND[1], f"process-only AUC {auc:.3f} outside {AUC_BAND}"


def test_no_feature_leaks_the_label(train):
    correlations = train[list(FEATURES)].corrwith(train["label"]).abs()
    assert correlations.max() < MAX_ABSOLUTE_CORRELATION, correlations.idxmax()


def test_die_temperature_stays_in_the_physical_window(train):
    """A die pinned at the clip ceiling means the thermal model has run away."""
    assert train["die_temp_c"].between(150.0, 300.0).all()
    assert train["die_temp_c"].std() > 5.0


def test_tool_wear_spans_a_realistic_share_of_die_life(train):
    wear = train["tool_wear_shots"]
    assert wear.max() > 40_000
    assert wear.std() > 10_000


def test_both_splits_see_comparable_wear(train, test):
    """Fresh-die-only test data would be a covariate shift, not a holdout."""
    assert abs(train["tool_wear_shots"].mean() - test["tool_wear_shots"].mean()) < 15_000


def test_grouping_columns_have_enough_levels(train, test):
    for frame in (train, test):
        assert frame["lot_id"].nunique() >= 3
        assert frame["die_id"].nunique() >= 3


def test_controllable_levers_carry_signal(train):
    """A prescriptive system needs its knobs to move risk."""
    from defectlab.twin.propensity import mechanism_indices, weighted_contributions

    spread = weighted_contributions(mechanism_indices(train)).std()
    for mechanism in ("shrinkage", "air_entrapment", "tool_wear"):
        assert spread[mechanism] > 0.1, f"{mechanism} is inert (std={spread[mechanism]:.3f})"
