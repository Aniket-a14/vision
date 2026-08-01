"""Gate thresholds from docs/04-execution-plan.md, enforced as tests.

Every gate here is a mean across twin seeds. A single seed cannot measure these:
alloy chemistry is drawn per lot, and a 715-row test split holds only ~4 lots, so
process-only AUC moves by more than the width of the band from one seed to the next.
"""

import numpy as np
import pandas as pd
import pytest

from defectlab.twin import FEATURES, TwinConfig, run_line, score

xgb = pytest.importorskip("xgboost")
metrics = pytest.importorskip("sklearn.metrics")

MAX_ABSOLUTE_CORRELATION = 0.35
AUC_BAND = (0.80, 0.88)
SEEDS = (42, 7, 99)
TRAIN_ROWS, TEST_ROWS = 6633, 715
TRAIN_PREVALENCE, TEST_PREVALENCE = 0.5666, 0.6336
TEST_SEED_OFFSET = 10_000
OVERSAMPLE = 4


def _split(seed: int, rows: int, prevalence: float) -> pd.DataFrame:
    """Mirror the pairing step, which consumes the early part of a longer run."""
    config = TwinConfig(seed=seed)
    return score(run_line(rows * OVERSAMPLE, config), config, prevalence).head(rows)


@pytest.fixture(scope="module")
def splits() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    return [
        (
            _split(seed, TRAIN_ROWS, TRAIN_PREVALENCE),
            _split(seed + TEST_SEED_OFFSET, TEST_ROWS, TEST_PREVALENCE),
        )
        for seed in SEEDS
    ]


def _process_only_auc(train: pd.DataFrame, test: pd.DataFrame) -> float:
    model = xgb.XGBClassifier(
        n_estimators=300,
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


@pytest.fixture(scope="module")
def process_aucs(splits) -> np.ndarray:
    return np.array([_process_only_auc(train, test) for train, test in splits])


def test_process_only_auc_is_realistic(process_aucs):
    """Above the band means the simulator is too easy to be credible."""
    mean = process_aucs.mean()
    assert AUC_BAND[0] <= mean <= AUC_BAND[1], (
        f"mean process-only AUC {mean:.3f} outside {AUC_BAND}"
    )


def test_single_seed_auc_cannot_be_trusted(process_aucs):
    """Guards the methodology: if this ever tightens, the gates may go back to one seed."""
    assert process_aucs.std(ddof=1) > 0.02, "seed spread collapsed; re-check the lot structure"


def test_no_feature_leaks_the_label(splits):
    correlations = [
        train[list(FEATURES)].corrwith(train["label"]).abs().max() for train, _ in splits
    ]
    assert np.mean(correlations) < MAX_ABSOLUTE_CORRELATION


def test_die_temperature_stays_in_the_physical_window(splits):
    """A die pinned at the clip ceiling means the thermal model has run away."""
    for train, _ in splits:
        assert train["die_temp_c"].between(150.0, 300.0).all()
        assert train["die_temp_c"].std() > 5.0


def test_tool_wear_spans_a_realistic_share_of_die_life(splits):
    for train, _ in splits:
        assert train["tool_wear_shots"].max() > 40_000
        assert train["tool_wear_shots"].std() > 10_000


def test_both_splits_see_comparable_wear(splits):
    """Fresh-die-only test data would be a covariate shift, not a holdout."""
    for train, test in splits:
        gap = abs(train["tool_wear_shots"].mean() - test["tool_wear_shots"].mean())
        assert gap < 15_000


def test_grouping_columns_have_enough_levels(splits):
    for train, test in splits:
        for frame in (train, test):
            assert frame["lot_id"].nunique() >= 3
            assert frame["die_id"].nunique() >= 3


def test_controllable_levers_carry_signal(splits):
    """A prescriptive system needs its knobs to move risk."""
    from defectlab.twin.propensity import mechanism_indices, weighted_contributions

    train = splits[0][0]
    spread = weighted_contributions(mechanism_indices(train)).std()
    for mechanism in ("shrinkage", "air_entrapment", "tool_wear"):
        assert spread[mechanism] > 0.1, f"{mechanism} is inert (std={spread[mechanism]:.3f})"
