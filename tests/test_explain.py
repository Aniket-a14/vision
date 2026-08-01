"""Grouped attribution over a small fusion model."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from defectlab.explain import (
    IMAGE_GROUP,
    PROCESS_GROUPS,
    ale,
    ale_table,
    anchor,
    assign,
    explain,
    group_of,
)
from defectlab.models.features import Modality, build_blocks
from defectlab.models.pipeline import FitConfig, fit
from defectlab.twin import FEATURES, TwinConfig, run_line, score

N_COMPONENTS = 6
EMBEDDING_DIM = 24


def _embeddings(frame: pd.DataFrame, rng: np.random.Generator, direction: np.ndarray):
    """A readable defect signal buried in noise, standing in for a real backbone."""
    signal = frame["true_defect_prob"].to_numpy()[:, None]
    return signal * direction + rng.normal(0, 0.5, (len(frame), EMBEDDING_DIM))


@pytest.fixture(scope="module")
def fitted():
    """A real fusion model: 11 process columns plus reduced image components."""
    config = TwinConfig(seed=3)
    labelled = score(run_line(1600, config), config, target_prevalence=0.5)
    train, test = labelled.iloc[:1200], labelled.iloc[1200:]
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(1, EMBEDDING_DIM))
    train_embeddings = _embeddings(train, rng, direction)
    test_embeddings = _embeddings(test, rng, direction)
    blocks = build_blocks(
        Modality.FUSION, train, test, train_embeddings, test_embeddings, N_COMPONENTS
    )
    model = fit(blocks.train, train["label"].to_numpy(), FitConfig(estimator="xgboost_fast"))
    return model, blocks


def test_every_process_feature_belongs_to_a_group():
    """An ungrouped feature would silently vanish from every explanation."""
    assert {name for members in PROCESS_GROUPS.values() for name in members} == set(FEATURES)


def test_image_components_collapse_into_one_group():
    assert group_of("img_pc07") == IMAGE_GROUP


def test_an_unknown_column_is_rejected():
    with pytest.raises(KeyError):
        group_of("mystery_sensor")


def test_assign_covers_every_column_exactly_once():
    names = [*FEATURES, "img_pc00", "img_pc01"]
    indices = assign(names)
    flat = sorted(position for group in indices.values() for position in group)
    assert flat == list(range(len(names)))


def test_attribution_has_one_column_per_group(fitted):
    model, blocks = fitted
    attribution = explain(model, blocks.test, blocks.names)
    assert attribution.values.shape == (len(blocks.test), len(attribution.groups))
    assert IMAGE_GROUP in attribution.groups


def test_attribution_reconstructs_the_model_ranking(fitted):
    """Attribution is on the margin; isotonic calibration must not reorder it."""
    model, blocks = fitted
    attribution = explain(model, blocks.test, blocks.names)
    margin = attribution.values.sum(axis=1) + attribution.base_value
    rank = pd.Series(model.score(blocks.test)).corr(pd.Series(margin), method="spearman")
    assert rank > 0.95


def test_importance_is_ranked_and_non_negative(fitted):
    model, blocks = fitted
    importance = explain(model, blocks.test, blocks.names).importance()
    assert (importance >= 0).all()
    assert list(importance) == sorted(importance, reverse=True)


def test_row_explanation_is_ordered_by_magnitude(fitted):
    model, blocks = fitted
    row = explain(model, blocks.test, blocks.names).explain_row(0)
    assert list(row.abs()) == sorted(row.abs(), reverse=True)


ALE_BINS = 10


@pytest.fixture
def grid() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(4000, 3))


def test_ale_recovers_a_known_linear_slope(grid):
    """The accumulation is defined at bin edges but reported at centres; a slope catches that."""
    curve = ale(lambda f: 2.0 * f[:, 0], grid, 0, "x0", bins=ALE_BINS)
    assert np.polyfit(curve.centres, curve.effect, 1)[0] == pytest.approx(2.0, abs=1e-6)


def test_ale_is_flat_for_an_unused_feature(grid):
    curve = ale(lambda f: 3.0 * f[:, 1], grid, 0, "x0", bins=ALE_BINS)
    assert np.abs(curve.effect).max() == pytest.approx(0.0, abs=1e-9)


def test_ale_survives_correlated_features():
    """Partial dependence would be biased here; ALE stays on the observed manifold."""
    rng = np.random.default_rng(1)
    shared = rng.normal(size=3000)
    features = np.column_stack([shared + 0.1 * rng.normal(size=3000), shared])
    curve = ale(lambda f: 2.0 * f[:, 0], features, 0, "x0", bins=ALE_BINS)
    assert np.polyfit(curve.centres, curve.effect, 1)[0] == pytest.approx(2.0, abs=1e-6)


def test_ale_is_centred_on_the_data(grid):
    curve = ale(lambda f: 2.0 * f[:, 0], grid, 0, "x0", bins=ALE_BINS)
    weighted = float(np.sum(curve.counts * curve.effect) / curve.counts.sum())
    assert weighted == pytest.approx(0.0, abs=1e-9)


def test_a_constant_feature_yields_an_empty_curve():
    """Quantile edges collapse to one value; that must not raise."""
    features = np.column_stack([np.ones(100), np.arange(100.0)])
    curve = ale(lambda f: f[:, 1], features, 0, "constant", bins=ALE_BINS)
    assert curve.span() == 0.0
    assert len(curve.effect) == 0


def test_ale_table_covers_every_named_column(grid):
    table = ale_table(lambda f: f[:, 0] + f[:, 1], grid, ["a", "b", "c"], bins=ALE_BINS)
    assert set(table["feature"]) == {"a", "b", "c"}
    assert table["shots"].sum() == len(grid) * 3


NAMES = [f"x{index}" for index in range(3)]


def _rule_model(features: np.ndarray) -> np.ndarray:
    """Ground truth: a defect needs both conditions, so a correct anchor names both."""
    return ((features[:, 0] > 1.0) & (features[:, 1] > 1.0)).astype(int)


def test_anchor_recovers_both_conditions_of_a_known_rule(grid):
    found = anchor(_rule_model, np.array([2.0, 2.0, 0.0]), grid, NAMES, seed=1)
    assert {predicate.name for predicate in found.predicates} == {"x0", "x1"}
    assert found.prediction == 1
    assert found.precision > 0.95


def test_anchor_contains_the_part_it_explains(grid):
    instance = np.array([2.0, 2.0, 0.0])
    found = anchor(_rule_model, instance, grid, NAMES, seed=1)
    assert found.holds(instance.reshape(1, -1))[0]


def test_anchor_coverage_is_the_share_of_shots_the_rule_admits(grid):
    found = anchor(_rule_model, np.array([2.0, 2.0, 0.0]), grid, NAMES, seed=1)
    assert found.coverage == pytest.approx(found.holds(grid).mean())
    assert 0.0 < found.coverage < 1.0


def test_a_majority_class_part_needs_no_predicates(grid):
    """Precision is measured against resampled data, so the empty rule already scores high."""
    found = anchor(_rule_model, np.array([-2.0, -2.0, 0.0]), grid, NAMES, seed=1)
    assert found.predicates == ()
    assert found.coverage == 1.0
    assert "no anchor" in found.rule()


def test_candidates_restrict_which_columns_may_be_anchored(grid):
    """Anchoring on an image component would produce a rule no operator can act on."""
    found = anchor(_rule_model, np.array([2.0, 2.0, 0.0]), grid, NAMES, candidates=[0], seed=1)
    assert {predicate.name for predicate in found.predicates} <= {"x0"}
