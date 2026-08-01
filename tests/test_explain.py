"""Grouped attribution over a small fusion model."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from defectlab.explain import IMAGE_GROUP, PROCESS_GROUPS, assign, explain, group_of
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
