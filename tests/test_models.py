import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from defectlab.models.evaluation import evaluate
from defectlab.models.features import Modality, build_blocks, fit_image_reducer
from defectlab.models.thresholds import (
    CostMatrix,
    alert_budget,
    choose,
    cost_optimal,
    neyman_pearson,
)
from defectlab.twin import FEATURES, TwinConfig, run_line, score


@pytest.fixture(scope="module")
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = TwinConfig(seed=11)
    labelled = score(run_line(1500, config), config, target_prevalence=0.5)
    return labelled.iloc[:1000], labelled.iloc[1000:]


@pytest.fixture(scope="module")
def embeddings() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(size=(1000, 128)), rng.normal(size=(500, 128))


def test_process_block_has_one_column_per_parameter(frames):
    blocks = build_blocks(Modality.PROCESS, *frames)
    assert blocks.train.shape[1] == len(FEATURES)
    assert blocks.names == list(FEATURES)


def test_vision_block_width_follows_the_component_count(frames, embeddings):
    blocks = build_blocks(Modality.VISION, *frames, *embeddings, n_components=16)
    assert blocks.train.shape[1] == 16


def test_fusion_block_concatenates_both_modalities(frames, embeddings):
    blocks = build_blocks(Modality.FUSION, *frames, *embeddings, n_components=16)
    assert blocks.train.shape[1] == len(FEATURES) + 16
    assert blocks.names[: len(FEATURES)] == list(FEATURES)


def test_vision_modality_requires_embeddings(frames):
    with pytest.raises(ValueError, match="embeddings are required"):
        build_blocks(Modality.VISION, *frames)


def test_reducer_is_fitted_on_train_only(embeddings):
    train, test = embeddings
    reducer = fit_image_reducer(train, n_components=8)
    assert reducer.transform(test).shape == (len(test), 8)


def test_block_balancing_keeps_modalities_comparable(frames, embeddings):
    blocks = build_blocks(Modality.FUSION, *frames, *embeddings, n_components=16)
    image_energy = blocks.train[:, len(FEATURES) :].var(axis=0).sum()
    assert 0.2 < image_energy / len(FEATURES) < 5.0


def test_escape_and_overkill_are_reported_separately():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    result = evaluate(labels, scores, threshold=0.5)
    assert result.escape_rate == 0.5
    assert result.overkill_rate == 0.5


def test_perfect_scores_give_zero_escape():
    labels = np.array([1, 1, 0, 0])
    result = evaluate(labels, np.array([0.99, 0.98, 0.01, 0.02]), threshold=0.5)
    assert result.escape_rate == 0.0
    assert result.roc_auc == 1.0


def test_expensive_escapes_push_the_threshold_below_half():
    rng = np.random.default_rng(2)
    labels = rng.binomial(1, 0.2, 4000)
    scores = np.clip(np.where(labels == 1, rng.beta(6, 3, 4000), rng.beta(3, 6, 4000)), 0, 1)
    threshold = cost_optimal(labels, scores, CostMatrix(escape=250.0, overkill=4.0))
    assert threshold < 0.5


def test_symmetric_costs_move_the_threshold_up():
    rng = np.random.default_rng(2)
    labels = rng.binomial(1, 0.2, 4000)
    scores = np.clip(np.where(labels == 1, rng.beta(6, 3, 4000), rng.beta(3, 6, 4000)), 0, 1)
    cheap = cost_optimal(labels, scores, CostMatrix(escape=250.0, overkill=4.0))
    even = cost_optimal(labels, scores, CostMatrix(escape=10.0, overkill=10.0))
    assert even > cheap


def test_neyman_pearson_bounds_the_escape_rate():
    rng = np.random.default_rng(8)
    labels = rng.binomial(1, 0.3, 5000)
    scores = np.clip(np.where(labels == 1, rng.beta(5, 3, 5000), rng.beta(2, 6, 5000)), 0, 1)
    threshold = neyman_pearson(scores, labels, max_escape_rate=0.05)
    realised = (scores[labels == 1] < threshold).mean()
    assert realised <= 0.05 + 1e-9


def test_alert_budget_caps_the_alarm_rate():
    rng = np.random.default_rng(9)
    scores = rng.uniform(0, 1, 10000)
    threshold = alert_budget(scores, max_alert_rate=0.05)
    assert (scores >= threshold).mean() == pytest.approx(0.05, abs=0.01)


def test_choose_respects_the_alert_budget():
    rng = np.random.default_rng(10)
    labels = rng.binomial(1, 0.3, 5000)
    scores = np.clip(np.where(labels == 1, rng.beta(5, 3, 5000), rng.beta(2, 6, 5000)), 0, 1)
    threshold = choose(labels, scores, CostMatrix(), max_alert_rate=0.05)
    assert (scores >= threshold).mean() <= 0.08
