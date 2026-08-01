"""End-to-end ablation machinery, driven by synthetic embeddings.

Embeddings are simulated as a noisy readout of the true defect probability, so image
quality can be varied without needing torch or the image dataset.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from defectlab.imaging import Regime
from defectlab.models.ablation import (
    MODALITIES,
    AblationInputs,
    RegimeData,
    component_ablation,
    degradation_sweep,
    fusion_gain,
    run,
    summarise,
)
from defectlab.models.pipeline import FitConfig
from defectlab.twin import TwinConfig, run_line, score

EMBEDDING_DIM = 48

# One fixed encoder direction shared by train and test, as a real backbone would be.
DIRECTION = np.random.default_rng(0).normal(size=(1, EMBEDDING_DIM))


def _embeddings(frame: pd.DataFrame, noise: float, seed: int) -> np.ndarray:
    """A readable defect signal buried in noise; higher noise means a worse camera."""
    rng = np.random.default_rng(seed)
    signal = frame["true_defect_prob"].to_numpy()[:, None]
    return signal * DIRECTION + rng.normal(0, noise, (len(frame), EMBEDDING_DIM))


@pytest.fixture(scope="module")
def inputs() -> AblationInputs:
    config = TwinConfig(seed=13)
    labelled = score(run_line(2400, config), config, target_prevalence=0.5)
    train, test = labelled.iloc[:1600], labelled.iloc[1600:]
    regimes = [
        RegimeData(Regime.LAB, _embeddings(train, 0.05, 1), _embeddings(test, 0.05, 2)),
        RegimeData(Regime.INLINE, _embeddings(train, 1.20, 3), _embeddings(test, 1.20, 4)),
    ]
    config = FitConfig(estimator="xgboost_fast", calibration_folds=2)
    return AblationInputs(train, test, regimes, n_components=16, fit_config=config)


def test_ablation_covers_every_modality_and_regime(inputs):
    results = run(inputs)
    assert len(results) == len(MODALITIES) * 2
    assert set(results["modality"]) == {m.value for m in MODALITIES}
    assert set(results["regime"]) == {"lab", "inline"}


def test_results_carry_the_required_columns(inputs):
    results = run(inputs)
    for column in ("roc_auc", "escape_rate", "overkill_rate", "threshold", "coverage_defect"):
        assert column in results.columns


def test_vision_degrades_between_regimes(inputs):
    """The whole argument depends on the inline regime actually being harder."""
    results = run(inputs).set_index(["modality", "regime"])
    lab = results.loc[("vision", "lab"), "roc_auc"]
    inline = results.loc[("vision", "inline"), "roc_auc"]
    assert inline < lab


def test_process_only_is_unaffected_by_imaging_regime(inputs):
    """Process signal is independent of camera quality; that is the fusion thesis."""
    results = run(inputs).set_index(["modality", "regime"])
    lab = results.loc[("process", "lab"), "roc_auc"]
    inline = results.loc[("process", "inline"), "roc_auc"]
    assert lab == pytest.approx(inline, abs=1e-9)


def test_fusion_beats_vision_under_inline_imaging(inputs):
    results = run(inputs).set_index(["modality", "regime"])
    fusion = results.loc[("fusion", "inline"), "roc_auc"]
    vision = results.loc[("vision", "inline"), "roc_auc"]
    assert fusion > vision


def test_conformal_coverage_holds_in_every_cell(inputs):
    results = run(inputs)
    assert (results["coverage_defect"] > 0.80).all()


def test_degradation_sweep_returns_a_row_per_severity_and_modality(inputs):
    severities = (0.1, 1.0)
    sweep = degradation_sweep(
        inputs,
        severities,
        lambda s: (
            _embeddings(inputs.train_frame, s, 21),
            _embeddings(inputs.test_frame, s, 22),
        ),
    )
    assert len(sweep) == len(severities) * len(MODALITIES)
    assert set(sweep["severity"]) == set(severities)


def test_component_ablation_sweeps_pca_width(inputs):
    table = component_ablation(inputs, inputs.regimes[1], [8, 16])
    assert list(table["n_components"]) == [8, 16]


def _tidy(gains: dict[float, list[float]]) -> pd.DataFrame:
    """A results table with a known fusion-minus-vision gain per seed and severity."""
    rows = []
    for severity, per_seed in gains.items():
        for seed, gain in enumerate(per_seed):
            common = {"seed": seed, "severity": severity, "regime": "inline"}
            rows.append({**common, "modality": "vision", "roc_auc": 0.90})
            rows.append({**common, "modality": "fusion", "roc_auc": 0.90 + gain})
    return pd.DataFrame(rows)


def test_fusion_gain_recovers_the_planted_delta():
    gain = fusion_gain(_tidy({1.0: [0.01, 0.02, 0.03]})).iloc[0]
    assert gain["mean"] == pytest.approx(0.02)
    assert gain["wins"] == 3
    assert gain["n"] == 3


def test_fusion_gain_pairs_within_seed_not_across_severities():
    """Pooling severities would halve the reported spread and inflate significance."""
    table = fusion_gain(_tidy({0.5: [0.00, 0.01, 0.02], 2.0: [0.04, 0.05, 0.06]}))
    assert list(table["severity"]) == [0.5, 2.0]
    assert table["mean"].tolist() == pytest.approx([0.01, 0.05])
    assert table["std"].tolist() == pytest.approx([0.01, 0.01])


def test_fusion_gain_reports_no_significance_when_seeds_disagree():
    """Direction alone is not evidence; four of five seeds positive can still be noise."""
    gain = fusion_gain(_tidy({1.0: [0.03, -0.02, 0.01, -0.03, 0.02]})).iloc[0]
    assert gain["p"] > 0.05


def test_summarise_keeps_severity_as_a_grouping_key():
    table = summarise(_tidy({0.5: [0.01], 2.0: [0.05]}))
    assert set(table.columns) >= {"modality", "regime", "severity", "mean", "std"}
    assert len(table) == 2 * 2
