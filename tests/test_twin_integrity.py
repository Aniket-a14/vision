"""Gate 1 checks. A failure here means the project's central claim is unsound."""

import numpy as np
import pandas as pd
import pytest

from defectlab.twin import FEATURES, TwinConfig, run_line, score

MAX_ABSOLUTE_CORRELATION = 0.35


@pytest.fixture(scope="module")
def labelled() -> pd.DataFrame:
    config = TwinConfig(seed=7)
    return score(run_line(6000, config), config, target_prevalence=0.567)


def test_prevalence_matches_the_calibration_target(labelled):
    assert labelled["label"].mean() == pytest.approx(0.567, abs=0.03)


def test_pour_temperature_shows_a_u_shape(labelled):
    """Defects must sit at both tails, not on one side."""
    defects = labelled.loc[labelled["label"] == 1, "pour_temp_c"]
    healthy = labelled.loc[labelled["label"] == 0, "pour_temp_c"]
    assert defects.std() > healthy.std()


def test_probabilities_are_bounded(labelled):
    assert labelled["true_defect_prob"].between(0.0, 1.0).all()


def test_labels_are_sampled_not_thresholded(labelled):
    """A thresholded label would make probability perfectly separate the classes."""
    overlap = labelled.groupby("label")["true_defect_prob"].agg(["min", "max"])
    assert overlap.loc[1, "min"] < overlap.loc[0, "max"]


def test_run_is_reproducible():
    config = TwinConfig(seed=99)
    pd.testing.assert_frame_equal(run_line(200, config), run_line(200, config))


def test_different_seeds_diverge():
    left = run_line(200, TwinConfig(seed=1))
    right = run_line(200, TwinConfig(seed=2))
    assert not left["pour_temp_c"].equals(right["pour_temp_c"])


def test_parameters_stay_inside_machine_limits(labelled):
    from defectlab.twin.parameters import BY_NAME

    for name in FEATURES:
        spec = BY_NAME[name]
        assert labelled[name].between(spec.lower, spec.upper).all(), name


def test_die_temperature_accumulates_rather_than_resetting():
    frame = run_line(400, TwinConfig(seed=3))
    early = frame["die_temp_c"].iloc[:50].mean()
    later = frame["die_temp_c"].iloc[-50:].mean()
    assert not np.isclose(early, later, atol=1e-6)


def test_tool_wear_is_monotonic():
    frame = run_line(300, TwinConfig(seed=4))
    assert frame["tool_wear_shots"].is_monotonic_increasing


def test_chemistry_is_constant_within_a_lot():
    frame = run_line(1200, TwinConfig(seed=5))
    spread = frame.groupby("lot_id")["si_content_pct"].nunique()
    assert (spread == 1).all()


def test_multiple_lots_are_produced():
    frame = run_line(1200, TwinConfig(seed=5))
    assert frame["lot_id"].nunique() > 1
