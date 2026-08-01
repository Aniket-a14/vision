"""Prior correction, PAF costing and the sensitivity band."""

import numpy as np
import pandas as pd
import pytest

from defectlab.economics import (
    ESCAPE_MULTIPLIER_RANGE,
    CostModel,
    cost_curve,
    for_parameter,
    inspect_everything,
    ledger,
    loss_table,
    multiplier_sweep,
    operate,
    optimal_threshold,
    outcome,
    prevalence,
    prevalence_sweep,
    shift,
    ship_everything,
    total_loss,
)
from defectlab.twin import FEATURES, TwinConfig, run_line, spec

COSTS = CostModel(scrap=12.0, inspection=3.0, escape_multiplier=25.0, prevention_per_shot=0.05)
SHOTS = 1000
LINE_PREVALENCE = 0.03


@pytest.fixture(scope="module")
def graded():
    """A separable-but-imperfect score, which is what a real gate has to price."""
    rng = np.random.default_rng(0)
    labels = rng.binomial(1, 0.5, 4000)
    scores = np.clip(rng.normal(0.35 + 0.3 * labels, 0.15), 0.001, 0.999)
    return labels, scores


def test_shift_moves_a_balanced_score_down_to_the_line_rate(graded):
    labels, scores = graded
    corrected = shift(scores, prevalence(labels), LINE_PREVALENCE)
    assert corrected.mean() < scores.mean()
    assert corrected.max() < 1.0


def test_shift_is_identity_when_the_prior_does_not_move(graded):
    _, scores = graded
    assert shift(scores, 0.5, 0.5) == pytest.approx(scores, abs=1e-6)


def test_shift_preserves_the_ranking(graded):
    """Correction rescales the odds monotonically; it must not reorder any two parts."""
    labels, scores = graded
    corrected = shift(scores, prevalence(labels), LINE_PREVALENCE)
    assert np.array_equal(np.argsort(scores), np.argsort(corrected))


def test_shift_round_trips(graded):
    _, scores = graded
    there = shift(scores, 0.5, LINE_PREVALENCE)
    assert shift(there, LINE_PREVALENCE, 0.5) == pytest.approx(scores, abs=1e-6)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_an_impossible_prior_is_rejected(bad, graded):
    _, scores = graded
    with pytest.raises(ValueError, match="strictly in"):
        shift(scores, bad, LINE_PREVALENCE)


def test_error_rates_are_conditional_on_the_true_class():
    """The test set runs near 50% defective; the rates must not carry that mixture."""
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    measured = outcome(labels, scores, 0.5)
    assert measured.escape_rate == pytest.approx(0.5)
    assert measured.overkill_rate == pytest.approx(0.25)


def test_a_class_with_no_members_has_no_error_rate():
    measured = outcome(np.ones(4), np.full(4, 0.9), 0.5)
    assert measured.overkill_rate == 0.0


def test_the_ledger_splits_into_the_four_paf_buckets(graded):
    labels, scores = graded
    entry = ledger(labels, scores, 0.5, LINE_PREVALENCE, COSTS, SHOTS)
    parts = entry.prevention + entry.appraisal + entry.internal_failure + entry.external_failure
    assert entry.total == pytest.approx(parts)
    assert entry.copq == pytest.approx(entry.total - entry.prevention)


def test_a_caught_defect_is_still_scrapped(graded):
    """Detection saves the escape, not the part; internal failure never goes to zero."""
    labels, scores = graded
    entry = ledger(labels, scores, 0.005, LINE_PREVALENCE, COSTS, SHOTS)
    assert entry.escapes == pytest.approx(0.0)
    assert entry.internal_failure == pytest.approx(SHOTS * LINE_PREVALENCE * COSTS.scrap)


def test_shipping_everything_pays_only_external_failure():
    entry = ship_everything(LINE_PREVALENCE, COSTS, SHOTS)
    assert entry.appraisal == 0.0
    assert entry.total == pytest.approx(SHOTS * LINE_PREVALENCE * COSTS.escape)


def test_inspecting_everything_pays_appraisal_on_every_shot():
    entry = inspect_everything(LINE_PREVALENCE, COSTS, SHOTS)
    assert entry.external_failure == 0.0
    assert entry.appraisal == pytest.approx(SHOTS * COSTS.inspection)


def test_a_baseline_carries_no_prevention_cost():
    """Neither baseline runs a model, so charging them for one would flatter the gate."""
    assert ship_everything(LINE_PREVALENCE, COSTS, SHOTS).prevention == 0.0


def test_the_cost_curve_covers_the_threshold_grid(graded):
    labels, scores = graded
    curve = cost_curve(labels, scores, LINE_PREVALENCE, COSTS, SHOTS)
    assert curve["threshold"].is_monotonic_increasing
    assert (curve["alert_rate"].between(0.0, 1.0)).all()


def test_the_optimum_is_the_cheapest_point_on_the_curve(graded):
    labels, scores = graded
    curve = cost_curve(labels, scores, LINE_PREVALENCE, COSTS, SHOTS)
    best = optimal_threshold(labels, scores, LINE_PREVALENCE, COSTS, SHOTS)
    cheapest = ledger(labels, scores, best, LINE_PREVALENCE, COSTS, SHOTS).per_shot
    assert cheapest == pytest.approx(curve["per_shot"].min())


def test_a_costlier_escape_lowers_the_threshold(graded):
    """The whole argument for a gate is this asymmetry; if it does not bite, nothing else holds."""
    labels, scores = graded
    cautious = optimal_threshold(
        labels, scores, LINE_PREVALENCE, COSTS.with_multiplier(50.0), SHOTS
    )
    relaxed = optimal_threshold(labels, scores, LINE_PREVALENCE, COSTS.with_multiplier(2.0), SHOTS)
    assert cautious < relaxed


def test_a_rarer_defect_raises_the_threshold(graded):
    """At 1% defective most alarms are false, so the gate has to become more selective."""
    labels, scores = graded
    rare = optimal_threshold(labels, scores, 0.01, COSTS, SHOTS)
    common = optimal_threshold(labels, scores, 0.20, COSTS, SHOTS)
    assert rare > common


def test_the_gate_beats_both_do_nothing_baselines(graded):
    labels, scores = graded
    point = operate(labels, scores, LINE_PREVALENCE, COSTS, SHOTS)
    assert point.savings_vs_ship > 0.0
    assert point.savings_vs_inspect > 0.0


def test_the_operating_frame_names_all_three_policies(graded):
    labels, scores = graded
    frame = operate(labels, scores, LINE_PREVALENCE, COSTS, SHOTS).frame()
    assert list(frame.index) == ["gate", "ship_all", "inspect_all"]


def test_the_multiplier_sweep_spans_the_stated_range(graded):
    labels, scores = graded
    band = multiplier_sweep(labels, scores, LINE_PREVALENCE, COSTS, shots=SHOTS)
    assert (band["escape_multiplier"].min(), band["escape_multiplier"].max()) == pytest.approx(
        ESCAPE_MULTIPLIER_RANGE
    )
    assert band["savings_vs_ship"].is_monotonic_increasing


def test_the_prevalence_sweep_returns_one_row_per_rate(graded):
    labels, scores = graded
    rates = np.array([0.01, 0.03, 0.05])
    band = prevalence_sweep(labels, scores, COSTS, rates, shots=SHOTS)
    assert band["prevalence"].to_numpy() == pytest.approx(rates)


TOLERANCE_SIGMAS = 3.0


def test_taguchi_loss_is_zero_on_target():
    curve = for_parameter("pour_temp_c", COSTS.scrap)
    assert curve.loss(np.array([spec("pour_temp_c").nominal])) == pytest.approx(0.0)


def test_taguchi_loss_reaches_the_anchor_at_the_tolerance():
    """k is fixed by exactly one point: the loss A0 at the tolerance D0."""
    curve = for_parameter("pour_temp_c", COSTS.scrap, TOLERANCE_SIGMAS)
    at_limit = spec("pour_temp_c").nominal + TOLERANCE_SIGMAS * spec("pour_temp_c").spread
    assert curve.loss(np.array([at_limit])) == pytest.approx(COSTS.scrap)


def test_taguchi_loss_is_symmetric_about_the_target():
    curve = for_parameter("die_temp_c", COSTS.scrap)
    offsets = np.array([-20.0, 20.0]) + spec("die_temp_c").nominal
    assert curve.loss(offsets)[0] == pytest.approx(curve.loss(offsets)[1])


def test_taguchi_penalises_a_part_that_passes_conformance():
    """Being inside the machine limits earns no credit; distance from target still costs."""
    bounds = spec("pour_temp_c")
    inside = np.array([bounds.nominal + 0.5 * (bounds.upper - bounds.nominal)])
    assert for_parameter("pour_temp_c", COSTS.scrap).loss(inside)[0] > 0.0


def test_loss_table_ranks_parameters_and_shares_sum_to_one():
    frame = run_line(600, TwinConfig(seed=5))
    table = loss_table(frame, COSTS)
    assert table["mean_loss"].is_monotonic_decreasing
    assert table["share"].sum() == pytest.approx(1.0)


def test_an_undisturbed_parameter_sits_at_a_baseline_ratio_of_one():
    """A draw at the declared spread costs A0 / sigmas^2 whatever A0 is; the ratio divides it out."""
    rng = np.random.default_rng(0)
    bounds = spec("hold_time_s")
    frame = pd.DataFrame({"hold_time_s": rng.normal(bounds.nominal, bounds.spread, 20000)})
    assert loss_table(frame, COSTS)["baseline_ratio"].iloc[0] == pytest.approx(1.0, abs=0.05)


def test_the_baseline_ratio_does_not_depend_on_the_anchor():
    """The absolute loss is only as good as A0; the ratio is not hostage to it."""
    frame = run_line(600, TwinConfig(seed=5))
    cheap = loss_table(frame, COSTS.with_multiplier(1.0))
    dear = loss_table(frame, CostModel(scrap=500.0))
    assert cheap["baseline_ratio"].to_numpy() == pytest.approx(dear["baseline_ratio"].to_numpy())


def test_loss_table_excludes_tool_wear():
    """Tool wear has no nominal to sit on; a quadratic around 25k shots would be meaningless."""
    frame = run_line(200, TwinConfig(seed=5))
    assert "tool_wear_shots" not in set(loss_table(frame, COSTS)["parameter"])


def test_total_loss_is_one_number_per_shot():
    frame = run_line(300, TwinConfig(seed=5))
    losses = total_loss(frame, COSTS)
    assert losses.shape == (len(frame),)
    assert (losses >= 0.0).all()


def test_total_loss_sums_the_per_parameter_curves():
    frame = run_line(120, TwinConfig(seed=5))
    scored = [name for name in FEATURES if name != "tool_wear_shots"]
    by_hand = sum(for_parameter(name, COSTS.scrap).loss(frame[name]) for name in scored)
    assert total_loss(frame, COSTS) == pytest.approx(by_hand)


def test_a_missing_column_is_simply_not_scored():
    frame = pd.DataFrame({"pour_temp_c": [690.0, 700.0]})
    assert loss_table(frame, COSTS)["parameter"].tolist() == ["pour_temp_c"]
