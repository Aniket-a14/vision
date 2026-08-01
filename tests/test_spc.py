"""Control limits, the eight Nelson rules, and what each of them is supposed to catch."""

import numpy as np
import pytest

from defectlab.spc import (
    RULES,
    apply_ewma,
    apply_i_mr,
    apply_xbar_r,
    evaluate,
    factors,
    fit_ewma,
    fit_i_mr,
    fit_xbar_r,
)

SUBGROUP = 5
PHASE_ONE_GROUPS = 40


@pytest.fixture(scope="module")
def stable():
    """A process in control: the Phase I window every chart here is fitted on."""
    rng = np.random.default_rng(0)
    return rng.normal(100.0, 2.0, (PHASE_ONE_GROUPS, SUBGROUP))


def test_an_unsupported_subgroup_size_is_rejected():
    with pytest.raises(ValueError, match="subgroup size"):
        factors(11)


def test_sigma_comes_from_the_mean_range_not_the_overall_sd(stable):
    """Within-subgroup sigma is the point; the overall sd would absorb any shift."""
    assert fit_xbar_r(stable).process_sigma == pytest.approx(2.0, rel=0.15)


def test_the_plotted_statistic_has_its_own_sigma(stable):
    """Zones are read off the subgroup mean, which is sqrt(n) tighter than a single part."""
    chart = fit_xbar_r(stable)
    assert chart.location.sigma == pytest.approx(chart.process_sigma / np.sqrt(SUBGROUP))


def test_the_derived_limits_agree_with_the_tabulated_a2(stable):
    """A2 = 3 / (d2 sqrt(n)). Deriving and tabulating must give the same limit or one is wrong."""
    chart = fit_xbar_r(stable)
    mean_range = chart.dispersion.centre
    half_width = chart.location.upper - chart.location.centre
    assert half_width == pytest.approx(factors(SUBGROUP).a2 * mean_range, rel=0.01)


def test_a_stable_process_barely_signals(stable):
    """All eight rules on in-control data should stay quiet on the overwhelming majority."""
    chart = fit_xbar_r(stable)
    assert apply_xbar_r(chart, stable)["signal"].mean() < 0.10


def test_limits_are_frozen_so_a_drift_still_signals(stable):
    """The whole reason Phase I is separate: re-fitting would follow the drift and hide it."""
    chart = fit_xbar_r(stable)
    drifted = stable + np.linspace(0.0, 6.0, len(stable))[:, None]
    assert apply_xbar_r(chart, drifted)["signal"].to_numpy()[-10:].all()
    assert not apply_xbar_r(fit_xbar_r(drifted), drifted)["signal"].to_numpy()[-10:].all()


def test_a_subgroup_of_the_wrong_size_is_rejected(stable):
    chart = fit_xbar_r(stable)
    with pytest.raises(ValueError, match="fitted for subgroups"):
        apply_xbar_r(chart, np.zeros((5, 3)))


def test_the_range_chart_catches_a_variance_change_the_mean_chart_misses(stable):
    """Spread can triple with the mean untouched; that is what the paired chart is for."""
    chart = fit_xbar_r(stable)
    spread = (stable - stable.mean(axis=1, keepdims=True)) * 3.0 + stable.mean(
        axis=1, keepdims=True
    )
    assert apply_xbar_r(chart, spread)["dispersion_signal"].mean() > 0.5


def test_individuals_chart_uses_the_moving_range():
    rng = np.random.default_rng(1)
    series = rng.normal(50.0, 1.5, 300)
    chart = fit_i_mr(series)
    assert chart.location.sigma == pytest.approx(1.5, rel=0.15)
    assert apply_i_mr(chart, series)["signal"].mean() < 0.10


def test_individuals_chart_needs_a_pair():
    with pytest.raises(ValueError, match="at least two points"):
        fit_i_mr(np.array([1.0]))


def _first_signal(flags: np.ndarray) -> int:
    """Run length to the first alarm; the length of the run itself if it never fires."""
    hits = np.flatnonzero(flags)
    return int(hits[0]) if len(hits) else len(flags)


def test_ewma_beats_shewhart_to_a_small_shift():
    """A 1-sigma shift is EWMA's reason for existing. Shewhart's rule 1 needs three sigma, so it
    only catches such a shift when noise happens to carry a point over -- late, and by luck."""
    rng = np.random.default_rng(2)
    phase_one = rng.normal(0.0, 1.0, 400)
    ewma_runs, shewhart_runs = [], []
    for _ in range(30):
        shifted = rng.normal(1.0, 1.0, 60)
        ewma_runs.append(_first_signal(apply_ewma(fit_ewma(phase_one), shifted)["signal"]))
        rule_one = apply_i_mr(fit_i_mr(phase_one), shifted)["beyond_3_sigma"]
        shewhart_runs.append(_first_signal(rule_one.to_numpy()))
    assert np.median(ewma_runs) < np.median(shewhart_runs) / 3.0


def test_ewma_limits_widen_then_settle():
    """The band is narrow at start-up because the statistic has not accumulated variance yet."""
    chart = fit_ewma(np.random.default_rng(3).normal(0.0, 1.0, 200))
    band = chart.band(100)
    assert band[0] < band[10] < band[-1]
    assert band[-1] == pytest.approx(band[-2], rel=1e-3)


def _zones(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=float)


def test_rule_1_flags_a_single_excursion():
    flags = evaluate(_zones([0.0, 0.1, 3.5, 0.2]))
    assert flags["beyond_3_sigma"].tolist() == [False, False, True, False]


def test_rule_2_needs_nine_on_one_side():
    assert not evaluate(_zones([0.5] * 8))["nine_one_side"].any()
    assert evaluate(_zones([0.5] * 9))["nine_one_side"].iloc[-1]


def test_rule_2_flags_only_the_last_point_of_the_run():
    """One event must produce one alarm; flagging the whole run is why dashboards get muted."""
    flags = evaluate(_zones([0.5] * 9))["nine_one_side"]
    assert flags.sum() == 1


def test_rule_3_catches_six_rising_points():
    assert evaluate(_zones([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))["six_trending"].iloc[-1]


def test_rule_3_ignores_five_rising_points():
    assert not evaluate(_zones([0.1, 0.2, 0.3, 0.4, 0.5]))["six_trending"].any()


def test_rule_4_catches_overcontrol():
    zigzag = _zones([(-1.0) ** index * 0.5 for index in range(14)])
    assert evaluate(zigzag)["fourteen_alternating"].iloc[-1]


def test_rule_5_needs_two_of_three_on_the_same_side():
    assert evaluate(_zones([2.5, 0.1, 2.4]))["two_of_three_beyond_2"].iloc[-1]
    assert not evaluate(_zones([2.5, 0.1, -2.4]))["two_of_three_beyond_2"].any()


def test_rule_6_needs_four_of_five_on_the_same_side():
    assert evaluate(_zones([1.5, 1.6, 0.1, 1.7, 1.8]))["four_of_five_beyond_1"].iloc[-1]


def test_rule_7_catches_hugging_the_centre_line():
    """Fifteen points too close to centre usually means stratified sampling, not a good process."""
    assert evaluate(np.full(15, 0.2))["fifteen_hugging"].iloc[-1]


def test_rule_8_catches_a_bimodal_process():
    """Eight points outside 1 sigma on alternating sides: two populations, one chart."""
    zones = _zones([1.5, -1.5] * 4)
    assert evaluate(zones)["eight_avoiding"].iloc[-1]
    assert not evaluate(zones)["nine_one_side"].any()


def test_every_rule_has_a_column():
    frame = evaluate(np.zeros(20))
    assert list(frame.columns) == list(RULES)


def test_no_rule_fires_before_its_window_is_full():
    """A short series must not trigger a nine-point rule on four points of data."""
    assert not evaluate(np.full(4, 0.5)).drop(columns=["beyond_3_sigma"]).to_numpy().any()
