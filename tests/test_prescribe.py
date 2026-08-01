"""The action space, the greedy search, and whether the advice survives a wrong simulator."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from defectlab.prescribe import (
    Action,
    Recommendation,
    apply,
    levers,
    perturbed_weights,
    reachable,
    recommend,
    scale_sweep,
    stability,
)
from defectlab.prescribe import fit as fit_surrogate
from defectlab.twin import (
    CONTROLLABLE,
    LOT_LEVEL,
    MECHANISM_WEIGHTS,
    SETPOINTS,
    TwinConfig,
    run_line,
    spec,
)

SURROGATE_SHOTS = 6000


@pytest.fixture(scope="module")
def surrogate():
    return fit_surrogate(shots=SURROGATE_SHOTS, seed=1)


@pytest.fixture(scope="module")
def risky():
    """The worst shot in a run, which is the one an operator would actually be asked about."""
    frame = run_line(600, TwinConfig(seed=7))
    return frame


def test_chemistry_is_never_a_lever():
    """An operator cannot change the iron content of a lot already charged into the furnace."""
    assert not set(levers()) & set(LOT_LEVEL)


def test_tool_wear_is_never_a_lever():
    assert "tool_wear_shots" not in levers()


def test_die_temperature_is_controllable_but_not_a_setpoint():
    """It is a thermal state driven by the dynamics, so it cannot be dialled in for one shot."""
    assert "die_temp_c" in CONTROLLABLE
    assert "die_temp_c" not in SETPOINTS
    assert "die_temp_c" not in levers()


def test_every_lever_is_a_setpoint():
    assert set(levers()) <= set(SETPOINTS)


def test_reachable_values_respect_the_ramp_limit():
    """A 50 C jump in pour temperature is not a recommendation, it is a scrapped shift."""
    bounds = spec("pour_temp_c")
    values = reachable("pour_temp_c", bounds.nominal)
    assert values.min() >= bounds.nominal - bounds.ramp_limit
    assert values.max() <= bounds.nominal + bounds.ramp_limit


def test_reachable_values_respect_the_machine_limits():
    bounds = spec("fast_shot_velocity_ms")
    values = reachable("fast_shot_velocity_ms", bounds.upper)
    assert values.max() <= bounds.upper
    assert values.min() >= bounds.lower


def test_reachable_collapses_when_the_current_value_is_pinned():
    """At a limit with no room to move, the only feasible value is where the shot already is."""
    bounds = spec("hold_time_s")
    assert len(reachable("hold_time_s", bounds.lower, grid=5)) >= 1


def test_apply_writes_only_the_named_setpoints():
    reading = {"pour_temp_c": 690.0, "hold_time_s": 6.0}
    moved = apply(reading, (Action("hold_time_s", 6.0, 7.0),))
    assert moved == {"pour_temp_c": 690.0, "hold_time_s": 7.0}


def test_apply_with_no_actions_is_the_identity():
    reading = {"pour_temp_c": 690.0, "hold_time_s": 6.0}
    assert apply(reading, ()) == reading


def test_the_surrogate_reproduces_the_twin_ordering(surrogate):
    """It is a stand-in for the simulator; if it does not rank shots the same way it is useless."""
    from defectlab.twin.propensity import evaluate

    design = run_line(400, TwinConfig(seed=11))
    truth = evaluate(design, np.random.default_rng(0), noise_sd=0.0).logit
    rank = pd.Series(surrogate.logit(design)).corr(pd.Series(truth), method="spearman")
    assert rank > 0.95


def test_a_recommendation_lowers_the_surrogate_risk(surrogate, risky):
    reading = risky.iloc[int(surrogate.risk(risky).argmax())].to_dict()
    advice = recommend(surrogate, reading)
    assert advice.improvement > 0.0
    assert advice.risk_after < advice.risk_before


def test_a_recommendation_respects_the_sparsity_cap(surrogate, risky):
    """Six simultaneous changes cannot be executed, attributed, or rolled back."""
    reading = risky.iloc[int(surrogate.risk(risky).argmax())].to_dict()
    assert len(recommend(surrogate, reading, max_actions=2).actions) <= 2


def test_a_recommendation_never_names_the_same_lever_twice(surrogate, risky):
    reading = risky.iloc[int(surrogate.risk(risky).argmax())].to_dict()
    names = [action.name for action in recommend(surrogate, reading).actions]
    assert len(names) == len(set(names))


def test_every_proposed_move_is_reachable(surrogate, risky):
    """The search must not propose anything the machine cannot do on the next shot."""
    reading = risky.iloc[int(surrogate.risk(risky).argmax())].to_dict()
    for action in recommend(surrogate, reading).actions:
        bounds = spec(action.name)
        assert abs(action.delta) <= bounds.ramp_limit + 1e-9
        assert bounds.lower - 1e-9 <= action.proposed <= bounds.upper + 1e-9


def test_a_safe_shot_gets_no_advice_at_all(surrogate, risky):
    """Nothing to fix must produce nothing to do. A shot at risk 0.0001 needs no setpoint change."""
    reading = risky.iloc[int(surrogate.risk(risky).argmin())].to_dict()
    advice = recommend(surrogate, reading)
    assert advice.actions == ()
    assert "no feasible" in advice.describe()


def test_lowering_the_gate_lets_a_safe_shot_be_optimised(surrogate, risky):
    """The gate is a policy, not a limit of the search; dropping it must re-enable advice."""
    reading = risky.iloc[int(surrogate.risk(risky).argmin())].to_dict()
    assert recommend(surrogate, reading, min_risk=0.0).actions


def test_perturbed_weights_stay_positive():
    """A negative mechanism weight would mean porosity prevents defects."""
    weights = perturbed_weights(np.random.default_rng(0), 0.5)
    assert all(value > 0.0 for value in weights.values())
    assert set(weights) == set(MECHANISM_WEIGHTS)


def test_perturbation_actually_moves_the_weights():
    weights = perturbed_weights(np.random.default_rng(0), 0.35)
    assert any(abs(weights[name] - MECHANISM_WEIGHTS[name]) > 1e-6 for name in weights)


def test_advice_survives_the_simulator_being_wrong(surrogate, risky):
    """The anti-circularity test: scoring advice against its own twin proves only that it copied it."""
    reading = risky.iloc[int(surrogate.risk(risky).argmax())].to_dict()
    advice = recommend(surrogate, reading)
    assert stability(advice, reading, trials=100, scale=0.35).rate > 0.9


def test_an_empty_recommendation_changes_nothing_under_perturbation(risky):
    reading = risky.iloc[0].to_dict()
    empty = Recommendation((), 0.5, 0.5)
    assert stability(empty, reading, trials=20).median_margin_gain == pytest.approx(0.0)


def test_the_scale_sweep_reports_one_row_per_scale(surrogate, risky):
    """Stability is quoted against how wrong the weights may be; one number would hide that."""
    reading = risky.iloc[int(surrogate.risk(risky).argmax())].to_dict()
    advice = recommend(surrogate, reading)
    table = scale_sweep(advice, reading, scales=(0.2, 0.5), trials=50)
    assert list(table["scale"]) == [0.2, 0.5]
    assert (table["stability"] <= 1.0).all()
