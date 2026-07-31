import numpy as np
import pytest

from defectlab.twin import constants as k
from defectlab.twin import physics


def test_hydrogen_solubility_rises_with_temperature():
    temps = np.array([650.0, 700.0, 750.0])
    assert np.all(np.diff(physics.hydrogen_solubility(temps)) > 0)


def test_solubility_matches_reference_point():
    at_reference = physics.hydrogen_solubility(k.H2_REFERENCE_TEMP_C)
    assert at_reference == pytest.approx(k.H2_SOLUBILITY_LIQUID, rel=1e-6)


def test_solidification_cliff_is_roughly_twentyfold():
    ratio = k.H2_SOLUBILITY_LIQUID / k.H2_SOLUBILITY_SOLID
    assert 18.0 < ratio < 21.0


def test_fe_critical_follows_taylor_relation():
    assert physics.fe_critical(np.array([9.5])) == pytest.approx(0.6625, abs=1e-4)
    assert physics.fe_critical(np.array([11.0])) == pytest.approx(0.775, abs=1e-4)


def test_fe_critical_is_silicon_dependent_not_fixed():
    low, high = physics.fe_critical(np.array([8.5])), physics.fe_critical(np.array([12.0]))
    assert high > low


def test_beta_platelets_only_above_critical_iron():
    si = np.array([10.0, 10.0])
    fe = np.array([0.4, 1.2])
    index = physics.beta_platelet_index(fe, si)
    assert index[0] == 0.0
    assert index[1] > 0.0


def test_iron_is_two_sided_low_iron_solders():
    die_temp = np.array([260.0, 260.0])
    low_iron = physics.soldering_index(np.array([0.35]), die_temp[:1])
    healthy_iron = physics.soldering_index(np.array([0.95]), die_temp[:1])
    assert low_iron > 0.0
    assert healthy_iron == 0.0


def test_sludge_factor_weights_manganese_and_chromium():
    assert physics.sludge_factor(1.0, 0.2, 0.1) == pytest.approx(1.0 + 0.4 + 0.3)


def test_sludge_penalty_activates_past_the_limit():
    assert physics.sludge_index(0.8, 0.2) == 0.0
    assert physics.sludge_index(1.6, 0.4) > 0.0


def test_feeding_efficiency_saturates():
    at_knee = physics.feeding_efficiency(np.array([k.PRESSURE_SATURATION_MPA]))
    far_past = physics.feeding_efficiency(np.array([3 * k.PRESSURE_SATURATION_MPA]))
    assert far_past - at_knee < 0.25
    assert far_past < 1.0


def test_excess_pressure_creates_flash():
    low = physics.flash_index(np.array([50.0]), np.array([30000.0]))
    high = physics.flash_index(np.array([120.0]), np.array([30000.0]))
    assert low == 0.0
    assert high > 0.0


def test_air_entrapment_is_non_monotonic_in_both_velocities():
    slow = np.array([0.15, 0.28, 0.45])
    fast = np.full(3, 2.80)
    index = physics.air_entrapment_index(slow, fast)
    assert index[1] < index[0]
    assert index[1] < index[2]


def test_gate_temperature_is_below_pour_temperature():
    assert physics.gate_temperature(np.array([700.0]))[0] == pytest.approx(
        700.0 - k.RUNNER_TO_GATE_LOSS_C
    )


def test_cold_die_raises_misrun_risk():
    cold = physics.misrun_index(np.array([660.0]), np.array([150.0]), np.array([10.0]))
    warm = physics.misrun_index(np.array([660.0]), np.array([250.0]), np.array([10.0]))
    assert cold > warm


def test_low_silicon_reduces_fluidity():
    poor = physics.misrun_index(np.array([700.0]), np.array([230.0]), np.array([8.0]))
    good = physics.misrun_index(np.array([700.0]), np.array([230.0]), np.array([11.0]))
    assert poor > good
