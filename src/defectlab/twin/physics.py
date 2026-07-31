"""Physically-grounded defect mechanisms for Al-Si HPDC.

Every function is a pure transform of process state; none may see a label.
"""

from __future__ import annotations

import numpy as np

from . import constants as k

Array = np.ndarray


def hydrogen_solubility(temp_c: Array | float) -> Array:
    """Sieverts-law solubility in the melt, mL H2 per 100 g."""
    temp_k = np.asarray(temp_c, dtype=float) + 273.15
    reference_k = k.H2_REFERENCE_TEMP_C + 273.15
    return k.H2_SOLUBILITY_LIQUID * np.exp(
        -k.H2_DISSOLUTION_SLOPE * (1.0 / temp_k - 1.0 / reference_k)
    )


def dissolved_hydrogen(pour_temp_c: Array, exposure: Array | float = 1.0) -> Array:
    """Melt picks up hydrogen in proportion to its solubility ceiling and holding exposure."""
    return hydrogen_solubility(pour_temp_c) * np.clip(np.asarray(exposure, dtype=float), 0.0, 2.0)


def gas_porosity_index(pour_temp_c: Array, exposure: Array | float = 1.0) -> Array:
    """Supersaturation against the solid solubility limit drives gas porosity."""
    supersaturation = dissolved_hydrogen(pour_temp_c, exposure) / k.H2_SOLUBILITY_SOLID - 1.0
    return np.maximum(supersaturation, 0.0) / 20.0


def fe_critical(si_pct: Array) -> Array:
    """Taylor's relation: the beta-Al5FeSi threshold scales with silicon."""
    return k.FE_CRIT_SI_SLOPE * np.asarray(si_pct, dtype=float) + k.FE_CRIT_INTERCEPT


def beta_platelet_index(fe_pct: Array, si_pct: Array) -> Array:
    """Pre-eutectic beta platelets block interdendritic feeding above Fe_crit."""
    excess = np.asarray(fe_pct, dtype=float) - fe_critical(si_pct)
    return np.maximum(excess, 0.0) / 0.15


def soldering_index(fe_pct: Array, die_temp_c: Array) -> Array:
    """Low iron plus a hot die lets the melt weld itself to H13 steel."""
    iron_deficit = np.maximum(k.FE_SOLDERING_FLOOR_PCT - np.asarray(fe_pct, dtype=float), 0.0) / 0.2
    thermal_drive = np.maximum(np.asarray(die_temp_c, dtype=float) - k.DIE_TEMP_MAX_C, 0.0) / 30.0
    return iron_deficit * (1.0 + thermal_drive)


def sludge_factor(fe_pct: Array, mn_pct: Array, cr_pct: Array | float = 0.0) -> Array:
    """SF = Fe + 2*Mn + 3*Cr; hard primary intermetallics form above the limit."""
    return (
        np.asarray(fe_pct, dtype=float)
        + k.SLUDGE_MN_WEIGHT * np.asarray(mn_pct, dtype=float)
        + k.SLUDGE_CR_WEIGHT * np.asarray(cr_pct, dtype=float)
    )


def sludge_index(fe_pct: Array, mn_pct: Array, cr_pct: Array | float = 0.0) -> Array:
    excess = sludge_factor(fe_pct, mn_pct, cr_pct) - k.SLUDGE_FACTOR_LIMIT
    return np.maximum(excess, 0.0) / 0.4


def feeding_efficiency(pressure_mpa: Array) -> Array:
    """Intensification feeds shrinkage with diminishing returns past the knee."""
    ratio = np.asarray(pressure_mpa, dtype=float) / k.PRESSURE_SATURATION_MPA
    return np.tanh(ratio)


def flash_index(pressure_mpa: Array, tool_wear_shots: Array) -> Array:
    """Excess pressure on a worn die pushes metal into the parting line."""
    overpressure = (
        np.maximum(np.asarray(pressure_mpa, dtype=float) - k.PRESSURE_SATURATION_MPA, 0.0) / 30.0
    )
    wear = np.asarray(tool_wear_shots, dtype=float) / 60000.0
    return overpressure * (0.4 + wear)


def shrinkage_index(
    pressure_mpa: Array, cooling_time_s: Array, hold_time_s: Array, beta: Array
) -> Array:
    """Unfed solidification shrinkage; pressure only helps while it is still held."""
    unfed = 1.0 - feeding_efficiency(pressure_mpa)
    short_hold = np.maximum(5.5 - np.asarray(hold_time_s, dtype=float), 0.0) / 1.5
    quench_deficit = np.maximum(11.0 - np.asarray(cooling_time_s, dtype=float), 0.0) / 3.5
    return unfed * (1.0 + 0.8 * short_hold + 0.6 * quench_deficit) + 0.30 * beta


def gate_temperature(pour_temp_c: Array) -> Array:
    """Melt arrives at the gate well below pour temperature."""
    return np.asarray(pour_temp_c, dtype=float) - k.RUNNER_TO_GATE_LOSS_C


def cold_shut_index(pour_temp_c: Array, die_temp_c: Array, fast_shot_velocity_ms: Array) -> Array:
    """Fronts that meet below the liquidus fail to fuse."""
    cold_spot_c = np.asarray(die_temp_c, dtype=float) - k.DIE_COLD_SPOT_DELTA_C
    front_temp = gate_temperature(pour_temp_c) + 0.25 * (cold_spot_c - k.DIE_TEMP_MIN_C)
    thermal_deficit = np.maximum(k.LIQUIDUS_C - front_temp, 0.0) / 25.0
    slow_fill = np.maximum(2.4 - np.asarray(fast_shot_velocity_ms, dtype=float), 0.0) / 0.6
    return thermal_deficit * (1.0 + slow_fill)


def misrun_index(pour_temp_c: Array, die_temp_c: Array, si_pct: Array) -> Array:
    """Cavity never fills: cold melt, cold die, and poor silicon fluidity."""
    superheat = gate_temperature(pour_temp_c) - k.LIQUIDUS_C
    thermal_deficit = np.maximum(20.0 - superheat, 0.0) / 20.0
    die_deficit = np.maximum(k.DIE_TEMP_MIN_C - np.asarray(die_temp_c, dtype=float), 0.0) / 25.0
    fluidity_deficit = np.maximum(9.0 - np.asarray(si_pct, dtype=float), 0.0) / 1.5
    return thermal_deficit * (1.0 + die_deficit) + 0.4 * fluidity_deficit


def air_entrapment_index(slow_shot_velocity_ms: Array, fast_shot_velocity_ms: Array) -> Array:
    """Both plunger phases have a critical velocity; either side of it entrains air."""
    slow = _quadratic_penalty(slow_shot_velocity_ms, optimum=0.28, scale=0.09)
    fast = _quadratic_penalty(fast_shot_velocity_ms, optimum=2.80, scale=0.55)
    return slow + fast


def _quadratic_penalty(value: Array, optimum: float, scale: float) -> Array:
    return ((np.asarray(value, dtype=float) - optimum) / scale) ** 2
