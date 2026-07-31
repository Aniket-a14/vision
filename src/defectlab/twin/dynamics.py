"""Stateful line dynamics: thermal inertia, tool wear, alloy lots and sensor drift."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import constants as k
from .parameters import BY_NAME


@dataclass(frozen=True, slots=True)
class LineConfig:
    shots_per_lot: int = 220
    shots_per_shift: int = 700
    die_warmup_gain: float = 0.11
    die_cool_per_second: float = 0.9
    wear_thermal_gain: float = 0.02
    sensor_drift_sd: float = 0.015
    starting_wear_shots: float = 0.0


@dataclass(frozen=True, slots=True)
class AlloyLot:
    lot_id: int
    si_content_pct: float
    fe_content_pct: float
    mn_content_pct: float


@dataclass(slots=True)
class LineState:
    shot_index: int = 0
    die_temp_c: float = k.DIE_TEMP_MIN_C
    tool_wear_shots: float = 0.0
    lot: AlloyLot = field(default_factory=lambda: AlloyLot(0, 10.5, 0.9, 0.28))
    shift_id: int = 0
    sensor_bias: dict[str, float] = field(default_factory=dict)


def initial_state(config: LineConfig, rng: np.random.Generator) -> LineState:
    return LineState(
        die_temp_c=k.DIE_TEMP_MIN_C,
        tool_wear_shots=config.starting_wear_shots,
        lot=sample_lot(0, rng),
        sensor_bias={name: 0.0 for name in _DRIFTING_SENSORS},
    )


def sample_lot(lot_id: int, rng: np.random.Generator) -> AlloyLot:
    """Chemistry is fixed per furnace charge, not per shot."""
    return AlloyLot(
        lot_id=lot_id,
        si_content_pct=float(rng.normal(BY_NAME["si_content_pct"].nominal, 0.55)),
        fe_content_pct=float(abs(rng.normal(BY_NAME["fe_content_pct"].nominal, 0.18))),
        mn_content_pct=float(abs(rng.normal(BY_NAME["mn_content_pct"].nominal, 0.06))),
    )


def advance(
    state: LineState, setpoints: dict[str, float], config: LineConfig, rng: np.random.Generator
) -> LineState:
    """Step the line one shot forward under the given setpoints."""
    die_temp = _next_die_temperature(state, setpoints, config)
    wear = _next_tool_wear(state, die_temp, config)
    shot_index = state.shot_index + 1
    return LineState(
        shot_index=shot_index,
        die_temp_c=die_temp,
        tool_wear_shots=wear,
        lot=_next_lot(state, shot_index, config, rng),
        shift_id=shot_index // config.shots_per_shift,
        sensor_bias=_next_sensor_bias(state, config, rng),
    )


def observed(state: LineState, setpoints: dict[str, float]) -> dict[str, float]:
    """What the historian records: true state seen through drifting sensors."""
    reading = dict(setpoints)
    reading["die_temp_c"] = state.die_temp_c
    reading["tool_wear_shots"] = state.tool_wear_shots
    reading["si_content_pct"] = state.lot.si_content_pct
    reading["fe_content_pct"] = state.lot.fe_content_pct
    reading["mn_content_pct"] = state.lot.mn_content_pct
    return _apply_sensor_bias(reading, state.sensor_bias)


_DRIFTING_SENSORS: tuple[str, ...] = (
    "pour_temp_c",
    "die_temp_c",
    "intensification_pressure_mpa",
)


def _next_die_temperature(
    state: LineState, setpoints: dict[str, float], config: LineConfig
) -> float:
    """Die heats toward the melt and is pulled back by the cooling interval."""
    pour_temp = setpoints["pour_temp_c"]
    cooling_time = setpoints["cooling_time_s"]
    heating = config.die_warmup_gain * (pour_temp - state.die_temp_c)
    cooling = config.die_cool_per_second * cooling_time
    return float(np.clip(state.die_temp_c + heating - cooling, 60.0, 400.0))


def _next_tool_wear(state: LineState, die_temp_c: float, config: LineConfig) -> float:
    """Wear accrues per shot plus an integral of thermal exposure above the window."""
    thermal_excess = max(die_temp_c - k.DIE_TEMP_MAX_C, 0.0)
    return state.tool_wear_shots + 1.0 + config.wear_thermal_gain * thermal_excess


def _next_lot(
    state: LineState, shot_index: int, config: LineConfig, rng: np.random.Generator
) -> AlloyLot:
    if shot_index % config.shots_per_lot != 0:
        return state.lot
    return sample_lot(state.lot.lot_id + 1, rng)


def _next_sensor_bias(
    state: LineState, config: LineConfig, rng: np.random.Generator
) -> dict[str, float]:
    """Slow random walk; this is what the drift monitor is meant to catch."""
    step = rng.normal(0.0, config.sensor_drift_sd, len(_DRIFTING_SENSORS))
    return {
        name: state.sensor_bias.get(name, 0.0) + float(s)
        for name, s in zip(_DRIFTING_SENSORS, step, strict=True)
    }


def _apply_sensor_bias(reading: dict[str, float], bias: dict[str, float]) -> dict[str, float]:
    out = dict(reading)
    for name, drift in bias.items():
        if name in out:
            out[name] += drift * BY_NAME[name].spread
    return out


def with_starting_wear(config: LineConfig, shots: float) -> LineConfig:
    return replace(config, starting_wear_shots=shots)
