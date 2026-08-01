"""Orchestrates the digital twin: setpoints, line state, propensity, labels."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit

from . import dynamics, parameters, propensity
from .dynamics import LineConfig, LineState


@dataclass(frozen=True, slots=True)
class TwinConfig:
    seed: int = 42
    noise_sd: float = propensity.DEFAULT_NOISE_SD
    signal_gain: float = propensity.DEFAULT_SIGNAL_GAIN
    line: LineConfig = field(default_factory=LineConfig)


@dataclass(frozen=True, slots=True)
class Shot:
    index: int
    reading: dict[str, float]
    lot_id: int
    die_id: int
    shift_id: int


def draw_setpoints(rng: np.random.Generator) -> dict[str, float]:
    """Operator-chosen targets for one shot; state variables are excluded."""
    return {name: parameters.sample_scalar(name, rng) for name in SETPOINTS}


def run_line(n_shots: int, config: TwinConfig) -> pd.DataFrame:
    """Simulate a contiguous production run and return the historian view."""
    rng = np.random.default_rng(config.seed)
    records = [shot.reading | _context(shot) for shot in _iter_shots(n_shots, config, rng)]
    return parameters.clip_to_limits(pd.DataFrame.from_records(records))


def stream_line(config: TwinConfig, start_shot: int = 0) -> Iterator[Shot]:
    """Unbounded generator used by the live demo service."""
    rng = np.random.default_rng(config.seed + start_shot)
    state = dynamics.initial_state(config.line, rng)
    while True:
        state, shot = _step(state, config, rng)
        yield shot


def score(
    frame: pd.DataFrame, config: TwinConfig, target_prevalence: float | None = None
) -> pd.DataFrame:
    """Attach true defect probability and a sampled label. Never reads an existing label."""
    rng = np.random.default_rng(config.seed + 1)
    raw = propensity.evaluate(frame, rng, noise_sd=config.noise_sd, signal_gain=config.signal_gain)
    intercept = _intercept_for(raw.logit, target_prevalence)
    shifted = raw.logit + intercept
    result = propensity.PropensityResult(shifted, expit(shifted), raw.mechanisms)
    return _assemble(frame, result, rng)


def _intercept_for(logit: np.ndarray, target_prevalence: float | None) -> float:
    if target_prevalence is None:
        return 0.0
    return propensity.calibrate_intercept(logit, target_prevalence)


def _assemble(
    frame: pd.DataFrame, result: propensity.PropensityResult, rng: np.random.Generator
) -> pd.DataFrame:
    out = frame.copy()
    out["true_defect_prob"] = result.probability
    out["label"] = rng.binomial(1, result.probability)
    out["dominant_mechanism"] = result.dominant_mechanism().to_numpy()
    return out


def _iter_shots(n_shots: int, config: TwinConfig, rng: np.random.Generator) -> Iterator[Shot]:
    state = dynamics.initial_state(config.line, rng)
    for _ in range(n_shots):
        state, shot = _step(state, config, rng)
        yield shot


def _step(state: LineState, config: TwinConfig, rng: np.random.Generator) -> tuple[LineState, Shot]:
    setpoints = draw_setpoints(rng)
    reading = dynamics.observed(state, setpoints)
    shot = Shot(state.shot_index, reading, state.lot.lot_id, state.die.die_id, state.shift_id)
    return dynamics.advance(state, setpoints, config.line, rng), shot


def _context(shot: Shot) -> dict[str, float | int]:
    return {
        "shot_index": shot.index,
        "lot_id": shot.lot_id,
        "die_id": shot.die_id,
        "shift_id": shot.shift_id,
    }


# What the operator writes into the machine. `die_temp_c` is controllable on paper but is a
# thermal state here, driven by the dynamics rather than dialled in, so it is not a setpoint.
SETPOINTS: tuple[str, ...] = (
    "pour_temp_c",
    "intensification_pressure_mpa",
    "slow_shot_velocity_ms",
    "fast_shot_velocity_ms",
    "hold_time_s",
    "cooling_time_s",
)
