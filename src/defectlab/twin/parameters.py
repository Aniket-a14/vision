"""Process parameter definitions and their sampling distributions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class Actionability(StrEnum):
    """Whether an operator can move a parameter, and on what timescale."""

    IMMEDIATE = "immediate"
    SLOW = "slow"
    LOT_LEVEL = "lot_level"
    MAINTENANCE = "maintenance"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    unit: str
    nominal: float
    spread: float
    lower: float
    upper: float
    actionability: Actionability
    ramp_limit: float | None = None

    @property
    def is_controllable(self) -> bool:
        return self.actionability in (Actionability.IMMEDIATE, Actionability.SLOW)


SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("pour_temp_c", "C", 690.0, 16.0, 640.0, 740.0, Actionability.SLOW, 15.0),
    ParameterSpec("die_temp_c", "C", 225.0, 18.0, 150.0, 310.0, Actionability.SLOW, 20.0),
    ParameterSpec(
        "intensification_pressure_mpa",
        "MPa",
        78.0,
        12.0,
        40.0,
        130.0,
        Actionability.IMMEDIATE,
        25.0,
    ),
    ParameterSpec(
        "slow_shot_velocity_ms", "m/s", 0.28, 0.05, 0.12, 0.55, Actionability.IMMEDIATE, 0.06
    ),
    ParameterSpec(
        "fast_shot_velocity_ms", "m/s", 2.80, 0.35, 1.60, 4.20, Actionability.IMMEDIATE, 0.40
    ),
    ParameterSpec("hold_time_s", "s", 6.0, 1.0, 2.0, 12.0, Actionability.IMMEDIATE, 1.0),
    ParameterSpec("cooling_time_s", "s", 12.0, 2.2, 5.0, 25.0, Actionability.IMMEDIATE, 2.0),
    ParameterSpec("si_content_pct", "%", 10.5, 0.7, 8.0, 13.0, Actionability.LOT_LEVEL),
    ParameterSpec("fe_content_pct", "%", 0.78, 0.20, 0.20, 2.00, Actionability.LOT_LEVEL),
    ParameterSpec("mn_content_pct", "%", 0.28, 0.08, 0.05, 0.70, Actionability.LOT_LEVEL),
    ParameterSpec(
        "tool_wear_shots", "shots", 25000.0, 14000.0, 0.0, 60000.0, Actionability.MAINTENANCE
    ),
)

BY_NAME: dict[str, ParameterSpec] = {spec.name: spec for spec in SPECS}
FEATURES: tuple[str, ...] = tuple(spec.name for spec in SPECS)
CONTROLLABLE: tuple[str, ...] = tuple(s.name for s in SPECS if s.is_controllable)
LOT_LEVEL: tuple[str, ...] = tuple(
    s.name for s in SPECS if s.actionability is Actionability.LOT_LEVEL
)


def spec(name: str) -> ParameterSpec:
    return BY_NAME[name]


def clip_to_limits(frame: pd.DataFrame) -> pd.DataFrame:
    """Clamp every known parameter column to its physical machine limits."""
    out = frame.copy()
    for column in out.columns.intersection(pd.Index(FEATURES)):
        bounds = BY_NAME[column]
        out[column] = out[column].clip(bounds.lower, bounds.upper)
    return out


def sample_scalar(name: str, rng: np.random.Generator) -> float:
    """Single draw for one parameter, clamped. Hot path, so no pandas here."""
    bounds = BY_NAME[name]
    return float(np.clip(_draw(bounds, 1, rng)[0], bounds.lower, bounds.upper))


def sample_independent(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw uncorrelated parameters; the line dynamics add realistic structure later."""
    columns = {spec.name: _draw(spec, n, rng) for spec in SPECS}
    return clip_to_limits(pd.DataFrame(columns))


def sample_uniform_envelope(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Randomised interventional design used to fit the prescriptive surrogate."""
    columns = {s.name: rng.uniform(s.lower, s.upper, n) for s in SPECS}
    return pd.DataFrame(columns)


def _draw(spec: ParameterSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    if spec.name == "tool_wear_shots":
        return rng.uniform(spec.lower, spec.upper, n)
    if spec.name in ("fe_content_pct", "mn_content_pct"):
        return _draw_gamma(spec, n, rng)
    return rng.normal(spec.nominal, spec.spread, n)


def _draw_gamma(spec: ParameterSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    """Impurities are right-skewed, so match the mean and spread with a gamma."""
    shape = (spec.nominal / spec.spread) ** 2
    scale = spec.spread**2 / spec.nominal
    return rng.gamma(shape, scale, n)
