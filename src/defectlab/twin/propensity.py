"""Structural causal model mapping physics indices to a defect propensity.

Causal order is always parameters -> propensity -> sampled label. Never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import expit

from . import physics

# Weights follow published HPDC ANOVA: plunger velocities dominate porosity.
MECHANISM_WEIGHTS: Final[dict[str, float]] = {
    "air_entrapment": 1.15,
    "gas_porosity": 1.05,
    "shrinkage": 1.60,
    "cold_shut": 0.80,
    "misrun": 0.30,
    "beta_platelets": 0.30,
    "soldering": 0.32,
    "sludge": 0.18,
    "flash": 0.45,
    "tool_wear": 1.10,
}

# Tuned together so process-only AUC lands at ~0.85 with no feature correlating past 0.35.
DEFAULT_NOISE_SD: Final = 1.0
DEFAULT_SIGNAL_GAIN: Final = 3.0


@dataclass(frozen=True, slots=True)
class PropensityResult:
    logit: np.ndarray
    probability: np.ndarray
    mechanisms: pd.DataFrame

    def dominant_mechanism(self) -> pd.Series:
        """Which mechanism is unusually high for this shot, not which has the largest offset."""
        return self.excess().idxmax(axis=1)

    def excess(self) -> pd.DataFrame:
        return self.mechanisms - self.mechanisms.median(axis=0)


def mechanism_indices(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every physical failure mechanism for each shot."""
    beta = physics.beta_platelet_index(frame["fe_content_pct"], frame["si_content_pct"])
    return pd.DataFrame(
        {
            "air_entrapment": physics.air_entrapment_index(
                frame["slow_shot_velocity_ms"], frame["fast_shot_velocity_ms"]
            ),
            "gas_porosity": physics.gas_porosity_index(
                frame["pour_temp_c"], _holding_exposure(frame)
            ),
            "shrinkage": physics.shrinkage_index(
                frame["intensification_pressure_mpa"],
                frame["cooling_time_s"],
                frame["hold_time_s"],
                beta,
            ),
            "cold_shut": physics.cold_shut_index(
                frame["pour_temp_c"], frame["die_temp_c"], frame["fast_shot_velocity_ms"]
            ),
            "misrun": physics.misrun_index(
                frame["pour_temp_c"], frame["die_temp_c"], frame["si_content_pct"]
            ),
            "beta_platelets": beta,
            "soldering": physics.soldering_index(frame["fe_content_pct"], frame["die_temp_c"]),
            "sludge": physics.sludge_index(frame["fe_content_pct"], frame["mn_content_pct"]),
            "flash": physics.flash_index(
                frame["intensification_pressure_mpa"], frame["tool_wear_shots"]
            ),
            "tool_wear": _tool_wear_index(frame["tool_wear_shots"]),
        },
        index=frame.index,
    )


def weighted_contributions(
    indices: pd.DataFrame, weights: dict[str, float] | None = None
) -> pd.DataFrame:
    """Weights are overridable so `prescribe` can ask what advice survives them being wrong."""
    return indices.mul(pd.Series(weights or MECHANISM_WEIGHTS), axis=1)


def evaluate(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    noise_sd: float = DEFAULT_NOISE_SD,
    intercept: float = 0.0,
    signal_gain: float = DEFAULT_SIGNAL_GAIN,
    weights: dict[str, float] | None = None,
) -> PropensityResult:
    """Deterministic physics plus irreducible process noise."""
    contributions = weighted_contributions(mechanism_indices(frame), weights) * signal_gain
    signal = contributions.sum(axis=1).to_numpy()
    noise = rng.normal(0.0, noise_sd, len(frame))
    logit = signal + noise + intercept
    return PropensityResult(logit, expit(logit), contributions)


def calibrate_intercept(
    logit: np.ndarray, target_prevalence: float, tolerance: float = 1e-9
) -> float:
    """Bisect the intercept so mean predicted prevalence matches the image split."""
    low, high = -40.0, 40.0
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if expit(logit + middle).mean() < target_prevalence:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _tool_wear_index(shots: pd.Series) -> pd.Series:
    return (shots / 60000.0).clip(lower=0.0) ** 1.5


def _holding_exposure(frame: pd.DataFrame) -> pd.Series:
    """Longer cycles hold the melt hotter for longer, raising hydrogen pickup."""
    return 0.7 + 0.03 * frame["cooling_time_s"] + 0.02 * frame["hold_time_s"]
