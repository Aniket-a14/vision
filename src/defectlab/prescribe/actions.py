"""The action space: what an operator can actually change, and by how much in one step.

Three separate constraints, and conflating them produces advice that cannot be followed.
`Actionability` says whether a parameter is a lever at all -- alloy chemistry is fixed once the
lot is charged, and telling a shift to lower iron content is not a recommendation. The machine
limits say where the parameter can physically go. The ramp limit says how far it can move in
one shot, which is what stops the search proposing a 50 C jump in pour temperature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..twin import SETPOINTS, spec

DEFAULT_GRID = 9


@dataclass(frozen=True, slots=True)
class Action:
    """One proposed setpoint move, in the parameter's own units."""

    name: str
    current: float
    proposed: float

    @property
    def delta(self) -> float:
        return self.proposed - self.current

    def describe(self) -> str:
        unit = spec(self.name).unit
        return f"{self.name}: {self.current:.4g} -> {self.proposed:.4g} ({self.delta:+.3g} {unit})"


def levers() -> tuple[str, ...]:
    """Setpoints only. A lot-level or maintenance parameter is not a shift-level action."""
    return tuple(name for name in SETPOINTS if spec(name).is_controllable)


def reachable(name: str, current: float, grid: int = DEFAULT_GRID) -> np.ndarray:
    """Feasible values for the next shot: inside the machine limits and inside the ramp limit."""
    bounds = spec(name)
    reach = bounds.ramp_limit if bounds.ramp_limit is not None else float("inf")
    lower = max(bounds.lower, current - reach)
    upper = min(bounds.upper, current + reach)
    if upper <= lower:
        return np.array([current])
    return np.linspace(lower, upper, grid)


def apply(reading: dict[str, float], actions: tuple[Action, ...]) -> dict[str, float]:
    """A copy of the shot with the proposed setpoints written in."""
    return reading | {action.name: action.proposed for action in actions}
