"""Minimal-change recommendation for one flagged shot.

Greedy coordinate search: try every feasible value of every lever, keep the single best move,
repeat. Like the anchor search this is deterministic and cheap, and like the anchor search it
can settle for less than an exhaustive optimum would find.

The sparsity cap is the point, not a shortcut. A recommendation that moves six setpoints at
once cannot be executed by a shift, cannot be attributed if it works, and cannot be rolled back
cleanly if it does not. Three moves is already generous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from ..config import settings
from .actions import DEFAULT_GRID, Action, apply, levers
from .surrogate import Surrogate

MAX_ACTIONS = 3

# Below this risk there is nothing to fix, and advice becomes noise-chasing: a shot at 0.0001
# would still collect three setpoint changes worth several logits of margin it does not need.
# Prescription belongs on flagged shots only, so the default matches the line's base rate.
MIN_RISK = settings.target_defect_rate

# In logit units. The search works on the margin because probability saturates: a shot already
# at risk 0.9999 barely moves in probability however much the margin improves, so a
# probability-based search silently returns no advice for exactly the shots that need it most.
MIN_IMPROVEMENT = 0.01


@dataclass(frozen=True, slots=True)
class Recommendation:
    """What to change, and what the surrogate expects it to buy."""

    actions: tuple[Action, ...]
    risk_before: float
    risk_after: float
    margin_gain: float = 0.0

    @property
    def improvement(self) -> float:
        """Reported in probability, because that is what an operator can act on."""
        return self.risk_before - self.risk_after

    def describe(self) -> str:
        if not self.actions:
            return "no feasible single-shot change lowers the risk"
        lines = [action.describe() for action in self.actions]
        return "\n".join(
            [
                *lines,
                f"risk {self.risk_before:.4f} -> {self.risk_after:.4f} "
                f"({self.improvement:+.4f}), margin {self.margin_gain:+.3f} logits",
            ]
        )


def recommend(
    surrogate: Surrogate,
    reading: dict[str, float],
    *,
    max_actions: int = MAX_ACTIONS,
    grid: int = DEFAULT_GRID,
    min_risk: float = MIN_RISK,
) -> Recommendation:
    """Grow a set of setpoint moves until nothing feasible helps or the cap is reached."""
    start = surrogate.logit_of(reading)
    if expit(start) < min_risk:
        return Recommendation((), float(expit(start)), float(expit(start)), 0.0)
    chosen: list[Action] = []
    current = start
    pool = list(levers())
    for _ in range(max_actions):
        action, margin = _best_move(surrogate, reading, chosen, pool, grid)
        if action is None or current - margin < MIN_IMPROVEMENT:
            break
        chosen.append(action)
        pool.remove(action.name)
        current = margin
    return _result(surrogate, reading, chosen, start, current)


def _result(
    surrogate: Surrogate,
    reading: dict[str, float],
    chosen: list[Action],
    start: float,
    end: float,
) -> Recommendation:
    """Search in logits, report in probability; the two are not interchangeable near saturation."""
    return Recommendation(tuple(chosen), float(expit(start)), float(expit(end)), start - end)


def _best_move(
    surrogate: Surrogate,
    reading: dict[str, float],
    chosen: list[Action],
    pool: list[str],
    grid: int,
) -> tuple[Action | None, float]:
    """One extra lever at a time; already-chosen moves stay fixed while the next is searched."""
    scored = [_best_for(surrogate, reading, chosen, name, grid) for name in pool]
    if not scored:
        return None, float("inf")
    return min(scored, key=lambda pair: pair[1])


def _best_for(
    surrogate: Surrogate, reading: dict[str, float], chosen: list[Action], name: str, grid: int
) -> tuple[Action, float]:
    """Evaluate one lever's whole reachable range in a single batched call."""
    import pandas as pd

    from .actions import reachable

    values = reachable(name, float(reading[name]), grid)
    frame = pd.DataFrame([apply(reading, tuple(chosen))] * len(values))
    frame[name] = values
    margins = surrogate.logit(frame)
    best = int(np.argmin(margins))
    return Action(name, float(reading[name]), float(values[best])), float(margins[best])
