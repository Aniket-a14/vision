"""Shewhart chart factors, ASTM STP-15D.

Tabulated rather than derived: d2 is the expected range of n standard normals, which has no
closed form worth computing at runtime, and every other factor follows from it.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_SUBGROUP = 2
MAX_SUBGROUP = 10


@dataclass(frozen=True, slots=True)
class Factors:
    """d2 converts mean range to sigma; A2, D3 and D4 give the limits directly."""

    d2: float
    a2: float
    d3: float
    d4: float


FACTORS: dict[int, Factors] = {
    2: Factors(1.128, 1.880, 0.000, 3.267),
    3: Factors(1.693, 1.023, 0.000, 2.574),
    4: Factors(2.059, 0.729, 0.000, 2.282),
    5: Factors(2.326, 0.577, 0.000, 2.114),
    6: Factors(2.534, 0.483, 0.000, 2.004),
    7: Factors(2.704, 0.419, 0.076, 1.924),
    8: Factors(2.847, 0.373, 0.136, 1.864),
    9: Factors(2.970, 0.337, 0.184, 1.816),
    10: Factors(3.078, 0.308, 0.223, 1.777),
}

# Individuals charts use the moving range of pairs, so they read the n = 2 row.
E2_INDIVIDUALS = 3.0 / FACTORS[MIN_SUBGROUP].d2
D2_INDIVIDUALS = FACTORS[MIN_SUBGROUP].d2


def factors(subgroup: int) -> Factors:
    """An unsupported subgroup size is a caller error, not something to interpolate."""
    if subgroup not in FACTORS:
        raise ValueError(f"subgroup size must be {MIN_SUBGROUP}-{MAX_SUBGROUP}; got {subgroup}")
    return FACTORS[subgroup]
