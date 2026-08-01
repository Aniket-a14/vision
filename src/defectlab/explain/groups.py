"""Physically meaningful feature groups.

A fusion model has 11 process columns and 30 image components. Per-column attribution is
unreadable and an individual principal component means nothing to a process engineer, so
attribution is reported over groups that map onto something a foundry can actually change.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..twin import FEATURES

IMAGE_PREFIX = "img_pc"
IMAGE_GROUP = "image"

PROCESS_GROUPS: dict[str, tuple[str, ...]] = {
    "thermal": ("pour_temp_c", "die_temp_c", "cooling_time_s"),
    "pressure": ("intensification_pressure_mpa", "hold_time_s"),
    "fill": ("slow_shot_velocity_ms", "fast_shot_velocity_ms"),
    "chemistry": ("si_content_pct", "fe_content_pct", "mn_content_pct"),
    "tooling": ("tool_wear_shots",),
}


def group_of(name: str) -> str:
    """Which group a column belongs to; an unmapped column is a bug, not a default case."""
    if name.startswith(IMAGE_PREFIX):
        return IMAGE_GROUP
    for group, members in PROCESS_GROUPS.items():
        if name in members:
            return group
    raise KeyError(f"column {name!r} belongs to no group; add it to PROCESS_GROUPS")


def assign(names: Sequence[str]) -> dict[str, list[int]]:
    """Column indices per group, in the order the groups first appear."""
    indices: dict[str, list[int]] = {}
    for position, name in enumerate(names):
        indices.setdefault(group_of(name), []).append(position)
    return indices


def _unmapped() -> tuple[str, ...]:
    mapped = {name for members in PROCESS_GROUPS.values() for name in members}
    return tuple(name for name in FEATURES if name not in mapped)


# A feature added to the twin without a group would silently vanish from every explanation.
if _unmapped():
    raise RuntimeError(f"process features are ungrouped: {_unmapped()}")
