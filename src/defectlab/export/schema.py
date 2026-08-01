"""The export contract: what tables leave this system, at what grain, keyed how.

This is a star schema because Power BI's engine is built for one. A single wide table would
render, but every measure would then have to guard against double-counting when a shot appears
once per attribution group, and slicers on parameter metadata would have nothing to attach to.

The contract is declared here and checked on write. A dashboard that silently loses a column is
a worse failure than one that refuses to build, because it fails at the viva rather than in CI.
"""

from __future__ import annotations

from dataclasses import dataclass

FACT_SHOT = "fact_shot"
FACT_PRODUCTION = "fact_production"
FACT_ATTRIBUTION = "fact_attribution"
FACT_SPC = "fact_spc"
FACT_COST_CURVE = "fact_cost_curve"
DIM_PARAMETER = "dim_parameter"
DIM_GROUP = "dim_group"
DIM_RULE = "dim_rule"
DIM_DATE = "dim_date"

# One shot a minute. Fixes the timestamp axis and matches the cycle assumed by the ISA-18.2
# alarms-per-hour figure in the economics layer; the two must not disagree.
SECONDS_PER_SHOT = 60.0
EPOCH = "2026-03-02 06:00:00"


@dataclass(frozen=True, slots=True)
class TableSpec:
    """One exported table: its grain, its key, and the columns a report may rely on."""

    name: str
    grain: str
    key: tuple[str, ...]
    columns: tuple[str, ...]

    @property
    def is_fact(self) -> bool:
        return self.name.startswith("fact_")


TABLES: dict[str, TableSpec] = {
    # The evaluation set carries no timestamp on purpose. It is oversampled and grouped by
    # label -- lag-1 label autocorrelation 0.997, a run of 453 identical labels -- so it is not a
    # production sequence, and stamping a clock on it would invent a time axis that never existed.
    FACT_SHOT: TableSpec(
        FACT_SHOT,
        "one row per evaluated part",
        ("shot_id",),
        (
            "shot_id",
            "lot_id",
            "die_id",
            "shift_id",
            "risk",
            "label",
            "flagged",
            "outcome",
            "dominant_mechanism",
        ),
    ),
    # A genuine contiguous run. Everything with a time axis hangs off this table: the clock, the
    # date dimension, and every control chart.
    FACT_PRODUCTION: TableSpec(
        FACT_PRODUCTION,
        "one row per shot of a contiguous production run",
        ("shot_id",),
        ("shot_id", "timestamp", "lot_id", "die_id", "shift_id", "risk", "label"),
    ),
    FACT_ATTRIBUTION: TableSpec(
        FACT_ATTRIBUTION,
        "one row per shot and feature group",
        ("shot_id", "group"),
        ("shot_id", "group", "contribution"),
    ),
    FACT_SPC: TableSpec(
        FACT_SPC,
        "one row per shot and chart",
        ("shot_id", "chart"),
        ("shot_id", "chart", "value", "centre", "lower", "upper", "signal", "rule"),
    ),
    FACT_COST_CURVE: TableSpec(
        FACT_COST_CURVE,
        "one row per candidate threshold",
        ("threshold",),
        ("threshold", "per_shot", "escape_rate", "overkill_rate", "alert_rate"),
    ),
    DIM_PARAMETER: TableSpec(
        DIM_PARAMETER,
        "one row per process parameter",
        ("parameter",),
        (
            "parameter",
            "unit",
            "nominal",
            "lower",
            "upper",
            "actionability",
            "group",
            "is_lever",
        ),
    ),
    DIM_GROUP: TableSpec(
        DIM_GROUP, "one row per feature group", ("group",), ("group", "description")
    ),
    DIM_RULE: TableSpec(
        DIM_RULE, "one row per Nelson rule", ("rule",), ("rule", "number", "description")
    ),
    DIM_DATE: TableSpec(
        DIM_DATE,
        "one row per calendar day covered by the run",
        ("date",),
        ("date", "year", "month", "day", "weekday", "shift_label"),
    ),
}


def spec(name: str) -> TableSpec:
    if name not in TABLES:
        raise KeyError(f"{name!r} is not part of the export contract")
    return TABLES[name]
