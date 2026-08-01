"""Builders for each exported table.

Pure transformations over already-computed inputs. Nothing here fits a model or runs the twin,
so the whole contract is testable in milliseconds without touching the image caches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..explain.groups import PROCESS_GROUPS, group_of
from ..spc import RULES
from ..twin import FEATURES, SETPOINTS
from ..twin import spec as parameter_spec
from .schema import EPOCH, SECONDS_PER_SHOT

GROUP_DESCRIPTIONS: dict[str, str] = {
    "thermal": "Melt and die temperatures, and the cooling that follows",
    "pressure": "Intensification pressure and hold time",
    "fill": "Plunger velocities: the dominant porosity path",
    "chemistry": "Alloy composition, fixed once the lot is charged",
    "tooling": "Die wear across the campaign",
    "image": "Reduced components of the inline camera embedding",
}

RULE_DESCRIPTIONS: dict[str, str] = {
    "beyond_3_sigma": "One point outside the control limits",
    "nine_one_side": "Nine points in a row on one side of centre: a shift",
    "six_trending": "Six points in a row rising or falling: a trend",
    "fourteen_alternating": "Fourteen points alternating: overcontrol",
    "two_of_three_beyond_2": "Two of three points beyond 2 sigma, same side",
    "four_of_five_beyond_1": "Four of five points beyond 1 sigma, same side",
    "fifteen_hugging": "Fifteen points inside 1 sigma: stratification",
    "eight_avoiding": "Eight points outside 1 sigma: a bimodal process",
}

OUTCOMES = ("true_negative", "false_positive", "false_negative", "true_positive")


@dataclass(frozen=True, slots=True)
class ExportInputs:
    """Two grains kept apart on purpose.

    `shots` is the held-out evaluation set: oversampled, grouped by label, and the only honest
    source of model metrics. `production` is a contiguous run of the line, and is the only
    honest source of anything with a time axis. Mixing them produced a control chart signalling
    on 48 % of points, every one of them an artefact of row order.
    """

    shots: pd.DataFrame
    risk: np.ndarray
    threshold: float
    cost_curve: pd.DataFrame
    attribution: pd.DataFrame | None = None
    spc: pd.DataFrame | None = None
    production: pd.DataFrame | None = None
    production_risk: np.ndarray | None = None


def fact_shot(inputs: ExportInputs) -> pd.DataFrame:
    frame = inputs.shots.reset_index(drop=True)
    flagged = inputs.risk >= inputs.threshold
    labels = frame["label"].to_numpy().astype(bool)
    return pd.DataFrame(
        {
            "shot_id": np.arange(len(frame)),
            "lot_id": frame.get("lot_id", pd.Series(0, index=frame.index)),
            "die_id": frame.get("die_id", pd.Series(0, index=frame.index)),
            "shift_id": frame.get("shift_id", pd.Series(0, index=frame.index)),
            "risk": inputs.risk,
            "label": labels.astype(int),
            "flagged": flagged.astype(int),
            "outcome": _outcomes(labels, flagged),
            "dominant_mechanism": frame.get(
                "dominant_mechanism", pd.Series("unknown", index=frame.index)
            ),
        }
    )


PRODUCTION_COLUMNS = ["shot_id", "timestamp", "lot_id", "die_id", "shift_id", "risk", "label"]


def fact_production(inputs: ExportInputs) -> pd.DataFrame:
    """The line as it actually ran. Everything with a clock on it hangs off this table."""
    if inputs.production is None or inputs.production_risk is None:
        return pd.DataFrame(columns=PRODUCTION_COLUMNS)
    frame = inputs.production.reset_index(drop=True)
    return pd.DataFrame(
        {
            "shot_id": np.arange(len(frame)),
            "timestamp": _timestamps(len(frame)),
            "lot_id": frame.get("lot_id", pd.Series(0, index=frame.index)),
            "die_id": frame.get("die_id", pd.Series(0, index=frame.index)),
            "shift_id": frame.get("shift_id", pd.Series(0, index=frame.index)),
            "risk": inputs.production_risk,
            "label": frame["label"].to_numpy().astype(int),
        }
    )


def fact_attribution(inputs: ExportInputs) -> pd.DataFrame:
    """Long format: Power BI slices a group dimension far better than it unpivots columns."""
    if inputs.attribution is None:
        return pd.DataFrame(columns=["shot_id", "group", "contribution"])
    wide = inputs.attribution.reset_index(drop=True)
    wide.insert(0, "shot_id", np.arange(len(wide)))
    long = wide.melt(id_vars="shot_id", var_name="group", value_name="contribution")
    return long.sort_values(["shot_id", "group"], ignore_index=True)


SPC_COLUMNS = ["shot_id", "chart", "value", "centre", "lower", "upper", "signal", "rule"]


def fact_spc(inputs: ExportInputs) -> pd.DataFrame:
    if inputs.spc is None:
        return pd.DataFrame(columns=SPC_COLUMNS)
    return inputs.spc.reset_index(drop=True)


def fact_cost_curve(inputs: ExportInputs) -> pd.DataFrame:
    columns = ["threshold", "per_shot", "escape_rate", "overkill_rate", "alert_rate"]
    return inputs.cost_curve[columns].reset_index(drop=True)


def dim_parameter(_: ExportInputs | None = None) -> pd.DataFrame:
    """Parameter metadata, so a slicer can filter by what an operator may actually change."""
    rows = [_parameter_row(name) for name in FEATURES]
    return pd.DataFrame(rows)


def dim_group(_: ExportInputs | None = None) -> pd.DataFrame:
    known = [*PROCESS_GROUPS, "image"]
    return pd.DataFrame(
        {"group": known, "description": [GROUP_DESCRIPTIONS[name] for name in known]}
    )


def dim_rule(_: ExportInputs | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rule": list(RULES),
            "number": list(range(1, len(RULES) + 1)),
            "description": [RULE_DESCRIPTIONS[name] for name in RULES],
        }
    )


def dim_date(inputs: ExportInputs) -> pd.DataFrame:
    """A real date table, because Power BI time intelligence will not work without one.

    Built from the production run, which is the only table here with a genuine time axis.
    """
    count = len(inputs.production) if inputs.production is not None else 1
    stamps = pd.Series(_timestamps(count))
    days = pd.to_datetime(stamps.dt.date.unique())
    return pd.DataFrame(
        {
            "date": days,
            "year": days.year,
            "month": days.month,
            "day": days.day,
            "weekday": days.day_name(),
            "shift_label": ["day" if index % 2 == 0 else "night" for index in range(len(days))],
        }
    )


def _parameter_row(name: str) -> dict[str, object]:
    bounds = parameter_spec(name)
    return {
        "parameter": name,
        "unit": bounds.unit,
        "nominal": bounds.nominal,
        "lower": bounds.lower,
        "upper": bounds.upper,
        "actionability": str(bounds.actionability),
        "group": group_of(name),
        "is_lever": int(name in SETPOINTS and bounds.is_controllable),
    }


def _timestamps(count: int) -> pd.Series:
    start = pd.Timestamp(EPOCH)
    return pd.Series(start + pd.to_timedelta(np.arange(count) * SECONDS_PER_SHOT, unit="s"))


def _outcomes(labels: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Named rather than numeric: a dashboard legend reading 0/1/2/3 helps nobody."""
    return np.array(OUTCOMES)[labels.astype(int) * 2 + flagged.astype(int)]
