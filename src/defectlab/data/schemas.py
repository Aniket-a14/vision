"""Runtime data contracts. These run as tests, not as documentation."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

from ..twin.parameters import BY_NAME


def limits(name: str) -> dict[str, float]:
    spec = BY_NAME[name]
    return {"min_value": spec.lower, "max_value": spec.upper}


class ShotSchema(pa.DataFrameModel):
    shot_index: Series[int] = pa.Field(ge=0)
    lot_id: Series[int] = pa.Field(ge=0)
    shift_id: Series[int] = pa.Field(ge=0)

    pour_temp_c: Series[float] = pa.Field(in_range=limits("pour_temp_c"))
    die_temp_c: Series[float] = pa.Field(in_range=limits("die_temp_c"))
    intensification_pressure_mpa: Series[float] = pa.Field(
        in_range=limits("intensification_pressure_mpa")
    )
    slow_shot_velocity_ms: Series[float] = pa.Field(in_range=limits("slow_shot_velocity_ms"))
    fast_shot_velocity_ms: Series[float] = pa.Field(in_range=limits("fast_shot_velocity_ms"))
    hold_time_s: Series[float] = pa.Field(in_range=limits("hold_time_s"))
    cooling_time_s: Series[float] = pa.Field(in_range=limits("cooling_time_s"))
    si_content_pct: Series[float] = pa.Field(in_range=limits("si_content_pct"))
    fe_content_pct: Series[float] = pa.Field(in_range=limits("fe_content_pct"))
    mn_content_pct: Series[float] = pa.Field(in_range=limits("mn_content_pct"))
    tool_wear_shots: Series[float] = pa.Field(ge=0)

    class Config:
        strict = False
        coerce = True


class LabelledShotSchema(ShotSchema):
    true_defect_prob: Series[float] = pa.Field(in_range={"min_value": 0.0, "max_value": 1.0})
    label: Series[int] = pa.Field(isin=[0, 1])
    dominant_mechanism: Series[str]


class PairedShotSchema(LabelledShotSchema):
    part_id: Series[str]
    image_path: Series[str]
    split: Series[str] = pa.Field(isin=["train", "test"])
