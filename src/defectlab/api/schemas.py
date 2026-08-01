"""Request and response bodies.

Setpoints are validated against the twin's own machine limits, so a physically impossible shot
is rejected at the edge rather than scored. A model asked to extrapolate outside its training
envelope will still return a confident number, and that number is worthless.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..twin import FEATURES, spec


class ShotRequest(BaseModel):
    """One shot's process telemetry. Every parameter is required; there is no sensible default."""

    readings: dict[str, float] = Field(..., description="parameter name to measured value")

    @field_validator("readings")
    @classmethod
    def _complete_and_in_range(cls, value: dict[str, float]) -> dict[str, float]:
        missing = set(FEATURES) - set(value)
        if missing:
            raise ValueError(f"missing parameters: {sorted(missing)}")
        for name, reading in value.items():
            _check_range(name, reading)
        return value


def _check_range(name: str, reading: float) -> None:
    """Outside the machine limits the model is extrapolating, however confident it sounds."""
    if name not in FEATURES:
        raise ValueError(f"unknown parameter: {name}")
    bounds = spec(name)
    if not bounds.lower <= reading <= bounds.upper:
        raise ValueError(
            f"{name}={reading} is outside the machine limits [{bounds.lower}, {bounds.upper}]"
        )


class ScoreResponse(BaseModel):
    risk: float
    threshold: float
    flagged: bool
    prediction_set: list[int] = Field(..., description="conformal set at the configured alpha")
    abstained: bool = Field(..., description="the set holds both classes, so the model declines")
    model_version: str
    audit_hash: str


class ActionResponse(BaseModel):
    parameter: str
    current: float
    proposed: float
    delta: float
    unit: str


class PrescribeResponse(BaseModel):
    actions: list[ActionResponse]
    risk_before: float
    risk_after: float
    margin_gain: float
    stability: float = Field(..., description="share of perturbed simulators that still improve")


class AuditResponse(BaseModel):
    intact: bool
    entries: int
    broken_at: int | None = None
    head: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    audit_entries: int
