"""Request and response bodies.

Setpoints are validated against the twin's own machine limits, so a physically impossible shot
is rejected at the edge rather than scored. A model asked to extrapolate outside its training
envelope will still return a confident number, and that number is worthless.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from ..twin import FEATURES, spec
from .reasons import Reason


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


class PredicateResponse(BaseModel):
    parameter: str
    lower: float
    upper: float


class ExplainResponse(BaseModel):
    rule: str
    prediction: int
    precision: float = Field(..., description="how often the rule fixes the gate's verdict")
    coverage: float = Field(..., description="share of the line the rule applies to")
    predicates: list[PredicateResponse]


class OverrideRequest(BaseModel):
    """An operator disagreeing with the gate, on the record.

    `explanation_shown` is what the screen displayed when the decision was taken. It has to come
    from the client because only the client knows what was rendered, and re-deriving it later
    would record the explanation the model gives *now* -- which is the one thing an audit of a
    past decision must not do. It is therefore an attestation, not a verified fact.
    """

    audit_hash: str = Field(..., description="hash of the score being overridden")
    defective: bool = Field(..., description="the operator's verdict, not the model's")
    reason: Reason
    note: str = ""
    explanation_shown: str = Field("", description="the rule on screen when the call was made")

    @model_validator(mode="after")
    def _note_required_for_other(self) -> OverrideRequest:
        if self.reason is Reason.OTHER and not self.note.strip():
            raise ValueError("reason 'other' requires a note")
        return self


class OverrideResponse(BaseModel):
    audit_hash: str
    overrides: str = Field(..., description="the score entry this override answers")


class AuditResponse(BaseModel):
    intact: bool
    entries: int
    broken_at: int | None = None
    head: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    audit_entries: int
