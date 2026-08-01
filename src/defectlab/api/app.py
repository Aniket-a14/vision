"""The FastAPI application: score, explain, prescribe, stream, audit.

The model is fitted once during startup rather than per request. Fitting takes seconds and the
result is deterministic given the seed, so a restart serves an identical model -- which is the
property an audit trail needs if its hashes are to mean anything across a redeploy.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..config import settings
from . import reasons, scoring, stream
from .audit import AuditLog
from .schemas import (
    ActionResponse,
    AuditResponse,
    ExplainResponse,
    HealthResponse,
    OverrideRequest,
    OverrideResponse,
    PrescribeResponse,
    ScoreResponse,
    ShotRequest,
)

SSE_MEDIA_TYPE = "text/event-stream"
STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
MAX_AUDIT_PAGE = 500
MAX_STREAM_SHOTS = 10000

# The Vite dev server. Production serves the built bundle from the same origin and needs none.
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@dataclass(slots=True)
class State:
    """Everything fitted at startup. Held on the app, not in a module global."""

    scorer: scoring.Scorer
    audit: AuditLog
    surrogate: object | None = None
    explainer: object | None = None


def create_app(seed: int = 42, estimator: str = "xgboost") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.defectlab = State(
            scorer=scoring.build(seed=seed, estimator=estimator),
            audit=AuditLog(settings.paths.processed / "audit.jsonl"),
        )
        yield

    app = FastAPI(title="DefectLab", version="1.1", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    _register(app)
    return app


def _state(app: FastAPI) -> State:
    return app.state.defectlab


def _register(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        state = _state(app)
        return HealthResponse(
            status="ok",
            model_version=state.scorer.version,
            audit_entries=len(state.audit),
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ShotRequest) -> ScoreResponse:
        """Every scored shot is written to the audit chain before the answer is returned."""
        state = _state(app)
        risk = state.scorer.risk(request.readings)
        labels, abstained = state.scorer.prediction_set(request.readings)
        flagged = risk >= state.scorer.threshold
        entry = state.audit.append(
            "score",
            {
                "readings": request.readings,
                "risk": risk,
                "threshold": state.scorer.threshold,
                "flagged": flagged,
                "model_version": state.scorer.version,
            },
        )
        return ScoreResponse(
            risk=risk,
            threshold=state.scorer.threshold,
            flagged=flagged,
            prediction_set=labels,
            abstained=abstained,
            model_version=state.scorer.version,
            audit_hash=entry.hash,
        )

    @app.post("/prescribe", response_model=PrescribeResponse)
    def prescribe(request: ShotRequest) -> PrescribeResponse:
        """Setpoint advice for one shot, with how well it survives a perturbed simulator."""
        from ..prescribe import recommend, stability
        from ..twin import spec

        state = _state(app)
        advice = recommend(_surrogate(state), request.readings)
        survives = stability(advice, request.readings, trials=100)
        return PrescribeResponse(
            actions=[
                ActionResponse(
                    parameter=action.name,
                    current=action.current,
                    proposed=action.proposed,
                    delta=action.delta,
                    unit=spec(action.name).unit,
                )
                for action in advice.actions
            ],
            risk_before=advice.risk_before,
            risk_after=advice.risk_after,
            margin_gain=advice.margin_gain,
            stability=survives.rate,
        )

    @app.post("/explain", response_model=ExplainResponse)
    def explain(request: ShotRequest) -> ExplainResponse:
        """The anchor rule for one shot, grown against the served threshold."""
        return ExplainResponse(**_explainer(_state(app)).rule(request.readings))

    @app.get("/parameters")
    def parameters() -> list[dict]:
        """Machine limits and actionability, so the UI never hard-codes a second copy of the
        physics. A sandbox slider that allows an impossible setpoint teaches the wrong thing."""
        from ..twin import FEATURES, spec

        return [
            {
                "name": name,
                "unit": spec(name).unit,
                "nominal": spec(name).nominal,
                "lower": spec(name).lower,
                "upper": spec(name).upper,
                "actionability": spec(name).actionability.value,
                "ramp_limit": spec(name).ramp_limit,
            }
            for name in FEATURES
        ]

    @app.get("/reasons")
    def override_reasons() -> list[dict]:
        """Served rather than hard-coded in the UI, so the two cannot drift apart."""
        return reasons.catalogue()

    @app.post("/override", response_model=OverrideResponse)
    def override(request: OverrideRequest) -> OverrideResponse:
        """Record an operator disagreeing with the gate, against the decision they saw."""
        state = _state(app)
        if not state.audit.contains(request.audit_hash):
            raise HTTPException(status_code=404, detail="no such decision in the audit log")
        entry = state.audit.append(
            "override",
            {
                "overrides": request.audit_hash,
                "defective": request.defective,
                "reason": request.reason.value,
                "note": request.note,
                "explanation_shown": request.explanation_shown,
                "model_version": state.scorer.version,
            },
        )
        return OverrideResponse(audit_hash=entry.hash, overrides=request.audit_hash)

    @app.get("/stream")
    def live(
        interval: float = Query(1.0, ge=0.05, le=60.0),
        limit: int | None = Query(None, ge=1, le=MAX_STREAM_SHOTS),
    ) -> StreamingResponse:
        """SSE of the running line. One-way traffic, so plain HTTP beats a socket.

        `limit` bounds the feed. The live demo omits it and runs forever; anything that needs
        the response to actually end -- a test, a curl, a fixed-length replay -- sets it.
        """
        state = _state(app)
        return StreamingResponse(
            stream.shots(state.scorer, interval=interval, limit=limit),
            media_type=SSE_MEDIA_TYPE,
            headers=STREAM_HEADERS,
        )

    @app.get("/audit", response_model=AuditResponse)
    def audit() -> AuditResponse:
        state = _state(app)
        result = state.audit.verify()
        return AuditResponse(
            intact=result.intact,
            entries=result.entries,
            broken_at=result.broken_at,
            head=state.audit.head,
        )

    @app.get("/audit/entries")
    def audit_entries(limit: int = Query(50, ge=1, le=MAX_AUDIT_PAGE)) -> list[dict]:
        return [entry.as_dict() for entry in _state(app).audit.entries(limit)]


def _surrogate(state: State):
    """Fitted on first use: it costs seconds, and a health check should not pay for it."""
    from ..prescribe import fit

    if state.surrogate is None:
        state.surrogate = fit(shots=8000, seed=42)
    if state.surrogate is None:
        raise HTTPException(status_code=503, detail="surrogate unavailable")
    return state.surrogate


def _explainer(state: State):
    """Also first-use: the anchor background costs a twin run."""
    from . import explaining

    if state.explainer is None:
        state.explainer = explaining.build(state.scorer)
    return state.explainer
