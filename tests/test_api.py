"""The serving layer: audit chain integrity, input validation, and the endpoints."""

import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from defectlab.api import stream
from defectlab.api.audit import GENESIS, AuditLog, digest_of
from defectlab.api.schemas import ShotRequest
from defectlab.api.stream import format_event
from defectlab.twin import FEATURES, spec

PAYLOAD = {"risk": 0.42, "flagged": True}


def _readings(**overrides: float) -> dict[str, float]:
    base = {name: spec(name).nominal for name in FEATURES}
    return {**base, **overrides}


def test_the_first_entry_points_at_genesis():
    log = AuditLog()
    assert log.append("score", PAYLOAD).previous == GENESIS


def test_each_entry_commits_to_its_predecessor():
    log = AuditLog()
    first = log.append("score", PAYLOAD)
    second = log.append("score", PAYLOAD)
    assert second.previous == first.hash
    assert log.head == second.hash


def test_a_clean_chain_verifies():
    log = AuditLog()
    for _ in range(5):
        log.append("score", PAYLOAD)
    result = log.verify()
    assert result.intact
    assert result.entries == 5
    assert result.broken_at is None


def test_editing_a_past_decision_breaks_the_chain():
    """The whole point: a silently altered verdict must not still verify."""
    log = AuditLog()
    for _ in range(5):
        log.append("score", PAYLOAD)
    tampered = log.entries()[2]
    log._entries[2] = type(tampered)(
        tampered.index,
        tampered.timestamp,
        tampered.event,
        digest_of({"risk": 0.01, "flagged": False}),
        tampered.previous,
        tampered.hash,
    )
    result = log.verify()
    assert not result.intact
    assert result.broken_at == 2


def test_removing_an_entry_breaks_the_chain():
    log = AuditLog()
    for _ in range(5):
        log.append("score", PAYLOAD)
    del log._entries[2]
    assert not log.verify().intact


def test_the_digest_ignores_key_order():
    """The same decision must digest the same however the dict was assembled."""
    assert digest_of({"a": 1, "b": 2}) == digest_of({"b": 2, "a": 1})


def test_a_different_payload_digests_differently():
    assert digest_of({"risk": 0.42}) != digest_of({"risk": 0.43})


def test_the_log_survives_a_restart(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditLog(path)
    for _ in range(3):
        first.append("score", PAYLOAD)
    reopened = AuditLog(path)
    assert len(reopened) == 3
    assert reopened.verify().intact
    assert reopened.head == first.head


def test_a_restarted_log_continues_the_same_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditLog(path)
    first.append("score", PAYLOAD)
    reopened = AuditLog(path)
    assert reopened.append("score", PAYLOAD).previous == first.head
    assert AuditLog(path).verify().intact


def test_a_complete_reading_is_accepted():
    assert ShotRequest(readings=_readings()).readings


def test_a_missing_parameter_is_rejected():
    incomplete = _readings()
    del incomplete["pour_temp_c"]
    with pytest.raises(ValueError, match="missing parameters"):
        ShotRequest(readings=incomplete)


def test_a_reading_outside_the_machine_limits_is_rejected():
    """Outside its envelope the model extrapolates and still sounds confident."""
    with pytest.raises(ValueError, match="outside the machine limits"):
        ShotRequest(readings=_readings(pour_temp_c=2000.0))


def test_an_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown parameter"):
        ShotRequest(readings={**_readings(), "vibes": 1.0})


def test_sse_framing_is_well_formed():
    frame = format_event({"risk": 0.5}, "shot")
    assert frame.startswith("event: shot\ndata: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame.split("data: ")[1].strip()) == {"risk": 0.5}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from defectlab.api.app import create_app

    app = create_app(seed=11, estimator="xgboost_fast")
    with TestClient(app) as started:
        started.app.state.defectlab.audit = AuditLog(
            tmp_path_factory.mktemp("audit") / "audit.jsonl"
        )
        yield started


def test_health_reports_the_model_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_scoring_returns_a_calibrated_risk(client):
    body = client.post("/score", json={"readings": _readings()}).json()
    assert 0.0 <= body["risk"] <= 1.0
    assert body["flagged"] == (body["risk"] >= body["threshold"])


def test_scoring_is_written_to_the_audit_chain(client):
    before = client.get("/audit").json()["entries"]
    response = client.post("/score", json={"readings": _readings()}).json()
    after = client.get("/audit").json()
    assert after["entries"] == before + 1
    assert after["head"] == response["audit_hash"]
    assert after["intact"]


def test_a_rejected_shot_returns_422_and_is_not_audited(client):
    """A shot that never scored must leave no decision in the log."""
    before = client.get("/audit").json()["entries"]
    response = client.post("/score", json={"readings": _readings(die_temp_c=9000.0)})
    assert response.status_code == 422
    assert client.get("/audit").json()["entries"] == before


def test_the_served_threshold_is_not_the_fitted_one(client):
    """`FittedModel.threshold` is the optimum at the research prevalence. Comparing a 3 %-scale
    risk against it mixes two scales, and flagged 83 % of a nominal line before it was caught."""
    scorer = client.app.state.defectlab.scorer
    assert scorer.threshold > scorer.model.threshold


def test_the_served_gate_stays_inside_the_alarm_budget(client):
    """A gate an operator cannot keep up with is not a gate. ISA-18.2 caps it at 12 alarms/hour."""
    import itertools

    import pandas as pd

    from defectlab.api.scoring import MAX_ALERT_RATE
    from defectlab.economics import shift
    from defectlab.twin import TwinConfig, stream_line

    scorer = client.app.state.defectlab.scorer
    shots = itertools.islice(stream_line(TwinConfig(seed=5)), 400)
    frame = pd.DataFrame([{n: float(s.reading[n]) for n in FEATURES} for s in shots])
    risk = shift(
        scorer.model.score(frame[list(FEATURES)].to_numpy()),
        scorer.source_prevalence,
        scorer.target_prevalence,
    )
    assert (risk >= scorer.threshold).mean() <= MAX_ALERT_RATE


def test_audit_entries_are_paged(client):
    client.post("/score", json={"readings": _readings()})
    entries = client.get("/audit/entries", params={"limit": 1}).json()
    assert len(entries) == 1
    assert entries[0]["hash"]


def _collect(scorer, count: int) -> list[dict]:
    """Drive the generator directly. Reading an unbounded SSE body through TestClient hangs on
    teardown, and the generator is the thing worth testing anyway."""

    async def drain() -> list[str]:
        return [frame async for frame in stream.shots(scorer, interval=0.0, limit=count)]

    return [json.loads(frame.split("data: ")[1].strip()) for frame in asyncio.run(drain())]


def test_the_stream_scores_each_shot(client):
    frames = _collect(client.app.state.defectlab.scorer, 3)
    assert len(frames) == 3
    assert all(0.0 <= frame["risk"] <= 1.0 for frame in frames)


def test_the_stream_advances_through_the_line(client):
    frames = _collect(client.app.state.defectlab.scorer, 3)
    indices = [frame["shot_index"] for frame in frames]
    assert indices == sorted(indices)
    assert indices[-1] > indices[0]


def test_a_bounded_stream_terminates(client):
    """Without `limit` the response never ends, which is right for a demo and useless for a test."""
    response = client.get("/stream", params={"interval": 0.05, "limit": 2})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.count("event: shot") == 2
