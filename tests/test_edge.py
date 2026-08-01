"""The MQTT edge: topics, wire format, delivery guarantees, and the gate's idempotency."""

import itertools

import pytest

from defectlab.api.audit import AuditLog
from defectlab.edge import codec, gate, line, topics
from defectlab.edge.transport import LoopbackTransport, _matches
from defectlab.twin import FEATURES, TwinConfig, stream_line


def _shots(count: int, seed: int = 3):
    return list(itertools.islice(stream_line(TwinConfig(seed=seed)), count))


def test_topics_are_namespaced_by_cell():
    assert topics.telemetry("cell-02") == "defectlab/cell-02/telemetry"
    assert topics.cell_of(topics.verdict("cell-02")) == "cell-02"


def test_a_foreign_topic_is_rejected():
    with pytest.raises(ValueError, match="not a defectlab topic"):
        topics.cell_of("plc/cell-01/telemetry")


def test_telemetry_is_at_most_once_and_verdicts_at_least_once():
    """The asymmetry is the design: a stale reading is noise, a lost reject is a shipped defect."""
    assert topics.AT_MOST_ONCE < topics.AT_LEAST_ONCE


@pytest.mark.parametrize(
    ("pattern", "topic", "expected"),
    [
        ("defectlab/+/telemetry", "defectlab/cell-01/telemetry", True),
        ("defectlab/+/telemetry", "defectlab/cell-01/verdict", False),
        ("defectlab/+/telemetry", "defectlab/a/b/telemetry", False),
        ("defectlab/cell-01/telemetry", "defectlab/cell-01/telemetry", True),
    ],
)
def test_single_level_wildcards_match_one_segment(pattern, topic, expected):
    assert _matches(pattern, topic) is expected


def test_a_shot_encodes_every_feature():
    payload = codec.telemetry_payload(_shots(1)[0])
    assert set(payload["readings"]) == set(FEATURES)
    assert payload["shot_index"] == 0


def test_the_payload_round_trips():
    payload = codec.telemetry_payload(_shots(1)[0])
    assert codec.decode(codec.encode(payload))["readings"] == payload["readings"]


def test_a_payload_from_another_schema_is_refused():
    """Better a loud failure than a consumer reading a field that has quietly changed meaning."""
    raw = codec.encode({"shot_index": 1}).replace(b'"schema": 1', b'"schema": 99')
    with pytest.raises(ValueError, match="unsupported payload schema"):
        codec.decode(raw)


def test_the_line_publishes_one_message_per_shot():
    transport = LoopbackTransport()
    produced = line.run(transport, line.LineConfig(cycle_s=0.0, limit=5), TwinConfig(seed=1))
    telemetry = [t for t, _ in transport.published if t == topics.telemetry()]
    assert produced == 5
    assert len(telemetry) == 5


def test_the_line_announces_itself_and_its_stop():
    transport = LoopbackTransport()
    line.run(transport, line.LineConfig(cycle_s=0.0, limit=2), TwinConfig(seed=1))
    states = [p["state"] for t, p in transport.published if t == topics.status()]
    assert states == ["running", "stopped"]


def test_the_status_is_retained_for_a_late_subscriber():
    """A dashboard opened mid-shift must not wait a cycle to learn the cell is running."""
    transport = LoopbackTransport()
    line.announce(transport, topics.DEFAULT_CELL, "running")
    seen: list[dict] = []
    transport.subscribe(topics.status(), lambda _t, raw: seen.append(codec.decode(raw)))
    assert seen and seen[0]["state"] == "running"


def test_shot_indices_advance():
    transport = LoopbackTransport()
    line.run(transport, line.LineConfig(cycle_s=0.0, limit=4), TwinConfig(seed=1))
    indices = [p["shot_index"] for t, p in transport.published if t == topics.telemetry()]
    assert indices == sorted(indices)


class _Scorer:
    """The gate's contract with the model, without paying to fit one."""

    version = "test-1"
    threshold = 0.5

    def __init__(self, risk: float = 0.9):
        self._risk = risk
        self.calls = 0

    def risk(self, readings):
        self.calls += 1
        return self._risk

    def prediction_set(self, readings):
        return [1], False


def _wire(scorer, audit=None):
    transport = LoopbackTransport()
    subscriber = gate.Gate(scorer, audit=audit)
    subscriber.listen(transport)
    return transport, subscriber


def test_every_published_shot_gets_a_verdict():
    transport, subscriber = _wire(_Scorer())
    line.run(transport, line.LineConfig(cycle_s=0.0, limit=3), TwinConfig(seed=1))
    verdicts = [p for t, p in transport.published if t == topics.verdict()]
    assert len(verdicts) == 3
    assert subscriber.scored == 3


def test_the_verdict_carries_the_key_it_answers():
    """Telemetry and verdict are separate topics, so the join has to be in the payload."""
    transport, _ = _wire(_Scorer())
    line.run(transport, line.LineConfig(cycle_s=0.0, limit=1), TwinConfig(seed=1))
    telemetry = next(p for t, p in transport.published if t == topics.telemetry())
    verdict = next(p for t, p in transport.published if t == topics.verdict())
    assert verdict["shot_index"] == telemetry["shot_index"]
    assert verdict["lot_id"] == telemetry["lot_id"]


def test_the_flag_follows_the_threshold():
    transport, _ = _wire(_Scorer(risk=0.1))
    line.run(transport, line.LineConfig(cycle_s=0.0, limit=1), TwinConfig(seed=1))
    assert next(p for t, p in transport.published if t == topics.verdict())["flagged"] is False


def test_a_redelivered_shot_is_scored_once():
    """QoS 1 is at-least-once, so the gate will see duplicates and must not double-count them."""
    scorer = _Scorer()
    transport, subscriber = _wire(scorer)
    payload = codec.telemetry_payload(_shots(1)[0])
    for _ in range(3):
        transport.publish(topics.telemetry(), payload, qos=topics.AT_LEAST_ONCE)
    assert subscriber.scored == 1
    assert subscriber.duplicates == 2
    assert scorer.calls == 1


def test_a_duplicate_is_not_audited_twice(tmp_path):
    """The chain must record the decisions the line made, not the redeliveries the broker made."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    transport, _ = _wire(_Scorer(), audit=audit)
    payload = codec.telemetry_payload(_shots(1)[0])
    for _ in range(4):
        transport.publish(topics.telemetry(), payload, qos=topics.AT_LEAST_ONCE)
    assert len(audit) == 1
    assert audit.verify().intact


def test_an_audited_verdict_returns_its_hash(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    transport, subscriber = _wire(_Scorer(), audit=audit)
    payload = subscriber.on_message(transport, codec.encode(codec.telemetry_payload(_shots(1)[0])))
    assert payload["audit_hash"] == audit.head


def test_a_gate_without_an_audit_still_publishes():
    transport, subscriber = _wire(_Scorer())
    payload = subscriber.on_message(transport, codec.encode(codec.telemetry_payload(_shots(1)[0])))
    assert "audit_hash" not in payload


def test_a_gate_ignores_another_cells_telemetry():
    transport, subscriber = _wire(_Scorer())
    transport.publish(topics.telemetry("cell-99"), codec.telemetry_payload(_shots(1)[0]))
    assert subscriber.scored == 0


class _FakeClient:
    """Records the paho calls in order. No broker is installed, and the ordering is the risk."""

    def __init__(self, *args, **kwargs):
        self.calls: list[str] = []
        self.will = None
        self.on_message = None
        _FakeClient.last = self

    def will_set(self, topic, payload, qos=0, retain=False):
        self.calls.append("will_set")
        self.will = (topic, payload, qos, retain)

    def connect(self, host, port, keepalive=0):
        self.calls.append("connect")

    def loop_start(self):
        self.calls.append("loop_start")

    def loop_stop(self):
        self.calls.append("loop_stop")

    def disconnect(self):
        self.calls.append("disconnect")

    def publish(self, topic, payload, qos=0, retain=False):
        self.calls.append("publish")

    def subscribe(self, topic, qos=0):
        self.calls.append("subscribe")


@pytest.fixture
def paho_client(monkeypatch):
    paho = pytest.importorskip("paho.mqtt.client")
    monkeypatch.setattr(paho, "Client", _FakeClient)
    return _FakeClient


def test_the_will_is_registered_before_the_connection(paho_client):
    """A broker cannot publish a will it was never told about, so the order is load-bearing."""
    from defectlab.edge.transport import MqttTransport

    transport = MqttTransport(cell="cell-07")
    calls = paho_client.last.calls
    assert calls.index("will_set") < calls.index("connect")
    assert transport is not None


def test_the_will_announces_the_cell_offline_and_is_retained(paho_client):
    """This is what SSE cannot do: a crashed line announces itself instead of just going quiet."""
    from defectlab.edge.transport import MqttTransport

    MqttTransport(cell="cell-07")
    topic, payload, qos, retain = paho_client.last.will
    assert topic == topics.status("cell-07")
    assert codec.decode(payload)["state"] == "offline"
    assert qos == topics.AT_LEAST_ONCE
    assert retain


def test_closing_disconnects_cleanly_so_the_will_is_suppressed(paho_client):
    from defectlab.edge.transport import MqttTransport

    MqttTransport().close()
    assert paho_client.last.calls[-2:] == ["loop_stop", "disconnect"]


def test_the_seen_set_stays_bounded():
    """A shift is thousands of shots; the gate must not grow a set for the life of the process."""
    subscriber = gate.Gate(_Scorer())
    subscriber._seen = set(range(gate.SEEN_LIMIT))
    assert subscriber._accept(gate.SEEN_LIMIT + 1)
    assert len(subscriber._seen) == 1
