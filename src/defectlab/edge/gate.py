"""The consumer: score every shot the line publishes, and publish the verdict back.

It scores with `api.scoring.Scorer` -- the same object the HTTP endpoint serves. Two transports,
one model, one threshold, one audit chain, so a decision cannot depend on how it was asked for.

At-least-once delivery means a redelivered shot will arrive twice. Rescoring it is harmless, but
auditing it twice is not: the chain would record two decisions where the line made one. So the
gate drops anything it has already seen, keyed on `shot_index`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..api.audit import AuditLog
from ..api.scoring import Scorer
from . import codec, topics
from .transport import Transport

SEEN_LIMIT = 100_000


@dataclass(slots=True)
class Gate:
    """A subscriber that turns telemetry into audited verdicts."""

    scorer: Scorer
    cell: str = topics.DEFAULT_CELL
    audit: AuditLog | None = None
    scored: int = 0
    duplicates: int = 0
    flagged: int = 0
    _seen: set[int] = field(default_factory=set)

    def listen(self, transport: Transport) -> None:
        """Subscribe at QoS 1: a verdict that was never delivered is a part that shipped."""
        transport.subscribe(
            topics.telemetry(self.cell),
            lambda topic, raw: self.on_message(transport, raw),
            qos=topics.AT_LEAST_ONCE,
        )

    def on_message(self, transport: Transport, raw: bytes) -> dict | None:
        message = codec.decode(raw)
        if not self._accept(message["shot_index"]):
            self.duplicates += 1
            return None
        payload = self._judge(message)
        transport.publish(
            topics.verdict(self.cell), payload, qos=topics.AT_LEAST_ONCE, retain=False
        )
        return payload

    def _accept(self, shot_index: int) -> bool:
        """Bounded memory: a shift is thousands of shots, and redelivery is immediate or never."""
        if shot_index in self._seen:
            return False
        if len(self._seen) >= SEEN_LIMIT:
            self._seen.clear()
        self._seen.add(shot_index)
        return True

    def _judge(self, message: dict) -> dict:
        readings = message["readings"]
        risk = self.scorer.risk(readings)
        labels, abstained = self.scorer.prediction_set(readings)
        extra = {
            "prediction_set": labels,
            "abstained": abstained,
            "model_version": self.scorer.version,
        }
        payload = codec.verdict_payload(message, risk, self.scorer.threshold, extra)
        self.scored += 1
        self.flagged += int(payload["flagged"])
        return self._record(payload, readings)

    def _record(self, payload: dict, readings: dict[str, float]) -> dict:
        """Same chain as the HTTP endpoint, so the transport leaves no trace in the evidence."""
        if self.audit is None:
            return payload
        entry = self.audit.append("score", {"readings": readings, **payload})
        return {**payload, "audit_hash": entry.hash}
