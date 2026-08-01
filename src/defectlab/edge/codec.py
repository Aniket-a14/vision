"""Wire format for MQTT payloads.

JSON, not pickle or a binary schema. The payload has to be readable by MQTT Explorer, a Node-RED
flow and a browser without any of them importing this package -- that interoperability is most of
what a broker is for.

Every message carries `shot_index`. It is the idempotency key: QoS 1 delivers at least once, so a
consumer will eventually see the same shot twice and must be able to tell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..twin import FEATURES
from ..twin.simulator import Shot

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Telemetry:
    """One shot as it leaves the machine: readings, provenance, no verdict."""

    shot_index: int
    lot_id: int
    die_id: int
    shift_id: int
    readings: dict[str, float]

    @classmethod
    def of(cls, shot: Shot) -> Telemetry:
        return cls(
            shot_index=shot.index,
            lot_id=shot.lot_id,
            die_id=shot.die_id,
            shift_id=shot.shift_id,
            readings={name: round(float(shot.reading[name]), 4) for name in FEATURES},
        )


def encode(payload: dict) -> bytes:
    """Stamp the schema version so an old consumer can refuse rather than misread."""
    return json.dumps({"schema": SCHEMA_VERSION, **payload}).encode()


def decode(raw: bytes) -> dict:
    payload = json.loads(raw.decode())
    seen = payload.get("schema")
    if seen != SCHEMA_VERSION:
        raise ValueError(f"unsupported payload schema: {seen}")
    return payload


def telemetry_payload(shot: Shot) -> dict:
    reading = Telemetry.of(shot)
    return {
        "shot_index": reading.shot_index,
        "lot_id": reading.lot_id,
        "die_id": reading.die_id,
        "shift_id": reading.shift_id,
        "readings": reading.readings,
    }


def verdict_payload(message: dict, risk: float, threshold: float, extra: dict) -> dict:
    """A decision, carrying the telemetry key it answers so the two can be joined downstream."""
    return {
        "shot_index": message["shot_index"],
        "lot_id": message["lot_id"],
        "risk": round(risk, 6),
        "threshold": round(threshold, 6),
        "flagged": risk >= threshold,
        **extra,
    }


def status_payload(state: str, cell: str, detail: str = "") -> dict:
    return {"state": state, "cell": cell, "detail": detail}
