"""The topic namespace and the quality-of-service policy.

Topics are hierarchical so a client can subscribe to one cell or to the whole plant with a
wildcard, which is the reason to use MQTT rather than a bespoke socket in the first place.

The QoS split is the load-bearing decision. Telemetry goes out at-most-once: a reading that
arrives late is already wrong, and a retransmitted one is worse than a dropped one. Verdicts go
at-least-once, because a lost reject means a defective part ships. At-least-once admits
duplicates, so the consumer has to be idempotent -- see `gate.Gate`.
"""

from __future__ import annotations

AT_MOST_ONCE = 0
AT_LEAST_ONCE = 1

ROOT = "defectlab"
DEFAULT_CELL = "cell-01"

TELEMETRY = "telemetry"
VERDICT = "verdict"
STATUS = "status"


def telemetry(cell: str = DEFAULT_CELL) -> str:
    """Raw process readings, one message per shot."""
    return f"{ROOT}/{cell}/{TELEMETRY}"


def verdict(cell: str = DEFAULT_CELL) -> str:
    """The gate's decision about a shot it saw on the telemetry topic."""
    return f"{ROOT}/{cell}/{VERDICT}"


def status(cell: str = DEFAULT_CELL) -> str:
    """Retained liveness. A client connecting mid-run learns the cell state immediately."""
    return f"{ROOT}/{cell}/{STATUS}"


def all_cells(leaf: str) -> str:
    """Plant-wide subscription: every cell's telemetry, verdicts or status."""
    return f"{ROOT}/+/{leaf}"


def cell_of(topic: str) -> str:
    """Recover the cell from a topic delivered through a wildcard subscription."""
    parts = topic.split("/")
    expected = 3
    if len(parts) != expected or parts[0] != ROOT:
        raise ValueError(f"not a defectlab topic: {topic}")
    return parts[1]
