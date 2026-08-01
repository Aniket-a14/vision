"""The producer: a machine publishing its own telemetry.

This is the half of the demo that makes it a line rather than a web page talking to itself. It
knows nothing about the model, the threshold or the cost of a defect -- a real PLC does not
either. It publishes what it measured and stops.

Shots are emitted on a wall clock, not as fast as the twin can generate them, because the point
of the exercise is a consumer that keeps up with a cycle time.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from ..twin import TwinConfig, stream_line
from ..twin.simulator import Shot
from . import codec, topics
from .transport import Transport

DEFAULT_CYCLE_S = 1.0


@dataclass(frozen=True, slots=True)
class LineConfig:
    cell: str = topics.DEFAULT_CELL
    cycle_s: float = DEFAULT_CYCLE_S
    limit: int | None = None


def announce(transport: Transport, cell: str, state: str, detail: str = "") -> None:
    """Retained, so a dashboard opened mid-shift sees the cell state without waiting a cycle."""
    transport.publish(
        topics.status(cell),
        codec.status_payload(state, cell, detail),
        qos=topics.AT_LEAST_ONCE,
        retain=True,
    )


def publish_shot(transport: Transport, cell: str, shot: Shot) -> None:
    """At-most-once: a reading that needed retransmitting is already stale."""
    transport.publish(
        topics.telemetry(cell),
        codec.telemetry_payload(shot),
        qos=topics.AT_MOST_ONCE,
    )


def run(transport: Transport, config: LineConfig, twin: TwinConfig | None = None) -> int:
    """Publish until the limit, then hand back how many shots went out."""
    announce(transport, config.cell, "running")
    produced = 0
    try:
        for shot in _shots(twin, config.limit):
            publish_shot(transport, config.cell, shot)
            produced += 1
            if config.cycle_s:
                time.sleep(config.cycle_s)
    finally:
        announce(transport, config.cell, "stopped", f"{produced} shots")
    return produced


def _shots(twin: TwinConfig | None, limit: int | None) -> Iterator[Shot]:
    for produced, shot in enumerate(stream_line(twin or TwinConfig()), start=1):
        yield shot
        if limit is not None and produced >= limit:
            return
