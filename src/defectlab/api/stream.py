"""Server-sent events for the live line.

SSE rather than WebSockets: this traffic is one-way, and SSE is plain HTTP, so it survives
proxies, reconnects on its own, and needs no client library. A bidirectional protocol here would
be more machinery for a channel that never carries anything upstream.

The generator sleeps between shots so a browser sees a line running, not a wall of history. It
also yields control on every iteration, which matters because the twin's step is synchronous and
would otherwise block the event loop for the whole run.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from ..twin import FEATURES, TwinConfig, stream_line
from .scoring import Scorer

DEFAULT_INTERVAL = 1.0
HEARTBEAT = ": keep-alive\n\n"


def format_event(payload: dict, event: str = "shot") -> str:
    """SSE framing: a named event, one JSON line, terminated by a blank line."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def shots(
    scorer: Scorer,
    config: TwinConfig | None = None,
    interval: float = DEFAULT_INTERVAL,
    limit: int | None = None,
) -> AsyncIterator[str]:
    """Yield scored shots forever, or `limit` of them when a test needs an end."""
    for produced, shot in enumerate(stream_line(config or TwinConfig()), start=1):
        readings = {name: float(shot.reading[name]) for name in FEATURES}
        yield format_event(_payload(scorer, shot, readings))
        if limit is not None and produced >= limit:
            return
        await asyncio.sleep(interval)


def _payload(scorer: Scorer, shot, readings: dict[str, float]) -> dict:
    risk = scorer.risk(readings)
    labels, abstained = scorer.prediction_set(readings)
    return {
        "shot_index": shot.index,
        "lot_id": shot.lot_id,
        "die_id": shot.die_id,
        "shift_id": shot.shift_id,
        "risk": round(risk, 6),
        "flagged": risk >= scorer.threshold,
        "prediction_set": labels,
        "abstained": abstained,
        "readings": {name: round(value, 4) for name, value in readings.items()},
    }
