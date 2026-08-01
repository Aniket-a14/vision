"""Throughput reporting for extraction passes that run for tens of minutes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

LOG = logging.getLogger("defectlab.imaging")

REPORT_EVERY_BATCHES = 10


@dataclass(slots=True)
class Progress:
    """Tracks images per second and the remaining time for one extraction pass."""

    total: int
    label: str
    every: int = REPORT_EVERY_BATCHES
    done: int = 0
    batches: int = 0
    started: float = field(default_factory=time.perf_counter)

    def update(self, count: int) -> None:
        self.done += count
        self.batches += 1
        if self.batches % self.every == 0 or self.done >= self.total:
            LOG.info("%s", self.line())

    def line(self) -> str:
        rate = self.rate()
        remaining = (self.total - self.done) / rate if rate else 0.0
        percent = 100.0 * self.done / self.total if self.total else 100.0
        return (
            f"{self.label}  {self.done:>5}/{self.total} ({percent:5.1f}%)  "
            f"{rate:5.1f} img/s  eta {clock(remaining)}"
        )

    def rate(self) -> float:
        elapsed = self.elapsed()
        return self.done / elapsed if elapsed > 0.0 else 0.0

    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def clock(seconds: float) -> str:
    minutes, remainder = divmod(int(max(seconds, 0.0)), 60)
    return f"{minutes:02d}:{remainder:02d}"
