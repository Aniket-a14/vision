"""Hash-chained audit log for scrap-or-ship decisions.

Every entry commits to its predecessor, so altering or removing any past decision invalidates
every hash after it. A quality system that cannot show *which* model made a call, on what input,
at what threshold, is not auditable, and "the model said so" is not a defence to a customer.

What this does not do, stated plainly because the distinction matters: the chain proves
**integrity and ordering**, not **authenticity**. Anyone who can append can also recompute the
whole chain from genesis. Real tamper-evidence needs the head hash published somewhere the
writer does not control -- a customer's system, a timestamping authority, another party's log.
The chain makes silent edits impossible; it does not make a determined operator honest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

GENESIS = "0" * 64
ENCODING = "utf-8"


@dataclass(frozen=True, slots=True)
class Entry:
    """One recorded decision. `digest` covers the payload; `hash` covers the entry and the chain."""

    index: int
    timestamp: str
    event: str
    digest: str
    previous: str
    hash: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Verification:
    """Whether the chain holds, and where it first stops holding."""

    intact: bool
    entries: int
    broken_at: int | None = None


def digest_of(payload: dict) -> str:
    """Sorted keys, so the same decision always digests the same regardless of dict order."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode(ENCODING)).hexdigest()


def entry_hash(index: int, timestamp: str, event: str, digest: str, previous: str) -> str:
    material = f"{index}|{timestamp}|{event}|{digest}|{previous}"
    return hashlib.sha256(material.encode(ENCODING)).hexdigest()


class AuditLog:
    """Append-only. There is deliberately no update or delete."""

    def __init__(self, path: Path | None = None) -> None:
        self._entries: list[Entry] = []
        self._path = path
        if path is not None and path.exists():
            self._entries = _load(path)

    def append(self, event: str, payload: dict) -> Entry:
        entry = Entry(
            index=len(self._entries),
            timestamp=datetime.now(UTC).isoformat(),
            event=event,
            digest=digest_of(payload),
            previous=self.head,
            hash="",
        )
        sealed = _seal(entry)
        self._entries.append(sealed)
        self._persist(sealed)
        return sealed

    @property
    def head(self) -> str:
        return self._entries[-1].hash if self._entries else GENESIS

    def entries(self, limit: int | None = None) -> list[Entry]:
        return self._entries[-limit:] if limit else list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def verify(self) -> Verification:
        """Walk the chain from genesis; report the first index that does not reconcile."""
        previous = GENESIS
        for position, entry in enumerate(self._entries):
            if entry.index != position or entry.previous != previous:
                return Verification(False, len(self._entries), position)
            if entry.hash != _seal(entry).hash:
                return Verification(False, len(self._entries), position)
            previous = entry.hash
        return Verification(True, len(self._entries))

    def _persist(self, entry: Entry) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding=ENCODING) as handle:
            handle.write(json.dumps(entry.as_dict()) + "\n")


def _seal(entry: Entry) -> Entry:
    """Recomputes the hash from the entry's own fields; used to write and to verify."""
    computed = entry_hash(entry.index, entry.timestamp, entry.event, entry.digest, entry.previous)
    return Entry(entry.index, entry.timestamp, entry.event, entry.digest, entry.previous, computed)


def _load(path: Path) -> list[Entry]:
    lines = path.read_text(encoding=ENCODING).splitlines()
    return [Entry(**json.loads(line)) for line in lines if line.strip()]
