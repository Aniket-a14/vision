"""Serving layer: scoring, prescription, a live SSE feed and a hash-chained audit log."""

from .audit import AuditLog, Entry, Verification, digest_of, entry_hash
from .scoring import Scorer, build

__all__ = [
    "AuditLog",
    "Entry",
    "Scorer",
    "Verification",
    "build",
    "digest_of",
    "entry_hash",
]
