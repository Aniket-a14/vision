"""Reason codes for an operator override.

A free-text box produces a log nobody can aggregate. A fixed vocabulary produces one you can
count, so "the model is wrong about tooling marks" becomes a number rather than an anecdote --
and that number is what tells you which failure to fix next.

`OTHER` exists because a closed vocabulary that cannot express the truth gets filled in with the
nearest wrong code, which is worse than an escape hatch. It carries a mandatory note, and a
rising share of `OTHER` is itself the signal that the vocabulary needs extending.
"""

from __future__ import annotations

from enum import StrEnum


class Reason(StrEnum):
    """Why an operator disagreed with the gate."""

    KNOWN_TOOLING_MARK = "known_tooling_mark"
    COSMETIC_ONLY = "cosmetic_only"
    VISUAL_CONFIRMATION = "visual_confirmation"
    DOWNSTREAM_REWORK = "downstream_rework"
    SAMPLE_OR_TRIAL_PART = "sample_or_trial_part"
    PROCESS_CHANGE_KNOWN = "process_change_known"
    OTHER = "other"


LABELS: dict[Reason, str] = {
    Reason.KNOWN_TOOLING_MARK: "Known tooling mark, not a defect",
    Reason.COSMETIC_ONLY: "Cosmetic only, within spec",
    Reason.VISUAL_CONFIRMATION: "Visual inspection disagrees",
    Reason.DOWNSTREAM_REWORK: "Will be reworked downstream",
    Reason.SAMPLE_OR_TRIAL_PART: "Sample or trial part",
    Reason.PROCESS_CHANGE_KNOWN: "Known process change not in the model",
    Reason.OTHER: "Other (note required)",
}


def catalogue() -> list[dict[str, object]]:
    """The vocabulary the UI renders, served rather than hard-coded in two places."""
    return [
        {
            "code": reason.value,
            "label": LABELS[reason],
            "note_required": reason is Reason.OTHER,
        }
        for reason in Reason
    ]
