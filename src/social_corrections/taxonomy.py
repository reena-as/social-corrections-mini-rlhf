"""Canonical taxonomy of social failures, derived from the midway analysis.

Every training example and every evaluation judgment uses these labels so the
per-failure-type breakdown reported in the final paper is consistent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureType:
    label: str
    short: str
    description: str


FAILURE_TYPES: dict[str, FailureType] = {
    ft.label: ft
    for ft in [
        FailureType(
            label="Overly Harsh or Judgmental Language",
            short="harsh",
            description=(
                "Assistant uses blunt negative judgments ('wrong', 'bad') where "
                "a more specific, collaborative phrasing would serve the task "
                "equally well."
            ),
        ),
        FailureType(
            label="Overconfidence / Lack of Uncertainty",
            short="overconfident",
            description=(
                "Assistant asserts a claim with unwarranted certainty and no "
                "hedge where the underlying knowledge is uncertain or "
                "context-dependent."
            ),
        ),
        FailureType(
            label="Lack of Acknowledgment",
            short="no_ack",
            description=(
                "Assistant responds to an emotionally laden or effortful user "
                "turn without acknowledging the user's perspective, effort, or "
                "confusion first."
            ),
        ),
        FailureType(
            label="Direct or Commanding Tone",
            short="commanding",
            description=(
                "Assistant issues bare imperatives ('Just redo it', 'Fix "
                "everything') rather than suggestions or collaborative framings."
            ),
        ),
        FailureType(
            label="Negative Framing Instead of Constructive Framing",
            short="negative_framing",
            description=(
                "Assistant frames feedback around what is wrong rather than "
                "what could be improved or tried next."
            ),
        ),
        FailureType(
            label="Emotional Mismatch (Too Cold or Abrupt)",
            short="cold",
            description=(
                "Assistant's register is colder or more terse than the user's "
                "turn invites, producing a dismissive feel even when factually "
                "correct."
            ),
        ),
    ]
}


ALL_LABELS: list[str] = list(FAILURE_TYPES.keys())
SHORT_TO_LABEL: dict[str, str] = {ft.short: ft.label for ft in FAILURE_TYPES.values()}


def canonicalize(label: str) -> str:
    """Map a possibly-abbreviated label onto the canonical label, or raise."""
    if label in FAILURE_TYPES:
        return label
    if label in SHORT_TO_LABEL:
        return SHORT_TO_LABEL[label]
    # Fuzzy-ish: case-insensitive exact match
    for canonical in FAILURE_TYPES:
        if canonical.lower() == label.lower():
            return canonical
    raise KeyError(f"Unknown failure type: {label!r}")
