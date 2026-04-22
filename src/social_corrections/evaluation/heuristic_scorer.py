"""Lightweight heuristic scorer for social appropriateness.

Refactored from the midway notebook into an importable module. Counts four
binary dimensions (acknowledgment, constructive, hedging, contains_harsh_word)
plus an awkwardness flag. These are cheap proxies that are useful as fast
sanity checks but should NOT be the only signal reported in the paper -- use
the Stanford-Politeness classifier and SOTOPIA-Eval for the main numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

ACK_WORDS: tuple[str, ...] = (
    "i see", "i understand", "thanks for", "you're on the right track",
    "let's", "of course",
)

CONSTRUCTIVE_WORDS: tuple[str, ...] = (
    "improve", "together", "try", "consider", "help", "go through",
    "take a closer look", "work through", "step by step",
)

HEDGE_WORDS: tuple[str, ...] = (
    "might", "may", "could", "likely", "appears", "seems",
)

HARSH_WORDS: tuple[str, ...] = (
    "wrong", "bad", "stupid", "dumb", "horrible",
)

AWKWARD_PATTERNS: tuple[str, ...] = (
    "the a ",
    "may not be correct every time",
    "it’s simple, you’re overthinking it",
)


@dataclass
class HeuristicScores:
    acknowledgment: int
    constructive: int
    hedging: int
    contains_harsh_word: int
    awkward_or_limited: int

    def as_dict(self) -> dict[str, int]:
        return {
            "acknowledgment": self.acknowledgment,
            "constructive": self.constructive,
            "hedging": self.hedging,
            "contains_harsh_word": self.contains_harsh_word,
            "awkward_or_limited": self.awkward_or_limited,
        }

    @property
    def composite(self) -> int:
        """A toy composite score: +1 per positive, -1 per negative dimension."""
        return (
            self.acknowledgment + self.constructive + self.hedging
            - self.contains_harsh_word - self.awkward_or_limited
        )


def score(text: str) -> HeuristicScores:
    t = text.lower()
    return HeuristicScores(
        acknowledgment=int(any(w in t for w in ACK_WORDS)),
        constructive=int(any(w in t for w in CONSTRUCTIVE_WORDS)),
        hedging=int(any(w in t for w in HEDGE_WORDS)),
        contains_harsh_word=int(any(w in t for w in HARSH_WORDS)),
        awkward_or_limited=int(any(p in t for p in AWKWARD_PATTERNS)),
    )


def aggregate(texts: Iterable[str]) -> dict[str, float]:
    """Return per-dimension means + composite mean over an iterable of texts."""
    rows = [score(t) for t in texts]
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "acknowledgment": 0.0,
            "constructive": 0.0,
            "hedging": 0.0,
            "contains_harsh_word": 0.0,
            "awkward_or_limited": 0.0,
            "composite": 0.0,
        }
    return {
        "n": n,
        "acknowledgment": sum(r.acknowledgment for r in rows) / n,
        "constructive": sum(r.constructive for r in rows) / n,
        "hedging": sum(r.hedging for r in rows) / n,
        "contains_harsh_word": sum(r.contains_harsh_word for r in rows) / n,
        "awkward_or_limited": sum(r.awkward_or_limited for r in rows) / n,
        "composite": sum(r.composite for r in rows) / n,
    }
