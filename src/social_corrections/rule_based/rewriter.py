"""Deterministic rule-based social post-processor.

This is the ``System B`` baseline from the revised plan. It is a direct
refactor of the midway notebook into a testable module. The rewriter operates
purely on the assistant's reply text; it does not look at dialogue context.

Determinism: in the midway notebook the acknowledgment phrase was chosen with
``random.choice``, which made the output non-reproducible. Here we accept a
``seed`` and pick deterministically.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

ACKNOWLEDGMENT_PHRASES: tuple[str, ...] = (
    "I see what you mean.",
    "I understand why that might be frustrating.",
    "Thanks for pointing that out.",
    "I see your reasoning.",
    "I understand why that could be confusing.",
)

HEDGING_REPLACEMENTS: dict[str, str] = {
    "definitely": "likely",
    "obviously": "it seems",
    "clearly": "it appears",
    "always": "often",
    "never": "may not",
    "best way": "a good way",
}

HARSH_REPLACEMENTS: dict[str, str] = {
    "wrong": "may not be correct",
    "bad": "could be improved",
    "dumb": "not very effective",
    "horrible": "could be improved",
    "stupid": "not a strong approach",
}

# Exact-string replacements for the most common blunt replies. These are curated
# to stay idiomatic; the generic word-replacement rules often produce awkward
# output on these (e.g. "That's may not be correct.").
FULL_SENTENCE_REPLACEMENTS: dict[str, str] = {
    "That’s wrong.": "I see what you’re trying to do, but there may be an issue here.",
    "That's wrong.": "I see what you’re trying to do, but there may be an issue here.",
    "Your code is bad.": "I think there might be a small issue in your code. Let’s take a closer look.",
    "It doesn’t.": "I see your reasoning, but there may be a small mistake.",
    "It doesn't.": "I see your reasoning, but there may be a small mistake.",
    "You didn’t.": "You’re on the right track, but there may be a small error.",
    "You didn't.": "You’re on the right track, but there may be a small error.",
    "Just redo it.": "You might want to revise it and try again step by step.",
    "Fix everything.": "There are a few important areas to improve, and we can work through them together.",
    "Just read the instructions.": "Let’s go through the instructions together and break them down.",
    "I already explained it.": "Of course — let me explain it again in a different way.",
    "That’s not a good idea.": "I see what you’re aiming for, though this approach may have some challenges.",
    "That's not a good idea.": "I see what you’re aiming for, though this approach may have some challenges.",
}

ABRUPT_STARTS: tuple[str, ...] = (
    "that’s", "that's", "you didn’t", "you didn't", "it doesn’t", "it doesn't",
    "just", "no", "wrong", "bad",
)


def _replace_case_insensitive(text: str, old: str, new: str) -> str:
    pattern = re.compile(rf"\b{re.escape(old)}\b", re.IGNORECASE)
    return pattern.sub(new, text)


def _starts_abruptly(text: str) -> bool:
    lower = text.strip().lower()
    return any(lower.startswith(start) for start in ABRUPT_STARTS)


@dataclass
class RuleBasedRewriter:
    """Applies the politeness rules in a deterministic order.

    The order is deliberate: full-sentence lookup first (highest-signal), then
    harsh-word softening, then hedging, then acknowledgment prepending, then
    collaborative reframing for imperatives.
    """

    acknowledgment_phrases: tuple[str, ...] = field(default_factory=lambda: ACKNOWLEDGMENT_PHRASES)
    hedging: dict[str, str] = field(default_factory=lambda: dict(HEDGING_REPLACEMENTS))
    harsh: dict[str, str] = field(default_factory=lambda: dict(HARSH_REPLACEMENTS))
    full_sentences: dict[str, str] = field(default_factory=lambda: dict(FULL_SENTENCE_REPLACEMENTS))
    seed: int = 0

    def __call__(self, text: str) -> str:
        rewritten = text.strip()

        # 1. Full-sentence lookup
        if rewritten in self.full_sentences:
            return self.full_sentences[rewritten]

        # 2. Harsh word softening
        for harsh_word, soft_word in self.harsh.items():
            rewritten = _replace_case_insensitive(rewritten, harsh_word, soft_word)

        # 3. Hedging of overconfident terms
        for strong_word, hedged_word in self.hedging.items():
            rewritten = _replace_case_insensitive(rewritten, strong_word, hedged_word)

        # 4. Prepend acknowledgment to short/abrupt replies
        if len(rewritten.split()) <= 6 or _starts_abruptly(rewritten):
            # Deterministic pick: combine seed with a stable hash of the input
            import hashlib
            h = hashlib.md5(rewritten.encode("utf-8")).digest()
            seed_int = (self.seed ^ int.from_bytes(h[:8], "big")) & 0x7FFFFFFFFFFFFFFF
            rng = random.Random(seed_int)
            ack = rng.choice(self.acknowledgment_phrases)
            rewritten = f"{ack} {rewritten}"

        # 5. Collaborative reframing of bare imperatives
        lower = rewritten.lower()
        if lower.startswith("you should"):
            rewritten = re.sub(r"^[Yy]ou should", "You might consider", rewritten)
        elif lower.startswith("do this"):
            rewritten = re.sub(r"^[Dd]o this", "You could try this", rewritten)

        return rewritten


# Module-level convenience for the notebook-style callers
_default_rewriter = RuleBasedRewriter()


def rewrite_response(text: str) -> str:
    """Apply the default rule-based rewriter."""
    return _default_rewriter(text)
