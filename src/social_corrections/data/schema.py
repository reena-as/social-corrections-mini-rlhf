"""Core data schema used by every stage of the pipeline.

A CorrectionPair is the atomic unit: a user turn, a socially-flawed assistant
turn, a corrected assistant turn, and metadata. Both SFT and DPO datasets are
derived from these pairs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..taxonomy import canonicalize


@dataclass
class CorrectionPair:
    """A single (bad, better) correction example.

    Attributes:
        user: The user turn that elicited the assistant reply.
        bad: The socially-flawed assistant reply.
        better: The corrected assistant reply.
        failure_type: Canonical taxonomy label.
        context: Optional multi-turn dialogue context preceding ``user``. Each
            element is a dict with keys ``role`` and ``content``. This is what
            makes the SOTOPIA-harvested examples long-horizon rather than
            single-turn.
        source: Where this example came from (e.g. ``"seed"``, ``"politerewrite"``,
            ``"sotopia_harvest"``).
        pair_id: Optional stable identifier.
    """

    user: str
    bad: str
    better: str
    failure_type: str
    context: list[dict[str, str]] = field(default_factory=list)
    source: str = "seed"
    pair_id: str | None = None

    def __post_init__(self) -> None:
        self.failure_type = canonicalize(self.failure_type)
        # Light normalization: strip whitespace but preserve user typography
        self.user = self.user.strip()
        self.bad = self.bad.strip()
        self.better = self.better.strip()

    # ---- conversation helpers ----
    def prompt_messages(self, system: str | None = None) -> list[dict[str, str]]:
        """The (system + context + user) messages that precede either reply."""
        msgs: list[dict[str, str]] = []
        if system is not None:
            msgs.append({"role": "system", "content": system})
        msgs.extend(self.context)
        msgs.append({"role": "user", "content": self.user})
        return msgs

    def to_sft_example(self, system: str | None = None) -> dict[str, Any]:
        """Tinker SFT format: ``{"messages": [...]}`` with the ``better`` reply."""
        return {
            "messages": self.prompt_messages(system) + [
                {"role": "assistant", "content": self.better}
            ]
        }

    def to_dpo_example(self, system: str | None = None) -> dict[str, Any]:
        """DPO-style preference record.

        We use a simple, portable JSON format:
            {
              "prompt_messages": [...],
              "chosen": "...",
              "rejected": "...",
              "failure_type": "...",
              "source": "...",
            }
        The training script converts this into Tinker's ``LabeledComparison``
        at runtime.
        """
        return {
            "prompt_messages": self.prompt_messages(system),
            "chosen": self.better,
            "rejected": self.bad,
            "failure_type": self.failure_type,
            "source": self.source,
            "pair_id": self.pair_id,
        }


# ---- serialization ----

def load_correction_pairs(path: str | Path) -> list[CorrectionPair]:
    """Load a JSON array of correction pairs.

    Accepts the legacy midway schema where the "better" field is named
    ``"better"`` OR ``"target_better"``.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: list[CorrectionPair] = []
    for i, ex in enumerate(raw):
        better = ex.get("better") or ex.get("target_better")
        if better is None:
            raise ValueError(f"Row {i} missing 'better'/'target_better': {ex}")
        out.append(
            CorrectionPair(
                user=ex["user"],
                bad=ex["bad"],
                better=better,
                failure_type=ex["failure_type"],
                context=ex.get("context", []),
                source=ex.get("source", "seed"),
                pair_id=ex.get("pair_id") or f"seed-{i:04d}",
            )
        )
    return out


def dump_correction_pairs(pairs: Iterable[CorrectionPair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in pairs], f, ensure_ascii=False, indent=2)


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Write an iterable of JSON-serializable records to a JSONL file. Returns count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
