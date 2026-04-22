"""Adapt PoliteRewrite-style data into our CorrectionPair schema.

PoliteRewrite is a dataset of (impolite, polite) sentence pairs. We:
    1. Load from HuggingFace if available, else from a local JSON/JSONL.
    2. Filter to assistant-shaped examples (short, imperative or declarative,
       second-person, not first-person venting).
    3. Pair each with a synthetic user turn drawn from a small template set, so
       the result fits the CorrectionPair schema.
    4. Assign a ``failure_type`` heuristically.

This is deliberately a WEAK SUPERVISION SOURCE: the labels are noisy and the
user turns are synthetic. We rely on the SOTOPIA-harvested data for harder,
in-distribution examples. PoliteRewrite provides volume, SOTOPIA provides
quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from ..taxonomy import ALL_LABELS
from ..utils import data_dir, ensure_dir
from .schema import CorrectionPair, dump_correction_pairs

# Synthetic user turns that plausibly elicit short, blunt assistant replies.
# These are intentionally generic so the rewrite's meaning is preserved.
SYNTHETIC_USER_TURNS = [
    "Can you take a look at my work?",
    "Is this right?",
    "What do you think?",
    "Does this make sense?",
    "Can you help me understand this?",
    "I'm not sure if this is correct.",
    "What should I do next?",
    "Can you review this for me?",
    "Is this a good approach?",
    "Did I do this correctly?",
]


def _looks_assistant_shaped(text: str) -> bool:
    """Heuristic filter. Keep short, direct, second-person-or-imperative replies."""
    t = text.strip()
    if not t or len(t) > 240:
        return False
    if len(t.split()) > 40:
        return False
    # Drop first-person venting
    lower = t.lower()
    if lower.startswith(("i hate", "i think you", "i love", "i feel", "i'm so")):
        return False
    # Must end with a sentence-ending punctuation
    if t[-1] not in ".!?":
        return False
    return True


def _guess_failure_type(impolite: str) -> str:
    lower = impolite.lower()
    if any(w in lower for w in ("wrong", "bad", "stupid", "dumb", "awful")):
        return "Overly Harsh or Judgmental Language"
    if any(w in lower for w in ("definitely", "obviously", "clearly", "always", "never")):
        return "Overconfidence / Lack of Uncertainty"
    if lower.startswith(("just ", "do ", "fix ", "stop ", "go ")):
        return "Direct or Commanding Tone"
    if any(w in lower for w in ("not ", "don't", "doesn't", "can't")):
        return "Negative Framing Instead of Constructive Framing"
    # Default
    return "Emotional Mismatch (Too Cold or Abrupt)"


def adapt_pairs(
    rows: Iterable[dict[str, str]],
    max_examples: int | None = None,
) -> list[CorrectionPair]:
    """Convert raw (impolite, polite) rows into CorrectionPairs."""
    out: list[CorrectionPair] = []
    for i, row in enumerate(rows):
        impolite = (row.get("impolite") or row.get("source") or row.get("bad") or "").strip()
        polite = (row.get("polite") or row.get("target") or row.get("better") or "").strip()
        if not impolite or not polite:
            continue
        if not _looks_assistant_shaped(impolite):
            continue
        if not _looks_assistant_shaped(polite):
            continue
        # Pick a user turn deterministically by hashing the impolite text
        idx = int(hashlib.md5(impolite.encode("utf-8")).hexdigest(), 16) % len(
            SYNTHETIC_USER_TURNS
        )
        user = SYNTHETIC_USER_TURNS[idx]
        ft = _guess_failure_type(impolite)
        assert ft in ALL_LABELS
        pid = f"politerewrite-{hashlib.md5(impolite.encode('utf-8')).hexdigest()[:10]}"
        out.append(
            CorrectionPair(
                user=user,
                bad=impolite,
                better=polite,
                failure_type=ft,
                source="politerewrite",
                pair_id=pid,
            )
        )
        if max_examples is not None and len(out) >= max_examples:
            break
    return out


def load_local(path: Path) -> list[dict[str, str]]:
    """Load a local PoliteRewrite dump (JSON array or JSONL).

    Expected keys per row: some subset of {impolite, polite, source, target, bad, better}.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    # Treat as JSONL
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def try_load_hf(split: str = "train", max_examples: int = 2000) -> list[dict[str, str]] | None:
    """Best-effort HF load. Returns None if `datasets` is missing or the dataset
    identifier doesn't resolve -- callers should then fall back to `load_local`.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return None
    candidate_ids = [
        "WYLing/PoliteRewrite",
        "cmu-lti/polite-rewrite",
        "politerewrite/politerewrite",
    ]
    for ds_id in candidate_ids:
        try:
            ds = load_dataset(ds_id, split=split)  # noqa
        except Exception:
            continue
        rows: list[dict[str, str]] = []
        for i, ex in enumerate(ds):
            rows.append(dict(ex))
            if i + 1 >= max_examples:
                break
        return rows
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Adapt PoliteRewrite data to CorrectionPairs.")
    ap.add_argument("--local", help="Path to a local PoliteRewrite JSON/JSONL dump.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-raw", type=int, default=2000, help="Max raw rows to load.")
    ap.add_argument(
        "--max-examples", type=int, default=500, help="Max adapted CorrectionPairs to emit."
    )
    ap.add_argument(
        "--out",
        default=str(data_dir() / "processed" / "politerewrite_adapted.json"),
        help="Output path for adapted CorrectionPairs.",
    )
    args = ap.parse_args()

    rows: list[dict[str, str]] | None
    if args.local:
        rows = load_local(Path(args.local))
    else:
        rows = try_load_hf(split=args.split, max_examples=args.max_raw)

    if not rows:
        raise SystemExit(
            "Could not load PoliteRewrite data. Pass --local <path> to a JSON/JSONL "
            "file with fields (impolite, polite) or similar."
        )

    pairs = adapt_pairs(rows, max_examples=args.max_examples)
    ensure_dir(Path(args.out).parent)
    dump_correction_pairs(pairs, args.out)
    print(f"Wrote {len(pairs)} adapted pairs to {args.out}")


if __name__ == "__main__":
    main()
