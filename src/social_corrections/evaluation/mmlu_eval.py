"""MMLU subset evaluation for the no-regression check (Evaluation Layer 3).

Usage:
    python -m social_corrections.evaluation.mmlu_eval \\
        --system base --model gpt-4o-mini --n 200 \\
        --out data/processed/mmlu_base.json

We evaluate on a random subset of MMLU questions (default 200) balanced across
subjects. Each question is asked as a chat-style four-choice multiple-choice
prompt; we parse the first A/B/C/D character in the reply.
"""
from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..inference import make_client
from ..inference.model_client import BaseModelClient
from ..utils import ensure_dir, write_json

CHOICES = ["A", "B", "C", "D"]


@dataclass
class MMLUResult:
    subject: str
    question: str
    gold: str
    predicted: str
    correct: bool


def _format_prompt(question: str, choices: list[str]) -> list[dict[str, str]]:
    opts = "\n".join(f"{letter}. {c}" for letter, c in zip(CHOICES, choices))
    user = (
        "Answer the following multiple-choice question. Reply with only one "
        "letter: A, B, C, or D.\n\n"
        f"Question: {question}\n\n{opts}\n\nAnswer:"
    )
    return [
        {"role": "system", "content": "You are a careful and knowledgeable test-taker."},
        {"role": "user", "content": user},
    ]


def _parse_answer(text: str) -> str:
    """Return first A/B/C/D character; default to empty string if none found."""
    m = re.search(r"[ABCD]", text.upper())
    return m.group(0) if m else ""


def _load_mmlu(n: int, seed: int) -> list[dict[str, Any]]:
    """Load a balanced MMLU subset via the ``datasets`` library."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=seed)
    subset = ds.select(range(min(n, len(ds))))
    return [dict(x) for x in subset]


def evaluate(client: BaseModelClient, n: int, seed: int) -> list[MMLUResult]:
    random.seed(seed)
    items = _load_mmlu(n=n, seed=seed)
    results: list[MMLUResult] = []
    for i, item in enumerate(items):
        # MMLU provides integer 'answer' + choices list
        gold_idx = int(item["answer"])
        gold_letter = CHOICES[gold_idx]
        messages = _format_prompt(item["question"], item["choices"])
        reply = client.chat(messages, temperature=0.0, max_tokens=4)
        pred = _parse_answer(reply)
        results.append(
            MMLUResult(
                subject=item.get("subject", "unknown"),
                question=item["question"],
                gold=gold_letter,
                predicted=pred,
                correct=(pred == gold_letter),
            )
        )
        if (i + 1) % 25 == 0:
            acc = sum(r.correct for r in results) / len(results)
            print(f"[MMLU] {i + 1}/{len(items)} acc={acc:.3f}")
    return results


def build_client(args) -> BaseModelClient:
    if args.system == "base":
        return make_client("openai", model=args.model)
    if args.system == "rule_based":
        base = make_client("openai", model=args.model)
        return make_client("rule_based", base_client=base)
    if args.system == "tinker":
        return make_client(
            "tinker",
            model_path=args.tinker_model_path,
            model_name=args.tinker_base_model,
        )
    if args.system == "hf":
        return make_client("hf", model_id=args.hf_model_id)
    raise ValueError(f"Unknown --system {args.system!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MMLU no-regression evaluation.")
    ap.add_argument("--system", choices=["base", "rule_based", "tinker", "hf"], required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--tinker-model-path", default=None)
    ap.add_argument("--tinker-base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--hf-model-id", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client = build_client(args)
    results = evaluate(client, n=args.n, seed=args.seed)

    correct = sum(1 for r in results if r.correct)
    total = len(results)
    acc = correct / total if total else 0.0

    # Per-subject breakdown
    from collections import defaultdict
    by_subject = defaultdict(lambda: [0, 0])  # [correct, total]
    for r in results:
        by_subject[r.subject][1] += 1
        if r.correct:
            by_subject[r.subject][0] += 1

    payload = {
        "system": args.system,
        "args": vars(args),
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "by_subject": {
            s: {"correct": c, "total": t, "accuracy": c / t if t else 0.0}
            for s, (c, t) in sorted(by_subject.items())
        },
        "per_question": [
            {
                "subject": r.subject,
                "question": r.question,
                "gold": r.gold,
                "predicted": r.predicted,
                "correct": r.correct,
            }
            for r in results
        ],
    }
    ensure_dir(Path(args.out).parent)
    write_json(payload, args.out)
    print(f"\nFinal accuracy: {acc:.3f} ({correct}/{total})")
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
