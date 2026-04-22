"""Evaluation Layer 1: sentence-level politeness on held-out correction prompts.

For each (user, context) in the held-out test set, ask each system for a
reply, then score with:
    (a) the heuristic scorer from ``heuristic_scorer``
    (b) optionally the Stanford-trained ``politeness_classifier``

Produces a JSON file with per-example and aggregate scores.

Usage:
    python -m social_corrections.evaluation.sentence_level_eval \\
        --test-path data/processed/test_sft.jsonl \\
        --system base --model gpt-4o-mini \\
        --out data/processed/layer1_base.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..data.schema import read_jsonl
from ..inference import make_client
from ..inference.model_client import BaseModelClient
from ..utils import ensure_dir, write_json
from .heuristic_scorer import aggregate, score


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
    ap = argparse.ArgumentParser(description="Sentence-level politeness eval.")
    ap.add_argument("--system", choices=["base", "rule_based", "tinker", "hf"], required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--tinker-model-path", default=None)
    ap.add_argument("--tinker-base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--hf-model-id", default=None)
    ap.add_argument("--test-path", required=True, help="JSONL in SFT format.")
    ap.add_argument("--politeness-clf", default=None, help="Path to a trained politeness classifier.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    rows = read_jsonl(args.test_path)
    client = build_client(args)

    generations: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        messages = row["messages"]
        # Drop the final assistant turn (it's the target); we want the model's own reply.
        prompt_msgs = [m for m in messages[:-1]]
        reply = client.chat(prompt_msgs, temperature=args.temperature, max_tokens=args.max_tokens)
        gold_reply = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else ""
        h = score(reply)
        generations.append(
            {
                "index": i,
                "prompt_messages": prompt_msgs,
                "gold_reply": gold_reply,
                "model_reply": reply,
                "heuristic": h.as_dict(),
                "composite": h.composite,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"[layer1] {i + 1}/{len(rows)} done")

    # Aggregate
    agg = aggregate(g["model_reply"] for g in generations)

    # Optional classifier scores
    if args.politeness_clf:
        from .politeness_classifier import PolitenessClassifier

        clf = PolitenessClassifier.load(args.politeness_clf)
        texts = [g["model_reply"] for g in generations]
        preds = clf.predict(texts)
        for g, p in zip(generations, preds):
            g["prob_polite"] = p.prob_polite
            g["polite_label"] = p.label
        agg["mean_prob_polite"] = sum(p.prob_polite for p in preds) / max(1, len(preds))

    payload = {
        "system": args.system,
        "args": vars(args),
        "aggregate": agg,
        "generations": generations,
    }
    ensure_dir(Path(args.out).parent)
    write_json(payload, args.out)
    print(f"\nWrote {len(generations)} generations to {args.out}")
    print("Aggregate:")
    for k, v in agg.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
