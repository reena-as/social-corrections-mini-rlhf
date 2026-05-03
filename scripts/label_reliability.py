#!/usr/bin/env python
"""Assess GPT-4o vs. human agreement on correction-pair failure_type labels.

Stratified-samples up to --n-samples pairs from the correction dataset,
asks GPT-4o to independently classify each pair's failure type using the
canonical taxonomy, and writes a per-pair agreement report plus summary
statistics (overall agreement %, confusion matrix, per-type precision).

Usage:
    python scripts/label_reliability.py \\
        --n-samples 30 \\
        --out data/processed/label_reliability.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from typing import Any

from social_corrections.inference.model_client import OpenAIModelClient
from social_corrections.taxonomy import ALL_LABELS, FAILURE_TYPES, canonicalize
from social_corrections.utils.io import data_dir, write_json

_SYSTEM_TEMPLATE = (
    "You are an expert at identifying social communication failures in AI assistant responses.\n"
    "Given a user message, a BAD assistant response, and an IMPROVED response, classify the "
    "failure type of the bad response using exactly one label from the taxonomy below.\n\n"
    "Taxonomy:\n{taxonomy}\n\n"
    'Respond with a JSON object: {{"label": "<exact label from taxonomy>", "rationale": "<one sentence>"}}\n'
    "No prose outside JSON."
)


def _taxonomy_block() -> str:
    return "\n".join(
        f'- "{label}": {FAILURE_TYPES[label].description}' for label in ALL_LABELS
    )


def _classify(
    client: OpenAIModelClient,
    user: str,
    bad: str,
    better: str,
    taxonomy_block: str,
) -> tuple[str, str]:
    system = _SYSTEM_TEMPLATE.format(taxonomy=taxonomy_block)
    user_msg = (
        f"User message:\n{user}\n\n"
        f"Bad response:\n{bad}\n\n"
        f"Improved response:\n{better}\n\n"
        "Which failure type best describes the bad response?"
    )
    raw = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=200,
    )
    txt = raw.strip()
    if txt.startswith("```"):
        lines = txt.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()
    data = json.loads(txt)
    raw_label = str(data.get("label", ""))
    rationale = str(data.get("rationale", ""))
    try:
        canonical = canonicalize(raw_label)
    except KeyError:
        canonical = raw_label
    return canonical, rationale


def _stratified_sample(pairs: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_type: dict[str, list] = defaultdict(list)
    for p in pairs:
        by_type[p["failure_type"]].append(p)

    per_type = max(1, n // len(by_type))
    sampled: list[dict] = []
    for ft in sorted(by_type):
        pool = by_type[ft]
        sampled.extend(rng.sample(pool, min(per_type, len(pool))))

    if len(sampled) < n:
        taken = {p["pair_id"] for p in sampled}
        rest = [p for p in pairs if p["pair_id"] not in taken]
        sampled.extend(rng.sample(rest, min(n - len(sampled), len(rest))))

    return sampled[:n]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assess GPT-4o vs. human agreement on correction-pair failure_type labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--pairs-path",
        default=str(data_dir() / "processed" / "correction_pairs_all.json"),
        help="Path to correction_pairs_all.json.",
    )
    ap.add_argument("--n-samples", type=int, default=30,
                    help="Max pairs to score (stratified by failure_type).")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--out", default="data/processed/label_reliability.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.pairs_path, encoding="utf-8") as f:
        pairs = json.load(f)

    sampled = _stratified_sample(pairs, args.n_samples, args.seed)
    n_types = len({p["failure_type"] for p in sampled})
    print(f"Sampled {len(sampled)} pairs (stratified across {n_types} failure types).\n")

    taxonomy_block = _taxonomy_block()
    client = OpenAIModelClient(model=args.judge_model)

    records: list[dict[str, Any]] = []
    for i, pair in enumerate(sampled):
        orig = pair["failure_type"]
        print(f"[{i+1}/{len(sampled)}] {pair['pair_id']} ...", end=" ", flush=True)
        try:
            predicted, rationale = _classify(
                client, pair["user"], pair["bad"], pair["better"], taxonomy_block
            )
            agreed = predicted == orig
            marker = "+" if agreed else "-"
            print(f"[{marker}] {predicted!r}")
        except Exception as exc:
            print(f"ERROR: {exc}")
            predicted, rationale, agreed = "", "", False

        records.append({
            "pair_id": pair["pair_id"],
            "original_label": orig,
            "gpt4o_label": predicted,
            "agreed": agreed,
            "gpt4o_rationale": rationale,
        })

    total = len(records)
    n_agreed = sum(1 for r in records if r["agreed"])
    overall_pct = n_agreed / total if total else 0.0

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        confusion[r["original_label"]][r["gpt4o_label"]] += 1

    # Precision: when GPT-4o predicts label X, how often is the true label also X?
    per_type_precision: dict[str, float | None] = {}
    for label in ALL_LABELS:
        tp = sum(1 for r in records if r["gpt4o_label"] == label and r["original_label"] == label)
        fp = sum(1 for r in records if r["gpt4o_label"] == label and r["original_label"] != label)
        per_type_precision[label] = tp / (tp + fp) if (tp + fp) > 0 else None

    print(f"\nOverall agreement: {overall_pct:.1%}  ({n_agreed}/{total})")
    print("\nPer-failure-type precision:")
    for label, prec in per_type_precision.items():
        prec_str = f"{prec:.1%}" if prec is not None else "n/a"
        print(f"  {label}: {prec_str}")

    write_json(
        {
            "judge_model": args.judge_model,
            "n_sampled": total,
            "per_pair": records,
            "summary": {
                "overall_agreement": overall_pct,
                "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
                "per_failure_type_precision": per_type_precision,
            },
        },
        args.out,
    )
    print(f"\nWrote results → {args.out}")


if __name__ == "__main__":
    main()
