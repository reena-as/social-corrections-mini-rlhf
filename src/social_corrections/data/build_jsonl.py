"""Convert CorrectionPairs into Tinker-ready SFT and DPO JSONL files.

Produces six files under data/processed/:
    - train_sft.jsonl, val_sft.jsonl, test_sft.jsonl
    - train_dpo.jsonl, val_dpo.jsonl, test_dpo.jsonl

Split: a deterministic 80/10/10 by hash of pair_id, so the same example always
lands in the same split across reruns. This matters for DPO training that
loads checkpoints from SFT runs.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ..utils import data_dir, ensure_dir
from .schema import CorrectionPair, load_correction_pairs, write_jsonl

DEFAULT_SYSTEM = (
    "You are a helpful, socially aware assistant. You pursue the user's goal "
    "while remaining honest, calibrated, and considerate of the user's "
    "perspective and effort."
)


def _bucket(pair_id: str) -> str:
    """Deterministic 80/10/10 split based on md5(pair_id)."""
    h = int(hashlib.md5(pair_id.encode("utf-8")).hexdigest(), 16) % 10
    if h < 8:
        return "train"
    if h < 9:
        return "val"
    return "test"


def build(
    pairs: list[CorrectionPair],
    out_dir: Path,
    system_prompt: str = DEFAULT_SYSTEM,
) -> dict[str, int]:
    """Write SFT + DPO JSONL files; returns per-file counts."""
    ensure_dir(out_dir)

    buckets: dict[str, list[CorrectionPair]] = {"train": [], "val": [], "test": []}
    for i, p in enumerate(pairs):
        pid = p.pair_id or f"unknown-{i:06d}"
        buckets[_bucket(pid)].append(p)

    counts: dict[str, int] = {}
    for split, split_pairs in buckets.items():
        sft_path = out_dir / f"{split}_sft.jsonl"
        dpo_path = out_dir / f"{split}_dpo.jsonl"
        counts[f"{split}_sft"] = write_jsonl(
            (p.to_sft_example(system=system_prompt) for p in split_pairs), sft_path
        )
        counts[f"{split}_dpo"] = write_jsonl(
            (p.to_dpo_example(system=system_prompt) for p in split_pairs), dpo_path
        )
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT and DPO JSONL from correction pairs.")
    ap.add_argument(
        "--input",
        default=str(data_dir() / "correction_dataset.json"),
        help="Path to a JSON array of correction pairs.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(data_dir() / "processed"),
        help="Output directory for JSONL files.",
    )
    ap.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="System prompt prepended to every example.",
    )
    args = ap.parse_args()

    pairs = load_correction_pairs(args.input)
    counts = build(pairs, Path(args.out_dir), system_prompt=args.system)
    print(f"Loaded {len(pairs)} correction pairs from {args.input}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
