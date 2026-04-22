"""Merge correction-pair sources into a single curated dataset.

Inputs: any number of JSON files containing arrays of correction pairs.
Output: ``data/processed/correction_pairs_all.json``, with duplicates removed
and a per-source + per-failure-type summary printed.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ..utils import data_dir, ensure_dir
from .schema import CorrectionPair, dump_correction_pairs, load_correction_pairs


def _dedupe_key(p: CorrectionPair) -> str:
    h = hashlib.md5()
    h.update(p.user.encode("utf-8"))
    h.update(b"||")
    h.update(p.bad.encode("utf-8"))
    h.update(b"||")
    h.update(p.better.encode("utf-8"))
    return h.hexdigest()


def merge(inputs: list[Path]) -> list[CorrectionPair]:
    seen: set[str] = set()
    out: list[CorrectionPair] = []
    for path in inputs:
        if not path.exists():
            print(f"[merge] Skipping missing input: {path}")
            continue
        pairs = load_correction_pairs(path)
        kept = 0
        for p in pairs:
            k = _dedupe_key(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
            kept += 1
        print(f"[merge] {path}: loaded {len(pairs)}, kept {kept} after dedupe")
    return out


def summarize(pairs: list[CorrectionPair]) -> None:
    from collections import Counter

    by_source = Counter(p.source for p in pairs)
    by_ft = Counter(p.failure_type for p in pairs)
    print(f"\nTotal: {len(pairs)} correction pairs")
    print("\nBy source:")
    for k, v in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print("\nBy failure type:")
    for k, v in sorted(by_ft.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge correction-pair datasets.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            str(data_dir() / "correction_dataset.json"),
            str(data_dir() / "processed" / "politerewrite_adapted.json"),
            str(data_dir() / "processed" / "sotopia_harvested_pairs.json"),
        ],
    )
    ap.add_argument(
        "--out",
        default=str(data_dir() / "processed" / "correction_pairs_all.json"),
    )
    args = ap.parse_args()

    pairs = merge([Path(p) for p in args.inputs])
    ensure_dir(Path(args.out).parent)
    dump_correction_pairs(pairs, args.out)
    print(f"\nWrote merged dataset to {args.out}")
    summarize(pairs)


if __name__ == "__main__":
    main()
