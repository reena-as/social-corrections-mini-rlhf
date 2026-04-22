#!/usr/bin/env python
"""Run all three evaluation layers for one system.

Usage:
    python scripts/run_all_eval.py --system base --model gpt-4o-mini \\
        --out-dir data/processed/eval_base/

This is just a convenience wrapper that calls the three eval entry points in
sequence so you don't have to remember the flags. For production runs you
should call each one individually so you can parallelize and checkpoint.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(mod: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", mod] + args
    print(f"\n$ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["base", "rule_based", "tinker", "hf"], required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--tinker-model-path", default=None)
    ap.add_argument("--tinker-base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--hf-model-id", default=None)

    ap.add_argument("--partner-model", default="gpt-4o")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--episodes-per-scenario", type=int, default=3)
    ap.add_argument("--mmlu-n", type=int, default=100)
    ap.add_argument("--politeness-clf", default=None)
    ap.add_argument("--test-sft-path", default="data/processed/test_sft.jsonl")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = ["--system", args.system, "--model", args.model]
    if args.tinker_model_path:
        common += ["--tinker-model-path", args.tinker_model_path]
    if args.tinker_base_model:
        common += ["--tinker-base-model", args.tinker_base_model]
    if args.hf_model_id:
        common += ["--hf-model-id", args.hf_model_id]

    # Layer 1
    l1 = ["--test-path", args.test_sft_path, "--out", str(out_dir / "layer1.json")]
    if args.politeness_clf:
        l1 += ["--politeness-clf", args.politeness_clf]
    _run("social_corrections.evaluation.sentence_level_eval", common + l1)

    # Layer 2
    _run(
        "social_corrections.evaluation.sotopia_eval",
        common
        + [
            "--partner-model", args.partner_model,
            "--judge-model", args.judge_model,
            "--episodes-per-scenario", str(args.episodes_per_scenario),
            "--out", str(out_dir / "layer2_sotopia.json"),
        ],
    )

    # Layer 3
    _run(
        "social_corrections.evaluation.mmlu_eval",
        common + ["--n", str(args.mmlu_n), "--out", str(out_dir / "layer3_mmlu.json")],
    )

    print(f"\nAll evaluations written to {out_dir}/")


if __name__ == "__main__":
    main()
