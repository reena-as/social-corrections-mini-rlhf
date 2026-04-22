#!/usr/bin/env python
"""Produce comparison plots across systems from eval JSON outputs.

Usage:
    python scripts/plot_results.py \\
        --base-dir data/processed/eval_base \\
        --rule-dir data/processed/eval_rule_based \\
        --sft-dir data/processed/eval_sft \\
        --dpo-dir data/processed/eval_dpo \\
        --out-dir data/processed/plots

Each --*-dir should contain layer1.json, layer2_sotopia.json, and layer3_mmlu.json
produced by run_all_eval.py. Missing files are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEMS = ["base", "rule_based", "sft", "dpo"]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _plot_bars(labels, values, title, ylabel, out_path, ylim=None, horizontal_zero=False):
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if horizontal_zero:
        ax.axhline(0, color="black", linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--rule-dir", default=None)
    ap.add_argument("--sft-dir", default=None)
    ap.add_argument("--dpo-dir", default=None)
    ap.add_argument("--out-dir", default="data/processed/plots")
    args = ap.parse_args()

    dirs = {
        "base": args.base_dir,
        "rule_based": args.rule_dir,
        "sft": args.sft_dir,
        "dpo": args.dpo_dir,
    }
    out_dir = Path(args.out_dir)

    # Layer 1: composite heuristic score
    labels, values = [], []
    for s in SYSTEMS:
        d = dirs.get(s)
        if not d:
            continue
        data = _load(Path(d) / "layer1.json")
        if not data:
            continue
        labels.append(s)
        values.append(float(data["aggregate"].get("composite", 0.0)))
    if labels:
        _plot_bars(
            labels, values,
            "Layer 1: heuristic composite politeness score",
            "composite (higher is better)",
            out_dir / "layer1_composite.png",
        )

    # Layer 2: per-dimension SOTOPIA scores
    rows: dict[str, dict[str, float]] = {}
    for s in SYSTEMS:
        d = dirs.get(s)
        if not d:
            continue
        data = _load(Path(d) / "layer2_sotopia.json")
        if not data:
            continue
        rows[s] = {k: v for k, v in data["aggregate"].items() if isinstance(v, (int, float))}
    if rows:
        for dim in ["goal", "social_rules", "relationship", "believability"]:
            labels = [s for s in SYSTEMS if s in rows and dim in rows[s]]
            values = [rows[s][dim] for s in labels]
            if labels:
                _plot_bars(
                    labels, values,
                    f"Layer 2: SOTOPIA {dim}",
                    dim,
                    out_dir / f"layer2_{dim}.png",
                    horizontal_zero=(dim in ("social_rules", "relationship")),
                )

    # Layer 3: MMLU accuracy
    labels, values = [], []
    for s in SYSTEMS:
        d = dirs.get(s)
        if not d:
            continue
        data = _load(Path(d) / "layer3_mmlu.json")
        if not data:
            continue
        labels.append(s)
        values.append(float(data.get("accuracy", 0.0)))
    if labels:
        _plot_bars(
            labels, values,
            "Layer 3: MMLU accuracy (no-regression check)",
            "accuracy",
            out_dir / "layer3_mmlu.png",
            ylim=(0, 1),
        )

    print(f"\nAll plots written under {out_dir}/")


if __name__ == "__main__":
    main()
