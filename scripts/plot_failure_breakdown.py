#!/usr/bin/env python
"""Grouped bar chart of goal and relationship scores by failure type × system.

Reads aggregate from failure_type_breakdown.json and produces a two-panel
figure saved to data/processed/final_paper/figures/failure_type_breakdown.png.

Usage:
    python scripts/plot_failure_breakdown.py \\
        --in data/processed/failure_type_breakdown.json \\
        --out data/processed/final_paper/figures/failure_type_breakdown.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from social_corrections.taxonomy import FAILURE_TYPES
from social_corrections.utils.io import data_dir

_DIMS = ("goal", "relationship")
_SYSTEM_ORDER = ["base", "rule_based", "sft", "dpo"]
_SYSTEM_COLORS = {
    "base": "#4C72B0",
    "rule_based": "#DD8452",
    "sft": "#55A868",
    "dpo": "#C44E52",
}
_DIM_YLIM = {
    "goal": (0.0, 10.0),
    "relationship": (-5.0, 5.0),
}


def _short(label: str) -> str:
    return FAILURE_TYPES[label].short if label in FAILURE_TYPES else label


def _plot(
    aggregate: dict[str, dict[str, dict[str, Any]]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    # Ordered failure types: alphabetical by short label, "none" last
    all_fts = sorted(
        [ft for ft in aggregate if ft != "none"],
        key=_short,
    )
    if "none" in aggregate:
        all_fts.append("none")

    # Systems present in any bucket, in canonical order
    all_sys_in_data = {sys for ft in all_fts for sys in aggregate[ft]}
    systems = [s for s in _SYSTEM_ORDER if s in all_sys_in_data]
    # Append any system not in canonical order
    systems += sorted(all_sys_in_data - set(systems))

    n_fts = len(all_fts)
    n_sys = len(systems)
    bar_width = 0.7 / max(n_sys, 1)
    x_positions = list(range(n_fts))
    offsets = [((si - (n_sys - 1) / 2) * bar_width) for si in range(n_sys)]

    fig_w = max(9, n_fts * (n_sys * bar_width + 0.5))
    fig, axes = plt.subplots(len(_DIMS), 1, figsize=(fig_w, 4.5 * len(_DIMS)))

    for ax, dim in zip(axes, _DIMS):
        for si, sys in enumerate(systems):
            vals = [
                aggregate[ft].get(sys, {}).get(dim, float("nan"))
                for ft in all_fts
            ]
            xs = [x + offsets[si] for x in x_positions]
            bars = ax.bar(
                xs,
                [0 if math.isnan(v) else v for v in vals],
                width=bar_width * 0.92,
                label=sys,
                color=_SYSTEM_COLORS.get(sys, None),
                alpha=0.85,
            )
            for bar, v in zip(bars, vals):
                if not math.isnan(v):
                    label_y = bar.get_height() + (0.08 if v >= 0 else -0.25)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        label_y,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_title(f"SOTOPIA {dim} by dominant failure type", fontsize=11)
        ax.set_ylabel(dim)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [_short(ft) for ft in all_fts],
            rotation=30,
            ha="right",
            fontsize=9,
        )
        ax.set_ylim(*_DIM_YLIM.get(dim, (None, None)))
        if _DIM_YLIM.get(dim, (0, 1))[0] < 0:
            ax.axhline(0, color="black", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot grouped bar charts of goal and relationship by failure type × system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--in",
        dest="input",
        default="data/processed/failure_type_breakdown.json",
        help="Path to failure_type_breakdown.json produced by failure_type_breakdown.py.",
    )
    ap.add_argument(
        "--out",
        default=str(
            data_dir() / "processed" / "final_paper" / "figures" / "failure_type_breakdown.png"
        ),
    )
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    aggregate = data.get("aggregate")
    if not aggregate:
        print("Error: 'aggregate' key missing from input file.")
        raise SystemExit(1)

    _plot(aggregate, Path(args.out))


if __name__ == "__main__":
    main()
