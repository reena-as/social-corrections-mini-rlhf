#!/usr/bin/env python
"""Plot long-horizon evaluation results vs. short-horizon baseline.

Produces two figures:
  1. longhorizon_goal_by_turn.png — line chart of mean goal score by turn-count
     quartile (1–5, 6–10, 11–15, 16+), one line per system.
  2. longhorizon_vs_short.png — side-by-side bar chart comparing aggregate goal
     and relationship scores on long-horizon vs. short-horizon episodes, per system.

Usage:
    python scripts/plot_longhorizon.py \\
        --longhorizon-glob "data/processed/eval_longhorizon_*.json" \\
        --shorthorizon-glob "data/processed/eval_*/layer2_sotopia.json" \\
        --out-dir data/processed/final_paper/figures
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from social_corrections.utils.io import data_dir

_SYSTEM_ORDER = ["base", "rule_based", "sft", "dpo"]
_SYSTEM_COLORS = {
    "base": "#4C72B0",
    "rule_based": "#DD8452",
    "sft": "#55A868",
    "dpo": "#C44E52",
}
_SYSTEM_MARKERS = {
    "base": "o",
    "rule_based": "s",
    "sft": "^",
    "dpo": "D",
}

# Turn-count quartile bins (inclusive bounds, last bin is open-ended)
_BINS: list[tuple[str, int, int]] = [
    ("1–5",   1,  5),
    ("6–10",  6, 10),
    ("11–15", 11, 15),
    ("16+",   16, 9999),
]


def _bin_label(turn_count: int) -> str | None:
    for label, lo, hi in _BINS:
        if lo <= turn_count <= hi:
            return label
    return None


def _load_longhorizon(pattern: str) -> dict[str, list[dict[str, Any]]]:
    """Return {system: [per_episode dicts]}."""
    by_system: dict[str, list] = defaultdict(list)
    for path in sorted(glob_mod.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        system = data.get("system", Path(path).stem.replace("eval_longhorizon_", ""))
        by_system[system].extend(data.get("per_episode", []))
    return dict(by_system)


def _load_shorthorizon_aggregates(pattern: str) -> dict[str, dict[str, float]]:
    """Return {system: aggregate_scores_dict}."""
    result: dict[str, dict[str, float]] = {}
    for path in sorted(glob_mod.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        system = data.get("system", Path(path).parent.name.replace("eval_", ""))
        agg = {k: v for k, v in data.get("aggregate", {}).items() if isinstance(v, float)}
        if agg:
            result[system] = agg
    return result


def _mean(vals: list[float]) -> float | None:
    finite = [v for v in vals if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else None


def _plot_goal_by_turn(
    by_system: dict[str, list[dict[str, Any]]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    bin_labels = [b[0] for b in _BINS]

    systems = [s for s in _SYSTEM_ORDER if s in by_system]
    systems += sorted(set(by_system) - set(systems))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False

    for sys in systems:
        episodes = by_system[sys]
        bin_means: list[float] = []
        for label, lo, hi in _BINS:
            bucket_goals = [
                ep["scores"].get("goal", float("nan"))
                for ep in episodes
                if lo <= ep.get("turn_count", 0) <= hi
            ]
            m = _mean(bucket_goals)
            bin_means.append(m if m is not None else float("nan"))

        if all(math.isnan(v) for v in bin_means):
            continue
        any_data = True
        ax.plot(
            bin_labels,
            bin_means,
            marker=_SYSTEM_MARKERS.get(sys, "o"),
            color=_SYSTEM_COLORS.get(sys, None),
            label=sys,
            linewidth=1.8,
            markersize=7,
        )
        for x, y in zip(bin_labels, bin_means):
            if not math.isnan(y):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7.5)

    if not any_data:
        print(f"Warning: no data to plot for {out_path.name}")
        return

    ax.set_title("Mean goal score by turn-count quartile (long-horizon)", fontsize=11)
    ax.set_xlabel("Turn-count quartile (# agent turns completed)")
    ax.set_ylabel("Mean goal score (0–10)")
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_vs_short(
    by_system_long: dict[str, list[dict[str, Any]]],
    short_aggregates: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    dims = ("goal", "relationship")
    dim_ylim = {"goal": (0.0, 10.0), "relationship": (-5.0, 5.0)}

    # Build long-horizon aggregates
    long_aggregates: dict[str, dict[str, float]] = {}
    for sys, episodes in by_system_long.items():
        for dim in dims:
            vals = [ep["scores"].get(dim, float("nan")) for ep in episodes]
            m = _mean(vals)
            if m is not None:
                long_aggregates.setdefault(sys, {})[dim] = m

    all_systems = [s for s in _SYSTEM_ORDER if s in long_aggregates or s in short_aggregates]
    all_systems += sorted((set(long_aggregates) | set(short_aggregates)) - set(all_systems))

    n_sys = len(all_systems)
    bar_width = 0.35
    offsets = [-bar_width / 2, bar_width / 2]
    horizon_labels = ["short-horizon", "long-horizon"]
    horizon_hatches = ["", "//"]

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4.5))

    for ax, dim in zip(axes, dims):
        x_positions = list(range(n_sys))
        for hi, (h_label, h_agg, hatch) in enumerate(
            zip(horizon_labels, [short_aggregates, long_aggregates], horizon_hatches)
        ):
            vals = [h_agg.get(sys, {}).get(dim, float("nan")) for sys in all_systems]
            xs = [x + offsets[hi] for x in x_positions]
            bars = ax.bar(
                xs,
                [0 if math.isnan(v) else v for v in vals],
                width=bar_width * 0.92,
                label=h_label,
                color=[_SYSTEM_COLORS.get(sys, "#999999") for sys in all_systems],
                alpha=0.75 if hi == 0 else 0.95,
                hatch=hatch,
                edgecolor="white",
            )
            for bar, v in zip(bars, vals):
                if not math.isnan(v):
                    label_y = bar.get_height() + (0.1 if v >= 0 else -0.4)
                    ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"SOTOPIA {dim}: long- vs. short-horizon", fontsize=10)
        ax.set_ylabel(dim)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(all_systems, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(*dim_ylim.get(dim, (None, None)))
        if dim_ylim.get(dim, (0, 1))[0] < 0:
            ax.axhline(0, color="black", linewidth=0.6)

    # Shared legend for short vs long (use pattern patches)
    import matplotlib.patches as mpatches  # type: ignore
    legend_handles = [
        mpatches.Patch(facecolor="#aaaaaa", alpha=0.75, label="short-horizon"),
        mpatches.Patch(facecolor="#aaaaaa", alpha=0.95, hatch="//", label="long-horizon"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    default_out = str(data_dir() / "processed" / "final_paper" / "figures")
    ap = argparse.ArgumentParser(
        description="Plot long-horizon vs. short-horizon SOTOPIA evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--longhorizon-glob",
        default="data/processed/eval_longhorizon_*.json",
        help="Glob for long-horizon eval JSON files.",
    )
    ap.add_argument(
        "--shorthorizon-glob",
        default="data/processed/eval_*/layer2_sotopia.json",
        help="Glob for short-horizon layer2_sotopia.json files.",
    )
    ap.add_argument("--out-dir", default=default_out,
                    help="Directory for output PNG files.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    lh_files = sorted(glob_mod.glob(args.longhorizon_glob))
    sh_files = sorted(glob_mod.glob(args.shorthorizon_glob))

    if not lh_files:
        print(f"Error: no long-horizon files matched {args.longhorizon_glob!r}")
        raise SystemExit(1)

    by_system_long = _load_longhorizon(args.longhorizon_glob)
    short_aggregates = _load_shorthorizon_aggregates(args.shorthorizon_glob)

    total_lh = sum(len(v) for v in by_system_long.values())
    print(f"Long-horizon: {total_lh} episodes across {len(by_system_long)} system(s).")
    if short_aggregates:
        print(f"Short-horizon: aggregates for {sorted(short_aggregates)} loaded.")
    else:
        print("Short-horizon: no files matched — bar chart will show long-horizon only.")

    _plot_goal_by_turn(
        by_system_long,
        out_dir / "longhorizon_goal_by_turn.png",
    )
    _plot_vs_short(
        by_system_long,
        short_aggregates,
        out_dir / "longhorizon_vs_short.png",
    )


if __name__ == "__main__":
    main()
