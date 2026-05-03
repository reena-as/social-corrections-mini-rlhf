#!/usr/bin/env python
"""Tag SOTOPIA episodes with a dominant failure type and aggregate SOTOPIA scores.

For each episode, runs the failure judge (data/prompts/failure_judge.txt) over
every AGENT turn, determines the dominant failure type (mode of flagged types,
or "none"), then aggregates mean scores grouped by system × dominant_failure_type.

Usage:
    python scripts/failure_type_breakdown.py \\
        --eval-results-glob "data/processed/eval_*/layer2_sotopia.json" \\
        --out data/processed/failure_type_breakdown.json
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from social_corrections.inference.model_client import OpenAIModelClient
from social_corrections.taxonomy import ALL_LABELS, FAILURE_TYPES
from social_corrections.utils.io import data_dir, read_text, write_json

_SCORE_DIMS = ("goal", "relationship", "social_rules", "believability")


def _failure_type_list() -> str:
    return "\n".join(f'- "{label}"' for label in ALL_LABELS)


def _judge_turn(
    client: OpenAIModelClient,
    system_prompt: str,
    turn_content: str,
    failure_type_list: str,
) -> dict[str, Any]:
    user_msg = (
        f"Failure types:\n{failure_type_list}\n\n"
        f"Reply to evaluate:\n{turn_content}"
    )
    raw = client.chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=200,
    )
    txt = raw.strip()
    if txt.startswith("```"):
        lines = txt.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()
    return json.loads(txt)


def _dominant_failure_type(turn_flags: list[dict[str, Any]]) -> str:
    types = [
        f["failure_type"]
        for f in turn_flags
        if f.get("flagged") and f.get("failure_type")
    ]
    if not types:
        return "none"
    counts = Counter(types)
    # Mode; ties broken alphabetically for determinism
    return min(counts, key=lambda t: (-counts[t], t))


def _load_episodes(pattern: str) -> list[dict[str, Any]]:
    paths = sorted(glob_mod.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {pattern!r}")
    episodes: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        system = data.get("system", Path(path).parent.name)
        for ep in data.get("per_episode", []):
            episodes.append({"_system": system, **ep})
    return episodes


def _aggregate(
    per_episode: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    sums: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "goal": 0.0, "relationship": 0.0, "social_rules": 0.0, "believability": 0.0})
    )
    for ep in per_episode:
        ft = ep["dominant_failure_type"]
        sys = ep["system"]
        scores = ep["scores"]
        bucket = sums[ft][sys]
        bucket["n"] += 1
        for dim in _SCORE_DIMS:
            bucket[dim] += scores.get(dim, 0.0)

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for ft, by_sys in sums.items():
        out[ft] = {}
        for sys, bucket in by_sys.items():
            n = bucket["n"]
            out[ft][sys] = {"n": n}
            for dim in _SCORE_DIMS:
                out[ft][sys][dim] = round(bucket[dim] / n, 3)
    return out


def _short(label: str) -> str:
    return FAILURE_TYPES[label].short if label in FAILURE_TYPES else label


def _print_table(aggregate: dict[str, dict[str, dict[str, Any]]]) -> None:
    ft_w, sys_w, n_w, dim_w = 20, 12, 4, 9
    header = (
        f"{'Failure Type':<{ft_w}} {'System':<{sys_w}} {'N':>{n_w}}"
        + "".join(f"  {d:>{dim_w}}" for d in _SCORE_DIMS)
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    # "none" last, rest alphabetical by short label
    sorted_fts = sorted(
        aggregate.keys(),
        key=lambda ft: (ft == "none", _short(ft)),
    )
    for ft in sorted_fts:
        short = _short(ft)
        for j, (sys, vals) in enumerate(sorted(aggregate[ft].items())):
            label_col = short if j == 0 else ""
            dim_cols = "".join(f"  {vals.get(d, 0.0):>{dim_w}.3f}" for d in _SCORE_DIMS)
            print(f"{label_col:<{ft_w}} {sys:<{sys_w}} {vals['n']:>{n_w}}{dim_cols}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tag SOTOPIA episodes by dominant failure type and aggregate SOTOPIA scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--eval-results-glob",
        default="data/processed/eval_*/layer2_sotopia.json",
        help="Glob for layer2_sotopia.json eval files.",
    )
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--out", default="data/processed/failure_type_breakdown.json")
    args = ap.parse_args()

    try:
        all_episodes = _load_episodes(args.eval_results_glob)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    n_systems = len({e["_system"] for e in all_episodes})
    print(f"Loaded {len(all_episodes)} episodes across {n_systems} system(s).\n")

    judge_prompt = read_text(data_dir() / "prompts" / "failure_judge.txt")
    failure_type_list = _failure_type_list()
    client = OpenAIModelClient(model=args.judge_model)

    per_episode: list[dict[str, Any]] = []
    for i, ep in enumerate(all_episodes):
        sid = ep.get("scenario_id", "")
        ep_idx = ep.get("episode_index", i)
        sys = ep["_system"]
        agent_turns = [t for t in ep.get("transcript", []) if t.get("role") == "agent"]

        print(
            f"[{i+1}/{len(all_episodes)}] {sys} / {sid}:ep{ep_idx}"
            f" — {len(agent_turns)} agent turn(s) ...",
            end=" ",
            flush=True,
        )

        turn_flags: list[dict[str, Any]] = []
        for ti, turn in enumerate(agent_turns):
            try:
                result = _judge_turn(client, judge_prompt, turn["content"], failure_type_list)
                turn_flags.append({
                    "turn_index": ti,
                    "flagged": bool(result.get("flagged", False)),
                    "failure_type": result.get("failure_type", ""),
                    "rationale": result.get("rationale", ""),
                })
            except Exception as exc:
                turn_flags.append({
                    "turn_index": ti,
                    "flagged": False,
                    "failure_type": "",
                    "rationale": f"ERROR: {exc}",
                })

        dom = _dominant_failure_type(turn_flags)
        n_flagged = sum(1 for f in turn_flags if f["flagged"])
        print(f"{n_flagged}/{len(agent_turns)} flagged → {dom}")

        per_episode.append({
            "episode_id": f"{sid}:ep{ep_idx}",
            "system": sys,
            "scenario_id": sid,
            "dominant_failure_type": dom,
            "turn_flags": turn_flags,
            "scores": ep.get("scores", {}),
        })

    aggregate = _aggregate(per_episode)

    print()
    _print_table(aggregate)

    write_json({"per_episode": per_episode, "aggregate": aggregate}, args.out)
    print(f"Wrote {len(per_episode)} episodes → {args.out}")


if __name__ == "__main__":
    main()
