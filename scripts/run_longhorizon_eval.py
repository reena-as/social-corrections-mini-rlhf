#!/usr/bin/env python
"""Run SOTOPIA-style long-horizon evaluation (20+ turn episodes) for one system.

Reuses the core episode runner and judge from sotopia_eval.py. Output schema
is identical to layer2_sotopia.json so existing plotting scripts work, with
one extra field per episode: turn_count (number of agent turns completed
before a participant stopped responding).

Usage:
    python scripts/run_longhorizon_eval.py --system base --model gpt-4o-mini \\
        --out data/processed/eval_longhorizon_base.json

    python scripts/run_longhorizon_eval.py --system tinker \\
        --tinker-model-path <path> \\
        --tinker-base-model meta-llama/Llama-3.1-8B-Instruct \\
        --out data/processed/eval_longhorizon_sft.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from social_corrections.evaluation.llm_judge import aggregate_scores, score_episode
from social_corrections.evaluation.sotopia_eval import (
    _run_episode_with_clients,
    build_agent_client,
)
from social_corrections.inference import make_client
from social_corrections.utils.io import data_dir, ensure_dir, write_json


def _load_scenarios(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Long-horizon SOTOPIA evaluation (20+ turns) for one system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # System selection — mirrors run_all_eval.py / sotopia_eval.py
    ap.add_argument("--system", choices=["base", "rule_based", "tinker", "hf"], required=True)
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model name (for --system base or rule_based).")
    ap.add_argument("--tinker-model-path", default=None)
    ap.add_argument("--tinker-base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--hf-model-id", default=None)

    ap.add_argument("--partner-model", default="gpt-4o")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--episodes-per-scenario", type=int, default=3)
    ap.add_argument(
        "--scenarios-path",
        default=str(data_dir() / "prompts" / "sotopia_scenarios_longhorizon.json"),
    )
    ap.add_argument(
        "--out", default=None,
        help="Output path. Defaults to data/processed/eval_longhorizon_{system}.json.",
    )
    args = ap.parse_args()

    if args.out is None:
        args.out = f"data/processed/eval_longhorizon_{args.system}.json"

    scenarios = _load_scenarios(args.scenarios_path)
    print(f"Loaded {len(scenarios)} scenarios from {args.scenarios_path}")
    print(f"System: {args.system}  |  episodes/scenario: {args.episodes_per_scenario}\n")

    agent_client = build_agent_client(args)
    partner_client = make_client("openai", model=args.partner_model)
    judge_client = make_client("openai", model=args.judge_model)

    all_scores = []
    all_results: list[dict[str, Any]] = []

    for scen in scenarios:
        sid = scen.get("scenario_id", "?")
        max_turns = scen.get("max_turns", 22)
        for ep_i in range(args.episodes_per_scenario):
            transcript = _run_episode_with_clients(
                scen, agent_client, partner_client, seed=ep_i
            )
            turn_count = sum(1 for t in transcript if t.get("role") == "agent")

            scores = score_episode(judge_client, scen, transcript, agent_role="agent")
            all_scores.append(scores)
            all_results.append(
                {
                    "scenario_id": sid,
                    "episode_index": ep_i,
                    "turn_count": turn_count,
                    "max_turns": max_turns,
                    "transcript": transcript,
                    "scores": scores.scores,
                    "rationales": scores.rationales,
                }
            )
            print(
                f"[{sid}:{ep_i}] turns={turn_count}/{max_turns}"
                f"  goal={scores.scores.get('goal', 0):.1f}"
                f"  soc={scores.scores.get('social_rules', 0):.1f}"
                f"  rel={scores.scores.get('relationship', 0):.1f}"
            )

    aggregate = aggregate_scores(all_scores)
    payload: dict[str, Any] = {
        "system": args.system,
        "args": vars(args),
        "aggregate": aggregate,
        "per_episode": all_results,
    }
    ensure_dir(Path(args.out).parent)
    write_json(payload, args.out)

    print(f"\nWrote {len(all_results)} episodes → {args.out}")
    print("Aggregate scores:")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
