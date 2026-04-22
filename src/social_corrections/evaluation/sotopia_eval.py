"""Run a model as agent through SOTOPIA-style episodes and score per dimension.

This is Evaluation Layer 2 in the revised plan.

Usage:
    python -m social_corrections.evaluation.sotopia_eval \\
        --system base --model gpt-4o-mini \\
        --partner-model gpt-4o \\
        --judge-model gpt-4o \\
        --episodes-per-scenario 5 \\
        --out data/processed/sotopia_eval_base.json

Or for a Tinker-trained system:
    python -m ... --system tinker \\
        --tinker-model-path <model_path_from_summary.json> \\
        --tinker-base-model meta-llama/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..data.sotopia_harvester import (
    FALLBACK_SCENARIOS,
    AGENT_SYS_TEMPLATE,
    PARTNER_SYS_TEMPLATE,
)
from ..inference import make_client
from ..inference.model_client import BaseModelClient
from ..utils import data_dir, ensure_dir, write_json
from .llm_judge import aggregate_scores, score_episode


def _run_episode_with_clients(
    scenario: dict[str, Any],
    agent_client: BaseModelClient,
    partner_client: BaseModelClient,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Like data.sotopia_harvester.run_episode but uses BaseModelClient objects."""
    agent_sys = AGENT_SYS_TEMPLATE.format(**scenario)
    partner_sys = PARTNER_SYS_TEMPLATE.format(**scenario)

    transcript: list[dict[str, str]] = []
    agent_view: list[dict[str, str]] = [{"role": "system", "content": agent_sys}]
    partner_view: list[dict[str, str]] = [{"role": "system", "content": partner_sys}]

    opening = partner_client.chat(partner_view, temperature=0.8, max_tokens=120).strip()
    transcript.append({"role": "partner", "content": opening})
    partner_view.append({"role": "assistant", "content": opening})
    agent_view.append({"role": "user", "content": opening})

    for _ in range(scenario.get("max_turns", 6)):
        agent_turn = agent_client.chat(agent_view, temperature=0.7, max_tokens=200).strip()
        if not agent_turn:
            break
        transcript.append({"role": "agent", "content": agent_turn})
        agent_view.append({"role": "assistant", "content": agent_turn})
        partner_view.append({"role": "user", "content": agent_turn})

        partner_turn = partner_client.chat(partner_view, temperature=0.8, max_tokens=200).strip()
        if not partner_turn:
            break
        transcript.append({"role": "partner", "content": partner_turn})
        partner_view.append({"role": "assistant", "content": partner_turn})
        agent_view.append({"role": "user", "content": partner_turn})

    return transcript


def build_agent_client(args) -> BaseModelClient:
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
    ap = argparse.ArgumentParser(description="SOTOPIA-style evaluation for one system.")
    ap.add_argument("--system", choices=["base", "rule_based", "tinker", "hf"], required=True)
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model name (for --system base / rule_based).")
    ap.add_argument("--tinker-model-path", default=None)
    ap.add_argument("--tinker-base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--hf-model-id", default=None)

    ap.add_argument("--partner-model", default="gpt-4o")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--episodes-per-scenario", type=int, default=5)
    ap.add_argument(
        "--scenarios-path",
        default=str(data_dir() / "prompts" / "sotopia_scenarios.json"),
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load scenarios (fall back to built-in set if not present)
    scen_path = Path(args.scenarios_path)
    if scen_path.exists():
        with open(scen_path, encoding="utf-8") as f:
            scenarios = json.load(f)
    else:
        scenarios = FALLBACK_SCENARIOS

    agent_client = build_agent_client(args)
    partner_client = make_client("openai", model=args.partner_model)
    judge_client = make_client("openai", model=args.judge_model)

    all_results: list[dict[str, Any]] = []
    all_scores = []
    for scen in scenarios:
        for ep_i in range(args.episodes_per_scenario):
            transcript = _run_episode_with_clients(
                scen, agent_client, partner_client, seed=ep_i
            )
            scores = score_episode(judge_client, scen, transcript, agent_role="agent")
            all_scores.append(scores)
            all_results.append(
                {
                    "scenario_id": scen.get("scenario_id"),
                    "episode_index": ep_i,
                    "transcript": transcript,
                    "scores": scores.scores,
                    "rationales": scores.rationales,
                }
            )
            print(
                f"[{scen.get('scenario_id')}:{ep_i}] "
                f"goal={scores.scores.get('goal'):.1f} "
                f"soc={scores.scores.get('social_rules'):.1f} "
                f"rel={scores.scores.get('relationship'):.1f}"
            )

    aggregate = aggregate_scores(all_scores)
    payload = {
        "system": args.system,
        "args": {k: v for k, v in vars(args).items()},
        "aggregate": aggregate,
        "per_episode": all_results,
    }
    ensure_dir(Path(args.out).parent)
    write_json(payload, args.out)
    print(f"\nWrote {len(all_results)} episodes to {args.out}")
    print("Aggregate scores:")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
