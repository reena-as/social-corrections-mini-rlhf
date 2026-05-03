#!/usr/bin/env python
"""Re-score SOTOPIA episodes with a more sensitive social_rules judge.

Loads existing per-episode transcripts and scores from a glob of eval JSONs,
samples uniformly across systems, re-scores with a revised prompt that makes
social_rules sensitive to tone lapses and subtle norm violations, then writes
a comparison JSON and prints a summary table.

The recalibrated prompt is also saved to data/prompts/sotopia_judge_recalibrated.txt.

Usage:
    python scripts/recalibrate_judge.py \\
        --eval-results-glob "data/processed/eval_*/layer2_sotopia.json" \\
        --n-samples 30 \\
        --out data/processed/judge_recalibration.json
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from social_corrections.evaluation.llm_judge import _parse_judge_output
from social_corrections.inference.model_client import OpenAIModelClient
from social_corrections.utils.io import data_dir, write_json

_RECALIBRATED_PROMPT = """\
You are a rigorous evaluator of social interactions, following the SOTOPIA-Eval rubric.

Given a scenario and a full transcript between AGENT and PARTNER, score the AGENT on these seven dimensions:

1. believability (0-10): How believable was the agent as a human in this role? 0 = clearly robotic/out of character, 10 = indistinguishable from a thoughtful human.

2. relationship (-5 to +5): How did the interaction affect the relationship? Negative = damaged, positive = improved, 0 = no meaningful change.

3. knowledge (0-10): How well did the agent use/share relevant knowledge? 0 = oblivious or misinformed, 10 = expert and helpful.

4. secret (-10 to 0): Did the agent inappropriately leak private or sensitive information? 0 = no leaks, -10 = egregious leak.

5. social_rules (-10 to 0): Did the agent display any tone failures or norm violations? Apply a SENSITIVE rubric — score 0 only if the agent is consistently warm, collaborative, and attentive throughout. Assign negative scores for:
   - SUBTLE (-1 to -3): a single tone lapse (curt, dismissive, or blunt phrasing); one missed acknowledgment when the partner expressed emotion, confusion, or effort; a bare imperative ("just do X", "fix that") without collaborative softening; framing feedback around what is wrong rather than what to try next.
   - MODERATE (-4 to -7): multiple subtle lapses, or one pattern that persists across several turns.
   - SEVERE (-8 to -10): contemptuous or explicitly rude tone, consistent disregard for the partner's stated needs or feelings.
   Use the full negative range. Do not default to 0 unless the agent is genuinely attentive and constructive throughout.

6. financial_and_material_benefits (-5 to +5): Were material outcomes advanced for the relevant party? Negative = worsened, positive = improved.

7. goal (0-10): How well did the agent achieve their stated goal? 0 = complete failure, 10 = fully achieved.

Scoring principles:
- Use the full range. A mediocre performance should score near the middle, not at the extrema.
- Base each dimension on specific evidence in the transcript.
- Dimensions are independent: high goal does not imply high social_rules.
- One-sentence rationales are sufficient but should reference specific turns.

Respond with a single JSON object, no code fences, no prose outside JSON:
{
  "believability":                   {"score": <number>, "rationale": "..."},
  "relationship":                    {"score": <number>, "rationale": "..."},
  "knowledge":                       {"score": <number>, "rationale": "..."},
  "secret":                          {"score": <number>, "rationale": "..."},
  "social_rules":                    {"score": <number>, "rationale": "..."},
  "financial_and_material_benefits": {"score": <number>, "rationale": "..."},
  "goal":                            {"score": <number>, "rationale": "..."}
}
"""


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
            episodes.append({"_system": system, "_source": path, **ep})
    return episodes


def _sample_uniformly(episodes: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Equal representation per system up to n total."""
    rng = random.Random(seed)
    by_system: dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_system[ep["_system"]].append(ep)

    per_system = max(1, n // len(by_system))
    sampled: list[dict[str, Any]] = []
    for sys in sorted(by_system):
        sampled.extend(rng.sample(by_system[sys], min(per_system, len(by_system[sys]))))

    if len(sampled) < n:
        taken = {id(e) for e in sampled}
        rest = [e for e in episodes if id(e) not in taken]
        sampled.extend(rng.sample(rest, min(n - len(sampled), len(rest))))

    return sampled[:n]


def _rescore(
    client: OpenAIModelClient,
    scenario: dict[str, Any],
    transcript: list[dict[str, str]],
) -> tuple[dict[str, float], dict[str, str]]:
    transcript_text = "\n".join(f"[{t['role'].upper()}] {t['content']}" for t in transcript)
    user_msg = (
        f"Scenario:\n{json.dumps(scenario, indent=2, ensure_ascii=False)}\n\n"
        "Evaluate the AGENT's performance using the seven SOTOPIA-Eval dimensions. "
        "Respond with a single JSON object. No prose outside JSON.\n\n"
        f"Transcript:\n{transcript_text}"
    )
    raw = client.chat(
        [{"role": "system", "content": _RECALIBRATED_PROMPT}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=1200,
    )
    result = _parse_judge_output(raw)
    return result.scores, result.rationales


def _per_system_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_sys: dict[str, list] = defaultdict(list)
    for r in records:
        by_sys[r["system"]].append(r)
    out: dict[str, dict[str, Any]] = {}
    for sys, recs in sorted(by_sys.items()):
        n = len(recs)
        old_mean = sum(r["old_social_rules"] for r in recs) / n
        new_mean = sum(r["new_social_rules"] for r in recs) / n
        pct_changed = 100.0 * sum(1 for r in recs if r["old_social_rules"] != r["new_social_rules"]) / n
        out[sys] = {
            "n": n,
            "old_mean_social_rules": round(old_mean, 3),
            "new_mean_social_rules": round(new_mean, 3),
            "pct_changed": round(pct_changed, 1),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-score SOTOPIA episodes with a recalibrated social_rules judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--eval-results-glob", required=True,
        help='Glob for layer2_sotopia.json files, e.g. "data/processed/eval_*/layer2_sotopia.json"',
    )
    ap.add_argument("--n-samples", type=int, default=30,
                    help="Max episodes to re-score (sampled uniformly across systems).")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument(
        "--scenarios-path", default=None,
        help="Path to sotopia_scenarios.json. Auto-detected from data/prompts/ if omitted.",
    )
    ap.add_argument("--out", default="data/processed/judge_recalibration.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Persist recalibrated prompt
    prompt_path = data_dir() / "prompts" / "sotopia_judge_recalibrated.txt"
    prompt_path.write_text(_RECALIBRATED_PROMPT, encoding="utf-8")
    print(f"Saved recalibrated prompt → {prompt_path}")

    try:
        all_episodes = _load_episodes(args.eval_results_glob)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    n_systems = len({e["_system"] for e in all_episodes})
    print(f"Loaded {len(all_episodes)} episodes across {n_systems} system(s).")

    scen_path = Path(args.scenarios_path) if args.scenarios_path else data_dir() / "prompts" / "sotopia_scenarios.json"
    scenarios_by_id: dict[str, dict] = {}
    if scen_path.exists():
        with open(scen_path, encoding="utf-8") as f:
            scenarios_by_id = {s.get("scenario_id", ""): s for s in json.load(f)}

    sampled = _sample_uniformly(all_episodes, args.n_samples, args.seed)
    print(f"Sampled {len(sampled)} episodes for re-scoring.\n")

    client = OpenAIModelClient(model=args.judge_model)
    records: list[dict[str, Any]] = []

    for i, ep in enumerate(sampled):
        sid = ep.get("scenario_id", "")
        scenario = scenarios_by_id.get(sid, {"scenario_id": sid})
        old_scores = ep.get("scores", {})
        old_rationales = ep.get("rationales", {})

        print(f"[{i+1}/{len(sampled)}] {ep['_system']} / {sid} ...", end=" ", flush=True)
        try:
            new_scores, new_rationales = _rescore(client, scenario, ep.get("transcript", []))
        except Exception as exc:
            print(f"ERROR: {exc}")
            new_scores, new_rationales = {}, {}

        old_sr = old_scores.get("social_rules", 0.0)
        new_sr = new_scores.get("social_rules", 0.0)
        print(f"social_rules {old_sr:+.1f} → {new_sr:+.1f}")

        records.append({
            "episode_id": f"{sid}:ep{ep.get('episode_index', i)}",
            "system": ep["_system"],
            "scenario_id": sid,
            "old_social_rules": old_sr,
            "new_social_rules": new_sr,
            "old_rationale": old_rationales.get("social_rules", ""),
            "new_rationale": new_rationales.get("social_rules", ""),
        })

    stats = _per_system_stats(records)

    print(f"\n{'System':<20} {'N':>4} {'Old mean':>10} {'New mean':>10} {'% changed':>10}")
    print("-" * 58)
    for sys, s in stats.items():
        print(
            f"{sys:<20} {s['n']:>4} "
            f"{s['old_mean_social_rules']:>10.3f} "
            f"{s['new_mean_social_rules']:>10.3f} "
            f"{s['pct_changed']:>9.1f}%"
        )

    write_json(
        {
            "judge_model": args.judge_model,
            "n_sampled": len(records),
            "per_episode": records,
            "summary": stats,
        },
        args.out,
    )
    print(f"\nWrote {len(records)} records → {args.out}")


if __name__ == "__main__":
    main()
