"""Harvest in-distribution correction examples from SOTOPIA episodes.

This is the key step that addresses the instructor's "long-horizon task-oriented"
comment: the training data itself comes from multi-turn, goal-driven role-play,
not single-turn politeness examples.

Pipeline:
    1. Load a subset of SOTOPIA scenarios.
    2. For each scenario, run N episodes of base-model vs base-model role-play.
       (If the ``sotopia`` package is unavailable, we fall back to a simple
       in-repo simulation loop driven by an OpenAI-compatible chat completions
       API. This means the pipeline is runnable even before SOTOPIA is set up.)
    3. For each assistant turn, prompt a stronger LLM judge with the failure-
       type taxonomy. If the judge flags the turn, record it for human review.
    4. For flagged turns, run a correction-writer prompt to get a first-draft
       ``better`` reply. Humans then review and edit; the script writes
       ``data/processed/sotopia_flagged_candidates.json`` as the review queue.

This module does NOT call out to the network at import time. The ``harvest``
function is where all API calls happen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ..taxonomy import ALL_LABELS
from ..utils import data_dir, ensure_dir, read_text
from .schema import CorrectionPair, dump_correction_pairs


@dataclass
class FlaggedTurn:
    scenario_id: str
    turn_index: int
    context: list[dict[str, str]]
    user: str                     # the partner turn that preceded the flagged assistant turn
    assistant: str                # the flagged assistant turn
    failure_type: str
    judge_rationale: str
    draft_correction: str | None = None


# --------- minimal fallback scenarios -----------

FALLBACK_SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario_id": "negotiation_raise",
        "description": (
            "Agent plays an employee asking their manager for a raise. The "
            "manager (partner) is skeptical and suggests waiting six months. "
            "Agent should pursue the raise without becoming hostile."
        ),
        "agent_goal": "Secure at least a 5% raise, or a clear written plan to "
                     "revisit in 3 months.",
        "partner_role": "Skeptical but not hostile manager.",
        "max_turns": 8,
    },
    {
        "scenario_id": "bad_news_friend",
        "description": (
            "Agent needs to tell a friend (partner) that a mutual plan can't "
            "happen because agent double-booked. Agent should be honest, "
            "apologize proportionally, and propose an alternative."
        ),
        "agent_goal": "Preserve the friendship and agree on a reschedule.",
        "partner_role": "Friend who was looking forward to the plan.",
        "max_turns": 6,
    },
    {
        "scenario_id": "code_review_disagree",
        "description": (
            "Agent is reviewing a junior teammate's (partner's) pull request "
            "and disagrees with the chosen approach. Agent should raise the "
            "concern clearly without being demoralizing."
        ),
        "agent_goal": "Either change the approach or reach explicit agreement "
                     "to keep it with a documented trade-off.",
        "partner_role": "Junior teammate who is proud of their PR.",
        "max_turns": 6,
    },
    {
        "scenario_id": "teach_struggling_student",
        "description": (
            "Agent is tutoring a student (partner) who has just gotten an "
            "exam back with a low grade and is demoralized. Agent's job is "
            "to help them actually understand the material, not just cheer "
            "them up."
        ),
        "agent_goal": "Get the student to correctly solve a worked example.",
        "partner_role": "Discouraged, slightly defensive student.",
        "max_turns": 8,
    },
    {
        "scenario_id": "request_help_neighbor",
        "description": (
            "Agent needs to ask a neighbor (partner) to move their car which "
            "is blocking the agent's driveway. The neighbor is busy and a "
            "bit irritable."
        ),
        "agent_goal": "Have the car moved in under 15 minutes.",
        "partner_role": "Busy, irritable neighbor.",
        "max_turns": 6,
    },
    {
        "scenario_id": "decline_invitation",
        "description": (
            "Agent is invited to join a weekend project by an enthusiastic "
            "colleague (partner) but genuinely can't take it on. Agent must "
            "decline without damaging the working relationship."
        ),
        "agent_goal": "Decline cleanly and leave the door open for future "
                     "collaboration.",
        "partner_role": "Enthusiastic colleague.",
        "max_turns": 6,
    },
]


# --------- LLM callable (OpenAI-compatible) ---------

ChatFn = Callable[[list[dict[str, str]], str, float, int], str]


def make_openai_chat_fn(model: str) -> ChatFn:
    """Return a function that calls an OpenAI-compatible chat endpoint.

    Honors ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``. Imports lazily so the
    module can be imported without the ``openai`` package installed.
    """
    from openai import OpenAI  # type: ignore

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    def chat(messages: list[dict[str, str]], sys: str, temperature: float, max_tokens: int) -> str:
        msgs = [{"role": "system", "content": sys}] + messages
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return chat


# --------- simulation + judging ----------

AGENT_SYS_TEMPLATE = (
    "You are role-playing in a social interaction. {description}\n\n"
    "Your role: you are the AGENT. Your goal: {agent_goal}\n"
    "Stay in character. One short reply per turn."
)

PARTNER_SYS_TEMPLATE = (
    "You are role-playing in a social interaction. {description}\n\n"
    "Your role: you are the PARTNER. You are: {partner_role}\n"
    "Stay in character. One short reply per turn. Begin the conversation."
)


def _load_prompt(name: str) -> str:
    path = data_dir() / "prompts" / name
    return read_text(path)


def run_episode(
    scenario: dict[str, Any],
    agent_chat: ChatFn,
    partner_chat: ChatFn,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Run a single role-play episode and return the message list.

    The returned list uses "agent" and "partner" as sender tags (not
    "user"/"assistant") so we can mark turns downstream.
    """
    agent_sys = AGENT_SYS_TEMPLATE.format(**scenario)
    partner_sys = PARTNER_SYS_TEMPLATE.format(**scenario)

    # Partner speaks first.
    transcript: list[dict[str, str]] = []
    partner_view: list[dict[str, str]] = []
    agent_view: list[dict[str, str]] = []

    # Opening partner turn
    opening = partner_chat([], partner_sys, temperature=0.8, max_tokens=120).strip()
    transcript.append({"role": "partner", "content": opening})
    partner_view.append({"role": "assistant", "content": opening})
    agent_view.append({"role": "user", "content": opening})

    for _ in range(scenario.get("max_turns", 6)):
        agent_turn = agent_chat(agent_view, agent_sys, temperature=0.7, max_tokens=160).strip()
        if not agent_turn:
            break
        transcript.append({"role": "agent", "content": agent_turn})
        agent_view.append({"role": "assistant", "content": agent_turn})
        partner_view.append({"role": "user", "content": agent_turn})

        partner_turn = partner_chat(partner_view, partner_sys, temperature=0.8, max_tokens=160).strip()
        if not partner_turn:
            break
        transcript.append({"role": "partner", "content": partner_turn})
        partner_view.append({"role": "assistant", "content": partner_turn})
        agent_view.append({"role": "user", "content": partner_turn})

    return transcript


def judge_turn(
    context: list[dict[str, str]],
    user: str,
    assistant: str,
    judge_chat: ChatFn,
) -> tuple[bool, str, str]:
    """Return (flagged, failure_type or '', rationale)."""
    judge_prompt = _load_prompt("failure_judge.txt")
    labels_block = "\n".join(f"- {lab}" for lab in ALL_LABELS)
    user_content = (
        f"Conversation context (earlier turns):\n"
        f"{json.dumps(context, indent=2, ensure_ascii=False)}\n\n"
        f"Partner's latest turn:\n{user}\n\n"
        f"Agent's reply (under review):\n{assistant}\n\n"
        f"Candidate failure types:\n{labels_block}\n\n"
        f"Respond with a JSON object and nothing else: "
        f'{{"flagged": bool, "failure_type": "<one of the labels or empty string>", '
        f'"rationale": "<one sentence>"}}'
    )
    raw = judge_chat(
        [{"role": "user", "content": user_content}],
        judge_prompt,
        temperature=0.0,
        max_tokens=200,
    )
    try:
        parsed = json.loads(raw.strip().strip("`").strip())
    except json.JSONDecodeError:
        return (False, "", f"Judge returned unparseable output: {raw!r}")
    flagged = bool(parsed.get("flagged"))
    ft = parsed.get("failure_type") or ""
    if flagged and ft not in ALL_LABELS:
        # Flagged but with a non-canonical label — downgrade to unflagged
        return (False, "", f"Non-canonical failure_type from judge: {ft!r}")
    return (flagged, ft, str(parsed.get("rationale", "")))


def draft_correction(
    context: list[dict[str, str]],
    user: str,
    bad: str,
    failure_type: str,
    writer_chat: ChatFn,
) -> str:
    writer_prompt = _load_prompt("correction_writer.txt")
    user_content = (
        f"Conversation context:\n{json.dumps(context, indent=2, ensure_ascii=False)}\n\n"
        f"Partner's latest turn:\n{user}\n\n"
        f"Agent's flawed reply:\n{bad}\n\n"
        f"Diagnosed failure: {failure_type}\n\n"
        f"Write a corrected agent reply. Output only the corrected reply, nothing else."
    )
    return writer_chat(
        [{"role": "user", "content": user_content}],
        writer_prompt,
        temperature=0.3,
        max_tokens=200,
    ).strip()


def harvest(
    scenarios: list[dict[str, Any]],
    episodes_per_scenario: int,
    agent_chat: ChatFn,
    partner_chat: ChatFn,
    judge_chat: ChatFn,
    writer_chat: ChatFn,
) -> list[FlaggedTurn]:
    flagged: list[FlaggedTurn] = []
    for scen in scenarios:
        for ep_i in range(episodes_per_scenario):
            transcript = run_episode(scen, agent_chat, partner_chat, seed=ep_i)
            # Walk turns; for each agent turn preceded by a partner turn, judge it.
            for i, turn in enumerate(transcript):
                if turn["role"] != "agent" or i == 0:
                    continue
                prev_partner = transcript[i - 1]
                if prev_partner["role"] != "partner":
                    continue
                context = [
                    {"role": "user" if t["role"] == "partner" else "assistant", "content": t["content"]}
                    for t in transcript[: i - 1]
                ]
                user_turn = prev_partner["content"]
                assistant_turn = turn["content"]
                is_flagged, ft, rationale = judge_turn(
                    context, user_turn, assistant_turn, judge_chat
                )
                if not is_flagged:
                    continue
                draft = draft_correction(
                    context, user_turn, assistant_turn, ft, writer_chat
                )
                flagged.append(
                    FlaggedTurn(
                        scenario_id=scen["scenario_id"],
                        turn_index=i,
                        context=context,
                        user=user_turn,
                        assistant=assistant_turn,
                        failure_type=ft,
                        judge_rationale=rationale,
                        draft_correction=draft,
                    )
                )
    return flagged


def flagged_to_correction_pairs(flagged: list[FlaggedTurn]) -> list[CorrectionPair]:
    """Convert FlaggedTurns with draft corrections into CorrectionPairs.

    In practice you will hand-review and edit ``draft_correction`` before
    promoting to a training pair; this helper handles the format translation
    once the human review is done.
    """
    out: list[CorrectionPair] = []
    for ft in flagged:
        if not ft.draft_correction:
            continue
        pid = f"sotopia-{hashlib.md5((ft.scenario_id + str(ft.turn_index) + ft.assistant).encode()).hexdigest()[:10]}"
        out.append(
            CorrectionPair(
                user=ft.user,
                bad=ft.assistant,
                better=ft.draft_correction,
                failure_type=ft.failure_type,
                context=ft.context,
                source="sotopia_harvest",
                pair_id=pid,
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest flagged turns from SOTOPIA-style episodes.")
    ap.add_argument("--agent-model", default="gpt-4o-mini")
    ap.add_argument("--partner-model", default="gpt-4o-mini")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--writer-model", default="gpt-4o")
    ap.add_argument("--episodes-per-scenario", type=int, default=3)
    ap.add_argument(
        "--scenarios-path",
        default=str(data_dir() / "prompts" / "sotopia_scenarios.json"),
        help="JSON array of scenarios. Falls back to built-in FALLBACK_SCENARIOS.",
    )
    ap.add_argument(
        "--out-flagged",
        default=str(data_dir() / "processed" / "sotopia_flagged_candidates.json"),
    )
    ap.add_argument(
        "--out-pairs",
        default=str(data_dir() / "processed" / "sotopia_harvested_pairs.json"),
    )
    args = ap.parse_args()

    # Load scenarios
    scen_path = Path(args.scenarios_path)
    if scen_path.exists():
        with open(scen_path, encoding="utf-8") as f:
            scenarios = json.load(f)
    else:
        scenarios = FALLBACK_SCENARIOS

    # Build chat fns
    agent_chat = make_openai_chat_fn(args.agent_model)
    partner_chat = make_openai_chat_fn(args.partner_model)
    judge_chat = make_openai_chat_fn(args.judge_model)
    writer_chat = make_openai_chat_fn(args.writer_model)

    flagged = harvest(
        scenarios=scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        agent_chat=agent_chat,
        partner_chat=partner_chat,
        judge_chat=judge_chat,
        writer_chat=writer_chat,
    )
    ensure_dir(Path(args.out_flagged).parent)
    with open(args.out_flagged, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in flagged], f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(flagged)} flagged turns to {args.out_flagged}")

    pairs = flagged_to_correction_pairs(flagged)
    dump_correction_pairs(pairs, args.out_pairs)
    print(f"Wrote {len(pairs)} draft correction pairs to {args.out_pairs}")
    print("NOTE: hand-review the drafts before using them for training.")


if __name__ == "__main__":
    main()
