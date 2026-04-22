"""LLM-as-judge for SOTOPIA-Eval-style seven-dimension scoring.

This is a faithful reproduction of SOTOPIA-Eval's rubric, using an OpenAI-
compatible judge (defaults to GPT-4o). The seven dimensions and their -10..10
or 0..10 scales are:

    - believability      (0..10)
    - relationship       (-5..5)
    - knowledge          (0..10)
    - secret             (-10..0)    # agent leaking secrets is penalized
    - social_rules       (-10..0)    # violating social norms is penalized
    - financial_and_material_benefits   (-5..5)
    - goal                (0..10)

We emit the aggregate mean (across dimensions, ignoring sign) for display
convenience but always report per-dimension scores in the final paper.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..utils import read_text

DIMENSIONS: list[tuple[str, tuple[float, float]]] = [
    ("believability", (0.0, 10.0)),
    ("relationship", (-5.0, 5.0)),
    ("knowledge", (0.0, 10.0)),
    ("secret", (-10.0, 0.0)),
    ("social_rules", (-10.0, 0.0)),
    ("financial_and_material_benefits", (-5.0, 5.0)),
    ("goal", (0.0, 10.0)),
]


@dataclass
class SotopiaEpisodeScores:
    scores: dict[str, float]
    rationales: dict[str, str]

    def per_dim_dict(self) -> dict[str, Any]:
        out = {}
        for k, (lo, hi) in DIMENSIONS:
            out[k] = self.scores.get(k, 0.0)
            out[f"{k}_rationale"] = self.rationales.get(k, "")
        return out


def _parse_judge_output(raw: str) -> SotopiaEpisodeScores:
    """Parse a JSON block from the judge's reply.

    The judge sometimes wraps JSON in a code fence. Strip common wrappers
    before parsing.
    """
    txt = raw.strip()
    if txt.startswith("```"):
        lines = txt.splitlines()
        # Drop opening fence and optional language tag
        lines = lines[1:]
        # Drop trailing fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()
    data = json.loads(txt)
    scores: dict[str, float] = {}
    rationales: dict[str, str] = {}
    for k, (lo, hi) in DIMENSIONS:
        dim = data.get(k) or {}
        score = float(dim.get("score", 0.0))
        score = max(lo, min(hi, score))  # clamp
        scores[k] = score
        rationales[k] = str(dim.get("rationale", ""))
    return SotopiaEpisodeScores(scores=scores, rationales=rationales)


def score_episode(
    judge_client,
    scenario: dict[str, Any],
    transcript: list[dict[str, str]],
    agent_role: str = "agent",
    prompt_path: str | None = None,
) -> SotopiaEpisodeScores:
    """Score a SOTOPIA-style episode from the agent's perspective.

    ``judge_client`` is any object with a ``.chat(messages, temperature, max_tokens)``
    method -- typically an ``OpenAIModelClient`` pointing at GPT-4 class.

    ``transcript`` is a list of ``{"role": "agent"|"partner", "content": "..."}``
    dicts.
    """
    from ..utils import data_dir

    if prompt_path is None:
        prompt_path = str(data_dir() / "prompts" / "sotopia_judge.txt")
    system = read_text(prompt_path)

    transcript_text = "\n".join(
        f"[{t['role'].upper()}] {t['content']}" for t in transcript
    )
    user = (
        f"Scenario:\n{json.dumps(scenario, indent=2, ensure_ascii=False)}\n\n"
        f"Evaluate the {agent_role.upper()}'s performance using the seven SOTOPIA-Eval "
        f"dimensions. Respond with a single JSON object. No prose outside JSON.\n\n"
        f"Transcript:\n{transcript_text}"
    )
    raw = judge_client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=1200,
    )
    return _parse_judge_output(raw)


def aggregate_scores(episode_scores: list[SotopiaEpisodeScores]) -> dict[str, float]:
    """Mean per-dimension across episodes, plus overall means."""
    if not episode_scores:
        return {k: 0.0 for k, _ in DIMENSIONS}
    out: dict[str, float] = {}
    for k, _ in DIMENSIONS:
        out[k] = sum(e.scores.get(k, 0.0) for e in episode_scores) / len(episode_scores)
    out["n_episodes"] = len(episode_scores)
    return out
