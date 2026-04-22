"""Unified chat interface across all four systems in the comparison.

Every evaluation script talks to one of these clients. They share a single
interface:

    client.chat(messages: list[dict], temperature: float, max_tokens: int) -> str

Which client to use is selected by ``make_client(kind, **kwargs)`` where
``kind`` is one of ``"base"``, ``"rule_based"``, ``"tinker"``, ``"hf"``.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..rule_based import RuleBasedRewriter


class BaseModelClient(ABC):
    """Interface every model client implements."""

    name: str = "base"

    @abstractmethod
    def chat(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 256
    ) -> str:
        ...


@dataclass
class OpenAIModelClient(BaseModelClient):
    """OpenAI-compatible endpoint (works for OpenAI, vLLM, Together, etc.)."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    name: str = "openai"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = self.base_url or os.environ.get("OPENAI_BASE_URL") or None
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 256
    ) -> str:
        client = self._ensure()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


@dataclass
class RuleBasedModelClient(BaseModelClient):
    """Wrap another client; apply the deterministic politeness rewriter to its output.

    This is System B in the plan. Give it a ``base_client`` to generate a raw
    reply, then the rewriter post-processes.
    """

    base_client: BaseModelClient
    rewriter: RuleBasedRewriter = field(default_factory=RuleBasedRewriter)
    name: str = "rule_based"

    def chat(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 256
    ) -> str:
        raw = self.base_client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return self.rewriter(raw)


@dataclass
class TinkerModelClient(BaseModelClient):
    """Sample from a Tinker-trained model by its saved ``model_path``.

    You get the ``model_path`` from ``sampling_client.model_path`` after
    training, or from the ``summary.json`` the training scripts write.
    """

    model_path: str
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # needed for tokenizer/renderer
    base_url: str | None = None
    name: str = "tinker"

    def __post_init__(self) -> None:
        self._sampling_client = None
        self._renderer = None

    def _ensure(self):
        if self._sampling_client is None:
            import tinker  # type: ignore
            from tinker_cookbook import model_info, renderers  # type: ignore
            from tinker_cookbook.tokenizer_utils import get_tokenizer  # type: ignore

            service = tinker.ServiceClient(base_url=self.base_url)
            self._sampling_client = service.create_sampling_client(
                model_path=self.model_path
            )
            tokenizer = get_tokenizer(self.model_name)
            renderer_name = model_info.get_recommended_renderer_name(self.model_name)
            self._renderer = renderers.get_renderer(renderer_name, tokenizer)
        return self._sampling_client, self._renderer

    def chat(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 256
    ) -> str:
        import tinker  # type: ignore

        sampling_client, renderer = self._ensure()
        # Render the prompt; Tinker's sampling takes token IDs.
        token_ids = renderer.build_generation_prompt(messages)
        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, stop=renderer.get_stop_sequences()
        )
        future = sampling_client.sample(
            prompt=tinker.ModelInput.from_ints(token_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )
        result = future.result()
        out_tokens = result.sequences[0].tokens
        return renderer.parse_generation(out_tokens).strip()


@dataclass
class HFModelClient(BaseModelClient):
    """Run a local Hugging Face model. Useful for base-model evaluation if you
    want to avoid API costs during development.
    """

    model_id: str
    device: str = "auto"
    name: str = "hf"

    def __post_init__(self) -> None:
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
            import torch  # type: ignore

            tok = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if self.device != "auto" else "auto",
            )
            self._pipe = pipeline("text-generation", model=model, tokenizer=tok)
        return self._pipe

    def chat(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 256
    ) -> str:
        pipe = self._ensure()
        out = pipe(
            messages,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            return_full_text=False,
        )
        # HF pipelines return a list of dicts; be permissive about schema.
        if isinstance(out, list) and out:
            item = out[0]
            if isinstance(item, dict):
                if "generated_text" in item:
                    gt = item["generated_text"]
                    if isinstance(gt, list):
                        # chat format: [{"role": "assistant", "content": "..."}]
                        for msg in reversed(gt):
                            if msg.get("role") == "assistant":
                                return (msg.get("content") or "").strip()
                    return str(gt).strip()
        return ""


def make_client(kind: str, **kwargs) -> BaseModelClient:
    """Factory. ``kind`` is one of: 'openai' | 'tinker' | 'hf' | 'rule_based'.

    For ``rule_based`` you must pass ``base_client`` (another BaseModelClient).
    """
    kind = kind.lower()
    if kind == "openai":
        return OpenAIModelClient(**kwargs)
    if kind == "tinker":
        return TinkerModelClient(**kwargs)
    if kind == "hf":
        return HFModelClient(**kwargs)
    if kind == "rule_based":
        return RuleBasedModelClient(**kwargs)
    raise ValueError(f"Unknown client kind: {kind!r}")
