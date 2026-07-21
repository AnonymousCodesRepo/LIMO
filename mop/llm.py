"""LLM client, model configuration, and response types.

Wraps OpenAI-compatible APIs (vLLM, etc.) for semantic operator calls.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum

from openai import OpenAI

logger = logging.getLogger(__name__)


# ── Model configuration ──────────────────────────────────────────


class ModelTier(Enum):
    SMALL = "small"
    LARGE = "large"


# ── Token pricing ────────────────────────────────────────────────
# Cloud open-source inference list prices (USD per token). Source: DeepInfra
# public pricing.
#   Large = Llama-3.1-70B-Instruct : $0.52 / 1M input, $0.75 / 1M output
#   Small = Qwen3.5-0.8B (3-way)   : $0.01 / 1M input, $0.05 / 1M output
# The streaming runner does NOT import this module — it mirrors these values as
# local constants (pipeline/runner/run_experiment.py); keep the two in sync when
# prices change.
LARGE_COST_PER_INPUT_TOKEN = 0.52e-6
LARGE_COST_PER_OUTPUT_TOKEN = 0.75e-6
SMALL_COST_PER_INPUT_TOKEN = 0.01e-6
SMALL_COST_PER_OUTPUT_TOKEN = 0.05e-6
# Embedding model = sentence-transformers/all-mpnet-base-v2: $0.005 / 1M tokens
EMBED_COST_PER_TOKEN = 0.005e-6


@dataclass
class ModelConfig:
    """Configuration for a single LLM endpoint."""

    name: str
    base_url: str
    tier: ModelTier = ModelTier.LARGE
    api_key: str = "dummy"
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    max_context_length: int = 16384
    default_temperature: float = 0.0
    default_max_tokens: int = 2048
    extra_body: dict = field(default_factory=dict)


class ModelRegistry:
    """Register and retrieve model configurations."""

    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}

    def register(self, config: ModelConfig) -> None:
        self._models[config.name] = config

    def get(self, name: str) -> ModelConfig:
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' not registered. Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def list_models(self) -> list[str]:
        return list(self._models.keys())


def default_registry() -> ModelRegistry:
    """Create a registry with default local model configs from env vars."""
    registry = ModelRegistry()

    base_url = os.environ.get(
        "MOP_LLM_BASE_URL", "http://localhost:8102/v1"
    )
    api_key = os.environ.get("MOP_LLM_API_KEY", "dummy")
    model_name = os.environ.get(
        "MOP_LLM_MODEL",
        "Meta-Llama-3.1-70B-Instruct",
    )

    registry.register(
        ModelConfig(
            name=model_name,
            base_url=base_url,
            api_key=api_key,
            tier=ModelTier.LARGE,
            max_context_length=16384,
            cost_per_input_token=LARGE_COST_PER_INPUT_TOKEN,
            cost_per_output_token=LARGE_COST_PER_OUTPUT_TOKEN,
        )
    )

    return registry


# ── LLM response ─────────────────────────────────────────────────


@dataclass
class TokenLogprob:
    """Log-probability for a single token."""

    token: str
    logprob: float
    top_logprobs: dict[str, float] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from a single LLM call."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = ""
    token_logprobs: list[TokenLogprob] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ── LLM client ───────────────────────────────────────────────────


class LLMClient:
    """Synchronous LLM client for vLLM / OpenAI-compatible endpoints."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        if model_config is None:
            reg = registry or default_registry()
            model_config = reg.get(reg.list_models()[0])
        self._config = model_config
        self._client = OpenAI(
            base_url=model_config.base_url,
            api_key=model_config.api_key,
        )

    @property
    def model_name(self) -> str:
        return self._config.name

    def call(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> LLMResponse:
        """Make a single chat completion call."""
        model = model or self._config.name
        temperature = (
            temperature if temperature is not None else self._config.default_temperature
        )
        max_tokens = max_tokens or self._config.default_max_tokens

        t0 = time.time()
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if logprobs:
            kwargs["logprobs"] = True
            if top_logprobs is not None:
                kwargs["top_logprobs"] = top_logprobs
        if self._config.extra_body:
            kwargs["extra_body"] = self._config.extra_body
        response = self._client.chat.completions.create(**kwargs)
        latency = time.time() - t0

        choice = response.choices[0]
        usage = response.usage

        # Parse logprobs if available
        token_logprobs: list[TokenLogprob] = []
        if logprobs and choice.logprobs and choice.logprobs.content:
            for tok_info in choice.logprobs.content:
                top = {}
                if tok_info.top_logprobs:
                    top = {t.token: t.logprob for t in tok_info.top_logprobs}
                token_logprobs.append(
                    TokenLogprob(
                        token=tok_info.token,
                        logprob=tok_info.logprob,
                        top_logprobs=top,
                    )
                )

        return LLMResponse(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_seconds=latency,
            model=model,
            token_logprobs=token_logprobs,
        )

    def call_batch(
        self,
        message_lists: list[list[dict[str, str]]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[LLMResponse]:
        """Process multiple prompts sequentially."""
        return [
            self.call(
                msgs, model=model, temperature=temperature, max_tokens=max_tokens
            )
            for msgs in message_lists
        ]

    def compute_cost(self, response: LLMResponse) -> float:
        """Compute USD cost for a response based on model config."""
        return (
            response.prompt_tokens * self._config.cost_per_input_token
            + response.completion_tokens * self._config.cost_per_output_token
        )
