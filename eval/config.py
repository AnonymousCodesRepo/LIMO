"""Dataclass configs for an evaluation run.

One dataclass per pipeline stage so each is independently swappable. The
top-level ``EvalConfig`` is what gets serialised into the run's results
JSON header so a finished run is fully reproducible from its own output.

Stage modules (``StageSpec``) carry a ``name`` and a free-form ``kwargs``
dict that is forwarded verbatim to the stage's ``build()`` factory.
Field-level validation lives close to the construction site (``run.py``)
so this file stays a pure schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageSpec:
    """One stage's name + constructor kwargs.

    ``name`` is matched against the stage's ``REGISTRY``. ``kwargs`` is
    forwarded to the constructor as-is — runtime objects (LLM clients,
    embedding clients, the retriever instance for joint-feature routers)
    are injected by the build code, not declared here.
    """

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data slice + ordering."""

    dataset: str = "cuad"
    """Document-text source. One of the keys in ``eval.data.DATASETS``:
      * ``"cuad"`` — CUAD legal contracts, paragraph-level text.
      * ``"opp115"`` — OPP-115 privacy-policy segments.
      * ``"hoc"`` — Hallmarks of Cancer PubMed abstracts.
    """

    queries: str = "phase_a"
    """Query slice spec.

    Sentinels:
      * ``"phase_a"`` — the canonical 6-query phase_a slice.
      * ``"all_cuad"`` — every cuad_* query in the dataset's pairs file
        (38 queries for CUAD).
      * ``"all_eval"`` / ``"all"`` — every query in the dataset's pairs file.
    Otherwise: a comma-separated list of explicit query names.
    """

    within_query_order: str = "doc_id_asc"
    """One of: ``doc_id_asc`` | ``random`` | ``doc_length_asc`` |
    ``doc_length_desc``."""

    order_seed: int = 0
    limit_per_query: int | None = None
    point_embedding_cache: str | None = None


@dataclass
class ExperiencePoolConfig:
    """Pre-supplied experience pool."""

    source: str = "phase_a"
    """Short name (``phase_a``) registered in
    ``pipeline.experience.REGISTRY``, OR a path to a JSONL file, OR
    ``"empty"`` / ``"none"`` to start with no pool."""


@dataclass
class PretrainConfig:
    """Optional warm-start of router / retriever from offline pretraining."""

    mode: str = "none"
    """One of:
      * ``"none"`` — no warm-start (default).
      * ``"load"`` — load existing checkpoints from
        ``router_checkpoint`` / ``retriever_snapshot``.
      * ``"fit"`` — run the offline pretraining pipeline (synth →
        joint pretrain of router + utility estimator, paper Section 6)
        into ``output_dir`` before the eval, then load both artifacts.
    """

    router_checkpoint: str | None = None
    retriever_snapshot: str | None = None
    output_dir: str | None = None
    fit: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """LLM endpoints + the large-prediction cache."""

    # Endpoint + served-model-name defaults. All five are overridable via
    # environment variables so the same code can run on any server that
    # exposes the LLMs/embedding service on the same ports. CLI
    # `--set llms.<field>=...` still wins over the env var.
    small_url: str = field(
        default_factory=lambda: os.environ.get(
            "MOP_SMALL_URL", "http://localhost:8105/v1"
        )
    )
    small_model: str = field(
        default_factory=lambda: os.environ.get(
            "MOP_SMALL_MODEL", "Qwen3.5-0.8B"
        )
    )
    large_url: str = field(
        default_factory=lambda: os.environ.get(
            "MOP_LARGE_URL", "http://localhost:8102/v1"
        )
    )
    large_model: str = field(
        default_factory=lambda: os.environ.get(
            "MOP_LARGE_MODEL", "Meta-Llama-3.1-70B-Instruct"
        )
    )
    embed_url: str = field(
        default_factory=lambda: os.environ.get(
            "MOP_EMBED_URL", "http://localhost:8200/embed"
        )
    )

    request_timeout: float = 60.0
    max_tokens: int = 10
    temperature: float = 0.0
    small_logprobs: bool = True
    top_logprobs: int = 20

    small_prompt_mode: str = "3way"
    """Prompt regime for the small LLM. ``"3way"`` (default) uses
    True/False/Unsure with 3-way logprob extraction; ``"2way"`` is an
    opt-in Yes/No ablation. The large LLM always uses the clean 2-way
    prompt."""

    small_prompt_style: str = "default"
    """System-prompt style for the small LLM. ``"default"`` is the standard
    wording; ``"skeptical"`` is an anti-Yes-bias fact-checking framing
    (FEVER) that lets the small model answer "No" when evidence is absent.
    See pipeline.common.prompts.SMALL_3WAY_SYSTEMS."""

    large_prediction_cache: list[str] = field(default_factory=list)
    """List of CSV / JSONL source paths for cached zero-shot 70B
    predictions. When non-empty, escalations look up the cached
    prediction instead of calling the live large LLM."""

    large_prediction_cache_persist: str | None = None
    """Optional JSONL path to which fresh (query, doc, prediction, raw)
    rows are appended on cache misses."""

    large_extra_body: dict = field(
        default_factory=lambda: json.loads(
            os.environ.get("MOP_LARGE_EXTRA_BODY", "{}")
        )
    )
    """Extra request body forwarded verbatim to every LIVE large-model call
    made by the online experience generator (synthesis). Used to pass a
    served model's provider-specific switches — e.g. Qwen3.5's
    ``{"chat_template_kwargs": {"enable_thinking": false}}`` to suppress the
    reasoning trace so synthesis stays short and the generation-cost metric is
    clean. Settable via the ``MOP_LARGE_EXTRA_BODY`` env var (JSON) or
    ``llms.large_extra_body`` in the config. Empty for Llama-family models."""


@dataclass
class RunnerConfig:
    """Streaming-runner concurrency + cadence."""

    workers: int = 1
    # Concurrency of the background experience-generation executor
    # (workers>1 only; each generation is a serial explain+synthesize LLM
    # chain that overlaps point processing). Kept smaller than `workers`
    # so generation can't starve the small-model server.
    gen_workers: int = 4
    progress_every: int = 25
    # Frozen-inference mode: the runner still routes and records the router's
    # score, but performs NO online updates (no router refit, no retriever
    # counter update, no experience generation). Used to evaluate a pretrained
    # system with all learned components frozen.
    frozen: bool = False


@dataclass
class ReportConfig:
    """What to write and where."""

    output: str = ""
    """Required. Path (absolute or repo-relative) for the main results
    JSON."""

    record_sidecar: bool = False
    """Write a sidecar JSON capturing rich generator + retriever
    recordings. The sidecar lands at ``<output>.recording.json`` unless
    ``record_output`` is set explicitly."""

    record_output: str | None = None

    frontier_baseline: str | None = None
    """Optional path to a JSON file containing the no-pretrain Pareto
    frontier. When set, the report includes ``delta_acc`` / ``delta_f1``
    at the run's escalation rate."""


@dataclass
class EvalConfig:
    """Top-level config — one of these is loaded from YAML / JSON per run."""

    data: DataConfig
    experience_pool: ExperiencePoolConfig
    experience_retr: StageSpec
    fewshot_retr: StageSpec
    router: StageSpec
    reporting: ReportConfig

    # Optional stages (None means "skip").
    experience_gen: StageSpec | None = None
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    llms: LLMConfig = field(default_factory=LLMConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
