"""Stage builders.

Thin wrappers around the existing ``pipeline.<stage>.build(...)``
factories. The wrappers exist to inject runtime objects (LLM clients,
embedding clients, the retriever instance for joint-feature routers)
that don't belong in a YAML config, and to apply per-stage post-build
hooks (loading pretrained snapshots, retriever-side prefetch).

Each builder takes the matching ``StageSpec`` plus whatever runtime
context it needs and returns the constructed instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI

from pipeline import experience as experience_stage
from pipeline.common.embeddings import EmbeddingClient
from pipeline.common.large_prediction_cache import LargePredictionCache
from pipeline.common.types import DataPoint, Experience
from pipeline.experience_generator import build as build_exp_generator
from pipeline.experience_retriever import build as build_exp_retriever
from pipeline.fewshot_retriever import build as build_fs_retriever
from pipeline.router import build as build_router

from .config import (
    ExperiencePoolConfig,
    LLMConfig,
    PretrainConfig,
    StageSpec,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _abs_path(p: str) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


# ── experience pool ──────────────────────────────────────────────────────


def build_experience_pool(cfg: ExperiencePoolConfig) -> list[Experience]:
    if cfg.source in ("empty", "none", ""):
        return []
    return experience_stage.load(cfg.source)


# ── experience retriever ─────────────────────────────────────────────────


def build_experience_retriever(
    spec: StageSpec,
    *,
    experiences: list[Experience],
    embed_client: EmbeddingClient,
    point_cache_path: str | None = None,
    pretrain: PretrainConfig | None = None,
) -> tuple[Any, int]:
    """Construct an experience retriever and apply pretrain warm-start.

    Returns ``(retriever, k)`` where ``k`` is the number of experiences
    to retrieve per point (popped from ``spec.kwargs``; defaults to the
    paper's k = 8).
    """
    kwargs = dict(spec.kwargs)
    k = int(kwargs.pop("k", 8))

    point_cache = None
    if point_cache_path:
        from pipeline.experience_retriever.topk_semantic import load_point_cache
        point_cache = load_point_cache(_abs_path(point_cache_path))

    retriever = build_exp_retriever(
        spec.name,
        experiences=experiences,
        embed_client=embed_client,
        point_cache=point_cache,
        **kwargs,
    )

    if (
        pretrain is not None
        and pretrain.mode in ("load", "fit")
        and pretrain.retriever_snapshot is not None
        and hasattr(retriever, "load_offline_snapshot")
    ):
        retriever.load_offline_snapshot(str(_abs_path(pretrain.retriever_snapshot)))

    return retriever, k


# ── fewshot retriever ────────────────────────────────────────────────────


def build_fewshot_retriever(
    spec: StageSpec, *, embed_client: EmbeddingClient,
) -> tuple[Any, int]:
    """Returns ``(retriever, k)`` — same convention as the experience
    retriever builder."""
    kwargs = dict(spec.kwargs)
    k = int(kwargs.pop("k", 2))
    retriever = build_fs_retriever(
        spec.name, embed_client=embed_client, **kwargs,
    )
    return retriever, k


# ── router ───────────────────────────────────────────────────────────────


CUAD_CONFIDENCE_VETO: float | None = None


def build_router_stage(
    spec: StageSpec,
    *,
    experience_retriever: Any,
    embed_client: EmbeddingClient,
    pretrain: PretrainConfig | None = None,
    small_client: OpenAI | None = None,
    small_model: str | None = None,
    request_timeout: float = 60.0,
    dataset: str | None = None,
) -> Any:
    kwargs = dict(spec.kwargs)

    if spec.name == "lightgbm":
        kwargs.setdefault("experience_retriever", experience_retriever)
    elif spec.name == "automix":
        # AutoMix issues its own small-model self-verification calls, so it
        # needs the live small client + model name (runtime objects, not YAML).
        if small_client is None or small_model is None:
            raise ValueError(
                "router=automix requires small_client + small_model "
                "(inject them via build_router_stage)"
            )
        kwargs.setdefault("small_client", small_client)
        kwargs.setdefault("small_model", small_model)
        kwargs.setdefault("request_timeout", request_timeout)

    if (
        pretrain is not None
        and pretrain.mode in ("load", "fit")
        and pretrain.router_checkpoint is not None
        and spec.name == "lightgbm"
    ):
        kwargs.setdefault(
            "pretrained_checkpoint",
            str(_abs_path(pretrain.router_checkpoint)),
        )


    if spec.name == "lightgbm":
        if dataset == "cuad":
            if CUAD_CONFIDENCE_VETO is not None:
                kwargs.setdefault("confidence_veto", CUAD_CONFIDENCE_VETO)
        elif kwargs.get("confidence_veto") is not None:
            print(
                f"[build_router_stage] confidence_veto is cuad-only; ignoring "
                f"it for dataset={dataset!r}.",
                flush=True,
            )
            kwargs.pop("confidence_veto", None)

    return build_router(spec.name, **kwargs)


def router_needs_large(spec: StageSpec, has_online_generator: bool) -> bool:
    """Whether a large-LLM client must be reachable for this run."""
    if has_online_generator:
        return True
    return spec.name in (
        "all_big",
        "automix",
        "confidence_threshold",
        "lightgbm",
    )


# ── experience generator ─────────────────────────────────────────────────


def build_generator(
    spec: StageSpec | None,
    *,
    small_client: OpenAI,
    small_model: str,
    large_client: OpenAI | None,
    large_model: str | None,
    request_timeout: float,
    large_extra_body: dict | None = None,
) -> Any | None:
    if spec is None:
        return None
    if large_client is None or large_model is None:
        raise ValueError(
            f"experience_gen={spec.name!r} requires a reachable large model"
        )
    return build_exp_generator(
        spec.name,
        small_client=small_client,
        small_model=small_model,
        large_client=large_client,
        large_model=large_model,
        request_timeout=request_timeout,
        large_extra_body=large_extra_body,
        **spec.kwargs,
    )


# ── LLM clients + caches ─────────────────────────────────────────────────


def build_small_client(cfg: LLMConfig) -> OpenAI:
    client = OpenAI(base_url=cfg.small_url, api_key="dummy")
    try:
        client.with_options(timeout=15).chat.completions.create(
            model=cfg.small_model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=3, temperature=0.0,
        )
    except Exception as e:
        raise SystemExit(f"small model unreachable: {e}")
    return client


def build_large_client(cfg: LLMConfig) -> OpenAI:
    client = OpenAI(base_url=cfg.large_url, api_key="dummy")
    try:
        client.with_options(timeout=20).chat.completions.create(
            model=cfg.large_model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=3, temperature=0.0,
        )
    except Exception as e:
        raise SystemExit(f"large model unreachable: {e}")
    return client


def build_large_prediction_cache(cfg: LLMConfig) -> LargePredictionCache | None:
    if not cfg.large_prediction_cache:
        return None
    sources = [str(_abs_path(s)) for s in cfg.large_prediction_cache if s]
    persist = (
        str(_abs_path(cfg.large_prediction_cache_persist))
        if cfg.large_prediction_cache_persist else None
    )
    return LargePredictionCache(sources=sources, persist_to=persist)


