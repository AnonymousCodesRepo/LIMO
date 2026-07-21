"""End-to-end eval orchestration.

Parses a YAML / JSON config into an ``EvalConfig``, builds every stage
through the registry-style ``builders`` module, runs the streaming
cascade via ``pipeline.runner.StreamingRunner``, and writes a single
results JSON (plus an optional sidecar with rich generator + retriever
recordings). Stages are independently swappable via the YAML; nothing is
method-specific here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.common.embeddings import EmbeddingClient
from pipeline.common.recording import (
    RecordingDiscrepancyGenerator,
    RecordingExperienceRetriever,
)
from pipeline.runner import StreamingRunner

from .builders import (
    build_experience_pool,
    build_experience_retriever,
    build_fewshot_retriever,
    build_generator,
    build_large_client,
    build_large_prediction_cache,
    build_router_stage,
    build_small_client,
    router_needs_large,
)
from .config import EvalConfig
from .data import DATASETS, load_points, resolve_query_list
from .pretrain import fit_warm_start
from .reporting import (
    build_results,
    print_summary,
    write_recording_sidecar,
    write_results,
)


@dataclass
class PipelineBundle:
    """Everything ``run()`` needs after the stages are wired together.

    Exposed so alternative drivers (e.g. the two-phase router-detector study
    in ``eval/router_detector_eval.py``) can build the exact same pipeline,
    then take control of how ``runner.run`` is called instead of running the
    single straight-through pass ``run()`` does.
    """

    runner: StreamingRunner
    data: list
    queries: list
    router: Any
    generator: Any
    exp_retriever: Any
    large_cache: Any


def build_pipeline(cfg: EvalConfig) -> PipelineBundle:
    """Wire every stage from ``cfg`` without running it."""
    print("\n=== EVAL CONFIG ===")
    from dataclasses import asdict
    print(json.dumps(asdict(cfg), indent=2))

    # If the user requested an end-to-end "fit from scratch" pretrain,
    # run synth → joint pretrain (resumable: router + utility estimator
    # trained together, paper Section 6) and then proceed as if mode
    # were "load" against the freshly trained artifacts.
    if cfg.pretrain.mode == "fit":
        produced_ckpt, produced_snap = fit_warm_start(
            cfg.pretrain, dataset=cfg.data.dataset
        )
        if (
            cfg.pretrain.router_checkpoint
            and cfg.pretrain.router_checkpoint != produced_ckpt
        ):
            print(
                f"[pretrain.fit] overriding pretrain.router_checkpoint "
                f"{cfg.pretrain.router_checkpoint!r} with the just-trained "
                f"checkpoint at {produced_ckpt!r}.",
                flush=True,
            )
        cfg.pretrain.router_checkpoint = produced_ckpt
        cfg.pretrain.retriever_snapshot = produced_snap

    embed_client = EmbeddingClient(url=cfg.llms.embed_url)

    experiences = build_experience_pool(cfg.experience_pool)
    print(f"[cfg] experiences loaded: {len(experiences)}", flush=True)

    exp_retriever, k_exp = build_experience_retriever(
        cfg.experience_retr,
        experiences=experiences,
        embed_client=embed_client,
        point_cache_path=cfg.data.point_embedding_cache,
        pretrain=cfg.pretrain,
    )

    if cfg.reporting.record_sidecar:
        exp_retriever = RecordingExperienceRetriever(exp_retriever)

    fs_retriever, k_fs = build_fewshot_retriever(
        cfg.fewshot_retr, embed_client=embed_client,
    )

    queries = resolve_query_list(cfg.data.queries, dataset=cfg.data.dataset)
    print(f"[cfg] queries ({len(queries)}): {','.join(queries)}", flush=True)
    data = load_points(
        queries,
        dataset=cfg.data.dataset,
        within_query_order=cfg.data.within_query_order,
        order_seed=cfg.data.order_seed,
        limit_per_query=cfg.data.limit_per_query,
    )
    print(f"[cfg] data points (in-order): {len(data)}", flush=True)
    for q in queries:
        n = sum(1 for p in data if p.query_name == q)
        print(f"[cfg]   {q}: {n}", flush=True)

    # Built before the router because some routers (e.g. AutoMix, which issues
    # its own small-model self-verification calls) need the small client as a
    # constructor dependency.
    small_client = build_small_client(cfg.llms)

    router = build_router_stage(
        cfg.router,
        experience_retriever=exp_retriever,
        embed_client=embed_client,
        pretrain=cfg.pretrain,
        small_client=small_client,
        small_model=cfg.llms.small_model,
        request_timeout=cfg.llms.request_timeout,
        dataset=cfg.data.dataset,
    )

    # Stage prefetch (router AND retriever both expose .prefetch via
    # their base class — default is a no-op).
    router.prefetch(data)
    exp_retriever.prefetch(data)

    needs_large = router_needs_large(
        cfg.router, has_online_generator=(cfg.experience_gen is not None),
    )
    large_client = build_large_client(cfg.llms) if needs_large else None

    generator = build_generator(
        cfg.experience_gen,
        small_client=small_client,
        small_model=cfg.llms.small_model,
        large_client=large_client,
        large_model=cfg.llms.large_model if large_client is not None else None,
        request_timeout=cfg.llms.request_timeout,
        large_extra_body=cfg.llms.large_extra_body,
    )
    if cfg.reporting.record_sidecar and generator is not None:
        if cfg.experience_gen.name != "online_discrepancy":
            raise ValueError(
                "record_sidecar=True only supports the online_discrepancy "
                f"generator; got {cfg.experience_gen.name!r}"
            )
        generator = RecordingDiscrepancyGenerator(
            small_client=small_client,
            small_model=cfg.llms.small_model,
            large_client=large_client,
            large_model=cfg.llms.large_model,
            request_timeout=cfg.llms.request_timeout,
            **cfg.experience_gen.kwargs,
        )
    if generator is not None:
        # Per-query totals for the cost-aware position gate (t* rule).
        generator.prefetch(data)

    large_cache = build_large_prediction_cache(cfg.llms)

    runner = StreamingRunner(
        experience_retriever=exp_retriever,
        fewshot_retriever=fs_retriever,
        router=router,
        small_client=small_client,
        small_model=cfg.llms.small_model,
        large_client=large_client,
        large_model=cfg.llms.large_model if large_client is not None else None,
        k_experiences=k_exp,
        k_fewshot=k_fs,
        max_tokens=cfg.llms.max_tokens,
        temperature=cfg.llms.temperature,
        request_timeout=cfg.llms.request_timeout,
        small_logprobs=cfg.llms.small_logprobs,
        top_logprobs=cfg.llms.top_logprobs,
        workers=cfg.runner.workers,
        gen_workers=cfg.runner.gen_workers,
        experience_generator=generator,
        large_prediction_cache=large_cache,
        small_prompt_mode=cfg.llms.small_prompt_mode,
        small_prompt_style=cfg.llms.small_prompt_style,
        large_system_prompt=DATASETS[cfg.data.dataset].large_system_prompt,
    )
    runner.frozen = bool(cfg.runner.frozen)

    print(
        f"[cfg] small_prompt_mode={cfg.llms.small_prompt_mode} "
        f"style={cfg.llms.small_prompt_style}; "
        "large prompt is always clean 2-way (no experiences, no fewshot)",
        flush=True,
    )

    return PipelineBundle(
        runner=runner,
        data=data,
        queries=queries,
        router=router,
        generator=generator,
        exp_retriever=exp_retriever,
        large_cache=large_cache,
    )


def run(cfg: EvalConfig) -> dict[str, Any]:
    """Build the pipeline from ``cfg``, run it, write results, return them."""
    b = build_pipeline(cfg)

    t0 = time.perf_counter()
    records, _ = b.runner.run(b.data, progress_every=cfg.runner.progress_every)
    wall = time.perf_counter() - t0

    blob = build_results(
        config=cfg,
        queries=b.queries,
        records=records,
        wall_clock_seconds=wall,
        router=b.router,
        generator=b.generator,
        retriever=b.exp_retriever,
        online_experiences=b.runner.online_experiences,
        large_cache=b.large_cache,
    )

    if getattr(b.runner, "cost_aborted", False):
        blob["aborted"] = True
        blob["abort_info"] = getattr(b.runner, "_abort_info", None)

    out_path = write_results(blob, cfg.reporting)
    if cfg.reporting.record_sidecar:
        write_recording_sidecar(
            generator=b.generator,
            retriever=b.exp_retriever,
            output=out_path,
            record_output=cfg.reporting.record_output,
        )

    print_summary(
        blob=blob,
        queries=b.queries,
        needs_large=b.runner.large_client is not None,
        out_path=out_path,
    )
    return blob
