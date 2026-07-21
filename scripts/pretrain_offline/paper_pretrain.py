"""Unsupervised Self-Pretraining (LIMO Section 6).

Runs the integrated cost-aware active-labeling loop in
``pipeline/label_synth/paper_pretrain.py`` and writes the warm-start
artifacts the online system consumes:

    <out-dir>/rollouts.jsonl       labeled + augmented router training rows
    <out-dir>/care_snapshot.json   warm-started utility estimator
    <out-dir>/experiences.jsonl    bootstrapped experience pool E_0
    <out-dir>/router_lgbm.pkl      router checkpoint (LightGBMRouterCheckpoint)
    <out-dir>/manifest.json        provenance + paths + diagnostics

Stage 1 (contrastive query synthesis) is produced by
``scripts/pretrain_offline/synth_queries_multidoc.py`` -- pass its
``candidates.jsonl`` via --candidates. The experience pool starts EMPTY and is
grown from the small/large disagreements discovered during labeling (the "no
historical queries" bootstrap).

Usage:

    bash run.sh scripts/pretrain_offline/paper_pretrain.py \\
        --candidates  <dir>/candidates.jsonl \\
        --dataset     opp115 \\
        --out-dir     <dir>/paper_pretrain/opp115 \\
        --rounds 4 --budget-per-round 100 100 100 100 \\
        --k-experiences 8 --feature-dim 34
"""

from __future__ import annotations
import os

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import experience as experience_stage  # noqa: E402
from pipeline.common.confidence import (  # noqa: E402
    extract_yes_no_features,
    extract_yes_no_features_3way,
)
from pipeline.common.embeddings import EmbeddingClient  # noqa: E402
from pipeline.common.prompts import (  # noqa: E402
    build_replay_messages,
    build_replay_messages_3way,
    parse_yes_no,
)
from pipeline.experience_generator.online_discrepancy import (  # noqa: E402
    OnlineDiscrepancyGenerator,
)
from pipeline.experience_retriever import build as build_exp_retriever  # noqa: E402
from pipeline.label_synth.paper_pretrain import (  # noqa: E402
    PaperPretrainConfig,
    run_paper_pretrain,
)
from pipeline.query_synth import load_candidates  # noqa: E402
from pipeline.trainer.retriever.offline_care import CARESnapshot  # noqa: E402
from pipeline.trainer.router.offline_lgbm import train_router_offline  # noqa: E402
from eval.data import DATASETS  # noqa: E402


SMALL_MODEL_PATH = os.environ.get("MOP_SMALL_MODEL", "Qwen3.5-0.8B")
SMALL_MODEL_URL = os.environ.get("MOP_SMALL_URL", "http://localhost:8105/v1")
LARGE_MODEL_PATH = os.environ.get("MOP_LARGE_MODEL", "Meta-Llama-3.1-70B-Instruct")
LARGE_MODEL_URL = os.environ.get("MOP_LARGE_URL", "http://localhost:8102/v1")
EMBED_URL = os.environ.get("MOP_EMBED_URL", "http://localhost:8200/embed")


def _load_docs(dataset: str) -> dict[int, str]:
    spec = DATASETS[dataset]
    merged = pd.read_csv(spec.doc_dir / spec.merged_file)
    return {int(r.document_id): str(getattr(r, spec.text_col))
            for r in merged.itertuples()}


def _make_small_call_fn(*, client, model, temperature, max_tokens, timeout,
                        prompt_mode="3way"):
    """(predicate, doc_text, qdesc, experiences) -> small features.

    ``prompt_mode`` selects the small prompt/feature regime and MUST match the
    eval-time ``llms.small_prompt_mode`` so the rollout feature distributions
    match what the router sees online (the point-feature block shares its
    keys across both modes; only the distributions differ).
    """
    build = (build_replay_messages if prompt_mode == "2way"
             else build_replay_messages_3way)
    extract = (extract_yes_no_features if prompt_mode == "2way"
               else extract_yes_no_features_3way)

    def _call(predicate, doc_text, qdesc, experiences):
        msgs = build(
            document_text=doc_text, query_description=qdesc,
            retrieved_experiences=(experiences or None), fewshot_demos=None)
        resp = client.with_options(timeout=timeout).chat.completions.create(
            model=model, messages=msgs, temperature=temperature,
            max_tokens=max_tokens, logprobs=True, top_logprobs=20)
        return extract(resp)
    return _call


def _make_large_call_fn(*, client, model, temperature, max_tokens, timeout):
    """(predicate, doc_text, qdesc) -> (prediction, raw). Clean 2-way prompt."""
    def _call(predicate, doc_text, qdesc):
        msgs = build_replay_messages(
            document_text=doc_text, query_description=qdesc,
            retrieved_experiences=None, fewshot_demos=None)
        resp = client.with_options(timeout=timeout).chat.completions.create(
            model=model, messages=msgs, temperature=temperature,
            max_tokens=max_tokens)
        raw = (resp.choices[0].message.content or "").strip()
        return parse_yes_no(raw), raw
    return _call


def _snapshot_retriever(retriever, out_path: Path, config: dict) -> dict:
    snap = CARESnapshot(
        mu=retriever._blr_mu.tolist(),
        A_inv=[row.tolist() for row in retriever._blr_A_inv],
        n_blr_updates=int(retriever._blr_n_updates),
        feature_dim=int(retriever._d_care),
        stats=dict(getattr(retriever, "_care_stats", {})),
        config=config,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(snap), indent=2))
    return asdict(snap)


def _dump_experiences(experiences, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for e in experiences:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True,
                   help="candidates.jsonl from synth_queries_multidoc (Stage 1)")
    p.add_argument("--dataset", default="cuad", choices=sorted(DATASETS))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--budget-per-round", type=int, nargs="+",
                   default=[100, 100, 100, 100],
                   help="paper: a fixed budget distributed evenly (4 x 100)")
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="exploration mix fraction; paper uses 0 "
                        "(pure proportional sampling)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-experiences", type=int, default=8)
    p.add_argument("--scoring-pool-size", type=int, default=400,
                   help="uniform candidate subset scored per round "
                        "(paper Section 6.2 Step 1); 0 = score all remaining")
    p.add_argument("--no-augment", action="store_true",
                   help="disable leave-out-experience augmentation (ablation)")
    p.add_argument("--aug-variants", nargs="+", default=["loo"],
                   choices=["loo", "noexp"],
                   help="augmentation variants to emit (default: loo, "
                        "the paper's leave-one-out set)")
    p.add_argument("--same-query-only", action="store_true", default=False,
                   help="restrict retrieval to experiences from the SAME "
                        "query (restrict_to_source_query); off by default "
                        "-- cross-query retrieval matches online serving")
    p.add_argument("--cross-query", dest="same_query_only", action="store_false",
                   help="allow cross-query retrieval (the default)")
    p.add_argument("--no-seed", action="store_true",
                   help="disable experience-pool seeding (ablation: empty pool)")
    p.add_argument("--max-aug-small-calls", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--min-for-refit", type=int, default=100)
    p.add_argument("--acq-conf-cap", type=float, default=None,
                   help="only label candidates with small-model confidence < "
                        "this. Default off = paper behavior.")
    # Experience seeding (online discrepancy generator) knobs.
    p.add_argument("--max-per-query", type=int, default=50)
    p.add_argument("--domain", default=None,
                   help="experience-prompt domain hint; default maps from "
                        "--dataset (cuad=legal, opp115=privacy-policy, "
                        "hoc=cancer-biology)")
    # Router fit (final checkpoint) knobs.
    p.add_argument("--feature-dim", type=int, default=34,
                   help="34 = joint (point+CARE), 14 = point-only ablation")
    p.add_argument("--router-n-estimators", type=int, default=200)
    p.add_argument("--router-learning-rate", type=float, default=0.05)
    p.add_argument("--router-num-leaves", type=int, default=31)
    p.add_argument("--router-min-child-samples", type=int, default=20)
    p.add_argument("--router-reg-lambda", type=float, default=1.0)
    p.add_argument("--router-calibration", default="none",
                   choices=["auto", "sigmoid", "isotonic", "none"])
    p.add_argument("--router-class-weight", default="balanced",
                   choices=["none", "balanced"])
    p.add_argument("--router-backend", default="sklearn_hgb",
                   help="sklearn_hgb | lightgbm | rf | mlp")
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[256, 256],
                   help="(backend=mlp) hidden-layer sizes")
    p.add_argument("--mlp-alpha", type=float, default=1e-4,
                   help="(backend=mlp) L2 regularization")
    p.add_argument("--mlp-max-iter", type=int, default=300,
                   help="(backend=mlp) max training iterations")
    p.add_argument("--no-ips", action="store_true",
                   help="train the final router head with uniform example "
                        "weights (no inverse-propensity correction)")
    p.add_argument("--blend-lambda", type=float, default=20.0)
    p.add_argument("--prompt-mode", default="3way", choices=["3way", "2way"],
                   help="small prompt/feature regime; MUST match eval-time "
                        "llms.small_prompt_mode")
    p.add_argument("--small-temperature", type=float, default=0.0)
    p.add_argument("--small-max-tokens", type=int, default=10)
    p.add_argument("--large-temperature", type=float, default=0.0)
    p.add_argument("--large-max-tokens", type=int, default=10)
    p.add_argument("--request-timeout", type=float, default=60.0)
    p.add_argument("--small-url", default=SMALL_MODEL_URL)
    p.add_argument("--small-model", default=SMALL_MODEL_PATH)
    p.add_argument("--large-url", default=LARGE_MODEL_URL)
    p.add_argument("--large-model", default=LARGE_MODEL_PATH)
    p.add_argument("--restrict-docs", default="",
                   help="optional JSON list of doc_ids to keep (smoke test)")
    p.add_argument("--max-candidates", type=int, default=0,
                   help="keep only the first N candidates (0 = all; smoke test)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"
    snapshot_path = out_dir / "care_snapshot.json"
    pool_path = out_dir / "experiences.jsonl"
    ckpt_path = out_dir / "router_lgbm.pkl"
    manifest_path = out_dir / "manifest.json"

    cands = load_candidates(args.candidates)
    if args.restrict_docs:
        keep = set(int(x) for x in json.loads(args.restrict_docs))
        cands = [c for c in cands if int(c.doc_id) in keep]
    if args.max_candidates and len(cands) > args.max_candidates:
        cands = cands[: args.max_candidates]
    if not cands:
        raise SystemExit(f"no candidates after filtering: {args.candidates}")
    docs = _load_docs(args.dataset)
    print(f"[paper] {len(cands)} candidates, {len(docs)} docs "
          f"(dataset={args.dataset})", flush=True)

    embed = EmbeddingClient(url=EMBED_URL)
    # CARE-PQ utility estimator starting from an EMPTY pool (bootstrap).
    retriever = build_exp_retriever(
        "care_pq", experiences=[], embed_client=embed,
        prior_precision=1.0, blend_lambda=float(args.blend_lambda),
        restrict_to_source_query=bool(args.same_query_only))
    print(f"[paper] CARE-PQ retriever built with empty pool "
          f"(same_query_only={args.same_query_only})", flush=True)

    small_client = OpenAI(base_url=args.small_url, api_key="dummy")
    large_client = OpenAI(base_url=args.large_url, api_key="dummy")
    small_call = _make_small_call_fn(
        client=small_client, model=args.small_model,
        temperature=args.small_temperature, max_tokens=args.small_max_tokens,
        timeout=args.request_timeout, prompt_mode=args.prompt_mode)
    large_call = _make_large_call_fn(
        client=large_client, model=args.large_model,
        temperature=args.large_temperature, max_tokens=args.large_max_tokens,
        timeout=args.request_timeout)

    generator = None
    if not args.no_seed:
        domain = args.domain or {"cuad": "legal", "opp115": "privacy-policy",
                                 "hoc": "cancer-biology"}.get(args.dataset,
                                                              "legal")
        generator = OnlineDiscrepancyGenerator(
            small_client=small_client, small_model=args.small_model,
            large_client=large_client, large_model=args.large_model,
            max_per_query=int(args.max_per_query),
            temperature=0.0, request_timeout=args.request_timeout,
            domain=domain)
        print(f"[paper] experience seeding ON "
              f"(max_per_query={args.max_per_query}, domain={domain})",
              flush=True)
    else:
        print("[paper] experience seeding OFF (ablation: pool stays empty)",
              flush=True)

    cfg = PaperPretrainConfig(
        n_rounds=int(args.rounds),
        budget_per_round=tuple(int(b) for b in args.budget_per_round),
        epsilon=float(args.epsilon), seed=int(args.seed),
        k_experiences=int(args.k_experiences),
        scoring_pool_size=(int(args.scoring_pool_size)
                           if args.scoring_pool_size > 0 else None),
        augment_leave_out=(not args.no_augment),
        aug_variants=tuple(args.aug_variants),
        max_aug_small_calls=int(args.max_aug_small_calls),
        workers=int(args.workers), min_for_refit=int(args.min_for_refit),
        n_estimators=int(args.router_n_estimators),
        learning_rate=float(args.router_learning_rate),
        num_leaves=int(args.router_num_leaves),
        min_child_samples=int(args.router_min_child_samples),
        reg_lambda=float(args.router_reg_lambda),
        calibration_method=args.router_calibration,
        class_weight=args.router_class_weight,
        acq_conf_cap=args.acq_conf_cap)
    print(f"[paper] config: {cfg}", flush=True)

    t0 = time.time()
    diag = run_paper_pretrain(
        candidates=cands, docs=docs, small_call_fn=small_call,
        large_call_fn=large_call, retriever=retriever, generator=generator,
        output_rollouts_path=rollouts_path, config=cfg,
        progress_cb=lambda m: print(f"[paper] {m}", flush=True))
    diag["loop_seconds"] = round(time.time() - t0, 1)

    # Snapshot the warm-started utility estimator + dump the bootstrapped pool.
    snap = _snapshot_retriever(
        retriever, snapshot_path,
        config={"source": "paper_pretrain", "dataset": args.dataset,
                "k_experiences": int(args.k_experiences),
                "blend_lambda": float(args.blend_lambda),
                "prompt_mode": args.prompt_mode})
    n_pool = _dump_experiences(list(retriever.experiences), pool_path)
    print(f"[paper] snapshot n_blr_updates={snap['n_blr_updates']}, "
          f"pool={n_pool} experiences", flush=True)

    # Train the final router checkpoint from the rollout buffer.
    print(f"[paper] training router (feature_dim={args.feature_dim})...",
          flush=True)
    metrics = train_router_offline(
        rollouts_path=rollouts_path, output_path=ckpt_path,
        n_estimators=int(args.router_n_estimators),
        learning_rate=float(args.router_learning_rate),
        num_leaves=int(args.router_num_leaves),
        min_child_samples=int(args.router_min_child_samples),
        reg_lambda=float(args.router_reg_lambda),
        calibration_method=args.router_calibration,
        class_weight=args.router_class_weight,
        ips=(not args.no_ips),
        mlp_hidden=tuple(args.mlp_hidden), mlp_alpha=float(args.mlp_alpha),
        mlp_max_iter=int(args.mlp_max_iter),
        backend=args.router_backend, feature_dim=int(args.feature_dim),
        seed=int(args.seed))

    manifest = {
        "method": "paper_pretrain (LIMO Section 6, joint warm-start)",
        "dataset": args.dataset,
        "candidates_path": args.candidates,
        "feature_dim": int(args.feature_dim),
        "artifacts": {
            "rollouts": str(rollouts_path),
            "care_snapshot": str(snapshot_path),
            "experiences": str(pool_path),
            "router_checkpoint": str(ckpt_path),
        },
        "n_experiences_final": n_pool,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(cfg).items()},
        "loop_diagnostics": diag,
        "router_metrics": metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[paper] DONE. manifest -> {manifest_path}", flush=True)
    print(json.dumps({"n_main": diag.get("n_main_rollouts"),
                      "n_aug": diag.get("n_aug_rollouts"),
                      "n_seeded": diag.get("n_experiences_seeded"),
                      "final_pool": diag.get("final_pool_size"),
                      "loop_seconds": diag.get("loop_seconds"),
                      "router_val": metrics.get("val", metrics)}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
