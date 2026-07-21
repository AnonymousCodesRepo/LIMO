"""Driver: multi-document synthetic query synthesis.

Generates ``n_predicates`` corpus-realistic yes/no predicates (the contrastive
way) and applies EACH to ``docs_per_query`` documents, so every predicate is a
real-query-like group of (predicate, document) pairs. Required for
same-source-query retrieval to transfer experiences during pretraining.

Usage:

    bash run.sh scripts/pretrain_offline/synth_queries_multidoc.py \\
        --output <dir>/candidates.jsonl --dataset opp115 \\
        --n-predicates 40 --docs-per-query 30 --n-clusters 20 --rng-seed 0
"""

from __future__ import annotations
import os

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.data import DATASETS  # noqa: E402
from pipeline.common.embeddings import EmbeddingClient  # noqa: E402
from pipeline.query_synth import dump_candidates  # noqa: E402
from pipeline.query_synth.multidoc import (  # noqa: E402
    MultiDocConfig,
    synthesize_multidoc,
)

LARGE_MODEL_PATH = os.environ.get("MOP_LARGE_MODEL", "Meta-Llama-3.1-70B-Instruct")
LARGE_MODEL_URL = os.environ.get("MOP_LARGE_URL", "http://localhost:8102/v1")
EMBED_URL = os.environ.get("MOP_EMBED_URL", "http://localhost:8200/embed")


def _load_docs(dataset: str) -> dict[int, str]:
    spec = DATASETS[dataset]
    merged = pd.read_csv(spec.doc_dir / spec.merged_file)
    return {int(r.document_id): str(getattr(r, spec.text_col))
            for r in merged.itertuples()}


def _make_generate_fn(*, client, model, temperature, max_tokens, request_timeout):
    def _gen(prompt: str) -> str:
        resp = client.with_options(timeout=request_timeout).chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens)
        return (resp.choices[0].message.content or "").strip()
    return _gen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="cuad", choices=sorted(DATASETS))
    p.add_argument("--n-predicates", type=int, default=40)
    p.add_argument("--docs-per-query", type=int, default=50)
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--large-url", default=LARGE_MODEL_URL)
    p.add_argument("--large-model", default=LARGE_MODEL_PATH)
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    docs = _load_docs(args.dataset)
    print(f"[multidoc] loaded {len(docs)} docs (dataset={args.dataset})",
          flush=True)

    embed_client = EmbeddingClient(url=EMBED_URL)
    large_client = OpenAI(base_url=args.large_url, api_key="dummy")
    gen_fn = _make_generate_fn(
        client=large_client, model=args.large_model,
        temperature=args.temperature, max_tokens=args.max_tokens,
        request_timeout=args.request_timeout)

    cfg = MultiDocConfig(
        n_predicates=args.n_predicates, docs_per_query=args.docs_per_query,
        n_clusters=args.n_clusters,
        rng_seed=args.rng_seed, verify_with_order_swap=not args.no_verify)
    print(f"[multidoc] config: {cfg}", flush=True)

    t0 = time.time()
    cands = synthesize_multidoc(
        doc_collection=docs, embed_fn=lambda t: embed_client.embed(t),
        generate_fn=gen_fn, config=cfg)
    n = dump_candidates(out_path, cands)
    diag = (cands[0].meta.get("__multidoc_diagnostics__")
            if cands and "__multidoc_diagnostics__" in cands[0].meta else None)
    print(f"[multidoc] wrote {n} candidates -> {out_path} in "
          f"{time.time() - t0:.1f}s", flush=True)
    if diag:
        print(f"[multidoc] diagnostics: {json.dumps(diag, indent=2)}", flush=True)
    with open(out_path.with_suffix(".multidoc.json"), "w") as f:
        json.dump({"n_candidates": n, "config": cfg.__dict__,
                   "diagnostics": diag}, f, indent=2)


if __name__ == "__main__":
    main()
