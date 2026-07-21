"""Multi-document synthetic queries.

Mines a contrastive yes/no predicate (as in ``doc_contrastive``) and applies
that ONE predicate to MANY documents. All those documents share the predicate
text, hence the same query, so an experience distilled on one document is
retrievable on the others during pretraining — unlike the singleton queries
``doc_contrastive`` produces.

For each accepted predicate p:
  * the discriminating pair (d+, d-) is always included (Yes / No);
  * ``docs_per_query - 2`` further documents are sampled from the corpus, with
    unknown silver label (the large model fixes their label during active
    labeling), giving a non-degenerate label spread for p across the corpus.

Output is the same ``Candidate`` JSONL as ``doc_contrastive`` — the pretrain
loop derives the shared query name from the predicate text, so no schema
change is needed.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .doc_contrastive import (
    _GEN_PROMPT,
    _VERIFY_PROMPT,
    _parse_predicate_json,
    _parse_yes_no,
)
from .types import Candidate


@dataclass
class MultiDocConfig:
    n_predicates: int = 40        # number of distinct predicates (queries)
    docs_per_query: int = 50      # documents each predicate is applied to
    n_clusters: int = 20
    rng_seed: int = 0
    verify_with_order_swap: bool = True
    max_gen_attempts_mult: int = 6


def _cand_id(predicate: str, doc_id: int) -> str:
    h = hashlib.sha1(f"{predicate}|{doc_id}".encode("utf-8")).hexdigest()[:16]
    return f"md_{h}"


def synthesize_multidoc(
    *,
    doc_collection: dict[int, str],
    embed_fn: Callable[[list[str]], np.ndarray],
    generate_fn: Callable[[str], str],
    config: MultiDocConfig | None = None,
) -> list[Candidate]:
    """Generate ``n_predicates`` predicates, each applied to ``docs_per_query``
    documents. Returns the flat Candidate list (shared predicate text groups
    documents into one query downstream)."""
    cfg = config or MultiDocConfig()
    rng = np.random.default_rng(cfg.rng_seed)
    from sklearn.cluster import KMeans

    doc_ids = sorted(doc_collection.keys())
    if len(doc_ids) < max(cfg.n_clusters, cfg.docs_per_query):
        raise ValueError(f"corpus too small: {len(doc_ids)} docs")
    texts = [doc_collection[d] for d in doc_ids]
    embs = embed_fn(texts)
    km = KMeans(n_clusters=cfg.n_clusters,
                random_state=int(rng.integers(0, 2**31 - 1)), n_init=10).fit(embs)
    by_cluster: dict[int, list[int]] = {}
    for did, lab in zip(doc_ids, km.labels_):
        by_cluster.setdefault(int(lab), []).append(int(did))
    cluster_ids = [c for c, lst in by_cluster.items() if lst]

    accepted: list[Candidate] = []
    seen_predicates: set[str] = set()
    n_gen = n_parse_fail = n_verify_fail = 0
    t0 = time.time()
    attempts = 0
    max_attempts = cfg.n_predicates * cfg.max_gen_attempts_mult
    pred_idx = 0
    while len(seen_predicates) < cfg.n_predicates and attempts < max_attempts:
        attempts += 1
        ci, cj = rng.choice(cluster_ids, size=2, replace=False)
        d_plus = int(rng.choice(by_cluster[int(ci)]))
        d_minus = int(rng.choice(by_cluster[int(cj)]))
        if d_plus == d_minus:
            continue
        n_gen += 1
        prompt = _GEN_PROMPT.format(
            doc_plus=doc_collection[d_plus],
            doc_minus=doc_collection[d_minus])
        try:
            parsed = _parse_predicate_json(generate_fn(prompt))
        except Exception:
            parsed = None
        if parsed is None or not isinstance(parsed.get("predicate"), str):
            n_parse_fail += 1
            continue
        predicate = parsed["predicate"].strip()
        if not predicate or len(predicate) > 500 or predicate in seen_predicates:
            n_parse_fail += 1
            continue
        if cfg.verify_with_order_swap:
            try:
                v_plus = _parse_yes_no(generate_fn(_VERIFY_PROMPT.format(
                    doc=doc_collection[d_plus],
                    predicate=predicate)))
                v_minus = _parse_yes_no(generate_fn(_VERIFY_PROMPT.format(
                    doc=doc_collection[d_minus],
                    predicate=predicate)))
            except Exception:
                n_verify_fail += 1
                continue
            if v_plus != "Yes" or v_minus != "No":
                n_verify_fail += 1
                continue

        seen_predicates.add(predicate)
        # Documents this predicate is applied to: the discriminating pair, plus
        # a random sample of the rest of the corpus.
        chosen = {d_plus, d_minus}
        others = [d for d in doc_ids if d not in chosen]
        rng.shuffle(others)
        for d in others[: max(0, cfg.docs_per_query - 2)]:
            chosen.add(int(d))
        for d in sorted(chosen):
            if d == d_plus:
                exp = "Yes"
            elif d == d_minus:
                exp = "No"
            else:
                exp = ""   # unknown; the large model labels it during labeling
            accepted.append(Candidate(
                cand_id=_cand_id(predicate, d),
                predicate=predicate,
                doc_id=int(d),
                expected_answer=exp,
                source="multidoc",
                meta={"predicate_idx": pred_idx, "paired_plus": d_plus,
                      "paired_minus": d_minus,
                      "role": ("plus" if d == d_plus else
                               "minus" if d == d_minus else "extra")},
            ))
        pred_idx += 1

    diagnostics = {
        "n_predicates": len(seen_predicates),
        "docs_per_query": cfg.docs_per_query,
        "n_candidates": len(accepted),
        "n_gen_attempts": n_gen,
        "n_parse_fail": n_parse_fail,
        "n_verify_fail": n_verify_fail,
        "wall_seconds": round(time.time() - t0, 2),
    }
    if accepted:
        accepted[0].meta["__multidoc_diagnostics__"] = diagnostics
    return accepted
