"""Method B — doc-contrastive predicate mining.

For each cross-cluster document pair (d+, d-), prompt the generator LLM to
emit a yes/no predicate p such that p(d+) = YES and p(d-) = NO on a
substantive (not stylistic) ground. Each accepted pair contributes two
candidates to the synthetic pool:

    (p, d+,  expected_answer="Yes")
    (p, d-,  expected_answer="No")

Forcing the predicate to *discriminate* a concrete doc pair avoids the
trivial ("is this a contract?") and under-grounded ("does this discuss
IP?") predicates that single-doc prompting produces: trivial predicates
fail the discrimination check in self-verification, and under-grounded
ones are filtered when their YES/NO labels collapse on re-prompting. Each
accepted pair also yields one YES and one NO label for free, keeping the
silver-label distribution balanced.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .types import Candidate


@dataclass
class DocContrastiveConfig:
    n_clusters: int = 20
    n_pairs: int = 40             # number of (d+, d-) pairs to generate
                                  # (paper §7.1: 40 synthesized queries)
    pairs_per_cluster_pair: int = 1  # how many doc pairs to sample per cluster pair
    rng_seed: int = 0
    # Verification: re-prompt the LLM with the doc order swapped to confirm
    # the predicate's truth value flips. Rejects predicates the LLM can't
    # answer consistently.
    verify_with_order_swap: bool = True
    # RandQuery ablation: sample the two documents UNIFORMLY at random
    # instead of the KMeans cross-cluster sampler (isolates the effect of
    # doc-pair selection). n_clusters is ignored when on.
    random_doc_pairs: bool = False


_GEN_PROMPT = """You will compare two documents from the same corpus.

DOCUMENT A:
{doc_plus}

DOCUMENT B:
{doc_minus}

Write a single yes/no question (a "predicate") about the substantive content of these documents such that:
- The answer for DOCUMENT A is YES.
- The answer for DOCUMENT B is NO.
- The predicate must concern substantive content (facts, statements, obligations, properties, or claims made in the text), not stylistic differences (formatting, length, or the names of people and organizations).
- The predicate must be answerable from the document text alone.

Output ONLY a JSON object with this exact shape (no preamble, no markdown fence):
{{"predicate": "<your yes/no question>", "rationale": "<one short sentence>"}}
"""


_VERIFY_PROMPT = """Read the document and answer the yes/no question with exactly one word: Yes or No.

DOCUMENT:
{doc}

QUESTION: {predicate}

Answer (Yes or No):"""


def _sample_doc_pairs_random(
    doc_ids: list[int],
    n_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    """RandQuery ablation: uniformly sample n_pairs distinct (d+, d-) pairs.

    No clustering. Cluster ids are reported as -1 to mark the random source.
    Returns: list of (doc_id_plus, doc_id_minus, -1, -1).
    """
    pairs: list[tuple[int, int, int, int]] = []
    attempts = 0
    max_attempts = n_pairs * 8
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        d_plus, d_minus = (int(x) for x in rng.choice(doc_ids, size=2, replace=False))
        pairs.append((d_plus, d_minus, -1, -1))
    return pairs


def _sample_doc_pairs(
    doc_ids: list[int],
    embeddings: np.ndarray,
    n_clusters: int,
    n_pairs: int,
    pairs_per_cluster_pair: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    """Cluster docs and sample cross-cluster pairs.

    Returns: list of (doc_id_plus, doc_id_minus, cluster_plus, cluster_minus).
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=int(rng.integers(0, 2**31 - 1)),
                n_init=10).fit(embeddings)
    labels = km.labels_
    by_cluster: dict[int, list[int]] = {}
    for did, lab in zip(doc_ids, labels):
        by_cluster.setdefault(int(lab), []).append(int(did))

    cluster_ids = [c for c, lst in by_cluster.items() if len(lst) >= 1]
    pairs: list[tuple[int, int, int, int]] = []
    attempts = 0
    max_attempts = n_pairs * 8
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        ci, cj = rng.choice(cluster_ids, size=2, replace=False)
        # Sample one doc from each cluster.
        if not by_cluster[int(ci)] or not by_cluster[int(cj)]:
            continue
        for _ in range(pairs_per_cluster_pair):
            d_plus = int(rng.choice(by_cluster[int(ci)]))
            d_minus = int(rng.choice(by_cluster[int(cj)]))
            if d_plus == d_minus:
                continue
            pairs.append((d_plus, d_minus, int(ci), int(cj)))
            if len(pairs) >= n_pairs:
                break
    return pairs


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_predicate_json(raw: str) -> dict | None:
    """Extract the JSON object from the generator's reply.

    Handles common deviations: leading 'json' fence, surrounding prose,
    trailing commentary. Returns None on unrecoverable failure.
    """
    raw = raw.strip()
    # Strip markdown fences if the LLM emitted any.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = _JSON_BLOCK_RE.search(raw)
    if m is None:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _parse_yes_no(raw: str) -> str | None:
    """Coerce an LLM reply to 'Yes' / 'No' / None. Permissive."""
    s = (raw or "").strip().lower()
    # Match leading token to avoid false-positive on docs containing both.
    m = re.match(r"\s*(yes|no)\b", s)
    if m is None:
        return None
    return "Yes" if m.group(1) == "yes" else "No"


def _cand_id(d_plus: int, d_minus: int, predicate: str, role: str) -> str:
    h = hashlib.sha1(
        f"{d_plus}|{d_minus}|{predicate}|{role}".encode("utf-8")
    ).hexdigest()[:16]
    return f"dc_{h}"


def synthesize(
    *,
    doc_collection: dict[int, str],
    embed_fn: Callable[[list[str]], np.ndarray],
    generate_fn: Callable[[str], str],
    config: DocContrastiveConfig | None = None,
) -> list[Candidate]:
    """Run Method B end-to-end.

    Parameters
    ----------
    doc_collection : {doc_id: doc_text}
    embed_fn       : callable(list[str]) -> (N, D) float array. Injected so
                     the synthesizer stays IO-agnostic.
    generate_fn    : callable(prompt: str) -> raw_reply: str, wrapping the
                     generator LLM endpoint (caller owns timeouts / retries).
    config         : DocContrastiveConfig

    Returns
    -------
    list[Candidate]  — accepted (predicate, doc) pairs after generation +
    optional order-swap verification. Each accepted predicate contributes
    exactly two candidates (one for d+ with expected="Yes", one for d-
    with expected="No").
    """
    cfg = config or DocContrastiveConfig()
    rng = np.random.default_rng(cfg.rng_seed)
    src = "doc_random" if cfg.random_doc_pairs else "doc_contrastive"

    doc_ids = sorted(doc_collection.keys())
    if cfg.random_doc_pairs:
        if len(doc_ids) < 2:
            raise ValueError(f"need >=2 docs, got {len(doc_ids)}")
        pairs = _sample_doc_pairs_random(doc_ids, n_pairs=cfg.n_pairs, rng=rng)
    else:
        if len(doc_ids) < cfg.n_clusters * 2:
            raise ValueError(
                f"doc collection too small for {cfg.n_clusters} clusters: "
                f"got {len(doc_ids)} docs"
            )
        texts = [doc_collection[d] for d in doc_ids]
        embs = embed_fn(texts)
        if embs.shape[0] != len(doc_ids):
            raise RuntimeError("embed_fn returned wrong row count")
        pairs = _sample_doc_pairs(
            doc_ids, embs,
            n_clusters=cfg.n_clusters,
            n_pairs=cfg.n_pairs,
            pairs_per_cluster_pair=cfg.pairs_per_cluster_pair,
            rng=rng,
        )

    accepted: list[Candidate] = []
    n_gen = 0
    n_parse_fail = 0
    n_verify_fail = 0
    t0 = time.time()
    for d_plus, d_minus, c_plus, c_minus in pairs:
        n_gen += 1
        prompt = _GEN_PROMPT.format(
            doc_plus=doc_collection[d_plus],
            doc_minus=doc_collection[d_minus],
        )
        try:
            raw = generate_fn(prompt)
        except Exception:
            n_parse_fail += 1
            continue
        parsed = _parse_predicate_json(raw)
        if parsed is None or not isinstance(parsed.get("predicate"), str):
            n_parse_fail += 1
            continue
        predicate = parsed["predicate"].strip()
        if not predicate or len(predicate) > 500:
            n_parse_fail += 1
            continue

        # Optional order-swap verification: ask the LLM to answer the
        # predicate against each doc separately. Reject if the LLM's own
        # reply doesn't match the YES/NO labels we induced from the
        # generation prompt.
        if cfg.verify_with_order_swap:
            try:
                v_plus = _parse_yes_no(generate_fn(_VERIFY_PROMPT.format(
                    doc=doc_collection[d_plus],
                    predicate=predicate,
                )))
                v_minus = _parse_yes_no(generate_fn(_VERIFY_PROMPT.format(
                    doc=doc_collection[d_minus],
                    predicate=predicate,
                )))
            except Exception:
                n_verify_fail += 1
                continue
            if v_plus != "Yes" or v_minus != "No":
                n_verify_fail += 1
                continue

        meta = {
            "paired_doc_plus": d_plus,
            "paired_doc_minus": d_minus,
            "cluster_plus": c_plus,
            "cluster_minus": c_minus,
            "rationale": parsed.get("rationale", ""),
        }
        accepted.append(Candidate(
            cand_id=_cand_id(d_plus, d_minus, predicate, "plus"),
            predicate=predicate,
            doc_id=d_plus,
            expected_answer="Yes",
            source=src,
            meta={**meta, "role": "plus"},
        ))
        accepted.append(Candidate(
            cand_id=_cand_id(d_plus, d_minus, predicate, "minus"),
            predicate=predicate,
            doc_id=d_minus,
            expected_answer="No",
            source=src,
            meta={**meta, "role": "minus"},
        ))

    diagnostics = {
        "n_pairs_attempted": n_gen,
        "n_parse_fail": n_parse_fail,
        "n_verify_fail": n_verify_fail,
        "n_accepted_pairs": len(accepted) // 2,
        "wall_seconds": round(time.time() - t0, 2),
    }
    # Stash on the first record's meta so the driver can persist it.
    if accepted:
        accepted[0].meta["__synth_diagnostics__"] = diagnostics
    else:
        # Without this the caller silently writes an empty candidates file.
        # A high n_parse_fail usually means the generator endpoint rejected
        # every request (most often a wrong served-model name).
        print(f"[synth] WARNING: no candidates accepted -- {diagnostics}",
              file=sys.stderr, flush=True)
    return accepted
