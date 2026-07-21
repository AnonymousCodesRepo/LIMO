"""Shared plumbing for the UQE and lightProxy proxy-cascade baselines.

Both baselines replace the small LLM entirely with a cheap logistic-regression
(LR) proxy trained on frozen all-mpnet-base-v2 document embeddings, using the
large LLM M_L (Llama-3.1-70B) zero-shot predictions as the supervision labels.
When a row is served by M_L its prediction = the cached 70B answer; otherwise it
is served by the proxy. Final accuracy / F1 are always measured against the
expert gold labels (ground_truth), matching the rest of the repo.

This module handles the parts both baselines share:
  * loading (doc, query) pairs + gold labels via ``eval.data.load_points``
  * fetching M_L (70B) labels from the cached zero-shot CSV
  * embedding the documents with the mpnet server (cached to disk as .npz)
  * a single-class-safe LR fit-and-predict helper

The two runner scripts (run_uqe.py, run_lightproxy.py) import from here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.data import PRED_DIR, load_points, resolve_query_list  # noqa: E402
from pipeline.common.embeddings import EmbeddingClient  # noqa: E402
from pipeline.common.large_prediction_cache import LargePredictionCache  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


def yn_to_int(s: str) -> int | None:
    """'Yes' -> 1, 'No' -> 0, anything else (incl. 'UNKNOWN') -> None."""
    if s == "Yes":
        return 1
    if s == "No":
        return 0
    return None


@dataclass
class QueryData:
    """Per-query arrays, all aligned by row (docs sorted by doc_id ascending)."""

    query_name: str
    doc_ids: np.ndarray            # int   [N]
    emb: np.ndarray                # f32   [N, 768]
    gold: np.ndarray              # int   [N]  (1 = Yes, 0 = No)
    ml_int: np.ndarray            # int   [N]  (1 / 0, or -1 when M_L = UNKNOWN)
    ml_str: list[str] = field(default_factory=list)   # 'Yes' / 'No' / 'UNKNOWN'

    @property
    def n(self) -> int:
        return len(self.doc_ids)


def _embed_documents(
    dataset: str,
    doc_ids: list[int],
    texts: list[str],
    *,
    cache_dir: Path,
    embed_url: str | None,
) -> dict[int, np.ndarray]:
    """Return {doc_id: 768-d embedding}, cached to ``cache_dir`` as .npz.

    all-mpnet-base-v2 is deterministic, so the vectors are bit-identical across
    runs that embed the same text, guaranteeing doc_id alignment."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{dataset}_docembeds.npz"
    if cache.exists():
        blob = np.load(cache)
        cached = {int(d): blob["emb"][i] for i, d in enumerate(blob["doc_ids"])}
        missing = [d for d in doc_ids if d not in cached]
        if not missing:
            return cached

    client = EmbeddingClient(url=embed_url) if embed_url else EmbeddingClient()
    emb = client.embed(texts)                       # [N, 768] float32
    out = {int(d): emb[i] for i, d in enumerate(doc_ids)}
    np.savez(
        cache,
        doc_ids=np.asarray(doc_ids, dtype=np.int64),
        emb=emb.astype(np.float32),
    )
    return out


def load_dataset(
    dataset: str,
    *,
    queries: str = "all",
    cache_dir: Path,
    embed_url: str | None = None,
    ml_cache_sources: list[str | Path] | None = None,
) -> tuple[dict[str, QueryData], dict]:
    """Load every query of ``dataset`` into aligned per-query arrays.

    Returns (by_query, stats). ``stats`` reports M_L coverage so degenerate
    inputs (missing large-model labels) are visible in the logs.

    ``ml_cache_sources`` selects which zero-shot cache supplies the large-model
    (M_L) labels. Defaults to the Llama-3.1-70B cache; pass the 397B canonical
    csv to run the baseline with 397B as the large model.
    """
    qlist = resolve_query_list(queries, dataset=dataset)
    points = load_points(qlist, dataset=dataset)

    uniq: dict[int, str] = {}
    for p in points:
        uniq.setdefault(p.doc_id, p.doc_text)
    doc_ids_sorted = sorted(uniq)
    id2emb = _embed_documents(
        dataset,
        doc_ids_sorted,
        [uniq[d] for d in doc_ids_sorted],
        cache_dir=cache_dir,
        embed_url=embed_url,
    )

    if ml_cache_sources is None:
        ml_cache_sources = [PRED_DIR / f"Llama-3.1-70B-Instruct_zeroshot_{dataset}.csv"]
    ml_cache = LargePredictionCache(sources=[Path(s) for s in ml_cache_sources])

    by_q: dict[str, list] = {}
    for p in points:
        by_q.setdefault(p.query_name, []).append(p)

    out: dict[str, QueryData] = {}
    n_ml_unknown = 0
    n_gold_bad = 0
    for q, pts in by_q.items():
        pts = sorted(pts, key=lambda p: p.doc_id)
        dids = [p.doc_id for p in pts]
        emb = np.stack([id2emb[d] for d in dids]).astype(np.float32)
        gold = []
        ml_int = []
        ml_str = []
        for p in pts:
            g = yn_to_int(p.ground_truth)
            if g is None:
                n_gold_bad += 1
                g = 0
            gold.append(g)
            v = ml_cache.lookup(p)
            pred = v[0] if v else "UNKNOWN"
            ml_str.append(pred)
            mi = yn_to_int(pred)
            if mi is None:
                n_ml_unknown += 1
                mi = -1
            ml_int.append(mi)
        out[q] = QueryData(
            query_name=q,
            doc_ids=np.asarray(dids, dtype=np.int64),
            emb=emb,
            gold=np.asarray(gold, dtype=np.int64),
            ml_int=np.asarray(ml_int, dtype=np.int64),
            ml_str=ml_str,
        )

    stats = {
        "dataset": dataset,
        "n_queries": len(out),
        "n_docs_unique": len(doc_ids_sorted),
        "n_pairs": len(points),
        "n_ml_unknown": n_ml_unknown,
        "n_gold_nonbinary": n_gold_bad,
        "ml_cache": ml_cache.stats(),
    }
    return out, stats


def fit_proxy_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    C: float = 1.0,
    class_weight: str | None = "balanced",
    max_iter: int = 1000,
) -> np.ndarray:
    """P(positive) for every row of ``X_eval``, single-class-safe.

    When the training labels have only one class (common on skewed queries)
    the LR would refuse to fit, so we return a constant probability equal to
    that class."""
    classes = np.unique(y_train)
    if len(classes) < 2:
        return np.full(len(X_eval), float(int(classes[0])), dtype=np.float64)
    lr = LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter)
    lr.fit(X_train, y_train)
    pos_col = list(lr.classes_).index(1)
    return lr.predict_proba(X_eval)[:, pos_col]


def confusion_from_served(
    served_pred_int: np.ndarray,
    served_is_unknown: np.ndarray,
    gold: np.ndarray,
) -> dict[str, int]:
    """Confusion counts under the repo's binary_metrics convention.

    ``served_pred_int`` is 1/0 (Yes/No); rows flagged in ``served_is_unknown``
    are parse failures (excluded from tp/fp/fn/tn, matching eval.metrics)."""
    tp = fp = fn = tn = pf = 0
    for i in range(len(gold)):
        if served_is_unknown[i]:
            pf += 1
            continue
        g = int(gold[i])
        p = int(served_pred_int[i])
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 0:
            tn += 1
        elif g == 0 and p == 1:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "parse_fail": pf}
