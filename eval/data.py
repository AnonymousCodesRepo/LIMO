"""Data slice resolution + DataPoint loading.

Each dataset declares its own document table, text column, and the
(document, query, ground-truth) "pairs" file. CUAD reads its pairs from the
70B zero-shot prediction CSV; OPP-115 and HoC ship a self-contained
``pairs_ground_truth.csv`` next to their documents.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.common.prompts import DEFAULT_LARGE_SYSTEM
from pipeline.common.types import DataPoint


REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "examples" / "predictions" / "zeroshot_all_queries"
LEGALBENCH_DIR = REPO_ROOT / "datasets" / "LegalBench"
OPP115_DIR = REPO_ROOT / "datasets" / "opp115"
HOC_DIR = REPO_ROOT / "datasets" / "hoc"


# Domain-neutral base system sentence for the large-LLM 2-way prompt, used by
# non-legal datasets (e.g. HoC cancer abstracts). Same structure as
# ``DEFAULT_LARGE_SYSTEM`` with the "legal" framing removed.
NEUTRAL_LARGE_SYSTEM = (
    "You are a careful document classifier. Given a document and a "
    "yes/no question, output only `Yes` or `No`."
)


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset's documents, queries, and labels live.

    ``doc_dir`` holds ``merged_file`` (document body) and
    ``query_name_mapping.csv`` (query descriptions). ``pairs_file`` is the
    (document_id, query_name, ground_truth, ...) source used to enumerate
    pairs and read ground-truth labels. ``large_system_prompt`` is the base
    system sentence for the large-LLM 2-way prompt; it is dataset-specific so
    the domain framing matches the corpus (CUAD keeps its exact legal wording
    for cache bit-exactness).
    """

    doc_dir: Path
    merged_file: str
    text_col: str
    pairs_file: Path
    large_system_prompt: str = DEFAULT_LARGE_SYSTEM


DATASETS: dict[str, DatasetSpec] = {
    # CUAD: 482 long commercial contracts, paragraph-level text; 38 binary
    # clause-type queries. The 70B zero-shot CSV doubles as the pairs file
    # (ground truth) and the large-prediction cache.
    "cuad": DatasetSpec(
        doc_dir=LEGALBENCH_DIR / "cuad",
        merged_file="merged_df_paragraph.csv",
        text_col="merged_paragraph",
        pairs_file=PRED_DIR / "Llama-3.1-70B-Instruct_zeroshot_cuad.csv",
        large_system_prompt=DEFAULT_LARGE_SYSTEM,
    ),
    # OPP-115 privacy-policy clauses, segment-level. One segment = one unit;
    # nine binary practice-category queries (union of >=1 annotator => Yes,
    # absence => No). Ships a self-contained pairs_ground_truth.csv.
    "opp115": DatasetSpec(
        doc_dir=OPP115_DIR,
        merged_file="merged_df.csv",
        text_col="merged_text",
        pairs_file=OPP115_DIR / "pairs_ground_truth.csv",
        large_system_prompt=DEFAULT_LARGE_SYSTEM,
    ),
    # Hallmarks of Cancer, document-level (one PubMed abstract = one unit).
    # Ten binary hallmark queries; ground truth is the union over sentence-level
    # annotations (any sentence hits a hallmark => Yes, absence => No).
    # Non-legal, so uses the neutral system prompt.
    "hoc": DatasetSpec(
        doc_dir=HOC_DIR,
        merged_file="merged_df.csv",
        text_col="merged_text",
        pairs_file=HOC_DIR / "pairs_ground_truth.csv",
        large_system_prompt=NEUTRAL_LARGE_SYSTEM,
    ),
}


PHASE_A_QUERIES: list[str] = [
    "cuad_affiliate_license-licensee",
    "cuad_rofr-rofo-rofn",
    "cuad_renewal_term",
    "cuad_anti-assignment",
    "cuad_cap_on_liability",
    "cuad_governing_law",
]


def resolve_query_list(spec: str | None, *, dataset: str = "cuad") -> list[str]:
    """Resolve a query slice spec to a concrete list of query names.

    ``all_cuad`` / ``all_eval`` / ``all`` read the query set from the given
    dataset's pairs file. ``phase_a`` (or empty) is the default six-query CUAD
    slice. Anything else is treated as a comma-separated list.
    """
    if spec is None or spec == "" or spec == "phase_a":
        return list(PHASE_A_QUERIES)
    if spec == "all_cuad":
        return _read_query_set(dataset, prefix="cuad")
    if spec in ("all_eval", "all"):
        return _read_query_set(dataset, prefix=None)
    return [q.strip() for q in spec.split(",") if q.strip()]


def _read_query_set(dataset: str, prefix: str | None) -> list[str]:
    if dataset not in DATASETS:
        raise ValueError(
            f"unknown dataset: {dataset!r} (known: {sorted(DATASETS)})"
        )
    seen: set[str] = set()
    with open(DATASETS[dataset].pairs_file) as f:
        for r in csv.DictReader(f):
            q = r["query_name"]
            if not q:
                continue
            if prefix is None or q.startswith(prefix):
                seen.add(q)
    return sorted(seen)


def load_points(
    queries: list[str],
    *,
    dataset: str = "cuad",
    within_query_order: str = "doc_id_asc",
    order_seed: int = 0,
    limit_per_query: int | None = None,
) -> list[DataPoint]:
    """Load every (doc, query) pair for the requested queries, ordered.

    Query-block order follows the input list. Within each query, docs are
    ordered by ``within_query_order`` ∈ {doc_id_asc, random,
    doc_length_asc, doc_length_desc}.
    """
    if dataset not in DATASETS:
        raise ValueError(
            f"unknown dataset: {dataset!r} (known: {sorted(DATASETS)})"
        )
    spec = DATASETS[dataset]

    with open(spec.pairs_file) as f:
        rows = list(csv.DictReader(f))

    merged = pd.read_csv(spec.doc_dir / spec.merged_file)
    docs = {int(r.document_id): str(getattr(r, spec.text_col)) for r in merged.itertuples()}
    names = {int(r.document_id): str(r.document_name) for r in merged.itertuples()}
    qmap = pd.read_csv(spec.doc_dir / "query_name_mapping.csv")
    q_desc = dict(zip(
        qmap["query_name"].str.strip(), qmap["query_description"].str.strip()
    ))

    by_q: dict[str, list[DataPoint]] = {q: [] for q in queries}
    qset = set(queries)
    for r in rows:
        q = r["query_name"]
        if q not in qset:
            continue
        doc_id = int(r["document_id"])
        if doc_id not in docs:
            continue
        by_q[q].append(DataPoint(
            doc_id=doc_id,
            doc_name=names.get(doc_id, ""),
            doc_text=docs[doc_id],
            query_name=q,
            query_description=q_desc.get(q, ""),
            ground_truth=r["ground_truth"],
        ))

    ordered: list[DataPoint] = []
    for q in queries:
        pts = by_q[q]
        if within_query_order == "doc_id_asc":
            pts.sort(key=lambda p: p.doc_id)
        elif within_query_order == "random":
            r = random.Random(int(order_seed) + hash(q) % 1000)
            r.shuffle(pts)
        elif within_query_order == "doc_length_asc":
            pts.sort(key=lambda p: (len(p.doc_text or ""), p.doc_id))
        elif within_query_order == "doc_length_desc":
            pts.sort(key=lambda p: (-len(p.doc_text or ""), p.doc_id))
        else:
            raise ValueError(
                f"unknown within_query_order: {within_query_order!r}"
            )
        if limit_per_query is not None:
            pts = pts[:limit_per_query]
        ordered.extend(pts)
    return ordered
