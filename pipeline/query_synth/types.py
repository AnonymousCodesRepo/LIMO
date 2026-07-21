"""Shared types for query synthesis stage.

A Candidate is one (predicate, doc) pair the labeler will roll out through
the cascade. The schema is intentionally compact and deterministic so the
downstream stages can stream-process the JSONL.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Candidate:
    """One synthetic (predicate, doc) candidate."""

    cand_id: str
    predicate: str
    doc_id: int
    expected_answer: str            # "Yes" | "No"  (per the source method's intended label)
    source: str                      # e.g. "doc_contrastive"
    # Method-specific provenance: the contrastive partner doc, the cluster
    # ids the doc pair was sampled from, the LLM call meta, etc. Free-form
    # to avoid coupling the schema to one method's needs.
    meta: dict[str, Any] = field(default_factory=dict)


def dump_candidates(path: str | Path, candidates: Iterable[Candidate]) -> int:
    """Append candidates to a JSONL file. Returns number of records written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "a") as f:
        for c in candidates:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_candidates(path: str | Path) -> list[Candidate]:
    p = Path(path)
    out: list[Candidate] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(Candidate(
                cand_id=r["cand_id"],
                predicate=r["predicate"],
                doc_id=int(r["doc_id"]),
                expected_answer=r["expected_answer"],
                source=r["source"],
                meta=r.get("meta", {}),
            ))
    return out
