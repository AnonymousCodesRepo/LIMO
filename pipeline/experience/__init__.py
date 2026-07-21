"""Experience stage: load a pre-generated experience pool from disk.

Entry points:
  - `load(path_or_name)` — accepts a filesystem path or a short name
    registered in `REGISTRY` (e.g. "phase_a").
  - `REGISTRY` — maps short names to bundled JSONL files under
    `pipeline/experience/data/`.

Note: paper-correct cascade evals start from an EMPTY pool with online
generation; the bundled `phase_a` pool is kept for offline trainer parity.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.common.types import Experience


DATA_DIR = Path(__file__).resolve().parent / "data"

REGISTRY: dict[str, Path] = {
    "phase_a": DATA_DIR / "phase_a.jsonl",
}


def resolve(path_or_name: str | Path) -> Path:
    """Map a short name to its bundled path, or return the path unchanged."""
    key = str(path_or_name)
    if key in REGISTRY:
        return REGISTRY[key]
    return Path(path_or_name)


def load(path_or_name: str | Path) -> list[Experience]:
    """Read an experience-pool JSONL file.

    Only the subset of fields needed downstream is kept; extras are ignored.
    """
    out: list[Experience] = []
    p = resolve(path_or_name)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(Experience(
                experience_id=r["experience_id"],
                source_query=r["source_query"],
                source_doc_id=int(r["source_doc_id"]),
                source_doc_excerpt=r.get("source_doc_excerpt", ""),
                experience_text=r.get("experience_text", ""),
                applicability_signal=r.get("applicability_signal", ""),
            ))
    return out
