"""Cached large-LLM predictions for the cascade runner.

When the router escalates, the runner normally invokes the live large LLM.
For experiments — where the same (query_name, doc_id) is escalated
repeatedly across configs — the prediction can instead be read from a
precomputed cache to save wall-clock time.

Semantic note: the cache stores the plain zero-shot large-LLM output (no
retrieved experiences, no fewshot demos). The live ``_call_large`` in the
runner builds a richer prompt that does include them; using the cache
trades that signal for speed and for determinism (every config sees the
same large answer for a given (query, doc), so comparisons are exact).

Opt-in via the runner's ``large_prediction_cache`` parameter and the
``--large-prediction-cache`` CLI flag.

Cache shape: ``{(query_name, doc_id): (prediction, raw)}``.
"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Iterable

from pipeline.common.prompts import parse_yes_no
from pipeline.common.types import DataPoint


class LargePredictionCache:
    """In-memory cache of pre-computed large-LLM predictions.

    Constructor inputs are lists of source files, merged in order (later
    sources overwrite earlier ones on key collision).

    Supported source formats:
      * CSV with columns at minimum ``query_name``, ``document_id``,
        ``prediction`` (raw read from ``raw_response`` if present, else
        ``prediction``).
      * JSONL with one record per line:
        ``{"query_name": str, "doc_id": int,
           "prediction": "Yes"|"No"|"UNKNOWN", "raw": str}``.

    An internal lock protects ``_table`` against concurrent ``store()``
    writes (used only when ``persist_to`` is set).
    """

    def __init__(
        self,
        sources: Iterable[str | Path] | None = None,
        *,
        persist_to: str | Path | None = None,
        normalize_predictions: bool = True,
    ):
        self._table: dict[tuple[str, int], tuple[str, str]] = {}
        self._lock = threading.Lock()
        self._persist_to: Path | None = (
            Path(persist_to) if persist_to is not None else None
        )
        self._normalize = bool(normalize_predictions)
        self._stats = {
            "n_loaded": 0,
            "n_lookup_hits": 0,
            "n_lookup_misses": 0,
            "n_stores": 0,
        }
        for src in sources or []:
            self._load_one(Path(src))

    # ── Loading ──────────────────────────────────────────────────────────

    def _load_one(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            self._load_csv(path)
        elif suffix in (".jsonl", ".json"):
            self._load_jsonl(path)
        else:
            raise ValueError(
                f"unknown cache source extension {suffix!r} (expect .csv / .jsonl)"
            )

    def _load_csv(self, path: Path) -> None:
        with open(path) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            id_col = (
                "doc_id" if "doc_id" in cols
                else "document_id" if "document_id" in cols
                else None
            )
            if id_col is None:
                raise ValueError(
                    f"{path}: CSV must contain 'doc_id' or 'document_id'"
                )
            n_loaded = 0
            for row in reader:
                try:
                    qn = row["query_name"]
                    did = int(row[id_col])
                except (KeyError, ValueError):
                    continue
                pred = (row.get("prediction") or "").strip()
                raw = (row.get("raw_response") or pred).strip()
                if self._normalize and pred:
                    pred_norm = parse_yes_no(pred)
                    if pred_norm == "UNKNOWN":
                        pred_norm = parse_yes_no(raw)
                    pred = pred_norm
                if not pred:
                    continue
                self._table[(qn, did)] = (pred, raw)
                n_loaded += 1
        self._stats["n_loaded"] += n_loaded

    def _load_jsonl(self, path: Path) -> None:
        n_loaded = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qn = rec.get("query_name")
                try:
                    did = int(rec.get("doc_id"))
                except (TypeError, ValueError):
                    continue
                pred = (rec.get("prediction") or "").strip()
                raw = (rec.get("raw") or pred).strip()
                if self._normalize and pred:
                    pred = parse_yes_no(pred) or "UNKNOWN"
                if not pred or qn is None:
                    continue
                self._table[(qn, did)] = (pred, raw)
                n_loaded += 1
        self._stats["n_loaded"] += n_loaded

    # ── Lookup / store ────────────────────────────────────────────────────

    def lookup(self, point: DataPoint) -> tuple[str, str] | None:
        with self._lock:
            v = self._table.get((point.query_name, point.doc_id))
        if v is None:
            self._stats["n_lookup_misses"] += 1
            return None
        self._stats["n_lookup_hits"] += 1
        return v

    def store(self, point: DataPoint, prediction: str, raw: str) -> None:
        """Insert a fresh prediction (e.g. one we had to compute live).

        Persists to ``persist_to`` (JSONL append) when set so subsequent
        runs benefit. Idempotent under repeated calls with the same key.
        """
        key = (point.query_name, point.doc_id)
        with self._lock:
            self._table[key] = (prediction, raw)
            self._stats["n_stores"] += 1
        if self._persist_to is not None:
            rec = {
                "query_name": point.query_name,
                "doc_id": point.doc_id,
                "prediction": prediction,
                "raw": raw,
            }
            self._persist_to.parent.mkdir(parents=True, exist_ok=True)
            # Append-only — duplicate keys are fine; loaders take the last.
            with open(self._persist_to, "a") as f:
                f.write(json.dumps(rec) + "\n")

    # ── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "n_entries": len(self._table),
                **self._stats,
                "hit_rate": (
                    self._stats["n_lookup_hits"]
                    / max(1, self._stats["n_lookup_hits"]
                          + self._stats["n_lookup_misses"])
                ),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._table)
