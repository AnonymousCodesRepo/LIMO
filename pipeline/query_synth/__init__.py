"""Offline query synthesis for unsupervised cascade pretraining.

A query-synth strategy reads the doc collection (and optionally the experience
pool) and emits a JSONL of `Candidate` records. Strategies are swappable; the
downstream `label_synth` and `trainer` modules consume the same Candidate
schema regardless of how it was produced.

Strategies
----------
* ``doc_contrastive`` — Method B: cross-cluster doc-pair contrastive
  predicate mining. Forces predicates to discriminate between two real
  documents, eliminating the trivial-predicate failure mode at generation
  time.
"""

from .types import Candidate, dump_candidates, load_candidates

__all__ = ["Candidate", "dump_candidates", "load_candidates"]
