"""Label synthesis for offline pretraining.

Consumes a stream of `Candidate` records (from `pipeline.query_synth`) and
produces `Rollout` records — one per labeled (predicate, doc) pair —
containing the silver label ``z = 1[small_pred == large_pred]`` together
with the feature vector the router head was built on, the small-LLM
confidence, and the importance-sampling probability ``q`` used to select
the candidate for large-LLM evaluation.

Strategies
----------
* ``active_acquisition`` — Method C: budget-aware iterative active
  learning over the candidate pool. Round 0 acquires by small-LLM
  entropy (no router yet); later rounds acquire by the in-training
  router's uncertainty p(1-p) over P(z=1 | features), mixed with an
  ε-uniform fraction. Importance-sampling weights ``1/q`` are emitted
  with each rollout for an unbiased fit.
"""

from .types import Rollout, dump_rollouts, load_rollouts

__all__ = ["Rollout", "dump_rollouts", "load_rollouts"]
