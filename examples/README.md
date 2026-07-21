# examples/

`predictions/zeroshot_all_queries/` holds the shipped model prediction caches:

* `Llama-3.1-70B-Instruct_zeroshot_{cuad,opp115,hoc}.csv` — large-model
  zero-shot verdicts over ALL (document, query) pairs. These double as the
  reference labels (agreement target) and as the escalation cache, so cascade
  evaluations rarely need the large model live. **Do not regenerate casually**:
  every evaluation is aligned to these files.
* `Qwen3.5-0.8B_confidence_3way_{opp115,hoc}.csv` — small-model 3-way
  zero-shot anchors.
* `GPT-5.6-Luna_zeroshot_opp115.csv` — commercial-large replay cache with real
  billed per-call costs (Section 7.7).

Collector scripts in this directory regenerate caches against your own
endpoints (`eval_zeroshot_all_queries.py`, `collect_*.py`); the generic
generator is `scripts/generate_predictions.py`.
