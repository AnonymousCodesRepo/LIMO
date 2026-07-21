"""Load per-(dataset, query) arrays for the Stretto baseline: the 0.8B log-odds,
the 70B gold label, and the human ground-truth label -- all from the cached
zero-shot predictions, no new inference for cuad / contract / opp115.

Join strategy (auto-detected from the 70B cache columns):
  * cuad / contract : 70B split cache has `document_name`; the 0.8B features live in
    the global `Qwen3.5-0.8B_confidence_3way.csv` (52 queries = 38 cuad + 14 contract)
    under the pre-split doc numbering, so we join on (document_name, query_name).
  * opp115 / hoc    : 70B cache has no `document_name`; the 0.8B features live in the
    per-dataset `Qwen3.5-0.8B_confidence_3way_{ds}.csv`, joined on (document_id, query_name).

log_odds = clip(logprob_true - logprob_false, -30, 30)  -> the accept/reject/unsure signal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(_REPO, "examples", "predictions", "zeroshot_all_queries")

# 70B split-cache file name per dataset (contract -> _contract.csv).
_P70 = lambda ds: os.path.join(CACHE, f"Llama-3.1-70B-Instruct_zeroshot_{ds}.csv")
_P08_GLOBAL = os.path.join(CACHE, "Qwen3.5-0.8B_confidence_3way.csv")
_P08_DS = lambda ds: os.path.join(CACHE, f"Qwen3.5-0.8B_confidence_3way_{ds}.csv")

LOGODDS_CLIP = 30.0


def _yn(series: pd.Series) -> np.ndarray:
    return (series.astype(str).str.strip().str.lower().isin({"yes", "1", "true"})).to_numpy().astype(np.int64)


@dataclass
class QueryData:
    dataset: str
    query_name: str
    qid: int
    doc_ids: np.ndarray      # (N,)
    log_odds: np.ndarray     # (N,) 0.8B accept/reject/unsure signal
    gold: np.ndarray         # (N,) 0/1 from the 70B prediction (the cascade gold)
    human: np.ndarray        # (N,) 0/1 from the human ground_truth


def load_dataset(ds: str, gold_cache: str | None = None,
                 small_conf: str | None = None) -> list[QueryData]:
    """Load per-query arrays. ``gold`` defaults to the Llama-3.1-70B prediction.

    ``gold_cache`` overrides the cascade gold with the prediction from another
    large-model zero-shot csv, joined on (document_id, query_name) — pass the
    397B canonical csv to run Stretto with 397B as the large model. The human
    ground_truth is unchanged.

    ``small_conf`` overrides the small-model 3-way confidence csv that supplies
    the log-odds (logprob_true/logprob_false) — pass the Llama-3.1-8B confidence
    cache to run Stretto with the 8B as the small model. Defaults to the 0.8B
    cache (per-dataset for opp115/hoc, the global one for cuad/contract)."""
    p70 = _P70(ds)
    if not os.path.exists(p70):
        raise FileNotFoundError(f"70B cache missing for {ds}: {p70}")
    df70 = pd.read_csv(p70)

    if "document_name" in df70.columns:            # cuad / contract
        p08 = small_conf or _P08_GLOBAL
        if not os.path.exists(p08):
            raise FileNotFoundError(f"global small-model cache missing: {p08}")
        df08 = pd.read_csv(p08, usecols=["document_name", "query_name",
                                         "logprob_true", "logprob_false"])
        key = ["document_name", "query_name"]
    else:                                          # opp115 / hoc
        p08 = small_conf or _P08_DS(ds)
        if not os.path.exists(p08):
            raise FileNotFoundError(
                f"per-dataset small-model cache missing for {ds}: {p08}\n"
                f"  (hoc has no confidence_3way cache yet -- generate it first)")
        df08 = pd.read_csv(p08, usecols=["document_id", "query_name",
                                         "logprob_true", "logprob_false"])
        key = ["document_id", "query_name"]

    df70 = df70.rename(columns={"prediction": "_gold_raw", "ground_truth": "_human_raw"})
    merged = df70.merge(df08, on=key, how="inner")
    if len(merged) < len(df70):
        missing = len(df70) - len(merged)
        print(f"[{ds}] WARNING: {missing}/{len(df70)} 70B rows had no matching 0.8B feature")

    lo = (merged["logprob_true"].to_numpy() - merged["logprob_false"].to_numpy())
    merged["_log_odds"] = np.clip(lo, -LOGODDS_CLIP, LOGODDS_CLIP)
    merged["_gold"] = _yn(merged["_gold_raw"])
    merged["_human"] = _yn(merged["_human_raw"])

    if gold_cache is not None:                     # swap the large-model gold (e.g. 397B)
        gdf = pd.read_csv(gold_cache, usecols=["document_id", "query_name", "prediction"])
        gmap = {(int(d), q): p for d, q, p in
                zip(gdf["document_id"], gdf["query_name"], gdf["prediction"])}
        new_gold, miss = [], 0
        for did, q in zip(merged["document_id"], merged["query_name"]):
            p = gmap.get((int(did), q))
            if p is None:
                miss += 1
                new_gold.append("No")            # fallback; flagged below
            else:
                new_gold.append(p)
        if miss:
            print(f"[{ds}] WARNING: {miss}/{len(merged)} rows missing in gold_cache "
                  f"{os.path.basename(gold_cache)}")
        merged["_gold"] = _yn(pd.Series(new_gold))

    out = []
    for qid, qname in enumerate(sorted(merged["query_name"].unique())):
        sub = merged[merged["query_name"] == qname].sort_values("document_id")
        out.append(QueryData(
            dataset=ds, query_name=qname, qid=qid,
            doc_ids=sub["document_id"].to_numpy(),
            log_odds=sub["_log_odds"].to_numpy().astype(np.float64),
            gold=sub["_gold"].to_numpy(),
            human=sub["_human"].to_numpy(),
        ))
    return out


def stratified_sample(gold: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Indices of a per-class stratified sample (so both classes appear)."""
    rng = np.random.default_rng(seed)
    idx = []
    for cls in (0, 1):
        pool = np.where(gold == cls)[0]
        if len(pool) == 0:
            continue
        k = max(1, int(round(frac * len(pool))))
        idx.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.sort(np.concatenate(idx)) if idx else np.arange(len(gold))
