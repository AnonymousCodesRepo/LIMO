"""Build self-contained ladder inputs for the Track A (KV-cache compression)
profiling job on the GPU cluster.

Per dataset, emits under --out (default tools/stretto/ladder_inputs/):
  {ds}_docs.csv    : document_id, document_name, doc_text          (unique docs)
  {ds}_queries.csv : query_name, query_description
  {ds}_pairs.csv   : document_id, document_name, query_name, in_sample70

in_sample70 marks the per-query stratified profiling sample for the expensive
70B compressed rungs (stratified by the 70B gold label, seed 43+qid, frac
min(0.15, cap/n) with cap=300). The cheap 0.8B compressed rungs run on ALL rows.

Doc texts come from the same dataset dirs the collectors use:
  cuad/contract -> datasets/LegalBench/{ds}/merged_df.csv (merged_text)
  opp115/hoc    -> datasets/{ds}/merged_df.csv            (merged_text)
Gold labels come from the cached 70B zero-shot predictions (via stretto_data).

Run locally (pandas/numpy only):  PYTHONPATH=. python3 tools/stretto/build_manifests.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stretto_data as sd  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOC_DIR = {
    "cuad": os.path.join(_REPO, "datasets", "LegalBench", "cuad"),
    "contract": os.path.join(_REPO, "datasets", "LegalBench", "contract"),
    "opp115": os.path.join(_REPO, "datasets", "opp115"),
    "hoc": os.path.join(_REPO, "datasets", "hoc"),
}

SAMPLE_CAP = 300
SAMPLE_FRAC = 0.15
SEED = 43


def build(ds: str, out_dir: str) -> None:
    ddir = DOC_DIR[ds]
    merged = pd.read_csv(os.path.join(ddir, "merged_df.csv"))
    docs = merged[["document_id", "document_name", "merged_text"]].rename(
        columns={"merged_text": "doc_text"})
    qmap = pd.read_csv(os.path.join(ddir, "query_name_mapping.csv"))
    queries = qmap[["query_name", "query_description"]].dropna()

    # pairs + the 70B-rung profiling sample, from the same loader run_stretto uses
    qdata = sd.load_dataset(ds)
    pair_rows = []
    for q in qdata:
        n = len(q.gold)
        frac = min(SAMPLE_FRAC, SAMPLE_CAP / max(n, 1))
        tr = sd.stratified_sample(q.gold, frac, SEED + q.qid)
        in_sample = np.zeros(n, dtype=bool)
        in_sample[tr] = True
        for i in range(n):
            pair_rows.append((int(q.doc_ids[i]), q.query_name, bool(in_sample[i])))
    pairs = pd.DataFrame(pair_rows, columns=["document_id", "query_name", "in_sample70"])
    name_by_id = dict(zip(docs["document_id"], docs["document_name"]))
    pairs.insert(1, "document_name", pairs["document_id"].map(name_by_id))

    # keep only docs actually referenced (LegalBench merged_df spans the pre-split corpus)
    used = set(pairs["document_id"])
    docs = docs[docs["document_id"].isin(used)]
    used_q = set(pairs["query_name"])
    queries = queries[queries["query_name"].isin(used_q)]

    if pairs["document_name"].isna().any():
        raise SystemExit(f"[{ds}] some pair doc_ids missing from merged_df -- id mismatch!")
    if len(queries) != len(used_q):
        missing = used_q - set(queries["query_name"])
        raise SystemExit(f"[{ds}] query descriptions missing for: {sorted(missing)[:5]}")

    os.makedirs(out_dir, exist_ok=True)
    docs.to_csv(os.path.join(out_dir, f"{ds}_docs.csv"), index=False)
    queries.to_csv(os.path.join(out_dir, f"{ds}_queries.csv"), index=False)
    pairs.to_csv(os.path.join(out_dir, f"{ds}_pairs.csv"), index=False)
    n70 = int(pairs["in_sample70"].sum())
    print(f"[{ds}] docs={len(docs)} queries={len(queries)} pairs={len(pairs)} "
          f"sample70={n70} ({n70/len(pairs):.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cuad,contract,opp115,hoc")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "ladder_inputs"))
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        build(ds.strip(), args.out)


if __name__ == "__main__":
    main()
